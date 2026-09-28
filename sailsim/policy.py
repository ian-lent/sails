"""The tactical helm: a sailor that looks at the wind and at the other boats.

THE BASELINE HELM IN course.py IS BLIND. It sails laylines and ignores the fleet,
which is deliberate — it is the control. This is the thing that replaces it, and
the thing a search will eventually tune, so every decision it makes is governed by
a named parameter rather than a constant buried in a branch.

Legibility is a design goal, not a nicety. The point of the project is to produce
decision rules a sailor can carry onto the water — "tack on anything beyond six
degrees, but not more often than every twenty seconds" — and a policy whose
behaviour cannot be stated in a sentence cannot produce one, however well it
sails.

THE ONE IDEA THIS IS BUILT ON. Every upwind and downwind tack decision reduces to
the same question: WHICH TACK POINTS CLOSER TO THE MARK? Sail that one.

That single rule turns out to be several pieces of classical advice at once:

  * In an oscillating breeze it is "tack on the headers". When you are headed, the
    other tack points closer, by exactly the size of the shift.
  * Off to one side of the course it is "sail the long tack". The tack that points
    back toward the mark is the one that closes the bearing.
  * Downwind it is the same rule with no modification at all, because a shift that
    heads you on a beat lifts you on a run and the geometry takes care of it.

So there is one comparison, not three special cases, and one threshold parameter
governing all of it. What stops the boat tacking on every ripple is the threshold
plus a minimum interval, and the trade between them is exactly what a parameter
sweep should explore: tacking costs boat lengths, and in a breeze it costs two to
three of them.

WHAT THIS STILL DOES NOT DO, since the list matters as much as the features:
covering a rival, splitting from the fleet to gain leverage, playing the favoured
side of a course with known geography, queueing at a crowded mark rounding, or any
pre-start play. It races the racecourse, not the opponents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geometry as geo
from .boat import Boat
from .course import Helm


@dataclass
class WindMemory:
    """A running estimate of the mean wind direction.

    Circular quantities cannot be averaged arithmetically — the mean of 350 and 10
    degrees is 0, not 180 — so this accumulates sine and cosine components and
    takes the angle of their resultant.

    The time constant is what makes the estimate a STRATEGY rather than a sensor.
    Short, and the mean chases every gust so nothing ever reads as a shift; long,
    and a genuine persistent trend is mistaken for a header for minutes. That the
    right value differs between an oscillating day and a persistent one is not a
    flaw in the model; it is the actual difficulty of the real problem.
    """

    time_constant_s: float = 120.0
    _sin: float = field(default=0.0, repr=False)
    _cos: float = field(default=0.0, repr=False)
    _seeded: bool = field(default=False, repr=False)

    def update(self, wind_from: float, dt: float) -> None:
        rad = math.radians(wind_from)
        if not self._seeded:
            self._sin, self._cos, self._seeded = math.sin(rad), math.cos(rad), True
            return
        alpha = 1.0 - math.exp(-dt / max(self.time_constant_s, 1e-6))
        self._sin += (math.sin(rad) - self._sin) * alpha
        self._cos += (math.cos(rad) - self._cos) * alpha

    @property
    def mean(self) -> float:
        return geo.wrap360(math.degrees(math.atan2(self._sin, self._cos)))

    def shift(self, wind_from: float) -> float:
        """Signed departure of the current wind from the running mean."""
        return geo.angle_diff(wind_from, self.mean)


@dataclass
class TacticalHelm(Helm):
    """A helm that tacks on shifts and works to keep its air clear.

    Parameters are the searchable surface. Defaults are plausible starting points,
    not answers — finding the answers is what the sweep in
    `experiments/shift_threshold.py` is for.
    """

    # How much closer to the mark the other tack must point before it is worth the
    # cost of getting there. The central parameter of the whole policy.
    # Default taken from experiments/shift_threshold.py rather than invented: the
    # cost curve bottoms out around 20-30 degrees of bearing advantage across 5-18
    # knots. A default of 8 made the boat chase the rhumb line, tacking back and
    # forth across it for 40 manoeuvres in a race with no shifts at all.
    shift_threshold_deg: float = 20.0
    # Floor on time between tactical tacks. Without it a shifty breeze produces a
    # boat that tacks on noise and pays for every one of them.
    min_tack_interval_s: float = 25.0
    # Wind-speed deficit that counts as dirty air, and how long to sit in it before
    # tacking out. Bailing instantly means tacking every time someone crosses; never
    # bailing means sailing a whole leg in someone's shadow.
    dirty_air_threshold: float = 0.06
    dirty_air_patience_s: float = 10.0
    # How far from laying the mark a tactical tack is still allowed. Near the
    # layline a tack is a commitment, not a tactic, and tacking there either
    # overstands or forces an immediate tack back.
    layline_guard_deg: float = 8.0
    # Require a tactical tack to be justified by an actual HEADER against the
    # running mean, not by geometry alone.
    #
    # Without this the "which tack points closer" rule has a flaw that only shows
    # up in steady wind: near the rhumb line the favoured tack flips as the boat
    # crosses it, so the boat tacks, overshoots, and tacks back. It logged 26
    # manoeuvres in a breeze with no shifts in it at all -- a slower rediscovery of
    # the same rhumb-line chasing that the very first cross-track helm did.
    #
    # The positional half of the rule is still doing useful work (it picks the long
    # tack at the start of a leg and it decides the layline tack); this only stops
    # it from re-deciding forever in the middle of the course. In a steady breeze
    # the boat now sails laylines, which is correct.
    require_header: bool = True

    memory: WindMemory = field(default_factory=WindMemory)
    _last_tack_t: float = field(default=-1e9, repr=False)
    _dirty_for_s: float = field(default=0.0, repr=False)
    _last_t: float = field(default=0.0, repr=False)

    # --- observation ---------------------------------------------------------
    def observe(self, wind_from: float, deficit: float, t: float) -> None:
        """Take in what the crew can actually see: the wind, and their own air."""
        dt = max(0.0, t - self._last_t)
        self._last_t = t
        self.memory.update(wind_from, dt)
        if deficit >= self.dirty_air_threshold:
            self._dirty_for_s += dt
        else:
            self._dirty_for_s = 0.0

    # --- the decision --------------------------------------------------------
    def target_heading(
        self,
        boat: Boat,
        mark_x: float,
        mark_y: float,
        tws_kt: float,
        wind_from: float,
        t: float = 0.0,
        deficit: float = 0.0,
    ) -> tuple[float, str | None]:
        self.observe(wind_from, deficit, t)

        if boat.leg != self._committed_leg:
            self._committed = 0
            self._committed_leg = boat.leg
            self._last_sign = 0
            self._dirty_for_s = 0.0

        to_mark = geo.bearing(boat.x, boat.y, mark_x, mark_y)
        mark_twa = geo.wrap180(wind_from - to_mark)
        close_hauled, _ = boat.polar.best_upwind(tws_kt)
        broad, _ = boat.polar.best_downwind(tws_kt)

        if abs(mark_twa) < close_hauled:
            desired = self._work(boat, to_mark, mark_twa, wind_from, close_hauled, t, upwind=True)
        elif abs(mark_twa) > broad:
            desired = self._work(boat, to_mark, mark_twa, wind_from, broad, t, upwind=False)
        else:
            self._committed = 0
            desired = to_mark

        return desired, self._maneuver(boat, desired, wind_from)

    def _work(
        self,
        boat: Boat,
        to_mark: float,
        mark_twa: float,
        wind_from: float,
        angle: float,
        t: float,
        upwind: bool,
    ) -> float:
        """Choose a tack, then sail it at the polar's best angle."""
        current = 1 if boat.twa(wind_from) >= 0 else -1
        if self._committed == 0:
            self._committed = self._better_tack(to_mark, wind_from, angle)
            self._last_tack_t = t

        # The layline still overrides everything: if the other tack lays the mark,
        # take it, whatever the shifts are doing. Sailing past the layline to chase
        # a shift is how a good strategic idea turns into an overstand.
        other = -self._committed
        if self._lays(mark_twa, angle, other, upwind):
            if other != self._committed:
                self._last_tack_t = t
            self._committed = other
            return geo.heading_for_twa(self._committed * angle, wind_from)

        if self._may_tack(mark_twa, angle, upwind, t) and self._should_tack(
            to_mark, wind_from, angle
        ):
            self._committed = -self._committed
            self._last_tack_t = t
            # Reset the dirty-air clock on tacking out. Without this the clock kept
            # running after the escape, the bail-out condition stayed true, and the
            # boat tacked straight back into the shadow it had just left — then out
            # again, for as long as the deficit lasted.
            self._dirty_for_s = 0.0

        return geo.heading_for_twa(self._committed * angle, wind_from)

    def _better_tack(self, to_mark: float, wind_from: float, angle: float) -> int:
        """Which tack points closer to the mark. The whole policy in one function."""
        return 1 if self._gain(1, to_mark, wind_from, angle) <= self._gain(
            -1, to_mark, wind_from, angle
        ) else -1

    @staticmethod
    def _gain(tack: int, to_mark: float, wind_from: float, angle: float) -> float:
        """How far off the mark this tack points, in degrees. Lower is better."""
        heading = geo.heading_for_twa(tack * angle, wind_from)
        return abs(geo.angle_diff(heading, to_mark))

    def headed_by(self, wind_from: float) -> float:
        """Degrees the current tack is headed relative to the running mean wind.

        Positive is headed, negative is lifted. On starboard the wind backing is a
        header, on port a veer is; one sign flip covers both.
        """
        return -self.memory.shift(wind_from) * self._committed

    def _should_tack(self, to_mark: float, wind_from: float, angle: float) -> bool:
        """Is the other tack better by more than the threshold, or is the air bad?"""
        here = self._gain(self._committed, to_mark, wind_from, angle)
        there = self._gain(-self._committed, to_mark, wind_from, angle)
        if here - there > self.shift_threshold_deg:
            if not self.require_header or self.headed_by(wind_from) > 0.0:
                return True
        # Bailing out of dirty air. Deliberately not conditioned on the shift: a
        # boat that has sat in someone's blanket for ten seconds is losing more
        # than any plausible shift is worth.
        return self._dirty_for_s >= self.dirty_air_patience_s

    def _may_tack(self, mark_twa: float, angle: float, upwind: bool, t: float) -> bool:
        """Guards: not too soon after the last tack, and not near a layline."""
        if t - self._last_tack_t < self.min_tack_interval_s:
            return False
        # How close either tack is to laying the mark. Inside the guard band a tack
        # is a layline commitment, and the layline branch above owns it.
        slack = self.layline_guard_deg
        if upwind:
            return abs(mark_twa) < angle - slack
        return abs(mark_twa) > angle + slack
