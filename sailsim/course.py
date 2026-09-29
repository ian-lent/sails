"""The course, and a helmsman that can get round it.

Two things live here, and they are separable on purpose. `Course` is geometry and
bookkeeping — where the marks are, which leg a boat is on, when it has finished.
`Helm` is the decision layer: given wind and position, what heading to steer.

The split matters because Helm is the thing we eventually replace. The whole point
of the project is to search over steering policies, so the baseline policy must be
swappable without touching geometry. Everything in Helm below is deliberately the
DUMBEST defensible sailor: it lays the mark, tacks at the layline, and knows
nothing about opponents, shifts, current or the fleet. That is the control against
which any cleverness must prove itself, and it is worth resisting the urge to make
it good — a strong baseline hides how much a policy actually adds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import geometry as geo
from .boat import Boat


@dataclass(frozen=True)
class Mark:
    name: str
    x: float
    y: float
    # Rounded when the boat comes within this radius AND on the correct side.
    radius_m: float = 8.0
    # Which side the mark is left on: "port" means keep it to port, which is the
    # standard windward-leeward rounding.
    #
    # THIS WAS MISSING ENTIRELY and it was the largest correctness bug in the
    # model. Rounding used to be a bare distance test, so boats passed whichever
    # side they happened to arrive on -- measured over one race, exactly 9 of 18
    # went each way. Half the fleet rounding backwards means boats meet HEAD ON at
    # the mark, which cannot happen in a real race and was generating most of the
    # fouls near marks. It also left rule 18 resting on nothing: "the inside boat"
    # is undefined until there is a side to be inside of.
    rounding: str = "port"
    # Bearing of the leg approaching this mark, filled in by the course builder.
    # The approach direction is what decides which side "port rounding" puts a
    # boat on, so it cannot be derived from the mark alone.
    approach_bearing: float = 0.0

    def side_sign(self) -> int:
        """+1 if the mark should pass down the boat's port side, -1 for starboard."""
        return 1 if self.rounding == "port" else -1

    @property
    def is_rounded(self) -> bool:
        """False for the finish, which is crossed rather than rounded."""
        return self.rounding in ("port", "starboard")

    def _offset_unit(self) -> tuple[float, float]:
        """Unit vector from the mark toward the side boats should pass on.

        ONE definition, used by both the steering target and the did-it-round-it
        test. The first version derived them separately -- a bearing offset for the
        gate and a cross product for the test -- and they came out opposite, so the
        gate pulled boats east of the mark while the test demanded they be west.
        Neither was obviously wrong on its own. Deriving one from the other removes
        the chance of that sign drifting again.

        For a port rounding the mark passes down the boat's port side, so the boat
        is to the RIGHT of the approach direction: +90 degrees from it.
        """
        rad = math.radians(self.approach_bearing + 90.0 * self.side_sign())
        return math.sin(rad), math.cos(rad)

    def gate_point(self, offset_m: float) -> tuple[float, float]:
        """The point boats should steer at, offset to the correct side of the mark.

        Steering at the mark itself funnels the fleet onto a single point from
        every direction. Steering at a point a length or two to the correct side
        turns the rounding into a queue, which is what it is on the water.
        """
        ux, uy = self._offset_unit()
        return self.x + ux * offset_m, self.y + uy * offset_m

    def passed_correct_side(self, x: float, y: float) -> bool:
        """Is this point on the side of the mark a boat should be rounding from?"""
        ux, uy = self._offset_unit()
        return (x - self.x) * ux + (y - self.y) * uy > 0.0


@dataclass
class Course:
    """A windward-leeward course as a sequence of marks to be rounded in order."""

    marks: tuple[Mark, ...]
    # Start line endpoints: (committee boat, pin).
    start_boat: tuple[float, float]
    start_pin: tuple[float, float]
    name: str = "W/L"

    @property
    def line_length_m(self) -> float:
        return geo.distance(*self.start_pin, *self.start_boat)

    @property
    def leg_count(self) -> int:
        return len(self.marks)

    def target_mark(self, boat: Boat) -> Mark | None:
        if boat.leg >= len(self.marks):
            return None
        return self.marks[boat.leg]

    def update_progress(self, boat: Boat, t: float) -> None:
        """Advance the boat's leg if it has reached its mark."""
        mark = self.target_mark(boat)
        if mark is None:
            return
        # Both conditions: close enough AND on the correct side. A boat that cuts
        # the wrong side of the mark has not rounded it, and its gate point pulls
        # it back around rather than letting it carry on up the course.
        if (
            geo.distance(boat.x, boat.y, mark.x, mark.y) <= mark.radius_m
            and (not mark.is_rounded or mark.passed_correct_side(boat.x, boat.y))
        ):
            boat.leg_times.append(t)
            boat.leg += 1
            if boat.leg >= len(self.marks) and boat.finished_at is None:
                boat.finished_at = t

    @staticmethod
    def for_conditions(
        tws_kt: float,
        polar,
        upwind_minutes: float = 5.5,
        laps: int = 2,
        line_length_m: float = 140.0,
        wind_from: float = 0.0,
        line_bias_deg: float = 0.0,
    ) -> "Course":
        """Size the beat so the upwind leg takes about `upwind_minutes`.

        This is what a race committee actually does — the course is resized for the
        day, not fixed. It matters here because leg DURATION, not leg length, is what
        the target racing is specified by (4-7 minutes upwind, 3-5 down), and the
        same 500 m beat is a 6.2-minute leg in 6 knots and a 4.6-minute leg in 16.

        A fixed-length default cannot hold both windows across the wind range; this
        keeps the upwind leg in its window by construction, and the downwind leg
        follows because the two VMGs move together.
        """
        _, vmg_kt = polar.best_upwind(tws_kt)
        beat = geo.ms(vmg_kt) * upwind_minutes * 60.0
        return Course.windward_leeward(beat, laps, line_length_m, wind_from, line_bias_deg)

    @staticmethod
    def windward_leeward(
        beat_length_m: float = 500.0,
        laps: int = 2,
        line_length_m: float = 140.0,
        wind_from: float = 0.0,
        line_bias_deg: float = 0.0,
    ) -> "Course":
        """A W/L course built square to the wind.

        The 500 m default is calibrated, not arbitrary: at this polar's upwind VMG in
        9 knots it is a 5.3-minute beat and a 4.2-minute run, both inside the 4-7 and
        3-5 minute windows the target racing runs to. (An earlier 730 m default gave
        a 7.9-minute beat — outside the window, and the reason a demo race took 27
        minutes.) Use `for_conditions` to hold the duration across wind speeds.
        """
        rad = math.radians(wind_from)
        ux, uy = math.sin(rad), math.cos(rad)  # unit vector toward the wind
        px, py = uy, -ux                        # perpendicular, to the right

        placed: list[tuple[str, float, float, float]] = []
        for lap in range(laps):
            placed.append((f"windward-{lap + 1}", ux * beat_length_m, uy * beat_length_m, 8.0))
            # Final lap finishes at the line rather than rounding the leeward mark.
            if lap < laps - 1:
                placed.append((f"leeward-{lap + 1}", 0.0, 0.0, 8.0))
        placed.append(("finish", 0.0, 0.0, 12.0))

        # Stamp each mark with the bearing of the leg that arrives at it, which is
        # what decides which side a port rounding puts a boat on.
        marks: list[Mark] = []
        prev = (0.0, 0.0)
        for name, mx, my, radius in placed:
            bearing = geo.bearing(prev[0], prev[1], mx, my) if (mx, my) != prev else wind_from
            # The finish is crossed, not rounded, so it takes no side.
            rounding = "none" if name == "finish" else "port"
            marks.append(Mark(name, mx, my, radius, rounding, bearing))
            prev = (mx, my)

        # The line is rotated about its midpoint by the bias. Positive bias puts the
        # PIN upwind, matching start.line_bias's sign convention.
        half = line_length_m / 2.0
        axis = math.radians(wind_from + 90.0 + line_bias_deg)
        ax, ay = math.sin(axis), math.cos(axis)
        return Course(
            marks=tuple(marks),
            start_boat=(ax * half, ay * half),
            start_pin=(-ax * half, -ay * half),
        )


@dataclass
class Helm:
    """A minimal sailor: hold a tack until the mark can be laid on the other one.

    THE BUG THIS REPLACES is worth recording, because it is the canonical way a
    sailing simulator looks plausible and is nonsense. The first version steered by
    cross-track error to the mark — tack whenever you are more than N metres off
    the direct line. That produces a boat that zigzags up the rhumb line, and it
    measured 92 TACKS PER RACE. A real college beat is one or two. Every tack was
    being paid for, so the fleet was sailing a physically consistent race of a kind
    nobody has ever sailed, and nothing in the output looked obviously wrong.

    The fix is to use the actual geometric criterion. You lay a mark on tack t when
    the TWA required to sail straight at it is at least your close-hauled angle AND
    on the correct side. Inside the cone bounded by the two laylines you cannot lay
    it on either tack, so you hold your tack until you reach a layline and then tack
    once. In steady wind that yields the one-tack beat, which is correct: every
    tacking pattern covers the same distance, so the fewest manoeuvres wins.

    `tack_bias` decides only which tack to take first when neither lays. That is
    the whole of this helm's "strategy", and it is a placeholder — tacking on
    shifts, covering, and playing the favoured side all belong to the policy layer
    that replaces this class.
    """

    tack_bias: float = 0.0
    # Degrees past the layline before committing, so a boat sitting exactly on it
    # does not re-decide every timestep. Small, because overstanding is a real cost.
    layline_margin_deg: float = 1.5
    _committed: int = field(default=0, repr=False)
    # Which leg the commitment belongs to. Without this the boat carries its tack
    # choice around the mark and starts the next leg already committed to a side it
    # decided on for the previous one.
    _committed_leg: int = field(default=-1, repr=False)
    # Sign of the TWA this helm last COMMANDED. Manoeuvres are detected against
    # this, not against the boat's actual heading — see _maneuver.
    _last_sign: int = field(default=0, repr=False)

    def target_heading(
        self,
        boat: Boat,
        mark_x: float,
        mark_y: float,
        tws_kt: float,
        wind_from: float,
        t: float = 0.0,
        deficit: float = 0.0,
        traffic: "Sequence[Boat]" = (),
    ) -> tuple[float, str | None]:
        """(heading to steer, manoeuvre kind if this heading commits one).

        `t`, `deficit` and `traffic` are what a tactical helm needs and this one
        ignores: the clock, how much wind the boat is being denied, and the other
        boats. They are on the base signature so the simulator drives every policy
        the same way — and so this helm stays a genuine control, blind by choice
        rather than by lacking the inputs.

        Manoeuvre detection is CENTRAL here rather than per-branch: any commanded
        heading that flips the sign of the true wind angle is a tack or a gybe, and
        deciding that in one place is what stops a boat changing tack for free. The
        earlier version charged tacks only inside the beat branch, so a boat that
        became able to lay the mark on the other tack simply swapped sides at no
        cost — invisible in the output, and worth more than any tactical decision.
        """
        if boat.leg != self._committed_leg:
            self._committed = 0
            self._committed_leg = boat.leg
            self._last_sign = 0

        to_mark = geo.bearing(boat.x, boat.y, mark_x, mark_y)
        mark_twa = geo.wrap180(wind_from - to_mark)
        close_hauled, _ = boat.polar.best_upwind(tws_kt)
        broad, _ = boat.polar.best_downwind(tws_kt)

        if abs(mark_twa) < close_hauled:
            desired = self._beat(boat, mark_x, mark_y, wind_from, close_hauled)
        elif abs(mark_twa) > broad:
            desired = self._run(boat, mark_x, mark_y, wind_from, broad)
        else:
            # The mark is fetchable on this angle: point at it.
            self._committed = 0
            desired = to_mark

        return desired, self._maneuver(boat, desired, wind_from)

    def _maneuver(self, boat: Boat, desired_heading: float, wind_from: float) -> str | None:
        """'tack' | 'gybe' | None for the transition this heading commands.

        COMPARED AGAINST THE LAST COMMAND, NOT THE BOAT. The earlier version
        compared the desired TWA to the boat's actual TWA, which lags because the
        turn rate is capped — so for the three or four timesteps a tack takes, the
        boat was still on the old tack while the command was on the new one, and the
        same single tack was detected and charged on every one of those steps.
        Manoeuvres arrived in bursts and the fleet paid four times for each. A helm
        knows when it decided to tack; it does not need to infer it from the hull.
        """
        want = geo.wrap180(wind_from - desired_heading)
        want_sign = 1 if want >= 0.0 else -1

        if self._last_sign == 0:
            self._last_sign = 1 if boat.twa(wind_from) >= 0.0 else -1
        previous, self._last_sign = self._last_sign, want_sign
        if want_sign == previous:
            return None

        # Dead-band on the wind axis: a boat wobbling across dead downwind by a
        # degree is not gybing, and jitter there must not be charged as a manoeuvre.
        if abs(abs(want) - 180.0) < 5.0 or abs(want) < 5.0:
            self._last_sign = previous
            return None
        return "tack" if abs(want) < 90.0 else "gybe"

    def _lays(self, mark_twa: float, angle: float, tack: int, upwind: bool) -> bool:
        """Could this tack sail straight to the mark from here?

        THE INEQUALITY REVERSES BETWEEN UPWIND AND DOWNWIND, and getting that wrong
        cost this simulator a thousand gybes per race.

        Upwind, close-hauled at 41 degrees: a mark dead upwind needs TWA 0, which is
        inside the no-go zone, so you cannot lay it; a mark at TWA 50 is a fetch and
        you can. You lay it when the required angle is WIDER than close-hauled.

        Downwind, broad-reaching at 155: a mark dead downwind needs TWA 180, which is
        slower than the angles you are sailing, so you cannot lay it and must gybe;
        a mark at TWA 140 is a reach you can bear away to. You lay it when the
        required angle is NARROWER than your running angle.

        Using the upwind test downwind made every leeward mark "layable" on both
        gybes at once, so the committed gybe flipped on every timestep.
        """
        if (mark_twa > 0) != (tack > 0):
            return False
        if upwind:
            return abs(mark_twa) >= angle - self.layline_margin_deg
        return abs(mark_twa) <= angle + self.layline_margin_deg

    def _beat(
        self, boat: Boat, mark_x: float, mark_y: float, wind_from: float, close_hauled: float
    ) -> float:
        """Hold the committed tack until the other tack reaches the layline."""
        return self._sail_angle(boat, mark_x, mark_y, wind_from, close_hauled, upwind=True)

    def _run(
        self, boat: Boat, mark_x: float, mark_y: float, wind_from: float, broad: float
    ) -> float:
        """Same logic downwind: gybe at the layline for the downwind angle.

        With no spinnaker this polar makes dead downwind genuinely slow, so the boat
        sails angles and gybes once per run. Whether real FJs and C420s gybe that
        little is directly checkable from GPS tracks — gybe count per run is one of
        the easiest observables there is, and it is a good first calibration target.
        """
        return self._sail_angle(boat, mark_x, mark_y, wind_from, broad, upwind=False)

    def _sail_angle(
        self,
        boat: Boat,
        mark_x: float,
        mark_y: float,
        wind_from: float,
        angle: float,
        upwind: bool,
    ) -> float:
        current = 1 if boat.twa(wind_from) >= 0 else -1
        if self._committed == 0:
            # First commitment on this leg: take the tack that heads toward the
            # mark's side of the course, nudged by this crew's bias.
            self._committed = self._opening_tack(boat, mark_x, mark_y, wind_from, current)

        # Does the other tack now lay the mark? Recompute the mark's TWA because
        # both the boat and the wind have moved.
        to_mark = geo.bearing(boat.x, boat.y, mark_x, mark_y)
        mark_twa = geo.wrap180(wind_from - to_mark)
        other = -self._committed
        if self._lays(mark_twa, angle, other, upwind):
            self._committed = other

        return geo.heading_for_twa(self._committed * angle, wind_from)

    def _opening_tack(
        self, boat: Boat, mark_x: float, mark_y: float, wind_from: float, current: int
    ) -> int:
        """Which tack to start the leg on, when neither lays the mark.

        Cross-track position decides it: head toward the side the mark is on, so the
        boat sails into the cone rather than out of it. `tack_bias` can override for
        a crew that prefers a side — the only strategic preference this helm has.
        """
        rad = math.radians(wind_from)
        px, py = math.cos(rad), -math.sin(rad)
        cross = (mark_x - boat.x) * px + (mark_y - boat.y) * py
        if abs(cross) < 1.0:
            return current if self.tack_bias == 0.0 else (1 if self.tack_bias > 0 else -1)
        return -1 if cross > 0 else 1
