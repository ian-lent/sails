"""One boat: state, kinematics, and what a manoeuvre costs.

MANOEUVRE COST IS THE PARAMETER THIS PROJECT TURNS ON, so it gets its own model
rather than a constant. The breakeven shift size for tacking is a direct function
of it: if a tack is free, tack on every wobble; if it costs three lengths, hold
through a lot of unpleasantness. Every strategic answer downstream inherits it.

The profile encoded here is YOURS, not mine, and it inverts what I would have
assumed from reading:

    light air   nearly free
    medium      more costly
    heavy       quite costly, 1-3 boat lengths

I had it backwards — I expected light air to be expensive because there is little
momentum to carry through the turn. The mechanism behind your version is
presumably that loss scales with the speed you had: in a breeze you are carrying
real speed, you dump most of it turning through the eye, and rebuilding takes time
against chop. In light air there is little speed to lose and a good roll tack can
return most of it. `tests/test_maneuver_cost.py` asserts the model reproduces your
numbers, so the claim is checked rather than described.

Two knobs, both per-boat so crews can differ:
  * `speed_loss` — the fraction of speed lost through the turn;
  * `recovery_s` — the time constant for rebuilding it.
Distance lost is roughly speed_loss * speed * recovery_s, which is why both must
grow with wind for the cost to grow as steeply as you describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geometry as geo
from .polar import Polar

# WHY THIS IS A "CARRY-THROUGH" MODEL AND NOT A "LOSS" MODEL.
#
# The first attempt charged an explicit speed penalty per tack. It double-counted:
# the polar already returns zero speed inside the no-go zone, so a boat steering
# through head-to-wind decelerates on its own, and adding a penalty on top made a
# light-air tack cost ~0.9 boat lengths — nowhere near the "nearly free" the user
# described. The polar collapse IS the cost mechanism; nothing needs adding to it.
#
# What the explicit model must supply instead is the thing that makes a light-air
# tack cheap: the ROLL. A well-rolled tack drives the rig through the turn, so the
# boat carries most of its entry speed across head-to-wind rather than obeying the
# polar. That is why the cost profile runs the way the user described rather than
# the way I first assumed — the question is not "how much do you lose" but "how
# much can the crew carry through", and carrying through is easy at 4 knots and
# nearly impossible at 18 against chop.
#
# THE THIRD TERM, AND WHY IT IS NEEDED.
#
# With only carry-through and turn rate, a tack at 18 knots cost barely more than
# one at 10 — because this polar's UPWIND speed plateaus near 4.7 knots (a hiking
# dinghy is limited by righting moment, not drag), so there is no more speed to
# lose in a breeze than in a blow. That contradicts the supplied profile, and the
# missing physics is RECOVERY: in a breeze you exit the tack overpowered, in chop,
# and rebuilding takes far longer than the same manoeuvre in flat light air. Heavy
# air is expensive because of how long you stay slow, not how slow you get.
#
# So `accel_s` is the speed-recovery time constant, and it stands in for SEA STATE
# as much as wind. That coupling is a simplification worth flagging: heavy air in
# flat water — a windy day on a protected bend of the river — should recover much
# faster than this profile implies. When sea state becomes its own input, this term
# moves there.
#
# (tws kt, carry-through fraction, turn rate deg/s, speed recovery time constant s).
MANEUVER_PROFILE: tuple[tuple[float, float, float, float], ...] = (
    (3.0, 0.60, 28.0, 2.2),
    (6.0, 0.55, 26.0, 2.8),
    (10.0, 0.45, 21.0, 3.8),
    (14.0, 0.38, 18.0, 5.0),
    (18.0, 0.30, 15.0, 6.2),
    (24.0, 0.24, 13.0, 7.2),
)

# A gybe in a non-spinnaker dinghy is cheaper than a tack: the boat turns through
# far less of the wind and never passes through the no-go zone at all, so the polar
# never collapses. Expressed as how much of the shortfall in carry-through is
# recovered — 1.0 would mean a gybe costs nothing.
GYBE_CARRY_RECOVERY = 0.75


def _interp_profile(tws_kt: float) -> tuple[float, float, float]:
    """(carry-through fraction, turn rate deg/s, recovery seconds) at this wind."""
    rows = MANEUVER_PROFILE
    if tws_kt <= rows[0][0]:
        return rows[0][1], rows[0][2], rows[0][3]
    if tws_kt >= rows[-1][0]:
        return rows[-1][1], rows[-1][2], rows[-1][3]
    for (w0, c0, r0, a0), (w1, c1, r1, a1) in zip(rows, rows[1:]):
        if w0 <= tws_kt <= w1:
            f = (tws_kt - w0) / (w1 - w0)
            return c0 + f * (c1 - c0), r0 + f * (r1 - r0), a0 + f * (a1 - a0)
    return rows[-1][1], rows[-1][2], rows[-1][3]


@dataclass
class Boat:
    """A single competitor.

    `speed_factor` and `handling` are where the fleet's spread lives. Because
    college racing rotates boats specifically to null out equipment differences,
    all variance belongs to the CREW — so these multiply the polar and the
    manoeuvre cost respectively, and nothing scales the hull. That is a modelling
    choice the format actually justifies, and it is worth keeping honest: if we
    ever see boat-dependent results, something has leaked.
    """

    boat_id: int
    name: str
    polar: Polar
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed_kt: float = 0.0
    # Crew quality, multiplying achievable polar speed. 1.0 is "sails the polar".
    speed_factor: float = 1.0
    # Manoeuvre skill. Below 1.0 is a better-than-nominal crew (cheaper tacks).
    handling: float = 1.0
    # Set per-manoeuvre from the wind-dependent profile; the default only applies
    # before the first one.
    max_turn_rate_deg_s: float = 24.0
    # While a manoeuvre is in progress the crew holds speed at this floor, which is
    # what a roll tack physically does. Zero when not manoeuvring.
    carry_floor_kt: float = 0.0
    maneuver_remaining_s: float = 0.0
    # Speed-recovery time constant, set from the wind-dependent profile.
    accel_tau_s: float = 3.0
    # Bookkeeping the analysis layer reads.
    leg: int = 0
    finished_at: float | None = None
    tacks: int = 0
    gybes: int = 0
    # Elapsed time at each mark rounding. Leg splits are how strategy gets
    # diagnosed — a policy that gains on the beat and gives it back on the run is
    # invisible in the finish time and obvious here.
    leg_times: list[float] = field(default_factory=list)
    # Lane quality. `dirty_air_s` is seconds spent with any measurable deficit and
    # `dirty_air_integral` is deficit-seconds, which is the honest one: five seconds
    # squarely behind a boat costs far more than thirty seconds clipping an edge.
    # Finishing position says a boat lost; these say whether bad air is why.
    dirty_air_s: float = 0.0
    dirty_air_integral: float = 0.0
    worst_deficit: float = 0.0
    # Start bookkeeping. `ocs` records that the boat was over at the gun; it stays
    # true after the boat has returned, because the penalty is the time it cost and
    # that must remain visible in the results.
    ocs: bool = False
    returning: bool = False
    start_side_m: float = 0.0
    # Position along the line at the gun: 0 at the pin, 1 at the committee boat.
    start_fraction: float = 0.5
    start_speed_kt: float = 0.0
    # Imposed during the pre-start when a boat is holding back. None means "sail".
    speed_cap_kt: float | None = None
    # Rules. `penalty_remaining_s` is time left spinning a two-turns penalty;
    # `fouls` counts infringements and `penalties_taken` completed turns. Contact
    # is counted separately because rule 14 binds both boats, not just the
    # give-way one.
    penalty_remaining_s: float = 0.0
    # Degrees turned so far in the current penalty. The spin ends when the circle
    # is COMPLETE, not when a clock says so — inferring it from elapsed time lost a
    # timestep to rounding and left the boat 8-15 degrees short of its turn.
    penalty_turned_deg: float = 0.0
    fouls: int = 0
    penalties_taken: int = 0
    contacts: int = 0
    gave_way_s: float = 0.0
    # Contact is counted per EPISODE, not per timestep. Two boats touching for six
    # seconds is one incident; without this it was counted twelve times and an
    # 18-boat race reported nearly eight thousand collisions.
    contact_cooldown_s: float = 0.0
    distance_sailed_m: float = 0.0
    # (t, x, y, heading, speed_kt, wind deficit, flags, leg). Tuples rather than
    # objects because an 18-boat race at 1 s resolution is ~24,000 of them and the
    # replay file has to stay small enough to open.
    track: list[tuple] = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return self.polar.boat_length_m

    @property
    def beam_m(self) -> float:
        return self.polar.beam_m

    def twa(self, wind_from: float) -> float:
        return geo.true_wind_angle(self.heading, wind_from)

    def tack(self, wind_from: float) -> str:
        return geo.tack_of(self.twa(wind_from))

    def target_speed_kt(self, tws_kt: float, wind_from: float) -> float:
        """Polar speed for the current angle, scaled by crew quality."""
        return self.polar.speed(tws_kt, self.twa(wind_from)) * self.speed_factor

    def begin_maneuver(self, tws_kt: float, kind: str) -> None:
        """Commit to a tack or gybe: set the carry-through floor and the turn rate.

        Nothing is "charged" here. The cost emerges from two things acting over the
        following seconds: the polar collapsing as the boat steers through the
        no-go zone, and the turn rate limiting how fast it gets out the other side.
        This only sets how much speed the crew manages to carry through against
        that collapse, which is what separates a good light-air roll tack from a
        survival tack in a breeze.
        """
        carry, turn_rate, recovery = _interp_profile(tws_kt)
        if kind == "gybe":
            # A gybe never crosses the no-go zone, so most of the shortfall in
            # carry-through simply does not apply.
            carry += (1.0 - carry) * GYBE_CARRY_RECOVERY
            self.gybes += 1
        else:
            self.tacks += 1

        # handling below 1.0 is a better crew: it recovers part of the shortfall.
        carry = min(1.0, carry + (1.0 - carry) * (1.0 - self.handling))
        self.max_turn_rate_deg_s = turn_rate
        self.accel_tau_s = recovery

        # A manoeuvre already in progress does not get a fresh full-speed floor —
        # being tacked on twice in quick succession should hurt more than once.
        floor = carry * self.speed_kt
        self.carry_floor_kt = floor if self.maneuver_remaining_s <= 0.0 else min(self.carry_floor_kt, floor)
        # Roughly how long the boat is turning, from the angle swept and the rate.
        self.maneuver_remaining_s = (90.0 if kind == "tack" else 60.0) / turn_rate

    def step(self, target_heading: float, tws_kt: float, wind_from: float, dt: float) -> None:
        """Advance one timestep toward a commanded heading."""
        self.heading = geo.turn_toward(self.heading, target_heading, self.max_turn_rate_deg_s, dt)

        target = self.target_speed_kt(tws_kt, wind_from)

        # While turning, the crew's roll holds speed at the carry floor against the
        # polar's collapse through the no-go zone.
        if self.maneuver_remaining_s > 0.0:
            target = max(target, self.carry_floor_kt)
            self.maneuver_remaining_s -= dt
            if self.maneuver_remaining_s <= 0.0:
                self.carry_floor_kt = 0.0

        # First-order approach to target speed. The lag matters even without
        # manoeuvres: a boat coming out of a lull does not snap to speed, and
        # acceleration off the start line is most of why the first thirty seconds
        # of a race decide so much.
        if self.speed_cap_kt is not None:
            target = min(target, self.speed_cap_kt)
        accel_tau = self.accel_tau_s if target > self.speed_kt else 2.0
        self.speed_kt += (target - self.speed_kt) * (1.0 - math.exp(-dt / accel_tau))
        effective = max(0.0, self.speed_kt)

        self.x, self.y = geo.step_position(self.x, self.y, self.heading, geo.ms(effective), dt)
        self.distance_sailed_m += geo.ms(effective) * dt

    def record(self, t: float, deficit: float = 0.0) -> None:
        """Snapshot for the replay.

        Position alone makes a replay of moving dots. Heading, speed, tack, lane
        quality and penalty state are what let someone watch a race and diagnose
        it: why that boat stalled, which side of the shift it was on, whether the
        pile-up at the mark looks like sailing or like a modelling artefact.
        """
        flags = 0
        if self.returning:
            flags |= 1
        if self.penalty_remaining_s > 0.0:
            flags |= 2
        if self.finished_at is not None:
            flags |= 4
        if self.ocs:
            flags |= 8
        self.track.append(
            (
                round(t, 1),
                round(self.x, 1),
                round(self.y, 1),
                round(self.heading, 1),
                round(self.speed_kt, 2),
                round(deficit, 3),
                flags,
                self.leg,
            )
        )
