"""The start: line bias, the approach, and being over early.

WHY THE LINE BIAS MATTERS MORE THAN IT SOUNDS. A line set eight degrees off square
on a 140 m line puts one end 140 * sin(8) = 19 m — about five boat lengths —
upwind of the other. Five lengths is worth more than almost any tactical decision
available on a five-minute beat, and it is handed out for free at the gun. That
arithmetic is why the favoured end is the first thing a sailor looks for, and why
a simulator that starts every boat on a square line cannot say anything useful
about college racing.

TWO SOURCES OF BIAS, modelled separately because they behave differently:

  * COMMITTEE ERROR is fixed. The line is laid by eye off a boat that is itself
    swinging on its anchor; a few degrees is normal and it does not change during
    the sequence. Once you have read it, it stays read.
  * WIND SHIFT is not. The line is square when it is laid and the breeze moves
    before the gun, so the bias at the start is the shift accumulated since. This
    is the component that punishes reading the line early and not looking again,
    and on a shifty river it can exceed the committee's error several times over.

Their sum is what a boat actually sails against, and the split matters because
only one half of it can be scouted in advance.

WHAT IS DELIBERATELY NOT MODELLED. Real pre-start boat-on-boat play — luffing
duels, hunting, barging, port-tack approaches, defending a hole to leeward — is a
game in its own right and needs the rules first. What is here is the part that
sets up the race: where on the line a boat aims, whether it arrives at speed, and
whether it is over. Boats do not yet fight each other for a slot, which means this
model will understate how hard a good start at a crowded end actually is.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from . import geometry as geo

# The user's specification: bias is drawn in this range, never zero. A perfectly
# square line is the one case that never happens on the water.
BIAS_MIN_DEG = 1.0
BIAS_MAX_DEG = 15.0


@dataclass(frozen=True)
class LineBias:
    """The bias on a line, and where it came from."""

    total_deg: float
    committee_deg: float
    shift_deg: float

    @property
    def favoured(self) -> str:
        """'pin' | 'boat'. Positive bias puts the pin upwind; see line_bias()."""
        return "pin" if self.total_deg > 0 else "boat"

    def advantage_m(self, line_length_m: float) -> float:
        """How much further upwind the favoured end is, in metres."""
        return line_length_m * abs(math.sin(math.radians(self.total_deg)))


def draw_bias(rng: random.Random, committee_share: float = 0.45) -> LineBias:
    """Draw a race's line bias: magnitude in [1, 15] degrees, random side.

    The magnitude is drawn first and then split, rather than drawing the two
    sources independently and adding them. Independent draws would sometimes
    cancel to a square line, which is the one outcome the specification excludes —
    and a distribution with a hole at zero is not what summing two symmetric
    errors produces.
    """
    magnitude = rng.uniform(BIAS_MIN_DEG, BIAS_MAX_DEG)
    total = magnitude if rng.random() < 0.5 else -magnitude
    # The split wobbles around the nominal share so the two sources are not a fixed
    # ratio of each other in every race.
    share = min(0.9, max(0.1, rng.gauss(committee_share, 0.15)))
    return LineBias(total_deg=total, committee_deg=total * share, shift_deg=total * (1.0 - share))


def line_bias(pin: tuple[float, float], boat: tuple[float, float], wind_from: float) -> float:
    """Signed bias of a line, in degrees. Positive means the PIN is favoured.

    A line square to the wind runs perpendicular to it, so its pin-to-boat bearing
    is wind_from + 90. The bias is how far the real line departs from that.
    """
    return geo.angle_diff(geo.bearing(pin[0], pin[1], boat[0], boat[1]), wind_from + 90.0)


def favoured_end(pin, boat, wind_from: float) -> str:
    """'pin' | 'boat' — which end is upwind right now."""
    return "pin" if line_bias(pin, boat, wind_from) > 0 else "boat"


def line_side(pin, boat, x: float, y: float, wind_from: float) -> float:
    """Signed distance from the LINE, in metres. POSITIVE is the course side.

    Perpendicular distance from the line itself, with the sign fixed by which side
    is upwind. Every point ON the line reads zero, whatever the bias, which is the
    property the whole start depends on: the line is where the line is, and the
    race committee does not sight it down the wind.

    AN EARLIER VERSION MEASURED FROM THE LINE'S MIDPOINT ALONG THE WIND AXIS, and
    the consequence was subtle and total. On a line biased eight degrees, the pin
    sits nearly ten metres upwind of the midpoint, so a boat sitting exactly at the
    pin read as ten metres OVER. The approach controller dutifully held it back,
    and the favoured end was neutralised for precisely the boats trying to use it.
    Measured across six races, starting at the favoured end was worth exactly
    nothing -- which is how the bug was found, because that result is absurd.
    """
    ax, ay = boat[0] - pin[0], boat[1] - pin[1]
    length = math.hypot(ax, ay)
    if length <= 0.0:
        return 0.0
    # Normal to the line, rotated so it points toward the wind.
    nx, ny = -ay / length, ax / length
    rad = math.radians(wind_from)
    if nx * math.sin(rad) + ny * math.cos(rad) < 0.0:
        nx, ny = -nx, -ny
    return (x - pin[0]) * nx + (y - pin[1]) * ny


def line_fraction(pin, boat, x: float, y: float) -> float:
    """Where along the line a point sits: 0 at the pin, 1 at the committee boat.

    Projected onto the line axis, so it stays meaningful for a boat that is not
    exactly on the line — which at the gun is every boat.
    """
    ax, ay = boat[0] - pin[0], boat[1] - pin[1]
    length_sq = ax * ax + ay * ay
    if length_sq <= 0.0:
        return 0.5
    return ((x - pin[0]) * ax + (y - pin[1]) * ay) / length_sq


@dataclass
class StartPlan:
    """One boat's intentions for the start.

    `timing_error_s` is the human part and it is signed: negative is a crew that
    arrives early and risks being over, positive is one that arrives late and
    starts in a hole. It is drawn per boat per race, because the same crew does not
    make the same mistake every time.
    """

    target_fraction: float
    # Positive means the crew begins its final approach TOO EARLY and risks being
    # over; negative means late, starting in a hole behind the front row.
    approach_error_s: float
    # How far below the line the boat holds station before the final approach.
    station_offset_m: float = 24.0
    # Metres short of the line the boat aims to be at the gun. A boat aiming to be
    # exactly on the line at zero is aiming at the OCS flag; good crews aim to be a
    # length short and accelerating.
    safety_margin_m: float = 3.0

    def target_point(self, pin, boat) -> tuple[float, float]:
        f = self.target_fraction
        return pin[0] + (boat[0] - pin[0]) * f, pin[1] + (boat[1] - pin[1]) * f


def draw_plans(
    boats,
    pin,
    boat_end,
    wind_from: float,
    rng: random.Random,
    crowding: float = 0.65,
) -> dict[int, StartPlan]:
    """Give every boat a place to aim for and a timing error.

    Boats crowd the favoured end, which is the whole reason the favoured end is not
    a free gift: `crowding` pulls the fleet's aim toward it, so the advantage in
    distance is partly paid back in dirty air and bad lanes. That trade is the
    interesting part of the start and it emerges here rather than being asserted —
    the interaction model does the rest.
    """
    favoured = favoured_end(pin, boat_end, wind_from)
    # Fraction 0 is the pin, 1 is the committee boat.
    pull = 0.0 if favoured == "pin" else 1.0
    plans: dict[int, StartPlan] = {}
    for b in boats:
        # Spread across the line, then dragged toward the favoured end.
        base = rng.uniform(0.04, 0.96)
        fraction = base + (pull - base) * crowding * rng.uniform(0.4, 1.0)
        plans[b.boat_id] = StartPlan(
            target_fraction=min(0.98, max(0.02, fraction)),
            # Centred slightly late, with an early tail: most crews leave something
            # in hand and a few get it wrong the expensive way. Deliberately not
            # symmetric — the distribution of start quality in a real fleet is not.
            approach_error_s=rng.gauss(-1.0, 2.4),
            station_offset_m=rng.uniform(18.0, 32.0),
            safety_margin_m=max(0.0, rng.gauss(3.0, 1.5)),
        )
    return plans


def approach_command(
    boat,
    plan: StartPlan,
    pin,
    boat_end,
    tws_kt: float,
    wind_from: float,
    seconds_to_gun: float,
) -> tuple[float, float | None]:
    """(heading to steer, speed cap or None) during the pre-start.

    TWO PHASES, which is what a start actually is.

    HOLDING. Until the final approach the boat sits below the line with no way on,
    pointed near head to wind. Luffing needs no special case: the polar already
    returns zero speed inside the no-go zone, so pointing a boat at the wind stops
    it, exactly as it does on the water.

    ACCELERATING. At a lead time computed from how far below the line the boat is
    and how fast it closes when close-hauled, it bears away to close-hauled and
    goes. The acceleration lag in Boat.step does the rest: a boat that starts this
    too late is still slow at the gun even if it is on the line, which is the real
    penalty for a bad start and the reason it cannot be modelled as position alone.

    The crew's error is in WHEN it starts that approach. `approach_error_s` shifts
    the lead: early risks being over, late means starting in a hole with the fleet
    rolling over the top.

    THE FIRST VERSION OF THIS WAS WRONG in a way worth recording. It regulated
    speed against distance to the target POINT while always steering close-hauled,
    so boats sailed straight through the line and all eighteen were OCS by 45-95
    metres. Distance below the LINE, along the wind axis, is the quantity that
    decides whether a boat is over; distance to a point on it is not.

    Starboard is assumed. A port-tack approach is a specialist move that depends on
    the rules this model does not have yet.
    """
    close_hauled, _ = boat.polar.best_upwind(tws_kt)
    sailing = geo.heading_for_twa(close_hauled, wind_from)
    full_speed = boat.target_speed_kt(tws_kt, wind_from)

    # Positive when the boat is still below the line, which is where it should be.
    below = -line_side(pin, boat_end, boat.x, boat.y, wind_from)

    # Closing rate to windward when close-hauled at full speed.
    closing_ms = geo.ms(full_speed) * math.cos(math.radians(close_hauled))
    ideal_lead = (below - plan.safety_margin_m) / max(closing_ms, 0.05)
    # Compensate for accelerating from a standstill. Without this the lead assumes
    # full closing speed from the instant the boat bears away, so every boat in the
    # fleet arrives systematically late and none is ever over — which is not what a
    # start line looks like. One time constant of acceleration is roughly the
    # distance deficit expressed as time, and crews learn exactly this allowance.
    lead = ideal_lead + boat.accel_tau_s + plan.approach_error_s

    if seconds_to_gun <= max(lead, 0.0):
        # Go. No cap: the whole point is to be at full speed when the gun fires.
        return sailing, None

    # Hold. Head to wind kills the boat's speed through the polar's no-go zone, and
    # a small cap stops it creeping over the line while it slows down.
    return geo.wrap360(wind_from), full_speed * 0.08
