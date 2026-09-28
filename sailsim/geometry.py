"""Angles and bearings, defined once.

This module exists because angle conventions are where sailing simulators rot
quietly. A sign error in true wind angle does not crash: the fleet just sails
slightly wrong for the rest of the project, and every strategic conclusion drawn
from it is wrong in a way no plot makes obvious. So the conventions are stated
here, used everywhere, and tested.

THE CONVENTIONS, all degrees:

  * Bearings are compass bearings. 0 = north, 90 = east, clockwise, [0, 360).
  * Wind direction is the direction the wind blows FROM, as every forecast and
    every sailor states it. "Wind 310" means a northwesterly. The vector the air
    actually travels along is the reciprocal, and the only place that matters is
    inside advection code, which must convert explicitly.
  * Position is metres in a local ENU frame: +x east, +y north. The Severn is
    small enough that a flat local frame costs nothing; converting to lat/lon is
    a presentation concern, not a simulation one.
  * TWA (true wind angle) is SIGNED, in (-180, 180]:
        twa = wrap180(wind_from - heading)
    Positive TWA means the wind comes over the starboard side, so the boat is on
    STARBOARD tack. Negative is port. Keeping the sign is what lets one number
    carry both the aerodynamic angle and the right-of-way state, instead of
    tracking a separate boolean that can disagree with the geometry.

Worked example, to pin the sign down: heading 000 (due north), wind from 045
(northeast). twa = +45. The wind is on the boat's starboard bow. Starboard tack.
"""

from __future__ import annotations

import math

# Both boats in scope are small dinghies; lengths are used to express distances
# in the unit sailors actually think in.
BOAT_LENGTH_M = {"fj": 4.03, "c420": 4.20}


def wrap360(angle: float) -> float:
    """Normalise to [0, 360)."""
    return angle % 360.0


def wrap180(angle: float) -> float:
    """Normalise to (-180, 180].

    The half-open direction matters: 180 must stay 180 (dead downwind on
    starboard by convention) rather than flipping to -180 and reading as port.
    """
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def angle_diff(a: float, b: float) -> float:
    """Signed smallest rotation from b to a, in (-180, 180]."""
    return wrap180(a - b)


def true_wind_angle(heading: float, wind_from: float) -> float:
    """Signed TWA. Positive = wind over starboard side = starboard tack."""
    return wrap180(wind_from - heading)


def heading_for_twa(twa: float, wind_from: float) -> float:
    """The heading that yields this signed TWA in this wind. Inverse of above."""
    return wrap360(wind_from - twa)


def tack_of(twa: float) -> str:
    """'starboard' | 'port' from signed TWA.

    Dead downwind (|twa| == 180) is called starboard by the convention in
    wrap180. Real boats settle this by which side the boom is on, which this
    model does not track; when we add rule 10 for run-in situations this needs
    revisiting rather than trusting the convention.
    """
    return "starboard" if twa >= 0.0 else "port"


def bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compass bearing from point 1 to point 2."""
    return wrap360(math.degrees(math.atan2(x2 - x1, y2 - y1)))


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in metres."""
    return math.hypot(x2 - x1, y2 - y1)


def step_position(x: float, y: float, heading: float, speed_ms: float, dt: float) -> tuple[float, float]:
    """Advance a position along a heading. Speed in m/s, dt in seconds."""
    rad = math.radians(heading)
    return x + speed_ms * math.sin(rad) * dt, y + speed_ms * math.cos(rad) * dt


def turn_toward(current: float, target: float, max_rate_deg_s: float, dt: float) -> float:
    """Rotate `current` toward `target`, capped at a turn rate.

    A rate cap rather than instant rotation is what makes a tack take time and
    therefore cost distance. Without it, direction changes are free and the whole
    question of when to tack becomes trivial and wrong.
    """
    delta = angle_diff(target, current)
    limit = max_rate_deg_s * dt
    if abs(delta) <= limit:
        return wrap360(target)
    return wrap360(current + math.copysign(limit, delta))


KNOTS_TO_MS = 0.514444


def knots(ms: float) -> float:
    return ms / KNOTS_TO_MS


def ms(knots_value: float) -> float:
    return knots_value * KNOTS_TO_MS


def apparent_wind(tws_kt: float, twa_deg: float, boat_speed_kt: float) -> tuple[float, float]:
    """(apparent wind speed, signed apparent wind angle) from true wind and boat speed.

    THIS IS THE FUNCTION THE WHOLE INTERACTION MODEL TURNS ON, because a boat's
    wind shadow lies downwind in the APPARENT wind, not the true wind, and the two
    differ most exactly where racing is densest — upwind.

    Beating in 8 knots at 41 degrees true, a dinghy doing 3 knots sees the wind
    about 30 degrees off the bow: eleven degrees further forward than the true
    angle. So the shadow is rotated forward by eleven degrees too. That is not a
    detail. It is the difference between "the bad air is directly behind the boat
    ahead" and the truth, which is that the bad air lies behind and to LEEWARD, and
    is why the escape from it is to sail higher or tack rather than to bear away.

    Sign convention follows true_wind_angle: positive is wind over the starboard
    side. The apparent angle always has the same sign as the true angle and a
    smaller magnitude, because the boat's own motion adds a headwind.
    """
    rad = math.radians(twa_deg)
    # Longitudinal component points at the bow; the boat's motion adds to it.
    along = tws_kt * math.cos(rad) + boat_speed_kt
    across = tws_kt * math.sin(rad)
    return math.hypot(along, across), math.degrees(math.atan2(across, along))


def apparent_wind_from(heading: float, wind_from: float, boat_speed_kt: float, tws_kt: float) -> float:
    """Compass bearing the APPARENT wind blows from, for a boat in this state."""
    twa = true_wind_angle(heading, wind_from)
    _, awa = apparent_wind(tws_kt, twa, boat_speed_kt)
    return wrap360(heading + awa)
