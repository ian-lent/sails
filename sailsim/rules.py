"""Part 2 of the Racing Rules of Sailing: when boats meet.

WHY THIS HAS TO EXIST BEFORE ANY POLICY SEARCH. An optimiser handed a simulator
without rules will find that barging at the committee boat works, that sailing
through a starboard-tacker is free, and that the inside berth at a mark can simply
be taken. It will then report those as strategy. Rules are not decoration on a
racing model; they are most of what makes the model's answers mean anything.

WHAT IS IMPLEMENTED. The right-of-way core that governs the situations this
simulator actually produces, plus the definitions they rest on:

    Rule 10  opposite tacks -- port keeps clear of starboard
    Rule 11  same tack, overlapped -- windward keeps clear of leeward
    Rule 12  same tack, not overlapped -- clear astern keeps clear of clear ahead
    Rule 13  while tacking -- past head to wind, keep clear until close-hauled
    Rule 14  avoid contact
    Rule 18  mark-room, on a three-hull-length zone
    Rule 44  the One-Turn Penalty (one tack and one gybe)

WHAT IS NOT, stated plainly because a half-implemented rulebook that does not say
so is worse than none. Each of these changes real tactical outcomes:

    Rule 15  acquiring right of way -- you must initially give room. Without it a
             boat can snap into a right-of-way position and instantly demand the
             other keep clear, which is not legal and is tactically potent.
    Rule 16  changing course -- a right-of-way boat must give room to keep clear.
             Its absence lets a leeward boat luff violently with no obligation.
    Rule 17  proper course -- a boat that becomes overlapped to leeward from clear
             astern within two lengths may not sail above her proper course. This
             is the rule that constrains the safe-leeward/leebow position, so the
             interaction model's most tactically loaded position is currently
             unconstrained. The most important omission on this list.
    Rule 19  room at an obstruction
    Rule 20  room to tack at an obstruction -- matters on a river with shorelines
    Rule 21  exoneration
    Rule 22  starting errors -- a boat returning to start keeps clear
    Rule 30  starting penalties (I, Z, U, black flag)
    Rule 31  touching a mark
    Rule 42  propulsion -- without it an optimiser may learn to pump

ONE MORE THING THE MODEL DOES NOT HAVE: protests. Here a foul is detected
geometrically and penalised immediately, which is closer to umpired team racing
than to protest-based fleet racing, where many fouls go unpunished and the threat
of a protest is itself tactical. Treat the foul rate here as an upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import geometry as geo
from .boat import Boat

# Rule 18's zone is three hull lengths from the mark, measured for the boat nearer
# to it. Definitions, RRS.
ZONE_LENGTHS = 3.0

# Rule 44.2: a One-Turn Penalty is ONE tack and ONE gybe in the same direction, so
# 360 degrees of turning. Turning through a full circle necessarily crosses head to
# wind once and dead downwind once, so spinning 360 degrees IS a tack and a gybe --
# nothing extra needs modelling to make that true.
#
# One turn rather than two because that is what the racing being modelled uses.
# College fleet racing runs short courses where a 720 is a disproportionate
# penalty, and the sailing instructions reduce rule 44.1 accordingly. An earlier
# version charged 720 degrees, which at a 24 deg/s turn rate cost about 30 seconds
# -- roughly a fifth of a beat, and enough that a single foul decided a race.
#
# The cost is not a fixed number of seconds: it is however long 360 degrees takes
# at the boat's own turn rate, which makes it more expensive in a breeze, exactly
# as it is on the water.
PENALTY_DEGREES = 360.0

# Seconds spent sailing clear of the fleet AFTER completing the turn. Rule 44.1
# requires a boat taking a penalty to get well clear, and without it boats spin in
# the middle of the traffic and set off cascading fouls.
#
# It is ADDITIONAL to the turn, not carved out of it. An earlier version spent the
# tail inside the penalty's own duration, so the boat peeled away partway through
# and never completed its circle -- 240 of the required 360 degrees, and 576 of 720
# back when it was a two-turns penalty. The penalty looked right in the results and
# was never actually sailed. The simulator now counts DEGREES TURNED rather than
# elapsed seconds, because even with the tail moved outside, inferring the turn
# from a clock lost a timestep to rounding and left the boat short of the circle.
PENALTY_CLEAR_S = 4.0


@dataclass(frozen=True)
class Encounter:
    """Who must keep clear of whom, and under which rule."""

    right_of_way: int
    give_way: int
    rule: str

    def obliges(self, boat_id: int) -> bool:
        return self.give_way == boat_id


def _bow(b: Boat) -> tuple[float, float]:
    return geo.step_position(b.x, b.y, b.heading, b.length_m / 2.0, 1.0)


def _stern(b: Boat) -> tuple[float, float]:
    return geo.step_position(b.x, b.y, geo.wrap360(b.heading + 180.0), b.length_m / 2.0, 1.0)


def clear_astern(a: Boat, b: Boat) -> bool:
    """Is `a` clear astern of `b`?

    RRS definition: a is clear astern of b when a's hull is behind a line abeam
    from the aftmost point of b's hull. "Abeam" is perpendicular to b's centreline,
    so the test is a projection onto b's heading — not onto the wind, and not onto
    the line joining them, both of which are tempting and wrong.
    """
    heading = math.radians(b.heading)
    hx, hy = math.sin(heading), math.cos(heading)
    ax, ay = _bow(a)
    sx, sy = _stern(b)
    return (ax - sx) * hx + (ay - sy) * hy < 0.0


def overlapped(a: Boat, b: Boat) -> bool:
    """Two boats overlap when neither is clear astern of the other.

    The RRS definition also makes boats overlap when a third between them overlaps
    both. That transitive case is not implemented, and it matters at crowded mark
    roundings where a three-boat chain confers mark-room that pairwise tests miss.
    """
    return not clear_astern(a, b) and not clear_astern(b, a)


def leeward_boat(a: Boat, b: Boat, wind_from: float) -> Boat:
    """Of two overlapped boats on the same tack, the one to leeward."""
    rad = math.radians(wind_from)
    # Unit vector pointing to leeward, i.e. the way the wind blows.
    lx, ly = -math.sin(rad), -math.cos(rad)
    return b if (b.x - a.x) * lx + (b.y - a.y) * ly > 0.0 else a


def is_tacking(b: Boat, wind_from: float) -> bool:
    """Rule 13: past head to wind and not yet on a close-hauled course.

    Approximated from the boat's manoeuvre state rather than tracked exactly: a
    boat mid-tack has a live manoeuvre and is pointing above close-hauled.
    """
    if b.maneuver_remaining_s <= 0.0:
        return False
    close_hauled, _ = b.polar.best_upwind(8.0)
    return abs(b.twa(wind_from)) < close_hauled


def right_of_way(a: Boat, b: Boat, wind_from: float) -> Encounter | None:
    """Which of two boats must keep clear, and why. None if they cannot interact.

    Precedence follows the rulebook: rule 13 first (a boat tacking keeps clear of
    everyone), then 10 if the tacks differ, then 11 or 12 by overlap.
    """
    a_tacking, b_tacking = is_tacking(a, wind_from), is_tacking(b, wind_from)
    if a_tacking and not b_tacking:
        return Encounter(right_of_way=b.boat_id, give_way=a.boat_id, rule="13")
    if b_tacking and not a_tacking:
        return Encounter(right_of_way=a.boat_id, give_way=b.boat_id, rule="13")
    if a_tacking and b_tacking:
        return None  # Both tacking: rule 13's second sentence, not modelled.

    a_tack, b_tack = a.tack(wind_from), b.tack(wind_from)
    if a_tack != b_tack:
        # Rule 10. Port keeps clear of starboard, whatever their overlap.
        port, starboard = (a, b) if a_tack == "port" else (b, a)
        return Encounter(right_of_way=starboard.boat_id, give_way=port.boat_id, rule="10")

    if overlapped(a, b):
        # Rule 11. Windward keeps clear of leeward.
        lee = leeward_boat(a, b, wind_from)
        wind_ward = b if lee is a else a
        return Encounter(right_of_way=lee.boat_id, give_way=wind_ward.boat_id, rule="11")

    # Rule 12. Clear astern keeps clear of clear ahead.
    astern, ahead = (a, b) if clear_astern(a, b) else (b, a)
    return Encounter(right_of_way=ahead.boat_id, give_way=astern.boat_id, rule="12")


def in_zone(b: Boat, mark_x: float, mark_y: float) -> bool:
    """Rule 18's zone: within three hull lengths of the mark."""
    return geo.distance(b.x, b.y, mark_x, mark_y) <= ZONE_LENGTHS * b.length_m


def mark_room(a: Boat, b: Boat, mark_x: float, mark_y: float, wind_from: float) -> Encounter | None:
    """Rule 18.2(b): who is entitled to mark-room, if anyone.

    Entitlement is decided by the overlap AT THE MOMENT the first boat reaches the
    zone, and it then persists even if the overlap is broken. This implementation
    tests the overlap continuously instead, which is the significant simplification
    here: a boat that breaks an overlap inside the zone keeps its entitlement in
    the rulebook and loses it here.

    Returns an encounter obliging the OUTSIDE boat to give room to the INSIDE one.
    Note this can reverse the right of way from rule 11 or 12 — which is the whole
    point of rule 18, and why a windward boat inside at the mark is entitled to
    room it would not otherwise get.
    """
    if not (in_zone(a, mark_x, mark_y) or in_zone(b, mark_x, mark_y)):
        return None
    if not overlapped(a, b):
        return None

    # Inside is the boat nearer the mark measured across the course, which for a
    # rounding is simply the one closer to it.
    a_dist = geo.distance(a.x, a.y, mark_x, mark_y)
    b_dist = geo.distance(b.x, b.y, mark_x, mark_y)
    inside, outside = (a, b) if a_dist < b_dist else (b, a)
    return Encounter(right_of_way=inside.boat_id, give_way=outside.boat_id, rule="18")


def hulls_touching(a: Boat, b: Boat) -> bool:
    """Rule 14: is there contact?

    Boats are treated as segments from bow to stern and contact is the segments
    coming within the boats' mean beam. Crude — a real hull is not a stick and a
    dinghy's boom reaches well outside it — but it is the right shape of test, and
    far better than the circles that would let boats interpenetrate bow to stern.
    """
    separation = _segment_distance(_bow(a), _stern(a), _bow(b), _stern(b))
    return separation < (a.beam_m + b.beam_m) / 2.0


def _segment_distance(p1, p2, q1, q2) -> float:
    """Shortest distance between two line segments."""

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    vx, vy = q2[0] - q1[0], q2[1] - q1[1]
    wx, wy = p1[0] - q1[0], p1[1] - q1[1]
    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    d = ux * wx + uy * wy
    e = vx * wx + vy * wy
    denom = a * c - b * b
    if denom < 1e-9:
        s, t = 0.0, clamp(e / c if c > 1e-9 else 0.0)
    else:
        s = clamp((b * e - c * d) / denom)
        t = clamp((a * e - b * d) / denom)
    cx = p1[0] + ux * s - (q1[0] + vx * t)
    cy = p1[1] + uy * s - (q1[1] + vy * t)
    return math.hypot(cx, cy)


# How close a projected pass counts as failing to keep clear, in mean hull lengths,
# and how far ahead to look. The RRS test is whether the right-of-way boat would
# have to take avoiding action, which is a question about her needing to change
# course at all -- not about imminent contact. An early version used half a hull
# length and an eight-second horizon, so boats sailed to within four metres before
# reacting and the fleet racked up fifty fouls a race. Sailors leave more room
# than that, and leave it earlier.
KEEP_CLEAR_LENGTHS = 1.8
KEEP_CLEAR_HORIZON_S = 14.0


def keeping_clear(give_way: Boat, row: Boat, dt: float, horizon_s: float = KEEP_CLEAR_HORIZON_S) -> bool:
    """Is the give-way boat keeping clear?

    The RRS definition is that the right-of-way boat can sail her course with no
    need to take avoiding action. Approximated here by projecting both boats
    forward on their current headings and asking whether they would touch. That is
    a fair reading for the converging cases this simulator produces and a poor one
    for the subtle ones, where keeping clear is a question of whether the other
    boat would have to change course *at all*, not whether contact is imminent.
    """
    steps = max(1, int(horizon_s / dt))
    gx, gy = give_way.x, give_way.y
    rx, ry = row.x, row.y
    g_speed = geo.ms(give_way.speed_kt) * dt
    r_speed = geo.ms(row.speed_kt) * dt
    for _ in range(steps):
        gx, gy = geo.step_position(gx, gy, give_way.heading, g_speed, 1.0)
        rx, ry = geo.step_position(rx, ry, row.heading, r_speed, 1.0)
        margin = KEEP_CLEAR_LENGTHS * (give_way.length_m + row.length_m) / 2.0
        if geo.distance(gx, gy, rx, ry) < margin:
            return False
    return True


def avoidance(give_way: Boat, row: Boat, rule: str, wind_from: float) -> tuple[float | None, float | None]:
    """(heading, speed cap) for a give-way boat taking avoiding action.

    One manoeuvre per rule, chosen to be what a sailor would actually do:

      * Rule 10 — DUCK. Bear away and aim astern of the starboard boat. Tacking is
        the other legal answer and often the better one, but it is a tactical
        choice rather than a keeping-clear obligation, so the policy layer should
        own it, not this function.
      * Rule 11 — the windward boat heads up to open the gap, and slows if it is
        already as high as it can sail.
      * Rules 12 and 18 — the boat astern, or the outside boat at a mark, slows.
        Crude for rule 18: the outside boat should be sailing WIDE, and a speed cap
        is a stand-in for room that is about timing rather than space.
    """
    if rule == "10":
        # Aim two lengths astern of the right-of-way boat.
        target = geo.step_position(
            row.x, row.y, geo.wrap360(row.heading + 180.0), 2.0 * row.length_m, 1.0
        )
        return geo.bearing(give_way.x, give_way.y, target[0], target[1]), None

    if rule == "11":
        twa = give_way.twa(wind_from)
        close_hauled, _ = give_way.polar.best_upwind(8.0)
        if abs(twa) <= close_hauled + 1.0:
            # Already pointing as high as it usefully can, so the only way to keep
            # clear is to drop back -- and it has to be a real drop. A gentle trim
            # off the top leaves the boats converging for a long time, which is how
            # a windward boat ends up fouling something it was nominally avoiding.
            return None, give_way.target_speed_kt(8.0, wind_from) * 0.45
        higher = math.copysign(max(abs(twa) - 12.0, close_hauled), twa)
        return geo.heading_for_twa(higher, wind_from), None

    # Rules 12 and 18.
    return None, give_way.speed_kt * 0.55


# Rule 14 binds the right-of-way boat too: she "shall avoid contact ... however
# she need not act to avoid contact until it is clear that the other boat is not
# keeping clear". This is how close a projected pass has to get before that
# becomes clear, in mean hull lengths -- tighter than KEEP_CLEAR_LENGTHS, because
# the right-of-way boat is entitled to sail her course right up until the point
# where holding it would mean hitting someone.
RULE_14_LENGTHS = 0.9
RULE_14_HORIZON_S = 6.0


def contact_imminent(a: Boat, b: Boat, dt: float) -> bool:
    """Would these two touch shortly if both held course? Rule 14's trigger."""
    steps = max(1, int(RULE_14_HORIZON_S / dt))
    ax, ay, bx, by = a.x, a.y, b.x, b.y
    a_step = geo.ms(a.speed_kt) * dt
    b_step = geo.ms(b.speed_kt) * dt
    margin = RULE_14_LENGTHS * (a.length_m + b.length_m) / 2.0
    for _ in range(steps):
        ax, ay = geo.step_position(ax, ay, a.heading, a_step, 1.0)
        bx, by = geo.step_position(bx, by, b.heading, b_step, 1.0)
        if geo.distance(ax, ay, bx, by) < margin:
            return True
    return False


def separation_heading(give_way: Boat, row: Boat) -> float:
    """Steer directly away from a boat already touching.

    Avoidance above is predictive, and predictions are no use once contact has
    happened. Without this the two hulls stay interpenetrated: they foul, the
    give-way boat spins a penalty on the spot, finishes it still touching, and
    fouls again. A race produced hundreds of penalties and boats ended up a tenth
    of a metre apart, welded together.

    This is not a physical collision response — boats have no mass here and do not
    bounce. It is the give-way boat doing the only thing left, which is to get out.
    """
    return geo.bearing(row.x, row.y, give_way.x, give_way.y)


def penalty_turn_seconds(b: Boat) -> float:
    """Time to turn the full 360 degrees, from this boat's own turn rate."""
    return PENALTY_DEGREES / max(b.max_turn_rate_deg_s, 1.0)


def penalty_seconds(b: Boat) -> float:
    """Total cost of a One-Turn Penalty: the circle, then getting clear."""
    return penalty_turn_seconds(b) + PENALTY_CLEAR_S
