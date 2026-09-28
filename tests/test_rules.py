"""Checks over Part 2: the right-of-way determinations and what follows from them.

The determinations are exactly checkable — the RRS says who keeps clear in a given
geometry and there is a right answer — so most of this file pins those. The
emergent section then asks the only question that matters afterwards: does a fleet
sailing under these rules behave differently from one sailing without them?

Run: python3 tests/test_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim import geometry as geo  # noqa: E402
from sailsim import rules as R  # noqa: E402
from sailsim.boat import Boat  # noqa: E402
from sailsim.course import Course  # noqa: E402
from sailsim.polar import C420  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.wind import UniformWind  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  \033[32m+\033[0m {name}")
    else:
        print(f"  \033[31mx\033[0m {name}" + (f"\n      {detail}" if detail else ""))
        FAILURES.append(name)


def group(name: str) -> None:
    print(f"\n\033[1m== {name} ==\033[0m")


WIND, TWS = 0.0, 8.0
CLOSE_HAULED, _ = C420.best_upwind(TWS)
L = C420.boat_length_m


def boat_on(tack: int, x: float, y: float, boat_id: int) -> Boat:
    b = Boat(boat_id, f"b{boat_id}", C420, x=x, y=y)
    b.heading = geo.heading_for_twa(tack * CLOSE_HAULED, WIND)
    b.speed_kt = b.target_speed_kt(TWS, WIND)
    return b


def offset(b: Boat, relative_bearing: float, metres: float) -> tuple[float, float]:
    """A point at a bearing relative to a boat's own heading."""
    return geo.step_position(b.x, b.y, geo.wrap360(b.heading + relative_bearing), metres, 1.0)


# --- definitions -------------------------------------------------------------
group("definitions: clear astern and overlap")

lead = boat_on(+1, 0.0, 0.0, 0)
astern = boat_on(+1, *offset(lead, 180.0, 12.0), boat_id=1)
check("a boat behind is clear astern", R.clear_astern(astern, lead))
check("and the boat ahead is not clear astern of it", not R.clear_astern(lead, astern))
check("they are therefore not overlapped", not R.overlapped(lead, astern))

abeam = boat_on(+1, *offset(lead, 90.0, 10.0), boat_id=2)
check("a boat abeam is overlapped", R.overlapped(lead, abeam))
check("overlap is symmetric", R.overlapped(abeam, lead) == R.overlapped(lead, abeam))

# The line is abeam of the OTHER boat's centreline, not perpendicular to the line
# joining them and not square to the wind. A boat displaced sideways but level is
# overlapped even though it is a long way away.
far_abeam = boat_on(+1, *offset(lead, 90.0, 60.0), boat_id=3)
check("overlap does not depend on how far apart they are", R.overlapped(lead, far_abeam))

# --- rule 10 -----------------------------------------------------------------
group("rule 10: opposite tacks")

stbd = boat_on(+1, 0.0, 0.0, 0)
port = boat_on(-1, 20.0, -20.0, 1)
enc = R.right_of_way(stbd, port, WIND)
check("an encounter is found", enc is not None)
check("port keeps clear of starboard", enc.give_way == port.boat_id, f"rule {enc.rule}")
check("and it is rule 10", enc.rule == "10")
check(
    "the answer does not depend on argument order",
    R.right_of_way(port, stbd, WIND).give_way == port.boat_id,
)

# Rule 10 does not care about overlap: opposite tacks settles it outright.
port_overlapped = boat_on(-1, *offset(stbd, 90.0, 8.0), boat_id=2)
enc_o = R.right_of_way(stbd, port_overlapped, WIND)
check(
    "rule 10 takes precedence over overlap",
    enc_o.rule == "10" and enc_o.give_way == port_overlapped.boat_id,
    f"got rule {enc_o.rule}",
)

# --- rule 11 -----------------------------------------------------------------
group("rule 11: same tack, overlapped")

windward = boat_on(+1, 0.0, 0.0, 0)
# To leeward on starboard is to port of the heading, which is down-wind of it.
leeward = boat_on(+1, *offset(windward, -90.0, 14.0), boat_id=1)
check("they are overlapped", R.overlapped(windward, leeward))
check(
    "the boat to leeward is identified as leeward",
    R.leeward_boat(windward, leeward, WIND) is leeward,
)
enc = R.right_of_way(windward, leeward, WIND)
check("windward keeps clear of leeward", enc.give_way == windward.boat_id, f"rule {enc.rule}")
check("and it is rule 11", enc.rule == "11")

mirror_w = boat_on(-1, 0.0, 0.0, 0)
mirror_l = boat_on(-1, *offset(mirror_w, 90.0, 14.0), boat_id=1)
check(
    "port tack mirrors it",
    R.right_of_way(mirror_w, mirror_l, WIND).give_way == mirror_w.boat_id,
)

# --- rule 12 -----------------------------------------------------------------
group("rule 12: same tack, not overlapped")

ahead = boat_on(+1, 0.0, 0.0, 0)
behind = boat_on(+1, *offset(ahead, 180.0, 15.0), boat_id=1)
enc = R.right_of_way(ahead, behind, WIND)
check("clear astern keeps clear of clear ahead", enc.give_way == behind.boat_id)
check("and it is rule 12", enc.rule == "12", f"got {enc.rule}")

# --- rule 13 -----------------------------------------------------------------
group("rule 13: while tacking")

tacker = boat_on(+1, 0.0, 0.0, 0)
tacker.heading = geo.heading_for_twa(10.0, WIND)  # past head to wind, not yet close-hauled
tacker.begin_maneuver(TWS, "tack")
holder = boat_on(+1, *offset(tacker, -90.0, 14.0), boat_id=1)
check("a boat past head to wind is tacking", R.is_tacking(tacker, WIND))
check("a boat sailing close-hauled is not", not R.is_tacking(holder, WIND))
enc = R.right_of_way(tacker, holder, WIND)
check("the tacking boat keeps clear", enc.give_way == tacker.boat_id, f"rule {enc.rule}")
check("and it is rule 13", enc.rule == "13")
check(
    "rule 13 outranks rule 11 (the tacker is to windward here would not matter)",
    enc.rule == "13",
)

# --- marks -------------------------------------------------------------------
group("rule 18: mark-room")

mark_x, mark_y = 0.0, 0.0
close = boat_on(+1, 0.0, -2.5 * L, 0)
far = boat_on(+1, 0.0, -20.0 * L, 1)
check("a boat within three lengths is in the zone", R.in_zone(close, mark_x, mark_y))
check("a boat well outside is not", not R.in_zone(far, mark_x, mark_y))
check(
    "the zone is exactly three hull lengths",
    R.in_zone(boat_on(+1, 0.0, -2.99 * L, 2), mark_x, mark_y)
    and not R.in_zone(boat_on(+1, 0.0, -3.01 * L, 3), mark_x, mark_y),
)

inside = boat_on(+1, 3.0, -1.5 * L, 0)
outside = boat_on(+1, *offset(inside, 90.0, 9.0), boat_id=1)
room = R.mark_room(inside, outside, mark_x, mark_y, WIND)
check("mark-room is owed in the zone when overlapped", room is not None)
if room is not None:
    nearer = inside if geo.distance(inside.x, inside.y, mark_x, mark_y) < geo.distance(
        outside.x, outside.y, mark_x, mark_y) else outside
    check("the inside boat is entitled to it", room.right_of_way == nearer.boat_id)
    check("and it is rule 18", room.rule == "18")

check(
    "no mark-room outside the zone",
    R.mark_room(far, boat_on(+1, 9.0, -20.0 * L, 4), mark_x, mark_y, WIND) is None,
)

# --- contact and penalty -----------------------------------------------------
group("rule 14 and rule 44")

a = boat_on(+1, 0.0, 0.0, 0)
b = boat_on(+1, 0.5, 0.5, 1)
check("overlapping hulls are touching", R.hulls_touching(a, b))
check("boats a few lengths apart are not", not R.hulls_touching(a, boat_on(+1, 0.0, 4 * L, 2)))

# An `or True` slipped into this assertion on the first draft, which made it pass
# unconditionally. A vacuous check is worse than no check: it reports green forever.
abeam_clear = boat_on(+1, *offset(a, 90.0, 3.0), boat_id=3)
check(
    "boats abeam by more than a beam's width are NOT touching",
    not R.hulls_touching(a, abeam_clear),
    f"separation 3.0 m, mean beam {C420.beam_m:.2f} m",
)
abeam_touching = boat_on(+1, *offset(a, 90.0, 1.0), boat_id=4)
check("boats abeam by less than a beam's width ARE touching",
      R.hulls_touching(a, abeam_touching))

slow_turner = boat_on(+1, 0.0, 0.0, 0)
slow_turner.max_turn_rate_deg_s = 15.0
fast_turner = boat_on(+1, 0.0, 0.0, 1)
fast_turner.max_turn_rate_deg_s = 30.0
check(
    "a two-turns penalty is 720 degrees, so it costs more when turning is slower",
    R.penalty_seconds(slow_turner) > R.penalty_seconds(fast_turner),
    f"{R.penalty_seconds(slow_turner):.0f}s vs {R.penalty_seconds(fast_turner):.0f}s",
)
print(f"      720 deg at 15 deg/s = {R.penalty_seconds(slow_turner):.0f}s, "
      f"at 30 deg/s = {R.penalty_seconds(fast_turner):.0f}s")

# --- avoidance ---------------------------------------------------------------
group("avoiding action")

stbd = boat_on(+1, 0.0, 0.0, 0)
port = boat_on(-1, 25.0, -25.0, 1)
heading, cap = R.avoidance(port, stbd, "10", WIND)
check("a port-tack boat given rule 10 alters course", heading is not None)
if heading is not None:
    # Ducking means bearing away: the new heading is further from the wind.
    before = abs(port.twa(WIND))
    after = abs(geo.wrap180(WIND - heading))
    check("it bears away to duck rather than pinching up", after > before,
          f"TWA {before:.0f} -> {after:.0f}")

windward = boat_on(+1, 0.0, 0.0, 0)
leeward = boat_on(+1, *offset(windward, -90.0, 10.0), boat_id=1)
_, cap11 = R.avoidance(windward, leeward, "11", WIND)
check("a windward boat already close-hauled keeps clear by slowing", cap11 is not None)

_, cap12 = R.avoidance(behind, ahead, "12", WIND)
check("a boat clear astern keeps clear by slowing", cap12 is not None)

check(
    "a boat already in contact is steered directly away",
    abs(
        geo.angle_diff(
            R.separation_heading(a, b), geo.bearing(b.x, b.y, a.x, a.y)
        )
    )
    < 1e-9,
)

# --- emergent ----------------------------------------------------------------
group("emergent: a fleet racing under the rules")

course = Course.for_conditions(TWS, C420, wind_from=WIND, line_bias_deg=8.0)

free_fleet = build_fleet(size=18, course=course, seed=3)
free = Simulator(course, UniformWind(TWS, WIND), start_seed=3, rules=False).run(free_fleet)
ruled_fleet = build_fleet(size=18, course=course, seed=3)
ruled = Simulator(course, UniformWind(TWS, WIND), start_seed=3, rules=True).run(ruled_fleet)

check("with rules off, nothing is enforced", free.fouls == 0 and free.contacts == 0)
check("with rules off the run says so, loudly",
      any("RULES ARE OFF" in note for note in free.estimates_used))
check("with rules on, boats spend time keeping clear",
      sum(b.gave_way_s for b in ruled_fleet) > 0.0)
check("the fleet still completes the course under the rules", ruled.finishers() == 18,
      f"{ruled.finishers()}/18")
print(f"      keeping clear: {sum(b.gave_way_s for b in ruled_fleet):.0f} boat-seconds; "
      f"{ruled.fouls} fouls, {ruled.contacts} contacts")
check("penalised boats actually sail their turns",
      sum(b.penalties_taken for b in ruled_fleet) > 0 or ruled.fouls == 0)
check(
    "fouling costs places",
    (lambda fouled, clean: (sum(fouled) / len(fouled) > sum(clean) / len(clean))
     if fouled and clean else True)(
        [i for i, (bid, _, _) in enumerate(ruled.order)
         if next(b for b in ruled_fleet if b.boat_id == bid).fouls > 0],
        [i for i, (bid, _, _) in enumerate(ruled.order)
         if next(b for b in ruled_fleet if b.boat_id == bid).fouls == 0],
    ),
)

# Rules must not turn the race into a procession of identical times, nor gridlock.
times = [t for _, _, t in ruled.order if t is not None]
check("the fleet is still spread out", max(times) - min(times) > 20.0,
      f"spread {max(times) - min(times):.0f}s")

print()
if FAILURES:
    print(f"\033[31m{len(FAILURES)} check(s) failed:\033[0m " + ", ".join(FAILURES))
    raise SystemExit(1)
print("\033[32mall checks passed\033[0m")
