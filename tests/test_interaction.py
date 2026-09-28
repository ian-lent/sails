"""Checks over wind shadow, backwind and lanes.

The geometry here is the part most likely to be subtly wrong and least likely to
look wrong, so these tests assert tactical facts a sailor would recognise rather
than just exercising the code. If the shadow ends up on the wrong side of the
boat, every conclusion this simulator ever produces about passing, starting and
covering will be confidently inverted, and nothing in a plot would give it away.

Run: python3 tests/test_interaction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim import geometry as geo  # noqa: E402
from sailsim.boat import Boat  # noqa: E402
from sailsim.course import Course, Helm  # noqa: E402
from sailsim.interaction import Disturbance, FleetWind  # noqa: E402
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


WIND_FROM = 0.0
TWS = 8.0
D = Disturbance()
BL = C420.boat_length_m


def beating(tack: int, x: float = 0.0, y: float = 0.0) -> Boat:
    """A boat close-hauled on the given tack at full speed, at a position."""
    close_hauled, _ = C420.best_upwind(TWS)
    b = Boat(0 if tack > 0 else 1, "stbd" if tack > 0 else "port", C420, x=x, y=y)
    b.heading = geo.heading_for_twa(tack * close_hauled, WIND_FROM)
    b.speed_kt = b.target_speed_kt(TWS, WIND_FROM)
    return b


def place(source: Boat, bearing_from_source: float, boat_lengths: float) -> tuple[float, float]:
    """A point at a bearing and distance from the source, in boat lengths."""
    return geo.step_position(source.x, source.y, bearing_from_source, boat_lengths * BL, 1.0)


# --- the shadow lies along apparent wind ------------------------------------
group("shadow axis")

src = beating(+1)  # starboard tack, wind from north
axis = D.shadow_axis(src, TWS, WIND_FROM)
true_downwind = geo.wrap360(WIND_FROM + 180.0)

check(
    "shadow does not lie along the TRUE downwind direction",
    abs(geo.angle_diff(axis, true_downwind)) > 5.0,
    f"axis {axis:.1f}, true downwind {true_downwind:.1f}",
)

# The rotation should equal the difference between apparent and true wind angle.
_, awa = geo.apparent_wind(TWS, src.twa(WIND_FROM), src.speed_kt)
expected_shift = src.twa(WIND_FROM) - awa
check(
    "shadow is rotated forward by exactly the apparent-wind shift",
    abs(abs(geo.angle_diff(axis, true_downwind)) - abs(expected_shift)) < 0.5,
    f"rotation {geo.angle_diff(axis, true_downwind):.1f}, apparent shift {expected_shift:.1f}",
)

# On starboard the wind crosses from starboard to port, so the shadow trails aft
# and to LEEWARD (port side). Relative bearing magnitude > 90 is aft.
relative = geo.angle_diff(axis, src.heading)
check(
    "shadow trails aft of the boat casting it",
    abs(relative) > 90.0,
    f"{relative:.1f} deg off the bow",
)
check(
    "on starboard tack the shadow trails to LEEWARD (port side)",
    relative < 0.0,
    f"{relative:.1f} deg off the bow; negative is to port",
)

mirror = beating(-1)
check(
    "port tack mirrors it exactly",
    abs(
        geo.angle_diff(D.shadow_axis(mirror, TWS, WIND_FROM), mirror.heading)
        + relative
    )
    < 1e-6,
)

# --- who is actually in it --------------------------------------------------
group("who sits in the bad air")

src = beating(+1)
in_shadow = place(src, axis, 3.0)
deficit_behind = D.shadow_at(src, in_shadow[0], in_shadow[1], TWS, WIND_FROM)
check("a boat three lengths down the shadow axis is slowed", deficit_behind > 0.1,
      f"deficit {deficit_behind:.3f}")

# Directly astern on the same track: only clipped, because the axis is rotated.
astern = place(src, geo.wrap360(src.heading + 180.0), 3.0)
deficit_astern = D.shadow_at(src, astern[0], astern[1], TWS, WIND_FROM)
check(
    "a boat directly astern is hurt LESS than one down the shadow axis",
    deficit_astern < deficit_behind,
    f"astern {deficit_astern:.3f} vs axis {deficit_behind:.3f}",
)

# The windward lane is the escape, and this is the single most tactically loaded
# assertion in the file: to windward and astern is clear air.
#
# On starboard tack the windward side is to STARBOARD, which is a POSITIVE relative
# bearing, so windward-astern is heading + 180 - 35. Writing + 35 puts the point to
# leeward instead — squarely in the shadow — which is exactly the sign error this
# whole file exists to catch, and it caught it in its own first draft.
windward_astern = place(src, geo.wrap360(src.heading + 180.0 - 35.0), 3.0)
check(
    "to windward and astern is CLEAR AIR (the windward lane)",
    D.shadow_at(src, windward_astern[0], windward_astern[1], TWS, WIND_FROM) == 0.0,
)
leeward_astern = place(src, geo.wrap360(src.heading + 180.0 + 35.0), 3.0)
check(
    "to leeward and astern is the bad place to be",
    D.shadow_at(src, leeward_astern[0], leeward_astern[1], TWS, WIND_FROM) > 0.1,
    f"deficit {D.shadow_at(src, leeward_astern[0], leeward_astern[1], TWS, WIND_FROM):.3f}",
)

# A FALSIFIABLE PREDICTION, recorded so it gets checked rather than assumed.
# Because the shed wake advects along the apparent wind, a boat sitting exactly in
# another's wake on the same tack is to WINDWARD of the bad air, and this model says
# it is in clear air at three lengths. That is a real consequence of the physics
# (the locus of shed parcels is the apparent wind direction, not the true one), but
# it is stronger than the usual advice that "directly behind is bad", and the cone
# half-angle is estimated. GPS tracks would settle it: the observable is whether a
# boat in another's exact wake actually slows.
astern_deficit = D.shadow_at(src, astern[0], astern[1], TWS, WIND_FROM)
print(f"      model predicts: directly astern at 3 BL -> deficit {astern_deficit:.3f} "
      f"(offset {abs(geo.angle_diff(geo.bearing(src.x, src.y, *astern), axis)):.1f} deg "
      f"vs {D.shadow_half_angle_deg:.0f} deg half-angle)")

ahead = place(src, src.heading, 3.0)
check(
    "a boat ahead is never in the shadow of the boat behind",
    D.shadow_at(src, ahead[0], ahead[1], TWS, WIND_FROM) == 0.0,
)

# --- decay ------------------------------------------------------------------
group("decay with distance")

near = place(src, axis, 1.0)
mid = place(src, axis, 4.0)
far = place(src, axis, 6.5)
beyond = place(src, axis, 9.0)
d_near = D.shadow_at(src, near[0], near[1], TWS, WIND_FROM)
d_mid = D.shadow_at(src, mid[0], mid[1], TWS, WIND_FROM)
d_far = D.shadow_at(src, far[0], far[1], TWS, WIND_FROM)
print(f"      deficit at 1 / 4 / 6.5 BL: {d_near:.3f} / {d_mid:.3f} / {d_far:.3f}")
check("deficit decays with distance", d_near > d_mid > d_far > 0.0)
check(
    "beyond the shadow length there is no effect",
    D.shadow_at(src, beyond[0], beyond[1], TWS, WIND_FROM) == 0.0,
)
check("a boat never shadows itself", D.shadow_at(src, src.x, src.y, TWS, WIND_FROM) == 0.0)
check(
    "deficit never exceeds the configured maximum",
    d_near <= D.shadow_max_deficit + 1e-9,
)

# --- backwind / the lee bow -------------------------------------------------
group("backwind (lee-bow)")

lee = beating(+1)  # the leeward boat, on starboard
# The windward boat sits to windward and aft of it. On starboard, windward is to
# the boat's starboard side, which is a positive relative bearing.
wind_side = place(lee, geo.wrap360(lee.heading + 180.0 - 40.0), 2.0)
bend = D.backwind_at(lee, wind_side[0], wind_side[1], TWS, WIND_FROM)
check("a boat to windward and aft is bent by the leeward boat", bend != 0.0, f"bend {bend:.2f}")

# A header on starboard means the wind comes further forward, i.e. wind_from falls.
check(
    "the bend HEADS a boat on the same tack (starboard: wind_from decreases)",
    bend < 0.0,
    f"bend {bend:+.2f} deg",
)

lee_side = place(lee, geo.wrap360(lee.heading + 180.0 + 40.0), 2.0)
check(
    "a boat to leeward and aft is not backwinded (it is in the shadow instead)",
    D.backwind_at(lee, lee_side[0], lee_side[1], TWS, WIND_FROM) == 0.0,
)
check(
    "a boat ahead is not backwinded",
    D.backwind_at(lee, *place(lee, lee.heading, 2.0), TWS, WIND_FROM) == 0.0,
)

port_lee = beating(-1)
port_wind_side = place(port_lee, geo.wrap360(port_lee.heading + 180.0 + 40.0), 2.0)
check(
    "on port tack the bend reverses sign",
    D.backwind_at(port_lee, port_wind_side[0], port_wind_side[1], TWS, WIND_FROM) > 0.0,
)

# --- composition and order independence -------------------------------------
group("composition")

a = beating(+1, 0.0, 0.0)
a.boat_id = 0
b = beating(+1, *place(a, axis, 2.0))
b.boat_id = 1
victim = beating(+1, *place(a, axis, 3.5))
victim.boat_id = 2

wind = UniformWind(speed_kt=TWS, direction_from=WIND_FROM)
fleet_wind = FleetWind.snapshot([a, b, victim], wind, 0.0, D)
speed_one, _, def_one = FleetWind.snapshot([a, victim], wind, 0.0, D).at(victim, TWS, WIND_FROM)
speed_two, _, def_two = fleet_wind.at(victim, TWS, WIND_FROM)
check("two boats ahead hurt more than one", def_two > def_one, f"{def_two:.3f} vs {def_one:.3f}")
check("wind is never driven below the floor", speed_two >= TWS * D.min_wind_fraction - 1e-9)

reversed_wind = FleetWind.snapshot([victim, b, a], wind, 0.0, D)
speed_rev, _, def_rev = reversed_wind.at(victim, TWS, WIND_FROM)
check(
    "result does not depend on fleet list order",
    abs(def_rev - def_two) < 1e-12,
    f"{def_rev} vs {def_two}",
)

alone = FleetWind.snapshot([victim], wind, 0.0, D)
speed_alone, dir_alone, def_alone = alone.at(victim, TWS, WIND_FROM)
check("a lone boat is undisturbed", def_alone == 0.0 and speed_alone == TWS)

# --- emergent behaviour in a real race --------------------------------------
group("emergent: lanes in an 18-boat race")

course = Course.for_conditions(TWS, C420, upwind_minutes=5.5, laps=2, wind_from=WIND_FROM)

clean_fleet = build_fleet(size=18, course=course, seed=7)
clean = Simulator(course, UniformWind(speed_kt=TWS, direction_from=WIND_FROM),
                  interaction=None).run(clean_fleet)

dirty_fleet = build_fleet(size=18, course=course, seed=7)
dirty = Simulator(course, UniformWind(speed_kt=TWS, direction_from=WIND_FROM),
                  interaction=D).run(dirty_fleet)

check("fleet still completes the course with interaction on", dirty.finishers() == 18,
      f"{dirty.finishers()}/18")

clean_times = [t for _, _, t in clean.order if t is not None]
dirty_times = [t for _, _, t in dirty.order if t is not None]
clean_spread = max(clean_times) - min(clean_times)
dirty_spread = max(dirty_times) - min(dirty_times)
print(f"      spread first-to-last: clean {clean_spread:.0f}s -> dirty {dirty_spread:.0f}s")
check(
    "dirty air stretches the fleet out",
    dirty_spread > clean_spread,
    f"clean {clean_spread:.0f}s, dirty {dirty_spread:.0f}s",
)

by_id = {b.boat_id: b for b in dirty_fleet}
finish_rank = {bid: i for i, (bid, _, _) in enumerate(dirty.order)}
leader = by_id[dirty.order[0][0]]
backmarker = by_id[dirty.order[-1][0]]
print(f"      leader dirty-air {leader.dirty_air_integral:.1f} def-s, "
      f"backmarker {backmarker.dirty_air_integral:.1f} def-s")
check(
    "the winner sailed in cleaner air than the last boat",
    leader.dirty_air_integral < backmarker.dirty_air_integral,
    f"leader {leader.dirty_air_integral:.1f} vs last {backmarker.dirty_air_integral:.1f}",
)

# Rank correlation between dirty air and finishing position. Not a proof of
# causation inside the model — being slow also puts you behind boats — but if the
# sign ever flips, the interaction model is doing something incoherent.
ranks = sorted(dirty_fleet, key=lambda b: b.dirty_air_integral)
positions = [finish_rank[b.boat_id] for b in ranks]
inversions = sum(
    1 for i in range(len(positions)) for j in range(i + 1, len(positions))
    if positions[i] > positions[j]
)
pairs = len(positions) * (len(positions) - 1) / 2
print(f"      concordance between clean air and finishing position: "
      f"{100 * (1 - inversions / pairs):.0f}%")
check("cleaner air goes with better finishes more often than not", inversions < pairs / 2)

check(
    "every boat records its lane quality",
    all(b.dirty_air_s >= 0.0 for b in dirty_fleet)
    and any(b.dirty_air_integral > 0.0 for b in dirty_fleet),
)
check(
    "interaction off means no dirty air recorded at all",
    all(b.dirty_air_integral == 0.0 for b in clean_fleet),
)

print()
if FAILURES:
    print(f"\033[31m{len(FAILURES)} check(s) failed:\033[0m " + ", ".join(FAILURES))
    raise SystemExit(1)
print("\033[32mall checks passed\033[0m")
