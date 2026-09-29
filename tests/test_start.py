"""Checks over the line, the approach, and being over early.

The load-bearing claim of this module is that the favoured end is worth something
real and that the model reproduces it, so the emergent test at the bottom is the
one that matters. Everything above it exists to make that test interpretable: if
the bias geometry is wrong, a correlation between start position and finish is
measuring nothing.

Run: python3 tests/test_start.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim.course import Course  # noqa: E402
from sailsim.polar import C420  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.start import (  # noqa: E402
    BIAS_MAX_DEG,
    BIAS_MIN_DEG,
    LineBias,
    draw_bias,
    favoured_end,
    line_bias,
    line_fraction,
    line_side,
)
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


TWS, WIND = 8.0, 0.0


# --- the line ----------------------------------------------------------------
group("line bias geometry")

for setting in (-15.0, -8.0, -1.0, 1.0, 8.0, 15.0):
    c = Course.windward_leeward(wind_from=WIND, line_bias_deg=setting)
    measured = line_bias(c.start_pin, c.start_boat, WIND)
    check(
        f"a line set at {setting:+.0f} deg measures {setting:+.0f}",
        abs(measured - setting) < 1e-6,
        f"measured {measured:+.3f}",
    )

c = Course.windward_leeward(wind_from=WIND, line_bias_deg=8.0)
check("positive bias favours the PIN", favoured_end(c.start_pin, c.start_boat, WIND) == "pin")
check("the favoured end is the one upwind", c.start_pin[1] > c.start_boat[1])
c_neg = Course.windward_leeward(wind_from=WIND, line_bias_deg=-8.0)
check("negative bias favours the BOAT", favoured_end(c_neg.start_pin, c_neg.start_boat, WIND) == "boat")

# The arithmetic that makes bias matter, checked against the closed form.
for deg in (2.0, 8.0, 15.0):
    course = Course.windward_leeward(wind_from=WIND, line_bias_deg=deg, line_length_m=140.0)
    expected = 140.0 * math.sin(math.radians(deg))
    got = LineBias(deg, 0.0, 0.0).advantage_m(course.line_length_m)
    check(
        f"{deg:.0f} deg on a 140 m line is worth {expected:.1f} m",
        abs(got - expected) < 0.01,
        f"got {got:.2f}",
    )
print(f"      8 deg = {140.0 * math.sin(math.radians(8.0)) / C420.boat_length_m:.1f} boat lengths")

# Bias is relative to the WIND, so the same line changes favour as the wind shifts.
# This is the half of the bias that cannot be scouted before the sequence.
shifted = line_bias(c.start_pin, c.start_boat, WIND + 12.0)
check(
    "a wind shift can reverse which end is favoured",
    favoured_end(c.start_pin, c.start_boat, WIND + 12.0) == "boat",
    f"bias was +8.0, after a 12 deg shift it is {shifted:+.1f}",
)

# --- drawing a race's bias ---------------------------------------------------
group("drawn bias, per the 1-15 degree specification")

rng = random.Random(3)
draws = [draw_bias(rng) for _ in range(4000)]
magnitudes = [abs(d.total_deg) for d in draws]
check(
    "every draw lands in [1, 15] degrees",
    all(BIAS_MIN_DEG <= m <= BIAS_MAX_DEG for m in magnitudes),
    f"range {min(magnitudes):.2f}..{max(magnitudes):.2f}",
)
check("no draw is a square line", min(magnitudes) >= BIAS_MIN_DEG)
check(
    "both ends get favoured",
    0.4 < sum(1 for d in draws if d.total_deg > 0) / len(draws) < 0.6,
)
check(
    "the two sources sum to the total",
    all(abs(d.committee_deg + d.shift_deg - d.total_deg) < 1e-9 for d in draws),
)
check(
    "both sources contribute in every draw",
    all(abs(d.committee_deg) > 1e-9 and abs(d.shift_deg) > 1e-9 for d in draws),
)
check(
    "favoured end agrees with the sign",
    all((d.favoured == "pin") == (d.total_deg > 0) for d in draws),
)
print(f"      mean magnitude {sum(magnitudes) / len(magnitudes):.1f} deg, "
      f"mean committee share {sum(abs(d.committee_deg) / abs(d.total_deg) for d in draws) / len(draws):.0%}")

# --- which side of the line --------------------------------------------------
group("over the line")

c = Course.windward_leeward(wind_from=WIND, line_bias_deg=0.0)
check("a point upwind of the line is on the course side", line_side(c.start_pin, c.start_boat, 0.0, 30.0, WIND) > 0)
check("a point downwind of it is not", line_side(c.start_pin, c.start_boat, 0.0, -30.0, WIND) < 0)
check("a point on it reads zero", abs(line_side(c.start_pin, c.start_boat, 0.0, 0.0, WIND)) < 1e-9)
check("the pin end reads 0 along the line", abs(line_fraction(c.start_pin, c.start_boat, *c.start_pin)) < 1e-9)
check("the boat end reads 1", abs(line_fraction(c.start_pin, c.start_boat, *c.start_boat) - 1.0) < 1e-9)
check("the midpoint reads 0.5", abs(line_fraction(c.start_pin, c.start_boat, 0.0, 0.0) - 0.5) < 1e-9)

# --- the sequence produces a plausible start ---------------------------------
group("the start itself")

course = Course.for_conditions(TWS, C420, wind_from=WIND, line_bias_deg=8.0)
ocs_counts, medians, gun_speeds = [], [], []
for seed in range(6):
    fleet = build_fleet(size=18, course=course, seed=seed)
    result = Simulator(course, UniformWind(TWS, WIND), start_seed=seed).run(fleet)
    ocs_counts.append(result.ocs_count)
    sides = sorted(b.start_side_m for b in fleet)
    medians.append(sides[len(sides) // 2])
    gun_speeds.append(max(b.start_speed_kt for b in fleet))

print(f"      OCS per race {ocs_counts}, median boat {sum(medians) / len(medians):+.1f} m from the line")
check("some boats are over early, but not the whole fleet", all(0 <= n <= 6 for n in ocs_counts),
      f"{ocs_counts}")
check("at least one race has someone over", sum(ocs_counts) > 0)
# SIX lengths, not the two a real fleet manages, and the gap is a known artefact
# worth stating rather than tuning away. With the rules on, a boat keeps clear at
# the start by SLOWING -- that is all the avoidance model has -- so the fleet
# decelerates into the gun instead of holding lanes and accelerating in them.
# Real pre-start play is lane discipline: you defend a hole to leeward and go. Until
# that exists the fleet will start further back than it should, and this bound is
# the measurable symptom of the omission.
check(
    "the median boat starts within six lengths of the line",
    all(-6.0 * C420.boat_length_m < m < 0.0 for m in medians),
    f"medians {[round(m, 1) for m in medians]}",
)
check("somebody is at racing speed on the gun", all(s > 2.5 for s in gun_speeds))

# Being over early has to COST something, or the model has no start risk in it.
fleet = build_fleet(size=18, course=course, seed=0)
result = Simulator(course, UniformWind(TWS, WIND), start_seed=0).run(fleet)
over = [b for b in fleet if b.ocs]
clear = [b for b in fleet if not b.ocs]
check("boats over the line were flagged", len(over) > 0)
check("every OCS boat returned below the line before racing", all(not b.returning for b in over))
# REPORTED, NOT ASSERTED, and the reason is worth recording. An earlier version
# asserted that being over early costs places, on the strength of a single race
# where OCS boats averaged 13th against 7.6th. Measured properly -- 12 races, all
# the OCS boats there are -- the gap is -0.17 +/- 1.10 places: nothing, at n=9.
#
# Two things could explain that and they need different fixes. The rate is low, so
# the sample is thin. And the model's OCS recovery is probably too cheap: a boat
# barely over sails back a few metres and rejoins, whereas the real cost of being
# over is losing your lane and having the fleet roll over you, which nothing here
# represents. Asserting a penalty the model does not actually impose would hide
# that, so this reports the number and the README carries the caveat.
if over and clear:
    ranks = {bid: i for i, (bid, _, _) in enumerate(result.order)}
    mean_over = sum(ranks[b.boat_id] for b in over) / len(over)
    mean_clear = sum(ranks[b.boat_id] for b in clear) / len(clear)
    print(f"      mean finish rank: OCS {mean_over:.1f} (n={len(over)}) vs "
          f"clear {mean_clear:.1f} (n={len(clear)}) — see README, not asserted")
check("OCS boats rejoin the race rather than being stranded",
      all(b.finished_at is not None for b in over) or not over)

# --- the control case --------------------------------------------------------
group("control: no starting sequence")

no_start_fleet = build_fleet(size=18, course=course, seed=0)
no_start = Simulator(course, UniformWind(TWS, WIND), prestart_s=0.0).run(no_start_fleet)
check("no sequence means nobody is over", no_start.ocs_count == 0)
check("no sequence means no start variance recorded",
      all(b.start_speed_kt == 0.0 for b in no_start_fleet))
check("the fleet still races", no_start.finishers() == 18)

# --- emergent: does the favoured end actually pay? ---------------------------
group("emergent: the favoured end pays")

# Crowding is switched OFF so boats spread evenly along the line. With crowding on,
# nearly everyone is at the favoured end and the comparison has nothing to compare;
# the trade between a favoured end and the dirty air of the crowd there is a real
# effect, but it is a different question from whether the end is worth anything.
favoured_ranks: list[float] = []
unfavoured_ranks: list[float] = []
for seed in range(10):
    bias_deg = 12.0 if seed % 2 == 0 else -12.0
    c = Course.for_conditions(TWS, C420, wind_from=WIND, line_bias_deg=bias_deg)
    fleet = build_fleet(size=18, course=c, seed=seed)
    res = Simulator(c, UniformWind(TWS, WIND), start_seed=seed, start_crowding=0.0).run(fleet)
    ranks = {bid: i for i, (bid, _, _) in enumerate(res.order)}
    # Fraction 0 is the pin. When the pin is favoured, a LOW fraction is the good end.
    for b in fleet:
        toward_favoured = b.start_fraction if bias_deg < 0 else 1.0 - b.start_fraction
        if toward_favoured > 0.6:
            favoured_ranks.append(ranks[b.boat_id])
        elif toward_favoured < 0.4:
            unfavoured_ranks.append(ranks[b.boat_id])

mean_fav = sum(favoured_ranks) / len(favoured_ranks)
mean_unfav = sum(unfavoured_ranks) / len(unfavoured_ranks)
print(f"      mean finish rank: favoured end {mean_fav:.1f} (n={len(favoured_ranks)})"
      f" vs unfavoured {mean_unfav:.1f} (n={len(unfavoured_ranks)})")
# A margin, not a bare inequality. Measured over 14 races the gap is +0.62 places
# and stable, so 0.2 is comfortably inside it while still failing if the effect
# disappears -- which is what happened when line_side was measured from the
# midpoint, and a bare "<" would have passed that on floating-point noise.
check(
    "starting near the favoured end gives a better average finish",
    mean_fav < mean_unfav - 0.2,
    f"favoured {mean_fav:.2f} vs unfavoured {mean_unfav:.2f} (gap {mean_unfav - mean_fav:+.2f})",
)
check("both groups are large enough to mean anything", len(favoured_ranks) > 15 and len(unfavoured_ranks) > 15)

print()
if FAILURES:
    print(f"\033[31m{len(FAILURES)} check(s) failed:\033[0m " + ", ".join(FAILURES))
    raise SystemExit(1)
print("\033[32mall checks passed\033[0m")
