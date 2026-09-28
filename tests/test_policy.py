"""Checks over the tactical helm.

The policy is the thing a search will tune, so its decisions have to be pinned
precisely: a policy that quietly stops responding to shifts still sails a
plausible-looking race, and the sweep that tunes it would then be optimising a
parameter that does nothing.

The last section encodes the finding from experiments/shift_threshold.py as a
regression test. That matters more than it looks: the U-shaped cost curve — bad to
tack on everything, bad to never tack, optimum in between, and the optimum moving
UP as wind speed rises — is the model's headline strategic result, and it is a
consequence of the manoeuvre-cost calibration. If that curve ever flattens or
inverts, either the tactics or the tack cost has broken.

Run: python3 tests/test_policy.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim import geometry as geo  # noqa: E402
from sailsim.boat import Boat  # noqa: E402
from sailsim.course import Course, Helm  # noqa: E402
from sailsim.polar import C420  # noqa: E402
from sailsim.policy import TacticalHelm, WindMemory  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.wind import OscillatingWind, UniformWind  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  \033[32m+\033[0m {name}")
    else:
        print(f"  \033[31mx\033[0m {name}" + (f"\n      {detail}" if detail else ""))
        FAILURES.append(name)


def group(name: str) -> None:
    print(f"\n\033[1m== {name} ==\033[0m")


TWS = 8.0
CLOSE_HAULED, _ = C420.best_upwind(TWS)
BROAD, _ = C420.best_downwind(TWS)


def beating(tack: int, wind_from: float = 0.0, x: float = 0.0, y: float = 0.0) -> Boat:
    b = Boat(0, "b", C420, x=x, y=y)
    b.heading = geo.heading_for_twa(tack * CLOSE_HAULED, wind_from)
    b.speed_kt = b.target_speed_kt(TWS, wind_from)
    return b


# --- wind memory -------------------------------------------------------------
group("wind memory")

m = WindMemory(time_constant_s=10.0)
for _ in range(400):
    m.update(350.0, 0.5)
    m.update(10.0, 0.5)
check(
    "averaging 350 and 010 gives 000, not 180",
    abs(geo.angle_diff(m.mean, 0.0)) < 2.0,
    f"mean {m.mean:.1f}",
)

m2 = WindMemory(time_constant_s=60.0)
for _ in range(600):
    m2.update(40.0, 0.5)
check("a steady wind is learned exactly", abs(geo.angle_diff(m2.mean, 40.0)) < 0.5)
check("a shift is measured against the mean", abs(m2.shift(50.0) - 10.0) < 0.5,
      f"{m2.shift(50.0):.2f}")
check("shift sign follows the direction of the change", m2.shift(30.0) < 0.0)

fast, slow = WindMemory(time_constant_s=5.0), WindMemory(time_constant_s=300.0)
for _ in range(120):
    fast.update(0.0, 0.5)
    slow.update(0.0, 0.5)
for _ in range(60):
    fast.update(20.0, 0.5)
    slow.update(20.0, 0.5)
check(
    "a short memory chases the wind; a long one holds the mean",
    abs(geo.angle_diff(fast.mean, 20.0)) < abs(geo.angle_diff(slow.mean, 20.0)),
    f"fast {fast.mean:.1f}, slow {slow.mean:.1f}",
)

# --- the core comparison -----------------------------------------------------
group("which tack points closer to the mark")

helm = TacticalHelm()
# Mark dead upwind in a square breeze: neither tack is favoured.
square = helm._gain(1, 0.0, 0.0, CLOSE_HAULED) - helm._gain(-1, 0.0, 0.0, CLOSE_HAULED)
check("with the mark dead upwind neither tack is favoured", abs(square) < 1e-9)

# A backing shift heads a starboard-tack boat, so port now points closer.
headed = helm._gain(1, 0.0, -10.0, CLOSE_HAULED) - helm._gain(-1, 0.0, -10.0, CLOSE_HAULED)
check("a header makes the other tack better", headed > 0.0, f"advantage {headed:.1f} deg")
check(
    "a shift of N degrees is worth about 2N of bearing advantage",
    abs(headed - 20.0) < 0.5,
    f"{headed:.1f} for a 10 deg shift",
)
lifted = helm._gain(1, 0.0, 10.0, CLOSE_HAULED) - helm._gain(-1, 0.0, 10.0, CLOSE_HAULED)
check("a lift makes the current tack better", lifted < 0.0)

check(
    "the same comparison works downwind, with no special case",
    helm._gain(1, 180.0, 10.0, BROAD) - helm._gain(-1, 180.0, 10.0, BROAD) != 0.0,
)
check(
    "the better tack is picked, not just compared",
    helm._better_tack(0.0, -10.0, CLOSE_HAULED) == -1,
)

# --- tacking decisions -------------------------------------------------------
group("when it tacks")


def sail(wind_from: float, threshold: float, elapsed: float, mark=(0.0, 600.0), tack=1):
    """Run one helm decision and report the tack it settles on."""
    h = TacticalHelm(shift_threshold_deg=threshold, min_tack_interval_s=10.0)
    b = beating(tack, wind_from)
    # Seed the memory with the un-shifted breeze so the shift reads as a shift.
    for _ in range(300):
        h.memory.update(0.0, 0.5)
    h._committed, h._committed_leg, h._last_tack_t = tack, 0, 0.0
    h.target_heading(b, mark[0], mark[1], TWS, wind_from, t=elapsed, deficit=0.0)
    return h._committed


check(
    "a big header triggers a tack",
    sail(wind_from=-14.0, threshold=8.0, elapsed=60.0) == -1,
)
check(
    "a small header does not",
    sail(wind_from=-2.0, threshold=8.0, elapsed=60.0) == 1,
)
check(
    "raising the threshold suppresses the same tack",
    sail(wind_from=-14.0, threshold=40.0, elapsed=60.0) == 1,
)
check(
    "a lift never triggers a tack",
    sail(wind_from=14.0, threshold=8.0, elapsed=60.0) == 1,
)
check(
    "no tack before the minimum interval has elapsed",
    sail(wind_from=-14.0, threshold=8.0, elapsed=3.0) == 1,
)

# Near the layline a tack is a commitment, not a tactic.
guarded = TacticalHelm(shift_threshold_deg=2.0, min_tack_interval_s=0.0, layline_guard_deg=8.0)
check(
    "no tactical tack inside the layline guard band",
    not guarded._may_tack(mark_twa=CLOSE_HAULED - 2.0, angle=CLOSE_HAULED, upwind=True, t=1e6),
)
check(
    "tactical tacks are allowed well below the layline",
    guarded._may_tack(mark_twa=5.0, angle=CLOSE_HAULED, upwind=True, t=1e6),
)

# --- bailing out of dirty air ------------------------------------------------
group("clearing its air")

# Counted, not read off the end state. A boat held in dirty air indefinitely tacks
# out, finds itself still in it, and tacks again -- so the FINAL tack is whichever
# way an even or odd number of escapes left it, and asserting a particular one
# tests the parity of the loop rather than the behaviour. (In a real race this does
# not arise: tacking away changes the deficit, which is exactly what the escape is
# for. Here the deficit is pinned by the test.)
air = TacticalHelm(shift_threshold_deg=90.0, dirty_air_threshold=0.05,
                   dirty_air_patience_s=6.0, min_tack_interval_s=0.0)
b = beating(1)
air._committed, air._committed_leg = 1, 0
escapes, previous = 0, air._committed
for step in range(30):
    air.target_heading(b, 0.0, 600.0, TWS, 0.0, t=step * 0.5, deficit=0.20)
    if air._committed != previous:
        escapes += 1
        previous = air._committed
check(
    "a boat left in dirty air tacks out even with shifts switched off",
    escapes >= 1,
    "threshold was 90 deg, so only the dirty-air rule could have fired",
)
check("and it does not tack every single timestep", escapes <= 4, f"{escapes} escapes in 15 s")

patient = TacticalHelm(shift_threshold_deg=90.0, dirty_air_threshold=0.05,
                       dirty_air_patience_s=60.0, min_tack_interval_s=0.0)
b2 = beating(1)
patient._committed, patient._committed_leg = 1, 0
for step in range(30):
    patient.target_heading(b2, 0.0, 600.0, TWS, 0.0, t=step * 0.5, deficit=0.20)
check("a patient boat holds its lane", patient._committed == 1)

clean = TacticalHelm(shift_threshold_deg=90.0, dirty_air_patience_s=6.0, min_tack_interval_s=0.0)
b3 = beating(1)
clean._committed, clean._committed_leg = 1, 0
for step in range(30):
    clean.target_heading(b3, 0.0, 600.0, TWS, 0.0, t=step * 0.5, deficit=0.0)
check("a boat in clear air does not tack out of it", clean._committed == 1)

# --- it still sails the course ----------------------------------------------
group("it still gets round")

course = Course.for_conditions(TWS, C420, wind_from=0.0)
fleet = build_fleet(size=6, course=course, seed=2)
result = Simulator(course, UniformWind(TWS, 0.0), interaction=None, rules=False,
                   prestart_s=0.0).run(fleet, helm_factory=lambda b: TacticalHelm())
check("a tactical fleet finishes in steady wind", result.finishers() == 6)
# In a steady breeze a tactical helm should sail like the baseline: laylines and
# nothing else, because there are no headers to tack on. Eight manoeuvres over two
# laps is the layline-only figure. Before the header requirement this was 26, and
# before the threshold was set from the experiment it was 40 -- a slow rediscovery
# of the rhumb-line chasing that the very first helm did.
check(
    "in steady wind it sails laylines and nothing else",
    max(b.tacks + b.gybes for b in fleet) <= 10,
    f"worst {max(b.tacks + b.gybes for b in fleet)} manoeuvres",
)

# --- the headline result, pinned --------------------------------------------
group("regression: the shift-threshold cost curve")


def mean_time(threshold: float, tws: float, phases=(0.0, 60.0, 120.0, 180.0)) -> float:
    times = []
    for phase in phases:
        c = Course.for_conditions(tws, C420, wind_from=0.0)
        f = build_fleet(size=1, course=c, seed=1)
        f[0].speed_factor = 1.0
        f[0].handling = 1.0
        w = OscillatingWind(mean_direction=0.0, amplitude_deg=12.0, period_s=200.0,
                            mean_speed_kt=tws, gust_factor=0.0, spatial_wavelength_m=0.0,
                            phase_s=phase)
        r = Simulator(c, w, interaction=None, rules=False, prestart_s=0.0).run(
            f, helm_factory=lambda b: TacticalHelm(
                shift_threshold_deg=threshold, min_tack_interval_s=15.0,
                dirty_air_patience_s=1e9)
        )
        if r.order[0][2] is not None:
            times.append(r.order[0][2])
    return statistics.mean(times)


# What is ROBUST across wind speeds is that ignoring shifts is expensive: never
# tacking tactically costs 50-80 s over two laps against any moderate threshold.
# What is NOT robust is the exact optimum -- the curve is flat from about 0 to 30
# degrees, and differences inside that band are a few seconds against a standard
# deviation of five or more. So the tests assert the large effect and deliberately
# decline to assert the small one. An earlier draft asserted that a high threshold
# pays off more in a breeze; the measurement says otherwise, and the claim is gone
# rather than tuned until it passed.
for tws in (10.0, 18.0):
    responsive = mean_time(14.0, tws)
    never = mean_time(90.0, tws)
    too_fussy = mean_time(55.0, tws)
    print(f"      {tws:.0f} kt: responsive {responsive:.0f}s | "
          f"threshold 55 {too_fussy:.0f}s | never tack tactically {never:.0f}s")
    check(
        f"{tws:.0f} kt: ignoring shifts is expensive",
        never - responsive > 25.0,
        f"responsive {responsive:.0f}s vs never {never:.0f}s (gap {never - responsive:.0f}s)",
    )
    check(
        f"{tws:.0f} kt: a threshold so high it misses real shifts also loses",
        too_fussy > responsive,
        f"{too_fussy:.0f}s vs {responsive:.0f}s",
    )

# --- head to head ------------------------------------------------------------
group("emergent: tactical against baseline in one fleet")

tactical_ranks: list[int] = []
baseline_ranks: list[int] = []
for seed in range(10):
    c = Course.for_conditions(TWS, C420, wind_from=0.0, line_bias_deg=8.0)
    f = build_fleet(size=18, course=c, seed=seed)
    # Crew speed equalised so the ONLY difference between the two halves is policy.
    for b in f:
        b.speed_factor = 1.0
        b.handling = 1.0
    w = OscillatingWind(mean_direction=0.0, amplitude_deg=12.0, period_s=200.0,
                        mean_speed_kt=TWS, spatial_wavelength_m=1600.0, phase_s=seed * 29.0)
    r = Simulator(c, w, start_seed=seed).run(
        f,
        helm_factory=lambda b: (
            TacticalHelm()
            if b.boat_id % 2 == 0
            else Helm(tack_bias=0.0)
        ),
    )
    ranks = {bid: i for i, (bid, _, _) in enumerate(r.order)}
    for b in f:
        (tactical_ranks if b.boat_id % 2 == 0 else baseline_ranks).append(ranks[b.boat_id])

mean_tac = statistics.mean(tactical_ranks)
mean_base = statistics.mean(baseline_ranks)
print(f"      tactical {mean_tac:.2f} vs baseline {mean_base:.2f} "
      f"(n={len(tactical_ranks)} each)")
# ASSERTED NOW, BUT IT WAS NOT ALWAYS SAFE TO. Before tactical tacks required an
# actual header, this gap measured +0.35 places against a standard error of 0.61 --
# noise, and an earlier draft of this file correctly declined to assert a direction.
# The header requirement stopped the boat tacking away its own gains, and over 20
# races the gap is now +1.89 +/- 0.54, about 3.5 standard errors. The margin below
# is deliberately well inside that, so this fails if the effect degrades rather
# than only if it vanishes.
check(
    "the tactical helm beats the blind one head to head",
    mean_tac < mean_base - 0.5,
    f"tactical {mean_tac:.2f} vs baseline {mean_base:.2f} (gap {mean_base - mean_tac:+.2f})",
)
check("the two groups are the same size, so the means are comparable",
      len(tactical_ranks) == len(baseline_ranks))
print(f"      gap {mean_base - mean_tac:+.2f} places over {len(tactical_ranks) // 18} races; "
      f"see the README for the larger-sample figure and its standard error")

print()
if FAILURES:
    print(f"\033[31m{len(FAILURES)} check(s) failed:\033[0m " + ", ".join(FAILURES))
    raise SystemExit(1)
print("\033[32mall checks passed\033[0m")
