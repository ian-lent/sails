"""How big must a shift be before tacking on it pays?

THE FIRST REAL QUESTION THIS SIMULATOR CAN ANSWER, and the shape of answer the
project is for: not a trajectory, but a decision rule with a breakeven.

Method. ONE BOAT, alone, with no fleet, no dirty air and no rules. That isolation
is the point — with other boats present, the answer would be contaminated by
traffic, and the question "when is a shift worth a tack" is about the wind and the
cost of tacking, nothing else. Each candidate threshold sails the same set of wind
realisations, and realisations differ by phase, because a race that opens on a
lift and one that opens on a header are different races.

What makes the answer non-trivial: tacking converts a shift into distance gained,
and costs distance to do it. The cost is the manoeuvre model — which the user
calibrated: nearly free in light air, one to three boat lengths in a breeze. So
the breakeven threshold should MOVE with wind speed, and a sweep that finds one
number for all conditions has probably found a bug.

    python3 experiments/shift_threshold.py            # default sweep
    python3 experiments/shift_threshold.py 16 14      # 16 kt, 14 deg amplitude
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim.course import Course  # noqa: E402
from sailsim.policy import TacticalHelm  # noqa: E402
from sailsim.polar import C420  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.wind import OscillatingWind  # noqa: E402

THRESHOLDS = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 55.0, 90.0]
PHASES = [0.0, 23.0, 47.0, 71.0, 95.0, 119.0, 143.0, 167.0, 191.0, 215.0]


def race(threshold: float, phase: float, tws: float, amplitude: float, period: float) -> float | None:
    """Seconds for one boat to sail two laps. None if it failed to finish."""
    course = Course.for_conditions(tws, C420, upwind_minutes=5.5, laps=2, wind_from=0.0)
    fleet = build_fleet(size=1, course=course, seed=1)
    fleet[0].speed_factor = 1.0
    fleet[0].handling = 1.0
    wind = OscillatingWind(
        mean_direction=0.0,
        amplitude_deg=amplitude,
        period_s=period,
        mean_speed_kt=tws,
        gust_factor=0.0,          # direction only: velocity shifts are a separate question
        spatial_wavelength_m=0.0,  # uniform in space, so this is purely temporal
        phase_s=phase,
    )
    result = Simulator(
        course, wind,
        interaction=None,   # alone on the course
        rules=False,        # nobody to give way to
        prestart_s=0.0,     # the start is a different experiment
    ).run(fleet, helm_factory=lambda b: TacticalHelm(
        shift_threshold_deg=threshold,
        min_tack_interval_s=15.0,
        dirty_air_patience_s=1e9,  # no fleet, so this must never fire
    ))
    return result.order[0][2]


def sweep(tws: float, amplitude: float, period: float) -> None:
    print(f"\n  {tws:.0f} kt, oscillation +/-{amplitude:.0f} deg on a {period:.0f} s period")
    print(f"  {'threshold':>10} {'mean':>9} {'stdev':>7} {'tacks':>7}   {'':<24}")
    rows = []
    for threshold in THRESHOLDS:
        times, tacks = [], []
        for phase in PHASES:
            course = Course.for_conditions(tws, C420, upwind_minutes=5.5, laps=2, wind_from=0.0)
            fleet = build_fleet(size=1, course=course, seed=1)
            fleet[0].speed_factor = 1.0
            fleet[0].handling = 1.0
            wind = OscillatingWind(
                mean_direction=0.0, amplitude_deg=amplitude, period_s=period,
                mean_speed_kt=tws, gust_factor=0.0, spatial_wavelength_m=0.0,
                phase_s=phase,
            )
            result = Simulator(
                course, wind, interaction=None, rules=False, prestart_s=0.0
            ).run(fleet, helm_factory=lambda b: TacticalHelm(
                shift_threshold_deg=threshold, min_tack_interval_s=15.0,
                dirty_air_patience_s=1e9,
            ))
            t = result.order[0][2]
            if t is not None:
                times.append(t)
                tacks.append(fleet[0].tacks + fleet[0].gybes)
        if not times:
            continue
        mean = statistics.mean(times)
        rows.append((threshold, mean, statistics.pstdev(times), statistics.mean(tacks)))

    best = min(r[1] for r in rows)
    for threshold, mean, sd, tk in rows:
        label = "tack on everything" if threshold == 0.0 else (
            "never tack tactically" if threshold >= 90.0 else "")
        bar = "#" * int(round((mean - best) / 2.0))
        marker = "  <-- best" if mean == best else ""
        print(f"  {threshold:>10.0f} {mean:>8.1f}s {sd:>6.1f}s {tk:>7.1f}   {bar}{marker} {label}")
    print(f"  spread between best and worst: {max(r[1] for r in rows) - best:.1f} s")


if __name__ == "__main__":
    tws = float(sys.argv[1]) if len(sys.argv) > 1 else None
    amp = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    print(__doc__.split("\n")[0])
    if tws is not None:
        sweep(tws, amp, 200.0)
    else:
        # The interesting comparison: the same oscillation in air where a tack is
        # nearly free, and in air where it costs two boat lengths.
        for wind_speed in (5.0, 10.0, 18.0):
            sweep(wind_speed, amp, 200.0)
