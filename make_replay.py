"""Run a race and write a replay file for replay/index.html.

    python3 make_replay.py                       # default: 18 boats, oscillating 8 kt
    python3 make_replay.py --wind uniform --kt 14
    python3 make_replay.py --baseline            # the blind helm, for comparison
    python3 make_replay.py --out out/my_race.json
"""

from __future__ import annotations

import argparse

from sailsim.course import Course, Helm
from sailsim.policy import TacticalHelm
from sailsim.polar import C420
from sailsim.replay import export
from sailsim.sim import Simulator, build_fleet
from sailsim.wind import OscillatingWind, PersistentShift, UniformWind


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wind", default="oscillating", choices=["oscillating", "uniform", "persistent"])
    p.add_argument("--kt", type=float, default=8.0)
    p.add_argument("--boats", type=int, default=18)
    p.add_argument("--bias", type=float, default=8.0, help="line bias in degrees")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--baseline", action="store_true", help="use the blind layline helm")
    p.add_argument("--no-rules", action="store_true")
    p.add_argument("--out", default="out/replay.json")
    # One second is a good default: fine enough that manoeuvres look like
    # manoeuvres, coarse enough that the file stays a couple of megabytes.
    p.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    args = p.parse_args()

    course = Course.for_conditions(
        args.kt, C420, upwind_minutes=5.5, laps=args.laps, wind_from=0.0, line_bias_deg=args.bias
    )
    if args.wind == "uniform":
        wind = UniformWind(speed_kt=args.kt, direction_from=0.0)
    elif args.wind == "persistent":
        wind = PersistentShift(start_direction=0.0, rate_deg_per_min=2.5, mean_speed_kt=args.kt)
    else:
        wind = OscillatingWind(
            mean_direction=0.0, amplitude_deg=12.0, period_s=200.0,
            mean_speed_kt=args.kt, spatial_wavelength_m=1600.0,
        )

    fleet = build_fleet(size=args.boats, course=course, seed=args.seed)
    helm_factory = (lambda b: Helm(tack_bias=(b.boat_id % 5 - 2) / 2.0)) if args.baseline else (
        lambda b: TacticalHelm()
    )
    result = Simulator(
        course, wind, start_seed=args.seed, record_every_s=args.every,
        rules=not args.no_rules,
    ).run(fleet, helm_factory=helm_factory)

    document = export(result, course, wind, args.out)
    frames = sum(len(b["frames"]) for b in document["boats"])
    import os

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB, {frames:,} frames, "
          f"{len(document['boats'])} boats, {document['meta']['duration_s']:.0f}s)")
    print(f"  helm: {'baseline (blind)' if args.baseline else 'tactical'}   "
          f"wind: {args.wind} {args.kt:.0f} kt   bias: {document['meta']['line_bias_deg']:+.1f} deg "
          f"({document['meta']['favoured_end']} favoured)")
    print(f"  {document['meta']['fouls']} fouls, {document['meta']['ocs']} OCS")
    print("\nOpen replay/index.html in a browser and load that file.")


if __name__ == "__main__":
    main()
