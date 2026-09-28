"""Run an 18-boat race and draw it, so the tracks can be eyeballed.

Looking at the plot is not a substitute for the tests, but it catches a different
class of error: tracks that are individually legal and collectively absurd. The
92-tacks bug would have been obvious here in a second and took a test to find.

    python3 run_demo.py                 # oscillating wind, the default
    python3 run_demo.py uniform 12      # steady 12 knots
    python3 run_demo.py persistent 8    # a righty through the race
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sailsim.course import Course  # noqa: E402
from sailsim.polar import C420  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.wind import OscillatingWind, PersistentShift, UniformWind  # noqa: E402

WIND_FROM = 20.0  # Roughly down the Severn's axis. A placeholder, not a measurement.


def build_wind(kind: str, speed: float):
    if kind == "uniform":
        return UniformWind(speed_kt=speed, direction_from=WIND_FROM)
    if kind == "persistent":
        return PersistentShift(start_direction=WIND_FROM, rate_deg_per_min=2.5, mean_speed_kt=speed)
    return OscillatingWind(
        mean_direction=WIND_FROM,
        amplitude_deg=12.0,
        period_s=210.0,
        mean_speed_kt=speed,
        # A phase gradient across the course, so the two sides are not in step and
        # left-versus-right is a real question rather than a coin flip.
        spatial_wavelength_m=1600.0,
    )


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "oscillating"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0

    # Sized so the beat lands in the 4-7 minute window at this wind speed.
    course = Course.for_conditions(speed, C420, upwind_minutes=5.5, laps=2, wind_from=WIND_FROM)
    fleet = build_fleet(size=18, boat_class="c420", course=course)
    result = Simulator(course, build_wind(kind, speed), record_every_s=2.0).run(fleet)

    beat_m = (course.marks[0].x ** 2 + course.marks[0].y ** 2) ** 0.5
    print(f"\nwind: {kind} at {speed:.0f} kt from {WIND_FROM:.0f}   course: 2 laps, {beat_m:.0f} m beat")
    print(result.table())
    won = result.order[0][2]
    print(f"\nwinner {won:.0f}s ({won / 60:.1f} min)" if won else "\nno finisher")
    beats = [b for b in fleet]
    print(
        f"manoeuvres: {min(b.tacks + b.gybes for b in beats)}-{max(b.tacks + b.gybes for b in beats)}"
        f"  |  spread first to last: "
        f"{(result.order[-1][2] or 0) - (result.order[0][2] or 0):.0f}s"
    )
    print("\nPROVENANCE — read every number above in this light:")
    for note in result.estimates_used:
        print(f"  * {note}")

    fig, ax = plt.subplots(figsize=(7.5, 10.5))
    ax.set_facecolor("#0b1622")
    fig.patch.set_facecolor("#0b1622")

    # Leaders drawn bright, the rest dim, so the plot reads as a fleet rather than
    # as spaghetti. Finish order, not boat id, sets the colour.
    order = [bid for bid, _, _ in result.order]
    by_id = {b.boat_id: b for b in fleet}
    for rank, bid in enumerate(order):
        b = by_id[bid]
        if not b.track:
            continue
        xs = [p[1] for p in b.track]
        ys = [p[2] for p in b.track]
        if rank < 3:
            ax.plot(xs, ys, lw=1.8, alpha=0.95, zorder=5, label=f"{rank + 1}. {b.name}")
        else:
            ax.plot(xs, ys, lw=0.7, alpha=0.30, color="#7fb3d5", zorder=2)

    for m in course.marks:
        ax.plot(m.x, m.y, marker="o", ms=9, color="#f5a623", zorder=6)
        ax.annotate(m.name, (m.x, m.y), textcoords="offset points", xytext=(9, 5),
                    color="#f5a623", fontsize=8)
    (bx, by), (px, py) = course.start_boat, course.start_pin
    ax.plot([bx, px], [by, py], color="#e8e8e8", lw=2.0, zorder=6)
    ax.annotate("start / finish", ((bx + px) / 2, by - 30), color="#e8e8e8", fontsize=8, ha="center")

    ax.set_aspect("equal")
    ax.set_title(
        f"18 boats, C420 (no spinnaker) — {kind} wind {speed:.0f} kt\n"
        "ESTIMATED polar and SYNTHETIC wind: shape only, not a Severn prediction",
        color="#e8e8e8", fontsize=10,
    )
    ax.set_xlabel("east (m)", color="#9bb")
    ax.set_ylabel("north (m)", color="#9bb")
    ax.tick_params(colors="#9bb", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3f52")
    ax.grid(color="#16263a", lw=0.5)
    ax.legend(loc="lower right", fontsize=8, facecolor="#0f1e2e", edgecolor="#2a3f52",
              labelcolor="#e8e8e8")

    out = f"out/race_{kind}_{speed:.0f}kt.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
