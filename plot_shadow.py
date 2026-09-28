"""Draw the disturbance field around a beating boat.

The shadow's position relative to the boat is the one thing in this model that is
both easy to get backwards and impossible to notice being backwards from race
results alone. This draws it, so it can be checked by eye against what a sailor
knows: bad air trails aft and to LEEWARD, the windward lane is clear.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sailsim import geometry as geo  # noqa: E402
from sailsim.boat import Boat  # noqa: E402
from sailsim.interaction import Disturbance  # noqa: E402
from sailsim.polar import C420  # noqa: E402

TWS, WIND_FROM = 8.0, 0.0
D = Disturbance()
BL = C420.boat_length_m

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
fig.patch.set_facecolor("#0b1622")

for ax, tack, label in ((axes[0], +1, "starboard tack"), (axes[1], -1, "port tack")):
    close_hauled, _ = C420.best_upwind(TWS)
    src = Boat(0, "src", C420, x=0.0, y=0.0)
    src.heading = geo.heading_for_twa(tack * close_hauled, WIND_FROM)
    src.speed_kt = src.target_speed_kt(TWS, WIND_FROM)

    span = 9 * BL
    n = 260
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    field = np.zeros((n, n))
    bend = np.zeros((n, n))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            field[i, j] = D.shadow_at(src, float(xx), float(yy), TWS, WIND_FROM)
            bend[i, j] = D.backwind_at(src, float(xx), float(yy), TWS, WIND_FROM)

    ax.set_facecolor("#0b1622")
    im = ax.pcolormesh(xs / BL, ys / BL, field, cmap="inferno", vmin=0, vmax=D.shadow_max_deficit,
                       shading="auto")
    ax.contour(xs / BL, ys / BL, np.abs(bend), levels=[1.0, 4.0], colors="#4fd1c5",
               linewidths=1.1, linestyles="--")

    # The boat, its heading, the true wind, and the apparent wind.
    hx, hy = np.sin(np.radians(src.heading)), np.cos(np.radians(src.heading))
    ax.arrow(0, 0, hx * 2.2, hy * 2.2, width=0.09, color="#ffffff", zorder=6, length_includes_head=True)
    ax.plot(0, 0, "o", ms=9, color="#ffffff", zorder=7)

    axis = D.shadow_axis(src, TWS, WIND_FROM)
    ax_x, ax_y = np.sin(np.radians(axis)), np.cos(np.radians(axis))
    ax.plot([0, ax_x * 7.5], [0, ax_y * 7.5], color="#f6e05e", lw=1.6, zorder=6,
            label="apparent-wind shadow axis")
    tdx, tdy = np.sin(np.radians(180.0)), np.cos(np.radians(180.0))
    ax.plot([0, tdx * 7.5], [0, tdy * 7.5], color="#fc8181", lw=1.4, ls=":", zorder=6,
            label="true downwind (NOT the axis)")

    ax.set_title(f"{label} — beating, {TWS:.0f} kt\nwind from {WIND_FROM:.0f}",
                 color="#e8e8e8", fontsize=10)
    ax.set_xlabel("boat lengths (east)", color="#9bb")
    ax.set_ylabel("boat lengths (north)", color="#9bb")
    ax.set_aspect("equal")
    ax.tick_params(colors="#9bb", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3f52")
    ax.legend(loc="upper left", fontsize=8, facecolor="#0f1e2e", edgecolor="#2a3f52",
              labelcolor="#e8e8e8")

cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("wind speed deficit", color="#e8e8e8")
cb.ax.tick_params(colors="#9bb")
fig.suptitle("Wind shadow (colour) and backwind header (dashed) — ESTIMATED magnitudes",
             color="#e8e8e8", fontsize=11)
fig.savefig("out/shadow.png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote out/shadow.png")
