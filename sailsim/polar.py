"""Boat speed as a function of wind speed and true wind angle.

PROVENANCE WARNING, and it is the most important comment in this project.

These numbers are ESTIMATES. There is no published velocity prediction program
for a college CFJ or a non-spinnaker Club 420 that I can point at. The tables
below were constructed to have the right qualitative shape for a small
non-planing-to-marginally-planing displacement dinghy without a spinnaker:

  * a no-go zone inside roughly 35 degrees;
  * upwind speed that plateaus near 5 knots because the boat is pinned by crew
    righting moment long before it is pinned by drag;
  * peak speed on a beam-to-broad reach;
  * dead downwind markedly slower than a broad reach, because with no spinnaker
    the rig has very little projected area square to the wind;
  * the best upwind angle widening as wind drops (foot for speed in light air)
    and the best downwind angle squaring up as wind builds.

Every one of those is a claim about shape, not magnitude. Do not read a strategic
conclusion off this polar and take it to a regatta. `Polar.is_measured` is False
for these tables and the simulator reports it, so a result derived from estimates
cannot silently be mistaken for one derived from data.

REPLACING THIS WITH REAL DATA is the highest-value calibration step available.
Two routes, in order of preference:

  1. GPS tracks. Speed-over-ground binned by TWA against a known wind gives a
     measured polar directly. The wind reference is the hard part, not the boat
     data: a course-side anemometer or a committee-boat reading is enough to bin
     by, given the short legs involved.
  2. Timed runs. Upwind and downwind VMG over a known distance at a handful of
     wind speeds pins the two points strategy actually depends on. Far less data
     than a full polar and it constrains the part that matters.

Until then, treat the shapes as a scaffold whose job is to make the rest of the
infrastructure testable.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .geometry import BOAT_LENGTH_M

# TWA sample points, shared by every row. Symmetric boats, so only 0..180 is
# stored and the sign is applied by the caller.
TWA_GRID: tuple[float, ...] = (
    0, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
)

# Boat speed in KNOTS per true wind speed row, aligned to TWA_GRID.
# Club 420, no spinnaker, two-up, nominal crew weight.
C420_TABLE: dict[float, tuple[float, ...]] = {
    2.0:  (0.0, 0.0, 1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.3, 2.3, 2.3, 2.2, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 1.0),
    4.0:  (0.0, 0.0, 2.0, 2.6, 2.9, 3.1, 3.4, 3.6, 3.7, 3.7, 3.6, 3.5, 3.3, 3.1, 2.8, 2.5, 2.2, 2.0, 1.9),
    6.0:  (0.0, 0.0, 2.7, 3.4, 3.7, 3.9, 4.2, 4.4, 4.5, 4.5, 4.5, 4.4, 4.2, 3.9, 3.7, 3.4, 3.1, 2.8, 2.7),
    8.0:  (0.0, 0.0, 3.1, 3.9, 4.2, 4.4, 4.7, 4.9, 5.1, 5.2, 5.2, 5.1, 5.0, 4.8, 4.5, 4.2, 3.8, 3.5, 3.4),
    12.0: (0.0, 0.0, 3.5, 4.4, 4.7, 4.9, 5.4, 5.8, 6.0, 6.2, 6.3, 6.3, 6.2, 6.0, 5.7, 5.4, 5.0, 4.6, 4.4),
    16.0: (0.0, 0.0, 3.6, 4.6, 4.9, 5.1, 5.8, 6.3, 6.7, 6.9, 7.1, 7.2, 7.2, 7.0, 6.7, 6.4, 5.9, 5.4, 5.2),
    20.0: (0.0, 0.0, 3.7, 4.7, 5.0, 5.2, 6.0, 6.6, 7.0, 7.3, 7.6, 7.8, 7.8, 7.6, 7.3, 7.0, 6.5, 6.0, 5.8),
}

# The FJ is shorter and a little slower. A flat scalar is a placeholder for a
# real second table: the two boats almost certainly differ in SHAPE, not just
# magnitude, most likely downwind and in a breeze. Since college regattas sail
# one class at a time, this only matters when comparing across venues.
FJ_SPEED_SCALE = 0.95


@dataclass(frozen=True)
class Polar:
    """Interpolated speed lookup for one boat class."""

    name: str
    boat_length_m: float
    # Beam matters only for contact: rules.hulls_touching treats boats as segments
    # and asks whether they come within the mean beam.
    beam_m: float
    table: dict[float, tuple[float, ...]]
    scale: float = 1.0
    is_measured: bool = False
    source: str = "estimated shape, not measured — see module docstring"
    _winds: tuple[float, ...] = field(init=False, repr=False, default=())
    _vmg_cache: dict = field(init=False, repr=False, default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_winds", tuple(sorted(self.table)))
        # Memo for the VMG optima. These are pure functions of wind speed, and the
        # helm asks for both on every timestep for every boat — 36 solves per step
        # in an 18-boat fleet, each sweeping 600 candidate angles. Uncached that is
        # about 90 seconds per race, which is survivable once and ruinous for the
        # thousands of races a policy search needs. Wind speed is quantised to
        # 0.05 kt for the key: far finer than any real measurement, and it makes
        # the cache hit on the continuously-varying wind a gust model produces.
        object.__setattr__(self, "_vmg_cache", {})

    def speed(self, tws_kt: float, twa_deg: float) -> float:
        """Boat speed in knots. TWA may be signed; boats are symmetric."""
        twa = abs(twa_deg)
        if twa > 180.0:
            twa = 360.0 - twa
        lo, hi, frac = self._bracket(tws_kt)
        return self.scale * (
            (1.0 - frac) * self._interp_twa(self.table[lo], twa)
            + frac * self._interp_twa(self.table[hi], twa)
        )

    def _bracket(self, tws: float) -> tuple[float, float, float]:
        """Neighbouring wind rows and the blend fraction between them.

        Clamps outside the table rather than extrapolating. Extrapolating a polar
        is how a simulator ends up sailing 9 knots in 30 knots of wind, which is
        not a speed these boats reach in any condition — they capsize first, and
        that is a regime this model does not represent at all.
        """
        winds = self._winds
        if tws <= winds[0]:
            return winds[0], winds[0], 0.0
        if tws >= winds[-1]:
            return winds[-1], winds[-1], 0.0
        i = bisect.bisect_left(winds, tws)
        lo, hi = winds[i - 1], winds[i]
        return lo, hi, (tws - lo) / (hi - lo)

    @staticmethod
    def _interp_twa(row: tuple[float, ...], twa: float) -> float:
        if twa <= TWA_GRID[0]:
            return row[0]
        if twa >= TWA_GRID[-1]:
            return row[-1]
        i = bisect.bisect_left(TWA_GRID, twa)
        a, b = TWA_GRID[i - 1], TWA_GRID[i]
        frac = (twa - a) / (b - a)
        return (1.0 - frac) * row[i - 1] + frac * row[i]

    def best_upwind(self, tws_kt: float) -> tuple[float, float]:
        """(TWA, VMG in knots) maximising progress to windward. Cached."""
        return self._cached_vmg(tws_kt, upwind=True)

    def best_downwind(self, tws_kt: float) -> tuple[float, float]:
        """(TWA, VMG in knots) maximising progress to leeward. Cached."""
        return self._cached_vmg(tws_kt, upwind=False)

    def _cached_vmg(self, tws_kt: float, upwind: bool) -> tuple[float, float]:
        key = (round(tws_kt * 20.0), upwind)
        hit = self._vmg_cache.get(key)
        if hit is None:
            sweep = range(300, 900) if upwind else range(900, 1801)
            hit = self._best_vmg(tws_kt, sweep, upwind=upwind)
            self._vmg_cache[key] = hit
        return hit

    def _best_vmg(self, tws_kt: float, tenth_degrees: range, upwind: bool) -> tuple[float, float]:
        best_twa, best_vmg = 0.0, -1.0
        for tenths in tenth_degrees:
            twa = tenths / 10.0
            vmg = self.speed(tws_kt, twa) * math.cos(math.radians(twa))
            if not upwind:
                vmg = -vmg
            if vmg > best_vmg:
                best_twa, best_vmg = twa, vmg
        return best_twa, best_vmg

    def no_go_limit(self, tws_kt: float) -> float:
        """Closest TWA that still produces movement, to a tenth of a degree."""
        for tenths in range(200, 900):
            if self.speed(tws_kt, tenths / 10.0) > 0.05:
                return tenths / 10.0
        return 90.0


C420 = Polar(name="C420", boat_length_m=BOAT_LENGTH_M["c420"], beam_m=1.63, table=C420_TABLE)
FJ = Polar(
    name="FJ",
    boat_length_m=BOAT_LENGTH_M["fj"],
    beam_m=1.52,
    table=C420_TABLE,
    scale=FJ_SPEED_SCALE,
    source="estimated: C420 shape scaled by %.2f — see module docstring" % FJ_SPEED_SCALE,
)

CLASSES: dict[str, Polar] = {"c420": C420, "fj": FJ}
