"""The simulation loop, and the fleet that goes round the course.

Deliberately absent, and each one is a named place to add nuance rather than an
oversight:

  * BOAT-ON-BOAT INTERACTION. No wind shadow, no backwind, no safe leeward. Boats
    currently sail through each other. This is the largest single omission and the
    first thing worth adding, because without it there are no lanes, and without
    lanes there is no reason to care where you start or whom you tack on.
  * THE RULES. No right of way, no mark-room, no penalties. Any policy search run
    against this simulator will learn illegal moves, so rules must land before
    optimisation does.
  * THE START. Boats currently begin on the line at speed. Real college racing is
    decided disproportionately in the thirty seconds either side of the gun, so
    this is the second omission by importance.
  * CURRENT. The Severn is tidal. A cross-course current gradient moves laylines
    and changes line bias, and it is not here.

What IS here is the spine: a fleet of independent agents, each reading the wind at
its own position and time, each sailing its own angle at its own speed, stepped
forward together and scored at the end. That is what everything above bolts onto.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import geometry as geo
from .boat import Boat
from .course import Course, Helm
from .polar import CLASSES, Polar
from .wind import WindField

# A helm factory takes a boat and returns the policy that steers it. Injecting it
# rather than constructing it inside the loop is what will let a search compare
# policies — and what lets a test hold the policy fixed while varying boat speed.
HelmFactory = "Callable[[Boat], Helm]"


@dataclass
class RaceResult:
    """Finish order plus the provenance needed to read it honestly."""

    order: list[tuple[int, str, float | None]]
    elapsed_s: float
    wind_name: str
    polar_name: str
    estimates_used: list[str]
    boats: list[Boat] = field(repr=False, default_factory=list)

    def finishers(self) -> int:
        return sum(1 for _, _, t in self.order if t is not None)

    def table(self) -> str:
        lines = [f"{'pos':>3}  {'boat':<12} {'finish':>8}  {'tacks':>5} {'gybes':>5} {'dist m':>7}"]
        by_id = {b.boat_id: b for b in self.boats}
        for i, (boat_id, name, t) in enumerate(self.order, start=1):
            b = by_id[boat_id]
            finish = f"{t:7.1f}s" if t is not None else "     DNF"
            lines.append(
                f"{i:>3}  {name:<12} {finish:>8}  {b.tacks:>5} {b.gybes:>5} {b.distance_sailed_m:>7.0f}"
            )
        return "\n".join(lines)


def build_fleet(
    size: int = 18,
    boat_class: str = "c420",
    course: Course | None = None,
    seed: int = 7,
    speed_spread: float = 0.025,
) -> list[Boat]:
    """A fleet spread along the start line, with per-crew variation.

    `speed_spread` is the standard deviation of the crew speed factor. At 2.5% the
    fastest and slowest crews differ by roughly 10% — which sounds small and is
    not: over a five-minute beat that is tens of seconds, far more than any single
    tactical decision on this course. Whether 2.5% is right for a college fleet is
    an empirical question your results data can answer, by looking at how much of
    finishing position is explained by crew identity across rotations.
    """
    rng = random.Random(seed)
    polar: Polar = CLASSES[boat_class]
    boats: list[Boat] = []

    if course is None:
        course = Course.windward_leeward()
    (bx, by), (px, py) = course.start_boat, course.start_pin

    for i in range(size):
        # Evenly along the line, mid-line boats neither favoured nor punished —
        # there is no line bias yet because there is no start model.
        f = (i + 0.5) / size
        x = bx + (px - bx) * f
        y = by + (py - by) * f
        boats.append(
            Boat(
                boat_id=i,
                name=f"{boat_class.upper()}-{i + 1:02d}",
                polar=polar,
                x=x,
                y=y,
                speed_factor=max(0.85, rng.gauss(1.0, speed_spread)),
                handling=max(0.7, rng.gauss(1.0, 0.08)),
            )
        )
    return boats


@dataclass
class Simulator:
    course: Course
    wind: WindField
    dt: float = 0.5
    max_time_s: float = 2400.0
    record_every_s: float = 5.0

    def run(self, boats: list[Boat], helm_factory=None) -> RaceResult:
        """Race the fleet. `helm_factory(boat) -> Helm` overrides the default policy.

        The default spreads tack_bias across the fleet so 18 boats do not trace one
        identical line. That is scaffolding, not a model of strategic preference —
        and it means the default fleet differs in DECISIONS as well as speed, which
        is worth remembering when reading any result off it.
        """
        if helm_factory is None:
            def helm_factory(b: Boat) -> Helm:
                return Helm(tack_bias=(b.boat_id % 5 - 2) / 2.0)
        helms = {b.boat_id: helm_factory(b) for b in boats}

        # Point every boat close-hauled on starboard and give it way on. Not a
        # start: a start is a separate model. This just avoids the first ten
        # seconds being boats accelerating from rest in a random direction.
        tws0, wdir0 = self.wind.at(0.0, 0.0, 0.0)
        close_hauled, _ = boats[0].polar.best_upwind(tws0)
        for b in boats:
            b.heading = geo.heading_for_twa(close_hauled, wdir0)
            b.speed_kt = b.target_speed_kt(tws0, wdir0) * 0.9

        t = 0.0
        next_record = 0.0
        while t < self.max_time_s:
            if all(b.finished_at is not None for b in boats):
                break
            recording = t >= next_record
            for b in boats:
                if b.finished_at is not None:
                    continue
                tws, wdir = self.wind.at(b.x, b.y, t)
                mark = self.course.target_mark(b)
                if mark is None:
                    continue
                heading, maneuver = helms[b.boat_id].target_heading(b, mark.x, mark.y, tws, wdir)
                if maneuver is not None:
                    b.begin_maneuver(tws, maneuver)
                b.step(heading, tws, wdir, self.dt)
                self.course.update_progress(b, t)
                if recording:
                    b.record(t)
            if recording:
                next_record += self.record_every_s
            t += self.dt

        # Unfinished boats sort last, by how far round they got — a placeholder for
        # scoring that will need to be real once races are abandoned or time out.
        def key(b: Boat) -> tuple[float, float]:
            if b.finished_at is not None:
                return (0.0, b.finished_at)
            return (1.0, -(b.leg * 1e6 + b.distance_sailed_m))

        ordered = sorted(boats, key=key)
        estimates = []
        if not self.wind.is_measured:
            estimates.append(f"wind field '{self.wind.name}' is synthetic, not measured")
        if not boats[0].polar.is_measured:
            estimates.append(f"polar '{boats[0].polar.name}': {boats[0].polar.source}")

        return RaceResult(
            order=[(b.boat_id, b.name, b.finished_at) for b in ordered],
            elapsed_s=t,
            wind_name=self.wind.name,
            polar_name=boats[0].polar.name,
            estimates_used=estimates,
            boats=boats,
        )
