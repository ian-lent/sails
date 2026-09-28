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
from .interaction import Disturbance, FleetWind
from . import start as start_mod
import math

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
    bias: "start_mod.LineBias | None" = None
    favoured_end: str | None = None
    ocs_count: int = 0

    def finishers(self) -> int:
        return sum(1 for _, _, t in self.order if t is not None)

    def table(self) -> str:
        lines = [
            f"{'pos':>3}  {'boat':<12} {'finish':>8}  {'tacks':>5} {'gybes':>5} "
            f"{'dist m':>7} {'dirty s':>8} {'def-s':>7}"
        ]
        by_id = {b.boat_id: b for b in self.boats}
        for i, (boat_id, name, t) in enumerate(self.order, start=1):
            b = by_id[boat_id]
            finish = f"{t:7.1f}s" if t is not None else "     DNF"
            lines.append(
                f"{i:>3}  {name:<12} {finish:>8}  {b.tacks:>5} {b.gybes:>5} "
                f"{b.distance_sailed_m:>7.0f} {b.dirty_air_s:>8.0f} {b.dirty_air_integral:>7.1f}"
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
    # Length of the starting sequence. Zero skips it entirely and drops boats onto
    # the line at speed, which is the control case for measuring what the start is
    # worth. College sequences are longer than this; 90 s is the part where
    # positioning actually decides anything.
    prestart_s: float = 90.0
    start_seed: int = 17
    # How hard the fleet crowds the favoured end. Zero spreads boats evenly, which
    # is what isolates the line's advantage from the dirty air that crowding causes.
    start_crowding: float = 0.65
    # None disables boat-on-boat interaction entirely, which is the control case:
    # any claim that dirty air caused something should be checked by running the
    # same race with this off.
    interaction: Disturbance | None = field(default_factory=Disturbance)

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

        tws0, wdir0 = self.wind.at(0.0, 0.0, 0.0)
        close_hauled, _ = boats[0].polar.best_upwind(tws0)
        pin, line_boat = self.course.start_pin, self.course.start_boat

        plans: dict[int, start_mod.StartPlan] = {}
        if self.prestart_s > 0.0:
            rng = random.Random(self.start_seed)
            plans = start_mod.draw_plans(
                boats, pin, line_boat, wdir0, rng, crowding=self.start_crowding
            )
            for b in boats:
                # Heading FIRST. target_speed_kt reads the boat's current heading,
                # and a freshly built boat is heading 000 — which against a
                # northerly is dead head to wind, so the polar returns zero and the
                # fleet gets placed exactly ON the line instead of below it. Every
                # boat was then over at the gun, which read as a broken controller
                # and was a broken initialisation.
                b.heading = geo.heading_for_twa(close_hauled, wdir0)
                full_kt = b.target_speed_kt(tws0, wdir0)
                b.speed_kt = full_kt * rng.uniform(0.4, 0.9)

                target = plans[b.boat_id].target_point(pin, line_boat)
                # Scatter the fleet below the line, roughly a sequence's run away.
                # Bunched rather than strung out: by the last ninety seconds a fleet
                # is jockeying near the line, not spread over a minute of sailing.
                back = rng.uniform(0.30, 0.60) * self.prestart_s * geo.ms(full_kt)
                rad = math.radians(wdir0)
                b.x = target[0] - math.sin(rad) * back + rng.gauss(0.0, 8.0)
                b.y = target[1] - math.cos(rad) * back + rng.gauss(0.0, 8.0)
        else:
            # Control case: everyone on the line at speed, no sequence.
            for b in boats:
                b.heading = geo.heading_for_twa(close_hauled, wdir0)
                b.speed_kt = b.target_speed_kt(tws0, wdir0) * 0.9

        gun_checked = False
        t = -self.prestart_s
        next_record = 0.0
        while t < self.max_time_s:
            if all(b.finished_at is not None for b in boats):
                break
            recording = t >= next_record
            # Snapshot the fleet BEFORE anyone moves, so every boat is disturbed by
            # the same frozen world and the result cannot depend on list order.
            fleet_wind = (
                FleetWind.snapshot(boats, self.wind, t, self.interaction)
                if self.interaction is not None
                else None
            )
            for b in boats:
                if b.finished_at is not None:
                    continue
                tws, wdir = self.wind.at(b.x, b.y, t)
                if fleet_wind is not None:
                    tws, wdir, deficit = fleet_wind.at(b, tws, wdir)
                    if deficit > 0.01:
                        b.dirty_air_s += self.dt
                        b.dirty_air_integral += deficit * self.dt
                        b.worst_deficit = max(b.worst_deficit, deficit)
                if t < 0.0 and plans:
                    # Pre-start: approach the chosen slot, regulating speed.
                    heading, cap = start_mod.approach_command(
                        b, plans[b.boat_id], pin, line_boat, tws, wdir, -t
                    )
                    b.speed_cap_kt = cap
                    b.step(heading, tws, wdir, self.dt)
                    if recording:
                        b.record(t)
                    continue

                b.speed_cap_kt = None
                if b.returning:
                    # Sail back below the line before racing. The penalty is the
                    # time this costs, not a number added at the end — which is
                    # what makes being over early expensive in the right way.
                    if start_mod.line_side(pin, line_boat, b.x, b.y, wdir) < -2.0:
                        b.returning = False
                    else:
                        rad = math.radians(wdir)
                        heading = geo.wrap360(math.degrees(math.atan2(-math.sin(rad), -math.cos(rad))))
                        b.step(heading, tws, wdir, self.dt)
                        if recording:
                            b.record(t)
                        continue

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
            # No sequence means no gun and no OCS: boats were placed on the line
            # at speed, so flagging them over would be an artefact of the setup.
            if not gun_checked and t >= 0.0 and plans:
                gun_checked = True
                _, gun_dir = self.wind.at(0.0, 0.0, 0.0)
                for b in boats:
                    b.start_side_m = start_mod.line_side(pin, line_boat, b.x, b.y, gun_dir)
                    b.start_fraction = start_mod.line_fraction(pin, line_boat, b.x, b.y)
                    b.start_speed_kt = b.speed_kt
                    if b.start_side_m > 0.0:
                        b.ocs = True
                        b.returning = True

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
        if self.prestart_s > 0.0:
            estimates.append("start timing errors and line bias are drawn, not observed")
        if self.interaction is not None:
            estimates.append("wind shadow and backwind magnitudes are estimated, not measured")
        if not self.wind.is_measured:
            estimates.append(f"wind field '{self.wind.name}' is synthetic, not measured")
        if not boats[0].polar.is_measured:
            estimates.append(f"polar '{boats[0].polar.name}': {boats[0].polar.source}")

        measured = start_mod.line_bias(pin, line_boat, wdir0)
        return RaceResult(
            bias=start_mod.LineBias(measured, float("nan"), float("nan")),
            favoured_end=start_mod.favoured_end(pin, line_boat, wdir0),
            ocs_count=sum(1 for b in boats if b.ocs),
            order=[(b.boat_id, b.name, b.finished_at) for b in ordered],
            elapsed_s=t,
            wind_name=self.wind.name,
            polar_name=boats[0].polar.name,
            estimates_used=estimates,
            boats=boats,
        )
