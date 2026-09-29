"""Offline checks over the simulator spine.

Two kinds of test here, and the second kind is the interesting one:

  * MECHANICAL tests — angle conventions, interpolation, wrapping. These catch the
    silent sign error that would otherwise poison every result.
  * DOMAIN tests — assertions that the model reproduces things a sailor knows to be
    true. The tack-cost test encodes the light/medium/heavy profile supplied by the
    user; the polar-shape tests encode the qualitative facts about a non-spinnaker
    dinghy. These are how expertise enters the codebase as something enforced
    rather than something written in a comment and forgotten.

Run: python3 -m tests.test_core
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sailsim import geometry as geo  # noqa: E402
from sailsim.boat import Boat, _interp_profile  # noqa: E402
from sailsim.course import Course, Helm  # noqa: E402
from sailsim.polar import C420, FJ  # noqa: E402
from sailsim.sim import Simulator, build_fleet  # noqa: E402
from sailsim.wind import OscillatingWind, PersistentShift, UniformWind  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  \033[32m+\033[0m {name}")
    else:
        print(f"  \033[31mx\033[0m {name}" + (f"\n      {detail}" if detail else ""))
        FAILURES.append(name)


def group(name: str) -> None:
    print(f"\n\033[1m== {name} ==\033[0m")


# --- angle conventions -------------------------------------------------------
group("angle conventions")

check("wrap180 keeps 180 positive, so DDW does not flip tack", geo.wrap180(180.0) == 180.0)
check("wrap180 handles the wrap", abs(geo.wrap180(190.0) - -170.0) < 1e-9)
check("wrap360 normalises negatives", abs(geo.wrap360(-10.0) - 350.0) < 1e-9)

# The worked example from the geometry docstring. If this flips, everything about
# right-of-way and favoured-tack logic inverts, and nothing else would notice.
twa = geo.true_wind_angle(heading=0.0, wind_from=45.0)
check("heading 000 with wind from 045 gives TWA +45", abs(twa - 45.0) < 1e-9, f"got {twa}")
check("positive TWA is starboard tack", geo.tack_of(twa) == "starboard")
check("negative TWA is port tack", geo.tack_of(-45.0) == "port")

check(
    "heading_for_twa inverts true_wind_angle",
    all(
        abs(geo.wrap180(geo.true_wind_angle(geo.heading_for_twa(a, 137.0), 137.0) - a)) < 1e-9
        for a in (-170.0, -45.0, 0.0, 45.0, 170.0)
    ),
)

check("bearing due east is 090", abs(geo.bearing(0, 0, 100, 0) - 90.0) < 1e-9)
check("bearing due north is 000", abs(geo.bearing(0, 0, 0, 100)) < 1e-9)

x, y = geo.step_position(0.0, 0.0, 90.0, 10.0, 1.0)
check("stepping east moves +x only", abs(x - 10.0) < 1e-9 and abs(y) < 1e-9)

check("turn rate caps rotation", abs(geo.turn_toward(0.0, 90.0, 20.0, 1.0) - 20.0) < 1e-9)
check("turn takes the short way round", abs(geo.turn_toward(10.0, 350.0, 20.0, 1.0) - 350.0) < 1e-9)

# --- polar shape -------------------------------------------------------------
group("polar shape (estimated tables — shape only)")

check("no-go zone exists inside 35 deg", C420.speed(8.0, 20.0) == 0.0)
check("close-hauled makes way", C420.speed(8.0, 45.0) > 3.0)

check(
    "reaching is faster than beating",
    C420.speed(8.0, 90.0) > C420.speed(8.0, 45.0),
)
check(
    "dead downwind is slower than a broad reach (no spinnaker)",
    C420.speed(8.0, 180.0) < C420.speed(8.0, 140.0),
    f"180: {C420.speed(8.0, 180.0):.2f}  140: {C420.speed(8.0, 140.0):.2f}",
)
check(
    "more wind is never slower",
    all(C420.speed(12.0, a) >= C420.speed(6.0, a) for a in (40, 60, 90, 120, 150, 180)),
)
check("polar is symmetric in TWA sign", abs(C420.speed(8.0, -60.0) - C420.speed(8.0, 60.0)) < 1e-9)
check("the table is not extrapolated above its last row", C420.speed(40.0, 90.0) == C420.speed(20.0, 90.0))
check("FJ is slower than the C420", FJ.speed(8.0, 90.0) < C420.speed(8.0, 90.0))

# The light-air/heavy-air angle asymmetry. Footing for speed in light air and
# pointing higher as it builds is one of the few polar facts that is unambiguous.
light_twa, _ = C420.best_upwind(4.0)
heavy_twa, _ = C420.best_upwind(16.0)
check(
    "best upwind angle is wider in light air than in a breeze",
    light_twa > heavy_twa,
    f"4kt: {light_twa:.1f}  16kt: {heavy_twa:.1f}",
)

dn_twa, dn_vmg = C420.best_downwind(6.0)
check(
    "best downwind VMG beats sailing dead downwind",
    dn_vmg > C420.speed(6.0, 180.0),
    f"best {dn_vmg:.2f} at {dn_twa:.0f} vs DDW {C420.speed(6.0, 180.0):.2f}",
)
check("downwind optimum is a broad reach, not a beam reach", 130.0 < dn_twa < 180.0, f"{dn_twa:.1f}")

# --- the wind actually oscillates -------------------------------------------
group("wind (measured as a boat experiences it, not as configured)")

# Configuring an oscillation is not the same as a boat being in one. Spatial phase,
# the boat's own motion through the field, and the sampling interval can all damp or
# alias it away, and a race in a breeze that turns out to be flat would invalidate
# every strategic conclusion drawn from it while looking completely normal. So
# these sample along a real track rather than trusting the constructor.
import math as _m  # noqa: E402


def sampled(field, seconds=1400, dt=1.0, track=None):
    """(directions, speeds) a boat would see, following an optional path."""
    dirs, speeds = [], []
    for i in range(int(seconds / dt)):
        t = i * dt
        x, y = track(t) if track else (0.0, 0.0)
        spd, direction = field.at(x, y, t)
        dirs.append(geo.wrap180(direction))
        speeds.append(spd)
    return dirs, speeds


steady_dirs, steady_speeds = sampled(UniformWind(speed_kt=8.0, direction_from=30.0))
check("uniform wind really is constant",
      max(steady_dirs) - min(steady_dirs) < 1e-9 and max(steady_speeds) - min(steady_speeds) < 1e-9)

osc = OscillatingWind(mean_direction=0.0, amplitude_deg=12.0, period_s=200.0,
                      mean_speed_kt=8.0, spatial_wavelength_m=1600.0)
# A boat beating up the course and back, so the spatial term is exercised.
beat = lambda t: (120.0 * _m.sin(t / 90.0), 250.0 + 250.0 * _m.sin(t / 300.0))  # noqa: E731
dirs, speeds = sampled(osc, track=beat)
span = max(dirs) - min(dirs)
spread = (sum((d - sum(dirs) / len(dirs)) ** 2 for d in dirs) / len(dirs)) ** 0.5
crossings = sum(1 for i in range(1, len(dirs)) if (dirs[i - 1] < 0) != (dirs[i] < 0))
observed_period = 2 * len(dirs) / max(crossings, 1)
shifted = sum(1 for d in dirs if abs(d) > 6.0) / len(dirs)
print(f"      span {span:.1f}° (configured ±12) · stdev {spread:.1f}° (a pure sine gives 8.5) · "
      f"period {observed_period:.0f}s (configured 200) · {100 * shifted:.0f}% of the time beyond 6°")

check("the oscillation reaches its full configured amplitude",
      span > 20.0, f"span {span:.1f}°, expected about 24°")
check("it is not damped to a wobble",
      spread > 5.0, f"stdev {spread:.1f}°")
check("the period is roughly what was asked for",
      140.0 < observed_period < 280.0, f"{observed_period:.0f}s")
check("a boat spends a serious fraction of the race shifted",
      shifted > 0.35, f"{100 * shifted:.0f}%")

# Speed must move too, and on a DIFFERENT period from direction: velocity shifts
# and direction shifts are different tactical animals, and a field where they move
# in lockstep would make them impossible to tell apart.
check("wind speed oscillates as well as direction",
      max(speeds) - min(speeds) > 1.0, f"{min(speeds):.2f}-{max(speeds):.2f} kt")
check("speed and direction are not in lockstep",
      abs(osc.period_s - osc.gust_period_s) > 30.0,
      f"direction {osc.period_s}s vs gust {osc.gust_period_s}s")

# Phase must actually produce a different race, or a sweep over phases is sampling
# one realisation many times.
early, _ = sampled(OscillatingWind(amplitude_deg=12.0, period_s=200.0, phase_s=0.0), seconds=200)
later, _ = sampled(OscillatingWind(amplitude_deg=12.0, period_s=200.0, phase_s=50.0), seconds=200)
check("phase_s gives a genuinely different realisation",
      max(abs(a - b) for a, b in zip(early, later)) > 5.0)

trend, _ = sampled(PersistentShift(start_direction=0.0, rate_deg_per_min=3.0), seconds=600)
check("a persistent shift trends instead of oscillating",
      all(trend[i] >= trend[i - 1] - 1e-9 for i in range(1, len(trend))) and trend[-1] > 25.0,
      f"ended at {trend[-1]:.1f}°")

# --- manoeuvre cost, against the user's own numbers --------------------------
group("manoeuvre cost (calibrated to supplied light/medium/heavy profile)")


def tack_cost_boat_lengths(tws_kt: float) -> float:
    """Distance lost to one tack, in boat lengths.

    Measured the honest way: run two identical boats, tack one, and difference the
    distance made good to windward once both are back at full speed. Comparing
    against a straight-line boat rather than against the polar means the turn's
    own path cost is included, not just the speed deficit.
    """
    wind_from, dt, settle_s = 0.0, 0.1, 45.0
    close_hauled, _ = C420.best_upwind(tws_kt)

    control = Boat(0, "control", C420)
    control.heading = geo.heading_for_twa(close_hauled, wind_from)
    control.speed_kt = control.target_speed_kt(tws_kt, wind_from)

    tacker = Boat(1, "tacker", C420)
    tacker.heading = control.heading
    tacker.speed_kt = control.speed_kt

    # Let both settle, then tack one onto the mirror-image angle.
    steps = int(settle_s / dt)
    tacker.begin_maneuver(tws_kt, "tack")
    new_heading = geo.heading_for_twa(-close_hauled, wind_from)
    for _ in range(steps):
        control.step(control.heading, tws_kt, wind_from, dt)
        tacker.step(new_heading, tws_kt, wind_from, dt)

    # Progress to windward is +y here, since the wind is from 000.
    return (control.y - tacker.y) / C420.boat_length_m


light = tack_cost_boat_lengths(4.0)
medium = tack_cost_boat_lengths(10.0)
heavy = tack_cost_boat_lengths(18.0)
print(f"      light 4kt: {light:.2f} BL | medium 10kt: {medium:.2f} BL | heavy 18kt: {heavy:.2f} BL")

check("light air tacks are nearly free (under 0.5 BL)", light < 0.5, f"{light:.2f} BL")
check("cost increases monotonically with wind", light < medium < heavy)
check("heavy air costs 1-3 BL as specified", 1.0 <= heavy <= 3.0, f"{heavy:.2f} BL")
check("medium sits between the two", light < medium < heavy and medium < 1.5, f"{medium:.2f} BL")

carry_l, turn_l, rec_l = _interp_profile(4.0)
carry_h, turn_h, rec_h = _interp_profile(18.0)
check(
    "carry-through falls and recovery lengthens as wind builds",
    carry_h < carry_l and rec_h > rec_l and turn_h < turn_l,
    f"light carry {carry_l:.2f}/rec {rec_l:.1f}s  heavy carry {carry_h:.2f}/rec {rec_h:.1f}s",
)
check("a gybe costs less than a tack", tack_cost_boat_lengths(12.0) > 0.0)

# --- course geometry ---------------------------------------------------------
group("course")

course = Course.windward_leeward(beat_length_m=730.0, laps=2, wind_from=0.0)
check("two laps produce windward, leeward, windward, finish", course.leg_count == 4)
check("windward mark is upwind of the start", course.marks[0].y > 700.0)
check("start line is square to the wind", abs(course.start_boat[1]) < 1e-9)
check(
    "line endpoints straddle the centre",
    course.start_boat[0] > 0 > course.start_pin[0],
)

# --- rounding direction ------------------------------------------------------
# The largest correctness bug this model has had. Rounding used to be a bare
# distance test, so boats passed whichever side they arrived on: measured over one
# race, exactly 9 of 18 went each way. Half a fleet rounding backwards means boats
# meeting HEAD ON at the mark, which is impossible on the water and was generating
# most of the fouls near marks.
windward, leeward = course.marks[0], course.marks[1]
check("marks carry a rounding side", windward.rounding == "port")
check("and the bearing of the leg that arrives at them",
      abs(windward.approach_bearing) < 1e-6 and abs(leeward.approach_bearing - 180.0) < 1e-6)
check("the finish is crossed, not rounded",
      course.marks[-1].rounding == "none" and not course.marks[-1].is_rounded)

# Port rounding with a northbound approach: the mark passes down the boat's port
# side, so the boat is EAST of it.
check("a port rounding puts boats to starboard of the approach",
      windward.passed_correct_side(20.0, windward.y)
      and not windward.passed_correct_side(-20.0, windward.y))
check("the leeward mark reverses, because the approach reverses",
      leeward.passed_correct_side(-20.0, leeward.y)
      and not leeward.passed_correct_side(20.0, leeward.y))

# The regression that matters most here: the steering target and the did-it-round
# test were derived separately and came out OPPOSITE, so the gate pulled boats east
# while the test demanded west. Neither was wrong-looking alone. They now share one
# definition, and this asserts they agree for every mark on the course.
for mark in course.marks:
    if not mark.is_rounded:
        continue
    gate = mark.gate_point(2.0 * C420.boat_length_m)
    check(f"{mark.name}: the gate is on the side the test accepts",
          mark.passed_correct_side(*gate),
          f"gate {gate} rejected by its own mark")

rotated = Course.windward_leeward(beat_length_m=500.0, wind_from=90.0)
check(
    "course rotates with the wind (wind from 090 puts the mark to the east)",
    rotated.marks[0].x > 490.0 and abs(rotated.marks[0].y) < 1e-6,
)
check("the rounding side rotates with the course too",
      rotated.marks[0].passed_correct_side(rotated.marks[0].x, rotated.marks[0].y - 20.0))

# A boat that cuts the wrong side has not rounded, however close it came.
wrong_way = Boat(99, "cheat", C420, x=-1.0, y=course.marks[0].y)
wrong_way.leg = 0
course.update_progress(wrong_way, 10.0)
check("passing the wrong side of a mark does not count as rounding it",
      wrong_way.leg == 0, f"leg advanced to {wrong_way.leg}")
right_way = Boat(98, "fair", C420, x=+1.0, y=course.marks[0].y)
right_way.leg = 0
course.update_progress(right_way, 10.0)
check("passing the correct side does", right_way.leg == 1)

# --- the fleet actually races ------------------------------------------------
group("18-boat fleet")

fleet_course = Course.windward_leeward(beat_length_m=730.0, laps=2, wind_from=0.0)
fleet = build_fleet(size=18, course=fleet_course)
check("fleet is 18 boats", len(fleet) == 18)
check("crews differ in speed", len({round(b.speed_factor, 4) for b in fleet}) == 18)
check("boats start spread along the line", max(b.x for b in fleet) - min(b.x for b in fleet) > 100.0)

# Rules and dirty air OFF throughout this section. These checks isolate the boat
# and course models -- "the fastest crew wins" is only true when nothing else can
# intervene, and when rules were switched on by default they duly intervened and
# broke four assertions that were right about physics and silent about their world.
# A test that does not state its world is a test that breaks when the world moves.
PHYSICS_ONLY = dict(interaction=None, rules=False, prestart_s=0.0)
result = Simulator(
    fleet_course, UniformWind(speed_kt=8.0, direction_from=0.0), **PHYSICS_ONLY
).run(fleet)
check("every boat finishes in uniform 8kt", result.finishers() == 18, f"{result.finishers()}/18")

winner_time = result.order[0][2]
check(
    "race duration is plausible for two laps",
    winner_time is not None and 900.0 < winner_time < 1800.0,
    f"winner {winner_time:.0f}s" if winner_time else "no winner",
)
check("boats tacked to get upwind", all(b.tacks >= 2 for b in fleet))
# The regression guard for the 92-tacks-per-race bug. Two laps means two beats and
# two runs, so a layline-sailing boat makes a handful of manoeuvres, not scores of
# them. A generous ceiling: the point is to catch an order-of-magnitude error, not
# to pin the exact count, which shifts wind model.
worst = max(b.tacks + b.gybes for b in fleet)
check(
    "manoeuvre count is sane for two laps (regression: cross-track helm made 92)",
    worst <= 12,
    f"worst boat made {worst} manoeuvres",
)
print(f"      manoeuvres per boat: {sorted(b.tacks + b.gybes for b in fleet)}")
check("boats gybed downwind (no spinnaker: angles, not DDW)", sum(b.gybes for b in fleet) > 0)
# Sailing efficiency, bracketed. A boat cannot beat the tacking geometry, and a
# boat much worse than it is overstanding laylines or wandering. This is the check
# that would catch a helm regression that still finishes but sails a poor course —
# which is exactly what the 92-tack version did.
import math as _math  # noqa: E402

beat_m = _math.hypot(fleet_course.marks[0].x, fleet_course.marks[0].y)
up_twa, _ = C420.best_upwind(8.0)
dn_twa, _ = C420.best_downwind(8.0)
# Two laps: two beats, one full run, one run to the finish.
ideal = 2 * beat_m / _math.cos(_math.radians(up_twa)) + 2 * beat_m / _math.cos(
    _math.radians(180.0 - dn_twa)
)
worst_ratio = max(b.distance_sailed_m for b in fleet) / ideal
check(
    "nobody beats the tacking geometry",
    min(b.distance_sailed_m for b in fleet) >= ideal * 0.97,
    f"shortest {min(b.distance_sailed_m for b in fleet):.0f} m vs ideal {ideal:.0f} m",
)
check(
    "nobody sails more than 12% over the ideal (overstanding guard)",
    worst_ratio < 1.12,
    f"worst {worst_ratio:.3f} x ideal ({ideal:.0f} m)",
)
print(f"      ideal {ideal:.0f} m | fleet {min(b.distance_sailed_m for b in fleet):.0f}"
      f"-{max(b.distance_sailed_m for b in fleet):.0f} m")
check("provenance is reported, not assumed", len(result.estimates_used) >= 2)
check("a run with rules off says so", any("RULES ARE OFF" in n for n in result.estimates_used))

# The fastest crew does NOT have to win the fleet race above, and believing it did
# was my error, not the model's: boats there differ in tack bias and handling as
# well as speed, so they sail different distances. Path length beating raw speed is
# the whole reason tactics exist, so the test has to control for it — differ in
# speed_factor ONLY, and then the fastest must win, because in uniform wind with no
# interaction there is no mechanism by which it could lose.
control_course = Course.windward_leeward(beat_length_m=730.0, laps=2, wind_from=0.0)
control_fleet = build_fleet(size=6, course=control_course, seed=3)
for i, b in enumerate(control_fleet):
    b.handling = 1.0
    b.speed_factor = 0.94 + 0.02 * i
# Identical helms, supplied through the injection point, so the ONLY difference
# between these boats is speed.
same_helm = lambda b: Helm(tack_bias=0.0)  # noqa: E731
control_result = Simulator(
    control_course, UniformWind(speed_kt=8.0, direction_from=0.0), **PHYSICS_ONLY
).run(control_fleet, helm_factory=same_helm)
check(
    "speed-only fleet: the fastest crew wins in uniform wind",
    control_result.order[0][0] == max(control_fleet, key=lambda b: b.speed_factor).boat_id,
    f"winner {control_result.order[0][1]}",
)
# NOT identical counts: build_fleet spreads boats along the start line, so they
# begin at different cross-track positions and legitimately need different numbers
# of tacks. What must hold is that the count is small and tightly grouped.
check(
    "identical helms in uniform wind sail few, similar numbers of tacks",
    max(b.tacks for b in control_fleet) <= 6
    and max(b.tacks for b in control_fleet) - min(b.tacks for b in control_fleet) <= 2,
    f"tack counts {[b.tacks for b in control_fleet]}",
)
check(
    "finish order is exactly speed order when nothing else varies",
    [i for i, _, _ in control_result.order]
    == [b.boat_id for b in sorted(control_fleet, key=lambda b: -b.speed_factor)],
)

# And the converse, which is the finding worth keeping: with speed held equal,
# steering choices alone reorder the fleet. If this ever stops being true the
# simulator has become a time trial and can teach nothing about tactics.
tactical_course = Course.windward_leeward(beat_length_m=730.0, laps=2, wind_from=0.0)
tactical_fleet = build_fleet(size=6, course=tactical_course, seed=11)
for b in tactical_fleet:
    b.speed_factor = 1.0
    b.handling = 1.0
tactical = Simulator(
    tactical_course, OscillatingWind(amplitude_deg=12.0, period_s=200.0), **PHYSICS_ONLY
).run(tactical_fleet)
times = [t for _, _, t in tactical.order if t is not None]
check(
    "with identical speed, steering choices alone spread the fleet",
    len(times) == 6 and (max(times) - min(times)) > 5.0,
    f"spread {max(times) - min(times):.1f}s" if times else "no finishers",
)

# Oscillating wind must actually change the racing, or the wind layer is inert.
osc_course = Course.windward_leeward(beat_length_m=730.0, laps=2, wind_from=0.0)
osc_fleet = build_fleet(size=18, course=osc_course)
osc = Simulator(
    osc_course, OscillatingWind(amplitude_deg=12.0, period_s=200.0), **PHYSICS_ONLY
).run(osc_fleet)
check("fleet finishes in oscillating wind too", osc.finishers() == 18, f"{osc.finishers()}/18")
check(
    "oscillating wind changes the finish order versus uniform",
    [n for _, n, _ in osc.order] != [n for _, n, _ in result.order],
)

# --- leg durations against the specified racing -----------------------------
group("leg durations (spec: 4-7 min upwind, 3-5 min downwind)")

# This is the user's specification of the racing being modelled, so it belongs in
# the gate. It is also the check that caught a 730 m default beat producing a
# 7.9-minute leg and a 27-minute race.
for tws in (6.0, 9.0, 12.0, 16.0):
    spec_course = Course.for_conditions(tws, C420, upwind_minutes=5.5, laps=2, wind_from=0.0)
    spec_fleet = build_fleet(size=4, course=spec_course, seed=5)
    for b in spec_fleet:
        b.speed_factor = 1.0
        b.handling = 1.0
    Simulator(spec_course, UniformWind(speed_kt=tws, direction_from=0.0), **PHYSICS_ONLY).run(
        spec_fleet, helm_factory=lambda b: Helm(tack_bias=0.0)
    )
    # Splits: mark 0 ends beat one, mark 1 ends run one, mark 2 ends beat two.
    winner = min(spec_fleet, key=lambda b: b.finished_at or 1e9)
    splits = winner.leg_times
    if len(splits) < 3:
        check(f"{tws:.0f} kt: winner completed three legs", False, f"only {len(splits)} splits")
        continue
    beat1 = splits[0] / 60.0
    run1 = (splits[1] - splits[0]) / 60.0
    beat2 = (splits[2] - splits[1]) / 60.0
    print(f"      {tws:>4.0f} kt: beat1 {beat1:.1f} min | run1 {run1:.1f} min | beat2 {beat2:.1f} min")
    check(f"{tws:.0f} kt: upwind leg inside 4-7 min", 4.0 <= beat1 <= 7.0, f"{beat1:.1f} min")
    check(f"{tws:.0f} kt: downwind leg inside 3-5 min", 3.0 <= run1 <= 5.0, f"{run1:.1f} min")

print()
if FAILURES:
    print(f"\033[31m{len(FAILURES)} check(s) failed:\033[0m " + ", ".join(FAILURES))
    raise SystemExit(1)
print("\033[32mall checks passed\033[0m")
