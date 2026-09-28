# sailsim — college fleet racing simulator

Infrastructure for simulating college fleet racing (FJ / C420, no spinnaker) in
order to search for strategic and tactical policy. Severn River is the target
venue. **Nothing here is calibrated to real data yet** — see Provenance.

    python3 tests/test_core.py        # spine: geometry, polar, manoeuvres, course
    python3 tests/test_interaction.py # shadow, backwind, lanes
    python3 tests/test_start.py       # line bias, the approach, OCS
    python3 tests/test_rules.py       # right of way, mark-room, penalties
    python3 plot_shadow.py            # draw the disturbance field
    python3 run_demo.py               # 18 boats, oscillating wind, writes a plot
    python3 run_demo.py uniform 12    # steady 12 kt
    python3 run_demo.py persistent 8  # a righty through the race

## Layout

| Module | Role |
|---|---|
| `geometry.py` | Angle conventions, stated once and tested. Bearings, signed TWA, tack sign. |
| `polar.py` | Boat speed from (TWS, TWA). **Estimated shapes**, with VMG optima cached. |
| `wind.py` | Wind field interface: uniform, oscillating, persistent. Swappable by design. |
| `boat.py` | Agent state, kinematics, and the manoeuvre-cost model. |
| `interaction.py` | Wind shadow, backwind, and lane quality. Boats disturbing each other. |
| `start.py` | Line bias, the pre-start approach, and being over early. |
| `rules.py` | RRS Part 2: right of way, mark-room, contact, penalties. |
| `course.py` | Marks, legs, splits, and `Helm` — the policy layer that gets replaced. |
| `sim.py` | The fleet loop. Takes a `helm_factory` so policies can be swapped. |

## What the model does and does not have

Present: 18 independent agents, each reading wind at its own position and time,
each sailing its own angle at its own speed; polar-driven speed with acceleration
lag; wind-dependent manoeuvre cost; layline-based steering; per-leg splits;
course sizing for conditions.

Absent, in rough order of how much they matter:

1. **The rules.** No right of way, no mark-room, no penalties. A policy search run
   against this simulator would learn illegal moves, so rules must land before
   optimisation does.
3. **The start.** Boats begin on the line at speed. College racing is decided
   disproportionately in the thirty seconds either side of the gun.
4. **Current.** The Severn is tidal. Cross-course current moves laylines and line
   bias.

## Provenance

Two things are estimated and both are labelled at runtime — `RaceResult` carries
`estimates_used` and the demo prints it:

* **Polars are shapes, not measurements.** No published VPP exists for a college
  CFJ or non-spinnaker C420. The tables have the right qualitative form for the
  boat type; the magnitudes are not to be trusted. GPS tracks would fix this.
* **Wind fields are synthetic.** The oscillating and persistent fields are
  scaffolds for testing machinery, not models of the Severn.

**The timescale problem.** Legs are 4-7 minutes upwind, 3-5 down. Standard marine
stations publish 2- or 6-minute averages — roughly one whole leg — which gives
climatology and destroys exactly the shift structure strategy consists of. Closing
that gap needs high-rate anemometry on station, or inference from GPS tracks, or a
physical model of the river. See the `wind.py` docstring.

## The start

The line carries a bias drawn per race, magnitude **1–15 degrees**, from two
sources modelled separately because they behave differently:

* **Committee error** is fixed — the line is laid by eye off a boat swinging on
  its anchor. Once read, it stays read.
* **Wind shift since the line was laid** is not. On a shifty river it can exceed
  the committee's error several times over, and it is the half that punishes
  reading the line early and not looking again.

Why it matters: eight degrees on a 140 m line puts one end **19 m — about 4.6 boat
lengths — upwind** of the other, handed out free at the gun.

The sequence has two phases. Boats hold station below the line with no way on
(luffing needs no special case — the polar already returns zero speed inside the
no-go zone), then bear away and accelerate at a lead time computed from how far
below the line they are. The crew's error is in *when* they start that approach:
early risks being over, late means starting in a hole. Acceleration lag does the
rest, so a boat that goes late is still slow at the gun even if it is on the line.

Boats over at the gun are OCS and must sail back below the line before racing.
**The penalty is the time that costs, not a number added at the end.**

Measured behaviour: ~1 boat over per 18-boat race, median boat about two lengths
below the line at the gun, gun speeds ranging from zero (a blown start) to full.
Starting near the favoured end is worth **+0.62 places** on average over 14 races.

That number is lower than sailing intuition suggests for a 29 m advantage, and the
likely reasons are all model limitations worth knowing: boats do not fight for the
favoured end, the helm does not use the advantage strategically, and crew speed
variance is large relative to it. A good calibration target.

`Simulator(prestart_s=0)` skips the sequence entirely — the control case for
measuring what the start is worth.

## The rules

RRS Part 2, the right-of-way core: **rule 10** (port keeps clear of starboard),
**11** (windward keeps clear of leeward), **12** (clear astern keeps clear),
**13** (tacking), **14** (avoid contact, binding *both* boats), **18** (mark-room
on a three-length zone) and **44** (the two-turns penalty), plus the definitions
they rest on — clear astern, overlap, windward/leeward, the zone.

A penalty is 720° of turning at the boat's own turn rate, so it costs 24 s for a
quick-turning boat and 48 s for a slow one — more expensive in a breeze, as on the
water.

**Not implemented, and each changes real outcomes:** rules 15 (acquiring right of
way), **17 (proper course — the rule that constrains the leebow, and the most
important omission)**, 16, 19, 20, 21, 22, 30, 31, and 42 (propulsion — without it
an optimiser may learn to pump). There are also no protests: a foul here is
detected geometrically and penalised immediately, which is closer to umpired team
racing than to protest-based fleet racing, so **treat the foul rate as an upper
bound**.

`Simulator(rules=False)` disables enforcement and the run says so loudly in its
provenance. A policy search must never be run against it.

## Calibration encoded as tests

Domain knowledge lives in `tests/test_core.py` as assertions rather than comments:

* **Manoeuvre cost** — light air nearly free, medium more, heavy 1-3 boat lengths.
  Currently measures 0.14 / 0.82 / 2.29 BL at 4 / 10 / 18 kt.
* **Leg durations** — 4-7 min upwind, 3-5 min downwind, checked at 6/9/12/16 kt.
* **Sailing efficiency** — the fleet must not beat the tacking geometry, nor sail
  more than 12% over it.
* **Manoeuvre count** — a regression guard against the bug below.
* **Line bias** — that a line set at N degrees measures N, that the favoured end
  is the upwind one, that the advantage matches `L·sin(θ)`, and that a wind shift
  can reverse which end is favoured.
* **The favoured end pays**, with a margin rather than a bare inequality.
* **Right of way** — every determination above, in explicit geometry, plus
  precedence (rule 10 outranks overlap; rule 13 outranks rule 11).
* **Shadow geometry** — that it trails aft and to leeward, that the windward lane
  is clear, that backwind heads a boat on the same tack, and that port mirrors
  starboard exactly.

## Bugs worth remembering

* **92 tacks per race.** The first helm steered by cross-track error, producing a
  boat that zigzagged up the rhumb line. Every tack was paid for, so the race was
  physically consistent and completely unlike sailing. Fixed by using real layline
  geometry; guarded by a test on manoeuvre count.
* **1000 gybes per race.** The layline test was applied with the same inequality
  upwind and downwind. It reverses: upwind you lay when the required angle is
  *wider* than close-hauled, downwind when it is *narrower* than your running
  angle. Using the upwind form downwind made both gybes "layable" at once.
* **Manoeuvres charged four times each.** Detection compared desired heading to the
  boat's actual heading, which lags because turn rate is capped — so the same tack
  was detected on every timestep of the turn. A helm knows when it decided to tack.
* **730 m default beat.** Gave a 7.9-minute leg, outside the specified window, and
  a 27-minute race. Now sized from the polar for a target leg duration.
* **The whole fleet OCS by 45–95 m.** The first approach controller regulated speed
  against distance to a target *point* while always steering close-hauled, so boats
  sailed straight through the line. Distance below the *line*, along the wind axis,
  is what decides whether a boat is over.
* **Every boat placed on the line instead of below it.** `back` was computed from
  `target_speed_kt` before the boat's heading was set, and a fresh boat heads 000 —
  dead head to wind against a northerly — so the polar returned zero. It read as a
  broken controller and was a broken initialisation.
* **7,911 collisions in one race.** Contact was counted per *timestep*, not per
  episode, and nothing separated boats once they overlapped — so they fouled, spun
  a penalty on the spot, finished it still touching, and fouled again. Fixed with
  contact episodes, a separation heading, penalised boats sailing clear, and the
  half of rule 14 that binds the right-of-way boat. Chain: 7911 → 80 → 60 → 27.
* **Four control tests that did not state their world.** "The fastest crew wins"
  is only true when nothing can intervene; switching rules on by default duly made
  things intervene and broke assertions that were right about physics and silent
  about their assumptions. Each now sets its own world explicitly.
* **A vacuous assertion.** `check(..., R.hulls_touching(a, b) or True)` passes
  forever. Worse than no test.
* **The favoured end worth exactly nothing.** `line_side` measured from the line's
  *midpoint* along the wind axis, so on a biased line a boat at the favoured end
  read as already over and the controller held it back, neutralising the advantage
  for precisely the boats trying to use it. Found because a null result there is
  absurd, not because anything crashed.
