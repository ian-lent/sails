# sailsim — college fleet racing simulator

Infrastructure for simulating college fleet racing (FJ / C420, no spinnaker) in
order to search for strategic and tactical policy. Severn River is the target
venue. **Nothing here is calibrated to real data yet** — see Provenance.

    python3 tests/test_core.py        # spine: geometry, polar, manoeuvres, course
    python3 tests/test_interaction.py # shadow, backwind, lanes
    python3 tests/test_start.py       # line bias, the approach, OCS
    python3 tests/test_rules.py       # right of way, mark-room, penalties
    python3 tests/test_policy.py      # the tactical helm
    python3 experiments/shift_threshold.py   # when is a shift worth a tack?
    python3 make_replay.py                   # writes out/replay.json
    # then open replay/index.html and load that file
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
| `policy.py` | The tactical helm: tacks on shifts, works to keep its air clear. |
| `replay.py` | Export a finished race as JSON for the browser replay. |
| `replay/index.html` | The replay viewer. No dependencies, no build step, one file. |
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

## The policy layer

`course.Helm` is the blind baseline — laylines only, ignores the fleet. It stays,
because it is the control. `policy.TacticalHelm` is the thing a search tunes, and
every decision it makes is governed by a named parameter rather than a constant
buried in a branch: the point of the project is decision rules a sailor can carry
onto the water, and a policy whose behaviour cannot be stated in a sentence cannot
produce one.

**One idea does the work.** Every tack and gybe decision reduces to: *which tack
points closer to the mark?* That single comparison is "tack on the headers" in an
oscillating breeze, "sail the long tack" when you are off to one side, and the
correct rule downwind with no modification — because a shift that heads you on a
beat lifts you on a run, and the geometry handles it.

**Plus one condition that turned out to matter more than the threshold.** A
tactical tack must be justified by an actual header against a running mean of the
wind, not by geometry alone. Without it, the favoured tack flips as the boat
crosses the rhumb line, so it tacks, overshoots, and tacks back: **26 manoeuvres in
a breeze with no shifts in it at all**. With it, a steady breeze produces 8 — the
layline-only figure. The running mean is a circular average (the mean of 350° and
10° is 0°, not 180°), and its time constant is a strategy choice, not a sensor
setting: too short and nothing reads as a shift, too long and a persistent trend
reads as a header for minutes.

### Result: when is a shift worth a tack?

`experiments/shift_threshold.py` sweeps the threshold over ten wind phases with a
**single boat** — no fleet, no dirty air, no rules — because the question is about
the wind and the cost of tacking, and traffic would contaminate it.

| | 10 kt | 18 kt |
|---|---|---|
| Never tack tactically | 1181 s | 1090 s |
| Threshold 55° (misses real shifts) | 1170 s | 1097 s |
| **Responsive (0–30°)** | **1117–1127 s** | **1046–1057 s** |

**The robust finding is that ignoring shifts costs 50–80 seconds over two laps.**
The exact threshold is second-order: the curve is flat from about 0° to 30°, and
differences inside that band are a few seconds against a standard deviation of
five or more. An earlier draft claimed the optimum rises with wind speed; the
measurement after the header requirement says otherwise, and the claim is gone
rather than tuned until it passed.

### Head to head

Mixed fleet, crew speed equalised so **only policy differs**, dirty air and rules
on, 20 races: tactical **7.56** vs baseline **9.44** mean finish rank — a gap of
**+1.89 ± 0.54 places (about 3.5σ)**.

Worth knowing how that number moved. Before tactical tacks required a header, the
same comparison gave **+0.35 ± 0.61** — indistinguishable from zero, because the
policy was tacking away its own gains. One condition took it from noise to a solid
effect.

## Watching a race

```
python3 make_replay.py                  # 18 boats, oscillating 8 kt -> out/replay.json
python3 make_replay.py --baseline       # the blind helm, for comparison
python3 make_replay.py --wind uniform --kt 14 --bias 12
```

Then open `replay/index.html` and load the file. It is a **file picker, not a
fetch**: browsers block `fetch()` on `file://`, so a page that auto-loaded its data
would only work behind a web server. This one works by double-clicking it, and the
file never leaves your machine.

Play/pause (or space), scrub, 1×–16×, arrow keys to step. Boats are drawn as hulls
pointing where they are going, coloured **green on starboard and red on port** —
the colours already in a sailor's head — or by dirty air, or by finishing position.
Purple is a boat spinning a penalty; an amber ring is a boat returning after being
over early. Click a boat for its tack, TWA, speed, leg and wind deficit.

The viewer reads `frame_fields` from the file rather than assuming the frame
layout, and refuses a schema it does not know, so an old file cannot silently
render as boats sailing backwards. A two-lap race at 1 s resolution is about 1 MB.

Two bugs the replay caught within a minute of first rendering, both invisible in
every test and plot up to that point:

* **The whole fleet started 40–70 m to leeward of the pin.** Boats were set up
  straight downwind of their target, but a close-hauled approach also travels
  sideways — about 50 m left over a 60 m run to windward. Real crews set up to
  leeward and behind and reach up; the fleet now backs down the reciprocal of the
  course it will actually sail.
* Mark labels stacked illegibly, because a two-lap course rounds the same buoy
  twice and the marks share coordinates.

## Mark roundings

Every mark carries a **rounding side** (`port` by default, the standard W/L
rounding) and the bearing of the leg that arrives at it — the approach direction
is what decides which side a port rounding puts a boat on, so it cannot be derived
from the mark alone. Boats steer at a **gate point** a length and a half to the
correct side, not at the buoy, which turns the rounding into a queue instead of
funnelling the fleet onto one point from every direction. A boat that cuts the
wrong side has not rounded, and its gate pulls it back around.

The steering target and the did-it-round-it test are derived from **one** vector,
because the first version derived them separately and they came out opposite: the
gate pulled boats east of the mark while the test demanded west. Neither looked
wrong alone.

### The duck

A port-tack boat approaching the windward mark inside 14 lengths, with a
starboard-tacker crossing ahead, **bears away and passes astern** — provided there
is space behind the starboard layline. If it can already lay the mark on starboard
it is *at* the layline, and ducking from there would sail past it, so the
manoeuvre declines and the ordinary layline logic tacks instead. That single
condition is what stops the duck becoming an automatic overstand.

This lives in the policy, not the rules, and the distinction matters: rule 10
already forces a port boat to keep clear, and `rules.avoidance` already ducks — but
only once the boats are nearly converging, as a last-second obligation. This is
the deliberate version: bear away early, take the transom cleanly, arrive at the
layline with speed. On the water that is worth several boat lengths.

### What it was worth

| | before | after |
|---|---|---|
| Boats rounding the **correct** side | **9 of 18** | **18 of 18** |
| Fouls within 45 m of a windward mark | 7 | **3** |
| Total fouls per race | 23 | 19 |

Peak crowding is 5 boats within 4 lengths of the mark — less congested than a real
windward mark, so the remaining fouls are not a density artefact.

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
* **Not asserted, deliberately:** that being over early costs places. Measured
  across 12 races it is −0.17 ± 1.10 (n=9) — nothing. Either the OCS rate is too
  low to measure, or the model's recovery is too cheap: a boat barely over sails
  back a few metres and rejoins, where the real cost is losing your lane and
  having the fleet roll over you. An earlier version asserted it on one race.
* **Right of way** — every determination above, in explicit geometry, plus
  precedence (rule 10 outranks overlap; rule 13 outranks rule 11).
* **Rounding** — that the gate and the side test agree for every mark on the
  course, that cutting the wrong side does not count as rounding, and that the
  duck fires below the layline, declines at it, and is always a bear-away.
* **The shift-threshold curve** — that ignoring shifts is expensive and that a
  threshold high enough to miss real shifts also loses. The flat middle of the
  curve is deliberately *not* asserted.
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
* **Half the fleet rounding marks backwards.** Rounding was a bare distance test,
  so boats passed whichever side they arrived on — exactly 9 of 18 each way, which
  means boats meeting head on at the buoy. It survived five test suites and was
  found by scrubbing the replay to a rounding and measuring what it showed. It
  also left rule 18 resting on nothing, since "the inside boat" is undefined until
  there is a side to be inside of.
* **A policy that tacked away its own gains.** "Sail the tack pointing closer to
  the mark" is correct, and near the rhumb line the answer flips every few seconds.
  With a default threshold of 8° it made 40 manoeuvres in steady wind, and its
  head-to-head advantage measured as noise. Requiring an actual header fixed both.
* **An assertion that was right to refuse.** An earlier draft declined to claim
  the tactical helm beat the baseline, because at the time it measured +0.35 ± 0.61.
  It became assertable only after the model changed, not after the test was retried.
* **A vacuous assertion.** `check(..., R.hulls_touching(a, b) or True)` passes
  forever. Worse than no test.
* **The favoured end worth exactly nothing.** `line_side` measured from the line's
  *midpoint* along the wind axis, so on a biased line a boat at the favoured end
  read as already over and the controller held it back, neutralising the advantage
  for precisely the boats trying to use it. Found because a null result there is
  absurd, not because anything crashed.
