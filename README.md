# sailsim — college fleet racing simulator

Infrastructure for simulating college fleet racing (FJ / C420, no spinnaker) in
order to search for strategic and tactical policy. Severn River is the target
venue. **Nothing here is calibrated to real data yet** — see Provenance.

    python3 tests/test_core.py        # 50 offline checks, no deps beyond numpy
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
| `course.py` | Marks, legs, splits, and `Helm` — the policy layer that gets replaced. |
| `sim.py` | The fleet loop. Takes a `helm_factory` so policies can be swapped. |

## What the model does and does not have

Present: 18 independent agents, each reading wind at its own position and time,
each sailing its own angle at its own speed; polar-driven speed with acceleration
lag; wind-dependent manoeuvre cost; layline-based steering; per-leg splits;
course sizing for conditions.

Absent, in rough order of how much they matter:

1. **Boat-on-boat interaction.** No wind shadow, no backwind, no safe leeward.
   Boats sail through each other. Without this there are no lanes, and without
   lanes there is no reason to care where you start or whom you tack on.
2. **The rules.** No right of way, no mark-room, no penalties. A policy search run
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

## Calibration encoded as tests

Domain knowledge lives in `tests/test_core.py` as assertions rather than comments:

* **Manoeuvre cost** — light air nearly free, medium more, heavy 1-3 boat lengths.
  Currently measures 0.14 / 0.82 / 2.29 BL at 4 / 10 / 18 kt.
* **Leg durations** — 4-7 min upwind, 3-5 min downwind, checked at 6/9/12/16 kt.
* **Sailing efficiency** — the fleet must not beat the tacking geometry, nor sail
  more than 12% over it.
* **Manoeuvre count** — a regression guard against the bug below.

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
