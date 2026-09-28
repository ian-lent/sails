"""Wind field: what the boat feels at a place and a time.

The interface is the point of this module, not the implementations. Every field
answers one question — `at(x, y, t) -> (speed_kt, direction_from_deg)` — so the
simulator never knows whether it is running in flat steady wind, a synthetic
oscillation, or a field reconstructed from measurements. Strategic conclusions are
almost entirely a function of which field is installed, so making them swappable
and labelled is the difference between an experiment and an anecdote.

ON THE SEVERN, AND THE TIMESCALE PROBLEM YOU FLAGGED.

Your legs are 4-7 minutes upwind and 3-5 down. That number is the whole
difficulty with using a shore station, and it is worth being blunt about it:

  * Standard marine station feeds publish 2-minute or 6-minute averages, often
    only hourly for archives. A 6-minute average is roughly ONE ENTIRE LEG. It
    tells you the mean wind and the diurnal pattern. It cannot tell you the shift
    structure inside a leg, and shift structure inside a leg is what strategy on
    a short course consists of.
  * Averaging also destroys the statistic that matters most. A 6-minute mean of a
    12-degree oscillation with a 3-minute period reads as no oscillation at all.
  * So a station gives us the CLIMATOLOGY layer — what to expect at 1400 in May,
    sea breeze onset, the gradient direction — and nothing about the tactical
    layer. Both are needed; only one is freely available.

Closing that gap needs one of:
  * a high-rate anemometer (1 Hz, or 10-second means) logged on station during
    racing, which is the cheapest real fix;
  * inference from GPS tracks — a fleet of boats is a distributed wind sensor, and
    with enough tracks the shift field can be estimated from what boats did
    rather than measured directly. This is the interesting approach and it
    inverts the data problem, since tracks are easier to get than anemometry;
  * a physical model of the river, which for a shore-bound tidal venue means
    shoreline bend, channelling along the river axis, land shadow, sea-breeze
    overlay and tidal current — the Severn has all five. Expensive, and it still
    needs data to validate.

Until a measured field exists, `is_measured` stays False and results carry the
label. The oscillating field below is a scaffold for testing the machinery, not a
model of the Severn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class WindField(Protocol):
    """A wind field answers one question, everywhere, at any time."""

    name: str
    is_measured: bool

    def at(self, x: float, y: float, t: float) -> tuple[float, float]:
        """(speed in knots, direction the wind blows FROM in degrees)."""
        ...


@dataclass
class UniformWind:
    """Steady, spatially flat. The control case.

    Worth keeping and worth running first for every experiment: in uniform wind,
    every strategic question collapses and only boat handling and boat-on-boat
    interaction remain. Anything the simulator claims here is a claim about
    tactics, with strategy held out. It is the cleanest way to tell the two apart.
    """

    speed_kt: float = 8.0
    direction_from: float = 0.0
    name: str = "uniform"
    is_measured: bool = False

    def at(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.speed_kt, self.direction_from


@dataclass
class OscillatingWind:
    """Direction oscillating about a mean, optionally phase-shifted across the course.

    The classic teaching case: a periodic shift means the favoured tack alternates
    and tacking on the shifts pays, as against a persistent trend where it does
    not. Two knobs beyond the obvious:

      * `spatial_wavelength_m` gives the oscillation a phase gradient across the
        course, so the two sides of the beat are not in the same phase at the same
        moment. With it set, a boat's gain depends on WHERE it is, not just when —
        which is the minimum needed for left-versus-right to be a real question.
      * `gust_factor` modulates speed independently of direction. Velocity shifts
        and direction shifts are different tactical animals and conflating them is
        a common modelling error.

    Still a scaffold: a real venue's shifts are neither sinusoidal nor stationary.
    """

    mean_direction: float = 0.0
    amplitude_deg: float = 10.0
    period_s: float = 180.0
    mean_speed_kt: float = 8.0
    gust_factor: float = 0.15
    gust_period_s: float = 95.0
    spatial_wavelength_m: float = 0.0
    # Shifts the oscillation in time. A race that starts on a lift and one that
    # starts on a header are different races, so a sweep that does not vary this is
    # measuring one realisation of the wind and calling it the wind.
    phase_s: float = 0.0
    name: str = "oscillating"
    is_measured: bool = False

    def at(self, x: float, y: float, t: float) -> tuple[float, float]:
        phase = 2.0 * math.pi * (t + self.phase_s) / self.period_s
        if self.spatial_wavelength_m > 0.0:
            # Phase advances across the course, so one side leads the other.
            phase += 2.0 * math.pi * x / self.spatial_wavelength_m
        direction = self.mean_direction + self.amplitude_deg * math.sin(phase)
        # Deliberately a different period from the shift, so speed and direction
        # do not move in lockstep and produce a spuriously simple world.
        gust = 1.0 + self.gust_factor * math.sin(
            2.0 * math.pi * (t + self.phase_s) / self.gust_period_s
        )
        return self.mean_speed_kt * gust, direction % 360.0


@dataclass
class PersistentShift:
    """Direction trending one way through the race.

    The counterpart to oscillation, and the reason the two must be separable: the
    correct play reverses. In a persistent shift you sail toward the new wind and
    accept being temporarily headed; in an oscillation that is exactly wrong. Any
    policy the simulator learns should be tested against both, because a policy
    that cannot tell them apart is worse than no policy.
    """

    start_direction: float = 0.0
    rate_deg_per_min: float = 3.0
    mean_speed_kt: float = 8.0
    name: str = "persistent"
    is_measured: bool = False

    def at(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.mean_speed_kt, (self.start_direction + self.rate_deg_per_min * t / 60.0) % 360.0
