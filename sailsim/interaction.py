"""Boats disturbing each other's wind: shadow, backwind, and clear lanes.

This is what turns the simulator from a time trial into racing. Without it there
is no reason to care where you start, whom you tack on, or which side of another
boat you pass — every strategic question collapses into "sail fast".

TWO DISTINCT EFFECTS, because they behave differently and are escaped differently.

1. SHADOW (blanketing). The sail removes momentum from the air, so downwind of a
   boat there is a cone of slower wind. Modelled as a SPEED DEFICIT.

   The cone lies along the APPARENT wind, which is the single most consequential
   detail in this module. Beating, the apparent wind is around ten degrees further
   forward than the true wind, so the shadow is rotated forward with it — it
   trails aft and to LEEWARD of the boat casting it, not straight down the true
   wind. This is why a boat directly astern on the same tack is only clipped by
   the edge of it, while a boat astern and to leeward is squarely in it, and why
   the escape is to sail higher or tack rather than to bear away. Get this wrong
   and the model produces confident, wrong advice about passing lanes.

2. BACKWIND (the lee-bow effect). A boat's sail does not only absorb wind, it
   DEFLECTS it. To windward and aft of a boat, the flow is bent rather than
   slowed, so a boat there is HEADED. Modelled as a DIRECTION CHANGE, not a speed
   loss, which is what makes it tactically different: you cannot foot through a
   header, you can only tack or sail lower.

   This is the mechanism behind leebowing — putting yourself to leeward and
   slightly ahead so the other boat is headed and has to tack away. It is worth
   saying plainly that the aerodynamics here are debated; what is not debated is
   the tactical phenomenon, and a direction bend reproduces it from a plausible
   mechanism rather than by special-casing the situation.

MAGNITUDES ARE ESTIMATES, like the polar. The shape is defensible — deficit
strongest near the source and decaying with distance, a cone of order fifteen
degrees, effective range of a handful of boat lengths. The numbers are not
measured. Fleet GPS tracks would calibrate this better than anything else: the
observable is how much a boat slows when it crosses behind another, and that is
directly visible in tracks.

ORDER INDEPENDENCE. Every boat's disturbance is computed from the AMBIENT wind in
a first pass, then applied in a second. Without that, boat 1's effective wind
would depend on boat 0 having already been updated this timestep, and the fleet's
behaviour would depend on the order of a list — a bug that is invisible in any
single race and poisons every statistic gathered over many.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import geometry as geo
from .boat import Boat


@dataclass(frozen=True)
class Disturbance:
    """Shape and strength of the wind one boat takes from another.

    Lengths are in BOAT LENGTHS, because that is the unit sailors judge these
    distances in and it keeps the model class-agnostic between the FJ and C420.
    """

    # --- shadow (speed deficit, downwind along the apparent wind) ---
    shadow_length_bl: float = 7.0
    shadow_half_angle_deg: float = 16.0
    # Fraction of wind speed removed at the source. Real blanketing close aboard is
    # near-total; 0.55 is a deliberately conservative starting point.
    shadow_max_deficit: float = 0.55

    # --- backwind (direction bend, to windward and aft) ---
    backwind_length_bl: float = 3.5
    backwind_max_header_deg: float = 8.0

    # Floor on the wind a boat can be left with. Several overlapping shadows in a
    # crowded start would otherwise multiply down to nearly zero, and a boat
    # becalmed mid-fleet is an artefact of naive composition, not an observation.
    min_wind_fraction: float = 0.25

    def shadow_axis(self, source: Boat, tws_kt: float, wind_from: float) -> float:
        """Compass bearing the shadow extends along, from the source boat."""
        apparent_from = geo.apparent_wind_from(source.heading, wind_from, source.speed_kt, tws_kt)
        return geo.wrap360(apparent_from + 180.0)

    def shadow_at(
        self, source: Boat, x: float, y: float, tws_kt: float, wind_from: float
    ) -> float:
        """Fraction of wind speed removed at (x, y) by this source. 0 = clear air."""
        distance = geo.distance(source.x, source.y, x, y)
        reach = self.shadow_length_bl * source.length_m
        if distance <= 1e-6 or distance >= reach:
            return 0.0

        offset = abs(geo.angle_diff(geo.bearing(source.x, source.y, x, y),
                                    self.shadow_axis(source, tws_kt, wind_from)))
        if offset >= self.shadow_half_angle_deg:
            return 0.0

        # Linear decay along the cone, cosine taper across it. The taper matters:
        # a hard edge makes boats flicker between full shadow and clear air as they
        # drift, which shows up as implausible speed chatter in the tracks.
        along = 1.0 - distance / reach
        across = math.cos(math.radians(90.0 * offset / self.shadow_half_angle_deg))
        return self.shadow_max_deficit * along * across

    def backwind_at(
        self, source: Boat, x: float, y: float, tws_kt: float, wind_from: float
    ) -> float:
        """Signed direction bend in degrees applied at (x, y) by this source.

        Positive rotates `wind_from` clockwise. The sign is chosen so the bend
        HEADS a boat on the same tack as the source, which is the tactical content
        of the lee-bow: on starboard a header means the wind comes further forward,
        which is a decrease in `wind_from`.
        """
        distance = geo.distance(source.x, source.y, x, y)
        reach = self.backwind_length_bl * source.length_m
        if distance <= 1e-6 or distance >= reach:
            return 0.0

        twa = source.twa(wind_from)
        if twa == 0.0:
            return 0.0

        # Where the target sits relative to the source's own heading.
        relative = geo.angle_diff(geo.bearing(source.x, source.y, x, y), source.heading)
        # Aft of the beam only: ahead of a boat the flow is upwash, not backwind.
        if abs(relative) <= 90.0:
            return 0.0
        # Windward side only. On starboard tack (twa > 0) the wind comes over the
        # starboard side, so windward is positive relative bearing.
        if (relative > 0.0) != (twa > 0.0):
            return 0.0

        strength = (1.0 - distance / reach) * self.backwind_max_header_deg
        # Heading a starboard-tack boat means decreasing wind_from.
        return -strength if twa > 0.0 else strength


@dataclass
class FleetWind:
    """Ambient wind modified by the fleet, computed order-independently.

    Built fresh each timestep from a snapshot of the fleet. The snapshot is the
    point: sources are frozen at the start of the step, so no boat sees a
    half-updated world and results do not depend on list order.
    """

    disturbance: Disturbance
    sources: tuple[tuple[Boat, float, float], ...]

    @classmethod
    def snapshot(cls, boats, wind, t: float, disturbance: Disturbance) -> "FleetWind":
        """Freeze every boat with the ambient wind it currently sits in."""
        frozen = []
        for b in boats:
            if b.finished_at is not None:
                # A finished boat has left the course; it stops casting a shadow.
                continue
            tws, wdir = wind.at(b.x, b.y, t)
            frozen.append((b, tws, wdir))
        return cls(disturbance=disturbance, sources=tuple(frozen))

    def at(self, target: Boat, tws_kt: float, wind_from: float) -> tuple[float, float, float]:
        """(effective wind speed, effective direction, total deficit) for one boat.

        The deficit is returned as well as applied, because it is the observable
        that makes "lanes" analysable — a boat's finishing position tells you it
        lost, and its accumulated time in dirty air tells you why.
        """
        clear = 1.0
        bend = 0.0
        for source, source_tws, source_dir in self.sources:
            if source.boat_id == target.boat_id:
                continue
            # Multiplicative attenuation: two boats each taking 40% leave 36% of
            # the wind, not 20%. Shadows compose by what they pass through, and
            # summing deficits sends a boat negative in a crowd.
            clear *= 1.0 - self.disturbance.shadow_at(
                source, target.x, target.y, source_tws, source_dir
            )
            bend += self.disturbance.backwind_at(
                source, target.x, target.y, source_tws, source_dir
            )

        clear = max(clear, self.disturbance.min_wind_fraction)
        return tws_kt * clear, geo.wrap360(wind_from + bend), 1.0 - clear
