"""Export a finished race as JSON for the browser replay.

The replay page is a separate thing from the simulator on purpose: races are
generated here, watched there, and the only contract between them is this file.
That means a race can be replayed months later, on another machine, without
Python — and it means the viewer cannot quietly become a second, divergent
implementation of the model.

SIZE IS THE DESIGN CONSTRAINT. An 18-boat race recorded every second is about
24,000 frames, and a naive JSON object per frame (`{"t": 100.0, "x": 59.5, ...}`)
runs to several megabytes of mostly key names. Frames are therefore flat arrays in
a documented order, and values are rounded at the point of recording rather than
here, so the rounding is visible next to the thing being rounded.

Provenance travels with the race. The estimates list from RaceResult goes into the
file and the page shows it, because a replay is exactly where someone is most
likely to forget that the polar is a guess and the wind is synthetic.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from . import start as start_mod
from .course import Course
from .sim import RaceResult
from .wind import WindField

# Bumped when the frame layout changes, so a new page cannot silently misread an
# old file — the same contract the infra-atlas pipeline documents use.
SCHEMA = 1

# The order of values in each frame. Written into the file so the page reads the
# layout rather than assuming it.
FRAME_FIELDS = ["t", "x", "y", "heading", "speed_kt", "deficit", "flags", "leg"]

FLAG_MEANINGS = {
    "1": "returning after being over early",
    "2": "spinning a penalty",
    "4": "finished",
    "8": "was over the line at the gun",
}


def export(
    result: RaceResult,
    course: Course,
    wind: WindField,
    path: str,
    wind_sample_s: float = 5.0,
) -> dict[str, Any]:
    """Write a replay file and return the document."""
    marks = [
        {"name": m.name, "x": round(m.x, 1), "y": round(m.y, 1), "radius": m.radius_m}
        for m in course.marks
    ]

    # The wind is sampled at the course centre only. A spatially varying field is
    # not captured, so the page's wind readout is "the breeze on the course", not
    # what any particular boat is in — which is why the per-boat deficit is carried
    # in the frames instead of being recomputed there.
    duration = max(
        (b.track[-1][0] for b in result.boats if b.track), default=0.0
    )
    start_t = min((b.track[0][0] for b in result.boats if b.track), default=0.0)
    samples = []
    t = start_t
    while t <= duration:
        speed, direction = wind.at(0.0, 0.0, t)
        samples.append([round(t, 1), round(direction, 1), round(speed, 2)])
        t += wind_sample_s

    bias = start_mod.line_bias(course.start_pin, course.start_boat, samples[0][1] if samples else 0.0)

    document = {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "frame_fields": FRAME_FIELDS,
        "flag_meanings": FLAG_MEANINGS,
        "meta": {
            "wind": result.wind_name,
            "polar": result.polar_name,
            "fouls": result.fouls,
            "contacts": result.contacts,
            "ocs": result.ocs_count,
            "favoured_end": result.favoured_end,
            "line_bias_deg": round(bias, 1),
            "start_t": round(start_t, 1),
            "duration_s": round(duration, 1),
            # The whole point of carrying this: a replay is where it is easiest to
            # forget that none of this is measured.
            "estimates": result.estimates_used,
        },
        "course": {
            "marks": marks,
            "start_pin": [round(course.start_pin[0], 1), round(course.start_pin[1], 1)],
            "start_boat": [round(course.start_boat[0], 1), round(course.start_boat[1], 1)],
        },
        "wind_samples": samples,
        "boats": [],
        "order": [
            {"boat_id": bid, "name": name, "finish_s": (round(t, 1) if t is not None else None)}
            for bid, name, t in result.order
        ],
    }

    finish_rank = {bid: i + 1 for i, (bid, _, _) in enumerate(result.order)}
    for b in result.boats:
        document["boats"].append(
            {
                "id": b.boat_id,
                "name": b.name,
                "length_m": round(b.length_m, 2),
                "beam_m": round(b.beam_m, 2),
                "finish_rank": finish_rank.get(b.boat_id),
                "finish_s": round(b.finished_at, 1) if b.finished_at is not None else None,
                "tacks": b.tacks,
                "gybes": b.gybes,
                "fouls": b.fouls,
                "ocs": b.ocs,
                "dirty_air_s": round(b.dirty_air_s, 1),
                "frames": [list(f) for f in b.track],
            }
        )

    with open(path, "w") as fh:
        json.dump(document, fh, separators=(",", ":"))
    return document
