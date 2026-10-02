# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-demand clips of the demonstration: ``view_demo_clip``.

``view_task_video`` hands the planner a map of the demonstration - its phases and the
moments where the gripper opens or closes. This tool serves the map's destinations: a
short, densely sampled strip of the frames LIBERO actually recorded during the human
teleoperation, so the planner can watch HOW a grasp or a transport was done at the point
where it is about to do the same thing itself.

The strip is one image: up to 20 frames laid out in a grid, each captioned with its step
and gripper state. The frames are the recorded 128 px observations, not a re-render, and
the store holds them as memory-mapped arrays so a clip reads only the steps it needs.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from rpent.tools.toolkit import readonly
from rpent.utils.logging import get_logger

logger = get_logger("libero.demo_clip")

#: Frames in one strip, and how many of them sit on a row.
MAX_CLIP_FRAMES = 20
GRID_COLS = 5

CLIP_CAVEAT = (
    "These are the frames the human teleoperator's cameras recorded, at the resolution "
    "they were recorded at (128 px), in the demonstration's own layout. Read the MOTION "
    "from them - approach direction, wrist orientation, where on the object the fingers "
    "close, how far the gripper descends before closing, what moves first after it "
    "closes, how the object is set down before release. Do NOT read positions from them: "
    "re-localize every entity in your own camera images."
)

TOOL_SPEC: dict[str, Any] = {
    "name": "view_demo_clip",
    "description": (
        "Watch one short clip of the demonstration at full frame rate, instead of the "
        "sparse keyframes. Ask for a MOMENT (the ~2 s around a grasp or a release, e.g. "
        "moment='grasp 1') when you are about to close or open the gripper, or a PHASE "
        "(a whole interval between gripper transitions, e.g. phase=3) when you want to "
        "see how a stretch of motion was executed. view_task_video lists both. "
        "Read-only: it does not move the robot."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "moment": {
                "type": "string",
                "description": "Moment name from view_task_video, e.g. 'grasp 1' or 'release 2'.",
            },
            "phase": {
                "type": "integer",
                "minimum": 1,
                "description": "Phase id from view_task_video, e.g. 3.",
            },
            "camera": {
                "type": "string",
                "enum": ["agentview", "wrist", "both"],
                "description": (
                    "Which view to show; 'both' stacks wrist under agentview. Defaults to "
                    "'both' for a moment (the wrist view is where the contact is legible) "
                    "and 'agentview' for a phase (gross motion across the scene)."
                ),
            },
        },
        "required": [],
    },
}


def _pick_steps(a: int, b: int, limit: int = MAX_CLIP_FRAMES) -> list[int]:
    """Evenly spaced step indices covering ``[a, b)``, at most ``limit`` of them."""
    span = max(1, b - a)
    n = min(limit, span)
    return sorted({int(s) for s in np.linspace(a, b - 1, num=n)})


def _gripper_at(phases: list[dict], step: int) -> str:
    for ph in phases:
        lo, hi = ph.get("steps", [0, 0])
        if lo <= step < hi:
            return str(ph.get("gripper", "?"))
    return "?"


def _strip(
    tiles: list[tuple[int, str, list[np.ndarray]]],
    *,
    title: str,
    side: int,
) -> bytes:
    """Compose the captioned grid; ``tiles`` is (step, gripper, [views]) per frame."""
    from PIL import Image, ImageDraw

    pad, cap, head = 4, 13, 15
    rows = -(-len(tiles) // GRID_COLS)
    views = max(1, len(tiles[0][2]))
    cell_h = side * views + cap
    w = GRID_COLS * (side + pad) + pad
    h = head + rows * (cell_h + pad) + pad
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((pad, 3), title, fill=(20, 20, 20))
    for i, (step, grip, frames) in enumerate(tiles):
        x = pad + (i % GRID_COLS) * (side + pad)
        y = head + (i // GRID_COLS) * (cell_h + pad)
        draw.text((x + 1, y + 1), f"s{step} {grip}", fill=(20, 20, 20))
        for v, arr in enumerate(frames):
            img.paste(Image.fromarray(np.asarray(arr, dtype=np.uint8)), (x, y + cap + v * side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def clip(
    root: Path,
    index: dict[str, Any],
    *,
    moment: str | None = None,
    phase: int | None = None,
    camera: str | None = None,
) -> dict[str, Any]:
    """Payload for one clip: what it is, which steps it covers, and the strip image."""
    phases = list(index.get("phases") or [])
    moments = list(index.get("moments") or [])
    if not phases:
        return {"error": "this demonstration has no clip data; use view_task_video"}

    chosen: dict[str, Any] | None = None
    if moment is not None:
        want = str(moment).strip().lower()
        chosen = next((m for m in moments if str(m.get("name", "")).lower() == want), None)
        if chosen is None:
            return {
                "error": f"no moment named {moment!r}",
                "moments": [m.get("name") for m in moments],
                "phases": [p.get("id") for p in phases],
            }
        kind = str(chosen.get("kind", ""))
        joint = chosen.get("joint") or {}
        if kind == "end":
            what = f"the last {chosen['seconds']}s of the successful demonstration"
        elif kind == "articulation":
            what = (
                f"moment '{chosen['name']}' - the fixture is moved at step "
                f"{chosen['step']} (joint {joint.get('from')} -> {joint.get('to')})"
            )
        else:
            what = f"moment '{chosen['name']}' - the gripper {kind}s at step {chosen['step']}"
    elif phase is not None:
        chosen = next((p for p in phases if int(p.get("id", -1)) == int(phase)), None)
        if chosen is None:
            return {
                "error": f"no phase with id {phase}",
                "phases": [p.get("id") for p in phases],
                "moments": [m.get("name") for m in moments],
            }
        what = f"phase {chosen['id']} - {chosen['label']}"
    else:
        return {
            "error": "pass either moment= or phase=",
            "moments": [m.get("name") for m in moments],
            "phases": [p.get("id") for p in phases],
        }

    a, b = (int(v) for v in chosen["steps"])
    steps = _pick_steps(a, b)
    if camera is None:
        camera = "both" if moment is not None else "agentview"
    wanted = ["agentview", "wrist"] if camera == "both" else [camera if camera in ("agentview", "wrist") else "agentview"]
    arrays = {}
    for name in wanted:
        path = root / f"{name}.npy"
        try:
            arrays[name] = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            logger.warning("clip array unreadable: %s (%s)", path, exc)
            return {"error": f"clip frames for {name} are not in the store"}

    side = int(index.get("clip_side") or next(iter(arrays.values())).shape[1])
    tiles = [(s, _gripper_at(phases, s), [arrays[n][s] for n in wanted]) for s in steps]
    hz = int(index.get("control_hz") or 20)
    title = (
        f"{index.get('video_id', 'demo')}  |  {what}  |  steps {a}-{b} "
        f"({(b - a) / hz:.1f}s at {hz} Hz, {len(steps)} frames shown)  |  {'+'.join(wanted)}"
    )
    return {
        "video_id": index.get("video_id"),
        "demonstrated_task_language": index.get("task_language"),
        "showing": what,
        "steps": [a, b],
        "seconds": round((b - a) / hz, 1),
        "frames_shown": len(steps),
        "stride": max(1, round((b - a) / max(1, len(steps)))),
        "camera": "+".join(wanted),
        # What the closure actually achieved: the human regrasps, so a moment named
        # "grasp 1" is not necessarily sub-goal 1, and only this says which it was.
        **({"while_closed": chosen["while_closed"]} if "while_closed" in chosen else {}),
        "caveat": CLIP_CAVEAT,
        "_image_series_bytes": [_strip(tiles, title=title, side=side)],
    }


@readonly
def view_demo_clip(
    moment: str | None = None,
    phase: int | None = None,
    camera: str | None = None,
    *,
    video: Any,
) -> dict[str, Any]:
    """Tool handler: one densely sampled clip of the cell's demonstration."""
    return clip(video.root, video.index, moment=moment, phase=phase, camera=camera)


# --------------------------------------------------------------------------------------
# The map: the demonstration's structure, in text, with no images at all.
# --------------------------------------------------------------------------------------

TOOL_SPEC_MAP: dict[str, Any] = {
    "name": "view_task_map",
    "description": (
        "Read the structure of this task's demonstration: how long it is, its phases "
        "(the intervals between gripper transitions) and its moments (where the gripper "
        "closes or opens). Returns TEXT ONLY - no images. Call it before you plan, then "
        "call view_demo_clip to watch the phases and moments that matter. Read-only."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

MAP_HOW_TO_USE = (
    "This map has no pictures in it. To see the scene or the motion, pull a clip: "
    "view_demo_clip({'moment': '<name>'}) for the ~2 s around a moment, "
    "view_demo_clip({'phase': <id>}) for a whole interval. Each phase says what it "
    "ACHIEVED - which object it moved and into which region, or which fixture it opened "
    "or closed - so match the phase to the sub-goal you are about to attempt and watch "
    "THAT one. `task_goal` below is the success condition of the demonstrated episode, "
    "stated as predicates over `task_objects` and `task_regions`; every predicate in it "
    "corresponds to a phase here. Moments are the instants inside those phases: a gripper "
    "closing or opening, or a fixture being moved."
)


def task_map(index: dict[str, Any], *, cross_task: bool = False) -> dict[str, Any]:
    """Text payload: provenance, caveats, phases, moments and a coarse trajectory sketch."""
    from robots.libero import task_video as tv

    phases = list(index.get("phases") or [])
    moments = list(index.get("moments") or [])
    if not phases:
        return {"error": "this demonstration has no map; the store predates build_clips.py"}
    hz = int(index.get("control_hz") or 20)
    n_steps = int(index.get("n_steps") or 0)
    # The keyframe rows carry a usable trajectory sketch, but their `event` labels come
    # from a pick-and-place heuristic that mislabels non-transport manipulation (a knob
    # turn reads as "lifted"). The phases below say the same thing correctly, so the
    # sketch ships without them.
    sketch = [
        {k: f[k] for k in ("step", "t_frac", "gripper", "eef_xyz") if k in f}
        for f in (index.get("frames") or [])
    ]
    payload = {
        "video_id": index.get("video_id"),
        "demonstrated_task_language": index.get("task_language"),
        "source": index.get("source"),
        "caveat": tv.CROSS_TASK_LAYOUT_CAVEAT if cross_task else tv.CAVEAT,
        "task_goal": index.get("task_goal"),
        "task_objects": index.get("task_objects"),
        "task_regions": index.get("task_regions"),
        "n_steps": n_steps,
        "control_hz": hz,
        "seconds": round(n_steps / max(1, hz), 1),
        "phases": phases,
        "moments": [{k: v for k, v in m.items() if k != "id"} for m in moments],
        "trajectory_sketch": sketch,
        "how_to_use": MAP_HOW_TO_USE,
    }
    if cross_task:
        payload["your_instruction"] = "read task_language from view_env_state; it may differ"
        payload["if_they_differ"] = tv.CROSS_TASK_CAVEAT
    return payload


@readonly
def view_task_map(*, video: Any, cross_task: bool = False) -> dict[str, Any]:
    """Tool handler: the demonstration's structure, text only."""
    return task_map(video.index, cross_task=cross_task)
