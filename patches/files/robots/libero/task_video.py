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

"""Per-task video memory: a demonstration of THIS task shown to the planner in-context.

The store is built offline (``video_icl/build_videos.py``) and holds, per video, an
``index.json`` plus labelled contact sheets. At run time the cell's video is resolved by
suite family + task id, and ``view_task_video`` returns its keyframes as image blocks so
the planner can watch the demonstration before planning. It is read-only: it never
advances the environment and never reveals evaluation-scene coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.tools.toolkit import readonly
from rpent.utils.logging import get_logger

logger = get_logger("libero.task_video")

#: Contact sheets returned by one ``view_task_video`` call.
SHEETS_PER_PART = 4

#: "full" stores (``video_icl/build_full_videos.py``) hold the complete recording as images of
#: one second of consecutive frames; this many of them come back per ``view_task_video`` call.
FULL_KIND = "full_video"
FULL_PER_PART = 6
FULL_DESCRIPTION = (
    "The complete recording of the demonstration: every frame its cameras recorded, in time "
    "order, at the control rate and the recorded resolution (128 px). Each image is one second "
    "of consecutive frames, left to right then top to bottom; each frame shows the agentview "
    "camera above the wrist camera and is stamped with its demonstration step."
)

CROSS_TASK_CAVEAT = (
    "The instruction this demonstration was recorded for is stated above as "
    "'demonstrated_task_language'. YOUR instruction is whatever `task_language` in "
    "view_env_state says, and it may name a DIFFERENT object or destination: this suite "
    "rewrites goals. When they differ, take only the TECHNIQUE from the video (approach "
    "direction, grasp point on that kind of object, carry height, release condition) and "
    "apply it to YOUR target. Never execute the demonstrated goal."
)


CAVEAT = (
    "This is a demonstration of the SAME task in a DIFFERENT layout: object positions, "
    "plate/basket positions and the robot start pose differ from your scene. Take the "
    "sub-goal order, which object goes where, the grasp point on each object, the "
    "approach direction, the carry height and the release behaviour. Do NOT reuse any "
    "coordinate from it — re-localize every entity in your own images."
)

#: Layout caveat for goal-rewritten suites: the same warning as ``CAVEAT`` without calling
#: the demonstration "the SAME task", which the cross-task note below contradicts.
CROSS_TASK_LAYOUT_CAVEAT = CAVEAT.replace(
    "a demonstration of the SAME task in a DIFFERENT layout",
    "a demonstration of the ORIGINAL task of this scene (see 'if_they_differ'), in a DIFFERENT layout",
)

TOOL_SPEC: dict[str, Any] = {
    "name": "view_task_video",
    "description": (
        "Watch the demonstration video of THIS task from the video memory layer. "
        "Returns labelled keyframes (agentview and wrist, captioned with step, gripper "
        "state, end-effector xyz and the grasp/release events) as images. Read-only: it "
        "does not move the robot. Call it before planning; pass part=2, 3, ... for the "
        "later keyframes of a long demonstration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "part": {
                "type": "integer",
                "minimum": 1,
                "description": "1-based page of keyframes (default 1, the earliest).",
            }
        },
        "required": [],
    },
}


#: ``--task-video-mode full``: the same tool name, describing the complete recording instead
#: of keyframes.
TOOL_SPEC_FULL: dict[str, Any] = {
    "name": "view_task_video",
    "description": (
        "Watch the demonstration video of THIS task from the video memory layer: the COMPLETE "
        "recording, every frame at the control rate, as images of one second of consecutive "
        "frames each (agentview above wrist, each frame stamped with its step). Read-only: it "
        "does not move the robot. Call it before planning; pass part=2, 3, ... for the later "
        "seconds of the demonstration."
    ),
    "input_schema": TOOL_SPEC["input_schema"],
}


#: ``--task-video-mode sheets_min | full_min`` (minimal-description ablation, 2026-09-28):
#: images only; the store (``video_icl/build_plain_videos.py``) carries no text at all.
TOOL_SPEC_MIN: dict[str, Any] = {
    "name": "view_task_video",
    "description": (
        "Show the demonstration of this task as images. Read-only: it does not move the "
        "robot. Pass part=2, 3, ... for the later parts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"part": {"type": "integer", "minimum": 1, "description": "1-based part (default 1, the earliest)."}},
        "required": [],
    },
}


@dataclass(frozen=True)
class TaskVideo:
    """One demonstration video in the store."""

    root: Path
    index: dict[str, Any]

    @property
    def video_id(self) -> str:
        return str(self.index.get("video_id", self.root.name))

    @property
    def provenance(self) -> str:
        """One line for the prompt: what this video is and where it came from."""
        return str(self.index.get("source") or "a recorded demonstration of this task")

    @property
    def is_full(self) -> bool:
        """True for a complete-recording store built by ``video_icl/build_full_videos.py``."""
        return self.index.get("kind") == FULL_KIND

    @property
    def per_part(self) -> int:
        return FULL_PER_PART if self.is_full else SHEETS_PER_PART

    @property
    def n_parts(self) -> int:
        sheets = self.index.get("sheets") or []
        return max(1, -(-len(sheets) // self.per_part))

    def part(self, part: int, *, cross_task: bool = False) -> dict[str, Any]:
        """Payload for one page: captions plus the sheet images of that page."""
        sheets = list(self.index.get("sheets") or [])
        if not sheets:
            return {"error": f"video {self.video_id} has no sheets"}
        part = max(1, min(int(part), self.n_parts))
        start = (part - 1) * self.per_part
        page = sheets[start : start + self.per_part]
        shown = {i for sheet in page for i in sheet.get("frames", [])}
        frames = [f for f in self.index.get("frames", []) if f.get("i") in shown]
        images: list[bytes] = []
        for sheet in page:
            path = self.root / str(sheet.get("file"))
            try:
                images.append(path.read_bytes())
            except OSError as exc:  # missing sheet must not kill the run
                logger.warning("task video sheet unreadable: %s (%s)", path, exc)
        if self.index.get("minimal"):
            # Minimal-description ablation: the images and the paging, nothing else.
            return {"part": part, "n_parts": self.n_parts, "_image_series_bytes": images}
        if self.is_full:
            return self._full_payload(part, page, images, cross_task=cross_task)
        payload = {
            "video_id": self.video_id,
            "demonstrated_task_language": self.index.get("task_language"),
            "source": self.index.get("source"),
            "caveat": CROSS_TASK_LAYOUT_CAVEAT if cross_task else CAVEAT,
            "part": part,
            "n_parts": self.n_parts,
            "frames_in_this_part": frames,
            "total_keyframes": len(self.index.get("frames", [])),
            "_image_series_bytes": images,
        }
        if cross_task:
            payload["your_instruction"] = "read task_language from view_env_state; it may differ"
            payload["if_they_differ"] = CROSS_TASK_CAVEAT
        return payload

    def _full_payload(self, part: int, page: list[dict[str, Any]], images: list[bytes], *, cross_task: bool) -> dict[str, Any]:
        """One page of the complete recording: which seconds it covers, and its images."""
        hz = int(self.index.get("control_hz") or 20)
        steps = [int(s) for sheet in page for s in sheet.get("frames", [])]
        a, b = (min(steps), max(steps)) if steps else (0, 0)
        payload = {
            "video_id": self.video_id,
            "demonstrated_task_language": self.index.get("task_language"),
            "source": self.index.get("source"),
            "caveat": CROSS_TASK_LAYOUT_CAVEAT if cross_task else CAVEAT,
            "what_this_is": FULL_DESCRIPTION,
            "part": part,
            "n_parts": self.n_parts,
            "steps_in_this_part": [a, b],
            "seconds_in_this_part": [round(a / hz, 2), round((b + 1) / hz, 2)],
            "total_frames": int(self.index.get("n_steps") or 0),
            "control_hz": hz,
            "_image_series_bytes": images,
        }
        if cross_task:
            payload["your_instruction"] = "read task_language from view_env_state; it may differ"
            payload["if_they_differ"] = CROSS_TASK_CAVEAT
        return payload


def load_store(root: str | Path | None) -> dict[tuple[str, int], TaskVideo]:
    """Index every video under ``root`` by (suite family, task id)."""
    if not root:
        return {}
    base = Path(root).expanduser()
    if not base.is_dir():
        logger.warning("task video store not found: %s", base)
        return {}
    store: dict[tuple[str, int], TaskVideo] = {}
    for index_path in sorted(base.glob("*/index.json")):
        try:
            index = json.loads(index_path.read_text())
            key = (str(index["suite_family"]), int(index["task_id"]))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("skipping malformed task video %s: %s", index_path, exc)
            continue
        store.setdefault(key, TaskVideo(index_path.parent, index))
    return store


def suite_family(suite: str) -> str:
    """``libero_10_swap`` -> ``libero_10``; the video is per base task, not per regime."""
    name = str(suite)
    for suffix in ("_task", "_swap", "_lan", "_object"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve(root: str | Path | None, suite: str | None, task_id: int | None) -> TaskVideo | None:
    """The video for this cell, or None when the store has none."""
    if suite is None or task_id is None:
        return None
    video = load_store(root).get((suite_family(suite), int(task_id)))
    if video is None:
        logger.warning("no task video for %s task %s under %s", suite, task_id, root)
    return video


@readonly
def view_task_video(part: int = 1, *, video: TaskVideo, cross_task: bool = False) -> dict[str, Any]:
    """Tool handler: return one page of the cell's demonstration keyframes."""
    return video.part(part, cross_task=cross_task)
