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

"""Contracts for the per-task video memory layer."""

from __future__ import annotations

import json
from pathlib import Path

from robots.libero import task_video
from robots.libero.prompt_bundle import system_prompt, user_prompt
from rpent.robots.prompt_bundle import PromptBundle
from rpent.tools.toolkit import ToolResult, _is_readonly

PROMPT_VARS = {
    "suite": "libero_10_swap",
    "task": 4,
    "seed": 1,
    "memory_suite": "libero_10_swap",
    "memory_task": 4,
    "recipe_tag": "10_swap_t4_s1",
    "mode": "eval",
    "memory_profile": "hf",
    "memory_dir": "/memory/libero",
    "reference_tag": "10_swap_t4_s0",
    "memory_inbox": "/memory/libero/_internal/inbox/x",
    "session_number": 1,
    "session_max": 1,
    "output_dir": "/out",
    "task_video_id": "libero_10_t4_demo_0",
    "task_video_source": "LIBERO-10 human teleoperation demonstration; layout differs",
    "task_video_parts": 2,
}


def _store(tmp_path: Path, n_sheets: int = 6) -> Path:
    root = tmp_path / "videos" / "libero_10_t4_demo_0"
    root.mkdir(parents=True)
    frames = [
        {"i": i, "step": i * 10, "t_frac": i / n_sheets, "event": "", "gripper": "open", "eef_xyz": [0.0, 0.0, 0.6]}
        for i in range(1, n_sheets + 1)
    ]
    index = {
        "video_id": "libero_10_t4_demo_0",
        "suite_family": "libero_10",
        "task_id": 4,
        "task_language": "put the white mug on the left plate",
        "source": "LIBERO-10 human teleoperation demonstration",
        "frames": frames,
        "sheets": [{"file": f"sheet_{i:02d}.png", "frames": [i]} for i in range(1, n_sheets + 1)],
    }
    (root / "index.json").write_text(json.dumps(index))
    for i in range(1, n_sheets + 1):
        (root / f"sheet_{i:02d}.png").write_bytes(f"png-{i}".encode())
    return root.parent


def test_resolve_matches_every_perturbation_suite_of_the_base_task(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for suite in ("libero_10", "libero_10_swap", "libero_10_task", "libero_10_lan", "libero_10_object"):
        video = task_video.resolve(store, suite, 4)
        assert video is not None and video.video_id == "libero_10_t4_demo_0", suite
    assert task_video.resolve(store, "libero_10_swap", 7) is None
    assert task_video.resolve(None, "libero_10_swap", 4) is None
    assert task_video.resolve(tmp_path / "missing", "libero_10_swap", 4) is None


def test_parts_page_the_sheets_and_carry_only_their_own_captions(tmp_path: Path) -> None:
    video = task_video.resolve(_store(tmp_path, n_sheets=6), "libero_10_swap", 4)
    assert video is not None and video.n_parts == 2
    first = video.part(1)
    assert first["part"] == 1 and first["n_parts"] == 2
    assert first["_image_series_bytes"] == [b"png-1", b"png-2", b"png-3", b"png-4"]
    assert [f["i"] for f in first["frames_in_this_part"]] == [1, 2, 3, 4]
    second = video.part(2)
    assert second["_image_series_bytes"] == [b"png-5", b"png-6"]
    assert [f["i"] for f in second["frames_in_this_part"]] == [5, 6]
    # out-of-range parts clamp instead of erroring mid-run
    assert video.part(9)["part"] == 2 and video.part(0)["part"] == 1


def test_tool_result_emits_one_image_block_per_sheet(tmp_path: Path) -> None:
    video = task_video.resolve(_store(tmp_path, n_sheets=6), "libero_10_swap", 4)
    assert video is not None
    result = ToolResult(name="view_task_video", result=task_video.view_task_video(1, video=video))
    kinds = [block["type"] for block in result.content_blocks]
    assert kinds == ["text", "image", "image", "image", "image"]
    assert "_image_series_bytes" not in result.content_blocks[0]["text"]
    assert "put the white mug on the left plate" in result.content_blocks[0]["text"]
    assert _is_readonly(task_video.view_task_video)


def test_mismatched_store_never_announces_itself_in_the_prompt(tmp_path: Path) -> None:
    """The blind control: nothing the planner sees may identify a swapped demonstration."""
    store = _store(tmp_path)
    index_path = store / "libero_10_t4_demo_0" / "index.json"
    index = json.loads(index_path.read_text())
    index["donor_task_id"] = 9  # what make_variants.py records for a swapped video
    index_path.write_text(json.dumps(index))
    video = task_video.resolve(store, "libero_10_swap", 4)
    assert video is not None
    payload = video.part(1)
    payload.pop("_image_series_bytes")
    assert "donor" not in json.dumps(payload) and "mismatch" not in json.dumps(payload).lower()
    bundle = PromptBundle(system=system_prompt, user=user_prompt)
    text = bundle.render(
        "system",
        variables={**PROMPT_VARS, "task_video": True, "task_video_source": video.provenance},
    )
    assert "donor" not in text and "mismatch" not in text.lower()


def test_missing_sheet_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    store = _store(tmp_path, n_sheets=2)
    (store / "libero_10_t4_demo_0" / "sheet_02.png").unlink()
    video = task_video.resolve(store, "libero_10_swap", 4)
    assert video is not None
    assert video.part(1)["_image_series_bytes"] == [b"png-1"]


def test_prompt_section_and_step_appear_only_when_a_video_is_bound() -> None:
    bundle = PromptBundle(system=system_prompt, user=user_prompt)
    with_video = bundle.render("system", variables={**PROMPT_VARS, "task_video": True})
    without = bundle.render("system", variables={**PROMPT_VARS, "task_video": False})
    assert "TASK VIDEO MEMORY" in with_video and "view_task_video" in with_video
    assert "human teleoperation demonstration" in with_video and "2 part(s)" in with_video
    assert "TASK VIDEO" not in without and "view_task_video" not in without
    # the baseline prompt is untouched: the video build only prepends one step
    assert without in with_video or len(with_video) > len(without)
    local = bundle.render(
        "system", variables={**PROMPT_VARS, "memory_profile": "local", "task_video": True}
    )
    assert "TASK VIDEO MEMORY" in local and "LOCAL exploration corpus" in local


def test_local_profile_prompt_carries_the_cross_task_block_when_flagged():
    """The local memory profile has its own prompt assembly; the goal-rewritten warning
    must reach the planner there too, not only through the hf profile's assembly."""
    from robots.libero.prompt_bundle import system_prompt
    from robots.libero.prompts import system as parts

    base = {"memory_profile": "local", "task_video": True, "task_video_id": "v", "task_video_source": "s", "task_video_parts": 2}
    same = str(system_prompt({**base, "task_video_cross_task": False}))
    cross = str(system_prompt({**base, "task_video_cross_task": True}))
    assert "THIS SUITE REWRITES GOALS" not in same
    assert "THIS SUITE REWRITES GOALS" in cross
    assert parts.TASK_VIDEO_CROSS_TASK.strip()[:40] in cross
    assert "demonstration of THIS task" not in cross


def test_cross_task_payload_never_calls_the_demonstration_the_same_task(tmp_path):
    from robots.libero.task_video import TaskVideo

    index = {"video_id": "v", "task_language": "orig", "source": "s", "frames": [{"i": 1}], "sheets": [{"file": "sheet_01.png", "frames": [1]}]}
    (tmp_path / "sheet_01.png").write_bytes(b"png")
    video = TaskVideo(root=tmp_path, index=index)
    same, cross = video.part(1), video.part(1, cross_task=True)
    assert "SAME task" in same["caveat"] and "if_they_differ" not in same
    assert "SAME task" not in cross["caveat"]
    assert "ORIGINAL task" in cross["caveat"] and "Never execute the demonstrated goal" in cross["if_they_differ"]
