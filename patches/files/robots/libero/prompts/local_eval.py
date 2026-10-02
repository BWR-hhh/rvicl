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

"""Local-memory variant of the single-attempt LIBERO evaluation prompt."""

from __future__ import annotations

from collections.abc import Mapping

from robots.libero.prompts import system as base
from rpent.prompt.utils import Numbered, PromptNode

(
    _,
    STEP_READ_GUIDES,
    _,
    STEP_INSPECT_INITIAL,
    STEP_PERCEPTION_PASS,
    STEP_EXECUTE,
    STEP_PRIMITIVES,
    STEP_RECOVERY,
    STEP_FINISH,
) = base.WORKFLOW_STEPS

MEMORY_PROFILE = """Use the LOCAL exploration corpus for this evaluation. Its three layers have
different jobs; use every layer that is available:

1. GLOBAL: `{{memory_dir}}/global/` — reusable robot/perception/primitive lessons.
2. SUITE: `{{memory_dir}}/suite/suite_libero10_<regime>_t{{memory_task}}.md` — the
   task/regime strategy, validated ranges, and failure table.
3. TASK: `{{memory_dir}}/task_only/{{reference_tag}}.json` plus
   `{{memory_dir}}/task_only/{{reference_tag}}_recipe.jsonl` — the matched successful
   audit and command order from seed 0.

Read the task pair and the exact suite leaf when present, then select only the
relevant global leaves through `MEMORY.md`. Recipes are technique references,
not coordinates: re-localize every entity in the current image. Never read
`_internal/` during evaluation."""

MEMORY_PROFILE_EMPTY = """There is NO text memory for this evaluation: the corpus is empty, by design.
Do not go looking for suite strategies, task audits or recipes — none exist, and time spent
searching for them is wasted. Your priors are (a) the demonstration video in the video memory
layer, and (b) the guides listed below, which describe the tools and the calibration, not this
task. Everything else you must derive yourself from the current scene: pick the parameters
(step sizes, chunk budgets, grasp offsets) from what you observe, verify after every primitive,
and adjust. State in your audit that you worked without text memory."""


STEP_READ_EMPTY_MEMORY = """THERE IS NO TEXT MEMORY. Skip every memory read: `{{memory_dir}}` is empty.
Go straight to the task video and the guides, then inspect the initial state."""


STEP_READ_LOCAL_MEMORY = """READ EACH AVAILABLE LOCAL MEMORY LAYER FIRST:
- task audit: `{{memory_dir}}/task_only/{{reference_tag}}.json`
- task recipe: `{{memory_dir}}/task_only/{{reference_tag}}_recipe.jsonl`
- suite leaf: find the matching task/regime leaf under `{{memory_dir}}/suite/`
- global index: `{{memory_dir}}/MEMORY.md`, then only relevant leaves under
  `{{memory_dir}}/global/`

If a layer is absent, state that explicitly and continue with the available
validated layers. Record the exact files used in final `strategy_notes`. Treat
absolute coordinates as stale and re-derive them from this scene."""

WORKFLOW_STEPS = (
    STEP_READ_LOCAL_MEMORY,
    STEP_READ_GUIDES,
    STEP_INSPECT_INITIAL,
    STEP_PERCEPTION_PASS,
    STEP_EXECUTE,
    STEP_PRIMITIVES,
    STEP_RECOVERY,
    STEP_FINISH,
)


def system_prompt(variables: "Mapping[str, object] | None" = None) -> PromptNode:
    """Assemble the local suite/task/global evaluation prompt."""
    video = bool((variables or {}).get("task_video"))
    # Goal-rewritten suites: the demonstration shows the ORIGINAL task, and the planner must
    # be told so in the prompt itself, not only inside the tool's payload.
    cross_task = video and bool((variables or {}).get("task_video_cross_task"))
    # "map" replaces the preloaded keyframes with a text map plus on-demand clips.
    _mode = str((variables or {}).get("task_video_mode") or "sheets")
    video_map = video and _mode == "map"
    # "both": the keyframe sheets AND the map with on-demand clips.
    video_both = video and _mode == "both"
    # "loop": same three views as "both", control loop instead of behavioural rules.
    video_loop = video and _mode == "loop"
    # "full" (ablation): the complete recording instead of keyframes; plain description, no rules.
    video_full = video and _mode == "full"
    # "sheets_min" / "full_min" (ablation): images only, one-paragraph description, no watch step.
    video_min = video and _mode in ("sheets_min", "full_min")
    empty = bool((variables or {}).get("memory_corpus_empty"))
    # LIBERO-Plus cells: every arm is told what the benchmark perturbs and where its memory
    # comes from; a video arm is also told the demonstration is of the unperturbed scene.
    plus = bool((variables or {}).get("libero_plus"))
    # Ablation arm: the behavioural rules of the video prompt, no demonstration at all.
    rules_only = bool((variables or {}).get("task_rules")) and not video
    base_steps = (
        (STEP_READ_EMPTY_MEMORY, *WORKFLOW_STEPS[1:]) if empty else WORKFLOW_STEPS
    )
    watch = (
        base.STEP_LOOP
        if video_loop
        else (
            base.STEP_WATCH_AND_MAP
            if video_both
            else (base.STEP_READ_TASK_MAP if video_map else base.STEP_WATCH_TASK_VIDEO)
        )
    )
    steps = (watch, *base_steps) if (video and not video_min) else base_steps
    return {
        "ROLE AND EVALUATION": base.ROLE_AND_EVALUATION,
        **({"LIBERO-PLUS EPISODE": base.LIBERO_PLUS_EPISODE} if plus else {}),
        **(
            {
                (
                    "TASK VIDEO MEMORY — a demonstration of the ORIGINAL task (yours is rewritten)"
                    if cross_task
                    else (
                        "TASK VIDEO MEMORY — read the map of THIS task's demonstration"
                        if video_map
                        else (
                            "TASK VIDEO MEMORY — watch THIS task's demonstration and read its map"
                            if (video_both or video_loop)
                            else "TASK VIDEO MEMORY — watch the demonstration of THIS task"
                        )
                    )
                ): (
                    base.TASK_VIDEO_LOOP
                    if video_loop
                    else (
                        base.TASK_VIDEO_BOTH
                        if video_both
                        else (base.TASK_VIDEO_MAP if video_map else (base.TASK_VIDEO_FULL if video_full else base.TASK_VIDEO))
                    )
                )
                + ("\n\n" + base.TASK_VIDEO_CROSS_TASK if cross_task else "")
                + ("\n\n" + base.TASK_VIDEO_PLUS if plus else "")
            }
            if (video and not video_min)
            else {}
        ),
        **(
            {"TASK DEMONSTRATION": (base.TASK_VIDEO_MIN_FULL if _mode == "full_min" else base.TASK_VIDEO_MIN_KEYFRAMES)}
            if video_min
            else {}
        ),
        **(
            {"EXECUTION RULES — recovery, placement, enclosures, posture under load": base.TASK_RULES_NOVIDEO}
            if rules_only
            else {}
        ),
        (
            "MEMORY PROFILE — NO TEXT MEMORY IN THIS RUN"
            if empty
            else "MEMORY PROFILE — LOCAL SUITE + TASK + GLOBAL"
        ): (MEMORY_PROFILE_EMPTY if empty else MEMORY_PROFILE),
        "PROVEN LEVERS": base.PROVEN_LEVERS,
        "RUNTIME": base.RUNTIME,
        "YOUR GOAL": base.GOAL,
        "RULES (NON-NEGOTIABLE)": base.RULES,
        "LOCALIZATION": base.LOCALIZATION,
        "FIRST-STEP ALGORITHM": base.PERCEPTION_ALGORITHM,
        "WORKFLOW": Numbered(steps),
        "KEY HYPERPARAMETERS": base.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": base.OUTPUT_DISCIPLINE,
    }


__all__ = ["system_prompt", "WORKFLOW_STEPS"]
