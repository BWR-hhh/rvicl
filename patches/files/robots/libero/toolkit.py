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

"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero import demo_clip as demo_clip_mod
from robots.libero import task_video as task_video_mod
from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("libero_toolkit")


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO robot."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
        task_video: "task_video_mod.TaskVideo | None" = None,
        task_video_cross_task: bool = False,
        task_video_mode: str = "sheets",
    ) -> None:
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported LIBERO toolkit mode: {mode!r}")
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._mode = mode
        self._task_video = task_video
        self._task_video_cross_task = bool(task_video_cross_task)
        # "sheets": the keyframe contact sheets. "map": a text map of the demonstration
        # plus view_demo_clip, which serves densely sampled clips on demand.
        self._task_video_mode = str(task_video_mode or "sheets")
        # "loop" mode enforcement, all no-ops unless mode == "loop": the control loop the
        # prompt describes is held by the harness rather than by prose. Four prompt versions
        # asked the planner to run it and on the cells that mattered it did not; the base
        # prompt's SINGLE-ATTEMPT exit ended the episode at turn 3-8 with zero clips pulled.
        self._loop = self._task_video_mode == "loop"
        self._loop_clip_counts: dict[tuple, int] = {}
        self._loop_action_calls = 0
        self._loop_grasp_calls = 0
        self._loop_finish_refusals = 0
        self._loop_calls_at_refusal = -1
        self._solved: bool = False
        self._attempt: int = 1
        # Bound the resettable attempts owned by this planner session.
        self._attempts_per_session: int = max(0, int(attempts_per_session))
        self._session_attempt: int = 1
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        # These read-only handlers need the run's EnvState bound in. Every
        # other spec binds to a primitive-driver method and captures state by
        # default unless that method is explicitly marked @readonly.
        state_handlers = {
            "view_env_state": partial(libero_tools.view_env_state, state=self._state),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(libero_tools.back_project, state=self._state),
            "segment": partial(self._primitives.segment, state=self._state),
        }
        for spec in libero_tools.TOOLS_SPEC:
            name = spec["name"]
            if name == "reset" and self._mode != "exploration":
                continue
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue  # spec without a backing primitive method
                handler = partial(self._execute_primitive, name, handler)
            self.add_tool(name, spec, handler)
        if self._task_video is not None and self._task_video_mode in ("both", "loop"):
            self.add_tool(
                task_video_mod.TOOL_SPEC["name"],
                task_video_mod.TOOL_SPEC,
                partial(
                    task_video_mod.view_task_video,
                    video=self._task_video,
                    cross_task=self._task_video_cross_task,
                ),
            )
            self.add_tool(
                demo_clip_mod.TOOL_SPEC_MAP["name"],
                demo_clip_mod.TOOL_SPEC_MAP,
                partial(
                    demo_clip_mod.view_task_map,
                    video=self._task_video,
                    cross_task=self._task_video_cross_task,
                ),
            )
            self.add_tool(
                demo_clip_mod.TOOL_SPEC["name"],
                demo_clip_mod.TOOL_SPEC,
                partial(demo_clip_mod.view_demo_clip, video=self._task_video),
            )
        elif self._task_video is not None and self._task_video_mode == "map":
            self.add_tool(
                demo_clip_mod.TOOL_SPEC_MAP["name"],
                demo_clip_mod.TOOL_SPEC_MAP,
                partial(
                    demo_clip_mod.view_task_map,
                    video=self._task_video,
                    cross_task=self._task_video_cross_task,
                ),
            )
            self.add_tool(
                demo_clip_mod.TOOL_SPEC["name"],
                demo_clip_mod.TOOL_SPEC,
                partial(demo_clip_mod.view_demo_clip, video=self._task_video),
            )
        elif self._task_video is not None and self._task_video_mode in ("sheets_min", "full_min"):
            # Minimal-description ablation: images only, no text in the result or on the images.
            want_full = self._task_video_mode == "full_min"
            if not self._task_video.index.get("minimal") or self._task_video.is_full != want_full:
                raise ValueError(
                    f"--task-video-mode {self._task_video_mode} needs a matching plain store "
                    "(video_icl/build_plain_videos.py), got " + str(self._task_video.root)
                )
            self.add_tool(
                task_video_mod.TOOL_SPEC_MIN["name"],
                task_video_mod.TOOL_SPEC_MIN,
                partial(task_video_mod.view_task_video, video=self._task_video, cross_task=False),
            )
        elif self._task_video is not None and self._task_video_mode == "full":
            # Ablation: the complete recording instead of keyframes, same tool name.
            if not self._task_video.is_full:
                raise ValueError(
                    "--task-video-mode full needs a complete-recording store "
                    "(video_icl/build_full_videos.py), got " + str(self._task_video.root)
                )
            self.add_tool(
                task_video_mod.TOOL_SPEC_FULL["name"],
                task_video_mod.TOOL_SPEC_FULL,
                partial(
                    task_video_mod.view_task_video,
                    video=self._task_video,
                    cross_task=self._task_video_cross_task,
                ),
            )
        elif self._task_video is not None:
            self.add_tool(
                task_video_mod.TOOL_SPEC["name"],
                task_video_mod.TOOL_SPEC,
                partial(
                    task_video_mod.view_task_video,
                    video=self._task_video,
                    cross_task=self._task_video_cross_task,
                ),
            )
        if self._mode == "exploration":
            reset_spec = next(
                spec for spec in libero_tools.TOOLS_SPEC if spec["name"] == "reset"
            )
            self.add_tool("reset", reset_spec, self._reset_episode)
            finish_spec, finish_handler = self._tools["finish"]
            self.add_tool(
                "finish", finish_spec, partial(self._guarded_finish, finish_handler)
            )
        if self._loop and self._task_video is not None:
            self.add_tool(
                demo_clip_mod.TOOL_SPEC["name"],
                demo_clip_mod.TOOL_SPEC,
                partial(
                    self._loop_clip,
                    partial(demo_clip_mod.view_demo_clip, video=self._task_video),
                ),
            )
            finish_spec, finish_handler = self._tools["finish"]
            self.add_tool(
                "finish", finish_spec, partial(self._loop_finish, finish_handler)
            )
            logger.info(
                "[loop] harness enforcement active: finish gate (clips>=1, pi0_pick>=6, actions>=20), "
                "clip cap (same phase/moment at most twice)"
            )

    # ---- "loop" mode enforcement ------------------------------------------------
    @readonly
    def _loop_clip(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
        """The same phase or moment at most twice; the third pull is refused."""
        key = (kwargs.get("phase"), kwargs.get("moment"))
        n = self._loop_clip_counts.get(key, 0) + 1
        self._loop_clip_counts[key] = n
        if n > 2:
            what = f"phase {key[0]}" if key[0] is not None else f"moment {key[1]!r}"
            return {
                "error": "clip refused",
                "reason": (
                    f"You have already watched {what} twice; a recording has nothing new "
                    "on a third viewing. The clip answers WHAT to do and you are stuck on "
                    "HOW, which the embodiment table answers: change the primitive, the "
                    "approach direction or the pre-pose."
                ),
                "viewings": n - 1,
            }
        return inner(**kwargs)

    @readonly
    def _loop_finish(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse to end an unsolved episode while the loop has rungs untried."""
        if self.solved():
            return inner(**kwargs)
        clips = sum(self._loop_clip_counts.values())
        grasps = self._loop_grasp_calls
        acts = self._loop_action_calls
        missing = []
        if clips < 1:
            missing.append("no clip has been pulled (loop step 1: watch the sub-goal's phase or moment before its first action)")
        if grasps < 6:
            missing.append(f"only {grasps} grasp attempts (the table gives pi0_pick 3, then rung a gives 3 more at a new approach)")
        if acts < 20:
            missing.append(f"only {acts} primitive calls in an episode with 100 turns")
        stuck = (
            self._loop_finish_refusals >= 3
            and self._loop_action_calls == self._loop_calls_at_refusal
        )
        if not missing or stuck:
            return inner(**kwargs)
        self._loop_finish_refusals += 1
        self._loop_calls_at_refusal = self._loop_action_calls
        return {
            "error": "finish refused",
            "reason": (
                "The task is not solved and the loop has rungs untried: "
                + "; ".join(missing)
                + ". The base prompt's SINGLE-ATTEMPT exit does not apply here - that rule "
                "forbids `reset`, it does not license stopping early. Continue: pull the "
                "clip for the sub-goal you are on if you have not, run the table's "
                "attempts back to back, test progress on the task object, escalate one "
                "rung at a time. `finish` is accepted once the task is solved or the "
                "turn budget is spent."
            ),
            "refusals": self._loop_finish_refusals,
        }

    def _execute_primitive(self, name: str, handler: Any, **kwargs: Any) -> Any:
        if self._loop:
            self._loop_action_calls += 1
            if name in ("pi0_pick", "pick", "pi0_doubled"):
                self._loop_grasp_calls += 1
        self._primitives.begin_primitive(name)
        try:
            return handler(**kwargs)
        finally:
            self._primitives.end_primitive()

    @readonly
    def _guarded_finish(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse to end an unsolved session while attempts remain."""
        budget = self._attempts_per_session
        if budget and not self.solved() and self._session_attempt < budget:
            remaining = budget - self._session_attempt
            return {
                "error": "finish refused",
                "reason": (
                    f"This session has {remaining} of its {budget} attempts left "
                    "and the task is not solved. Archive this attempt, call "
                    "`reset`, and try another approach."
                ),
            }
        return inner(**kwargs)

    def _reset_episode(self, reason: str) -> dict[str, Any]:
        """Restart the episode while preserving the full exploration trace."""
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": (
                    f"This session's attempt budget is spent ({budget} attempts). "
                    "Archive the attempt, update the handoff notes, and call "
                    "`finish` so the next session can continue."
                ),
            }
        self._attempt += 1
        self._session_attempt += 1
        result = self._primitives.reset_episode(reason=reason)
        result["attempt"] = self._attempt
        result["notice"] = (
            f"Episode restarted; this is attempt {self._attempt}. The original "
            "layout was restored. Re-run perception before acting."
        )
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        self._solved |= record.terminated
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = libero_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.update(result)
        return out

    @property
    def primitives(self) -> "libero_tools.LiberoPrimitives":
        """Return the action primitives this toolkit drives."""
        return self._primitives

    def init_primitives(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Finalize collected data and save the episode video independently."""
        try:
            episode = self._primitives.finalize_flywheel()
            if episode is not None:
                logger.info("flywheel episode finalized: %s", episode)
        except Exception as e:
            logger.warning("failed to finalize flywheel episode: %s", e)

        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            logger.warning(f"failed to save episode video: {e}")

    def solved(self) -> bool:
        """Return whether this run has completed the task."""
        return self._solved

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(
            self._state, recipe_tag, output_dir=get_output_dir()
        )
