"""Build the per-task video memory store used by RPent's ``view_task_video`` tool.

One LIBERO human teleoperation demonstration per task is turned into a small set of
keyframes (grasp / lift / carry / release events plus uniform fill), re-rendered from the
recorded simulator states at high resolution in the SAME orientation RPent uses for
``agentview_high.png`` / ``wrist_high.png``, and composed into labelled contact sheets.

The store layout is one directory per video:

    <store>/<video_id>/index.json
    <store>/<video_id>/sheet_01.png ...

Usage:
    python build_videos.py --out task_videos --tasks 0-9
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Demonstration roots and the base-task suite they belong to, keyed by suite family.
#: Where the original LIBERO demonstrations live (<root>/libero_10/*_demo.hdf5 etc.).
DEMO_ROOT_BASE = os.environ.get("LIBERO_DEMO_ROOT", "datasets/libero")
FAMILIES = {
    "libero_10": (os.path.join(DEMO_ROOT_BASE, "libero_10"), "libero_10_swap"),
    "libero_object": (os.path.join(DEMO_ROOT_BASE, "libero_object"), "libero_object_swap"),
    "libero_goal": (os.path.join(DEMO_ROOT_BASE, "libero_goal"), "libero_goal_swap"),
    "libero_spatial": (os.path.join(DEMO_ROOT_BASE, "libero_spatial"), "libero_spatial_swap"),
}
DEMO_ROOT = FAMILIES["libero_10"][0]
SUITE_FAMILY = "libero_10"
#: Cameras rendered per keyframe, in RPent's naming.
CAMERAS = ("agentview", "robot0_eye_in_hand")


@dataclass
class Keyframe:
    i: int
    step: int
    t_frac: float
    event: str
    gripper: str
    eef_xyz: list[float]


def _gripper_events(actions: np.ndarray) -> tuple[list[int], list[int]]:
    """Steps where the gripper command switches to close / to open."""
    cmd = np.sign(np.asarray(actions)[:, -1])
    closes, opens = [], []
    for t in range(1, len(cmd)):
        if cmd[t] > 0 and cmd[t - 1] <= 0:
            closes.append(t)
        if cmd[t] < 0 and cmd[t - 1] >= 0:
            opens.append(t)
    return closes, opens


def _lift_step(eef_z: np.ndarray, close_t: int, window: int = 60) -> int:
    """First local peak of the end-effector height after a grasp."""
    end = min(len(eef_z) - 1, close_t + window)
    if end <= close_t:
        return close_t
    return int(close_t + np.argmax(eef_z[close_t:end]))


def select_keyframes(actions: np.ndarray, eef_pos: np.ndarray, gripper_states: np.ndarray, max_frames: int = 16) -> list[Keyframe]:
    """Event-driven keyframes: start, per grasp (pre / close / lift), per release
    (pre / open), end, then uniform fill up to ``max_frames``."""
    T = len(actions)
    closes, opens = _gripper_events(actions)
    eef_z = np.asarray(eef_pos)[:, 2]
    picks: list[tuple[int, str]] = [(0, "start")]
    for n, c in enumerate(closes, 1):
        picks.append((max(0, c - 10), f"pre-grasp {n}"))
        picks.append((c, f"gripper closes {n}"))
        picks.append((_lift_step(eef_z, c), f"lifted {n}"))
    for n, o in enumerate(opens, 1):
        picks.append((max(0, o - 12), f"over target {n}"))
        picks.append((o, f"gripper opens {n}"))
    picks.append((T - 1, "end (task succeeded)"))
    # de-duplicate on step, keep the first label, then uniform fill
    seen: dict[int, str] = {}
    for step, label in picks:
        seen.setdefault(int(step), label)
    for step in np.linspace(0, T - 1, num=max_frames, dtype=int):
        if len(seen) >= max_frames:
            break
        seen.setdefault(int(step), "")
    steps = sorted(seen)[:max_frames]
    out = []
    for i, step in enumerate(steps, 1):
        opening = float(np.asarray(gripper_states)[step][0] - np.asarray(gripper_states)[step][1])
        out.append(
            Keyframe(
                i=i,
                step=int(step),
                t_frac=round(step / max(1, T - 1), 3),
                event=seen[step],
                gripper="closed" if abs(opening) < 0.05 else "open",
                eef_xyz=[round(float(v), 3) for v in np.asarray(eef_pos)[step]],
            )
        )
    return out


def _sheet(frames: list[tuple[Keyframe, np.ndarray, np.ndarray]], side: int) -> "Image.Image":
    """One contact sheet: one row per keyframe, agentview | wrist, captioned."""
    from PIL import Image, ImageDraw

    pad, bar = 6, 26
    w = side * 2 + pad * 3
    h = (side + bar + pad) * len(frames) + pad
    sheet = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for r, (kf, agent, wrist) in enumerate(frames):
        y = pad + r * (side + bar + pad)
        caption = (
            f"frame {kf.i}  step {kf.step}  t={kf.t_frac:.2f}  gripper {kf.gripper}"
            f"  eef xyz {kf.eef_xyz}" + (f"  <{kf.event}>" if kf.event else "")
        )
        draw.text((pad, y + 6), caption, fill=(20, 20, 20))
        for c, img in enumerate((agent, wrist)):
            im = Image.fromarray(img).resize((side, side))
            sheet.paste(im, (pad + c * (side + pad), y + bar))
        draw.text((pad + 4, y + bar + 4), "agentview", fill=(255, 255, 0))
        draw.text((pad + side + pad + 4, y + bar + 4), "wrist", fill=(255, 255, 0))
    return sheet


def build_task_video(task_id: int, out_root: Path, *, family: str = "libero_10", demo: str = "demo_0", max_frames: int = 16, render_side: int = 512, sheet_side: int = 448, rows_per_sheet: int = 2) -> dict:
    """Render one task's demonstration video into the store; returns its index dict."""
    import h5py
    import liberopro.liberopro as lp
    from liberopro.liberopro import benchmark as pb
    from liberopro.liberopro.envs import OffScreenRenderEnv

    demo_root, ref_suite = FAMILIES[family]
    bench = pb.get_benchmark_dict()[ref_suite]()
    task = bench.get_task(task_id)
    matches = [f for f in sorted(glob.glob(os.path.join(demo_root, "*_demo.hdf5"))) if os.path.basename(f)[: -len("_demo.hdf5")].endswith(task.name)]
    if not matches:
        raise FileNotFoundError(f"no {family} demo for task {task_id} ({task.name})")
    demo_path = matches[0]
    base_name = os.path.basename(demo_path)[: -len("_demo.hdf5")]

    with h5py.File(demo_path, "r") as h:
        g = h["data"][demo]
        states = g["states"][:]
        actions = g["actions"][:]
        eef_pos = g["obs/ee_pos"][:]
        gripper_states = g["obs/gripper_states"][:]

    keyframes = select_keyframes(actions, eef_pos, gripper_states, max_frames=max_frames)
    bddl = os.path.join(lp.get_libero_path("bddl_files"), family, base_name + ".bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=render_side, camera_widths=render_side, camera_names=list(CAMERAS))
    env.seed(0)
    env.reset()
    rendered = []
    try:
        for kf in keyframes:
            obs = env.set_init_state(states[kf.step])
            # RPent stores agentview_high.png / wrist_high.png as the vertically
            # flipped raw buffer; match that orientation exactly.
            rendered.append((kf, np.asarray(obs["agentview_image"])[::-1], np.asarray(obs["robot0_eye_in_hand_image"])[::-1]))
    finally:
        env.close()

    video_id = f"{family}_t{task_id}_{demo}"
    out = out_root / video_id
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("sheet_*.png"):
        f.unlink()
    sheets = []
    for s, start in enumerate(range(0, len(rendered), rows_per_sheet), 1):
        chunk = rendered[start : start + rows_per_sheet]
        name = f"sheet_{s:02d}.png"
        _sheet(chunk, sheet_side).save(out / name)
        sheets.append({"file": name, "frames": [kf.i for kf, _, _ in chunk]})

    index = {
        "video_id": video_id,
        "suite_family": family,
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "source": f"LIBERO human teleoperation demonstration ({base_name}, {demo}); layout differs from the evaluation scene",
        "n_steps": int(len(states)),
        "render_side": render_side,
        "frames": [asdict(kf) for kf in keyframes],
        "sheets": sheets,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="task_videos")
    p.add_argument("--tasks", default="0-9", help="task ids, e.g. '0-9' or '3,4'")
    p.add_argument("--family", default="libero_10", choices=sorted(FAMILIES))
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--max-frames", type=int, default=16)
    a = p.parse_args()
    ids: list[int] = []
    for part in a.tasks.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    out_root = Path(a.out)
    for task_id in ids:
        index = build_task_video(task_id, out_root, family=a.family, demo=a.demo, max_frames=a.max_frames)
        print(f"t{index['task_id']}: {len(index['frames'])} frames, {len(index['sheets'])} sheets — {index['task_language']}", flush=True)


if __name__ == "__main__":
    main()
