"""Add on-demand clip data to an existing per-task video store.

``build_videos.py`` writes the 16-keyframe contact sheets that ``view_task_video``
serves. This script is a strictly additive second pass over the SAME store: it never
touches the sheets or the ``frames`` list, and it needs no renderer, because the clips
are the frames LIBERO actually recorded during the human teleoperation
(``obs/agentview_rgb`` / ``obs/eye_in_hand_rgb``), not a re-render of the sim states.

Per video it adds:

    <store>/<video_id>/agentview.npy    (T, 128, 128, 3) uint8, RPent orientation
    <store>/<video_id>/wrist.npy        (T, 128, 128, 3) uint8, RPent orientation

and merges ``control_hz``, ``phases`` and ``moments`` into ``index.json``. Phases are the
intervals between gripper-command transitions; moments are windows straddling the
transitions themselves, because the closing/opening transient falls exactly on a phase
boundary and would otherwise be split across two clips.

Usage:
    python build_clips.py --store task_videos --tasks 0-9
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

#: Where the original LIBERO demonstrations live (<root>/libero_10/*_demo.hdf5 etc.).
DEMO_ROOT_BASE = os.environ.get("LIBERO_DEMO_ROOT", "datasets/libero")
FAMILIES = {
    "libero_10": os.path.join(DEMO_ROOT_BASE, "libero_10"),
    "libero_object": os.path.join(DEMO_ROOT_BASE, "libero_object"),
    "libero_goal": os.path.join(DEMO_ROOT_BASE, "libero_goal"),
    "libero_spatial": os.path.join(DEMO_ROOT_BASE, "libero_spatial"),
}

#: LIBERO demonstrations are recorded at 20 Hz.
CONTROL_HZ = 20
#: A moment's clip spans this many steps before / after the gripper transition.
MOMENT_BEFORE, MOMENT_AFTER = 24, 16
#: Height change (m) above which a phase counts as rising / falling.
DZ_EPS = 0.03
#: Lateral travel (m) above which a phase counts as transporting.
DXY_EPS = 0.05


def _transitions(actions: np.ndarray) -> list[int]:
    """Steps where the gripper COMMAND changes sign: the phase boundaries."""
    cmd = np.sign(np.asarray(actions)[:, -1])
    return [t for t in range(1, len(cmd)) if cmd[t] != cmd[t - 1]]


def _phase_label(closed: bool, dz: float, dxy: float) -> str:
    """A short honest descriptor; the numbers travel alongside it in the index.

    An empty gripper that descends is approaching something even when it also travels
    laterally, so the vertical component decides the verb and the lateral component is
    reported alongside it rather than replacing it.
    """
    across = " and moving across the scene" if dxy > DXY_EPS else ""
    if not closed:
        if dz < -DZ_EPS:
            return f"approach (empty gripper, descending{across})"
        if dz > DZ_EPS:
            return f"retreat (empty gripper, rising{across})"
        if dxy > DXY_EPS:
            return "travel (empty gripper, moving across the scene)"
        return "hold (empty gripper, almost no translation)"
    if dxy > DXY_EPS:
        return "lift and carry (holding)" if dz > DZ_EPS else "carry (holding, moving across the scene)"
    if dz > DZ_EPS:
        return "lift (holding, rising in place)"
    if dz < -DZ_EPS:
        return "press down (holding, descending in place)"
    return "manipulate in place (holding, almost no translation)"


def describe(actions: np.ndarray, ee_pos: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Phases (intervals) and moments (windows around gripper transitions)."""
    T = len(actions)
    cmd = np.sign(np.asarray(actions)[:, -1])
    xyz = np.asarray(ee_pos)
    bounds = [0] + _transitions(actions) + [T]

    phases: list[dict] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        dz = float(xyz[e - 1, 2] - xyz[s, 2])
        dxy = float(np.linalg.norm(xyz[e - 1, :2] - xyz[s, :2]))
        closed = bool(cmd[s] > 0)
        phases.append(
            {
                "id": i + 1,
                "steps": [int(s), int(e)],
                "seconds": round((e - s) / CONTROL_HZ, 1),
                "gripper": "closed" if closed else "open",
                "dz": round(dz, 3),
                "dxy": round(dxy, 3),
                "label": _phase_label(closed, dz, dxy),
            }
        )

    # A moment is named by its position in time, but WHAT the closure achieved is what
    # tells a reader whether it was a sub-goal or a fumble: the human regrasps, and a
    # closure that is released a second later without moving anything is not sub-goal 1.
    # Rather than guess intent, each moment carries the closed phase's outcome verbatim.
    by_start = {int(ph["steps"][0]): ph for ph in phases}
    by_end = {int(ph["steps"][1]): ph for ph in phases}
    moments = []
    for t in _transitions(actions):
        kind = "grasp" if cmd[t] > 0 else "release"
        held = by_start.get(int(t)) if kind == "grasp" else by_end.get(int(t))
        a_, b_ = max(0, t - MOMENT_BEFORE), min(T, t + MOMENT_AFTER)
        n = sum(1 for m in moments if m["kind"] == kind) + 1
        entry = {
            "name": f"{kind} {n}",
            "kind": kind,
            "step": int(t),
            "steps": [int(a_), int(b_)],
            "seconds": round((b_ - a_) / CONTROL_HZ, 1),
            "eef_xyz": [round(float(v), 3) for v in xyz[t]],
        }
        if held is not None:
            moved = held["dz"] > DZ_EPS or held["dxy"] > DXY_EPS
            entry["while_closed"] = {
                "seconds": held["seconds"],
                "dz": held["dz"],
                "dxy": held["dxy"],
                "transported_something": bool(moved),
            }
        moments.append(entry)

    # The gripper command does not always reopen before the episode ends, so the final
    # release can be missing from the transitions entirely (LIBERO-10 t5). The last
    # seconds are where success becomes visible, so every demonstration gets them.
    tail = max(0, T - (MOMENT_BEFORE + MOMENT_AFTER))
    moments.append(
        {
            "name": "end",
            "kind": "end",
            "step": int(T - 1),
            "steps": [int(tail), int(T)],
            "seconds": round((T - tail) / CONTROL_HZ, 1),
            "eef_xyz": [round(float(v), 3) for v in xyz[T - 1]],
            "note": "the final seconds of the successful demonstration",
        }
    )
    return phases, moments


def build_clips(video_dir: Path, *, family: str, demo: str = "demo_0") -> dict:
    """Add clip arrays + phase/moment tables to one video directory, in place."""
    import h5py

    index = json.loads((video_dir / "index.json").read_text())
    task_name = index["task_name"]
    root = FAMILIES[family]
    matches = [
        f
        for f in sorted(glob.glob(os.path.join(root, "*_demo.hdf5")))
        if os.path.basename(f)[: -len("_demo.hdf5")].endswith(task_name)
    ]
    if not matches:
        raise FileNotFoundError(f"no {family} demo for {task_name}")

    with h5py.File(matches[0], "r") as h:
        g = h["data"][demo]
        actions = g["actions"][:]
        ee_pos = g["obs/ee_pos"][:]
        # RPent stores agentview_high.png / wrist_high.png as the vertically flipped
        # raw buffer; the recorded observations are in the same raw convention, so the
        # same flip puts the demonstration in the orientation the planner's own cameras use.
        agent = np.asarray(g["obs/agentview_rgb"][:])[:, ::-1]
        wrist = np.asarray(g["obs/eye_in_hand_rgb"][:])[:, ::-1]

    np.save(video_dir / "agentview.npy", np.ascontiguousarray(agent))
    np.save(video_dir / "wrist.npy", np.ascontiguousarray(wrist))

    phases, moments = describe(actions, ee_pos)
    index["control_hz"] = CONTROL_HZ
    index["clip_side"] = int(agent.shape[1])
    index["phases"] = phases
    index["moments"] = moments
    tmp = video_dir / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(video_dir / "index.json")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="task_videos")
    p.add_argument("--tasks", default="0-9", help="task ids, e.g. '0-9' or '3,4'")
    p.add_argument("--family", default="libero_10", choices=sorted(FAMILIES))
    p.add_argument("--demo", default="demo_0")
    a = p.parse_args()
    ids: list[int] = []
    for part in a.tasks.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    for task_id in ids:
        d = Path(a.store) / f"{a.family}_t{task_id}_{a.demo}"
        if not (d / "index.json").is_file():
            print(f"t{task_id}: no video in store, skipped", flush=True)
            continue
        ix = build_clips(d, family=a.family, demo=a.demo)
        print(
            f"t{task_id}: {ix['n_steps']} steps -> {len(ix['phases'])} phases, "
            f"{len(ix['moments'])} moments  [{ix['task_language']}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
