"""Give the clip map object-level meaning: what each phase ACHIEVED, not how the arm moved.

``build_clips.py`` segments a demonstration from the gripper command alone, so its phase
labels are kinematic - "lift and carry (holding)", "travel (empty gripper)". That is enough
to find a grasp, and not enough to find anything else: on LIBERO-10 t9 the planner watched
the grasp twice and never watched the insertion it was failing at, because the insertion is
labelled "lift and carry" and the microwave door closing is labelled "travel".

This pass replays the recorded simulator states through a headless LIBERO env (no renderer,
no GPU) and reads, at each phase boundary, where every task object is and where every
articulated fixture's joint sits. From that it writes, per phase, which object moved and
into which named region, and it adds MOMENTS for articulation events - a door closing is as
addressable as a grasp. It also copies the task's object inventory and its goal predicates
out of the BDDL, so the map states the sub-goals instead of leaving them to be inferred.

Run it after build_clips.py:
    python build_semantics.py --store task_videos --tasks 0-9
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
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

#: Displacement (m) above which an object counts as having been moved by a phase. While the
#: gripper is closed the held object is being transported on purpose, so a small threshold is
#: right; while it is open, anything that moves is incidental contact or an object settling
#: after release, and reporting that as "pushes X" is worse than saying nothing.
MOVED_EPS_HELD = 0.02
MOVED_EPS_FREE = 0.06
#: Fraction of a fixture joint's own travel that counts as having articulated it. A
#: threshold in absolute units cannot work here: a microwave door is a hinge measured in
#: radians (1.8 rad of travel) and a cabinet drawer is a slide measured in metres (0.17 m),
#: so any constant that catches the door silently drops the drawer.
ARTIC_FRAC = 0.3
#: Fallback when a joint declares no usable range.
ARTIC_EPS = 0.05
#: A moved object is reported as landing "in" the nearest named site within this radius (m).
REGION_EPS = 0.12
#: States are sampled this finely when locating an articulation event.
SCAN_STRIDE = 5
#: Steps before / after an articulation event that its clip covers.
ARTIC_BEFORE, ARTIC_AFTER = 30, 20


def _bddl_section(text: str, head: str) -> str:
    """The balanced-paren body of a top-level ``(:head ...)`` block."""
    i = text.find(f"(:{head}")
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
    return ""


def read_bddl(path: str) -> dict:
    """Object inventory and goal predicates, verbatim, straight out of the problem file."""
    text = Path(path).read_text()
    objects = []
    for line in _bddl_section(text, "objects").splitlines()[1:]:
        line = line.strip().rstrip(")")
        if " - " in line:
            names, _, kind = line.partition(" - ")
            objects += [{"name": n, "type": kind.strip()} for n in names.split()]
    goal = _bddl_section(text, "goal")
    goal = re.sub(r"\s+", " ", goal[len("(:goal") : -1]).strip() if goal else ""
    return {"objects": objects, "goal": goal}


def _label(moved: list[dict], artic: list[dict], closed: bool, fallback: str) -> str:
    """What the phase achieved; falls back to the kinematic label when nothing changed."""
    parts = []
    for m in moved:
        where = f" into {m['region']}" if m.get("region") else ""
        verb = "carries" if closed else "pushes"
        parts.append(f"{verb} {m['object']}{where}")
    for a in artic:
        parts.append(f"{'closes' if a['closing'] else 'opens'} {a['fixture']}")
    if not parts:
        return fallback
    return "; ".join(parts) + f"  [{fallback}]"


def annotate(video_dir: Path, *, family: str, demo: str = "demo_0") -> dict:
    """Rewrite one video's phase labels and moments in object-level terms."""
    import h5py
    import liberopro.liberopro as lp
    from liberopro.liberopro.envs.env_wrapper import ControlEnv

    index = json.loads((video_dir / "index.json").read_text())
    task_name = index["task_name"]
    bddl = os.path.join(lp.get_libero_path("bddl_files"), family, task_name + ".bddl")
    spec = read_bddl(bddl)

    demo_file = [
        f
        for f in sorted(glob.glob(os.path.join(FAMILIES[family], "*_demo.hdf5")))
        if os.path.basename(f)[: -len("_demo.hdf5")].endswith(task_name)
    ][0]
    with h5py.File(demo_file, "r") as h:
        states = h["data"][demo]["states"][:]

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
    )
    try:
        e = env.env
        model = e.sim.model
        env.reset()
        obj_ids = dict(getattr(e, "obj_body_id", {}) or {})
        fixtures = list(getattr(e, "fixtures_dict", {}) or {})
        joints = {
            name: [
                model.joint_id2name(i)
                for i in range(model.njnt)
                if model.joint_id2name(i) and model.joint_id2name(i).startswith(name)
            ]
            for name in fixtures
        }
        # Per-joint threshold, scaled to that joint's own declared travel.
        limits = {}
        for js in joints.values():
            for j in js:
                lo, hi = model.jnt_range[model.joint_name2id(j)]
                span = float(hi) - float(lo)
                limits[j] = max(ARTIC_EPS, ARTIC_FRAC * span) if span > 0 else ARTIC_EPS
        sites = [model.site_id2name(i) for i in range(model.nsite)]

        def read(step: int) -> dict:
            env.set_init_state(states[min(step, len(states) - 1)])
            d = e.sim.data
            return {
                "pos": {k: np.array(d.body_xpos[v], dtype=float) for k, v in obj_ids.items()},
                "jnt": {
                    j: float(d.qpos[model.get_joint_qpos_addr(j)])
                    for js in joints.values()
                    for j in js
                },
                "site": {s: np.array(d.site_xpos[model.site_name2id(s)], dtype=float) for s in sites},
            }

        # Boundaries of every phase, plus a scan grid for locating articulation events.
        scan = sorted({0, len(states) - 1, *range(0, len(states), SCAN_STRIDE),
                       *(int(b) for ph in index["phases"] for b in ph["steps"])})
        snap = {s: read(s) for s in scan}
    finally:
        env.close()

    manipulable = {o["name"] for o in spec["objects"]}

    for ph in index["phases"]:
        a, b = (int(v) for v in ph["steps"])
        lo, hi = snap[min(scan, key=lambda s: abs(s - a))], snap[min(scan, key=lambda s: abs(s - (b - 1)))]
        moved = []
        held = ph["gripper"] == "closed"
        for name in obj_ids:
            if name not in manipulable:
                continue
            d = float(np.linalg.norm(hi["pos"][name] - lo["pos"][name]))
            if d <= (MOVED_EPS_HELD if held else MOVED_EPS_FREE):
                continue
            end = hi["pos"][name]
            # An object's own site sits at distance zero, and a start region says nothing
            # about where it ended up; neither is the destination we want to name.
            near = [
                (float(np.linalg.norm(hi["site"][s] - end)), s)
                for s in hi["site"]
                if not s.endswith(("_init_region", "_default_site"))
                and not s.startswith((name, "robot", "gripper"))
            ]
            best = min(near) if near else (1e9, "")
            moved.append(
                {
                    "object": name,
                    "displacement_m": round(d, 3),
                    "region": best[1] if best[0] <= REGION_EPS else None,
                }
            )
        artic = []
        for fixture, js in joints.items():
            for j in js:
                dq = hi["jnt"][j] - lo["jnt"][j]
                if abs(dq) > limits[j]:
                    artic.append(
                        {
                            "fixture": fixture,
                            "joint": j,
                            "from": round(lo["jnt"][j], 3),
                            "to": round(hi["jnt"][j], 3),
                            # The joints rest at 0 when shut, so shrinking magnitude is closing.
                            "closing": abs(hi["jnt"][j]) < abs(lo["jnt"][j]),
                        }
                    )
        # Idempotent: a second pass must rewrite the kinematic label, not its own output.
        ph["kinematics"] = ph.get("kinematics", ph["label"])
        ph["label"] = _label(moved, artic, held, ph["kinematics"])
        if moved:
            ph["moves"] = moved
        if artic:
            ph["articulates"] = artic

    # Articulation moments: the door closing is a sub-goal, so it must be addressable.
    extra = []
    for fixture, js in joints.items():
        for j in js:
            series = [(s, snap[s]["jnt"][j]) for s in scan]
            total = series[-1][1] - series[0][1]
            if abs(total) <= limits[j]:
                continue
            half = series[0][1] + total / 2
            step = next(
                (s for s, q in series if (q >= half if total > 0 else q <= half)),
                series[-1][0],
            )
            closing = abs(series[-1][1]) < abs(series[0][1])
            extra.append(
                {
                    "name": f"{'close' if closing else 'open'} {fixture}",
                    "kind": "articulation",
                    "step": int(step),
                    "steps": [
                        max(0, int(step) - ARTIC_BEFORE),
                        min(len(states), int(step) + ARTIC_AFTER),
                    ],
                    "seconds": round((ARTIC_BEFORE + ARTIC_AFTER) / (index.get("control_hz") or 20), 1),
                    "joint": {"name": j, "from": round(series[0][1], 3), "to": round(series[-1][1], 3)},
                }
            )
    if extra:
        end = [m for m in index["moments"] if m.get("kind") == "end"]
        rest = [m for m in index["moments"] if m.get("kind") not in ("end", "articulation")]
        index["moments"] = sorted(rest + extra, key=lambda m: m["step"]) + end

    index["task_objects"] = spec["objects"]
    index["task_goal"] = spec["goal"]
    index["task_regions"] = [
        s
        for s in sites
        if not s.endswith(("_init_region", "_default_site"))
        and not s.startswith(("robot", "gripper"))
    ]
    tmp = video_dir / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(video_dir / "index.json")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="task_videos")
    p.add_argument("--tasks", default="0-9")
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
            print(f"t{task_id}: not in store, skipped", flush=True)
            continue
        ix = annotate(d, family=a.family, demo=a.demo)
        print(f"t{task_id}: goal {ix['task_goal']}", flush=True)
        for ph in ix["phases"]:
            print(f"    P{ph['id']} {ph['steps']}  {ph['label']}", flush=True)
        print(f"    moments: {[m['name'] for m in ix['moments']]}", flush=True)


if __name__ == "__main__":
    main()
