"""Repair three LIBERO-PRO goal-rewritten tasks that ship with malformed BDDL goals.

libero_10_task t2 and libero_spatial_task t3/t7 have unbalanced parentheses in (:goal ...).
LIBERO-PRO's parser then silently falls back to an unrelated libero_10 problem, the layout
pruning drops every state, and the shipped .pruned_init is an empty array, so the task cannot
be built at all. This script (1) backs up the originals, (2) fixes the goal to the exact form
the sibling tasks use, (3) regenerates the task's layouts with the fixed file, and (4) writes
them in the same format (float64 array of flattened MuJoCo states, torch.save).

Layout generation: env.seed(s) + reset() samples placements from the BDDL (:init) regions, as
LIBERO does. Objects spawn slightly above their support and settle: the shipped layouts of the
same scene drop up to 7 cm vertically over 20 no-op steps with ~0 horizontal motion. A layout is
kept only if the goal is false at reset and after settling, and every movable object settles no
more than it does in the shipped layouts of the same scene: horizontal move <= shipped max + 1 cm
(at least 2 cm), drop <= shipped max + 3 cm (spawn heights differ a lot between scenes: 7 cm in
the L10 kitchen, 16 cm for a bowl stacked on a box in the spatial scenes). The first 50 kept are written, and
the xy spread of the objects is compared with the shipped layouts of the same scene.
LIBERO-PRO's own pruning code is not shipped, so this is a documented reconstruction.

Usage: python patch_liberopro_tasks.py fix      # backup + BDDL fix + parse check (fast)
       python patch_liberopro_tasks.py layouts  # regenerate the 3 .pruned_init files
       python patch_liberopro_tasks.py verify   # RLinf/RPent loading path
"""
from __future__ import annotations

import hashlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np

PATCH_ROOT = Path(os.environ.get("LIBERO_PRO_PATCH_ROOT", Path(__file__).resolve().parent))
PATCHES = [
    ("libero_10_task", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
     "  And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))\n",
     "  (And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))\n",
     [["turnon", "flat_stove_1"], ["on", "chefmate_8_frypan_1", "flat_stove_1_cook_region"]]),
    ("libero_spatial_task", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
     "  (On akita_black_bowl_2 plate_1))\n", "  (And (On akita_black_bowl_2 plate_1))\n",
     [["on", "akita_black_bowl_2", "plate_1"]]),
    ("libero_spatial_task", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
     "  (On akita_black_bowl_2 plate_1))\n", "  (And (On akita_black_bowl_2 plate_1))\n",
     [["on", "akita_black_bowl_2", "plate_1"]]),
]
N_LAYOUTS, SETTLE_STEPS, XY_MARGIN_M, MIN_XY_M, DROP_MARGIN_M, N_CALIBRATION = 50, 20, 0.01, 0.02, 0.03, 5


def paths(suite: str, name: str) -> tuple[Path, Path]:
    from liberopro.liberopro import get_libero_path
    return (Path(get_libero_path("bddl_files")) / suite / f"{name}.bddl",
            Path(get_libero_path("init_states")) / suite / f"{name}.pruned_init")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fix() -> None:
    from liberopro.liberopro.envs.bddl_utils import _core_robosuite_parse_problem
    manifest = PATCH_ROOT / "manifest.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {}
    for suite, name, old, new, expected_goal in PATCHES:
        bddl, init = paths(suite, name)
        backup = PATCH_ROOT / "originals" / suite
        backup.mkdir(parents=True, exist_ok=True)
        for src in (bddl, init):
            dst = backup / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                record[f"{suite}/{src.name}"] = {"original_sha256": sha(dst), "original_bytes": dst.stat().st_size}
        text = bddl.read_text()
        if new in text and old not in text.replace(new, ""):
            print(f"{suite}/{name}: BDDL already fixed")
        else:
            assert text.count(old) == 1, f"{suite}/{name}: expected goal line not found exactly once"
            bddl.write_text(text.replace(old, new))
            print(f"{suite}/{name}: BDDL goal fixed")
        parsed = _core_robosuite_parse_problem(str(bddl))  # raises instead of falling back
        assert parsed["goal_state"] == expected_goal, (parsed["goal_state"], expected_goal)
        record[f"{suite}/{bddl.name}"]["patched_sha256"] = sha(bddl)
        print(f"   parses cleanly; goal = {parsed['goal_state']}; language = {' '.join(parsed['language_instruction'])}")
    manifest.write_text(json.dumps(record, indent=2))


def layouts() -> None:
    import torch
    from liberopro.liberopro.envs import OffScreenRenderEnv
    for suite, name, _old, _new, _goal in PATCHES:
        bddl, init = paths(suite, name)
        rec = json.loads((PATCH_ROOT / "manifest.json").read_text())
        if "layouts" in rec.get(f"{suite}/{init.name}", {}):
            print(f"{suite}/{name}: layouts already generated, left untouched (runs may be using them)")
            continue
        t0 = time.time()
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        inner = env.env
        movable = list(getattr(inner, "objects_dict", {}).keys())
        body_ids = {o: inner.obj_body_id[o] for o in movable}
        noop = np.zeros(7); noop[-1] = -1.0
        # calibrate settling limits on the shipped layouts of the same scene
        shipped_cal = torch.load(Path(init.parent.parent) / suite.replace("_task", "") / init.name, weights_only=False)
        max_xy = {o: 0.0 for o in body_ids}; max_drop = {o: 0.0 for o in body_ids}
        for st in shipped_cal[:N_CALIBRATION]:
            env.reset(); env.set_init_state(st)
            p0 = {o: inner.sim.data.body_xpos[b].copy() for o, b in body_ids.items()}
            for _ in range(SETTLE_STEPS):
                env.step(noop)
            for o, b in body_ids.items():
                d = inner.sim.data.body_xpos[b] - p0[o]
                max_xy[o] = max(max_xy[o], float(np.linalg.norm(d[:2]))); max_drop[o] = max(max_drop[o], float(-d[2]))
        lim_xy = {o: max(MIN_XY_M, max_xy[o] + XY_MARGIN_M) for o in body_ids}
        lim_drop = {o: max_drop[o] + DROP_MARGIN_M for o in body_ids}
        print(f"   {name[:40]}: settle limits from shipped layouts (xy, drop) m: " + str({o: (round(lim_xy[o], 3), round(lim_drop[o], 3)) for o in body_ids}), flush=True)
        kept, spawn_xy, tried, rejected = [], [], 0, {"goal_at_reset": 0, "unstable": 0, "duplicate": 0}
        for seed in range(2000):
            tried += 1
            np.random.seed(seed); env.seed(seed); env.reset()
            state = np.array(inner.sim.get_state().flatten(), dtype=np.float64)
            if inner._check_success():
                rejected["goal_at_reset"] += 1; continue
            if any(np.allclose(state, k) for k in kept):
                rejected["duplicate"] += 1; continue
            pos0 = {o: inner.sim.data.body_xpos[b].copy() for o, b in body_ids.items()}
            for _ in range(SETTLE_STEPS):
                env.step(noop)
            too_far = any(np.linalg.norm((inner.sim.data.body_xpos[b] - pos0[o])[:2]) > lim_xy[o]
                          or pos0[o][2] - inner.sim.data.body_xpos[b][2] > lim_drop[o] for o, b in body_ids.items())
            if too_far or inner._check_success():
                rejected["unstable"] += 1; continue
            kept.append(state); spawn_xy.append({o: pos0[o][:2].copy() for o in body_ids})
            if tried % 10 == 0 or len(kept) % 10 == 0:
                print(f"   {name[:40]}: kept {len(kept)}/{N_LAYOUTS} after {tried} tries {rejected}", flush=True)
            if len(kept) == N_LAYOUTS:
                break
        # xy spread of the generated layouts vs the shipped layouts of the same scene
        base_suite = suite.replace("_task", "")
        shipped = torch.load(Path(init.parent.parent) / base_suite / init.name, weights_only=False)
        ship_xy = []
        for st in shipped[:N_LAYOUTS]:
            env.reset(); env.set_init_state(st)
            ship_xy.append({o: inner.sim.data.body_xpos[b][:2].copy() for o, b in body_ids.items()})
        env.close()
        spread = {o: (round(float(np.std([d[o] for d in spawn_xy], axis=0).mean()), 4),
                      round(float(np.std([d[o] for d in ship_xy], axis=0).mean()), 4)) for o in body_ids}
        print(f"   xy std per object (generated, shipped {base_suite}): {spread}", flush=True)
        assert len(kept) == N_LAYOUTS, f"{suite}/{name}: only {len(kept)} layouts after {tried} tries {rejected}"
        arr = np.stack(kept)
        torch.save(arr, init)
        rec_path = PATCH_ROOT / "manifest.json"; rec = json.loads(rec_path.read_text())
        rec[f"{suite}/{init.name}"]["settle_limits_m"] = {o: [round(lim_xy[o], 4), round(lim_drop[o], 4)] for o in body_ids}
        rec[f"{suite}/{init.name}"].update({"patched_sha256": sha(init), "layouts": int(arr.shape[0]), "state_dim": int(arr.shape[1]),
                                           "tried": tried, "rejected": rejected, "rule": f"goal false at reset and after {SETTLE_STEPS} no-op steps; per-object xy move and drop within shipped max (+{XY_MARGIN_M} m xy, +{DROP_MARGIN_M} m drop) over {N_CALIBRATION} shipped layouts",
                                           "xy_std_generated_vs_shipped": spread})
        rec_path.write_text(json.dumps(rec, indent=2))
        print(f"{suite}/{name}: wrote {arr.shape} (tried {tried}, rejected {rejected}) in {time.time()-t0:.0f}s", flush=True)


def verify() -> None:
    from liberopro.liberopro import benchmark as pb, get_libero_path
    for suite, name, *_ in PATCHES:
        b = pb.get_benchmark_dict()[suite]()
        tid = next(i for i in range(b.n_tasks) if b.get_task(i).name == name)
        states = b.get_task_init_states(tid)
        # dimension must match a working sibling layout of the same scene in another suite
        base_suite = suite.replace("_task", "")
        base = Path(get_libero_path("init_states")) / base_suite / f"{name}.pruned_init"
        import torch
        base_dim = torch.load(base, weights_only=False).shape[1] if base.exists() else None
        print(f"{suite} t{tid}: {len(states)} layouts, dim {states.shape[1]} (same scene in {base_suite}: dim {base_dim})")


if __name__ == "__main__":
    {"fix": fix, "layouts": layouts, "verify": verify}[sys.argv[1]]()
