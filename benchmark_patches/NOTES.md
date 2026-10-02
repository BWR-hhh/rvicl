# LIBERO-PRO patches applied to this deployment

Installed package: the `liberopro` package of the RPent environment. Originals are kept under `originals/`,
hashes and generation stats in `manifest.json`. Patch script: `benchmark_patches/patch_liberopro_tasks.py`.

## What was broken

Three goal-rewritten tasks ship with unbalanced parentheses in `(:goal ...)`:

| task | shipped goal | fixed goal |
|---|---|---|
| libero_10_task t2 ("turn on the stove and put the pan on it") | `And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))` | `(And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))` |
| libero_spatial_task t3 ("... bowl on the top of the cabinet ...") | `(On akita_black_bowl_2 plate_1))` | `(And (On akita_black_bowl_2 plate_1))` |
| libero_spatial_task t7 (same language) | `(On akita_black_bowl_2 plate_1))` | `(And (On akita_black_bowl_2 plate_1))` |

The fixed forms are exactly those of the correctly-formed sibling tasks in the same suites.
For t3/t7 the (:init) places `akita_black_bowl_2` on `wooden_cabinet_1_top_side`, matching
the rewritten instruction.

Consequence of the bug: `robosuite_parse_problem` catches the parse error and silently loads
`os.listdir(bddl_files/libero_10)[0]` (an unrelated kitchen problem). LIBERO-PRO's layout
pruning then dropped every state and shipped an empty `.pruned_init` (364 bytes, shape (0,)),
so RPent's env_server crashes on `seed % 0` before the planner starts. The authors' shipped
memory contains a solved seed-0 record for libero_10_task t2, so their LIBERO-PRO copy did not
have this problem: the paper's L10-T / Spat-T numbers cover all 10 tasks.

## How the layouts were rebuilt

LIBERO-PRO does not ship its layout generation or pruning code. For each fixed task: build the
env from the fixed BDDL, `np.random.seed(s); env.seed(s); env.reset()` for s = 0, 1, ...;
record `sim.get_state().flatten()` (float64). Objects spawn slightly above their support and
settle, and how far depends on the scene (the shipped layouts drop up to 7 cm in the L10 kitchen
scene and up to 16 cm for a bowl stacked on a box in the spatial scenes, with ~0 horizontal
motion). A layout is kept only if the goal is false at reset and after 20 no-op steps, and every
movable object settles no more than in the shipped layouts of the same scene: horizontal move
<= shipped max + 1 cm (at least 2 cm), drop <= shipped max + 3 cm. (libero_10_task t2 was
generated with a flat 2 cm / 10 cm limit, which equals this rule for its scene.) Exact duplicates are dropped; the first 50
are written with `torch.save` (same format as the shipped files). The object xy spread of the
generated layouts is compared with the shipped layouts of the same scene (see manifest).
Report results on these three tasks as "patched task", separately identifiable.

## Result (2026-09-17 06:12)

| task | layouts | state dim (shipped same scene) | accepted / tried | object xy std, generated vs shipped |
|---|---|---|---|---|
| libero_10_task t2 | 50 | 47 (47) | 50 / 50 | frypan 1.45 vs 1.29 cm, moka pot 1.36 vs 1.42 cm |
| libero_spatial_task t3 | 50 | 92 (92) | 50 / 50 | bowl_1 0.87 vs 0.95, bowl_2 1.98 vs 1.90, plate 0.87 vs 0.90 cm |
| libero_spatial_task t7 | 50 | 92 (92) | 50 / 50 | bowl_1 0.89 vs 1.04, bowl_2 1.65 vs 1.71, plate 0.87 vs 0.88 cm |

All three build and reset through RPent's own `robots/libero/env_server.make_env` (the path that
used to raise ZeroDivisionError), and `scripts/run_eval.py` no longer skips them.
