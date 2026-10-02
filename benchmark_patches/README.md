# LIBERO-PRO benchmark fix

Three goal-rewritten LIBERO-PRO tasks ship with malformed BDDL goals (unbalanced parentheses) and,
as a consequence, empty layout files: `libero_10_task` t2 ("turn on the stove and put the pan on
it"), `libero_spatial_task` t3 and t7 (bowl on the cabinet to the plate). The parser falls back to
an unrelated problem and the environment cannot be built, so these cells would be missing from any
evaluation. `NOTES.md` documents the bug, the fix and how the layouts were regenerated;
`manifest.json` holds the hashes of the shipped and patched files and the generation statistics;
`originals/` keeps the shipped files.

Apply once, inside the RPent environment (the script edits the installed `liberopro` package and
backs the originals up under `LIBERO_PRO_PATCH_ROOT`, default: this directory):

```bash
python benchmark_patches/patch_liberopro_tasks.py fix       # back up, fix the goals, parse check
python benchmark_patches/patch_liberopro_tasks.py layouts   # regenerate 50 layouts per task (needs the simulator)
python benchmark_patches/patch_liberopro_tasks.py verify    # re-check the installed files against manifest.json
```

Report results on these three tasks as patched tasks.
