# The per-task video store

One directory per base task, 40 in all (`libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, tasks 0-9), named `<family>_t<task>_demo_0`:

```
index.json        task, language, provenance, keyframe list, phases, moments, task objects/goal/regions
sheet_01..08.png  16 keyframes, two per sheet: agentview | wrist, captioned with step, gripper, eef xyz
agentview.npy     (T, 128, 128, 3) uint8 - every recorded frame, for view_demo_clip
wrist.npy         (T, 128, 128, 3) uint8
```

The source is `demo_0` of each task in the original LIBERO datasets (human teleoperation). LIBERO-PRO
ships no demonstrations; its `*_swap` suites keep the original goals and its `*_task` suites rewrite
them, which is why those cells run with `--task-video-cross-task`.

## Use the released store

Unpack `task_videos_libero.tar.gz` (checksum in the `.sha256` file beside it) so that the result is
`<workspace>/task_videos/<video_id>/...`, or point `RVICL_VIDEO_STORE` at wherever you put it.

## Rebuild it

Needs the RPent environment (`liberopro`, EGL rendering) and the LIBERO datasets
(`libero_10/*_demo.hdf5` etc.; `LIBERO_DEMO_ROOT` points at the directory that holds the four
family folders). Three passes per family, in this order:

```bash
export LIBERO_DEMO_ROOT=/path/to/datasets/libero
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
for fam in libero_spatial libero_object libero_goal libero_10; do
  python video_store/build_videos.py    --family $fam --tasks 0-9 --out   ../task_videos   # keyframes, re-rendered at 512 px from the recorded sim states
  python video_store/build_clips.py     --family $fam --tasks 0-9 --store ../task_videos   # recorded 128 px frames + kinematic phases/moments
  python video_store/build_semantics.py --family $fam --tasks 0-9 --store ../task_videos   # object-level phase meaning from a headless replay
done
```

`build_videos.py` renders the keyframes in RPent's own camera orientation so the planner's
`agentview_high.png` / `wrist_high.png` and the demonstration look alike; the selection is
event-driven (start, pre-grasp, close, lift, over-target, open, plus uniform fill, 16 frames).
`build_clips.py` is strictly additive and needs no renderer. `build_semantics.py` replays the
recorded states through a headless LIBERO environment and writes, per phase, which object moved
into which named region and which fixture opened or closed, and adds moments for articulation
events (a door closing is a moment the planner can ask for).
