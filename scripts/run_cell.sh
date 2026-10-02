#!/bin/bash
# Run one LIBERO-PRO cell with the six-rule video prompt (the RV-ICL method).
# Usage: scripts/run_cell.sh <suite> <task> <seed> [extra rpent args...]
#   e.g. scripts/run_cell.sh libero_10_swap 4 1
#        scripts/run_cell.sh libero_10_task 2 1 --task-video-cross-task   (goal-rewriting *_task suites)
# Requires the patched RPent on PATH (its venv activated), the video store, and the services from
# start_services.sh (or omit the endpoints and rpent starts its own Pi0.5 + SAM 3 per cell).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${RVICL_ROOT:-$(cd "$HERE/.." && pwd)}"
RPENT="${RPENT_DIR:-$ROOT/RPent}"
STORE="${RVICL_VIDEO_STORE:-$ROOT/task_videos}"
MEMORY="${RVICL_MEMORY_DIR:-$ROOT/memory/libero}"
SUITE="$1"; TASK="$2"; SEED="$3"; shift 3
OUT="$ROOT/runs/$(date +%Y%m%d_%H%M%S)_${SUITE}_t${TASK}_s${SEED}"
mkdir -p "$OUT"
python "$HERE/scripts/check_final_prompts.py" libero --quiet
ENDPOINTS=()
[ -n "${RPENT_VLA_ENDPOINT:-}" ] && ENDPOINTS+=(--vla-endpoint "$RPENT_VLA_ENDPOINT")
[ -n "${RPENT_SAM3_ENDPOINT:-}" ] && ENDPOINTS+=(--sam3-endpoint "$RPENT_SAM3_ENDPOINT")
cd "$RPENT"
echo "output dir: $OUT"
exec rpent --robot libero --libero-type "${LIBERO_TYPE:-pro}" \
  --suite "$SUITE" --task "$TASK" --seed "$SEED" \
  --planner "${RVICL_PLANNER:-codex}" --model "${RVICL_MODEL:-gpt-6-astra}" --reasoning-effort "${RVICL_REASONING_EFFORT:-low}" \
  --memory-profile local --memory-dir "$MEMORY" \
  --task-video-dir "$STORE" --task-video-mode both \
  --cuda-device "${CUDA_DEVICE:-0}" \
  --max-turns 100 --planner-timeout-s 5000 --max-episode-steps 10000 \
  --output-dir "$OUT" "${ENDPOINTS[@]}" "$@"
