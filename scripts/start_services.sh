#!/bin/bash
# Start the shared Pi0.5 and SAM 3 services once; every cell of a sweep then reuses them.
# Usage: scripts/start_services.sh [gpu] [vla_port] [sam3_port]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${RVICL_ROOT:-$(cd "$HERE/.." && pwd)}"
RPENT="${RPENT_DIR:-$ROOT/RPent}"
GPU=${1:-0}; VLA_PORT=${2:-8220}; SAM3_PORT=${3:-8114}
: "${PI05_CHECKPOINT_PATH:?set PI05_CHECKPOINT_PATH}" "${SAM3_CHECKPOINT_PATH:?set SAM3_CHECKPOINT_PATH}"
mkdir -p "$ROOT/runs/services"
cd "$RPENT"
setsid nohup python rpent/robots/components/pi05_vla_server.py --embodiment libero \
  --transport http --host 127.0.0.1 --port "$VLA_PORT" --cuda-device "$GPU" \
  > "$ROOT/runs/services/vla_server_$VLA_PORT.log" 2>&1 &
setsid nohup python rpent/robots/components/sam3_server.py \
  --transport http --host 127.0.0.1 --port "$SAM3_PORT" --cuda-device "$GPU" \
  > "$ROOT/runs/services/sam3_server_$SAM3_PORT.log" 2>&1 &
echo "starting Pi0.5 on 127.0.0.1:$VLA_PORT and SAM 3 on 127.0.0.1:$SAM3_PORT (GPU $GPU); logs in $ROOT/runs/services"
echo "export RPENT_VLA_ENDPOINT=http://127.0.0.1:$VLA_PORT"
echo "export RPENT_SAM3_ENDPOINT=http://127.0.0.1:$SAM3_PORT"
