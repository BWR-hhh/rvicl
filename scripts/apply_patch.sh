#!/bin/bash
# Check out the RPent (HarnessVLA) commit this work is based on and apply the RV-ICL LIBERO patch.
# Usage: scripts/apply_patch.sh [target_dir]   (default: ../RPent next to this repository)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$HERE/../RPent}"
BASE=d6daf3413dd203f44bbd4a9f3f08c19db0c2063c   # RPent main, 2026-09-15 (#182)
if [ ! -d "$TARGET/.git" ]; then
  git clone https://github.com/RLinf/RPent.git "$TARGET"
fi
cd "$TARGET"
git fetch --quiet origin
git checkout --quiet "$BASE"
git apply --check "$HERE/patches/rvicl-libero.patch"
git apply "$HERE/patches/rvicl-libero.patch"
echo "applied patches/rvicl-libero.patch on RPent@$BASE in $TARGET"
echo "next: pip install -e \".[libero-pro]\" in a Python 3.11 environment (see README)"
