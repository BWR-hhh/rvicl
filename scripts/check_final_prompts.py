"""Refuse to drift from the frozen video-ICL prompts.

Usage: python scripts/check_final_prompts.py libero [--quiet]

Evaluates ``robots.libero.prompts.system.TASK_VIDEO_BOTH`` (the ``--task-video-mode both``
prompt) and compares its sha256 with ``prompts/libero_task_video_both.sha256``. Exit 0
when they match, 1 when they do not. ``run_eval.py`` runs this before launching a ``both`` sweep
so a stray edit cannot silently change what a LIBERO "both" result means.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # this repository
RPENT = Path(os.environ.get("RPENT_DIR", ROOT.parent / "RPent"))
FROZEN = {
    "libero": ("robots.libero.prompts.system", "TASK_VIDEO_BOTH", ROOT / "prompts" / "libero_task_video_both.sha256"),
}


def check(embodiment: str) -> tuple[bool, str, str]:
    module, attr, sha_file = FROZEN[embodiment]
    sys.path.insert(0, str(RPENT))
    text = getattr(importlib.import_module(module), attr)
    live = hashlib.sha256(text.encode()).hexdigest()
    frozen = sha_file.read_text().split()[0]
    return live == frozen, live, frozen


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    emb = args[0] if args else "libero"
    ok, live, frozen = check(emb)
    if not quiet or not ok:
        print(f"{emb} TASK_VIDEO_BOTH sha256 live={live[:16]} frozen={frozen[:16]} -> {'OK' if ok else 'DRIFTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
