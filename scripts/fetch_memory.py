#!/usr/bin/env python
"""Download the LIBERO memory corpus at the revision this pipeline was run with.

    python scripts/fetch_memory.py            # -> <workspace>/memory/libero
    python scripts/fetch_memory.py <dir>      # -> <dir>/libero

RPent syncs its memory from the Hugging Face dataset RLinf/RPent-memory; the dataset was
reorganised on 2026-09-25/26, so the pipeline pins the earlier revision. Runs read the corpus
through ``--memory-profile local --memory-dir <dir>/libero`` and never sync it again.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = "RLinf/RPent-memory"
REVISION = "8d34642e420e35ea8a883ddabe52471433920177"  # 2026-09-08


def main() -> None:
    from huggingface_hub import snapshot_download

    root = Path(os.environ.get("RVICL_ROOT", Path(__file__).resolve().parents[2]))
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "memory"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                      allow_patterns=["libero/**"], local_dir=str(target))
    n = sum(1 for p in (target / "libero").rglob("*") if p.is_file() and ".cache" not in p.parts)
    print(f"{n} files in {target / 'libero'} (RLinf/RPent-memory @ {REVISION[:8]})")


if __name__ == "__main__":
    main()
