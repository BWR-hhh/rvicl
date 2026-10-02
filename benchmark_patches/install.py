#!/usr/bin/env python
"""Install the task files under benchmark_patches/files/ into the liberopro package of the active
environment (bddl_files/ and init_states/). The shipped files are kept beside them as *.orig.

    python benchmark_patches/install.py          # install
    python benchmark_patches/install.py --check  # report only
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "files"


def main() -> int:
    from liberopro.liberopro import get_libero_path

    check = "--check" in sys.argv
    roots = {".bddl": Path(get_libero_path("bddl_files")), ".pruned_init": Path(get_libero_path("init_states"))}
    pending = 0
    for src in sorted(HERE.rglob("*")):
        if not src.is_file():
            continue
        dst = roots[src.suffix] / src.relative_to(HERE)
        same = dst.exists() and hashlib.sha256(dst.read_bytes()).digest() == hashlib.sha256(src.read_bytes()).digest()
        print(f"{'ok      ' if same else 'install '}{dst}")
        if same or check:
            pending += not same
            continue
        if dst.exists() and not dst.with_suffix(dst.suffix + ".orig").exists():
            shutil.copy(dst, dst.with_suffix(dst.suffix + ".orig"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return 1 if (check and pending) else 0


if __name__ == "__main__":
    raise SystemExit(main())
