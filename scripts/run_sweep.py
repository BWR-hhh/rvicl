#!/usr/bin/env python
"""Run LIBERO-PRO cells with the six-rule video prompt and score them from the simulator.

    python scripts/run_sweep.py --suites all --tasks 0-9 --seeds 1-3 --gpu 0 --parallel 2
    python scripts/run_sweep.py --suites libero_10_swap --tasks 4 --seeds 1 --dry-run

One row per attempt is appended to <out>/results.csv and the sweep resumes from it: a cell with a
valid row is skipped. A cell is solved when the run's state manifest (states.json) reports
``terminated``. Provider-side rejections (quota, policy, a stream the SDK never recovered) make the
attempt invalid and it is retried up to --attempts times; an episode itself is never re-run.

Paths come from the workspace layout (see README): RVICL_ROOT (default: the parent of this
repository), RPENT_DIR, RVICL_VIDEO_STORE, RVICL_MEMORY_DIR. The Pi0.5 and SAM 3 services are
taken from RPENT_VLA_ENDPOINT / RPENT_SAM3_ENDPOINT when set; otherwise rpent starts its own.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RVICL_ROOT", HERE.parents[1]))
REPO = Path(os.environ.get("RPENT_DIR", ROOT / "RPent"))
STORE = Path(os.environ.get("RVICL_VIDEO_STORE", ROOT / "task_videos"))
MEMORY = Path(os.environ.get("RVICL_MEMORY_DIR", ROOT / "memory" / "libero"))

SUITES = (
    "libero_spatial_swap", "libero_spatial_task", "libero_object_swap", "libero_object_task",
    "libero_goal_swap", "libero_goal_task", "libero_10_task", "libero_10_swap",
)
FIELDS = ("suite", "task", "seed", "cell", "attempt", "solved", "valid", "invalid_reason",
          "exit_code", "wall_s", "input_tokens", "output_tokens", "tool_calls", "out_dir")

#: Provider-side failures: the episode never ran, so the attempt is retried, not scored.
INVALID_MARKERS = (
    "violating our usage policy",
    "stream disconnected before completion",
    "stream closed before response.completed",
    "API Error: 502", "API Error: 504", "API Error: 529",
    "env_server exited with code",
    "insufficient_quota", "invalid_api_key",
    "response_too_many_failed_attempts", "exceeded retry limit",
)
#: The planner SDK recovers from these by itself; fatal only when the run ends there.
RECOVERABLE = ("stream disconnected before completion", "stream closed before response.completed")
_TOOL_LINE = re.compile(r"\[tool(?:<-|->)\] [a-z_][a-z0-9_]*:")


@dataclass(frozen=True)
class Cell:
    suite: str
    task: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.suite.replace('libero_', '')}_t{self.task}_s{self.seed}"


def parse_ids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def rpent_binary() -> str:
    venv = REPO / ".venv" / "bin" / "rpent"
    if venv.exists():
        return str(venv)
    found = shutil.which("rpent")
    if not found:
        raise SystemExit(f"rpent not found: no {venv} and nothing on PATH (activate the RPent environment)")
    return found


def command(cell: Cell, args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        rpent_binary(), "--robot", "libero", "--libero-type", "pro",
        "--suite", cell.suite, "--task", str(cell.task), "--seed", str(cell.seed),
        "--planner", args.planner, "--model", args.model, "--reasoning-effort", args.reasoning_effort,
        "--memory-profile", "local", "--memory-dir", str(MEMORY),
        "--max-turns", str(args.max_turns), "--planner-timeout-s", str(args.planner_timeout_s),
        "--max-episode-steps", str(args.max_episode_steps),
        "--cuda-device", str(args.gpu), "--output-dir", str(out_dir),
    ]
    cmd += ["--task-video-dir", str(STORE)]
    if cell.suite.endswith("_task"):
        # goal-rewriting suites: the demonstration was recorded for the original instruction
        cmd += ["--task-video-cross-task"]
    if args.vla_endpoint:
        cmd += ["--vla-endpoint", args.vla_endpoint]
    if args.sam3_endpoint:
        cmd += ["--sam3-endpoint", args.sam3_endpoint]
    return cmd


def environment(args: argparse.Namespace, worker: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_GL": "egl",
                "MUJOCO_EGL_DEVICE_ID": str(args.gpu), "LIBERO_TYPE": "pro"})
    # Concurrent Codex sessions must not share one CODEX_HOME (its sqlite state locks under
    # parallel writes); the same holds for the Claude Agent SDK's config dir.
    for var, default in (("CODEX_HOME", ROOT / "codex_home"), ("CLAUDE_CONFIG_DIR", ROOT / "claude_home")):
        home = Path(env.get(var) or default) / f"w{worker}"
        home.mkdir(parents=True, exist_ok=True)
        env[var] = str(home)
    env.pop("RPENT_TASK_VIDEO_DIR", None)
    return env


def invalid_reason(out_dir: Path) -> str:
    """Why this attempt must not count as a failure, or '' when the episode really ran."""
    if (out_dir / "KILLED_BY_OPERATOR").exists():
        return "killed by operator"
    log = out_dir / "run.stdout.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    for marker in INVALID_MARKERS:
        if marker not in text:
            continue
        if marker in RECOVERABLE and "[tool<-]" in text[text.rindex(marker):]:
            continue  # the SDK reconnected and the planner kept working
        return marker
    if not _TOOL_LINE.search(text):
        return "planner made no call"  # never looked at the scene: not an attempt at the task
    return ""


def outcome(out_dir: Path) -> dict[str, object]:
    """Ground-truth success (simulator ``terminated``) plus planner usage for one finished run."""
    solved = False
    manifest = out_dir / "states.json"
    if manifest.exists():
        try:
            records = json.loads(manifest.read_text())
            entries = records.get("steps", records.get("records", [])) if isinstance(records, dict) else records
            if isinstance(entries, dict):
                entries = list(entries.values())
            solved = any(bool((r or {}).get("terminated")) for r in entries)
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    stats: dict[str, object] = {}
    transcripts = sorted(out_dir.glob("transcript_*.json"))
    if transcripts:
        try:
            record = json.loads(transcripts[-1].read_text())
            usage = record.get("stats") or {}
            stats = {"input_tokens": usage.get("total_input_tokens"),
                     "output_tokens": usage.get("total_output_tokens"),
                     "tool_calls": usage.get("tool_calls")}
        except (OSError, ValueError):
            pass
    reason = "" if solved else invalid_reason(out_dir)
    return {"solved": int(solved), "valid": int(not reason), "invalid_reason": reason, **stats}


def kill_tree(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=10)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def run_attempt(cell: Cell, attempt: int, args: argparse.Namespace, worker: int) -> dict[str, object]:
    out_dir = Path(args.out) / f"{cell.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(cell, args, out_dir)
    row: dict[str, object] = {f: "" for f in FIELDS}
    row.update(suite=cell.suite, task=cell.task, seed=cell.seed, cell=cell.tag, attempt=attempt, out_dir=str(out_dir))
    if args.dry_run:
        print("   ", " ".join(cmd[1:]), flush=True)
        return {**row, "solved": 0, "valid": 0, "invalid_reason": "dry run"}
    started = time.time()
    with (out_dir / "run.stdout.log").open("w") as log:
        # rpent spawns env_server and the planner; a session of its own lets a timeout take
        # the whole tree down instead of orphaning a planner that keeps calling the API.
        proc = subprocess.Popen(cmd, cwd=REPO, env=environment(args, worker), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=args.cell_timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            exit_code = -9
    o = outcome(out_dir)
    if exit_code == -9 and o["invalid_reason"] not in RECOVERABLE:
        o.update(valid=1, invalid_reason="")  # running out of budget is a real attempt
    return {**row, **o, "exit_code": exit_code, "wall_s": round(time.time() - started, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="all", help="comma list of suites, or 'all' for the eight LIBERO-PRO suites")
    p.add_argument("--tasks", default="0-9")
    p.add_argument("--seeds", default="1-3", help="seed 0 is the seed the shipped memory was built on; keep it out")
    p.add_argument("--out", default=str(ROOT / "runs" / "sweep"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--parallel", type=int, default=1, help="cells in flight (each adds ~3 GB host memory)")
    p.add_argument("--planner", default="codex")
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--planner-timeout-s", type=int, default=5000)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--cell-timeout-s", type=int, default=7200)
    p.add_argument("--attempts", type=int, default=3, help="re-runs for provider-side rejections")
    p.add_argument("--retry-backoff-s", type=int, default=180)
    p.add_argument("--vla-endpoint", default=os.environ.get("RPENT_VLA_ENDPOINT", ""))
    p.add_argument("--sam3-endpoint", default=os.environ.get("RPENT_SAM3_ENDPOINT", ""))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    suites = list(SUITES) if args.suites == "all" else [s.strip() for s in args.suites.split(",") if s.strip()]
    cells = [Cell(s, t, sd) for s in suites for sd in parse_ids(args.seeds) for t in parse_ids(args.tasks)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    done = set()
    if results.exists():
        with results.open() as f:
            done = {r["cell"] for r in csv.DictReader(f) if r.get("valid") == "1"}
    todo = [c for c in cells if c.tag not in done]
    print(f"{len(todo)} cells to run ({len(cells) - len(todo)} already valid in {results})", flush=True)

    lock = threading.Lock()

    def write(row: dict[str, object]) -> None:
        with lock, results.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(FIELDS), extrasaction="ignore")
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    def run(cell: Cell, worker: int) -> None:
        for attempt in range(1, args.attempts + 1):
            row = run_attempt(cell, attempt, args, worker)
            if not args.dry_run:
                write(row)
            tag = "SOLVED" if row["solved"] else ("failed" if row["valid"] else f"INVALID ({row['invalid_reason']})")
            print(f"{time.strftime('%H:%M:%S')} {cell.tag} attempt {attempt}: {tag} ({row['wall_s']} s)", flush=True)
            if row["valid"] or args.dry_run or attempt == args.attempts:
                return
            time.sleep(args.retry_backoff_s)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for i, cell in enumerate(todo):
            pool.submit(run, cell, i % args.parallel)

    if results.exists():
        latest: dict[str, dict[str, str]] = {}
        with results.open() as f:
            for r in csv.DictReader(f):
                if r.get("valid") == "1":
                    latest[r["cell"]] = r
        print("\n| suite | solved |\n|---|---|")
        total = n = 0
        for s in suites:
            rows = [r for r in latest.values() if r["suite"] == s]
            k = sum(int(r["solved"]) for r in rows)
            total += k
            n += len(rows)
            print(f"| {s} | {k}/{len(rows)} |")
        print(f"| total | {total}/{n} |")


if __name__ == "__main__":
    main()
