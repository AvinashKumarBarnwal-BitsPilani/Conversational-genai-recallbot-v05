"""Task 4(a) ablation: Session A then Session B for each configuration.

Every session is its own OS process (src.dialogue.run_session), so Session B
of each config only knows what that config's Session A saved to disk.
Experiments are named t4_<config>; run IDs are t4_<config>_A and t4_<config>_B.

    python scripts/run_ablation.py                     # all four configs
    python scripts/run_ablation.py --configs hybrid_mem
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("no_rag", "hybrid", "hybrid_sem", "hybrid_mem")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--prefix", default="t4")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for config in args.configs:
        experiment = f"{args.prefix}_{config}"
        for session in ("A", "B"):
            command = [sys.executable, "-m", "src.dialogue.run_session", "--session", session,
                       "--config", config, "--experiment", experiment, "--weighting", "A"]
            started = time.time()
            print(f"\n=== {experiment} Session {session} ===", flush=True)
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
            for line in completed.stdout.splitlines():
                if line.startswith(("[run_session", "    ->", "[probe")):
                    print(line[:170], flush=True)
            if completed.returncode != 0:
                print(completed.stdout[-3000:], completed.stderr[-3000:], sep="\n")
                print(f"FAILED: {experiment} Session {session}")
                return 1
            print(f"--- {experiment} Session {session} done in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
