#!/usr/bin/env python3
"""Compute remaining evaluate indices and times from ablation log."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "src" / "optimus1" / "conf" / "benchmark"

EVAL_RE = re.compile(r"Evaluate Task:\s*(.+?)\s+in 1 times")


def load_tasks(benchmark: str) -> list[dict]:
    path = BENCHMARK_DIR / f"{benchmark}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tasks = sorted(data["all_task"], key=lambda t: t["id"])
    return tasks


def count_episodes(log_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not log_path.is_file():
        return counts
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                counts[m.group(1).strip()] += 1
    return counts


def compute_resume(benchmark: str, log_path: Path, times_per_task: int) -> tuple[str, int] | None:
    tasks = load_tasks(benchmark)
    counts = count_episodes(log_path)

    for task in tasks:
        instruction = task["instruction"].strip()
        done = min(counts.get(instruction, 0), times_per_task)
        if done < times_per_task:
            remaining = times_per_task - done
            if remaining < times_per_task:
                return f"[{task['id']}]", remaining
            remaining_ids = [t["id"] for t in tasks if t["id"] >= task["id"]]
            return "[" + ",".join(str(i) for i in remaining_ids) + "]", times_per_task
    return None


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: ablation_compute_resume.py <wooden|stone> <log> <times_per_task>", file=sys.stderr)
        sys.exit(2)

    benchmark, log, times_s = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    result = compute_resume(benchmark, log, times_s)
    if result is None:
        print("DONE")
        sys.exit(0)
    eval_str, times = result
    print(eval_str, times)


if __name__ == "__main__":
    main()
