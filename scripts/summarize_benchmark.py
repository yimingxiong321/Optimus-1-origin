#!/usr/bin/env python3
"""Summarize Optimus-1 benchmark logs into SR / AS / AT tables.

Episode-level metrics are parsed from Hydra or tee logs:

    store plan of <task> to .../plan/{success,failed}/...
    Evaluate Task: <task> in 1 times, sum steps: N

Paper-style metrics:
    SR = success / episodes
    AS = mean steps over successful episodes (∞ if none succeed)
    AT = AS / 20  (MineRL ~20 FPS)

Examples:

    python scripts/summarize_benchmark.py
    python scripts/summarize_benchmark.py --benchmark wooden --out logs/summaries
    python scripts/summarize_benchmark.py --logs logs/eval/2026-08-29/02-32/main.log
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS_PER_SECOND = 20.0

STORE_PLAN_RE = re.compile(
    r"store plan of\s+(?P<task>.+?)\s+to\s+\S*?plan/(?P<status>success|failed)/",
    re.IGNORECASE,
)
EVAL_TASK_RE = re.compile(
    r"Evaluate Task:\s+(?P<task>.+?)\s+in\s+\d+\s+times,\s+sum steps:\s+(?P<steps>\d+)",
)
SUMMARY_RE = re.compile(r"^Summary:\s+(?P<body>\{.*\})\s*$")
STEP_MONITOR_RE = re.compile(r"'StepMonitor':\s*(\d+)")
COMPLETED_RE = re.compile(r"All tasks are completed!")
FAILED_RE = re.compile(r"Some tasks are not completed!")
MARKUP_RE = re.compile(r"\[/?[^\]]+\]")
YAML_TASK_RE = re.compile(r"instruction:\s*([^}\n]+)")


@dataclass
class Episode:
    task: str
    status: str
    steps: int | None
    source: str


@dataclass
class TaskStats:
    task: str
    episodes: int = 0
    success: int = 0
    failed: int = 0
    success_steps: list[int] = field(default_factory=list)
    all_steps: list[int] = field(default_factory=list)

    @property
    def sr(self) -> float:
        return self.success / self.episodes if self.episodes else 0.0

    @property
    def as_steps(self) -> float | None:
        if not self.success_steps:
            return None
        return sum(self.success_steps) / len(self.success_steps)

    @property
    def at_seconds(self) -> float | None:
        avg = self.as_steps
        return None if avg is None else avg / STEPS_PER_SECOND


def strip_markup(text: str) -> str:
    return MARKUP_RE.sub("", text)


def iter_log_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.log")))
    return files


def parse_log(path: Path) -> list[Episode]:
    text = strip_markup(path.read_text(errors="replace"))
    episodes: list[Episode] = []
    pending_status: str | None = None
    pending_task: str | None = None
    pending_steps: int | None = None

    def flush() -> None:
        nonlocal pending_status, pending_task, pending_steps
        if pending_task and pending_status:
            episodes.append(
                Episode(
                    task=normalize_task(pending_task),
                    status=pending_status,
                    steps=pending_steps,
                    source=str(path),
                )
            )
        pending_status = None
        pending_task = None
        pending_steps = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if COMPLETED_RE.search(line):
            pending_status = "success"
            continue
        if FAILED_RE.search(line):
            pending_status = "failed"
            continue

        store = STORE_PLAN_RE.search(line)
        if store:
            if pending_task and pending_status:
                flush()
            pending_task = store.group("task").strip()
            pending_status = store.group("status").lower()
            continue

        summary = SUMMARY_RE.search(line)
        if summary:
            steps = sum(int(x) for x in STEP_MONITOR_RE.findall(summary.group("body")))
            pending_steps = steps
            if pending_task and pending_status:
                flush()
            continue

        eval_match = EVAL_TASK_RE.search(line)
        if eval_match:
            task = eval_match.group("task").strip()
            steps = int(eval_match.group("steps"))
            if pending_task and normalize_task(pending_task) != normalize_task(task):
                flush()
            pending_task = task
            pending_steps = steps
            if pending_status:
                flush()
            continue

    flush()
    return episodes


def normalize_task(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).strip()


def load_benchmark_tasks(benchmark: str) -> list[str]:
    yaml_path = REPO_ROOT / "src/optimus1/conf/benchmark" / f"{benchmark}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"benchmark yaml not found: {yaml_path}")
    tasks: list[str] = []
    for line in yaml_path.read_text().splitlines():
        match = YAML_TASK_RE.search(line)
        if match:
            tasks.append(normalize_task(match.group(1)))
    return tasks


def aggregate(episodes: list[Episode], last_n: int | None = None) -> dict[str, TaskStats]:
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.task].append(ep)

    stats: dict[str, TaskStats] = {}
    for task, items in by_task.items():
        chosen = items[-last_n:] if last_n else items
        row = TaskStats(task=task)
        for ep in chosen:
            row.episodes += 1
            if ep.steps is not None:
                row.all_steps.append(ep.steps)
            if ep.status == "success":
                row.success += 1
                if ep.steps is not None:
                    row.success_steps.append(ep.steps)
            else:
                row.failed += 1
        stats[task] = row
    return stats


def format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "∞"
    return f"{value:.{digits}f}"


def print_table(ordered: list[TaskStats]) -> None:
    headers = ["Task", "N", "Success", "Fail", "SR", "AS", "AT(s)"]
    rows: list[list[str]] = []
    for row in ordered:
        rows.append(
            [
                row.task,
                str(row.episodes),
                str(row.success),
                str(row.failed),
                f"{row.sr:.2%}",
                format_num(row.as_steps),
                format_num(row.at_seconds),
            ]
        )
    if not rows:
        print("No episodes found.")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    if len(ordered) > 1:
        n = sum(r.episodes for r in ordered)
        ok = sum(r.success for r in ordered)
        as_vals = [r.as_steps for r in ordered if r.as_steps is not None]
        overall_as = sum(as_vals) / len(as_vals) if as_vals else None
        print()
        print(
            f"Overall  N={n}  Success={ok}  SR={ok / n:.2%}  "
            f"AS={format_num(overall_as)}  AT={format_num(None if overall_as is None else overall_as / STEPS_PER_SECOND)}"
        )


def write_outputs(ordered: list[TaskStats], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    json_path = out_dir / f"{name}.json"

    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["task", "episodes", "success", "failed", "sr", "as", "at"])
        for row in ordered:
            writer.writerow(
                [
                    row.task,
                    row.episodes,
                    row.success,
                    row.failed,
                    f"{row.sr:.6f}",
                    "" if row.as_steps is None else f"{row.as_steps:.4f}",
                    "" if row.at_seconds is None else f"{row.at_seconds:.4f}",
                ]
            )

    lines = [
        f"# Benchmark summary: {name}",
        "",
        "| Task | N | Success | Fail | SR | AS | AT (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            f"| {row.task} | {row.episodes} | {row.success} | {row.failed} | "
            f"{row.sr:.2%} | {format_num(row.as_steps)} | {format_num(row.at_seconds)} |"
        )
    if len(ordered) > 1:
        n = sum(r.episodes for r in ordered)
        ok = sum(r.success for r in ordered)
        as_vals = [r.as_steps for r in ordered if r.as_steps is not None]
        overall_as = sum(as_vals) / len(as_vals) if as_vals else None
        lines += [
            "",
            f"Overall: N={n}, Success={ok}, SR={ok / n:.2%}, "
            f"AS={format_num(overall_as)}, AT={format_num(None if overall_as is None else overall_as / STEPS_PER_SECOND)}",
            "",
            "AS / AT are averages over **successful** episodes only (∞ if a task never succeeded).",
        ]
    md_path.write_text("\n".join(lines) + "\n")

    payload = []
    for row in ordered:
        item = asdict(row)
        item["sr"] = row.sr
        item["as"] = row.as_steps
        item["at"] = row.at_seconds
        payload.append(item)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {csv_path}\nWrote {md_path}\nWrote {json_path}")


def default_logs() -> list[Path]:
    hydra = [
        REPO_ROOT / "logs/eval/2026-08-29/00-18/main.log",
        REPO_ROOT / "logs/eval/2026-08-29/02-32/main.log",
    ]
    found = [p for p in hydra if p.exists()]
    if found:
        return found
    tee = [
        REPO_ROOT / "logs/wooden_benchmark_30x.log",
        REPO_ROOT / "logs/wooden_benchmark_30x_resume_v3.log",
    ]
    return [p for p in tee if p.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="*",
        type=Path,
        help="Log files or directories. Default: wooden Hydra + tee logs.",
    )
    parser.add_argument(
        "--benchmark",
        default="wooden",
        help="Only keep tasks listed in conf/benchmark/<name>.yaml. Use 'all' for every parsed task.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Keep only the last N episodes per task (useful after resume/re-runs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "logs/summaries",
        help="Directory for csv/md/json outputs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output basename. Default: <benchmark>_benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in (args.logs or [])]
    files = iter_log_files(log_paths) if log_paths else default_logs()
    if not files:
        print("No log files found.", file=sys.stderr)
        return 1

    episodes: list[Episode] = []
    for path in files:
        found = parse_log(path)
        print(f"Parsed {len(found):4d} episodes from {path}")
        episodes.extend(found)

    if args.benchmark != "all":
        allowed = {normalize_task(t) for t in load_benchmark_tasks(args.benchmark)}
        episodes = [ep for ep in episodes if ep.task in allowed]

    stats = aggregate(episodes, last_n=args.last)
    if args.benchmark != "all":
        order = [normalize_task(t) for t in load_benchmark_tasks(args.benchmark)]
        ordered = [stats[t] for t in order if t in stats]
        missing = [t for t in order if t not in stats]
        if missing:
            print("Missing tasks (no episodes):", ", ".join(missing), file=sys.stderr)
    else:
        ordered = [stats[k] for k in sorted(stats)]

    print()
    print_table(ordered)
    write_outputs(ordered, args.out, args.name or f"{args.benchmark}_benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
