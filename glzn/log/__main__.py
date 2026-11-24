"""Inspect existing GLZN run logs from the terminal."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .reader import summarize_log


def _log_paths(paths: Sequence[Path]) -> list[Path]:
    logs = set()
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            logs.add(path.resolve())
        elif path.is_dir():
            logs.update(p.resolve() for p in path.rglob("metrics.jsonl") if p.is_file())
        else:
            raise FileNotFoundError(f"{path} does not exist.")
    return sorted(logs)


def _age(seconds: float) -> str:
    if abs(seconds) < 60:
        return f"{seconds:.0f}s"
    if abs(seconds) < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _print_table(rows: list[tuple[str, ...]]) -> None:
    columns = zip(*rows, strict=True)
    widths = [max(map(len, column)) for column in columns]
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glzn.log",
        description="Summarize existing GLZN run logs.",
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, metavar="PATH",
        help="Log files or directories to search for metrics.jsonl.",
    )
    parser.add_argument(
        "--window", type=int, default=100, metavar="RECORDS",
        help="Maximum recent complete lines per log (default: 100).",
    )
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive.")
    try:
        paths = _log_paths(args.paths)
    except OSError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    if not paths:
        parser.exit(1, f"{parser.prog}: no metrics.jsonl files found.\n")

    rows = [(
        "Run", "Epoch", "Updates", "Last record", "Updates/min",
        "AMP skips", "Nonfinite", "Records", "Log",
    )]
    for path in paths:
        try:
            summary = summarize_log(path, window=args.window)
        except OSError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        if summary.invalid_lines:
            print(f"{path}: {summary.invalid_lines} invalid complete line(s) in the window.", file=sys.stderr)
        latest = summary.latest
        if latest is None:
            print(f"{path}: no complete GLZN records in the window.", file=sys.stderr)
            continue
        rate = summary.updates_per_minute
        rows.append((
            latest.run_id, str(latest.epoch), str(latest.fullstep), _age(time.time() - latest.time),
            f"{rate:.1f}" if rate is not None else "-", str(summary.amp_skips),
            str(summary.nonfinite), str(summary.records), str(path),
        ))
    if len(rows) == 1:
        return 1
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
