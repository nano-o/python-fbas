#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any


def _is_valid_dataset_name(name: str) -> bool:
    return name.lower().count("orgs") < 2


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("file_name") or not row.get("test_type"):
                continue
            file_name = row["file_name"]
            if not _is_valid_dataset_name(file_name):
                continue
            try:
                elapsed = float(row.get("elapsed_seconds", "0") or "0")
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid elapsed_seconds in {path}: {row.get('elapsed_seconds')}"
                ) from exc
            rows.append(
                {
                    "file_name": file_name,
                    "test_type": row["test_type"],
                    "status": row.get("status", "unknown"),
                    "elapsed": elapsed,
                }
            )
    return rows


def _summaries(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    test_types = sorted({r["test_type"] for r in results})
    for test_type in test_types:
        case_results = [r for r in results if r["test_type"] == test_type]
        times = [
            r["elapsed"]
            for r in case_results
            if r["status"] == "success"
        ]
        total = len(case_results)
        summaries[test_type] = {
            "total": total,
            "success": sum(1 for r in case_results if r["status"] == "success"),
            "timeout": sum(1 for r in case_results if r["status"] == "timeout"),
            "error": sum(1 for r in case_results if r["status"] == "error"),
            "avg": statistics.mean(times) if times else None,
            "median": statistics.median(times) if times else None,
        }
    return summaries


def _compare_metric(
    *,
    metric: str,
    baseline: float | None,
    current: float | None,
    max_regression_pct: float,
) -> str | None:
    if baseline is None or current is None:
        return None
    if baseline <= 0:
        return None
    regression_pct = (current - baseline) / baseline * 100.0
    if regression_pct > max_regression_pct:
        return (
            f"{metric} regression {regression_pct:.1f}% "
            f"(baseline {baseline:.3f}s, current {current:.3f}s)"
        )
    return None


def _compare_rate(
    *,
    label: str,
    baseline: dict[str, Any],
    current: dict[str, Any],
    allow_increase: bool,
) -> str | None:
    if allow_increase:
        return None
    baseline_total = baseline["total"]
    current_total = current["total"]
    if baseline_total == 0 or current_total == 0:
        return None
    baseline_rate = baseline[label] / baseline_total
    current_rate = current[label] / current_total
    if current_rate > baseline_rate:
        return (
            f"{label} rate increased "
            f"(baseline {baseline_rate:.2%}, current {current_rate:.2%})"
        )
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two perf_test CSV reports and flag regressions.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline CSV report.")
    parser.add_argument("--current", required=True, help="Current CSV report.")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=20.0,
        help="Max allowed regression (percent) for avg/median runtime.",
    )
    parser.add_argument(
        "--allow-timeout-increase",
        action="store_true",
        help="Do not fail when timeout rate increases.",
    )
    parser.add_argument(
        "--allow-error-increase",
        action="store_true",
        help="Do not fail when error rate increases.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_results = _load_results(Path(args.baseline))
    current_results = _load_results(Path(args.current))

    baseline_summary = _summaries(baseline_results)
    current_summary = _summaries(current_results)

    issues: list[str] = []
    test_types = sorted(set(baseline_summary) | set(current_summary))
    if not test_types:
        print("No comparable results after filtering.", file=sys.stderr)
        return 1

    for test_type in test_types:
        base = baseline_summary.get(test_type)
        cur = current_summary.get(test_type)
        if base is None or cur is None:
            issues.append(f"{test_type}: missing data in baseline or current report")
            continue

        for metric in ("avg", "median"):
            issue = _compare_metric(
                metric=metric,
                baseline=base[metric],
                current=cur[metric],
                max_regression_pct=args.max_regression_pct,
            )
            if issue:
                issues.append(f"{test_type}: {issue}")

        timeout_issue = _compare_rate(
            label="timeout",
            baseline=base,
            current=cur,
            allow_increase=args.allow_timeout_increase,
        )
        if timeout_issue:
            issues.append(f"{test_type}: {timeout_issue}")

        error_issue = _compare_rate(
            label="error",
            baseline=base,
            current=cur,
            allow_increase=args.allow_error_increase,
        )
        if error_issue:
            issues.append(f"{test_type}: {error_issue}")

    if issues:
        print("Performance regressions detected:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print("No performance regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
