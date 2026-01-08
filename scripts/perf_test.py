#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from python_fbas.serialization import detect_format  # noqa: E402
from python_fbas.solver import solvers as available_solvers  # noqa: E402

DEFAULT_DATASET_DIR = ROOT_DIR / "tests" / "test_data" / "random"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "tests" / "perf" / "perf_results"
DEFAULT_TIMEOUT_SECONDS = 30.0

TEST_CASES = [
    {"name": "disjoint-quorums", "command": "check-intersection"},
    {"name": "minimal-splitting-set", "command": "min-splitting-set"},
    {"name": "minimal-blocking-set", "command": "min-blocking-set"},
]


def _default_sat_solver() -> str:
    preferred = "cryptominisat5"
    if preferred in available_solvers:
        return preferred
    return available_solvers[0] if available_solvers else preferred


def _truncate(value: str, limit: int = 300) -> str:
    if len(value) <= limit:
        return value
    return value[:limit - 3] + "..."


def _sanitize_details(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _is_valid_dataset_name(name: str) -> bool:
    return name.lower().count("orgs") < 2


def _detect_dataset_format(
    path: Path,
    format_cache: dict[Path, tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    cached = format_cache.get(path)
    if cached is not None:
        return cached
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        result = (None, f"read-error: {exc}")
        format_cache[path] = result
        return result

    detected = detect_format(data)
    if detected in ("stellarbeat", "python-fbas"):
        result = (detected, None)
    else:
        result = (None, f"unknown-format: {detected}")
    format_cache[path] = result
    return result


def _run_with_timeout(cmd: list[str], timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    process: subprocess.Popen[str] | None = None

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT_DIR,
            preexec_fn=os.setsid,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            elapsed = time.perf_counter() - start
            return {
                "success": process.returncode == 0,
                "elapsed": elapsed,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
                "returncode": process.returncode,
            }
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.1)
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                process.kill()
                process.wait()
            except (OSError, ProcessLookupError):
                pass

            return {
                "success": False,
                "elapsed": timeout,
                "stdout": "",
                "stderr": "TIMEOUT",
                "timed_out": True,
                "returncode": None,
            }
    except Exception as exc:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        elapsed = time.perf_counter() - start
        return {
            "success": False,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "returncode": None,
        }


def _build_command(
    *,
    dataset_path: Path,
    subcommand: str,
    sat_solver: str,
    max_sat_algo: str,
    cardinality_encoding: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={dataset_path}",
        "--sat-solver",
        sat_solver,
        "--max-sat-algo",
        max_sat_algo,
        "--cardinality-encoding",
        cardinality_encoding,
        subcommand,
    ]


def _summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in TEST_CASES:
        name = case["name"]
        case_results = [r for r in results if r["test_type"] == name]
        times = [
            r["elapsed"]
            for r in case_results
            if r["status"] == "success"
        ]
        stats: dict[str, Any] = {
            "test_type": name,
            "success": sum(1 for r in case_results if r["status"] == "success"),
            "timeout": sum(1 for r in case_results if r["status"] == "timeout"),
            "error": sum(1 for r in case_results if r["status"] == "error"),
            "avg": statistics.mean(times) if times else None,
            "median": statistics.median(times) if times else None,
            "variance": statistics.pvariance(times) if times else None,
        }
        summary.append(stats)
    return summary


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = ["file_name", "test_type", "status", "elapsed_seconds", "details"]
    ordered = sorted(results, key=lambda r: (r["file_name"], r["test_type"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "file_name": row["file_name"],
                    "test_type": row["test_type"],
                    "status": row["status"],
                    "elapsed_seconds": f"{row['elapsed']:.3f}",
                    "details": row["details"],
                }
            )


def _write_markdown(
    path: Path,
    *,
    results: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    skipped: dict[str, str],
    timestamp: str,
    timeout: float,
    dataset_dir: Path,
    dataset_limit: int | None,
    max_datasets: int,
    sat_solver: str,
    max_sat_algo: str,
    cardinality_encoding: str,
) -> None:
    dataset_names = {r["file_name"] for r in results}
    lines = [
        "# Python-FBAS Performance Results",
        "",
        f"- Timestamp: {timestamp}",
        f"- Dataset directory: {dataset_dir}",
        f"- Timeout per run: {timeout:.1f}s",
        f"- Datasets executed: {len(dataset_names)}",
        f"- Dataset limit: {dataset_limit} of {max_datasets}"
        if dataset_limit is not None
        else f"- Dataset limit: none ({max_datasets} available)",
        f"- Total runs: {len(results)}",
        f"- sat_solver: {sat_solver}",
        f"- max_sat_algo: {max_sat_algo}",
        f"- cardinality-encoding: {cardinality_encoding}",
        "",
        "## Summary (success-only statistics; variance is population variance)",
        "",
        "| test_type | success | timeout | error | avg_s | median_s | variance_s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for summary in summaries:
        avg = f"{summary['avg']:.3f}" if summary["avg"] is not None else "n/a"
        median = (
            f"{summary['median']:.3f}" if summary["median"] is not None else "n/a"
        )
        variance = (
            f"{summary['variance']:.3f}"
            if summary["variance"] is not None
            else "n/a"
        )
        lines.append(
            "| {test_type} | {success} | {timeout} | {error} | {avg} | {median} |"
            " {variance} |".format(
                test_type=summary["test_type"],
                success=summary["success"],
                timeout=summary["timeout"],
                error=summary["error"],
                avg=avg,
                median=median,
                variance=variance,
            )
        )

    if skipped:
        lines += [
            "",
            "## Skipped datasets",
            "",
            "| file_name | reason |",
            "| --- | --- |",
        ]
        for name, reason in sorted(skipped.items()):
            lines.append(f"| {name} | {_truncate(reason)} |")

    lines += [
        "",
        "## Results",
        "",
        "| file_name | test_type | status | elapsed_s | details |",
        "| --- | --- | --- | --- | --- |",
    ]

    ordered = sorted(results, key=lambda r: (r["file_name"], r["test_type"]))
    for row in ordered:
        lines.append(
            "| {file_name} | {test_type} | {status} | {elapsed:.3f} | {details} |"
            .format(
                file_name=row["file_name"],
                test_type=row["test_type"],
                status=row["status"],
                elapsed=row["elapsed"],
                details=_truncate(row["details"]),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run python-fbas performance sweeps and emit reports.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing FBAS JSON datasets (default: random datasets).",
    )
    parser.add_argument(
        "--dataset-order",
        choices=("name", "size"),
        default="name",
        help="Order datasets by filename or by descending file size.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write markdown and CSV reports.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout per run in seconds.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit the number of datasets (alphabetical order).",
    )
    parser.add_argument(
        "--sat-solver",
        default=_default_sat_solver(),
        help="SAT solver to use.",
    )
    parser.add_argument(
        "--max-sat-algo",
        default="RC2",
        help="MaxSAT algorithm to use (LSU or RC2).",
    )
    parser.add_argument(
        "--cardinality-encoding",
        default="totalizer",
        help="Cardinality encoding to use (naive or totalizer).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if some runs fail or time out.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    timeout = args.timeout
    dataset_limit = args.max_datasets

    if args.sat_solver not in available_solvers:
        raise SystemExit(
            "Unsupported --sat-solver {!r}. Choose one of: {}.".format(
                args.sat_solver, available_solvers
            )
        )
    if args.max_sat_algo not in ("LSU", "RC2"):
        raise SystemExit(
            "Unsupported --max-sat-algo {!r}. Use LSU or RC2.".format(
                args.max_sat_algo
            )
        )
    if args.cardinality_encoding not in ("naive", "totalizer"):
        raise SystemExit(
            "Unsupported --cardinality-encoding {!r}. Use naive or totalizer."
            .format(args.cardinality_encoding)
        )
    if dataset_limit is not None and dataset_limit < 1:
        raise SystemExit("--max-datasets must be >= 1.")

    dataset_paths = [
        path for path in dataset_dir.glob("*.json")
        if _is_valid_dataset_name(path.name)
    ]
    if args.dataset_order == "size":
        dataset_paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    else:
        dataset_paths.sort()
    all_dataset_count = len(dataset_paths)
    if not dataset_paths:
        print(f"No datasets found in {dataset_dir}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    format_cache: dict[Path, tuple[str | None, str | None]] = {}

    processed_datasets = 0
    for dataset_path in dataset_paths:
        if dataset_limit is not None and processed_datasets >= dataset_limit:
            break
        detected, reason = _detect_dataset_format(dataset_path, format_cache)
        if detected is None:
            skipped[dataset_path.name] = reason or "unknown"
            continue

        for case in TEST_CASES:
            cmd = _build_command(
                dataset_path=dataset_path,
                subcommand=case["command"],
                sat_solver=args.sat_solver,
                max_sat_algo=args.max_sat_algo,
                cardinality_encoding=args.cardinality_encoding,
            )
            result = _run_with_timeout(cmd, timeout=timeout)

            if result["timed_out"]:
                status = "timeout"
                details = "TIMEOUT"
            elif result["success"]:
                status = "success"
                details = ""
            else:
                status = "error"
                raw_details = (
                    result["stderr"]
                    or result["stdout"]
                    or f"exit code {result['returncode']}"
                )
                details = _truncate(_sanitize_details(raw_details))

            results.append(
                {
                    "file_name": dataset_path.name,
                    "test_type": case["name"],
                    "status": status,
                    "elapsed": result["elapsed"],
                    "details": details,
                }
            )
            print(
                "PERF {test_type} {file_name}: {status} ({elapsed:.3f}s)".format(
                    test_type=case["name"],
                    file_name=dataset_path.name,
                    status=status,
                    elapsed=result["elapsed"],
                )
            )
        processed_datasets += 1

    if not results and not skipped:
        print("No results were produced.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"perf_results_{timestamp}.md"
    csv_path = output_dir / f"perf_results_{timestamp}.csv"
    summaries = _summaries(results)
    _write_csv(csv_path, results)
    _write_markdown(
        md_path,
        results=results,
        summaries=summaries,
        skipped=skipped,
        timestamp=timestamp,
        timeout=timeout,
        dataset_dir=dataset_dir,
        dataset_limit=dataset_limit,
        max_datasets=all_dataset_count,
        sat_solver=args.sat_solver,
        max_sat_algo=args.max_sat_algo,
        cardinality_encoding=args.cardinality_encoding,
    )
    print(f"Perf results written to {md_path} and {csv_path}")

    had_failures = any(r["status"] != "success" for r in results)
    if had_failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
