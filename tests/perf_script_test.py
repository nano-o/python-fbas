import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from python_fbas.solver import solvers as available_solvers


DATASETS = [
    "almost_symmetric_network_5_orgs_delete_prob_factor_1.json",
    "almost_symmetric_network_6_orgs_delete_prob_factor_2.json",
    "almost_symmetric_network_8_orgs_delete_prob_factor_2.json",
]


def _run_cmd(cmd, root_dir):
    return subprocess.run(
        cmd,
        cwd=root_dir,
        capture_output=True,
        text=True,
    )


def _assert_ok(result, label):
    if result.returncode != 0:
        raise AssertionError(
            "{label} failed\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                label=label,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )


def _load_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "file_name": row["file_name"],
                    "test_type": row["test_type"],
                    "status": row.get("status"),
                    "elapsed": float(row["elapsed_seconds"]),
                }
            )
    return rows


def _stage_datasets(dataset_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        source = dataset_dir / name
        if not source.exists():
            raise AssertionError(f"Missing dataset: {source}")
        dest = target_dir / name
        try:
            dest.symlink_to(source)
        except OSError:
            shutil.copy2(source, dest)


def test_perf_script_smoke(tmp_path):
    if not available_solvers:
        pytest.skip("No SAT solvers available in pysat.")

    root_dir = Path(__file__).resolve().parents[1]
    perf_script = root_dir / "scripts" / "perf_test.py"
    compare_script = root_dir / "scripts" / "compare_perf.py"
    dataset_dir = root_dir / "tests" / "test_data" / "random"
    output_dir = tmp_path / "perf_results"
    staged_dir = tmp_path / "datasets"
    _stage_datasets(dataset_dir, staged_dir)

    perf_cmd = [
        sys.executable,
        str(perf_script),
        "--dataset-dir",
        str(staged_dir),
        "--max-datasets",
        str(len(DATASETS)),
        "--timeout",
        "30",
        "--output-dir",
        str(output_dir),
        "--sat-solver",
        available_solvers[0],
    ]
    perf_result = _run_cmd(perf_cmd, root_dir)
    _assert_ok(perf_result, "perf_test.py")

    md_files = list(output_dir.glob("*.md"))
    csv_files = list(output_dir.glob("*.csv"))
    assert md_files, "Expected a markdown report from perf_test.py"
    assert csv_files, "Expected a CSV report from perf_test.py"

    csv_files.sort()
    latest_csv = csv_files[-1]
    results = _load_csv(latest_csv)
    dataset_names = {row["file_name"] for row in results}
    assert dataset_names == set(DATASETS)

    compare_cmd = [
        sys.executable,
        str(compare_script),
        "--baseline",
        str(latest_csv),
        "--current",
        str(latest_csv),
        "--max-regression-pct",
        "0",
    ]
    compare_result = _run_cmd(compare_cmd, root_dir)
    _assert_ok(compare_result, "compare_perf.py")
