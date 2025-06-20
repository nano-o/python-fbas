from pathlib import Path
import logging
import sys
import pytest
import importlib
import python_fbas
from python_fbas import main, config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_PKG = PROJECT_ROOT / "python_fbas"
if LOCAL_PKG not in Path(python_fbas.__file__).resolve().parents:
    python_fbas = importlib.reload(importlib.import_module("python_fbas"))


DATA_DIR = Path(__file__).parent / "test_data"


def run_cmd(argv):
    args = main.parse_args(argv)
    main.configure(args)
    return args.func(args)


def test_parse_update_cache():
    args = main.parse_args(["update-cache"])
    assert args.command == "update-cache"
    assert args.func is main.cmd_update_cache


def test_configure_sets_globals():
    args = main.parse_args([
        "--cardinality-encoding",
        "naive",
        "--sat-solver",
        "minisat22",
        "--max-sat-algo",
        "RC2",
        "--log-level",
        "INFO",
        "update-cache",
    ])
    main.configure(args)
    assert config.card_encoding == "naive"
    assert config.sat_solver == "minisat22"
    assert config.max_sat_algo == "RC2"
    assert logging.getLogger().level == logging.INFO


def test_prepare_fbas_group_by_validation():
    args = main.parse_args([
        "--fbas",
        str(DATA_DIR / "circular_1.json"),
        "--group-by",
        "homeDomain",
        "min-blocking-set",
    ])
    main.configure(args)
    with pytest.raises(ValueError):
        main.cmd_min_blocking_set(args)


def test_cmd_check_intersection_fast(capsys):
    ret = run_cmd([
        "--fbas",
        str(DATA_DIR / "circular_2.json"),
        "check-intersection",
        "--fast",
    ])
    captured = capsys.readouterr().out
    assert "Intersection-check result" in captured
    assert ret == 0


def test_cmd_min_blocking_set(capsys):
    ret = run_cmd([
        "--fbas",
        str(DATA_DIR / "circular_2.json"),
        "min-blocking-set",
    ])
    captured = capsys.readouterr().out
    assert "No blocking set found" in captured
    assert ret == 0


def test_cmd_history_loss(capsys):
    ret = run_cmd([
        "--fbas",
        str(DATA_DIR / "circular_1.json"),
        "history-loss",
    ])
    captured = capsys.readouterr().out
    assert "Minimal history-loss critical set cardinality" in captured
    assert ret == 0


@pytest.mark.xfail
def test_cmd_top_tier_without_qbf():
    run_cmd([
        "--fbas",
        str(DATA_DIR / "circular_2.json"),
        "top-tier",
    ])

