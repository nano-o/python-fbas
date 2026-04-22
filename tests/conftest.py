import os

import pytest


def _run_network_tests(config: pytest.Config) -> bool:
    return config.getoption("--run-network") or os.environ.get("PYTEST_RUN_NETWORK") == "1"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require live network access",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _run_network_tests(config):
        return

    skip_network = pytest.mark.skip(
        reason="requires live network access; use --run-network or set PYTEST_RUN_NETWORK=1"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
