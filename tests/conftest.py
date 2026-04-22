import os

import pytest

DEFAULT_NETWORK_URL = "https://radar.withobsrvr.com/api/v1/node"
DEFAULT_NETWORK_ALT_URL = "https://api.stellaratlas.io/v1/node"


def _run_network_tests(config: pytest.Config) -> bool:
    return config.getoption("--run-network") or os.environ.get("PYTEST_RUN_NETWORK") == "1"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require live network access",
    )
    parser.addoption(
        "--network-url",
        default=os.environ.get("PYTEST_NETWORK_URL", DEFAULT_NETWORK_URL),
        help="live network URL to use in network-marked tests",
    )
    parser.addoption(
        "--network-alt-url",
        default=os.environ.get("PYTEST_NETWORK_ALT_URL", DEFAULT_NETWORK_ALT_URL),
        help="alternate live network URL to use in network-marked tests",
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


@pytest.fixture
def network_url(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--network-url"))


@pytest.fixture
def network_alt_url(pytestconfig: pytest.Config, network_url: str) -> str:
    alt_url = str(pytestconfig.getoption("--network-alt-url"))
    if alt_url == network_url:
        pytest.skip("--network-alt-url must differ from --network-url for this test")
    return alt_url


@pytest.fixture
def network_config_file(tmp_path: pytest.TempPathFactory, network_url: str) -> str:
    config_path = tmp_path / "python-fbas-network.cfg"
    config_path.write_text(f'stellar_data_url: "{network_url}"\n', encoding="utf-8")
    return str(config_path)
