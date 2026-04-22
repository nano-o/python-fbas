# Development notes

## AI/dev container (bind mount)

This setup keeps your repo on the host (good for Emacs) and mounts it into a container that has Codex/Claude installed.

Build and start:
```bash
USER_UID=$(id -u) USER_GID=$(id -g) docker compose -f docker-compose.ai.yml up -d --build
```

Enter the container:
```bash
docker compose -f docker-compose.ai.yml exec ai bash
```

The first start installs `.[dev,qbf]` into a venv at `/home/developer/.venv`. You can override extras:
```bash
PYTHON_FBAS_EXTRAS=".[dev]" docker compose -f docker-compose.ai.yml up -d --build
```

Codex auth is shared from the host by bind-mounting `~/.codex` read-write. Log in once on the host, then restart the container:
```bash
codex login --device-auth
docker compose -f docker-compose.ai.yml up -d
```

To update dependencies, `rm ~/.python-fbas-container-setup` and re-run the container, or delete the venv at `~/.venv`.

## Network tests

Tests marked `@pytest.mark.network` are skipped by default. Enable them with:

```bash
python3 -m pytest --run-network
```

The live data source used by network tests is configurable. The primary URL
defaults to `https://radar.withobsrvr.com/api/v1/node` and the alternate URL
defaults to `https://api.stellaratlas.io/v1/node`.

Override them with pytest flags:

```bash
python3 -m pytest --run-network \
  --network-url=https://api.stellaratlas.io/v1/node \
  --network-alt-url=https://radar.withobsrvr.com/api/v1/node
```

Or use environment variables:

```bash
PYTEST_RUN_NETWORK=1 \
PYTEST_NETWORK_URL=https://api.stellaratlas.io/v1/node \
PYTEST_NETWORK_ALT_URL=https://radar.withobsrvr.com/api/v1/node \
python3 -m pytest
```

Some tests exercise the default configured URL versus an explicitly provided
alternate URL, so `PYTEST_NETWORK_ALT_URL` should differ from
`PYTEST_NETWORK_URL`.

## Performance tests

Performance regression sweeps are run via `scripts/perf_test.py`. By default it
executes three workloads (disjoint quorums, minimal splitting set, minimal
blocking set) over all JSON files in `tests/test_data/random/` that are valid
FBAS inputs, and writes timestamped `.md` and `.csv` reports to
`tests/perf/perf_results/`.
Datasets with filenames containing "orgs" twice are skipped.

Example:
```bash
python3 scripts/perf_test.py --timeout 30 --max-datasets 25 --dataset-order size
```

The default SAT solver is `cryptominisat5` when available; otherwise the first
available solver from pysat is used. The script exits non-zero if any run
fails or times out (use `--allow-failures` to always exit 0).

The test suite includes a smoke test (`tests/perf_script_test.py`) that runs the
script against fixed datasets  to ensure it keeps working.

Use `scripts/compare_perf.py` to compare two CSV reports and flag regressions:

```bash
python3 scripts/compare_perf.py \
  --baseline tests/perf/perf_results/perf_results_20260101_120000.csv \
  --current tests/perf/perf_results/perf_results_20260102_120000.csv \
  --max-regression-pct 15
```
