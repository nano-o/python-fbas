# AGENTS.md

Short guide for working in this repo.

## Project summary
- python-fbas analyzes Federated Byzantine Agreement Systems (FBAS) using SAT/MaxSAT/QBF solvers.
- CLI entry point: `python-fbas` (from `python_fbas.main:main`).

## Key paths
- `python_fbas/`: core library and CLI implementation.
- `tests/`: pytest-based tests and fixtures.
- `benchmark/`: benchmarking docs and results.
- `python-fbas.cfg.example`: sample config.

## Common commands
- Run tests: `python3 -m pytest`
- Type check: `mypy python_fbas/`
- Lint: `ruff check python_fbas/`
- Format: `autopep8 --in-place --recursive python_fbas/`

## Notes
- Optional QBF support comes from `pyqbf` (installed via `.[qbf]`).
- Default data source is the Stellar network URL in `config.py` and can be overridden via CLI/config file.

## Architecture map
- `python_fbas/main.py`: CLI entry point and command routing.
- `python_fbas/config.py`: configuration loading and defaults.
- `python_fbas/fbas_graph.py`: FBAS graph and quorum set data structures.
- `python_fbas/fbas_graph_analysis.py`: core analysis algorithms and encodings.
- `python_fbas/propositional_logic.py`: propositional/CNF helpers used by encodings.
- `python_fbas/solver.py`: solver backends and wrappers.
- `python_fbas/pubnet_data.py`: Stellar network data fetching and caching.
- `python_fbas/*serializer*.py` and `python_fbas/serialization.py`: JSON import/export utilities.
