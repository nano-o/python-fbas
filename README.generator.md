# FBAS generator command (random-sybil-attack-fbas)

This document describes the synthetic FBAS generator used for Sybil-attack
experiments. The command builds a directed org graph with thresholds and then
exports an FBAS from it.

## Quick start

Generate a random Sybil-attack FBAS and print the JSON:
```bash
python-fbas random-sybil-attack-fbas --orgs 10 --sybils 5 --print-fbas
```

Use the generator config example:
```bash
python-fbas random-sybil-attack-fbas --generator-config python-fbas.generator.cfg.example --print-fbas
```

Enable two Sybil clusters with a bridge restriction:
```bash
python-fbas random-sybil-attack-fbas \
  --num-sybil-clusters 2 \
  --orgs 10 \
  --sybils 5 \
  --sybils-cluster-2 5 \
  --sybil-bridge-orgs 2 \
  --print-fbas
```

Pick a minimal quorum instead of a random quorum (requires QBF support):
```bash
python-fbas random-sybil-attack-fbas --quorum-selection min --print-fbas
```

Plot the generated org graph:
```bash
python-fbas random-sybil-attack-fbas --plot
```

## Graph shape and roles

The generator builds a directed graph of orgs and then maps each org to a
validator with a threshold-based quorum set. Nodes get a `role` attribute:
- `honest` and `attacker` come from the original graph and its sampled quorum.
- `sybil` are nodes in Sybil cluster 1 (with `sybil_cluster=1`).
- `sybil` are nodes in Sybil cluster 2 (with `sybil_cluster=2`).
- `sybil_sybil_bridge` are the bridge nodes between Sybil clusters when
  `num_sybil_clusters=2`.

When `num_sybil_clusters=2`, the overall shape is a "double hourglass":
```
honest/attackers -> sybil cluster 1 -> sybil_sybil_bridge -> sybil cluster 2
```

Additional cross-edges can be enabled with `connect_*` flags, and each edge
family has its own probability parameter. See `python-fbas.generator.cfg.example`
for the full list.

## Configuration files

There are three relevant config files:
- `python-fbas.cfg`: global options (data source, solvers, etc).
- `python-fbas.generator.cfg`: generator parameters.
- `python-fbas.sybil-detection.cfg`: sybil detection parameters.

You can generate sample configs with:
```bash
python-fbas show-config > python-fbas.cfg
python-fbas show-generator-config > python-fbas.generator.cfg
python-fbas show-sybil-detection-config > python-fbas.sybil-detection.cfg
```

The generator command looks for `python-fbas.generator.cfg` and
`python-fbas.sybil-detection.cfg` in the current directory by default.
You can override them with `--generator-config` and `--sybil-detection-config`.

Use `--config-dir DIR` to point the command at a directory containing the
config files above (it will also load `python-fbas.cfg` if present).

## Reproducible runs

By default, the generator records the effective configuration of each run
under `runs/` as:
- `python-fbas.generator.cfg`
- `python-fbas.sybil-detection.cfg`
- `command.txt`

Use `--runs-dir` to change the output directory, and `--no-record-run` to
disable recording.

If you do not supply `--seed`, the command generates one, prints it to stderr,
and writes it into the recorded generator config so the run can be reproduced.

To reproduce a recorded run, point the generator at the run directory:
```bash
python-fbas random-sybil-attack-fbas --config-dir runs/20250101-120000
```

If `--config-dir` points inside `--runs-dir`, recording is skipped to avoid
creating duplicate run folders.

## Output and plotting

- `--print-fbas`: print the generated FBAS JSON to stdout.
- `--plot`: draw the org graph layout (roles, edges, thresholds).
- `--plot-with-trust`: shade nodes by trust scores computed with bounded
  capacity-limited propagation (`steps`, `capacity`).
- `--plot-with-trustrank`: shade nodes by TrustRank scores (personalized
  PageRank with `trustrank_alpha`, `trustrank_epsilon`).
- `--plot-with-maxflow`: shade nodes by max-flow scores with
  `maxflow_seed_capacity`. If `maxflow_sweep` is enabled, it prints bimodality
  coefficients and shows a small sweep plot. Set `maxflow_mode: equal-outflow`
  to enforce equal outflow per node (requires `python-fbas[lp]`).

Only one of `--plot-with-trust`, `--plot-with-trustrank`, or
`--plot-with-maxflow` can be enabled at a time.

### Plot details

- Trust seeds are chosen from honest nodes (`seed_count`), or all honest nodes
  if `seed_count` exceeds the honest set. If no honest nodes exist, seeds are
  drawn from all nodes.
- Node labels include trust scores when a trust plot is selected.
- Legend and edge styles:
  - Honest nodes (black outline), attackers (red outline), sybils (no outline),
    and sybil-bridge nodes (tan fill).
  - Original honest/attacker edges are solid gray.
  - Attacker -> sybil edges are solid red.
  - Sybil -> honest edges are dashed gray.
  - Sybil -> sybil edges are dotted gray.

## Key parameters

Shape controls:
- `orgs`: number of honest orgs in the original graph.
- `sybils`: size of Sybil cluster 1.
- `num_sybil_clusters`: 0 (no sybils), 1 (single cluster), or 2 (two clusters).
- `sybils_cluster_2`: size of Sybil cluster 2 (when `num_sybil_clusters=2`).
- `sybil_bridge_orgs`: number of `sybil_sybil_bridge` nodes (when `num_sybil_clusters=2`).

Quorum selection:
- `quorum_selection`: `random` (default) or `min` (requires QBF).

Edge probabilities and toggles:
- `original_edge_probability`: density within the honest graph.
- `sybil_sybil_edge_probability` and `sybil2_sybil2_edge_probability`: density
  within each Sybil cluster.
- `attacker_to_sybil_edge_probability`: edges from attackers to Sybil 1.
- `sybil_to_*`, `sybil2_to_*`, `sybil_bridge_to_*`: cross-group edges enabled
  by the corresponding `connect_*` flags.

See `python-fbas.generator.cfg.example` for the complete list of edge
probabilities and `connect_*` flags.

## Sybil detection settings

Sybil detection parameters are configured separately via
`python-fbas.sybil-detection.cfg` or `--sybil-detection-*` flags. The detection
step uses max-flow and trust ranking on the generated graph.

Use `python-fbas show-sybil-detection-config` to see all available settings.
