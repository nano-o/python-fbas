# python-fbas

A tool to analyze Federated Byzantine Agreement Systems (FBAS), as used in the Stellar network, using automated constraint solvers.

## What it does

python-fbas can find:
- **Disjoint quorums**: Two quorums that do not intersect
- **Minimal splitting set**: Smallest set of nodes needed to split the network
- **Minimal blocking set**: Smallest set of nodes needed to halt the network
- **Minimal history-critical set**: Smallest set of nodes that could cause history loss
- **Top-tier**: The union of all minimal quorums

## Related work

[fbas_analyzer](https://github.com/trudi-group/fbas_analyzer) is another FBAS analysis tool; as of July 2025, explorers like https://radar.withobsrvr.com/ and https://stellaratlas.io/ use it in their analysis backend.
In comparison, python-fbas aims for higher scalability.
See [benchmark/BENCHMARK_README.md](benchmark/BENCHMARK_README.md) and [sample results](benchmark/benchmark_results/benchmark_results_20251222_172738.md).
Another related project is [Stellar Observatory](https://github.com/andrenarchy/stellar-observatory).

## Technical approach

- **SAT encoding**: Finding disjoint quorums → SAT instance
- **MaxSAT encoding**: Finding minimal-size splitting/blocking sets → MaxSAT instance
- **QBF encoding**: Finding a minimal quorum → QBF instance
- **Solvers**: [pysat](https://pysathq.github.io/) for SAT/MaxSAT, [pyqbf](https://qbf.pages.sai.jku.at/pyqbf/) for QBF
- **Custom CNF transformation**: Faster than pysat's built-in transformation
- **Totalizer encoding**: Efficient cardinality constraints (see [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a9481bf4ce2b5c20d2e282dd69dcb92bddcc36c9))

## Installation

### Docker image

The tool is available as a pre-built image at https://hub.docker.com/r/giulianolosa/python-fbas/.

```bash
docker pull giulianolosa/python-fbas:latest
```

Run a command:
```bash
docker run --rm -it giulianolosa/python-fbas:latest --help
```

If you reference local files (for example, `--fbas=tests/...` or `--config-file=...`), mount the project directory:
```bash
docker run --rm -v "$PWD:/work" -w /work giulianolosa/python-fbas:latest <args>
```

### Local installation

Optionally create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install the package:
```
pip install .[qbf]
```
If this fails because pyqbf fails to build (which tends to happen), disable QBF support:
```
pip install .
```
In this case, computing minimal quorums and computing the top-tier (defined as the union of all minimal quorums) will not be available.

## Usage

### Basic commands

**Get help:**
```bash
python-fbas --help
```

**Show current configuration:**
```bash
python-fbas show-config
```
*Displays the current effective configuration, including default values. This shows what data source and solver settings will be used.*

### Analysis commands

*The following examples use the default Stellar network data source. You can specify a different URL with `--fbas=https://your-url.com/api` or use a local JSON file with `--fbas=path/to/file.json`.*

**Check quorum intersection:**
```bash
python-fbas check-intersection
```

**Find minimal splitting set:**
```bash
python-fbas min-splitting-set
```

**Find minimal blocking set:**
```bash
python-fbas min-blocking-set
```

**Find top-tier (union of minimal quorums):**
```bash
python-fbas top-tier
```
*Note: Only meaningful if the FBAS has quorum intersection*

**Sample a random quorum (requires UniGen):**
```bash
python-fbas random-quorum
```
*Requires `pyunigen` (install with `pip install .[unigen]`).*

**Generate a random Sybil-attack FBAS (or top-tier if sybils=0):**
```bash
python-fbas random-sybil-attack-fbas --orgs 5 --sybils 3
python-fbas random-sybil-attack-fbas --generator-config python-fbas.generator.cfg.example
```
*Use `python-fbas show-generator-config` to print generator defaults. Add `--plot` to visualize the generated org graph (not the full FBAS), or `--plot-with-trust` to shade nodes by trust from a random honest org. You can pass `--sybil-detection-config` or `--sybil-detection-steps`/`--sybil-detection-capacity` to tune the trust heuristic.*

**Find history-critical set:**
```bash
python-fbas history-loss
```
*Finds validators that could cause history loss if they stop publishing valid history archives.*

**Export FBAS to JSON:**
```bash
python-fbas to-json
```
*Converts the loaded FBAS to JSON format (see [Export to JSON](#export-to-json) for format options).*

**Show validator metadata:**
```bash
python-fbas validator-metadata GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH
```
*Prints the metadata fields associated with the validator (as JSON).*

### Options

**Group validators by attribute:**
```bash
python-fbas --group-by=homeDomain min-splitting-set
```
*Computes the minimal number of home domains (organizations) that must be corrupted to create disjoint quorums.*

**Restrict analysis to validators that are reachable from a given validator:**
```bash
python-fbas --reachable-from GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH min-splitting-set
```
*Useful to avoid surprising results from validators with unusual configurations*

TODO: allow specifying the validator by name

**Displaying validators:**
```bash
python-fbas --validator-display=name min-splitting-set
```
Options: `both` (default), `id`, `name`

**Force cache update:**
```bash
python-fbas --update-cache min-splitting-set
```
*Forces cache update before analysis.*

**Use configuration file:**
```bash
python-fbas --config-file=my-config.yaml min-splitting-set
```
*Loads settings from YAML configuration file. CLI options take precedence over config file.*

**Generate configuration file:**
```bash
python-fbas show-config > my-config.yaml
```
*Creates a configuration file with current settings that can be edited and reused*

### Configuration Files

python-fbas supports YAML configuration files to set default values for command-line options.

**Automatic detection:** Create a file named `python-fbas.cfg` in your current directory and it will be automatically loaded.

**Generate a configuration file from current settings:**
```bash
python-fbas show-config > ./python-fbas.cfg
```

**Explicit path:** Use `--config-file=path/to/config.yaml` to specify a custom config file path.

**Example configuration file:**
```yaml
# Data source
stellar_data_url: "https://radar.withobsrvr.com/api/v1/node"

# Solver settings
sat_solver: "minisat22"
card_encoding: "totalizer"
max_sat_algo: "RC2"

# Output settings
validator_display: "name"

# Optional settings
group_by: "homeDomain"
output: "problem.cnf"
```

**Precedence order:**
1. Command-line arguments (highest priority)
2. Configuration file values
3. Built-in defaults (lowest priority)

**View current configuration:**
Use `python-fbas show-config` to see the current effective configuration, including values from config files and defaults. The output is valid YAML that can be saved as a config file.

See `python-fbas.cfg.example` for a complete example with all available options.

### Data sources

**Use the default Stellar network:**
The default data source is `https://radar.withobsrvr.com/api/v1/node`. Use `python-fbas show-config` to see the current data source.

**Use a custom Stellar network URL:**
```bash
python-fbas --fbas=https://api.stellaratlas.io/v1/node check-intersection
```

**Use a local JSON file:**
```bash
python-fbas --fbas=tests/test_data/small/circular_1.json check-intersection
```
**Using Docker (with local file):**
```bash
docker run --rm -v "$PWD:/work" -w /work giulianolosa/python-fbas:latest --fbas=tests/test_data/small/circular_1.json check-intersection
```

**Update cache:**
```bash
# Update cache for default URL
python-fbas update-cache

# Update cache for specific URL
python-fbas --fbas=https://radar.withobsrvr.com/api/v1/node update-cache
```
*Updates cached Stellar network data. Useful when you want fresh data without waiting for automatic cache invalidation.*

### Data formats

python-fbas can read FBAS data in two formats:

**Stellarbeat format** (traditional): A JSON array of validator objects, each containing a `publicKey` and optional `quorumSet`. This is the format used by stellarbeat.io and similar network explorers.

**Python-fbas format** (efficient): A JSON object with separate `validators` and `qsets` sections. This format is more compact and efficient for large networks as it avoids duplicating identical quorum sets.

python-fbas automatically detects the input format when loading data.

### Export to JSON

**Convert loaded FBAS to JSON:**
```bash
# Export in python-fbas format (default)
python-fbas --fbas=tests/test_data/small/circular_1.json to-json

# Export in stellarbeat format
python-fbas --fbas=tests/test_data/small/circular_1.json to-json --format=stellarbeat

# Export current Stellar network in stellarbeat format
python-fbas to-json --format=stellarbeat
```
*Converts the loaded FBAS to JSON format and prints to stdout. Useful for format conversion or creating snapshots of network data.*
*Note: log messages are printed to stderr; use `--log-level=ERROR` or redirect stderr (e.g. `2>/dev/null`) to suppress them if you are piping to a file that should be valid JSON.*

## Development

For AI/dev container setup and contributor notes, see `DEVELOPMENT.md`.
