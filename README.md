# python-fbas

A tool to analyze Federated Byzantine Agreement Systems (FBAS), as used in the Stellar network, using automated constraint solvers.

## What it does

python-fbas can find:
- **Find disjoint quorums**: Two quorums that do not intersect
- **Find a minimal splitting set**: A minimal-cardinality set of nodes needed to split the network
- **Find a minimal blocking set**: A minimal-cardinality set of nodes needed to halt the network
- **Find a minimal history-critical sets**: A minimal-cardinality set of nodes that could cause history loss
- **Compute the top-tier**: The union of all minimal quorums

python-fbas handles much larger FBASs than [fbas_analyzer](https://github.com/trudi-group/fbas_analyzer) or the quorum-intersection checker of [stellar-core](https://github.com/stellar/stellar-core/).

## Technical approach

- **SAT encoding**: Finding disjoint quorums → SAT instance
- **MaxSAT encoding**: Finding minimal-size splitting/blocking sets → MaxSAT instance
- **QBF encoding**: Finding a minimal quorum → QBF instance (we could do this we MaxSAT too)
- **Solvers**: [pysat](https://pysathq.github.io/) for SAT/MaxSAT, [pyqbf](https://qbf.pages.sai.jku.at/pyqbf/) for QBF
- **Custom CNF transformation**: Faster than pysat's built-in transformation
- **Totalizer encoding**: Efficient cardinality constraints (see [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a9481bf4ce2b5c20d2e282dd69dcb92bddcc36c9))

## Installation

Either use the docker image or install locally.

### Docker image

The functionality is available as a pre-built docker image available at https://hub.docker.com/r/giulianolosa/python-fbas/.

Use `docker pull giulianolosa/python-fbas` to pull the latest version.

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

Run the tests:
```
pip install pytest
python3 -m pytest
```

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

**Using Docker:**
```bash
docker run --rm -it giulianolosa/python-fbas:latest python-fbas show-config
```

### Analysis commands

*The following examples use the default Stellar network data source. Each command shows what data source is being used and whether it's using cached data. You can specify a different URL with `--fbas=https://your-url.com/api` or use a local JSON file with `--fbas=path/to/file.json`.*

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

**Find history-critical set:**
```bash
python-fbas history-loss
```
*Finds validators that could cause history loss if they stop publishing valid history archives*

### Options

**Group validators by attribute:**
```bash
python-fbas --group-by=homeDomain min-splitting-set
```
*Computes minimal number of home domains (i.e. organizations) that must be corrupted to create disjoint quorums*

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
*Forces cache update before analysis*

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

python-fbas supports YAML configuration files to set default values for command-line options. This is useful for avoiding repetitive CLI arguments.

**Automatic detection:**
Create a file named `python-fbas.cfg` in your current directory and it will be automatically loaded.

**Generate a configuration file from current settings:**
```bash
python-fbas show-config > ./python-fbas.cfg
```

**Explicit path:**
Use `--config-file=path/to/config.yaml` to specify a custom config file path.

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
python-fbas --fbas=tests/test_data/circular_1.json check-intersection
```

**Update cache:**
```bash
# Update cache for default URL
python-fbas update-cache

# Update cache for specific URL
python-fbas --fbas=https://api.stellaratlas.io/v1/node update-cache
```
*Updates cached Stellar network data. Useful when you want fresh data without waiting for automatic cache invalidation.*
