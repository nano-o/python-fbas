# python-fbas

A tool to analyze Federated Byzantine Agreement Systems (FBAS), as used in the Stellar network, using automated constraint solvers.

## What it does

python-fbas can find:
- **Find disjoint quorums**: Two quorums that do not intersect
- **Find a minimal splitting set**: Minimal-cardinality set of nodes needed to split the network
- **Find a minimal blocking set**: Minimal-cardinality set of nodes needed to halt the network
- **Find a minimal history-critical sets**: Minimal-cardinality set of nodes that could cause history loss
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

Get help:
```bash
python-fbas
```

Using Docker:
```bash
docker run --rm -it giulianolosa/python-fbas:latest python-fbas
```

### Analysis commands

**Check quorum intersection:**
```bash
python-fbas --fbas=pubnet check-intersection
```

**Find minimal splitting set:**
```bash
python-fbas --fbas=pubnet min-splitting-set
```

**Find minimal blocking set:**
```bash
python-fbas --fbas=pubnet min-blocking-set
```

**Find top-tier (union of minimal quorums):**
```bash
python-fbas --fbas=pubnet top-tier
```
*Note: Only meaningful if the FBAS has quorum intersection*

**Find history-critical set:**
```bash
python-fbas --fbas=pubnet history-loss
```
*Finds validators that could cause history loss if they stop publishing valid history archives*

### Advanced options

**Group validators by attribute:**
```bash
python-fbas --fbas=pubnet --group-by=homeDomain min-splitting-set
```
*Computes minimal number of home domains (i.e. organizations) that must be corrupted to create disjoint quorums*

**Restrict analysis to validators that are reachable  from a given validator in the FBAS graph:**
```bash
python-fbas --reachable-from GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH min-splitting-set
```
*Useful to avoid surprising results from validators with unusual configurations*

TODO: allow specifying the validator by name

**Customize validator display:**
```bash
python-fbas --fbas=pubnet --validator-display=name min-splitting-set
```
Options: `both` (default), `id`, `name`

### Data sources

**Use custom JSON file:**
```bash
python-fbas --fbas=tests/test_data/random/almost_symmetric_network_13_orgs_delete_prob_factor_1.json check-intersection
```

**Update Stellar network cache:**
```bash
python-fbas update-cache
```
*Updates cached data from the URL configured in `config.py`*
