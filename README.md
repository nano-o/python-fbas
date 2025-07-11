# python-fbas

python-fbas is a tool to analyze Federated Byzantine Agreement Systems (FBAS), as used in the Stellar network, using automated constraint solvers like SAT solvers.
It can find disjoint quorums, minimal-cardinality splitting sets, minimal-cardinality blocking sets, and minimal-cardinality history-critical sets.

python-fbas seems to be able to handle much larger FBASs than [fbas_analyzer](https://github.com/trudi-group/fbas_analyzer) or the quorum-intersection checker of [stellar-core](https://github.com/stellar/stellar-core/).

## Technical highlights

We encode the problem of finding disjoint quorums as a SAT instance.
For minimal splitting/blocking sets, we encode the problem as a MaxSAT instances.
For minimal quorums, we encode the problem as a QBF instance.
We use [pysat](https://pysathq.github.io/) to solve the SAT/MaxSAT instances and [pyqbf](https://qbf.pages.sai.jku.at/pyqbf/) for the QBF instances.

Most SAT solvers expect input in conjunctive normal form (CNF), but it is easier to work with full propositional logic (using and, or, implies, not, etc. without restrictions).
pysat implements a transformation from propositional logic to CNF, but our benchmarks showed that it is way too slow for our purposes.
Thus we use our own CNF transformation, which is faster for our use case (see `to_cnf` in [`propositional_logic.py`](./python_fbas/propositional_logic.py)).

When reasoning about a FBAS, we often encounter cardinality constraints that assert that at least k out of n variables must be true.
To encode this efficiently in propositional logic, we use the totalizer encoding provided by pysat.
The paper [Efficient CNF Encoding of Boolean Cardinality Constraints](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a9481bf4ce2b5c20d2e282dd69dcb92bddcc36c9) describes the theory behind it, and it is a nice trick.

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

Run the main script and obtain the help message:
```
python-fbas
```
If you are using the docker image:
```
docker run --rm -it giulianolosa/python-fbas:latest python-fbas
```

To check whether the current Stellar network has quorum intersection:
```
python-fbas --fbas=pubnet check-intersection
```

To determine what the top-tier (i.e. the union of all minimal quorums) is:
```
python-fbas --fbas=pubnet top-tier
```
Note that this will not make sense if the FBAS does not have quorum intersection.

To determine the minimal number of nodes that, if corrupted, can split the network:
```
python-fbas --fbas=pubnet min-splitting-set
```
To determine the minimal number of nodes that, if corrupted, can halt the network:
```
python-fbas --fbas=pubnet min-blocking-set
```

For the `min-splitting-set` and `min-blocking-set` commands, you can group validators by attribute.
For example:
```
python-fbas --fbas=pubnet --group-by=homeDomain min-splitting-set
```
This computes the minimal number of home domains that must be corrupted in order to create disjoint quorums.

You might get surprising results due to a single validators having a weird configuration, and you might not care about this problematic validator.
In this case it helps to restrict the analysis to the validators that are reachable from some validator you care about.
For example, to restrict the FBAS to what is reachable from one of SDF's validators:
```
python-fbas --group-by=homeDomain --reachable-from GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH min-splitting-set
```

A history-critical set is a set of validators such that, together with the validators that currently have history-archive errors, form a quorum; in a worst-case scenario, if this quorum does not publish useable history archives, it would be possible to loose network history.
To find a minimal-cardinality history-critical set:
```
python-fbas --fbas=pubnet history-loss
```

Finally, you can also provide a FBAS to check in JSON format:
```
python-fbas --fbas=tests/test_data/random/almost_symmetric_network_13_orgs_delete_prob_factor_1.json check-intersection
```

By default, validators are displayed using both their ID and name. You can customize this using the `--validator-display` flag:

```
python-fbas --fbas=pubnet --validator-display=name min-splitting-set
```

Available options:
- `both` (default): Shows both validator ID and name
- `id`: Shows only the validator ID
- `name`: Shows only the validator name

Note that data about the Stellar network is cached in a local file the first time it is needed.
To update the cache (using the URL set in `config.py`):
```
python-fbas update-cache
```
