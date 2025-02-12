# Constellation

In this package we provide functionality related to Constellation overlays.

Constellation is described in an upcoming FC 2025 paper.

## Installation

Either use the docker image or install locally.

### Docker image

The functionality is available as a pre-built docker image available at https://hub.docker.com/r/giulianolosa/constellation/.

Use `docker pull giulianolosa/constellation` to pull the latest version.

### Local installation

This is an alternative to using the docker image.

Start in the top-level directory (i.e., the grand-parent of the directory this file belongs to).
Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```
Next, run `pip install .` (or `pip install -e .` to keep updated during development).
Finally, go to `python_fbas/constellation/brute-force-search/` and run `make install` (note this calls the gcc compiler).

## Jupyter Notebooks

There are several notebooks in `python_fbas/constellation/notebooks/` which allow reproducing some of the results in the paper.

You can use the docker image to access them from a browser.
Run `docker run --rm -p 8888:8888 giulianolosa/constellation:latest start_jupyter`
Then look for the URL to access in the output of this command.

## Commands

There are two main commands available: `python-fbas` and `constellation`, which each prints its own help.
`python-fbas` is additionally documented at the root of the project.

To compute the Constellation clusters when there are 200 organizations where 100 have threshold 130 and 100 have threshold 140, try
```
constellation compute-clusters --thresholds 100 130 100 140 --max-num-clusters=5 --min-cluster-size=33
```

As the number of different thresholds increases, the search space increases a log.
To keep runtime low, we need to reduce the maximal number of clusters allowed and increase the minimal cluster size.
For example:
```
constellation compute-clusters --thresholds 50 130 50 135 50 140 50 145 --max-num-clusters=3 --min-cluster-size=60
```

To compute the Constellation overlay graph give a single-universe, regular FBAS in JSON format:
```
constellation compute-overlay --fbas data/11_orgs_single_univ.json --output overlay.json
```

To run a command using the docker image, run e.g.
```
docker run --rm -it giulianolosa/constellation:latest constellation compute-overlay --fbas python_fbas/constellation/data/11_orgs_single_univ.json --output=/dev/stdout
```
