# Task 1 - Earthquake Damage Prediction

## Quickstart

After cloning the repository, place the datafiles inside the `data` directory.

To run the the pipeline with environment setup and installation, use the following command if you are on Linux/MacOS:

```shell
make
```

which defaults to `make all`.

Alternatively, run the pipeline manually with `dmgpred` after having installed the required packages, e.g. with `pip install .`

## Pipeline CLI arguments

Additional metrics can be specified using the `--add-metrics` argument. It must be a list of strings, where each string is the name of a metric to be calculated. The available metrics are documented on sklearns documentation [page](https://scikit-learn.org/stable/modules/model_evaluation.html). For example:

```shell
dmgpred --add-metrics="f1_macro,roc_auc_ovr"
```

or just `dmgpred` to use the defaults, which are MCC and F1 Micro averaged.

See the help for the pipeline `dmgpred --help` for more information on the arguments.

## Development Guide

Use the makefile (`make`) or the following commands to setup your dev environment:

Setup a virtual environment and install the required packages using the following commands:
Posix:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

or on Windows:

```shell
python -m venv .venv
./.venv/Scripts/activate
```

Then install the package with

```shell
pip install -e '.[dev]'
```

and the precommit hooks with

```shell
pre-commit install
```

and optionally run on all files:

```shell
pre-commit run --all-files
```
