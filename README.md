# Task 2 - Apoptosis Failure Detection

## Quickstart

After cloning the repository, place the datafiles inside the `data` directory.

The training data must be named `train_set_p53mutant.parquet` with labels `train_labels_p53mutant.csv`.
The test data must be named `test_data_p53_mutant.parquet`

To run the the pipeline with environment setup and installation, use the following command if you are on Linux/MacOS:

```shell
make
```

which defaults to `make all`.

Alternatively, run the pipeline manually with `afp` after having installed the required packages, e.g. with `pip install .`

## Pipeline CLI arguments

To see all available arguments, run:
```shell
apopfail --help
```

When running the pipeline, the following arguments are available:
- `--mode`: The mode of the pipeline. Either `binary` or `occ`. Default is `occ`.
- `--refit`: Refit the model on the full training data before predicting the test data. Default is (--no-refit) False to save time.
  Ignored when `--mode binary` is used. When the pipeline is run through `make run`, it is activated by default.
- `subsample`: The fraction of the training data to use. Default is None, i.e. use all training data. Useful for testing.


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
