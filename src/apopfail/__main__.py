"""Main Script of the pipeline."""

import pandas as pd

from apopfail.cleaning import clean
from apopfail.evaluate import evaluate
from apopfail.featurize import featurize
from apopfail.train import train


def main():
    """Run Prediction Pipeline."""
    print("Running Prediction Pipeline...")


if __name__ == "__main__":
    main()
