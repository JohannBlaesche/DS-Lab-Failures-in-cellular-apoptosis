"""Main Script of the pipeline."""

import sys
import time
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier

from apopfail.compare import compare_occ_models
from apopfail.evaluate import evaluate
from apopfail.model import clean, get_pipeline, train
from apopfail.utils.loading import load_data

DATA_PATH = "./data"
OUTPUT_PATH = "./output"
TEST_VALUES_PATH = f"{DATA_PATH}/test_data_p53_mutant.parquet"
TRAIN_VALUES_PATH = f"{DATA_PATH}/train_set_p53mutant.parquet"
TRAIN_LABELS_PATH = f"{DATA_PATH}/train_labels_p53mutant.csv"
SUBMISSION_PATH = f"{OUTPUT_PATH}/Mandalorians_prediction.csv"
TARGET = "5408"


@click.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["occ", "binary"], case_sensitive=False),
    default="occ",
    help="Choose the mode of the pipeline. 'occ' for one class classification, 'binary' for binary classification.",  # noqa: E501
)
@click.option(
    "--subsample",
    "-s",
    type=float,
    default=None,
    help="Subsample the data with the given ratio to decrease training time for testing purposes.",  # noqa: E501
)
@click.option(
    "--refit/--no-refit",
    default=False,
    help="Refit the model on the full dataset after training for best "
    "submission score. Turn it off to decrease pipeline runtime.",
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(
        ["debug", "info", "success", "warning", "error"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def main(log_level, mode, subsample, refit):
    """Run Prediction Pipeline."""
    setup_logger(log_level)
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="X has feature names"
    )
    np.random.seed(0)
    start = time.perf_counter()
    set_config(transform_output="pandas")
    logger.info("Loading the data...")
    X_train, X_test, y_train = load_data(root=".")

    X_train, y_train = clean(X_train, y_train, subsample=subsample)

    if mode == "occ":
        # only use models from pyod! not sklearn outlier detectors
        """'
        contamination = y_train.mean()
        clf = IForest(
            contamination=contamination, n_jobs=-1, random_state=0, behaviour="new"
        )
        model = get_pipeline(clf=clf)
        logger.info("Running the OCC pipeline...")

        model, _ = occ(model, X_train, y_train, refit=refit)
        """
        model = compare_occ_models(X_train, y_train, n_repeats=3, skip_existing=True)
        y_pred = model.predict(X_test)

    elif mode == "binary":
        clf = RandomForestClassifier()
        model = get_pipeline(clf=clf)
        logger.info("Training the model on full dataset...")
        model = train(model, X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info("Evaluating the model...")
        _ = evaluate(model, X_train, y_train, n_folds=5)

    else:
        raise ValueError("Invalid mode.")

    y_pred = pd.Series(y_pred, index=X_test.index, name=TARGET)
    y_pred = y_pred.map({0: "inactive", 1: "active"})

    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    y_pred.to_csv(SUBMISSION_PATH)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")
    end = time.perf_counter()
    logger.success(f"Finished in {end - start: .2f} seconds.")


def setup_logger(level: str):
    """Set up the logger."""
    logger_config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "{time:DD-MMM-YYYY HH:mm:ss} | {level: <8} | {message}",
                "level": level.upper(),
            },
            {"sink": f"{OUTPUT_PATH}/logs/apopfail.log", "rotation": "1 day"},
        ],
    }
    logger.enable("apopfail")
    logger.configure(**logger_config)


if __name__ == "__main__":
    main()
