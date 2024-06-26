"""Main Script of the pipeline."""

import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn import set_config
from sklearn.linear_model import LogisticRegression

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
    "--log-level",
    default="info",
    type=click.Choice(
        ["debug", "info", "success", "warning", "error"], case_sensitive=False
    ),
)
def main(log_level):
    """Run Prediction Pipeline."""
    setup_logger(log_level)
    np.random.seed(0)
    start = time.perf_counter()
    set_config(transform_output="pandas")
    logger.info("Loading the data...")
    X_train, X_test, y_train = load_data(root=".")

    X_train, y_train = clean(X_train, y_train)
    X_test = clean(X_test)

    assert X_train.columns == X_test.columns

    clf = LogisticRegression()  # baseline classifier to start with
    model = get_pipeline(clf=clf, sampler=SMOTE(random_state=0))
    logger.info("Training the model on full dataset...")
    model = train(model, X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index=X_test.index)
    # y_pred = y_pred.map({-1: "active", 1: "inactive"})
    logger.info("Evaluating the model...")
    _ = evaluate(model, X_train, y_train, n_folds=5)
    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    submission = pd.DataFrame({TARGET: y_pred}).set_index(X_test.index)
    submission.to_csv(SUBMISSION_PATH)
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
            {"sink": f"{OUTPUT_PATH}/apopfail.log", "rotation": "1 day"},
        ],
    }
    logger.enable("apopfail")
    logger.configure(**logger_config)


if __name__ == "__main__":
    main()
