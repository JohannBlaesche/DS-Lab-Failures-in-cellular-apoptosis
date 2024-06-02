"""Main Script of the pipeline."""

import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import dump
from loguru import logger
from sklearn import set_config

from dmgpred.cleaning import clean
from dmgpred.evaluate import evaluate
from dmgpred.featurize import featurize
from dmgpred.train import train
from dmgpred.tune_lgbm import run_optimization

DATA_PATH = "./data"
OUTPUT_PATH = "./output"
TEST_VALUES_PATH = f"{DATA_PATH}/test_values.csv"
TRAIN_VALUES_PATH = f"{DATA_PATH}/train_values.csv"
TRAIN_LABELS_PATH = f"{DATA_PATH}/train_labels.csv"
SUBMISSION_PATH = f"{OUTPUT_PATH}/Mandalorians_prediction.csv"
INDEX_COL = "building_id"
TARGET = "damage_grade"


@click.command()
@click.option(
    "--add-metrics",
    default=None,
    help="Additional scoring metrics to report in evaluation.",
)
@click.option(
    "--n-folds",
    default=5,
    help="Number of folds for cross-validation.",
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(
        ["debug", "info", "success", "warning", "error"], case_sensitive=False
    ),
)
@click.option("--no-gpu", default=False, is_flag=True, help="Use CPU for training.")
@click.option(
    "--tune",
    default=False,
    is_flag=True,
    help="Run hyperparameter tuning of pipeline models with Optuna.",
)
@click.option(
    "--n-trials",
    default=100,
    help="Number of trials for hyperparameter optimization for each model.",
)
def main(add_metrics, n_folds, log_level, no_gpu, tune, n_trials):
    """Run the Damage Prediction Pipeline.

    This pipeline consists of four steps, namely cleaning, featurization,
    training and evaluation.

    The cleaned and featurized data is used to train a model on the full dataset and
    to evaluate the model using cross-validation with StratifiedKFold splits.
    The model is then used to predict the
    damage grade of the test data and the results are saved to a CSV file.
    """
    # a simple timer, could use TQDM later on for progress bars
    setup_logger(log_level)
    np.random.seed(0)
    start = time.perf_counter()

    # keep pandas output in transform
    set_config(transform_output="pandas")
    logger.info("Preparing the data...")
    X_test = pd.read_csv(TEST_VALUES_PATH, index_col=INDEX_COL)
    X_train = pd.read_csv(TRAIN_VALUES_PATH, index_col=INDEX_COL)

    # need building id as index here,
    # otherwise it is interpreted as multi-output classification
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col=INDEX_COL)
    y_train = y_train[TARGET] - 1

    X_train, X_test = clean(X_train, X_test)
    X_train, X_test = featurize(X_train, X_test)

    use_gpu = not no_gpu
    if tune:
        run_optimization(
            X_train, y_train, n_trials=n_trials, use_gpu=use_gpu, study_name="catboost"
        )

        # run_optimization(
        #     X_train, y_train, n_trials=n_trials, use_gpu=use_gpu, study_name="xgb"
        # )

        # run_optimization(
        #     X_train, y_train, n_trials=n_trials, use_gpu=use_gpu, study_name="lgbm"
        # )

    logger.info("Training the model on full dataset...")
    model = train(X_train, y_train, use_gpu=use_gpu)
    dump(model, f"{OUTPUT_PATH}/trained_model.pkl")
    logger.info(f"Model saved to {OUTPUT_PATH}/trained_model.pkl")
    y_pred = model.predict(X_test) + 1

    if add_metrics is not None:
        add_metrics = {metric: metric for metric in add_metrics.split(",")}

    logger.info(f"Evaluating the model with {n_folds}-fold Cross-Validation...")
    _ = evaluate(
        model,
        X_train,
        y_train,
        n_folds=n_folds,
        additional_scoring=add_metrics,
    )

    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    submission = pd.DataFrame({INDEX_COL: X_test.index, TARGET: y_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
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
            {"sink": f"{OUTPUT_PATH}/dmgpred.log", "rotation": "1 day"},
        ],
    }
    logger.enable("dmgpred")
    logger.configure(**logger_config)


if __name__ == "__main__":
    main()
