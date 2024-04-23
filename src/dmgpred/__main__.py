"""Main Script of the pipeline."""

import time

import numpy as np
import pandas as pd

from dmgpred.cleaning import clean
from dmgpred.evaluate import evaluate
from dmgpred.featurize import featurize
from dmgpred.train import train

DATA_PATH = "./data/"
TEST_VALUES_PATH = f"{DATA_PATH}/test_values.csv"
TRAIN_VALUES_PATH = f"{DATA_PATH}/train_values.csv"
TRAIN_LABELS_PATH = f"{DATA_PATH}/train_labels.csv"
SUBMISSION_PATH = "./output/Mandalorians_prediction.csv"


def main():
    """Run Prediction Pipeline."""
    # a simple timer, could use TQDM later on for progress bars
    np.random.seed(0)
    start = time.perf_counter()

    X_test = pd.read_csv(TEST_VALUES_PATH)
    X_train = pd.read_csv(TRAIN_VALUES_PATH)

    # need building id as index here,
    # otherwise it is interpreted as multi-output classification
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col="building_id")

    X_train, X_test = clean(X_train, X_test)
    X_train, X_test = featurize(X_train, X_test)

    model = train(X_train, y_train)
    y_pred = model.predict(X_test)

    # TODO: implement evaluation with custom metrics
    # e.g. with command-line arguments, or later with Hydra
    score = evaluate(model, X_train, y_train)
    print(f"Matthews Correlation Coefficient: {score: .4f}")

    submission = pd.DataFrame(
        {"building_id": X_test["building_id"], "damage_grade": y_pred}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    end = time.perf_counter()
    print(f"Finished in {end - start: .2f} seconds.")


if __name__ == "__main__":
    main()
