"""Main Script of the pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from apopfail.evaluate import evaluate
from apopfail.model import get_pipeline

DATA_PATH = "./data"
OUTPUT_PATH = "./output"
TEST_VALUES_PATH = f"{DATA_PATH}/test_set_p53mutant.parquet"
TRAIN_VALUES_PATH = f"{DATA_PATH}/train_set_p53mutant.parquet"
TRAIN_LABELS_PATH = f"{DATA_PATH}/train_labels_p53mutant.csv"
SUBMISSION_PATH = f"{OUTPUT_PATH}/Mandalorians_prediction.csv"
TARGET = "5408"


def main():
    """Run Prediction Pipeline."""
    X_test = pd.read_parquet(TEST_VALUES_PATH)
    X_train = pd.read_parquet(TRAIN_VALUES_PATH)
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col=0, names=["target"], skiprows=1)[
        "target"
    ]
    y_train = y_train.map({"inactive": 0, "active": 1}).astype(np.int8)
    clf = IsolationForest()  # baseline classifier to start with
    model = get_pipeline(clf=clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_results = evaluate(model, X_train, y_train)
    print(eval_results)
    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    submission = pd.DataFrame({X_test.index, y_pred.reshape(-1)})
    submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
