"""Main Script of the pipeline."""

from pathlib import Path

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
INDEX_COL = "Unnamed: 0"
TARGET = "5408"


def main():
    """Run Prediction Pipeline."""
    X_test = pd.read_parquet(TEST_VALUES_PATH).set_index(INDEX_COL)
    X_train = pd.read_parquet(TRAIN_VALUES_PATH).set_index(INDEX_COL)
    y_train = pd.read_csv(TRAIN_LABELS_PATH)
    clf = IsolationForest()  # baseline classifier to start with
    model = get_pipeline(clf=clf)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    eval_results = evaluate(model, X_train, y_train)
    print(eval_results)
    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    submission = pd.DataFrame({INDEX_COL: X_test.index, TARGET: y_pred.reshape(-1)})
    submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
