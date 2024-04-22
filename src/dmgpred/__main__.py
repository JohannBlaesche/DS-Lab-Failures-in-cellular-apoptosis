"""Main Script of the pipeline."""

import time

import pandas as pd

from dmgpred.cleaning import clean
from dmgpred.evaluate import evaluate
from dmgpred.featurize import featurize
from dmgpred.train import train

TEST_VALUES_PATH = "./data/test_values.csv"
TRAIN_VALUES_PATH = "./data/train_values.csv"
TRAIN_LABELS_PATH = "./data/train_labels.csv"
SUBMISSION_PATH = "./output/Mandalorians_prediction.csv"


def main():
    """Run Prediction Pipeline."""
    start = time.perf_counter()
    X_test = pd.read_csv(TEST_VALUES_PATH)
    X_train = pd.read_csv(TRAIN_VALUES_PATH)
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col="building_id")

    X_train, X_test = clean(X_train, X_test)
    X_train, X_test = featurize(X_train, X_test)
    model = train(X_train, y_train)

    # TODO: implement evaluation with custom metrics
    score = evaluate(model, X_train, y_train)
    print(f"Matthews Correlation Coefficient: {score: .4f}")

    y_pred = model.predict(X_test)
    submission = pd.DataFrame(
        {"building_id": X_test["building_id"], "damage_grade": y_pred}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    end = time.perf_counter()
    print(f"Finished in {end - start: .2f} seconds.")


if __name__ == "__main__":
    main()
