"""Main Script of the pipeline."""

import pandas as pd

from dmgpred.cleaning import clean
from dmgpred.evaluate import evaluate
from dmgpred.featurize import featurize
from dmgpred.train import train


def main():
    """Run Prediction Pipeline."""
    test_values = pd.read_csv("./data/test_values.csv")
    train_values = pd.read_csv("./data/train_values.csv")
    train_labels = pd.read_csv("./data/train_labels.csv")
    sample_submission = pd.read_csv("./data/submission_format.csv")

    X_train, y_train, X_test = clean(train_values, train_labels, test_values)
    X_train, X_test = featurize(X_train, X_test)
    model = train(X_train, y_train)

    score, y_pred = evaluate(model, X_train, y_train, X_test)
    print(f"Score: {score}")

    submission = sample_submission.copy()
    submission["damage_grade"] = y_pred
    submission.to_csv("../output/Mandalorians_prediction.csv", index=False)


if __name__ == "__main__":
    main()
