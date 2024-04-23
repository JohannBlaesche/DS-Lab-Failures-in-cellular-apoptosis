"""Training step in the pipeline."""

import pandas as pd
from sklearn.dummy import DummyClassifier


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Run the training step."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model
