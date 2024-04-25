"""Training step in the pipeline."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from dmgpred.cleaning import get_normalization_pipeline


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Run the training step."""
    normalizer = get_normalization_pipeline()
    model = Pipeline(
        [
            ("normalizer", normalizer),
            (
                "clf",
                DummyClassifier(strategy="most_frequent"),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model
