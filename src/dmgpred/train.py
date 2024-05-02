"""Training step in the pipeline."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from dmgpred.cleaning import get_normalization_pipeline
from dmgpred.featurize import get_encoder


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Run the training step."""
    normalizer = get_normalization_pipeline()
    encoder = get_encoder(X_train)
    model = Pipeline(
        [
            ("normalizer", normalizer),
            ("encode", encoder),
            (
                "clf",
                RandomForestClassifier(max_depth=3, n_estimators=50),
            ),
        ],
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model
