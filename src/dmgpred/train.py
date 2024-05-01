"""Training step in the pipeline."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
                RandomForestClassifier(random_state=42, max_depth=3, n_estimators=50),
            ),
        ],
        verbose=True,
    )
    print(X_train.info())
    model.fit(X_train, y_train.to_numpy().ravel())
    return model
