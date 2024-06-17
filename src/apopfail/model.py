"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def train():
    """Run the training step."""
    pass


def get_pipeline(clf=None) -> Pipeline:
    """Get prediction pipeline.

    Parameters
    ----------
    clf : object, optional
        Classifier to use in the pipeline.
        If None, the pipeline will not include a classifier.

    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline
        Pipeline to use in the prediction step.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("reducer", PCA(n_components=0.95)),
    ]

    if clf is not None:
        steps.append(("clf", clf))

    return Pipeline(steps=steps)
