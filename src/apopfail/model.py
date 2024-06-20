"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def train(model, X, y=None):
    """Fit the model on the dataset."""
    return model.fit(X, y)


def clean(X):
    """Remove rows with only nans and duplicate rows."""
    X = X.dropna(how="all")
    X = X.drop_duplicates()
    return X


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
        ("cleaner", FunctionTransformer(func=clean)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("reducer", PCA(n_components=0.95)),
        ("sampler", RandomUnderSampler()),
    ]

    if clf is not None:
        steps.append(("clf", clf))

    return Pipeline(steps=steps)
