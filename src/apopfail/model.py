"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def train(model, X, y=None):
    """Fit the model on the dataset."""
    return model.fit(X, y)


def clean(X, y=None):
    """Remove rows with only nans and duplicate rows."""
    if y is not None:
        X = X.join(y)
    X = X.dropna(how="all")
    X = X.dropna(how="all", axis=1)
    X = X.drop_duplicates()
    if y is None:
        return X

    y = X["target"]
    X = X.drop(columns=["target"])
    return X, y


def get_pipeline(*, clf=None, scaler=None, reducer=None, sampler=None) -> Pipeline:
    """Get prediction pipeline.

    Parameters
    ----------
    clf : BaseEstimator, optional
        Classifier to use in the pipeline.
        If None, the pipeline will not include a classifier.
    scaler : scikit learn transformer, optional
        If None, use RobustScaler as default.
    reducer : scikit learn transformer, optional
        If None, use PCA with n_components=0.99
    sampler: imblearn sampler, optional
        If None, no sampling is done in the pipeline.

    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline
        Pipeline to use in the prediction step.
    """
    steps = [
        # ("cleaner", FunctionSampler(validate=False, func=clean)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", scaler or RobustScaler()),
        ("reducer", reducer or PCA(n_components=0.99)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))

    if clf is not None:
        steps.append(("clf", clf))

    return Pipeline(steps=steps)
