"""Training step in the pipeline."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skorch.helper import DataFrameTransformer
from xgboost import XGBClassifier

from apopfail.neuralnet import build_nn


def train(model, X, y=None):
    """Fit the model on the dataset."""
    return model.fit(X, y)


def clean(X, y, subsample: float | None = None):
    """Remove rows with only nans and duplicate rows."""
    X = X.join(y)
    X = X.dropna(how="all")
    X = X.dropna(how="all", axis=1)
    X = X.drop_duplicates()

    if subsample is not None:
        logger.warning(f"Subsampling the data with ratio {subsample}")
        X = X.sample(frac=subsample, random_state=0)

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
        If None, use StandardScaler as default.
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
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", scaler or StandardScaler()),
        ("reducer", reducer or PCA(n_components=0.99)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))

    if clf is not None:
        if clf.__class__.__name__.endswith("NeuralNetBinaryClassifier"):
            steps.append(("flatten", DataFrameTransformer()))
        steps.append(("clf", clf))

    return Pipeline(steps=steps)


def build_model():
    """Build the model for the pipeline."""
    neural_clf = build_nn(input_size=3000, pos_weight=1.3)
    nn_pipe = get_pipeline(
        clf=neural_clf,
        reducer=PCA(n_components=3000),
        # reducer="passthrough",
        sampler=SMOTE(random_state=0, sampling_strategy=0.2),
    )
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        reg_lambda=10,
        reg_alpha=2,
        random_state=0,
        subsample=1,
        colsample_bytree=0.8,
        min_child_weight=0.5,
        n_jobs=-1,
    )
    xgb_pipe = get_pipeline(  # noqa: F841
        clf=xgb, sampler=SMOTE(random_state=0, sampling_strategy=0.5)
    )
    logistic = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="saga",
        max_iter=1000,
        random_state=0,
        n_jobs=-1,
    )
    logistic_pipe = get_pipeline(  # noqa: F841
        clf=logistic, sampler=SMOTE(random_state=0, sampling_strategy=0.5)
    )
    # return nn_pipe
    return VotingClassifier(
        estimators=[
            ("nn", nn_pipe)
        ],  # , ("xgb", xgb_pipe), ("logistic", logistic_pipe)],
        voting="soft",
    )
