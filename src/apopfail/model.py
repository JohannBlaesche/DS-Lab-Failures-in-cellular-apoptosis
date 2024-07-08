"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skorch.helper import DataFrameTransformer
from torch import nn


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
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", scaler or StandardScaler()),
        ("reducer", reducer or PCA(n_components=0.99)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))

    if clf is not None:
        if clf.__class__.__name__ == "NeuralNetBinaryClassifier":
            steps.append(("flatten", DataFrameTransformer()))
        steps.append(("clf", clf))

    return Pipeline(steps=steps)


class ApopfailNeuralNet(nn.Module):
    """Simple Neural Network for binary classification."""

    def __init__(self, input_size=5408, p=0.2):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, X, **kwargs):
        """Forward pass of the neural network."""
        return self.seq(X)
