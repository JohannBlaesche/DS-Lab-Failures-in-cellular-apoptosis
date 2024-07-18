"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch import nn
from umap import UMAP


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
    # clf = NeuralNetBinaryClassifier(
    #     ApopfailNeuralNet,
    #     max_epochs=15,
    #     lr=0.001,
    #     optimizer=Adam,
    #     criterion=nn.BCEWithLogitsLoss,
    #     device="cuda",
    #     optimizer__weight_decay=0.001,
    # )
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", scaler or RobustScaler()),
        ("reducer", reducer or UMAP(n_components=200)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))

    if clf is not None:
        steps.append(("clf", clf))

    return Pipeline(steps=steps)


class ApopfailNeuralNet(nn.Module):
    """Simple Neural Network for binary classification."""

    def __init__(self, nonlin=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(900, 256)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, X, **kwargs):
        """Forward pass of the neural network."""
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.output(X)
        X = nn.Flatten(start_dim=0)(X)
        return X
