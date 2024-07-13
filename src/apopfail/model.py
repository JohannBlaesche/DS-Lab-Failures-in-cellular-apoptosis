"""Training step in the pipeline."""

from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
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


class ApopfailBlock(nn.Module):
    """Simple block for a neural network."""

    def __init__(self, in_features, out_features, activation, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation(),
            nn.Dropout(p),
        )

    def forward(self, X, **kwargs):
        """Forward pass of the block."""
        return self.block(X)


class ApopfailNeuralNet(nn.Module):
    """Simple Neural Network for binary classification."""

    def __init__(self, input_size=5408, p=0.2, activation=nn.GELU):
        super().__init__()

        self.seq = nn.Sequential(
            ApopfailBlock(input_size, 256, activation, p),
            ApopfailBlock(256, 128, activation, p),
            ApopfailBlock(128, 64, activation, p),
            nn.Linear(64, 1),
        )

    def forward(self, X, **kwargs):
        """Forward pass of the neural network."""
        return self.seq(X)


def build_nn(input_size=5408, p=0.2, activation=nn.LeakyReLU, monitor="valid_loss"):
    """Build a simple neural network for binary classification."""
    early_stopping = EarlyStopping(patience=20, monitor=monitor, load_best=True)
    ap = EpochScoring(
        scoring="average_precision",
        lower_is_better=False,
        name="valid_ap",
        on_train=False,
    )
    recall = EpochScoring(
        scoring="recall",
        lower_is_better=False,
        name="valid_recall",
        on_train=False,
    )
    clf = NeuralNetBinaryClassifier(
        ApopfailNeuralNet,
        module__activation=activation,
        module__input_size=input_size,
        module__p=p,
        max_epochs=200,
        lr=3e-4,
        device="cuda",
        optimizer__weight_decay=3e-4,
        iterator_train__shuffle=True,
        callbacks=[early_stopping, ap, recall],
    )
    return clf
