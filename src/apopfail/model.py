"""Training step in the pipeline."""

import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import DataFrameTransformer
from torch import nn
from xgboost import XGBClassifier


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
        if clf.__class__.__name__.endswith("NeuralNetBinaryClassifier"):
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

    def __init__(self, input_size=5408, p=0.3, activation=nn.GELU):
        super().__init__()

        self.seq = nn.Sequential(
            ApopfailBlock(input_size, 256, activation, p),
            ApopfailBlock(256, 128, activation, p),
            ApopfailBlock(128, 64, activation, p),
            ApopfailBlock(64, 32, activation, p),
            nn.Linear(64, 1),
        )

    def forward(self, X, **kwargs):
        """Forward pass of the neural network."""
        return self.seq(X)


# if the net is in a VotingClassifier, we need to use this subclass
class MyNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    """Custom NeuralNetBinaryClassifier to use custom module."""

    def fit(self, X, y, **fit_params):
        """Fit the model on the dataset."""
        y = y.astype("float32")
        return super().fit(X, y=y, **fit_params)


def build_nn(input_size=5408, p=0.3, activation=nn.LeakyReLU, monitor="valid_loss"):
    """Build a simple neural network for binary classification."""
    early_stopping = EarlyStopping(patience=10, monitor=monitor, load_best=True)
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
    clf = MyNeuralNetBinaryClassifier(
        ApopfailNeuralNet,
        module__activation=activation,
        module__input_size=input_size,
        module__p=p,
        max_epochs=500,
        lr=3e-4,
        device="cuda",
        optimizer__weight_decay=5e-2,
        iterator_train__shuffle=True,
        criterion__pos_weight=torch.tensor([0.9]),
        callbacks=[early_stopping, ap, recall],
    )
    return clf


def build_model():
    """Build the model for the pipeline."""
    nn = build_nn(input_size=3500)
    nn_pipe = get_pipeline(
        clf=nn,
        reducer=PCA(n_components=3500),
        # reducer="passthrough",
        # sampler=SMOTE(random_state=0, sampling_strategy=0.2),
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
    xgb_pipe = get_pipeline(clf=xgb, sampler=SMOTE(random_state=0))

    # return nn_pipe
    return VotingClassifier(
        estimators=[("nn", nn_pipe), ("xgb", xgb_pipe)], voting="soft"
    )
