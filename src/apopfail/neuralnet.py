"""Neural net classifier implementation."""

import torch
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from torch import nn


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
            nn.Linear(32, 1),
        )

    def forward(self, X, **kwargs):
        """Forward pass of the neural network."""
        return self.seq(X)


# if the net is in a VotingClassifier, we need to use this subclass
# because votingclassifier transforms y with a LabelEncoder
# which changes the type of y
class MyNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    """Custom NeuralNetBinaryClassifier to use custom module."""

    def fit(self, X, y, **fit_params):
        """Fit the model on the dataset."""
        y = y.astype("float32")
        return super().fit(X, y=y, **fit_params)


def build_nn(
    input_size=5408,
    p=0.3,
    activation=nn.LeakyReLU,
    monitor="valid_loss",
    pos_weight=0.9,
):
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
        # if sampling is used, set pos_weight < 1 (increase Precision), else > 1 (increase Recall)
        criterion__pos_weight=torch.tensor([pos_weight]),
        callbacks=[early_stopping, ap, recall],
        verbose=False,
    )
    return clf
