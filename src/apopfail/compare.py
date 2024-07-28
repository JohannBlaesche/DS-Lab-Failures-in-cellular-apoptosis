"""Model comparison."""

import json
import os
from pathlib import Path

import numpy as np
from loguru import logger
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cof import COF
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import StandardScaler
from pyod.models.deep_svdd import DeepSVDD
from tqdm import tqdm

from apopfail.model import get_pipeline
from apopfail.occ import occ


def compare_occ_models(X, y, n_repeats, skip_existing=True):
    """Compare One-Class Classification models."""
    models = build_model_dict()
    result_dict = {}

    for model_key, model in tqdm(models.items(), desc="Iterating Models"):
        metrics = {}

        path = Path("output", "scores")
        path.mkdir(parents=True, exist_ok=True)
        model_scores_path = path / f"{model_key}_scores.json"
        if skip_existing and os.path.exists(model_scores_path):
            continue

        random_states = np.random.randint(low=0, high=1000, size=n_repeats)

        for i in tqdm(random_states, leave=False, desc="Computing Metrics"):
            model, scores = occ(
                model, X, y, random_state=i, refit=False, model_name=model_key
            )
            for metric, value in scores.items():
                vals = metrics.get(metric, [])
                vals.append(value)
                metrics[metric] = vals

        # Save intermediate results for each model as JSON
        with open(model_scores_path, "w") as outfile:
            json.dump(metrics, outfile, indent=4)

        for metric, values in metrics.items():
            logger.info(
                f"{model_key} {metric}: {np.mean(values): .4f} (±{np.std(values): .4f})"
            )

        result_dict[model_key] = metrics

    target_metric = "Average Precision"
    best_model_key = None
    best_model_score = 0

    for model_key, metrics in result_dict.items():
        model_results = metrics[target_metric]
        result_mean = np.mean(model_results)
        result_std = np.std(model_results)
        logger.info(f"{model_key}: {result_mean: .4f} (±{result_std: .4f})")
        if result_mean > best_model_score:
            best_model_score = result_mean
            best_model_key = model_key

    logger.info(
        f"Chose the model {best_model_key} with average score {best_model_score: .4f}"
    )

    best_model = models[best_model_key]
    return best_model


def build_model_dict():
    """Build a list of models."""
    autoencoder = AutoEncoder(
        random_state=0,
        preprocessing=False,
        batch_size=256,
        epoch_num=50,
        verbose=0,
        hidden_neuron_list=[128, 64],
        dropout_rate=0.2,
        device="cuda",
        use_compile=True,
        hidden_activation_name="leaky_relu",
    )
    lof = LOF()
    cof = COF()
    ocsvm = OCSVM()
    lunar = LUNAR(n_neighbours=1)

    lof_model = get_pipeline(clf=lof)
    cof_model = get_pipeline(clf=cof)
    ocsvm_model = get_pipeline(clf=ocsvm)
    lunar_model = get_pipeline(clf=lunar)

    n_features = 5408
    deep_svdd = DeepSVDD(
        n_features=n_features,
        hidden_neurons=[128, 64, 32],
        hidden_activation="leaky_relu",
        contamination=0.2,
        batch_size=32,
        preprocessing=False,
        epochs=50,
        validation_size=0.2,
        random_state=0,
    )
    reducer = "passthrough"
    deep_svdd_model = get_pipeline(clf=deep_svdd, reducer=reducer)  # noqa: F841

    model_dict = {
        # "deep_svdd": deep_svdd_model,
        "autoencoder no dimension reduction": get_pipeline(
            clf=autoencoder, reducer="passthrough"
        ),
        "lof": lof_model,
        "cof": cof_model,
        "ocsvm": ocsvm_model,
        "lunar": lunar_model,
    }
    for model in model_dict.values():
        model.set_params(clf__contamination=0.3)
    return model_dict
