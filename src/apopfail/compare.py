"""Model comparison."""

import json
import os

import numpy as np
from loguru import logger
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from tqdm import tqdm

from apopfail.model import get_pipeline
from apopfail.occ import occ


def compare_occ_models(X, y, n_repeats, skip_existing=True):
    """Compare One-Class Classification models."""
    models = build_model_dict()
    result_dict = {}

    for model_key, model in tqdm(models.items()):
        metrics = {
            "ROC AUC": [],
            "Average Precision": [],
            "Recall (Sensitivity)": [],
            "F2 Score": [],
        }

        if skip_existing and os.path.exists(f"output/{model_key}_scores.json"):
            logger.info("Skipping existing model...")
            continue

        logger.info("Running model: " + model_key)

        for i in tqdm(range(n_repeats), leave=False):
            model, scores = occ(model, X, y, random_state=i, refit=False)
            for metric, value in scores.items():
                if metric in metrics:
                    metrics[metric].append(value)

        # Save intermediate results for each model as JSON
        with open(f"output/{model_key}_scores.json", "w") as outfile:
            json.dump(metrics, outfile, indent=4)

        result_dict[model_key] = metrics

    target_metric = "Average Precision"
    best_model_key = None
    best_model_score = 0

    for model_key, metrics in result_dict.items():
        model_results = metrics[target_metric]
        result_mean = np.mean(model_results)
        result_std = np.std(model_results)
        logger.info(f"{model_key}: {result_mean: .4f}(±{result_std: .4f})")
        if result_mean > best_model_score:
            best_model_score = result_mean
            best_model_key = model_key

    logger.info(
        f"Chose the model {best_model_key} with average score of {best_model_score}"
    )

    best_model = models[best_model_key]
    return best_model


def build_model_dict():
    """Build a list of models."""
    iforest = IForest(random_state=0)
    abod = ABOD()
    autoencoder = AutoEncoder(
        random_state=0,
        preprocessing=False,
        epoch_num=25,
        verbose=0,
        hidden_neuron_list=[256, 128, 64],
        dropout_rate=0.1,
    )

    model_dict = {
        "isolation_forest": iforest,
        "abod": abod,
        "autoencoder": autoencoder,
    }
    for model_key, model in model_dict.items():
        model.set_params(contamination=0.01)
        model_dict[model_key] = get_pipeline(clf=model)
    return model_dict
