"""Model comparison."""

import json

import numpy as np
from loguru import logger
from pyod.models.iforest import IForest

from apopfail.model import get_pipeline
from apopfail.occ import occ


def compare_occ_models(X, y, n_repeats):
    """Compare One-Class Classification models."""
    models = build_model_dict()
    result_dict = {}

    for model_key, model in models.items():
        metrics = {
            "ROC AUC": [],
            "Average Precision": [],
            "Recall (Sensitivity)": [],
            "F2 Score": [],
        }

        for i in range(n_repeats):
            model, scores = occ(model, X, y, random_state=i)
            for metric, value in scores.items():
                if metric in metrics:
                    metrics[metric].append(value)

        # Save intermediate results for each model as JSON
        with open(f"output/{model_key}_scores.json", "w") as outfile:
            json.dump(metrics, outfile)

        result_dict[model_key] = metrics

    # create a dict with just one target metric that could be used for ranking
    target_metric = "Average Precision"
    target_metric_dict = {
        model_key: metrics[target_metric] for model_key, metrics in result_dict.items()
    }

    best_model_key, best_score = max(
        (
            (model_key, np.mean(metrics))
            for model_key, metrics in target_metric_dict.items()
        ),
        key=lambda item: item[1],
    )
    logger.info(f"Chose the model {best_model_key} with average score of {best_score}")

    best_model = models[best_model_key]
    return best_model


def build_model_dict():
    """Build a list of models."""
    clf = IForest()
    model = get_pipeline(clf=clf)
    model_dict = {"isolation_forest": model}
    return model_dict
