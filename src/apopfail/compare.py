"""Model comparison."""

import json

import numpy as np
from loguru import logger
from pyod.models.cof import COF
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

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
            logger.info(f"Round {i} of OCC with model {model_key}.")
            model, scores = occ(model, X, y, random_state=i, refit=False)
            for metric, value in scores.items():
                if metric in metrics:
                    metrics[metric].append(value)

        # Save intermediate results for each model as JSON
        with open(f"output/{model_key}_scores_0.1.json", "w") as outfile:
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
        f"Chose the model {best_model_key} with average score {best_model_score: .4f}"
    )

    best_model = models[best_model_key]
    return best_model


def build_model_dict():
    """Build a dict of models."""
    cont = 0.1
    lof = LOF(contamination=cont)
    cof = COF(contamination=cont)
    ocsvm = OCSVM(contamination=cont)

    lof_model = get_pipeline(clf=lof)
    cof_model = get_pipeline(clf=cof)
    ocsvm_model = get_pipeline(clf=ocsvm)
    model_dict = {
        "lof": lof_model,
        "cof": cof_model,
        "ocsvm": ocsvm_model,
    }
    return model_dict
