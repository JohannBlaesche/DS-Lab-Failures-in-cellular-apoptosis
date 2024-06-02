"""Optimize hyperparameters for a given model using Optuna."""

import json
import logging
from typing import Literal

import joblib
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from dmgpred.tune.catboost import (
    get_catboost_objective,
)
from dmgpred.tune.lgbm import get_lgbm_objective
from dmgpred.tune.xgboost import get_xgb_objective
from dmgpred.utils.eval import evaluate_clf
from dmgpred.utils.logging import InterceptHandler


def run_optimization(
    X,
    y,
    n_trials=100,
    random_state=0,
    study_name=Literal["lgbm", "xgb", "catboost"],
    use_gpu=True,
) -> dict:
    """Tune the model with optuna."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_full,
    )

    if study_name == "xgb":
        objective, X_val_processed = get_xgb_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )
    elif study_name == "catboost":
        objective, X_val_processed = get_catboost_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )
    else:
        objective, X_val_processed = get_lgbm_objective(
            X_train, y_train, X_val, y_val, use_gpu=use_gpu
        )

    storage_name = f"sqlite:///./output/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, timeout=1 * 60 * 60
    )

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value:.4f}")

    logger.info("Evaluating parameters on hold-out test set...")

    fit_kwargs = {"clf__eval_set": [(X_val_processed, y_val)]}

    if study_name == "xgb":
        clf = XGBClassifier(**best_params)
        fit_kwargs["clf__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y_train
        )
    elif study_name == "catboost":
        clf = CatBoostClassifier(**best_params)
    else:
        clf = LGBMClassifier(**best_params)
        fit_kwargs["clf__callbacks"] = [lgb.early_stopping(50, verbose=False)]

    clf, _ = evaluate_clf(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        **fit_kwargs,
    )
    with open(f"./output/{study_name}_best_clf.pkl", "wb") as f:
        joblib.dump(clf, f)
    best_iter = clf.best_iteration_
    best_params["n_estimators"] = best_iter
    best_params.pop("early_stopping_rounds", None)
    with open(f"./output/{study_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    return best_params
