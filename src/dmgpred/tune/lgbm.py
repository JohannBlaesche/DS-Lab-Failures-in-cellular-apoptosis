"""Tune LGBM for the DMGPred project."""

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef

from dmgpred.train import get_pipeline


def get_lgbm_param_space(trial, use_gpu=True, random_state=0):
    """Return the parameter space for LGBM to tune."""
    max_depth = trial.suggest_int("max_depth", 8, 63)
    num_leaves = trial.suggest_int("num_leaves", 7, 3000)
    device = "gpu" if use_gpu else "cpu"
    n_jobs = -1
    param_space = {
        "objective": trial.suggest_categorical("objective", ["multiclass"]),
        "num_class": trial.suggest_categorical("num_class", [3]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
        "n_jobs": trial.suggest_categorical("n_jobs", [n_jobs]),
        "device": trial.suggest_categorical("device", [device]),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "n_estimators": trial.suggest_int("n_estimators", 100_000, 100_000),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.05),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 2),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
    }
    return param_space


def get_lgbm_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_lgbm_param_space(trial, use_gpu=use_gpu)
        clf = LGBMClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
            clf__callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed
