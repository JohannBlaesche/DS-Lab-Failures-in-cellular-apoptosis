"""Tune CatBoost."""

from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef

from dmgpred.train import get_pipeline


def get_catboost_param_space(trial, use_gpu=True):
    """Return the parameter space for CatBoost to tune."""
    param_space = {
        "iterations": trial.suggest_int("iterations", 100_000, 100_000),
        "logging_level": trial.suggest_categorical("logging_level", ["Silent"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.05),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["SqrtBalanced"]
        ),
        "random_seed": trial.suggest_categorical("random_seed", [0]),
        "use_best_model": trial.suggest_categorical("use_best_model", [True]),
        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        # "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "border_count": trial.suggest_int("border_count", 32, 128, log=True),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec"]),
        "od_wait": trial.suggest_categorical("od_wait", [50]),
        "od_pval": trial.suggest_categorical("od_pval", [1e-6]),
        "eval_metric": trial.suggest_categorical("eval_metric", ["MCC"]),
    }
    return param_space


def get_catboost_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_catboost_param_space(trial, use_gpu=use_gpu)
        clf = CatBoostClassifier(**param_space)
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed
