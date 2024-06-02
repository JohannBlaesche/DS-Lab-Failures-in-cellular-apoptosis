"""Tune XGBoost."""

from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from dmgpred.train import get_pipeline


def get_xgb_param_space(trial, use_gpu=True):
    """Return the parameter space for XGBoost to tune."""
    device = "gpu" if use_gpu else "cpu"
    n_jobs = 1 if use_gpu else -1
    param_space = {
        "objective": trial.suggest_categorical("objective", ["multi:softmax"]),
        "random_state": trial.suggest_categorical("random_state", [0]),
        "verbosity": trial.suggest_categorical("verbosity", [0]),
        "n_jobs": trial.suggest_categorical("n_jobs", [n_jobs]),
        "device": trial.suggest_categorical("device", [device]),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 50),
        "max_depth": trial.suggest_int("max_depth", 10, 63),
        "num_leaves": trial.suggest_int("num_leaves", 200, 3300),
        "n_estimators": trial.suggest_int("n_estimators", 100_000, 100_000),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.04),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_split_loss": trial.suggest_float("min_split_loss", 0, 2),
    }
    return param_space


def get_xgb_objective(X_train, y_train, X_val, y_val, use_gpu=True):
    """Return the objective function for LGBM."""
    preprocessor = get_pipeline(X_train)
    preprocessor.fit(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    def objective(trial):
        """Tune LGBM."""
        param_space = get_xgb_param_space(trial, use_gpu=use_gpu)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
        clf = XGBClassifier(**param_space, callbacks=[pruning_callback])
        model = get_pipeline(X_train, clf)
        model.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val_processed, y_val)],
            clf__sample_weight=sample_weight,
            clf__verbose=False,
        )
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        return score

    return objective, X_val_processed
