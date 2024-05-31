import json  # noqa: D100

import optuna
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from dmgpred.train import get_pipeline


def tune_XGBoost(X, y, n_trials=250, random_state=0):
    """Tune the model with optuna."""
    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    def objective(trial):
        """Tune XGBoost."""
        config = {
            "objective": trial.suggest_categorical("objective", ["multi:softprob"]),
            # "eval_metric": ["mlogloss", "merror"],
            # "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-1, log=True),  # noqa: E501
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.0906481523921039]
            ),
            # "n_estimators": trial.suggest_int("n_estimators",500,2000, step=50),
            "n_estimators": trial.suggest_categorical("n_estimators", [1850]),
            # "max_depth": trial.suggest_int("max_depth", 1, 10),
            "max_depth": trial.suggest_categorical("max_depth", [5]),
            # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "subsample": trial.suggest_categorical("subsample", [0.7296057237367]),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.7504806008221163]
            ),
            # "gamma": trial.suggest_int("gamma", 0, 6),
            # "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [6]),
            # "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-2, 0.1, 1, 100])
            # "reg_lambda": trial.suggest_int("reg_lambda", [0,150])
            "reg_lambda": trial.suggest_categorical("reg_lambda", [103]),
            "enable_categorical": trial.suggest_categorical(
                "enable_categorical", [True]
            ),
            "random_state": trial.suggest_categorical("random_state", [0]),
            "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
            "seed": trial.suggest_categorical("seed", [0]),
            "device": trial.suggest_categorical("device", ["cuda"]),
        }
        clf = XGBClassifier(**config)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = matthews_corrcoef(y_test, y_pred)
        return score

    study_name = "optimize_xgboost"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name, direction="maximize")
    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value}")
    with open(f"./output/{study_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best params", best_params)
    return best_params
