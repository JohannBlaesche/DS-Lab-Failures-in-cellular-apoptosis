"""Hyperparameter tuning for the model."""

import json

import joblib
import optuna
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from loguru import logger
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from dmgpred.cleaning import get_normalizer
from dmgpred.featurize import get_encoder
from dmgpred.train import get_pipeline

# set_config(enable_metadata_routing=True)


def tune_optuna(X, y, n_trials=100, random_state=0):
    """Tune the model with optuna."""
    # create train-val-test split
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

    # TODO: create notebook for viszualization of optuna study
    #       (to choose better parameter ranges)
    # TODO: catboost cat_features without preprocessing
    #       --> error when defining cat_features
    def objective(trial):
        """Tune Catboost."""
        space = {
            "iterations": trial.suggest_int("iterations", 100, 10000, log=True),
            "logging_level": trial.suggest_categorical("logging_level", ["Silent"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-1, 0.3),
            "task_type": trial.suggest_categorical("task_type", ["GPU"]),
            "depth": trial.suggest_int("depth", 2, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            # "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "border_count": trial.suggest_int("border_count", 1, 128, log=True),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec"]),
            "od_wait": trial.suggest_categorical("od_wait", [50]),
            "od_pval": trial.suggest_categorical("od_pval", [1e-10]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["MCC"]),
            # "cat_features": trial.suggest_categorical("cat_features", [cat_features])
        }

        pre_Pipeline = Pipeline(
            [
                ("normalizer", get_normalizer()),
                ("encoder", get_encoder(X_train)),
            ],
            verbose=False,
        )

        pre_Pipeline.fit(X_train, y_train)
        X_test_fitted = pre_Pipeline.transform(X_test)
        clf = CatBoostClassifier(**space)
        model = get_pipeline(X_train, clf)
        model.fit(X_train, y_train, clf__eval_set=(X_test_fitted, y_test))
        y_pred = model.predict(X_val)
        score = matthews_corrcoef(y_val, y_pred)
        logger.info(f"Final score on test set: {score:.4f}")
        return score

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name="optimize_lgbm",
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
    )
    logger.info(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Study completed with best score: {study.best_value}")

    with open("./output/catboost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    joblib.dump(study, "./output/catboost_study.pkl")

    logger.info("Evaluating parameters on hold-out test set...")
    clf = CatBoostClassifier(**best_params)
    model = get_pipeline(X_train, clf)
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test)
    score = matthews_corrcoef(y_test, y_pred)
    logger.info(f"Final score on test set: {score:.4f}")

    return CatBoostClassifier(**best_params)


# # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_metadata_routing.html
# class RouterConsumerClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):

#     def __init__(self, estimator):
#         self.estimator = estimator

#     def get_metadata_routing(self):
#         router = (
#             MetadataRouter(owner=self.__class__.__name__)
#             # defining metadata routing request values for usage in the meta-estimator
#             .add_self_request(self)
#             # defining metadata routing request values for usage in the sub-estimator
#             .add(
#                 estimator=self.estimator,
#                 method_mapping=MethodMapping()
#                 .add(caller="fit", callee="fit")
#                 .add(caller="predict", callee="predict")
#                 .add(caller="score", callee="score")
#                 .add(caller="predict_proba", callee="predict_proba")
#             )
#         )
#         return router

#     # Since `sample_weight` is used and consumed here, it should be defined as
#     # an explicit argument in the method's signature. All other metadata which
#     # are only routed, will be passed as `**fit_params`:
#     def fit(self, X, y, eval_set, **fit_params):
#         if self.estimator is None:
#             raise ValueError("estimator cannot be None!")

#         # check_metadata(self, eval_set=eval_set)

#         # We add `eval_set` to the `fit_params` dictionary.
#         if eval_set is not None:
#             fit_params["eval_set"] = eval_set

#         request_router = get_routing_for_object(self)
#         request_router.validate_metadata(params=fit_params, method="fit")
#         routed_params = request_router.route_params(params=fit_params,
#                                                       caller="fit")
#         self.estimator_ = clone(self.estimator).fit(X, y,
#                                                     **routed_params.estimator.fit)
#         self.classes_ = self.estimator_.classes_
#         return self

#     def predict(self, X, **predict_params):
#         check_is_fitted(self)
#         # As in `fit`, we get a copy of the object's MetadataRouter,
#         request_router = get_routing_for_object(self)
#         # we validate the given metadata,
#         request_router.validate_metadata(params=predict_params, method="predict")
#         # and then prepare the input to the underlying ``predict`` method.
#         routed_params = request_router.route_params(
#             params=predict_params, caller="predict"
#         )
#         return self.estimator_.predict(X, **routed_params.estimator.predict)

#     def predict_proba(self, X, **predict_params):
#         request_router = get_routing_for_object(self)
#         request_router.validate_metadata(params=predict_params,#
#                                           method="predict_proba")
#         routed_params = request_router.route_params(
#             params=predict_params, caller="predict_proba"
#         )
#         return self.estimator_.predict_proba(X,
#                               **routed_params.estimator.predict_proba)
