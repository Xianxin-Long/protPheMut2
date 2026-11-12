"""Tune and refit the four base models"""

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike
import optuna
import sklearn
from sklearn import calibration
from sklearn import ensemble
from sklearn import impute
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm
import xgboost


CvSplits = Iterable[tuple[np.ndarray, np.ndarray]]
_FIT_ERRORS = (
    ValueError, RuntimeError, xgboost.core.XGBoostError,
)


def brier_loss(y_true: ArrayLike, proba: ArrayLike) -> float:
    """Compute half the mean summed squared probability error.

    Args:
        y_true: Integer class indices corresponding to probability columns.
        proba: Matrix of shape (n_samples, n_classes), in fixed class order.

    Returns:
        Brier loss, equal to positive-class MSE for binary classification.
    """
    probabilities = np.asarray(proba, dtype=float)
    return metrics.brier_score_loss(
        y_true,
        probabilities,
        labels=np.arange(probabilities.shape[1]),
        scale_by_half=True,
    )


def _make_pipeline(estimator, *, standardize: bool = False):
    """Keep imputation and optional scaling inside each model's fit boundary."""
    steps = [impute.SimpleImputer(strategy="median", keep_empty_features=True)]
    if standardize:
        steps.append(preprocessing.StandardScaler())
    return pipeline.make_pipeline(*steps, estimator)


def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Use the earliest completed trial within 1e-6 of the minimum loss."""
    minimum = study.best_value
    return next(
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value <= minimum + 1e-6
    )


def _tune_and_refit(
    cv_splits: CvSplits,
    x_train: ArrayLike,
    y_train: ArrayLike,
    suggest_params: Callable,
    make_model: Callable,
    n_trials: int,
    sampler_seed: int,
):
    """Share sequential tuning, sample-weighted scoring, and final refitting."""
    features = np.asarray(x_train, dtype=float)
    labels = np.asarray(y_train)
    splits = list(cv_splits)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, n_startup_trials=10
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fold_losses = []
        fold_sizes = []
        for fit_idx, val_idx in splits:
            model = make_model(params, fit_idx)
            model.fit(features[fit_idx], labels[fit_idx])
            probabilities = model.predict_proba(features[val_idx])
            fold_losses.append(brier_loss(labels[val_idx], probabilities))
            fold_sizes.append(len(val_idx))
        return float(np.average(fold_losses, weights=fold_sizes))

    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=_FIT_ERRORS)
    best_trial = _select_best_trial(study)
    model = make_model(best_trial.params, np.arange(len(labels)))
    model.fit(features, labels)
    return model


def train_elastic_net(
    cv_splits: CvSplits,
    X_train: ArrayLike,
    y_train: ArrayLike,
    n_trials: int = 50,
    sampler_seed: int = 42,
) -> pipeline.Pipeline:
    """Tune Elastic Net logistic regression and refit on all training rows.

    Args:
        cv_splits: Fixed grouped fit/validation positions relative to X_train.
        X_train: Numeric training features; NaN is imputed within each fit.
        y_train: Integer labels 0..K-1 in the predefined class order.
        n_trials: Number of Optuna attempts, defaulting to 50.
        sampler_seed: Seed for sequential TPE and model randomness.

    Returns:
        Fitted imputation/scaling/classifier pipeline.
    """
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def make_model(params: dict, fit_idx: np.ndarray) -> pipeline.Pipeline:
        del fit_idx
        # sklearn < 1.8 also requires an explicit Elastic Net penalty.
        legacy = {}
        if tuple(map(int, sklearn.__version__.split(".")[:2])) < (1, 8):
            legacy["penalty"] = "elasticnet"
        return _make_pipeline(linear_model.LogisticRegression(
            **params, **legacy, solver="saga", max_iter=10000,
            random_state=sampler_seed,
        ), standardize=True)

    return _tune_and_refit(cv_splits, X_train, y_train, suggest_params,
                          make_model, n_trials, sampler_seed)


def train_rbf_svm(
    cv_splits: CvSplits,
    X_train: ArrayLike,
    y_train: ArrayLike,
    n_trials: int = 50,
    sampler_seed: int = 42,
    *,
    groups: ArrayLike,
) -> calibration.CalibratedClassifierCV:
    """Tune RBF-SVM with training-only grouped sigmoid probability calibration.

    Args:
        cv_splits: Fixed grouped CV row positions within X_train.
        X_train: Numeric training features, allowing NaN.
        y_train: Integer labels 0..K-1 in the predefined class order.
        n_trials: Number of Optuna attempts, defaulting to 50.
        sampler_seed: Seed for TPE, SVM, and calibration partition generation.
        groups: Required Site values aligned with the training rows.

    Returns:
        Fitted calibrated predictor.
        Two-fold grouped scores train sigmoid mappings; preprocessing and raw
        SVM are then refitted on the full current training partition. Multiclass
        OvR sigmoid probabilities are normalized to sum to one.
    """
    labels = np.asarray(y_train)
    sites = np.asarray(groups)
    splits = list(cv_splits)
    # Freeze internal folds once, relative to each current fitting subset.
    splitter = model_selection.StratifiedGroupKFold(
        n_splits=2, shuffle=True, random_state=sampler_seed
    )
    calibration_folds = {
        tuple(fit): list(splitter.split(labels[fit], labels[fit], sites[fit]))
        for fit in [fit for fit, _ in splits] + [np.arange(len(labels))]
    }

    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical(
                "gamma", ["scale", 0.001, 0.01, 0.1]
            ),
        }

    def make_model(params: dict, fit_idx: np.ndarray):
        raw_svm = _make_pipeline(svm.SVC(
            **params, kernel="rbf", decision_function_shape="ovr",
            random_state=sampler_seed,
        ), standardize=True)
        return calibration.CalibratedClassifierCV(
            estimator=raw_svm, method="sigmoid",
            cv=calibration_folds[tuple(fit_idx)], ensemble=False, n_jobs=1,
        )

    return _tune_and_refit(splits, X_train, y_train, suggest_params,
                          make_model, n_trials, sampler_seed)


def train_extra_trees(
    cv_splits: CvSplits,
    X_train: ArrayLike,
    y_train: ArrayLike,
    n_trials: int = 50,
    sampler_seed: int = 42,
) -> pipeline.Pipeline:
    """Tune 500-tree ExtraTrees with fold-local median imputation.

    Args:
        cv_splits: Fixed grouped CV row positions within X_train.
        X_train: Numeric training features, allowing NaN.
        y_train: Integer labels 0..K-1 in the predefined class order.
        n_trials: Number of Optuna attempts, defaulting to 50.
        sampler_seed: Seed for sequential TPE and tree randomness.

    Returns:
        Fitted imputation/classifier pipeline.
    """
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [2, 5, 10, 20]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", [0.5, 0.7, 1.0]
            ),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, None]),
        }

    def make_model(params: dict, fit_idx: np.ndarray) -> pipeline.Pipeline:
        del fit_idx
        return _make_pipeline(ensemble.ExtraTreesClassifier(
            **params, n_estimators=500, random_state=sampler_seed,
            class_weight=None, n_jobs=1,
        ))

    return _tune_and_refit(cv_splits, X_train, y_train, suggest_params,
                          make_model, n_trials, sampler_seed)


def train_xgboost(
    cv_splits: CvSplits,
    X_train: ArrayLike,
    y_train: ArrayLike,
    n_trials: int = 50,
    sampler_seed: int = 42,
) -> pipeline.Pipeline:
    """Tune binary/multiclass XGBoost without early stopping.

    Args:
        cv_splits: Fixed grouped CV row positions within X_train.
        X_train: Numeric training features, allowing NaN.
        y_train: Integer labels 0..K-1 in the predefined class order.
        n_trials: Number of Optuna attempts, defaulting to 50.
        sampler_seed: Seed for sequential TPE and tree randomness.

    Returns:
        Fitted imputation/classifier pipeline.
    """
    n_classes = len(np.unique(y_train))

    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [100, 200, 300]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "min_child_weight": trial.suggest_categorical(
                "min_child_weight", [1, 5, 10]
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0, 30.0, log=True
            ),
        }

    def make_model(params: dict, fit_idx: np.ndarray) -> pipeline.Pipeline:
        del fit_idx
        objective_params = {"objective": "binary:logistic"}
        if n_classes > 2:
            objective_params = {
                "objective": "multi:softprob", "num_class": n_classes
            }
        return _make_pipeline(xgboost.XGBClassifier(
            **params, **objective_params, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, random_state=sampler_seed, n_jobs=1,
        ))

    return _tune_and_refit(cv_splits, X_train, y_train, suggest_params,
                          make_model, n_trials, sampler_seed)
