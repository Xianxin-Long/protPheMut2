"""Regression tests for grouped splits and probability model training."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.ml_pipeline import data
from src.ml_pipeline import train


def make_dataset(n_classes=3):
    """Creates numeric features with repeated sites and missing values."""
    rng = np.random.default_rng(18)
    groups = np.repeat(np.arange(18), n_classes)
    labels = np.tile(np.arange(n_classes), 18)
    features = np.column_stack([
        np.arange(len(labels)),
        labels + rng.normal(0, 0.25, len(labels)),
        rng.normal(size=len(labels)),
        np.full(len(labels), np.nan),
    ])
    features[::7, 2] = np.nan
    frame = pd.DataFrame({"Site": groups, "label": labels})
    frame.index = np.arange(len(frame)) * 7 + 101
    return features, labels, groups, frame


def assert_probability_matrix(model, features, n_classes):
    """Checks complete class order and valid predicted probability vectors."""
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(features), n_classes)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(model.classes_, np.arange(n_classes))


def test_holdout_keeps_sites_together_and_returns_row_positions():
    _, _, groups, frame = make_dataset()
    fit_indices, test_indices = data.split_by_position(frame)
    assert set(groups[fit_indices]).isdisjoint(groups[test_indices])
    np.testing.assert_array_equal(
        np.sort(np.r_[fit_indices, test_indices]), np.arange(len(frame))
    )
    assert len(np.unique(groups[test_indices])) == 4
    repeated = data.split_by_position(frame)
    for actual, expected in zip(repeated, (fit_indices, test_indices)):
        np.testing.assert_array_equal(actual, expected)


def test_cv_indices_are_local_to_training_frame_and_cover_it_once():
    _, _, _, frame = make_dataset()
    fit_indices, _ = data.split_by_position(frame)
    subset = frame.iloc[fit_indices]
    folds = data.cross_validation_split(subset)
    assert len(folds) == 3
    validation_indices = []
    for train_indices, valid_indices in folds:
        fit_frame = subset.iloc[train_indices]
        valid_frame = subset.iloc[valid_indices]
        assert set(fit_frame["Site"]).isdisjoint(valid_frame["Site"])
        assert set(fit_frame["label"]) == {0, 1, 2}
        assert set(valid_frame["label"]) == {0, 1, 2}
        validation_indices.extend(valid_indices)
    np.testing.assert_array_equal(
        np.sort(validation_indices), np.arange(len(subset))
    )


def test_binary_brier_is_positive_class_mean_squared_error():
    labels = np.array([0, 1, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    assert train.brier_loss(labels, probabilities) == pytest.approx(
        (0.2**2 + 0.3**2 + 0.6**2) / 3
    )


def test_multiclass_brier_uses_half_sum_of_squared_errors():
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
    expected = ((0.04 + 0.01 + 0.01) + (0.04 + 0.25 + 0.49)) / 4
    assert train.brier_loss(np.array([0, 2]), probabilities) == (
        pytest.approx(expected)
    )


@pytest.mark.parametrize("trainer_name", [
    "train_elastic_net", "train_rbf_svm", "train_extra_trees",
    "train_xgboost",
])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_training_returns_usable_probabilities_with_missing_data(
    trainer_name, n_classes
):
    features, labels, groups, frame = make_dataset(n_classes)
    folds = data.cross_validation_split(frame)
    trainer = getattr(train, trainer_name)
    assert inspect.signature(trainer).parameters["n_trials"].default == 30
    options = {"groups": groups} if trainer_name == "train_rbf_svm" else {}
    # Deliberately retain nonconsecutive labels on the pandas row index.
    feature_frame = pd.DataFrame(features, index=frame.index)
    label_series = pd.Series(labels, index=frame.index)
    model = trainer(
        folds, feature_frame, label_series, n_trials=1, **options
    )
    assert_probability_matrix(model, feature_frame.iloc[:8], n_classes)
    assert len(model.study_.trials) == 1
    assert model.best_trial_number_ == 0
    assert model.best_params_ == model.study_.trials[0].params
    distributions = model.study_.trials[0].distributions
    if trainer_name == "train_elastic_net":
        classifier = model.steps[-1][1]
        assert classifier.solver == "saga"
        assert classifier.max_iter == 10000
        assert distributions["C"].log
        assert (distributions["C"].low, distributions["C"].high) == (
            0.01, 10
        )
        assert not distributions["l1_ratio"].log
        assert (
            distributions["l1_ratio"].low,
            distributions["l1_ratio"].high,
        ) == (0, 1)
    elif trainer_name == "train_rbf_svm":
        classifier = model.estimator.steps[-1][1]
        assert classifier.kernel == "rbf"
        assert not hasattr(classifier, "predict_proba")
        assert classifier.decision_function_shape == "ovr"
        assert distributions["C"].log
        assert (distributions["C"].low, distributions["C"].high) == (
            0.1, 10
        )
        assert distributions["gamma"].choices == (
            "scale", 0.001, 0.01, 0.1
        )
    elif trainer_name == "train_extra_trees":
        classifier = model.steps[-1][1]
        assert classifier.n_estimators == 500
        assert classifier.class_weight is None
        assert distributions["min_samples_leaf"].choices == (2, 5, 10, 20)
        assert distributions["max_features"].choices == (0.5, 0.7, 1.0)
        assert distributions["max_depth"].choices == (5, 10, None)
    else:
        classifier_params = model.steps[-1][1].get_params()
        assert classifier_params["subsample"] == 0.8
        assert classifier_params["colsample_bytree"] == 0.8
        assert classifier_params["reg_alpha"] == 0
        assert classifier_params.get("early_stopping_rounds") is None
        expected_objective = (
            "binary:logistic" if n_classes == 2 else "multi:softprob"
        )
        assert classifier_params["objective"] == expected_objective
        assert distributions["max_depth"].low == 2
        assert distributions["max_depth"].high == 4
        assert distributions["n_estimators"].choices == (100, 200, 300)
        assert distributions["min_child_weight"].choices == (1, 5, 10)
        for name, expected_range in (
            ("learning_rate", (0.01, 0.1)), ("reg_lambda", (1, 30))
        ):
            assert distributions[name].log
            assert (
                distributions[name].low, distributions[name].high
            ) == expected_range


def test_cv_objective_weights_rows_instead_of_unequal_folds(monkeypatch):
    labels = np.array([
        1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    ])
    features = np.arange(48, dtype=float).reshape(24, 2)
    all_indices = np.arange(len(labels))
    folds = [
        (np.setdiff1d(all_indices, valid), valid)
        for valid in (all_indices[:4], all_indices[4:12], all_indices[12:])
    ]

    def constant_probabilities(self, values):
        return np.tile([0.8, 0.2], (len(values), 1))

    monkeypatch.setattr(
        LogisticRegression, "predict_proba", constant_probabilities
    )
    model = train.train_elastic_net(folds, features, labels, n_trials=1)
    expected = np.mean((labels - 0.2)**2)
    assert model.study_.trials[0].value == pytest.approx(expected)


def test_svm_calibration_refits_preprocessing_inside_grouped_partitions(
    monkeypatch
):
    features, labels, groups, frame = make_dataset()
    folds = data.cross_validation_split(frame)
    original_calibration_fit = CalibratedClassifierCV.fit
    original_imputer_fit = SimpleImputer.fit
    active_fit_rows = []
    calibration_scopes = []
    preprocessing_scopes = []

    def audit_calibration_fit(self, values, targets, *args, **kwargs):
        row_ids = np.asarray(values)[:, 0].astype(int)
        assert self.ensemble is False
        assert self.method == "sigmoid"
        inner_folds = list(self.cv)
        assert len(inner_folds) == 2
        allowed = {frozenset(row_ids)}
        for fit_indices, valid_indices in inner_folds:
            fit_ids = row_ids[fit_indices]
            valid_ids = row_ids[valid_indices]
            assert set(groups[fit_ids]).isdisjoint(groups[valid_ids])
            assert set(labels[fit_ids]) == {0, 1, 2}
            assert set(labels[valid_ids]) == {0, 1, 2}
            allowed.add(frozenset(fit_ids))
        calibration_scopes.append(frozenset(row_ids))
        active_fit_rows.append(allowed)
        try:
            return original_calibration_fit(
                self, values, targets, *args, **kwargs
            )
        finally:
            active_fit_rows.pop()

    def audit_imputer_fit(self, values, targets=None, **kwargs):
        row_ids = frozenset(np.asarray(values)[:, 0].astype(int))
        assert active_fit_rows, "Imputation occurred outside calibration."
        assert row_ids in active_fit_rows[-1]
        preprocessing_scopes.append(row_ids)
        return original_imputer_fit(self, values, targets, **kwargs)

    monkeypatch.setattr(
        CalibratedClassifierCV, "fit", audit_calibration_fit
    )
    monkeypatch.setattr(SimpleImputer, "fit", audit_imputer_fit)
    model = train.train_rbf_svm(
        folds, features, labels, n_trials=1, groups=groups
    )
    expected_scopes = [frozenset(fit) for fit, _ in folds]
    expected_scopes.append(frozenset(range(len(features))))
    assert calibration_scopes == expected_scopes
    # Each calibration performs two OOF fits and a full-partition refit.
    assert len(preprocessing_scopes) == 3 * len(expected_scopes)
    assert_probability_matrix(model, features[:6], 3)


def test_fit_failure_is_recorded_and_next_trial_can_succeed(monkeypatch):
    features, labels, _, frame = make_dataset(2)
    folds = data.cross_validation_split(frame)
    original_fit = LogisticRegression.fit
    fit_calls = 0

    def fail_first_fit(self, values, targets, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 1:
            raise ValueError("Injected fitting failure.")
        return original_fit(self, values, targets, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", fail_first_fit)
    model = train.train_elastic_net(folds, features, labels, n_trials=2)
    assert [trial.state.name for trial in model.study_.trials] == [
        "FAIL", "COMPLETE"
    ]
    assert model.best_trial_number_ == 1
    assert_probability_matrix(model, features[:6], 2)


def test_all_nonconverged_trials_report_failure(monkeypatch):
    features, labels, _, frame = make_dataset(2)
    folds = data.cross_validation_split(frame)
    original_create_study = train.optuna.create_study
    created_studies = []

    def capture_study(*args, **kwargs):
        study = original_create_study(*args, **kwargs)
        created_studies.append(study)
        return study

    def nonconverging_fit(self, values, targets, *args, **kwargs):
        warnings.warn("Injected nonconvergence.", ConvergenceWarning)
        raise AssertionError("Convergence warnings must fail the trial.")

    monkeypatch.setattr(LogisticRegression, "fit", nonconverging_fit)
    monkeypatch.setattr(train.optuna, "create_study", capture_study)
    with pytest.raises(ValueError):
        train.train_elastic_net(folds, features, labels, n_trials=2)
    trials = created_studies[0].trials
    assert len(trials) == 2
    assert all(trial.state.name == "FAIL" for trial in trials)


def test_numerically_tied_trials_choose_earlier_configuration(monkeypatch):
    features, labels, _, frame = make_dataset(2)
    folds = data.cross_validation_split(frame)
    original_fit = LogisticRegression.fit
    fit_calls = 0

    def count_fit(self, values, targets, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, values, targets, *args, **kwargs)

    def nearly_tied_loss(targets, probabilities):
        trial_number = (fit_calls - 1) // len(folds)
        return 0.3 - trial_number * 5e-7

    monkeypatch.setattr(LogisticRegression, "fit", count_fit)
    monkeypatch.setattr(train, "brier_loss", nearly_tied_loss)
    model = train.train_elastic_net(folds, features, labels, n_trials=2)
    trials = model.study_.trials
    assert trials[1].value < trials[0].value
    assert model.best_trial_number_ == 0
    assert model.best_params_ == trials[0].params
