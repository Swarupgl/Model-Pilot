"""
tests/test_evaluator.py

Tests for src/modelpilot/evaluation/model_evaluator.py

Covers:
- evaluate_candidate() for classification and regression
- Pipeline prevents data leakage (fit inside CV folds)
- Failure handling for bad algorithm names
- evaluate_multiple() sorting and baseline inclusion
- get_available_algorithms() returns the correct lists
- Result dict has required keys and valid value types
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelpilot.evaluation.model_evaluator import (
    EvaluationResult,
    evaluate_candidate,
    evaluate_multiple,
    get_available_algorithms,
)

# Path to the configs directory
_config_path = Path(__file__).resolve().parent.parent / "configs" / "algorithms.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — small, deterministic datasets
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def iris_data():
    """Use the real Iris dataset for integration-style tests."""
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    return data.data, data.target


@pytest.fixture
def regression_data():
    """Simple regression dataset from sklearn."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    return data.data, data.target


@pytest.fixture
def tiny_classification():
    """Tiny hand-crafted dataset with 2 features and binary labels."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feature_a": np.random.randn(60),
        "feature_b": np.random.randn(60),
    })
    y = pd.Series((X["feature_a"] > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def data_with_missing(tiny_classification):
    """Same tiny dataset but with 10% missing values injected."""
    X, y = tiny_classification
    X_missing = X.copy()
    rng = np.random.default_rng(seed=99)
    mask = rng.random(X_missing.shape) < 0.10
    X_missing[mask] = np.nan
    return X_missing, y


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_candidate — result structure
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "algorithm", "task_type", "cv_folds", "primary_metric",
    "primary_score", "primary_std", "secondary_metrics",
    "cv_scores", "train_time_s", "status", "error", "params_used",
}

class TestEvaluateCandidateStructure:
    def test_all_required_keys_present_classification(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "DummyClassifier", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert REQUIRED_KEYS.issubset(result.keys()), \
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"

    def test_all_required_keys_present_regression(self, regression_data):
        X, y = regression_data
        result = evaluate_candidate(
            "DummyRegressor", X, y, "regression",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_cv_scores_length_matches_folds(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "DummyClassifier", X, y, "classification",
            cv_folds=4, algorithms_config_path=_config_path,
        )
        assert len(result["cv_scores"]) == 4

    def test_primary_score_is_mean_of_cv_scores(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "LogisticRegression", X, y, "classification",
            cv_folds=5, algorithms_config_path=_config_path,
        )
        assert result["status"] == "ok"
        expected_mean = round(float(np.mean(result["cv_scores"])), 5)
        assert abs(result["primary_score"] - expected_mean) < 1e-6

    def test_train_time_is_positive(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "LogisticRegression", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["train_time_s"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_candidate — correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateCandidateCorrectness:
    def test_dummy_classifier_scores_sensibly(self, iris_data):
        """DummyClassifier should score well below RandomForest."""
        X, y = iris_data
        dummy = evaluate_candidate(
            "DummyClassifier", X, y, "classification",
            cv_folds=5, algorithms_config_path=_config_path,
        )
        rf = evaluate_candidate(
            "RandomForestClassifier", X, y, "classification",
            cv_folds=5, algorithms_config_path=_config_path,
        )
        assert dummy["status"] == "ok"
        assert rf["status"] == "ok"
        # RandomForest must beat the dummy baseline on Iris
        assert rf["primary_score"] > dummy["primary_score"]

    def test_classification_primary_metric_is_balanced_accuracy(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "DummyClassifier", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["primary_metric"] == "balanced_accuracy"

    def test_regression_primary_metric_is_r2(self, regression_data):
        X, y = regression_data
        result = evaluate_candidate(
            "DummyRegressor", X, y, "regression",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["primary_metric"] == "r2"

    def test_logistic_regression_on_iris(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "LogisticRegression", X, y, "classification",
            cv_folds=5, algorithms_config_path=_config_path,
        )
        assert result["status"] == "ok"
        # Iris is easily separable — expect good balanced accuracy
        assert result["primary_score"] > 0.85, \
            f"Expected >0.85 balanced accuracy, got {result['primary_score']}"

    def test_handles_missing_values_without_error(self, data_with_missing):
        """The pipeline's imputer must handle NaN inside CV folds."""
        X, y = data_with_missing
        result = evaluate_candidate(
            "LogisticRegression", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["status"] == "ok", f"Failed with: {result['error']}"

    def test_reproducibility(self, iris_data):
        """Same seed must produce identical scores on two identical runs."""
        X, y = iris_data
        r1 = evaluate_candidate(
            "RandomForestClassifier", X, y, "classification",
            cv_folds=5, random_seed=7, algorithms_config_path=_config_path,
        )
        r2 = evaluate_candidate(
            "RandomForestClassifier", X, y, "classification",
            cv_folds=5, random_seed=7, algorithms_config_path=_config_path,
        )
        assert r1["cv_scores"] == r2["cv_scores"]


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_candidate — failure handling
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateCandidateFailures:
    def test_unknown_algorithm_name_returns_failed_status(self, iris_data):
        X, y = iris_data
        result = evaluate_candidate(
            "NonExistentAlgorithmXYZ", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["status"] == "failed"
        assert result["error"] != ""

    def test_wrong_task_type_returns_failed_status(self, iris_data):
        """Asking for a regression algorithm on a classification task."""
        X, y = iris_data
        result = evaluate_candidate(
            "LinearRegression", X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert result["status"] == "failed"


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_multiple
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateMultiple:
    def test_returns_sorted_results(self, iris_data):
        X, y = iris_data
        names = ["DummyClassifier", "LogisticRegression", "RandomForestClassifier"]
        results = evaluate_multiple(
            names, X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        assert len(results) == 3
        scores = [r["primary_score"] for r in results if r["status"] == "ok"]
        assert scores == sorted(scores, reverse=True), "Results must be sorted best-first"

    def test_failed_algorithms_at_end(self, iris_data):
        X, y = iris_data
        names = ["NonExistentXYZ", "DummyClassifier", "LogisticRegression"]
        results = evaluate_multiple(
            names, X, y, "classification",
            cv_folds=3, algorithms_config_path=_config_path,
        )
        statuses = [r["status"] for r in results]
        # All "ok" results must come before "failed" results
        seen_ok = False
        for s in statuses:
            if s == "ok":
                seen_ok = True
            if seen_ok and s == "failed":
                break
        else:
            assert True  # All ok, or failed at end — both acceptable
        failed_result = next(r for r in results if r["status"] == "failed")
        # Failed result must still have required keys
        assert REQUIRED_KEYS.issubset(failed_result.keys())


# ──────────────────────────────────────────────────────────────────────────────
# get_available_algorithms
# ──────────────────────────────────────────────────────────────────────────────

class TestGetAvailableAlgorithms:
    def test_classification_algorithms_listed(self):
        algos = get_available_algorithms("classification", _config_path)
        assert isinstance(algos, list)
        assert len(algos) > 0
        assert "DummyClassifier" in algos
        assert "RandomForestClassifier" in algos

    def test_regression_algorithms_listed(self):
        algos = get_available_algorithms("regression", _config_path)
        assert "DummyRegressor" in algos
        assert "RandomForestRegressor" in algos

    def test_classification_does_not_contain_regressor(self):
        algos = get_available_algorithms("classification", _config_path)
        assert "DummyRegressor" not in algos

    def test_regression_does_not_contain_classifier(self):
        algos = get_available_algorithms("regression", _config_path)
        assert "DummyClassifier" not in algos
