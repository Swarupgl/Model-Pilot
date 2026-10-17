"""Tests for the meta-learning performance predictor."""

import pytest
import numpy as np

from modelpilot.meta_learning.performance_predictor import (
    PerformancePredictor,
    evaluate_meta_predictor_loo,
)


@pytest.fixture
def mock_meta_dataset():
    # 3 distinct datasets, each evaluated across 3 algorithms
    records = []
    base_vectors = {
        "ds_A": [2.0, 1.0, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.2, 0.1, 0.0, 0.3, 0.5],
        "ds_B": [2.5, 1.2, 0.08, 0.8, 0.2, 0.05, 0.0, 0.1, 3.0, 1.5, 0.9, 0.4, 0.2, 0.1, 0.4, 0.6],
        "ds_C": [3.0, 1.5, 0.10, 0.5, 0.5, 0.10, 0.0, 0.2, 4.0, 2.0, 0.8, 0.5, 0.3, 0.2, 0.5, 0.7],
    }
    scores = {
        "ds_A": {"RandomForestClassifier": 0.96, "LogisticRegression": 0.90, "DummyClassifier": 0.50},
        "ds_B": {"RandomForestClassifier": 0.92, "LogisticRegression": 0.85, "DummyClassifier": 0.40},
        "ds_C": {"RandomForestClassifier": 0.88, "LogisticRegression": 0.82, "DummyClassifier": 0.35},
    }

    for ds_name, vec in base_vectors.items():
        for algo, score in scores[ds_name].items():
            records.append({
                "dataset_name": ds_name,
                "dataset_fingerprint": f"fp_{ds_name}",
                "meta_features": {"vector": vec},
                "algorithm": algo,
                "primary_score": score,
            })
    return records


def test_predictor_fit_and_predict(mock_meta_dataset):
    predictor = PerformancePredictor(random_seed=42)
    predictor.fit(mock_meta_dataset)

    query_meta = {"vector": [2.2, 1.1, 0.06, 0.9, 0.1, 0.02, 0.0, 0.05, 2.0, 1.2, 0.95, 0.3, 0.15, 0.05, 0.35, 0.55]}
    predictions = predictor.predict(query_meta)

    assert len(predictions) == 3
    # RandomForest should be predicted highest based on training data
    assert predictions[0]["algorithm"] == "RandomForestClassifier"
    assert predictions[0]["predicted_score"] > predictions[-1]["predicted_score"]
    assert "uncertainty_std" in predictions[0]
    assert "confidence_interval_95" in predictions[0]
    assert predictions[0]["confidence"] > 0.0
    assert not predictions[0]["is_out_of_distribution"]


def test_predictor_ood_detection(mock_meta_dataset):
    predictor = PerformancePredictor(random_seed=42)
    predictor.fit(mock_meta_dataset)

    # Completely alien vector with huge values
    alien_meta = {"vector": [100.0] * 16}
    predictions = predictor.predict(alien_meta)

    assert predictions[0]["is_out_of_distribution"] is True
    # Confidence should be penalized for extreme distance
    assert predictions[0]["confidence"] < 0.6


def test_leave_one_dataset_out_evaluation(mock_meta_dataset):
    metrics = evaluate_meta_predictor_loo(mock_meta_dataset)

    assert metrics["n_datasets_evaluated"] == 3
    assert metrics["mean_absolute_error"] < 0.15
    assert metrics["mean_top1_selection_regret"] >= 0.0
    assert "mean_spearman_rank_correlation" in metrics
