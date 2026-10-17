"""Tests for algorithm selection and recommendation."""

import pytest
from modelpilot.data.loader import load_builtin
from modelpilot.diagnosis.dataset_diagnostics import diagnose_dataset
from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features
from modelpilot.selection.algorithm_selector import (
    rank_algorithms_heuristic,
    recommend_algorithms_from_history,
)


def test_heuristic_ranking_classification():
    ds = load_builtin("iris")
    diag = diagnose_dataset(ds["df"], target_column=ds["target"], task_type="classification")
    rankings = rank_algorithms_heuristic("classification", diag)

    assert len(rankings) > 0
    # Top algorithm should have a high score
    assert rankings[0]["heuristic_score"] >= rankings[-1]["heuristic_score"]
    # Check that rationales are included
    assert len(rankings[0]["rationale"]) > 0
    # Baseline dummy classifier should be at the lower tier
    dummy_item = next(r for r in rankings if r["algorithm"] == "DummyClassifier")
    assert dummy_item["heuristic_score"] < rankings[0]["heuristic_score"]


def test_heuristic_ranking_regression():
    ds = load_builtin("diabetes")
    diag = diagnose_dataset(ds["df"], target_column=ds["target"], task_type="regression")
    rankings = rank_algorithms_heuristic("regression", diag)

    assert len(rankings) > 0
    assert any("ensemble" in r["tags"] or "linear" in r["tags"] for r in rankings[:3])


def test_historical_recommendation_empty_history(tmp_path):
    store = ExperimentStore(tmp_path / "empty.db")
    meta = {"vector": [1.0] * 16}
    rec = recommend_algorithms_from_history(meta, "classification", store)
    assert rec is None


def test_historical_recommendation_with_history(tmp_path):
    store = ExperimentStore(tmp_path / "populated.db")
    ds_iris = load_builtin("iris")
    meta_iris = extract_meta_features(ds_iris["df"], ds_iris["target"], "classification")

    evals = [
        {"algorithm": "RandomForestClassifier", "primary_metric": "balanced_accuracy", "primary_score": 0.98, "status": "ok"},
        {"algorithm": "LogisticRegression", "primary_metric": "balanced_accuracy", "primary_score": 0.94, "status": "ok"},
        {"algorithm": "DummyClassifier", "primary_metric": "balanced_accuracy", "primary_score": 0.33, "status": "ok"},
    ]
    store.record_experiment(
        experiment_id="hist-1",
        timestamp_utc="2026-10-17T10:00:00Z",
        dataset_name="iris",
        dataset_fingerprint=meta_iris["dataset_fingerprint"],
        task_type="classification",
        target_column="target",
        n_rows=150,
        n_features=4,
        meta_features=meta_iris,
        config={},
        evaluations=evals,
    )

    recs = recommend_algorithms_from_history(meta_iris, "classification", store)
    assert recs is not None
    assert len(recs) == 3
    assert recs[0]["algorithm"] == "RandomForestClassifier"
    assert recs[0]["historical_weighted_score"] >= recs[1]["historical_weighted_score"]
    assert recs[0]["evidence_count"] == 1
