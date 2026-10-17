"""Tests for the SQLite experiment store."""

import pytest
from modelpilot.experiments.experiment_store import ExperimentStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_history.db"
    return ExperimentStore(db_path=db_file)


def test_record_and_retrieve_experiment(store):
    evals = [
        {
            "algorithm": "LogisticRegression",
            "params_used": {"max_iter": 500},
            "primary_metric": "balanced_accuracy",
            "primary_score": 0.95,
            "primary_std": 0.02,
            "cv_scores": [0.93, 0.97],
            "secondary_metrics": {"accuracy": 0.96},
            "train_time_s": 0.12,
            "status": "ok",
            "error": "",
        },
        {
            "algorithm": "DummyClassifier",
            "params_used": {},
            "primary_metric": "balanced_accuracy",
            "primary_score": 0.33,
            "primary_std": 0.0,
            "cv_scores": [0.33, 0.33],
            "secondary_metrics": {"accuracy": 0.33},
            "train_time_s": 0.01,
            "status": "ok",
            "error": "",
        },
    ]

    store.record_experiment(
        experiment_id="exp-001",
        timestamp_utc="2026-10-17T12:00:00Z",
        dataset_name="test_iris",
        dataset_fingerprint="fp12345",
        task_type="classification",
        target_column="target",
        n_rows=150,
        n_features=4,
        meta_features={"log_n_rows": 2.176, "vector": [2.176] * 16},
        config={"cv_folds": 5},
        evaluations=evals,
    )

    all_exps = store.get_all_experiments()
    assert len(all_exps) == 1
    assert all_exps[0]["experiment_id"] == "exp-001"
    assert all_exps[0]["dataset_name"] == "test_iris"

    lr_evals = store.get_evaluations_for_algorithm("LogisticRegression")
    assert len(lr_evals) == 1
    assert lr_evals[0]["primary_score"] == 0.95

    fp_evals = store.get_dataset_experiments("fp12345")
    assert len(fp_evals) == 2
    assert fp_evals[0]["algorithm"] == "LogisticRegression"

    meta_ds = store.get_meta_dataset("classification")
    assert len(meta_ds) == 2
    assert meta_ds[0]["dataset_name"] == "test_iris"
