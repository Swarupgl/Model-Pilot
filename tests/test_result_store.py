"""
tests/test_result_store.py

Tests for src/modelpilot/experiments/result_store.py

Covers:
- Saving an experiment creates a JSON file
- The saved file contains all required top-level fields
- Experiment ID is unique across two saves
- load_experiment round-trips correctly
- list_experiments discovers saved files
- best_algorithm is correctly identified
- Output directory is created if it does not exist
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelpilot.experiments.result_store import (
    list_experiments,
    load_experiment,
    save_experiment,
)

# Shared mock evaluation results for tests
MOCK_RESULTS = [
    {
        "algorithm": "DummyClassifier",
        "task_type": "classification",
        "cv_folds": 5,
        "primary_metric": "balanced_accuracy",
        "primary_score": 0.33,
        "primary_std": 0.01,
        "secondary_metrics": {"accuracy": 0.35},
        "cv_scores": [0.32, 0.33, 0.34, 0.33, 0.33],
        "train_time_s": 0.05,
        "status": "ok",
        "error": "",
        "params_used": {"strategy": "most_frequent"},
    },
    {
        "algorithm": "RandomForestClassifier",
        "task_type": "classification",
        "cv_folds": 5,
        "primary_metric": "balanced_accuracy",
        "primary_score": 0.96,
        "primary_std": 0.02,
        "secondary_metrics": {"accuracy": 0.97},
        "cv_scores": [0.94, 0.96, 0.97, 0.96, 0.97],
        "train_time_s": 1.23,
        "status": "ok",
        "error": "",
        "params_used": {"n_estimators": 100},
    },
]


class TestSaveExperiment:
    def test_creates_json_file(self, tmp_path):
        path = save_experiment(
            dataset_source="builtin:iris",
            task_type="classification",
            target_column="target",
            evaluation_results=MOCK_RESULTS,
            output_dir=tmp_path,
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "dir"
        assert not new_dir.exists()
        path = save_experiment(
            dataset_source="builtin:iris",
            task_type="classification",
            target_column="target",
            evaluation_results=MOCK_RESULTS,
            output_dir=new_dir,
        )
        assert path.exists()

    def test_saved_file_contains_required_fields(self, tmp_path):
        path = save_experiment(
            dataset_source="builtin:iris",
            task_type="classification",
            target_column="target",
            evaluation_results=MOCK_RESULTS,
            output_dir=tmp_path,
        )
        with open(path) as f:
            record = json.load(f)

        required = {
            "experiment_id", "timestamp_utc", "dataset_source",
            "task_type", "target_column", "config",
            "n_candidates_evaluated", "results", "best_algorithm",
        }
        assert required.issubset(record.keys())

    def test_experiment_id_is_unique(self, tmp_path):
        p1 = save_experiment("builtin:iris", "classification", "target",
                             MOCK_RESULTS, output_dir=tmp_path)
        p2 = save_experiment("builtin:iris", "classification", "target",
                             MOCK_RESULTS, output_dir=tmp_path)
        r1 = json.loads(p1.read_text())
        r2 = json.loads(p2.read_text())
        assert r1["experiment_id"] != r2["experiment_id"]

    def test_best_algorithm_is_highest_scorer(self, tmp_path):
        path = save_experiment("builtin:iris", "classification", "target",
                               MOCK_RESULTS, output_dir=tmp_path)
        record = json.loads(path.read_text())
        assert record["best_algorithm"]["algorithm"] == "RandomForestClassifier"
        assert abs(record["best_algorithm"]["primary_score"] - 0.96) < 1e-6

    def test_all_failed_results_best_algorithm_is_none(self, tmp_path):
        failed = [{**MOCK_RESULTS[0], "status": "failed", "primary_score": float("nan")}]
        path = save_experiment("builtin:iris", "classification", "target",
                               failed, output_dir=tmp_path)
        record = json.loads(path.read_text())
        assert record["best_algorithm"] is None

    def test_n_candidates_evaluated_matches_results_length(self, tmp_path):
        path = save_experiment("builtin:iris", "classification", "target",
                               MOCK_RESULTS, output_dir=tmp_path)
        record = json.loads(path.read_text())
        assert record["n_candidates_evaluated"] == len(MOCK_RESULTS)

    def test_config_is_stored(self, tmp_path):
        config = {"cv_folds": 5, "random_seed": 42}
        path = save_experiment("builtin:iris", "classification", "target",
                               MOCK_RESULTS, experiment_config=config,
                               output_dir=tmp_path)
        record = json.loads(path.read_text())
        assert record["config"]["cv_folds"] == 5


class TestLoadExperiment:
    def test_round_trip(self, tmp_path):
        path = save_experiment("builtin:wine", "classification", "target",
                               MOCK_RESULTS, output_dir=tmp_path)
        record = load_experiment(path)
        assert record["dataset_source"] == "builtin:wine"
        assert len(record["results"]) == 2


class TestListExperiments:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        result = list_experiments(tmp_path)
        assert result == []

    def test_nonexistent_dir_returns_empty_list(self, tmp_path):
        result = list_experiments(tmp_path / "does_not_exist")
        assert result == []

    def test_discovers_saved_files(self, tmp_path):
        save_experiment("builtin:iris", "classification", "target",
                        MOCK_RESULTS, output_dir=tmp_path)
        save_experiment("builtin:wine", "classification", "target",
                        MOCK_RESULTS, output_dir=tmp_path)
        found = list_experiments(tmp_path)
        assert len(found) == 2
        assert all(p.suffix == ".json" for p in found)
