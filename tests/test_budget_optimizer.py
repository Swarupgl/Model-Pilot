"""Tests for selective evaluation and budget optimization."""

import pytest
from modelpilot.data.loader import load_builtin
from modelpilot.optimization.budget_optimizer import SelectiveEvaluator


def test_prioritize_candidates_ucb():
    evaluator = SelectiveEvaluator(max_candidates=2, exploration_weight=1.5)
    candidates = [
        {"algorithm": "AlgoA", "predicted_score": 0.85, "uncertainty_std": 0.02},  # UCB = 0.85 + 1.5*0.02 = 0.88
        {"algorithm": "AlgoB", "predicted_score": 0.80, "uncertainty_std": 0.10},  # UCB = 0.80 + 1.5*0.10 = 0.95
        {"algorithm": "AlgoC", "predicted_score": 0.70, "uncertainty_std": 0.05},  # UCB = 0.70 + 1.5*0.05 = 0.775
    ]

    ranked = evaluator.prioritize_candidates(candidates)
    # AlgoB has higher exploration potential (UCB=0.95) than AlgoA (UCB=0.88)
    assert ranked[0]["algorithm"] == "AlgoB"
    assert ranked[1]["algorithm"] == "AlgoA"
    assert ranked[2]["algorithm"] == "AlgoC"


def test_selective_evaluation_budget_limit():
    ds = load_builtin("iris")
    evaluator = SelectiveEvaluator(max_candidates=2)

    candidates = [
        {"algorithm": "KNeighborsClassifier", "predicted_score": 0.95, "uncertainty_std": 0.02},
        {"algorithm": "LogisticRegression", "predicted_score": 0.90, "uncertainty_std": 0.02},
        {"algorithm": "DecisionTreeClassifier", "predicted_score": 0.85, "uncertainty_std": 0.05},
        {"algorithm": "DummyClassifier", "predicted_score": 0.33, "uncertainty_std": 0.01},
    ]

    prioritized = evaluator.prioritize_candidates(candidates)
    res = evaluator.run_selective_evaluation(
        prioritized_candidates=prioritized,
        X=ds["X"],
        y=ds["y"],
        task_type="classification",
        cv_folds=3,
    )

    assert res["candidates_evaluated"] == 2
    assert res["candidates_skipped"] == 2
    assert res["estimated_compute_saved_ratio"] == 0.5
    assert res["best_candidate"] is not None
    assert res["best_candidate"]["primary_score"] > 0.8
