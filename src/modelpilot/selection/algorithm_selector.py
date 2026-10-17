"""Algorithm selector module for ModelPilot.

Provides transparent heuristic ranking and historical similarity-based
recommendations for machine learning candidate algorithms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import STANDARD_FEATURE_NAMES

logger = logging.getLogger(__name__)


def _load_available_algorithms(task_type: str, config_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "algorithms.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get(task_type, [])


def rank_algorithms_heuristic(
    task_type: str,
    diagnostics: Dict[str, Any],
    algorithms_config_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Generate transparent, rule-based heuristic rankings for candidate algorithms.

    Args:
        task_type: 'classification' or 'regression'.
        diagnostics: Output dictionary from diagnose_dataset().
        algorithms_config_path: Path to algorithms.yaml.

    Returns:
        List of candidate dictionaries sorted by heuristic priority score,
        with detailed rationale for each score.
    """
    candidates = _load_available_algorithms(task_type, algorithms_config_path)
    n_rows = diagnostics["dataset_shape"]["n_rows"]
    n_features = diagnostics["dataset_shape"]["n_features"]
    dim_ratio = diagnostics["dataset_shape"]["dim_ratio"]
    sparsity = diagnostics["dataset_shape"]["sparsity"]
    imbalance = (
        diagnostics["target_diagnostics"].get("imbalance_ratio", 1.0)
        if task_type == "classification"
        else 1.0
    )

    ranked_candidates = []

    for algo in candidates:
        name = algo["name"]
        tags = algo.get("tags", [])
        base_score = 50.0
        reasons: List[str] = []

        if "baseline" in tags:
            base_score = 10.0
            reasons.append("Always evaluated as a reference baseline.")
        else:
            # Ensembles (RandomForest, ExtraTrees, GradientBoosting)
            if "ensemble" in tags or "boosting" in tags:
                base_score += 30.0
                reasons.append("Ensembles handle non-linear relationships and mixed data robustly.")
                if n_rows < 1000:
                    base_score += 5.0
                    reasons.append("Manageable sample size enables fast ensemble convergence.")
                if imbalance > 3.0:
                    base_score += 5.0
                    reasons.append("Ensembles typically maintain resilience under class imbalance.")

            # Linear models (LogisticRegression, Ridge, LinearRegression)
            if "linear" in tags:
                base_score += 20.0
                reasons.append("Linear models provide fast, interpretable baselines.")
                if dim_ratio > 0.10:
                    base_score += 10.0
                    reasons.append("Favorable when feature dimension is significant relative to sample count.")
                if sparsity > 0.30:
                    base_score += 5.0
                    reasons.append("Performs efficiently on sparse representations.")

            # Instance-based (KNeighbors)
            if "instance_based" in tags:
                base_score += 15.0
                if n_rows > 5000:
                    base_score -= 20.0
                    reasons.append("Penalized: distance queries scale poorly on large sample sizes.")
                elif n_features > 50:
                    base_score -= 15.0
                    reasons.append("Penalized: high dimensionality degrades Euclidean distance metrics.")
                else:
                    base_score += 10.0
                    reasons.append("Compact dataset size favors local neighborhood representations.")

            # Support Vector Machines
            if "kernel" in tags:
                base_score += 15.0
                if n_rows > 10000:
                    base_score -= 30.0
                    reasons.append("Penalized: cubic training complexity prohibitive for large sample sizes.")
                else:
                    base_score += 10.0
                    reasons.append("Effective for non-linear boundaries on small to medium datasets.")

            # Naive Bayes
            if "assumes_independence" in tags:
                base_score += 10.0
                reasons.append("Fast probabilistic estimation under conditional independence assumption.")

        ranked_candidates.append({
            "algorithm": name,
            "heuristic_score": round(base_score, 1),
            "recommendation_type": "heuristic",
            "tags": tags,
            "rationale": reasons,
        })

    ranked_candidates.sort(key=lambda x: x["heuristic_score"], reverse=True)
    return ranked_candidates


def recommend_algorithms_from_history(
    target_meta_features: Dict[str, Any],
    task_type: str,
    store: ExperimentStore,
    top_k_datasets: int = 3,
    algorithms_config_path: Optional[str | Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Recommend algorithms using historical dataset similarity.

    Args:
        target_meta_features: Extracted meta-features for the query dataset.
        task_type: 'classification' or 'regression'.
        store: SQLite experiment store containing past runs.
        top_k_datasets: Number of nearest historical datasets to consider.
        algorithms_config_path: Path to algorithms.yaml.

    Returns:
        List of recommended candidates with similarity-weighted historical scores,
        or None if insufficient history is recorded.
    """
    records = store.get_meta_dataset(task_type=task_type)
    if not records:
        logger.info("Insufficient history for similarity-based recommendation.")
        return None

    # Group historical evaluations by unique dataset fingerprint
    datasets_meta: Dict[str, Dict[str, Any]] = {}
    algo_scores_by_dataset: Dict[str, Dict[str, float]] = {}

    for r in records:
        fp = r["dataset_fingerprint"]
        if fp not in datasets_meta:
            datasets_meta[fp] = r["meta_features"]
            algo_scores_by_dataset[fp] = {}
        algo_scores_by_dataset[fp][r["algorithm"]] = r["primary_score"]

    if len(datasets_meta) < 1:
        return None

    # Compute Euclidean distance in normalized meta-feature space
    target_vec = np.array(target_meta_features.get("vector", []), dtype=float)
    if len(target_vec) != len(STANDARD_FEATURE_NAMES):
        return None

    similarities: List[Tuple[str, float]] = []
    for fp, meta in datasets_meta.items():
        hist_vec = np.array(meta.get("vector", []), dtype=float)
        if len(hist_vec) != len(target_vec):
            continue

        # Normalized Euclidean distance
        diff = target_vec - hist_vec
        denom = np.abs(target_vec) + np.abs(hist_vec) + 1e-6
        norm_diff = diff / denom
        dist = float(np.linalg.norm(norm_diff))
        sim = 1.0 / (1.0 + dist)
        similarities.append((fp, sim))

    if not similarities:
        return None

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_datasets = similarities[:top_k_datasets]

    # Aggregate performance of algorithms across nearest datasets
    all_algos = _load_available_algorithms(task_type, algorithms_config_path)
    recommendations = []

    for algo_def in all_algos:
        name = algo_def["name"]
        scores = []
        weights = []
        evidence = []

        for fp, sim in top_datasets:
            if name in algo_scores_by_dataset[fp]:
                score = algo_scores_by_dataset[fp][name]
                scores.append(score)
                weights.append(sim)
                evidence.append({"fingerprint": fp, "similarity": round(sim, 3), "score": round(score, 4)})

        if scores:
            weighted_avg = float(np.average(scores, weights=weights))
            recommendations.append({
                "algorithm": name,
                "historical_weighted_score": round(weighted_avg, 4),
                "recommendation_type": "historical_similarity",
                "evidence_count": len(scores),
                "evidence": evidence,
            })

    if not recommendations:
        return None

    recommendations.sort(key=lambda x: x["historical_weighted_score"], reverse=True)
    return recommendations
