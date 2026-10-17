"""Meta-learning performance predictor for ModelPilot.

Trains predictive models over dataset meta-features and algorithm identities
to estimate expected validation performance, quantify prediction uncertainty,
and detect out-of-distribution datasets before full model training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import STANDARD_FEATURE_NAMES

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """Supervised meta-learning predictor with tree-variance uncertainty estimation."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.model: Optional[RandomForestRegressor] = None
        self.algorithms_: List[str] = []
        self.train_meta_vectors_: Optional[np.ndarray] = None
        self.train_records_count_: int = 0
        self.global_algo_means_: Dict[str, float] = {}
        self.ood_distance_threshold_: float = 2.5

    def fit(self, meta_dataset: List[Dict[str, Any]]) -> "PerformancePredictor":
        """Train the meta-learning performance model on historical experiment records.

        Args:
            meta_dataset: List of records from ExperimentStore.get_meta_dataset().
        """
        if not meta_dataset:
            raise ValueError("Meta-dataset is empty. Cannot fit performance predictor.")

        self.train_records_count_ = len(meta_dataset)

        # Extract unique algorithms and compute historical mean baselines
        algos = sorted(list({r["algorithm"] for r in meta_dataset}))
        self.algorithms_ = algos

        algo_scores: Dict[str, List[float]] = {a: [] for a in algos}
        for r in meta_dataset:
            algo_scores[r["algorithm"]].append(r["primary_score"])
        self.global_algo_means_ = {
            a: float(np.mean(scores)) if scores else 0.5 for a, scores in algo_scores.items()
        }

        # Build feature matrix: [meta_features_vector, one_hot_encoded_algorithm]
        X_rows = []
        y_rows = []
        meta_vecs = []

        for r in meta_dataset:
            vec = r["meta_features"].get("vector", [])
            if len(vec) != len(STANDARD_FEATURE_NAMES):
                continue

            meta_vecs.append(vec)
            algo = r["algorithm"]
            one_hot = [1.0 if algo == a else 0.0 for a in self.algorithms_]
            feature_row = list(vec) + one_hot
            X_rows.append(feature_row)
            y_rows.append(r["primary_score"])

        if not X_rows:
            raise ValueError("No valid meta-feature vectors found in meta-dataset.")

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=float)

        self.train_meta_vectors_ = np.array(meta_vecs, dtype=float)

        # Compute empirical OOD threshold based on 95th percentile of pairwise distances
        if len(self.train_meta_vectors_) > 2:
            dists = []
            for i in range(len(self.train_meta_vectors_)):
                for j in range(i + 1, len(self.train_meta_vectors_)):
                    dists.append(np.linalg.norm(self.train_meta_vectors_[i] - self.train_meta_vectors_[j]))
            if dists:
                self.ood_distance_threshold_ = float(np.percentile(dists, 95)) * 1.5

        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=self.random_seed,
            max_depth=6,
            min_samples_leaf=1,
        )
        self.model.fit(X, y)
        logger.info("Fitted meta-predictor on %d evaluations across %d algorithms.", len(X), len(algos))
        return self

    def predict(
        self,
        meta_features: Dict[str, Any],
        candidate_algorithms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Predict expected performance and uncertainty for candidate algorithms.

        Args:
            meta_features: Query dataset meta-features dictionary.
            candidate_algorithms: List of candidate names to evaluate. Defaults to
                all known algorithms.

        Returns:
            List of prediction dictionaries sorted by predicted score descending.
        """
        if self.model is None or not self.algorithms_:
            raise RuntimeError("PerformancePredictor must be fitted before predict().")

        query_vec = np.array(meta_features.get("vector", []), dtype=float)
        if len(query_vec) != len(STANDARD_FEATURE_NAMES):
            raise ValueError(
                f"Query vector length ({len(query_vec)}) != expected ({len(STANDARD_FEATURE_NAMES)})."
            )

        # Check for Out-of-Distribution (OOD)
        min_distance = float("inf")
        if self.train_meta_vectors_ is not None and len(self.train_meta_vectors_) > 0:
            diffs = self.train_meta_vectors_ - query_vec
            distances = np.linalg.norm(diffs, axis=1)
            min_distance = float(np.min(distances))

        is_ood = min_distance > self.ood_distance_threshold_

        candidates = candidate_algorithms or self.algorithms_
        results = []

        for algo in candidates:
            # Baseline mean
            hist_mean = self.global_algo_means_.get(algo, 0.5)

            if algo in self.algorithms_:
                one_hot = [1.0 if algo == a else 0.0 for a in self.algorithms_]
                x_sample = np.array([list(query_vec) + one_hot], dtype=float)

                # Predictions from all individual decision trees in the forest
                tree_preds = np.array([tree.predict(x_sample)[0] for tree in self.model.estimators_])
                pred_score = float(np.mean(tree_preds))
                pred_std = float(np.std(tree_preds))
                ci_lower = max(0.0, pred_score - 1.96 * pred_std)
                ci_upper = min(1.0, pred_score + 1.96 * pred_std)
            else:
                pred_score = hist_mean
                pred_std = 0.25
                ci_lower = max(0.0, pred_score - 0.5)
                ci_upper = min(1.0, pred_score + 0.5)

            # Confidence level reflects distance & ensemble agreement
            uncertainty_penalty = min(1.0, (min_distance / self.ood_distance_threshold_) * 0.4)
            confidence = max(0.0, round(float(1.0 - pred_std - uncertainty_penalty), 3))

            results.append({
                "algorithm": algo,
                "predicted_score": round(pred_score, 4),
                "uncertainty_std": round(pred_std, 4),
                "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
                "confidence": confidence,
                "historical_mean_baseline": round(hist_mean, 4),
                "is_out_of_distribution": is_ood,
                "min_meta_distance": round(min_distance, 3),
            })

        results.sort(key=lambda x: x["predicted_score"], reverse=True)
        return results


def evaluate_meta_predictor_loo(
    meta_dataset: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Perform Leave-One-Dataset-Out cross-validation on historical records.

    Ensures zero cross-dataset leakage: a dataset is strictly in the meta-test
    fold or meta-train fold, never both.

    Returns:
        Evaluation metrics: MAE, mean Spearman rank correlation, top-1 regret.
    """
    fingerprints = sorted(list({r["dataset_fingerprint"] for r in meta_dataset}))
    if len(fingerprints) < 2:
        return {
            "error": "At least 2 distinct datasets required for Leave-One-Dataset-Out evaluation."
        }

    maes = []
    spearmans = []
    regrets = []

    for test_fp in fingerprints:
        train_data = [r for r in meta_dataset if r["dataset_fingerprint"] != test_fp]
        test_data = [r for r in meta_dataset if r["dataset_fingerprint"] == test_fp]

        if not train_data or not test_data:
            continue

        predictor = PerformancePredictor()
        try:
            predictor.fit(train_data)
        except Exception:
            continue

        test_meta = test_data[0]["meta_features"]
        algos = [r["algorithm"] for r in test_data]
        actual_scores = {r["algorithm"]: r["primary_score"] for r in test_data}

        preds = predictor.predict(test_meta, candidate_algorithms=algos)
        pred_map = {p["algorithm"]: p["predicted_score"] for p in preds}

        y_true = [actual_scores[a] for a in algos]
        y_pred = [pred_map.get(a, 0.5) for a in algos]

        # MAE
        maes.append(mean_absolute_error(y_true, y_pred))

        # Spearman rank correlation
        if len(y_true) >= 3:
            rho, _ = spearmanr(y_true, y_pred)
            if not np.isnan(rho):
                spearmans.append(float(rho))

        # Top-1 Regret: actual score of best overall vs. actual score of predicted #1
        best_actual_score = max(y_true)
        top1_predicted_algo = preds[0]["algorithm"]
        top1_actual_score = actual_scores[top1_predicted_algo]
        regret = best_actual_score - top1_actual_score
        regrets.append(max(0.0, regret))

    return {
        "n_datasets_evaluated": len(fingerprints),
        "mean_absolute_error": round(float(np.mean(maes)), 4) if maes else 0.0,
        "mean_spearman_rank_correlation": round(float(np.mean(spearmans)), 4) if spearmans else 0.0,
        "mean_top1_selection_regret": round(float(np.mean(regrets)), 4) if regrets else 0.0,
    }
