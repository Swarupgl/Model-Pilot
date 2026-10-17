"""Selective model evaluation and budget optimizer for ModelPilot.

Prioritizes candidates using uncertainty-aware acquisition (Upper Confidence Bound),
enforces evaluation budgets (max candidates / time limits), and benchmarks compute savings.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from modelpilot.evaluation.model_evaluator import evaluate_candidate, EvaluationResult

logger = logging.getLogger(__name__)


class SelectiveEvaluator:
    """Orchestrates budget-aware, uncertainty-guided candidate evaluation."""

    def __init__(
        self,
        max_candidates: int = 3,
        max_time_seconds: Optional[float] = None,
        exploration_weight: float = 1.0,
    ):
        """Initialize selective evaluator.

        Args:
            max_candidates: Maximum number of algorithms to evaluate.
            max_time_seconds: Optional global wall-clock budget in seconds.
            exploration_weight: Beta parameter in UCB acquisition score:
                acquisition = predicted_score + beta * uncertainty_std.
        """
        self.max_candidates = max_candidates
        self.max_time_seconds = max_time_seconds
        self.exploration_weight = exploration_weight

    def prioritize_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Order candidate algorithms according to Upper Confidence Bound (UCB).

        Args:
            candidates: List of candidate prediction dicts containing 'algorithm',
                'predicted_score' (or 'heuristic_score'), and optionally 'uncertainty_std'.

        Returns:
            Prioritized candidate list with computed acquisition scores.
        """
        prioritized = []
        for cand in candidates:
            c = dict(cand)
            mu = c.get("predicted_score")
            if mu is None:
                # Normalize heuristic score to [0, 1] range if present
                mu = c.get("heuristic_score", 50.0) / 100.0

            sigma = c.get("uncertainty_std", 0.10)
            ucb = float(mu + self.exploration_weight * sigma)
            c["acquisition_score"] = round(ucb, 4)
            prioritized.append(c)

        prioritized.sort(key=lambda x: x["acquisition_score"], reverse=True)
        return prioritized

    def run_selective_evaluation(
        self,
        prioritized_candidates: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int = 5,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute evaluation only on top budgeted candidates and record savings.

        Args:
            prioritized_candidates: Prioritized candidates from prioritize_candidates().
            X: Feature matrix.
            y: Target vector.
            task_type: 'classification' or 'regression'.
            cv_folds: Number of CV folds.
            random_seed: Random seed.

        Returns:
            Dictionary with selected evaluations, best candidate found,
            wall-clock time, and computational savings analysis.
        """
        selected_candidates = prioritized_candidates[: self.max_candidates]
        evaluations: List[EvaluationResult] = []
        start_time = time.perf_counter()
        stopped_early_due_to_time = False

        for cand in selected_candidates:
            if self.max_time_seconds:
                elapsed = time.perf_counter() - start_time
                if elapsed >= self.max_time_seconds:
                    stopped_early_due_to_time = True
                    logger.warning("Time budget exceeded (%.2fs >= %.2fs). Stopping evaluation.", elapsed, self.max_time_seconds)
                    break

            algo_name = cand["algorithm"]
            res = evaluate_candidate(
                algorithm_name=algo_name,
                X=X,
                y=y,
                task_type=task_type,
                cv_folds=cv_folds,
                random_seed=random_seed,
            )
            # Annotate with prior prediction information
            res["prior_predicted_score"] = cand.get("predicted_score")
            res["prior_uncertainty"] = cand.get("uncertainty_std")
            res["acquisition_score"] = cand.get("acquisition_score")
            evaluations.append(res)

        total_elapsed = round(time.perf_counter() - start_time, 3)

        successful = [e for e in evaluations if e["status"] == "ok"]
        best_candidate = (
            max(successful, key=lambda e: e.get("primary_score", float("-inf")))
            if successful
            else None
        )

        n_skipped = max(0, len(prioritized_candidates) - len(evaluations))
        estimated_savings_ratio = round(float(n_skipped / max(len(prioritized_candidates), 1)), 3)

        return {
            "evaluations": evaluations,
            "best_candidate": best_candidate,
            "total_elapsed_s": total_elapsed,
            "candidates_evaluated": len(evaluations),
            "candidates_skipped": n_skipped,
            "estimated_compute_saved_ratio": estimated_savings_ratio,
            "stopped_early_due_to_time": stopped_early_due_to_time,
        }
