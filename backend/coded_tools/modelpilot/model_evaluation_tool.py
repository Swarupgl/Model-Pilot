"""Neuro-SAN coded tool adapter for ModelPilot model evaluation."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from neuro_san.interfaces.coded_tool import CodedTool

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.evaluation.model_evaluator import evaluate_candidate
from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features
from modelpilot.optimization.budget_optimizer import SelectiveEvaluator

logger = logging.getLogger(__name__)


class ModelEvaluationTool(CodedTool):
    """Evaluates candidate machine learning models under cross-validation and records results."""

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """Execute model evaluation.

        Args:
            args: Dictionary with keys:
                - dataset_name: Built-in dataset name (default 'iris')
                - csv_path: Path to CSV dataset (optional)
                - target_column: Name of target column (optional)
                - algorithms: List of algorithm names (optional, defaults to top 3)
                - max_candidates: Maximum number of models to evaluate (default 3)
                - cv_folds: Cross-validation folds (default 5)
            sly_data: Neuro-SAN session context (unused).

        Returns:
            Evaluation summary with primary/secondary metrics and best model identification.
        """
        try:
            csv_path = args.get("csv_path")
            dataset_name = args.get("dataset_name", "iris")
            target_col = args.get("target_column")
            max_cand = int(args.get("max_candidates", 3))
            cv_folds = int(args.get("cv_folds", 5))

            if csv_path:
                ds = load_csv(csv_path, target_column=target_col)
            else:
                ds = load_builtin(dataset_name, target_column=target_col)

            meta = extract_meta_features(ds["df"], ds["target"], ds["task_type"])

            specified_algos = args.get("algorithms")
            if specified_algos and isinstance(specified_algos, list):
                candidates = [{"algorithm": a} for a in specified_algos]
            else:
                if ds["task_type"] == "classification":
                    candidates = [
                        {"algorithm": "DummyClassifier"},
                        {"algorithm": "LogisticRegression"},
                        {"algorithm": "RandomForestClassifier"},
                    ]
                else:
                    candidates = [
                        {"algorithm": "DummyRegressor"},
                        {"algorithm": "Ridge"},
                        {"algorithm": "RandomForestRegressor"},
                    ]

            evaluator = SelectiveEvaluator(max_candidates=max_cand)
            prioritized = evaluator.prioritize_candidates(candidates)
            run_summary = evaluator.run_selective_evaluation(
                prioritized_candidates=prioritized,
                X=ds["X"],
                y=ds["y"],
                task_type=ds["task_type"],
                cv_folds=cv_folds,
            )

            # Record in historical SQLite database
            store = ExperimentStore()
            exp_id = str(uuid.uuid4())
            ts = datetime.now(tz=timezone.utc).isoformat()
            store.record_experiment(
                experiment_id=exp_id,
                timestamp_utc=ts,
                dataset_name=dataset_name if not csv_path else Path(csv_path).stem,
                dataset_fingerprint=meta["dataset_fingerprint"],
                task_type=ds["task_type"],
                target_column=ds["target"],
                n_rows=ds["n_rows"],
                n_features=ds["n_features"],
                meta_features=meta,
                config={"cv_folds": cv_folds, "max_candidates": max_cand},
                evaluations=run_summary["evaluations"],
            )

            results_clean = []
            for e in run_summary["evaluations"]:
                results_clean.append({
                    "algorithm": e["algorithm"],
                    "primary_metric": e["primary_metric"],
                    "primary_score": e["primary_score"],
                    "primary_std": e["primary_std"],
                    "train_time_s": e["train_time_s"],
                    "status": e["status"],
                })

            best = run_summary["best_candidate"]
            best_clean = {
                "algorithm": best["algorithm"],
                "primary_metric": best["primary_metric"],
                "primary_score": best["primary_score"],
            } if best else None

            return {
                "status": "success",
                "experiment_id": exp_id,
                "dataset_source": ds["source"],
                "task_type": ds["task_type"],
                "candidates_evaluated": run_summary["candidates_evaluated"],
                "total_elapsed_s": run_summary["total_elapsed_s"],
                "evaluations": results_clean,
                "best_candidate": best_clean,
            }
        except Exception as exc:
            logger.error("ModelEvaluationTool error: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}
