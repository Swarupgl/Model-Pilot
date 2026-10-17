"""Neuro-SAN coded tool adapter for ModelPilot algorithm recommendation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Union
from neuro_san.interfaces.coded_tool import CodedTool

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.diagnosis.dataset_diagnostics import diagnose_dataset
from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features
from modelpilot.selection.algorithm_selector import (
    rank_algorithms_heuristic,
    recommend_algorithms_from_history,
)

logger = logging.getLogger(__name__)


class AlgorithmRecommendationTool(CodedTool):
    """Recommends candidate algorithms via transparent heuristic ranking and historical similarity."""

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """Generate algorithm recommendations.

        Args:
            args: Dictionary with keys:
                - dataset_name: Built-in dataset name (default 'iris')
                - csv_path: Path to CSV dataset (optional)
                - target_column: Name of target column (optional)
            sly_data: Neuro-SAN session context (unused).

        Returns:
            Ranked candidate algorithms with explicit rationale and historical evidence.
        """
        try:
            csv_path = args.get("csv_path")
            dataset_name = args.get("dataset_name", "iris")
            target_col = args.get("target_column")

            if csv_path:
                ds = load_csv(csv_path, target_column=target_col)
            else:
                ds = load_builtin(dataset_name, target_column=target_col)

            diag = diagnose_dataset(ds["df"], ds["target"], ds["task_type"])
            meta = extract_meta_features(ds["df"], ds["target"], ds["task_type"])

            # 1. Transparent heuristic ranking
            heuristic_ranks = rank_algorithms_heuristic(ds["task_type"], diag)

            # 2. Historical similarity recommendation if stored history exists
            store = ExperimentStore()
            historical_recs = recommend_algorithms_from_history(
                target_meta_features=meta,
                task_type=ds["task_type"],
                store=store,
            )

            return {
                "status": "success",
                "dataset_source": ds["source"],
                "task_type": ds["task_type"],
                "heuristic_recommendations": heuristic_ranks[:5],
                "historical_recommendations": historical_recs[:5] if historical_recs else None,
                "has_historical_evidence": historical_recs is not None,
            }
        except Exception as exc:
            logger.error("AlgorithmRecommendationTool error: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}
