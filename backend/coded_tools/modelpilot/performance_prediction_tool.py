"""Neuro-SAN coded tool adapter for ModelPilot meta-learning performance prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from neuro_san.interfaces.coded_tool import CodedTool

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features
from modelpilot.meta_learning.performance_predictor import PerformancePredictor

logger = logging.getLogger(__name__)


class PerformancePredictionTool(CodedTool):
    """Predicts candidate validation scores, uncertainty intervals, and OOD status."""

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """Execute performance prediction.

        Args:
            args: Dictionary with keys:
                - dataset_name: Built-in dataset name (default 'iris')
                - csv_path: Path to CSV dataset (optional)
                - target_column: Name of target column (optional)
            sly_data: Neuro-SAN session context (unused).

        Returns:
            Dictionary containing predicted scores, uncertainty bounds, and OOD alerts.
        """
        try:
            csv_path = args.get("csv_path")
            dataset_name = args.get("dataset_name", "iris")
            target_col = args.get("target_column")

            if csv_path:
                ds = load_csv(csv_path, target_column=target_col)
            else:
                ds = load_builtin(dataset_name, target_column=target_col)

            meta = extract_meta_features(ds["df"], ds["target"], ds["task_type"])

            store = ExperimentStore()
            historical_data = store.get_meta_dataset(task_type=ds["task_type"])

            if not historical_data or len(historical_data) < 3:
                return {
                    "status": "insufficient_history",
                    "message": (
                        "Fewer than 3 historical experiments recorded. "
                        "Evaluate models first or rely on heuristic recommendations."
                    ),
                    "dataset_source": ds["source"],
                    "task_type": ds["task_type"],
                }

            predictor = PerformancePredictor()
            predictor.fit(historical_data)
            predictions = predictor.predict(meta)

            return {
                "status": "success",
                "dataset_source": ds["source"],
                "task_type": ds["task_type"],
                "predictions": predictions,
                "is_out_of_distribution": predictions[0]["is_out_of_distribution"] if predictions else False,
            }
        except Exception as exc:
            logger.error("PerformancePredictionTool error: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}
