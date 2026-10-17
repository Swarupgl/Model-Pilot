"""Neuro-SAN coded tool adapter for ModelPilot dataset diagnosis."""

from __future__ import annotations

import logging
from typing import Any, Dict, Union
from neuro_san.interfaces.coded_tool import CodedTool

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.diagnosis.dataset_diagnostics import diagnose_dataset
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features

logger = logging.getLogger(__name__)


class DatasetDiagnosticsTool(CodedTool):
    """Diagnoses dataset dimensions, missingness, class imbalance, and data health risks."""

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """Execute dataset diagnostics.

        Args:
            args: Dictionary with keys:
                - dataset_name: Built-in dataset name ('iris', 'wine', 'diabetes', etc.)
                - csv_path: Path to CSV dataset (optional)
                - target_column: Name of target column (optional)
            sly_data: Neuro-SAN session context (unused).

        Returns:
            Structured diagnostics report including quality flags and advisories.
        """
        try:
            csv_path = args.get("csv_path")
            dataset_name = args.get("dataset_name", "iris")
            target_col = args.get("target_column")

            if csv_path:
                ds = load_csv(csv_path, target_column=target_col)
            else:
                ds = load_builtin(dataset_name, target_column=target_col)

            diag = diagnose_dataset(
                df=ds["df"],
                target_column=ds["target"],
                task_type=ds["task_type"],
            )

            meta = extract_meta_features(
                df=ds["df"],
                target_column=ds["target"],
                task_type=ds["task_type"],
            )

            return {
                "status": "success",
                "dataset_source": ds["source"],
                "target_column": ds["target"],
                "task_type": ds["task_type"],
                "dataset_shape": diag["dataset_shape"],
                "missing_values": diag["missing_values"],
                "duplicate_rows": diag["duplicate_rows"],
                "target_diagnostics": diag["target_diagnostics"],
                "advisories": diag["advisories"],
                "dataset_fingerprint": meta["dataset_fingerprint"],
            }
        except Exception as exc:
            logger.error("DatasetDiagnosticsTool error: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}
