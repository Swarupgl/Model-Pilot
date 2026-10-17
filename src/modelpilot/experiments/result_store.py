"""
src/modelpilot/experiments/result_store.py

Phase 1 experiment persistence — saves structured experiment records as JSON files.

Phase 3 will replace this with a SQLite database (experiment_store.py).
This module remains as a lightweight fallback and for easy inspection of results.

Design decisions:
- Each experiment run is saved as one JSON file in outputs/experiments/.
- Files are named with a timestamp + dataset name so they are human-readable.
- We do NOT aggregate or query here — that is the job of the database layer.
- The schema is kept compatible with the planned SQLite schema in Phase 3.
"""

from __future__ import annotations

import itertools
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Monotonic counter ensures unique IDs even when uuid4() is called in rapid
# succession within the same process (observed in Python 3.14 test runs).
_id_counter = itertools.count()


def _unique_experiment_id() -> str:
    """Return a standard RFC 4122 random UUID string."""
    return str(uuid.uuid4())

logger = logging.getLogger(__name__)


# Public API

def save_experiment(
    dataset_source: str,
    task_type: str,
    target_column: str,
    evaluation_results: list[dict],
    experiment_config: Optional[dict] = None,
    output_dir: str | Path = "outputs/experiments",
) -> Path:
    """
    Save a complete experiment run (one dataset, multiple algorithm results) to JSON.

    Parameters
    ----------
    dataset_source : str
        Identifier for the dataset (e.g. "builtin:iris" or "csv:my_data.csv").
    task_type : str
        "classification" or "regression".
    target_column : str
        Name of the target column.
    evaluation_results : list[dict]
        List of EvaluationResult dicts from model_evaluator.evaluate_multiple().
    experiment_config : dict, optional
        Configuration used for this run (cv_folds, random_seed, etc.).
    output_dir : str or Path
        Directory where the JSON file will be saved. Created if not present.

    Returns
    -------
    Path — the saved JSON file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = _unique_experiment_id()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

# Derive a safe filename
    dataset_slug = dataset_source.replace(":", "_").replace("/", "_").replace("\\", "_")
    ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = experiment_id[:8]
    filename = f"{ts_compact}_{short_id}_{dataset_slug}.json"
    file_path = output_dir / filename

# Build the experiment record
    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "dataset_source": dataset_source,
        "task_type": task_type,
        "target_column": target_column,
        "config": experiment_config or {},
        "n_candidates_evaluated": len(evaluation_results),
        "results": evaluation_results,
        "best_algorithm": _find_best(evaluation_results),
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=_json_serializer)

    logger.info("Experiment saved: %s", file_path)
    return file_path


def load_experiment(file_path: str | Path) -> dict:
    """
    Load a previously saved experiment JSON file.

    Parameters
    ----------
    file_path : str or Path
        Path to the JSON file produced by save_experiment().

    Returns
    -------
    Experiment record dict.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments(output_dir: str | Path = "outputs/experiments") -> list[Path]:
    """
    Return a sorted list of all saved experiment JSON files in output_dir.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.json"))


# Internal helpers

def _find_best(results: list[dict]) -> Optional[dict]:
    """Return the best successful result by primary_score."""
    successful = [r for r in results if r.get("status") == "ok"]
    if not successful:
        return None
    best = max(successful, key=lambda r: r.get("primary_score", float("-inf")))
    return {
        "algorithm": best["algorithm"],
        "primary_metric": best["primary_metric"],
        "primary_score": best["primary_score"],
        "primary_std": best["primary_std"],
    }


def _json_serializer(obj: Any) -> Any:
    """Handle types that json.dump cannot serialise by default."""
    if hasattr(obj, "item"):
        # numpy scalar → Python scalar
        return obj.item()
    if hasattr(obj, "tolist"):
        # numpy array → Python list
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
