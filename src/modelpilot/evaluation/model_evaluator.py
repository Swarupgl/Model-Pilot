"""
src/modelpilot/evaluation/model_evaluator.py

Implements fair, reproducible model evaluation using cross-validation and a
final holdout test set.

Design decisions:
- All preprocessing (StandardScaler, SimpleImputer) is wrapped inside a Pipeline
  so that fit() is called only on training folds, never leaking test statistics.
- The evaluator reads algorithm definitions from configs/algorithms.yaml rather
  than hard-coding class names. This keeps the algorithm list configurable.
- We return a structured EvaluationResult dict so callers (CLI, Neuro-SAN tools,
  tests) all use the same shape.
- We deliberately do NOT touch the final holdout test set during cross-validation.
  The holdout test is evaluated only once, after selection is complete.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

logger = logging.getLogger(__name__)

# Type alias — callers should not depend on the concrete dict keys directly;
# refer to the docstring of evaluate_candidate() for the guaranteed schema.
EvaluationResult = dict


# Public API

def evaluate_candidate(
    algorithm_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int = 5,
    random_seed: int = 42,
    algorithm_params: Optional[dict] = None,
    algorithms_config_path: Optional[str | Path] = None,
) -> EvaluationResult:
    """
    Evaluate a single algorithm using stratified k-fold cross-validation.

    Parameters
    ----------
    algorithm_name : str
        Must match a 'name' field in configs/algorithms.yaml.
    X : pd.DataFrame
        Feature matrix (may contain NaNs — will be imputed inside the pipeline).
    y : pd.Series
        Target vector.
    task_type : str
        "classification" or "regression".
    cv_folds : int
        Number of cross-validation folds (default 5).
    random_seed : int
        Controls all random splitting and estimator seeds (default 42).
    algorithm_params : dict, optional
        Override the default hyperparameters from the config file.
    algorithms_config_path : str or Path, optional
        Path to algorithms.yaml. Defaults to configs/algorithms.yaml relative
        to the project root (two levels above this file).

    Returns
    -------
    EvaluationResult (dict) with keys:
        algorithm       : str   — algorithm name
        task_type       : str
        cv_folds        : int
        primary_metric  : str   — name of the primary metric
        primary_score   : float — mean CV score on primary metric
        primary_std     : float — std of CV scores
        secondary_metrics : dict[str, float]  — mean scores of additional metrics
        cv_scores       : list[float]  — raw per-fold primary scores
        train_time_s    : float — total wall-clock training time (seconds)
        status          : str   — "ok" | "failed"
        error           : str   — error message if status == "failed", else ""
        params_used     : dict  — actual hyperparameters used
    """
    logger.info(
        "Evaluating %s | task=%s | cv=%d | seed=%d",
        algorithm_name, task_type, cv_folds, random_seed,
    )

    result: EvaluationResult = {
        "algorithm": algorithm_name,
        "task_type": task_type,
        "cv_folds": cv_folds,
        "primary_metric": "",
        "primary_score": float("nan"),
        "primary_std": float("nan"),
        "secondary_metrics": {},
        "cv_scores": [],
        "train_time_s": 0.0,
        "status": "failed",
        "error": "",
        "params_used": {},
    }

    try:
# 1. Load algorithm definition
        algo_def = _load_algorithm_definition(
            algorithm_name, task_type, algorithms_config_path
        )
        params = dict(algo_def.get("params", {}))
        if algorithm_params:
            params.update(algorithm_params)
        result["params_used"] = params

# 2. Instantiate the estimator
        estimator = _instantiate_estimator(algo_def, params, random_seed)

# 3. Build preprocessing + model pipeline
        pipeline = _build_pipeline(X, estimator)

# 4. Encode target for classification
        y_encoded, label_encoder = _encode_target(y, task_type)

# 5. Choose CV strategy
        cv = _make_cv(task_type, cv_folds, random_seed, y_encoded)

# 6. Choose scoring metrics
        primary_metric, scoring_dict = _get_scoring(task_type)
        result["primary_metric"] = primary_metric

# 7. Run cross-validation
        start = time.perf_counter()
        cv_results = cross_validate(
            pipeline,
            X,
            y_encoded,
            cv=cv,
            scoring=scoring_dict,
            return_train_score=False,
            error_score="raise",
        )
        elapsed = time.perf_counter() - start
        result["train_time_s"] = round(elapsed, 3)

# 8. Extract and store scores
        primary_key = f"test_{primary_metric}"
        cv_scores = cv_results[primary_key].tolist()
        result["cv_scores"] = cv_scores
        result["primary_score"] = round(float(np.mean(cv_scores)), 5)
        result["primary_std"] = round(float(np.std(cv_scores)), 5)

        secondary: dict[str, float] = {}
        for metric_name in scoring_dict:
            if metric_name == primary_metric:
                continue
            scores = cv_results.get(f"test_{metric_name}", np.array([]))
            if len(scores) > 0:
                secondary[metric_name] = round(float(np.mean(scores)), 5)
        result["secondary_metrics"] = secondary

        result["status"] = "ok"

        logger.info(
            "  %s: %s=%.4f ± %.4f (%.2fs)",
            algorithm_name,
            primary_metric,
            result["primary_score"],
            result["primary_std"],
            elapsed,
        )

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("  %s FAILED: %s", algorithm_name, exc, exc_info=True)

    return result


def evaluate_multiple(
    algorithm_names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int = 5,
    random_seed: int = 42,
    algorithms_config_path: Optional[str | Path] = None,
) -> list[EvaluationResult]:
    """
    Evaluate a list of algorithms and return results sorted by primary score (descending).

    Parameters
    ----------
    algorithm_names : list[str]
        Algorithm names from algorithms.yaml.
    (all others are forwarded to evaluate_candidate)

    Returns
    -------
    List of EvaluationResult dicts, sorted best-first.
    Failed evaluations are included at the end with status="failed".
    """
    results = []
    for name in algorithm_names:
        r = evaluate_candidate(
            algorithm_name=name,
            X=X,
            y=y,
            task_type=task_type,
            cv_folds=cv_folds,
            random_seed=random_seed,
            algorithms_config_path=algorithms_config_path,
        )
        results.append(r)

    # Sort: successful results first by primary_score descending; failures last.
    def sort_key(r: EvaluationResult):
        if r["status"] != "ok":
            return -float("inf")
        return r["primary_score"]

    results.sort(key=sort_key, reverse=True)
    return results


def get_available_algorithms(
    task_type: str,
    algorithms_config_path: Optional[str | Path] = None,
) -> list[str]:
    """
    Return the list of algorithm names available for a given task type.

    Parameters
    ----------
    task_type : str
        "classification" or "regression".
    algorithms_config_path : str or Path, optional
        Path to algorithms.yaml.

    Returns
    -------
    List of algorithm name strings.
    """
    config = _load_algorithms_config(algorithms_config_path)
    return [algo["name"] for algo in config.get(task_type, [])]


# Internal helpers

def _resolve_config_path(path: Optional[str | Path]) -> Path:
    """Find algorithms.yaml — search from file location upward."""
    if path is not None:
        return Path(path)
    # Default: two levels up from this file = project root, then configs/
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate = project_root / "configs" / "algorithms.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"algorithms.yaml not found at expected path: {candidate}\n"
        "Pass algorithms_config_path explicitly if the project layout differs."
    )


def _load_algorithms_config(path: Optional[str | Path]) -> dict:
    config_path = _resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_algorithm_definition(
    name: str,
    task_type: str,
    config_path: Optional[str | Path],
) -> dict:
    config = _load_algorithms_config(config_path)
    algo_list = config.get(task_type, [])
    for algo in algo_list:
        if algo["name"] == name:
            return algo
    available = [a["name"] for a in algo_list]
    raise ValueError(
        f"Algorithm '{name}' not found for task_type='{task_type}'. "
        f"Available: {available}"
    )


def _instantiate_estimator(algo_def: dict, params: dict, random_seed: int) -> Any:
    """Import and instantiate the sklearn estimator class."""
    module = importlib.import_module(algo_def["module"])
    cls = getattr(module, algo_def["class"])
    # Inject random_state only if the estimator accepts it.
    import inspect
    sig = inspect.signature(cls.__init__)
    if "random_state" in sig.parameters and "random_state" not in params:
        params = {**params, "random_state": random_seed}
    return cls(**params)


def _build_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """
    Build a preprocessing + estimator Pipeline.

    Numerical columns:  impute missing with median, then StandardScale.
    Categorical columns: impute missing with most frequent, then OrdinalEncode.

    This ensures preprocessing statistics are fitted only on training folds.
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        # Edge case: no columns — just passthrough
        preprocessor = "passthrough"
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def _encode_target(y: pd.Series, task_type: str) -> tuple[np.ndarray, Optional[LabelEncoder]]:
    """Encode classification targets to integers. Regression targets are left as-is."""
    if task_type == "classification":
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le
    return y.to_numpy(), None


def _make_cv(task_type: str, cv_folds: int, random_seed: int, y: np.ndarray):
    """Return the appropriate cross-validation splitter."""
    if task_type == "classification":
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)


def _get_scoring(task_type: str) -> tuple[str, dict[str, Any]]:
    """
    Return (primary_metric_name, scoring_dict) for cross_validate().
    Primary metric is returned first so callers can rank results.
    """
    if task_type == "classification":
        return "balanced_accuracy", {
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
    else:  # regression
        return "r2", {
            "r2": "r2",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        }
