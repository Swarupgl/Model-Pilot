"""
src/modelpilot/data/loader.py

Responsible for loading datasets from CSV files or scikit-learn built-in datasets,
validating basic data quality, and detecting the task type (classification vs regression).

Design decisions:
- Returns a plain Python dict (a LoadedDataset) rather than a custom class,
  so callers don't need to import a type just to inspect results.
- Does NOT modify the data (no dropping, no imputing) — diagnosis and preprocessing
  are handled by separate modules. The loader only reports problems it finds.
- Task type detection is heuristic; the caller must confirm with the user when ambiguous.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public type alias — a dict describing a loaded dataset.
# Keys are documented here so callers know what to expect.
# ──────────────────────────────────────────────────────────────────────────────
LoadedDataset = dict  # See load_csv / load_builtin for the guaranteed keys.


def load_csv(
    path: str | Path,
    target_column: Optional[str] = None,
) -> LoadedDataset:
    """
    Load a CSV file into a pandas DataFrame and run basic validation.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the CSV file.
    target_column : str, optional
        Name of the column to treat as the prediction target.
        If None, the last column is used and a warning is logged.

    Returns
    -------
    LoadedDataset (dict) with keys:
        df          : pd.DataFrame  — full dataset (features + target)
        X           : pd.DataFrame  — feature columns only
        y           : pd.Series     — target column
        target      : str           — name of the target column
        task_type   : str           — "classification" | "regression" | "unknown"
        n_rows      : int
        n_features  : int
        warnings    : list[str]     — non-fatal issues found during loading
        source      : str           — "csv:<path>"

    Raises
    ------
    FileNotFoundError  — if the path does not exist.
    ValueError         — if the DataFrame is empty or target column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info("Loading CSV from: %s", path)
    df = pd.read_csv(path)

    return _validate_and_package(
        df=df,
        target_column=target_column,
        source=f"csv:{path}",
    )


def load_builtin(name: str, target_column: Optional[str] = None) -> LoadedDataset:
    """
    Load one of the bundled scikit-learn sample datasets by name.

    Supported names
    ---------------
    "iris"             — 150 rows, 4 features, 3-class classification
    "breast_cancer"    — 569 rows, 30 features, binary classification
    "wine"             — 178 rows, 13 features, 3-class classification
    "diabetes"         — 442 rows, 10 features, regression
    "california"       — 20640 rows, 8 features, regression

    Parameters
    ----------
    name : str
        Dataset name (case-insensitive).
    target_column : str, optional
        Override the default target column name.

    Returns
    -------
    LoadedDataset (same keys as load_csv).

    Raises
    ------
    ValueError — if the name is not recognised.
    """
    name = name.lower().strip()
    logger.info("Loading built-in dataset: %s", name)

    loaders = {
        "iris": _load_iris,
        "breast_cancer": _load_breast_cancer,
        "wine": _load_wine,
        "diabetes": _load_diabetes,
        "california": _load_california,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown built-in dataset '{name}'. "
            f"Supported: {sorted(loaders.keys())}"
        )

    df, default_target = loaders[name]()
    chosen_target = target_column or default_target

    return _validate_and_package(
        df=df,
        target_column=chosen_target,
        source=f"builtin:{name}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_and_package(
    df: pd.DataFrame,
    target_column: Optional[str],
    source: str,
) -> LoadedDataset:
    """
    Shared validation + packaging logic for both CSV and built-in loaders.
    """
    warnings: list[str] = []

    # ── Basic structural checks ───────────────────────────────────────────────
    if df.empty:
        raise ValueError("Dataset is empty (0 rows after loading).")

    if len(df.columns) < 2:
        raise ValueError(
            f"Dataset has only {len(df.columns)} column(s). "
            "Need at least one feature and one target."
        )

    # ── Resolve target column ─────────────────────────────────────────────────
    if target_column is None:
        target_column = df.columns[-1]
        warnings.append(
            f"No target column specified — using last column '{target_column}'. "
            "Confirm this is correct."
        )
    elif target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ── Detect task type ──────────────────────────────────────────────────────
    task_type, task_warning = _detect_task_type(y, target_column)
    if task_warning:
        warnings.append(task_warning)

    # ── Soft warnings ─────────────────────────────────────────────────────────
    if df.isnull().values.any():
        missing_count = df.isnull().sum().sum()
        warnings.append(
            f"Dataset contains {missing_count} missing value(s). "
            "Preprocessing will be required before training."
        )

    if len(df) < 50:
        warnings.append(
            f"Dataset is very small ({len(df)} rows). "
            "Cross-validation estimates may be unreliable."
        )

    result: LoadedDataset = {
        "df": df,
        "X": X,
        "y": y,
        "target": target_column,
        "task_type": task_type,
        "n_rows": len(df),
        "n_features": len(X.columns),
        "warnings": warnings,
        "source": source,
    }

    _log_summary(result)
    return result


def _detect_task_type(y: pd.Series, column_name: str) -> tuple[str, str]:
    """
    Heuristically detect whether the target suggests classification or regression.

    Returns
    -------
    (task_type, warning_message)
    warning_message is empty string if detection is confident.
    """
    n_unique = y.nunique()
    n_total = len(y)
    dtype = y.dtype
    dtype_str = str(dtype)  # handles pandas 3 StringDtype ("string[python]" etc.)

    # pandas 3 uses pd.StringDtype for string columns — catch by dtype name
    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or dtype_str.startswith("string")   # pandas 3 StringDtype
        or dtype_str == "str"
    ):
        return "classification", ""

    if isinstance(dtype, pd.CategoricalDtype):  # replaces deprecated is_categorical_dtype
        return "classification", ""

    if pd.api.types.is_integer_dtype(dtype):
        if n_unique <= 20:
            return "classification", ""
        else:
            return "regression", (
                f"Target '{column_name}' is integer with {n_unique} unique values — "
                "treating as regression. Override if this is ordinal classification."
            )

    if pd.api.types.is_float_dtype(dtype):
        # Use a relative threshold: few unique values *relative to dataset size*
        # to avoid misclassifying tiny regression datasets with few rows.
        absolute_threshold = 10
        relative_threshold = max(0.05 * n_total, absolute_threshold)  # 5% of rows, min 10
        if n_unique <= min(absolute_threshold, relative_threshold) and n_unique < n_total * 0.05:
            return "classification", (
                f"Target '{column_name}' is float but has only {n_unique} unique values — "
                "treating as classification. Confirm this is correct."
            )
        return "regression", ""

    return "unknown", (
        f"Cannot determine task type for target '{column_name}' "
        f"(dtype={dtype}, unique={n_unique}). Please specify task_type manually."
    )


def _log_summary(ds: LoadedDataset) -> None:
    logger.info(
        "Dataset loaded | rows=%d | features=%d | target='%s' | task=%s | source=%s",
        ds["n_rows"],
        ds["n_features"],
        ds["target"],
        ds["task_type"],
        ds["source"],
    )
    for w in ds["warnings"]:
        logger.warning("  [LOADER WARNING] %s", w)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in dataset helpers — convert sklearn Bunch → DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _load_iris() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    df = df.drop(columns=["target_y"] if "target_y" in df.columns else [])
    # sklearn as_frame already includes 'target' — rename for clarity
    return df, "target"


def _load_breast_cancer() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return data.frame.copy(), "target"


def _load_wine() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.frame.copy(), "target"


def _load_diabetes() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    return data.frame.copy(), "target"


def _load_california() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return data.frame.copy(), "MedHouseVal"
