"""Dataset diagnostics module for ModelPilot.

Provides comprehensive, non-destructive quality checks and risk assessments
on tabular datasets, distinguishing observed facts from diagnostic advisories.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def diagnose_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    correlation_threshold: float = 0.90,
    leakage_threshold: float = 0.98,
) -> Dict[str, Any]:
    """Run non-destructive diagnostic checks on a dataset.

    Args:
        df: Input DataFrame containing features and target column.
        target_column: Name of the target variable.
        task_type: Task type, either 'classification' or 'regression'.
        correlation_threshold: Pearson correlation above which feature pairs
            are flagged as collinear.
        leakage_threshold: Absolute correlation with target above which a
            feature is flagged for potential target leakage.

    Returns:
        Structured dictionary containing dataset overview, missingness stats,
        duplicate checks, target statistics, collinearity analysis, and
        prioritized diagnostic advisories.
    """
    if df.empty:
        raise ValueError("Cannot diagnose an empty DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_rows, n_cols = df.shape
    n_features = X.shape[1]
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    advisories: List[Dict[str, str]] = []

    # 1. Sparsity & Missingness
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    missing_rate = round(float(missing_cells / total_cells), 4)

    missing_by_col = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            rate = round(float(cnt / n_rows), 4)
            missing_by_col[col] = {"count": cnt, "ratio": rate}
            if rate > 0.40:
                advisories.append({
                    "severity": "warning",
                    "category": "missing_values",
                    "message": f"Column '{col}' has {rate*100:.1f}% missing values."
                })

    zero_count = int((df[num_cols] == 0).sum().sum()) if num_cols else 0
    sparsity = round(float(zero_count / (n_rows * len(num_cols))), 4) if num_cols else 0.0

    # 2. Duplicate rows & Constant features
    duplicate_rows = int(df.duplicated().sum())
    duplicate_row_rate = round(float(duplicate_rows / n_rows), 4)
    if duplicate_rows > 0:
        advisories.append({
            "severity": "info",
            "category": "duplicates",
            "message": f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_row_rate*100:.1f}%)."
        })

    constant_features = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    quasi_constant = []
    for col in X.columns:
        if col not in constant_features:
            top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq >= 0.99:
                quasi_constant.append(col)

    if constant_features:
        advisories.append({
            "severity": "warning",
            "category": "constants",
            "message": f"Features with zero variance (single value): {constant_features}."
        })
    if quasi_constant:
        advisories.append({
            "severity": "info",
            "category": "quasi_constants",
            "message": f"Features with >99% dominant value: {quasi_constant}."
        })

    # 3. High feature-to-sample ratio
    dim_ratio = round(float(n_features / n_rows), 4)
    if dim_ratio > 0.20:
        advisories.append({
            "severity": "warning",
            "category": "dimensionality",
            "message": f"High feature-to-sample ratio ({dim_ratio:.3f}). Regularization recommended to avoid overfitting."
        })

    # 4. Target distribution & imbalance
    target_stats: Dict[str, Any] = {"task_type": task_type}
    if task_type == "classification":
        class_counts = y.value_counts(dropna=False).to_dict()
        class_ratios = y.value_counts(normalize=True, dropna=False).to_dict()
        n_classes = len(class_counts)
        min_class_ratio = min(class_ratios.values()) if class_ratios else 0.0
        max_class_ratio = max(class_ratios.values()) if class_ratios else 0.0
        imbalance_ratio = round(float(max_class_ratio / (min_class_ratio + 1e-9)), 2)

        # Normalized class entropy (1.0 = balanced, 0.0 = completely degenerate)
        p = np.array(list(class_ratios.values()))
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))
        norm_entropy = round(float(entropy / np.log2(n_classes)), 4) if n_classes > 1 else 0.0

        target_stats.update({
            "n_classes": n_classes,
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "class_ratios": {str(k): round(float(v), 4) for k, v in class_ratios.items()},
            "imbalance_ratio": imbalance_ratio,
            "normalized_entropy": norm_entropy,
        })

        if imbalance_ratio > 4.0:
            advisories.append({
                "severity": "warning",
                "category": "class_imbalance",
                "message": f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1). Balanced accuracy or F1 recommended."
            })
    else:
        y_num = pd.to_numeric(y, errors="coerce").dropna()
        target_stats.update({
            "mean": round(float(y_num.mean()), 4) if not y_num.empty else 0.0,
            "std": round(float(y_num.std()), 4) if len(y_num) > 1 else 0.0,
            "skewness": round(float(y_num.skew()), 4) if len(y_num) > 2 else 0.0,
            "kurtosis": round(float(y_num.kurt()), 4) if len(y_num) > 3 else 0.0,
        })
        if abs(target_stats["skewness"]) > 2.0:
            advisories.append({
                "severity": "info",
                "category": "target_skew",
                "message": f"Target exhibits significant skewness ({target_stats['skewness']:.2f}). Log transformation may improve linear models."
            })

    # 5. Collinearity & Potential Target Leakage
    collinear_pairs = []
    potential_leakage = []

    if len(num_cols) >= 2:
        corr_matrix = X[num_cols].corr().abs()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if pd.notna(val) and val >= correlation_threshold:
                    collinear_pairs.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(val), 4)
                    })

    if collinear_pairs:
        advisories.append({
            "severity": "info",
            "category": "multicollinearity",
            "message": f"Found {len(collinear_pairs)} feature pairs with correlation >= {correlation_threshold}."
        })

    # Leakage check against target
    if pd.api.types.is_numeric_dtype(y) and num_cols:
        y_numeric = y.astype(float)
        for col in num_cols:
            valid_mask = X[col].notna() & y_numeric.notna()
            if valid_mask.sum() > 10:
                corr = abs(float(np.corrcoef(X.loc[valid_mask, col], y_numeric[valid_mask])[0, 1]))
                if pd.notna(corr) and corr >= leakage_threshold:
                    potential_leakage.append({"feature": col, "target_correlation": round(corr, 4)})

    if potential_leakage:
        advisories.append({
            "severity": "warning",
            "category": "target_leakage",
            "message": f"Potential target leakage: {potential_leakage}. Verify feature collection timing."
        })

    return {
        "dataset_shape": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_features": n_features,
            "numerical_features": len(num_cols),
            "categorical_features": len(cat_cols),
            "memory_mb": memory_mb,
            "dim_ratio": dim_ratio,
            "sparsity": sparsity,
        },
        "missing_values": {
            "total_missing_cells": missing_cells,
            "missing_ratio": missing_rate,
            "columns_with_missing": missing_by_col,
        },
        "duplicate_rows": {
            "count": duplicate_rows,
            "ratio": duplicate_row_rate,
        },
        "constant_features": constant_features,
        "quasi_constant_features": quasi_constant,
        "collinear_pairs": collinear_pairs,
        "potential_leakage": potential_leakage,
        "target_diagnostics": target_stats,
        "advisories": advisories,
    }
