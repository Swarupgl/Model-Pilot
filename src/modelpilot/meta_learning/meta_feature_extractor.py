"""Dataset meta-feature extractor for ModelPilot.

Extracts reproducible, standardized numerical representations of datasets
for cross-dataset meta-learning, similarity ranking, and performance prediction.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "1.0.0"

# Standard meta-feature keys in fixed order for vector representations
STANDARD_FEATURE_NAMES = [
    "log_n_rows",
    "log_n_features",
    "dim_ratio",
    "num_feature_ratio",
    "cat_feature_ratio",
    "missing_ratio",
    "duplicate_row_ratio",
    "sparsity",
    "class_count",
    "class_imbalance_ratio",
    "class_entropy",
    "mean_feature_variance",
    "mean_feature_skew",
    "mean_feature_kurtosis",
    "mean_abs_correlation",
    "max_abs_correlation",
]


def compute_dataset_fingerprint(df: pd.DataFrame, target_column: str) -> str:
    """Compute a deterministic SHA-256 hash representing dataset structure and content.

    Args:
        df: Input DataFrame.
        target_column: Target column name.

    Returns:
        Hexadecimal hash string representing the dataset fingerprint.
    """
    hasher = hashlib.sha256()
    # Hash column names and types
    schema_str = f"target:{target_column};" + ";".join(f"{c}:{df[c].dtype}" for c in sorted(df.columns))
    hasher.update(schema_str.encode("utf-8"))

    # Hash shape and summary statistics
    hasher.update(f"shape:{df.shape}".encode("utf-8"))

    # Sample head and tail for content integrity
    sample_df = pd.concat([df.head(20), df.tail(20)]) if len(df) > 40 else df
    content_repr = sample_df.to_string(index=False)
    hasher.update(content_repr.encode("utf-8"))

    return hasher.hexdigest()[:16]


def extract_meta_features(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """Extract standard meta-features from a dataset.

    Args:
        df: Input DataFrame containing features and target column.
        target_column: Name of the target column.
        task_type: Task type, either 'classification' or 'regression'.

    Returns:
        Dictionary containing:
            - extractor_version: Extractor version string.
            - dataset_fingerprint: Hexadecimal dataset fingerprint.
            - features: Dictionary mapping meta-feature names to float values.
            - vector: List of float values in standard feature order.
            - feature_names: List of feature names corresponding to vector indices.
    """
    if df.empty:
        raise ValueError("Cannot extract meta-features from an empty DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_rows, n_cols = df.shape
    n_features = X.shape[1]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 1. Structural features
    log_n_rows = float(np.log10(max(n_rows, 1)))
    log_n_features = float(np.log10(max(n_features, 1)))
    dim_ratio = float(n_features / max(n_rows, 1))
    num_ratio = float(len(num_cols) / max(n_features, 1))
    cat_ratio = float(len(cat_cols) / max(n_features, 1))

    # 2. Quality and representation
    total_cells = n_rows * n_cols
    missing_ratio = float(df.isna().sum().sum() / max(total_cells, 1))
    duplicate_row_ratio = float(df.duplicated().sum() / max(n_rows, 1))

    zero_count = int((df[num_cols] == 0).sum().sum()) if num_cols else 0
    num_cells = n_rows * len(num_cols) if num_cols else 1
    sparsity = float(zero_count / max(num_cells, 1))

    # 3. Target properties
    if task_type == "classification":
        counts = y.value_counts(normalize=True, dropna=False)
        class_count = float(len(counts))
        min_p = float(counts.min()) if not counts.empty else 0.0
        max_p = float(counts.max()) if not counts.empty else 0.0
        imbalance_ratio = float(max_p / (min_p + 1e-9))
        p = counts.values
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))
        norm_entropy = float(entropy / np.log2(class_count)) if class_count > 1 else 0.0
    else:
        class_count = 0.0
        imbalance_ratio = 1.0
        norm_entropy = 0.0

    # 4. Statistical properties of numerical features
    if num_cols and n_rows > 3:
        # Standardize numerical features locally to get scale-independent variance summaries
        num_data = X[num_cols].apply(pd.to_numeric, errors="coerce")
        variances = num_data.var(skipna=True).dropna()
        skews = num_data.skew(skipna=True).abs().dropna()
        kurtoses = num_data.kurt(skipna=True).dropna()

        mean_var = float(variances.mean()) if not variances.empty else 0.0
        mean_skew = float(skews.mean()) if not skews.empty else 0.0
        mean_kurt = float(kurtoses.mean()) if not kurtoses.empty else 0.0

        if len(num_cols) >= 2:
            corr = num_data.corr().abs().to_numpy()
            triu_indices = np.triu_indices_from(corr, k=1)
            corr_values = corr[triu_indices]
            corr_values = corr_values[~np.isnan(corr_values)]
            mean_corr = float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
            max_corr = float(np.max(corr_values)) if len(corr_values) > 0 else 0.0
        else:
            mean_corr = 0.0
            max_corr = 0.0
    else:
        mean_var = 0.0
        mean_skew = 0.0
        mean_kurt = 0.0
        mean_corr = 0.0
        max_corr = 0.0

    features_dict = {
        "log_n_rows": round(log_n_rows, 4),
        "log_n_features": round(log_n_features, 4),
        "dim_ratio": round(dim_ratio, 4),
        "num_feature_ratio": round(num_ratio, 4),
        "cat_feature_ratio": round(cat_ratio, 4),
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_row_ratio": round(duplicate_row_ratio, 4),
        "sparsity": round(sparsity, 4),
        "class_count": class_count,
        "class_imbalance_ratio": round(imbalance_ratio, 4),
        "class_entropy": round(norm_entropy, 4),
        "mean_feature_variance": round(mean_var, 4),
        "mean_feature_skew": round(mean_skew, 4),
        "mean_feature_kurtosis": round(mean_kurt, 4),
        "mean_abs_correlation": round(mean_corr, 4),
        "max_abs_correlation": round(max_corr, 4),
    }

    feature_vector = [features_dict[name] for name in STANDARD_FEATURE_NAMES]
    fingerprint = compute_dataset_fingerprint(df, target_column)

    return {
        "extractor_version": EXTRACTOR_VERSION,
        "dataset_fingerprint": fingerprint,
        "features": features_dict,
        "vector": feature_vector,
        "feature_names": STANDARD_FEATURE_NAMES,
    }
