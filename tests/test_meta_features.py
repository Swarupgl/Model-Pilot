"""Tests for meta-feature extractor module."""

import numpy as np
import pandas as pd
import pytest

from modelpilot.data.loader import load_builtin
from modelpilot.meta_learning.meta_feature_extractor import (
    STANDARD_FEATURE_NAMES,
    compute_dataset_fingerprint,
    extract_meta_features,
)


def test_extract_meta_features_iris():
    ds = load_builtin("iris")
    res = extract_meta_features(ds["df"], target_column=ds["target"], task_type="classification")

    assert res["extractor_version"] == "1.0.0"
    assert len(res["vector"]) == len(STANDARD_FEATURE_NAMES)
    assert not any(np.isnan(v) for v in res["vector"])

    assert res["features"]["class_count"] == 3
    assert res["features"]["num_feature_ratio"] == 1.0
    assert res["features"]["cat_feature_ratio"] == 0.0


def test_extract_meta_features_determinism():
    ds = load_builtin("wine")
    res1 = extract_meta_features(ds["df"], target_column=ds["target"], task_type="classification")
    res2 = extract_meta_features(ds["df"], target_column=ds["target"], task_type="classification")

    assert res1["vector"] == res2["vector"]
    assert res1["dataset_fingerprint"] == res2["dataset_fingerprint"]


def test_fingerprint_sensitivity():
    df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    df2 = pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [0, 1, 0]})

    fp1 = compute_dataset_fingerprint(df1, "y")
    fp2 = compute_dataset_fingerprint(df2, "y")
    assert fp1 != fp2


def test_extract_meta_features_regression():
    ds = load_builtin("diabetes")
    res = extract_meta_features(ds["df"], target_column=ds["target"], task_type="regression")

    assert res["features"]["class_count"] == 0.0
    assert len(res["vector"]) == len(STANDARD_FEATURE_NAMES)
    assert not any(np.isnan(v) for v in res["vector"])


def test_meta_features_error():
    with pytest.raises(ValueError):
        extract_meta_features(pd.DataFrame(), "target")
