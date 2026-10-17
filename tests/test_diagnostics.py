"""Tests for dataset diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from modelpilot.diagnosis.dataset_diagnostics import diagnose_dataset
from modelpilot.data.loader import load_builtin


def test_diagnose_iris_classification():
    ds = load_builtin("iris")
    diag = diagnose_dataset(ds["df"], target_column=ds["target"], task_type="classification")

    assert diag["dataset_shape"]["n_rows"] == 150
    assert diag["dataset_shape"]["n_features"] == 4
    assert diag["dataset_shape"]["numerical_features"] == 4
    assert diag["target_diagnostics"]["n_classes"] == 3
    assert diag["target_diagnostics"]["imbalance_ratio"] == 1.0
    assert diag["missing_values"]["total_missing_cells"] == 0


def test_diagnose_diabetes_regression():
    ds = load_builtin("diabetes")
    diag = diagnose_dataset(ds["df"], target_column=ds["target"], task_type="regression")

    assert diag["dataset_shape"]["n_rows"] == 442
    assert diag["target_diagnostics"]["task_type"] == "regression"
    assert "mean" in diag["target_diagnostics"]
    assert "std" in diag["target_diagnostics"]


def test_diagnose_missing_and_constants():
    df = pd.DataFrame({
        "num_const": [1.0, 1.0, 1.0, 1.0, 1.0],
        "with_missing": [np.nan, np.nan, np.nan, 2.0, 3.0],
        "normal": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0, 1, 0, 1, 0],
    })
    diag = diagnose_dataset(df, target_column="target", task_type="classification")

    assert "num_const" in diag["constant_features"]
    assert "with_missing" in diag["missing_values"]["columns_with_missing"]
    assert diag["missing_values"]["columns_with_missing"]["with_missing"]["count"] == 3

    categories = [adv["category"] for adv in diag["advisories"]]
    assert "constants" in categories
    assert "missing_values" in categories


def test_diagnose_imbalance():
    df = pd.DataFrame({
        "feat": list(range(20)),
        "target": [0] * 18 + [1] * 2,
    })
    diag = diagnose_dataset(df, target_column="target", task_type="classification")
    assert diag["target_diagnostics"]["imbalance_ratio"] >= 4.0
    categories = [adv["category"] for adv in diag["advisories"]]
    assert "class_imbalance" in categories


def test_diagnose_leakage():
    df = pd.DataFrame({
        "leak_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "random_feat": [0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6, 0.1, 0.9, 0.4],
        "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    })
    diag = diagnose_dataset(df, target_column="target", task_type="regression")
    assert any(leak["feature"] == "leak_feature" for leak in diag["potential_leakage"])
    categories = [adv["category"] for adv in diag["advisories"]]
    assert "target_leakage" in categories


def test_diagnose_errors():
    with pytest.raises(ValueError, match="empty DataFrame"):
        diagnose_dataset(pd.DataFrame(), "target")

    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not present"):
        diagnose_dataset(df, "missing_col")
