"""
tests/test_loader.py

Tests for src/modelpilot/data/loader.py

Covers:
- Loading each built-in dataset
- Correct column detection and task type inference
- Error handling for invalid paths and missing target columns
- Handling datasets with NaN values
- Edge cases: single feature, numeric targets with few unique values
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from modelpilot.data.loader import load_builtin, load_csv


# ──────────────────────────────────────────────────────────────────────────────
# Built-in dataset tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadBuiltin:
    def test_iris_loads(self):
        ds = load_builtin("iris")
        assert ds["n_rows"] == 150
        assert ds["n_features"] == 4
        assert ds["task_type"] == "classification"
        assert ds["target"] == "target"
        assert ds["source"] == "builtin:iris"

    def test_breast_cancer_loads(self):
        ds = load_builtin("breast_cancer")
        assert ds["n_rows"] == 569
        assert ds["n_features"] == 30
        assert ds["task_type"] == "classification"

    def test_wine_loads(self):
        ds = load_builtin("wine")
        assert ds["n_rows"] == 178
        assert ds["task_type"] == "classification"

    def test_diabetes_loads(self):
        ds = load_builtin("diabetes")
        assert ds["task_type"] == "regression"
        assert ds["n_features"] == 10

    def test_california_loads(self):
        ds = load_builtin("california")
        assert ds["task_type"] == "regression"
        assert ds["n_rows"] == 20640

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown built-in dataset"):
            load_builtin("nonexistent_dataset")

    def test_case_insensitive(self):
        ds = load_builtin("IRIS")
        assert ds["task_type"] == "classification"

    def test_returns_dataframe_and_series(self):
        ds = load_builtin("iris")
        assert isinstance(ds["df"], pd.DataFrame)
        assert isinstance(ds["X"], pd.DataFrame)
        assert isinstance(ds["y"], pd.Series)

    def test_X_does_not_contain_target(self):
        ds = load_builtin("iris")
        assert ds["target"] not in ds["X"].columns

    def test_n_features_matches_X_shape(self):
        ds = load_builtin("iris")
        assert ds["n_features"] == ds["X"].shape[1]

    def test_n_rows_matches_shape(self):
        ds = load_builtin("iris")
        assert ds["n_rows"] == ds["X"].shape[0]
        assert ds["n_rows"] == len(ds["y"])


# ──────────────────────────────────────────────────────────────────────────────
# CSV loader tests (using in-memory strings instead of real files)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def _write_tmp_csv(self, tmp_path, content: str, filename="data.csv") -> str:
        p = tmp_path / filename
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_basic_classification_csv(self, tmp_path):
        csv = """\
            feature1,feature2,label
            1.0,2.0,cat
            3.0,4.0,dog
            5.0,6.0,cat
            7.0,8.0,dog
            9.0,0.0,cat
        """
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="label")
        assert ds["task_type"] == "classification"
        assert ds["target"] == "label"
        assert ds["n_features"] == 2
        assert ds["n_rows"] == 5

    def test_basic_regression_csv(self, tmp_path):
        # Need enough rows AND enough unique values for the detector to be confident.
        # Use 30 rows each with a distinct float price — well above the 5% threshold.
        import random
        random.seed(1)
        rows = "\n".join(f"{i+1}.0,{i*2+0.5},{round(random.uniform(100, 999), 2)}" for i in range(30))
        csv = "x1,x2,price\n" + rows
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="price")
        assert ds["task_type"] == "regression"

    def test_last_column_used_when_target_not_specified(self, tmp_path):
        csv = """\
            a,b,c
            1,2,0
            3,4,1
        """
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path)
        assert ds["target"] == "c"
        # Should warn about auto-selection
        assert any("last column" in w for w in ds["warnings"])

    def test_missing_target_column_raises(self, tmp_path):
        csv = """\
            a,b
            1,2
        """
        path = self._write_tmp_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="not found in dataset"):
            load_csv(path, target_column="nonexistent")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_csv_with_missing_values_warns(self, tmp_path):
        csv = """\
            a,b,target
            1.0,,0
            ,4.0,1
            5.0,6.0,0
        """
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="target")
        # Must warn but NOT raise
        assert any("missing" in w.lower() for w in ds["warnings"])
        assert ds["n_rows"] == 3  # rows are kept — loader never drops data

    def test_small_dataset_warns(self, tmp_path):
        rows = "\n".join(f"{i},{i+1},a" for i in range(10))
        csv = "x,y,label\n" + rows
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="label")
        assert any("small" in w.lower() for w in ds["warnings"])

    def test_integer_target_with_few_values_is_classification(self, tmp_path):
        csv = """\
            a,b,target
            1,2,0
            3,4,1
            5,6,0
            7,8,1
        """
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="target")
        assert ds["task_type"] == "classification"

    def test_float_target_with_many_values_is_regression(self, tmp_path):
        import random
        random.seed(0)
        rows = "\n".join(f"{i},{i*1.1},{random.uniform(0,100):.4f}" for i in range(100))
        csv = "x,y,price\n" + rows
        path = self._write_tmp_csv(tmp_path, csv)
        ds = load_csv(path, target_column="price")
        assert ds["task_type"] == "regression"
