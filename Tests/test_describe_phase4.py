# -*- coding: utf-8 -*-
"""
Tests for Phase 4: describe() function.

Validates:
- All calling conventions (Series, formula, column-list, keywords, GroupBy)
- Multi-DV with "Variable" column
- Marginal (iv), cell (by), pivot (over), mixed formula
- Statistical correctness against numpy/pandas
- Edge cases (NaN, single observation)
"""

import pytest
import numpy as np
import pandas as pd

from researchpy.statistics import describe


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_df():
    """Simple DataFrame for basic tests."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "a", "b", "b", "b"],
    })


@pytest.fixture
def grouped_df():
    """DataFrame with multiple grouping variables."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "z": [10, 20, 30, 40, 50, 60, 70, 80],
        "x": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "k": ["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"],
    })


@pytest.fixture
def nan_df():
    """DataFrame with missing values."""
    return pd.DataFrame({
        "y": [1, 2, np.nan, 4, 5, np.nan],
        "g": ["a", "a", "a", "b", "b", "b"],
    })


# ============================================================================
# Ungrouped (Convention 1: bare array/Series)
# ============================================================================

class TestDescribeUngrouped:
    """Tests for describe without grouping."""

    def test_returns_dataframe(self):
        result = describe([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self):
        result = describe([1, 2, 3, 4, 5])
        assert len(result) == 1

    def test_expected_columns(self):
        result = describe([1, 2, 3, 4, 5])
        expected_cols = ["N", "N Missing", "Mean", "Median", "SD", "Min", "Q1", "Q3", "Max", "IQR"]
        assert list(result.columns) == expected_cols

    def test_series_input(self):
        s = pd.Series([1, 2, 3, 4, 5], name="test")
        result = describe(s)
        assert isinstance(result, pd.DataFrame)
        assert result["N"].iloc[0] == 5

    def test_values_correct(self):
        data = [1, 2, 3, 4, 5]
        result = describe(data)
        assert result["N"].iloc[0] == 5
        assert result["N Missing"].iloc[0] == 0
        assert result["Mean"].iloc[0] == 3.0
        assert result["Median"].iloc[0] == 3.0
        assert result["Min"].iloc[0] == 1.0
        assert result["Max"].iloc[0] == 5.0

    def test_sd_matches_numpy(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = describe(data)
        expected_sd = round(float(np.std(data, ddof=1)), 4)
        assert result["SD"].iloc[0] == expected_sd

    def test_quartiles_correct(self):
        data = [1, 2, 3, 4, 5]
        result = describe(data)
        assert result["Q1"].iloc[0] == np.percentile(data, 25)
        assert result["Q3"].iloc[0] == np.percentile(data, 75)
        assert result["IQR"].iloc[0] == np.percentile(data, 75) - np.percentile(data, 25)

    def test_handles_nan(self):
        result = describe([1, 2, np.nan, 4, 5])
        assert result["N"].iloc[0] == 4
        assert result["N Missing"].iloc[0] == 1
        # Mean of [1,2,4,5] = 3.0
        assert result["Mean"].iloc[0] == 3.0


# ============================================================================
# Convention 2: Formula
# ============================================================================

class TestDescribeFormula:
    """Tests for describe with formula input."""

    def test_single_factor(self, simple_df):
        result = describe("y ~ C(g)", simple_df)
        assert "g" in result.columns
        assert "N" in result.columns
        assert "Mean" in result.columns
        assert len(result) == 2

    def test_formula_values(self, simple_df):
        result = describe("y ~ C(g)", simple_df)
        a_row = result[result["g"] == "a"]
        b_row = result[result["g"] == "b"]
        # group a: [1,2,3], mean=2, N=3
        # group b: [4,5,6], mean=5, N=3
        assert a_row["N"].iloc[0] == 3
        assert a_row["Mean"].iloc[0] == 2.0
        assert b_row["N"].iloc[0] == 3
        assert b_row["Mean"].iloc[0] == 5.0

    def test_interaction_formula(self, grouped_df):
        result = describe("y ~ C(x):C(k)", grouped_df)
        assert "x" in result.columns
        assert "k" in result.columns
        assert len(result) == 4  # 2x2 cells

    def test_star_formula(self, grouped_df):
        result = describe("y ~ C(x)*C(k)", grouped_df)
        # Pivot treated as flat grouped with by+over
        assert "x" in result.columns
        assert "k" in result.columns


# ============================================================================
# Convention 3: Column list
# ============================================================================

class TestDescribeColumnList:
    """Tests for describe with column list input."""

    def test_single_column(self, simple_df):
        result = describe(["y"], simple_df)
        assert isinstance(result, pd.DataFrame)
        assert result["N"].iloc[0] == 6

    def test_multiple_columns(self, grouped_df):
        result = describe(["y", "z"], grouped_df)
        assert "Variable" in result.columns
        assert len(result) == 2
        assert list(result["Variable"]) == ["y", "z"]


# ============================================================================
# Convention 4: Keywords
# ============================================================================

class TestDescribeKeywords:
    """Tests for describe with keyword arguments."""

    def test_by_single(self, simple_df):
        result = describe(dv="y", by="g", data=simple_df)
        assert "g" in result.columns
        assert len(result) == 2

    def test_iv_marginal(self, grouped_df):
        result = describe(dv="y", iv=["x", "k"], data=grouped_df)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        # 2 levels of x + 2 levels of k = 4 rows
        assert len(result) == 4

    def test_multi_dv(self, grouped_df):
        result = describe(dv=["y", "z"], by="x", data=grouped_df)
        assert "Variable" in result.columns
        # 2 DVs × 2 groups = 4 rows
        assert len(result) == 4

    def test_over_pivot(self, grouped_df):
        result = describe(dv="y", by="x", over="k", data=grouped_df)
        # Treated as flat grouped with x+k
        assert "x" in result.columns
        assert "k" in result.columns
        assert len(result) == 4


# ============================================================================
# Convention 5: GroupBy
# ============================================================================

class TestDescribeGroupBy:
    """Tests for describe with GroupBy input."""

    def test_series_groupby(self, simple_df):
        grouped = simple_df.groupby("g")["y"]
        result = describe(grouped)
        assert "g" in result.columns
        assert len(result) == 2


# ============================================================================
# Mixed formula
# ============================================================================

class TestDescribeMixed:
    """Tests for describe with mixed formula (sub_specs)."""

    def test_mixed_formula(self, grouped_df):
        result = describe("y ~ C(x) + C(x):C(k)", grouped_df)
        assert "Term" in result.columns
        assert "Level" in result.columns
        terms = result["Term"].unique()
        assert "x" in terms
        assert "x:k" in terms

    def test_mixed_formula_has_all_stats(self, grouped_df):
        result = describe("y ~ C(x) + C(x):C(k)", grouped_df)
        expected_cols = ["Term", "Level", "N", "N Missing", "Mean", "Median",
                         "SD", "Min", "Q1", "Q3", "Max", "IQR"]
        assert list(result.columns) == expected_cols


# ============================================================================
# Edge cases
# ============================================================================

class TestDescribeEdgeCases:
    """Edge cases for describe."""

    def test_all_nan_group(self):
        df = pd.DataFrame({
            "y": [np.nan, np.nan, 3, 4],
            "g": ["a", "a", "b", "b"],
        })
        result = describe("y ~ C(g)", df)
        a_row = result[result["g"] == "a"]
        assert a_row["N"].iloc[0] == 0
        assert a_row["N Missing"].iloc[0] == 2
        assert np.isnan(a_row["Mean"].iloc[0])

    def test_nan_in_data(self, nan_df):
        result = describe("y ~ C(g)", nan_df)
        a_row = result[result["g"] == "a"]
        # group a: [1, 2, nan] → N=2, N Missing=1
        assert a_row["N"].iloc[0] == 2
        assert a_row["N Missing"].iloc[0] == 1

    def test_single_observation_per_group(self):
        df = pd.DataFrame({
            "y": [10, 20, 30],
            "g": ["a", "b", "c"],
        })
        result = describe("y ~ C(g)", df)
        # SD should be NaN with only 1 observation
        assert np.isnan(result["SD"].iloc[0])
        # But other stats should be fine
        assert result["N"].iloc[0] == 1
        assert result["Mean"].iloc[0] == 10.0


# ============================================================================
# Statistical correctness validation
# ============================================================================

class TestDescribeStatisticalCorrectness:
    """Validate describe() output against pandas/numpy."""

    def test_grouped_mean_matches_pandas(self, simple_df):
        result = describe("y ~ C(g)", simple_df)
        pandas_means = simple_df.groupby("g")["y"].mean()
        for g_val in ["a", "b"]:
            row = result[result["g"] == g_val]
            assert row["Mean"].iloc[0] == round(pandas_means[g_val], 4)

    def test_grouped_sd_matches_pandas(self, simple_df):
        result = describe("y ~ C(g)", simple_df)
        pandas_sds = simple_df.groupby("g")["y"].std()
        for g_val in ["a", "b"]:
            row = result[result["g"] == g_val]
            assert row["SD"].iloc[0] == round(pandas_sds[g_val], 4)

    def test_grouped_count_matches_pandas(self, simple_df):
        result = describe("y ~ C(g)", simple_df)
        pandas_counts = simple_df.groupby("g")["y"].count()
        for g_val in ["a", "b"]:
            row = result[result["g"] == g_val]
            assert row["N"].iloc[0] == pandas_counts[g_val]

