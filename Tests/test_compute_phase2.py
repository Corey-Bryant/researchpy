# -*- coding: utf-8 -*-
"""
Tests for Phase 2: Unified computation layer (_compute.py).

Validates:
- _compute_grouped_flat produces correct results for both linear (matrix) and non-linear paths
- mean/median/skewness/kurtosis still work grouped after removing fallback_grouped_func
- Mixed-formula routing end-to-end produces Term-tagged stacked DataFrames
- All calling conventions produce expected output through the unified pipeline
"""

import pytest
import numpy as np
import pandas as pd

from researchpy.statistics import mean, median, skewness, kurtosis
from researchpy.statistics._compute import _compute_grouped_flat, _compute_mixed
from researchpy.core.spec import ComputeSpec, TermSpec, resolve


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grouped_df():
    """DataFrame with multiple grouping variables for testing."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "k": ["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"],
        "z": ["m", "m", "f", "f", "f", "f", "m", "m"],
    })


@pytest.fixture
def simple_df():
    """Simple DataFrame for basic grouped tests."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "b", "b", "c", "c"],
    })


# ============================================================================
# _compute_grouped_flat — linear path (matrix_stat)
# ============================================================================

class TestComputeGroupedFlatLinear:
    """Tests for _compute_grouped_flat with matrix_stat (linear path)."""

    def test_mean_single_group(self, simple_df):
        result = _compute_grouped_flat(
            simple_df, "y", ["g"],
            scalar_func=np.nanmean,
            matrix_stat="mean",
            decimals=4,
            stat_label="Mean",
        )
        assert list(result.columns) == ["g", "Mean"]
        assert len(result) == 3
        # group a: (1+2)/2 = 1.5, b: (3+4)/2 = 3.5, c: (5+6)/2 = 5.5
        assert result.loc[result["g"] == "a", "Mean"].values[0] == 1.5
        assert result.loc[result["g"] == "b", "Mean"].values[0] == 3.5
        assert result.loc[result["g"] == "c", "Mean"].values[0] == 5.5

    def test_multi_group(self, grouped_df):
        result = _compute_grouped_flat(
            grouped_df, "y", ["x", "k"],
            scalar_func=np.nanmean,
            matrix_stat="mean",
            decimals=4,
            stat_label="Mean",
        )
        assert "x" in result.columns
        assert "k" in result.columns
        assert "Mean" in result.columns
        # 4 cells: a:hi, a:lo, b:hi, b:lo
        assert len(result) == 4

    def test_count(self, simple_df):
        result = _compute_grouped_flat(
            simple_df, "y", ["g"],
            scalar_func=np.nanmean,  # not used for linear path
            matrix_stat="count",
            decimals=4,
            stat_label="N",
        )
        # Each group has 2 observations
        assert all(result["N"] == 2)


# ============================================================================
# _compute_grouped_flat — non-linear path (scalar iteration)
# ============================================================================

class TestComputeGroupedFlatNonLinear:
    """Tests for _compute_grouped_flat with matrix_stat=None (non-linear path)."""

    def test_median_single_group(self, simple_df):
        result = _compute_grouped_flat(
            simple_df, "y", ["g"],
            scalar_func=np.nanmedian,
            matrix_stat=None,
            decimals=4,
            stat_label="Median",
        )
        assert list(result.columns) == ["g", "Median"]
        assert len(result) == 3
        assert result.loc[result["g"] == "a", "Median"].values[0] == 1.5
        assert result.loc[result["g"] == "b", "Median"].values[0] == 3.5
        assert result.loc[result["g"] == "c", "Median"].values[0] == 5.5

    def test_multi_group_non_linear(self, grouped_df):
        result = _compute_grouped_flat(
            grouped_df, "y", ["x", "k"],
            scalar_func=np.nanmedian,
            matrix_stat=None,
            decimals=4,
            stat_label="Median",
        )
        assert "x" in result.columns
        assert "k" in result.columns
        assert "Median" in result.columns
        assert len(result) == 4

    def test_handles_nan(self):
        df = pd.DataFrame({
            "y": [1, np.nan, 3, 4, np.nan, 6],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        result = _compute_grouped_flat(
            df, "y", ["g"],
            scalar_func=np.nanmedian,
            matrix_stat=None,
            decimals=4,
            stat_label="Median",
        )
        # group a: median of [1, 3] = 2.0; group b: median of [4, 6] = 5.0
        assert result.loc[result["g"] == "a", "Median"].values[0] == 2.0
        assert result.loc[result["g"] == "b", "Median"].values[0] == 5.0

    def test_empty_group_returns_nan(self):
        """If all values in a group are NaN, result should be NaN."""
        df = pd.DataFrame({
            "y": [np.nan, np.nan, 3, 4],
            "g": ["a", "a", "b", "b"],
        })
        result = _compute_grouped_flat(
            df, "y", ["g"],
            scalar_func=np.nanmedian,
            matrix_stat=None,
            decimals=4,
            stat_label="Median",
        )
        assert np.isnan(result.loc[result["g"] == "a", "Median"].values[0])
        assert result.loc[result["g"] == "b", "Median"].values[0] == 3.5


# ============================================================================
# Simplified stat functions (no fallback_grouped_func)
# ============================================================================

class TestSimplifiedStatFunctions:
    """Verify that mean, median, skewness, kurtosis still work grouped."""

    def test_mean_grouped_formula(self, simple_df):
        result = mean("y ~ C(g)", simple_df)
        # Should produce grouped DataFrame with MultiIndex
        assert result.index.name == "g"
        assert "Mean" in result.columns
        assert result.loc["a", "Mean"] == 1.5
        assert result.loc["b", "Mean"] == 3.5
        assert result.loc["c", "Mean"] == 5.5

    def test_median_grouped_formula(self, simple_df):
        result = median("y ~ C(g)", simple_df)
        assert result.index.name == "g"
        assert "Median" in result.columns
        assert result.loc["a", "Median"] == 1.5
        assert result.loc["b", "Median"] == 3.5

    def test_median_grouped_keywords(self, simple_df):
        result = median(dv="y", by="g", data=simple_df)
        assert result.index.name == "g"
        assert result.loc["c", "Median"] == 5.5

    def test_median_marginal(self, grouped_df):
        result = median("y ~ C(x) + C(k)", grouped_df)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        assert "Median" in result.columns
        # x factor: a median = 2.5 (1,2,5,6 → median=3.5), b median = 5.5 (3,4,7,8 → median=5.5)
        x_rows = result[result["Factor"] == "x"]
        assert len(x_rows) == 2

    def test_median_pivot(self, grouped_df):
        result = median("y ~ C(x)*C(k)", grouped_df)
        # Should be a pivot table
        assert result.index.name == "x"
        assert "hi" in result.columns or "lo" in result.columns

    def test_skewness_grouped(self, simple_df):
        result = skewness("y ~ C(g)", simple_df)
        assert result.index.name == "g"
        assert "Skewness" in result.columns

    def test_kurtosis_grouped(self, simple_df):
        result = kurtosis("y ~ C(g)", simple_df)
        assert result.index.name == "g"
        assert "Kurtosis" in result.columns


# ============================================================================
# Mixed-formula end-to-end routing
# ============================================================================

class TestMixedFormulaEndToEnd:
    """Tests for the full mixed-formula pipeline: parse → route → compute → stack."""

    def test_mean_mixed_formula(self, grouped_df):
        """y ~ C(x) + C(k):C(z) should produce a Term-tagged stacked DataFrame."""
        result = mean("y ~ C(x) + C(k):C(z)", grouped_df)
        assert "Term" in result.columns
        assert "Level" in result.columns
        assert "Mean" in result.columns

        # Should have rows for term "x" and term "k:z"
        terms = result["Term"].unique()
        assert "x" in terms
        assert "k:z" in terms

    def test_mixed_formula_term_x_values(self, grouped_df):
        """Check that the main effect term computes correctly."""
        result = mean("y ~ C(x) + C(k):C(z)", grouped_df)
        x_rows = result[result["Term"] == "x"]
        # x=a: values [1,2,5,6] → mean=3.5; x=b: values [3,4,7,8] → mean=5.5
        a_mean = x_rows[x_rows["Level"] == "a"]["Mean"].values[0]
        b_mean = x_rows[x_rows["Level"] == "b"]["Mean"].values[0]
        assert a_mean == 3.5
        assert b_mean == 5.5

    def test_mixed_formula_interaction_values(self, grouped_df):
        """Check that the interaction term computes correctly."""
        result = mean("y ~ C(x) + C(k):C(z)", grouped_df)
        kz_rows = result[result["Term"] == "k:z"]
        # Interaction groups: hi:f, hi:m, lo:f, lo:m
        assert len(kz_rows) == 4

    def test_median_mixed_formula(self, grouped_df):
        """Mixed formula should also work for non-linear stats."""
        result = median("y ~ C(x) + C(k):C(z)", grouped_df)
        assert "Term" in result.columns
        assert "Median" in result.columns
        terms = result["Term"].unique()
        assert "x" in terms
        assert "k:z" in terms

    def test_mixed_formula_preserves_order(self, grouped_df):
        """Main effects should appear before interactions in the output."""
        result = mean("y ~ C(x) + C(k):C(z)", grouped_df)
        # First rows should be term "x", then "k:z"
        first_term = result.iloc[0]["Term"]
        assert first_term == "x"

    def test_mixed_formula_single_dv(self, grouped_df):
        """Single DV should not have a 'Variable' column."""
        result = mean("y ~ C(x) + C(k):C(z)", grouped_df)
        assert "Variable" not in result.columns


# ============================================================================
# Regression tests for existing patterns through unified pipeline
# ============================================================================

class TestUnifiedPipelineRegression:
    """Ensure all existing patterns work identically through the unified pipeline."""

    def test_mean_scalar(self):
        result = mean(pd.Series([1, 2, 3, 4, 5]))
        assert result == 3.0

    def test_mean_cell_multiindex(self, grouped_df):
        result = mean("y ~ C(x):C(k)", grouped_df)
        # Should have MultiIndex with x and k
        assert result.index.names == ["x", "k"]

    def test_mean_pivot(self, grouped_df):
        result = mean("y ~ C(x)*C(k)", grouped_df)
        # Pivot table: rows=x, cols=k
        assert result.index.name == "x"
        assert set(result.columns) == {"hi", "lo"}

    def test_mean_column_list(self, grouped_df):
        result = mean(["y"], grouped_df)
        # Single column, no groups → scalar
        assert result == 4.5  # mean of 1-8

    def test_median_scalar(self):
        result = median([1, 2, 3, 4, 5])
        assert result == 3.0

