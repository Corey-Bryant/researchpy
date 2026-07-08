# -*- coding: utf-8 -*-
"""
Tests for the researchpy.statistics subpackage.

Validates:
- All 5 calling conventions
- All 3 output layouts (marginal, cell, pivot)
- Real group variable names in output
- Statistical correctness against scipy/numpy
"""

import pytest
import numpy as np
import pandas as pd
import scipy.stats

from researchpy.statistics import (
    n_obs,
    n_missing,
    percent_missing,
    mean,
    median,
    mode,
    quartiles,
    percentile,
    iqr,
    variance,
    standard_deviation,
    standard_error,
    value_range,
    coefficient_of_variation,
    confidence_interval,
    skewness,
    kurtosis,
    estable,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_series():
    return pd.Series([1, 2, 3, 4, 5], name="test_var")


@pytest.fixture
def series_with_nan():
    return pd.Series([1, 2, np.nan, 4, 5], name="test_nan")


@pytest.fixture
def grouped_df():
    """DataFrame with two grouping variables for testing all layouts."""
    return pd.DataFrame({
        "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "disease": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "A", "B", "C"],
        "drug": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
    })


# ============================================================================
# Convention 1: Series/Array (scalar output)
# ============================================================================

class TestConvention1Scalar:
    """Test Convention 1: stat(series_or_array) → scalar."""

    def test_n_obs_series(self, simple_series):
        assert n_obs(simple_series) == 5

    def test_n_obs_with_nan(self, series_with_nan):
        assert n_obs(series_with_nan) == 4

    def test_n_obs_list(self):
        assert n_obs([1, 2, 3]) == 3

    def test_mean_series(self, simple_series):
        assert mean(simple_series) == 3.0

    def test_mean_with_nan(self, series_with_nan):
        assert mean(series_with_nan) == 3.0

    def test_mean_matches_numpy(self, simple_series):
        expected = float(np.nanmean(simple_series.to_numpy()))
        assert mean(simple_series) == round(expected, 4)

    def test_median_series(self, simple_series):
        assert median(simple_series) == 3.0

    def test_median_even(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_variance_series(self, simple_series):
        expected = float(np.nanvar(simple_series.to_numpy(), ddof=1))
        assert variance(simple_series) == round(expected, 4)

    def test_standard_deviation_series(self, simple_series):
        expected = float(np.nanstd(simple_series.to_numpy(), ddof=1))
        assert standard_deviation(simple_series) == round(expected, 4)

    def test_standard_error_series(self, simple_series):
        expected = float(scipy.stats.sem(simple_series.to_numpy(), nan_policy='omit'))
        assert standard_error(simple_series) == round(expected, 4)

    def test_skewness_symmetric(self):
        result = skewness([1, 2, 3, 4, 5])
        assert abs(result) < 0.01  # Symmetric data has ~0 skewness

    def test_skewness_matches_scipy(self, simple_series):
        expected = float(scipy.stats.skew(simple_series.to_numpy(), nan_policy='omit'))
        assert skewness(simple_series) == round(expected, 4)

    def test_kurtosis_pearson_default(self, simple_series):
        expected = float(scipy.stats.kurtosis(simple_series.to_numpy(), fisher=False, nan_policy='omit'))
        assert kurtosis(simple_series) == round(expected, 4)

    def test_kurtosis_fisher(self, simple_series):
        expected = float(scipy.stats.kurtosis(simple_series.to_numpy(), fisher=True, nan_policy='omit'))
        assert kurtosis(simple_series, fisher=True) == round(expected, 4)

    def test_confidence_interval_basic(self, simple_series):
        lower, upper = confidence_interval(simple_series)
        assert lower < 3.0 < upper  # CI should contain the mean

    def test_confidence_interval_99(self, simple_series):
        lower_95, upper_95 = confidence_interval(simple_series, confidence_level=0.95)
        lower_99, upper_99 = confidence_interval(simple_series, confidence_level=0.99)
        # 99% CI should be wider than 95% CI
        assert lower_99 < lower_95
        assert upper_99 > upper_95


# ============================================================================
# Convention 2: Formula string
# ============================================================================

class TestConvention2Formula:
    """Test Convention 2: stat('y ~ C(x)', df) → DataFrame with group names."""

    def test_mean_formula_single_factor(self, grouped_df):
        result = mean("y ~ C(disease)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "disease" in result.index.names
        assert "Mean" in result.columns

    def test_mean_formula_values_correct(self, grouped_df):
        result = mean("y ~ C(disease)", grouped_df)
        # Group A: [10, 20, 30, 100] → mean = 40
        assert result.loc["A", "Mean"] == 40.0

    def test_n_obs_formula(self, grouped_df):
        result = n_obs("y ~ C(disease)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "disease" in result.index.names
        # Each group has 4 observations
        assert (result["N"] == 4).all()

    def test_variance_formula(self, grouped_df):
        result = variance("y ~ C(disease)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "disease" in result.index.names
        assert "Variance" in result.columns


# ============================================================================
# Convention 3: Column list
# ============================================================================

class TestConvention3ColumnList:
    """Test Convention 3: stat(['y', 'z'], df) → DataFrame multi-DV."""

    def test_mean_column_list(self, grouped_df):
        # Add a second numeric column
        df = grouped_df.copy()
        df["z"] = df["y"] * 2
        result = mean(["y", "z"], df)
        assert isinstance(result, pd.DataFrame)
        assert "Variable" in result.columns
        assert "Mean" in result.columns
        assert len(result) == 2

    def test_mean_column_list_values(self, grouped_df):
        df = grouped_df.copy()
        df["z"] = df["y"] * 2
        result = mean(["y", "z"], df)
        y_mean = result[result["Variable"] == "y"]["Mean"].iloc[0]
        z_mean = result[result["Variable"] == "z"]["Mean"].iloc[0]
        assert z_mean == y_mean * 2


# ============================================================================
# Convention 4: Keywords (by, iv, over)
# ============================================================================

class TestConvention4Keywords:
    """Test Convention 4: stat(dv=, by=, iv=, over=, data=)."""

    def test_mean_by_single(self, grouped_df):
        """by= produces cell means with MultiIndex."""
        result = mean(dv="y", by="disease", data=grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "disease" in result.index.names
        assert "Mean" in result.columns

    def test_mean_by_multi(self, grouped_df):
        """Multiple by= produces MultiIndex."""
        result = mean(dv="y", by=["disease", "drug"], data=grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["disease", "drug"]

    def test_mean_iv_marginal(self, grouped_df):
        """iv= produces marginal (stacked) results."""
        result = mean(dv="y", iv=["disease", "drug"], data=grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        # Should have disease levels (A, B, C) + drug levels (x, y) = 5 rows
        assert len(result) == 5
        assert set(result["Factor"].unique()) == {"disease", "drug"}

    def test_mean_over_pivot(self, grouped_df):
        """by= + over= produces pivot table."""
        result = mean(dv="y", by="disease", over="drug", data=grouped_df)
        assert isinstance(result, pd.DataFrame)
        # Rows = disease levels, columns = drug levels
        assert "disease" in result.index.names or result.index.name == "disease"
        assert "x" in result.columns and "y" in result.columns

    def test_iv_xor_by_raises(self, grouped_df):
        """iv and by/over cannot be used together."""
        with pytest.raises(ValueError, match="Cannot use 'iv' together with 'by'"):
            mean(dv="y", iv="disease", by="drug", data=grouped_df)

    def test_over_requires_by_raises(self, grouped_df):
        """over requires by."""
        with pytest.raises(ValueError, match="'over' requires 'by'"):
            mean(dv="y", over="drug", data=grouped_df)


# ============================================================================
# Convention 5: Formula with operators (+, :, *)
# ============================================================================

class TestConvention5FormulaOperators:
    """Test Convention 5: formula with +, :, * operators."""

    def test_formula_plus_marginal(self, grouped_df):
        """+ produces marginal (stacked) results."""
        result = mean("y ~ C(disease) + C(drug)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        assert len(result) == 5  # 3 disease + 2 drug

    def test_formula_colon_cell(self, grouped_df):
        """': produces cell means with MultiIndex."""
        result = mean("y ~ C(disease):C(drug)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        assert "disease" in result.index.names
        assert "drug" in result.index.names

    def test_formula_star_pivot(self, grouped_df):
        """* produces pivot table."""
        result = mean("y ~ C(disease)*C(drug)", grouped_df)
        assert isinstance(result, pd.DataFrame)
        # Should be a pivot: rows=disease, columns=drug
        assert "disease" in result.index.names or result.index.name == "disease"
        assert "x" in result.columns and "y" in result.columns


# ============================================================================
# GroupBy convention
# ============================================================================

class TestGroupByConvention:
    """Test GroupBy object as input with real group names."""

    def test_groupby_series_real_names(self, grouped_df):
        """GroupBy SeriesGroupBy preserves actual group variable name."""
        result = estable(grouped_df.groupby("disease")["y"], stats=["N", "Mean"])
        assert "disease" in result.columns
        assert "N" in result.columns
        assert "Mean" in result.columns

    def test_groupby_multi_real_names(self, grouped_df):
        """Multi-level GroupBy preserves all group variable names."""
        result = estable(
            grouped_df.groupby(["disease", "drug"])["y"], stats=["N", "Mean"]
        )
        assert "disease" in result.columns
        assert "drug" in result.columns

    def test_groupby_values_correct(self, grouped_df):
        """GroupBy results match manual computation."""
        result = estable(grouped_df.groupby("disease")["y"], stats=["N", "Mean"])
        row_a = result[result["disease"] == "A"]
        assert row_a["N"].iloc[0] == 4
        assert row_a["Mean"].iloc[0] == 40.0


# ============================================================================
# estable() dispatcher
# ============================================================================

class TestEstable:
    """Test the estable() unified dispatcher."""

    def test_series_default_stats(self, simple_series):
        result = estable(simple_series)
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert result["Name"].iloc[0] == "test_var"

    def test_series_custom_stats(self, simple_series):
        result = estable(simple_series, stats=["N", "Mean", "SD"])
        assert "N" in result.columns
        assert "Mean" in result.columns
        assert "SD" in result.columns

    def test_formula_grouped(self, grouped_df):
        result = estable("y ~ C(disease)", grouped_df, stats=["N", "Mean"])
        assert "disease" in result.columns
        assert len(result) == 3

    def test_iv_marginal_estable(self, grouped_df):
        result = estable(dv="y", iv=["disease", "drug"], data=grouped_df, stats=["N", "Mean"])
        assert "Factor" in result.columns
        assert "Level" in result.columns
        # disease (A,B,C) + drug (x,y) = 5 rows
        assert len(result) == 5

    def test_return_type_dictionary(self, simple_series):
        result = estable(simple_series, stats=["N", "Mean"], return_type="Dictionary")
        assert isinstance(result, dict)

    def test_invalid_return_type_raises(self, simple_series):
        with pytest.raises(ValueError, match="Unsupported return_type"):
            estable(simple_series, return_type="xml")

    def test_invalid_stat_name_raises(self, simple_series):
        with pytest.raises(ValueError, match="Unknown statistic"):
            estable(simple_series, stats=["NotARealStat"])


# ============================================================================
# Statistical correctness
# ============================================================================

class TestStatisticalCorrectness:
    """Validate statistical results against scipy/numpy."""

    def test_grouped_mean_matches_pandas(self, grouped_df):
        """Grouped mean from matrix engine matches pandas groupby."""
        result = mean(dv="y", by="disease", data=grouped_df)
        expected = grouped_df.groupby("disease")["y"].mean()
        for grp in ["A", "B", "C"]:
            assert abs(result.loc[grp, "Mean"] - expected[grp]) < 0.001

    def test_grouped_variance_matches_pandas(self, grouped_df):
        """Grouped variance from matrix engine matches pandas groupby."""
        result = variance(dv="y", by="disease", data=grouped_df)
        expected = grouped_df.groupby("disease")["y"].var(ddof=1)
        for grp in ["A", "B", "C"]:
            assert abs(result.loc[grp, "Variance"] - expected[grp]) < 0.01

    def test_grouped_sd_matches_pandas(self, grouped_df):
        """Grouped SD from matrix engine matches pandas groupby."""
        result = standard_deviation(dv="y", by="disease", data=grouped_df)
        expected = grouped_df.groupby("disease")["y"].std(ddof=1)
        for grp in ["A", "B", "C"]:
            assert abs(result.loc[grp, "SD"] - expected[grp]) < 0.01

    def test_grouped_count_matches_pandas(self, grouped_df):
        """Grouped count from matrix engine matches pandas groupby."""
        result = n_obs(dv="y", by="disease", data=grouped_df)
        expected = grouped_df.groupby("disease")["y"].count()
        for grp in ["A", "B", "C"]:
            assert result.loc[grp, "N"] == expected[grp]

    def test_pivot_values_correct(self, grouped_df):
        """Pivot table values match manual computation."""
        result = mean(dv="y", by="disease", over="drug", data=grouped_df)
        # Group A, drug x: [10, 30] → mean = 20
        # Group A, drug y: [20, 100] → mean = 60
        assert result.loc["A", "x"] == 20.0
        assert result.loc["A", "y"] == 60.0


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_n_missing(self):
        assert n_missing([1, 2, np.nan, 4]) == 1
        assert n_missing([1, 2, 3]) == 0

    def test_percent_missing(self):
        assert percent_missing([1, np.nan, 3, np.nan]) == 50.0

    def test_ci_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            confidence_interval([5.0])

    def test_ci_invalid_level_raises(self):
        with pytest.raises(ValueError, match="confidence_level must be between"):
            confidence_interval([1, 2, 3], confidence_level=1.5)

    def test_mode_unimodal(self):
        assert mode([1, 2, 2, 3]) == 2.0

    def test_mode_multimodal(self):
        result = mode([1, 1, 2, 2, 3])
        assert isinstance(result, list)
        assert 1.0 in result and 2.0 in result

    def test_quartiles(self):
        result = quartiles([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)
        assert "Q1" in result.columns and "Q2" in result.columns and "Q3" in result.columns
        assert result["Q2"].iloc[0] == 3.0

    def test_iqr(self):
        result = iqr([1, 2, 3, 4, 5])
        assert result == 2.0

    def test_value_range(self):
        assert value_range([1, 2, 3, 4, 5]) == 4.0

    def test_cv_zero_mean_raises(self):
        with pytest.raises(ValueError, match="mean is zero"):
            coefficient_of_variation([-1, 0, 1])

    def test_missing_column_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            mean(dv="nonexistent", by="disease", data=grouped_df)

