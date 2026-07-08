# -*- coding: utf-8 -*-
"""
Tests for Phase 3: Extended quartiles, percentile, and iqr functions.

Validates:
- Ungrouped: single-row wide DataFrame output
- Grouped (formula, keywords): wide DataFrame with group columns
- Marginal (iv): stacked with Factor/Level columns
- Statistical correctness against numpy
- Edge cases (NaN handling, validation errors)
"""

import pytest
import numpy as np
import pandas as pd

from researchpy.statistics import quartiles, percentile, iqr


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_df():
    """Simple DataFrame for basic grouped tests."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "a", "b", "b", "b"],
    })


@pytest.fixture
def grouped_df():
    """DataFrame with multiple grouping variables."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "k": ["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"],
    })


# ============================================================================
# quartiles() — ungrouped
# ============================================================================

class TestQuartilesUngrouped:
    """Tests for quartiles without grouping."""

    def test_returns_dataframe(self):
        result = quartiles([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self):
        result = quartiles([1, 2, 3, 4, 5])
        assert len(result) == 1

    def test_columns(self):
        result = quartiles([1, 2, 3, 4, 5])
        assert list(result.columns) == ["Q1", "Q2", "Q3"]

    def test_values_match_numpy(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = quartiles(data)
        expected_q1 = np.percentile(data, 25)
        expected_q2 = np.percentile(data, 50)
        expected_q3 = np.percentile(data, 75)
        assert result["Q1"].iloc[0] == round(expected_q1, 4)
        assert result["Q2"].iloc[0] == round(expected_q2, 4)
        assert result["Q3"].iloc[0] == round(expected_q3, 4)

    def test_series_input(self):
        s = pd.Series([1, 2, 3, 4, 5], name="test")
        result = quartiles(s)
        assert isinstance(result, pd.DataFrame)
        assert result["Q2"].iloc[0] == 3.0

    def test_handles_nan(self):
        result = quartiles([1, 2, np.nan, 4, 5])
        # Q2 of [1, 2, 4, 5] = 3.0
        assert result["Q2"].iloc[0] == 3.0

    def test_multiple_dvs(self, grouped_df):
        result = quartiles(["y"], grouped_df)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# quartiles() — grouped
# ============================================================================

class TestQuartilesGrouped:
    """Tests for quartiles with grouping."""

    def test_formula_single_factor(self, simple_df):
        result = quartiles("y ~ C(g)", simple_df)
        assert "g" in result.columns
        assert "Q1" in result.columns
        assert "Q2" in result.columns
        assert "Q3" in result.columns
        assert len(result) == 2  # groups a, b

    def test_formula_values(self, simple_df):
        result = quartiles("y ~ C(g)", simple_df)
        # group a: [1, 2, 3], Q2 = 2.0
        # group b: [4, 5, 6], Q2 = 5.0
        a_row = result[result["g"] == "a"]
        b_row = result[result["g"] == "b"]
        assert a_row["Q2"].iloc[0] == 2.0
        assert b_row["Q2"].iloc[0] == 5.0

    def test_keyword_by(self, simple_df):
        result = quartiles(dv="y", by="g", data=simple_df)
        assert "g" in result.columns
        assert len(result) == 2

    def test_keyword_iv_marginal(self, grouped_df):
        result = quartiles(dv="y", iv=["x", "k"], data=grouped_df)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        assert "Q1" in result.columns

    def test_multi_group_interaction(self, grouped_df):
        result = quartiles("y ~ C(x):C(k)", grouped_df)
        assert "x" in result.columns
        assert "k" in result.columns
        assert len(result) == 4  # 2x2 cells

    def test_star_expansion(self, grouped_df):
        result = quartiles("y ~ C(x)*C(k)", grouped_df)
        # Pivot output: groups = by+over
        assert "x" in result.columns
        assert "k" in result.columns


# ============================================================================
# percentile() — ungrouped
# ============================================================================

class TestPercentileUngrouped:
    """Tests for percentile without grouping."""

    def test_single_percentile_returns_dataframe(self):
        result = percentile([1, 2, 3, 4, 5], q=50)
        assert isinstance(result, pd.DataFrame)
        assert "P50" in result.columns
        assert result["P50"].iloc[0] == 3.0

    def test_multiple_percentiles(self):
        result = percentile([1, 2, 3, 4, 5], q=[25, 50, 75])
        assert list(result.columns) == ["P25", "P50", "P75"]
        assert result["P50"].iloc[0] == 3.0

    def test_values_match_numpy(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = percentile(data, q=[10, 25, 50, 75, 90])
        for q_val in [10, 25, 50, 75, 90]:
            col = f"P{q_val}"
            expected = round(float(np.percentile(data, q_val)), 4)
            assert result[col].iloc[0] == expected

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1, 2, 3], q=101)

    def test_invalid_percentile_in_list_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1, 2, 3], q=[25, -5, 75])


# ============================================================================
# percentile() — grouped
# ============================================================================

class TestPercentileGrouped:
    """Tests for percentile with grouping."""

    def test_formula_grouped(self, simple_df):
        result = percentile("y ~ C(g)", simple_df, q=[25, 50, 75])
        assert "g" in result.columns
        assert "P25" in result.columns
        assert "P50" in result.columns
        assert "P75" in result.columns
        assert len(result) == 2

    def test_keyword_by(self, simple_df):
        result = percentile(dv="y", by="g", data=simple_df, q=[10, 90])
        assert "g" in result.columns
        assert "P10" in result.columns
        assert "P90" in result.columns

    def test_grouped_values(self, simple_df):
        result = percentile("y ~ C(g)", simple_df, q=50)
        # group a: [1,2,3] → P50=2.0; group b: [4,5,6] → P50=5.0
        a_row = result[result["g"] == "a"]
        b_row = result[result["g"] == "b"]
        assert a_row["P50"].iloc[0] == 2.0
        assert b_row["P50"].iloc[0] == 5.0


# ============================================================================
# iqr() — ungrouped
# ============================================================================

class TestIqrUngrouped:
    """Tests for iqr without grouping."""

    def test_returns_float(self):
        result = iqr([1, 2, 3, 4, 5])
        assert isinstance(result, float)

    def test_value_correct(self):
        # Q1=2.0, Q3=4.0, IQR=2.0
        assert iqr([1, 2, 3, 4, 5]) == 2.0

    def test_series_input(self):
        result = iqr(pd.Series([1, 2, 3, 4, 5]))
        assert result == 2.0

    def test_handles_nan(self):
        result = iqr([1, 2, np.nan, 4, 5])
        # [1, 2, 4, 5]: Q1=1.75, Q3=4.25, IQR=2.5
        expected = np.percentile([1, 2, 4, 5], 75) - np.percentile([1, 2, 4, 5], 25)
        assert result == round(expected, 4)


# ============================================================================
# iqr() — grouped
# ============================================================================

class TestIqrGrouped:
    """Tests for iqr with grouping."""

    def test_formula_grouped(self, simple_df):
        result = iqr("y ~ C(g)", simple_df)
        assert result.index.name == "g"
        assert "IQR" in result.columns

    def test_keyword_by(self, simple_df):
        result = iqr(dv="y", by="g", data=simple_df)
        assert result.index.name == "g"
        assert "IQR" in result.columns

    def test_grouped_values(self, simple_df):
        result = iqr("y ~ C(g)", simple_df)
        # group a: [1,2,3] → Q1=1.5, Q3=2.5, IQR=1.0
        # group b: [4,5,6] → Q1=4.5, Q3=5.5, IQR=1.0
        assert result.loc["a", "IQR"] == 1.0
        assert result.loc["b", "IQR"] == 1.0

    def test_marginal(self, grouped_df):
        result = iqr("y ~ C(x) + C(k)", grouped_df)
        assert "Factor" in result.columns
        assert "Level" in result.columns
        assert "IQR" in result.columns

    def test_pivot(self, grouped_df):
        result = iqr("y ~ C(x)*C(k)", grouped_df)
        # Pivot: rows=x, cols=k
        assert result.index.name == "x"

