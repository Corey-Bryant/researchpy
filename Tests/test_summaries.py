# -*- coding: utf-8 -*-
"""
Tests for the researchpy.descriptive subpackage.

Validates against scipy/numpy known results.
"""

import pytest
import numpy as np
import pandas as pd
import scipy.stats

from researchpy.descriptive import (
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
    SummaryResult,
    summarize,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_series():
    return pd.Series([1, 2, 3, 4, 5], name="test_var")


@pytest.fixture
def series_with_nan():
    return pd.Series([1, 2, np.nan, 4, 5], name="has_nan")


@pytest.fixture
def simple_df():
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})


@pytest.fixture
def grouped_series():
    df = pd.DataFrame({"group": ["a", "a", "a", "b", "b", "b"], "val": [1, 2, 3, 4, 5, 6]})
    return df.groupby("group")["val"]


# ---------------------------------------------------------------------------
# Observation module tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_n_obs_series(self, simple_series):
        assert n_obs(simple_series) == 5

    def test_n_obs_with_nan(self, series_with_nan):
        assert n_obs(series_with_nan) == 4

    def test_n_obs_list(self):
        assert n_obs([1.0, 2.0, 3.0]) == 3

    def test_n_missing_no_nans(self, simple_series):
        assert n_missing(simple_series) == 0

    def test_n_missing_with_nans(self, series_with_nan):
        assert n_missing(series_with_nan) == 1

    def test_percent_missing_no_nans(self, simple_series):
        assert percent_missing(simple_series) == 0.0

    def test_percent_missing_with_nans(self, series_with_nan):
        assert percent_missing(series_with_nan) == 20.0


# ---------------------------------------------------------------------------
# Central tendency module tests
# ---------------------------------------------------------------------------

class TestCentralTendency:
    def test_mean_simple(self, simple_series):
        assert mean(simple_series) == 3.0

    def test_mean_with_nan(self, series_with_nan):
        # (1 + 2 + 4 + 5) / 4 = 3.0
        assert mean(series_with_nan) == 3.0

    def test_mean_matches_numpy(self, simple_series):
        expected = float(np.nanmean(simple_series.values))
        assert mean(simple_series) == expected

    def test_median_simple(self, simple_series):
        assert median(simple_series) == 3.0

    def test_median_even(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_mode_unimodal(self):
        assert mode([1, 2, 2, 3]) == 2.0

    def test_mode_multimodal(self):
        result = mode([1, 1, 2, 2, 3])
        assert result == [1.0, 2.0]

    def test_quartiles_simple(self, simple_series):
        result = quartiles(simple_series)
        assert "Q1" in result
        assert "Q2" in result
        assert "Q3" in result
        assert result["Q2"] == 3.0  # Median

    def test_percentile_scalar(self, simple_series):
        result = percentile(simple_series, q=50)
        assert result == 3.0

    def test_percentile_list(self, simple_series):
        result = percentile(simple_series, q=[25, 50, 75])
        assert "P25" in result
        assert "P50" in result
        assert "P75" in result

    def test_percentile_out_of_range(self, simple_series):
        with pytest.raises(ValueError):
            percentile(simple_series, q=101)

    def test_iqr_simple(self, simple_series):
        q1, q3 = np.nanpercentile(simple_series.values, [25, 75])
        expected = float(q3 - q1)
        assert iqr(simple_series) == expected


# ---------------------------------------------------------------------------
# Dispersion module tests
# ---------------------------------------------------------------------------

class TestDispersion:
    def test_variance_simple(self, simple_series):
        expected = float(np.nanvar(simple_series.values, ddof=1))
        assert variance(simple_series) == expected

    def test_variance_matches_pandas(self, simple_series):
        expected = float(simple_series.var())
        assert abs(variance(simple_series) - expected) < 1e-10

    def test_standard_deviation_simple(self, simple_series):
        expected = float(np.nanstd(simple_series.values, ddof=1))
        assert abs(standard_deviation(simple_series) - expected) < 1e-10

    def test_standard_error_simple(self, simple_series):
        expected = float(scipy.stats.sem(simple_series.values, nan_policy='omit'))
        assert abs(standard_error(simple_series) - expected) < 1e-10

    def test_value_range_simple(self, simple_series):
        assert value_range(simple_series) == 4.0

    def test_coefficient_of_variation(self):
        data = [10, 10, 10, 10, 10]
        assert coefficient_of_variation(data) == 0.0

    def test_cv_zero_mean_raises(self):
        with pytest.raises(ValueError, match="undefined when the mean is zero"):
            coefficient_of_variation([-1, 0, 1])


# ---------------------------------------------------------------------------
# Intervals module tests
# ---------------------------------------------------------------------------

class TestIntervals:
    def test_confidence_interval_basic(self, simple_series):
        lower, upper = confidence_interval(simple_series, confidence_level=0.95)
        # Validate against scipy directly
        n = 5
        se = scipy.stats.sem(simple_series.values, nan_policy='omit')
        expected_lower, expected_upper = scipy.stats.t.interval(
            0.95, n - 1, loc=np.mean(simple_series.values), scale=se
        )
        assert abs(lower - round(expected_lower, 4)) < 1e-10
        assert abs(upper - round(expected_upper, 4)) < 1e-10

    def test_confidence_interval_99(self, simple_series):
        lower_95, upper_95 = confidence_interval(simple_series, confidence_level=0.95)
        lower_99, upper_99 = confidence_interval(simple_series, confidence_level=0.99)
        # 99% CI should be wider than 95% CI
        assert lower_99 < lower_95
        assert upper_99 > upper_95

    def test_confidence_interval_invalid_level(self, simple_series):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            confidence_interval(simple_series, confidence_level=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            confidence_interval(simple_series, confidence_level=0.0)

    def test_confidence_interval_too_few_obs(self):
        with pytest.raises(ValueError, match="At least 2"):
            confidence_interval([5.0])


# ---------------------------------------------------------------------------
# Shape module tests
# ---------------------------------------------------------------------------

class TestShape:
    def test_skewness_symmetric(self):
        # Symmetric data should have ~0 skew
        data = [1, 2, 3, 4, 5]
        assert abs(skewness(data)) < 1e-10

    def test_skewness_right_skewed(self):
        data = [1, 2, 3, 4, 100]
        assert skewness(data) > 0

    def test_skewness_matches_scipy(self, simple_series):
        expected = float(scipy.stats.skew(simple_series.values, nan_policy='omit'))
        assert abs(skewness(simple_series) - expected) < 1e-10

    def test_kurtosis_pearson_default(self, simple_series):
        expected = float(scipy.stats.kurtosis(simple_series.values, fisher=False, nan_policy='omit'))
        assert abs(kurtosis(simple_series) - expected) < 1e-10

    def test_kurtosis_fisher(self, simple_series):
        expected = float(scipy.stats.kurtosis(simple_series.values, fisher=True, nan_policy='omit'))
        assert abs(kurtosis(simple_series, fisher=True) - expected) < 1e-10


# ---------------------------------------------------------------------------
# UnivariateSummary tests
# ---------------------------------------------------------------------------

class TestSummaryResult:
    def test_to_dataframe(self):
        result = SummaryResult(name="age", statistics={"N": 100, "Mean": 35.4})
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["Name"] == "age"
        assert df.iloc[0]["N"] == 100
        assert df.iloc[0]["Mean"] == 35.4

    def test_to_dict(self):
        result = SummaryResult(name="age", statistics={"N": 100, "Mean": 35.4})
        d = result.to_dict()
        assert d["Name"] == "age"
        assert d["N"] == 100
        assert d["Mean"] == 35.4

    def test_no_name(self):
        result = SummaryResult(statistics={"N": 50})
        df = result.to_dataframe()
        assert "Name" not in df.columns


# ---------------------------------------------------------------------------
# Dispatcher (summarize) tests
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_series_default_stats(self, simple_series):
        result = summarize(simple_series)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["Name"] == "test_var"
        assert result.iloc[0]["N"] == 5
        assert result.iloc[0]["Mean"] == 3.0

    def test_series_custom_stats(self, simple_series):
        result = summarize(simple_series, stats=["N", "Mean", "SD"])
        assert list(result.columns) == ["Name", "N", "Mean", "SD"]

    def test_series_name_override(self, simple_series):
        result = summarize(simple_series, name="custom_name")
        assert result.iloc[0]["Name"] == "custom_name"

    def test_dataframe(self, simple_df):
        result = summarize(simple_df, stats=["N", "Mean"])
        assert len(result) == 2
        assert result.iloc[0]["Name"] == "A"
        assert result.iloc[1]["Name"] == "B"
        assert result.iloc[0]["Mean"] == 3.0
        assert result.iloc[1]["Mean"] == 8.0

    def test_groupby_series(self, grouped_series):
        result = summarize(grouped_series, stats=["N", "Mean"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Group "a" has values [1, 2, 3] -> mean = 2.0
        row_a = result[result["Group"] == "a"].iloc[0]
        assert row_a["N"] == 3
        assert row_a["Mean"] == 2.0
        # Group "b" has values [4, 5, 6] -> mean = 5.0
        row_b = result[result["Group"] == "b"].iloc[0]
        assert row_b["N"] == 3
        assert row_b["Mean"] == 5.0

    def test_return_type_dictionary(self, simple_series):
        result = summarize(simple_series, stats=["N", "Mean"], return_type="Dictionary")
        assert isinstance(result, dict)
        assert result["N"] == 5
        assert result["Mean"] == 3.0

    def test_invalid_return_type(self, simple_series):
        with pytest.raises(ValueError, match="Unsupported return_type"):
            summarize(simple_series, return_type="XML")

    def test_invalid_stat_name(self, simple_series):
        with pytest.raises(ValueError, match="Unknown statistic"):
            summarize(simple_series, stats=["N", "InvalidStat"])

    def test_list_input(self):
        result = summarize([1, 2, 3, 4, 5], name="my_list", stats=["N", "Mean"])
        assert result.iloc[0]["Name"] == "my_list"
        assert result.iloc[0]["N"] == 5
        assert result.iloc[0]["Mean"] == 3.0

    def test_with_nan(self, series_with_nan):
        result = summarize(series_with_nan, stats=["N", "N Missing", "Mean"])
        assert result.iloc[0]["N"] == 4
        assert result.iloc[0]["N Missing"] == 1
        assert result.iloc[0]["Mean"] == 3.0

    def test_ci_in_output(self, simple_series):
        result = summarize(simple_series, stats=["CI"], ci_level=0.95)
        col_name = "95% Conf. Interval"
        assert col_name in result.columns
        ci_val = result.iloc[0][col_name]
        assert isinstance(ci_val, tuple)
        assert len(ci_val) == 2
        assert ci_val[0] < ci_val[1]


# ---------------------------------------------------------------------------
# Validation tests (edge cases)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="no valid"):
            summarize(pd.Series([np.nan, np.nan, np.nan]))

    def test_single_value(self):
        # Should work for stats that don't require > 1 obs
        result = summarize([5.0], stats=["N", "Mean", "Min", "Max"])
        assert result.iloc[0]["N"] == 1
        assert result.iloc[0]["Mean"] == 5.0

    def test_single_value_ci_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            summarize([5.0], stats=["CI"])

    def test_large_dataset(self):
        np.random.seed(42)
        data = np.random.normal(100, 15, size=10000)
        result = summarize(data, name="large", stats=["N", "Mean", "SD"])
        assert result.iloc[0]["N"] == 10000
        # Mean should be close to 100
        assert abs(result.iloc[0]["Mean"] - 100) < 1.0
        # SD should be close to 15
        assert abs(result.iloc[0]["SD"] - 15) < 1.0

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            summarize(pd.Series(["a", "b", "c"]), stats=["N"])

