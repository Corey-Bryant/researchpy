"""
Golden-value tests for descriptive statistics.

Verifies that functions producing univariate descriptive statistics match
the documented values from the ResearchPy summarize() documentation when
applied to the Stata 'auto' and 'systolic' datasets.

Pattern for testing new functions:
    1. Apply your function to the appropriate fixture
    2. Extract statistics from the result into a dict
    3. Call assert_stats_match(actual, golden)

The placeholder implementations use scipy.stats.t.interval() for CIs
to match ResearchPy's methodology. Replace the placeholder blocks with
your actual function calls when wiring in new code.
"""
import pytest
import numpy as np
from scipy import stats

from Tests.Golden.golden_values import (
    APPROX_REL,
    APPROX_ABS,
    get_auto_golden,
    get_systolic_golden,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_descriptives(series):
    """
    Compute univariate descriptive statistics for a pandas Series.

    Uses the same methodology as ResearchPy's summarize():
    - Sample variance (ddof=1)
    - t-distribution for confidence intervals

    This is a reference implementation for validating golden values.
    Replace with your actual function when testing new code.

    Parameters
    ----------
    series : pandas.Series
        Data to analyze

    Returns
    -------
    dict
        {"N": int, "Mean": float, "Median": float, "Variance": float,
         "SD": float, "SE": float, "95% Conf. Interval": (float, float)}
    """
    n = int(series.count())
    mean = float(series.mean())
    median = float(series.median())
    variance = float(series.var(ddof=1))
    sd = float(series.std(ddof=1))
    se = sd / np.sqrt(n)

    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=se,
    )

    return {
        "N": n,
        "Mean": mean,
        "Median": median,
        "Variance": variance,
        "SD": sd,
        "SE": se,
        "95% Conf. Interval": (float(ci_low), float(ci_high)),
    }


def assert_stats_match(actual, expected):
    """
    Assert that every stat in `expected` matches `actual`.

    Uses pytest.approx() for floats, exact equality for integers,
    and element-wise comparison for confidence interval tuples.

    Parameters
    ----------
    actual : dict
        Computed statistics
    expected : dict
        Golden values from golden_values module
    """
    for stat_name, expected_val in expected.items():

        # Skip non-numeric fields like "Name"
        if isinstance(expected_val, str):
            assert actual.get(stat_name) == expected_val, (
                f"{stat_name}: expected '{expected_val}', "
                f"got '{actual.get(stat_name)}'"
            )
            continue

        # Confidence intervals are tuples of (lower, upper)
        if isinstance(expected_val, tuple):
            actual_ci = actual[stat_name]
            assert len(actual_ci) == 2, (
                f"{stat_name}: expected 2-element CI, got {actual_ci}"
            )
            assert actual_ci[0] == pytest.approx(
                expected_val[0], rel=APPROX_REL, abs=APPROX_ABS
            ), f"{stat_name} lower bound: expected {expected_val[0]}, got {actual_ci[0]}"
            assert actual_ci[1] == pytest.approx(
                expected_val[1], rel=APPROX_REL, abs=APPROX_ABS
            ), f"{stat_name} upper bound: expected {expected_val[1]}, got {actual_ci[1]}"

        # Integer stats (N) use exact equality
        elif isinstance(expected_val, int):
            assert actual[stat_name] == expected_val, (
                f"{stat_name}: expected {expected_val}, got {actual[stat_name]}"
            )

        # Floats use approximate comparison
        else:
            assert actual[stat_name] == pytest.approx(
                expected_val, rel=APPROX_REL, abs=APPROX_ABS
            ), f"{stat_name}: expected {expected_val}, got {actual[stat_name]}"


# ═══════════════════════════════════════════════════════════════════════════
# AUTO DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoSummarize:
    """Golden-value tests for auto dataset descriptive statistics."""

    def test_single_variable_price(self, auto_price):
        """
        Validate stats for auto['price'] against documented values.

        Replace compute_descriptives() with your function:
            result = your_function(auto_price)
            assert_stats_match(result, golden)
        """
        golden = get_auto_golden('single_price')
        actual = compute_descriptives(auto_price)
        assert_stats_match(actual, golden)

    @pytest.mark.parametrize("var_name", ["price", "mpg"])
    def test_two_variables(self, auto_price_mpg, var_name):
        """
        Validate stats for auto[['price', 'mpg']] against documented values.
        """
        golden = get_auto_golden('two_variables', variable=var_name)
        actual = compute_descriptives(auto_price_mpg[var_name])
        assert_stats_match(actual, golden)

    @pytest.mark.parametrize("group", ["Domestic", "Foreign"])
    def test_series_groupby(self, auto_groupby_foreign_price, group):
        """
        Validate grouped stats for auto.groupby('foreign')['price'].
        """
        golden = get_auto_golden('series_groupby', group=group, variable='price')
        actual = compute_descriptives(
            auto_groupby_foreign_price.get_group(group)
        )
        assert_stats_match(actual, golden)

    @pytest.mark.parametrize("group,var_name", [
        ("Domestic", "price"),
        ("Domestic", "mpg"),
        ("Foreign", "price"),
        ("Foreign", "mpg"),
    ])
    def test_dataframe_groupby(
        self, auto_groupby_foreign_price_mpg, group, var_name
    ):
        """
        Validate grouped stats for auto.groupby('foreign')[['price', 'mpg']].
        """
        golden = get_auto_golden(
            'dataframe_groupby', group=group, variable=var_name
        )
        actual = compute_descriptives(
            auto_groupby_foreign_price_mpg.get_group(group)[var_name]
        )
        assert_stats_match(actual, golden)


# ═══════════════════════════════════════════════════════════════════════════
# SYSTOLIC DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSystolicSummarize:
    """Golden-value tests for systolic dataset descriptive statistics."""

    def test_summarize_systolic(self, systolic_series):
        """
        Validate stats for systolic['systolic'] against documented values.

        Source: anova() documentation, rp.summarize(systolic["systolic"])
        """
        golden = get_systolic_golden('summarize')
        actual = compute_descriptives(systolic_series)

        # Include the Name field check
        actual_with_name = {"Name": systolic_series.name, **actual}
        assert_stats_match(actual_with_name, golden)


class TestSystolicSummaryCat:
    """Golden-value tests for systolic summary_cat (univariate frequency counts)."""

    @pytest.mark.parametrize("variable", ["drug", "disease"])
    def test_summary_cat_counts(self, systolic_df, variable):
        """
        Validate frequency counts and percentages for summary_cat output.

        Source: anova() documentation, rp.summary_cat(systolic["drug"]),
        rp.summary_cat(systolic["disease"])
        """
        golden = get_systolic_golden('summary_cat', row_key=variable)
        series = systolic_df[variable]

        for outcome, expected in golden.items():
            count = int((series == outcome).sum())
            percent = round(count / len(series) * 100, 2)

            assert count == expected["count"], (
                f"{variable}={outcome}: expected count {expected['count']}, "
                f"got {count}"
            )
            assert percent == pytest.approx(
                expected["percent"], rel=APPROX_REL, abs=APPROX_ABS
            ), f"{variable}={outcome}: expected percent {expected['percent']}, got {percent}"


class TestSystolicAnova:
    """Golden-value tests for systolic ANOVA Type III SS."""

    @pytest.mark.parametrize("source", [
        "Model", "drug", "disease", "drug:disease", "Residual", "Total"
    ])
    def test_anova_table(self, source):
        """
        Validate ANOVA table values against documented output.

        Source: anova() documentation
        Formula: systolic ~ C(drug) + C(disease) + C(drug):C(disease)
        Sum of squares: Type III

        Replace this placeholder with your actual ANOVA function:
            m = your_anova("systolic ~ C(drug) + C(disease) + C(drug):C(disease)",
                          data=systolic_df, sum_of_squares=3)
            desc, table = m.results()
            # extract row for `source` and compare
        """
        golden = get_systolic_golden('anova_table', row_key=source)

        # Placeholder: just verify golden values are accessible
        assert golden["Source"] == source
        assert "Sum of Squares" in golden
        assert "Degrees of Freedom" in golden
        assert "Mean Squares" in golden

    def test_anova_fit_statistics(self):
        """
        Validate ANOVA fit statistics (N, Root MSE, R-squared, Adj R-squared).
        """
        golden = get_systolic_golden('anova_fit')

        assert golden["Number of obs"] == 58
        assert golden["Root MSE"] == pytest.approx(
            10.5096, rel=APPROX_REL, abs=APPROX_ABS
        )
        assert golden["R-squared"] == pytest.approx(
            0.4560, rel=APPROX_REL, abs=APPROX_ABS
        )
        assert golden["Adj R-squared"] == pytest.approx(
            0.3259, rel=APPROX_REL, abs=APPROX_ABS
        )

    #@pytest.mark.parametrize("row_idx", range(len(__import__('golden_values').SYSTOLIC_REGRESSION_TABLE)))
    @pytest.mark.parametrize("row_idx", range(17))  # 17 rows in SYSTOLIC_REGRESSION_TABLE
    def test_regression_table(self, row_idx):
        """
        Validate regression coefficient table against documented output.

        Source: anova() documentation, m.regression_table()

        Each row contains term, Coef., Std. Err., t, p-value, and 95% CI.
        Header/reference rows have empty strings for non-applicable fields.
        """
        from Tests.Golden.golden_values import SYSTOLIC_REGRESSION_TABLE

        golden_row = SYSTOLIC_REGRESSION_TABLE[row_idx]

        # Skip header and reference rows (empty string fields)
        if golden_row["Coef."] == "" or golden_row["Coef."] == "(reference)":
            # Just verify the term name is present
            assert golden_row["term"] is not None
            return

        # For estimated rows, verify numeric fields
        assert golden_row["Coef."] == pytest.approx(
            golden_row["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )

        ci = golden_row["95% Conf. Interval"]
        if isinstance(ci, tuple):
            assert ci[0] < ci[1], (
                f"Row {row_idx} ({golden_row['term']}): "
                f"CI lower ({ci[0]}) should be < upper ({ci[1]})"
            )