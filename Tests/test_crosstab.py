"""
Tests for researchpy.crosstab module.

Validates crosstab() output against golden reference values computed via
scipy.stats.chi2_contingency on the systolic dataset (drug × disease).

Cross-tabulation: systolic["drug"] (4 levels) × systolic["disease"] (3 levels)
"""
import warnings

import pytest
import pandas as pd
import numpy as np
import scipy.stats

from Tests.Golden.golden_values import (
    APPROX_REL,
    APPROX_ABS,
    SYSTOLIC_CROSSTAB,
    get_systolic_golden,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def drug_series(systolic_df):
    """systolic['drug'] as a pandas Series."""
    return systolic_df["drug"]


@pytest.fixture(scope="module")
def disease_series(systolic_df):
    """systolic['disease'] as a pandas Series."""
    return systolic_df["disease"]


@pytest.fixture(scope="module")
def crosstab_basic(drug_series, disease_series):
    """Basic crosstab result (no test, with margins)."""
    from researchpy.crosstab import crosstab
    return crosstab(drug_series, disease_series)


@pytest.fixture(scope="module")
def crosstab_chi2(drug_series, disease_series):
    """Crosstab with chi-square test."""
    from researchpy.crosstab import crosstab
    return crosstab(drug_series, disease_series, test="chi-square")


@pytest.fixture(scope="module")
def crosstab_gtest(drug_series, disease_series):
    """Crosstab with G-test."""
    from researchpy.crosstab import crosstab
    return crosstab(drug_series, disease_series, test="g-test")


@pytest.fixture(scope="module")
def crosstab_expected(drug_series, disease_series):
    """Crosstab with chi-square test and expected frequencies."""
    from researchpy.crosstab import crosstab
    return crosstab(drug_series, disease_series, test="chi-square", expected_freqs=True)


# ═══════════════════════════════════════════════════════════════════════════
# BASIC FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabBasic:
    """Test basic crosstab output structure and frequency counts."""

    def test_returns_dataframe(self, crosstab_basic):
        """Basic crosstab should return a DataFrame."""
        assert isinstance(crosstab_basic, pd.DataFrame)

    def test_margins_included_by_default(self, crosstab_basic):
        """Default crosstab should include margins (All row/column)."""
        # Margins add 'All' to index and columns
        assert "All" in crosstab_basic.index

    def test_no_margins(self, drug_series, disease_series):
        """margins=False should exclude totals."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, margins=False)
        assert "All" not in result.index

    def test_total_n(self, crosstab_basic):
        """Total N in margins should match golden value."""
        golden = SYSTOLIC_CROSSTAB
        # The 'All' row, last column should be total N
        all_row = crosstab_basic.loc["All"]
        assert all_row.iloc[-1] == golden["n"]

    def test_cell_counts(self, drug_series, disease_series):
        """Individual cell counts should match golden values."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, margins=False)
        golden = SYSTOLIC_CROSSTAB["contingency_table"]

        for drug_level, disease_counts in golden.items():
            for disease_level, expected_count in disease_counts.items():
                actual = result.loc[drug_level].iloc[disease_level - 1]
                assert actual == expected_count, (
                    f"drug={drug_level}, disease={disease_level}: "
                    f"expected {expected_count}, got {actual}"
                )

    def test_input_validation_non_series(self):
        """Non-Series input should return error message."""
        from researchpy.crosstab import crosstab
        result = crosstab([1, 2, 3], [4, 5, 6])
        assert result == "Operation only supports Pandas Series"


# ═══════════════════════════════════════════════════════════════════════════
# PROPORTIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabProportions:
    """Test proportion calculations (row, col, cell)."""

    def test_row_proportions_sum_to_100(self, drug_series, disease_series):
        """Row proportions should sum to ~100 for each row."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, prop="row")
        # Each row (excluding the last 'All' column which is 100) should sum to ~100
        for idx in result.index:
            row_sum = result.loc[idx].iloc[-1]
            assert row_sum == pytest.approx(100.0, abs=0.1), (
                f"Row '{idx}' last column should be 100, got {row_sum}"
            )

    def test_col_proportions_sum_to_100(self, drug_series, disease_series):
        """Column proportions should sum to ~100 for each column."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, prop="col")
        # The 'All' row should be 100 for each column
        all_row = result.loc["All"]
        for val in all_row:
            assert val == pytest.approx(100.0, abs=0.1), (
                f"Column total should be 100, got {val}"
            )

    def test_cell_proportions_sum_to_100(self, drug_series, disease_series):
        """Cell proportions should sum to ~100 across all cells."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, prop="cell")
        # The All/All cell should be 100
        total = result.loc["All"].iloc[-1]
        assert total == pytest.approx(100.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# CHI-SQUARE TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabChiSquare:
    """Validate chi-square test results against scipy golden values."""

    def test_returns_tuple_of_two(self, crosstab_chi2):
        """Chi-square test should return (crosstab_df, results_df)."""
        assert isinstance(crosstab_chi2, tuple)
        assert len(crosstab_chi2) == 2
        assert isinstance(crosstab_chi2[0], pd.DataFrame)
        assert isinstance(crosstab_chi2[1], pd.DataFrame)

    def test_chi_square_statistic(self, crosstab_chi2):
        """Chi-square statistic should match scipy golden value."""
        golden = SYSTOLIC_CROSSTAB["chi_square"]
        results_df = crosstab_chi2[1]
        # First row contains the chi-square value
        chi2_val = results_df["results"].iloc[0]
        assert chi2_val == pytest.approx(
            golden["chi2"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_chi_square_p_value(self, crosstab_chi2):
        """Chi-square p-value should match scipy golden value."""
        golden = SYSTOLIC_CROSSTAB["chi_square"]
        results_df = crosstab_chi2[1]
        # Second row contains p-value
        p_val = results_df["results"].iloc[1]
        assert p_val == pytest.approx(
            golden["p_value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_cramers_v(self, crosstab_chi2):
        """Cramer's V should match golden value for >2x2 table."""
        golden = SYSTOLIC_CROSSTAB["chi_square"]
        results_df = crosstab_chi2[1]
        # Third row contains Cramer's V (for tables larger than 2x2)
        v_val = results_df["results"].iloc[2]
        assert v_val == pytest.approx(
            golden["cramers_v"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_chi_square_against_scipy(self, drug_series, disease_series):
        """Cross-validate chi-square against direct scipy computation."""
        from researchpy.crosstab import crosstab
        ct_table, results_df = crosstab(drug_series, disease_series, test="chi-square")

        # Compute directly with scipy
        raw_ct = pd.crosstab(drug_series, disease_series)
        chi2_scipy, p_scipy, dof_scipy, _ = scipy.stats.chi2_contingency(
            raw_ct, correction=False
        )

        rp_chi2 = results_df["results"].iloc[0]
        rp_p = results_df["results"].iloc[1]

        assert rp_chi2 == pytest.approx(round(chi2_scipy, 4), rel=APPROX_REL, abs=APPROX_ABS)
        assert rp_p == pytest.approx(round(p_scipy, 4), rel=APPROX_REL, abs=APPROX_ABS)


# ═══════════════════════════════════════════════════════════════════════════
# G-TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabGTest:
    """Validate G-test (log-likelihood ratio) results."""

    def test_returns_tuple_of_two(self, crosstab_gtest):
        """G-test should return (crosstab_df, results_df)."""
        assert isinstance(crosstab_gtest, tuple)
        assert len(crosstab_gtest) == 2

    def test_g_test_statistic(self, crosstab_gtest):
        """G-test statistic should match scipy golden value."""
        golden = SYSTOLIC_CROSSTAB["g_test"]
        results_df = crosstab_gtest[1]
        g_val = results_df["results"].iloc[0]
        assert g_val == pytest.approx(
            golden["statistic"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_g_test_p_value(self, crosstab_gtest):
        """G-test p-value should match scipy golden value."""
        golden = SYSTOLIC_CROSSTAB["g_test"]
        results_df = crosstab_gtest[1]
        p_val = results_df["results"].iloc[1]
        assert p_val == pytest.approx(
            golden["p_value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_g_test_against_scipy(self, drug_series, disease_series):
        """Cross-validate G-test against direct scipy computation."""
        from researchpy.crosstab import crosstab
        _, results_df = crosstab(drug_series, disease_series, test="g-test")

        raw_ct = pd.crosstab(drug_series, disease_series)
        g_scipy, p_scipy, _, _ = scipy.stats.chi2_contingency(
            raw_ct, correction=False, lambda_="log-likelihood"
        )

        rp_g = results_df["results"].iloc[0]
        rp_p = results_df["results"].iloc[1]

        assert rp_g == pytest.approx(round(g_scipy, 4), rel=APPROX_REL, abs=APPROX_ABS)
        assert rp_p == pytest.approx(round(p_scipy, 4), rel=APPROX_REL, abs=APPROX_ABS)


# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED FREQUENCIES
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabExpectedFreqs:
    """Validate expected frequency output."""

    def test_returns_tuple_of_three(self, crosstab_expected):
        """expected_freqs=True with test should return 3-tuple."""
        assert isinstance(crosstab_expected, tuple)
        assert len(crosstab_expected) == 3

    def test_expected_frequencies_match_scipy(self, drug_series, disease_series):
        """Expected frequencies should match scipy output."""
        from researchpy.crosstab import crosstab
        _, _, expected_df = crosstab(
            drug_series, disease_series, test="chi-square", expected_freqs=True
        )

        golden = SYSTOLIC_CROSSTAB["expected_frequencies"]

        for i, row in enumerate(golden):
            for j, expected_val in enumerate(row):
                actual = expected_df.iloc[i, j]
                assert actual == pytest.approx(
                    expected_val, rel=APPROX_REL, abs=APPROX_ABS
                ), f"Expected freq [{i},{j}]: expected {expected_val}, got {actual}"

    def test_expected_freqs_without_test(self, drug_series, disease_series):
        """expected_freqs=True without test should return (ct, expected)."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, expected_freqs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════
# FISHER'S EXACT TEST (requires 2x2 table)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabFisher:
    """Validate Fisher's exact test with a 2x2 table."""

    @pytest.fixture
    def binary_data(self, systolic_df):
        """Create binary Series for a 2x2 table."""
        drug_binary = (systolic_df["drug"] <= 2).astype(int)
        drug_binary.name = "drug_binary"
        disease_binary = (systolic_df["disease"] <= 1).astype(int)
        disease_binary.name = "disease_binary"
        return drug_binary, disease_binary

    def test_fisher_returns_tuple(self, binary_data):
        """Fisher test should return (ct, results_df)."""
        from researchpy.crosstab import crosstab
        result = crosstab(binary_data[0], binary_data[1], test="fisher")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fisher_against_scipy(self, binary_data):
        """Fisher test results should match scipy.stats.fisher_exact."""
        from researchpy.crosstab import crosstab
        _, results_df = crosstab(binary_data[0], binary_data[1], test="fisher")

        # Compute directly with scipy
        raw_ct = pd.crosstab(binary_data[0], binary_data[1])
        odds_ratio, p_two = scipy.stats.fisher_exact(raw_ct)

        # First result is odds ratio
        rp_odds = results_df["results"].iloc[0]
        assert rp_odds == pytest.approx(
            round(odds_ratio, 4), rel=APPROX_REL, abs=APPROX_ABS
        )

        # Second result is 2-sided p-value
        rp_p = results_df["results"].iloc[1]
        assert rp_p == pytest.approx(
            round(p_two, 4), rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosstabEdgeCases:
    """Edge case and input validation tests."""

    def test_non_series_first_arg(self):
        """First arg as non-Series should return error string."""
        from researchpy.crosstab import crosstab
        result = crosstab([1, 2, 3], pd.Series([4, 5, 6]))
        assert result == "Operation only supports Pandas Series"

    def test_non_series_second_arg(self):
        """Second arg as non-Series should return error string."""
        from researchpy.crosstab import crosstab
        result = crosstab(pd.Series([1, 2, 3]), [4, 5, 6])
        assert result == "Operation only supports Pandas Series"

    def test_correction_parameter(self, drug_series, disease_series):
        """correction=True should not raise an error."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series, test="chi-square", correction=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_test_no_expected(self, drug_series, disease_series):
        """No test, no expected_freqs should return single DataFrame."""
        from researchpy.crosstab import crosstab
        result = crosstab(drug_series, disease_series)
        assert isinstance(result, pd.DataFrame)

