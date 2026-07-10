"""
Tests for researchpy.anova module.

Validates ANOVA (Type I, II, III) output against golden reference values from
the systolic dataset (source: ResearchPy anova() documentation).

Formula: systolic ~ C(drug) + C(disease) + C(drug):C(disease)
Sum of squares: Type III (default)

Golden values validated against Stata output and trusted by the scientific community.
"""
import warnings

import pytest
import pandas as pd
import numpy as np

from Tests.Golden.golden_values import (
    APPROX_REL,
    APPROX_ABS,
    SYSTOLIC_ANOVA_FIT,
    SYSTOLIC_ANOVA_TABLE,
    SYSTOLIC_REGRESSION_TABLE,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def anova_model_type3(systolic_df):
    """
    Fit ANOVA Type III model on systolic data, suppressing deprecation warnings.

    Formula matches the anova() documentation example.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from researchpy.anova import anova
        model = anova(
            "systolic ~ C(drug) + C(disease) + C(drug):C(disease)",
            data=systolic_df,
            sum_of_squares=3,
        )
    return model


@pytest.fixture(scope="module")
def anova_model_type1(systolic_df):
    """Fit ANOVA Type I model on systolic data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from researchpy.anova import anova
        model = anova(
            "systolic ~ C(drug) + C(disease) + C(drug):C(disease)",
            data=systolic_df,
            sum_of_squares=1,
        )
    return model


@pytest.fixture(scope="module")
def anova_results_dict(anova_model_type3):
    """ANOVA Type III results returned as Dictionary."""
    return anova_model_type3.results(return_type="Dictionary", decimals=4,
                                     pretty_format=True)


@pytest.fixture(scope="module")
def anova_results_df(anova_model_type3):
    """ANOVA Type III results returned as DataFrames."""
    return anova_model_type3.results(return_type="Dataframe", decimals=4,
                                     pretty_format=True)


# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATION WARNING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaDeprecation:
    """Verify that anova class issues deprecation warnings on instantiation."""

    def test_anova_emits_deprecation_warning(self, systolic_df):
        """Instantiating anova should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from researchpy.anova import anova
            _ = anova("systolic ~ C(drug)", data=systolic_df, sum_of_squares=1)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, "Expected at least one DeprecationWarning"

        # Check that the anova-specific message is present
        messages = [str(x.message) for x in deprecation_warnings]
        assert any("anova" in msg.lower() for msg in messages), (
            f"Expected 'anova' in deprecation message, got: {messages}"
        )

    def test_anova_deprecation_mentions_migration(self, systolic_df):
        """Deprecation message should mention the migration path."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from researchpy.anova import anova
            _ = anova("systolic ~ C(drug)", data=systolic_df, sum_of_squares=1)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        messages = [str(x.message) for x in deprecation_warnings]
        assert any("researchpy.models.multivariable" in msg for msg in messages), (
            f"Expected migration path in message, got: {messages}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FIT STATISTICS (Type III)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaFitStatistics:
    """Validate ANOVA Type III model fit statistics against golden values."""

    def test_number_of_observations(self, anova_model_type3):
        """Model should have correct number of observations."""
        assert anova_model_type3.nobs == SYSTOLIC_ANOVA_FIT["Number of obs"]

    def test_root_mse(self, anova_model_type3):
        """Root MSE should match golden value."""
        assert anova_model_type3.model_data["root_mse"] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Root MSE"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_r_squared(self, anova_model_type3):
        """R-squared should match golden value."""
        assert anova_model_type3.model_data["r squared"] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_adj_r_squared(self, anova_model_type3):
        """Adjusted R-squared should match golden value."""
        assert anova_model_type3.model_data["r squared adj."] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Adj R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_fit_stats_in_results_output(self, anova_results_dict):
        """Fit statistics from results() should match golden values."""
        desc = anova_results_dict[0]
        assert desc["Number of obs = "] == SYSTOLIC_ANOVA_FIT["Number of obs"]
        assert desc["Root MSE = "] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Root MSE"], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert desc["R-squared = "] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert desc["Adj R-squared = "] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Adj R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# ANOVA TABLE — MODEL ROW (Type III)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaModelRow:
    """Validate the Model row in ANOVA Type III output."""

    def test_model_ss(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["sum_of_square_model"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_df(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["degrees_of_freedom_model"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_ms(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["msr"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_f_value(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["f_value_model"] == pytest.approx(
            golden["F value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_p_value(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["f_p_value_model"] == pytest.approx(
            golden["p-value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_eta_squared(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["Eta squared"] == pytest.approx(
            golden["Eta squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_omega_squared(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert anova_model_type3.model_data["Omega squared"] == pytest.approx(
            golden["Omega squared"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# ANOVA TABLE — FACTOR EFFECTS (Type III)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaFactorEffects:
    """Validate factor-level effects (drug, disease, interaction) for Type III."""

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_sum_of_squares(self, anova_model_type3, factor_idx, source_key):
        """Factor SS should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["Sum of Squares"][factor_idx]
        assert actual == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_degrees_of_freedom(self, anova_model_type3, factor_idx, source_key):
        """Factor df should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["Degrees of Freedom"][factor_idx]
        assert actual == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_mean_squares(self, anova_model_type3, factor_idx, source_key):
        """Factor MS should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["Mean Squares"][factor_idx]
        assert actual == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_f_value(self, anova_model_type3, factor_idx, source_key):
        """Factor F-value should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["F value"][factor_idx]
        assert actual == pytest.approx(
            golden["F value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_p_value(self, anova_model_type3, factor_idx, source_key):
        """Factor p-value should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["p-value"][factor_idx]
        assert actual == pytest.approx(
            golden["p-value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_eta_squared(self, anova_model_type3, factor_idx, source_key):
        """Factor partial Eta squared should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["Eta squared"][factor_idx]
        assert actual == pytest.approx(
            golden["Eta squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("factor_idx,source_key", [
        (0, "drug"),
        (1, "disease"),
        (2, "drug:disease"),
    ])
    def test_factor_omega_squared(self, anova_model_type3, factor_idx, source_key):
        """Factor partial Omega squared should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE[source_key]
        actual = anova_model_type3.factor_effects["Omega squared"][factor_idx]
        assert actual == pytest.approx(
            golden["Omega squared"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# ANOVA TABLE — RESIDUAL AND TOTAL ROWS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaResidualTotal:
    """Validate Residual and Total rows in ANOVA output."""

    def test_residual_ss(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert anova_model_type3.model_data["sum_of_square_residual"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_residual_df(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert anova_model_type3.model_data["degrees_of_freedom_residual"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_residual_ms(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert anova_model_type3.model_data["mse"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_ss(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert anova_model_type3.model_data["sum_of_square_total"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_df(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert anova_model_type3.model_data["degrees_of_freedom_total"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_ms(self, anova_model_type3):
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert anova_model_type3.model_data["mst"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION TABLE (via anova.regression_table())
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaRegressionTable:
    """Validate anova.regression_table() against golden values."""

    @pytest.fixture(scope="class")
    def reg_table(self, anova_model_type3):
        """Get regression table as DataFrame."""
        return anova_model_type3.regression_table(return_type="Dataframe", decimals=4)

    def test_regression_table_returns_dataframe(self, reg_table):
        """regression_table() should return a DataFrame."""
        assert isinstance(reg_table, pd.DataFrame)

    def test_intercept_coefficient(self, reg_table):
        """Intercept coefficient should match golden value."""
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        assert reg_table["Coef."].iloc[0] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_std_err(self, reg_table):
        """Intercept standard error should match golden value."""
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        assert reg_table["Std. Err."].iloc[0] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_t_stat(self, reg_table):
        """Intercept t-statistic should match golden value."""
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        assert reg_table["t"].iloc[0] == pytest.approx(
            golden["t"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# TYPE I SUM OF SQUARES
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaTypeI:
    """Validate ANOVA Type I SS properties and consistency."""

    def test_type1_factor_effects_exist(self, anova_model_type1):
        """Type I model should have factor_effects attribute."""
        assert hasattr(anova_model_type1, "factor_effects")
        assert "Source" in anova_model_type1.factor_effects
        assert "Sum of Squares" in anova_model_type1.factor_effects

    def test_type1_factor_count(self, anova_model_type1):
        """Type I should have 3 factor effects (drug, disease, interaction)."""
        sources = anova_model_type1.factor_effects["Source"]
        assert len(sources) == 3

    def test_type1_ss_sum_equals_model_ss(self, anova_model_type1):
        """
        For Type I, sum of factor SS should equal Model SS.

        This is a defining property of sequential (Type I) sums of squares.
        """
        factor_ss_sum = sum(anova_model_type1.factor_effects["Sum of Squares"])
        model_ss = anova_model_type1.model_data["sum_of_square_model"]
        assert factor_ss_sum == pytest.approx(model_ss, rel=1e-6)

    def test_type1_all_ss_positive(self, anova_model_type1):
        """All Type I factor SS values should be non-negative."""
        for ss in anova_model_type1.factor_effects["Sum of Squares"]:
            assert ss >= 0, f"Negative SS found: {ss}"

    def test_type1_all_f_values_positive(self, anova_model_type1):
        """All Type I F-values should be positive."""
        for f_val in anova_model_type1.factor_effects["F value"]:
            assert f_val > 0, f"Non-positive F-value: {f_val}"

    def test_type1_p_values_in_range(self, anova_model_type1):
        """All Type I p-values should be between 0 and 1."""
        for p_val in anova_model_type1.factor_effects["p-value"]:
            assert 0 <= p_val <= 1, f"p-value out of range: {p_val}"

    def test_type1_effect_sizes_in_range(self, anova_model_type1):
        """Partial Eta squared values should be between 0 and 1."""
        for eta in anova_model_type1.factor_effects["Eta squared"]:
            assert 0 <= eta <= 1, f"Eta squared out of range: {eta}"


# ═══════════════════════════════════════════════════════════════════════════
# RETURN TYPE AND OUTPUT FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaReturnTypes:
    """Validate that ANOVA results() returns correct types."""

    def test_returns_dataframe_tuple(self, anova_results_df):
        """results(return_type='Dataframe') should return tuple of 2 DataFrames."""
        assert isinstance(anova_results_df, tuple)
        assert len(anova_results_df) == 2
        for item in anova_results_df:
            assert isinstance(item, pd.DataFrame)

    def test_returns_dictionary_tuple(self, anova_results_dict):
        """results(return_type='Dictionary') should return tuple of 2 dicts."""
        assert isinstance(anova_results_dict, tuple)
        assert len(anova_results_dict) == 2
        for item in anova_results_dict:
            assert isinstance(item, dict)

    def test_invalid_return_type(self, anova_model_type3, capsys):
        """Invalid return_type should print error message."""
        result = anova_model_type3.results(return_type="InvalidType")
        assert result is None
        captured = capsys.readouterr()
        assert "Not a valid return type" in captured.out

    def test_pretty_format_includes_blank_rows(self, anova_results_dict):
        """Pretty format should include separator blank rows in Source column."""
        table = anova_results_dict[1]
        # Pretty format adds empty string separators in Source column
        assert "" in table["Source"]


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES AND CONSISTENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnovaEdgeCases:
    """Edge case and consistency tests for ANOVA."""

    def test_ss_decomposition(self, anova_model_type3):
        """SS Total should equal SS Model + SS Residual."""
        ss_total = anova_model_type3.model_data["sum_of_square_total"]
        ss_model = anova_model_type3.model_data["sum_of_square_model"]
        ss_resid = anova_model_type3.model_data["sum_of_square_residual"]
        assert ss_total == pytest.approx(ss_model + ss_resid, rel=1e-6)

    def test_df_decomposition(self, anova_model_type3):
        """df Total should equal df Model + df Residual."""
        df_total = anova_model_type3.model_data["degrees_of_freedom_total"]
        df_model = anova_model_type3.model_data["degrees_of_freedom_model"]
        df_resid = anova_model_type3.model_data["degrees_of_freedom_residual"]
        assert df_total == df_model + df_resid

    def test_factor_df_sum_equals_model_df(self, anova_model_type3):
        """Sum of factor degrees of freedom should equal model df."""
        factor_df_sum = sum(anova_model_type3.factor_effects["Degrees of Freedom"])
        model_df = anova_model_type3.model_data["degrees_of_freedom_model"]
        assert factor_df_sum == pytest.approx(model_df, rel=1e-6)

    def test_predict_method_exists(self, anova_model_type3):
        """ANOVA model should have a predict method."""
        assert hasattr(anova_model_type3, "predict")
        assert callable(anova_model_type3.predict)

    def test_nobs_matches_data(self, anova_model_type3, systolic_df):
        """Number of observations should match input data length."""
        assert anova_model_type3.nobs == len(systolic_df)

