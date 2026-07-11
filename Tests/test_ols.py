"""
Tests for researchpy.ols module.

Validates OLS regression output against golden reference values from the
systolic dataset (source: ResearchPy anova() documentation). Since anova
extends ols, the model-level statistics and regression coefficient table
produced by ols are validated here using the same trusted golden values.

Formula: systolic ~ C(drug) + C(disease) + C(drug):C(disease)
"""
import warnings

import pytest
import pandas as pd

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
def ols_model(systolic_df):
    """
    Fit OLS model on systolic data, suppressing deprecation warnings.

    Formula matches the anova() documentation example.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from researchpy.ols import ols
        model = ols("systolic ~ C(drug) + C(disease) + C(drug):C(disease)",
                    data=systolic_df)
    return model


@pytest.fixture(scope="module")
def ols_results_dict(ols_model):
    """OLS results returned as Dictionary (descriptives, anova_table, regression_table)."""
    return ols_model.results(return_type="Dictionary", decimals=4, pretty_format=True)


@pytest.fixture(scope="module")
def ols_results_df(ols_model):
    """OLS results returned as DataFrames."""
    return ols_model.results(return_type="Dataframe", decimals=4, pretty_format=True)


# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATION WARNING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsDeprecation:
    """Verify that ols class issues deprecation warnings on instantiation."""

    def test_ols_emits_deprecation_warning(self, systolic_df):
        """Instantiating ols should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from researchpy.ols import ols
            _ = ols("systolic ~ C(drug)", data=systolic_df)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, "Expected at least one DeprecationWarning"

        # Check that the ols-specific message is present
        messages = [str(x.message) for x in deprecation_warnings]
        assert any("ols" in msg.lower() for msg in messages), (
            f"Expected 'ols' in deprecation message, got: {messages}"
        )

    def test_model_emits_deprecation_warning(self, systolic_df):
        """Instantiating model (parent class) should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from researchpy.model import model
            _ = model("systolic ~ C(drug)", data=systolic_df)

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning from model class"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FIT STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsFitStatistics:
    """Validate OLS model fit statistics against golden values."""

    def test_number_of_observations(self, ols_model):
        """Model should have correct number of observations."""
        assert ols_model.nobs == SYSTOLIC_ANOVA_FIT["Number of obs"]

    def test_root_mse(self, ols_model):
        """Root MSE should match golden value."""
        assert ols_model.model_data["root_mse"] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Root MSE"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_r_squared(self, ols_model):
        """R-squared should match golden value."""
        assert ols_model.model_data["r squared"] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_adj_r_squared(self, ols_model):
        """Adjusted R-squared should match golden value."""
        assert ols_model.model_data["r squared adj."] == pytest.approx(
            SYSTOLIC_ANOVA_FIT["Adj R-squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_results_descriptives_dict(self, ols_results_dict):
        """Descriptives returned as dict should contain correct fit stats."""
        desc = ols_results_dict[0]
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
# OVERALL MODEL ANOVA TABLE (Model / Residual / Total)
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsModelTable:
    """Validate OLS model-level ANOVA table (Model, Residual, Total rows)."""

    def test_model_sum_of_squares(self, ols_model):
        """Model SS should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["sum_of_square_model"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_degrees_of_freedom(self, ols_model):
        """Model degrees of freedom should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["degrees_of_freedom_model"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_mean_squares(self, ols_model):
        """Model mean squares should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["msr"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_f_value(self, ols_model):
        """Model F-value should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["f_value_model"] == pytest.approx(
            golden["F value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_p_value(self, ols_model):
        """Model p-value should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["f_p_value_model"] == pytest.approx(
            golden["p-value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_eta_squared(self, ols_model):
        """Model Eta squared should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["Eta squared"] == pytest.approx(
            golden["Eta squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_model_omega_squared(self, ols_model):
        """Model Omega squared should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Model"]
        assert ols_model.model_data["Omega squared"] == pytest.approx(
            golden["Omega squared"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_residual_sum_of_squares(self, ols_model):
        """Residual SS should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert ols_model.model_data["sum_of_square_residual"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_residual_degrees_of_freedom(self, ols_model):
        """Residual degrees of freedom should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert ols_model.model_data["degrees_of_freedom_residual"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_residual_mean_squares(self, ols_model):
        """Residual mean squares (MSE) should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Residual"]
        assert ols_model.model_data["mse"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_sum_of_squares(self, ols_model):
        """Total SS should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert ols_model.model_data["sum_of_square_total"] == pytest.approx(
            golden["Sum of Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_degrees_of_freedom(self, ols_model):
        """Total degrees of freedom should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert ols_model.model_data["degrees_of_freedom_total"] == pytest.approx(
            golden["Degrees of Freedom"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_total_mean_squares(self, ols_model):
        """Total mean squares should match golden value."""
        golden = SYSTOLIC_ANOVA_TABLE["Total"]
        assert ols_model.model_data["mst"] == pytest.approx(
            golden["Mean Squares"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION COEFFICIENT TABLE
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsRegressionTable:
    """Validate OLS regression coefficient table against golden values."""

    @pytest.fixture(scope="class")
    def reg_table_df(self, ols_model):
        """Extract regression table as DataFrame."""
        _, _, reg = ols_model.results(return_type="Dataframe", decimals=4,
                                      pretty_format=True)
        return reg

    def test_intercept_coefficient(self, ols_results_dict):
        """Intercept coefficient should match golden value."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[0]  # Intercept row
        # Find the intercept value in the coefficient list
        assert reg["Coef."][0] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_std_err(self, ols_results_dict):
        """Intercept standard error should match golden value."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        assert reg["Std. Err."][0] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_t_statistic(self, ols_results_dict):
        """Intercept t-statistic should match golden value."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        assert reg["t"][0] == pytest.approx(
            golden["t"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_p_value(self, ols_results_dict):
        """Intercept p-value should match golden value (essentially 0)."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        # p-value is 0.0000 in golden (< 0.0001)
        assert reg["p-value"][0] == pytest.approx(
            golden["p-value"], abs=1e-3
        )

    def test_intercept_confidence_interval(self, ols_results_dict):
        """Intercept 95% CI should match golden value."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[0]
        ci = reg["95% Conf. Interval"][0]
        assert ci[0] == pytest.approx(
            golden["95% Conf. Interval"][0], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert ci[1] == pytest.approx(
            golden["95% Conf. Interval"][1], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("row_idx,golden_idx", [
        (3, 3),   # drug level 2
        (4, 4),   # drug level 3
        (5, 5),   # drug level 4
    ])
    def test_drug_coefficients(self, ols_results_dict, row_idx, golden_idx):
        """Drug factor coefficients should match golden values."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[golden_idx]

        assert reg["Coef."][row_idx] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert reg["Std. Err."][row_idx] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert reg["t"][row_idx] == pytest.approx(
            golden["t"], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert reg["p-value"][row_idx] == pytest.approx(
            golden["p-value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    @pytest.mark.parametrize("row_idx,golden_idx", [
        (8, 8),   # disease level 2
        (9, 9),   # disease level 3
    ])
    def test_disease_coefficients(self, ols_results_dict, row_idx, golden_idx):
        """Disease factor coefficients should match golden values."""
        reg = ols_results_dict[2]
        golden = SYSTOLIC_REGRESSION_TABLE[golden_idx]

        assert reg["Coef."][row_idx] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert reg["Std. Err."][row_idx] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# RETURN TYPE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsReturnTypes:
    """Validate that OLS results() returns correct types."""

    def test_returns_dataframe_tuple(self, ols_results_df):
        """results(return_type='Dataframe') should return tuple of 3 DataFrames."""
        assert isinstance(ols_results_df, tuple)
        assert len(ols_results_df) == 3
        for item in ols_results_df:
            assert isinstance(item, pd.DataFrame)

    def test_returns_dictionary_tuple(self, ols_results_dict):
        """results(return_type='Dictionary') should return tuple of 3 dicts."""
        assert isinstance(ols_results_dict, tuple)
        assert len(ols_results_dict) == 3
        for item in ols_results_dict:
            assert isinstance(item, dict)

    def test_invalid_return_type(self, ols_model, capsys):
        """Invalid return_type should print error message."""
        result = ols_model.results(return_type="InvalidType")
        assert result is None
        captured = capsys.readouterr()
        assert "Not a valid return type" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestOlsEdgeCases:
    """Edge case tests for OLS."""

    def test_model_data_keys_exist(self, ols_model):
        """All expected keys should be present in model_data."""
        expected_keys = [
            "J", "I", "H", "betas",
            "sum_of_square_total", "sum_of_square_model", "sum_of_square_residual",
            "degrees_of_freedom_model", "degrees_of_freedom_residual",
            "degrees_of_freedom_total",
            "msr", "mse", "mst", "root_mse",
            "f_value_model", "f_p_value_model",
            "r squared", "r squared adj.",
            "Eta squared", "Epsilon squared", "Omega squared",
        ]
        for key in expected_keys:
            assert key in ols_model.model_data, f"Missing key: {key}"

    def test_ss_decomposition(self, ols_model):
        """SS Total should equal SS Model + SS Residual."""
        ss_total = ols_model.model_data["sum_of_square_total"]
        ss_model = ols_model.model_data["sum_of_square_model"]
        ss_resid = ols_model.model_data["sum_of_square_residual"]
        assert ss_total == pytest.approx(ss_model + ss_resid, rel=1e-6)

    def test_df_decomposition(self, ols_model):
        """df Total should equal df Model + df Residual."""
        df_total = ols_model.model_data["degrees_of_freedom_total"]
        df_model = ols_model.model_data["degrees_of_freedom_model"]
        df_resid = ols_model.model_data["degrees_of_freedom_residual"]
        assert df_total == df_model + df_resid

    def test_r_squared_from_ss(self, ols_model):
        """R-squared should equal SS_model / SS_total."""
        r_sq = ols_model.model_data["r squared"]
        ss_model = ols_model.model_data["sum_of_square_model"]
        ss_total = ols_model.model_data["sum_of_square_total"]
        assert r_sq == pytest.approx(ss_model / ss_total, rel=1e-6)

    def test_f_value_from_mean_squares(self, ols_model):
        """F-value should equal MSR / MSE."""
        f_val = ols_model.model_data["f_value_model"]
        msr = ols_model.model_data["msr"]
        mse = ols_model.model_data["mse"]
        assert f_val == pytest.approx(msr / mse, rel=1e-6)

    def test_confidence_level_parameter(self, ols_model):
        """Results with conf_level=0.99 should produce wider intervals."""
        _, _, reg_95 = ols_model.results(return_type="Dictionary", decimals=8,
                                         conf_level=0.95)
        _, _, reg_99 = ols_model.results(return_type="Dictionary", decimals=8,
                                         conf_level=0.99)

        # 99% CI should be wider than 95% CI for the intercept
        ci_95 = reg_95["95% Conf. Interval"][0]
        ci_99 = reg_99["99% Conf. Interval"][0]

        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 > width_95

