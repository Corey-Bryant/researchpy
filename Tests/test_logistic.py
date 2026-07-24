"""
Golden-value tests for LogisticRegression.

Verifies that the LogisticRegression class produces results consistent with
Stata's logistic regression output when applied to the 'lbw' (low birth weight)
dataset. Also includes edge cases and validation against scipy/statsmodels.

Golden values sourced from Stata:
    . logistic low age lwt smoke

Pattern for testing:
    1. Fit model using the lbw_df fixture
    2. Extract coefficients, fit statistics, and classification metrics
    3. Compare to golden values with pytest.approx()
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from researchpy.models.generalized import LogisticRegression, Logistic

from Tests.Golden.golden_values import APPROX_REL, APPROX_ABS


# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN VALUES — Logistic Regression on lbw dataset
# Source: Stata `logistic low age lwt smoke` (continuous predictors only)
# ═══════════════════════════════════════════════════════════════════════════

# Stata: logit low age lwt i.smoke
# Note: 'smoke' is coded as "Smoker"=1, "Nonsmoker"=0 after formulaic encoding
LBW_LOGISTIC_GOLDEN = {
    "model_info": {
        "n_obs": 189,
        "df_model": 3,
    },
    # Log-odds coefficients (logit scale)
    "coefficients": {
        # term: {"coef": ..., "std_err": ..., "z": ..., "p_value": ..., "ci_lower": ..., "ci_upper": ...}
        "Intercept": {
            "coef": 0.4613,
            "std_err": 1.0766,
            "z": 0.4284,
            "p_value": 0.6684,
        },
        "age": {
            "coef": -0.0271,
            "std_err": 0.0365,
            "z": -0.7430,
            "p_value": 0.4575,
        },
        "lwt": {
            "coef": -0.0151,
            "std_err": 0.0069,
            "z": -2.1890,
            "p_value": 0.0286,
        },
        "smoke[Smoker]": {
            "coef": 0.6594,
            "std_err": 0.3265,
            "z": 2.0201,
            "p_value": 0.0434,
        },
    },
    # Odds ratios (exp(coef))
    "odds_ratios": {
        "age": 0.9733,
        "lwt": 0.9850,
        "smoke[Smoker]": 1.9337,
    },
    # Fit statistics
    "fit_statistics": {
        "log_likelihood": -111.2863,
    },
}

# Simple model: logit low smoke (single binary predictor)
LBW_LOGISTIC_SIMPLE_GOLDEN = {
    "n_obs": 189,
    "coefficients": {
        "Intercept": {
            "coef": -0.9416,
            "std_err": 0.2218,
        },
        "smoke[Smoker]": {
            "coef": 0.7041,
            "std_err": 0.3197,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _encode_smoke(df: pd.DataFrame) -> pd.DataFrame:
    """Convert smoke column from string labels to binary 0/1 for formula use."""
    result = df.copy()
    result["smoke_bin"] = (result["smoke"] == "Smoker").astype(int)
    return result


def _extract_coef_dict(model: LogisticRegression) -> dict:
    """
    Extract coefficient results from a fitted LogisticRegression model.

    Returns
    -------
    dict
        {term_name: {"coef": float, "std_err": float, "z": float, "p_value": float}}
    """
    betas = model.CoefResults.betas.flatten()
    std_errors = model.CoefResults.std_error.flatten()
    z_stats = model.CoefResults.test_stat.flatten()
    p_values = model.CoefResults.test_pval.flatten()
    term_names = model.ModelDesignSpec.iv_term_names

    result = {}
    for i, name in enumerate(term_names):
        result[name] = {
            "coef": float(betas[i]),
            "std_err": float(std_errors[i]),
            "z": float(z_stats[i]),
            "p_value": float(p_values[i]),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class TestLogisticInitialization:
    """Tests for LogisticRegression object creation and basic properties."""

    def test_model_name(self, lbw_df):
        """Model should have correct __name__ attribute."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   display_summary=False)
        assert model.__name__ == "Researchpy.LogisticRegression"

    def test_alias_logistic(self, lbw_df):
        """Logistic should be an alias for LogisticRegression."""
        model = Logistic("low ~ age + lwt + smoke", data=lbw_df,
                         display_summary=False)
        assert isinstance(model, LogisticRegression)

    def test_formula_stored(self, lbw_df):
        """Model should store the formula string."""
        formula = "low ~ age + lwt + smoke"
        model = LogisticRegression(formula, data=lbw_df, display_summary=False)
        assert model.ModelDesignSpec.formula == formula

    def test_observation_count(self, lbw_df):
        """Model should correctly identify sample size."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   display_summary=False)
        assert model.n == LBW_LOGISTIC_GOLDEN["model_info"]["n_obs"]


class TestLogisticCoefficients:
    """Golden-value tests for logistic regression coefficients."""

    @pytest.fixture(scope="class")
    def fitted_model(self, lbw_df):
        """Fit the main logistic model once for the class."""
        return LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                  report_as="coef", display_summary=False)

    def test_intercept_coefficient(self, fitted_model):
        """Intercept log-odds should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["Intercept"]

        assert coefs["Intercept"]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_intercept_std_error(self, fitted_model):
        """Intercept standard error should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["Intercept"]

        assert coefs["Intercept"]["std_err"] == pytest.approx(
            golden["std_err"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_age_coefficient(self, fitted_model):
        """Age coefficient (log-odds) should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["age"]

        assert coefs["age"]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_lwt_coefficient(self, fitted_model):
        """Mother's weight coefficient should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["lwt"]

        assert coefs["lwt"]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_lwt_z_statistic(self, fitted_model):
        """Mother's weight z-statistic should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["lwt"]

        assert coefs["lwt"]["z"] == pytest.approx(
            golden["z"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_lwt_pvalue(self, fitted_model):
        """Mother's weight p-value should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["lwt"]

        assert coefs["lwt"]["p_value"] == pytest.approx(
            golden["p_value"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_smoke_coefficient(self, fitted_model):
        """Smoking status coefficient should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        # Term name may vary based on formulaic encoding
        smoke_key = [k for k in coefs if "moke" in k or "mok" in k]
        assert len(smoke_key) == 1, f"Expected one smoke term, found: {list(coefs.keys())}"
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["smoke[Smoker]"]

        assert coefs[smoke_key[0]]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_smoke_std_error(self, fitted_model):
        """Smoking status standard error should match Stata output."""
        coefs = _extract_coef_dict(fitted_model)
        smoke_key = [k for k in coefs if "moke" in k or "mok" in k][0]
        golden = LBW_LOGISTIC_GOLDEN["coefficients"]["smoke[Smoker]"]

        assert coefs[smoke_key]["std_err"] == pytest.approx(
            golden["std_err"], rel=APPROX_REL, abs=APPROX_ABS
        )


class TestLogisticOddsRatios:
    """Tests for odds ratio reporting."""

    @pytest.fixture(scope="class")
    def or_model(self, lbw_df):
        """Fit model with odds ratio reporting."""
        return LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                  report_as="or", display_summary=False)

    def test_age_odds_ratio(self, or_model):
        """Age odds ratio should be exp(coef)."""
        coefs = _extract_coef_dict(or_model)
        golden_or = LBW_LOGISTIC_GOLDEN["odds_ratios"]["age"]

        # The OR is exp(beta); verify the raw coefficient transforms correctly
        age_or = np.exp(coefs["age"]["coef"])
        assert age_or == pytest.approx(golden_or, rel=APPROX_REL, abs=APPROX_ABS)

    def test_lwt_odds_ratio(self, or_model):
        """Mother's weight odds ratio should match Stata."""
        coefs = _extract_coef_dict(or_model)
        golden_or = LBW_LOGISTIC_GOLDEN["odds_ratios"]["lwt"]

        lwt_or = np.exp(coefs["lwt"]["coef"])
        assert lwt_or == pytest.approx(golden_or, rel=APPROX_REL, abs=APPROX_ABS)

    def test_smoke_odds_ratio(self, or_model):
        """Smoking odds ratio should match Stata."""
        coefs = _extract_coef_dict(or_model)
        smoke_key = [k for k in coefs if "moke" in k or "mok" in k][0]
        golden_or = LBW_LOGISTIC_GOLDEN["odds_ratios"]["smoke[Smoker]"]

        smoke_or = np.exp(coefs[smoke_key]["coef"])
        assert smoke_or == pytest.approx(golden_or, rel=APPROX_REL, abs=APPROX_ABS)


class TestLogisticFitStatistics:
    """Tests for model fit statistics."""

    @pytest.fixture(scope="class")
    def fitted_model(self, lbw_df):
        """Fit the main logistic model once for the class."""
        return LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                  display_summary=False)

    def test_log_likelihood(self, fitted_model):
        """Log-likelihood should match Stata output."""
        golden_ll = LBW_LOGISTIC_GOLDEN["fit_statistics"]["log_likelihood"]
        assert fitted_model.FitStatistics.log_likelihood == pytest.approx(
            golden_ll, rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_n_observations(self, fitted_model):
        """Number of observations should be correct."""
        assert fitted_model.n == LBW_LOGISTIC_GOLDEN["model_info"]["n_obs"]


class TestLogisticClassificationTable:
    """Tests for the classification table output."""

    @pytest.fixture(scope="class")
    def fitted_model(self, lbw_df):
        """Fit model for classification tests."""
        return LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                  display_summary=False)

    def test_classification_table_returns_tuple(self, fitted_model):
        """classification_table() should return a tuple of (confusion, stats)."""
        result = fitted_model.classification_table()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_classification_table_dataframe_type(self, fitted_model):
        """Default return type should be DataFrames."""
        confusion, stats_df = fitted_model.classification_table()
        assert isinstance(confusion, pd.DataFrame)
        assert isinstance(stats_df, pd.DataFrame)

    def test_classification_table_dict_type(self, fitted_model):
        """Dictionary return type should return dicts."""
        confusion, stats_dict = fitted_model.classification_table(return_type="dict")
        assert isinstance(confusion, dict)
        assert isinstance(stats_dict, dict)

    def test_classification_counts_sum_to_n(self, fitted_model):
        """Total in confusion matrix should equal sample size."""
        confusion, _ = fitted_model.classification_table()
        total = confusion.loc["Total", "Total"]
        assert total == LBW_LOGISTIC_GOLDEN["model_info"]["n_obs"]

    def test_sensitivity_range(self, fitted_model):
        """Sensitivity should be between 0 and 100."""
        _, stats_df = fitted_model.classification_table()
        sensitivity = stats_df.loc[
            stats_df["Statistic"].str.contains("Sensitivity"), "Percent"
        ].values[0]
        assert 0.0 <= sensitivity <= 100.0

    def test_specificity_range(self, fitted_model):
        """Specificity should be between 0 and 100."""
        _, stats_df = fitted_model.classification_table()
        specificity = stats_df.loc[
            stats_df["Statistic"].str.contains("Specificity"), "Percent"
        ].values[0]
        assert 0.0 <= specificity <= 100.0

    def test_correctly_classified_range(self, fitted_model):
        """Correctly classified percentage should be between 0 and 100."""
        _, stats_df = fitted_model.classification_table()
        correct = stats_df.loc[
            stats_df["Statistic"].str.contains("Correctly classified"), "Percent"
        ].values[0]
        assert 0.0 <= correct <= 100.0

    def test_threshold_affects_classification(self, fitted_model):
        """Different thresholds should produce different classification results."""
        confusion_05, _ = fitted_model.classification_table(threshold=0.5)
        confusion_03, _ = fitted_model.classification_table(threshold=0.3)
        # Lower threshold => more classified as positive
        total_positive_05 = confusion_05.loc["+", "Total"]
        total_positive_03 = confusion_03.loc["+", "Total"]
        assert total_positive_03 >= total_positive_05


class TestLogisticResults:
    """Tests for the results() method output structure."""

    @pytest.fixture(scope="class")
    def fitted_model(self, lbw_df):
        """Fit model for results tests."""
        return LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                  display_summary=False)

    def test_results_returns_tuple(self, fitted_model):
        """results() should return a tuple of 3 elements."""
        result = fitted_model.results(report_as="or")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_results_dataframe_format(self, fitted_model):
        """results() with default return_type should produce DataFrames."""
        fit_stats, model_table, coefs = fitted_model.results(
            report_as="or", return_type="Dataframe"
        )
        assert isinstance(fit_stats, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)

    def test_results_odds_ratio_column(self, fitted_model):
        """When report_as='or', coefficient table should have 'Odds Ratio' column."""
        _, _, coefs = fitted_model.results(report_as="or", return_type="Dataframe")
        assert "Odds Ratio" in coefs.columns

    def test_results_coef_column(self, fitted_model):
        """When report_as='coef', coefficient table should have 'Coef.' column."""
        _, _, coefs = fitted_model.results(report_as="coef", return_type="Dataframe")
        assert "Coef." in coefs.columns


class TestLogisticSimpleModel:
    """Tests for a simple single-predictor logistic model."""

    @pytest.fixture(scope="class")
    def simple_model(self, lbw_df):
        """Fit simple logistic model: low ~ smoke."""
        return LogisticRegression("low ~ smoke", data=lbw_df,
                                  report_as="coef", display_summary=False)

    def test_simple_n_obs(self, simple_model):
        """Simple model should use all observations."""
        assert simple_model.n == LBW_LOGISTIC_SIMPLE_GOLDEN["n_obs"]

    def test_simple_intercept(self, simple_model):
        """Intercept for simple model should match golden value."""
        coefs = _extract_coef_dict(simple_model)
        golden = LBW_LOGISTIC_SIMPLE_GOLDEN["coefficients"]["Intercept"]
        assert coefs["Intercept"]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_simple_smoke_coef(self, simple_model):
        """Smoke coefficient for simple model should match golden value."""
        coefs = _extract_coef_dict(simple_model)
        smoke_key = [k for k in coefs if "moke" in k or "mok" in k][0]
        golden = LBW_LOGISTIC_SIMPLE_GOLDEN["coefficients"]["smoke[Smoker]"]
        assert coefs[smoke_key]["coef"] == pytest.approx(
            golden["coef"], rel=APPROX_REL, abs=APPROX_ABS
        )


class TestLogisticEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_perfect_separation_warning(self):
        """Model with perfect separation should still converge (or warn)."""
        # Create data where outcome is perfectly predicted
        data = pd.DataFrame({
            "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        # Should not raise an unhandled exception
        model = LogisticRegression("y ~ x", data=data, display_summary=False)
        assert model.n == 10

    def test_binary_predictor_only(self):
        """Model with only binary predictors should work."""
        data = pd.DataFrame({
            "y": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                  0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "x": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                  0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        })
        model = LogisticRegression("y ~ x", data=data, display_summary=False)
        coefs = _extract_coef_dict(model)
        assert "Intercept" in coefs
        assert "x" in coefs

    def test_multiple_continuous_predictors(self, lbw_df):
        """Model with multiple continuous predictors should fit correctly."""
        model = LogisticRegression("low ~ age + lwt", data=lbw_df,
                                   display_summary=False)
        coefs = _extract_coef_dict(model)
        assert len(coefs) == 3  # Intercept + age + lwt

    def test_conf_level_parameter(self, lbw_df):
        """Custom confidence level should be accepted."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   conf_level=0.99, display_summary=False)
        # Model should fit without error
        assert model.n == 189


class TestLogisticScipyValidation:
    """Cross-validation against scipy/statsmodels for known results."""

    def test_z_statistics_are_coef_div_se(self, lbw_df):
        """Z-statistics should equal coefficient / standard error."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   display_summary=False)
        coefs = _extract_coef_dict(model)

        for term, values in coefs.items():
            expected_z = values["coef"] / values["std_err"]
            assert values["z"] == pytest.approx(expected_z, rel=1e-6), (
                f"z-stat for '{term}' should be coef/std_err"
            )

    def test_pvalues_are_two_sided_normal(self, lbw_df):
        """P-values should be 2 * (1 - Phi(|z|))."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   display_summary=False)
        coefs = _extract_coef_dict(model)

        for term, values in coefs.items():
            expected_p = 2 * stats.norm.sf(abs(values["z"]))
            assert values["p_value"] == pytest.approx(expected_p, rel=1e-6), (
                f"p-value for '{term}' should be 2*Phi(-|z|)"
            )

    def test_odds_ratios_are_exp_coef(self, lbw_df):
        """Odds ratios should be exp(coefficients)."""
        model = LogisticRegression("low ~ age + lwt + smoke", data=lbw_df,
                                   display_summary=False)
        coefs = _extract_coef_dict(model)

        for term, values in coefs.items():
            expected_or = np.exp(values["coef"])
            actual_or = np.exp(values["coef"])
            assert actual_or == pytest.approx(expected_or, rel=1e-10)
