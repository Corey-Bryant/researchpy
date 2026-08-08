"""
Tests for researchpy GLM with Gaussian family and Identity link.

Validates GLM regression output against golden reference values from the
glm-reg dataset (source: https://academicweb.nd.edu/~rwilliam/stats3/L01.pdf).

Formula: income ~ educ + jobexp + C(black)

Golden values transcribed from Stata output:
    . glm income educ jobexp i.black, family(gaussian) link(identity)

Note on reference levels:
    Stata uses white (black=0) as reference, reporting β_black = -2.55.
    Formulaic uses alphabetical ordering, making "black" the reference
    and reporting β_white = +2.55.  The magnitudes, SEs, |z|, p-values,
    and CI widths are identical; only the sign and intercept differ by
    the reference-level offset.
"""
import warnings

import pytest
import pandas as pd
import numpy as np

from Tests.Golden.golden_values import (
    APPROX_REL,
    APPROX_ABS,
    GLM_REG_FIT,
    GLM_REG_REGRESSION_TABLE,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model_fixture(glm_reg_df):
    """
    Fit GLM with Gaussian family / Identity link.

    Equivalent Stata command:
        glm income educ jobexp i.black, family(gaussian) link(identity)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from researchpy import GLM
        model = GLM(
            "income ~ educ + jobexp + C(black)",
            data=glm_reg_df,
            family="gaussian",
            link="identity",
            solver_options={"display": False},
            report_betas_as="coef",
            display_summary=False,
        )

    return model


@pytest.fixture(scope="module")
def results_dict(model_fixture):
    """GLM results returned as a Tuple of Dictionaries."""
    return model_fixture.results(
        return_type="Dictionary", report_betas_as="coef", pretty_format=True
    )


@pytest.fixture(scope="module")
def results_df(model_fixture):
    """GLM results returned as a Tuple of DataFrames."""
    return model_fixture.results(
        return_type="Dataframe", report_betas_as="coef", pretty_format=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN VALIDATION: FIT STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestFitStatistics:
    """Validate model fit statistics against Stata golden values."""

    def test_number_of_observations(self, model_fixture):
        """Model should have n=500 observations."""
        assert model_fixture.n == GLM_REG_FIT["n"]

    def test_residual_degrees_of_freedom(self, model_fixture):
        """Residual df = n - k = 500 - 4 = 496."""
        assert model_fixture.FitStatistics.df_residual == GLM_REG_FIT["df_residual"]

    def test_scale_parameter(self, model_fixture):
        """Scale parameter (dispersion) should match Stata: φ = Pearson/df_resid."""
        assert model_fixture.FitStatistics.scale_parameter == pytest.approx(
            GLM_REG_FIT["scale_parameter"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_log_likelihood(self, model_fixture):
        """Log-likelihood should match Stata."""
        assert float(model_fixture.FitStatistics.log_likelihood) == pytest.approx(
            GLM_REG_FIT["log_likelihood"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_deviance(self, model_fixture):
        """Deviance = Σ(y - μ)² should match Stata (Gaussian deviance = RSS)."""
        family = model_fixture.ModelDesignSpec.family
        eta = model_fixture.IV @ model_fixture.CoefResults.betas
        mu = family.link_inverse(eta)
        deviance = float(np.sum(family.deviance_residuals(model_fixture.DV, mu)))
        assert deviance == pytest.approx(
            GLM_REG_FIT["deviance"], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_pearson_chi_squared(self, model_fixture):
        """Pearson χ² = Σ[(y-μ)²/V(μ)] should match Stata (= deviance for Gaussian)."""
        family = model_fixture.ModelDesignSpec.family
        eta = model_fixture.IV @ model_fixture.CoefResults.betas
        mu = family.link_inverse(eta)
        variance_mu = family.variance(mu)
        pearson = float(np.sum((model_fixture.DV - mu) ** 2 / variance_mu))
        assert pearson == pytest.approx(
            GLM_REG_FIT["pearson"], rel=APPROX_REL, abs=APPROX_ABS
        )


# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN VALIDATION: REGRESSION COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionCoefficients:
    """Validate coefficient table against Stata golden values.

    Notes
    -----
    Formulaic encodes C(black) with "black" as reference (alphabetical),
    so the kept level is "white" with β_white = +2.55.  Stata uses white
    as reference and reports β_black = -2.55.  These are equivalent
    (same magnitude, opposite sign).  The intercept absorbs the offset:
        RP_intercept = Stata_intercept + Stata_β_black
    """

    # -- educ (invariant to reference level) --

    def test_educ_coefficient(self, model_fixture):
        """educ coefficient should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[0]
        betas = model_fixture.CoefResults.betas.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "educ" in t)
        assert betas[idx] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_educ_std_error(self, model_fixture):
        """educ standard error should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[0]
        se = model_fixture.CoefResults.std_error.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "educ" in t)
        assert se[idx] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_educ_z_statistic(self, model_fixture):
        """educ z-statistic should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[0]
        z = model_fixture.CoefResults.test_stat.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "educ" in t)
        assert z[idx] == pytest.approx(golden["z"], rel=1e-2, abs=0.01)

    def test_educ_confidence_interval(self, model_fixture):
        """educ 95% CI should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[0]
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "educ" in t)
        ci_lo = float(model_fixture.CoefResults.conf_int_lower[idx])
        ci_hi = float(model_fixture.CoefResults.conf_int_upper[idx])
        assert ci_lo == pytest.approx(
            golden["95% Conf. Interval"][0], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert ci_hi == pytest.approx(
            golden["95% Conf. Interval"][1], rel=APPROX_REL, abs=APPROX_ABS
        )

    # -- jobexp (invariant to reference level) --

    def test_jobexp_coefficient(self, model_fixture):
        """jobexp coefficient should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[1]
        betas = model_fixture.CoefResults.betas.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "jobexp" in t)
        assert betas[idx] == pytest.approx(
            golden["Coef."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_jobexp_std_error(self, model_fixture):
        """jobexp standard error should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[1]
        se = model_fixture.CoefResults.std_error.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "jobexp" in t)
        assert se[idx] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_jobexp_z_statistic(self, model_fixture):
        """jobexp z-statistic should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[1]
        z = model_fixture.CoefResults.test_stat.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "jobexp" in t)
        assert z[idx] == pytest.approx(golden["z"], rel=1e-2, abs=0.01)

    def test_jobexp_confidence_interval(self, model_fixture):
        """jobexp 95% CI should match golden value."""
        golden = GLM_REG_REGRESSION_TABLE[1]
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "jobexp" in t)
        ci_lo = float(model_fixture.CoefResults.conf_int_lower[idx])
        ci_hi = float(model_fixture.CoefResults.conf_int_upper[idx])
        assert ci_lo == pytest.approx(
            golden["95% Conf. Interval"][0], rel=APPROX_REL, abs=APPROX_ABS
        )
        assert ci_hi == pytest.approx(
            golden["95% Conf. Interval"][1], rel=APPROX_REL, abs=APPROX_ABS
        )

    # -- race coefficient (sign-flipped due to reference level swap) --
    # Stata: white=ref, β_black = -2.55136
    # ResearchPy: black=ref, β_white = +2.55136
    # |β| and SE are identical; z flips sign; p-value unchanged.

    def test_race_coefficient_magnitude(self, model_fixture):
        """Race coefficient magnitude should match golden value (sign may differ due to reference)."""
        golden = GLM_REG_REGRESSION_TABLE[4]  # Stata's black coefficient
        betas = model_fixture.CoefResults.betas.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "black" in t.lower())
        assert abs(betas[idx]) == pytest.approx(
            abs(golden["Coef."]), rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_race_std_error(self, model_fixture):
        """Race SE should match golden value (invariant to reference level)."""
        golden = GLM_REG_REGRESSION_TABLE[4]
        se = model_fixture.CoefResults.std_error.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "black" in t.lower())
        assert se[idx] == pytest.approx(
            golden["Std. Err."], rel=APPROX_REL, abs=APPROX_ABS
        )

    def test_race_z_magnitude(self, model_fixture):
        """Race |z| should match golden value (sign flips with reference swap)."""
        golden = GLM_REG_REGRESSION_TABLE[4]
        z = model_fixture.CoefResults.test_stat.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "black" in t.lower())
        assert abs(z[idx]) == pytest.approx(abs(golden["z"]), rel=1e-2, abs=0.01)

    def test_race_p_value(self, model_fixture):
        """Race p-value should match golden value (invariant to sign)."""
        golden = GLM_REG_REGRESSION_TABLE[4]
        pvals = model_fixture.CoefResults.test_pval.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "black" in t.lower())
        assert pvals[idx] == pytest.approx(golden["p-value"], abs=1e-3)

    def test_race_ci_width(self, model_fixture):
        """Race CI width should match golden value (invariant to reference)."""
        golden = GLM_REG_REGRESSION_TABLE[4]
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "black" in t.lower())
        ci_lo = float(model_fixture.CoefResults.conf_int_lower[idx])
        ci_hi = float(model_fixture.CoefResults.conf_int_upper[idx])
        rp_width = ci_hi - ci_lo
        golden_width = golden["95% Conf. Interval"][1] - golden["95% Conf. Interval"][0]
        assert rp_width == pytest.approx(golden_width, rel=APPROX_REL, abs=APPROX_ABS)

    # -- intercept (absorbed reference level offset) --
    # RP_intercept = Stata_intercept + Stata_β_black
    # because RP reference is "black" (baseline includes black effect)

    def test_intercept_coefficient(self, model_fixture):
        """Intercept should equal Stata intercept + Stata β_black (reference shift)."""
        golden_intercept = GLM_REG_REGRESSION_TABLE[5]["Coef."]
        golden_black = GLM_REG_REGRESSION_TABLE[4]["Coef."]
        expected = golden_intercept + golden_black  # -4.72676 + (-2.55136) = -7.27812

        betas = model_fixture.CoefResults.betas.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "intercept" in t.lower())
        assert betas[idx] == pytest.approx(expected, rel=APPROX_REL, abs=APPROX_ABS)

    def test_intercept_std_error(self, model_fixture):
        """Intercept SE should be positive and consistent with z = β/SE.

        Note: The intercept SE differs from Stata because the reference
        level is different (black=ref vs white=ref), changing which group
        the intercept represents and thus its variability.
        """
        se = model_fixture.CoefResults.std_error.flatten()
        betas = model_fixture.CoefResults.betas.flatten()
        z = model_fixture.CoefResults.test_stat.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "intercept" in t.lower())
        assert se[idx] > 0, "Intercept SE should be positive"
        assert z[idx] == pytest.approx(betas[idx] / se[idx], rel=1e-10)

    def test_intercept_z_statistic(self, model_fixture):
        """Intercept z-statistic should be consistent with β/SE."""
        betas = model_fixture.CoefResults.betas.flatten()
        se = model_fixture.CoefResults.std_error.flatten()
        z = model_fixture.CoefResults.test_stat.flatten()
        terms = list(model_fixture.CoefResults.term)
        idx = next(i for i, t in enumerate(terms) if "intercept" in t.lower())
        assert z[idx] == pytest.approx(betas[idx] / se[idx], rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# RETURN TYPE AND OUTPUT FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestReturnTypes:
    """Validate that results() returns correct types and structures."""

    def test_returns_dataframe_tuple(self, results_df):
        """results(return_type='Dataframe') should return tuple of 3 DataFrames."""
        assert isinstance(results_df, tuple)
        assert len(results_df) == 3
        for item in results_df:
            assert isinstance(item, pd.DataFrame)

    def test_returns_dictionary_tuple(self, results_dict):
        """results(return_type='Dictionary') should return tuple of 3 dicts."""
        assert isinstance(results_dict, tuple)
        assert len(results_dict) == 3
        for item in results_dict:
            assert isinstance(item, dict)

    def test_coefficient_table_has_expected_columns(self, results_df):
        """Coefficient DataFrame should have standard column names."""
        coef_df = results_df[2]
        expected_cols = {"Coef.", "Std. Err.", "z", "p-value", "95% Conf. Interval"}
        actual_cols = set(coef_df.columns)
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES AND CONSISTENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

class TestConsistency:
    """Internal consistency checks that must hold regardless of data."""

    def test_z_equals_coef_over_se(self, model_fixture):
        """z-statistic should equal Coef / SE for all parameters."""
        betas = model_fixture.CoefResults.betas.flatten()
        se = model_fixture.CoefResults.std_error.flatten()
        z = model_fixture.CoefResults.test_stat.flatten()
        expected_z = betas / se
        np.testing.assert_allclose(z, expected_z, rtol=1e-10)

    def test_p_values_in_range(self, model_fixture):
        """All p-values should be between 0 and 1."""
        pvals = model_fixture.CoefResults.test_pval.flatten()
        assert np.all(pvals >= 0), "Found negative p-value"
        assert np.all(pvals <= 1), "Found p-value > 1"

    def test_confidence_intervals_contain_point_estimate(self, model_fixture):
        """Each 95% CI should contain the point estimate."""
        betas = model_fixture.CoefResults.betas.flatten()
        ci_lo = model_fixture.CoefResults.conf_int_lower.flatten()
        ci_hi = model_fixture.CoefResults.conf_int_upper.flatten()
        for i in range(len(betas)):
            assert ci_lo[i] <= betas[i] <= ci_hi[i], (
                f"CI [{ci_lo[i]}, {ci_hi[i]}] does not contain beta={betas[i]} at index {i}"
            )

    def test_scale_parameter_equals_pearson_over_df(self, model_fixture):
        """Scale parameter should equal Pearson χ² / df_residual."""
        family = model_fixture.ModelDesignSpec.family
        eta = model_fixture.IV @ model_fixture.CoefResults.betas
        mu = family.link_inverse(eta)
        variance_mu = family.variance(mu)
        pearson = float(np.sum((model_fixture.DV - mu) ** 2 / variance_mu))
        df_resid = model_fixture.n - model_fixture.k
        expected_scale = pearson / df_resid
        assert model_fixture.FitStatistics.scale_parameter == pytest.approx(
            expected_scale, rel=1e-10
        )

    def test_dispersion_matches_family_estimate(self, model_fixture):
        """FitStatistics.scale_parameter should equal family.estimate_dispersion()."""
        family = model_fixture.ModelDesignSpec.family
        eta = model_fixture.IV @ model_fixture.CoefResults.betas
        mu = family.link_inverse(eta)
        phi = family.estimate_dispersion(model_fixture.DV, mu, model_fixture.n, model_fixture.k)
        assert model_fixture.FitStatistics.scale_parameter == pytest.approx(phi, rel=1e-10)

    def test_ci_width_increases_with_confidence_level(self, model_fixture):
        """Higher confidence level should produce wider intervals."""
        from scipy.stats import norm

        se = model_fixture.CoefResults.std_error.flatten()

        # 95% CI width
        z95 = norm.ppf(0.975)
        width_95 = 2 * z95 * se

        # 99% CI width
        z99 = norm.ppf(0.995)
        width_99 = 2 * z99 * se

        assert np.all(width_99 > width_95), "99% CI should be wider than 95% CI"
