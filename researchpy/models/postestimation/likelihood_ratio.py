# -*- coding: utf-8 -*-
"""
Likelihood Ratio Test

This module provides the LikelihoodRatioTest class for comparing nested models
using the likelihood ratio test statistic.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from researchpy.containers import TestResults


class LikelihoodRatioTest:
    """
    Likelihood Ratio Test for comparing nested models.

    This class performs likelihood ratio tests to compare:
    - A fitted model vs. its null (intercept-only) model
    - Two arbitrary nested models (full vs. restricted)

    The test statistic is: LR = -2 * (LL_restricted - LL_full)
    which follows a chi-squared distribution with df = k_full - k_restricted.

    Parameters
    ----------
    model : fitted model object
        The full/unrestricted model. Must have attributes:
        - logL: Log-likelihood of the fitted model (list or scalar)
        - k: Number of parameters
        - nobs: Number of observations
        - DV: Dependent variable array
        - IV: Independent variable design matrix
        - _family: Distribution family (e.g., "binomial")
        - _link: Link function (e.g., "logit")
    restricted_model : fitted model object, optional
        The restricted/null model. If None, compares to intercept-only null model.
        Must have logL and k attributes.
    store_null : bool, optional
        Whether to store the fitted null model for inspection. Default is True.

    Attributes
    ----------
    LR_chi2 : float
        The likelihood ratio chi-squared test statistic.
    df : int
        Degrees of freedom for the test.
    p_value : float
        p-value from chi-squared distribution.
    LL_full : float
        Log-likelihood of the full model.
    LL_restricted : float
        Log-likelihood of the restricted/null model.
    null_model : dict or None
        Dictionary containing null model information (if store_null=True and
        restricted_model was not provided).

    Examples
    --------
    Compare fitted model to null (intercept-only):

    >>> from researchpy.models.postestimation import LikelihoodRatioTest
    >>> lr_test = LikelihoodRatioTest(fitted_model)
    >>> lr_test.results()

    Compare two nested models:

    >>> full_model = Logistic("y ~ x1 + x2 + x3", data=df)
    >>> reduced_model = Logistic("y ~ x1", data=df)
    >>> lr_test = LikelihoodRatioTest(full_model, restricted_model=reduced_model)
    >>> print(f"LR Chi2: {lr_test.LR_chi2}, p-value: {lr_test.p_value}")

    Access null model details:

    >>> lr_test = LikelihoodRatioTest(model, store_null=True)
    >>> print(lr_test.null_model['betas'])  # Intercept estimate
    >>> print(lr_test.null_model['log_likelihood'])

    See Also
    --------
    LogisticRegression : Logistic regression model
    """

    def __init__(self, model, restricted_model=None, store_null=True):
        self.model = model
        self.restricted_model = restricted_model
        self.store_null = store_null
        self.null_model = None

        # Extract log-likelihood from full model
        if isinstance(model.logL, list):
            self.LL_full = model.logL[-1] if model.logL else None
        else:
            self.LL_full = model.logL

        # Get restricted model log-likelihood or fit null model
        if restricted_model is not None:
            # Comparing two fitted models
            if isinstance(restricted_model.logL, list):
                self.LL_restricted = restricted_model.logL[-1] if restricted_model.logL else None
            else:
                self.LL_restricted = restricted_model.logL
            self.df_restricted = restricted_model.k
        else:
            # Fit null (intercept-only) model
            self._fit_null_model()
            self.df_restricted = 1  # Only intercept

        # Compute test statistic
        self._compute_test()

    def _fit_null_model(self):
        """
        Fit intercept-only null model based on distribution family and link function.

        The null model fitting is tailored to the specific family/link combination
        of the full model to ensure proper comparison.
        """
        # Resolve the Family instance from ModelDesignSpec; fall back to
        # string-based look-up for backward compatibility.
        family_obj = getattr(self.model, 'ModelDesignSpec', None)
        if family_obj is not None:
            family_obj = getattr(family_obj, 'family', None)

        if family_obj is not None and hasattr(family_obj, 'name'):
            family = family_obj.name          # e.g. "binomial"
            link = family_obj.link            # e.g. "logit"
        else:
            # Legacy fallback
            family = getattr(self.model, '_family', 'binomial')
            link = getattr(self.model, '_link', 'logit')
        DV = self.model.DV
        nobs = self.model.n

        # Create intercept-only design matrix
        IV_null = np.ones((nobs, 1))

        # Smart initialization based on family
        if family == "binomial":
            y_mean = np.mean(DV)
            if 0 < y_mean < 1:
                intercept_init = np.log(y_mean / (1 - y_mean))
            else:
                intercept_init = 0.0
        elif family == "poisson":
            y_mean = np.mean(DV)
            intercept_init = np.log(max(y_mean, 1e-10))
        else:
            # Gaussian/normal
            intercept_init = np.mean(DV)

        # Define objective function based on family/link
        def null_neg_log_likelihood(params):
            params = np.atleast_2d(params).T
            linear_pred = IV_null @ params

            if family == "binomial" and link == "logit":
                p = expit(linear_pred)
                eps = 1e-15
                p = np.clip(p, eps, 1 - eps)
                return -np.sum(DV * np.log(p) + (1 - DV) * np.log(1 - p))

            elif family == "poisson" and link == "log":
                from scipy.special import gammaln
                mu = np.exp(linear_pred)
                mu = np.clip(mu, 1e-10, None)
                return -np.sum(DV * np.log(mu) - mu - gammaln(DV + 1))

            elif family == "gaussian":
                residuals = DV - linear_pred
                return 0.5 * np.sum(residuals ** 2)

            else:
                raise NotImplementedError(
                    f"Null model fitting not implemented for family='{family}', link='{link}'"
                )

        # Define gradient based on family/link
        def null_gradient(params):
            params = np.atleast_2d(params).T
            linear_pred = IV_null @ params

            if family == "binomial" and link == "logit":
                p = expit(linear_pred)
                return (-IV_null.T @ (DV - p)).flatten()

            elif family == "poisson" and link == "log":
                mu = np.exp(linear_pred)
                return (-IV_null.T @ (DV - mu)).flatten()

            elif family == "gaussian":
                return (IV_null.T @ (linear_pred - DV)).flatten()

            else:
                return None  # Will use numerical gradient

        # Fit null model
        null_result = minimize(
            fun=null_neg_log_likelihood,
            x0=np.array([intercept_init]),
            jac=null_gradient,
            method='BFGS',
            options={'maxiter': 100, 'gtol': 1e-6}
        )

        self.LL_restricted = -null_result.fun

        # Store null model details if requested
        if self.store_null:
            self.null_model = {
                'betas': null_result.x.reshape(-1, 1),
                'log_likelihood': -null_result.fun,
                'converged': null_result.success,
                'n_iterations': null_result.nit,
                'family': family,
                'link': link,
                'nobs': nobs
            }

    def _compute_test(self):
        """Compute the likelihood ratio test statistic and p-value."""
        # LR statistic: -2 * (LL_restricted - LL_full)
        self.LR_chi2 = -2 * (self.LL_restricted - self.LL_full)

        # Degrees of freedom
        if self.restricted_model is not None:
            self.df = self.model.k - self.restricted_model.k
        else:
            self.df = self.model.k - 1  # Full model params minus intercept

        # p-value from chi-squared distribution
        self.p_value = chi2.sf(self.LR_chi2, self.df)

    def results(self, return_type="Dataframe", decimals=4):
        """
        Return the likelihood ratio test results as a ``TestResults`` dataclass.

        Parameters
        ----------
        return_type : str, optional
            Format of returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        decimals : int, optional
            Number of decimal places for rounding. Default is 4.

        Returns
        -------
        TestResults
            A dataclass with fields:
            - ``test_name``: ``"Likelihood Ratio Test"``
            - ``statistics``: Test statistics (DataFrame or dict)
            - ``details``: dict with null model info (if available)

            Supports tuple unpacking::

                name, stats, details = lr_test.results()

            Or attribute access::

                result = lr_test.results()
                result.statistics
                result.details
        """
        results_dict = {
            "LR Chi-squared": round(self.LR_chi2, decimals),
            "Degrees of Freedom": self.df,
            "p-value": round(self.p_value, decimals) if self.p_value >= 10**(-decimals) else f"<{10**(-decimals)}",
            "Log-Likelihood (Full)": round(self.LL_full, decimals),
            "Log-Likelihood (Restricted)": round(self.LL_restricted, decimals)
        }

        if return_type.lower() in ["dataframe", "df"]:
            statistics = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
        elif return_type.lower() in ["dictionary", "dict"]:
            statistics = results_dict
        else:
            raise ValueError(
                "Not a valid return type option, please use either "
                "'Dataframe' or 'Dictionary'."
            )

        return TestResults(
            test_name="Likelihood Ratio Test",
            statistics=statistics,
            details={"null_model": self.null_model} if self.null_model is not None else None,
        )

    def summary(self):
        """Print a formatted summary of the likelihood ratio test."""
        print("\n" + "=" * 60)
        print("Likelihood Ratio Test")
        print("=" * 60)
        print(f"  LR Chi-squared({self.df}) = {self.LR_chi2:.4f}")
        print(f"  Prob > Chi-squared   = {self.p_value:.4g}")
        print("-" * 60)
        print(f"  Log-Likelihood (Full Model):       {self.LL_full:.4f}")
        print(f"  Log-Likelihood (Restricted Model): {self.LL_restricted:.4f}")

        if self.restricted_model is not None:
            print(f"\n  Comparison: Full model ({self.model.k} params) vs. "
                  f"Restricted model ({self.restricted_model.k} params)")
        else:
            print(f"\n  Comparison: Full model ({self.model.k} params) vs. "
                  f"Null model (intercept only)")

        print("=" * 60 + "\n")

    def __repr__(self):
        return (f"LikelihoodRatioTest(LR_chi2={self.LR_chi2:.4f}, "
                f"df={self.df}, p_value={self.p_value:.4g})")


# Convenience alias
LRTest = LikelihoodRatioTest
