# -*- coding: utf-8 -*-
"""
Likelihood Ratio Test

This module provides the LikelihoodRatioTest class for comparing nested models
using the likelihood ratio test statistic.
"""

from numpy import (
    ones, array,
)

from researchpy.statistics import _compute_pvalue
from researchpy.containers import TestResults, FitStatistics
from researchpy.optimize import (
    IRLS, mle_estimation_principal,
)


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
        - FitStatistics.log_likelihood: Log-likelihood of the fitted model
        - k: Number of parameters
        - n: Number of observations
        - DV: Dependent variable array
        - ModelDesignSpec.family: A Family instance
        - SolverOptions: Solver configuration
    restricted_model : fitted model object, optional
        The restricted/null model. If None, compares to intercept-only null model.
        Must have FitStatistics.log_likelihood and k attributes.
    store_null : bool, optional
        Whether to store the fitted null model details in
        ``FitStatistics.additional_stats["null_model"]``. Default is True.
    display_summary : bool, optional
        Whether to print the summary table on initialization. Default is True.
    table_decimals : dict, optional
        Dictionary overriding decimal places for formatted output.

    Attributes
    ----------
    FitStatistics : FitStatistics
        Dataclass containing all computed statistics:
        - ``test_stat``: LR chi-squared test statistic
        - ``test_stat_name``: "LR Chi^2"
        - ``df_model``: Degrees of freedom
        - ``test_pval``: p-value from chi-squared distribution
        - ``log_likelihood``: Log-likelihood of the full model
        - ``log_likelihood_restricted``: Log-likelihood of the restricted/null model
        - ``r_squared_pseudo``: McFadden's Pseudo R²
        - ``additional_stats["null_model"]``: Null model details (if store_null=True)

    Examples
    --------
    Compare fitted model to null (intercept-only):

    >>> from researchpy.models.postestimation import LikelihoodRatioTest
    >>> lr_test = LikelihoodRatioTest(fitted_model, display_summary=False)
    >>> lr_test.FitStatistics.test_stat  # LR chi-squared value
    >>> lr_test.FitStatistics.test_pval  # p-value

    Compare two nested models:

    >>> full_model = Logistic("y ~ x1 + x2 + x3", data=df)
    >>> reduced_model = Logistic("y ~ x1", data=df)
    >>> lr_test = LikelihoodRatioTest(full_model, restricted_model=reduced_model)

    Access null model details:

    >>> lr_test = LikelihoodRatioTest(model, store_null=True, display_summary=False)
    >>> lr_test.FitStatistics.additional_stats["null_model"]["betas"]
    >>> lr_test.FitStatistics.additional_stats["null_model"]["log_likelihood"]

    See Also
    --------
    LogisticRegression : Logistic regression model
    """

    def __init__(self, model, restricted_model=None, store_null=True, display_summary=True, table_decimals=None, **kwargs):
        self.model = model
        self.restricted_model = restricted_model
        self.store_null = store_null
        self.FitStatistics = FitStatistics()

        # Checking to see if the `self._table_decimals` attribute is defined, if it's not then check to see
        # if `model._table_decimals` attributes is defined (should always be), and if not then create it. Then if
        # a table_decimals dictionary is passed, update self._table_decimals with the provided
        # This is used to specify the number of decimal places to round to for different statistics in the summary table.
        if not hasattr(self, "_table_decimals"):
            if hasattr(model, "_table_decimals"):
                self._table_decimals = model._table_decimals

            else:
                self._table_decimals = {
                    "Coef."             : 2, "Std. Err.": 3, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                    "Root MSE"          : 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                    'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4
                }

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Extract log-likelihood from full model
        self.FitStatistics.log_likelihood = model.FitStatistics.log_likelihood

        # Get restricted model log-likelihood or fit null model
        if restricted_model is not None:
            # Comparing two fitted models
            if restricted_model.FitStatistics.log_likelihood:
                self.FitStatistics.log_likelihood_restricted = restricted_model.FitStatistics.log_likelihood
            else:
                self.FitStatistics.log_likelihood_restricted = None

            self.FitStatistics.df_restricted = restricted_model.k

        else:
            # Fit null (intercept-only) model
            self._fit_null_model()
            self.FitStatistics.df_restricted = 1 # Only intercept

        # Compute test statistic
        self._compute_test()

        if display_summary:
            self.summary()


    def _fit_null_model(self):
        """
        Fit intercept-only null model using user specified solver algorithm.

        Uses the same Family instance and solver approach as the full model
        to ensure consistent likelihood computation.
        """
        # Resolve the Family instance from ModelDesignSpec
        family = getattr(self.model, 'ModelDesignSpec', None)
        if family is not None:
            family = getattr(family, 'family', None)

        if family is None or not hasattr(family, 'name'):
            raise ValueError(
                "Cannot fit null model: the full model must have a "
                "ModelDesignSpec.family attribute (a Family instance)."
            )

        DV = self.model.DV
        self.FitStatistics.n = self.model.n

        # Create intercept-only design matrix
        IV_null = ones((self.FitStatistics.n, 1))

        # Smart initialization via Family
        intercept_init = family.initial_intercept(DV)

        # Fit null model using user specified solver algorithm
        # consistent with how GeneralizedLinearModel.fit() operates
        options = self.model.SolverOptions.to_scipy_options() | {"family" : self.model.ModelDesignSpec.family,
                                                                 "IV"     : IV_null,
                                                                 "DV"     : self.model.DV,
                                                                 "display": False,
                                                                 }

        null_result = mle_estimation_principal(fun=lambda p, *a: 0, x0=array([intercept_init]), method=IRLS,
                                               options=options
                                               )

        self.FitStatistics.log_likelihood_restricted = -null_result.fun
        self.FitStatistics.n_iterations = null_result.nit

        # Store null model details if requested
        if self.store_null:
            self.FitStatistics.additional_stats["null_model"] = {
                'betas': null_result.x.reshape(-1, 1),
                'log_likelihood': self.FitStatistics.log_likelihood_restricted,
                'converged': null_result.success,
                'n_iterations': null_result.nit,
                'family': family.name,
                'link': family.link,
                'n': self.FitStatistics.n
            }


    def _compute_test(self):
        """Compute the likelihood ratio test statistic and p-value."""
        # LR statistic: -2 * (LL_restricted - LL_full)
        self.FitStatistics.test_stat_name = "LR Chi^2"
        self.FitStatistics.test_stat = -2 * (self.FitStatistics.log_likelihood_restricted - self.model.FitStatistics.log_likelihood)

        # Degrees of freedom
        if self.restricted_model is not None:
            self.FitStatistics.df_model = self.model.k - self.restricted_model.k
        else:
            self.FitStatistics.df_model = self.model.k - 1  # Full model params minus intercept

        # p-value from chi-squared distribution
        self.FitStatistics.test_pval = _compute_pvalue(
                self.FitStatistics.test_stat,
                "chi2",
                df=self.FitStatistics.df_model,
                alternative="greater",
        )

        # McFadden's Pseudo R² = 1 - (LL_full / LL_null)
        if self.FitStatistics.log_likelihood is not None and self.FitStatistics.log_likelihood_restricted is not None and self.FitStatistics.log_likelihood_restricted != 0:
            self.FitStatistics.r_squared_pseudo = 1 - (self.FitStatistics.log_likelihood / self.FitStatistics.log_likelihood_restricted)


    def _get_fit_statistics(self, table_decimals=None, **kwargs):

        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        df_model = self.FitStatistics.df_model if self.FitStatistics.df_model is not None else ""
        test_stat_model = round(float(self.FitStatistics.test_stat), self._table_decimals.get('test_stat_model', 4))
        test_pval_model = round(float(self.FitStatistics.test_pval), self._table_decimals.get('test_stat_p', 4))
        log_likelihood = round(float(self.FitStatistics.log_likelihood), self._table_decimals.get('log_likelihood', 4))
        log_likelihood_restricted = round(float(self.FitStatistics.log_likelihood_restricted), self._table_decimals.get('log_likelihood', 4))
        pseudo_r2 = round(float(self.FitStatistics.r_squared_pseudo), self._table_decimals.get('R-squared', 4))
        n_iter = self.FitStatistics.n_iterations if self.FitStatistics.n_iterations is not None else None

        fit_statistics = {
            "test_stat_model": [f"LR Chi^2({df_model}) = {test_stat_model}"],
            "test_pval_model": [f"Prob > Chi^2 = {test_pval_model}"],
            "log_likelihood": [f"Log-Likelihood (Full) = {log_likelihood}"],
            "log_likelihood_restricted": [f"Log-Likelihood (Restricted) = {log_likelihood_restricted}"],
            "pseudo_r2": [f"Pseudo R^2 = {pseudo_r2}"],
            "n_iterations": [f"N iterations = {n_iter}"],
        }

        return fit_statistics


    def _get_TestResults(self, return_type="Dataframe", pretty_format=True, table_decimals=None,):
        ## Checking for valid return type ##
        if return_type.lower() not in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", "dictionary", "dict"]:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the fit statistics and coefficient results
        fit_statistics = self._get_fit_statistics(table_decimals=self._table_decimals)

        return fit_statistics


    def results(self, table_decimals=None):
        """
        Return the likelihood ratio test results as a ``TestResults`` dataclass.

        Parameters
        ----------
        table_decimals : dict, optional
            Dictionary specifying the number of decimal places for each statistic.
            Default is None, which uses the class's default settings.

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
        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        fit_statistics = self._get_fit_statistics(table_decimals=self._table_decimals)

        return TestResults(
                test_name="Likelihood Ratio Test",
                statistics=fit_statistics,
                details={"null_model": self.FitStatistics.additional_stats["null_model"]} if self.FitStatistics.additional_stats.get("null_model", None) is not None else None,
        )


    def summary(self, width=78):
        """Print a formatted summary of the likelihood ratio test."""
        print("\n" + "=" * width)
        print("Likelihood Ratio Test")
        print("=" * width)
        print(f"  {self.FitStatistics.test_stat_name}({self.FitStatistics.df_model}) = {self.FitStatistics.test_stat:.4f}")
        print(f"  Prob > Chi-squared   = {self.FitStatistics.test_pval:.4g}")
        print("-" * width)
        print(f"  Log-Likelihood (Full Model): {self.FitStatistics.log_likelihood:.4f}")
        if self.restricted_model is not None:
            print(f"  Log-Likelihood (Restricted Model): {self.FitStatistics.log_likelihood_restricted:.4f}")
        else:
            print(f"  Log-Likelihood (Null Model): {self.FitStatistics.log_likelihood_restricted:.4f}")


        if self.restricted_model is not None:
            print(f"\n  Comparison: Full model ({self.model.k} params) vs. "
                  f"Restricted model ({self.restricted_model.k} params)"
                  )
        else:
            print(f"\n  Comparison: Full model ({self.model.k} params) vs. "
                  f"Null model (intercept only)"
                  )

        print("=" * width + "\n")


    def __repr__(self):
        return (f"LikelihoodRatioTest({self.FitStatistics.test_stat_name}({self.FitStatistics.df_model}) = {self.FitStatistics.test_stat:.4f}, "
                f"Prob > Chi^2 = {self.FitStatistics.test_pval:.4f})")


# Convenience alias
LRTest = LikelihoodRatioTest
