import numpy as np

from researchpy.models.base import CoreModel
from researchpy.containers import SolverOptions, ModelResults

from researchpy.objective_functions.likelihood import neg_log_likelihood, gradient_neg_log_likelihood
from researchpy.optimize.iterative_algorithms import scipy_minimize, newton_raphson
from researchpy.models.postestimation import LikelihoodRatioTest

from researchpy.predict import predict


class GeneralModel(CoreModel):
    """

    This is a subclass of core_model for generalized statistical models such as logistic, poisson, and etc.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal", solver_options=None,
                 table_decimals=None):

        self.__name__ = "Researchpy.GeneralModel"

        if data is None:
            data = {}

        # Build SolverOptions: start with GeneralModel defaults, then overlay user input.
        base_defaults = SolverOptions(
            method="mle",
            algorithm="newton-raphson",
            obj_function="log-likelihood",
            tol=1e-7,
            max_iter=300,
            display=True
        )

        if solver_options is None:
            resolved_solver_options = base_defaults
        elif isinstance(solver_options, SolverOptions):
            resolved_solver_options = solver_options
        else:
            # User passed a dict — override base defaults with user values
            resolved_solver_options = base_defaults.with_overrides(solver_options)


        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_options=resolved_solver_options,
                         table_decimals=table_decimals)


    def __initialize_betas(self, initial_betas=None, initial_betas_method=None):

        if initial_betas is not None and initial_betas_method is not None:
            return("Warning: Both initial_betas and initial_betas_method were provided. Ignoring initial_betas_method and using provided initial_betas.")

        if initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.CoefResults.betas = initial_betas
            else:
                raise ValueError(f"initial_betas must be a numpy array of shape ({self.k}, 1), but got {initial_betas.shape}")

        elif initial_betas_method.lower() == "ols":
            self._CoreModel__ols_fit()

        # Default initialization based on model type
        else:
            if self.__name__ in ["researchpy.Logistic", "researchpy.Logit"]:
                """Initialize betas with smarter starting values."""
                # Start with zeros (better than OLS for binary outcomes)
                betas = np.ones((self.n, self.k))

                # Set intercept to log odds of outcome proportion
                y_mean = np.mean(self.DV)
                if 0 < y_mean < 1:
                    betas[0] = np.log(y_mean / (1 - y_mean))

                self.CoefResults.betas = betas.reshape(-1, 1)


    def _neg_log_likelihood(self, params, *args, **kwargs):

        return neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link,
            tracker=self._OptimizationTracker
        )


    def _gradient_neg_log_likelihood(self, params):
        """Gradient of negative log-likelihood."""
        return gradient_neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link
        )


    def __fit_model(self):
        """Fit the model using scipy.optimize with fallback."""

        self.logL = []
        self.nfev = -1
        converged = False

        # Try scipy.optimize first
        try:
            if self.solver_options.display:
                print(f"Starting optimization with {self.solver_options.algorithm}...\n")

            # Fit the full model
            result = scipy_minimize(
                fun=self._neg_log_likelihood,
                x0=self.CoefResults.betas.flatten(),
                jac=self._gradient_neg_log_likelihood,
                method=self.solver_options.algorithm,
                options=self.solver_options.to_scipy_options(),
                callback=None
            )

            if result.success:
                self.CoefResults.betas = result.x.reshape(-1, 1)
                self.logL.append(-result.fun)
                self.nfev = result.nfev
                converged = True

                # Perform Likelihood Ratio Test (full model vs null)
                self._lr_test = LikelihoodRatioTest(self, store_null=True)

                if self.solver_options.display:
                    print(f"")
                    print(f"")
            else:
                if self.solver_options.display:
                    print(f"Warning: scipy optimization did not converge ({result.message})")
                    print("Falling back to Newton-Raphson...")

        except Exception as e:
            if self.solver_options.display:
                print(f"scipy.optimize failed: {e}")
                print("Falling back to Newton-Raphson...")

        # Fallback to Newton-Raphson if scipy failed
        if not converged:
            betas, self.logL = newton_raphson(
                IV=self.IV,
                DV=self.DV,
                betas=self.CoefResults.betas,
                tol=self.solver_options.tol,
                max_iter=self.solver_options.max_iter,
                display=self.solver_options.display
            )
            self.CoefResults.betas = betas

            # Perform Likelihood Ratio Test for Newton-Raphson path
            self.nfev = len(self.logL)
            self._lr_test = LikelihoodRatioTest(self, store_null=True)

        # ---- Populate FitStatistics dataclass from LR test results ----
        ll_full = self.logL[-1] if self.logL else None
        ll_null = self._lr_test.LL_restricted

        self.FitStatistics.log_likelihood = ll_full
        self.FitStatistics.test_stat_name = "LR Chi^2"
        self.FitStatistics.test_stat = self._lr_test.LR_chi2
        self.FitStatistics.df_model = self._lr_test.df
        self.FitStatistics.test_pval = self._lr_test.p_value

        # AIC = -2·LL + 2·k
        if ll_full is not None:
            self.FitStatistics.aic = -2 * ll_full + 2 * self.k
            # BIC = -2·LL + k·ln(n)
            self.FitStatistics.bic = -2 * ll_full + self.k * np.log(self.n)

        # McFadden's Pseudo R² = 1 - (LL_full / LL_null)
        if ll_full is not None and ll_null is not None and ll_null != 0:
            self.FitStatistics.r_squared_pseudo = 1 - (ll_full / ll_null)

        self.FitStatistics.additional_stats = {
            "n_iterations": self.nfev,
            "ll_null": ll_null,
            "converged": converged or (self.logL is not None and len(self.logL) > 0),
        }


    def predict(self, estimate=None, trans=None, decimals=4, **kwargs):

        return predict(self, estimate=estimate, trans=trans, decimals=decimals, **kwargs)


    #--------------------------------------------------------------------------------------#
    #                  Results Methods (new flow)                                          #
    #--------------------------------------------------------------------------------------#
    def _get_fit_statistics(self, table_decimals=None, **kwargs) -> dict:
        """
        Build the fit statistics dictionary for MLE-based models.

        Reads from ``self.FitStatistics`` dataclass which is populated during
        model fitting.

        Parameters
        ----------
        table_decimals : dict or None
            Override decimal settings.

        Returns
        -------
        dict
            Fit statistics as {label: [formatted_string]} pairs.
        """
        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        df_model = self.FitStatistics.df_model if self.FitStatistics.df_model is not None else ""
        test_stat_model = round(float(self.FitStatistics.test_stat), self._table_decimals.get('test_stat_model', 4))
        test_pval_model = round(float(self.FitStatistics.test_pval), self._table_decimals.get('test_stat_p', 4))
        log_likelihood = round(float(self.FitStatistics.log_likelihood), self._table_decimals.get('log_likelihood', 4))
        pseudo_r2 = round(float(self.FitStatistics.r_squared_pseudo), self._table_decimals.get('R-squared', 4))
        n_iter = self.FitStatistics.additional_stats.get("n_iterations") if self.FitStatistics.additional_stats else None

        fit_statistics = {
            "n": [f"N = {self.n}"],
            "test_stat_model": [f"LR Chi^2({df_model}) = {test_stat_model}"],
            "test_pval_model": [f"Prob > Chi^2 = {test_pval_model}"],
            "log_likelihood": [f"Log likelihood = {log_likelihood}"],
            "pseudo_r2": [f"Pseudo R^2 = {pseudo_r2}"],
            "n_iterations": [f"N iterations = {n_iter}"],
        }

        return fit_statistics


    def _get_from_child(self, key: str, **kwargs):
        """
        Return LogisticRegression-specific data requested by the parent class.

        Parameters
        ----------
        key : str
            Identifier for the requested data. Supported keys:
            - ``"default_transform"`` : ``np.exp`` (odds ratios)
            - ``"test_stat_name"`` : ``"z"``
            - ``"additional_fit_stats"`` : extra logistic-specific fit stats
            - ``"report_options"`` : default reporting configuration
        **kwargs
            Additional context from the parent.

        Returns
        -------
        object
            The requested value, or ``None`` for unrecognized keys.
        """
        if key == "default_transform":
            return np.exp
        if key == "test_stat_name":
            return "z"
        if key == "additional_fit_stats":
            return {}
        if key == "report_options":
            return {"report_as": "or", "beta_type": "odds ratio"}

        return super()._get_from_child(key, **kwargs)


    def _get_coefficient_results(self, na_rep='', pretty_format=True, table_decimals=None,
                                    coef_transform=None) -> dict:
        """
        Build the coefficient results table using the parent CoreModel method.

        Parameters
        ----------
        na_rep : object
            Representation for missing values.
        pretty_format : bool
            Whether to format the output for display.
        table_decimals : dict or None
            Override decimal settings.
        coef_transform : callable or None
            Transformation function for coefficients and CIs (e.g., ``np.exp``
            to convert log-odds to odds ratios). Applied before rounding.

        Returns
        -------
        dict
            Coefficient table as a dictionary suitable for DataFrame conversion.
        """
        return super()._get_coefficient_results(pretty_format=pretty_format,
                                                table_decimals=table_decimals,
                                                coef_transform=coef_transform)


    def _get_ModelResults(self, return_type="Dataframe", pretty_format=True,
                          table_decimals=None, coef_transform=None) -> ModelResults:
        """
        Assemble the ModelResults dataclass for generalized (MLE) models.

        MLE models have no sum-of-squares decomposition, so ``model_table``
        is always ``None``.

        Parameters
        ----------
        return_type : str, optional
            ``"Dataframe"`` or ``"Dictionary"``. Default is ``"Dataframe"``.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places.
        coef_transform : callable or None
            Transformation function for coefficients and CIs (e.g., ``np.exp``
            for odds ratios). Applied before rounding. Default is ``None``.

        Returns
        -------
        ModelResults
        """
        ## Checking for valid return type ##
        if return_type.lower() not in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", "dictionary", "dict"]:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the fit statistics and coefficient results
        fit_statistics = self._get_fit_statistics(table_decimals=self._table_decimals)

        table_from_child = self._get_from_child("additional_fit_stats", fit_statistics=fit_statistics)

        coefficients = self._get_coefficient_results(pretty_format=pretty_format,
                                                     table_decimals=self._table_decimals,
                                                     coef_transform=coef_transform)

        self.ModelResults = ModelResults(
            model_name=self._get_model_display_name(),
            fit_statistics=fit_statistics,
            model_table=None,  # MLE models have no SS decomposition
            coefficients=coefficients,
        )

        return self.ModelResults


    def _get_results(self, return_type="Dataframe", pretty_format=True,
                     table_decimals=None, coef_transform=None):
        """
        Return the regression results.

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either ``"Dataframe"`` or
            ``"Dictionary"``. Default is ``"Dataframe"``.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.
        coef_transform : callable or None
            Transformation function for coefficients and CIs (e.g., ``np.exp``).

        Returns
        -------
        tuple
            If return_type is "Dataframe": (fit_statistics_df, None, coefficients_df)
            If return_type is "Dictionary": (fit_statistics_dict, None, coefficients_dict)
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        mr = self._get_ModelResults(return_type=return_type,
                                    pretty_format=pretty_format,
                                    table_decimals=self._table_decimals,
                                    coef_transform=coef_transform)


        if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe"]:
            if mr.model_table is None:
                return (
                    self.ModelResults.as_dataframe("fit_statistics", mr.fit_statistics),
                    None,
                    self.ModelResults.as_dataframe("coefficients", mr.coefficients)
                )

        else:
            return mr.fit_statistics, None, mr.coefficients


    #---------------------------------------------------------------------------#
    #                           Shared Summary Methods                          #
    #---------------------------------------------------------------------------#
    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the summary header for generalized models.

        Shows model name and log-likelihood value.

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        lines = [self._get_model_display_name()]

        if self.FitStatistics.log_likelihood is not None:
            lines.append(f"Log likelihood = {self.FitStatistics.log_likelihood:.4f}")

        return lines


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the summary header for generalized models.

        When *descriptives_df* is provided the values are read from that
        DataFrame via ``to_string(header=False)`` so the summary is driven
        entirely by the DataFrames returned from ``self.results()``.

        Otherwise falls back to rendering from ``self`` attributes directly.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()`` (index-oriented:
            stat names as index, values in column 0).

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        import pandas as pd

        if descriptives_df is not None:
            # Convert to index-oriented for to_string.
            if self.ModelResults.fit_statistics is not None:
                table = self.ModelResults.as_dataframe("fit_statistics", self.ModelResults.fit_statistics)
        else:
            if not isinstance(descriptives_df, pd.DataFrame):
                table = pd.DataFrame.from_dict(descriptives_df)
            else:
                table = descriptives_df.copy()


            desc_lines = table.to_string(
                header=False,
                index=False,
                justify="right"
            ).split("\n")

            return desc_lines


        # Fallback: build from FitStatistics dataclass
        lines = [f"Number of obs = {self.n:>8}"]

        if self.FitStatistics.test_stat is not None and self.FitStatistics.df_model is not None:
            lines.append(f"LR chi2({int(self.FitStatistics.df_model)})    = {self.FitStatistics.test_stat:>8.4f}")

        if self.FitStatistics.test_pval is not None:
            lines.append(f"Prob > chi2   = {self.FitStatistics.test_pval:>8.4f}")

        n_iter = self.FitStatistics.additional_stats.get("n_iterations") if self.FitStatistics.additional_stats else None
        if n_iter is not None:
            lines.append(f"N iterations  = {n_iter:>8}")

        return lines



