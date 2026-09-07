import numpy as np

from researchpy.containers import SolverOptions, ModelResults
from researchpy.models.base import BaseModel
from researchpy.models.postestimation import (
    LikelihoodRatioTest, predict, detect_separation
)
from researchpy.optimize import (
    OptimizationTracker,
    ols_estimation_principal, mle_estimation_principal,
    neg_log_likelihood, gradient_neg_log_likelihood,
    IRLS, check_conditioning, CONDITION_NUMBER_THRESHOLD,
)
from researchpy.statistics import _compute_pvalue



class GeneralizedLinearModel(BaseModel):
    """

    This is a subclass of BaseModel for generalized statistical models such as logistic, poisson, and etc.

    """

    def __init__(self, formula, data=None, conf_level=0.95,
                 family="gaussian",
                 link="identity",
                 solver_options=None,
                 table_decimals=None,
                 report_betas_as="coef",
                 initial_betas=None,
                 initial_betas_method="ols",
                 fit=True,
                 display_summary=True,
                 **kwargs):

        if data is None: data = {}

        #-------------------------------------------------#
        # -- Build a SolverOptions dataclass instance. -- #
        #-------------------------------------------------#
        # Subclasses (LinearModel, GeneralizedLinearModel) should resolve their own defaults and pass a fully-formed SolverOptions instance.
        # If None or dict arrives here, we fall back to the SolverOptions dataclass defaults.
        self.SolverOptions = SolverOptions(
                estimation_method="mle",
                obj_function="log-likelihood",
                algorithm="IRLS",
                tol=1e-6,
                tolerance=1e-6,
                logtolerance=0,
                max_iter=300,
                display=True,
                regularization=None,
                alpha=0.0
        )
        if isinstance(solver_options, SolverOptions):
            self.SolverOptions = self.SolverOptions.with_overrides(solver_options.to_dict())
        elif isinstance(solver_options, dict):
            self.SolverOptions = self.SolverOptions.with_overrides(solver_options)


        # -- Resolving coeficient test statistic name --
        if kwargs.get("test_stat_name", None) is not None and kwargs.get("test_stat_name", None) != '':
            test_stat_name = kwargs.pop("test_stat_name")
        else:
            test_stat_name = "z"


        # -- Calling BaseModel initialization method --
        super().__init__(formula=formula, data=data, conf_level=conf_level, family=family, link=link,
                         solver_options=self.SolverOptions, table_decimals=table_decimals,
                         test_stat_name=test_stat_name, **kwargs,
                         )

        self.__name__ = "Researchpy.GeneralizedLinearModel"
        self.ModelDesignSpec.model = self.__name__
        self.ModelDesignSpec.model_display_name = self._get_model_display_name()
        self.ModelDesignSpec.report_betas_as = report_betas_as

        #-------------------------------------------------------------------#
        # -- Initialize an optimization tracker instance for this model. -- #
        #-------------------------------------------------------------------#
        # This tracker can be used by optimization algorithms to
        # store and monitor the optimization process.
        self._OptimizationTracker = OptimizationTracker()

        if fit:
            # -- Initializing betas --
            self._initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

            # -- Fit the model --
            self.fit()

            # -- Compute standard errors and statistics --
            self._compute_coef_stats(confidence=conf_level)

            # -- Build ModelResults (results() sets self.ModelResults internally)
            self.results(report_betas_as=report_betas_as, return_type="Dataframe", pretty_format=True)

            # -- Display the model results summary --
            if display_summary:
                self.summary()


    def _initialize_betas(self, initial_betas: object = None, initial_betas_method: object = None) -> str | None:
        """
        Initialize the model coefficients (betas) based on user input or default methods.

        Priority of initialization:
            1. If `initial_betas` is provided, use it directly.
            2. If `initial_betas_method` using ordinary least-squares if the corresponding method (e.g., "ols") if specified.
            3. If neither is provided, use default initialization based on the model type (e.g., zeros for logistic regression). Default initialization is an array of ones with shape (k, 1), where k is the number of coefficients.
        """

        # -- Check for conflicting initialization options --#
        if initial_betas is not None and initial_betas_method is not None:
            return(
                "Warning: Both initial_betas and initial_betas_method were provided."
                "Ignoring initial_betas_method and using provided initial_betas."
            )

        # -- Initialize betas based on user-provided values --#
        elif initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.CoefResults.betas = initial_betas
            else:
                raise ValueError(f"initial_betas must be a numpy array of shape ({self.k}, 1), but got {initial_betas.shape}")

        # -- Initialize betas ussing an array of zeros --#
        elif initial_betas_method == "zeros":
            betas = np.zeros((self.k, 1))
            if self.__name__ in ["researchpy.LogisticRegression", "researchpy.Logit"]:
                # Set intercept to log odds of outcome proportion
                y_mean = np.mean(self.DV)
                if 0 < y_mean < 1:
                    betas[0] = np.log(y_mean / (1 - y_mean))
            self.CoefResults.betas = betas

        # -- Initialize betas using an array of ones --#
        elif initial_betas_method == "ones":
            self.CoefResults.betas = np.ones((self.k, 1))

        # -- Initialize betas using random values --#
        elif initial_betas_method == "random":
            self.CoefResults.betas = np.random.rand(self.k, 1)

        # -- Initialize betas using ordinary least-squares --#
        elif initial_betas_method.lower() == "ols":
            self.CoefResults.betas = ols_estimation_principal(self.IV, self.DV)

        else:
            # Default initialization for generalized models: array of ones
            self.CoefResults.betas = np.ones((self.k, 1))


    def _neg_log_likelihood(self, params, *args, **kwargs):

        return neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.SolverOptions,
            family=self.ModelDesignSpec.family,
            tracker=self._OptimizationTracker
        )


    def _gradient_neg_log_likelihood(self, params):
        """Gradient of negative log-likelihood."""
        return gradient_neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.SolverOptions,
            family=self.ModelDesignSpec.family
        )


    def fit(self):
        # Initialize betas if not already set (default is empty list from CoefResults)
        if not isinstance(self.CoefResults.betas, np.ndarray) or self.CoefResults.betas.size == 0:
            self.CoefResults.betas = np.ones((self.k, 1))


        if self.SolverOptions.display:
            print(f"Starting optimization with {self.SolverOptions.algorithm}...\n")


        # -- Solving algorithm Iteratively Reweighted Least-squares (IRLS) has to be executed differently than the other algorithms. --#
        if self.SolverOptions.algorithm.lower() in ['irls', 'iterative reweighted least-squares']:

            options = self.SolverOptions.to_scipy_options() | {"family": self.ModelDesignSpec.family,
                                                               "IV": self.IV,
                                                               "DV": self.DV,
                                                               "display": self.SolverOptions.display,}

            result = mle_estimation_principal(lambda p, *a: 0, x0=self.CoefResults.betas.flatten(),
                                              method=IRLS,
                                              callback=None,
                                              options=options
                                              )

        else:
            result = mle_estimation_principal(fun=self._neg_log_likelihood,
                                              x0=self.CoefResults.betas.flatten(),
                                              jac=self._gradient_neg_log_likelihood,
                                              method=self.SolverOptions.algorithm,
                                              callback=None,
                                              options=self.SolverOptions.to_scipy_options()
                                              )


        # -- Evaluating if the optimization converged and storing results accordingly. -- #
        # -- Record convergence diagnostics regardless of outcome --
        self.Diagnostics.converged = bool(result.success)
        self.Diagnostics.n_iterations = getattr(result, "nit", None) or getattr(result, "nfev", None)

        if result.success:
            self.CoefResults.betas = result.x.reshape(-1, 1)
            self.FitStatistics.log_likelihood = -result.fun
            self.FitStatistics.df_residual = self.FitStatistics.n - self.k

            # Perform Likelihood Ratio Test (full model vs null)
            lr_test = LikelihoodRatioTest(self, store_null=True, display_summary=False)
            self.FitStatistics._lr_test = lr_test

            if self.SolverOptions.display:
                print(f"")
                print(f"")

            # ---- Populate FitStatistics dataclass from LR test results ----
            ll_full = -result.fun
            ll_null = lr_test.FitStatistics.log_likelihood_restricted

            self.FitStatistics.test_stat_name = lr_test.FitStatistics.test_stat_name
            self.FitStatistics.test_stat = lr_test.FitStatistics.test_stat
            self.FitStatistics.df_model = lr_test.FitStatistics.df_model
            self.FitStatistics.test_pval = lr_test.FitStatistics.test_pval

            # AIC = -2·LL + 2·k
            self.FitStatistics.aic = -2 * ll_full + 2 * self.k
            # BIC = -2·LL + k·ln(n)
            self.FitStatistics.bic = -2 * ll_full + self.k * np.log(self.n)

            # McFadden's Pseudo R² = 1 - (LL_full / LL_null)
            if lr_test.FitStatistics.r_squared_pseudo:
                self.FitStatistics.r_squared_pseudo = lr_test.FitStatistics.r_squared_pseudo

            elif ll_full is not None and ll_null is not None and ll_null != 0:
                self.FitStatistics.r_squared_pseudo = 1 - (ll_full / ll_null)

            self.FitStatistics.additional_stats = {
                "n_iterations": result.nfev,
                "converged": result.success or (self.FitStatistics.log_likelihood is not None and len(self.FitStatistics.log_likelihood) > 0),
                "log_likelihood_null": ll_null,
            }

        else:
            if self.SolverOptions.display:
                print(f"Warning: {self.SolverOptions.estimation_method} using {self.SolverOptions.algorithm} did not converge ({result.message})")


    def _compute_coef_stats(self, confidence=0.95, distribution="normal", dof=None):
        """
        Compute standard errors, Wald test statistics, p-values, and
        confidence intervals using the GLM Fisher information matrix.

        Uses the Family instance from ModelDesignSpec to compute working
        weights generically across distribution families.

        Notes
        -----
        Covariance matrix: Cov(β) = φ · (X'WX)^{-1}
        where W = diag(working_weights) and φ is the dispersion parameter
        (φ = 1 for binomial and Poisson; estimated for Gaussian/Gamma).

        For Gaussian family: φ = RSS / (n − k), the mean squared error.

        Wald statistic: z = β / SE(β)  [or t for models with estimated dispersion]

        References
        ----------
        McCullagh & Nelder (1989), §2.4 — Estimation of the dispersion parameter.
        """
        family = self.ModelDesignSpec.family
        eta = self.IV @ self.CoefResults.betas       # linear predictor
        mu = family.link_inverse(eta)                # fitted values
        w = family.working_weights(eta).reshape(-1, 1)  # GLM working weights


        # ---------------------------------------------------------------
        # Estimate dispersion parameter φ via the family.
        # Binomial and Poisson return φ = 1 (known);
        # Gaussian and Gamma estimate from the Pearson chi-squared statistic:
        #   φ_hat = Σ[(y - μ)² / V(μ)] / (n - k)
        # ---------------------------------------------------------------
        self.FitStatistics.scale_parameter = family.estimate_dispersion(
            self.DV, mu, self.n, self.k
        )

        # ---------------------------------------------------------------
        # Covariance matrix: Cov(β) = φ · (X'WX)^{-1}
        #
        # Computed via QR decomposition of the weighted design matrix
        # for numerical stability (avoids issues with direct inversion
        # when the information matrix is near-singular):
        #   X_w = X · √W  →  QR = X_w  →  (X'WX)^{-1} = (R'R)^{-1} = R^{-1} R'^{-1}
        #
        # This mirrors the lstsq approach used by the IRLS solver.
        # ---------------------------------------------------------------
        dispersion = self.FitStatistics.scale_parameter
        sqrt_w = np.sqrt(np.clip(w, 1e-15, None))
        X_weighted = self.IV * sqrt_w  # (n, k) weighted design matrix

        cov_method = "inverse"
        conditioning = None

        try:
            # QR decomposition: X_w = Q @ R, where R is (k, k) upper-triangular
            Q, R = np.linalg.qr(X_weighted, mode='reduced')

            # inv() won't raise on *near*-singular R, so check conditioning
            # explicitly via the shared numerical diagnostics helper.
            conditioning = check_conditioning(R, threshold=CONDITION_NUMBER_THRESHOLD)
            if conditioning.rank_deficient:
                raise np.linalg.LinAlgError(
                    f"Ill-conditioned R (cond={conditioning.condition_number:.2e})"
                )

            # (X'WX)^{-1} = (R'R)^{-1} = R^{-1} @ R'^{-1}
            R_inv = np.linalg.inv(R)
            cov_matrix = dispersion * (R_inv @ R_inv.T)

        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse for rank-deficient / ill-conditioned cases
            XtWX = self.IV.T @ (self.IV * w)
            cov_matrix = dispersion * np.linalg.pinv(XtWX)
            cov_method = "pseudo-inverse"

        # Post-hoc guard: NaN/negative variance means numerical failure, not a valid SE
        diag = np.diag(cov_matrix)
        if not np.all(np.isfinite(diag)) or np.any(diag < 0):
            XtWX = self.IV.T @ (self.IV * w)
            cov_matrix = dispersion * np.linalg.pinv(XtWX)
            diag = np.diag(cov_matrix)
            cov_method = "pseudo-inverse"

        self.CoefResults.std_error = np.sqrt(np.maximum(diag, 0.0)).reshape(-1, 1)

        # -- Record numerical diagnostics --
        if conditioning is not None:
            self.Diagnostics.condition_number = conditioning.condition_number
            self.Diagnostics.rank_deficient = conditioning.rank_deficient
            self.Diagnostics.threshold_used = conditioning.threshold_used
        self.Diagnostics.cov_method = cov_method

        # -- Statistical model-fit diagnostics: separation (binomial only) --
        if getattr(family, "name", "").lower() == "binomial":
            sep = detect_separation(mu, warn=True)
            self.Diagnostics.separation_status = sep.status
            self.Diagnostics.boundary_fitted_count = sep.boundary_count

        # -- Build diagnostic messages for the summary footer --
        self.Diagnostics.build_messages()

        # Wald test statistics
        self.CoefResults.test_stat = self.CoefResults.betas / self.CoefResults.std_error

        # P-values: z-test for known dispersion, t-test for estimated
        if self.CoefResults.test_stat_name == "t":
            dof = self.n - self.k
            self.CoefResults.test_pval = _compute_pvalue(self.CoefResults.test_stat, "t", df=dof)
            self._confidence_interval(distribution="t", dof=dof)
        else:
            self.CoefResults.test_pval = _compute_pvalue(self.CoefResults.test_stat, "z")
            self._confidence_interval(distribution="normal")


    #-----------------------------------------------------------------------------------#
    # Post-estimation methods (predict, results, summary) are inherited from BaseModel. #
    # Subclasses can override these methods if needed for specialized behavior.         #
    #-----------------------------------------------------------------------------------#
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


    def _get_from_child(self, **kwargs):


        if type(self).__name__ == self.__class__.__name__:
            # -- Simple model table: Model name + Log likelihood
            return {"": [self._get_model_display_name(),
                         f"Distribution family = {self.ModelDesignSpec.family.name}",
                         f"Link function = {self.ModelDesignSpec.family.link}",
                         f"Log likelihood = {self.FitStatistics.log_likelihood:.4f}"]}

        else:
            raise NotImplementedError(
                    f"{type(self).__name__} must override _get_ModelResults() "
                    "to provide self.ModelResults."
            )


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

        table_from_child = self._get_from_child(key="model_table")

        coefficients = self._get_coefficient_results(pretty_format=pretty_format,
                                                     table_decimals=self._table_decimals,
                                                     coef_transform=coef_transform)


        self.ModelResults = ModelResults(
            model_name=self._get_model_display_name(),
            fit_statistics=fit_statistics,
            model_table=table_from_child,
            coefficients=coefficients,
        )

        return self.ModelResults


    def results(self, report_betas_as="coef", return_type="Dataframe", pretty_format=True,
                table_decimals=None, **kwargs
                ):
        """
        Return the logistic regression results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        report_betas_as : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw log-odds.
        return_type : str, optional
            ``"Dataframe"`` (default) or ``"Dictionary"``.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Logistic Regression"``
            - ``fit_statistics``: Combined fit statistics (DataFrame or dict)
            - ``model_table``: ``None`` (MLE-based model, no SS decomposition)
            - ``coefficients``: Coefficient / odds ratio table (DataFrame or dict)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, model_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.coefficients
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Determine the coefficient transform based on report_betas_as
        if report_betas_as.lower() in ["or", "odds ratio", "odds_ratio"]:
            transform = np.exp
            self._beta_type = "odds ratio"
        else:
            transform = None
            self._beta_type = "coef"

        # Use the new GeneralizedLinearModel flow to build ModelResults
        mr = self._get_ModelResults(return_type=return_type,
                                    pretty_format=pretty_format,
                                    table_decimals=self._table_decimals,
                                    coef_transform=transform
                                    )

        # Rename "Coef." column to "Odds Ratio" if reporting odds ratios
        if self._beta_type == "odds ratio":
            if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe"]:
                coef_df = self.ModelResults.as_dataframe("coefficients", mr.coefficients)

                if "Coef." in coef_df.columns:
                    coef_df = coef_df.rename(columns={"Coef.": "Odds Ratio"})

                self.ModelResults.coefficients = coef_df

            elif isinstance(mr.coefficients, dict) and "Coef." in mr.coefficients:
                self.ModelResults.coefficients = {
                    ('Odds Ratio' if k == 'Coef.' else k): v
                    for k, v in mr.coefficients.items()
                }

        if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", ]:
            return (self.ModelResults.as_dataframe("fit_statistics", mr.fit_statistics),
                    self.ModelResults.as_dataframe("model_table", mr.model_table),
                    self.ModelResults.as_dataframe("coefficients", mr.coefficients)
                    )

        else:
            return mr.fit_statistics, mr.model_table, mr.coefficients


    #---------------------------------------------------------------------------#
    #                           Shared Summary Methods                          #
    #---------------------------------------------------------------------------#
    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the summary header for generalized models.

        Shows model name, distribution family, link function, and log-likelihood value.

        Parameters
        ----------
        width : int
            Available character width.
        model_summary_df : DataFrame or None
            Summary DataFrame from ``self.results()``.

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        import pandas as pd

        # Resolve the table to render: prefer the explicit argument,
        # then ModelResults.fit_statistics, then hardcoded fallback.
        table = None

        if model_summary_df is not None:
            if not isinstance(model_summary_df, pd.DataFrame):
                table = pd.DataFrame.from_dict(model_summary_df)
            else:
                table = model_summary_df.copy()

        elif hasattr(self, 'ModelResults') and self.ModelResults.model_table is not None:
            table = self.ModelResults.as_dataframe("model_table", self.ModelResults.model_table)

        if table is not None:
            desc_lines = table.to_string(
                    header=False,
                    index=False,
                    justify="right"
            ).split("\n")
            return desc_lines

        # Fallback: build from  ModelDesignSpec and FitStatistics
        lines = [self._get_model_display_name()]

        if self.ModelDesignSpec.family.name is not None:
            lines.append(f"Distribution family = {self.ModelDesignSpec.family.name}")

        if self.ModelDesignSpec.family.link is not None:
            lines.append(f"Link function = {self.ModelDesignSpec.family.link}")

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

        # Resolve the table to render: prefer the explicit argument,
        # then ModelResults.fit_statistics, then hardcoded fallback.
        table = None

        if descriptives_df is not None:
            if not isinstance(descriptives_df, pd.DataFrame):
                table = pd.DataFrame.from_dict(descriptives_df)
            else:
                table = descriptives_df.copy()

        elif hasattr(self, 'ModelResults') and self.ModelResults.fit_statistics is not None:
            table = self.ModelResults.as_dataframe("fit_statistics", self.ModelResults.fit_statistics)

        if table is not None:
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

        if self.FitStatistics.r_squared_pseudo is not None:
            lines.append(f"Pseudo R2     = {self.FitStatistics.r_squared_pseudo:>8.4f}")

        n_iter = self.FitStatistics.additional_stats.get("n_iterations") if self.FitStatistics.additional_stats else None
        if n_iter is not None:
            lines.append(f"N iterations  = {n_iter:>8}")

        return lines




# Convenience aliases for users who prefer different naming conventions
GLM = GeneralizedLinearModel