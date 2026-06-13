from typing import Any
from pandas import DataFrame

from researchpy.models.base import CoreModel
from researchpy.core.containerclasses import FactorEffects, ModelResults, SolverOptions
from researchpy.utility import *


class LinearModel(CoreModel):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal", solver_options=None,
                 table_decimals=None):

        self.FactorEffects = FactorEffects()

        if data is None:
            data = {}

        # Build SolverOptions: start with LinearModel OLS defaults, then overlay user input.
        base_defaults = SolverOptions(
            method="ols",
            algorithm=None,
            obj_function="numeric",
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

        self.__name__ = "Researchpy.LinearModel"

        self.ModelFit.model = self.__name__
        self.ModelFit.model_display_name = self._get_model_display_name()


        # OLS fit to compute the coefficients (betas) — stored in self.CoefResults.betas
        self._CoreModel__ols_fit()

        # Compute the model sum of squares, degrees of freedom, mean squares, F-value, p-value,
        # and effect size measures — stored in self.ModelEffects
        self.__model_sum_of_square_stats()

        # Compute standard errors and confidence intervals — stored in self.CoefResults
        self.__compute_beta_se_and_stats()


    def __compute_beta_se_and_stats(self):
        variance_covariance_beta_matrix = self.__variance_covariance_beta_matrix(method="standard", to_return=True, add_to_self=False)

        ## Standard Errors
        self.CoefResults.std_error = np.sqrt(np.diag(variance_covariance_beta_matrix)).reshape(-1, 1)

        ## T-statistics
        self.CoefResults.test_stat = self.CoefResults.betas * (1 / self.CoefResults.std_error)

        ## Two-sided p-value
        self.CoefResults.test_pval = np.array([
            float((scipy.stats.t.sf(np.abs(t), self.ModelEffects.df_residual) * 2).item())
            for t in self.CoefResults.test_stat
        ])

        self._CoreModel__compute_confidence_intervals()


    def __variance_covariance_residual_matrix(self, method="standard", to_return=True, add_to_self=False):

        if method == "standard":
            H = self._hat_matrix(to_return=True)
            I = self._identity_matrix(to_return=True)
            variance_covariance_residual_matrix = np.asarray(self.ModelEffects.mse * (I - H))

        if add_to_self:
            self.variance_covariance_residual_matrix = variance_covariance_residual_matrix

        if to_return:
            return variance_covariance_residual_matrix


    def __variance_covariance_beta_matrix(self, method="standard", to_return=True, add_to_self=False):

        if method == "standard":
            try:
                variance_covariance_beta_matrix = np.asarray(self.ModelEffects.mse * np.linalg.inv(self.IV.T @ self.IV))
            except np.linalg.LinAlgError:
                variance_covariance_beta_matrix = np.asarray(self.ModelEffects.mse * np.linalg.pinv(self.IV.T @ self.IV))

        if add_to_self:
            self.variance_covariance_beta_matrix = variance_covariance_beta_matrix

        if to_return:
            return variance_covariance_beta_matrix


    def __model_sum_of_square_stats(self) -> None:

        predicted_y = self.IV @ self.CoefResults.betas     # predicted y values
        residuals = self.DV - predicted_y                   # Calculation of residuals (error)

        J = self._j_matrix(to_return=True)                  # Creating the J matrix

        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.ModelEffects.ss_total = float((self.DV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Model sum of squares (SSR)
        self.ModelEffects.ss_model = float((self.CoefResults.betas.T @ self.IV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Error sum of squares (SSE)
        self.ModelEffects.ss_residual = float((residuals.T @ residuals).item())


        ### Degrees of freedom
        # Model
        self.ModelEffects.df_model = return_numeric(np.linalg.matrix_rank(self.IV) - 1)

        # Error
        self.ModelEffects.df_residual = return_numeric(self.n - np.linalg.matrix_rank(self.IV))

        # Total
        self.ModelEffects.df_total = return_numeric(self.n - 1)

        ### Mean Square
        # Model (MSR)
        self.ModelEffects.msr = return_numeric(self.ModelEffects.ss_model * (1/self.ModelEffects.df_model))

        # Residual (error; MSE)
        self.ModelEffects.mse = return_numeric(self.ModelEffects.ss_residual * (1/self.ModelEffects.df_residual))

        #Total (MST)
        self.ModelEffects.mst = return_numeric(self.ModelEffects.ss_total * (1/self.ModelEffects.df_total))

        ## Root Mean Square Error
        self.ModelEffects.root_mse = float(np.sqrt(self.ModelEffects.mse))


        ### F-values
        # Model
        self.ModelEffects.test_stat_name = "F"
        self.ModelEffects.test_stat = return_numeric(self.ModelEffects.msr / self.ModelEffects.mse)
        self.ModelEffects.test_pval = return_numeric(
            scipy.stats.f.sf(self.ModelEffects.test_stat, self.ModelEffects.df_model, self.ModelEffects.df_residual)
        )

        ### Effect Size Measures
        # Model
        self.ModelEffects.r_squared = return_numeric(self.ModelEffects.ss_model / self.ModelEffects.ss_total)
        self.ModelEffects.r_squared_adj = return_numeric(
            1 - (self.ModelEffects.df_total / self.ModelEffects.df_residual) * (self.ModelEffects.ss_residual / self.ModelEffects.ss_total)
        )
        self.ModelEffects.eta_squared = self.ModelEffects.r_squared

        self.ModelEffects.epsilon_squared = return_numeric(
            (self.ModelEffects.df_model * (self.ModelEffects.msr - self.ModelEffects.mse)) / (self.ModelEffects.ss_total)
        )

        self.ModelEffects.omega_squared = return_numeric(
            (self.ModelEffects.df_model * (self.ModelEffects.msr - self.ModelEffects.mse)) / (self.ModelEffects.ss_total + self.ModelEffects.mse)
        )


        self.FitStatistics.test_stat_name = "F"
        self.FitStatistics.test_stat = self.ModelEffects.test_stat
        self.FitStatistics.test_pval = self.ModelEffects.test_pval
        self.FitStatistics.df_model = self.ModelEffects.df_model
        self.FitStatistics.df_residual = self.ModelEffects.df_residual
        self.FitStatistics.r_squared = self.ModelEffects.r_squared
        self.FitStatistics.r_squared_adj = self.ModelEffects.r_squared_adj
        self.FitStatistics.root_mse = self.ModelEffects.root_mse


    #--------------------------------------------------------------------------------------#
    #                  Shared Methods for dictionaries for self.ModelFit                   #
    #--------------------------------------------------------------------------------------#
    def _get_fit_statistics(self, table_decimals: object = None, *args: object) -> dict[str, list[str]]:

        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Constructing the model's metadata table (description), and anova table
        df_model = round(self.FitStatistics.df_model, self._table_decimals.get("Degrees of Freedom", 1))
        df_resid = round(self.FitStatistics.df_residual, self._table_decimals.get("Degrees of Freedom", 1))
        test_stat_name = self.FitStatistics.test_stat_name
        test_stat_model = round(self.FitStatistics.test_stat, self._table_decimals.get("test_stat", 4))
        test_pval_model = round(self.FitStatistics.test_pval, self._table_decimals.get("test_stat_p", 4))

        fit_statistics = {
            "n": [f"N = {self.n}"],
            "test_stat_model": [f"F({df_model}, {df_resid}) = {test_stat_model}"],
            "test_pval_model": [f"Prob > {test_stat_name} = {test_pval_model}"],
            "root_mse": [f"Root MSE = {round(self.ModelEffects.root_mse, self._table_decimals.get('Root MSE', 4))}"],
            "r_squared": [f"R-squared = {round(self.ModelEffects.r_squared, self._table_decimals.get('R-squared', 4))}"],
            "r_squared_adj": [f"Adj R-squared = {round(self.ModelEffects.r_squared_adj, self._table_decimals.get('Adj R-squared', 4))}"]
        }

        return fit_statistics


    def _get_sum_of_squares(self, include_test_stat_p: object = False, include_effect_sizes: object = False, factor_effects: object = False,
                            na_rep: object = '', pretty_format: object = True, table_decimals: object = None, *args: object) -> dict[Any, Any]:

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Use np.nan for missing cells when not pretty-formatting, '' otherwise
        blank = '' if pretty_format else []

        # --- Sum of Squares/ANOVA table columns that always appear ---
        base_columns_mapping = {
            "Source": "Source",
            "Sum of Squares": "SS",
            "Degrees of Freedom": "DF",
            "Mean Squares": "MS",
        }
        test_columns_mapping = {
            "test_stat": "F value",
            "test_pval": "p-value",
        }
        effect_size_column_mapping = {
            "Eta squared": "Eta^2",
            "Epsilon squared": "Epsilon^2",
            "Omega squared": "Omega^2"
        }
        table_columns_mapping = base_columns_mapping.copy() | (test_columns_mapping.copy() if include_test_stat_p else {}) | (effect_size_column_mapping.copy() if include_effect_sizes else {})

        # --- Top + Model rows ---
        top = {
            "Source": ["Model"],
            "SS": [round(self.ModelEffects.ss_model, self._table_decimals.get("Sum of Squares", 4))],
            "DF": [round(self.ModelEffects.df_model, self._table_decimals.get("Degrees of Freedom", 4))],
            "MS": [round(self.ModelEffects.msr, self._table_decimals.get("Mean Squares", 4))],
        }

        # --- IV rows ---
        if factor_effects:
            if self.FactorEffects.ss_type is None:
                raise ValueError(f"FactorEffects.ss_type not specified, must be set to an appropriate value to include factor effects in the table. See {type(FactorEffects)}.")

            middle = {}

            for k, col in base_columns_mapping.items():
                if k == "Source":
                    middle["Source"] = ([blank] if pretty_format else []) + [patsy_term_cleaner(term) for term in getattr(self.FactorEffects, k.lower())] + ([blank] if pretty_format else [])
                else:
                    dt = getattr(self.FactorEffects, col.lower())
                    middle[col] = ([blank] if pretty_format else []) + rounder(dt, self._table_decimals.get(col, 4), in_place=False) + ([blank] if pretty_format else [])
        else:
            middle = None

        # --- Bottom + Residual & Total rows ---
        n_blank_prefix = 1
        bottom_source = ["Residual", "Total"]
        bottom = {
            "Source": bottom_source,
            "SS": [round(self.ModelEffects.ss_residual,
                         self._table_decimals.get("Sum of Squares", 4)),
                   round(self.ModelEffects.ss_total,
                         self._table_decimals.get("Sum of Squares", 4))],
            "DF": [round(self.ModelEffects.df_residual,
                         self._table_decimals.get("Degrees of Freedom", 4)),
                   round(self.ModelEffects.df_total,
                         self._table_decimals.get("Degrees of Freedom", 4))],
            "MS": [round(self.ModelEffects.mse, self._table_decimals.get("Mean Squares", 4)),
                   round(self.ModelEffects.mst, self._table_decimals.get("Mean Squares", 4))],
        }

        # --- Adding F and p-values if requested ---
        if include_test_stat_p:
            for k, v in test_columns_mapping.items():
                attr_name = k.lower().replace(" ", "_")
                top[v] = [round(getattr(self.ModelEffects, attr_name), self._table_decimals.get(k, 4))]
                bottom[v] = [na_rep] * (n_blank_prefix * 2)

                if factor_effects:
                    dt = getattr(self.FactorEffects, attr_name)
                    middle[v] = ([blank] if pretty_format else []) + rounder(dt, self._table_decimals.get(k, 4), in_place=False) + ([blank] if pretty_format else [])


        # --- Adding effect size columns if requested ---
        if include_effect_sizes:
            for k, v in effect_size_column_mapping.items():
                attr_name = k.lower().replace(" ", "_")
                top[v] = [round(getattr(self.ModelEffects, attr_name), self._table_decimals.get("Effect size", 4))]
                bottom[v] = [na_rep] * (n_blank_prefix * 2)

                if factor_effects:
                    dt = getattr(self.FactorEffects, attr_name)
                    middle[v] = ([blank] if pretty_format else []) + rounder(dt, self._table_decimals.get("Effect size", 4), in_place=False) + ([blank] if pretty_format else [])


        # --- Combine parts of the table ---
        model_results = {}
        for col in list(table_columns_mapping.values()):
            model_results[col] = top[col] + (middle[col] if factor_effects else []) + bottom[col]


        return model_results

    def _get_coefficient_results(self, na_rep='', pretty_format=True, table_decimals=None,
                                    coef_transform=None) -> dict:

        return super()._get_coefficient_results(pretty_format=pretty_format,
                                                table_decimals=table_decimals,
                                                coef_transform=coef_transform)


    #-----------------------------------------------------------------------------------------------------------------#
    #       Setting the fit statistics, model table (sum of squares), and coefficient results in self.ModelResults    #
    #-----------------------------------------------------------------------------------------------------------------#
    def _get_ModelResults(self, include_test_stat_p: object = False, include_effect_sizes: object = False,
                          factor_effects: object = False, return_type: object = "Dataframe",
                          na_rep: object = '', pretty_format: object = True, table_decimals: object = None,
                          *args: object) -> ModelResults:


        ## Checking for valid return type ##
        if return_type.lower() not in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", "dictionary", "dict"]:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

        ## Resolving decimal places ##
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Build the fit statistics, model table, and coefficient results to be stored in self.ModelResults
        fit_statistics = self._get_fit_statistics(table_decimals=self._table_decimals, *args)

        model_table = self._get_sum_of_squares(pretty_format=pretty_format,
                                               na_rep=na_rep,
                                               include_test_stat_p=include_test_stat_p,
                                               factor_effects=factor_effects,
                                               include_effect_sizes=include_effect_sizes,
                                               table_decimals=self._table_decimals,
                                               *args)

        coefficients = self._get_coefficient_results(pretty_format=pretty_format, table_decimals=self._table_decimals)


        self.ModelResults = ModelResults(
                model_name= self.ModelFit.model_display_name,
                fit_statistics= fit_statistics,
                model_table= model_table,
                coefficients= coefficients,
        )

        return self.ModelResults


    def _get_results(self, include_test_stat_p: object = False, include_effect_sizes: object = False,
                     factor_effects: object = False, return_type: object = "Dataframe",
                     na_rep: object = '', pretty_format: object = True,
                     table_decimals: object = None, *args: object) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None] | tuple[Any, Any, Any]:
        """
        Return the regression results.

        Parameters
        ----------
        include_test_stat_p : bool, optional
            Whether to include test statistics and p-values in the output. Default is False.
        include_effect_sizes : bool, optional
            Whether to include effect size measures in the output. Default is False.
        factor_effects : bool, optional
            Whether to include factor effects in the output. Default is False.
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary". Default is "Dataframe".
        na_rep : object, optional
            Representation for missing values. Default is ''.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics. Default is None.

        Returns
        -------
        tuple
            If return_type is "Dataframe": (descriptives_df, model_results_df, coefficients_df)
            If return_type is "Dictionary": (descriptives_dict, model_results_dict, coefficients_dict)
        """


        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        mr = self._get_ModelResults(include_test_stat_p=include_test_stat_p,
                                    include_effect_sizes=include_effect_sizes,
                                    factor_effects=factor_effects,
                                    na_rep=na_rep,
                                    return_type=return_type,
                                    pretty_format=pretty_format,
                                    table_decimals=self._table_decimals)

        if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", ]:
            return (self.ModelResults.as_dataframe("fit_statistics", mr.fit_statistics),
                    self.ModelResults.as_dataframe("model_table", mr.model_table),
                    self.ModelResults.as_dataframe("coefficients", mr.coefficients))

        else:
            return mr.fit_statistics, mr.model_table, mr.coefficients


    #---------------------------------------------------------------------------#
    #                           Shared Summary Methods                          #
    #---------------------------------------------------------------------------#
    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the OLS summary header with an ANOVA source table.

        Shows model name followed by a Source/SS/df/MS table with Model, Residual,
        and Total rows.

        Parameters
        ----------
        width : int
            Total character width of the output.
        model_summary_df : DataFrame or None
            Model summary DataFrame from ``self.results()``.  When None the
            ANOVA mini-table is skipped (used by Anova subclass).

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        if model_summary_df is None:
            if self.ModelResults.model_table is None:
                return [self._get_model_display_name()]


        lines = []
        model_display = self._get_model_display_name()

        # Define column widths for ANOVA table
        col_w = {'source': 10, 'ss': 12, 'df': 4, 'ms': 12}
        table_w = col_w['source'] + col_w['ss'] + col_w['df'] + col_w['ms'] + 6

        def fmt_num(val, w, d=4):
            try:
                return f"{float(val):>{w}.{d}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        def fmt_int(val, w):
            try:
                return f"{int(val):>{w}}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        sep = "-" * col_w['source'] + "+" + "-" * (table_w - col_w['source'] - 1)

        lines.append(f"{model_display:<{table_w}}")
        lines.append(
            f"{'Source':>{col_w['source']}} | "
            f"{'SS':>{col_w['ss']}} "
            f"{'df':>{col_w['df']}} "
            f"{'MS':>{col_w['ms']}}"
        )
        lines.append(sep)

        me = self.ModelEffects
        lines.append(
            f"{'Model':>{col_w['source']}} | "
            f"{fmt_num(me.ss_model or 0, col_w['ss'])} "
            f"{fmt_int(me.df_model or 0, col_w['df'])} "
            f"{fmt_num(me.msr or 0, col_w['ms'])}"
        )
        lines.append(
            f"{'Residual':>{col_w['source']}} | "
            f"{fmt_num(me.ss_residual or 0, col_w['ss'])} "
            f"{fmt_int(me.df_residual or 0, col_w['df'])} "
            f"{fmt_num(me.mse or 0, col_w['ms'])}"
        )
        lines.append(sep)
        lines.append(
            f"{'Total':>{col_w['source']}} | "
            f"{fmt_num(me.ss_total or 0, col_w['ss'])} "
            f"{fmt_int(me.df_total or 0, col_w['df'])} "
            f"{fmt_num(me.mst or 0, col_w['ms'])}"
        )

        return lines



    # Pretty sure can delete __summary_header_left_anova_table
    def __summary_header_left_anova_table(self, width=78, model_summary_df=None, *args):

        # ---- Resolve model_summary table ----------------------------
        if model_summary_df is not None:
            if not isinstance(model_summary_df, pd.DataFrame):
                table = pd.DataFrame.from_dict(model_summary_df)

        else:
            if self.ModelResults.model_table is not None:
                if isinstance(self.ModelResults.model_table, pd.DataFrame):
                    table = self.ModelResults.model_table.copy()
                else:
                    table = self.ModelResults.as_dataframe("model_table", self.ModelResults.model_table)
            else:
                return [self._get_model_display_name()]


        # ---- Build the output string ------------------------------------
        sep = "-" * width

        table_str = table.to_string(
            index=False,
            na_rep="",
            justify="right",
        )

        lines = [sep,
                 table_str,
                 sep,
                 "Note: Effect size values for factors are partial.",
                 ]

        return "\n".join(lines)



    def _splice_f_stat_lines(self):
        """
        Return the F-statistic and Prob > F lines for the header right side.

        These are not stored in the descriptives DataFrame, so they are
        built from ``self.ModelEffects`` and spliced into the header.
        Shared by OLS and Anova header_right methods.

        Returns
        -------
        list of str
        """
        me = self.ModelEffects
        df_model = me.df_model or 0
        df_resid = me.df_residual or 0
        return [
            f"F({df_model}, {df_resid}) = "
            f"{(me.test_stat or 0):>8.2f}",
            f"Prob > F      = "
            f"{(me.test_pval or 0):>8.4f}",
        ]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the OLS summary header with fit statistics.

        When *descriptives_df* is provided the values are read from that
        DataFrame via ``to_string(header=False)`` so the summary is driven
        entirely by the DataFrames returned from ``self.results()``.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()`` (transposed,
            single-row with stat names as columns).

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        if descriptives_df is None:
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

        # Fallback: build from self.ModelEffects directly
        me = self.ModelEffects
        return ([f"Number of obs = {self.n:>8}",] + self._splice_f_stat_lines() +
                [f"R-squared     = {(me.r_squared or 0):>8.4f}",
                 f"Adj R-squared = {(me.r_squared_adj or 0):>8.4f}",
                 f"Root MSE      = {(me.root_mse or 0):>8.4f}",
                 ])

