
import scipy.stats
import patsy

from researchpy.utility import *
from researchpy.predict import predict
from researchpy.containers import ModelFit, FitStatistics, ModelEffects, CoefResults, ModelTerms, SolverOptions

from researchpy.optimize.trackers import OptimizationTracker





# Base model class for regression models. This class is not meant to be used directly, but rather to be inherited by
# specific regression model classes (e.g., OLS, Logistic, etc.). It contains common functionality and attributes that
# are shared across different types of regression models.
class CoreModel():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

    @property
    def CI_LEVEL(self):
        return self._CI_LEVEL
    @CI_LEVEL.setter
    def CI_LEVEL(self, conf_level):
        self._CI_LEVEL = float(conf_level)

    @property
    def conf_level(self):
        return self._CI_LEVEL
    @conf_level.setter
    def conf_level(self, conf_level):
        self._CI_LEVEL = float(conf_level)

    @property
    def obj_function(self):
        return self._obj_function
    @obj_function.setter
    def obj_function(self, obj_function):
        self._obj_function = obj_function

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_options=None, table_decimals=None):

        self.__name__ = "Researchpy.CoreModel"

        self._beta_type = "coef"

        if data is None:
            data = {}

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        # Build a SolverOptions dataclass instance.
        # Subclasses (LinearModel, GeneralModel) should resolve their own defaults
        # and pass a fully-formed SolverOptions instance. If None or dict arrives
        # here, we fall back to the SolverOptions dataclass defaults.
        if isinstance(solver_options, SolverOptions):
            self.solver_options = solver_options
        elif isinstance(solver_options, dict):
            self.solver_options = SolverOptions.from_dict(solver_options)
        else:
            self.solver_options = SolverOptions()

        self.obj_function = self.solver_options.obj_function

        self.CI_LEVEL = conf_level
        self.conf_level = conf_level

        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        if not hasattr(self, "_test_stat_name"):
            self._test_stat_name = "t" if family == "gaussian" else "z"
        self._family = family
        self._link = link
        self._CI_LEVEL = conf_level

        # Initialize an optimization tracker instance for this model. This tracker can be used by optimization
        # algorithms to store and monitor the optimization process.
        self._OptimizationTracker = OptimizationTracker()

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        # New dataclass-based term/column mapping
        self._model_terms = ModelTerms.from_design_info(self._IV_design_info)

        # Will be refractoring to use containers to clean up codebase and make it more modular. This ModelFit
        # dataclass will store the model design information and fit parameters that are common across different
        # regression models. By centralizing this information in a dataclass, it allows for cleaner code and easier
        # maintenance, as well as providing a standardized way to access model fit information across different model types.
        self.ModelFit = ModelFit(
            formula = formula_like,
            family = family,
            link = link,
            solver_method = self.solver_options.method,
            ci_level = conf_level,
            dv_term_names = self.DV.design_info.term_names,
            iv_term_names = list(self._model_terms.column_map.keys())
        )

        self.FitStatistics = FitStatistics(
            n = self.n,
            k = self.k,
            test_stat_name = "F"
        )
        self.ModelEffects = ModelEffects()

        self.CoefResults = CoefResults()
        self.CoefResults.term = self._model_terms.column_map.keys()
        #self.CoefResults.test_stat_name = "t" if self.ModelFit.family == "gaussian" else "z"

        ## Creating variable table information
        if not hasattr(self, "regression_table_info"):
            self.regression_table_info = {
                self.DV_name: [],
                "Coef.": [],
                "Std. Err.": [],
                f"{self._test_stat_name}": [],
                "p-value": [],
                f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []
            }


        # Checking to see if the `self._table_decimals` attribute is defined. If it's not then create it.
        # This is used to specify the number of decimal places to round to for different statistics in the summary
        # table. By defining it in the base class, it allows subclasses to override or update the decimal settings as
        # needed without having to redefine the entire dictionary.
        if not hasattr(self, "_table_decimals"):
            self._table_decimals = {
                "Coef.": 2, "Std. Err.": 3, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                        'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4
            }

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


    #---------------------------------------------------------------------------#
    #                       Shared Computational Methods                        #
    # --------------------------------------------------------------------------#
    def _hat_matrix(self, y=None, x=None, to_return=False, add_to_self=False):

        if y is not None and x is not None:
            try:
                H = np.asarray(x) @ np.linalg.inv(np.asarray(x.T) @ np.asarray(x)) @ np.asarray(x.T)
            except:
                H = np.asarray(x) @ np.linalg.pinv(np.asarray(x.T) @ np.asarray(x)) @ np.asarray(x.T)

        else:
            try:
                H = np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
            except:
                H = np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)


        if add_to_self:
            self.H = H

        if to_return:
            return H


    def _j_matrix(self, n=None, to_return=False, add_to_self=False):

        if n is None:
            n = self.n

        J = np.ones((n, n))

        if add_to_self:
            self.J = J

        if to_return:
            return J


    def _identity_matrix(self, n=None, to_return=False, add_to_self=False):

        if n is None:
            n = self.n

        I = np.identity(n)

        if add_to_self:
            self.I = I

        if to_return:
            return I



    def _eigenval_matrix(self, x=None, to_return=False, add_to_self=False):

        if x is None:
            x = self.IV

        # Eigenvalues
        eigvals = np.linalg.eigvals(np.asarray(x.T) @ np.asarray(x))

        if add_to_self:
            self.eigvals = np.asarray(eigvals)

        if to_return:
            return np.asarray(eigvals)


    def __ols_fit(self, y=None, x=None, to_return=False, add_to_self=False):

        if y is None: y = self.DV
        if x is None: x = self.IV

        # Eigenvalues
        eigvals = self._eigenval_matrix(to_return=True)

        # Estimation of betas
        try:
            betas = np.linalg.inv((np.asarray(x.T) @ np.asarray(x))) @ np.asarray(x.T) @ np.asarray(y)
        except:
            betas = np.linalg.pinv((np.asarray(x.T) @ np.asarray(x))) @ np.asarray(x.T) @ np.asarray(y)

        # Store in CoefResults dataclass
        self.CoefResults.betas = betas

        if add_to_self:
            self.betas = betas


        if to_return:
            return betas


    def __compute_confidence_intervals(self):
        conf_int_lower = []
        conf_int_upper = []

        for beta, se in zip(self.CoefResults.betas, self.CoefResults.std_error):

            try:
                lower, upper = scipy.stats.norm.interval(self._CI_LEVEL, loc=beta, scale=se)
                conf_int_lower.append(float(lower))
                conf_int_upper.append(float(upper))

            except TypeError:
                try:
                    conf_int_lower.append(lower.item())
                    conf_int_upper.append(upper.item())

                except:
                    conf_int_lower.append(np.nan)
                    conf_int_upper.append(np.nan)

        self.CoefResults.conf_int_lower = np.array(conf_int_lower)
        self.CoefResults.conf_int_upper = np.array(conf_int_upper)


    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    #---------------------------------------------------------------------------#
    #                   Shared Returning Results Methods                        #
    # --------------------------------------------------------------------------#
    def _get_coefficient_results(self, na_rep: object = '', pretty_format=True, table_decimals=None,
                                    coef_transform=None):
        """Build a prettified coefficient table as a dictionary.

        Uses ``self._model_terms`` for structural info (term names, factor
        flags, reference categories, cleaned column names) and
        ``self.CoefResults`` for the raw statistics.  ``CoefResults.term``
        remains the original Patsy column names; cleaned names are sourced
        from ``ModelTerms.column_map``.

        Parameters
        ----------
        na_rep : object
            Representation for missing values.
        pretty_format : bool
            Whether to format the output for display. Default is True.
        table_decimals : dict or None
            Override decimal settings.  Merged with ``self._table_decimals``.
        coef_transform : callable or None
            Optional transformation function applied to coefficients and
            confidence interval bounds **before** rounding for display.
            For example, ``np.exp`` to convert log-odds to odds ratios.
            Standard errors, test statistics, and p-values are NOT transformed.
            When ``None`` (default), no transformation is applied.

        Returns
        -------
        dict
            Keys are column header strings, values are equal-length lists.
            Intended to be converted to a DataFrame by the caller.
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # ---- Resolve sources -------------------------------------------------
        dv = self.ModelFit.dv_term_names[0]
        ci_level = int(self.ModelFit.ci_level * 100)
        test_stat_name = "t" if self.ModelFit.family == "gaussian" else "z"

        # Decimal settings
        d_coef = self._table_decimals.get("Coef.", 2)
        d_se = self._table_decimals.get("Std. Err.", 3)
        d_ts = self._table_decimals.get("test_stat", 4)
        d_p = self._table_decimals.get("test_stat_p", 4)
        d_ci = self._table_decimals.get("CI", 2)

        # Build positional index: original Patsy column name → array index
        coef_terms = list(self.CoefResults.term)
        col_to_idx = {col: i for i, col in enumerate(coef_terms)}

        # Column map for display names
        column_map = self._model_terms.column_map  # orig col → cleaned col

        # ---- Helper to extract a stats row for a given original column name --
        def _stats_row(orig_col):
            idx = col_to_idx[orig_col]

            # Extract raw values
            beta_raw = self.CoefResults.betas[idx]
            se_raw = self.CoefResults.std_error[idx]
            ts_raw = self.CoefResults.test_stat[idx]
            pv_raw = self.CoefResults.test_pval[idx]
            ci_lo_raw = self.CoefResults.conf_int_lower[idx]
            ci_hi_raw = self.CoefResults.conf_int_upper[idx]

            # Apply transformation to betas and CIs BEFORE rounding
            # (SE, test stat, p-value are invariant under monotonic transforms)
            if coef_transform is not None:
                beta = np.round(coef_transform(beta_raw), d_coef)
                ci_lo = np.round(coef_transform(ci_lo_raw), d_ci)
                ci_hi = np.round(coef_transform(ci_hi_raw), d_ci)
            else:
                beta = np.round(beta_raw, d_coef)
                ci_lo = np.round(ci_lo_raw, d_ci)
                ci_hi = np.round(ci_hi_raw, d_ci)

            se = np.round(se_raw, d_se)
            ts = np.round(ts_raw, d_ts)
            pv = np.round(pv_raw, d_p)

            return as_numeric(beta), as_numeric(se), as_numeric(ts), as_numeric(pv), [as_numeric(ci_lo), as_numeric(ci_hi)]


        # ---- Build output rows -----------------------------------------------
        col_dv = []
        col_coef = []
        col_se = []
        col_ts = []
        col_pv = []
        col_ci = []

        for term in self._model_terms:

            is_factor = (
                term.is_factor if not term.is_interaction
                else (any(term.is_factor) if isinstance(term.is_factor, list) else term.is_factor)
            )

            if not pretty_format:
                # Raw output — one row per estimated column, using original Patsy names
                for orig_col in term.columns:
                    col_dv.append(orig_col)
                    beta, se, ts, pv, ci = _stats_row(orig_col)
                    col_coef.append(beta)
                    col_se.append(se)
                    col_ts.append(ts)
                    col_pv.append(pv)
                    col_ci.append(ci)

            elif not is_factor:
                # Continuous variable or Intercept — one row per column
                for orig_col in term.columns:
                    col_dv.append(column_map[orig_col])
                    beta, se, ts, pv, ci = _stats_row(orig_col)
                    col_coef.append(beta)
                    col_se.append(se)
                    col_ts.append(ts)
                    col_pv.append(pv)
                    col_ci.append(ci)

            elif not term.is_interaction:
                # Simple factor — header row + reference row + level rows
                # Header row (term name, blank stats)
                col_dv.append(term.name)
                col_coef.append("")
                col_se.append("")
                col_ts.append("")
                col_pv.append("")
                col_ci.append("")

                # Reference row
                ref_label = str(term.reference) if term.reference is not None else ""
                col_dv.append(ref_label)
                col_coef.append("(reference)")
                col_se.append("")
                col_ts.append("")
                col_pv.append("")
                col_ci.append("")

                # Estimated level rows
                for orig_col in term.columns:
                    col_dv.append(column_map[orig_col])
                    beta, se, ts, pv, ci = _stats_row(orig_col)
                    col_coef.append(beta)
                    col_se.append(se)
                    col_ts.append(ts)
                    col_pv.append(pv)
                    col_ci.append(ci)

            else:
                # Interaction term — header row + estimated level rows only
                col_dv.append(term.name)
                col_coef.append("")
                col_se.append("")
                col_ts.append("")
                col_pv.append("")
                col_ci.append("")

                for orig_col in term.columns:
                    col_dv.append(column_map[orig_col])
                    beta, se, ts, pv, ci = _stats_row(orig_col)
                    col_coef.append(beta)
                    col_se.append(se)
                    col_ts.append(ts)
                    col_pv.append(pv)
                    col_ci.append(ci)



        return {
            dv: col_dv,
            "Coef.": col_coef,
            "Std. Err.": col_se,
            f"{test_stat_name}": col_ts,
            "p-value": col_pv,
            f"{ci_level}% Conf. Interval": col_ci,
        }


    def _get_ModelResults(self, return_type="Dataframe", pretty_format=True, table_decimals=None, **kwargs):

        raise NotImplementedError(
            f"{type(self).__name__} must override _get_ModelResults() "
            "to provide self.ModelResults."
        )

    #---------------------------------------------------------------------------#
    #                           Shared Summary Methods                          #
    #---------------------------------------------------------------------------#
    def _get_summary_parts(self):
        """
        Return the DataFrames needed by ``summary()`` to render output.

        Default implementation reads from ``self.ModelResults``.
        Subclasses that override ``results()`` and set ``self.ModelResults``
        during ``__init__`` do not need to override this method.

        Returns
        -------
        tuple of (model_table_df, fit_statistics_df, body_df)
            *model_table_df* – DataFrame for the header-left side (e.g.
            ANOVA decomposition mini-table for OLS; ``None`` for MLE models).
            *fit_statistics_df* – DataFrame with fit-statistics used to
            build the header right side.
            *body_df* – DataFrame used to build the summary body (coefficient
            table for regression, ANOVA table for Anova, etc.).
        """
        mr = self.ModelResults
        body_df = mr.coefficients if mr.coefficients is not None else mr.model_table
        return mr.model_table, mr.fit_statistics, body_df


    def summary(self, total_width=78, return_string=False, table_decimals=None):
        """
        Print a formatted summary of the model results to the terminal.

        Reads directly from ``self.ModelResults`` and composes the output
        from overridable building blocks:

        - **Header** (left + right side-by-side): ``_summary_header()``
        - **Body** (coefficient / ANOVA table): ``_summary_coef_table()``

        Parameters
        ----------
        total_width : int, optional
            Character width for the summary output. Default is 78.
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False (prints to terminal).
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.
            Supported keys: "Coef.", "Std. Err.", "test_stat", "test_stat_p",
            "CI", "Root MSE", "R-squared", "Adj R-squared", "Sum of Squares",
            "Degrees of Freedom", "Mean Squares", "Effect size".
            User-provided values override the base defaults.

        Returns
        -------
        str or None
            If *return_string* is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.
        """
        # Base decimal defaults — user values override these
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        fit_statistics_df = self.ModelResults.as_dataframe("fit_statistics", self.ModelResults.fit_statistics)
        model_table_df = self.ModelResults.as_dataframe("model_table", self.ModelResults.model_table)
        coefficients_df = self.ModelResults.as_dataframe("coefficients", self.ModelResults.coefficients)

        output_lines = []

        # === HEADER SECTION ===
        output_lines.append(
            self._summary_header(total_width, model_summary_df=model_table_df, descriptives_df=fit_statistics_df)
        )

        # === BODY SECTION ===
        output_lines.append(
            self._summary_coef_table(coefficients_df, total_width, table_decimals=self._table_decimals)
        )


        summary_str = "\n".join(output_lines)

        if return_string:
            return summary_str
        else:
            print(summary_str)
            return None


    def _get_model_display_name(self):
        """
        Return a human-readable display name for the model.

        Maps internal __name__ attributes to user-friendly display names.
        Subclasses can override to provide custom display names.

        Returns
        -------
        str
            Human-readable model name.
        """
        model_name = getattr(self, '__name__', 'Model').replace('Researchpy.', '').replace('researchpy.', '')
        model_display_names = {
            'OLS': 'Linear Regression (OLS)',
            'LinearRegression': 'Linear Regression (OLS)',
            'LinearModel': 'Linear Regression (OLS)',
            'lm': 'Linear Regression (OLS)',
            'LM': 'Linear Regression (OLS)',
            'Regress': 'Linear Regression (OLS)',
            'Anova': 'Analysis of Variance',
            'ANOVA': 'Analysis of Variance',
            'anova': 'Analysis of Variance',
            'LogisticRegression': 'Logistic Regression',
            'Logistic': 'Logistic Regression',
            'Logit': 'Logistic Regression',
            'GeneralModel': 'Generalized Linear Model',
            'CoreModel': 'Model'
        }
        return model_display_names.get(model_name, model_name)


    def _summary_header(self, width=78, model_summary_df=None, descriptives_df=None):
        """
        Build the header section of the summary output by composing
        left and right sub-sections side-by-side.

        Subclasses should override ``_summary_header_left()`` and/or
        ``_summary_header_right()`` to customize header content for their
        model type, rather than overriding this method directly.

        Parameters
        ----------
        width : int, optional
            Total character width of the output. Default is 78.
        descriptives_df : DataFrame or None, optional
            Descriptives DataFrame from ``self.results()``.  When provided,
            it is forwarded to ``_summary_header_right()`` so that subclasses
            can render fit statistics directly from the DataFrame.

        Returns
        -------
        str
            Formatted header string.
        """
        left_lines = self._summary_header_left(width, model_summary_df=model_summary_df)
        right_lines = self._summary_header_right(width, descriptives_df=descriptives_df)

        # Pad shorter list to match longer
        max_lines = max(len(left_lines), len(right_lines))
        while len(left_lines) < max_lines: left_lines.append("")
        while len(right_lines) < max_lines: right_lines.append("")

        # Calculate left column width and gap
        left_width = width * 55 // 100
        gap = "    "
        combined = []
        for left_text, right_text in zip(left_lines, right_lines):
            combined.append(f"{left_text:<{left_width}}{gap}{right_text}")

        return "\n".join(combined)


    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the summary header.

        Base implementation returns the model display name.
        Subclasses should override to add model-specific content
        (e.g., ANOVA source table for linear models, log-likelihood for MLE models).

        Parameters
        ----------
        width : int, optional
            Total character width of the output. Default is 78.
        model_summary_df : DataFrame or None, optional
            Model summary DataFrame from ``self.results()``.  When None
            (e.g. for ANOVA) only the display name is shown.

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """

        # ---- Resolve Conent ----------------------------------
        if model_summary_df is None:
            return [self._get_model_display_name()]

        return [self._get_model_display_name()]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the summary header.

        Base implementation returns the number of observations.
        Subclasses should override to add model-specific fit statistics
        (e.g., R-squared for linear models, LR chi-squared for MLE models).

        Parameters
        ----------
        width : int, optional
            Total character width of the output. Default is 78.
        descriptives_df : DataFrame or None, optional
            Descriptives DataFrame from ``self.results()``.  The base
            implementation ignores this; subclasses can use it to render
            fit statistics from the DataFrame.

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        return [f"Number of obs = {self.n:>8}"]


    def _summary_coef_table(self, coef_df, width=78, table_decimals=None):
        """
        Build the coefficient table section of the summary output using
        ``DataFrame.to_string()``.

        The DataFrame is expected to have been produced by
        ``self.results(return_type="Dataframe")`` and to contain columns for
        the DV name (variable labels), coefficient estimates, standard errors,
        test statistics, p-values, and confidence intervals.

        Parameters
        ----------
        coef_df : DataFrame
            Coefficient table DataFrame from ``self.results()``.
        width : int
            Total character width of the output.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.
            Supported keys: "Coef.", "Std. Err.", "test_stat", "test_stat_p",
            "CI".  Falls back to base defaults when not provided.

        Returns
        -------
        str
            Formatted coefficient table string.
        """
        # ---- Resolve decimal places -------------------------------------
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        table = coef_df.copy()

        # ---- Identify column names dynamically -------------------------
        dv_name = table.columns[0]  # First column is the variable name

        beta_col = "Coef."
        if "Odds Ratio" in table.columns:
            beta_col = "Odds Ratio"

        test_stat_label = self._test_stat_name
        ci_col_name = f"{int(self.CI_LEVEL * 100)}% Conf. Interval"

        # ---- Format CI list column into "[lower, upper]" strings --------
        if ci_col_name in table.columns:
            ci_dec = self._table_decimals.get("CI", 2)

            def _fmt_ci_combined(val):
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    lower = val[0]
                    upper = val[1]
                    try:
                        return f"[{float(lower):.{ci_dec}f}, {float(upper):.{ci_dec}f}]"
                    except (ValueError, TypeError):
                        return ""
                try:
                    if pd.isna(val):
                        return ""
                except (TypeError, ValueError):
                    pass
                return str(val)

            table[ci_col_name] = table[ci_col_name].apply(_fmt_ci_combined)

        # ---- Select and order display columns ---------------------------
        display_cols = [dv_name]
        for col in [beta_col, "Std. Err.", test_stat_label, "p-value",
                     ci_col_name]:
            if col in table.columns:
                display_cols.append(col)
        table = table[display_cols]

        # ---- Coerce numeric columns to float ----------------------------
        # Skip the beta column — it may contain "(reference)" strings
        # Skip the CI column — it is now a pre-formatted string
        for col in display_cols[1:]:
            if col in (beta_col, ci_col_name):
                continue
            table[col] = pd.to_numeric(table[col], errors="coerce")

        # ---- Per-column formatters (using shared static methods) -----------
        formatters = {
            dv_name:         self._fmt_str,
            beta_col:        self._fmt_beta(self._table_decimals.get("Coef.", 2)),
            "Std. Err.":     self._fmt_float(self._table_decimals.get("Std. Err.", 4)),
            test_stat_label: self._fmt_float(self._table_decimals.get("test_stat", 4)),
            "p-value":       self._fmt_float(self._table_decimals.get("test_stat_p", 4)),
            ci_col_name:     self._fmt_str,
        }
        # Only include formatters for columns actually in the table
        formatters = {k: v for k, v in formatters.items() if k in table.columns}

        # ---- Build the output string ------------------------------------
        sep = "-" * width

        table_str = table.to_string(
            index=False,
            na_rep="",
            formatters=formatters,
            justify="right",
        )

        lines = [sep, table_str, sep]

        return "\n".join(lines)


    # ------------------------------------------------------------------ #
    #              Shared formatters for DataFrame.to_string()           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt_float(decimals):
        """Return a ``to_string`` formatter: renders floats or blank for NaN."""
        def _f(val):
            try:
                if pd.isna(val):
                    return ""
            except (TypeError, ValueError):
                pass
            try:
                return f"{float(val):.{decimals}f}"
            except (ValueError, TypeError):
                return str(val)
        return _f

    @staticmethod
    def _fmt_int(val):
        """``to_string`` formatter: renders integers or blank for NaN."""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        try:
            return f"{int(val)}"
        except (ValueError, TypeError):
            return str(val)

    @staticmethod
    def _fmt_str(val):
        """``to_string`` formatter: keeps strings, blanks NaN."""
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        return str(val)

    @staticmethod
    def _fmt_beta(decimals):
        """``to_string`` formatter: renders '(reference)' strings or float."""
        def _f(val):
            if isinstance(val, str):
                return val  # e.g. "(reference)"
            try:
                if pd.isna(val):
                    return ""
            except (TypeError, ValueError):
                pass
            try:
                return f"{float(val):.{decimals}f}"
            except (ValueError, TypeError):
                return str(val)
        return _f

