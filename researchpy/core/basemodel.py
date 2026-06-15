# Used
import numpy as np
import scipy.stats
from scipy.special import expit, logit
import patsy
import pandas as pd


from researchpy.utility import *
from researchpy.predict import predict

from researchpy.optimize.trackers import OptimizationTracker





# Base model class for regression models. This class is not meant to be used directly, but rather to be inherited by
# specific regression model classes (e.g., OLS, Logistic, etc.). It contains common functionality and attributes that
# are shared across different types of regression models.
class BaseModel():
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
                 solver_method="ols", obj_function="numeric",
                 solver_options=None):

        self.__name__ = "Researchpy.CoreModel"

        self._beta_type = "coef"

        if data is None:
            data = {}
        if solver_options is None:
            solver_options = {}

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        base_solver_options = {"tol": 1e-7, "max_iter": 300, "display": True}
        self.solver_options = base_solver_options | solver_options

        self.obj_function = obj_function

        self.CI_LEVEL = conf_level
        self.conf_level = conf_level

        self.nobs = self.IV.shape[0]
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
        self._solver = {"method": solver_method,
                        "obj_function": obj_function,
                        "algorithm": solver_options.get("algorithm", None),
                        "tol": solver_options.get("tol", None),
                        "max_iter": solver_options.get("max_iter", None),
                        "display": solver_options.get("display", True)}

        # Initialize an optimization tracker instance for this model. This tracker can be used by optimization
        # algorithms to store and monitor the optimization process.
        self._OptimizationTracker = OptimizationTracker()

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        # Dictionary for storing model results and information. This will be populated by specific regression model
        # classes that inherit from this base class.
        if not hasattr(self, "model_data"):
            self.model_data = {}

        ## Creating variable table information
        if not hasattr(self, "regression_table_info"):
            self.regression_table_info = {self.DV_name: [],
                                          "Coef.": [],
                                          "Std. Err.": [],
                                          f"{self._test_stat_name}": [],
                                          "p-value": [],
                                          f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}


    def _hat_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            try:
                self.model_data["H"] = np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
            except:
                self.model_data["H"] = np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
                #print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")
        else:
            try:
                H = np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
            except:
                H = np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
                #print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")
            return H


    def _j_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            self.model_data["J"] = np.ones((self.nobs, self.nobs))
        else:
            J = np.ones((self.nobs, self.nobs))
            return J

    def _identity_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            self.model_data["I"] = np.identity(self.nobs)
        else:
            I = np.identity(self.nobs)
            return I


    def __ols_fit(self, add_to_model_data=True):
        # Eigenvalues
        self.eigvals = np.linalg.eigvals(np.asarray(self.IV.T) @ np.asarray(self.IV))


        # Estimation of betas
        if add_to_model_data:
            try:
                self.model_data["betas"] = np.linalg.inv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            except:
                self.model_data["betas"] = np.linalg.pinv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
        else:
            try:
                betas = np.linalg.inv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            except:
                betas = np.linalg.pinv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            return betas



    def __compute_confidence_intervals(self, add_to_model_data=True):
        conf_int_lower = []
        conf_int_upper = []

        for beta, se in zip(self.model_data["betas"], self.model_data["standard_errors"]):

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

        if add_to_model_data:
            self.model_data["conf_int_lower"] = np.array(conf_int_lower)
            self.model_data["conf_int_upper"] = np.array(conf_int_upper)
        else:
            return np.array(conf_int_lower), np.array(conf_int_upper)


    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def __regression_base_table(self):

        dv = list(self.regression_table_info)[0]

        # Creating the first table #
        terms = (pd.DataFrame.from_dict(self._patsy_factor_information, orient="index")).reset_index()
        terms.columns = ["term", "term_cleaned"]
        terms["intx"] = [1 if ":" in t else 0 for t in list(self._patsy_factor_information.keys())]
        terms["factor"] = [1 if "C(" in t else 0 for t in list(self._patsy_factor_information.keys())]

        # Creating the second table #
        term_levels = {"term_cleaned"      : [],
                       "term_level_cleaned": []}

        for key in self._rp_factor_information.keys():

            count = 1

            if key == 'Intercept' or terms[terms.term_cleaned == key].factor.item() == 0:
                term_levels["term_cleaned"].append(key)
                term_levels["term_level_cleaned"].append(self._rp_factor_information[key])

            else:
                for value in self._rp_factor_information[key]:

                    term_levels["term_cleaned"].append(key)

                    if count == 1:

                        term_levels["term_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(value)

                        count += 1
                    else:
                        term_levels["term_level_cleaned"].append(value)

        # Creating the third table #
        current_terms = (pd.DataFrame.from_dict(self._mapping, orient="index")).reset_index()
        current_terms.columns = [dv, "term_level_cleaned"]
        current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in self._mapping.keys()]

        # Joining the tables together #
        table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                         how="left", on="term_cleaned")

        table = pd.merge(table, current_terms,
                         how="left", on=["term_cleaned", "term_level_cleaned"])

        table = pd.merge(table, pd.DataFrame.from_dict(self.regression_table_info),
                         how="left", on=dv)

        # Cleaning up final table #
        table[dv] = table["term_level_cleaned"]
        table["Coef."] = table["Coef."].astype(object)

        for idx in table.index:
            if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(self._rp_factor_information.keys())[1:]:
                table.iloc[idx, 6] = "(reference)"
                table.iloc[idx, 7:] = np.nan

            else:
                if table.iloc[idx][dv] in list(self._rp_factor_information.keys())[1:] and pd.isnull(table.iloc[idx, 6]):
                    table.iloc[idx, 6:] = np.nan

        table = table[(table.intx == 0) |
                      ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

        return table.iloc[:, 5:]


    def __table_regression_results(self, return_type="Dataframe", pretty_format=True,
                                  decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 2, "test_stat_p": 4,
                                            "CI": 2, "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4,
                                            "Sum of Squares": 4, 'Degrees of Freedom': 1,
                                            'Mean Squares': 4, 'Effect size': 4},
                                 *args):

        base_decimals = {"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                         "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                         'Degrees of Freedom': 1,  'Mean Squares': 4, 'Effect size': 4}
        decimals = base_decimals | decimals

        try:
            self.regression_table_info[self._DV_design_info.term_names[0]] = self._IV_design_info.column_names
            self.regression_table_info["Coef."] = np.round(self.model_data["betas"].flatten(), decimals["Coef."]).tolist()
            self.regression_table_info["Std. Err."] = np.round(self.model_data["standard_errors"].flatten(), decimals["Std. Err."]).tolist()
            self.regression_table_info[f"{self._test_stat_name}"] = np.round(self.model_data["test_stat"].flatten(), decimals["test_stat"]).tolist()
            self.regression_table_info["p-value"] = np.round(self.model_data["test_stat_p_values"].flatten(), decimals["test_stat_p"]).tolist()
            self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"] = [list(x) for x in np.round(np.hstack((self.model_data["conf_int_lower"].flatten().reshape(-1, 1),
                                                                                                                              self.model_data["conf_int_upper"].flatten().reshape(-1, 1))), decimals["CI"]).tolist()]

        except:
            try:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"], self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"], self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"], self.model_data["conf_int_upper"]):

                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci.item(), decimals["CI"]),
                                                                                                      round(u_ci.item(), decimals["CI"])])
            except AttributeError:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"],
                                                                  self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"],
                                                                  self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"],
                                                                  self.model_data["conf_int_upper"]):

                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci, decimals["CI"]),
                                                                                                      round(u_ci, decimals["CI"])])

    def _get_summary_parts(self):
        """
        Return the DataFrames needed by ``summary()`` to render output.

        Subclasses **must** override this to call ``self.results()`` (with
        stdout suppressed if needed) and return the relevant DataFrames.

        Returns
        -------
        tuple of (descriptives_df, body_df)
            *descriptives_df* – index-oriented DataFrame with fit-statistics
            used to build the header right side.
            *body_df* – DataFrame used to build the summary body (coefficient
            table for regression, ANOVA table for Anova, etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_summary_parts() "
            "to provide DataFrames for summary()."
        )


    def summary(self, total_width=78, return_string=False, decimals=None):
        """
        Print a formatted summary of the model results to the terminal.

        Retrieves DataFrames via ``_get_summary_parts()`` and composes the
        output from overridable building blocks:

        - **Header** (left + right side-by-side): ``_summary_header()``
        - **Body** (coefficient / ANOVA table): ``_summary_coef_table()``

        Parameters
        ----------
        total_width : int, optional
            Character width for the summary output. Default is 78.
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False (prints to terminal).
        decimals : dict, optional
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
        base_decimals = {
            "Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4,
            "CI": 2, "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4,
            "Sum of Squares": 4, "Degrees of Freedom": 1, "Mean Squares": 4,
            "Effect size": 4,
        }
        if decimals is not None:
            base_decimals = base_decimals | decimals
        decimals = base_decimals


        model_summary_df, model_description_df, coef_df = self._get_summary_parts()



        output_lines = []

        # === HEADER SECTION ===
        output_lines.append(
            self._summary_header(total_width, model_summary_df=model_summary_df, descriptives_df=model_description_df)
        )

        # === BODY SECTION ===
        output_lines.append(
            self._summary_coef_table(coef_df, total_width, decimals=decimals)
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
            'LM': 'Linear Regression (OLS)',
            'Anova': 'Analysis of Variance',
            'LogisticRegression': 'Logistic Regression',
            'Logistic': 'Logistic Regression',
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
        right_lines = self._summary_header_right(
            width, descriptives_df=descriptives_df
        )

        # Pad shorter list to match longer
        max_lines = max(len(left_lines), len(right_lines))
        while len(left_lines) < max_lines:
            left_lines.append("")
        while len(right_lines) < max_lines:
            right_lines.append("")

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
            fit statistics from the DataFrame rather than ``self.model_data``.

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        return [f"Number of obs = {self.nobs:>8}"]


    # ------------------------------------------------------------------ #
    #              Shared formatters for DataFrame.to_string()             #
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


    def _summary_coef_table(self, coef_df, width=78, decimals=None):
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
        decimals : dict, optional
            Dictionary specifying decimal places for different statistics.
            Supported keys: "Coef.", "Std. Err.", "test_stat", "test_stat_p",
            "CI".  Falls back to base defaults when not provided.

        Returns
        -------
        str
            Formatted coefficient table string.
        """
        # ---- Resolve decimal places -------------------------------------
        base_decimals = {
            "Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4,
            "CI": 2,
        }
        if decimals is not None:
            base_decimals = base_decimals | decimals
        dec = base_decimals

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
            ci_dec = dec.get("CI", 2)

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
            beta_col:        self._fmt_beta(dec.get("Coef.", 2)),
            "Std. Err.":     self._fmt_float(dec.get("Std. Err.", 4)),
            test_stat_label: self._fmt_float(dec.get("test_stat", 4)),
            "p-value":       self._fmt_float(dec.get("test_stat_p", 4)),
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

