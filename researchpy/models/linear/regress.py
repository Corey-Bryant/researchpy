# -*- coding: utf-8 -*-
"""
Ordinary Least Squares (OLS) Regression

This module provides the OLS class for fitting linear regression models
using the ordinary least squares method.
"""

from researchpy.models.linear.linear_model import LinearModel


class Regress(LinearModel):
    """
    Ordinary Least Squares (OLS) regression for continuous outcomes.

    This class provides linear regression analysis using the ordinary least squares
    method, including:
    - Coefficient estimation
    - Standard errors and t-statistics
    - R-squared and adjusted R-squared
    - F-test for overall model significance
    - Effect size measures (Eta squared, Epsilon squared, Omega squared)
    - Confidence intervals
    - Variance-covariance matrices

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "y ~ x1 + x2").
        See https://patsy.readthedocs.io/en/latest/ for formula syntax.
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    conf_level : float, optional
        Confidence level for confidence intervals. Default is 0.95.

    Attributes
    ----------
    ModelEffects : ModelEffects
        Dataclass containing model-level results including:
        - sum_of_square_total, sum_of_square_model, sum_of_square_residual
        - degrees_of_freedom_model, degrees_of_freedom_residual, degrees_of_freedom_total
        - msr, mse, mst: Mean squares
        - root_mse: Root mean square error
        - model_test_stat, model_test_pval: F-statistic and p-value
        - r_squared, r_squared_adj: R-squared values
        - eta_squared, epsilon_squared, omega_squared: Effect sizes
    CoefResults : CoefResults
        Dataclass containing per-coefficient results including:
        - betas: Estimated coefficients
        - std_error: Standard errors of coefficients
        - test_stat: t-statistics
        - p_value: p-values for t-tests
        - conf_int_lower, conf_int_upper: Confidence interval bounds

    Examples
    --------
    >>> import researchpy as rp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]})
    >>> model = rp.Regress("y ~ x", data=df)
    >>> model.results()

    See Also
    --------
    LinearRegression : Alias for Regress
    LM : Alias for Regress (R-style shorthand)
    Anova : Analysis of Variance (inherits from OLS)
    """

    def __init__(self, formula, data=None, conf_level=0.95,
                 table_decimals=None,
                 display_summary=True, return_type="Dictionary",
                 na_rep='', pretty_format=True,):

        self._test_stat_name = "t"
        self.__name__ = "Researchpy.Regress"

        super().__init__(formula=formula, data=data, conf_level=conf_level, table_decimals=table_decimals)

        self.ModelDesignSpec.model = self.__name__
        self.ModelDesignSpec.model_display_name = self._get_model_display_name()

        # Build ModelResults (results() sets self.ModelResults internally)
        self.results(return_type=return_type, na_rep=na_rep, pretty_format=pretty_format, table_decimals=table_decimals)

        # Display the model results summary
        if display_summary:
            self.summary()


    def results(self, include_test_stat_p=False, include_effect_sizes=True,
                return_type="Dataframe", na_rep='', pretty_format=True, table_decimals=None, *args):
        """
        Return the regression results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        include_test_stat_p : bool, optional
            Whether to include p-values for test statistics in the results. Default is False.
        include_effect_sizes : bool, optional
            Whether to include effect sizes in the results. Default is True.
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        na_rep : object
            Representation for missing values in the results. Default is an empty string.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Linear Regression (OLS)"``
            - ``fit_statistics``: Descriptive fit statistics (DataFrame or dict)
            - ``model_table``: ANOVA decomposition table (DataFrame or dict)
            - ``coefficients``: Coefficient table (DataFrame or dict)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, model_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.coefficients
        """

        return  self._get_results(include_test_stat_p=include_test_stat_p,
                                  include_effect_sizes=include_effect_sizes,
                                  factor_effects=False,
                                  return_type=return_type,
                                  na_rep=na_rep,
                                  pretty_format=pretty_format,
                                  table_decimals=table_decimals)


    def _summary_header_anova(self, width=78, model_summary_df=None):
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
            return [self._get_model_display_name()]

        model_display = self.ModelDesignSpec.model_display_name

        table = self.ModelResults.model_table.copy()

        lines = []
        col_w = {}
        for col in table.columns.to_list():
            max_str_len = table[col].astype(str).str.len().max()
            col_w[col] = max_str_len.item()

        table_w = sum(col_w.values()) + 3 * (len(col_w) - 1)
        print(f"Total table width: {table_w}")
        # table_w = sum(col_w.values()) + 6
        # print(f"Total table width: {table_w}")

        lines_after_model_name = 3
        lines.append(f"{model_display:<{table_w}}{"\n" * lines_after_model_name}")



        for col_name, col_data in table.items():
            print(f"Column: {col_name}, Max length: {col_data.astype(str).str.len().max()}")
            print("\n")
            print(f"{col_name:>{col_w[col_name]}} |")
            print(" " + col_data.astype(str) + " |")

        for idx, row in table.iterrows():
            # row.index gives you the column names for THIS row
            for col in row.index:
                print(f"{col}: {row[col]}")




        print(table["Source"].to_string(index=False))

        col_as_string = ""
        space_btwn_headers = 1
        for ix, row in table.iterrows():
            if ix < 3: print(row.to_string(index=False))

            col_as_string = ""

            btwn_headers = " " * space_btwn_headers
            if col == "Source":
                col_as_string = f"{col:}"
                sep = "-" * col_w[col] + f"{btwn_headers}|" * (table_w - col_w[col] - 1)
            else:
                sep = "-" * col_w[col] + " | " * (table_w - col_w[col] - 1)


            # print(table[col].to_string(index=False) + " | ")
            lines.append(f"{col:>{col_w.get(col, 2)}} | ")

            table[col].to_string(index=False)

        print("\n".join(lines))


        for row in table.itertuples(index=False):
            print(row)

            print(f"{row.Source:<{col_w[row.Source]}} |")












        space_btwn_headers = 1
        for col in table.columns.to_list():
            btwn_headers = " " * space_btwn_headers

            if col == "Source":
                lines.append(f"{col:>{col_w[col]}} |{btwn_headers}")
            else:
                lines.append(f"{col:<{col_w[col]}}{btwn_headers}")


        sep = "-" * col_w['Source'] + "-" * (table_w - col_w['Source'] - 1)
        lines.append(sep)



        lines.append(sep)



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


# Convenience aliases for users who prefer different naming conventions
LinearRegression = Regress
LM = Regress

