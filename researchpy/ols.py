"""
Ordinary Least Squares (OLS) Regression

This module provides the OLS class for fitting linear regression models
using the ordinary least squares method.

DEPRECATION NOTICE:
    The `ols` class in this module is maintained for backward compatibility with v0.3.7.
    It will be deprecated in a future version.
    New code should use the new API modular structure introduced with this version.

    The `decimals` parameter in the `results()` method is deprecated and is being renamed
    to `table_decimals` in a future version for better clarity and consistency.

"""

import warnings
import numpy
import scipy.stats
import patsy
import pandas

from researchpy.model import model
from researchpy.utility import *
from researchpy.predict import predict


class ols(model):
    """

    .. deprecated::
        The `ols` class is deprecated and will be removed in a future version.
        Please use the new modular regression interface once available in v0.3.8.

    Parameters
    ----------
    formula_like: string
        A string which represents a valid Patsy formula; https://patsy.readthedocs.io/en/latest/

    data : array_like
        Array like data object.

    Returns
    -------
    Ordinary Least Squares regression object with assessible methods and stored class data. The class data
    which is stored is the following:


        self.model_data: dictionary object
            The following data is stored with the dictionary key ("Key"):
                J matrix ('J')
                Identify matrix ('I')
                Hat matrix ('H')
                Coeffeicients ('betas')
                Total Sum of Squares ('sum_of_square_total')
                Model Sum of Squares ('sum_of_square_model')
                Residual Sum of Squares ('sum_of_square_residual')
                Model Degrees of Freedom ('degrees_of_freedom_model')
                Residual Degrees of Freedom ('degrees_of_freedom_residual')
                Total Degrees of Freedom ('degrees_of_freedom_total')
                Model Mean Squares ('msr')
                Error Mean Squares ('mse')
                Total Mean Squares ('mst')
                Root Mean Square Error ('root_mse')
                Model F-value ('f_value_model')
                Model p-value ('f_p_value_model')
                R-sqaured ('r squared')
                Adjusted R-squared ('r squared adj.')
                Eta squared ('Eta squared')
                Epsilon squared ('Epsilon squared')
                Omega squared ('Omega squared')
=][]
    """

    def __init__(self, formula_like, data={}):
        # Issue deprecation warning
        warnings.warn(
            "The 'ols' class is deprecated and will be removed in a future version. "
            "A new modular regression interface will be available in v0.3.8. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(formula_like, data, matrix_type=1)

        self.model_data = {}

        ###########

        # J matrix of ones based on y
        self.model_data["J"] = numpy.ones((self.nobs, self.nobs))

        # identity matrix (I) based on x
        self.model_data["I"] = numpy.identity(self.nobs)

        # Eigenvalues
        self.eigvals = numpy.linalg.eigvals(self.IV.T @ self.IV)

        # Hat matrix
        try:
            self.model_data["H"] = self.IV @ numpy.linalg.inv(
                self.IV.T @ self.IV) @ self.IV.T
        except:
            self.model_data["H"] = self.IV @ numpy.linalg.pinv(
                self.IV.T @ self.IV) @ self.IV.T
            # print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")

        # Estimation of betas
        try:
            self.model_data["betas"] = numpy.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = numpy.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        predicted_y = self.IV @ self.model_data["betas"]

        # Calculation of residuals (error)
        residuals = self.DV - predicted_y

        ###  Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = (
            self.DV.T @ self.DV - (1 / self.nobs) * self.DV.T @ self.model_data["J"] @ self.DV).item()

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = (
            self.model_data["betas"].T @ self.IV.T @ self.DV - (1 / self.nobs) * self.DV.T @ self.model_data[
                "J"] @ self.DV).item()

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = (residuals.T @ residuals).item()

        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = numpy.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.nobs - numpy.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.nobs - 1

        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        #Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])

        ## Root Mean Square Error
        self.model_data["root_mse"] = float(numpy.sqrt(self.model_data["mse"]))

        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf(self.model_data["f_value_model"],
                                                              self.model_data["degrees_of_freedom_model"],
                                                              self.model_data["degrees_of_freedom_residual"])

        ### Effect Size Measures
        # Model
        self.model_data["r squared"] = (self.model_data["sum_of_square_model"] / self.model_data["sum_of_square_total"])
        self.model_data["r squared adj."] = 1 - (
                    self.model_data["degrees_of_freedom_total"] / self.model_data["degrees_of_freedom_residual"]) * (
                                                    self.model_data["sum_of_square_residual"] / self.model_data[
                                                "sum_of_square_total"])
        self.model_data["Eta squared"] = self.model_data["r squared"]

        self.model_data["Epsilon squared"] = (self.model_data["degrees_of_freedom_model"] * (
                    self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"])

        self.model_data["Omega squared"] = (self.model_data["degrees_of_freedom_model"] * (
                    self.model_data["msr"] - self.model_data["mse"])) / (
                                                       self.model_data["sum_of_square_total"] + self.model_data["mse"])

        ### Variance-covariance matrices
        # Non-robust - from Applied Linear Statistical Models, pg. 203
        self.variance_covariance_residual_matrix = numpy.matrix(self.model_data["mse"] * (self.model_data["I"] - self.model_data["H"]))

        try:
            self.variance_covariance_beta_matrix = numpy.matrix(
                self.model_data["mse"] * numpy.linalg.inv(self.IV.T @ self.IV))
        except:
            self.variance_covariance_beta_matrix = numpy.matrix(
                self.model_data["mse"] * numpy.linalg.pinv(self.IV.T @ self.IV))

        #if robust_standard_errors == False:
            #self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))
        #elif robust_standard_errors == True:
            ## Not implemented yet so it's same as non-robust
         #   self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))

    def results(self, return_type="Dataframe", decimals=4, pretty_format=True, conf_level=0.95):

        ### Standard Errors
        standard_errors = (numpy.array(numpy.sqrt(self.variance_covariance_beta_matrix.diagonal()))).T

        ### Confidence Intrvals
        conf_int_lower = []
        conf_int_upper = []

        for beta, se in zip(self.model_data["betas"], standard_errors):

            try:
                lower, upper = scipy.stats.t.interval(conf_level, self.model_data["degrees_of_freedom_residual"],
                                                      loc=beta, scale=se)

                conf_int_lower.append(numpy.asarray(lower).item())
                conf_int_upper.append(numpy.asarray(upper).item())

            except:

                conf_int_lower.append(numpy.nan)
                conf_int_upper.append(numpy.nan)

        ### T-stastics
        t_stastics = self.model_data["betas"] * (1 / standard_errors)
        # Two-sided p-value
        t_p_values = numpy.array([numpy.asarray(scipy.stats.t.sf(numpy.abs(
            t), self.model_data["degrees_of_freedom_residual"]) * 2).item() for t in t_stastics])

        ## Creating variable table information
        regression_description_info = {

            self._DV_design_info.term_names[0]: ["Coef.", "Std. Err.", "t", "p-value", "95% Conf. Interval"],

        }

        regression_info = {self._DV_design_info.term_names[0]: [],
                           "Coef.": [],
                           "Std. Err.": [],
                           "t": [],
                           "p-value": [],
                           f"{int(conf_level * 100)}% Conf. Interval": []}

        for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names, self.model_data["betas"],
                                                          standard_errors, t_stastics, t_p_values, conf_int_lower,
                                                          conf_int_upper):
            regression_info[self._DV_design_info.term_names[0]].append(column)
            regression_info["Coef."].append(round(beta[0], decimals))
            regression_info["Std. Err."].append(round(stderr[0], decimals))
            regression_info["t"].append(round(t[0], decimals))
            regression_info["p-value"].append(round(p, decimals))
            regression_info[f"{int(conf_level * 100)}% Conf. Interval"].append(
                [round(l_ci, decimals), round(u_ci, decimals)])

        regression_info = base_table(self._patsy_factor_information, self._mapping,
                                     self._rp_factor_information, pandas.DataFrame.from_dict(regression_info))

        if pretty_format == True:

            descriptives = {

                "Number of obs = ": self.nobs,
                "Root MSE = ": round(self.model_data["root_mse"], decimals),
                "R-squared = ": round(self.model_data["r squared"], decimals),
                "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

            }

            top = {

                "Source": ["Model", ''],
                "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals), ''],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals), ''],
                "Mean Squares": [round(self.model_data["msr"], decimals), ''],
                "F value": [round(self.model_data["f_value_model"], decimals), ''],
                "p-value": [round(self.model_data["f_p_value_model"], decimals), ''],
                "Eta squared": [round(self.model_data["Eta squared"], decimals), ''],
                "Epsilon squared": [round(self.model_data["Epsilon squared"], decimals), ''],
                "Omega squared": [round(self.model_data["Omega squared"], decimals), '']

            }

            bottom = {

                "Source": ["Residual", "Total"],
                "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals),
                                   round(self.model_data["sum_of_square_total"], decimals)],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals),
                                       round(self.model_data["degrees_of_freedom_total"], decimals)],
                "Mean Squares": [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                "F value": ['', ''],
                "p-value": ['', ''],
                "Eta squared": ['', ''],
                "Epsilon squared": ['', ''],
                "Omega squared": ['', '']

            }

            results = {

                "Source": top["Source"] + bottom["Source"],
                "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
                "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
                "F value": top["F value"] + bottom["F value"],
                "p-value": top["p-value"] + bottom["p-value"],
                "Eta squared": top["Eta squared"] + bottom["Eta squared"],
                "Epsilon squared": top["Epsilon squared"] + bottom["Epsilon squared"],
                "Omega squared": top["Omega squared"] + bottom["Omega squared"]

            }

        else:

            descriptives = {

                "Number of obs = ": self.nobs,
                "Root MSE = ": round(self.model_data["root_mse"], decimals),
                "R-squared = ": round(self.model_data["r squared"], decimals),
                "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

            }

            top = {

                "Source": ["Model"],
                "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals)],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals)],
                "Mean Squares": [round(self.model_data["msr"], decimals)],
                "F value": [round(self.model_data["f_value_model"], decimals)],
                "p-value": [round(self.model_data["f_p_value_model"], decimals)]

            }

            bottom = {

                "Source": ["Residual", "Total"],
                "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals),
                                   round(self.model_data["sum_of_square_total"], decimals)],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals),
                                       round(self.model_data["degrees_of_freedom_total"], decimals)],
                "Mean Squares": [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                "F value": [numpy.nan, numpy.nan],
                "p-value": [numpy.nan, numpy.nan]

            }

            results = {

                "Source": top["Source"] + bottom["Source"],
                "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
                "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
                "F value": top["F value"] + bottom["F value"],
                "p-value": top["p-value"] + bottom["p-value"]

            }

        if return_type == "Dataframe":

            return (pandas.DataFrame.from_dict(descriptives, orient="index"), pandas.DataFrame.from_dict(results),
                    pandas.DataFrame.from_dict(regression_info))

        elif return_type == "Dictionary":
            #return (descriptives, results, regression_info)   # apart of Feature_SyntaxEngine
            return (descriptives, results, regression_info.to_dict())

        else:

            print(
                "Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")


    def predict(self, estimate=None):
        return predict(self, estimate=estimate)