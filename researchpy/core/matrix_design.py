from formulaic import Formula
from formulaic.parser import DefaultFormulaParser

from researchpy.containers.multivariable import ModelTerms

from researchpy.core.syntax_engine import SyntaxSpec

import numpy as np


class DesignMatrix():
    """

    This is the core matrix design object for Researchpy and the matrix engine.

    """

    def __init__(self, formula: str, data: object = {}, output: str = "numpy",
                 include_intercept: bool = True, ensure_full_rank: bool = True, **kwargs: object, ) -> None:

        self.__name__ = "Researchpy.DesignMatrix"

        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = Formula(formula,
                     _parser=DefaultFormulaParser(include_intercept=include_intercept),
                     **kwargs,
                     ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)


        # Extract what we need
        self.DV = mm.lhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.IV = mm.rhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.model_terms = {"dv": ModelTerms.from_model_spec(mm.lhs.model_spec),
                            "iv": ModelTerms.from_model_spec(mm.rhs.model_spec)}


    #---------------------------------------------------#
    #               Shared Builder Methods              #
    #---------------------------------------------------#
    @staticmethod
    def from_formula(formula: str, data: object = {}, output: str = "numpy",
                     include_intercept: bool = True, ensure_full_rank: bool = True, **kwargs: object, ):

        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = Formula(formula,
                     _parser=DefaultFormulaParser(include_intercept=include_intercept),
                     **kwargs,
                     ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)


        # Extract what we need
        DV = mm.lhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        IV = mm.rhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        model_terms = {"dv": ModelTerms.from_model_spec(mm.lhs.model_spec),
                       "iv": ModelTerms.from_model_spec(mm.rhs.model_spec)}

        return DV, IV, model_terms


    #---------------------------------------------------------------------------#
    #                       Shared Computational Methods                        #
    # --------------------------------------------------------------------------#
    def _hat_matrix(self):
        try:
            return np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
        except:
            return np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)

    def _j_matrix(self, n=None,):
        if n is None:
            n = self.DV.shape[0]
        return np.ones((n, n))

    def _identity_matrix(self, n=None,):
        if n is None:
            n = self.DV.shape[0]

        return np.identity(n)

    def _eigenval_matrix(self,):
        return np.asanyarray(np.linalg.eigvals(np.asarray(self.IV.T) @ np.asarray(self.IV)))

