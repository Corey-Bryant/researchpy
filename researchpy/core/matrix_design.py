import scipy.stats
import formulaic

from typing import Any, Dict, List, Optional, Union

from researchpy.utility import variable_information



class DesignMatrices():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """
    def __init__(self, formula_like: str, data: object = {}, matrix_type: int = 1) -> None:
        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 0:
            formula_like = formula_like + " - 1"

        mm = formulaic.model_matrix(formula_like, data)
        self.DV = mm.lhs
        self.IV = mm.rhs

        # Model design information
        self.formula = formula_like
        self._DV_model_spec = self.DV.model_spec
        self._IV_model_spec = self.IV.model_spec

        ## My design information ##
        self.DV_name = list(self.DV.columns)[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(
            [str(t) for t in self.IV.model_spec.terms], list(self.IV.columns), data)



class DesignMatrix():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """
    def __init__(self, dv: Any = None, iv: Any = None, data={}, matrix_type=1,
                 return_type='matrix'):

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        formula = dv if isinstance(dv, str) else str(dv)
        if matrix_type == 0:
            formula = formula + " - 1"

        mm = formulaic.model_matrix(formula, data)
        self.DV = mm.lhs

        # Model design information
        self._DV_model_spec = self.DV.model_spec

        ## My design information ##
        self.DV_name = list(self.DV.columns)[0]



class DMatrix():

    def __init__(self, formula_like:str, data:object ={}, output:str ="numpy",
                 include_intercept:bool =True, ensure_full_rank:bool =True) -> None:


        '''         This approach builds a design matrix based on formula

        self.DV, self.IV = formulaic.Formula(
                formula_like,
                _parser=formulaic.parser.DefaultParser(include_intercept=include_intercept)
        ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank)
        '''

        # This uses the shorthand approach of the above
        if not include_intercept:
            formula_like = formula_like + " + 0"

        self.mm = formulaic.model_matrix(formula_like, data, output=output, ensure_full_rank=ensure_full_rank)
        self.DV = self.mm.lhs
        self.IV = self.mm.rhs


        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(
                [str(t) for t in self.IV.model_spec.terms],
                list(self.IV.model_spec.column_names),
                data
        )
