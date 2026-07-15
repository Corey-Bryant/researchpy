#import formulaic
import formulaic
from formulaic import Formula
from formulaic.parser import DefaultFormulaParser

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pandas import DataFrame as PandasDataFrame

from researchpy.core.syntax_engine import variable_information
from researchpy.containers import CoreDataclass, ModelTerms


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

    def __init__(self, formula:str, data:object ={}, output:str ="numpy",
                 include_intercept:bool =True, ensure_full_rank:bool =True, **kwargs, ) -> None:
        '''         This approach builds a design matrix based on formula

        self.DV, self.IV = formulaic.Formula(
                formula_like,
                _parser=formulaic.parser.DefaultFormulaParser(include_intercept=include_intercept)
        ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank)
        '''

        self.__name__ = "Researchpy.DesignMatrix"

        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = formulaic.Formula(formula,
                               _parser=DefaultFormulaParser(include_intercept=include_intercept),
                               **kwargs,
                               ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)

        # Extract what we need
        self.formula = formula
        self.DV = mm.lhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.IV = mm.rhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.model_terms = {"dv": ModelTerms.from_model_spec(mm.lhs.model_spec),
                            "iv": ModelTerms.from_model_spec(mm.rhs.model_spec)}


        # New dataclass-based term/column mapping (from formulaic ModelSpec)
        #self._model_terms = ModelTerms.from_model_spec(self.IV.model_spec)




class ModelMatrix():

    def __init__(self, formula_like:str, data:object ={}, output:str ="numpy",
                 include_intercept:bool =True, ensure_full_rank:bool =True) -> None:


        '''         This approach builds a design matrix based on formula

        self.DV, self.IV = formulaic.Formula(
                formula_like,
                _parser=formulaic.parser.DefaultFormulaParser(include_intercept=include_intercept)
        ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank)
        '''

        # This uses the shorthand approach of the above
        if not include_intercept:
            formula_like = formula_like + " + 0"

        self.ModelDesignSpec = ModelDesignSpec.from_formulaic(formula_like, data,
                                                              output=output,
                                                              ensure_full_rank=ensure_full_rank)
        self.DV = self.ModelDesignSpec.DV
        self.IV = self.ModelDesignSpec.IV

        # New dataclass-based term/column mapping (from formulaic ModelSpec)
        #self._model_terms = ModelTerms.from_model_spec(self.IV.model_spec)

        (self._patsy_factor_information,  #self.ModelDesignSpec.model_terms.term_map
         self._mapping,                     #self.ModelDesignSpec.model_terms.column_map
         self._rp_factor_information) = variable_information(
                [str(t) for t in self.IV.model_spec.terms],
                list(self.IV.model_spec.column_names),
                data
        )


@dataclass
class MatrixContainer(CoreDataclass):

    data: np.ndarray
    column_names: Optional[list[str]] = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __array__(self, dtype=None):
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def __getitem__(self, key):
        return self.data[key]

    def to_dataframe(self) -> PandasDataFrame:
        """Convert to labeled DataFrame."""
        if self.column_names is None:
            return PandasDataFrame(self.data)
        return PandasDataFrame(self.data, columns=self.column_names)



@dataclass
class ModelDesignSpec(CoreDataclass):
    """

    Standardized container for formulaic model output + Researchpy model fit information.

    The following attributes are defined:
    - ``formula``: The model formula as a string (e.g., "y ~ x1 + x2").
    - ``family``: The model family (e.g., "gaussian", "binomial").
    - ``link``: The link function used in the model (e.g., "identity", "logit").
    - ``n``: The number of observations used to fit the model.
    - ``k``: The number of predictors (including intercept) in the model.
    - ``ci_level``: The confidence interval level used for coefficient estimates (default is 0.95).
    - ``dv``: A list of dependent variable names (optional).
    - ``iv``: A list of independent variable names (optional).
    - ``model_display_name``: A user-friendly name for the model (optional).
    - ``model``: The internal name of the model (optional).

    """

    formula: Optional[str] = None
    model_spec: Optional[formulaic.ModelSpec] = None
    model_terms: Optional[ModelTerms] = None
    DV: Optional[MatrixContainer | np.ndarray] = None
    IV: Optional[MatrixContainer | np.ndarray] = None
    dv_term_names: Optional[list] = None
    iv_term_names: Optional[list] = None
    family: Optional[str] = None
    model: Optional[str] = None
    model_display_name: Optional[str] = None
    link: Optional[str] = None
    solver_method: Optional[str] = None
    ci_level: Optional[float] = 0.95
    additional_stats: Optional[dict] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.__name__ = "Researchpy.ModelFitSpec"


    @classmethod
    def from_formulaic(cls, formula: str, data: object, output: str = "numpy",
                       include_intercept: bool = True, ensure_full_rank: bool = True, **kwargs, ) -> "ModelFitSpec":
        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        #mm = formulaic.model_matrix(formula, data, output=output, ensure_full_rank=ensure_full_rank, **kwargs, )
        mm = Formula(formula,
                     _parser=DefaultFormulaParser(include_intercept=include_intercept),
                     **kwargs,
                     ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)

        # Extract what we need
        dv_term_names = (
            list(mm.lhs.model_spec.column_names)
            if hasattr(mm.lhs.model_spec, "column_names")
            else [str(t) for t in mm.lhs.model_spec.terms]
        )
        iv_term_names = (
            list(mm.rhs.model_spec.column_names)
            if hasattr(mm.rhs.model_spec, "column_names")
            else [str(t) for t in mm.rhs.model_spec.terms]
        )

        model_terms = ModelTerms.from_model_spec(mm.rhs.model_spec)
        #model_terms.dv = dv_term_names

        '''
        return cls(DV=MatrixContainer(np.asarray(mm.lhs), dv_term_names),
                   IV=MatrixContainer(np.asarray(mm.rhs), model_terms.columns),
                   model_spec=mm.model_spec,
                   model_terms=model_terms,
                   formula=formula,
                   dv_term_names=dv_term_names,
                   iv_term_names=iv_term_names,
                   metadata={
                                "include_intercept": include_intercept,
                                "ensure_full_rank" : ensure_full_rank,
                                "output"           : output,
                            } | kwargs,
                   )
        '''
        return cls(DV=mm.lhs,
                   IV=mm.rhs,
                   model_spec=mm.model_spec,
                   model_terms=model_terms,
                   formula=formula,
                   dv_term_names=dv_term_names,
                   iv_term_names=iv_term_names,
                   metadata={
                                "include_intercept": include_intercept,
                                "ensure_full_rank" : ensure_full_rank,
                                "output"           : output,
                            } | kwargs,
                   )