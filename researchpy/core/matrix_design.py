from formulaic import Formula
from formulaic.parser import DefaultFormulaParser

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from researchpy.containers import CoreDataclass, ModelTerms



class DesignMatrix():

    def __init__(self, formula:str, data:object ={}, output:str ="numpy",
                 include_intercept:bool =True, ensure_full_rank:bool =True, **kwargs, ) -> None:

        self.__name__ = "Researchpy.DesignMatrix"

        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = Formula(formula,
                     _parser=DefaultFormulaParser(include_intercept=include_intercept),
                     **kwargs,
                     ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)

        # Extract what we need
        self.formula = formula
        self.DV = mm.lhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.IV = mm.rhs                                                   # Maintains a formuliac.model_spec.ModelSpec
        self.model_terms = {"dv": ModelTerms.from_model_spec(mm.lhs.model_spec),
                            "iv": ModelTerms.from_model_spec(mm.rhs.model_spec)}




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
    model_terms: Optional[ModelTerms] = None
    DV: Optional[np.ndarray] = None
    IV: Optional[np.ndarray] = None
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

        return cls(DV=mm.lhs,
                   IV=mm.rhs,
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