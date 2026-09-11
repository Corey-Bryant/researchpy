"""
DesignMatrix — the design matrix construction layer.

Takes the output of :class:`~researchpy.engine.syntax.SyntaxParser` and
builds the numerical design matrices used by regression models, descriptive
statistics, and all other computation paths.

Heavily relies on ``formulaic.Formula.get_model_matrix`` and wraps
results in researchpy's own :class:`~researchpy.core.matrix_design.DesignMatrix`
and :class:`~researchpy.containers.multivariable.ModelTerms` containers.

Usage
-----
>>> from researchpy.engine.syntax import FormulaSpec
>>> from researchpy.engine.matrix import DesignMatrix
>>>
>>> spec = SyntaxParser.from_args("y ~ x1 + x2", df)
>>> engine = DesignMatrix.from_spec(spec)
>>> engine.DV          # numpy array / formulaic ModelMatrix (LHS)
>>> engine.IV          # numpy array / formulaic ModelMatrix (RHS)
>>> engine.model_terms # ModelTerms container
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from formulaic import Formula
from formulaic.parser import DefaultFormulaParser

from formulaic import model_matrix

from researchpy.containers.base import CoreDataclass
from researchpy.containers.multivariable import ModelTerms

if TYPE_CHECKING:
    from researchpy.engine.syntax import FormulaSpec


# ======================================================================
# DesignMatrix dataclass
# ======================================================================

@dataclass
class DesignMatrix(CoreDataclass):
    """Design matrix container built from a parsed syntax specification.

    Attributes
    ----------
    DV : array-like or None
        Left-hand side (dependent variable) design matrix.
        Retains the ``formulaic.ModelMatrix`` interface when built
        via :meth:`from_formula`.
    IV : array-like or None
        Right-hand side (independent variables) design matrix.
    model_terms : dict or None
        ``{"lhs": ModelTerms, "rhs": ModelTerms}`` mapping term metadata
        for both sides of the formula.
    formula : str or None
        The formula string used to build the matrices.
    n : int or None
        Number of observations (rows).
    k : int or None
        Number of predictors (columns in IV, including intercept if present).
    """

    DV: Optional[Any] = None
    IV: Optional[Any] = None
    model_terms: Optional[Dict[str, ModelTerms]] = None
    formula: Optional[str] = None
    n: Optional[int] = None
    k: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = "Researchpy.DesignMatrix"

        # Auto-populate n, k from array shapes
        if self.DV is not None and self.n is None:
            self.n = np.asarray(self.DV).shape[0]

        if self.IV is not None and self.k is None:
            self.k = np.asarray(self.IV).shape[1] if np.asarray(self.IV).ndim > 1 else 1


    # ==================================================================
    # Factory: from FormulaSpec spec
    # ==================================================================
    @classmethod
    def from_spec(
        cls,
        spec: "FormulaSpec",
        output: str = "numpy",
        include_intercept: bool = True,
        ensure_full_rank: bool = True,
        **kwargs,
    ) -> "DesignMatrix":
        """Build a DesignMatrix from a :class:`FormulaSpec` instance.

        Delegates to :meth:`from_formula` using the spec's formula and data.

        Parameters
        ----------
        spec : FormulaSpec
            Parsed syntax specification (must have ``formula`` and ``data``).
        output : str, optional
            Output format for ``formulaic``.  Default is ``"numpy"``.
        include_intercept : bool, optional
            Whether to include an intercept term.  Default is ``True``.
        ensure_full_rank : bool, optional
            Whether to ensure the design matrix is full rank.
            Default is ``True``.
        **kwargs
            Additional keyword arguments passed to ``formulaic``.

        Returns
        -------
        DesignMatrix

        Raises
        ------
        ValueError
            If the spec has no formula or no data.
        """
        if spec.formula is None:
            raise ValueError(
                "FormulaSpec spec has no formula.  Cannot build a design matrix "
                "without a formula string."
            )
        if spec.data is None:
            raise ValueError(
                "FormulaSpec spec has no data.  Cannot build a design matrix "
                "without a DataFrame."
            )

        return cls.from_formula(
            formula=spec.formula,
            data=spec.data,
            output=output,
            include_intercept=include_intercept,
            ensure_full_rank=ensure_full_rank,
            **kwargs,
        )


    # ==================================================================
    # Factory: from formula string + data
    # ==================================================================
    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: Any,
        output: str = "numpy",
        include_intercept: bool = True,
        ensure_full_rank: bool = True,
        **kwargs,
    ) -> "DesignMatrix":
        """Build a DesignMatrix from a formula string and DataFrame.

        Constructs a :class:`DesignMatrix` instance using ``formulaic.sugar.model_matrix`` to construct the
        ``DV (lhs)`` and ``IV (rhs)`` matrices, then wraps them in a ``DesignMatrix``
        instance together with :class:`ModelTerms` metadata. ``DV`` and ``IV`` retains maintains the
        ``formulaic.model_matrix.ModelMatrix`` interface.

        Within the ``model_terms`` attribute, the ``lhs`` key contains metadata for the dependent variable (``DV``),
        while the ``rhs`` key contains metadata for the independent variables (``IV``).


        Parameters
        ----------
        formula : str
            Wilkinson-style formula (e.g., ``"y ~ x1 + x2"``).
        data : pd.DataFrame
            Source data.
        output : str, optional
            Output type passed to formulaic (``"numpy"``, ``"pandas"``,
            ``"sparse"``).  Default is ``"numpy"``.
        include_intercept : bool, optional
            Whether to include an intercept.  Default is ``True``.
        ensure_full_rank : bool, optional
            Whether to drop aliased columns.  Default is ``True``.
        **kwargs
            Extra arguments forwarded to ``formulaic``.

        Returns
        -------
        DesignMatrix
        """
        mm = Formula(formula,
                          _parser=DefaultFormulaParser(include_intercept=include_intercept),
                          **kwargs,
                          ).get_model_matrix(data,
                                      output=output,
                                      ensure_full_rank=ensure_full_rank,**kwargs,
                                      )

        DV = mm.lhs
        IV = mm.rhs

        terms = {
            "lhs": ModelTerms.from_model_spec(mm.lhs.model_spec),
            "rhs": ModelTerms.from_model_spec(mm.rhs.model_spec),
        }
        #terms = ModelTerms.from_model_specs(mm.model_spec)

        return cls(
            DV=DV,
            IV=IV,
            model_terms=terms,
            formula=formula,
        )


    @classmethod
    def get_design_matrix(
        formula: str,
        data: Any,
        output: str = "numpy",
        include_intercept: bool = True,
        ensure_full_rank: bool = True,
        **kwargs,
    ) -> "DesignMatrix":
        """Build a DesignMatrix from a formula string and DataFrame.

        Constructs a :class:`DesignMatrix` instance using ``formulaic.sugar.model_matrix`` to construct the
        ``DV (lhs)`` and ``IV (rhs)`` matrices, then wraps them in a ``DesignMatrix``
        instance together with :class:`ModelTerms` metadata. ``DV`` and ``IV`` retains maintains the
        ``formulaic.model_matrix.ModelMatrix`` interface.

        Within the ``model_terms`` attribute, the ``lhs`` key contains metadata for the dependent variable (``DV``),
        while the ``rhs`` key contains metadata for the independent variables (``IV``).


        Parameters
        ----------
        formula : str
            Wilkinson-style formula (e.g., ``"y ~ x1 + x2"``).
        data : pd.DataFrame
            Source data.
        output : str, optional
            Output type passed to formulaic (``"numpy"``, ``"pandas"``,
            ``"sparse"``).  Default is ``"numpy"``.
        include_intercept : bool, optional
            Whether to include an intercept.  Default is ``True``.
        ensure_full_rank : bool, optional
            Whether to drop aliased columns.  Default is ``True``.
        **kwargs
            Extra arguments forwarded to ``formulaic``.

        Returns
        -------
        DesignMatrix
        """
        formula_str = formula
        if not include_intercept:
            formula_str = formula_str + " + 0"

        mm = model_matrix(formula_str, data=data, output=output,
                          ensure_full_rank=ensure_full_rank, **kwargs)

        DV = mm.lhs
        IV = mm.rhs

        terms = {
            "lhs": ModelTerms.from_model_spec(mm.lhs.model_spec),
            "rhs": ModelTerms.from_model_spec(mm.rhs.model_spec),
        }

        return DesignMatrix(
                DV=DV,
                IV=IV,
                model_terms=terms,
                formula=formula,
            )



    # ==================================================================
    # Computational helpers
    # ==================================================================
    def hat_matrix(self, IV: DesignMatrix) -> np.ndarray:
        """Compute the hat matrix H = X(X'X)⁻¹X'.

        Falls back to the Moore-Penrose pseudoinverse if X'X is singular.

        Returns
        -------
        np.ndarray
            Hat matrix of shape ``(n, n)``.
        """
        X = np.asarray(self.IV)
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        return X @ XtX_inv @ X.T

    def j_matrix(self, n: Optional[int] = None) -> np.ndarray:
        """Return an n×n matrix of ones (J matrix).

        Parameters
        ----------
        n : int or None
            Dimension.  Defaults to the number of observations.
        """
        size: int = n if n is not None else (self.n or 0)
        return np.ones((size, size))

    def identity_matrix(self, n: Optional[int] = None) -> np.ndarray:
        """Return an n×n identity matrix.

        Parameters
        ----------
        n : int or None
            Dimension.  Defaults to the number of observations.
        """
        size: int = n if n is not None else (self.n or 0)
        return np.identity(size)

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of X'X.

        Returns
        -------
        np.ndarray
            1-D array of eigenvalues.
        """
        X = np.asarray(self.IV)
        return np.linalg.eigvals(X.T @ X)

    # ==================================================================
    # Info / repr
    # ==================================================================
    def info(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        if self.formula:
            lines.append(f"  formula='{self.formula}'")
        if self.DV is not None:
            lines.append(f"  DV: {type(self.DV).__name__} shape={np.asarray(self.DV).shape}")
        if self.IV is not None:
            lines.append(f"  IV: {type(self.IV).__name__} shape={np.asarray(self.IV).shape}")
        lines.append(f"  n={self.n}, k={self.k}")

        if self.model_terms:
            for side, mt in self.model_terms.items():
                lines.append(f"  model_terms['{side}']: {len(mt)} term(s)")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.info()



