# -*- coding: utf-8 -*-
"""
Separation diagnostics for logistic / binomial GLMs.

Provides a statistical model-fit check for (quasi-)complete separation, which
occurs when a linear combination of predictors perfectly (or almost perfectly)
predicts the binary outcome.  Under separation the maximum-likelihood estimates
diverge and standard errors become unreliable.

This module owns the *statistical* interpretation of the fitted values; the
purely numerical conditioning checks live in ``researchpy.optimize.diagnostics``.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from researchpy.optimize import ModelWarning


@dataclass
class SeparationResult:
    """
    Result of a separation check.

    Attributes
    ----------
    status : str
        One of ``"none"``, ``"quasi"``, or ``"complete"``.
    boundary_count : int
        Number of fitted probabilities numerically at 0 or 1.
    message : str or None
        Human-readable diagnostic message, or ``None`` when ``status`` is
        ``"none"``.
    """

    status: str
    boundary_count: int
    message: Optional[str] = None


def detect_separation(fitted_probs: np.ndarray,
                      tol: float = 1e-8,
                      warn: bool = True) -> SeparationResult:
    """
    Detect (quasi-)complete separation from fitted probabilities.

    Parameters
    ----------
    fitted_probs : array-like
        Fitted probabilities from a logistic/binomial model.
    tol : float, optional
        Tolerance for treating a fitted probability as numerically 0 or 1.
        Default is ``1e-8``.
    warn : bool, optional
        If True (default), emit a :class:`~researchpy.optimize.ModelWarning`
        when separation is detected. Estimation still completes; this is a
        warning, not an error.

    Returns
    -------
    SeparationResult
        The separation status, boundary count, and message.

    Notes
    -----
    - **Complete separation**: fitted probabilities lie *only* at the
      boundaries (all ~0 or ~1), i.e., the model perfectly classifies every
      observation.
    - **Quasi-complete separation**: *some* fitted probabilities sit at the
      boundary while others do not.
    """
    p = np.asarray(fitted_probs, dtype=float).ravel()

    at_zero = p <= tol
    at_one = p >= 1.0 - tol
    boundary_mask = at_zero | at_one
    boundary_count = int(np.sum(boundary_mask))

    if boundary_count == 0:
        return SeparationResult(status="none", boundary_count=0, message=None)

    if bool(np.all(boundary_mask)):
        status = "complete"
        message = (
            "Complete separation detected. Some fitted probabilities are "
            "numerically 0 or 1; coefficient estimates and standard errors "
            "are unreliable."
        )
    else:
        status = "quasi"
        message = (
            "Quasi-complete separation detected. Standard errors for one or "
            "more terms may be inflated."
        )

    if warn:
        warnings.warn(message, ModelWarning, stacklevel=2)

    return SeparationResult(status=status, boundary_count=boundary_count, message=message)

