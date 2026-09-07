# -*- coding: utf-8 -*-
"""
Numerical diagnostics for model estimation.

This module provides reusable, purely-numerical checks used by the model
covariance computations (both OLS and GLM paths).  It intentionally contains
*no* statistical/family semantics, those live in ``researchpy.models.postestimation``.

The primary entry point is :func:`check_conditioning`, which inspects the
information matrix used to form the coefficient covariance matrix and reports
whether it is ill-conditioned / rank-deficient.  Callers use the result to
decide between a standard inverse and a pseudo-inverse fallback and to populate
``ModelDiagnostics``.
"""
from dataclasses import dataclass

import numpy as np


# Default condition-number threshold above which a matrix is treated as
# effectively rank-deficient for covariance estimation.  Kept as a module
# constant for now; may be promoted to a SolverOptions field if per-model
# control is needed.
CONDITION_NUMBER_THRESHOLD: float = 1e10


@dataclass
class ConditioningResult:
    """
    Result of a matrix conditioning check.

    Attributes
    ----------
    condition_number : float
        The 2-norm condition number of the inspected matrix. ``np.inf`` when
        the matrix is numerically singular.
    rank_deficient : bool
        True when the condition number is non-finite or exceeds *threshold*.
    threshold_used : float
        The threshold applied to flag rank deficiency.
    """

    condition_number: float
    rank_deficient: bool
    threshold_used: float


def check_conditioning(matrix: np.ndarray,
                       threshold: float = CONDITION_NUMBER_THRESHOLD
                       ) -> ConditioningResult:
    """
    Assess the numerical conditioning of a matrix.

    ``numpy.linalg.inv`` does not raise on a *near*-singular matrix; it silently
    returns an unstable inverse.  This helper detects that situation explicitly
    via the condition number so callers can route to a pseudo-inverse fallback.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to inspect (e.g., the upper-triangular ``R`` from a QR decomposition, or ``X'X`` / ``X'WX``).
    threshold : float, optional
        Condition-number threshold above which the matrix is flagged as
        rank-deficient. Defaults to :data:`CONDITION_NUMBER_THRESHOLD`.

    Returns
    -------
    ConditioningResult
        The computed condition number and rank-deficiency flag.
    """
    try:
        cond = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        cond = np.inf

    rank_deficient = (not np.isfinite(cond)) or (cond > threshold)

    return ConditioningResult(
        condition_number=cond,
        rank_deficient=rank_deficient,
        threshold_used=threshold,
    )

