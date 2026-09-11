# -*- coding: utf-8 -*-
"""
Centralized p-value computation for distribution-based hypothesis tests.

Provides a single function, ``compute_pvalue``, that resolves a reference
distribution by name and computes the survival-function-based p-value for
a given test statistic.  This eliminates the repeated pattern of importing
``scipy.stats`` ad-hoc across the models sub-module.

Supported distributions
-----------------------
- ``"t"``    — Student's t  (requires ``df``)
- ``"z"``    — Standard normal
- ``"f"``    — Fisher–Snedecor F  (requires ``df`` and ``df_denom``)
- ``"chi2"`` — Chi-squared  (requires ``df``)

Usage
-----
>>> from researchpy.statistics._hypothesis_test import compute_pvalue
>>> compute_pvalue(2.45, "t", df=30)                    # two-sided t-test
>>> compute_pvalue(4.12, "f", df=3, df_denom=96)        # one-sided F-test
>>> compute_pvalue(7.88, "chi2", df=2)                  # one-sided chi² test

References
----------
- Casella, G. & Berger, R. L. (2002). *Statistical Inference* (2nd ed.).
  Duxbury.
"""
from researchpy.statistics import _get_distribution

from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from scipy.stats import distributions

import numpy as np

def _compute_pvalue(test_stat: Union[float, int, np.floating, np.integer],
                    distribution: str,
                    *,
                    df: Union[int, float, None] = None,
                    df_denom: Union[int, float, None] = None,
                    alternative: str = "two-sided",
                    **kwargs) -> Union[float, int, np.ndarray]:
    """Compute a p-value from a test statistic and reference distribution.

    Parameters
    ----------
    test_stat : float, int, or np.ndarray
        The observed test statistic (scalar or array).
    distribution : str
        The name of the distribution to retrieve (case-insensitive), see researchpy.statistics._get_distribution.
        Currently supported distributions include:
        - "t" or "student's t" for Student's t-distribution
        - "z" or "normal" for Standard normal distribution
        - "f" for Fisher–Snedecor F-distribution
        - "chi2" or "chi-squared" for Chi-squared distribution
        - "bernoulli" or "ber" for Bernoulli distribution
        - "binomial" or "bin" for Binomial distribution
        - "poisson" or "poi" for Poisson distribution
    df : int or float, optional
        Degrees of freedom.  Required for ``"t"``, ``"chi2"``, and ``"f"`` (numerator df for F).
    df_denom : int or float, optional
        Denominator degrees of freedom.  Required only for ``"f"``.
    alternative : str, optional
        Direction of the test.  One of ``"two-sided"`` (default), `"greater"``, or ``"less"``.
        - ``"two-sided"`` — ``2 * sf(|stat|, ...)``.  Typical for t and z coefficient tests.
        - ``"greater"``   — ``sf(stat, ...)``.  Typical for F and chi² omnibus tests.
        - ``"less"``      — ``cdf(stat, ...)``.

    Returns
    -------
    float, int, or np.ndarray
        The p-value(s), same shape as *test_stat*.

    Raises
    ------
    ValueError
        If *distribution* is not recognised, required degrees of freedom are missing, or *alternative* is invalid.

    Examples
    --------
    Two-sided t-test p-value:
    >>> _compute_pvalue(2.45,"t",df=30)
    0.0203...

    One-sided F-test p-value:
    >>> _compute_pvalue(4.12,"f",df=3,df_denom=96,alternative="greater")
    0.0087...

    Chi-squared test p-value:
    >>> _compute_pvalue(7.88,"chi2",df=2,alternative="greater")
    0.0194...

    """
    # -- Importing scipy.stats.distributions -- #
    if isinstance(distribution, str):
        distribution = _get_distribution(distribution)
        if distribution == '': distribution = _get_distribution(distribution, supported_only=False)
    elif isinstance(distribution, (distributions.rv_continuous, distributions.rv_discrete)):
        pass
    else:
        raise ValueError(
                f"distribution must be a string and a name of a distribution to be passed to researchpy.statistics._get_distribution, got {type(distribution)}. "
                f"Uses _get_distribution() to retrieve a distribution by name.",
        )

    # ------------------------------------------------------------------
    # Resolve distribution and validate required parameters
    # ------------------------------------------------------------------
    if distribution.name == "norm":
        rv = distribution

    elif distribution.name == "t":
        if df is None:
            raise ValueError(
                    f"Degrees of freedom (df) are required for the t-distribution. "
                    f"Got df={df}.",
            )
        rv = distribution(df=df)

    elif distribution.name == "f":
        if df is None or df_denom is None:
            raise ValueError(
                    f"Both numerator (df) and denominator (df_denom) degrees of "
                    f"freedom are required for the F-distribution. "
                    f"Got df={df}, df_denom={df_denom}.",
            )
        rv = distribution(dfn=df, dfd=df_denom)

    elif distribution.name == "chi2":
        if df is None:
            raise ValueError(
                    f"Degrees of freedom (df) are required for the chi-squared "
                    f"distribution. Got df={df}.",
            )
        rv = distribution(df=df)

    else:
        try:
            rv = distribution
        except ValueError:
            supported  = "'t', 'z', 'f', 'chi2'"
            raise ValueError(
                    f"Unknown distribution '{distribution.name}'. "
                    f"Supported distributions: {supported}.",
            )

    # ------------------------------------------------------------------
    # Compute p-value based on alternative hypothesis direction
    # ------------------------------------------------------------------
    if alternative == "two-sided":
        pvalue = 2.0 * rv.sf(np.abs(test_stat))
    elif alternative == "greater":
        pvalue = rv.sf(test_stat)
    elif alternative == "less":
        pvalue = rv.cdf(test_stat)
    else:
        raise ValueError(
                f"Invalid alternative '{alternative}'. "
                f"Must be 'two-sided', 'greater', or 'less'.",
        )

    return pvalue
