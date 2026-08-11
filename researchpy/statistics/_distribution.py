# -*- coding: utf-8 -*-
"""
This module provides functions for working with statistical distributions, including retrieving distributions by name,
computing confidence intervals, and calculating p-values for hypothesis tests.

Functions:
    - get_distribution(name): Retrieve a scipy distribution by name with clear error messaging.
    - _confidence_interval(point_est, scale_error_est, distribution, confidence=0.95, dof=None, decimals=None): Compute a confidence interval for a point estimate.
    - _compute_pvalue(test_stat, distribution_name, alternative="two-sided", dof=None): Compute a p-value for a hypothesis test.
"""
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
from researchpy.utility import as_numeric
from researchpy.containers import TestResults

if TYPE_CHECKING:
    from scipy.stats import distributions





def get_distribution(name: Optional[str],
                     return_supported: bool = False,
                     supported_only: bool = True
                     ) -> "scipy.stats.distributions":
    """
    Retrieve a scipy distribution by name with clear error messaging.

    Parameters
    ----------
    name : str
        The name of the distribution to retrieve. Supported currently supported distributions include:
        - "t" or "student's t" for Student's t-distribution
        - "z" or "normal" for Standard normal distribution
        - "f" for Fisher–Snedecor F-distribution
        - "chi2" or "chi-squared" for Chi-squared distribution
    return_supported : bool, optional
        If True, returns the list of supported distributions. Default is False.

    Returns
    -------
    ``scipy.stats.distributions``

    Examples
    --------
    >>> from researchpy.statistics import get_distribution
    >>> dist = get_distribution("t")
    >>> dist.name
    't'
    >>> dist = get_distribution("normal") # == get_distribution("z")
    >>> dist.name
    'norm'
    >>> dist = get_distribution("f")
    >>> dist.name
    'f'
    >>> dist = get_distribution("chi2")
    >>> dist.name
    'chi2'
    """
    from scipy.stats import distributions

    # Map user-friendly names to scipy canonical names
    distribution_map = {
        "norm" : ["norm", "z", "normal", "gaussian"],
        "t"    : ["t", "student's t", "students t"],
        "chi2" : ["chi2", "chi^2", "chi-sq", "chi-squared"],
        "f"    : ["f", "fisher"],
    }

    name = name.lower().strip()

    if return_supported:
        supported = []
        for k, v in distribution_map.items():
            supported.append(k), supported.append(v)

        print("Supported distributions:")
        return list(distribution_map.items())

    if name not in distribution_map:
        if supported_only:
            raise ValueError(
                    f"Distribution '{name}' is not supported. "
                    f"Use `return_supported=True` to see the list of supported distributions."
            )


        if not hasattr(distributions, name):
            available = [n for n in dir(distributions) if isinstance(getattr(distributions, n),
                                                                     (distributions.rv_continuous, distributions.rv_discrete)
                                                                     )
                         ]
            raise ValueError(
                    f"Distribution '{name}' not found. "
                    f"Available options include: {', '.join(sorted(available)[:10])}..."
            )


    name = distribution_map[name]
    canonical_name = distribution_map[name]
    return getattr(distributions, canonical_name)




def _confidence_interval(point_est: Union[float, int, np.floating, np.integer],
                         scale_error_est: Union[float, int, np.floating, np.integer],
                         distribution: Union[str, distributions.rv_continuous, distributions.rv_discrete],
                         *,
                         confidence: Union[float, int, np.floating, np.integer] = 0.95,
                         dof: Union[float, int, np.floating, np.integer, None] = None,
                         decimals: Union[float, int, np.floating, np.integer, None] = None,
                         **kwargs: object,
                         ) -> TestResults:
    """
    Computes a confidence interval for a point estimate using the specified distribution.
s
    Parameters
    ----------
    :param point_est: The point estimate for which the confidence interval is to be computed.
    :type point_est: Union[float, int, np.floating, np.integer]
    :param scale_error_est: The standard error of the point estimate.
    :type scale_error_est: Union[float, int, np.floating, np.integer]
    :param distribution: The name of the distribution to use for computing the confidence interval.
    :type distribution: Union[str, distributions.rv_continuous, distributions.rv_discrete]. If str is passed, it will be
            resolved to a ``scipy.stats distribution`` using ``get_distribution()``.
    :param confidence: The confidence level for the interval.
    :type confidence: Union[float, int, np.floating, np.integer]
    :param dof: Degrees of freedom for the t-distribution.
    :type dof: Union[float, int, np.floating, np.integer, None]
    :param decimals: Number of decimal places to round the results to.
    :type decimals: Union[float, int, np.floating, np.integer, None]

    Returns
    -------
    :return: The confidence interval as a TestResults object.
    :rtype: TestResults

    Examples
    --------
    >>> from researchpy import _confidence_interval
    >>> point_est = 5.0
    >>> scale_error_est = 1.0
    >>> distribution_name = "norm"
    >>> confidence = 0.95
    >>> ci_result = _confidence_interval(point_est, scale_error_est, distribution_name, confidence=confidence)
    >>> print(ci_result.statistics)
    {'lower': 3.04, 'upper': 6.96}

    """
    # --------------------------- #
    # -- Validating Parameters -- #
    # ----------------------------#
    if confidence <= 0 or confidence >= 1 :
        raise ValueError(
                f"confidence must be between 0 and 1 (exclusive), got {confidence}. "
                f"For a 95% CI, use confidence=0.95.",
        )

    if isinstance(distribution, str):
        distribution = get_distribution(distribution)
    elif isinstance(distribution, (distributions.rv_continuous, distributions.rv_discrete)):
        pass
    else:
        raise ValueError(
                f"distribution must be a string or a scipy.stats distribution object, got {type(distribution)}. "
                f"Use get_distribution() to retrieve a distribution by name.",
        )

    #--------------------------------------#
    # -- Estimating Confidence Interval -- #
    #--------------------------------------#
    if distribution.name.lower() == "t":
        if dof is None:
            raise ValueError(
                    f"Degrees of freedom (dof) must be provided for t-distribution. "
            )
        lower, upper = distribution.interval(confidence, df=dof, loc=point_est, scale=scale_error_est)
    else:
        lower, upper = distribution.interval(confidence, loc=point_est, scale=scale_error_est)



    #--------------------------------#
    # -- Formatting and Returning -- #
    #--------------------------------#
    try:
        lower = as_numeric(lower)
        upper = as_numeric(upper)

        if decimals is not None and decimals >= 0:
            lower = round(lower, decimals)
            upper = round(upper, decimals)

    except Exception as e:
        print(
                f"Error: {e} --> Failed to convert lower and upper bounds to numeric; lower={lower}, upper={upper}."
                f"Returning as original values instead, i.e. returning as [{lower}, {upper}].",
        )

    test_name = f"{confidence * 100:.1f}% Conf. Interval"
    return TestResults(
            test_name=test_name,
            statistics={"lower": lower, "upper": upper},
            details={test_name: [f"Estimated using the {distribution.name} distribution"]},
    )



def _compute_pvalue(test_stat: Union[float, int, np.floating, np.integer],
                    distribution: Union[str, distributions.rv_continuous, distributions.rv_discrete],
                    *,
                    df: Union[int, float, None] = None,
                    df_denom: Union[int, float, None] = None,
                    alternative: str = "two-sided",
                    ) -> Union[TestResults, float, int, np.ndarray]:
    """Compute a p-value from a test statistic and reference distribution.

    Parameters
    ----------
    test_stat : float, int, or np.ndarray
        The observed test statistic (scalar or array).
    distribution : str or scipy.stats distribution object
        Name of the reference distribution.  One of ``"t"``, ``"z"``, ``"f"``, or ``"chi2"`` (case-insensitive).
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
        If *distribution* is not recognised, required degrees of freedom
        are missing, or *alternative* is invalid.

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
        distribution = get_distribution(distribution)
    elif isinstance(distribution, (distributions.rv_continuous, distributions.rv_discrete)):
        pass
    else:
        raise ValueError(
                f"distribution must be a string or a scipy.stats distribution object, got {type(distribution)}. "
                f"Use get_distribution() to retrieve a distribution by name.",
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













