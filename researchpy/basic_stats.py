import warnings
import numpy
import scipy.stats


def count(d):
    """

        Counts the number of non-missing observations.

    """

    return numpy.count_nonzero(~numpy.isnan(d))


def nanvar(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The variance of the non-missing data passed; calculated as numpy.nanvar(d, ddof = 1).

    """
    return numpy.nanvar(d, ddof=1)


def nanstd(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The standard deviation of the non-missing data passed; calculated as numpy.nanstd(d, ddof = 1).

    """
    return numpy.nanstd(d, ddof=1)


def nansem(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The standard error of the non-missing data passed; calculated as scipy.stats.sem(d, nan_policy= 'omit').

    """
    return scipy.stats.sem(d, nan_policy='omit')


def value_range(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The range of the data passed; calculated as numpy.nanmax(d) - numpy.nanmin(d).

    """
    min_val = numpy.nanmin(d)
    max_val = numpy.nanmax(d)

    return float(max_val - min_val)


def kurtosis(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The kurtosis of the distribution of the data passed using Pearson's definition; calculated as scipy.stats.kurtosis(d, fisher = False, nan_policy = 'omit').'

    """
    return float(scipy.stats.kurtosis(d, fisher=False, nan_policy='omit'))


def skew(d):
    """

    Parameters
    ----------
    d : array_like
        The data to be passed.

    Returns
    -------
    Float
        The skew of the distribution of the data passed; calculated as scipy.stats.skew(d, nan_policy = 'omit').

    """
    return float(scipy.stats.skew(d, nan_policy='omit'))


def confidence_interval(d, alpha=0.95, n=None, loc=None, scale=None, decimals=4, confidence=None):
    """

    .. deprecated:: v0.3.7.2
        The interface of ``confidence_interval`` is changing and will be updated in the future version.
        As of v0.4.0 a ``TestResults`` object will be returned instead of a list of bounds. The
        bounds are accessible as ``result.lower`` and ``result.upper``.
        The former ``alpha`` parameter will be renamed to ``confidence``. The former ``n``, ``loc``, and ``scale``
        parameters are removed; passing them will raise a ``TypeError`` as of v0.4.0.
        See ``_confidence_interval`` for the new interface. See `TestResults` for the new return type.

    Parameters
    ----------
    d : array_like
        The data being passed to the function.
        In the future, this will be replaced with ``point_est`` which is either an ``np.ndarray`` or a scalar.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    confidence : decimal (float), optional
        Future replacement for ``alpha``. Pass only one of ``alpha`` or ``confidence``.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).
        In the future, this will be replaced with ``point_est`` which is either an ``np.ndarray`` or a scalar.

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)
        In the future, this will be replaced with ``scale_error_est`` which is either a scalar or None.

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the confidence interval in a list as the [lower_bound, upper_bound].
        In the future, this will be replaced with a ``TestResults`` object with ``lower`` and ``upper`` attributes.

    Examples
    --------
    >>> confidence_interval([1, 2, 3, 4, 5], alpha=0.95)
    [1.036, 4.964]

    """
    warnings.warn(
            "`confidence_interval` interface and return is changing and will be updated in the future version (0.4.0). "
            "See `_confidence_interval` for the new interface. See `TestResults` for the new return type. " 
            "The return type will be a `TestResults` object with `lower` and `upper` attributes, and the `alpha` parameter is renamed to `confidence`. " 
            "The `n`, `loc`, and `scale` parameters are removed.",
            FutureWarning,
            stacklevel=2,
    )
    if confidence is not None:
        if alpha != 0.95:
            raise TypeError("Pass only one of 'alpha' or 'confidence'.")
        alpha = confidence

    if n == None:
        n = count(d) - 1
    if loc == None:
        central = numpy.nanmean(d)
    if scale == None:
        scaler = nansem(d)

    ci_intervals = list(scipy.stats.t.interval(alpha,
                                               n,
                                               loc=central,
                                               scale=scaler))

    idx = 0
    for value in ci_intervals:
        ci_intervals[idx] = round(value, decimals)
        idx += 1

    return ci_intervals


def l_ci(d, alpha=0.95, n=None, loc=None, scale=None, decimals=4):
    """

    Parameters
    ----------
    d : array_like
        The data being passed to the function.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the lower boud of confidence interval.

    """

    if n == None:
        n = count(d) - 1
    if loc == None:
        central = numpy.nanmean(d)
    if scale == None:
        scaler = nansem(d)

    l_ci, _ = scipy.stats.t.interval(alpha,
                                     n - 1,
                                     loc=central,
                                     scale=scaler)
    return round(l_ci, decimals)


def u_ci(d, alpha=0.95, n=None, loc=None, scale=None, decimals=4):
    """

    Parameters
    ----------
    d : array_like
        The data being passed to the function.

    alpha : decimal (float), optional
        Confidence interval range to be calculated. The default is 0.95.

    n : numeric, optional
        The number of observations - 1. The default is None and is calculated as numpy.count_nonzero(~numpy.isnan(d)) - 1.

    loc : float, optional
        The central measure of tendency to be used. The default is None, which will be calculated as numpy.nanmean(d) (the mean).

    scale : float, optional
        The variability measure to be used. The default is None, which will be calculated as scipy.stats.sem(d, nan_policy= 'omit') (the standard error)

    decimals : integer, optional
        How many decimals places to round to. The default is 4.

    Returns
    -------
    ci_intervals : List
        Returns the upper boud of confidence interval.

    """

    if n == None:
        n = count(d) - 1
    if loc == None:
        central = numpy.nanmean(d)
    if scale == None:
        scaler = nansem(d)

    _, u_ci = scipy.stats.t.interval(alpha,
                                     n - 1,
                                     loc=central,
                                     scale=scaler)
    return round(u_ci, decimals)
