# -*- coding: utf-8 -*-

from typing import Dict, Any

import scipy.stats

from researchpy.utility import as_numeric
from researchpy.containers import TestResults


# ---------------------------------------------------------------------------
# Registry: maps user-facing stat names to computation callables.
# Each callable takes (arr, **kwargs) and returns a scalar or structured value.
# ---------------------------------------------------------------------------

_DISTRIBUTION_REGISTRY: Dict[str, Any] = {
    "normal": scipy.stats.norm,
    "t": scipy.stats.t,
    "student's t": scipy.stats.t,
}



def _confidence_interval(point_est, scale_error_est, confidence=0.95,
                         distribution_name=None, distribution_object=None, dof=None,
                         decimals=None):




    #-----------------------------#
    # -- Validating parameters -- #
    #-----------------------------#
    # -- Validating the distribution -- #
    if distribution_name is None and distribution_object is None:
        raise ValueError(
                "Either distribution_name or distribution_object must be provided. If both are provided, distribution_object will be used."
        )

    if distribution_object:
        if distribution_name:
            print(
                    f"Both distribution_name and distribution_object are provided, naming distribution {distribution_name} (user supplied) and attempting to use {distribution_object}."
            )
        else:
            try:
                distribution_name = str(distribution_object.name)
            except AttributeError:
                try:
                    distribution_name = str(distribution_object.__class__)
                except:
                    distribution_name = str(distribution_object)

        if not isinstance(distribution_object, scipy.stats.rv_continuous) and not isinstance(distribution_object, scipy.stats.rv_discrete):
            raise ValueError(
                f"Invalid distribution_object: {distribution_object}, {type(distribution_object)}. Must be a scipy.stats continuous or discrete distribution."
            )

    else:
        if distribution_name.lower() not in _DISTRIBUTION_REGISTRY.keys():
            raise ValueError(
                    f"Unsupported distribution: {distribution_name}. Supported distributions are: {', '.join(_DISTRIBUTION_REGISTRY.keys())}."
                    "Or trying passing a scipy.stats distribution object directly, e.g. scipy.stats.norm, to distribution_object."
            )

        else:
            distribution_object = _DISTRIBUTION_REGISTRY[distribution_name.lower()]


    # -- Validating confidence -- #
    if confidence <= 0 or confidence >= 1:
        raise ValueError(
            f"confidence must be between 0 and 1 (exclusive), got {confidence}. "
            f"For a 95% CI, use confidence=0.95."
        )


    #--------------------------------------#
    # -- Estimating Confidence Interval -- #
    #--------------------------------------#
    if distribution_name.lower() not in ["t", "student's t"]:
        lower, upper = distribution_object.interval(confidence, loc=point_est, scale=scale_error_est)

    elif distribution_name.lower() in ["t", "student's t"]:
        if dof is None:
            raise ValueError(
                f"Degrees of freedom (dof) must be provided for t-distribution. "
                f"Got dof={dof}."
            )

        lower, upper = distribution_object.interval(confidence, df=dof, loc=point_est, scale=scale_error_est)


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
        print (
                f"Error: {e} --> Failed to convert lower and upper bounds to numeric; lower={lower}, upper={upper}."
                f"Returning as original values instead, i.e. returning as [{lower}, {upper}]."
        )


    test_name = f"{confidence*100:.1f}% Conf. Interval"
    return TestResults(
            test_name=test_name,
            statistics={"lower": lower, "upper": upper},
            details={test_name: [f"Estimated using the {distribution_name} distribution"]}
    )
