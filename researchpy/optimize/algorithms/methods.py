import numpy as np
from scipy.special import expit



def newton_raphson(IV, DV, betas, tol, max_iter, display):
    """Newton-Raphson optimization algorithm."""
    from scipy.optimize import OptimizeResult

    converged = False
    it = 0
    error = np.ones_like(betas)
    logL = []
    while np.any(error > tol) and it < max_iter:
        linear_pred = IV @ betas
        p = expit(linear_pred)  # Use expit for stability

        w = p * (1 - p).reshape(-1, 1)
        H = -(IV.T @ (IV * w))
        G = IV.T @ (DV - p)

        try:
            betas_new = betas - np.linalg.inv(H) @ G
        except np.linalg.LinAlgError:
            betas_new = betas - np.linalg.pinv(H) @ G

        error = np.abs(betas_new - betas)
        betas = betas_new

        ll = np.sum(DV * np.log(p + 1e-12) + (1 - DV) * np.log(1 - p + 1e-12))
        logL.append(ll)

        it += 1
        if display:
            print(f"NR Iteration {it}: Log-likelihood = {ll:.4f}")


    if np.any(error < tol) and (it < max_iter and it > 1): converged = True
    if display:
        if converged:
            print(f"Newton-Raphson completed in {it} iterations")

        elif not converged:
            print(f"Newton-Raphson did not converge after {it} iterations. "
                  f"Final ΔDeviance = {error:.2e}")

    #return converged, betas, logL
    return OptimizeResult(x=betas.ravel(),
                          success=converged,
                          fun=-logL[-1],                    # negative log-likelihood (minimisation convention)
                          nit=it,
                          nfev=it,                          # IRLS has no separate "function evaluations"
                          message=(f"IRLS converged in {it} iterations."
                                   if converged
                                   else f"IRLS did not converge after {it} iterations."),
                          # Extra attributes for ResearchPy diagnostics
                          logL=logL,
                          deviance=error,
                          )




def IRLS(fun, x0, args=(), **options):
    """Generic Iteratively Reweighted Least Squares (IRLS) solver.

    This function conforms to the ``scipy.optimize.minimize`` custom-method
    callable signature so it can be passed directly as the *method* argument::

        from scipy.optimize import minimize
        result = minimize(fun, x0, method=IRLS, options={...})

    IRLS fits Generalized Linear Models by iteratively solving weighted
    least-squares sub-problems.  At each iteration the algorithm:

    1. Computes the current linear predictor  η = X β.
    2. Obtains the working weights  W = (dμ/dη)² / V(μ)  and the
       working (adjusted) response  z = η + (y − μ) / (dμ/dη)  from
       the supplied ``Family`` object.
    3. Solves the weighted normal equations  (X'WX) β_new = X'Wz.

    All distribution-specific logic is encapsulated in the ``Family``
    object, making this solver generic across Binomial, Poisson,
    Gaussian, and any future families.

    Parameters
    ----------
    fun : callable
        Objective function (unused by IRLS, but required by the
        ``scipy.optimize.minimize`` custom-method API).
    x0 : np.ndarray
        Initial parameter estimates, shape ``(k,)``.
    args : tuple
        Extra positional arguments (unused, kept for API conformance).
    **options : dict
        Solver options.  The following keys are recognised:

        _Family : Family
            A ``researchpy.models.families.Family`` instance that
            provides ``link_inverse``, ``variance``, ``dmu_deta``,
            ``working_weights``, ``working_response``, and
            ``log_likelihood``.  **Required.**

            .. note:: Also accepted under the key ``"family"``.
        IV : np.ndarray
            Design matrix (independent variables), shape ``(n, k)``.
            **Required.**
        DV : np.ndarray
            Response vector (dependent variable), shape ``(n, 1)``
            or ``(n,)``.  **Required.**
        maxiter : int, optional
            Maximum number of IRLS iterations (default ``100``).
        tol : float, optional
            Convergence tolerance on the change in deviance between
            successive iterations (default ``1e-8``).
        display : bool, optional
            If ``True``, print iteration-level diagnostics to stdout
            (default ``False``).

    Returns
    -------
    scipy.optimize.OptimizeResult
        A result object with the following attributes:

        - ``x`` — final parameter estimates, shape ``(k,)``.
        - ``success`` — ``True`` if the algorithm converged.
        - ``fun`` — negative log-likelihood at the solution.
        - ``nit`` — number of iterations performed.
        - ``nfev`` — same as ``nit`` (no separate function evals).
        - ``message`` — human-readable convergence summary.
        - ``logL`` — list of log-likelihood values per iteration.
        - ``deviance`` — final model deviance.

    Raises
    ------
    ValueError
        If required options (``_Family``, ``IV``, ``DV``) are missing.

    References
    ----------
    - McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear
      Models* (2nd ed.), §2.5.
    - Green, P. J. (1984). Iteratively Reweighted Least Squares for
      Maximum Likelihood Estimation, and some Robust and Resistant
      Alternatives. *JRSS-B*, 46(2), 149–192.

    Examples
    --------
    >>> from scipy.optimize import minimize
    >>> from researchpy.models.families import BinomialFamily
    >>> result = minimize(
    ...     fun=lambda p, *a: 0,   # placeholder; IRLS ignores fun
    ...     x0=np.zeros(k),
    ...     method=IRLS,
    ...     options={"family": BinomialFamily(), "IV": X, "DV": y}
    ... )
    >>> result.x   # fitted coefficients
    """
    from scipy.optimize import OptimizeResult

    # ---- Unpack required options ----
    family = options.get("family") or options.get("_Family")
    IV = options.get("IV")
    DV = options.get("DV")

    if family is None:
        raise ValueError("IRLS requires a 'family' option (a Family instance).")
    if IV is None:
        raise ValueError("IRLS requires an 'IV' option (design matrix).")
    if DV is None:
        raise ValueError("IRLS requires a 'DV' option (response vector).")


    # ---- Unpack optional settings ----
    max_iter: int = options.get("maxiter", 100)
    tol: float = options.get("tol", 1e-8)
    display: bool = options.get("display", False)


    # ---- Ensure shapes ----
    betas = np.atleast_1d(x0).astype(float).reshape(-1, 1)
    DV = np.asarray(DV).reshape(-1, 1)
    IV = np.asarray(IV)

    n, k = IV.shape
    log_likelihood_history: list[float] = []
    converged = False
    prev_deviance = np.inf

    for iteration in range(1, max_iter + 1):
        # Step 1: Linear predictor
        eta = IV @ betas


        # Step 2: Fitted values and working components from Family
        mu = family.link_inverse(eta)
        w = family.working_weights(eta)             # shape (n, 1) or (n,)
        z = family.working_response(eta, DV)        # working response

        # Ensure w is a column vector for element-wise multiplication
        w = np.asarray(w).reshape(-1, 1)


        # Step 3: Weighted normal equations  (X'WX)β = X'Wz
        #   W is diagonal, so X'WX = X' diag(w) X = (X * sqrt(w))' (X * sqrt(w))
        sqrt_w = np.sqrt(np.clip(w, 1e-15, None))
        IV_w = IV * sqrt_w                          # weighted design matrix
        z_w = z * sqrt_w                            # weighted response
        try:
            # Solve via Cholesky or least-squares for numerical stability
            betas_new, _, _, _ = np.linalg.lstsq(IV_w, z_w, rcond=None)

        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            XtWX = IV.T @ (IV * w)
            XtWz = IV.T @ (z * w)
            betas_new = np.linalg.pinv(XtWX) @ XtWz
        #betas_new = _ols_estimation_principal(IV_w, z_w)       Could replace the try-except immediately above


        # Step 4: Convergence check via deviance
        mu_new = family.link_inverse(IV @ betas_new)
        deviance = float(np.sum(family.deviance_residuals(DV, mu_new)))

        ll = family.log_likelihood(DV, mu_new)
        log_likelihood_history.append(ll)

        deviance_change = abs(prev_deviance - deviance)

        if display:
            print(f"IRLS Iteration {iteration}: "
                  f"Deviance = {deviance:.6f}, "
                  f"ΔDeviance = {deviance_change:.2e}, "
                  f"Log-likelihood = {ll:.4f}")

        betas = betas_new

        if deviance_change < tol and iteration > 1:
            converged = True
            if display:
                print(f"IRLS converged in {iteration} iterations "
                      f"(ΔDeviance = {deviance_change:.2e} < tol = {tol:.2e})")
            break

        prev_deviance = deviance

    if not converged and display:
        print(f"IRLS did not converge after {iteration} iterations. "
              f"Final ΔDeviance = {deviance_change:.2e}")


    # ---- Build scipy-compatible result ----
    final_ll = log_likelihood_history[-1] if log_likelihood_history else np.nan

    return OptimizeResult(
        x=betas.ravel(),
        success=converged,
        fun=-final_ll,                  # negative log-likelihood (minimisation convention)
        nit=iteration,
        nfev=iteration,                 # IRLS has no separate "function evaluations"
        message=(f"IRLS converged in {iteration} iterations."
                 if converged
                 else f"IRLS did not converge after {iteration} iterations."),
        # Extra attributes for ResearchPy diagnostics
        logL=log_likelihood_history,
        deviance=deviance,
    )
