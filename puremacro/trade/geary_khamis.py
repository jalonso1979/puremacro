"""Multilateral Geary-Khamis Purchasing Power Parity (PPP) and Real GDP Engine.

This module ports and vectorizes MATLAB `Geary_Khamis.m` and `Resultadosl.m` (lines 50–95).
It provides:
- :func:`restore_capital_formation`: Restores net foreign investment into Gross Fixed
  Capital Formation (Category 2) before multilateral evaluation.
- :func:`solve_multilateral_ppp`: Solves the simultaneous system for international reference
  commodity prices (pi) and purchasing power parity exchange rates (PPP) using fixed-point
  iteration or direct closed-form linear matrix inversion.
- :func:`compute_geary_khamis`: High-level orchestration function returning a complete
  :class:`~puremacro.trade._results.GearyKhamisResult`.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero non-stdlib dependencies in the runtime path, fully vectorized.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence
import numpy as np

from puremacro.trade._results import GearyKhamisResult

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def restore_capital_formation(
    c: np.ndarray,
    XN: np.ndarray,
    matlab_compat: bool = True,
) -> np.ndarray:
    """Restore foreign investment XN to Category 2 (Gross Capital Formation).

    Replicates MATLAB `Resultadosl.m:52-55`. In the open-economy CGE general equilibrium
    model, category 2 of final demand represents purely domestic investment:
    ``I_domestic = cd(2) - invforT / ppfd(2)``. Adding net foreign investment ``XN``
    restores gross national capital formation:
    ``I_national = I_domestic + XN``, ensuring that final expenditure equals national GDP.

    Parameters
    ----------
    c : np.ndarray
        Final demand consumption tensor, shape ``(nfd, nc)`` or ``(1, nfd, nc)``.
    XN : np.ndarray
        Net foreign transfers / trade surplus, shape ``(nc - 1,)`` or ``(nc,)`` or 2D column.
    matlab_compat : bool, default True
        If True, reproduces the MATLAB `Resultadosl.m:10, 50` defect where country 77 (ROW)
        receives ``XN[76] = +np.sum(XN[:76])`` for exact paper table parity.
        If False, enforces theoretical global current account balance ``XN[76] = -np.sum(XN[:76])``.

    Returns
    -------
    np.ndarray
        Adjusted final demand consumption tensor with restored capital formation,
        matching the input shape of `c`.
    """
    c_arr = np.asarray(c, dtype=float).copy()
    nc = c_arr.shape[-1]

    xn_flat = np.asarray(XN, dtype=float).ravel()
    xn_full = np.zeros(nc, dtype=float)

    if len(xn_flat) == nc - 1:
        xn_full[: nc - 1] = xn_flat
        if matlab_compat:
            xn_full[nc - 1] = float(np.sum(xn_flat))
        else:
            xn_full[nc - 1] = -float(np.sum(xn_flat))
    elif len(xn_flat) == nc:
        xn_full = xn_flat.copy()
        if matlab_compat and xn_full[nc - 1] < 0 and np.sum(xn_full[: nc - 1]) > 0:
            xn_full[nc - 1] = float(np.sum(xn_full[: nc - 1]))
        elif not matlab_compat and xn_full[nc - 1] > 0 and np.sum(xn_full[: nc - 1]) > 0:
            xn_full[nc - 1] = -float(np.sum(xn_full[: nc - 1]))
    else:
        raise ValueError(
            f"Expected XN length {nc - 1} or {nc} for {nc} countries, got {len(xn_flat)}."
        )

    # Category 2 is index 1 in 0-indexed Python
    if c_arr.ndim == 3:
        c_arr[..., 1, :] += xn_full
    elif c_arr.ndim == 2:
        c_arr[1, :] += xn_full
    else:
        raise ValueError(f"Expected c to have 2 or 3 dimensions, got ndim={c_arr.ndim}.")

    return c_arr


def solve_multilateral_ppp(
    p: np.ndarray,
    q: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
    method: str = "iterative",
    normalize: str = "none",
    numeraire_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Solve the simultaneous Geary-Khamis multilateral Purchasing Power Parity system.

    Solves the coupled equations:
        pi_m = sum_k (p_mk / PPP_k) * (q_mk / Q_m)
        PPP_k = (sum_m p_mk * q_mk) / (sum_m pi_m * q_mk)

    Parameters
    ----------
    p : np.ndarray
        Purchaser price matrix, shape ``(M, K)`` or ``(1, M, K)``.
    q : np.ndarray
        Final absorption quantity volume matrix, shape ``(M, K)`` or ``(1, M, K)``.
    tol : float, default 1e-10
        Convergence tolerance on PPP exchange rates: ``sum(|PPP_new - PPP_old|)``.
    max_iter : int, default 1000
        Maximum iterations for iterative contraction mapping.
    method : {"iterative", "linear"}, default "iterative"
        Computational method:
        - 'iterative': Fixed-point contraction iteration initialized at ``PPP^{(0)} = 1``.
        - 'linear': Direct closed-form ``M x M`` linear system solve ``(I - A) pi = 0``
          with world expenditure normalization constraint.
    normalize : {"none", "pi1", "ppp_usa"}, default "none"
        Scale normalization:
        - 'none': Preserves natural unit of account pinned down by ``PPP^{(0)} = 1``
          (exact paper tables parity).
        - 'pi1' or 'pi_1': Normalizes reference price of category 1 to 1.0.
        - 'ppp_usa' or 'numeraire': Normalizes PPP exchange rate of United States to 1.0.
    numeraire_idx : int, optional
        Explicit country index to normalize to 1.0 when ``normalize='ppp_usa'`` or
        ``normalize='numeraire'``. Defaults to 73 if ``nc >= 74`` (matching standard WIOD 77-country layout),
        or ``nc - 1`` otherwise.

    Returns
    -------
    pi : np.ndarray, shape (M,)
        International reference commodity prices.
    ppp : np.ndarray, shape (K,)
        Purchasing power parity exchange rates per country.
    converged : bool
        Whether the solver attained convergence within `tol`.
    iterations : int
        Number of iterations executed.
    """
    p_mat = np.asarray(p, dtype=float)
    q_mat = np.asarray(q, dtype=float)

    if p_mat.ndim == 3 and p_mat.shape[0] == 1:
        p_mat = p_mat[0]
    if q_mat.ndim == 3 and q_mat.shape[0] == 1:
        q_mat = q_mat[0]

    if p_mat.shape != q_mat.shape:
        raise ValueError(
            f"Shape mismatch between prices {p_mat.shape} and quantities {q_mat.shape}."
        )

    nm, nc = p_mat.shape
    q_world = np.sum(q_mat, axis=1)  # Shape (M,)
    expenditure = np.sum(p_mat * q_mat, axis=0)  # Shape (K,)

    if np.any(q_world <= 0):
        raise ValueError("World consumption volume must be strictly positive for all categories.")

    method_str = method.lower().strip()
    converged = False
    it_count = 0

    if method_str == "iterative":
        share = q_mat / q_world[:, np.newaxis]  # (M, K)
        ppp0 = np.ones(nc, dtype=float)
        ppp = ppp0.copy()
        pi = np.ones(nm, dtype=float)

        for it in range(1, max_iter + 1):
            it_count = it
            pi = np.sum(p_mat * share / ppp0[np.newaxis, :], axis=1)
            denom = np.sum(q_mat * pi[:, np.newaxis], axis=0)
            ppp = np.where(denom > 0, expenditure / denom, 1.0)

            diff = float(np.sum(np.abs(ppp - ppp0)))
            if diff < tol:
                converged = True
                break
            ppp0 = ppp.copy()

        if not converged:
            pi = np.sum(p_mat * share / ppp0[np.newaxis, :], axis=1)
            ppp = ppp0

    elif method_str == "linear":
        term1 = (p_mat * q_mat) / q_world[:, np.newaxis]  # (M, K)
        term2 = q_mat / expenditure[np.newaxis, :]  # (M, K)
        a_mat = term1 @ term2.T  # (M, M)

        m_sys = np.eye(nm, dtype=float) - a_mat
        m_sys[-1, :] = q_world
        b = np.zeros(nm, dtype=float)
        b[-1] = np.sum(expenditure)

        pi = np.linalg.solve(m_sys, b)
        ppp = expenditure / np.sum(q_mat * pi[:, np.newaxis], axis=0)
        converged = True
        it_count = 1

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Valid options are 'iterative' and 'linear'."
        )

    # Post-hoc scale normalization
    norm_str = str(normalize).lower().strip()
    if norm_str in ("pi1", "pi_1"):
        scale = float(pi[0])
        if scale != 0.0:
            pi = pi / scale
            ppp = ppp * scale
    elif norm_str in ("ppp_usa", "usa", "numeraire"):
        if numeraire_idx is not None:
            target_idx = int(numeraire_idx)
            if target_idx < 0 or target_idx >= nc:
                raise IndexError(f"numeraire_idx {target_idx} is out of bounds for {nc} countries.")
        else:
            target_idx = 73 if nc >= 74 else (nc - 1)
        scale = float(ppp[target_idx])
        if scale != 0.0:
            ppp = ppp / scale
            pi = pi * scale
    elif norm_str in ("none", "default", ""):
        pass
    else:
        raise ValueError(
            f"Unsupported normalize option '{normalize}'. Valid options: 'none', 'pi1', 'ppp_usa'."
        )

    return pi, ppp, converged, it_count


def compute_geary_khamis(
    equilibrium_sol: Any,
    base_sol: Any = None,
    calib: TradeCalibrationResult | None = None,
    method: str = "iterative",
    matlab_compat: bool = True,
    normalize: str = "none",
    numeraire_idx: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> GearyKhamisResult:
    """Compute multilateral Geary-Khamis Purchasing Power Parity and real GDP welfare.

    Supports both high-level :class:`~puremacro.trade._results.TradeEquilibriumResult` inputs
    and raw array inputs ``(prices, quantities)``.

    Parameters
    ----------
    equilibrium_sol : TradeEquilibriumResult or np.ndarray
        Counterfactual equilibrium solution (or purchaser price matrix).
    base_sol : TradeEquilibriumResult or np.ndarray, optional
        Baseline equilibrium solution (or baseline quantity matrix) for growth calculation.
    calib : TradeCalibrationResult, optional
        Calibrated model parameters for metadata and country codes.
    method : {"iterative", "linear"}, default "iterative"
        Numerical solution method.
    matlab_compat : bool, default True
        Whether to replicate MATLAB post-processing sign convention for ROW.
    normalize : {"none", "pi1", "ppp_usa"}, default "none"
        Scale normalization strategy.
    numeraire_idx : int, optional
        Explicit country index for PPP normalization. If None and country codes are
        present, automatically resolves 'USA' or 'US'; otherwise defaults to 73 (if nc >= 74) or 0.
    tol : float, default 1e-10
        Stopping tolerance.
    max_iter : int, default 1000
        Maximum iterations.

    Returns
    -------
    GearyKhamisResult
        Complete Geary-Khamis multilateral parity solution container.
    """
    # Case A: Input is raw arrays (prices, quantities)
    if isinstance(equilibrium_sol, np.ndarray) and isinstance(base_sol, np.ndarray) and not hasattr(base_sol, "p_fd"):
        p = equilibrium_sol
        q = base_sol
        base_quantities = None
        c_codes: tuple[str, ...] = ()
        cat_codes: tuple[str, ...] = ()
        if calib is not None:
            c_codes = tuple(calib.country_codes)
    # Case B: Input is TradeEquilibriumResult
    else:
        sol = equilibrium_sol
        p_raw = getattr(sol, "p_fd", None)
        if p_raw is None:
            p_raw = getattr(sol, "pfd_sol", None)

        c_raw = getattr(sol, "c_fd", None)
        if c_raw is None:
            c_raw = getattr(sol, "c_sol", None)

        if p_raw is None or c_raw is None:
            raise ValueError(
                "equilibrium_sol must provide final demand prices (p_fd/pfd_sol) and quantities (c_fd/c_sol)."
            )

        xn = getattr(sol, "XN_sol", None)
        if xn is None:
            raise ValueError("equilibrium_sol must provide XN_sol.")

        p = np.asarray(p_raw, dtype=float)
        q = restore_capital_formation(c_raw, xn, matlab_compat=matlab_compat)

        base_quantities = None
        if base_sol is not None:
            c_base_raw = getattr(base_sol, "c_fd", None)
            if c_base_raw is None:
                c_base_raw = getattr(base_sol, "c_sol", None)
            xn_base = getattr(base_sol, "XN_sol", None)
            if c_base_raw is not None and xn_base is not None:
                base_quantities = restore_capital_formation(
                    c_base_raw, xn_base, matlab_compat=matlab_compat
                )
            elif isinstance(base_sol, np.ndarray):
                base_quantities = np.asarray(base_sol, dtype=float)

        c_codes = tuple(getattr(sol, "country_codes", ()))
        if not c_codes and calib is not None:
            c_codes = tuple(calib.country_codes)
        cat_codes = ("Consumption", "Investment", "DirectPurchasesAbroad")

    resolved_numeraire_idx = numeraire_idx
    if resolved_numeraire_idx is None and c_codes:
        for idx_c, code in enumerate(c_codes):
            if str(code).upper() in ("USA", "US"):
                resolved_numeraire_idx = idx_c
                break

    # Solve multilateral PPP
    pi, ppp, converged, iterations = solve_multilateral_ppp(
        p=p,
        q=q,
        tol=tol,
        max_iter=max_iter,
        method=method,
        normalize=normalize,
        numeraire_idx=resolved_numeraire_idx,
    )

    p_2d = p[0] if p.ndim == 3 else p
    q_2d = q[0] if q.ndim == 3 else q

    nominal_gdp = np.sum(p_2d * q_2d, axis=0)
    real_gdp = np.sum(q_2d * pi[:, np.newaxis], axis=0)
    world_nominal_gdp = float(np.sum(nominal_gdp))
    world_real_gdp = float(np.sum(real_gdp))

    gdp_growth = None
    if base_quantities is not None:
        qb_2d = base_quantities[0] if base_quantities.ndim == 3 else base_quantities
        gdp_base = np.sum(qb_2d, axis=0)
        gdp_growth = ((real_gdp / gdp_base) - 1.0) * 100.0

    if not c_codes:
        c_codes = tuple(f"C{i:02d}" for i in range(len(ppp)))
    if not cat_codes:
        cat_codes = tuple(f"Cat{m:02d}" for m in range(len(pi)))

    return GearyKhamisResult(
        pi=pi,
        ppp=ppp,
        real_gdp=real_gdp,
        nominal_gdp=nominal_gdp,
        world_real_gdp=world_real_gdp,
        world_nominal_gdp=world_nominal_gdp,
        gdp_growth=gdp_growth,
        real_gdp_growth=gdp_growth,
        country_codes=c_codes,
        category_codes=cat_codes,
        converged=converged,
        iterations=iterations,
        tolerance=tol,
        method=method,
        matlab_compat=matlab_compat,
        metadata={
            "normalize": normalize,
            "world_nominal_gdp": world_nominal_gdp,
            "world_real_gdp": world_real_gdp,
        },
    )


def compute_geary_khamis_ppp(
    equilibrium_sol: Any = None,
    base_sol: Any = None,
    *,
    pfd_sol: np.ndarray | None = None,
    c_sol: np.ndarray | None = None,
    c_base: np.ndarray | None = None,
    base_xn: np.ndarray | None = None,
    scen_xn: np.ndarray | None = None,
    country_codes: tuple[str, ...] | Sequence[str] = (),
    method: str = "iterative",
    matlab_compat: bool = True,
    normalize: str = "none",
    tol: float = 1e-10,
    max_iter: int = 1000,
    **kwargs: Any,
) -> GearyKhamisResult:
    """Convenience and backwards-compatibility wrapper for :func:`compute_geary_khamis`."""
    if equilibrium_sol is not None:
        return compute_geary_khamis(
            equilibrium_sol=equilibrium_sol,
            base_sol=base_sol,
            method=method,
            matlab_compat=matlab_compat,
            normalize=normalize,
            tol=tol,
            max_iter=max_iter,
        )

    if pfd_sol is None or c_sol is None:
        raise ValueError("Either equilibrium_sol or both pfd_sol and c_sol must be provided.")

    p = np.asarray(pfd_sol, dtype=float)
    q = np.asarray(c_sol, dtype=float)

    if scen_xn is not None:
        q = restore_capital_formation(q, scen_xn, matlab_compat=matlab_compat)

    qb = None
    if c_base is not None:
        qb = np.asarray(c_base, dtype=float)
        if base_xn is not None:
            qb = restore_capital_formation(qb, base_xn, matlab_compat=matlab_compat)

    p_2d = p[0] if p.ndim == 3 else p
    q_2d = q[0] if q.ndim == 3 else q

    pi, ppp, converged, iterations = solve_multilateral_ppp(
        p=p_2d,
        q=q_2d,
        tol=tol,
        max_iter=max_iter,
        method=method,
        normalize=normalize,
    )

    nominal_gdp = np.sum(p_2d * q_2d, axis=0)
    real_gdp = np.sum(q_2d * pi[:, np.newaxis], axis=0)
    world_nominal_gdp = float(np.sum(nominal_gdp))
    world_real_gdp = float(np.sum(real_gdp))

    gdp_growth = None
    if qb is not None:
        qb_2d = qb[0] if qb.ndim == 3 else qb
        gdp_base = np.sum(qb_2d, axis=0)
        gdp_growth = ((real_gdp / gdp_base) - 1.0) * 100.0

    c_tup = tuple(country_codes) if country_codes else tuple(f"C{i:02d}" for i in range(len(ppp)))
    cat_tup = ("Consumption", "Investment", "DirectPurchasesAbroad")

    return GearyKhamisResult(
        pi=pi,
        ppp=ppp,
        real_gdp=real_gdp,
        nominal_gdp=nominal_gdp,
        world_real_gdp=world_real_gdp,
        world_nominal_gdp=world_nominal_gdp,
        gdp_growth=gdp_growth,
        real_gdp_growth=gdp_growth,
        country_codes=c_tup,
        category_codes=cat_tup,
        converged=converged,
        iterations=iterations,
        tolerance=tol,
        method=method,
        matlab_compat=matlab_compat,
        metadata={
            "normalize": normalize,
            "world_nominal_gdp": world_nominal_gdp,
            "world_real_gdp": world_real_gdp,
        },
    )


__all__ = [
    "restore_capital_formation",
    "solve_multilateral_ppp",
    "compute_geary_khamis",
    "compute_geary_khamis_ppp",
]
