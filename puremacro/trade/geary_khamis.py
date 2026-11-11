"""Multilateral Geary-Khamis Purchasing Power Parity (PPP) and Real GDP Engine.

Geary (1958) and Khamis (1972). The module provides:
- :func:`restore_capital_formation`: Restores net foreign investment into gross capital
  formation (final-use category 2, the investment category) before multilateral evaluation.
- :func:`solve_multilateral_ppp`: Solves the simultaneous system for international reference
  commodity prices (pi) and purchasing power parity exchange rates (PPP) using fixed-point
  iteration or direct closed-form linear matrix inversion.
- :func:`compute_geary_khamis`: High-level orchestration function returning a complete
  :class:`~puremacro.trade._results.GearyKhamisResult`.

Pure NumPy, vectorised.
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
    p_fd: np.ndarray | None = None,
) -> np.ndarray:
    """Restore foreign investment XN to the investment category.

    In the open-economy model the investment category (index 1; index 0 when
    there is a single final-use category) holds purely domestic investment:
    ``I_domestic = cd(2) - invforT / P_fd(2)``. Adding net foreign investment
    deflated by the investment-category purchaser price ``XN / P_fd(2)``
    (or nominal ``XN`` when ``p_fd`` is omitted at unit baseline prices)
    restores gross national capital formation so that ``sum_m P_fd(m) * q(m)``
    equals national GDP identically and real quantities remain homogeneous of
    degree zero in nominal prices and incomes.

    Parameters
    ----------
    c : np.ndarray
        Final demand consumption tensor, shape ``(nfd, nc)`` or ``(1, nfd, nc)``.
    XN : np.ndarray
        Net foreign balances. With ``nc - 1`` entries the last country's balance
        is ``-sum(XN)`` (balances add to zero); with ``nc`` entries they are used
        as given.
    p_fd : np.ndarray, optional
        Final-demand purchaser price tensor, shape ``(nfd, nc)`` or ``(1, nfd, nc)``.
        When provided, ``XN`` is deflated by ``p_fd[..., investment, :]`` into
        real investment-category quantity units. When ``None`` (default), ``XN``
        is added directly (unit-price baseline convention).

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
        xn_full[nc - 1] = -float(np.sum(xn_flat))
    elif len(xn_flat) == nc:
        xn_full = xn_flat.copy()
    else:
        raise ValueError(
            f"Expected XN length {nc - 1} or {nc} for {nc} countries, got {len(xn_flat)}."
        )

    if c_arr.ndim not in (2, 3):
        raise ValueError(f"Expected c to have 2 or 3 dimensions, got ndim={c_arr.ndim}.")
    investment = 1 if c_arr.shape[-2] >= 2 else 0
    if p_fd is None:
        c_arr[..., investment, :] += xn_full
    else:
        p_arr = np.asarray(p_fd, dtype=float)
        p_2d = p_arr[0] if (p_arr.ndim == 3 and p_arr.shape[0] == 1) else p_arr
        expected_shape = (c_arr.shape[-2], nc)
        if p_2d.shape != expected_shape:
            raise ValueError(
                f"Expected p_fd shape {expected_shape} (or (1, {expected_shape[0]}, {nc})), got {p_arr.shape}."
            )
        p_inv = p_2d[investment, :]
        if not np.all(np.isfinite(p_inv)) or np.any(p_inv <= 0.0):
            raise ValueError("Investment-category purchaser prices p_fd must be finite and strictly positive.")
        c_arr[..., investment, :] += xn_full / p_inv

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
    normalize : {"none", "pi1", "numeraire"}, default "none"
        Scale normalization:
        - 'none': Keeps the unit of account pinned down by ``PPP^{(0)} = 1``.
        - 'pi1' or 'pi_1': Normalizes reference price of category 1 to 1.0.
        - 'numeraire': Normalizes the PPP of country ``numeraire_idx`` to 1.0.
    numeraire_idx : int, optional
        Country index whose PPP is set to 1.0; required with ``normalize='numeraire'``.

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
    elif norm_str == "numeraire":
        if numeraire_idx is None:
            raise ValueError("normalize='numeraire' needs numeraire_idx (the numeraire country's index)")
        target_idx = int(numeraire_idx)
        if target_idx < 0 or target_idx >= nc:
            raise IndexError(f"numeraire_idx {target_idx} is out of bounds for {nc} countries.")
        scale = float(ppp[target_idx])
        if scale != 0.0:
            ppp = ppp / scale
            pi = pi * scale
    elif norm_str in ("none", "default", ""):
        pass
    else:
        raise ValueError(
            f"Unsupported normalize option '{normalize}'. Valid options: 'none', 'pi1', 'numeraire'."
        )

    return pi, ppp, converged, it_count


def _numeraire_index(numeraire: str | None, country_codes: Sequence[str]) -> int | None:
    """Position of the numeraire country code, or None when no numeraire is requested."""
    if numeraire is None:
        return None
    if not isinstance(numeraire, str):
        raise TypeError("numeraire must be a country code (str)")
    codes = [str(c) for c in country_codes]
    if numeraire not in codes:
        raise ValueError(f"Unknown numeraire country code {numeraire!r}; known codes: {codes}")
    return codes.index(numeraire)


def _geary_khamis_result(
    p: np.ndarray,
    q: np.ndarray,
    base_quantities: np.ndarray | None,
    country_codes: tuple[str, ...],
    category_codes: tuple[str, ...],
    *,
    method: str,
    numeraire: str | None,
    tol: float,
    max_iter: int,
) -> GearyKhamisResult:
    p_2d = p[0] if p.ndim == 3 else p
    q_2d = q[0] if q.ndim == 3 else q
    if not country_codes:
        country_codes = tuple(f"C{i:02d}" for i in range(p_2d.shape[1]))
    numeraire_idx = _numeraire_index(numeraire, country_codes)
    pi, ppp, converged, iterations = solve_multilateral_ppp(
        p=p_2d,
        q=q_2d,
        tol=tol,
        max_iter=max_iter,
        method=method,
        normalize="none" if numeraire_idx is None else "numeraire",
        numeraire_idx=numeraire_idx,
    )

    nominal_gdp = np.sum(p_2d * q_2d, axis=0)
    real_gdp = np.sum(q_2d * pi[:, np.newaxis], axis=0)
    world_nominal_gdp = float(np.sum(nominal_gdp))
    world_real_gdp = float(np.sum(real_gdp))

    gdp_growth = None
    if base_quantities is not None:
        qb_2d = base_quantities[0] if base_quantities.ndim == 3 else base_quantities
        # Value the baseline quantities at the same international prices pi as
        # the counterfactual ones, so growth is a quantity index: invariant to
        # the numeraire (which rescales pi by a common factor) and zero when the
        # quantities do not change.
        gdp_base = np.sum(qb_2d * pi[:, np.newaxis], axis=0)
        gdp_growth = ((real_gdp / gdp_base) - 1.0) * 100.0

    if not category_codes:
        category_codes = tuple(f"Cat{m:02d}" for m in range(len(pi)))

    return GearyKhamisResult(
        pi=pi,
        ppp=ppp,
        real_gdp=real_gdp,
        nominal_gdp=nominal_gdp,
        world_real_gdp=world_real_gdp,
        world_nominal_gdp=world_nominal_gdp,
        gdp_growth=gdp_growth,
        real_gdp_growth=gdp_growth,
        country_codes=tuple(country_codes),
        category_codes=tuple(category_codes),
        converged=converged,
        iterations=iterations,
        tolerance=tol,
        method=method,
        metadata={
            "numeraire": numeraire,
            "world_nominal_gdp": world_nominal_gdp,
            "world_real_gdp": world_real_gdp,
        },
    )


def _category_codes(nfd: int) -> tuple[str, ...]:
    if nfd == 3:
        return ("Consumption", "Investment", "DirectPurchasesAbroad")
    return tuple(f"Cat{m:02d}" for m in range(nfd))


def compute_geary_khamis(
    equilibrium_sol: Any,
    base_sol: Any = None,
    calib: TradeCalibrationResult | None = None,
    method: str = "iterative",
    *,
    numeraire: str | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> GearyKhamisResult:
    """Compute multilateral Geary-Khamis Purchasing Power Parity and real GDP welfare.

    Supports both :class:`~puremacro.trade._results.TradeEquilibriumResult` inputs
    and raw array inputs ``(prices, quantities)``.

    Parameters
    ----------
    equilibrium_sol : TradeEquilibriumResult or np.ndarray
        Counterfactual equilibrium solution (or purchaser price matrix).
    base_sol : TradeEquilibriumResult or np.ndarray, optional
        Baseline equilibrium solution (or, with an array ``equilibrium_sol``,
        the quantity matrix).
    calib : TradeCalibrationResult, optional
        Calibrated model parameters, used for the country codes.
    method : {"iterative", "linear"}, default "iterative"
        Numerical solution method.
    numeraire : str, optional (keyword-only)
        Country code whose PPP is normalised to 1.0. None (default) keeps the
        unit of account fixed by the iteration's ``PPP = 1`` start; an unknown
        code raises ``ValueError``.
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
        p = np.asarray(equilibrium_sol, dtype=float)
        q = np.asarray(base_sol, dtype=float)
        c_codes: tuple[str, ...] = tuple(calib.country_codes) if calib is not None else ()
        return _geary_khamis_result(p, q, None, c_codes, (), method=method, numeraire=numeraire,
                                    tol=tol, max_iter=max_iter)

    # Case B: Input is TradeEquilibriumResult
    sol = equilibrium_sol
    p_raw = getattr(sol, "Pfd_final", None)
    if p_raw is None:
        p_raw = getattr(sol, "p_fd", None)
    if p_raw is None:
        p_raw = getattr(sol, "pfd_sol", None)

    c_raw = getattr(sol, "c_fd", None)
    if c_raw is None:
        c_raw = getattr(sol, "c_sol", None)

    if p_raw is None or c_raw is None:
        raise ValueError(
            "equilibrium_sol must provide final demand prices (Pfd_final/p_fd/pfd_sol) and quantities (c_fd/c_sol)."
        )

    xn = getattr(sol, "XN_sol", None)
    if xn is None:
        raise ValueError("equilibrium_sol must provide XN_sol.")

    p = np.asarray(p_raw, dtype=float)
    q = restore_capital_formation(c_raw, xn, p_fd=p)

    base_quantities = None
    if base_sol is not None:
        c_base_raw = getattr(base_sol, "c_fd", None)
        if c_base_raw is None:
            c_base_raw = getattr(base_sol, "c_sol", None)
        xn_base = getattr(base_sol, "XN_sol", None)
        if c_base_raw is not None and xn_base is not None:
            p_base_raw = getattr(base_sol, "Pfd_final", None)
            if p_base_raw is None:
                p_base_raw = getattr(base_sol, "p_fd", None)
            if p_base_raw is None:
                p_base_raw = getattr(base_sol, "pfd_sol", None)
            p_base_use = np.asarray(p_base_raw, dtype=float) if p_base_raw is not None else None
            base_quantities = restore_capital_formation(c_base_raw, xn_base, p_fd=p_base_use)
        elif isinstance(base_sol, np.ndarray):
            base_quantities = np.asarray(base_sol, dtype=float)

    c_codes = tuple(getattr(sol, "country_codes", ()) or ())
    if not c_codes and calib is not None:
        c_codes = tuple(calib.country_codes)
    return _geary_khamis_result(p, q, base_quantities, c_codes, _category_codes(p.shape[-2]),
                                method=method, numeraire=numeraire, tol=tol, max_iter=max_iter)


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
    numeraire: str | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> GearyKhamisResult:
    """Geary-Khamis PPP from results or from raw final-demand prices and quantities.

    With ``equilibrium_sol`` this is :func:`compute_geary_khamis`. Otherwise
    ``pfd_sol`` and ``c_sol`` (shape ``(nfd, nc)`` or ``(1, nfd, nc)``) are the
    purchaser prices and quantities; ``scen_xn``/``base_xn`` restore foreign
    investment (:func:`restore_capital_formation`) and ``c_base`` gives the
    baseline quantities for real GDP growth. ``numeraire`` is a country code
    resolved in ``country_codes``.
    """
    if equilibrium_sol is not None:
        return compute_geary_khamis(
            equilibrium_sol=equilibrium_sol,
            base_sol=base_sol,
            method=method,
            numeraire=numeraire,
            tol=tol,
            max_iter=max_iter,
        )

    if pfd_sol is None or c_sol is None:
        raise ValueError("Either equilibrium_sol or both pfd_sol and c_sol must be provided.")

    p = np.asarray(pfd_sol, dtype=float)
    q = np.asarray(c_sol, dtype=float)

    if scen_xn is not None:
        q = restore_capital_formation(q, scen_xn)

    qb = None
    if c_base is not None:
        qb = np.asarray(c_base, dtype=float)
        if base_xn is not None:
            qb = restore_capital_formation(qb, base_xn)

    return _geary_khamis_result(p, q, qb, tuple(str(c) for c in country_codes), _category_codes(p.shape[-2]),
                                method=method, numeraire=numeraire, tol=tol, max_iter=max_iter)


__all__ = [
    "restore_capital_formation",
    "solve_multilateral_ppp",
    "compute_geary_khamis",
    "compute_geary_khamis_ppp",
]
