"""Nonlinear General Equilibrium Formulation for puremacro.trade.

Vectorized equilibrium residual evaluation porting MATLAB `ff_equi.m` (lines 1–150).
Evaluates the 2,001-equation general equilibrium residual system F(x) = 0 across
goods market clearing, zero-profit pricing, labor and capital factor market clearing,
multilateral trade balance, and national fiscal budget consistency.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy,
zero dev-dependency imports, fully vectorized.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple
import numpy as np

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult


class EquilibriumVariables(NamedTuple):
    """Unpacked economic variables from the general equilibrium state vector."""

    p: np.ndarray        # Gross output prices, shape (1, ns, nc)
    y: np.ndarray        # Gross output quantities, shape (1, ns, nc)
    r: np.ndarray        # Capital rental rates, shape (1, 1, nc)
    w: np.ndarray        # Wage rates, shape (1, 1, nc)
    T: np.ndarray        # Fiscal revenues / transfers (in levels), shape (1, 1, nc)
    XN: np.ndarray       # Net foreign transfers (in levels), shape (nc - 1,)
    invforT: np.ndarray  # Complete foreign current accounts with global closure, shape (1, nc)


def unpack_equilibrium_vector(
    x: np.ndarray,
    ns: int = 11,
    nc: int = 77,
    nfd: int = 3,
    return_levels: bool = True,
) -> EquilibriumVariables:
    """Unpack state vector x into component economic tensors.

    Parameters
    ----------
    x : np.ndarray
        1D array of length 2*ns*nc + 3*nc + (nc - 1), representing
        [log(p); log(y); log(r); log(w); T; XN].
    ns : int, default 11
        Number of industrial sectors.
    nc : int, default 77
        Number of countries.
    nfd : int, default 3
        Number of final demand categories.
    return_levels : bool, default True
        If True, exponentiates log-prices, log-outputs, and log-factor-returns.

    Returns
    -------
    EquilibriumVariables
        NamedTuple with unpacked fields p, y, r, w, T, XN, and invforT.
    """
    x_flat = np.asarray(x, dtype=float).ravel()
    expected_len = 2 * ns * nc + 3 * nc + (nc - 1)
    if len(x_flat) != expected_len:
        raise ValueError(
            f"State vector length {len(x_flat)} does not match expected {expected_len} "
            f"for ns={ns}, nc={nc}, nfd={nfd}."
        )

    i_p_end = ns * nc
    i_y_end = 2 * ns * nc
    i_r_end = 2 * ns * nc + nc
    i_w_end = 2 * ns * nc + 2 * nc
    i_T_end = 2 * ns * nc + 3 * nc
    i_XN_end = 2 * ns * nc + 4 * nc - 1

    p_raw = x_flat[:i_p_end]
    y_raw = x_flat[i_p_end:i_y_end]
    r_raw = x_flat[i_y_end:i_r_end]
    w_raw = x_flat[i_r_end:i_w_end]
    T_raw = x_flat[i_w_end:i_T_end]
    XN_raw = x_flat[i_T_end:i_XN_end]

    if return_levels:
        p = np.exp(p_raw).reshape((1, ns, nc), order="F")
        y = np.exp(y_raw).reshape((1, ns, nc), order="F")
        r = np.exp(r_raw).reshape((1, 1, nc), order="F")
        w = np.exp(w_raw).reshape((1, 1, nc), order="F")
    else:
        p = p_raw.reshape((1, ns, nc), order="F")
        y = y_raw.reshape((1, ns, nc), order="F")
        r = r_raw.reshape((1, 1, nc), order="F")
        w = w_raw.reshape((1, 1, nc), order="F")

    T = T_raw.reshape((1, 1, nc), order="F")
    XN = XN_raw.copy()

    # Global trade balance closure: sum(invforT) == 0 identity
    # Country nc (index nc - 1) is determined residually
    invforT = np.zeros((1, nc), dtype=float)
    invforT[0, :nc - 1] = XN
    invforT[0, nc - 1] = -np.sum(XN)

    return EquilibriumVariables(p=p, y=y, r=r, w=w, T=T, XN=XN, invforT=invforT)


def pack_equilibrium_vector(
    p: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    XN: np.ndarray,
    log_transformed: bool = True,
) -> np.ndarray:
    """Pack economic tensors into the canonical 1D general equilibrium state vector.

    Packs variables in MATLAB `Main77c_11s.m` order:
    ``x = [log(p); log(y); log(r); log(w); T; XN]``.

    Parameters
    ----------
    p : np.ndarray
        Gross output prices, shape (1, ns, nc) or (ns*nc,).
    y : np.ndarray
        Gross output quantities, shape (1, ns, nc) or (ns*nc,).
    r : np.ndarray
        Capital rental rates, shape (1, 1, nc) or (nc,).
    w : np.ndarray
        Wage rates, shape (1, 1, nc) or (nc,).
    T : np.ndarray
        Government fiscal revenues/transfers (in levels), shape (1, 1, nc) or (nc,).
    XN : np.ndarray
        Net foreign transfers (in levels), shape (nc - 1,).
    log_transformed : bool, default True
        If True, applies np.log to p, y, r, and w. If False, assumes they are already logs.

    Returns
    -------
    np.ndarray
        1D state vector of shape (2*ns*nc + 3*nc + nc - 1,).
    """
    p_flat = np.asarray(p, dtype=float).flatten(order="F")
    y_flat = np.asarray(y, dtype=float).flatten(order="F")
    r_flat = np.asarray(r, dtype=float).flatten(order="F")
    w_flat = np.asarray(w, dtype=float).flatten(order="F")
    T_flat = np.asarray(T, dtype=float).flatten(order="F")
    XN_flat = np.asarray(XN, dtype=float).ravel()

    if log_transformed:
        p_part = np.log(p_flat)
        y_part = np.log(y_flat)
        r_part = np.log(r_flat)
        w_part = np.log(w_flat)
    else:
        p_part = p_flat
        y_part = y_flat
        r_part = r_flat
        w_part = w_flat

    return np.concatenate([p_part, y_part, r_part, w_part, T_flat, XN_flat])


def _get_ces_weights(calib: TradeCalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """Precompute and cache CES aggregation weights on the calibration instance to optimize evaluations."""
    cached = getattr(calib, "_ces_weights_cache", None)
    if cached is not None:
        return cached

    A_mat = np.sum(calib.a, axis=0)  # shape (ns, nc)
    A_mat_safe = np.where(A_mat > 0, A_mat, 1.0)
    omega = calib.a / A_mat_safe[np.newaxis, :, :]  # shape (ns*nc, ns, nc)
    cached_val = (A_mat, omega)
    try:
        object.__setattr__(calib, "_ces_weights_cache", cached_val)
    except Exception:
        pass
    return cached_val


def compute_equilibrium_residuals(
    x: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
    *,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    sigma: float = 0.0,
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
    **kwargs: Any,
) -> np.ndarray:
    """Vectorized evaluation of all 2,001 general equilibrium residual equations.

    Evaluates the system F(x) = 0 across:
    - Block 0: Goods market clearing (ns*nc equations)
    - Block 1: Zero-profit price condition (ns*nc equations)
    - Block 2: Labor market clearing (nc equations)
    - Block 3: Capital market clearing (nc equations)
    - Block 4: Multilateral trade balance / current account consistency (nc - 1 equations)
    - Block 5: Fiscal budget consistency (nc equations)

    Parameters
    ----------
    x : np.ndarray
        State vector of length 2*ns*nc + 3*nc + (nc - 1).
    calib : TradeCalibrationResult
        Calibrated model parameters from :func:`calibrate_trade_model`.
    tau : np.ndarray, optional
        Intermediate tariff multipliers (1 + tariff_rate).
        Shape (ns*nc, ns, nc), (ns, nc, ns, nc), or None (all ones).
    tau_fd : np.ndarray, optional
        Final demand tariff multipliers (1 + tariff_rate).
        Shape (ns*nc, nfd, nc), (ns, nc, nfd, nc), or None (all ones).
    tauf : np.ndarray, optional
        National intermediate tariff rates by importing country, shape (nc,) or (1, nc).
        Defaults to zeros if not provided.
    tauf_fd : np.ndarray, optional
        National final demand tariff rates by importing country, shape (nc,) or (1, nc).
        Defaults to zeros if not provided.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31
        ``w / (1 - alpha)^(1 - alpha)`` where ``^`` binds before ``/``.
        Achieves < 10^-14 precision against reference MATLAB solutions.
    sigma : float, default 0.0
        Intermediate sourcing elasticity of substitution (0.0 = pure Leontief).
    fiscal_closure : str, default "lump_sum"
        Fiscal regime for tariff revenues:
        - 'lump_sum': returned 100% to domestic households
        - 'capital_tax': recycled to reduce capital factor tax
        - 'labor_tax': recycled to reduce payroll/labor tax
        - 'strategic_subsidy': targeted intermediate manufacturing output subsidy
        - 'deficit_reduction': retained to improve fiscal surplus / debt retirement
    recycling_params : dict[str, Any], optional
        Optional parameters governing fiscal recycling targets and sectoral shares.
    capacity_margins : dict or np.ndarray, optional
        Spare capacity margins (kappa) on constrained upstream sectors.
    capacity_target_country : str, default "USA"
        Target country code for capacity constraints.
    penalty_scale : float, default 0.05
        Scale parameter zeta for C^2 smooth barrier penalty function.
    penalty_exponent : float, default 8.0
        Exponent eta for C^2 smooth barrier penalty function.

    Returns
    -------
    np.ndarray
        1D residual vector of length 2*ns*nc + 3*nc + (nc - 1).
    """
    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand

    # Expand tariff matrices if needed
    if tau_a is not None:
        tau = tau_a
    if taufd_a is not None:
        tau_fd = taufd_a

    if tau is None:
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
    elif tau.shape == (ns * nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float)
    elif tau.ndim == 4 and tau.shape == (ns, nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    elif tau.ndim == 4 and tau.shape == (11, 77, 11, 77):
        tau_a = np.asarray(tau, dtype=float).transpose(1, 0, 3, 2).reshape(ns * nc, ns, nc, order="F")
    else:
        tau_a = np.asarray(tau, dtype=float)

    if tau_fd is None:
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float)
    elif tau_fd.shape == (ns * nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float)
    elif tau_fd.ndim == 4 and tau_fd.shape == (ns, nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    else:
        taufd_a = np.asarray(tau_fd, dtype=float)

    tauf_vec = np.zeros(nc, dtype=float) if tauf is None else np.asarray(tauf, dtype=float).ravel()
    tauf_fd_vec = np.zeros(nc, dtype=float) if tauf_fd is None else np.asarray(tauf_fd, dtype=float).ravel()

    # 1. Unpack state vector
    vars_eq = unpack_equilibrium_vector(x, ns=ns, nc=nc, nfd=nfd, return_levels=True)
    p, ytot, r, w, T, XN, invforT = (
        vars_eq.p,
        vars_eq.y,
        vars_eq.r,
        vars_eq.w,
        vars_eq.T,
        vars_eq.XN,
        vars_eq.invforT,
    )
    p_vec = p.flatten(order="F")

    # Dynamic CES or Leontief intermediate unit cost and demand
    if sigma > 0.0:
        A_mat, omega = _get_ces_weights(calib)
        p_tau = p_vec[:, np.newaxis, np.newaxis] * tau_a
        p_tau_safe = np.maximum(p_tau, 1e-12)

        if abs(sigma - 1.0) < 1e-6:
            ln_p_tau = np.log(p_tau_safe)
            P_M = np.exp(np.sum(omega * ln_p_tau, axis=0))
        else:
            inner = np.sum(omega * (p_tau_safe ** (1.0 - sigma)), axis=0)
            inner_safe = np.maximum(inner, 1e-12)
            P_M = inner_safe ** (1.0 / (1.0 - sigma))

        inter_cost = (A_mat * P_M)[np.newaxis, :, :]
        rel_p = P_M[np.newaxis, :, :] / p_tau_safe
        x_mat = calib.a * ytot * (rel_p ** sigma)
    else:
        inter_cost = np.tensordot(p_vec, calib.a * tau_a, axes=(0, 0))[np.newaxis, :, :]
        x_mat = calib.a * ytot

    # Fiscal recycling adjustments
    r_use = r.copy()
    w_use = w.copy()
    tax_use = calib.tax.copy()
    sub_prod = np.zeros((1, ns, nc), dtype=float)
    delta_income = np.zeros((1, 1, nc), dtype=float)
    is_non_lumpsum = fiscal_closure not in ("lump_sum", "baseline", "")

    if is_non_lumpsum:
        target_c = recycling_params.get("target_country", "USA") if recycling_params else "USA"
        c_tgt = (
            calib.country_codes.index(target_c)
            if target_c in calib.country_codes
            else (calib.country_codes.index("USA") if "USA" in calib.country_codes else 0)
        )

        # Evaluate preliminary intermediate and final demand tariff collections for target country
        val_mat_pre = p_vec[:, np.newaxis, np.newaxis] * x_mat
        tariffs_interm_pre = np.sum((tau_a - 1.0) * val_mat_pre, axis=(0, 1))

        ppfd_pre = np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        l_endow_3d = calib.l_endow.reshape((1, 1, nc))
        k_endow_3d = calib.k_endow.reshape((1, 1, nc))
        Ycon_pre = w * l_endow_3d + r * k_endow_3d + T
        cd_pre = calib.theta * Ycon_pre / ppfd_pre
        tax_fd_arr = calib.tax_fd if calib.tax_fd is not None else np.zeros((1, nfd, nc))
        c_pre = cd_pre.copy()
        c_pre[:, 1:2, :] -= invforT.reshape((1, 1, nc)) / ppfd_pre[:, 1:2, :]
        xc_pre = calib.afd * (c_pre - tax_fd_arr * cd_pre)
        val_fd_pre = p_vec[:, np.newaxis, np.newaxis] * xc_pre
        tariffs_fd_pre = np.sum((taufd_a - 1.0) * val_fd_pre, axis=(0, 1))

        total_tariffs_by_c = tariffs_interm_pre + tariffs_fd_pre
        delta_rev = float(np.maximum(total_tariffs_by_c[c_tgt], 0.0))

        if fiscal_closure in ("capital_tax", "capital_tax_cut"):
            k_val = float(r[0, 0, c_tgt] * calib.k_endow.ravel()[c_tgt])
            if k_val > 1e-12:
                delta_tk = -delta_rev / k_val
                r_eff = max(float(r[0, 0, c_tgt] * (1.0 + delta_tk)), 0.05 * float(r[0, 0, c_tgt]))
                r_use[0, 0, c_tgt] = r_eff
            delta_income[0, 0, c_tgt] = delta_rev

        elif fiscal_closure in ("labor_tax", "payroll_tax_cut"):
            l_val = float(w[0, 0, c_tgt] * calib.l_endow.ravel()[c_tgt])
            if l_val > 1e-12:
                delta_tl = -delta_rev / l_val
                w_eff = max(float(w[0, 0, c_tgt] * (1.0 + delta_tl)), 0.05 * float(w[0, 0, c_tgt]))
                w_use[0, 0, c_tgt] = w_eff
            delta_income[0, 0, c_tgt] = delta_rev

        elif fiscal_closure in ("strategic_subsidy", "targeted_subsidy", "manufacturing_subsidy"):
            shares = recycling_params.get("subsidy_shares", {"MANU": 1.0}) if recycling_params else {"MANU": 1.0}
            for s_name, s_share in shares.items():
                if s_name in calib.sector_codes:
                    s_idx = calib.sector_codes.index(s_name)
                    y_sec = float(ytot[0, s_idx, c_tgt])
                    if y_sec > 1e-12:
                        sub_prod[0, s_idx, c_tgt] = (s_share * delta_rev) / y_sec
            delta_income[0, 0, c_tgt] = delta_rev

        elif fiscal_closure in ("deficit_reduction", "public_debt"):
            delta_income[0, 0, c_tgt] = delta_rev

    # 2. Paso 1 & 2: Zero-profit prices (pp) and factor demands (xl, xk)
    term_r = (r_use / calib.alpha) ** calib.alpha
    if replicate_matlab_precedence:
        term_w = w_use / ((1.0 - calib.alpha) ** (1.0 - calib.alpha))
    else:
        term_w = (w_use / (1.0 - calib.alpha)) ** (1.0 - calib.alpha)
    val_va = (1.0 / calib.beta) * (term_r * term_w)

    # Smooth capacity barrier penalty function
    pen_arr = np.zeros((1, ns, nc), dtype=float)
    if capacity_margins is not None:
        c_cap_idx = (
            calib.country_codes.index(capacity_target_country)
            if capacity_target_country in calib.country_codes
            else (calib.country_codes.index("USA") if "USA" in calib.country_codes else 0)
        )
        if isinstance(capacity_margins, dict):
            for k_sec, margin in capacity_margins.items():
                if isinstance(k_sec, str) and k_sec in calib.sector_codes:
                    s_i = calib.sector_codes.index(k_sec)
                    c_i = c_cap_idx
                elif isinstance(k_sec, tuple):
                    c_i, s_i = k_sec
                else:
                    continue
                y0_val = float(calib.ytot[0, s_i, c_i])
                y_bar = (1.0 + float(margin)) * y0_val
                if y_bar > 1e-12:
                    y_curr = float(ytot[0, s_i, c_i])
                    ratio = max(y_curr / y_bar, 0.0)
                    pen_arr[0, s_i, c_i] = float(penalty_scale * (ratio ** penalty_exponent))
        elif isinstance(capacity_margins, np.ndarray):
            y_bar = (1.0 + capacity_margins) * calib.ytot
            y_bar_safe = np.maximum(y_bar, 1e-12)
            ratio = np.maximum(ytot / y_bar_safe, 0.0)
            pen_arr = penalty_scale * (ratio ** penalty_exponent)

        val_va = val_va * (1.0 + pen_arr)

    pp = (val_va + inter_cost) / (1.0 - tax_use) - sub_prod
    mask_active = (ytot > 0)
    pp = np.where(mask_active, pp, 1.0)

    ratio_rw = ((1.0 - calib.alpha) * r_use) / (calib.alpha * w_use)
    ratio_wr = (calib.alpha * w_use) / ((1.0 - calib.alpha) * r_use)
    pen_factor = (1.0 + pen_arr) if capacity_margins is not None else 1.0
    xl = np.where(mask_active, (ytot / calib.beta) * (ratio_rw ** calib.alpha) * pen_factor, 0.0)
    xk = np.where(mask_active, (ytot / calib.beta) * (ratio_wr ** (1.0 - calib.alpha)) * pen_factor, 0.0)

    # 3. Paso 3: Final demand composite purchaser prices (ppfd)
    ppfd = np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]

    # 4. Paso 4: Final demand expenditure allocation and transfers
    l_endow_3d = calib.l_endow.reshape((1, 1, nc))
    k_endow_3d = calib.k_endow.reshape((1, 1, nc))
    Ycon = w * l_endow_3d + r * k_endow_3d + T - delta_income
    cd = calib.theta * Ycon / ppfd
    tax_fd_arr = calib.tax_fd if calib.tax_fd is not None else np.zeros((1, nfd, nc))
    Tax_c = tax_fd_arr * ppfd * cd

    c = cd.copy()
    c[:, 1:2, :] -= invforT.reshape((1, 1, nc)) / ppfd[:, 1:2, :]
    xc = calib.afd * (c - tax_fd_arr * cd)

    # 5. Paso 5: Intermediate demand quantities (already evaluated in x_mat above)

    # 6. Paso 6: Data model construction & Trade balance evaluation
    x_2d = x_mat.reshape((ns * nc, ns * nc), order="F")
    xc_2d = xc.reshape((ns * nc, nfd * nc), order="F")
    data_model = np.hstack([x_2d, xc_2d])

    pp_col = pp.reshape((ns * nc, 1), order="F")
    ppfd_row = ppfd.reshape((1, nfd * nc), order="F")
    data_model_p = np.hstack([
        np.repeat(pp_col, ns * nc, axis=1),
        np.repeat(ppfd_row, ns * nc, axis=0),
    ])
    data_model_val = data_model * data_model_p

    inter_val = data_model_val[:, :ns * nc].reshape(nc, ns, nc, ns)
    T_inter = np.sum(inter_val, axis=(1, 3))
    fd_val = data_model_val[:, ns * nc:].reshape(nc, ns, nc, nfd)
    T_fd = np.sum(fd_val, axis=(1, 3))

    np.fill_diagonal(T_inter, 0.0)
    np.fill_diagonal(T_fd, 0.0)

    X0 = np.sum(T_inter, axis=1)
    M0 = np.sum(T_inter, axis=0)
    XFD = np.sum(T_fd, axis=1)
    MFD = np.sum(T_fd, axis=0)

    invforT_realized = X0 + XFD - M0 - MFD

    # 7. Paso 7: Fiscal tax and tariff collections
    Tax_Total = np.sum(calib.tax * ytot, axis=1).ravel() + np.sum(Tax_c, axis=1).ravel()
    if is_non_lumpsum:
        val_mat_actual = p_vec[:, np.newaxis, np.newaxis] * x_mat
        tariffs_interm_actual = np.sum((tau_a - 1.0) * val_mat_actual, axis=(0, 1))
        val_fd_actual = p_vec[:, np.newaxis, np.newaxis] * xc
        tariffs_fd_actual = np.sum((taufd_a - 1.0) * val_fd_actual, axis=(0, 1))
        Tarifs_Totals = tariffs_interm_actual + tariffs_fd_actual
    else:
        Tarifs_Totals = M0 * tauf_vec + MFD * tauf_fd_vec

    # 8. Paso 8: Equilibrium residual equations
    ff0 = (ytot.reshape((ns * nc, 1), order="F") - np.sum(data_model, axis=1, keepdims=True)).ravel()
    ff1 = (pp.reshape((ns * nc, 1), order="F") - p.reshape((ns * nc, 1), order="F")).ravel()
    ff2 = (calib.l_endow.ravel() - np.sum(xl, axis=1).ravel())
    ff3 = (calib.k_endow.ravel() - np.sum(xk, axis=1).ravel())
    ff4 = (XN - invforT_realized[:nc - 1])
    ff5 = (T.ravel() - (Tax_Total + Tarifs_Totals))

    return np.concatenate([ff0, ff1, ff2, ff3, ff4, ff5])


def evaluate_equilibrium_residuals(
    x: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
    *,
    sigma: float = 0.0,
    fiscal_closure: str = "lump_sum",
    recycling_params: dict[str, Any] | None = None,
    capacity_margins: dict[str, float] | dict[tuple[int, int], float] | np.ndarray | None = None,
    capacity_target_country: str = "USA",
    penalty_scale: float = 0.05,
    penalty_exponent: float = 8.0,
) -> np.ndarray:
    """Evaluate equilibrium residuals with flexible argument alias support."""
    actual_tau = tau_a if tau_a is not None else tau
    actual_tau_fd = taufd_a if taufd_a is not None else tau_fd
    return compute_equilibrium_residuals(
        x=x,
        calib=calib,
        tau=actual_tau,
        tau_fd=actual_tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        replicate_matlab_precedence=replicate_matlab_precedence,
        sigma=sigma,
        fiscal_closure=fiscal_closure,
        recycling_params=recycling_params,
        capacity_margins=capacity_margins,
        capacity_target_country=capacity_target_country,
        penalty_scale=penalty_scale,
        penalty_exponent=penalty_exponent,
    )


__all__ = [
    "EquilibriumVariables",
    "unpack_equilibrium_vector",
    "pack_equilibrium_vector",
    "compute_equilibrium_residuals",
    "evaluate_equilibrium_residuals",
]
