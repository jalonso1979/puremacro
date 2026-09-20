"""Post-processing and flow accounting engine for puremacro.trade.

Ports and vectorizes MATLAB `ff_eval.m` (lines 1–179), reconstructing 4D bilateral
intermediate flows, 4D final demand deliveries, composite purchaser prices,
tariff revenue collections, macroeconomic GDP aggregates, and terms of trade.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy,
zero dev-dependency imports, fully vectorized.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np

from puremacro.trade.equilibrium import _evaluate_equilibrium, compute_equilibrium_residuals, unpack_equilibrium_vector

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def intermediate_matrix_to_tensor(
    matrix_847x847: np.ndarray,
    ns: int = 11,
    nc: int = 77,
) -> np.ndarray:
    """Convert intermediate transaction matrix to 4D bilateral tensor.

    Parameters
    ----------
    matrix_847x847 : np.ndarray
        2D matrix of shape (ns*nc, ns*nc).
    ns : int, default 11
        Number of industrial sectors.
    nc : int, default 77
        Number of countries.

    Returns
    -------
    np.ndarray
        4D tensor of shape (ns, nc, ns, nc) indexed as
        (s_src, c_src, s_dst, c_dst).
    """
    return np.asarray(matrix_847x847, dtype=float).reshape((ns, nc, ns, nc), order="F")


def intermediate_tensor_to_matrix(
    tensor_11x77x11x77: np.ndarray,
    ns: int = 11,
    nc: int = 77,
) -> np.ndarray:
    """Convert 4D bilateral intermediate tensor to 2D transaction matrix.

    Parameters
    ----------
    tensor_11x77x11x77 : np.ndarray
        4D tensor of shape (ns, nc, ns, nc).
    ns : int, default 11
        Number of industrial sectors.
    nc : int, default 77
        Number of countries.

    Returns
    -------
    np.ndarray
        2D matrix of shape (ns*nc, ns*nc).
    """
    return np.asarray(tensor_11x77x11x77, dtype=float).reshape((ns * nc, ns * nc), order="F")


def final_demand_matrix_to_tensor(
    matrix_847x231: np.ndarray,
    ns: int = 11,
    nc: int = 77,
    nfd: int = 3,
) -> np.ndarray:
    """Convert final demand transaction matrix to 4D bilateral tensor.

    Parameters
    ----------
    matrix_847x231 : np.ndarray
        2D matrix of shape (ns*nc, nfd*nc).
    ns : int, default 11
        Number of industrial sectors.
    nc : int, default 77
        Number of countries.
    nfd : int, default 3
        Number of final demand categories.

    Returns
    -------
    np.ndarray
        4D tensor of shape (ns, nc, nfd, nc) indexed as
        (s_src, c_src, fd_dst, c_dst).
    """
    return np.asarray(matrix_847x231, dtype=float).reshape((ns, nc, nfd, nc), order="F")


def final_demand_tensor_to_matrix(
    tensor_11x77x3x77: np.ndarray,
    ns: int = 11,
    nc: int = 77,
    nfd: int = 3,
) -> np.ndarray:
    """Convert 4D bilateral final demand tensor to 2D transaction matrix.

    Parameters
    ----------
    tensor_11x77x3x77 : np.ndarray
        4D tensor of shape (ns, nc, nfd, nc).
    ns : int, default 11
        Number of industrial sectors.
    nc : int, default 77
        Number of countries.
    nfd : int, default 3
        Number of final demand categories.

    Returns
    -------
    np.ndarray
        2D matrix of shape (ns*nc, nfd*nc).
    """
    return np.asarray(tensor_11x77x3x77, dtype=float).reshape((ns * nc, nfd * nc), order="F")


def compute_postprocessing_flows(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    matlab_compat: bool = True,
    **model_options: Any,
) -> dict[str, Any]:
    """Compute post-processing flows, transaction matrices, prices, and aggregates.

    The default is a vectorized port of MATLAB `ff_eval.m` (lines 1–179).
    ``accounting="consistent"`` instead reports producer-price bilateral trade,
    purchaser-price expenditure, nominal tax/factor-payment tables and a fixed-
    basket purchaser CPI. It returns the actual solved producer price, and
    metadata records all account residuals and the price/foreign-balance closure.

    Parameters
    ----------
    x_sol : np.ndarray
        Solved equilibrium state vector of length 2*ns*nc + 3*nc + (nc - 1).
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray, optional
        Intermediate tariff multipliers (1 + tariff_rate).
    tau_fd : np.ndarray, optional
        Final demand tariff multipliers (1 + tariff_rate).
    tauf : np.ndarray, optional
        National intermediate tariff rates by importing country, shape (nc,).
    tauf_fd : np.ndarray, optional
        National final demand tariff rates by importing country, shape (nc,).
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result used for CPI inflation weighting.

    Returns
    -------
    dict[str, Any]
        Dictionary with reconstructed flows, prices, aggregates, and tables.
    """
    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand

    blocks = _evaluate_equilibrium(
        x_sol, calib, tau, tau_fd, tauf, tauf_fd, replicate_matlab_precedence,
        **model_options,
    )
    if model_options.get("accounting", "legacy") == "consistent":
        from ._accounting import postprocess
        return postprocess(blocks, calib, base_result)
    p = blocks["p"]
    ytot = blocks["ytot"]
    r = blocks["r"]
    w = blocks["w"]
    T = blocks["T"]
    XN = blocks["XN"]
    invforT = blocks["invforT"]
    p_vec = blocks["p_vec"]
    pp = blocks["pp"]
    xl = blocks["xl"]
    xk = blocks["xk"]
    ppfd = blocks["ppfd"]
    tax_fd_arr = blocks["tax_fd_arr"]
    cd = blocks["cd"]
    Tax_c = blocks["Tax_c"]
    c = blocks["c"]
    xc = blocks["xc"]
    x_mat = blocks["x_mat"]
    tau_a = blocks["tau_a"]
    taufd_a = blocks["taufd_a"]
    Tarifs_Totals = blocks["Tarifs_Totals"]
    Pfd_final = (1.0 + tax_fd_arr) * ppfd
    interm_matrix = x_mat.reshape((ns * nc, ns * nc), order="F")
    interm_tensor = interm_matrix.reshape((ns, nc, ns, nc), order="F")

    fd_matrix = xc.reshape((ns * nc, nfd * nc), order="F")
    fd_tensor = fd_matrix.reshape((ns, nc, nfd, nc), order="F")

    # 6. Paso 6: Data model and valuation matrices
    data_model = np.hstack([interm_matrix, fd_matrix])  # (847, 1078)
    pp_col = pp.reshape((ns * nc, 1), order="F")
    ppfd_row = ppfd.reshape((1, nfd * nc), order="F")
    data_model_p = np.hstack([
        np.repeat(pp_col, ns * nc, axis=1),
        np.repeat(ppfd_row, ns * nc, axis=0),
    ])
    data_model_val = data_model * data_model_p

    # 7. Paso 7: Bilateral trade matrices & foreign investment realization
    inter_val = data_model_val[:, :ns * nc].reshape(nc, ns, nc, ns)
    T_inter = np.sum(inter_val, axis=(1, 3))
    fd_val = data_model_val[:, ns * nc:].reshape(nc, ns, nc, nfd)
    T_fd = np.sum(fd_val, axis=(1, 3))

    bilateral_gross = (T_inter + T_fd).copy()
    np.fill_diagonal(bilateral_gross, 0.0)

    exports = np.sum(bilateral_gross, axis=1)
    imports = np.sum(bilateral_gross, axis=0)
    net_exports = exports - imports

    # 8. Paso 8: Tariff collections & transaction tables
    tau_a_mat = tau_a.reshape(ns * nc, ns * nc, order="F")
    taufd_a_mat = taufd_a.reshape(ns * nc, nfd * nc, order="F")
    tariff_rates = np.hstack([tau_a_mat - 1.0, taufd_a_mat - 1.0])
    tariff_values = data_model_val
    if model_options.get("fiscal_closure", "lump_sum") not in ("lump_sum", "baseline", ""):
        tariff_values = data_model * p_vec[:, None]
    data_tariff = np.sum(tariff_values * tariff_rates, axis=0, keepdims=True)

    row_tax = np.hstack([
        (calib.tax * ytot).reshape(1, ns * nc, order="F"),
        Tax_c.reshape(1, nc * nfd, order="F"),
    ])
    row_xl = np.hstack([xl.reshape(1, ns * nc, order="F"), np.zeros((1, nc * nfd))])
    row_xk = np.hstack([xk.reshape(1, ns * nc, order="F"), np.zeros((1, nc * nfd))])

    data_model_vf = np.vstack([data_model, row_tax, row_xl, row_xk])
    data_tariff_vf = np.vstack([data_model, row_tax, data_tariff, row_xl, row_xk])

    # Extract tariff revenues by country
    tariff_row = data_tariff.ravel()
    tariffs_interm = tariff_row[:ns * nc].reshape(nc, ns)
    tariffs_fd = tariff_row[ns * nc:].reshape(nc, nfd)
    tariffs_total = np.sum(tariffs_interm, axis=1) + np.sum(tariffs_fd, axis=1)

    # 9. Macroeconomic aggregates: GDP at factor cost and market prices
    gdp_fc = w.ravel() * calib.l_endow.ravel() + r.ravel() * calib.k_endow.ravel()
    gdp_mp = gdp_fc + T.ravel()

    # 10. Price Indices: CPI and Terms of Trade
    # Domestic wage-deflated CPI
    c_adj = c.copy()
    if matlab_compat:
        if len(XN) == nc - 1:
            xn_adj = np.append(XN, float(np.sum(XN)))
        else:
            xn_adj = np.asarray(XN, dtype=float).copy()
            if xn_adj[nc - 1] < 0 and np.sum(xn_adj[: nc - 1]) > 0:
                xn_adj[nc - 1] = float(np.sum(xn_adj[: nc - 1]))
        c_adj[0, 1, :] += xn_adj
    else:
        c_adj[0, 1, :] += invforT[0, :]

    if base_result is not None and base_result.c_fd is not None:
        c_base = np.asarray(base_result.c_fd, dtype=float).copy()
        xn_b = np.asarray(base_result.XN_sol, dtype=float).ravel()
        if len(xn_b) == nc - 1:
            row_val = float(np.sum(xn_b)) if matlab_compat else -float(np.sum(xn_b))
            xn_b_full = np.append(xn_b, row_val)
        else:
            xn_b_full = xn_b.copy()
            if matlab_compat and xn_b_full[nc - 1] < 0 and np.sum(xn_b_full[: nc - 1]) > 0:
                xn_b_full[nc - 1] = float(np.sum(xn_b_full[: nc - 1]))
            elif not matlab_compat and xn_b_full[nc - 1] > 0 and np.sum(xn_b_full[: nc - 1]) > 0:
                xn_b_full[nc - 1] = -float(np.sum(xn_b_full[: nc - 1]))
        c_base[0, 1, :] += xn_b_full
    else:
        c_base = c_adj.copy()

    p_dom = ppfd / w
    denom_cpi = np.sum(c_base, axis=(0, 1))
    cpi_w = np.where(
        denom_cpi > 0,
        np.sum(p_dom * c_adj, axis=(0, 1)) / denom_cpi,
        1.0,
    )

    # Terms of trade (TOT = Px / Pm)
    qxX0_sol = interm_tensor.transpose(0, 2, 1, 3).copy()
    qxFD0_sol = fd_tensor.transpose(0, 2, 1, 3).copy()
    for k_idx in range(nc):
        qxX0_sol[:, :, k_idx, k_idx] = 0.0
        qxFD0_sol[:, :, k_idx, k_idx] = 0.0

    p_s_c = p[0]  # (ns, nc)
    q_exp_sec = qxX0_sol.sum(axis=(1, 3)) + qxFD0_sol.sum(axis=(1, 3))  # (ns, nc)
    val_exp_sec = p_s_c * q_exp_sec
    q_exp_tot = q_exp_sec.sum(axis=0)
    px = np.where(q_exp_tot > 0, val_exp_sec.sum(axis=0) / q_exp_tot, 1.0)

    q_imp_sec = qxX0_sol.sum(axis=(1, 2)) + qxFD0_sol.sum(axis=(1, 2))  # (ns, nc)
    v_imp_x = (p_s_c[:, None, :, None] * qxX0_sol).sum(axis=(0, 1, 2))
    v_imp_fd = (p_s_c[:, None, :, None] * qxFD0_sol).sum(axis=(0, 1, 2))
    v_imp_tot = v_imp_x + v_imp_fd
    q_imp_tot = q_imp_sec.sum(axis=0)
    pm = np.where(q_imp_tot > 0, v_imp_tot / q_imp_tot, 1.0)
    terms_of_trade = px / pm

    # 11. Residual vector evaluation
    residuals = blocks["residuals"]
    diff = float(np.sum(np.abs(residuals)))
    max_res = float(np.max(np.abs(residuals)))

    return {
        "p_sol": pp,
        "y_sol": ytot,
        "r_sol": r,
        "w_sol": w,
        "T_sol": T,
        "XN_sol": XN,
        "invforT": invforT,
        "intermediate_flows": interm_tensor,
        "intermediate_matrix": interm_matrix,
        "final_demand_flows": fd_tensor,
        "final_demand_matrix": fd_matrix,
        "p_fd": ppfd,
        "P_fd": Pfd_final,
        "c": c,
        "c_fd": c,
        "cd": cd,
        "bilateral_trade": bilateral_gross,
        "exports": exports,
        "imports": imports,
        "net_exports": net_exports,
        "tariffs": tariffs_total,
        "fiscal_tariffs": blocks["Tarifs_Totals"],
        "tariffs_interm": tariffs_interm,
        "tariffs_fd": tariffs_fd,
        "gdp": gdp_mp,
        "gdp_fc": gdp_fc,
        "cpi": cpi_w,
        "terms_of_trade": terms_of_trade,
        "data_model_vf": data_model_vf,
        "data_tariff_vf": data_tariff_vf,
        "qxX0": interm_tensor.transpose(0, 2, 1, 3),
        "qxFD0": fd_tensor.transpose(0, 2, 1, 3),
        "residuals": residuals,
        "residual_norm": diff,
        "diff": diff,
        "max_residual": max_res,
    }


def postprocess_trade_equilibrium(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
    converged: bool = True,
    iterations: int = 0,
    base_result: TradeEquilibriumResult | None = None,
    metadata: dict[str, Any] | None = None,
    matlab_compat: bool = True,
    **model_options: Any,
) -> TradeEquilibriumResult:
    """Post-process equilibrium solution and package into TradeEquilibriumResult container.

    Parameters
    ----------
    x_sol : np.ndarray
        Solved equilibrium state vector.
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray, optional
        Intermediate tariff multipliers.
    tau_fd : np.ndarray, optional
        Final demand tariff multipliers.
    tauf : np.ndarray, optional
        National intermediate tariff rates.
    tauf_fd : np.ndarray, optional
        National final demand tariff rates.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence.
    converged : bool, default True
        Solver convergence flag.
    iterations : int, default 0
        Number of iterations executed.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result for relative index calculation.
    metadata : dict[str, Any], optional
        Additional diagnostic and provenance metadata.
    matlab_compat : bool, default True
        Whether to apply MATLAB compatibility for ROW capital formation adjustment.

    Returns
    -------
    TradeEquilibriumResult
        Frozen dataclass with all flow accounting, prices, and aggregates populated.
    """
    from puremacro.trade._results import TradeEquilibriumResult

    flows = compute_postprocessing_flows(
        x_sol=x_sol,
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        replicate_matlab_precedence=replicate_matlab_precedence,
        base_result=base_result,
        matlab_compat=matlab_compat,
        **model_options,
    )

    meta = dict(metadata) if metadata is not None else {}
    meta.update(model_options)
    meta.update({
        "tariff_revenue_mode": model_options.get("tariff_revenue_mode", "legacy_national"),
        "fiscal_tariffs": flows["fiscal_tariffs"],
        "matlab_compat": matlab_compat,
        "replicate_matlab_precedence": replicate_matlab_precedence,
        "bilateral_trade": flows["bilateral_trade"],
        "tariffs_interm": flows["tariffs_interm"],
        "tariffs_fd": flows["tariffs_fd"],
        "data_model_vf": flows["data_model_vf"],
        "data_tariff_vf": flows["data_tariff_vf"],
    })

    meta.update(flows.get("accounting_metadata", {}))
    if meta.get("accounting") == "consistent":
        tolerance = float(meta.get("tol", 1e-8))
        physical = meta["physical_residuals"]
        converged = bool(converged and meta["demand_feasible"] and np.isfinite(physical).all()
                         and np.max(np.abs(physical)) <= tolerance
                         and np.isfinite(flows["max_residual"]) and flows["max_residual"] <= tolerance)

    return TradeEquilibriumResult(
        x_sol=np.asarray(x_sol, dtype=float).ravel(),
        p_sol=flows["p_sol"],
        y_sol=flows["y_sol"],
        r_sol=flows["r_sol"],
        w_sol=flows["w_sol"],
        T_sol=flows["T_sol"],
        XN_sol=flows["XN_sol"],
        intermediate_flows=flows["intermediate_flows"],
        final_demand_flows=flows["final_demand_flows"],
        p_fd=flows["p_fd"],
        c_fd=flows["c_fd"],
        gdp=flows["gdp"],
        gdp_fc=flows["gdp_fc"],
        imports=flows["imports"],
        exports=flows["exports"],
        tariffs=flows["tariffs"],
        cpi=flows["cpi"],
        terms_of_trade=flows["terms_of_trade"],
        c_sol=flows["c_fd"],
        pfd_sol=flows["p_fd"],
        Pfd_final=flows["P_fd"],
        qxX0_sol=flows["qxX0"],
        qxFD0_sol=flows["qxFD0"],
        data_model_vf=flows["data_model_vf"],
        data_tariff_vf=flows["data_tariff_vf"],
        converged=converged,
        iterations=iterations,
        diff=flows["diff"],
        residual_norm=flows["residual_norm"],
        max_residual=flows["max_residual"],
        residuals=flows["residuals"],
        country_codes=calib.country_codes,
        sector_codes=calib.sector_codes,
        metadata=meta,
    )


__all__ = [
    "intermediate_matrix_to_tensor",
    "intermediate_tensor_to_matrix",
    "final_demand_matrix_to_tensor",
    "final_demand_tensor_to_matrix",
    "compute_postprocessing_flows",
    "postprocess_trade_equilibrium",
]
