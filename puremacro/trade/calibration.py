"""Benchmark Parameter Calibration for CGE Trade Model.

Vectorized calibration engine porting `calibrar.m` (lines 1–205). Extracts structural
technology parameters (alpha, beta, a, afd), national endowments (k_endow, l_endow),
fiscal tax rates and transfers (tax, tax_fd, T), and trade balances (invforT) from
the empirical ICIO transaction matrix.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_SECTOR_CODES,
    RAW_45_SECTOR_CODES,
)


def calibrate_trade_model(
    data: np.ndarray,
    ns: int = 11,
    nc: int = 77,
    nfd: int = 3,
    country_codes: Sequence[str] | None = None,
    sector_codes: Sequence[str] | None = None,
    validate: bool = True,
) -> TradeCalibrationResult:
    """Calibrate structural parameters from the processed ICIO transaction table.

    ``tax_fd`` retains the MATLAB denominator (including foreign saving in the
    investment allocation). The consistent equilibrium mode reconstructs the
    tax share on actual final purchases from ``data_calibra``, excluding that
    financial saving. Its production charge is a tax-inclusive share of output
    revenue; interpreting aggregate source TLS this way is a model assumption.

    Parameters
    ----------
    data : np.ndarray
        Processed ICIO transaction matrix of shape (ns*nc + 3, ns*nc + nfd*nc).
        Rows: 0..ns*nc-1 intermediate goods, ns*nc net taxes, ns*nc+1 labor, ns*nc+2 capital.
        Columns: 0..ns*nc-1 intermediate uses, ns*nc..final demand uses.
    ns : int, default=11
        Number of composite industrial sectors (11 for aggregated baseline, 45 for raw ICIO).
    nc : int, default=77
        Number of countries/regions.
    nfd : int, default=3
        Number of final demand categories (Consumption, Investment, Direct Purchases).
    country_codes : sequence of str, optional
        List of ISO alpha-3 country identifiers.
    sector_codes : sequence of str, optional
        List of industry sector identifiers.
    validate : bool, default=True
        Whether to assert budget balance and calibration table reconstruction.

    Returns
    -------
    TradeCalibrationResult
        Frozen dataclass container encapsulating all structural calibrated parameters.
    """
    expected_rows = ns * nc + 3
    expected_cols = ns * nc + nfd * nc
    if data.shape != (expected_rows, expected_cols):
        # Auto-infer ns if shape matches a different sector count for (nc, nfd)
        if (data.shape[0] - 3) % nc == 0:
            inferred_ns = (data.shape[0] - 3) // nc
            if data.shape == (inferred_ns * nc + 3, inferred_ns * nc + nfd * nc):
                ns = inferred_ns
                expected_rows = ns * nc + 3
                expected_cols = ns * nc + nfd * nc
        if data.shape != (expected_rows, expected_cols):
            raise ValueError(
                f"Invalid ICIO data matrix shape {data.shape}. "
                f"Expected ({expected_rows}, {expected_cols}) for ns={ns}, nc={nc}, nfd={nfd}."
            )

    # -------------------------------------------------------------------------
    # 1. Gross Output Extraction (y)
    # -------------------------------------------------------------------------
    # Column sum of intermediate uses and value-added across all rows.
    # Columns are structured as: col = jc * ns + js (dest country jc, dest sector js).
    col_sums = np.sum(data[:, :ns * nc], axis=0)  # Shape (ns*nc,)
    y = col_sums.reshape(nc, ns).T[np.newaxis, :, :]  # Shape (1, ns, nc)

    # -------------------------------------------------------------------------
    # 2. Value Added Decomposition & Factor Endowments
    # -------------------------------------------------------------------------
    # Row ns*nc: Net production taxes less subsidies (t)
    # Row ns*nc + 1: Labor compensation (l = 2/3 VA)
    # Row ns*nc + 2: Gross capital return (k = 1/3 VA)
    t = data[ns * nc, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]      # (1, ns, nc)
    l = data[ns * nc + 1, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]  # (1, ns, nc)
    k = data[ns * nc + 2, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]  # (1, ns, nc)

    # National endowments and tax totals
    TT = np.sum(t, axis=1)  # (1, nc)
    LT = np.sum(l, axis=1)  # (1, nc)
    KT = np.sum(k, axis=1)  # (1, nc)

    # Final demand taxes (Row ns*nc, columns ns*nc .. ns*nc + nfd*nc - 1)
    tfd = data[ns * nc, ns * nc:].reshape(nc, nfd).T[np.newaxis, :, :]  # (1, nfd, nc)
    TTfd = np.sum(tfd, axis=1)  # (1, nc)

    # Total national income
    Income = TT + TTfd + LT + KT  # (1, nc)

    # -------------------------------------------------------------------------
    # 3. Intermediate and Final Demand Flow Tensors (4D & 3D)
    # -------------------------------------------------------------------------
    # 4D Intermediate flows: (origin_sector, origin_country, dest_sector, dest_country)
    x_4d = data[:ns * nc, :ns * nc].reshape(nc, ns, nc, ns).transpose(1, 0, 3, 2)  # (ns, nc, ns, nc)

    # 4D Final demand flows: (origin_sector, origin_country, category, dest_country)
    xfd_4d = data[:ns * nc, ns * nc:].reshape(nc, ns, nc, nfd).transpose(1, 0, 3, 2)  # (ns, nc, nfd, nc)

    # 3D Collapsed representations (origin_country*ns + origin_sector collapsed to ns*nc)
    x_3d = x_4d.transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)     # (ns*nc, ns, nc)
    xc_3d = xfd_4d.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)  # (ns*nc, nfd, nc)

    # -------------------------------------------------------------------------
    # 4. Bilateral Trade Balances & Current Account Surplus (invforT)
    # -------------------------------------------------------------------------
    # Bilateral country-to-country trade flows (summed across sectors)
    T_inter = np.sum(data[:ns * nc, :ns * nc].reshape(nc, ns, nc, ns), axis=(1, 3))  # (nc, nc) [orig_c, dest_c]
    T_fd = np.sum(data[:ns * nc, ns * nc:].reshape(nc, ns, nc, nfd), axis=(1, 3))     # (nc, nc) [orig_c, dest_c]

    # Exclude domestic absorption (diagonal)
    np.fill_diagonal(T_inter, 0.0)
    np.fill_diagonal(T_fd, 0.0)

    X0 = np.sum(T_inter, axis=1, keepdims=True).T   # (1, nc) Intermediate exports
    M0 = np.sum(T_inter, axis=0, keepdims=True)     # (1, nc) Intermediate imports
    XFD = np.sum(T_fd, axis=1, keepdims=True).T     # (1, nc) Final demand exports
    MFD = np.sum(T_fd, axis=0, keepdims=True)       # (1, nc) Final demand imports

    # Net foreign surplus (Current Account balance)
    invforT = (X0 + XFD - M0 - MFD)  # (1, nc)

    # -------------------------------------------------------------------------
    # 5. Final Expenditure Allocation & Consumption Shares (theta)
    # -------------------------------------------------------------------------
    c = np.sum(xc_3d, axis=0, keepdims=True) + tfd  # (1, nfd, nc)
    # Add net foreign investment to Investment category (index 1 if nfd >= 2, else index 0)
    inv_idx = 1 if nfd >= 2 else 0
    c[0, inv_idx, :] += invforT[0, :]

    if validate:
        budget_disc = float(np.max(np.abs(np.sum(c, axis=1) - Income)))
        if budget_disc > 1e-3:
            raise ValueError(f"Consumer budget balance discrepancy {budget_disc:.4e} exceeds 1e-3.")

    with np.errstate(divide="ignore", invalid="ignore"):
        inc_3d = Income.reshape(1, 1, nc)
        theta = np.divide(c, inc_3d, out=np.zeros_like(c), where=(inc_3d > 0))

        if validate:
            theta_disc = float(np.abs(np.sum(theta) - nc))
            if theta_disc > 1e-9:
                raise ValueError(f"Expenditure share sum discrepancy {theta_disc:.4e} exceeds 1e-9.")

        # ---------------------------------------------------------------------
        # 6. Technology Parameters (alpha, beta)
        # ---------------------------------------------------------------------
        mask_y = (y > 0)
        alpha = np.zeros_like(y)
        beta = np.zeros_like(y)

        ratio_kl = np.divide(k, l, out=np.zeros_like(k), where=(l > 0))
        alpha[mask_y] = ratio_kl[mask_y] / (1.0 + ratio_kl[mask_y])
        denom_beta = (k[mask_y] ** alpha[mask_y]) * (l[mask_y] ** (1.0 - alpha[mask_y]))
        beta[mask_y] = np.divide(
            y[mask_y], denom_beta, out=np.zeros_like(y[mask_y]), where=(denom_beta > 0)
        )

        # ---------------------------------------------------------------------
        # 7. Technical Input-Output & Final Demand Coefficients (a, afd)
        # ---------------------------------------------------------------------
        # 4D Technical coefficients
        y_4d = y.reshape(1, 1, ns, nc)
        a_4d = np.divide(x_4d, y_4d, out=np.zeros_like(x_4d), where=(y_4d > 0))  # (ns, nc, ns, nc)

        sum_fd_4d = np.sum(xfd_4d, axis=(0, 1), keepdims=True)
        afd_4d = np.divide(
            xfd_4d, sum_fd_4d, out=np.zeros_like(xfd_4d), where=(sum_fd_4d > 0)
        )  # (ns, nc, nfd, nc)

        # 3D Technical coefficients for fast solver evaluation
        a_3d = np.divide(x_3d, y, out=np.zeros_like(x_3d), where=(y > 0))  # (ns*nc, ns, nc)
        sum_xc = np.sum(xc_3d, axis=0, keepdims=True)
        afd_3d = np.divide(
            xc_3d, sum_xc, out=np.zeros_like(xc_3d), where=(sum_xc > 0)
        )  # (ns*nc, nfd, nc)

        # ---------------------------------------------------------------------
        # 8. Tax Rates & Transfers
        # ---------------------------------------------------------------------
        # Net production taxes on gross output
        tax = np.divide(t, y, out=np.zeros_like(t), where=(y != 0.0))  # (1, ns, nc)
        # Final demand taxes (Note: c can be negative for deficit countries, use c != 0)
        tax_fd = np.divide(tfd, c, out=np.zeros_like(tfd), where=(c != 0.0))  # (1, nfd, nc)

        # Baseline transfers: total tax collections in level
        T_base = (TT + TTfd).reshape(1, 1, nc)  # (1, 1, nc)

        # ---------------------------------------------------------------------
        # 9. Calibration Table Validation (data_calibra)
        # ---------------------------------------------------------------------
        Tax_c = tax_fd * c
        xl = np.zeros_like(y)
        xk = np.zeros_like(y)
        mask_prod = (beta > 0) & (alpha > 0) & (alpha < 1.0)
        if np.any(mask_prod):
            term_l = ((1.0 - alpha[mask_prod]) / alpha[mask_prod]) ** alpha[mask_prod]
            term_k = (alpha[mask_prod] / (1.0 - alpha[mask_prod])) ** (1.0 - alpha[mask_prod])
            xl[mask_prod] = (y[mask_prod] / beta[mask_prod]) * term_l
            xk[mask_prod] = (y[mask_prod] / beta[mask_prod]) * term_k

        x_rec = a_3d * y
        x_rec_2d = x_rec.transpose(0, 2, 1).reshape(ns * nc, ns * nc)
        xc_rec_2d = xc_3d.transpose(0, 2, 1).reshape(ns * nc, nfd * nc)

        row1 = np.hstack([x_rec_2d, xc_rec_2d])
        row2 = np.hstack([(tax * y).flatten(order='F').reshape(1, -1), Tax_c.flatten(order='F').reshape(1, -1)])
        row3 = np.hstack([xl.flatten(order='F').reshape(1, -1), np.zeros((1, nfd * nc))])
        row4 = np.hstack([xk.flatten(order='F').reshape(1, -1), np.zeros((1, nfd * nc))])

        data_calibra = np.vstack([row1, row2, row3, row4])

        if validate:
            calib_error = float(np.max(np.sum(np.abs(data_calibra - data), axis=0)))
            if calib_error > 1e-4:
                raise ValueError(f"Calibration table reconstruction error {calib_error:.4e} exceeds 1e-4.")

    # -------------------------------------------------------------------------
    # 10. Assemble Result Container
    # -------------------------------------------------------------------------
    resolved_countries = tuple(country_codes) if country_codes is not None else (
        CANONICAL_COUNTRY_CODES if nc == 77 else tuple(f"C{i:02d}" for i in range(nc))
    )
    resolved_sectors = tuple(sector_codes) if sector_codes is not None else (
        CANONICAL_SECTOR_CODES if ns == 11 else (
            RAW_45_SECTOR_CODES if ns == 45 else tuple(f"S{i:02d}" for i in range(ns))
        )
    )

    return TradeCalibrationResult(
        a=a_3d,
        afd=afd_3d,
        alpha=alpha,
        beta=beta,
        k_endow=KT,
        l_endow=LT,
        invforT=invforT,
        tax=tax,
        ytot=y,
        theta=theta,
        tax_fd=tax_fd,
        TT=TT,
        TTfd=TTfd,
        T=T_base,
        a_3d=a_3d,
        afd_3d=afd_3d,
        a_4d=a_4d,
        afd_4d=afd_4d,
        data_calibra=data_calibra,
        n_countries=nc,
        n_sectors=ns,
        n_final_demand=nfd,
        country_codes=resolved_countries,
        sector_codes=resolved_sectors,
    )


__all__ = [
    "calibrate_trade_model",
]
