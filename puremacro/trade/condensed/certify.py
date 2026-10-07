"""Independent ten-block raw-flow certificate of the condensed model.

The certificate re-evaluates the equilibrium conditions from the primitive
table flows ``Z, F, VA, TLS, TFD`` and the wedges, never from the calibrated
coefficients ``a, b, omega, theta`` or the solver's factorizations. With
``yhat_j = y_j / y0_j`` on every cell (``y0 > 0`` is a calibration gate) and
``qhat^f_k = q^f_k / B^f_k``:

| block | expression | scale |
|---|---|---|
| zero_profit_active | ``p_j y0_j yhat_j - [sum_i tau_ij Z_ij p_i + w_c(j) VA_j + TLS_j p_j] yhat_j`` | ``max(p_j y_j, 1)`` |
| goods_clearing | ``y_i - (Z yhat)_i - sum_{k,f} F^f_ik qhat^f_k - sum_k qV_ik`` | ``max(y_i, 1)`` |
| factor_markets | ``sum_{j in k} VA_j yhat_j - Fbar_k`` | ``Fbar_k`` |
| household_budget | ``sum_f E^f_k + E^V_k - (Y_k - XN_k)`` | ``Y0_k`` |
| national_income | ``Y_k - w_k Fbar_k - PT^raw_k - TR^raw_k`` (both rebuilt from flows) | ``Y0_k`` |
| tariff_revenue_identity | ``TR_k - TR^raw_k`` | ``Y0_k`` |
| closure_equation | ``XN_k - closure(w, Y)`` | ``Y0_k`` |
| current_account_identity | ``NX_k - XN_k`` with ``NX`` from bilateral producer values | ``Y0_k`` |
| world_adding_up | ``|sum XN|`` and ``|sum NX|`` | ``sum Y0`` |
| numeraire_identity | numeraire residual | ``sum Fbar`` or 1 |

Every block must be at most ``tol = 1e-10``. The two per-cell blocks are scaled
by ``max(p_j y_j, cell_floor)`` and ``max(y_i, cell_floor)`` with
``cell_floor = 1`` in the table's own units (the IO USD-million convention):
cells below one unit are judged at absolute ``1e-10``, so tables in large units
should pass a smaller ``cell_floor``. The certificate uses the state's basket
prices ``P`` and quantities ``q``, so it is independent of the calibrated
coefficients but not of the demand system that produced ``q``.
When baseline duties were separated, ``TLS`` and ``TFD`` are first netted of
``(tau0 - 1) * flow`` so benchmark customs collections are not counted twice.
These are floating-point verification criteria, not interval enclosures.

Phantom cells (empty cells that the table build filled with a ``1e-6`` own
consumption flow and ``1e-6`` value added) are excluded from the zero-profit
block only. Everywhere else they enter with their true ``yhat_j``, because the
model lets their output move and their value added is part of ``Fbar``. The IO
engine held ``yhat_j = 1`` on inactive cells, so its factor-market block
rejected valid equilibria of small-unit tables with empty cells (a violation of
``sum_phantom VA_j (yhat_j - 1) / Fbar_k``); this is a documented deviation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .calibration import CondensedCalibration
from .errors import CertificationFailure
from .model import CondensedState, _resolve_numeraire
from .table import ENDOGENOUS_FD, FD_CODES
from .tariffs import TariffWedges

CERTIFICATE_TOL = 1e-10

BLOCK_NAMES: tuple[str, ...] = (
    "zero_profit_active", "goods_clearing", "factor_markets", "household_budget", "national_income",
    "tariff_revenue_identity", "closure_equation", "current_account_identity", "world_adding_up",
    "numeraire_identity",
)


@dataclass(frozen=True)
class RawFlowCertificate:
    """Maximum scaled violation of each certificate block.

    Attributes
    ----------
    zero_profit_active, goods_clearing, factor_markets, household_budget,
    national_income, tariff_revenue_identity, closure_equation,
    current_account_identity, world_adding_up, numeraire_identity : float
    tol : float
        Gate applied to every block.
    cell_floor : float
        Floor of the per-cell scales ``max(p_j y_j, cell_floor)`` and ``max(y_i, cell_floor)``.
    metadata : dict
        Closure, numeraire and inventory treatment used.
    """

    zero_profit_active: float
    goods_clearing: float
    factor_markets: float
    household_budget: float
    national_income: float
    tariff_revenue_identity: float
    closure_equation: float
    current_account_identity: float
    world_adding_up: float
    numeraire_identity: float
    tol: float = CERTIFICATE_TOL
    cell_floor: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> dict[str, float]:
        """Block name to maximum scaled violation."""
        return {name: float(getattr(self, name)) for name in BLOCK_NAMES}

    @property
    def max_block(self) -> float:
        """Largest scaled violation over the ten blocks."""
        return max(self.blocks.values())

    @property
    def passed(self) -> bool:
        """True when every block is finite and at most ``tol``."""
        vals = np.array(list(self.blocks.values()))
        return bool(np.all(np.isfinite(vals)) and np.all(vals <= self.tol))

    @property
    def failed_blocks(self) -> tuple[str, ...]:
        """Names of the blocks that are non-finite or above ``tol``."""
        return tuple(k for k, v in self.blocks.items() if not (np.isfinite(v) and v <= self.tol))

    def to_dict(self) -> dict[str, Any]:
        """The ten block values plus ``tol``, ``cell_floor`` and ``passed``."""
        return {**self.blocks, "tol": self.tol, "cell_floor": self.cell_floor, "passed": self.passed}

    def to_dataframe(self) -> pd.DataFrame:
        """One row per block: maximum scaled violation and whether it passed."""
        rows = [(k, v, bool(np.isfinite(v) and v <= self.tol)) for k, v in self.blocks.items()]
        return pd.DataFrame(rows, columns=["Block", "Max scaled violation", "Passed"]).set_index("Block")

    def summary(self) -> str:
        """One line: the largest block, the tolerance and the failed blocks if any."""
        return (f"RawFlowCertificate: max block {self.max_block:.3e} (tol {self.tol:.0e}); "
                + ("all ten blocks passed" if self.passed else f"failed: {', '.join(self.failed_blocks)}"))

    def to_markdown(self, **kwargs: Any) -> str:
        """The block table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The block table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The block table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


def certify_raw_flows(
    calib: CondensedCalibration,
    wedges: TariffWedges,
    state: CondensedState,
    *,
    closure: str = "factor",
    numeraire: str = "factor",
    inventories: str = "tariffed",
    tol: float = CERTIFICATE_TOL,
    strict: bool = True,
    cell_floor: float = 1.0,
) -> RawFlowCertificate:
    """Evaluate the independent ten-block raw-flow certificate.

    Parameters
    ----------
    calib : CondensedCalibration
        Supplies the raw table and benchmark aggregates.
    wedges : TariffWedges
        The wedges the state was solved under.
    state : CondensedState
        The state to certify.
    closure, numeraire, inventories : str
        The model options used for the solve (they change blocks 7, 10 and the
        inventory duty terms).
    tol : float, default 1e-10
        Gate for every block.
    strict : bool, default True
        Raise :class:`CertificationFailure` on any block above ``tol``.
    cell_floor : float, default 1.0
        Floor of the per-cell scales in the zero-profit and goods-clearing
        blocks, ``max(p_j y_j, cell_floor)`` and ``max(y_i, cell_floor)``, in
        the table's own units (the IO engine's USD-million convention). Cells
        below the floor are judged at absolute ``tol``; use a smaller floor for
        tables whose cells are below one unit.

    Returns
    -------
    RawFlowCertificate
    """
    if closure not in ("factor", "gdp", "own"):
        raise ValueError(f"unknown closure {closure!r}")
    if inventories not in ("tariffed", "untariffed"):
        raise ValueError(f"unknown inventory treatment {inventories!r}")
    cell_floor = float(cell_floor)
    if not np.isfinite(cell_floor) or cell_floor <= 0:
        raise ValueError("cell_floor must be a positive finite number in the table's units")
    kind, numeraire_country = _resolve_numeraire(calib, numeraire)
    table = calib.table
    Z, F, VA, TLS, TFD = table.Z, table.F, table.VA, table.TLS, table.TFD
    tau_V = wedges.tau_V if inventories == "tariffed" else np.ones_like(wedges.tau_V)
    if calib.baseline_tariffs is not None:
        bt = calib.baseline_tariffs.wedges
        TLS = TLS - ((bt.tau - 1) * Z).sum(axis=0)
        TFD = TFD.copy()
        TFD[:, :3] -= ((bt.tau_fd - 1) * F[:, :, :3]).sum(axis=0)
        TFD[:, 3] -= ((bt.tau_V - 1) * F[:, :, 3]).sum(axis=0)
    n, s = table.n_countries, table.n_sectors
    cty = calib.country_of_cell
    act = calib.active
    w, p, y, Y, XN, E, q, EV, TR = (state.w, state.p, state.y, state.Y, state.XN, state.E, state.q,
                                    state.E_V, state.TR)
    y0 = calib.y0
    y_hat = y / y0  # every cell, phantoms included (their output moves in the model)
    endo = [FD_CODES.index(c) for c in ENDOGENOUS_FD]
    iv = FD_CODES.index("V")
    Fe = F[:, :, endo]
    B0 = Fe.sum(axis=0)
    q_hat = np.where(B0.T > 0, q / np.where(B0.T > 0, B0.T, 1.0), 1.0)
    qV = F[:, :, iv]
    blocks: dict[str, float] = {}

    interm_cost = (wedges.tau * Z) * p[:, None]
    cost_j = interm_cost.sum(axis=0) * y_hat + w[cty] * VA * y_hat + TLS * p * y_hat
    revenue_j = p * y0 * y_hat
    zp_err = np.abs(revenue_j - cost_j)[act] / np.maximum(p[act] * y[act], cell_floor)
    blocks["zero_profit_active"] = float(np.max(zp_err)) if zp_err.size else 0.0

    endo_fd_demand = np.einsum("ikf,fk->i", Fe, q_hat)
    deliveries = Z @ y_hat + endo_fd_demand + qV.sum(axis=1)
    blocks["goods_clearing"] = float(np.max(np.abs(y - deliveries) / np.maximum(y, cell_floor)))

    va_employed = np.bincount(cty, weights=VA * y_hat, minlength=n)
    blocks["factor_markets"] = float(np.max(np.abs(va_employed - calib.Fbar) / calib.Fbar))

    blocks["household_budget"] = float(np.max(np.abs(E.sum(axis=0) + EV - (Y - XN)) / calib.Y0))

    PG = state.P[ENDOGENOUS_FD.index("G"), :]
    PT_raw = (np.bincount(cty, weights=TLS * p * y_hat, minlength=n)
              + np.sum(TFD[:, endo] * (state.P / calib.P0).T * q_hat.T, axis=1)
              + TFD[:, iv] * PG / calib.P0[1])
    Z_flow = Z * y_hat[None, :] * p[:, None]
    TRI_raw = np.bincount(cty, weights=((wedges.tau - 1.0) * Z_flow).sum(axis=0), minlength=n)
    Fe_flow = Fe * q_hat.T[None, :, :] * p[:, None, None]
    TRF_raw = ((wedges.tau_fd - 1.0) * Fe_flow).sum(axis=(0, 2))
    TRV_raw = np.einsum("id,i->d", (tau_V - 1.0) * qV, p)
    TR_raw = TRI_raw + TRF_raw + TRV_raw
    blocks["national_income"] = float(np.max(np.abs(Y - w * calib.Fbar - PT_raw - TR_raw) / calib.Y0))
    blocks["tariff_revenue_identity"] = float(np.max(np.abs(TR - TR_raw) / calib.Y0))

    if closure == "factor":
        target_xn = calib.s_factor * float(w @ calib.Fbar)
    elif closure == "gdp":
        target_xn = calib.s_gdp * float(Y.sum())
    else:
        target_xn = calib.s_own * Y
        target_xn[calib.reference] = -(target_xn.sum() - target_xn[calib.reference])
    blocks["closure_equation"] = float(np.max(np.abs(XN - target_xn) / calib.Y0))

    Z_val = (Z * y_hat[None, :]) * p[:, None]
    FD_flow = (Fe * q_hat.T[None, :, :]).sum(axis=2) * p[:, None] + qV * p[:, None]
    bilateral = Z_val.reshape(n, s, n, s).sum(axis=(1, 3)) + FD_flow.reshape(n, s, n).sum(axis=1)
    NX = bilateral.sum(axis=1) - bilateral.sum(axis=0)
    blocks["current_account_identity"] = float(np.max(np.abs(NX - XN) / calib.Y0))

    y0_sum = float(calib.Y0.sum())
    blocks["world_adding_up"] = max(float(abs(XN.sum())) / y0_sum, float(abs(NX.sum())) / y0_sum)

    if kind == "factor":
        fbar_sum = float(calib.Fbar.sum())
        blocks["numeraire_identity"] = float(abs(w @ calib.Fbar - fbar_sum) / fbar_sum)
    else:
        blocks["numeraire_identity"] = float(abs(w[numeraire_country] - 1.0))

    cert = RawFlowCertificate(**blocks, tol=float(tol), cell_floor=cell_floor,
                              metadata={"closure": closure, "numeraire": numeraire, "inventories": inventories,
                                        "baseline_tariffs_separated": calib.baseline_tariffs is not None})
    if strict and not cert.passed:
        failed = {b: blocks[b] for b in cert.failed_blocks}
        raise CertificationFailure(
            f"certificate failed in {len(failed)} block(s) (tol={tol:.0e}): {failed}", blocks=blocks
        )
    return cert


__all__ = ["RawFlowCertificate", "certify_raw_flows"]
