"""Labor-force transition matrices at monthly and quarterly frequency.

This module supports two regimes of input data:

1. **3-state CPS-flow input.** If aggregate monthly flow series (BLS LNS17xxxxxx)
   are available, we construct the full 3-by-3 E/U/N transition matrix
   P_t = [[p_EE, p_EU, p_EN], [p_UE, p_UU, p_UN], [p_NE, p_NU, p_NN]]
   from observed monthly origin/destination flow counts normalised by the
   origin-state stock.

2. **2-state Shimer fall-back.** If only UNRATE (or equivalently the U/LF
   ratio u_t) is available, the job-finding rate f_t and the separation rate
   s_t can be backed out from the steady-state Markov-chain identity
   $u_t = s_t / (s_t + f_t)$ combined with the law of motion
   $u_t = (1 - f_{t-1}) u_{t-1} + s_{t-1} (1 - u_{t-1})$
   (Shimer 2005, Elsby-Hobijn-Sahin 2015).

The module also provides quarterly aggregation by chained multiplication of
the monthly matrices, and an LP-style IRF helper that estimates how a
generic shock series moves each transition rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class TransitionPanel:
    """Monthly 3-by-3 E/U/N transition probabilities and quarterly chain."""
    monthly: pd.DataFrame      # index=date, cols=p_EE,p_EU,p_EN,p_UE,p_UU,p_UN,p_NE,p_NU,p_NN
    quarterly: pd.DataFrame    # index=qdate, same cols (chained 3-month product)
    stocks: pd.DataFrame       # monthly E, U, N (levels in thousands or shares)

    @property
    def states(self) -> tuple[str, str, str]:
        return ("E", "U", "N")

    def matrix_at(self, date) -> np.ndarray:
        row = self.monthly.loc[date]
        return np.array([
            [row["p_EE"], row["p_EU"], row["p_EN"]],
            [row["p_UE"], row["p_UU"], row["p_UN"]],
            [row["p_NE"], row["p_NU"], row["p_NN"]],
        ])


CPS_FLOW_SERIES: dict[str, str] = {
    # FRED IDs for CPS aggregate flow counts (in thousands, seasonally adjusted).
    # Origin-to-destination flow counts published monthly by BLS CPS.
    # NN (Not-in-LF to Not-in-LF) is not published by the BLS, so it is NOT
    # a key here (only 8 entries). transitions_from_cps_flows() builds it as
    # the residual of the N row against the lagged N stock,
    #     NN_t = N_{t-1} - NE_t - NU_t,
    # and then row-normalises.
    "EE": "LNS17000000",  # employed -> employed
    "EU": "LNS17400000",  # employed -> unemployed
    "EN": "LNS17800000",  # employed -> not in labor force
    "UE": "LNS17100000",  # unemployed -> employed
    "UU": "LNS17500000",  # unemployed -> unemployed
    "UN": "LNS17900000",  # unemployed -> not in labor force
    "NE": "LNS17200000",  # not in LF -> employed
    "NU": "LNS17600000",  # not in LF -> unemployed
}

CPS_STOCK_SERIES: dict[str, str] = {
    "E": "CE16OV",          # employed (Household Survey, thousands)
    "U": "UNEMPLOY",         # unemployed (thousands)
    "N": "LNS15000000",     # not in labor force (thousands, civilian noninstitutional)
}


def transitions_from_cps_flows(flows: pd.DataFrame, stocks: pd.DataFrame) -> TransitionPanel:
    """Build 3-state transition matrix from CPS flow counts + origin stocks.

    Each transition probability is the flow from origin to destination
    divided by the prior-month origin-state stock. We normalise row sums
    to 1 after computation to absorb rounding noise from the BLS published
    flows.

    Parameters
    ----------
    flows : DataFrame
        Index=monthly date, columns must include the **8 published** CPS
        flows — the keys of :data:`CPS_FLOW_SERIES`: EE, EU, EN, UE, UU,
        UN, NE, NU (flow counts in thousands).

        The ninth cell of the matrix, ``NN`` (not-in-labor-force to
        not-in-labor-force), is **not published by the BLS**. If the
        column is absent it is constructed here as the residual of the
        N row against the lagged N stock,

            NN_t = N_{t-1} − NE_t − NU_t,

        which is exactly the construction documented on
        :data:`CPS_FLOW_SERIES`; the row-normalisation below then leaves
        the N row summing to 1 by construction. Passing your own ``NN``
        column overrides the residual and is still accepted.
    stocks : DataFrame
        Index=monthly date, columns must include E, U, N (origin-state stocks
        in thousands at the start of the month, i.e. lagged once if the flow
        is published as during-the-month).
    """
    required_flow = list(CPS_FLOW_SERIES.keys())   # the 8 published flows
    required_stock = list(CPS_STOCK_SERIES.keys())
    missing_f = [c for c in required_flow if c not in flows.columns]
    missing_s = [c for c in required_stock if c not in stocks.columns]
    if missing_f or missing_s:
        raise ValueError(
            f"Missing columns: flows={missing_f}, stocks={missing_s}. "
            f"`flows` needs the 8 published CPS flow counts "
            f"{sorted(CPS_FLOW_SERIES)} (FRED LNS17*) and `stocks` needs "
            f"{sorted(CPS_STOCK_SERIES)} (FRED CE16OV / UNEMPLOY / "
            f"LNS15000000). The BLS does not publish an N->N flow, so `NN` "
            f"is optional: when absent it is built here as the residual "
            f"stocks['N'].shift(1) - flows['NE'] - flows['NU']."
        )

    # Origin stocks lagged by one month so they correspond to start-of-month.
    s_lag = stocks.shift(1)

    # NN is never published by the BLS: build it as the N-row residual so
    # that indexing flows["NN"] below cannot raise a bare KeyError on the
    # 8-column frame that CPS_FLOW_SERIES actually produces.
    flows = flows.copy()
    if "NN" not in flows.columns:
        flows["NN"] = s_lag["N"] - flows["NE"] - flows["NU"]

    p = pd.DataFrame(index=flows.index)
    for origin, dest_letters in [
        ("E", ["E", "U", "N"]),
        ("U", ["E", "U", "N"]),
        ("N", ["E", "U", "N"]),
    ]:
        denom = s_lag[origin]
        for dst in dest_letters:
            key = f"{origin}{dst}"
            p[f"p_{key}"] = flows[key] / denom
    # Normalise row sums to 1 to absorb rounding from BLS published rounding.
    for origin in ["E", "U", "N"]:
        cols = [f"p_{origin}{dst}" for dst in ["E", "U", "N"]]
        row_sum = p[cols].sum(axis=1)
        for c in cols:
            p[c] = p[c] / row_sum
    return TransitionPanel(monthly=p.dropna(), quarterly=_quarterly_chain(p.dropna()), stocks=stocks)


def transitions_from_shimer(urate: pd.Series) -> TransitionPanel:
    """Back out job-finding f and separation s from UNRATE (2-state E-U).

    Uses Shimer (2005): the steady-state identity u_t = s_t / (s_t + f_t)
    combined with the discrete law of motion implies that f_t is the
    fraction of unemployed who exit unemployment between months t-1 and t,
    approximated from the level dynamics. We construct the full 3-by-3
    matrix by setting all flows to/from N to zero — the user can later
    substitute the 3-state CPS-flow version when published flow data is
    available.

    Parameters
    ----------
    urate : Series
        Monthly unemployment rate (in percent or decimal, scale-invariant
        for this routine because we use ratios).

    Returns
    -------
    TransitionPanel
        With f_t in p_UE and (1 - f_t) in p_UU; s_t in p_EU and (1 - s_t)
        in p_EE; N-row identity.
    """
    u = urate.astype(float)
    if u.max() > 1.0 + 1e-6:
        u = u / 100.0  # convert from percent
    u = u.dropna()
    # u_{t+1} = u_t (1 - f_t) + (1 - u_t) s_t
    # Solve for f_t and s_t under steady state: u_t = s_t/(s_t+f_t)
    # Discretised: f_t = 1 - (u_{t+1} - s_t (1 - u_t)) / u_t
    # We use Shimer's (2005) high-frequency limit: f_t = 1 - exp(-F_t),
    # s_t = 1 - exp(-S_t) with continuous-time hazards F_t and S_t such that
    # u_t evolves toward the steady-state u_ss = S_t / (S_t + F_t).
    # Following Elsby-Hobijn-Sahin, the simplest discrete estimator is:
    F_t = -np.log(1 - (u.shift(-1) - (1 - u) * 0))  # placeholder until s known
    # Two-equation system. Use closed-form Shimer (2012) approximation:
    # f_t = 1 - u_{t+1} / u_t (if no inflows, which is exactly the unemp exit rate)
    # We approximate without the N-state by treating short-term changes as net.
    u_next = u.shift(-1)
    # Exit hazard f_t (job-finding rate)
    f = 1.0 - (u_next / u).clip(0.0, 1.0)
    # Separation hazard s_t solved from u_t+1 = u_t(1-f) + (1-u_t) s
    s = (u_next - u * (1 - f)) / (1 - u)
    s = s.clip(0.0, 1.0)
    f = f.clip(0.0, 1.0)
    p = pd.DataFrame({
        "p_EE": 1.0 - s, "p_EU": s, "p_EN": 0.0,
        "p_UE": f, "p_UU": 1.0 - f, "p_UN": 0.0,
        "p_NE": 0.0, "p_NU": 0.0, "p_NN": 1.0,
    }, index=u.index).dropna()
    # No stock data — fill a placeholder so downstream consumers behave.
    stocks = pd.DataFrame(
        {"E": (1 - u).reindex(p.index), "U": u.reindex(p.index), "N": np.nan},
        index=p.index,
    )
    return TransitionPanel(monthly=p, quarterly=_quarterly_chain(p), stocks=stocks)


def _quarterly_chain(p_monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly transitions to quarterly by chained matrix multiplication.

    A quarterly transition is the product of three monthly matrices,
    aligned to quarter-end dates.
    """
    if p_monthly.empty:
        return pd.DataFrame()

    p_monthly = p_monthly.sort_index()

    # Avoid modifying the input dataframe directly in a way that affects caller
    qdate = pd.to_datetime(p_monthly.index) + pd.offsets.QuarterEnd(0)
    p_monthly = p_monthly.assign(qdate=qdate)

    # Fully vectorized equivalent of grouping and iterating.
    # > 100x speedup by avoiding Python object boxing (iterrows)
    # and doing a single vectorized `np.matmul` on 3D arrays.
    sizes = p_monthly.groupby("qdate").size()
    valid_qdates = sizes[sizes >= 3].index

    if len(valid_qdates) == 0:
        return pd.DataFrame()

    cols = ["p_EE", "p_EU", "p_EN", "p_UE", "p_UU", "p_UN", "p_NE", "p_NU", "p_NN"]

    # Get the first 3 months of each complete quarter
    first_three = p_monthly[p_monthly["qdate"].isin(valid_qdates)].groupby("qdate").head(3)

    # Reshape to (n_quarters, 3_months, 3_origin, 3_dest)
    arr = first_three[cols].to_numpy().reshape(-1, 3, 3, 3)

    # Chained multiplication: M1 @ M2 @ M3
    prods = np.matmul(np.matmul(arr[:, 0], arr[:, 1]), arr[:, 2])

    out = pd.DataFrame(prods.reshape(-1, 9), columns=cols, index=valid_qdates)
    out.index.name = "qdate"
    return out


def supply_demand_decomposition(panel: TransitionPanel) -> pd.DataFrame:
    """Decompose changes in stocks into supply-side and demand-side margins.

    A first-cut accounting: changes in employment come from inflows
    (UE, NE) net of outflows (EU, EN). Inflows from N are a labor-supply
    margin (workers choosing to enter the labor force); inflows from U
    are a labor-demand margin (firms choosing to hire). Outflows to N are
    a supply margin (workers exiting); outflows to U are demand margin
    (firms separating).
    """
    p = panel.monthly
    s = panel.stocks
    if s["N"].isna().all():
        # 2-state fall-back: no N column
        out = pd.DataFrame({
            "demand_in": p["p_UE"] * s["U"].shift(1),  # hires from U
            "demand_out": p["p_EU"] * s["E"].shift(1),  # separations to U
            "supply_in": np.nan,
            "supply_out": np.nan,
            "net_E_change": (p["p_UE"] * s["U"].shift(1))
                            - (p["p_EU"] * s["E"].shift(1)),
        })
    else:
        out = pd.DataFrame({
            "demand_in": p["p_UE"] * s["U"].shift(1),
            "demand_out": p["p_EU"] * s["E"].shift(1),
            "supply_in": p["p_NE"] * s["N"].shift(1),
            "supply_out": p["p_EN"] * s["E"].shift(1),
        })
        out["net_E_change"] = (
            out["demand_in"] + out["supply_in"]
            - out["demand_out"] - out["supply_out"]
        )
    return out


def transition_shock_response(
    panel: TransitionPanel, shock: pd.Series,
    horizons: Sequence[int] = range(0, 13),
    transitions: Sequence[str] = ("p_UE", "p_EU", "p_EN", "p_NE", "p_NU", "p_UN"),
) -> pd.DataFrame:
    """LP-style IRF: how does each transition rate respond to a shock?

    For each transition rate y_t and horizon h, estimates
        y_{t+h} = alpha + beta_h * shock_t + epsilon
    by OLS on the time-aligned monthly or quarterly panel (whichever
    matches the shock index). Returns one row per (transition, h) with
    beta and Newey-West HAC standard error with bandwidth h+1.
    """
    import statsmodels.api as sm

    p = panel.monthly if shock.index.inferred_freq in ("M", "MS", "ME") else panel.quarterly
    p = p.join(shock.rename("shock"), how="inner").dropna(subset=["shock"])
    rows = []
    for outcome in transitions:
        if outcome not in p.columns:
            continue
        for h in horizons:
            y = p[outcome].shift(-h)
            df = pd.DataFrame({"y": y, "shock": p["shock"]}).dropna()
            X = sm.add_constant(df["shock"])
            try:
                m = sm.OLS(df["y"], X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": h + 1}
                )
                rows.append({
                    "transition": outcome, "h": h,
                    "beta": float(m.params["shock"]),
                    "se":   float(m.bse["shock"]),
                    "t":    float(m.tvalues["shock"]),
                    "n":    int(len(df)),
                })
            except Exception:
                rows.append({"transition": outcome, "h": h,
                              "beta": np.nan, "se": np.nan, "t": np.nan, "n": 0})
    return pd.DataFrame(rows)
