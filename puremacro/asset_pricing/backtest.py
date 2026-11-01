"""Long-only historical portfolio accounting with explicit execution timing.

Returns at row t cover the interval ending t. Target decisions are dated after
that close. Default lag=2 executes at the next close and earns the subsequent
interval. Lag=1 explicitly assumes execution at the decision close. This is a
period accounting model, not an intraday execution or broker simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ._research import ResearchReport, dated_frame, finite, integer
from ._validate import _validate_returns

__all__ = ["BacktestResult", "backtest_portfolio", "walk_forward_weights", "replay_backtest"]


def _encode(frame: pd.DataFrame) -> dict[str, Any]:
    frequency = frame.index.freqstr
    # A custom holiday calendar cannot be reconstructed from the generic 'C'
    # alias. Its exact observed dates survive; omit a misleading frequency hint.
    if frequency is not None and pd.tseries.frequencies.to_offset(frequency) != frame.index.freq:
        frequency = None
    return {"dates": [x.isoformat() for x in frame.index],
            "timezone": str(frame.index.tz) if frame.index.tz is not None else None,
            "frequency": frequency, "columns": list(frame.columns),
            "values": [[None if pd.isna(x) else float(x) for x in row]
                       for row in frame.to_numpy()]}


def _decode(record: dict[str, Any]) -> pd.DataFrame:
    zone = record.get("timezone")
    dates = pd.to_datetime(record["dates"], utc=zone is not None)
    if zone is not None:
        dates = dates.tz_convert(zone)
    dates = pd.DatetimeIndex(dates, freq=record.get("frequency"))
    return pd.DataFrame(record["values"], index=dates,
                        columns=record["columns"], dtype=float)


def _json(record: Any) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BacktestResult(ResearchReport):
    """Owned ledger/weights/trades and a JSON replay recipe, without callbacks.

    Frozen attributes do not make pandas objects deeply immutable. ``to_json``
    exports the original input recipe independently of edits to display frames.
    Costs and turnover in the ledger are fractions of wealth before trading;
    weights are fractions after costs, before the period return.
    """

    ledger: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    input_sha256: str
    _recipe_json: str

    def to_dataframe(self) -> pd.DataFrame:
        return self.ledger.rename_axis("date").reset_index().copy()

    def to_json(self) -> str:
        return _json({"schema": "puremacro.backtest.v1", "input_sha256": self.input_sha256,
                      "recipe": json.loads(self._recipe_json)})

    def plot(self, ax: Any = None, show: bool = False):
        from ._results import _new_ax, _finish
        fig, ax, given = _new_ax(ax)
        ax.plot(self.ledger.index, self.ledger["wealth"])
        ax.set(xlabel="date", ylabel="wealth after costs", title="Historical portfolio")
        return _finish(fig, ax, given, show)


def backtest_portfolio(
    returns: pd.DataFrame, targets: pd.DataFrame, *, execution_lag: int = 2,
    cost_bps: float = 0.0, cash_returns: float | pd.Series = 0.0,
    initial_wealth: float = 1.0, eligibility: pd.DataFrame | None = None,
    check_scale: bool = True,
) -> BacktestResult:
    """Evaluate sparse, dated target weights with self-financing cash accounting.

    All weights are nonnegative and sum to at most one; the residual is cash.
    A wholly NaN target row means no trade. Other target rows must be complete.
    No-trade intervals retain actual holdings, allowing weights to drift.
    Missing returns are permitted only for assets with exactly zero exposure.
    A -100% return is allowed; values below -100% are invalid.

    ``eligibility`` has the same index/columns as returns and contains booleans
    known at the start of each interval. An ineligible held/targeted asset is
    an error: there is no invented liquidation price. The caller must supply
    actual delisting returns and timely liquidation decisions.

    Proportional costs charge each risky purchase/sale (not the cash leg).
    For pretrade risky weights v and desired post-cost weights w, solve
    q + c*sum(abs(q*w-v)) = 1. Wealth after costs is q times pretrade wealth.
    Targets beyond the last executable interval are retained in the recipe
    but produce no trades. External targets may contain user-created look-ahead;
    a lag alone cannot certify their information set.
    """
    r = dated_frame(returns, "returns", missing=True)
    w = dated_frame(targets, "targets", missing=True)
    lag = integer(execution_lag, "execution_lag")
    cost = finite(cost_bps, "cost_bps", 0) / 10_000
    if cost >= 1:
        raise ValueError("cost_bps must be below 10000")
    wealth = finite(initial_wealth, "initial_wealth", 0)
    if wealth == 0:
        raise ValueError("initial_wealth must be positive")
    if not r.columns.equals(w.columns) or not w.index.isin(r.index).all():
        raise ValueError("targets must use the exact return columns and a subset of return dates")
    a = r.to_numpy()
    if np.any(a < -1):
        raise ValueError("simple returns cannot be below -1")
    if check_scale:
        for col in r:
            observed = r[col].dropna()
            if len(observed):
                _validate_returns(observed, f"returns[{col}]")
    scheduled = w.reindex(r.index)
    complete = scheduled.notna().all(axis=1)
    if (scheduled.notna().any(axis=1) & ~complete).any():
        raise ValueError("a target row must be wholly missing or wholly specified")
    specified = scheduled.loc[complete].to_numpy()
    if np.any(specified < 0) or np.any(specified.sum(axis=1) > 1 + 1e-12):
        raise ValueError("target weights must be nonnegative and sum to at most one")
    # Roundoff-only normalization, never an economic leverage correction.
    if len(specified):
        scheduled.loc[complete] = specified / np.maximum(1, specified.sum(axis=1))[:, None]
    if isinstance(cash_returns, pd.Series):
        if not cash_returns.index.equals(r.index):
            raise ValueError("cash_returns must have exactly the return dates")
        rf = cash_returns.to_numpy(dtype=float, copy=True)
    else:
        rf = np.full(len(r), finite(cash_returns, "cash_returns"))
    if not np.isfinite(rf).all() or np.any(rf <= -1):
        raise ValueError("cash_returns must be finite and above -1")
    if eligibility is None:
        allowed = np.ones(r.shape, dtype=bool)
    else:
        if (not eligibility.index.equals(r.index) or not eligibility.columns.equals(r.columns)
                or not all(pd.api.types.is_bool_dtype(d) for d in eligibility.dtypes)
                or eligibility.isna().any().any()):
            raise ValueError("eligibility must be an exactly aligned boolean DataFrame")
        allowed = eligibility.to_numpy(dtype=bool, copy=True)
    active = scheduled.shift(lag).to_numpy()
    units = np.zeros(r.shape[1])  # asset wealth, not share counts
    cash = wealth
    held = np.zeros(r.shape)
    trades = np.zeros(r.shape)
    rows = []
    for t, date in enumerate(r.index):
        before = wealth
        previous = units / before
        target = active[t]
        traded = not np.isnan(target).all()
        q = 1.0
        if traded:
            q = brentq(lambda z: z + cost * np.abs(z * target - previous).sum() - 1,
                       0.0, 1.0, xtol=5e-15)
            trades[t] = q * target - previous
            units = before * q * target
            cash = before * q * (1 - target.sum())
        exposure = units / (before * q)
        if np.any((units > 0) & ~allowed[t]):
            raise ValueError(f"ineligible exposure at {date}; supply a timely liquidation decision")
        if np.any((units > 0) & ~np.isfinite(a[t])):
            raise ValueError(f"missing return for a held asset at {date}")
        held[t] = exposure
        turnover = float(np.abs(trades[t]).sum())
        fee = cost * turnover
        # Missing unheld values contribute exactly zero, not a fabricated return.
        pnl = float(np.dot(units[units > 0], a[t, units > 0]))
        cash_pnl = cash * rf[t]
        gross = (pnl + cash_pnl) / (before * q)
        units = np.where(units > 0, units * (1 + np.nan_to_num(a[t], nan=0.0)), 0.0)
        cash += cash_pnl
        wealth = float(units.sum() + cash)
        if not np.isfinite(wealth) or wealth <= 0:
            raise ValueError(f"portfolio exhausted or nonfinite at {date}")
        rows.append((gross, wealth / before - 1, wealth, turnover, fee,
                     1 - exposure.sum(), abs(wealth - before * (1 - fee) * (1 + gross))))
    ledger = pd.DataFrame(rows, index=r.index, columns=["gross_return", "net_return", "wealth",
                          "turnover", "cost", "cash_weight", "accounting_residual"])
    recipe = {"returns": _encode(r), "targets": _encode(w), "execution_lag": lag,
              "cost_bps": float(cost_bps), "cash_returns": _encode(pd.DataFrame({"cash": rf}, index=r.index)),
              "initial_wealth": float(initial_wealth), "eligibility": allowed.astype(int).tolist(),
              "check_scale": bool(check_scale)}
    encoded = _json(recipe)
    return BacktestResult(ledger, pd.DataFrame(held, index=r.index, columns=r.columns),
                          pd.DataFrame(trades, index=r.index, columns=r.columns),
                          hashlib.sha256(encoded.encode()).hexdigest(), encoded)


def replay_backtest(record: str) -> BacktestResult:
    """Verify and rerun a JSON input recipe; never deserialize executable code.

    The hash detects accidental changes, not malicious tampering. Results are
    recomputed by the currently installed engine; this is not a version pin.
    """
    obj = json.loads(record)
    if obj.get("schema") != "puremacro.backtest.v1":
        raise ValueError("unsupported backtest schema")
    recipe = obj["recipe"]
    if hashlib.sha256(_json(recipe).encode()).hexdigest() != obj.get("input_sha256"):
        raise ValueError("backtest input checksum mismatch")
    r = _decode(recipe["returns"])
    return backtest_portfolio(r, _decode(recipe["targets"]), execution_lag=recipe["execution_lag"],
        cost_bps=recipe["cost_bps"], cash_returns=_decode(recipe["cash_returns"]).iloc[:, 0],
        initial_wealth=recipe["initial_wealth"], check_scale=recipe["check_scale"],
        eligibility=pd.DataFrame(recipe["eligibility"], index=r.index, columns=r.columns).astype(bool))


def walk_forward_weights(
    observations: pd.DataFrame, policy: Callable[[pd.DataFrame], pd.Series], *,
    min_history: int = 20, rebalance_every: int = 1, window: int | None = None,
) -> pd.DataFrame:
    """Call a policy on a copy of the available history at each decision close.

    The callback must return a Series indexed by every asset, including explicit
    zero weights for excluded assets. Missing observations remain visible. This
    restricts supplied history, but cannot sandbox a callback's external state.
    Decisions remain unshifted; ``backtest_portfolio`` applies execution lag.
    """
    frame = dated_frame(observations, "observations", missing=True)
    minimum = integer(min_history, "min_history")
    step = integer(rebalance_every, "rebalance_every")
    if window is not None:
        window = integer(window, "window")
        if window < minimum:
            raise ValueError("window cannot be shorter than min_history")
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for t in range(minimum - 1, len(frame), step):
        start = 0 if window is None else max(0, t + 1 - window)
        decision = policy(frame.iloc[start:t + 1].copy())
        if not isinstance(decision, pd.Series) or not decision.index.equals(frame.columns):
            raise ValueError("policy must return a Series with the exact observation columns")
        vals = decision.to_numpy(dtype=float)
        if not np.isfinite(vals).all() or np.any(vals < 0) or vals.sum() > 1 + 1e-12:
            raise ValueError("policy weights must be finite, nonnegative and sum to at most one")
        out.iloc[t] = vals
    return out
