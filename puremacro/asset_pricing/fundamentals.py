"""Availability-dated fundamentals and explicit cash-flow valuation assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ._research import ResearchReport, finite, readonly

__all__ = ["fundamentals_asof", "fundamental_ratios", "ValuationResult",
           "discounted_cash_flow", "dividend_discount", "valuation_sensitivity"]


def fundamentals_asof(facts: pd.DataFrame, as_of: Any) -> pd.DataFrame:
    """Select latest available revisions, then latest period by asset/metric/unit/type.

    Required columns: asset, metric, unit, period_type (instant/quarterly/annual/
    ttm), period_end, available_at, value. Duration facts also require period_start;
    instant facts leave it missing. Availability means public information time,
    not fiscal period end. Naive times are interpreted as UTC. A date-only as_of
    is midnight, not end-of-day. Additional provenance columns are preserved.

    Revisions are selected within each exact reporting interval before choosing
    the latest period. Annual, quarterly, TTM and currency units remain separate;
    this function does not infer fiscal quarters, convert YTD figures, construct
    TTM, normalize custom taxonomies or equate accounting definitions.
    """
    required = {"asset","metric","unit","period_type","period_end","available_at","value"}
    if not isinstance(facts,pd.DataFrame) or not required.issubset(facts.columns):
        raise ValueError(f"facts require columns {sorted(required)}")
    out = facts.copy(deep=True)
    for column in ("asset","metric","unit","period_type"):
        if not out[column].map(lambda x: isinstance(x,str) and bool(x)).all():
            raise ValueError(f"{column} must contain nonempty strings")
    if not out.period_type.isin(["instant","quarterly","annual","ttm"]).all():
        raise ValueError("unknown period_type")
    if "period_start" not in out:
        out["period_start"] = pd.NaT
    for column in ("period_start","period_end","available_at"):
        out[column] = pd.to_datetime(out[column],utc=True,errors="raise")
    cutoff = pd.to_datetime(as_of,utc=True)
    if not isinstance(cutoff,pd.Timestamp) or pd.isna(cutoff):
        raise ValueError("as_of must be one finite timestamp")
    duration = out.period_type != "instant"
    if (out[["period_end","available_at"]].isna().any().any()
            or out.loc[duration,"period_start"].isna().any()
            or (out.loc[duration,"period_start"] > out.loc[duration,"period_end"]).any()
            or (out.period_end > out.available_at).any()
            or out.loc[~duration,"period_start"].notna().any()):
        raise ValueError("inconsistent reporting interval or availability time")
    out["value"] = pd.to_numeric(out.value,errors="raise")
    if not np.isfinite(out.value.to_numpy(dtype=float)).all():
        raise ValueError("fact values must be finite")
    group = ["asset","metric","unit","period_type"]
    interval = group+["period_start","period_end"]
    if out.duplicated(interval+["available_at"]).any():
        raise ValueError("ambiguous duplicate fact/revision timestamps")
    out = out.loc[out.available_at <= cutoff]
    out = out.sort_values("available_at").drop_duplicates(interval,keep="last")
    # Different starts sharing an end cannot be silently treated as equal periods.
    if out.duplicated(group+["period_end"]).any():
        raise ValueError("ambiguous durations with the same period end; normalize explicitly")
    return out.sort_values(["period_end","available_at"]).drop_duplicates(group,keep="last").sort_values(group).reset_index(drop=True)


def fundamental_ratios(*, price: float, shares_outstanding: float, revenue: float,
                       net_income: float, book_equity: float, debt: float, cash: float,
                       operating_cash_flow: float, capital_expenditure: float,
                       ebitda: float | None = None) -> pd.DataFrame:
    """Ratios from explicitly comparable accounting inputs, without guessed data.

    Monetary inputs use one currency and scale; flow inputs use the same period.
    Capital expenditure is positive cash outflow. Enterprise value is equity
    market capitalization + debt - cash (no preferred/minority adjustments are
    inferred). ROE here uses supplied book equity, not average book equity.
    Negative/zero earnings, equity, revenue or EBITDA denominators yield NaN
    with a status, rather than misleading negative valuation multiples.
    """
    price = finite(price,"price",0)
    shares = finite(shares_outstanding,"shares_outstanding",0)
    if price == 0 or shares == 0:
        raise ValueError("price and shares_outstanding must be positive")
    values = {"revenue":revenue,"net_income":net_income,"book_equity":book_equity,
              "debt":debt,"cash":cash,"operating_cash_flow":operating_cash_flow,
              "capital_expenditure":capital_expenditure}
    values = {k:finite(v,k,0 if k in ("debt","cash","capital_expenditure") else None) for k,v in values.items()}
    market = price*shares
    enterprise = market+values["debt"]-values["cash"]
    ebit = None if ebitda is None else finite(ebitda,"ebitda")
    rows = [("market_cap",market,"ok"),("enterprise_value",enterprise,"ok")]
    for metric,num,den in [
        ("price_earnings",market,values["net_income"]),
        ("price_book",market,values["book_equity"]),
        ("price_sales",market,values["revenue"]),
        ("ev_ebitda",enterprise,ebit),
        ("profit_margin",values["net_income"],values["revenue"]),
        ("return_on_equity",values["net_income"],values["book_equity"]),
        ("debt_equity",values["debt"],values["book_equity"]),
        ("free_cash_flow_yield",values["operating_cash_flow"]-values["capital_expenditure"],market),
    ]:
        status = "missing_denominator" if den is None else "nonpositive_denominator" if den <= 0 else "ok"
        rows.append((metric,float(num/den) if status == "ok" and den is not None else np.nan,status))
    return pd.DataFrame(rows,columns=["metric","value","status"])


@dataclass(frozen=True)
class ValuationResult(ResearchReport):
    kind: str
    enterprise_value: float
    equity_value: float
    value_per_share: float
    forecast_pv: float
    terminal_pv: float
    terminal_fraction: float
    shares_outstanding: float
    net_debt: float
    cash_flows: np.ndarray
    discount_rates: np.ndarray
    terminal_growth: float | None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"metric":["enterprise_value","equity_value","value_per_share",
                                       "forecast_pv","terminal_pv","terminal_fraction"],
                             "value":[self.enterprise_value,self.equity_value,self.value_per_share,
                                      self.forecast_pv,self.terminal_pv,self.terminal_fraction]})


def discounted_cash_flow(cash_flows: Sequence[float], discount_rate: float | Sequence[float], *,
                        terminal_growth: float | None = None, kind: str = "firm",
                        net_debt: float = 0, shares_outstanding: float = 1) -> ValuationResult:
    """Discount end-of-period cash flows with an optional Gordon terminal value.

    kind='firm': FCFF discounted at WACC, then subtract supplied net debt.
    kind='equity': FCFE discounted at cost of equity; no debt subtraction.
    kind='dividend': total dividends discounted at cost of equity, no debt subtraction.
    Rates may be scalar or one per forecast period; discount factors compound.
    Terminal cash flow is last_forecast*(1+growth), discounted at the LAST rate,
    which is assumed permanent beyond the forecast. Require last_rate>growth.
    All cash flows/rates are in matching currency, inflation and period units.
    Outputs are conditional valuations, not confidence intervals or predictions.
    Negative equity values remain visible; no liability-limited floor is imposed.
    """
    if kind not in ("firm","equity","dividend"):
        raise ValueError("kind must be firm, equity or dividend")
    cf = np.asarray(cash_flows,dtype=float)
    if cf.ndim != 1 or not len(cf) or not np.isfinite(cf).all():
        raise ValueError("cash_flows must be a nonempty finite vector")
    if kind == "dividend" and np.any(cf < 0):
        raise ValueError("dividends cannot be negative")
    rates = np.asarray(discount_rate,dtype=float)
    if rates.ndim == 0:
        rates = np.full(len(cf),float(rates))
    if rates.shape != cf.shape or not np.isfinite(rates).all() or np.any(rates <= -1):
        raise ValueError("discount rates must match cash flows and be finite and above -1")
    debt = finite(net_debt,"net_debt")
    shares = finite(shares_outstanding,"shares_outstanding",0)
    if shares == 0:
        raise ValueError("shares_outstanding must be positive")
    if kind != "firm" and debt != 0:
        raise ValueError("equity/dividend cash flows must not subtract net debt again")
    discount = np.cumprod(1+rates)
    if not np.isfinite(discount).all() or np.any(discount <= 0):
        raise ValueError("discount factors overflowed or underflowed")
    forecast = float(np.sum(cf/discount))
    terminal = 0.
    if terminal_growth is not None:
        growth = finite(terminal_growth,"terminal_growth")
        if growth <= -1 or growth >= rates[-1]:
            raise ValueError("terminal_growth must be above -1 and below the terminal discount rate")
        terminal = float(cf[-1]*(1+growth)/(rates[-1]-growth)/discount[-1])
    total = forecast+terminal
    equity = total-debt if kind == "firm" else total
    if not np.isfinite([forecast,terminal,equity,equity/shares]).all():
        raise ValueError("valuation overflowed")
    return ValuationResult(kind,total if kind == "firm" else np.nan,equity,equity/shares,
        forecast,terminal,terminal/total if total != 0 else np.nan,shares,debt,
        readonly(cf),readonly(rates),terminal_growth)


def dividend_discount(dividend_next: float, cost_of_equity: float, growth: float = 0) -> ValuationResult:
    """Constant-growth dividend value per share D1/(ke-g); shares normalized to 1."""
    dividend = finite(dividend_next,"dividend_next",0)
    return discounted_cash_flow([dividend],cost_of_equity,terminal_growth=growth,kind="dividend")


def valuation_sensitivity(cash_flows: Sequence[float], discount_rates: Sequence[float],
                          growth_rates: Sequence[float], **kwargs: Any) -> pd.DataFrame:
    """Scenario grid, not probability intervals; invalid rate-growth pairs retained.

    Other invalid inputs raise. Rates are constant within each scenario.
    """
    if "terminal_growth" in kwargs or "discount_rate" in kwargs:
        raise ValueError("sensitivity supplies terminal_growth and discount_rate")
    rows = []
    for rate in discount_rates:
        rate = finite(rate,"discount_rate")
        # Validate the full base valuation even if every terminal scenario fails.
        discounted_cash_flow(cash_flows,rate,**kwargs)
        for growth in growth_rates:
            growth = finite(growth,"terminal_growth")
            valid = -1 < growth < rate
            value = discounted_cash_flow(cash_flows,rate,terminal_growth=growth,**kwargs) if valid else None
            rows.append({"discount_rate":rate,"terminal_growth":growth,
                         "value_per_share":value.value_per_share if value else np.nan,
                         "terminal_fraction":value.terminal_fraction if value else np.nan,
                         "status":"ok" if valid else "invalid_terminal_growth"})
    return pd.DataFrame(rows,columns=["discount_rate","terminal_growth","value_per_share","terminal_fraction","status"])
