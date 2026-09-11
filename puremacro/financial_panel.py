"""Modular International Financial & Macroprudential Panel Builder.

Constructs aligned multi-country financial panels combining:
1. Sovereign debt yields (10Y and 2Y benchmarks) and term/sovereign spreads
   (via :func:`puremacro.fetch.financial.fetch_sovereign_yields` and
   :func:`puremacro.fetch.financial.compute_sovereign_spreads`).
2. Central bank policy rates
   (via :func:`puremacro.fetch.financial.fetch_policy_rates`).
3. BIS macroprudential indicators: credit-to-GDP gap, total credit, property prices
   (via :func:`puremacro.fetch.financial.fetch_bis_macroprudential`).
4. Financial conditions and systemic risk spreads: TED rate, HY OAS, EM OAS, NFCI
   (via :func:`puremacro.fetch.financial.fetch_financial_conditions`).
5. Standardized global commodity benchmark prices and indices
   (via :func:`puremacro.fetch.wb_pink_sheet.fetch_commodity_benchmarks`).

Supports automated frequency harmonization:
- Monthly ('M') to quarterly ('Q') via period-mean or period-last aggregation.
- Quarterly ('Q') to monthly ('M') via forward-repeat across the quarter's months.

Architectural Invariant:
Zero module-scope ``requests`` imports. All network access routes through
puremacro's cached data layer.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._codes import is_country

_SCHEMA_COLS = ["code", "date", "variable", "value", "sa_source", "source"]
_EMPTY = pd.DataFrame(
    columns=_SCHEMA_COLS + ["is_imputed"]
)


def _norm_var_name(var: str, target_freq: str) -> str:
    """Ensure variable name ends with the target frequency suffix ('_q' or '_m')."""
    if target_freq == "Q":
        if var.endswith("_q"):
            return var
        if var.endswith("_m"):
            return var[:-2] + "_q"
        return f"{var}_q"
    elif target_freq == "M":
        if var.endswith("_m"):
            return var
        if var.endswith("_q"):
            return var[:-2] + "_m"
        return f"{var}_m"
    return var


def _aggregate_monthly_to_quarterly(
    df_m: pd.DataFrame,
    harmonization: str = "mean",
) -> pd.DataFrame:
    """Aggregate monthly financial series to quarterly frequency ('QS').

    Parameters
    ----------
    df_m : pd.DataFrame
        Monthly long-form DataFrame.
    harmonization : str, default 'mean'
        Aggregation rule: 'mean' (quarterly average) or 'last' (end-of-quarter observation).

    Returns
    -------
    pd.DataFrame
        Quarterly long-form DataFrame.
    """
    if df_m.empty:
        return _EMPTY.copy()

    records: list[dict[str, object]] = []

    for (code, var), grp in df_m.groupby(["code", "variable"]):
        s = grp.set_index("date")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]

        use_last = (harmonization == "last") or ("policy_rate" in var) or ("cbrate" in var)
        if use_last:
            q_series = s.resample("QS").last()
            rule_name = "last"
        else:
            q_series = s.resample("QS").mean()
            rule_name = "mean"

        q_series = q_series.dropna()
        if q_series.empty:
            continue

        q_var = _norm_var_name(var, "Q")
        src = f"resampled_from_M({rule_name}):{grp['source'].iloc[0]}"
        sa = grp["sa_source"].iloc[0]

        for dt_val, val in q_series.items():
            records.append({
                "code": code,
                "date": pd.Timestamp(dt_val),
                "variable": q_var,
                "value": float(val),
                "sa_source": sa,
                "source": src,
                "is_imputed": False,
            })

    if not records:
        return _EMPTY.copy()

    out = pd.DataFrame(records)
    return out.drop_duplicates(subset=["code", "date", "variable"], keep="first")


def _project_quarterly_to_monthly(
    df_q: pd.DataFrame,
) -> pd.DataFrame:
    """Project quarterly series to monthly frequency by repeating across the quarter.

    Parameters
    ----------
    df_q : pd.DataFrame
        Quarterly long-form DataFrame with dates on quarter starts (1, 4, 7, 10).

    Returns
    -------
    pd.DataFrame
        Monthly long-form DataFrame with dates on the 1st of each month.
    """
    if df_q.empty:
        return _EMPTY.copy()

    records: list[dict[str, object]] = []

    for _, row in df_q.iterrows():
        base_date = pd.Timestamp(row["date"])
        yr = base_date.year
        q_month = base_date.month
        val = float(row["value"])
        var_name = str(row["variable"])
        m_var = _norm_var_name(var_name, "M")
        src = f"resampled_from_Q:{row['source']}"
        sa = row["sa_source"]

        for offset in (0, 1, 2):
            m = q_month + offset
            if m > 12:
                break
            records.append({
                "code": row["code"],
                "date": pd.Timestamp(f"{yr}-{m:02d}-01"),
                "variable": m_var,
                "value": val,
                "sa_source": "quarterly_ffill" if offset != 0 else sa,
                "source": src,
                "is_imputed": offset != 0,
            })

    if not records:
        return _EMPTY.copy()

    out = pd.DataFrame(records)
    return out.drop_duplicates(subset=["code", "date", "variable"], keep="first")


def build_financial_panel(
    codes: Sequence[str] | None = None,
    *,
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    frequency: str = "Q",
    harmonization: str = "mean",
    include_spreads: bool = True,
    benchmark_code: str = "USA",
    include_commodities: bool = True,
    include_conditions: bool = True,
    wide: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build multi-country financial panel merging sovereign yields, policy rates, credit gaps, and commodities.

    Parameters
    ----------
    codes : Sequence[str] | None, default None
        List of ISO-3 country codes. If None, queries all available countries.
    start_date : str, default '1990-01-01'
        Earliest start date string (YYYY-MM-DD).
    end_date : str | None, optional
        Latest end date string (YYYY-MM-DD).
    frequency : str, default 'Q'
        Target frequency: ``'Q'`` (quarterly) or ``'M'`` (monthly).
    harmonization : str, default 'mean'
        Aggregation rule for monthly to quarterly aggregation: ``'mean'`` or ``'last'``.
    include_spreads : bool, default True
        If True, computes term spreads and sovereign spreads vs benchmark.
    benchmark_code : str, default 'USA'
        Benchmark sovereign for risk spread calculations.
    include_commodities : bool, default True
        If True, includes global commodity benchmark prices under code ``'WLD'``.
    include_conditions : bool, default True
        If True, includes financial conditions and spreads (TED rate, HY OAS, NFCI, etc.).
    wide : bool, default False
        If True, returns pivoted matrix format where index is date and columns are formatted
        as ``{code}_{variable}``.
    refresh : bool, default False
        If True, refreshes remote cache.

    Returns
    -------
    pd.DataFrame
        Aligned financial panel in long-form (or wide matrix if ``wide=True``).
    """
    freq_norm = frequency.strip().upper()
    if freq_norm not in ("M", "Q"):
        raise ValueError(f"Unsupported frequency: '{frequency}'. Must be 'M' or 'Q'.")

    harm_norm = harmonization.strip().lower()
    if harm_norm not in ("mean", "last"):
        raise ValueError(f"Unsupported harmonization: '{harmonization}'. Must be 'mean' or 'last'.")

    if codes is not None:
        codes = [str(c).strip().upper() for c in codes]
        if len(codes) == 0:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date")) if wide else _EMPTY.copy()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    monthly_frames: list[pd.DataFrame] = []
    quarterly_frames: list[pd.DataFrame] = []

    # 1. Sovereign Yields & Spreads (monthly)
    try:
        from .fetch.financial import fetch_sovereign_yields, compute_sovereign_spreads
        b_code = benchmark_code.strip().upper()
        yield_codes: list[str] | None
        if codes is not None and include_spreads and b_code not in codes:
            yield_codes = list(codes) + [b_code]
        else:
            yield_codes = codes

        y_df = fetch_sovereign_yields(
            codes=yield_codes,
            start_date=start_date,
            maturities=("10Y", "2Y"),
            refresh=refresh,
        )
        if not y_df.empty:
            if include_spreads:
                spreads_df = compute_sovereign_spreads(y_df, benchmark_code=b_code, include_yields=False)
                if not spreads_df.empty:
                    monthly_frames.append(spreads_df)
            monthly_frames.append(y_df)
    except Exception:
        pass

    # 2. Policy Rates (monthly)
    try:
        from .fetch.financial import fetch_policy_rates
        p_df = fetch_policy_rates(
            codes=codes,
            start_date=start_date,
            refresh=refresh,
        )
        if not p_df.empty:
            monthly_frames.append(p_df)
    except Exception:
        pass

    # 3. Financial Conditions (monthly, USA / WLD)
    if include_conditions and (codes is None or "USA" in codes):
        try:
            from .fetch.financial import fetch_financial_conditions
            fc_df = fetch_financial_conditions(
                start_date=start_date,
                refresh=refresh,
            )
            if not fc_df.empty:
                monthly_frames.append(fc_df)
        except Exception:
            pass

    # 4. BIS Macroprudential (quarterly)
    try:
        from .fetch.financial import fetch_bis_macroprudential
        bis_df = fetch_bis_macroprudential(
            codes=codes,
            start_date=start_date,
            refresh=refresh,
        )
        if not bis_df.empty:
            quarterly_frames.append(bis_df)
    except Exception:
        pass

    # 5. Global Commodities (code='WLD')
    if include_commodities:
        try:
            from .fetch.wb_pink_sheet import fetch_commodity_benchmarks
            if freq_norm == "Q" and harm_norm == "last":
                # For 'last' quarterly aggregation, fetch monthly and aggregate via last
                comm_df = fetch_commodity_benchmarks(
                    frequency="M",
                    start_date=start_date,
                    refresh=refresh,
                )
                if not comm_df.empty:
                    monthly_frames.append(comm_df)
            else:
                comm_df = fetch_commodity_benchmarks(
                    frequency=freq_norm,
                    start_date=start_date,
                    refresh=refresh,
                )
                if not comm_df.empty:
                    if freq_norm == "M":
                        monthly_frames.append(comm_df)
                    else:
                        quarterly_frames.append(comm_df)
        except Exception:
            pass

    # Harmonize to target frequency
    final_frames: list[pd.DataFrame] = []

    if freq_norm == "Q":
        if monthly_frames:
            m_all = pd.concat(monthly_frames, ignore_index=True)
            q_resampled = _aggregate_monthly_to_quarterly(m_all, harmonization=harm_norm)
            if not q_resampled.empty:
                final_frames.append(q_resampled)
        if quarterly_frames:
            q_all = pd.concat(quarterly_frames, ignore_index=True)
            q_all["variable"] = q_all["variable"].apply(lambda v: _norm_var_name(v, "Q"))
            if "is_imputed" not in q_all.columns:
                q_all["is_imputed"] = False
            final_frames.append(q_all)
    else:  # freq_norm == "M"
        if quarterly_frames:
            q_all = pd.concat(quarterly_frames, ignore_index=True)
            m_projected = _project_quarterly_to_monthly(q_all)
            if not m_projected.empty:
                final_frames.append(m_projected)
        if monthly_frames:
            m_all = pd.concat(monthly_frames, ignore_index=True)
            m_all["variable"] = m_all["variable"].apply(lambda v: _norm_var_name(v, "M"))
            if "is_imputed" not in m_all.columns:
                m_all["is_imputed"] = False
            final_frames.append(m_all)

    if not final_frames:
        if wide:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return _EMPTY.copy()

    panel = pd.concat(final_frames, ignore_index=True)

    # Date filter
    if start_ts is not None:
        panel = panel[panel["date"] >= start_ts]
    if end_ts is not None:
        panel = panel[panel["date"] <= end_ts]

    if panel.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date")) if wide else _EMPTY.copy()

    # Code filter: preserve requested country codes + global benchmarks (WLD)
    if codes is not None:
        allowed = set(codes) | ({"WLD"} if (include_commodities or include_conditions) else set())
        panel = panel[panel["code"].isin(allowed)]

    if panel.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date")) if wide else _EMPTY.copy()

    panel = panel.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    panel = panel.sort_values(["code", "variable", "date"]).reset_index(drop=True)

    if wide:
        piv = panel.pivot_table(index="date", columns=["code", "variable"], values="value")
        piv.columns = [f"{c}_{v}" for c, v in piv.columns]
        piv.columns.name = None
        piv = piv.sort_index()
        return piv

    return panel


__all__ = ["build_financial_panel"]
