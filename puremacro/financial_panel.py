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

import warnings
from typing import Any, Sequence

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

    # ⚡ Bolt: Vectorize groupby loop by grouping on quarter start dates directly.
    # Yields 10x+ speedup on large frames by avoiding Python loop and .resample() per series.
    df_m = df_m.sort_values(["code", "variable", "date"]).drop_duplicates(
        ["code", "variable", "date"], keep="last"
    ).copy()

    df_m["q_date"] = pd.to_datetime(df_m["date"]).dt.to_period("Q").dt.start_time
    df_m["value"] = df_m["value"].astype(float)
    df_m = df_m.dropna(subset=["value"])

    if df_m.empty:
        return _EMPTY.copy()

    vars_series = df_m["variable"]
    is_last = (harmonization == "last") | vars_series.str.contains("policy_rate") | vars_series.str.contains("cbrate")

    mean_df = df_m[~is_last]
    last_df = df_m[is_last]

    res_frames = []

    if not mean_df.empty:
        g_mean = mean_df.groupby(["code", "variable", "q_date"], as_index=False)
        agg_mean = g_mean.agg(
            value=("value", "mean"),
            sa_source=("sa_source", "first"),
            source=("source", "first")
        )
        agg_mean["source"] = "resampled_from_M(mean):" + agg_mean["source"].astype(str)
        res_frames.append(agg_mean)

    if not last_df.empty:
        g_last = last_df.groupby(["code", "variable", "q_date"], as_index=False)
        agg_last = g_last.agg(
            value=("value", "last"),
            sa_source=("sa_source", "first"),
            source=("source", "first")
        )
        agg_last["source"] = "resampled_from_M(last):" + agg_last["source"].astype(str)
        res_frames.append(agg_last)

    if not res_frames:
        return _EMPTY.copy()

    out = pd.concat(res_frames, ignore_index=True)
    out = out.rename(columns={"q_date": "date"})
    out["variable"] = out["variable"].apply(lambda v: _norm_var_name(v, "Q"))
    out["is_imputed"] = False

    return out[["code", "date", "variable", "value", "sa_source", "source", "is_imputed"]]


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

    dates = df_q["date"]
    if dates.isna().any():
        raise ValueError("quarterly dates must be valid timestamps")
    if pd.api.types.is_datetime64_any_dtype(dates.dtype):
        years = dates.dt.year.to_numpy()
        months = dates.dt.month.to_numpy()
    else:
        # Preserve Timestamp's scalar parsing for mixed valid date formats
        # and local calendar months for mixed timezone offsets.
        codes, unique_dates = pd.factorize(dates, sort=False)
        parsed = [pd.Timestamp(value) for value in unique_dates]
        years = np.array([date.year for date in parsed])[codes]
        months = np.array([date.month for date in parsed])[codes]
    positions = np.repeat(np.arange(len(df_q)), 3)
    offsets = np.tile(np.arange(3), len(df_q))
    projected_months = months[positions] + offsets
    # Quarter starts cannot cross December. Retain the previous truncation
    # behavior for out-of-contract November/December observations as well.
    valid = projected_months <= 12
    positions, offsets = positions[valid], offsets[valid]
    projected_months = projected_months[valid]
    names = df_q["variable"].astype(str)
    normalized = {name: _norm_var_name(name, "M") for name in names.unique()}
    out = pd.DataFrame({
        "code": df_q["code"].to_numpy()[positions],
        "date": pd.to_datetime({
            "year": years[positions], "month": projected_months, "day": 1,
        }),
        "variable": names.map(normalized).to_numpy()[positions],
        "value": df_q["value"].astype(float).to_numpy()[positions],
        "sa_source": np.where(offsets != 0, "quarterly_ffill",
                              df_q["sa_source"].to_numpy()[positions]),
        "source": ("resampled_from_Q:" + df_q["source"].astype(str)).to_numpy()[positions],
        "is_imputed": offsets != 0,
    })
    return out.drop_duplicates(subset=["code", "date", "variable"], keep="first")


def _status_entry(
    status: str,
    df: pd.DataFrame | None = None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    if df is not None and not df.empty:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        obs_start = str(dates.min().date()) if not dates.empty else None
        obs_end = str(dates.max().date()) if not dates.empty else None
        return {
            "status": status,
            "rows": int(len(df)),
            "obs_start": obs_start,
            "obs_end": obs_end,
            "error": error,
        }
    return {
        "status": status,
        "rows": 0,
        "obs_start": None,
        "obs_end": None,
        "error": error,
    }


def _attach_status(df: pd.DataFrame, source_status: dict[str, dict[str, Any]]) -> pd.DataFrame:
    df.attrs["source_status"] = source_status
    return df


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
    include_credit: bool = True,
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
    include_credit : bool, default True
        If True, includes BIS macroprudential credit-to-GDP gaps, total credit, and property prices.
    wide : bool, default False
        If True, returns pivoted matrix format where index is date and columns are formatted
        as ``{code}_{variable}``.
    refresh : bool, default False
        If True, refreshes remote cache.

    Returns
    -------
    pd.DataFrame
        Aligned financial panel in long-form (or wide matrix if ``wide=True``), with
        ``panel.attrs['source_status']`` recording per-block status, row counts,
        observation bounds, and any upstream errors.
    """
    freq_norm = frequency.strip().upper()
    if freq_norm not in ("M", "Q"):
        raise ValueError(f"Unsupported frequency: '{frequency}'. Must be 'M' or 'Q'.")

    harm_norm = harmonization.strip().lower()
    if harm_norm not in ("mean", "last"):
        raise ValueError(f"Unsupported harmonization: '{harmonization}'. Must be 'mean' or 'last'.")

    source_status: dict[str, dict[str, Any]] = {}

    if codes is not None:
        codes = [str(c).strip().upper() for c in codes]
        if len(codes) == 0:
            empty_out = (
                pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
                if wide
                else _EMPTY.copy()
            )
            return _attach_status(empty_out, source_status)

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
            source_status["sovereign_yields"] = _status_entry("ok", y_df)
        else:
            source_status["sovereign_yields"] = _status_entry("empty")
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        source_status["sovereign_yields"] = _status_entry("failed", error=err_msg)
        warnings.warn(
            f"build_financial_panel: upstream 'sovereign_yields' fetch failed ({err_msg}); "
            "continuing with remaining blocks.",
            RuntimeWarning,
            stacklevel=2,
        )

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
            source_status["policy_rates"] = _status_entry("ok", p_df)
        else:
            source_status["policy_rates"] = _status_entry("empty")
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        source_status["policy_rates"] = _status_entry("failed", error=err_msg)
        warnings.warn(
            f"build_financial_panel: upstream 'policy_rates' fetch failed ({err_msg}); "
            "continuing with remaining blocks.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 3. Financial Conditions (monthly, USA / WLD)
    if not include_conditions or not (codes is None or "USA" in codes):
        source_status["financial_conditions"] = _status_entry("skipped")
    else:
        try:
            from .fetch.financial import fetch_financial_conditions
            fc_df = fetch_financial_conditions(
                start_date=start_date,
                refresh=refresh,
            )
            if not fc_df.empty:
                monthly_frames.append(fc_df)
                source_status["financial_conditions"] = _status_entry("ok", fc_df)
            else:
                source_status["financial_conditions"] = _status_entry("empty")
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            source_status["financial_conditions"] = _status_entry("failed", error=err_msg)
            warnings.warn(
                f"build_financial_panel: upstream 'financial_conditions' fetch failed ({err_msg}); "
                "continuing with remaining blocks.",
                RuntimeWarning,
                stacklevel=2,
            )

    # 4. BIS Macroprudential (quarterly)
    if not include_credit:
        source_status["bis_macroprudential"] = _status_entry("skipped")
    else:
        try:
            from .fetch.financial import fetch_bis_macroprudential
            bis_df = fetch_bis_macroprudential(
                codes=codes,
                start_date=start_date,
                refresh=refresh,
            )
            if not bis_df.empty:
                quarterly_frames.append(bis_df)
                source_status["bis_macroprudential"] = _status_entry("ok", bis_df)
            else:
                source_status["bis_macroprudential"] = _status_entry("empty")
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            source_status["bis_macroprudential"] = _status_entry("failed", error=err_msg)
            warnings.warn(
                f"build_financial_panel: upstream 'bis_macroprudential' fetch failed ({err_msg}); "
                "continuing with remaining blocks.",
                RuntimeWarning,
                stacklevel=2,
            )

    # 5. Global Commodities (code='WLD')
    if not include_commodities:
        source_status["commodities"] = _status_entry("skipped")
    else:
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
                    source_status["commodities"] = _status_entry("ok", comm_df)
                else:
                    source_status["commodities"] = _status_entry("empty")
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
                    source_status["commodities"] = _status_entry("ok", comm_df)
                else:
                    source_status["commodities"] = _status_entry("empty")
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            source_status["commodities"] = _status_entry("failed", error=err_msg)
            warnings.warn(
                f"build_financial_panel: upstream 'commodities' fetch failed ({err_msg}); "
                "continuing with remaining blocks.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        empty_res = (
            pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            if wide
            else _EMPTY.copy()
        )
        return _attach_status(empty_res, source_status)

    panel = pd.concat(final_frames, ignore_index=True)

    # Date filter
    if start_ts is not None:
        panel = panel[panel["date"] >= start_ts]
    if end_ts is not None:
        panel = panel[panel["date"] <= end_ts]

    if panel.empty:
        empty_res = (
            pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            if wide
            else _EMPTY.copy()
        )
        return _attach_status(empty_res, source_status)

    # Code filter: preserve requested country codes + global benchmarks (WLD)
    if codes is not None:
        allowed = set(codes) | ({"WLD"} if (include_commodities or include_conditions) else set())
        panel = panel[panel["code"].isin(allowed)]

    if panel.empty:
        empty_res = (
            pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            if wide
            else _EMPTY.copy()
        )
        return _attach_status(empty_res, source_status)

    panel = panel.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    panel = panel.sort_values(["code", "variable", "date"]).reset_index(drop=True)

    if wide:
        piv = panel.pivot_table(index="date", columns=["code", "variable"], values="value")
        piv.columns = [f"{c}_{v}" for c, v in piv.columns]
        piv.columns.name = None
        piv = piv.sort_index()
        return _attach_status(piv, source_status)

    return _attach_status(panel, source_status)


__all__ = ["build_financial_panel"]
