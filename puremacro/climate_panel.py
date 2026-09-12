"""Modular Climate & Emissions Panel Builder.

Constructs multi-country climate panels combining:
1. Greenhouse gas emissions from World Bank WDI and OECD SDMX Air Emissions Inventory
   (via :func:`puremacro.fetch.emissions.fetch_emissions_panel`).
2. Energy transition and primary energy consumption from Ember and WDI
   (via :func:`puremacro.fetch.energy_transition.fetch_energy_transition`).
3. National macroeconomic aggregates when available.

Supports automated frequency harmonization:
- Annual ('A') to quarterly ('Q') via forward-repeat or linear interpolation.
- Output formatting in standard long-form schema or pivoted wide-panel format.

Architectural Invariant:
Zero module-scope ``requests`` imports. All network access routes through
puremacro's cached data layer.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_SCHEMA_COLS = ["code", "date", "variable", "value", "sa_source", "source"]
_EMPTY = pd.DataFrame(
    columns=_SCHEMA_COLS + ["is_imputed"]
)


def _harmonize_annual_to_quarterly(
    df_annual: pd.DataFrame,
    harmonization: str = "repeat",
) -> pd.DataFrame:
    """Harmonize annual series (January 1) to quarterly frequency ('QS').

    Parameters
    ----------
    df_annual : pd.DataFrame
        Input DataFrame with columns [code, date, variable, value, sa_source, source].
    harmonization : str, default 'repeat'
        Method to project annual observations into quarterly periods:
        * 'repeat': replicates the annual observation across all four quarters (Q1-Q4).
        * 'interpolate': places the observation on Q1 (Jan 1) and linearly interpolates
          intermediate quarters Q2-Q4.

    Returns
    -------
    pd.DataFrame
        Harmonized quarterly panel with updated variable names ('*_q') and source tags.
    """
    if df_annual.empty:
        return _EMPTY.copy()

    records: list[dict[str, object]] = []

    for (code, var), grp in df_annual.groupby(["code", "variable"]):
        grp_sorted = grp.sort_values("date").dropna(subset=["value"])
        if grp_sorted.empty:
            continue

        q_var = var[:-2] + "_q" if var.endswith("_a") else f"{var}_q"

        if harmonization == "repeat":
            for row in grp_sorted.to_dict('records'):
                yr = row["date"].year
                val = float(row["value"])
                src = f"resampled_from_A:{row['source']}"
                for m in (1, 4, 7, 10):
                    records.append({
                        "code": code,
                        "date": pd.Timestamp(f"{yr}-{m:02d}-01"),
                        "variable": q_var,
                        "value": val,
                        "sa_source": "annual_repeated",
                        "source": src,
                        "is_imputed": m != 1,
                    })
        elif harmonization == "interpolate":
            s = grp_sorted.set_index("date")["value"].astype(float)
            s = s[~s.index.duplicated(keep="first")]
            if len(s) == 1:
                yr = s.index[0].year
                val = float(s.iloc[0])
                src = f"interpolated_from_A:{grp_sorted['source'].iloc[0]}"
                sa = grp_sorted["sa_source"].iloc[0]
                for m in (1, 4, 7, 10):
                    records.append({
                        "code": code,
                        "date": pd.Timestamp(f"{yr}-{m:02d}-01"),
                        "variable": q_var,
                        "value": val,
                        "sa_source": sa,
                        "source": src,
                        "is_imputed": m != 1,
                    })
            else:
                min_yr = s.index.min().year
                max_yr = s.index.max().year
                full_q_idx = pd.date_range(
                    start=pd.Timestamp(f"{min_yr}-01-01"),
                    end=pd.Timestamp(f"{max_yr}-10-01"),
                    freq="QS",
                )
                s_q = s.reindex(full_q_idx)
                s_interp = s_q.interpolate(method="linear").bfill().ffill()

                src = f"interpolated_from_A:{grp_sorted['source'].iloc[0]}"
                sa = "interpolated"
                orig_dates = set(grp_sorted["date"])

                for dt_val, val in s_interp.items():
                    records.append({
                        "code": code,
                        "date": pd.Timestamp(dt_val),
                        "variable": q_var,
                        "value": float(val),
                        "sa_source": sa,
                        "source": src,
                        "is_imputed": dt_val not in orig_dates,
                    })
        else:
            raise ValueError(
                f"Unknown harmonization method: '{harmonization}'. Must be 'repeat' or 'interpolate'."
            )

    if not records:
        return _EMPTY.copy()

    res = pd.DataFrame(records)
    res = res.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    return res


def _aggregate_quarterly_to_annual(
    df_q: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate quarterly macroeconomic series to annual frequency (January 1).

    Parameters
    ----------
    df_q : pd.DataFrame
        Quarterly DataFrame with columns [code, date, variable, value, sa_source, source].

    Returns
    -------
    pd.DataFrame
        Annual long-form DataFrame with dates on January 1 of each year, values averaged,
        and is_imputed=False.
    """
    if df_q.empty:
        return _EMPTY.copy()

    records: list[dict[str, object]] = []

    for (code, var), grp in df_q.groupby(["code", "variable"]):
        s = grp.set_index("date")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        a_series = s.resample("YS").mean().dropna()
        if a_series.empty:
            continue

        src = f"aggregated_from_Q:{grp['source'].iloc[0]}" if ("source" in grp.columns and not grp["source"].empty) else "aggregated_from_Q"
        sa = grp["sa_source"].iloc[0] if ("sa_source" in grp.columns and not grp["sa_source"].empty) else "none"

        for dt_val, val in a_series.items():
            records.append({
                "code": code,
                "date": pd.Timestamp(dt_val),
                "variable": var,
                "value": float(val),
                "sa_source": sa,
                "source": src,
                "is_imputed": False,
            })

    if not records:
        return _EMPTY.copy()

    out = pd.DataFrame(records)
    return out.drop_duplicates(subset=["code", "date", "variable"], keep="first")


def build_climate_panel(
    codes: Sequence[str] | None = None,
    *,
    start_year: int = 1990,
    end_year: int | None = None,
    frequency: str = "A",
    harmonization: str = "repeat",
    wide: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build multi-country climate panel merging emissions, energy, and transition indicators.

    Parameters
    ----------
    codes : Sequence[str] | None, default None
        List of ISO-3 country codes to include. If None, retrieves all available countries.
    start_year : int, default 1990
        Earliest calendar year to include.
    end_year : int | None, optional
        Latest calendar year to include. If None, retrieves up to latest available.
    frequency : str, default 'A'
        Target frequency: ``'A'`` (annual) or ``'Q'`` (quarterly).
    harmonization : str, default 'repeat'
        Frequency projection method for quarterly frequency:
        ``'repeat'`` (repeat annual value across Q1-Q4) or
        ``'interpolate'`` (linear interpolation across quarters).
    wide : bool, default False
        If True, returns a pivoted matrix where index is date and columns are formatted
        as ``{code}_{variable}``.
    refresh : bool, default False
        If True, refreshes remote cache.

    Returns
    -------
    pd.DataFrame
        Aligned climate panel in long-form (or wide matrix if ``wide=True``).
    """
    freq_norm = frequency.strip().upper()
    if freq_norm not in ("A", "Q"):
        raise ValueError(f"Unsupported frequency: '{frequency}'. Must be 'A' or 'Q'.")

    harm_norm = harmonization.strip().lower()
    if harm_norm not in ("repeat", "interpolate"):
        raise ValueError(
            f"Unsupported harmonization: '{harmonization}'. Must be 'repeat' or 'interpolate'."
        )

    if codes is not None:
        codes = [str(c).strip().upper() for c in codes]
        if len(codes) == 0:
            return pd.DataFrame() if wide else _EMPTY.copy()

    annual_frames: list[pd.DataFrame] = []
    quarterly_frames: list[pd.DataFrame] = []

    # 1. Fetch emissions (annual baseline)
    try:
        from .fetch.emissions import fetch_emissions_panel
        em_df = fetch_emissions_panel(
            codes=codes,
            start_year=start_year,
            end_year=end_year,
            frequency="A",
            refresh=refresh,
        )
        if not em_df.empty:
            if "is_imputed" not in em_df.columns:
                em_df["is_imputed"] = False
            annual_frames.append(em_df)
    except Exception:
        pass

    # 2. Fetch energy transition (annual baseline)
    try:
        from .fetch.energy_transition import fetch_energy_transition
        et_df = fetch_energy_transition(
            codes=codes,
            start_year=start_year,
            refresh=refresh,
        )
        if not et_df.empty:
            if "is_imputed" not in et_df.columns:
                et_df["is_imputed"] = False
            annual_frames.append(et_df)
    except Exception:
        pass

    # 3. National macro output (real GDP, population) if available
    try:
        from .build_panel import PANEL_Q_PATH
        if PANEL_Q_PATH.exists():
            macro_q = pd.read_parquet(PANEL_Q_PATH)
            macro_vars = {"log_gdp_real", "gdp_real", "pop", "population"}
            macro_sub = macro_q[macro_q["variable"].isin(macro_vars)].copy()
            if codes is not None:
                macro_sub = macro_sub[macro_sub["code"].isin(set(codes))]
            if not macro_sub.empty:
                if "is_imputed" not in macro_sub.columns:
                    macro_sub["is_imputed"] = False
                quarterly_frames.append(macro_sub)
    except Exception:
        pass

    final_frames: list[pd.DataFrame] = []

    if freq_norm == "Q":
        if annual_frames:
            a_all = pd.concat(annual_frames, ignore_index=True)
            q_from_a = _harmonize_annual_to_quarterly(a_all, harmonization=harm_norm)
            if not q_from_a.empty:
                final_frames.append(q_from_a)
        if quarterly_frames:
            q_all = pd.concat(quarterly_frames, ignore_index=True)
            final_frames.append(q_all)
    else:  # freq_norm == "A"
        if annual_frames:
            final_frames.extend(annual_frames)
        if quarterly_frames:
            q_all = pd.concat(quarterly_frames, ignore_index=True)
            a_from_q = _aggregate_quarterly_to_annual(q_all)
            if not a_from_q.empty:
                final_frames.append(a_from_q)

    if not final_frames:
        if wide:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return _EMPTY.copy()

    panel = pd.concat(final_frames, ignore_index=True)

    # Filter date range
    if start_year is not None:
        panel = panel[panel["date"] >= pd.Timestamp(f"{start_year}-01-01")]
    if end_year is not None:
        panel = panel[panel["date"] <= pd.Timestamp(f"{end_year}-12-31")]

    if panel.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date")) if wide else _EMPTY.copy()

    # Filter to requested country codes
    if codes is not None:
        panel = panel[panel["code"].isin(set(codes))]

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


__all__ = ["build_climate_panel"]
