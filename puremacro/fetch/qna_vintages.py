"""Multi-country Quarterly National Accounts (QNA) from historical vintages.

Enables real-time / vintage-aware macroeconomic research across 45+ countries
(all 38 OECD member states + G20 + emerging markets) from the 1960s to present.

Covers expenditure and output components:
- ``gdp_real``      : Real Gross Domestic Product (chained volume / constant prices, SA)
- ``con_real``      : Real Private Final Consumption Expenditure (SA)
- ``gfcf_real``     : Real Gross Fixed Capital Formation / Investment (SA)
- ``govcon_real``   : Real Government Final Consumption Expenditure (SA)
- ``exports_real``  : Real Exports of Goods and Services (SA)
- ``imports_real``  : Real Imports of Goods and Services (SA)
- ``deflator``      : GDP Implicit Price Deflator (SA)
- ``gdp_nom``       : Nominal Gross Domestic Product (National currency, SA)

Backends:
- FRED / ALFRED (Archival Federal Reserve Economic Data) for international QNA vintages
- Persistent caching via :class:`puremacro.vintages.AlfredVintageStore`
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ._classic import _safe_urlopen
from ..vintages import as_of, align_vintages, forecast_revision, AlfredVintageStore


# ─────────────────────────────────────────────────────────────────────────────
# 45+ Country Catalog & Series Mappings
# ─────────────────────────────────────────────────────────────────────────────

# ISO3 -> (ISO2, Country Name, Region/Group)
COUNTRY_METADATA: dict[str, tuple[str, str, str]] = {
    # G7 / Major Advanced
    "USA": ("US", "United States", "G7"),
    "GBR": ("GB", "United Kingdom", "G7"),
    "DEU": ("DE", "Germany", "G7"),
    "FRA": ("FR", "France", "G7"),
    "ITA": ("IT", "Italy", "G7"),
    "JPN": ("JP", "Japan", "G7"),
    "CAN": ("CA", "Canada", "G7"),
    # Other OECD Europe
    "AUT": ("AT", "Austria", "OECD_Europe"),
    "BEL": ("BE", "Belgium", "OECD_Europe"),
    "CHE": ("CH", "Switzerland", "OECD_Europe"),
    "CZE": ("CZ", "Czech Republic", "OECD_Europe"),
    "DNK": ("DK", "Denmark", "OECD_Europe"),
    "ESP": ("ES", "Spain", "OECD_Europe"),
    "EST": ("EE", "Estonia", "OECD_Europe"),
    "FIN": ("FI", "Finland", "OECD_Europe"),
    "GRC": ("GR", "Greece", "OECD_Europe"),
    "HUN": ("HU", "Hungary", "OECD_Europe"),
    "IRL": ("IE", "Ireland", "OECD_Europe"),
    "ISL": ("IS", "Iceland", "OECD_Europe"),
    "LTU": ("LT", "Lithuania", "OECD_Europe"),
    "LUX": ("LU", "Luxembourg", "OECD_Europe"),
    "LVA": ("LV", "Latvia", "OECD_Europe"),
    "NLD": ("NL", "Netherlands", "OECD_Europe"),
    "NOR": ("NO", "Norway", "OECD_Europe"),
    "POL": ("PL", "Poland", "OECD_Europe"),
    "PRT": ("PT", "Portugal", "OECD_Europe"),
    "SVK": ("SK", "Slovakia", "OECD_Europe"),
    "SVN": ("SI", "Slovenia", "OECD_Europe"),
    "SWE": ("SE", "Sweden", "OECD_Europe"),
    "TUR": ("TR", "Turkey", "OECD_Europe"),
    # OECD Asia-Pacific
    "AUS": ("AU", "Australia", "OECD_AsiaPacific"),
    "ISR": ("IL", "Israel", "OECD_AsiaPacific"),
    "KOR": ("KR", "Korea", "OECD_AsiaPacific"),
    "NZL": ("NZ", "New Zealand", "OECD_AsiaPacific"),
    # OECD Americas / Latin America
    "CHL": ("CL", "Chile", "OECD_Americas"),
    "COL": ("CO", "Colombia", "OECD_Americas"),
    "CRI": ("CR", "Costa Rica", "OECD_Americas"),
    "MEX": ("MX", "Mexico", "OECD_Americas"),
    # G20 / Emerging Markets
    "BRA": ("BR", "Brazil", "Emerging_G20"),
    "CHN": ("CN", "China", "Emerging_G20"),
    "IDN": ("ID", "Indonesia", "Emerging_G20"),
    "IND": ("IN", "India", "Emerging_G20"),
    "RUS": ("RU", "Russia", "Emerging_G20"),
    "ZAF": ("ZA", "South Africa", "Emerging_G20"),
    # Euro Area Aggregate
    "EA19": ("EZ", "Euro Area", "Aggregates"),
}

# Variable name -> (ALFRED OECD code template, US direct series ID, description)
# For OECD countries, ALFRED standard codes follow NAEXKP01<ISO2>Q652S etc.
VARIABLE_MAP: dict[str, tuple[str, str, str]] = {
    "gdp_real": (
        "NAEXKP01{iso2}Q652S",
        "GDPC1",
        "Gross Domestic Product, Real, Chained Volume, SA",
    ),
    "con_real": (
        "NAEXKP02{iso2}Q652S",
        "PCECC96",
        "Private Final Consumption Expenditure, Real, SA",
    ),
    "govcon_real": (
        "NAEXKP03{iso2}Q652S",
        "GCEC1",
        "Government Final Consumption Expenditure, Real, SA",
    ),
    "gfcf_real": (
        "NAEXKP04{iso2}Q652S",
        "GPDIC1",
        "Gross Fixed Capital Formation (Investment), Real, SA",
    ),
    "exports_real": (
        "NAEXKP06{iso2}Q652S",
        "EXPGSC1",
        "Exports of Goods and Services, Real, SA",
    ),
    "imports_real": (
        "NAEXKP07{iso2}Q652S",
        "IMPGSC1",
        "Imports of Goods and Services, Real, SA",
    ),
    "deflator": (
        "CPALTT01{iso2}Q659N",
        "GDPDEF",
        "GDP Implicit Price Deflator / Price Level, SA",
    ),
    "gdp_nom": (
        "NAEXCP01{iso2}Q652S",
        "GDP",
        "Gross Domestic Product, Nominal, National Currency, SA",
    ),
}

# Standard aliases for puremacro panel integration
VARIABLE_ALIASES = {
    "log_gdp_real": "gdp_real",
    "log_con_real": "con_real",
    "log_govcon_real": "govcon_real",
    "log_gfcf_real": "gfcf_real",
    "log_exports_real": "exports_real",
    "log_imports_real": "imports_real",
    "log_deflator": "deflator",
    "log_gdp_nom": "gdp_nom",
}


def _resolve_series_id(country: str, variable: str) -> str:
    """Resolve the ALFRED / FRED series ID for a (country, variable) pair."""
    country_clean = country.upper()
    var_clean = VARIABLE_ALIASES.get(variable, variable)
    if var_clean not in VARIABLE_MAP:
        raise ValueError(
            f"Unknown QNA variable {variable!r}. Available: {sorted(VARIABLE_MAP.keys())}"
        )
    if country_clean not in COUNTRY_METADATA:
        raise ValueError(
            f"Unknown country code {country!r}. Available: {sorted(COUNTRY_METADATA.keys())}"
        )
    iso2, _, _ = COUNTRY_METADATA[country_clean]
    template, us_id, _ = VARIABLE_MAP[var_clean]
    if country_clean == "USA" and us_id:
        return us_id
    return template.format(iso2=iso2)


def get_qna_vintage_catalog() -> pd.DataFrame:
    """Return a DataFrame listing all supported countries, variables, and ALFRED series IDs.

    Returns
    -------
    pd.DataFrame with columns ``[country, iso2, country_name, region, variable, series_id, description]``.
    """
    rows = []
    for c_iso3, (c_iso2, c_name, c_region) in sorted(COUNTRY_METADATA.items()):
        for var, (tpl, us_id, desc) in sorted(VARIABLE_MAP.items()):
            if c_iso3 == "USA" and us_id:
                sid = us_id
            else:
                sid = tpl.format(iso2=c_iso2)
            rows.append({
                "country": c_iso3,
                "iso2": c_iso2,
                "country_name": c_name,
                "region": c_region,
                "variable": var,
                "series_id": sid,
                "description": desc,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# QNAVintagePanel Class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QNAVintagePanel:
    """Multi-country, multi-variable Quarterly National Accounts vintage panel.

    Attributes
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns:
        ``[country, variable, date, vintage, value]``
        where ``date`` is the reference calendar period (quarter start),
        ``vintage`` is the publication date, and ``value`` is the published number.
    metadata : dict
        Provenance, provider, query parameters.
    """
    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.df.empty:
            required = {"country", "variable", "date", "vintage", "value"}
            missing = required - set(self.df.columns)
            if missing:
                raise ValueError(f"QNAVintagePanel missing required columns: {sorted(missing)}")
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df["vintage"] = pd.to_datetime(self.df["vintage"])
            self.df["value"] = pd.to_numeric(self.df["value"], errors="coerce")
            self.df = self.df.dropna(subset=["date", "vintage", "value"]).sort_values(
                ["country", "variable", "date", "vintage"]
            ).reset_index(drop=True)

    @property
    def countries(self) -> list[str]:
        """List of country codes present in the panel."""
        if self.df.empty:
            return []
        return sorted(self.df["country"].unique().tolist())

    @property
    def variables(self) -> list[str]:
        """List of variables present in the panel."""
        if self.df.empty:
            return []
        return sorted(self.df["variable"].unique().tolist())

    @property
    def vintages(self) -> list[pd.Timestamp]:
        """List of all distinct publication vintages."""
        if self.df.empty:
            return []
        return sorted(self.df["vintage"].unique().tolist())

    def as_of(
        self,
        vintage_date: str | pd.Timestamp,
        *,
        countries: Iterable[str] | None = None,
        variables: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Slice panel to the latest-known observation as of ``vintage_date``.

        Parameters
        ----------
        vintage_date : timestamp or ISO string; only vintages <= vintage_date are considered.
        countries : optional subset of country codes.
        variables : optional subset of variable names.

        Returns
        -------
        pd.DataFrame indexed by ``[country, date]`` with variable columns (wide),
        representing the macro dataset as it was known on ``vintage_date``.
        """
        v_dt = pd.Timestamp(vintage_date)
        sub = self.df[self.df["vintage"] <= v_dt]
        if countries is not None:
            c_set = {c.upper() for c in countries}
            sub = sub[sub["country"].isin(c_set)]
        if variables is not None:
            v_set = {v for v in variables}
            sub = sub[sub["variable"].isin(v_set)]
        if sub.empty:
            return pd.DataFrame()

        # For each (country, variable, date), keep the latest vintage <= v_dt
        latest = (
            sub.sort_values(["country", "variable", "date", "vintage"])
            .groupby(["country", "variable", "date"], as_index=False)
            .last()
        )
        piv = latest.pivot_table(
            index=["country", "date"], columns="variable", values="value"
        ).sort_index()
        piv.columns.name = None
        return piv

    def revision_matrix(
        self,
        country: str,
        variable: str,
    ) -> pd.DataFrame:
        """Construct the (observation_date × vintage_date) revision triangle.

        Each column represents a vintage release date; each row represents a reference quarter.
        Missing values (NaN) indicate the reference quarter had not yet been published
        as of that vintage date.

        Returns
        -------
        pd.DataFrame indexed by ``date`` with columns named after ``vintage``.
        """
        c = country.upper()
        sub = self.df[(self.df["country"] == c) & (self.df["variable"] == variable)]
        if sub.empty:
            return pd.DataFrame()
        return align_vintages(
            sub,
            vintage_dates=sorted(sub["vintage"].unique()),
            date_col="date",
            vintage_col="vintage",
            value_col="value",
        )

    def first_release(self, country: str, variable: str) -> pd.Series:
        """Extract the sequence of initial / advance releases for each reference quarter."""
        c = country.upper()
        sub = self.df[(self.df["country"] == c) & (self.df["variable"] == variable)]
        if sub.empty:
            return pd.Series(dtype=float, name="first_release")
        sub = sub.sort_values(["date", "vintage"])
        s = sub.groupby("date", as_index=True)["value"].first()
        s.name = f"{country}_{variable}_first_release"
        return s

    def latest_release(self, country: str, variable: str) -> pd.Series:
        """Extract the most recent / final release available in the panel for each reference quarter."""
        c = country.upper()
        sub = self.df[(self.df["country"] == c) & (self.df["variable"] == variable)]
        if sub.empty:
            return pd.Series(dtype=float, name="latest_release")
        sub = sub.sort_values(["date", "vintage"])
        s = sub.groupby("date", as_index=True)["value"].last()
        s.name = f"{country}_{variable}_latest_release"
        return s

    def revision_stats(self, country: str, variable: str) -> dict[str, Any]:
        """Compute revision properties: mean revision, variance, and news vs noise tests.

        Evaluates the difference between the latest release $y_T$ and the initial release $y_0$:
        $$\text{Revision}_t = y_{T,t} - y_{0,t}$$

        Also performs the Mankiw & Shapiro (1986) news vs noise regression:
        $$y_{T,t} - y_{0,t} = \alpha + \beta y_{0,t} + \varepsilon_t$$
        - Under **pure news** (initial release is an efficient rational expectation): $\beta = 0$.
        - Under **pure noise** (initial release is true value + independent measurement error): $\beta = -1$.
        """
        first = self.first_release(country, variable).dropna()
        latest = self.latest_release(country, variable).dropna()
        shared = first.index.intersection(latest.index)
        if len(shared) < 4:
            return {
                "n_obs": len(shared),
                "mean_revision": np.nan,
                "abs_mean_revision": np.nan,
                "std_revision": np.nan,
                "mankiw_shapiro_beta": np.nan,
                "mankiw_shapiro_pvalue": np.nan,
                "hypothesis": "insufficient_data",
            }
        y0 = first.loc[shared].values
        yT = latest.loc[shared].values
        rev = yT - y0

        # OLS regression: rev = alpha + beta * y0 + e
        X = np.column_stack([np.ones(len(y0)), y0])
        beta_hat, *_ = np.linalg.lstsq(X, rev, rcond=None)
        resid = rev - X @ beta_hat
        n, k = X.shape
        sigma2 = (resid @ resid) / max(1, n - k)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(max(0, cov[1, 1]))
        t_stat = beta_hat[1] / se_beta if se_beta > 1e-9 else 0.0

        # Approximate two-tailed p-value using standard normal
        from math import erfc, sqrt
        p_val = erfc(abs(t_stat) / sqrt(2.0))

        # Categorize
        if p_val > 0.05:
            hypothesis = "news"  # cannot reject beta=0
        elif abs(beta_hat[1] - (-1.0)) < 0.3:
            hypothesis = "noise"
        else:
            hypothesis = "mixed"

        return {
            "n_obs": int(len(shared)),
            "first_obs": str(shared.min())[:10],
            "last_obs": str(shared.max())[:10],
            "mean_revision": float(np.mean(rev)),
            "abs_mean_revision": float(np.mean(np.abs(rev))),
            "std_revision": float(np.std(rev, ddof=1)),
            "mankiw_shapiro_alpha": float(beta_hat[0]),
            "mankiw_shapiro_beta": float(beta_hat[1]),
            "mankiw_shapiro_se": float(se_beta),
            "mankiw_shapiro_tstat": float(t_stat),
            "mankiw_shapiro_pvalue": float(p_val),
            "hypothesis": hypothesis,
        }

    def to_long(self) -> pd.DataFrame:
        """Return raw long-form DataFrame ``[country, variable, date, vintage, value]``."""
        return self.df.copy()

    def to_panel_q(
        self,
        vintage_date: str | pd.Timestamp | None = None,
        *,
        log_transform: bool = False,
    ) -> pd.DataFrame:
        """Export to standard puremacro panel format ``[code, date, variable, value, sa_source, source]``.

        Parameters
        ----------
        vintage_date : if provided, takes the snapshot as of that vintage;
            if None, takes the latest release for each country/variable/date.
        log_transform : if True, logs positive values and prefixes variable names with ``log_``.
        """
        if self.df.empty:
            return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])

        if vintage_date is not None:
            v_dt = pd.Timestamp(vintage_date)
            sub = self.df[self.df["vintage"] <= v_dt]
        else:
            sub = self.df

        if sub.empty:
            return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])

        latest = (
            sub.sort_values(["country", "variable", "date", "vintage"])
            .groupby(["country", "variable", "date"], as_index=False)
            .last()
        )
        out = pd.DataFrame()
        out["code"] = latest["country"]
        out["date"] = latest["date"]
        var_name = latest["variable"]
        val = latest["value"].astype(float)

        if log_transform:
            valid_mask = val > 0
            latest = latest[valid_mask].copy()
            out = pd.DataFrame()
            out["code"] = latest["country"]
            out["date"] = latest["date"]
            out["variable"] = latest["variable"].apply(
                lambda v: f"log_{v}" if not v.startswith("log_") else v
            )
            out["value"] = np.log(latest["value"].astype(float))
        else:
            out["variable"] = var_name
            out["value"] = val

        out["sa_source"] = "alfred_oecd"
        out["source"] = "ALFRED_QNA_VINTAGES"
        return out.dropna(subset=["value"]).reset_index(drop=True)

    def filter(
        self,
        *,
        countries: Iterable[str] | None = None,
        variables: Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        start_vintage: str | None = None,
        end_vintage: str | None = None,
    ) -> "QNAVintagePanel":
        """Return a filtered sub-panel."""
        sub = self.df.copy()
        if countries is not None:
            c_set = {c.upper() for c in countries}
            sub = sub[sub["country"].isin(c_set)]
        if variables is not None:
            v_set = {v for v in variables}
            sub = sub[sub["variable"].isin(v_set)]
        if start_date is not None:
            sub = sub[sub["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            sub = sub[sub["date"] <= pd.Timestamp(end_date)]
        if start_vintage is not None:
            sub = sub[sub["vintage"] >= pd.Timestamp(start_vintage)]
        if end_vintage is not None:
            sub = sub[sub["vintage"] <= pd.Timestamp(end_vintage)]
        return QNAVintagePanel(df=sub, metadata={**self.metadata, "filtered": True})


# ─────────────────────────────────────────────────────────────────────────────
# Raw ALFRED Fetch Helper
# ─────────────────────────────────────────────────────────────────────────────

_ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={}"


def _fetch_series_alfred_vintages(
    series_id: str,
    *,
    store: AlfredVintageStore | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch vintage history for a single ALFRED series ID.

    Returns long DataFrame with columns ``[date, vintage, value]``.
    """
    if store is not None and not refresh and store.has_series(series_id):
        df_store = store.get(series_id)
        if not df_store.empty:
            return df_store.rename(columns={
                "observation_date": "date",
                "vintage_date": "vintage",
            })[["date", "vintage", "value"]].reset_index(drop=True)

    url = _ALFRED_GRAPH_URL.format(series_id)
    try:
        raw = _safe_urlopen(url, timeout=timeout)
        df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        # Fall back to store if available or return empty
        if store is not None and store.has_series(series_id):
            df_store = store.get(series_id)
            return df_store.rename(columns={
                "observation_date": "date",
                "vintage_date": "vintage",
            })[["date", "vintage", "value"]].reset_index(drop=True)
        return pd.DataFrame(columns=["date", "vintage", "value"])

    if df_raw.empty or len(df_raw.columns) < 2:
        return pd.DataFrame(columns=["date", "vintage", "value"])

    df_raw = df_raw.rename(columns={df_raw.columns[0]: "date"})
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    long = df_raw.melt(id_vars="date", var_name="vintage_col", value_name="value")
    # Extract trailing YYYY-MM-DD from column name like "GDPC1_2023-04-26"
    long["vintage"] = pd.to_datetime(
        long["vintage_col"].str.split("_").str[-1], errors="coerce"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["date", "vintage", "value"])
    out = long[["date", "vintage", "value"]].reset_index(drop=True)

    # Cache into SQLite store
    if store is not None and not out.empty:
        store_rows = out.assign(series_id=series_id).rename(columns={
            "date": "observation_date",
            "vintage": "vintage_date",
        })[["series_id", "observation_date", "vintage_date", "value"]]
        store.put_many(store_rows)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# High-Level API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_qna_vintages(
    countries: Iterable[str] | str | None = None,
    variables: Iterable[str] | str | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    start_vintage: str | None = None,
    end_vintage: str | None = None,
    provider: str = "alfred",
    store: AlfredVintageStore | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
) -> QNAVintagePanel:
    """Fetch quarterly national accounts across historical publication vintages for multiple countries.

    Parameters
    ----------
    countries : ISO3 country codes (e.g. ``["USA", "DEU", "GBR", "JPN", "FRA", "MEX"]``).
        Defaults to all 45+ supported economies if None.
    variables : QNA variable names (e.g. ``["gdp_real", "con_real", "gfcf_real"]``).
        Defaults to ``["gdp_real"]`` if None.
    start_date : optional start period for observations (e.g. ``"1970-01-01"``).
    end_date : optional end period for observations.
    start_vintage : optional earliest publication vintage to include.
    end_vintage : optional latest publication vintage to include.
    provider : data provider (default ``"alfred"``).
    store : persistent AlfredVintageStore instance; if None, uses default shared SQLite store.
    refresh : if True, forces live network fetch even if cached.
    timeout : network timeout per series in seconds.

    Returns
    -------
    QNAVintagePanel
        A panel containing vintage observations with slicing, revision analysis,
        and export capabilities.
    """
    if store is None:
        try:
            store = AlfredVintageStore()
        except Exception:
            store = None

    # Resolve target country list
    if countries is None:
        target_countries = sorted(COUNTRY_METADATA.keys())
    elif isinstance(countries, str):
        target_countries = [countries.upper()]
    else:
        target_countries = [c.upper() for c in countries]

    # Resolve target variable list
    if variables is None:
        target_variables = ["gdp_real"]
    elif isinstance(variables, str):
        target_variables = [VARIABLE_ALIASES.get(variables, variables)]
    else:
        target_variables = [VARIABLE_ALIASES.get(v, v) for v in variables]

    frames: list[pd.DataFrame] = []
    series_metadata: dict[str, str] = {}

    for c in target_countries:
        if c not in COUNTRY_METADATA:
            continue
        for var in target_variables:
            if var not in VARIABLE_MAP:
                continue
            try:
                sid = _resolve_series_id(c, var)
            except ValueError:
                continue

            series_metadata[f"{c}_{var}"] = sid
            df_v = _fetch_series_alfred_vintages(
                sid, store=store, refresh=refresh, timeout=timeout
            )
            if df_v.empty:
                continue
            df_v["country"] = c
            df_v["variable"] = var
            frames.append(df_v[["country", "variable", "date", "vintage", "value"]])

    if not frames:
        empty_df = pd.DataFrame(columns=["country", "variable", "date", "vintage", "value"])
        panel = QNAVintagePanel(df=empty_df, metadata={"provider": provider, "series_ids": series_metadata})
    else:
        full_df = pd.concat(frames, ignore_index=True)
        panel = QNAVintagePanel(df=full_df, metadata={"provider": provider, "series_ids": series_metadata})

    # Apply date / vintage filters if specified
    if any(p is not None for p in (start_date, end_date, start_vintage, end_vintage)):
        panel = panel.filter(
            start_date=start_date,
            end_date=end_date,
            start_vintage=start_vintage,
            end_vintage=end_vintage,
        )

    return panel


__all__ = [
    "QNAVintagePanel",
    "fetch_qna_vintages",
    "get_qna_vintage_catalog",
    "COUNTRY_METADATA",
    "VARIABLE_MAP",
    "VARIABLE_ALIASES",
]
