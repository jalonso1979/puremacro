"""Ciżkowicz-Ledóchowski-Rzońca (2023) narrative dataset of fiscal
actions in EU New Member States.

Covers 11 EU NMS — Poland, Czechia, Slovakia, Hungary, Lithuania,
Latvia, Estonia, Slovenia, Romania, Bulgaria, Croatia — for 2004-2019.
Each row records one announced action with explicit multi-year fiscal
effects in the ``Impact_t``, ``Impact_t+1``, ..., ``Impact_t+5``
columns. The loader maps each row directly to one ``NarrativeEvent``
with ``implementation_profile`` populated from the impact-column
values.

Source
------
Ciżkowicz, P., Ledóchowski, M., and Rzońca, A. (2023). "Narrative
dataset of fiscal actions in the EU new member states." Data in
Brief 49, 109446. DOI: 10.1016/j.dib.2023.109446
Data: https://data.mendeley.com/datasets/5xnhy8dxmh/2

Sign convention
---------------
Consolidations are positive in the source. ``flip_sign=True`` (default)
inverts to puremacro's expansionary-fiscal-shock convention.

Endogeneity filter
------------------
The source flags each action as ``exogenous`` (motivated by deficit
reduction or long-run growth) or ``endogenous`` (counter-cyclical).
``include_endogenous=False`` (default) loads only exogenous events,
matching the narrative-IV use case.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "eu_nms_2023_fiscal_actions.csv"
)

_EU_NMS_COUNTRIES = frozenset({
    "POL", "CZE", "SVK", "HUN", "LTU",
    "LVA", "EST", "SVN", "ROU", "BGR", "HRV",
})

# ISO2 → ISO3 (when source carries 2-letter codes).
_EU_NMS_ISO2_TO_ISO3: dict[str, str] = {
    "PL": "POL", "CZ": "CZE", "SK": "SVK", "HU": "HUN", "LT": "LTU",
    "LV": "LVA", "EE": "EST", "SI": "SVN", "RO": "ROU", "BG": "BGR",
    "HR": "HRV",
}

# Map fine-grained source-data category strings → canonical subtargets.
# Lookup is lowercased; unrecognized strings fall back to "general".
_CATEGORY_TO_SUBTARGET: dict[str, str] = {
    # Tax
    "personal income tax":   "tax",
    "corporate income tax":  "tax",
    "vat":                   "tax",
    "value added tax":       "tax",
    "excise":                "tax",
    "excise duty":           "tax",
    "social contributions":  "tax",
    "social security":       "tax",
    "tax":                   "tax",
    # Spending
    "public investment":     "expenditure",
    "public wages":          "expenditure",
    "public sector wages":   "expenditure",
    "transfers":             "expenditure",
    "consumption spending":  "expenditure",
    "subsidies":             "expenditure",
    "expenditure":           "expenditure",
    "spending":              "expenditure",
}

_SUBTARGET_TO_TARGET: dict[str, str] = {
    "tax":         "consumption",
    "expenditure": "both",
    "general":     "both",
}

_HORIZON_MAX = 5

_ENDOGENEITY_CANDIDATES: tuple[str, ...] = (
    "endogeneity", "endogenous", "type", "kind", "exogeneity",
)


def _normalize_country(raw) -> str | None:
    """Translate ISO2 / ISO3 / lowercase to canonical ISO3, or None."""
    s = str(raw).strip().upper()
    if not s:
        return None
    if s in _EU_NMS_COUNTRIES:
        return s
    if s in _EU_NMS_ISO2_TO_ISO3:
        return _EU_NMS_ISO2_TO_ISO3[s]
    return None


def _category_to_subtarget(raw_category: str) -> str:
    """Map source category string to canonical subtarget; default 'general'."""
    s = str(raw_category).strip().lower()
    return _CATEGORY_TO_SUBTARGET.get(s, "general")


def eu_nms_csv_to_events(
    df: pd.DataFrame,
    *,
    flip_sign: bool = True,
    include_endogenous: bool = False,
    within_year_rule: str = "uniform",
) -> list[NarrativeEvent]:
    """Coerce an EU NMS 2023 CSV into NarrativeEvents.

    Each row → one NarrativeEvent. ``implementation_profile`` is built
    directly from the ``Impact_t``, ``Impact_t+1``, ..., ``Impact_t+5``
    columns: each non-zero year is quarterized via ``annualize_to_quarters``.
    """
    from ._within_year import annualize_to_quarters

    cols = {c.lower(): c for c in df.columns}
    country_col = (cols.get("country") or cols.get("iso3") or cols.get("iso2")
                     or cols.get("iso") or cols.get("code"))
    year_col = cols.get("year") or cols.get("date") or cols.get("announcement_year")
    cat_col = cols.get("category") or cols.get("type") or cols.get("kind")
    gdp_col = cols.get("gdp") or cols.get("gdp_total") or cols.get("gdp_lcu")

    if country_col is None or year_col is None or cat_col is None:
        raise ValueError(
            "EU NMS 2023 CSV missing required columns; expected country, "
            "year, category."
        )

    impact_cols: list[tuple[int, str]] = []
    for k in range(_HORIZON_MAX + 1):
        suffix = "" if k == 0 else f"+{k}"
        col_name = f"impact_t{suffix}".lower()
        col_real = cols.get(col_name)
        if col_real is None:
            col_real = cols.get(f"impact_t{suffix}")
        if col_real is None:
            continue
        impact_cols.append((k, col_real))
    if not impact_cols:
        raise ValueError(
            "EU NMS 2023 CSV missing impact_t / impact_t+1 / ... columns."
        )

    endog_col = next(
        (cols[c] for c in _ENDOGENEITY_CANDIDATES if c in cols), None,
    )

    out: list[NarrativeEvent] = []

    for _, row in df.iterrows():
        if endog_col is not None:
            endog_label = str(row[endog_col]).strip().lower()
        else:
            endog_label = "exogenous"
        if not include_endogenous and endog_label != "exogenous":
            continue

        country = _normalize_country(row[country_col])
        if country is None or country not in _EU_NMS_COUNTRIES:
            continue

        year_val = str(row[year_col]).strip()
        if "." in year_val:
            year_val = year_val.split(".")[0]
        if year_val.isdigit():
            year = int(year_val)
        else:
            try:
                year = pd.Period(year_val, freq="Q").year
            except Exception:
                year = pd.Timestamp(year_val).year

        year_value_pairs: list[tuple[int, float]] = []
        for k, c in impact_cols:
            v = row[c]
            if v is None or pd.isna(v):
                continue
            v = float(v)
            if v == 0.0:
                continue
            year_value_pairs.append((year + k, v))
        if not year_value_pairs:
            continue

        total_value = sum(v for _, v in year_value_pairs)
        if total_value == 0.0:
            continue

        if gdp_col is not None:
            gdp_val = row[gdp_col]
            try:
                gdp_f = float(gdp_val)
            except (TypeError, ValueError):
                gdp_f = 0.0
            if gdp_f > 0:
                magnitude_pct = abs(total_value) / gdp_f * 100.0
                magnitude_unit = "pct_gdp"
            else:
                magnitude_pct = abs(total_value)
                magnitude_unit = "pct_gdp"
        else:
            magnitude_pct = abs(total_value)
            magnitude_unit = "pct_gdp"

        sign_raw = 1 if total_value > 0 else -1
        sign = -sign_raw if flip_sign else sign_raw

        raw_category = str(row[cat_col])
        subtarget = _category_to_subtarget(raw_category)
        target = _SUBTARGET_TO_TARGET[subtarget]

        profile_pairs: list[tuple[pd.Timestamp, float]] = []
        for y, val in year_value_pairs:
            year_weight = abs(val) / abs(total_value) if total_value else 0.0
            country_for_fy = country if within_year_rule == "fiscal_year" else None
            profile_pairs.extend(
                annualize_to_quarters(y, weight=year_weight,
                                      rule=within_year_rule,
                                      country=country_for_fy)
            )

        out.append(NarrativeEvent(
            date=pd.Timestamp(year, 1, 1),
            country=country,
            magnitude=magnitude_pct,
            magnitude_unit=magnitude_unit,
            target=target,
            subtarget=subtarget,
            sign=sign,
            confidence=1.0,
            source_text=f"EU NMS 2023 ({raw_category}, {endog_label}).",
            source_url="https://data.mendeley.com/datasets/5xnhy8dxmh/2",
            scoring_method="manual",
            metadata={
                "replication": "eu_nms_2023",
                "category": raw_category,
                "endogeneity": endog_label,
                "tranche_years": [y for y, _ in year_value_pairs],
                "within_year_rule": within_year_rule,
            },
            implementation_profile=profile_pairs,
        ))
    return out


def load(
    *,
    countries: list[str] | None = None,
    csv_path: str | Path | None = None,
    flip_sign: bool = True,
    include_endogenous: bool = False,
    within_year_rule: str = "uniform",
) -> dict[str, NarrativeInstrument]:
    """Load the EU NMS 2023 narrative-fiscal-action panel."""
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch EU NMS 2023 dataset. Download from "
                "https://data.mendeley.com/datasets/5xnhy8dxmh/2 (the "
                "DataStructuring_consistent_with_Alesina.xlsx file or its "
                "CSV equivalent) and pass csv_path=. Mirror upload to "
                "qmd-co is pending."
            ) from e

    events = eu_nms_csv_to_events(
        df, flip_sign=flip_sign,
        include_endogenous=include_endogenous,
        within_year_rule=within_year_rule,
    )
    by_country: dict[str, list[NarrativeEvent]] = {}
    for ev in events:
        by_country.setdefault(ev.country, []).append(ev)
    if countries is not None:
        wanted = {c.upper() for c in countries}
        by_country = {c: v for c, v in by_country.items() if c in wanted}
    return {
        c: NarrativeInstrument.from_events(evs, target=None,
                                            aggregation="sum")
        for c, evs in by_country.items()
    }


__all__ = ["eu_nms_csv_to_events", "load"]
