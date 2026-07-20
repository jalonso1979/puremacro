"""IMF WP/22/114 (Kirti, Liu, Martinez Peria, Mishra, Strasky 2022)
COVID-period announcement-level fiscal database.

Covers 74 countries × 28 granular policies × 2020. Each row records
one announced policy action. The loader filters to the fiscal subset
by default, aggregates same-quarter same-policy-category rows, and
emits one NarrativeEvent per (country, policy_category, quarter) cell.

Source
------
Kirti, D., Liu, Y., Martinez Peria, S., Mishra, P., and Strasky, J.
(2022). "Tracking Economic and Financial Policies During COVID-19:
An Announcement-Level Database." IMF Working Paper WP/22/114.
https://www.imf.org/en/Publications/WP/Issues/2022/06/03/
Tracking-Economic-and-Financial-Policies-During-COVID-19-An-
Announcement-Level-Database-518896

Sign convention
---------------
COVID fiscal announcements are overwhelmingly expansionary (relief,
stimulus). Source magnitudes are positive for expansionary spend or
tax cuts. Default ``flip_sign=True`` keeps puremacro's expansionary-
positive convention (no flip needed for COVID). This contrasts with
DGLP / Devries / Guajardo / Adler 2024, where the source positive =
consolidation and the loader flips to expansionary-negative.

Endogeneity
-----------
WP/22/114 doesn't separate exogenous from endogenous announcements
in the way the EU NMS dataset does. All COVID measures are arguably
counter-cyclical (responding to the pandemic shock); using them as
narrative IVs requires assuming the COVID shock itself is exogenous
to fiscal policy timing. Caller's call.
"""
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "imf_covid_2022_announcements.csv"
)

# Map fine-grained source-data policy categories → canonical subtargets.
# Lookup is lowercased; unrecognized fiscal-class strings fall back to
# "general".
_POLICY_TO_SUBTARGET: dict[str, str] = {
    # Tax-side
    "tax cut":               "tax",
    "tax_cut":               "tax",
    "vat cut":               "tax",
    "vat_cut":               "tax",
    "personal income tax":   "tax",
    "corporate income tax":  "tax",
    "tax deferral":          "tax",
    "tax_deferral":          "tax",
    "social contributions":  "tax",
    # Spending-side: investment / infrastructure
    "public investment":     "infra",
    "public_investment":     "infra",
    "infrastructure":        "infra",
    "capital spending":      "infra",
    # Spending-side: transfers
    "unemployment benefits": "transfers",
    "unemployment_benefits": "transfers",
    "cash transfers":        "transfers",
    "cash_transfers":        "transfers",
    "income support":        "transfers",
    "household transfers":   "transfers",
    "transfers":             "transfers",
    # Health / defense
    "health spending":       "health",
    "health_spending":       "health",
    "covid-19 health":       "health",
    "defense spending":      "defense",
    # Public wages
    "public wages":          "wages",
    "wage subsidies":        "wages",
    "wage_subsidies":        "wages",
    "furlough":              "wages",
    # Generic spending
    "expenditure":           "expenditure",
    "spending":              "expenditure",
    "subsidies":             "expenditure",
}

_SUBTARGET_TO_TARGET: dict[str, str] = {
    "tax":         "consumption",
    "transfers":   "consumption",
    "wages":       "consumption",
    "health":      "both",
    "defense":     "both",
    "infra":       "investment",
    "expenditure": "both",
    "general":     "both",
}

# Minimal ISO2 → ISO3 dict for the 74-country WP/22/114 set (most
# common cases). Expand as needed; unknown ISO2 codes fall through
# unchanged (and the country whitelist drops them later).
_ISO2_TO_ISO3: dict[str, str] = {
    "US": "USA", "GB": "GBR", "DE": "DEU", "FR": "FRA", "IT": "ITA",
    "ES": "ESP", "JP": "JPN", "CN": "CHN", "IN": "IND", "BR": "BRA",
    "MX": "MEX", "CA": "CAN", "AU": "AUS", "RU": "RUS", "ZA": "ZAF",
    "ID": "IDN", "TR": "TUR", "KR": "KOR", "AR": "ARG", "SA": "SAU",
    "NL": "NLD", "CH": "CHE", "SE": "SWE", "BE": "BEL", "AT": "AUT",
    "NO": "NOR", "DK": "DNK", "FI": "FIN", "GR": "GRC", "PT": "PRT",
    "IE": "IRL", "PL": "POL", "CZ": "CZE", "HU": "HUN", "IL": "ISR",
    "CL": "CHL", "CO": "COL", "PE": "PER", "TH": "THA", "MY": "MYS",
    "SG": "SGP", "PH": "PHL", "VN": "VNM", "EG": "EGY", "MA": "MAR",
    "NG": "NGA", "KE": "KEN", "GH": "GHA", "PK": "PAK", "BD": "BGD",
    "LK": "LKA", "NZ": "NZL", "RO": "ROU", "BG": "BGR", "HR": "HRV",
    "SK": "SVK", "SI": "SVN", "EE": "EST", "LV": "LVA", "LT": "LTU",
    "IS": "ISL", "MT": "MLT", "CY": "CYP", "LU": "LUX",
}


def _normalize_country(raw) -> str | None:
    """Accept ISO2 or ISO3 (case-insensitive); return canonical ISO3 or None."""
    s = str(raw).strip().upper()
    if not s:
        return None
    if len(s) == 3:
        return s  # assume ISO3
    if len(s) == 2:
        return _ISO2_TO_ISO3.get(s, s)  # fall through if unknown ISO2
    return None


def _policy_to_subtarget(raw_policy: str) -> str:
    """Map source policy category to canonical subtarget; default 'general'."""
    s = str(raw_policy).strip().lower()
    return _POLICY_TO_SUBTARGET.get(s, "general")


def imf_covid_2022_csv_to_events(
    df: pd.DataFrame,
    *,
    flip_sign: bool = True,
    include_non_fiscal: bool = False,
) -> list[NarrativeEvent]:
    """Coerce the WP/22/114 CSV into NarrativeEvents.

    Filters to ``policy_class == 'fiscal'`` unless ``include_non_fiscal=True``.
    Aggregates rows sharing ``(country, subtarget, quarter)`` into one event
    with summed magnitude and earliest announcement date.
    """
    cols = {c.lower(): c for c in df.columns}
    country_col = (cols.get("country") or cols.get("iso3") or cols.get("iso2")
                     or cols.get("iso") or cols.get("code"))
    date_col = (cols.get("date") or cols.get("announcement_date")
                  or cols.get("announce_date"))
    policy_col = (cols.get("policy") or cols.get("policy_category")
                    or cols.get("policy_type") or cols.get("category"))
    class_col = (cols.get("policy_class") or cols.get("area")
                   or cols.get("class"))
    mag_col = (cols.get("magnitude_pct_gdp") or cols.get("size_pct_gdp")
                 or cols.get("magnitude") or cols.get("size"))
    desc_col = (cols.get("description") or cols.get("text")
                  or cols.get("excerpt"))

    if country_col is None or date_col is None or policy_col is None:
        raise ValueError(
            "IMF COVID 2022 CSV missing required columns; expected "
            "country, date, policy."
        )

    # Aggregation buckets keyed by (country, subtarget, quarter_period).
    # Value: dict accumulating magnitude, earliest_date, raw_policies, descriptions.
    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "magnitude": 0.0,
        "earliest_date": None,
        "raw_policies": [],
        "descriptions": [],
        "target": None,
    })

    for _, row in df.iterrows():
        if class_col is not None:
            policy_class = str(row[class_col]).strip().lower()
        else:
            policy_class = "fiscal"  # if column missing, assume fiscal
        if not include_non_fiscal and policy_class != "fiscal":
            continue

        country = _normalize_country(row[country_col])
        if country is None:
            continue

        try:
            date = pd.Timestamp(row[date_col])
        except (ValueError, TypeError):
            continue
        if pd.isna(date):
            continue
        quarter_ts = pd.Period(date, freq="Q").to_timestamp()

        raw_policy = str(row[policy_col])
        subtarget = _policy_to_subtarget(raw_policy)
        target = _SUBTARGET_TO_TARGET.get(subtarget, "both")

        if mag_col is not None:
            mag_raw = row[mag_col]
            try:
                mag = float(mag_raw) if not pd.isna(mag_raw) else 0.0
            except (TypeError, ValueError):
                mag = 0.0
        else:
            mag = 0.0

        desc = str(row[desc_col]) if desc_col is not None else ""

        key = (country, subtarget, quarter_ts)
        bucket = buckets[key]
        bucket["magnitude"] += mag
        if bucket["earliest_date"] is None or date < bucket["earliest_date"]:
            bucket["earliest_date"] = date
        bucket["raw_policies"].append(raw_policy)
        if desc:
            bucket["descriptions"].append(desc)
        bucket["target"] = target

    out: list[NarrativeEvent] = []
    for (country, subtarget, quarter_ts), bucket in buckets.items():
        total_mag = bucket["magnitude"]
        # Keep events even with magnitude=0 (announcement happened, magnitude
        # not always reported). NarrativeEvent allows magnitude=0.
        sign_raw = 1 if total_mag >= 0 else -1
        sign = sign_raw if flip_sign else -sign_raw
        # COVID convention: source positive = expansionary. flip_sign=True
        # (default) preserves that as +1.

        # Concat descriptions truncated to 500 chars.
        joined_desc = " | ".join(bucket["descriptions"])[:500]
        if not joined_desc:
            joined_desc = (
                f"IMF COVID 2022 announcement: {country} "
                f"{subtarget} {quarter_ts.strftime('%Y-Q%q').replace('Q1', 'Q1')}"
            )

        out.append(NarrativeEvent(
            date=bucket["earliest_date"] or quarter_ts,
            country=country,
            magnitude=abs(total_mag),
            magnitude_unit="pct_gdp",
            target=bucket["target"] or "both",
            subtarget=subtarget,
            sign=sign,
            confidence=1.0,
            source_text=joined_desc,
            source_url=(
                "https://www.imf.org/en/Publications/WP/Issues/2022/06/03/"
                "Tracking-Economic-and-Financial-Policies-During-COVID-19-"
                "An-Announcement-Level-Database-518896"
            ),
            scoring_method="manual",
            metadata={
                "replication": "imf_covid_2022",
                "raw_policies": bucket["raw_policies"],
                "n_announcements": len(bucket["raw_policies"]),
            },
            implementation_profile=[],  # COVID measures = immediate; default fallback.
        ))
    return out


def load(
    *,
    countries: list[str] | None = None,
    csv_path: str | Path | None = None,
    flip_sign: bool = True,
    include_non_fiscal: bool = False,
) -> dict[str, NarrativeInstrument]:
    """Load the IMF WP/22/114 COVID announcement-level fiscal panel.

    Parameters
    ----------
    countries : optional ISO3 filter. ``None`` returns all reporters.
    csv_path : optional local path; default attempts mirror download.
    flip_sign : if True (default), preserve source's expansionary-
        positive convention. (Note: this is the OPPOSITE flip-sign
        rule of DGLP/Adler — those flip from consolidation-positive
        to expansionary-negative.)
    include_non_fiscal : if False (default), drop monetary / prudential /
        "other" rows. If True, all 28 policy classes flow through.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch IMF COVID 2022 announcement-level dataset. "
                "Download from the IMF WP archive (WP/22/114) and pass "
                "csv_path=. Mirror upload to qmd-co is pending."
            ) from e

    events = imf_covid_2022_csv_to_events(
        df, flip_sign=flip_sign, include_non_fiscal=include_non_fiscal,
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


__all__ = ["imf_covid_2022_csv_to_events", "load"]
