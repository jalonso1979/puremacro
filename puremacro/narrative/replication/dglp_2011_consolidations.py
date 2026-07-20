"""Devries-Guajardo-Leigh-Pescatori (2011) action-based fiscal-
consolidation dataset.

The DGLP dataset is the canonical multi-country narrative-fiscal
benchmark. Its 17 advanced economies (AUS, AUT, BEL, CAN, DNK, FIN,
FRA, DEU, IRL, ITA, JPN, NLD, PRT, ESP, SWE, GBR, USA) span 1978-2009
with 173 hand-coded consolidation episodes, each tagged as
**tax-based** or **expenditure-based** with magnitude in % of GDP. It
is used in 100+ papers (Alesina-Favero-Giavazzi, IMF WEO chapter on
expansionary austerity, every modern fiscal-multiplier panel).

Source
------
Devries, P., Guajardo, J., Leigh, D., and Pescatori, A. (2011). "A
New Action-Based Dataset of Fiscal Consolidation," IMF Working Paper
WP/11/158. The replication CSV ships in the WP archive. We keep a
mirror; the loader can also read a local CSV via ``csv_path=``.

Sign convention
---------------
DGLP magnitudes are positive for consolidations (tightenings). For
the puremacro narrative-IV convention — where ``sign × magnitude`` is
the *expansionary* fiscal-shock magnitude — we flip the sign to ``-1``.
Pass ``flip_sign=False`` to keep the original DGLP convention.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "dglp_2011_fiscal_consolidations.csv"
)

# DGLP cover the EU15 + USA + CAN + JPN + AUS.
_DGLP_COUNTRIES = (
    "AUS AUT BEL CAN DNK FIN FRA DEU IRL ITA JPN NLD PRT ESP SWE GBR USA"
).split()


def _dglp_extract_rows(df: pd.DataFrame) -> list[dict]:
    """Read a DGLP-format CSV into normalized per-row dicts.

    Returns one dict per (country, year, subtarget) with non-zero
    magnitude. Per-component rows (tax_pct_gdp + exp_pct_gdp) emit
    two dicts; total-only rows emit one dict tagged ``"general"``.
    """
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("year") or cols.get("quarter")
    country_col = (cols.get("country") or cols.get("iso3")
                     or cols.get("iso") or cols.get("code"))
    total_col = (cols.get("total") or cols.get("magnitude")
                   or cols.get("size_pct_gdp") or cols.get("size"))
    tax_col = (cols.get("tax_based") or cols.get("tax_pct_gdp")
                 or cols.get("tax"))
    exp_col = (cols.get("spending_based") or cols.get("exp_pct_gdp")
                 or cols.get("expenditure") or cols.get("spending"))
    if date_col is None or country_col is None:
        raise ValueError("DGLP CSV missing date/country columns.")

    def _year(raw) -> int:
        s = str(raw)
        if len(s) == 4 and s.isdigit():
            return int(s)
        try:
            return pd.Period(s, freq="Q").year
        except Exception:
            return pd.Timestamp(s).year

    out: list[dict] = []

    def _push(country, year, magnitude, target, subtarget):
        if magnitude is None or pd.isna(magnitude) or float(magnitude) == 0.0:
            return
        v = float(magnitude)
        out.append({
            "country": country,
            "year": year,
            "value": v,
            "target": target,
            "subtarget": subtarget,
        })

    for _, row in df.iterrows():
        country = str(row[country_col]).upper()
        year = _year(row[date_col])
        if tax_col is not None and exp_col is not None:
            _push(country, year, row[tax_col], target="consumption", subtarget="tax")
            _push(country, year, row[exp_col], target="both", subtarget="expenditure")
        elif total_col is not None:
            _push(country, year, row[total_col], target="both", subtarget="general")
    return out


def _group_consecutive(rows: list[dict]) -> list[dict]:
    """Group consecutive-year rows sharing (country, subtarget, sign-of-value).

    A gap year breaks the group. Returns a list of group dicts:
    {country, subtarget, target, sign, first_year, tranches=[(year, value), ...]}.
    """
    def _key(r):
        return (r["country"], r["subtarget"], 1 if r["value"] > 0 else -1)

    sorted_rows = sorted(rows, key=lambda r: (_key(r), r["year"]))
    groups: list[dict] = []
    current: dict | None = None
    for r in sorted_rows:
        k = _key(r)
        if current is None or current["_key"] != k or r["year"] != current["last_year"] + 1:
            if current is not None:
                current.pop("_key", None)
                groups.append(current)
            current = {
                "_key": k, "country": r["country"], "subtarget": r["subtarget"],
                "target": r["target"], "sign_of_value": k[2],
                "first_year": r["year"], "last_year": r["year"],
                "tranches": [(r["year"], r["value"])],
            }
        else:
            current["last_year"] = r["year"]
            current["tranches"].append((r["year"], r["value"]))
    if current is not None:
        current.pop("_key", None)
        groups.append(current)
    return groups


def dglp_csv_to_events(
    df: pd.DataFrame,
    *,
    flip_sign: bool,
    within_year_rule: str = "uniform",
) -> list[NarrativeEvent]:
    """Coerce a DGLP-format CSV into NarrativeEvents.

    Consecutive years for the same (country, subtarget, sign) collapse
    into one event with a multi-quarter ``implementation_profile``.
    Within-year quarterization follows ``within_year_rule`` (default
    ``"uniform"``).
    """
    from ._within_year import annualize_to_quarters

    rows = _dglp_extract_rows(df)
    groups = _group_consecutive(rows)

    out: list[NarrativeEvent] = []
    for g in groups:
        total_value = sum(v for _, v in g["tranches"])
        if total_value == 0.0:
            continue
        # Magnitude is the absolute total; sign is from group's sign-of-value
        # (with optional DGLP-convention flip).
        sign_raw = g["sign_of_value"]
        sign = -sign_raw if flip_sign else sign_raw
        # Build the implementation profile across years × within-year quarters.
        profile_pairs: list[tuple[pd.Timestamp, float]] = []
        for year, val in g["tranches"]:
            year_weight = val / total_value
            country_for_fy = g["country"] if within_year_rule == "fiscal_year" else None
            profile_pairs.extend(
                annualize_to_quarters(year, weight=year_weight,
                                      rule=within_year_rule,
                                      country=country_for_fy)
            )
        out.append(NarrativeEvent(
            date=pd.Timestamp(g["first_year"], 1, 1),
            country=g["country"],
            magnitude=abs(total_value),
            magnitude_unit="pct_gdp",
            target=g["target"],
            subtarget=g["subtarget"],
            sign=sign,
            confidence=1.0,
            source_text="DGLP (2011) action-based fiscal consolidation.",
            source_url="https://www.imf.org/en/Publications/WP/Issues/2016/12/31/A-New-Action-Based-Dataset-of-Fiscal-Consolidation-25022",
            scoring_method="manual",
            metadata={
                "replication": "dglp_2011",
                "tranche_years": [y for y, _ in g["tranches"]],
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
    within_year_rule: str = "uniform",
) -> dict[str, NarrativeInstrument]:
    """Load the DGLP 2011 action-based fiscal-consolidation panel.

    Parameters
    ----------
    countries : optional ISO3 filter (e.g. ``["USA", "GBR", "DEU"]``).
    csv_path : optional local path; default attempts mirror download.
    flip_sign : if True (default), invert the DGLP "tightening = positive"
        convention so puremacro's ``signed_magnitude`` is the
        *expansionary* fiscal shock.
    within_year_rule : ``"uniform"`` | ``"front_loaded"`` | ``"fiscal_year"``.
        Controls how each annual tranche projects onto quarters.

    Returns
    -------
    dict mapping ISO3 → :class:`NarrativeInstrument`. Each instrument's
    ``events`` list contains both tax-based and expenditure-based
    events (filter via ``target=`` at construction).
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch DGLP 2011 fiscal-consolidation series. "
                "Download the replication CSV from the IMF WP archive "
                "(WP/11/158) and pass csv_path=. Per-country panel "
                "datasets are also redistributed in Alesina-Favero-"
                "Giavazzi (2019) replication archives."
            ) from e
    events = dglp_csv_to_events(df, flip_sign=flip_sign,
                                 within_year_rule=within_year_rule)
    by_country: dict[str, list[NarrativeEvent]] = {}
    for ev in events:
        if ev.country in _DGLP_COUNTRIES:
            by_country.setdefault(ev.country, []).append(ev)
    if countries is not None:
        wanted = {c.upper() for c in countries}
        by_country = {c: v for c, v in by_country.items() if c in wanted}
    return {
        c: NarrativeInstrument.from_events(evs, target=None,
                                              aggregation="sum")
        for c, evs in by_country.items()
    }


__all__ = ["load"]
