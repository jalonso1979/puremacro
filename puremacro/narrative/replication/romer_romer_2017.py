"""Replication of Romer-Romer (2017) cross-country fiscal narrative
tax shocks.

A 14-country panel (US, UK, AUS, CAN, DNK, FRA, DEU, IRL, ITA, JPN,
NLD, NZL, ESP, SWE) of legislated tax changes, hand-coded with the
Romer-Romer 2010 exogenous/endogenous classification. The natural
benchmark for cross-country fiscal-IV work and a direct on-ramp for
the 33-country wedges-panel endgame.

Source: Romer, C. and Romer, D. (2017). "Hard Times: Why the Recovery
from the Great Recession Was So Slow" (working paper version) and the
authors' cross-country tax-narrative dataset.

Implementation timing
---------------------
The RR2017 cross-country tax CSV does not carry implementation dates
distinct from announcement quarters. Events emitted by this loader
carry the default empty ``implementation_profile``. Backfilling per-
country implementation timing would require manual annotation against
each country's legislative record (out of scope for v1).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "romer_romer_2017_panel_tax.csv"
)

# ISO2 → ISO3 fix-ups in case the source file uses 2-letter codes.
_ISO2_TO_ISO3 = {
    "US": "USA", "UK": "GBR", "GB": "GBR", "AU": "AUS", "CA": "CAN",
    "DK": "DNK", "FR": "FRA", "DE": "DEU", "IE": "IRL", "IT": "ITA",
    "JP": "JPN", "NL": "NLD", "NZ": "NZL", "ES": "ESP", "SE": "SWE",
}


def romer_romer_2017_csv_to_events(df: pd.DataFrame) -> list[NarrativeEvent]:
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("quarter")
    country_col = (cols.get("country") or cols.get("iso3")
                     or cols.get("iso") or cols.get("code"))
    val_col = (cols.get("exog") or cols.get("exog_tax") or cols.get("y")
                or cols.get("tax_shock") or cols.get("rr_exog"))
    if date_col is None or val_col is None or country_col is None:
        raise ValueError(
            "Romer-Romer 2017 panel CSV missing required columns "
            "(date, country, value)."
        )
    out: list[NarrativeEvent] = []
    for _, row in df.iterrows():
        try:
            date = pd.Period(str(row[date_col]), freq="Q").to_timestamp()
        except Exception:
            try:
                date = pd.Timestamp(row[date_col])
            except Exception:
                continue
        country = str(row[country_col]).upper()
        country = _ISO2_TO_ISO3.get(country, country)
        v = float(row[val_col])
        if v == 0.0 or pd.isna(v):
            continue
        out.append(NarrativeEvent(
            date=date, country=country,
            magnitude=abs(v), magnitude_unit="pct_gdp",
            target="consumption", subtarget="tax",
            sign=int(1 if v > 0 else -1),
            confidence=1.0,
            source_text="Romer-Romer (2017) cross-country tax change.",
            source_url="https://eml.berkeley.edu/~dromer/papers/",
            scoring_method="manual",
            metadata={"replication": "romer_romer_2017"},
        ))
    return out


def load(
    *,
    countries: list[str] | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, NarrativeInstrument]:
    """Load the Romer-Romer 2017 cross-country tax-shock dataset.

    Returns
    -------
    dict mapping ISO3 country code → :class:`NarrativeInstrument`.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Romer-Romer 2017 panel. Download from "
                "the authors' replication archive and pass csv_path=."
            ) from e
    events = romer_romer_2017_csv_to_events(df)
    by_country: dict[str, list[NarrativeEvent]] = {}
    for ev in events:
        by_country.setdefault(ev.country, []).append(ev)
    if countries is not None:
        wanted = {c.upper() for c in countries}
        by_country = {c: v for c, v in by_country.items() if c in wanted}
    return {
        c: NarrativeInstrument.from_events(evs, target="consumption",
                                              aggregation="sum")
        for c, evs in by_country.items()
    }


__all__ = ["load"]
