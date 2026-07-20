"""Replication of Cloyne (2013) UK narrative tax shocks.

Cloyne hand-codes UK tax changes from Hansard, Treasury budget
documents, and contemporary press, applying the Romer-Romer (2010)
exogenous/endogenous classification. The "exogenous" series isolates
tax changes motivated by deficit reduction or long-run growth — the
identifying variation for UK fiscal multipliers.

Source: Cloyne, J. (2013). "Discretionary Tax Changes and the
Macroeconomy: New Narrative Evidence from the United Kingdom."
American Economic Review 103(4), 1507-1528.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "cloyne_2013_uk_tax.csv"
)


def cloyne_csv_to_events(df: pd.DataFrame) -> list[NarrativeEvent]:
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("quarter")
    val_col = (cols.get("exog") or cols.get("exog_tax") or cols.get("y")
                or cols.get("tax_shock"))
    eff_col = cols.get("effective_date") or cols.get("effective_quarter")
    if date_col is None or val_col is None:
        raise ValueError(
            "Cloyne CSV missing date/value columns; expected 'date' and "
            "'exog' / 'exog_tax' / 'tax_shock'."
        )

    def _to_quarter(raw) -> pd.Timestamp | None:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        s = str(raw).strip()
        if not s:
            return None
        try:
            return pd.Period(s, freq="Q").to_timestamp()
        except Exception:
            try:
                return pd.Period(pd.Timestamp(s), freq="Q").to_timestamp()
            except Exception:
                return None

    out: list[NarrativeEvent] = []
    for _, row in df.iterrows():
        date = _to_quarter(row[date_col])
        if date is None:
            continue
        v = float(row[val_col])
        if v == 0.0 or pd.isna(v):
            continue
        profile: list[tuple[pd.Timestamp, float]] = []
        if eff_col is not None:
            eff_q = _to_quarter(row.get(eff_col))
            if eff_q is not None and eff_q >= date:
                profile = [(eff_q, 1.0)]
        out.append(NarrativeEvent(
            date=date,
            country="GBR",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="consumption",
            subtarget="tax",
            sign=int(1 if v > 0 else -1),
            confidence=1.0,
            source_text="Cloyne (2013) UK exogenous tax change.",
            source_url="https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1507",
            scoring_method="manual",
            metadata={"replication": "cloyne_2013_uk"},
            implementation_profile=profile,
        ))
    return out


def load(
    *,
    csv_path: str | Path | None = None,
    target: str = "consumption",
) -> NarrativeInstrument:
    """Return the Cloyne 2013 UK tax-shock instrument."""
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Cloyne 2013 UK tax series. Download "
                "the replication file from the AEA archive and pass "
                "csv_path=."
            ) from e
    events = cloyne_csv_to_events(df)
    return NarrativeInstrument.from_events(events, target=target,
                                              aggregation="sum")


__all__ = ["load"]
