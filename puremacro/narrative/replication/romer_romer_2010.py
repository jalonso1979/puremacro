"""Replication of Romer-Romer (2010) US tax narrative shocks.

Romer & Romer's ``Y`` series (their "exogenous tax shocks") quantifies
legislated tax changes between 1945 and 2007 motivated by (a) deficit
reduction or (b) long-run growth — but explicitly *not* counter-
cyclical. It is one of the canonical narrative IVs and is widely used
to identify fiscal-consumption effects via household disposable income.

Source: Romer, C. and Romer, D. (2010). "The Macroeconomic Effects of
Tax Changes." American Economic Review 100(3), 763-801.

By default the loader reads the frozen snapshot shipped as package data
(``puremacro/replication/data/tax14_narrative_tax_shocks.csv``), so it
works offline out of the box. Provenance of the snapshot: Valerie
Ramey's public "Macroeconomic Shocks and Their Propagation" (Handbook
of Macroeconomics, 2016) tax archive
(https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_tax.zip, sheet
``homtaxdat``, column ``EXOGENRRATIO``), quarterly 1945Q1-2007Q4 in
percent of GDP; regenerate with ``tools/gen_notebook_data_tax14.py``.
A local CSV path (``csv_path=``) or a one-shot download (``url=``)
override the snapshot.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ...replication._data import load_csv as _load_snapshot_csv
from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_SNAPSHOT = "tax14_narrative_tax_shocks"


def romer_romer_2010_csv_to_events(df: pd.DataFrame) -> list[NarrativeEvent]:
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("quarter")
    val_col = (
        cols.get("rr_exog") or cols.get("exog_tax") or cols.get("y")
        or cols.get("tax_shock")
    )
    eff_col = cols.get("effective_date") or cols.get("effective_quarter")
    if date_col is None or val_col is None:
        raise ValueError(
            "Romer-Romer CSV missing required columns; expected "
            "'date' and one of 'rr_exog' / 'exog_tax' / 'y' / 'tax_shock'."
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

        # Tax cut → expansionary effect on disposable income → +1.
        # Romer-Romer convention: positive value = tax *cut* (expansionary).
        # We propagate that sign into ``sign``.
        out.append(NarrativeEvent(
            date=date,
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="consumption",
            subtarget="tax",
            sign=int(1 if v > 0 else -1),
            confidence=1.0,
            source_text="Romer-Romer (2010) exogenous tax change.",
            source_url="https://eml.berkeley.edu/~dromer/papers/",
            scoring_method="manual",
            metadata={"replication": "romer_romer_2010"},
            implementation_profile=profile,
        ))
    return out


def load(
    *,
    csv_path: str | Path | None = None,
    url: str | None = None,
    target: str = "consumption",
) -> NarrativeInstrument:
    """Return the Romer-Romer 2010 tax-shock instrument.

    Source resolution order:

    1. ``csv_path`` — a local CSV (columns per
       :func:`romer_romer_2010_csv_to_events`).
    2. ``url`` — one-shot download of such a CSV.
    3. default — the frozen snapshot shipped as package data
       (``puremacro/replication/data/tax14_narrative_tax_shocks.csv``,
       from Ramey's Handbook-of-Macroeconomics tax archive), i.e. the
       loader works offline with no arguments.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif url is not None:
        try:
            raw = safe_get_bytes(url)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch the Romer-Romer 2010 tax-shock series "
                f"from url={url!r}. Either retry, download the CSV manually "
                "and pass its path as csv_path=, or call load() with no "
                "arguments to use the shipped snapshot "
                "(puremacro/replication/data/tax14_narrative_tax_shocks.csv)."
            ) from e
    else:
        df = _load_snapshot_csv(_SNAPSHOT)
    events = romer_romer_2010_csv_to_events(df)
    return NarrativeInstrument.from_events(events, target=target,
                                              aggregation="sum")


__all__ = ["load"]
