"""Replication of Mertens-Ravn (2013) US tax narrative shocks.

Mertens & Ravn refine the Romer-Romer 2010 series by separating
*unanticipated* and *anticipated* tax changes. Their proxy SVAR
identification of unanticipated shocks (mtr_unanticipated_q) is the
go-to instrument for fiscal-tax IV in the modern empirical-macro
literature.

Source: Mertens, K. and Ravn, M. O. (2013). "The Dynamic Effects of
Personal and Corporate Income Tax Changes in the United States."
American Economic Review 103(4), 1212-1247.

By default the loader reads the frozen snapshot shipped as package data
(``puremacro/replication/data/tax14_narrative_tax_shocks.csv``), so it
works offline out of the box. Provenance of the snapshot: Valerie
Ramey's public "Macroeconomic Shocks and Their Propagation" (Handbook
of Macroeconomics, 2016) tax archive
(https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_tax.zip, sheet
``homtaxdat``; ``mtr_u`` = column ``TAXU``, unanticipated, ``mtr_a`` =
column ``taxnews``, anticipated), quarterly 1945Q1-2007Q4 in percent of
GDP; regenerate with ``tools/gen_notebook_data_tax14.py``. A local CSV
path (``csv_path=``) or a one-shot download (``url=``) override the
snapshot.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ...replication._data import load_csv as _load_snapshot_csv
from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


_SNAPSHOT = "tax14_narrative_tax_shocks"


def mertens_ravn_csv_to_events(df: pd.DataFrame, *, kind: str) -> list[NarrativeEvent]:
    """Coerce an MR-format CSV into NarrativeEvents.

    ``kind`` selects which series within the CSV to use:
        - "unanticipated" : MR's ``mtr_u`` series (default).
        - "anticipated"   : ``mtr_a``.
        - "all"           : ``mtr_total`` (= u + a).

    For ``kind="anticipated"``, if the CSV carries an ``impl_q`` (or
    ``implementation_quarter``) column, its value is used to populate
    ``implementation_profile`` — reflecting that anticipated changes are
    announced before they take effect. For other kinds the default empty
    profile applies (announcement = implementation).
    """
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("quarter")
    val_col = {
        "unanticipated": cols.get("mtr_u") or cols.get("unanticipated") or cols.get("mr_u"),
        "anticipated":   cols.get("mtr_a") or cols.get("anticipated") or cols.get("mr_a"),
        "all":           cols.get("mtr_total") or cols.get("total") or cols.get("mr_total"),
    }.get(kind)
    impl_col = cols.get("impl_q") or cols.get("implementation_quarter")
    if date_col is None or val_col is None:
        raise ValueError(
            f"Mertens-Ravn CSV missing date/value columns for kind={kind!r}"
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
        try:
            date = pd.Period(str(row[date_col]), freq="Q").to_timestamp()
        except Exception:
            try:
                date = pd.Timestamp(row[date_col])
            except Exception:
                continue
        v = float(row[val_col])
        if v == 0.0 or pd.isna(v):
            continue
        profile: list[tuple[pd.Timestamp, float]] = []
        if kind == "anticipated" and impl_col is not None:
            impl_q = _to_quarter(row.get(impl_col))
            if impl_q is not None and impl_q >= date:
                profile = [(impl_q, 1.0)]
        out.append(NarrativeEvent(
            date=date,
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="consumption",
            subtarget="tax",
            sign=int(1 if v > 0 else -1),
            confidence=1.0,
            source_text=f"Mertens-Ravn (2013) {kind} tax change.",
            source_url="https://karelmertens.com/research/",
            scoring_method="manual",
            metadata={"replication": "mertens_ravn_2013", "kind": kind},
            implementation_profile=profile,
        ))
    return out


def load(
    *,
    kind: str = "unanticipated",
    csv_path: str | Path | None = None,
    url: str | None = None,
    target: str = "consumption",
) -> NarrativeInstrument:
    """Load the Mertens-Ravn 2013 tax-shock instrument.

    Parameters
    ----------
    kind : ``"unanticipated"`` (default), ``"anticipated"``, or ``"all"``.
    csv_path : optional local path (takes precedence over ``url``).
    url : optional one-shot download of an MR-format CSV.
    target : passed through to :class:`NarrativeInstrument`.

    With neither ``csv_path`` nor ``url`` the frozen snapshot shipped as
    package data is read
    (``puremacro/replication/data/tax14_narrative_tax_shocks.csv``, from
    Ramey's Handbook-of-Macroeconomics tax archive), so the loader works
    offline with no arguments. If ``kind="all"`` and the CSV carries no
    total column, ``mtr_total`` is derived as ``mtr_u + mtr_a`` when
    both components are present (the snapshot ships only the two
    components).
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif url is not None:
        try:
            raw = safe_get_bytes(url)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch the Mertens-Ravn 2013 series from "
                f"url={url!r}. Either retry, download the CSV manually and "
                "pass its path as csv_path=, or call load() with no "
                "arguments to use the shipped snapshot "
                "(puremacro/replication/data/tax14_narrative_tax_shocks.csv)."
            ) from e
    else:
        df = _load_snapshot_csv(_SNAPSHOT)
    if kind == "all":
        cols = {c.lower(): c for c in df.columns}
        has_total = any(k in cols for k in ("mtr_total", "total", "mr_total"))
        u_col = cols.get("mtr_u") or cols.get("unanticipated") or cols.get("mr_u")
        a_col = cols.get("mtr_a") or cols.get("anticipated") or cols.get("mr_a")
        if not has_total and u_col is not None and a_col is not None:
            df = df.assign(
                mtr_total=df[u_col].fillna(0.0) + df[a_col].fillna(0.0)
            )
    events = mertens_ravn_csv_to_events(df, kind=kind)
    return NarrativeInstrument.from_events(events, target=target,
                                              aggregation="sum")


__all__ = ["load"]
