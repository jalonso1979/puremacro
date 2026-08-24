"""``qna_long_panel`` — the longest quarterly national accounts we can build.

The OECD spine (:func:`puremacro.fetch.qna_panel`) is extended backwards
per country by ratio-splicing archived national vintages onto it::

    long = qna_long_panel(["ESP", "JPN"])
    long.loc["JPN"].index.min()      # 1955-04-01, vs 1994-01-01 from OECD

Column schema is exactly :func:`~puremacro.fetch.qna_panel`'s, so
:func:`~puremacro.fetch.qna_identity` and friends work unchanged, plus
one ``src_<column>`` per value column recording which vintage produced
each quarter.

WHICH COUNTRIES, AND WHY SO FEW
-------------------------------
Only where an archived source demonstrably reaches further back than
the OECD already does. That was measured, not assumed, and most
candidates failed:

* **Eurostat** publishes the same numbers as the OECD — the splice
  ratio is 1.0 — and has no United Kingdom at all. It buys zero
  quarters for zero countries.
* **IMF** ties the OECD on seven of eight economies checked and is
  twelve quarters *shorter* on the United States.
* **INEGI** serves nothing before 1993 and is identical to the OECD
  after dividing by four; **IBGE** ties the OECD at 1996 exactly.
* **CEPALSTAT** adds nothing clean; its Mexican 1980-92 join has a
  ratio drifting 11% across the overlap, which is precisely the
  condition :func:`.._splice.ratio_splice` refuses to treat as a splice.
* **Germany** reaches 1970 only in *volumes*: its pre-1991 nominal
  accounts cover West Germany, and rescaling those onto unified
  Germany would fabricate an East German economy.

Those findings are recorded in :data:`KNOWN_GAPS` rather than being
lost, because "we checked and it buys nothing" is a different statement
from "we did not check".

READ THE SEAMS
--------------
A spliced series is a claim about the past built from vintages that
disagree. ``return_seams=True`` hands back the evidence: the ratio at
each join, the overlap it was estimated on, and how much it drifted.
A seam whose ratio held steady is a rebasing and is safe; a seam whose
ratio wandered means the two vintages disagree about *growth*, and the
level you get depends on the anchor. For Spain, GDP and household
consumption are steady back to 1970 while investment is not — so the
answer differs by column, and only the seam table will tell you that.
"""
from __future__ import annotations

import warnings

import pandas as pd

from ._splice import RATIO_DRIFT_WARN, expenditure_residual, splice_frame


#: The expenditure columns a long panel carries. Deliberately the
#: headline aggregates only: the archived vintages publish coarser
#: asset and durability splits than the modern accounts, and mapping
#: those onto the modern column names would misstate what they are.
LONG_PANEL_COLUMNS: tuple[str, ...] = (
    "gdp", "cons_hh", "cons_gov", "inv", "capform", "exports", "imports",
)

#: ISO3 -> the callable returning that country's archived segments,
#: newest first, spine excluded.
LONG_PANEL_SOURCES: dict[str, str] = {
    "ESP": "puremacro.fetch.longpanel.ine_es:spain_segments",
    "JPN": "puremacro.fetch.longpanel.esri_jp:japan_segments",
}

#: Countries deliberately not extended, and the measured reason.
KNOWN_GAPS: dict[str, str] = {
    "DEU": "Pre-1991 German nominal accounts cover West Germany only. A "
           "ratio splice onto unified Germany would fabricate an East "
           "German economy back to 1970. The Bundesbank publishes an "
           "officially linked 1970- series in VOLUMES (BBNZ1 Q.DE.Y.H.*.L) "
           "which is the correct route if you need real rather than "
           "nominal.",
    "MEX": "INEGI serves nothing before 1993 and is identical to the OECD "
           "after dividing by four. CEPALSTAT's 1980-92 Mexico is GDP only, "
           "with a vintage ratio drifting 0.754-0.837 across the overlap.",
    "BRA": "IBGE's quarterly accounts start 1996Q1, the same quarter as the "
           "OECD, across all seven candidate tables. A hard floor.",
    "CHL": "Banco Central reaches 1990Q1, 24 quarters before the OECD, but "
           "only for gdp/inv/exports/imports and only NSA, via ~1 MB HTML "
           "scrapes with no JSON route.",
    "GBR": "The OECD already carries the UK from 1960Q1. Eurostat has no UK "
           "row at all.",
    "USA": "The OECD already carries the US from 1947Q1.",
    "KOR": "The OECD already carries Korea from 1960Q1.",
}


def _load(path: str):
    mod, _, attr = path.partition(":")
    import importlib
    return getattr(importlib.import_module(mod), attr)


def qna_long_panel(
    codes=None,
    *,
    start: str = "1947",
    columns: tuple[str, ...] = LONG_PANEL_COLUMNS,
    drift_warn: float = RATIO_DRIFT_WARN,
    drop_unstable: bool = False,
    return_seams: bool = False,
    spine: pd.DataFrame | None = None,
    timeout: float = 120.0,
    use_cache: bool = True,
    refresh: bool = False,
):
    """Quarterly national accounts, extended back per country.

    Parameters
    ----------
    codes : ISO3 codes. ``None`` takes everything
        :data:`LONG_PANEL_SOURCES` can extend. Codes with no archived
        source pass through with the OECD span unchanged — that is not
        an error, and :data:`KNOWN_GAPS` says why for the ones that
        were checked.
    columns : which expenditure columns to splice.
    drift_warn : coefficient-of-variation threshold above which a seam
        is reported unstable.
    drop_unstable : blank the pre-seam quarters of any column whose
        splice ratio drifted more than ``drift_warn``. Off by default,
        because silently deleting data is its own failure mode — but a
        warning is easy to miss in a notebook, so this is offered for
        anyone who would rather have a gap than a number they did not
        notice was shaky.
    spine : override the OECD panel (mostly for testing).
    return_seams : also return the tidy seam table.

    Returns
    -------
    pd.DataFrame indexed by ``(code, date)`` with ``columns`` plus a
    ``src_<column>`` for each, or ``(frame, seams)`` when
    ``return_seams``.
    """
    if isinstance(codes, str):
        codes = [codes]
    wanted = ([c.upper() for c in codes] if codes is not None
              else sorted(LONG_PANEL_SOURCES))

    if spine is None:
        from ..oecd_qna_panel import qna_panel
        spine = qna_panel(wanted, start=start, refresh=refresh)
    spine = spine.reset_index()

    frames, seam_rows = [], []
    for code in wanted:
        own = spine[spine["code"] == code].set_index("date").sort_index()
        cols = [c for c in columns if c in own.columns]
        if own.empty or not cols:
            warnings.warn(
                f"qna_long_panel: the OECD spine returned nothing usable for "
                f"{code!r}; it is skipped rather than extended.",
                UserWarning, stacklevel=2,
            )
            continue
        segments = [("oecd", own[cols])]
        loader = LONG_PANEL_SOURCES.get(code)
        if loader is not None:
            try:
                segments += _load(loader)(timeout=timeout, use_cache=use_cache)
            except Exception as exc:
                warnings.warn(
                    f"qna_long_panel: archived sources for {code!r} could not "
                    f"be fetched ({type(exc).__name__}: {exc}); falling back "
                    "to the OECD span alone.",
                    UserWarning, stacklevel=2,
                )

        values, prov, seams = splice_frame(
            segments, columns=cols, drift_warn=drift_warn)

        if drop_unstable:
            for col, ss in seams.items():
                bad = [s for s in ss if not s.stable]
                if not bad or col not in values.columns:
                    continue
                cutoff = max(s.date for s in bad)
                mask = values.index < cutoff
                values.loc[mask, col] = float("nan")
                prov.loc[mask, col] = "dropped_unstable"

        out = values.copy()
        for col in prov.columns:
            out[f"src_{col}"] = prov[col]
        out.insert(0, "code", code)
        out.index.name = "date"
        frames.append(out.reset_index().set_index(["code", "date"]))

        for col, ss in seams.items():
            for s in ss:
                seam_rows.append({
                    "code": code, "column": col, "older": s.older,
                    "newer": s.newer, "date": s.date, "overlap_n": s.overlap_n,
                    "ratio": s.ratio, "ratio_drift": s.ratio_drift,
                    "ratio_min": s.ratio_min, "ratio_max": s.ratio_max,
                    "stable": s.stable,
                })

    panel = (pd.concat(frames).sort_index() if frames
             else pd.DataFrame(columns=["code", "date", *columns]).set_index(
                 ["code", "date"]))
    seam_table = pd.DataFrame(seam_rows, columns=[
        "code", "column", "older", "newer", "date", "overlap_n", "ratio",
        "ratio_drift", "ratio_min", "ratio_max", "stable"])

    if len(seam_table) and not seam_table["stable"].all():
        n = int((~seam_table["stable"]).sum())
        worst = seam_table.loc[seam_table["ratio_drift"].idxmax()]
        warnings.warn(
            f"qna_long_panel: {n} of {len(seam_table)} seams are unstable — "
            f"the two vintages disagree about growth, not only level. Worst: "
            f"{worst['code']} {worst['column']} at the {worst['older']} seam, "
            f"drift {worst['ratio_drift']:.2%} over {int(worst['overlap_n'])} "
            "quarters. Pass return_seams=True to see them all, or "
            "drop_unstable=True to blank them.",
            UserWarning, stacklevel=2,
        )

    return (panel, seam_table) if return_seams else panel


def long_panel_residual(panel: pd.DataFrame) -> pd.Series:
    """Expenditure residual per ``(code, date)``.

    Columns are spliced independently, so this is not zero and is not
    meant to be — its size is the honest measure of how much the splice
    and the underlying vintages fail to add up.
    """
    parts = []
    for code, sub in panel.groupby(level="code"):
        r = expenditure_residual(sub.droplevel("code"))
        parts.append(pd.concat({code: r}, names=["code"]))
    return (pd.concat(parts) if parts else pd.Series(dtype=float))


__all__ = [
    "LONG_PANEL_COLUMNS",
    "LONG_PANEL_SOURCES",
    "KNOWN_GAPS",
    "qna_long_panel",
    "long_panel_residual",
]
