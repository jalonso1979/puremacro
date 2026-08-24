"""Which editions of a vintage archive carry an *unadjusted* series.

WHY THIS EXISTS
---------------
``OECD.SDD.STES,DSD_STES_REVISIONS`` has six dimensions and none of them is
seasonal adjustment. It carries whatever the OECD ingested at the time, and
for twelve reference areas the early editions are the **raw** series — the
archive switched over area by area between 2000 and 2007 without saying so
anywhere in the metadata.

Nothing downstream survives that. Run a news-vs-noise test on the panel as it
arrives and Sweden comes back with ``beta = -0.96`` and a revision standard
deviation of 5.1 percentage points a quarter; Turkey with ``-0.96`` and 10.9.
Read literally, those say that essentially all of the variance of the Swedish
and Turkish first releases is measurement error. What they actually say is
that Sweden's first estimate of 2002Q4 was ``+16%`` and its current one is
``+0.07%``, because the first was not seasonally adjusted. The "revision"
being regressed is a seasonal factor. The same contamination lands on any HP
trend fitted through such an edition, so a real-time output gap is hit too.

This is not a defect in the connectors — they report the archive faithfully —
and it cannot be read off the metadata, because the archive does not record
it. It can be read off the **data**, and that is what this module does.

THE SCREEN
----------
For each ``(country, edition)`` pair, on that edition's *own* quarterly log
growth (so a rebasing, which multiplies a whole edition by a constant,
cancels), over a window that excludes the pandemic:

1. ``F`` of quarterly dummies — is the quarterly pattern there at all?
2. ``seasonal_range`` — the strongest quarterly mean minus the weakest, in
   percentage points of quarterly growth. Is it *big*?

An edition is called unadjusted when **both** exceed their thresholds. Neither
does the job alone. The ``F`` is a significance test, and with a hundred
quarters it rejects for residual seasonality far too small to matter — which
is what would flag Iceland's genuinely adjusted recent editions. The range is
the discriminating statistic, but on its own a short or outlier-ridden edition
can show a large apparent spread with no systematic pattern, and the ``F`` is
what refuses to flag it.

The default thresholds are not conventional critical values. On the STES
archive the two populations separate by more than an order of magnitude on
both axes, and ``SEASONAL_RANGE_MIN`` sits in the empty gap between them:
adjusted editions reach at most 4.1 pp, and the mildest genuinely raw series
(Portugal's) starts at 4.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import VintagePanel

#: Minimum F of quarterly dummies for an edition to be called unadjusted.
SEASONAL_F_MIN = 10.0

#: Minimum peak-to-trough range of the four quarterly mean growth rates, in
#: percentage points. Sits in the empty gap between the two populations.
SEASONAL_RANGE_MIN = 4.25

#: Quarters needed before an edition is scored at all.
SEASONAL_MIN_OBS = 16

#: Default scoring window: long enough to identify a seasonal, and stopping
#: before a pandemic that would masquerade as one.
SEASONAL_WINDOW = ("1995-01-01", "2020-01-01")


def seasonal_signature(
    panel: VintagePanel | pd.DataFrame,
    *,
    window: tuple[str, str] = SEASONAL_WINDOW,
    f_min: float = SEASONAL_F_MIN,
    range_min: float = SEASONAL_RANGE_MIN,
    min_obs: int = SEASONAL_MIN_OBS,
) -> pd.DataFrame:
    """Score every ``(country, variable, edition)`` for residual seasonality.

    Vectorised over editions — one four-row group-mean per (country,
    variable), not one regression per edition, of which a full STES panel has
    eleven thousand.

    Returns
    -------
    pd.DataFrame with columns ``[country, variable, vintage, n, F,
    seasonal_range, unadjusted]``, one row per scored edition.
    """
    df = panel.df if isinstance(panel, VintagePanel) else panel
    #: The column set is part of the contract: callers index straight
    #: into it, so an empty panel must still come back shaped correctly
    #: rather than as a bare DataFrame that raises a KeyError.
    cols = ["country", "variable", "vintage", "n", "F", "seasonal_range",
            "unadjusted"]
    if df is None or len(df) == 0 or not {"country", "variable", "vintage",
                                          "date", "value"} <= set(df.columns):
        empty = pd.DataFrame(columns=cols)
        return empty.astype({"unadjusted": bool})
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    out = []
    for (country, variable), sub in df.groupby(["country", "variable"],
                                               sort=True):
        piv = (sub.pivot_table(index="date", columns="vintage", values="value")
                  .sort_index())
        g = 100.0 * np.log(piv).diff()
        g = g[(g.index >= lo) & (g.index < hi)]
        if g.empty:
            continue
        q = g.index.quarter
        n = g.notna().sum()
        m = g.groupby(q).mean()
        n_q = g.groupby(q).count()
        ssb = (n_q * (m - g.mean()) ** 2).sum()          # between quarters
        ssw = ((g - m.reindex(q).to_numpy()) ** 2).sum()  # within
        with np.errstate(invalid="ignore", divide="ignore"):
            F = (ssb / 3.0) / (ssw / (n - 4))
        out.append(pd.DataFrame({
            "country": country, "variable": variable, "vintage": g.columns,
            "n": n.to_numpy(), "F": F.to_numpy(),
            "seasonal_range": (m.max() - m.min()).to_numpy(),
        }))
    if not out:
        return pd.DataFrame(columns=["country", "variable", "vintage", "n", "F",
                                     "seasonal_range", "unadjusted"])
    sig = pd.concat(out, ignore_index=True)
    sig = sig[(sig["n"] >= min_obs) & np.isfinite(sig["F"])].reset_index(drop=True)
    sig["unadjusted"] = (sig["F"] > f_min) & (sig["seasonal_range"] >= range_min)
    return sig


def drop_unadjusted_editions(
    panel: VintagePanel,
    *,
    window: tuple[str, str] = SEASONAL_WINDOW,
    f_min: float = SEASONAL_F_MIN,
    range_min: float = SEASONAL_RANGE_MIN,
    min_obs: int = SEASONAL_MIN_OBS,
) -> tuple[VintagePanel, pd.DataFrame]:
    """Remove the editions :func:`seasonal_signature` calls unadjusted.

    Returns ``(panel, report)``. The report is one row per affected country
    with how many editions went, the first and last of them, and their median
    seasonal range — because a screen that does not say what it removed is
    indistinguishable from a bug.

    An area never loses everything: on the STES archive every flagged edition
    is an early one, so what a country loses is the first years of its own
    first releases, not a slice out of the middle.
    """
    sig = seasonal_signature(panel, window=window, f_min=f_min,
                             range_min=range_min, min_obs=min_obs)
    if sig.empty:
        report = pd.DataFrame(columns=["editions", "first", "last",
                                       "median_range"])
        report.index.name = "country"
        return panel, report
    bad = sig[sig["unadjusted"]]
    flagged = set(map(tuple, bad[["country", "variable", "vintage"]]
                      .itertuples(index=False, name=None)))
    keep = [k not in flagged for k in zip(panel.df["country"],
                                          panel.df["variable"],
                                          panel.df["vintage"])]
    kept = VintagePanel(df=panel.df[keep].copy(),
                        metadata={**panel.metadata,
                                  "unadjusted_editions": len(flagged)})
    if bad.empty:
        report = pd.DataFrame(columns=["editions", "first", "last",
                                       "median_range"])
        report.index.name = "country"
        return kept, report
    report = (bad.groupby("country")
                 .agg(editions=("vintage", "size"),
                      first=("vintage", "min"),
                      last=("vintage", "max"),
                      median_range=("seasonal_range", "median")))
    return kept, report


__all__ = ["seasonal_signature", "drop_unadjusted_editions",
           "SEASONAL_F_MIN", "SEASONAL_RANGE_MIN", "SEASONAL_MIN_OBS",
           "SEASONAL_WINDOW"]
