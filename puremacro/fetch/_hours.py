"""Detect a reference area whose hours are published on the wrong time base.

The OECD publishes some reference areas' hours per *week*, or at an *annual
rate*, under exactly the same ``UNIT_MEASURE`` and ``UNIT_MULT`` as everyone
else's per-period figure. Nothing in the SDMX message distinguishes them: the
series still moves correctly and only its level is absurd, which is the one
direction a caller is least likely to sanity-check. On the quarterly flow it is
Chile (weekly, 13x) and Costa Rica (annual rate, 0.25x); on the annual flow it
is New Zealand (weekly, 52x).

The detection is the same arithmetic on both, and only the bands and the
candidate factors differ, so it lives here once and is parameterised by period
rather than written twice with two chances to drift. :func:`hours_scale_factors`
is the whole of it; the callers own their own bands, because "hours per worker
per quarter" and "hours per worker per year" are different numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hours_scale_factors(heads: pd.Series, hours: pd.Series, *,
                        implausible: tuple[float, float],
                        plausible: tuple[float, float],
                        scales: dict[float, str]) -> dict[str, float]:
    """``{code: factor}`` for every reference area whose hours need rescaling.

    ``heads`` and ``hours`` are indexed alike, with the reference area on level
    0; heads arrive in thousands and hours in millions, so the ratio is hours
    per worker per period once the ``10**3`` between them is undone.

    A code is a candidate only if its median ratio falls *outside*
    ``implausible`` — a band set far wider than any real economy, so it can
    only ever fire on an order-of-magnitude error and never on a country that
    merely works short weeks. A candidate factor from ``scales`` is then
    accepted only if it lands the median *inside* ``plausible``, which is
    tighter: a rescaling that does not produce a believable working period is
    not applied at all and the series is left exactly as published. A reference
    area that publishes hours but no heads cannot be checked and is left alone.
    """
    both = pd.concat({"h": hours, "p": heads}, axis=1).dropna()
    if both.empty:
        return {}
    ratio = (both["h"] * 1e3 / both["p"]).groupby(level=0).median()

    lo, hi = implausible
    ok_lo, ok_hi = plausible
    out: dict[str, float] = {}
    for code, r in ratio.items():
        if not np.isfinite(r) or lo <= r <= hi:
            continue
        for factor in scales:
            if ok_lo <= r * factor <= ok_hi:
                out[str(code)] = factor
                break
    return out
