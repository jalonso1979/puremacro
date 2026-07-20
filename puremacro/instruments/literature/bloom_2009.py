"""Bloom (2009) uncertainty shock indicator series.

Bloom identifies 17 large-uncertainty episodes from major political,
economic, or financial events with associated stock-volatility spikes.
Each event is a discrete announcement-month date. This loader emits a
monthly indicator series (value 1.0 at each event month, 0.0 elsewhere)
over the standard sample Jan-1962 to Dec-2008.

The 17 dates are the canonical list from Bloom (2009, Econometrica)
Table A.1, plus the credit crunch episode of October 2008 (added
after the paper's submission cutoff and widely adopted in the
subsequent uncertainty-shocks literature). They are baked into this
module rather than fetched from a website because the list is small,
stable, and reproducible without network access.

Reference
---------
Bloom, N. (2009). The impact of uncertainty shocks. Econometrica 77(3), 623-685.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._core import Instrument


BLOOM_2009_EVENTS: tuple[str, ...] = (
    "1962-10-01",  # Cuban Missile Crisis
    "1963-11-01",  # Assassination of JFK
    "1966-08-01",  # Vietnam buildup
    "1970-05-01",  # Cambodia / Kent State
    "1973-12-01",  # OPEC I, Arab-Israeli War
    "1974-10-01",  # Franklin National
    "1978-11-01",  # OPEC II
    "1980-03-01",  # Afghanistan, Iran hostage crisis
    "1982-10-01",  # Monetary cycle turning point
    "1987-11-01",  # Black Monday
    "1990-10-01",  # Gulf War I
    "1997-11-01",  # Asian Crisis
    "1998-09-01",  # Russian / LTCM
    "2001-09-01",  # 9/11
    "2002-09-01",  # WorldCom and Enron
    "2003-02-01",  # Gulf War II
    "2008-10-01",  # Credit crunch / GFC peak
)


_REFERENCE = (
    "Bloom, N. (2009). The impact of uncertainty shocks. "
    "Econometrica 77(3), 623-685."
)


def load() -> Instrument:
    """Return Bloom (2009) uncertainty-event indicator series.

    Returns
    -------
    Instrument
        Monthly series spanning Jan-1962 through Dec-2008 (564 obs).
        Value 1.0 at each of the 17 event months; 0.0 elsewhere.
        Category ``"literature"``, frequency ``"M"``.
    """
    idx = pd.date_range("1962-01-01", "2008-12-01", freq="MS")
    s = pd.Series(np.zeros(len(idx)), index=idx, name="bloom_2009_uncertainty")
    for date in BLOOM_2009_EVENTS:
        ts = pd.Timestamp(date)
        if ts in s.index:
            s.loc[ts] = 1.0
    return Instrument(
        series=s,
        name="bloom_2009_uncertainty",
        source="Bloom 2009 large-uncertainty episodes (Table A.1)",
        category="literature",
        frequency="M",
        metadata={
            "reference": _REFERENCE,
            "event_dates": list(BLOOM_2009_EVENTS),
        },
    )


__all__ = ["load", "BLOOM_2009_EVENTS"]
