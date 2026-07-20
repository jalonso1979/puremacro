"""Shared regime dates across the uncertainty_examples library.

Two distinct schemes — pick by *scope*:

* **Global / cross-country** (``REGIMES_GLOBAL``): six bins.
  Separates the Great Recession (US-led financial shock) from the Euro
  Sovereign Debt crisis (Europe-specific sovereign stress). Use this for
  ICIO and the cross-country panels (panel_M, panel_Q).

      PreGR     1995Q1–2006Q4 — pre-financial-crisis baseline.
      GR        2007Q1–2009Q4 — Great Recession.
      EuroDebt  2010Q1–2013Q4 — peripheral sovereign stress (PIIGS).
      Inter     2014Q1–2019Q4 — calm decade between crises.
      COVID     2020Q1–2021Q4 — pandemic.
      PostCOVID 2022Q1–       — high-inflation / rate-hike era.

* **US-only** (``REGIMES_US``): five bins. The Euro Sovereign Debt
  crisis is irrelevant for US states, so it is folded back into the
  pre-COVID period.

      PreGR     1995Q1–2006Q4
      GR        2007Q1–2009Q4
      PreCOVID  2010Q1–2019Q4 (single bin)
      COVID     2020Q1–2021Q4
      PostCOVID 2022Q1–

* **US political / Trump-era** (``REGIMES_US_POLITICAL``): an
  alternative for US analyses where political shifts matter (EPU,
  policy uncertainty). Five bins by presidency:

      Bush43    2001Q1–2008Q4
      Obama     2009Q1–2016Q4
      Trump1    2017Q1–2020Q4
      Biden     2021Q1–2024Q4
      Trump2    2025Q1–
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Regime:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive


REGIMES_GLOBAL = (
    Regime("PreGR",     pd.Timestamp("1995-01-01"), pd.Timestamp("2006-12-31")),
    Regime("GR",        pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31")),
    Regime("EuroDebt",  pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-31")),
    Regime("Inter",     pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
    Regime("COVID",     pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    Regime("PostCOVID", pd.Timestamp("2022-01-01"), pd.Timestamp("2099-12-31")),
)

REGIMES_US = (
    Regime("PreGR",     pd.Timestamp("1995-01-01"), pd.Timestamp("2006-12-31")),
    Regime("GR",        pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31")),
    Regime("PreCOVID",  pd.Timestamp("2010-01-01"), pd.Timestamp("2019-12-31")),
    Regime("COVID",     pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    Regime("PostCOVID", pd.Timestamp("2022-01-01"), pd.Timestamp("2099-12-31")),
)

REGIMES_US_POLITICAL = (
    Regime("Bush43",    pd.Timestamp("2001-01-01"), pd.Timestamp("2008-12-31")),
    Regime("Obama",     pd.Timestamp("2009-01-01"), pd.Timestamp("2016-12-31")),
    Regime("Trump1",    pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-31")),
    Regime("Biden",     pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31")),
    Regime("Trump2",    pd.Timestamp("2025-01-01"), pd.Timestamp("2099-12-31")),
)

# Backwards-compatible alias: ``REGIMES`` defaults to the global scheme.
REGIMES = REGIMES_GLOBAL


def _assign(dates: pd.Series | pd.DatetimeIndex, scheme) -> pd.Series:
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    out = pd.Series(["OTHER"] * len(dates), index=dates.index, dtype="object")
    for r in scheme:
        mask = (dates >= r.start) & (dates <= r.end)
        out.loc[mask] = r.name
    return out


def assign_regime(dates: pd.Series | pd.DatetimeIndex,
                  scope: str = "global") -> pd.Series:
    """Return a regime label per date.

    Parameters
    ----------
    dates  : Series or DatetimeIndex of dates to label.
    scope  : one of {"global", "us", "us_political"}.
    """
    if scope == "global":
        return _assign(dates, REGIMES_GLOBAL)
    if scope == "us":
        return _assign(dates, REGIMES_US)
    if scope == "us_political":
        return _assign(dates, REGIMES_US_POLITICAL)
    raise ValueError(f"unknown scope {scope!r}; "
                     f"expected one of 'global', 'us', 'us_political'")
