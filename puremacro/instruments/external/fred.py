"""FRED (Federal Reserve Economic Data) generic series loader.

FRED is the public economic-data archive of the Federal Reserve Bank of
St. Louis, hosting tens of thousands of macro / financial / regional
series. Access requires a free API key (register at
https://fred.stlouisfed.org/docs/api/api_key.html).

This loader fetches a single series by ID via the
``/fred/series/observations`` endpoint, parses the JSON response, and
returns an :class:`Instrument`. The frequency is supplied by the caller
(FRED's metadata can also report it but we accept it explicitly so
catalog entries are self-documenting).

Reference
---------
Federal Reserve Bank of St. Louis (n.d.). FRED® Economic Data API.
https://fred.stlouisfed.org/docs/api/fred/
"""
from __future__ import annotations

import json
import urllib.parse

import pandas as pd

from .._core import Instrument
from .._helpers import _json_to_instrument
from ...narrative.sources._http import safe_get_text
from ... import credentials


_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _build_url(
    series_id: str,
    api_key: str,
    *,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> str:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start is not None:
        params["observation_start"] = observation_start
    if observation_end is not None:
        params["observation_end"] = observation_end
    return _BASE + "?" + urllib.parse.urlencode(params)


def load(
    *,
    series_id: str,
    api_key: str | None = None,
    frequency: str = "M",
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> Instrument:
    """Load a FRED series as an :class:`Instrument`.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``"FEDFUNDS"``, ``"NFCI"``).
    api_key : str | None
        Free FRED API key. If None, resolved via
        :func:`puremacro.credentials.require` (checks ``FRED_API_KEY``
        env var and any configured credential store).
    frequency : str, default ``"M"``
        Pandas-style frequency code recorded on the resulting
        Instrument. Pass ``"W"`` for weekly, ``"D"`` for daily, etc.
        FRED returns the series at its native frequency; this kwarg
        is metadata only (no resampling).
    observation_start, observation_end : str | None
        Optional ISO date strings (``"YYYY-MM-DD"``) restricting the
        FRED query window.

    Returns
    -------
    Instrument
        Series indexed by observation date, name ``f"fred_{series_id}"``,
        category ``"external_csv"``.
    """
    key = credentials.require("fred", explicit=api_key)
    url = _build_url(
        series_id, key,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    try:
        text = safe_get_text(url)
        payload = json.loads(text)
    except Exception as e:
        # Suppress the exception chain to prevent the API key in the URL
        # from leaking into traceback output. Caller still sees the type
        # of failure via the message.
        raise RuntimeError(
            f"Could not fetch FRED series {series_id!r} ({type(e).__name__}). "
            f"Verify the series ID exists at https://fred.stlouisfed.org/"
            f"series/{series_id} and that the API key is valid."
        ) from None
    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(
            f"FRED returned no observations for {series_id!r}. Check the "
            f"series ID and the requested date window."
        )
    return _json_to_instrument(
        observations,
        name=f"fred_{series_id}",
        source=f"FRED series {series_id}",
        frequency=frequency,
        date_field="date",
        value_field="value",
        metadata={
            "reference": (
                "Federal Reserve Bank of St. Louis. FRED Economic Data. "
                f"https://fred.stlouisfed.org/series/{series_id}"
            ),
            "series_id": series_id,
        },
    )


__all__ = ["load"]
