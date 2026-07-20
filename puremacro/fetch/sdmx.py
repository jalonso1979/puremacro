"""Generic SDMX-CSV fetcher for OECD, Eurostat, ECB, and IMF SDMX Central.

SDMX-CSV (Statistical Data and Metadata eXchange — CSV variant) is a
W3C-stewarded wire format that all major statistical agencies expose.
Each provider has its own base URL but the response shape is
consistent: dimension columns + ``TIME_PERIOD`` + ``OBS_VALUE`` +
optional attribute columns.

Generic ``sdmx_get(provider, dataflow, key, ...)`` returns the raw
DataFrame. ``oecd_sdmx_instrument(...)`` wraps the most common case
(one country × one indicator → :class:`puremacro.instruments.Instrument`).

References
----------
SDMX-CSV format: https://sdmx.org/?page_id=4345
OECD SDMX API:  https://sdmx.oecd.org/public/rest/
Eurostat SDMX:   https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1
ECB SDW:         https://data-api.ecb.europa.eu/service
IMF SDMX:        https://sdmxcentral.imf.org/ws/public/sdmxapi/rest
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._http import safe_get_bytes
from ..instruments import Instrument


# Provider URL templates — fill with {dataflow} and {key} placeholders.
# format=csvfile / csvdata triggers SDMX-CSV per provider's convention.
_PROVIDERS: dict[str, str] = {
    "oecd": "https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}?format=csvfilewithlabels",
    "eurostat": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataflow}/{key}?format=SDMX-CSV",
    "ecb": "https://data-api.ecb.europa.eu/service/data/{dataflow}/{key}?format=csvdata",
    "imf": "https://sdmxcentral.imf.org/ws/public/sdmxapi/rest/data/{dataflow}/{key}?format=sdmx-csv",
    "ilostat": "https://sdmx.ilo.org/rest/data/{dataflow}/{key}?format=csv&dimensionAtObservation=AllDimensions",
}

_REFERENCE_TEMPLATE = (
    "SDMX-CSV from provider {provider!r}, dataflow {dataflow!r}. "
    "See https://sdmx.org/?page_id=4345 for format spec."
)


def sdmx_get(
    *,
    provider: str,
    dataflow: str,
    key: str = "all",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch an SDMX-CSV response and return as a DataFrame.

    Parameters
    ----------
    provider : str
        One of ``"oecd"``, ``"eurostat"``, ``"ecb"``, ``"imf"``. Each maps
        to a known base URL template (see :data:`_PROVIDERS`).
    dataflow : str
        Provider-specific dataflow ID (e.g. ``"DSD_STAN"`` for OECD-STAN).
    key : str, default ``"all"``
        SDMX dot-separated dimension key. ``"all"`` returns everything;
        e.g. ``"USA"`` filters by REF_AREA.
    csv_path : str | Path | None
        Optional local path to a pre-downloaded SDMX-CSV file. When
        None, attempt the live network fetch. Note: when ``csv_path``
        is set, the ``key`` argument is ignored — the entire CSV is
        returned and the caller is responsible for any further
        filtering.

    Returns
    -------
    pd.DataFrame
        Raw SDMX-CSV columns: dimension cols + ``TIME_PERIOD`` +
        ``OBS_VALUE`` + attributes. Filtering is the caller's job
        (use :func:`oecd_sdmx_instrument` for the common case).

    Raises
    ------
    ValueError
        If ``provider`` is not in the known providers whitelist.
    RuntimeError
        If the network fetch fails (and no ``csv_path=`` was provided).
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"provider {provider!r} not in known providers: "
            f"{sorted(_PROVIDERS.keys())}"
        )
    if csv_path is not None:
        return pd.read_csv(csv_path)
    url = _PROVIDERS[provider].format(dataflow=dataflow, key=key)
    try:
        raw = safe_get_bytes(url)
    except Exception:
        raise RuntimeError(
            f"Could not fetch SDMX from {provider!r} (dataflow={dataflow!r}, "
            f"key={key!r}). Verify the dataflow ID at the provider's portal "
            f"and pass csv_path= with a local download to skip the network."
        ) from None
    return pd.read_csv(io.BytesIO(raw))


def oecd_sdmx_instrument(
    *,
    dataset: str,
    country: str,
    indicator: str,
    csv_path: str | Path | None = None,
    frequency: str = "A",
    measure_col: str = "MEASURE",
) -> Instrument:
    """Convenience: fetch one (country × indicator) slice from OECD-SDMX
    and return as an :class:`Instrument`.

    Parameters
    ----------
    dataset : str
        OECD dataflow ID (e.g. ``"DSD_STAN"`` for STAN industrial data,
        ``"DF_FUNCTIONAL"`` for fiscal indicators).
    country : str
        ISO3 country code, matching the ``REF_AREA`` column.
    indicator : str
        Code matching the ``measure_col`` column (default ``MEASURE``).
        For STAN this is e.g. ``"VALADD"`` (value added),
        ``"EMPN"`` (employment), etc.
    csv_path : str | Path | None
        Optional local SDMX-CSV.
    frequency : str, default ``"A"``
        Currently only ``"A"`` (annual) is supported. Quarterly and
        monthly OECD datasets require manual handling via
        :func:`sdmx_get`. A ``ValueError`` is raised for other values.
    measure_col : str, default ``"MEASURE"``
        Column name carrying the indicator code. SDMX naming varies
        slightly across OECD dataflows.

    Returns
    -------
    Instrument
        Time series indexed by ``TIME_PERIOD``, name
        ``f"oecd_{dataset}_{country}_{indicator}"``,
        category ``"external_csv"``.
    """
    if frequency != "A":
        raise ValueError(
            f"oecd_sdmx_instrument currently only handles annual data "
            f"(frequency='A'). Got frequency={frequency!r}. For quarterly "
            f"or monthly OECD datasets, use sdmx_get() directly and "
            f"construct the Instrument manually."
        )
    df = sdmx_get(provider="oecd", dataflow=dataset, key=country,
                  csv_path=csv_path)
    expected = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", measure_col}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"OECD SDMX response missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. Pass measure_col= if the indicator "
            f"column has a non-default name."
        )
    available_countries = sorted(df["REF_AREA"].dropna().unique().tolist())
    if country not in available_countries:
        raise ValueError(
            f"country {country!r} not in OECD response; available: "
            f"{available_countries}"
        )
    sub = df[(df["REF_AREA"] == country) & (df[measure_col] == indicator)].copy()
    if sub.empty:
        raise ValueError(
            f"no rows in OECD response for country={country!r}, "
            f"indicator={indicator!r}. Available indicators for {country}: "
            f"{sorted(df[df['REF_AREA'] == country][measure_col].dropna().unique().tolist())}"
        )
    sub = sub.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    dates = pd.to_datetime(sub["TIME_PERIOD"].astype(str) + "-01-01")
    name = f"oecd_{dataset}_{country}_{indicator}"
    series = pd.Series(
        sub["OBS_VALUE"].astype(float).values,
        index=dates,
        name=name,
    ).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=f"OECD SDMX {dataset} ({country}, {indicator})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE_TEMPLATE.format(
                provider="oecd", dataflow=dataset,
            ),
            "provider": "oecd",
            "dataset": dataset,
            "country": country,
            "indicator": indicator,
        },
    )


__all__ = ["sdmx_get", "oecd_sdmx_instrument"]
