"""ECB Data Portal — Real Time Database (the EABCN area-wide RTDB).

Keyless SDMX at ``https://data-api.ecb.europa.eu/service``.

TWO THINGS THAT SURPRISE PEOPLE
-------------------------------
**1. The RTD flow has no vintage dimension.** Its DSD (``ECB_RTD1``) is
``FREQ.REF_AREA.ADJUSTMENT.RT_ECON_CONCEPT.RT_DENOM`` and nothing else.
A plain data request therefore returns the *current* vintage only, and
returns it silently — a well-formed response with one value per period
and no hint that the archive was not consulted. Vintages come from the
SDMX 2.1 ``includeHistory=true`` parameter, which adds ``ACTION``,
``VALID_FROM`` and ``VALID_TO`` columns and emits one row per past
version of each observation.

**2. ``asOf`` does not work.** The SDMX ``asOf`` parameter is accepted
and then ignored: asking for a 2016 snapshot returns today's numbers.
Never rely on it; filter ``VALID_FROM`` yourself, which is what
:func:`puremacro.vintages.as_of` does once the data is tidy.

COVERAGE IS NARROW
------------------
Only three reference areas exist in this flow: ``S0`` (euro area),
``JP`` and ``US``. There is no Germany, no Spain, no individual member
state. For country-level euro-area vintages use the OECD STES
revisions archive (:mod:`puremacro.fetch.realtime.oecd_stes`), which
covers 42 economies.

What RTD does give is depth: version history for euro-area GDP runs
back to 2001, and ``S0`` is the euro area on a *moving* composition —
membership changes across vintages, so levels are not
composition-consistent over the long run.
"""
from __future__ import annotations

import io

import pandas as pd

from ..._http import safe_get_bytes, safe_get_bytes_cached
from ._base import VintagePanel, normalize_vintage_frame, register_provider
from .catalog import SeriesSpec, register_catalog


ECB_RTD_URL = (
    "https://data-api.ecb.europa.eu/service/data/RTD/{key}"
    "?includeHistory=true&format=csvdata"
)

_UA = "puremacro (real-time vintage reader)"

#: ISO3 -> the flow's REF_AREA code. ``EA`` is the moving-composition
#: euro area the EABCN database is defined on.
ECB_RTD_AREAS: dict[str, str] = {"EA": "S0", "JPN": "JP", "USA": "US"}

#: Canonical variable -> (RT_ECON_CONCEPT, RT_DENOM, units).
ECB_RTD_CONCEPTS: dict[str, tuple[str, str, str]] = {
    "gdp_real": ("G_GDPM_TO_C", "E", "level"),
    "gdp_nom": ("G_GDPM_TO_U", "E", "level"),
    "deflator": ("G_GDPM_TO_D", "E", "index"),
}


def build_key(area: str, concept: str, denom: str, *, freq: str = "Q",
              adjustment: str = "S") -> str:
    """Assemble an RTD series key: ``FREQ.REF_AREA.ADJ.CONCEPT.DENOM``."""
    return f"{freq}.{area}.{adjustment}.{concept}.{denom}"


def parse_ecb_history_csv(raw: bytes | str) -> pd.DataFrame:
    """Parse an ``includeHistory=true`` CSV into ``[date, vintage, value]``.

    Pure function. ``VALID_FROM`` is the dissemination timestamp of that
    version and becomes the vintage, normalised to a date.

    ``ACTION == "Delete"`` rows carry no ``OBS_VALUE``: they record a
    withdrawal, not an observation, and are dropped. Note the
    consequence — a period that was published and later withdrawn keeps
    its last real value here rather than reverting to missing.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or not raw.strip():
        return pd.DataFrame(columns=["date", "vintage", "value"])
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    needed = {"TIME_PERIOD", "OBS_VALUE", "VALID_FROM"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"ECB RTD response missing {sorted(missing)} — was "
            f"includeHistory=true set? Got {sorted(df.columns)[:12]}..."
        )
    if "ACTION" in df.columns:
        df = df[df["ACTION"].astype(str).str.lower() != "delete"]
    period = df["TIME_PERIOD"].astype(str)
    out = pd.DataFrame({
        "date": pd.PeriodIndex(period, freq="Q").to_timestamp(),
        "vintage": pd.to_datetime(df["VALID_FROM"], errors="coerce",
                                  utc=True, format="mixed").dt.tz_localize(None).dt.normalize(),
        "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
    }).dropna(subset=["date", "vintage", "value"])
    return (out.drop_duplicates(subset=["date", "vintage"], keep="last")
               .sort_values(["date", "vintage"]).reset_index(drop=True))


def fetch_ecb_rtd_vintages(
    country: str = "EA",
    variable: str = "gdp_real",
    *,
    freq: str = "Q",
    timeout: float = 120.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long ``[date, vintage, value]`` for one RTD series."""
    c = str(country).upper()
    if c not in ECB_RTD_AREAS:
        raise ValueError(
            f"ECB RTD covers only {sorted(ECB_RTD_AREAS)}; got {country!r}. "
            "For country-level euro-area vintages use provider 'oecd_stes'."
        )
    if variable not in ECB_RTD_CONCEPTS:
        raise ValueError(
            f"no ECB RTD concept mapped for {variable!r}; available: "
            f"{sorted(ECB_RTD_CONCEPTS)}"
        )
    concept, denom, _units = ECB_RTD_CONCEPTS[variable]
    key = build_key(ECB_RTD_AREAS[c], concept, denom, freq=freq)
    getter = safe_get_bytes_cached if use_cache else safe_get_bytes
    return parse_ecb_history_csv(
        getter(ECB_RTD_URL.format(key=key), timeout, user_agent=_UA))


def fetch_ecb_rtd_panel(
    countries, variables, *, timeout: float = 120.0, use_cache: bool = True,
    **_ignored,
) -> VintagePanel:
    """Registry entry point."""
    frames, failed = [], {}
    for country in countries:
        c = str(country).upper()
        if c not in ECB_RTD_AREAS:
            continue
        for variable in variables:
            if variable not in ECB_RTD_CONCEPTS:
                continue
            concept, denom, units = ECB_RTD_CONCEPTS[variable]
            try:
                long = fetch_ecb_rtd_vintages(
                    c, variable, timeout=timeout, use_cache=use_cache)
            except Exception as exc:
                failed[f"{c}:{variable}"] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(normalize_vintage_frame(
                long, country=c, variable=variable, provider="ecb_rtd",
                series_id=build_key(ECB_RTD_AREAS[c], concept, denom),
                units=units,
            ))
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=[
              "country", "variable", "date", "vintage", "value", "provider",
              "series_id", "units"]))
    return VintagePanel(df=df, metadata={"provider": "ecb_rtd",
                                         "failed": failed})


def _register() -> None:
    register_catalog("ecb_rtd", {
        iso3: {
            var: SeriesSpec(
                build_key(area, concept, denom), units, "ecb_rtd",
                "EABCN area-wide real-time database; vintages via "
                "includeHistory")
            for var, (concept, denom, units) in ECB_RTD_CONCEPTS.items()
        }
        for iso3, area in ECB_RTD_AREAS.items()
    })
    register_provider("ecb_rtd", fetch_ecb_rtd_panel, list(ECB_RTD_AREAS))


__all__ = [
    "ECB_RTD_URL",
    "ECB_RTD_AREAS",
    "ECB_RTD_CONCEPTS",
    "build_key",
    "parse_ecb_history_csv",
    "fetch_ecb_rtd_vintages",
    "fetch_ecb_rtd_panel",
]
