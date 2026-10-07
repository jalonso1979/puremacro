"""US Census Bureau Microdata API: ACS PUMS and CPS basic monthly.

Returns :class:`~puremacro.fetch.micro.MicroFrame` objects whose design
is set from the survey's own documentation:

* **ACS PUMS** — the main weight plus its 80 successive-difference
  replicate weights are always requested (``PWGTP``/``PWGTP1..80`` for
  persons, ``WGTP``/``WGTP1..80`` for households), so ``.mean()`` and
  friends return standard errors the Census Bureau would recognise.
* **CPS basic monthly** — weights only. The public files carry neither
  replicate weights nor strata/PSU, so standard errors are reported as
  ``NaN`` rather than understated.

THE 50-VARIABLE CAP
-------------------
The API serves at most 50 variables per request. Weights alone are 81
columns, so requests are split into chunks that each repeat the record
identifiers (``SERIALNO`` + ``SPORDER`` for PUMS) and are re-joined on
them. A record missing from any chunk is an error, not a silent drop.

KEY AND CACHE
-------------
The API requires a key (free: https://api.census.gov/data/key_signup.html),
resolved through :mod:`puremacro.credentials` (``CENSUS_API_KEY`` or
``[census].api_key``). It is sent as a secret parameter of
:func:`puremacro.fetch._http.cached_get`, so it never reaches the on-disk
cache, the manifest, or an error message.

GEOGRAPHY
---------
PUMS is published at state and PUMA level only. Pass the API's own
syntax: ``geography="state:06"`` or ``geography="public use microdata
area:*", within="state:06"``.

DOLLAR AMOUNTS
--------------
ACS income and housing-cost variables are in the survey year's nominal
dollars *before* the ``ADJINC`` / ``ADJHSG`` factors are applied
(divide the factor by 10**6). Request ``ADJINC`` and adjust explicitly;
this module does not do it silently.
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ... import credentials
from .. import _http
from ._design import MicroFrame, SurveyDesign

API_ROOT = "https://api.census.gov/data"
MAX_VARIABLES = 50
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
           "oct", "nov", "dec")

PUMS_IDS = ("SERIALNO", "SPORDER")
CPS_IDS = ("HRHHID", "HRHHID2", "PULINENO")


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------
def _predicates(where: Mapping | None) -> dict[str, str]:
    """``{"AGEP": (25, 64), "SEX": 1, "ST": ["06", "36"]}`` -> API predicates."""
    out = {}
    for var, val in (where or {}).items():
        if isinstance(val, tuple) and len(val) == 2:
            out[var] = f"{val[0]}:{val[1]}"
        elif isinstance(val, (list, set)):
            out[var] = ",".join(str(v) for v in val)
        else:
            out[var] = str(val)
    return out


def _chunks(ids: Sequence[str], variables: Sequence[str],
            limit: int | None = None) -> list[list[str]]:
    room = (limit or MAX_VARIABLES) - len(ids)
    if room <= 0:
        raise ValueError("identifier columns alone exceed the variable cap")
    return [list(variables[i:i + room]) for i in range(0, len(variables), room)] or [[]]


def _parse(payload: bytes) -> pd.DataFrame:
    """Census JSON (list of rows, header first) -> string DataFrame.

    The API answers HTTP 204 with an empty body when no record matches.
    """
    if not payload or not payload.strip():
        return pd.DataFrame()
    rows = json.loads(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[1:], columns=rows[0])


def _get(url: str, params: dict, *, api_key: str | None, refresh: bool,
         timeout: int) -> pd.DataFrame:
    key = credentials.require("census", explicit=api_key)
    raw = _http.cached_get(url, params=params, secret_params={"key": key},
                           refresh=refresh, timeout=timeout)
    return _parse(raw)


def _to_numeric(df: pd.DataFrame, keep_text: Iterable[str]) -> pd.DataFrame:
    """Convert every column that is fully numeric; leave codes like
    ``SERIALNO`` (and anything with letters) as strings."""
    keep_text = set(keep_text)
    for c in df.columns:
        if c in keep_text:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == df[c].notna().sum():
            df[c] = conv
    return df


def fetch_records(url: str, variables: Sequence[str], ids: Sequence[str], *,
                  geography: str | None = None, within: str | None = None,
                  where: Mapping | None = None, api_key: str | None = None,
                  refresh: bool = False, timeout: int = 180) -> pd.DataFrame:
    """Pull ``variables`` from a Microdata API endpoint, chunking past the
    50-variable cap and re-joining on ``ids``."""
    ids = list(ids)
    payload = [v for v in dict.fromkeys(variables) if v not in ids]
    base = _predicates(where)
    if geography:
        base["for"] = geography
    if within:
        base["in"] = within
    merged: pd.DataFrame | None = None
    for chunk in _chunks(ids, payload):
        df = _get(url, {"get": ",".join(ids + chunk), **base}, api_key=api_key,
                  refresh=refresh, timeout=timeout)
        if df.empty:
            return pd.DataFrame(columns=ids + payload)
        if df.duplicated(ids).any():
            raise ValueError(f"{url}: identifiers {ids} do not uniquely "
                             "identify records; cannot re-join chunks")
        if merged is None:
            merged = df
            continue
        df = df.drop(columns=[c for c in df.columns
                              if c in merged.columns and c not in ids])
        joined = merged.merge(df, on=ids, how="outer", indicator=True)
        if (joined["_merge"] != "both").any():
            raise ValueError(f"{url}: chunks returned different record sets; "
                             "refusing to splice them")
        merged = joined.drop(columns="_merge")
    return merged


def variables(dataset: str = "acs/acs1/pums", year: int = 2023, *,
              refresh: bool = False, timeout: int = 60) -> pd.DataFrame:
    """Codebook for an endpoint: name, label, predicate type, group."""
    raw = _http.cached_get(f"{API_ROOT}/{year}/{dataset}/variables.json",
                           refresh=refresh, timeout=timeout)
    v = json.loads(raw).get("variables", {})
    rows = [{"name": k, "label": d.get("label", ""),
             "predicate_type": d.get("predicateType", ""),
             "group": d.get("group", "")} for k, d in v.items()]
    return pd.DataFrame(rows).sort_values("name", ignore_index=True)


# ---------------------------------------------------------------------------
# ACS PUMS
# ---------------------------------------------------------------------------
def fetch_acs_pums(
    year: int,
    variables: Sequence[str],
    *,
    survey: str = "acs1",
    unit: str = "person",
    geography: str | None = None,
    within: str | None = None,
    where: Mapping | None = None,
    replicate_weights: bool = True,
    api_key: str | None = None,
    refresh: bool = False,
    timeout: int = 180,
) -> MicroFrame:
    """ACS PUMS records with their variance design.

    Parameters
    ----------
    year : survey year (1-year) or final year of the 5-year period.
    variables : PUMS variable names, e.g. ``["AGEP", "WAGP", "ADJINC"]``.
    survey : ``"acs1"`` or ``"acs5"``.
    unit : ``"person"`` (one row per person, ``PWGTP``) or ``"household"``
        (one row per occupied housing unit — the ``SPORDER == 1`` record —
        weighted by ``WGTP``). Group-quarters persons carry ``WGTP == 0``.
    geography, within : API ``for=`` / ``in=`` clauses (state or PUMA).
    where : predicates, e.g. ``{"AGEP": (25, 64), "SEX": 2}``.
    replicate_weights : set False to skip the 80 replicates (point
        estimates only; ``se`` becomes ``NaN``).

    Examples
    --------
    >>> acs = fetch_acs_pums(2023, ["AGEP", "WAGP", "ADJINC"],
    ...                      geography="state:11")          # doctest: +SKIP
    >>> acs.mean("WAGP")                                     # doctest: +SKIP
    """
    if survey not in ("acs1", "acs5"):
        raise ValueError("survey must be 'acs1' or 'acs5'")
    if unit not in ("person", "household"):
        raise ValueError("unit must be 'person' or 'household'")
    design = (SurveyDesign.acs_pums(unit) if replicate_weights else
              SurveyDesign.weights_only(
                  "PWGTP" if unit == "person" else "WGTP",
                  note="replicate weights not requested"))
    url = f"{API_ROOT}/{year}/acs/{survey}/pums"
    wanted = list(variables) + design.columns()
    df = fetch_records(url, wanted, PUMS_IDS, geography=geography,
                       within=within, where=where, api_key=api_key,
                       refresh=refresh, timeout=timeout)
    df = _to_numeric(df, keep_text=["SERIALNO"])
    if unit == "household":
        df = df[df["SPORDER"] == 1].reset_index(drop=True)
    return MicroFrame(
        data=df, design=design, unit=unit, source="census.acs_pums",
        vintage=f"{year} {survey}",
        query=_describe(url, wanted, geography, within, where))


# ---------------------------------------------------------------------------
# CPS basic monthly
# ---------------------------------------------------------------------------
def fetch_cps_basic(
    year: int,
    month: int | str,
    variables: Sequence[str],
    *,
    weight: str = "PWCMPWGT",
    geography: str | None = None,
    where: Mapping | None = None,
    api_key: str | None = None,
    refresh: bool = False,
    timeout: int = 180,
) -> MicroFrame:
    """CPS basic monthly person records.

    ``weight`` defaults to ``PWCMPWGT``, the composited final weight BLS
    uses for labour-force estimates; ``PWSSWGT`` is the second-stage
    weight. Standard errors are ``NaN``: the public CPS carries no
    replicate weights or design variables, and a weights-only standard
    error would understate the truth. Use the BLS generalised variance
    function parameters if you need an approximation.
    """
    if isinstance(month, str):
        mon = month.lower()[:3]
    elif 1 <= int(month) <= 12:
        mon = _MONTHS[int(month) - 1]
    else:
        raise ValueError(f"month {month!r} not in 1..12")
    if mon not in _MONTHS:
        raise ValueError(f"month {month!r} not understood")
    url = f"{API_ROOT}/{year}/cps/basic/{mon}"
    wanted = list(variables) + [weight]
    df = fetch_records(url, wanted, CPS_IDS, geography=geography, where=where,
                       api_key=api_key, refresh=refresh, timeout=timeout)
    df = _to_numeric(df, keep_text=["HRHHID", "HRHHID2"])
    design = SurveyDesign.weights_only(
        weight, note="CPS public files publish no replicate weights or "
                     "design variables; standard errors not computed")
    return MicroFrame(data=df, design=design, unit="person",
                      source="census.cps_basic", vintage=f"{year}-{mon}",
                      query=_describe(url, wanted, geography, None, where))


def _describe(url, wanted, geography, within, where) -> str:
    parts = [f"get={len(wanted)} variables"]
    if geography:
        parts.append(f"for={geography}")
    if within:
        parts.append(f"in={within}")
    for k, v in _predicates(where).items():
        parts.append(f"{k}={v}")
    return url + " ? " + " & ".join(parts)


__all__ = ["fetch_acs_pums", "fetch_cps_basic", "fetch_records", "variables",
           "MAX_VARIABLES", "PUMS_IDS", "CPS_IDS"]
