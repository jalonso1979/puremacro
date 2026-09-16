"""Schema drift telemetry and canaries for real-time data providers.

Detects when an upstream REST service changes its JSON key names, date
formats, or response envelope without standard HTTP error codes, and
triggers graceful fallback to persistent SQLite cache or portable
``.pmz`` cartridges.

Policy
------
Every LatAm parser and fetch path takes ``on_drift``, one of
:data:`DRIFT_POLICIES`:

- ``"raise"`` (default): raise :class:`SchemaDriftError`. The fetch
  layer then falls back to the locally stored snapshots when it has
  any, and re-raises otherwise.
- ``"warn"``: emit :class:`SchemaDriftWarning` and let the parser try
  anyway — its own date fallbacks (an ISO date where ``DD/MM/YYYY`` was
  expected, say) often still recover the payload.
- ``"ignore"``: parse silently.

Whatever the policy, a ``schema_drift`` event is recorded in the
``connector_events`` table (subject to the
``PUREMACRO_NARRATIVE_TELEMETRY=0`` kill-switch) so ``connector_health``
sees the drift even when a fallback hid it from the caller.
"""
from __future__ import annotations

import json
import re
import warnings
from typing import Any, Callable

import pandas as pd


class SchemaDriftError(ValueError):
    """Raised when an upstream payload violates the provider's expected schema."""


class SchemaDriftWarning(UserWarning):
    """Emitted when schema drift is detected and fallback is engaged."""


#: Accepted values of ``on_drift``.
DRIFT_POLICIES = ("raise", "warn", "ignore")


def _coerce_json(payload: Any) -> Any:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8-sig", errors="ignore")
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _brief(payload: Any, limit: int = 200) -> str:
    """A short, JSON-ish rendering of an unexpected payload for messages.

    Upstream error bodies ("this series is limited to 10 years of daily
    data", "token expired") arrive as JSON objects where a list was
    expected; showing the text is what lets the user act on it.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:                                 # pragma: no cover
        text = repr(payload)
    return text if len(text) <= limit else text[:limit] + "..."


def _sample_observations(items: list[Any], max_full: int = 500, sample_size: int = 20) -> list[tuple[int, Any]]:
    """Return list of (original_index, item) pairs to validate.

    If length <= max_full, inspect all items.
    Otherwise inspect the first sample_size and last sample_size items.
    """
    n = len(items)
    if n <= max_full:
        return list(enumerate(items))
    head = list(enumerate(items[:sample_size]))
    tail_start = max(sample_size, n - sample_size)
    tail = [(tail_start + i, item) for i, item in enumerate(items[tail_start:])]
    return head + tail


def _bcch_observations(data: dict) -> tuple[Any, str | None]:
    """Locate the observation list in a SIETE response.

    SIETE emits ``Series.Obs`` (capital O). The lowercase ``obs`` spelling,
    under ``Series`` or at the root, is still accepted for payloads and
    fixtures written against it.

    Returns ``(observations, key)``; ``key`` is None when nothing was found.
    """
    series = data.get("Series")
    containers = [series] if isinstance(series, dict) else []
    containers.append(data)
    for container in containers:
        for key in ("Obs", "obs"):
            if key in container:
                return container[key], key
    return None, None


def _record_drift(provider: str, fallback_used: str) -> None:
    try:
        from ... import _cache_db
        _cache_db.record_connector_event(provider, "schema_drift", fallback_used)
    except Exception:                                 # pragma: no cover
        pass


def handle_drift(
    provider: str,
    reason: str,
    on_drift: str = "raise",
    *,
    stacklevel: int = 2,
) -> None:
    """Apply an ``on_drift`` policy to one detected drift.

    Records a ``schema_drift`` telemetry event in every mode, then
    raises, warns, or returns according to the policy.
    """
    policy = str(on_drift).lower()
    if policy not in DRIFT_POLICIES:
        raise ValueError(
            f"on_drift={on_drift!r} is not one of {DRIFT_POLICIES}"
        )
    message = f"Schema drift detected for {provider}: {reason}"
    if policy == "raise":
        _record_drift(provider, "none")
        raise SchemaDriftError(message)
    if policy == "warn":
        _record_drift(provider, "warning_emitted")
        warnings.warn(message, SchemaDriftWarning, stacklevel=stacklevel)
        return
    _record_drift(provider, "ignored")


class SchemaCanary:
    """Validator for provider response payloads."""

    @staticmethod
    def validate_banxico(payload: Any) -> tuple[bool, str]:
        """Validate Banco de México SIE API response.

        Expected schema:
        {
          "bmx": {
            "series": [
              {
                "idSerie": "...",
                "titulo": "...",
                "datos": [
                  {"fecha": "DD/MM/YYYY", "dato": "1,234.56"}
                ]
              }
            ]
          }
        }
        ``dato`` is ``"N/E"`` where the observation is missing.
        """
        try:
            data = _coerce_json(payload)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"

        if not isinstance(data, dict):
            return False, f"expected dict root, got {type(data).__name__}"
        if "bmx" not in data or not isinstance(data["bmx"], dict):
            return False, "missing 'bmx' envelope object"
        series_list = data["bmx"].get("series")
        if not isinstance(series_list, list) or len(series_list) == 0:
            return False, "missing or empty 'bmx.series' array"
        first = series_list[0]
        if not isinstance(first, dict):
            return False, "bmx.series element is not an object"
        datos = first.get("datos")
        if datos is None:
            return False, "missing 'datos' array in bmx.series[0]"
        if not isinstance(datos, list):
            return False, f"'datos' is not a list ({type(datos).__name__})"
        if len(datos) > 0:
            sample = _sample_observations(datos)
            for i, obs in sample:
                if not isinstance(obs, dict):
                    return False, f"datos[{i}] is not a dict"
                if "fecha" not in obs or "dato" not in obs:
                    return False, f"datos[{i}] missing 'fecha' or 'dato' key"
                # Check date structure: strict DD/MM/YYYY or YYYY-MM-DD
                fecha = str(obs["fecha"]).strip()
                if "/" in fecha:
                    parts = fecha.split("/")
                    if len(parts) != 3 or len(parts[0]) != 2 or len(parts[1]) != 2 or len(parts[2]) != 4:
                        return False, f"datos[{i}] date format {fecha!r} does not match DD/MM/YYYY"
                    try:
                        pd.to_datetime(fecha, format="%d/%m/%Y")
                    except Exception as exc:
                        return False, f"datos[{i}] date {fecha!r} invalid DD/MM/YYYY: {exc}"
                elif "-" in fecha:
                    parts = fecha.split("-")
                    if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
                        return False, f"datos[{i}] date format {fecha!r} does not match YYYY-MM-DD"
                    try:
                        pd.to_datetime(fecha, format="%Y-%m-%d")
                    except Exception as exc:
                        return False, f"datos[{i}] date {fecha!r} invalid YYYY-MM-DD: {exc}"
                else:
                    return False, f"datos[{i}] date format {fecha!r} does not match DD/MM/YYYY or YYYY-MM-DD"
        return True, ""

    @staticmethod
    def validate_inegi(payload: Any) -> tuple[bool, str]:
        """Validate INEGI BIE / Indicadores API response.

        Expected schema:
        {
          "Series": [
            {
              "INDICADOR": "...",
              "FREQ": "...",
              "OBSERVATIONS": [
                {"TIME_PERIOD": "YYYY/NN", "OBS_VALUE": "..."}
              ]
            }
          ]
        }
        ``TIME_PERIOD`` is ``"YYYY"``, ``"YYYY/MM"`` (monthly) or
        ``"YYYY/NN"`` with NN in 1..4 (quarterly).
        """
        try:
            data = _coerce_json(payload)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"

        if not isinstance(data, dict):
            return False, f"expected dict root, got {type(data).__name__}"
        series_list = data.get("Series")
        if not isinstance(series_list, list) or len(series_list) == 0:
            return False, "missing or empty 'Series' array"
        first = series_list[0]
        if not isinstance(first, dict):
            return False, "'Series' element is not an object"
        obs_list = first.get("OBSERVATIONS")
        if obs_list is None or not isinstance(obs_list, list):
            return False, "missing or invalid 'OBSERVATIONS' array in Series[0]"
        if len(obs_list) > 0:
            sample = _sample_observations(obs_list)
            for i, obs in sample:
                if not isinstance(obs, dict):
                    return False, f"OBSERVATIONS[{i}] is not an object"
                if "TIME_PERIOD" not in obs or "OBS_VALUE" not in obs:
                    return False, f"OBSERVATIONS[{i}] missing 'TIME_PERIOD' or 'OBS_VALUE'"
                tp = str(obs["TIME_PERIOD"]).strip()
                if not tp:
                    return False, f"OBSERVATIONS[{i}] empty 'TIME_PERIOD'"
                valid_tp = False
                if "/" in tp:
                    parts = tp.split("/")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        year, num = int(parts[0]), int(parts[1])
                        if 1900 <= year <= 2100 and 1 <= num <= 12:
                            valid_tp = True
                elif "Q" in tp.upper():
                    try:
                        pd.Period(tp.upper(), freq="Q")
                        valid_tp = True
                    except Exception:
                        valid_tp = False
                else:
                    try:
                        pd.to_datetime(tp)
                        valid_tp = True
                    except Exception:
                        valid_tp = False
                if not valid_tp:
                    return False, f"OBSERVATIONS[{i}] invalid TIME_PERIOD date format {tp!r}"
        return True, ""

    @staticmethod
    def validate_bcb(payload: Any) -> tuple[bool, str]:
        """Validate Banco Central do Brasil SGS API response.

        Expected schema (``formato=json``):
        [
          {"data": "DD/MM/YYYY", "valor": "1.23"}
        ]
        An object where the list should be is an upstream error body
        (for instance the 10-year window limit on daily series); its
        text is surfaced in the reason.
        """
        try:
            data = _coerce_json(payload)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"

        if not isinstance(data, list):
            if isinstance(data, dict):
                return False, f"expected list root, got dict: {_brief(data)}"
            return False, f"expected list root, got {type(data).__name__}"
        if len(data) > 0:
            sample = _sample_observations(data)
            for i, obs in sample:
                if not isinstance(obs, dict):
                    return False, f"element[{i}] is not an object"
                if "data" not in obs or "valor" not in obs:
                    return False, f"element[{i}] missing 'data' or 'valor'"
                d_str = str(obs["data"]).strip()
                if not re.match(r"^\d{2}/\d{2}/\d{4}$", d_str):
                    return False, f"element[{i}] date {d_str!r} does not match DD/MM/YYYY"
                try:
                    pd.to_datetime(d_str, format="%d/%m/%Y")
                except Exception as exc:
                    return False, f"element[{i}] date {d_str!r} invalid DD/MM/YYYY: {exc}"
        return True, ""

    @staticmethod
    def validate_bcch(payload: Any) -> tuple[bool, str]:
        """Validate Banco Central de Chile SIETE API response.

        Expected schema (``function=GetSeries``):
        {
          "Codigo": 0,
          "Descripcion": "",
          "Series": {
            "descripEsp": "...",
            "seriesId": "...",
            "Obs": [
              {"indexDateString": "DD-MM-YYYY", "value": "123.4", "statusCode": "OK"}
            ]
          },
          "SeriesInfos": []
        }
        ``value`` is ``"NaN"`` where the observation is missing. The
        lowercase ``obs`` spelling (under ``Series`` or at the root) is
        also accepted.
        """
        try:
            data = _coerce_json(payload)
        except Exception as exc:
            return False, f"invalid JSON: {exc}"

        if not isinstance(data, dict):
            return False, f"expected dict root, got {type(data).__name__}"
        # Check error code
        code = data.get("Codigo")
        if code is not None and code != 0:
            desc = data.get("Descripcion", "Unknown error")
            return False, f"BCCh API returned error code {code}: {desc}"
        obs_list, key = _bcch_observations(data)
        if key is None:
            return False, (
                "missing 'Series' or 'obs' object in BCCh response "
                "(expected Series.Obs)"
            )

        if not isinstance(obs_list, list):
            return False, f"'{key}' is not a list ({type(obs_list).__name__})"
        if len(obs_list) > 0:
            sample = _sample_observations(obs_list)
            for i, obs in sample:
                if not isinstance(obs, dict):
                    return False, f"{key}[{i}] is not an object"
                if "indexDateString" not in obs or "value" not in obs:
                    return False, f"{key}[{i}] missing 'indexDateString' or 'value'"
                d_str = str(obs["indexDateString"]).strip()
                if not d_str:
                    return False, f"{key}[{i}] empty 'indexDateString'"
                valid_date = False
                try:
                    if "-" in d_str and len(d_str.split("-")[0]) == 2:
                        pd.to_datetime(d_str, format="%d-%m-%Y")
                        valid_date = True
                    else:
                        pd.to_datetime(d_str)
                        valid_date = True
                except Exception:
                    valid_date = False
                if not valid_date:
                    return False, f"{key}[{i}] invalid indexDateString date format {d_str!r}"
        return True, ""

    @classmethod
    def check(
        cls,
        provider: str,
        payload: Any,
        *,
        raise_on_drift: bool = False,
        on_drift: str | None = None,
    ) -> tuple[bool, str]:
        """Validate ``payload`` against ``provider``'s schema and apply a policy.

        ``on_drift`` (``"raise"`` / ``"warn"`` / ``"ignore"``) takes
        precedence; when it is None the older ``raise_on_drift`` flag
        selects ``"raise"`` (True) or ``"warn"`` (False). Providers
        without a validator pass. Returns ``(ok, reason)`` whenever it
        returns at all; a ``schema_drift`` telemetry event is recorded
        for every drift, whatever the policy.
        """
        prov = str(provider).lower()
        policy = on_drift if on_drift is not None else (
            "raise" if raise_on_drift else "warn")
        # Checked before the payload, not only once drift occurs: a typo
        # in the policy must fail on the first call, not months later
        # inside a fallback path where it reads as a fetch failure.
        if str(policy).lower() not in DRIFT_POLICIES:
            raise ValueError(
                f"on_drift={policy!r} is not one of {DRIFT_POLICIES}"
            )
        validators: dict[str, Callable[[Any], tuple[bool, str]]] = {
            "banxico": cls.validate_banxico,
            "inegi": cls.validate_inegi,
            "bcb": cls.validate_bcb,
            "bcch": cls.validate_bcch,
        }
        validator = validators.get(prov)
        if validator is None:
            return True, ""
        ok, reason = validator(payload)
        if not ok:
            handle_drift(prov, reason, policy, stacklevel=3)
        return ok, reason


def validate_payload(
    provider: str,
    payload: Any,
    *,
    raise_on_drift: bool = False,
    on_drift: str | None = None,
) -> tuple[bool, str]:
    """Convenience functional interface for :meth:`SchemaCanary.check`."""
    return SchemaCanary.check(
        provider, payload, raise_on_drift=raise_on_drift, on_drift=on_drift,
    )


__all__ = [
    "DRIFT_POLICIES",
    "SchemaDriftError",
    "SchemaDriftWarning",
    "SchemaCanary",
    "handle_drift",
    "validate_payload",
]
