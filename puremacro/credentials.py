"""Centralised API-key resolution for puremacro fetchers.

Resolves keys in priority order:
  1. Explicit `explicit=` kwarg passed by the caller.
  2. Environment variables in the service's registry, tried in order.
  3. TOML config file (default: ``~/.puremacro/credentials.toml``).
  4. None.

Lookup is side-effect-free. Use ``get()`` when missing == valid;
use ``require()`` when missing == error (raises
``MissingCredentialError`` with a researcher-actionable message).
``get_credential`` / ``require_credential`` are aliases of the two.

Two-part credentials
--------------------
Some services (Banco Central de Chile's SIETE API) authenticate with a
user *and* a password rather than one token. For those,
``get()``/``require()`` resolve the **user** (``BCCH_API_USER`` or
``[bcch].user`` — ``[bcch].api_key`` is still read for compatibility)
and :func:`get_password` resolves the **password** (``BCCH_API_PASS``
or ``[bcch].password``). ``require()`` insists on both; a password
alone never satisfies a lookup. See :data:`PASSWORD_ENV_VARS`.

Use ``status()`` from a notebook to see which services are configured
without leaking the actual key values.
"""
from __future__ import annotations

import os
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ServiceCredentialSpec:
    """Per-service registry entry."""
    name: str
    env_vars: tuple[str, ...]
    signup_url: str
    description: str


SERVICES: dict[str, ServiceCredentialSpec] = {
    "fred": ServiceCredentialSpec(
        name="fred",
        env_vars=("FRED_API_KEY", "PUREMACRO_FRED_API_KEY"),
        signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
        description="FRED + ALFRED real-time macro data (St. Louis Fed)",
    ),
    "bea": ServiceCredentialSpec(
        name="bea",
        env_vars=("BEA_API_KEY", "PUREMACRO_BEA_API_KEY"),
        signup_url="https://apps.bea.gov/API/signup/",
        description="BEA NIPA / regional / industry tables",
    ),
    "anthropic": ServiceCredentialSpec(
        name="anthropic",
        env_vars=("ANTHROPIC_API_KEY", "PUREMACRO_ANTHROPIC_API_KEY"),
        signup_url="https://console.anthropic.com/settings/keys",
        description="LLM-scored narrative kernels (narrative.scoring.llm)",
    ),
    "openai": ServiceCredentialSpec(
        name="openai",
        env_vars=("OPENAI_API_KEY", "PUREMACRO_OPENAI_API_KEY"),
        signup_url="https://platform.openai.com/api-keys",
        description="OpenAI provider for the LLM kernel (alternative to Anthropic)",
    ),
    "census": ServiceCredentialSpec(
        name="census",
        env_vars=("CENSUS_API_KEY", "PUREMACRO_CENSUS_API_KEY"),
        signup_url="https://api.census.gov/data/key_signup.html",
        description="Census BFS / ACS connectors",
    ),
    "banxico": ServiceCredentialSpec(
        name="banxico",
        env_vars=("BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY"),
        signup_url="https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html",
        description="Banco de México SIE API",
    ),
    "inegi": ServiceCredentialSpec(
        name="inegi",
        env_vars=("INEGI_API_KEY", "PUREMACRO_INEGI_API_KEY"),
        signup_url="https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/",
        description="INEGI Banco de Indicadores / BIE API",
    ),
    "bcch": ServiceCredentialSpec(
        name="bcch",
        env_vars=("BCCH_API_USER", "PUREMACRO_BCCH_API_USER"),
        signup_url="https://si3.bcentral.cl/estadisticas/principal1/registro/index.html",
        description="Banco Central de Chile Base de Datos Estadísticos (SIETE API; user + password)",
    ),
}

#: Services whose credential is a (user, password) pair. ``env_vars``
#: in :data:`SERVICES` name the *user*; these name the *password*. The
#: TOML file carries them as ``[service].user`` and ``[service].password``.
PASSWORD_ENV_VARS: dict[str, tuple[str, ...]] = {
    "bcch": ("BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS"),
}


def default_config_path() -> Path:
    """`$PUREMACRO_CREDENTIALS_FILE` if set; else
       `$XDG_CONFIG_HOME/puremacro/credentials.toml` if XDG_CONFIG_HOME set;
       else `~/.puremacro/credentials.toml`."""
    env = os.environ.get("PUREMACRO_CREDENTIALS_FILE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "puremacro" / "credentials.toml"
    return Path.home() / ".puremacro" / "credentials.toml"


_CONFIG_CACHE: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Read the TOML config file once per process; cache the parsed dict.
       Returns {} on missing file. Warns + returns {} on malformed TOML."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = default_config_path()
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        with open(path, "rb") as f:
            _CONFIG_CACHE = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        warnings.warn(
            f"puremacro.credentials: failed to parse {path}: {e}. "
            f"Falling back to env-vars only.",
            UserWarning,
            stacklevel=2,
        )
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def _config_section(service: str) -> dict[str, Any]:
    section = _load_config().get(service)
    return section if isinstance(section, dict) else {}


def _config_keys(service: str) -> tuple[str, ...]:
    """TOML keys that carry the token (or the user, for a pair service)."""
    return ("user", "api_key") if service in PASSWORD_ENV_VARS else ("api_key",)


class MissingCredentialError(RuntimeError):
    """Raised by `require()` when a fetcher needs an API key and none is found.

    Message structure (assertable in tests):
        "<description> needs an API key. Checked env vars (in order):
         <var1>, <var2>. Checked config file: <path> (<found|not found>).
         Get a free key at: <signup_url>"

    For a two-part service the first sentence reads "needs a user and a
    password" and the env-var list names both parts:
        "Checked env vars (in order): <user vars> (user); <password
         vars> (password)."
    """


def get(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve an API key for `service` (None if not found).

    For a two-part service (see :data:`PASSWORD_ENV_VARS`) this is the
    *user*; the password comes from :func:`get_password`.
    """
    if service not in SERVICES:
        raise KeyError(
            f"Unknown service {service!r}. Known: {sorted(SERVICES.keys())}"
        )
    if explicit:
        return explicit
    spec = SERVICES[service]
    for var in spec.env_vars:
        v = os.environ.get(var)
        if v:
            return v
    section = _config_section(service)
    for key in _config_keys(service):
        v = section.get(key)
        if v:
            return str(v)
    return None


def get_password(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve the password half of a two-part credential (None if not found).

    Same precedence as :func:`get`: explicit argument, then the env vars
    in :data:`PASSWORD_ENV_VARS`, then ``[service].password`` in the
    TOML file. Services without a password part always resolve to None.
    """
    if service not in SERVICES:
        raise KeyError(
            f"Unknown service {service!r}. Known: {sorted(SERVICES.keys())}"
        )
    if explicit:
        return explicit
    for var in PASSWORD_ENV_VARS.get(service, ()):
        v = os.environ.get(var)
        if v:
            return v
    v = _config_section(service).get("password")
    return str(v) if v else None


def require(service: str, *, explicit: str | None = None) -> str:
    """Like `get(service)` but raises `MissingCredentialError` on miss.

    For a two-part service both the user and the password must resolve;
    the user is returned.
    """
    key = get(service, explicit=explicit)
    needs_password = service in PASSWORD_ENV_VARS
    if key and (not needs_password or get_password(service)):
        return key
    spec = SERVICES[service]
    cfg_path = default_config_path()
    if needs_password:
        cfg_status = (
            "found but no [{0}].user / [{0}].password".format(service)
            if cfg_path.exists() else "not found"
        )
        raise MissingCredentialError(
            f"{spec.description} needs a user and a password. "
            f"Checked env vars (in order): {', '.join(spec.env_vars)} (user); "
            f"{', '.join(PASSWORD_ENV_VARS[service])} (password). "
            f"Checked config file: {cfg_path} ({cfg_status}). "
            f"Register at: {spec.signup_url}"
        )
    cfg_status = "found but no [{}].api_key".format(service) if cfg_path.exists() else "not found"
    raise MissingCredentialError(
        f"{spec.description} needs an API key. "
        f"Checked env vars (in order): {', '.join(spec.env_vars)}. "
        f"Checked config file: {cfg_path} ({cfg_status}). "
        f"Get a free key at: {spec.signup_url}"
    )


def status() -> pd.DataFrame:
    """Return one row per service: ['service', 'configured', 'source',
       'description', 'signup_url']. Never includes the actual key value.

       A two-part service counts as configured only when both the user
       and the password resolve; ``source`` says which part is missing.
    """
    rows = []
    for name, spec in SERVICES.items():
        source = "missing"
        configured = False
        for var in spec.env_vars:
            if os.environ.get(var):
                source = f"env:{var}"
                configured = True
                break
        if not configured:
            section = _config_section(name)
            if any(section.get(k) for k in _config_keys(name)):
                source = "config_file"
                configured = True
        if configured and name in PASSWORD_ENV_VARS and not get_password(name):
            configured = False
            source = (
                f"{source} (no password: set "
                f"{' or '.join(PASSWORD_ENV_VARS[name])} or [{name}].password)"
            )
        rows.append({
            "service": name,
            "configured": configured,
            "source": source,
            "description": spec.description,
            "signup_url": spec.signup_url,
        })
    return pd.DataFrame(rows)


get_credential = get
require_credential = require


__all__ = [
    "ServiceCredentialSpec",
    "SERVICES",
    "PASSWORD_ENV_VARS",
    "MissingCredentialError",
    "default_config_path",
    "get",
    "get_credential",
    "get_password",
    "require",
    "require_credential",
    "status",
]
