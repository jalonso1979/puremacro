"""Centralised API-key resolution for puremacro fetchers.

Resolves keys in priority order:
  1. Explicit `explicit=` kwarg passed by the caller.
  2. Environment variables in the service's registry, tried in order.
  3. TOML config file (default: ``~/.puremacro/credentials.toml``).
  4. None.

Lookup is side-effect-free. Use ``get()`` when missing == valid;
use ``require()`` when missing == error (raises
``MissingCredentialError`` with a researcher-actionable message).

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


class MissingCredentialError(RuntimeError):
    """Raised by `require()` when a fetcher needs an API key and none is found.

    Message structure (assertable in tests):
        "<description> needs an API key. Checked env vars (in order):
         <var1>, <var2>. Checked config file: <path> (<found|not found>).
         Get a free key at: <signup_url>"
    """


def get(service: str, *, explicit: str | None = None) -> str | None:
    """Resolve an API key for `service` (None if not found)."""
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
    cfg = _load_config()
    return cfg.get(service, {}).get("api_key") or None


def require(service: str, *, explicit: str | None = None) -> str:
    """Like `get(service)` but raises `MissingCredentialError` on miss."""
    key = get(service, explicit=explicit)
    if key:
        return key
    spec = SERVICES[service]
    cfg_path = default_config_path()
    cfg_status = "found but no [{}].api_key".format(service) if cfg_path.exists() else "not found"
    raise MissingCredentialError(
        f"{spec.description} needs an API key. "
        f"Checked env vars (in order): {', '.join(spec.env_vars)}. "
        f"Checked config file: {cfg_path} ({cfg_status}). "
        f"Get a free key at: {spec.signup_url}"
    )


def status() -> pd.DataFrame:
    """Return one row per service: ['service', 'configured', 'source',
       'description', 'signup_url']. Never includes the actual key value."""
    cfg = _load_config()
    rows = []
    for name, spec in SERVICES.items():
        source = "missing"
        configured = False
        for var in spec.env_vars:
            if os.environ.get(var):
                source = f"env:{var}"
                configured = True
                break
        if not configured and cfg.get(name, {}).get("api_key"):
            source = "config_file"
            configured = True
        rows.append({
            "service": name,
            "configured": configured,
            "source": source,
            "description": spec.description,
            "signup_url": spec.signup_url,
        })
    return pd.DataFrame(rows)


__all__ = [
    "ServiceCredentialSpec",
    "SERVICES",
    "MissingCredentialError",
    "default_config_path",
    "get",
    "require",
    "status",
]
