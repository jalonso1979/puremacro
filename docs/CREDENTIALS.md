> 🇬🇧 English · 🇪🇸 [Español](es/CREDENTIALS.md)

# Credentials

> Available from puremacro **0.66.0** onwards.

`puremacro.credentials` is the single place every API-keyed fetcher in
puremacro reads its key. It resolves keys in priority order:

1. **Explicit kwarg** — `credentials.get("fred", explicit=...)` or any
   fetcher's `api_key=` parameter wins everything.
2. **Environment variables** — each service has a primary alias
   (`FRED_API_KEY`) and a `PUREMACRO_`-prefixed secondary
   (`PUREMACRO_FRED_API_KEY`); first hit wins.
3. **TOML config file** — `~/.puremacro/credentials.toml` (overridable
   via `$PUREMACRO_CREDENTIALS_FILE` or `$XDG_CONFIG_HOME`).
4. **None** — `get()` returns `None`; `require()` raises
   `MissingCredentialError` with a researcher-actionable message.

## Quickstart

```python
import puremacro.credentials as creds

# See what's configured (never leaks the actual key values):
creds.status()
#       service  configured           source                                         description                                 signup_url
# 0        fred        True  env:FRED_API_KEY            FRED + ALFRED real-time macro data (...)  https://fred.stlouisfed.org/...
# 1         bea       False          missing               BEA NIPA / regional / industry tables  https://apps.bea.gov/API/signup/
# ...

# Resolve a key (None if not found):
key = creds.get("anthropic")

# Or require it (raises with a helpful message):
key = creds.require("anthropic")
# MissingCredentialError: LLM-scored narrative kernels (...) needs an
# API key. Checked env vars (in order): ANTHROPIC_API_KEY,
# PUREMACRO_ANTHROPIC_API_KEY. Checked config file:
# /Users/you/.puremacro/credentials.toml (not found). Get a free key
# at: https://console.anthropic.com/settings/keys
```

## Config file format

`~/.puremacro/credentials.toml` (optional; create if you prefer not to set env vars):

```toml
[fred]
api_key = "abc123..."

[bea]
api_key = "..."

[anthropic]
api_key = "sk-ant-..."

[openai]
api_key = "sk-..."

[census]
api_key = "..."
```

Missing sections fall back to env vars. The file is read once per
process and cached. Malformed TOML emits a `UserWarning` and falls
through to env-vars-only — never blocks credential resolution.

## Known services

| Service     | Used by                                        | Sign up                                                  |
|-------------|------------------------------------------------|----------------------------------------------------------|
| `fred`      | `fetch.fred`, `fetch.fred_states`, FRB Phil    | https://fred.stlouisfed.org/docs/api/api_key.html        |
| `bea`       | `fetch.bea_cainc`, `fetch.bea_industry_shares` | https://apps.bea.gov/API/signup/                         |
| `anthropic` | `narrative.scoring.llm` (Anthropic provider)   | https://console.anthropic.com/settings/keys              |
| `openai`    | `narrative.scoring.llm` (OpenAI provider)      | https://platform.openai.com/api-keys                     |
| `census`    | (forward-declared; no current consumer)        | https://api.census.gov/data/key_signup.html              |

> ⚠️ The `census` service is forward-declared (registered in
> `SERVICES` for future direct Census API connectors, e.g., ACS or
> per-state BFS series not mirrored on FRED). Current `fetch.census_bfs`
> pulls Census BFS data via FRED — see the `fred` row above.

## For implementers (adding a new fetcher)

```python
from puremacro import credentials

def fetch_my_thing(*, api_key: str | None = None) -> pd.DataFrame:
    key = credentials.require("my_service", explicit=api_key)
    # ... use key in your HTTP calls ...
```

The AST lint test
`tests/test_credentials/test_no_direct_env_get_in_fetch.py`
fails the build if you read `os.environ.get("*_API_KEY")` directly
in `puremacro/{fetch,narrative/scoring,narrative/indices,instruments}/`.

To add a new known service, append a `ServiceCredentialSpec` entry to
`puremacro/credentials.py::SERVICES`. The service registry test
verifies every entry has the required fields and an HTTPS signup URL.
