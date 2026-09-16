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

One service authenticates with a **user and a password** rather than a
single token (Banco Central de Chile). For those, `get()` / `require()`
resolve the *user* and `get_password()` resolves the *password*; see
[Two-part credentials](#two-part-credentials-bcch).

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

[banxico]
api_key = "..."

[inegi]
api_key = "..."

[bcch]
user = "..."
password = "..."
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
| `banxico`   | `fetch.realtime.banxico`                       | https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html |
| `inegi`     | `fetch.realtime.inegi`                         | https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/ |
| `bcch`      | `fetch.realtime.bcch`                          | https://si3.bcentral.cl/estadisticas/principal1/registro/index.html |

> ⚠️ The `census` service is forward-declared (registered in
> `SERVICES` for future direct Census API connectors, e.g., ACS or
> per-state BFS series not mirrored on FRED). Current `fetch.census_bfs`
> pulls Census BFS data via FRED — see the `fred` row above.

### Real-time Latin American connectors

The three services added in 3.4.0 back the snapshot connectors described
in [`docs/real_time_data.md`](real_time_data.md). Banco Central do Brasil
(`bcb`) needs no credential at all and therefore has no entry.

| Service | Env vars (in resolution order) | TOML keys | Where the credential travels |
|---|---|---|---|
| `banxico` | `BANXICO_API_KEY`, `BMX_TOKEN`, `PUREMACRO_BANXICO_API_KEY` | `[banxico].api_key` | the `Bmx-Token` **request header** |
| `inegi` | `INEGI_API_KEY`, `PUREMACRO_INEGI_API_KEY` | `[inegi].api_key` | a segment of the **URL path** |
| `bcch` | user: `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` · password: `BCCH_API_PASS`, `PUREMACRO_BCCH_API_PASS` | `[bcch].user`, `[bcch].password` (`[bcch].api_key` still read as the user, for compatibility) | both in the **query string** (`user=`, `pass=`) |

Because the INEGI token sits in the path and the BCCh pair in the query
string, either can end up in a URL that an exception or a log line
carries. The BCCh connector scrubs the user and the password — raw and
URL-encoded — from its warnings and from the URL urllib hangs on its
errors; treat any URL you print yourself as a secret.

### Two-part credentials (`bcch`)

```python
import puremacro.credentials as creds

creds.get("bcch")           # the USER   (BCCH_API_USER / [bcch].user)
creds.get_password("bcch")  # the PASSWORD (BCCH_API_PASS / [bcch].password)
creds.require("bcch")       # raises unless BOTH resolve
```

`PASSWORD_ENV_VARS` names the services whose credential is a pair. A
password alone never satisfies a lookup: with only `BCCH_API_PASS` set,
`get("bcch")` returns `None` and `status()` reports `bcch` as not
configured. With only the user set, `status()` reports the half that is
missing — `env:BCCH_API_USER (no password: set BCCH_API_PASS or
PUREMACRO_BCCH_API_PASS or [bcch].password)` — and `require("bcch")`
raises naming both lists.

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
