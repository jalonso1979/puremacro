> 🇬🇧 English · 🇪🇸 [Español](es/real_time_latam.md)

# Latin America Real-Time Connectors, Schema Canaries & .pmz Cartridges

`puremacro.fetch.realtime` gains four connectors for Latin American central banks and statistical agencies — `banxico` (Banco de México, SIE), `inegi` (INEGI, Banco de Indicadores / BIE), `bcb` (Banco Central do Brasil, SGS) and `bcch` (Banco Central de Chile, SIETE) — together with three pieces of plumbing they depend on: a **schema canary** (`puremacro.fetch.realtime.canary`) that checks each provider's JSON envelope before it is parsed, a **snapshot store** (the `realtime_vintages` table of the SQLite cache database, `puremacro._cache_db`) that turns repeated captures into vintages, and a **portable `.pmz` cartridge** (`puremacro.fetch.realtime.cartridge`, built on `puremacro.pocket`) that moves a `VintagePanel` to a machine with no network and no credentials. Credentials are resolved through `puremacro.credentials` (services `banxico`, `inegi` and `bcch`; `bcb` is keyless). The connectors register with the provider registry, so the general entry point `vintage_panel(...)` of [real_time_data.md](real_time_data.md) reaches them through `providers=`.

One fact shapes everything on this page, so it comes first rather than in a footnote: **none of these four sources publishes an archive of past editions**. puremacro manufactures vintages by snapshotting.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 Vintages by snapshot, not by archive

ALFRED, the Bundesbank real-time database and the OECD STES revisions archive keep every edition a statistical office ever published. Banxico's SIE, INEGI's BIE, the BCB's SGS and the BCCh's SIETE do not: each request returns the **current** series, overwritten in place, and the payload carries no vintage field. `puremacro.fetch.realtime.banxico.MEXICO_VINTAGE_NOTE` records that determination for Mexico, and the same holds for Brazil and Chile.

The four connectors therefore build vintages the only way they can. Every successful fetch is parsed into a tidy `[date, vintage, value]` frame whose `vintage` column is the **snapshot date**, the day *this machine* captured the series — `pd.Timestamp.now().normalize()`, i.e. local midnight of today, unless you pass `vintage_date=` — and the rows are written into the `realtime_vintages` table of the local SQLite cache, keyed by `(provider, country, series_id, observation_date, vintage_date)`. Writing $y_t^{(v)}$ for the value of reference period $t$ carried by the snapshot captured on day $v$, one fetch contributes the set

$$S_v = \{(t, v, y_t^{(v)}) : t \in \text{payload}_v\},$$

and the local archive after captures on days $v_1 < v_2 < \dots < v_K$ is $\bigcup_k S_{v_k}$. The consequences are immediate and the rest of the page keeps coming back to them:

- A single fetch yields exactly **one** edition. `coverage()` reports `n_vintages == 1` and no revision test is possible.
- Revision history accumulates **only across captures on different days**. A re-fetch on the same day hits the same primary key and overwrites it (`INSERT OR REPLACE`).
- The vintage date is when *you* captured, not when the institution released. `VINTAGE_SEMANTICS["banxico"]` and its three siblings say exactly that, and `VintagePanel.vintage_semantics()` returns the text for every provider present.
- For the historical editions of Mexican quarterly GDP use provider `oecd_stes`, which archives INEGI's series with 329 monthly editions from 1999-02 (see [real_time_data.md](real_time_data.md)).

### 1.2 From snapshots to revisions

Given the archive, the objects the rest of `puremacro.fetch.realtime` works with are the standard real-time data-set constructs of Croushore and Stark (2001). The **revision triangle** places reference dates on rows and capture dates on columns, $T[t, v] = y_t^{(v)}$, with `NaN` wherever period $t$ was not yet in the payload on day $v$. The **as-of cross-section** on day $v^*$ keeps, for every $t$, the last capture not later than $v^*$:

$$y_t^{\,\text{as of } v^*} = y_t^{(v^\dagger)}, \qquad v^\dagger = \max\{v \le v^* : t \in \text{payload}_v\}.$$

For a series catalogued in **levels** or as an **index**, the transform applied within each vintage column before any test is the log-difference in percent, $x_t^{(v)} = 100\,[\ln y_t^{(v)} - \ln y_{t-1}^{(v)}]$; series catalogued as a **rate** or a **growth rate** are used as they stand (`UNITS_TRANSFORM`). Numerators and denominators are never mixed across vintages. The preliminary estimate of $t$ is its first capture, the final is its latest, and the revision is $r_t = x_t^{f} - x_t^{p}$ — provided the first capture can be observed at all. `revision_frame` drops every reference period that ended before the earliest vintage column ($t < v_1$), because the edition captured on $v_1$ already carries a revised number for it (`require_observable_first=True`, which `VintagePanel.revisions` and `news_or_noise` do not expose; call `puremacro.vintages.revision_frame(panel.long(...), require_observable_first=False)` for the naive reading). With snapshot vintages this means the revision sample starts at the first capture date, not at the start of the series: everything captured on day one is history that arrived already revised.

The Mankiw and Shapiro (1986) test pair is the two regressions

$$r_t = \alpha_p + \beta_p\, x_t^{p} + \varepsilon_{p,t}, \qquad r_t = \alpha_f + \beta_f\, x_t^{f} + \varepsilon_{f,t}.$$

Under **news** the preliminary estimate is an efficient forecast and the revision is orthogonal to it, $\beta_p = 0$; under **noise** the preliminary estimate is the truth plus measurement error, $x_t^p = x_t^f + u_t$, so $r_t = -u_t$ is orthogonal to the final, $\beta_f = 0$, and negatively related to the preliminary. The reported `noise_share` is $\max(0, -\hat\beta_p)$, the share of the preliminary estimate's variance attributable to $u_t$ under the pure noise model. Standard errors are heteroskedasticity-robust by default (`hac_lags=0`) and Newey-West HAC with `hac_lags="auto"` or an integer bandwidth. With snapshot vintages this machinery is only as informative as the calendar of captures behind it: $K$ captures on $K$ distinct days give at most $K$ columns.

### 1.3 The snapshot store

```sql
CREATE TABLE IF NOT EXISTS realtime_vintages (
    provider         TEXT NOT NULL,
    country          TEXT NOT NULL,
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD, the snapshot date
    value            REAL,
    PRIMARY KEY (provider, country, series_id, observation_date, vintage_date)
);
CREATE INDEX IF NOT EXISTS realtime_vintages_idx
    ON realtime_vintages(provider, country, series_id, vintage_date);
```

The table lives in the same file as the HTTP cache and the ALFRED store, `~/.cache/puremacro/cache.db` or `$PUREMACRO_HTTP_CACHE_DIR` ([CACHE_DB.md](CACHE_DB.md)), and `bootstrap_schema` creates it next to `connector_events` (`schema_version` component `realtime_vintages`, version 1). Three helpers in `puremacro._cache_db` wrap it: `store_realtime_vintages` (insert-or-replace, returns the number of rows written), `query_realtime_vintages` (optionally restricted to the editions with `vintage_date <= ...`, the raw material of the as-of cross-section; `VintagePanel.as_of` then keeps the latest edition per period) and `record_connector_event` (one row per telemetry event).

The four connectors differ only in how they build the request and parse the body; everything after that — canary policy, cache write, history read-back, fallback, telemetry — lives in one shared helper (`puremacro.fetch.realtime._snapshot.fetch_snapshot_vintages`), so every `fetch_*_vintages` call follows the same decision tree:

1. Resolve the credential (Banxico, INEGI, BCCh; BCCh needs both halves). If it does not resolve and `use_cache=True`, return the stored snapshots when there are any — no request, no warning, no event — and otherwise raise `MissingCredentialError`.
2. Request the live series with `urllib` (`User-Agent: puremacro (real-time vintage reader)`, `timeout=60.0` by default) and run the parser, which runs the canary under the `on_drift` policy of Section 1.4.
3. With `use_cache=True`, store the parsed rows and log the event `('success', 'none')`. Then, with `history=True` (the default), **return every snapshot stored locally for this series** — today's included, one vintage per snapshot date — which is what a revision test needs; `history=False` returns only the snapshot just fetched.
4. If the payload parsed to an empty frame and the store holds rows, warn `... received empty observations; falling back to cached vintages.` and log `('fallback', 'sqlite_cache')`.
5. If anything raised — an HTTP error, a timeout, a `SchemaDriftError` — and the store holds rows, warn `... failed (<exc>); falling back to cached vintages.` (`SchemaDriftWarning` for drift, `UserWarning` otherwise) and log `('fallback', 'sqlite_cache')`. With a cold store the exception propagates.

`use_cache=False` opts out of the store entirely — it neither reads nor writes `realtime_vintages` — so the live snapshot comes back exactly as parsed and any failure is raised rather than papered over. A snapshot that cannot be *stored* is still returned, and a telemetry write that fails never breaks the fetch that emitted it. For the two credentialed providers that carry secrets in the URL, the failure message and the `url` / `filename` attributes urllib hangs on its exceptions are scrubbed before they are shown.

The panel entry points `fetch_*_panel` catch whatever propagates and record it in `metadata["failed"]` under `"<ISO3>:<variable>"`; `vintage_panel` re-keys that as `"<ISO3>:<variable>:<provider>"` and never raises for a failed series.

### 1.4 Schema canaries

A REST service that renames a key or changes a date format keeps answering `200 OK`. `SchemaCanary` catches that class of drift before it becomes a silently empty panel. Each validator is a pure function returning `(ok, reason)`; the parsers call them and apply a policy, so drift is handled like any other fetch failure in the tree above. Payloads may be `bytes`, `str` or already-decoded JSON. Observation lists longer than 500 elements are sampled: the first 20 and the last 20 elements are inspected.

| Provider | Root | Envelope | Per observation | Date rule |
|---|---|---|---|---|
| `banxico` | dict | `bmx.series[0].datos` (a list, possibly empty) | keys `fecha`, `dato` | `DD/MM/YYYY` or `YYYY-MM-DD`, parsed strictly |
| `inegi` | dict | `Series[0].OBSERVATIONS` (a list) | keys `TIME_PERIOD`, `OBS_VALUE` | `YYYY/NN` with $1900 \le YYYY \le 2100$ and $1 \le NN \le 12$, a `YYYYQn`-style period, or anything `pd.to_datetime` accepts |
| `bcb` | list | the list itself | keys `data`, `valor` | exactly `DD/MM/YYYY` |
| `bcch` | dict | `Codigo` absent or `0`; the observation list under `Series.Obs` — the spelling SIETE emits — with the lowercase `obs`, under `Series` or at the root, still accepted | keys `indexDateString`, `value` | `DD-MM-YYYY`, or anything `pd.to_datetime` accepts |

**The `on_drift` policy.** Every LatAm parser and fetch path takes `on_drift`, one of `DRIFT_POLICIES = ("raise", "warn", "ignore")`, and `canary.handle_drift(provider, reason, on_drift)` applies it:

- `"raise"` (the default): raise `SchemaDriftError`, a `ValueError`. The fetch layer turns that into a cache fallback when the store holds anything and re-raises otherwise.
- `"warn"`: emit `SchemaDriftWarning` (a `UserWarning`) and let the parser try anyway — its own date fallbacks, an ISO date where `DD/MM/YYYY` was expected for instance, often still recover the payload.
- `"ignore"`: parse silently.

Whatever the policy, a `('schema_drift', ...)` event is recorded in `connector_events` — with `fallback_used` `'none'`, `'warning_emitted'` or `'ignored'` respectively — so `connector_health()` sees the drift even when a fallback hid it from the caller. A policy string outside `DRIFT_POLICIES` raises `ValueError` on the first call, before the payload is looked at, rather than months later inside a fallback path.

`SchemaCanary.check(provider, payload, *, raise_on_drift=False, on_drift=None)` and its functional twin `validate_payload` carry both spellings: `on_drift` wins, and when it is `None` the older `raise_on_drift` flag selects `"raise"` (`True`) or `"warn"` (`False`). Both return `(ok, reason)` whenever they return at all, and an unknown provider name passes through as `(True, "")`.

### 1.5 The `.pmz` cartridge container

`pack_realtime_cartridge` and `load_realtime_cartridge` are thin wrappers over `puremacro.pocket`, so a real-time cartridge *is* a pocket cartridge:

- a **plain zip** (`ZIP_STORED`) with two kinds of member: `manifest.json` and one `frames/<name>.npz` per frame. A real-time cartridge holds `data`, the tidy `VintagePanel.df` with the eight `VINTAGE_COLUMNS`, and — when you pack a `VintagePanel` that carries metadata — a second frame `vintage_metadata`, a two-column key / JSON-value table that `load_realtime_cartridge` turns back into `panel.metadata`, so `failed`, `missing`, `freq`, `provider_used` and friends survive the round trip. Values that are not JSON-native (a `DataFrame` of dropped editions, a `Timestamp`) are stored through `str` rather than dropped, so the cartridge still says what was there;
- each frame is an **npz archive** — one NumPy array per column plus a JSON schema recording dtypes, index type and column names (`puremacro.runtime.store`). No parquet, no pyarrow and **no pickle**: object columns holding anything but strings are refused at pack time, and `np.load(..., allow_pickle=False)` is used at load time;
- the manifest records `format = "puremacro-cartridge"`, `version = 1`, a `provenance` block (`created` UTC timestamp, `puremacro_version`, `source`, `vintage`, `notes`, `call`, `host`) and, per frame, `name`, `n_rows`, `n_cols`, `columns`, `index`, `n_bytes` and the **SHA-256** of the stored npz payload.

`load_realtime_cartridge(path, verify=True)` reads the zip through `pocket.load`, which always compares each payload's digest with the manifest (a mismatch raises `pocket.CartridgeError`); with `verify=True` it additionally re-encodes the loaded frame and compares again (`Cartridge.verify()`), and `verify=False` skips only that second check. It returns a `VintagePanel` whose `metadata` carries `cartridge_path`, `provenance_source`, `provenance_vintage` and `provenance_notes`. The digest depends on the data alone, not on the clock, which is what makes the re-encode check possible.

Is it safe on a file you did not produce? Against **code execution**, yes: the format contains no pickle, members are read from memory by name and never extracted to disk, and the manifest is parsed with `json`. Against **tampering**, no: the SHA-256 is an unsigned checksum, so anyone can repack a modified frame with a matching digest. The `pocket` module states it precisely — cartridges are a transport format, not a trust boundary — and a hostile archive can still cost memory to inflate. The relationship to `puremacro.pocket` is containment: `pocket.load(path)` opens a real-time cartridge as a generic `Cartridge`, `pocket.inspect_cartridge(path)` reads its manifest without decoding any data, and `pocket.to_base64` / `pocket.from_base64` move it by clipboard. `pack_realtime_cartridge` only fixes the defaults (`source="LatAm Regional Real-Time Ecosystem"`, `vintage=` today's UTC date, a standard note) and `load_realtime_cartridge` only adds the `VintagePanel` wrapping.

---

## 2. Methodological & Provider Options

| | `banxico` | `inegi` | `bcb` | `bcch` |
|---|---|---|---|---|
| Institution | Banco de México, SIE | INEGI, Banco de Indicadores / BIE | Banco Central do Brasil, SGS | Banco Central de Chile, SIETE |
| Country served | `MEX` | `MEX` | `BRA` | `CHL` |
| Credential | token | token | none | user + password |
| Where it travels | request header `Bmx-Token` | path segment of the request URL | — | query string (`user=`, `pass=`), URL-encoded |
| Endpoint constant | `BANXICO_SERIES_URL` | `INEGI_SERIES_URL` | `BCB_SGS_URL` | `BCCH_SIETE_URL` (built by `_build_bcch_url`) |
| Value coercion | drops `N/E`, `NaN`, `null`; strips thousands commas | strips thousands commas | comma is the decimal separator | comma is the decimal separator |
| Vintage semantics | snapshot date | snapshot date | snapshot date | snapshot date |

The catalogue entries `BANXICO_SERIES`, `INEGI_SERIES`, `BCB_SERIES` and `BCCH_SERIES` in `puremacro.fetch.realtime.catalog` are registered at import time, so `vintage_catalog()` lists them and `providers_for("MEX", "cpi")` returns `['banxico', 'inegi']`. The `units` column decides the default revision transform of Section 1.2 and the `freq` column is enforced by `vintage_panel` (below). Entries whose identifier could not be confirmed against the live service carry the marker `VERIFY ONLINE` in their `note`; `pytest -m network` is the audit that clears them.

| Provider | Variable | Series id | Units | Freq | Catalogue note |
|---|---|---|---|---|---|
| `banxico` | `policy_rate` | `SF61745` | rate | D | Tasa objetivo (tasa de interés interbancaria a 1 día), diaria |
| `banxico` | `cpi` | `SP1` | index | M | Índice Nacional de Precios al Consumidor (INPC general), mensual |
| `banxico` | `activity` | `SR17631` | index | M | IGAE índice general, mensual — `VERIFY ONLINE` |
| `inegi` | `gdp_real` | `735848` | level | Q | PIB trimestral, valores constantes 2018, desestacionalizado — `VERIFY ONLINE` |
| `inegi` | `cpi` | `628197` | index | M | INPC general, mensual — `VERIFY ONLINE` |
| `inegi` | `activity` | `736184` | index | M | IGAE índice general, mensual — `VERIFY ONLINE` |
| `bcb` | `gdp_real` | `22099` | index | Q | PIB trimestral - dados dessazonalizados - índice encadeado (média 1995 = 100) |
| `bcb` | `cpi` | `433` | growth_mom | M | IPCA - variação % mensal |
| `bcb` | `policy_rate` | `432` | rate | D | Taxa de juros - Meta Selic definida pelo Copom (% a.a.), diária |
| `bcb` | `activity` | `24363` | index | M | IBC-Br, dessazonalizado, mensal — `VERIFY ONLINE` |
| `bcch` | `gdp_real` | `F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T` | level | Q | PIB volumen a precios del año anterior encadenado, ref. 2018, trimestral — `VERIFY ONLINE` |
| `bcch` | `cpi` | `F074.IPC.IND.Z.Z.C.M` | index | M | IPC general, índice base 2023 = 100, mensual — `VERIFY ONLINE` |
| `bcch` | `policy_rate` | `F022.TPM.TPO.D001.NO.Z.D` | rate | D | Tasa de Política Monetaria (TPM), diaria |
| `bcch` | `activity` | `F032.IMC.IND.Z.Z.EP18.Z.Z.0.M` | index | M | IMACEC Índice Mensual de Actividad Económica, mensual |

Two of these entries are worth reading twice. BCB `433` is the IPCA **monthly percentage change**, not an index level, so its units are `growth_mom` and no revision helper will log-difference an already-differenced series. BCB `22099` is the IBGE quarterly seasonally adjusted chained-volume index (`22109` is the unadjusted one); `4380`, which looks like a GDP series, is "PIB mensal - valores correntes", monthly and nominal. On the Chilean side the SIETE codes end in the Spanish frequency letter (`D`, `M`, `T`, `A`) — none ends in `.Q` — and the IPC lives under chapter `F074`, not `F073`, which is exchange rates.

`policy_rate` and `activity` are the two canonical variables this page adds. `policy_rate` ("Central bank policy rate / target interest rate") carries the aliases `tpm`, `selic`, `tasa_objetivo`, `interest_rate`, `mp_rate`, `central_bank_rate`, `target_rate` and `overnight_rate`; `activity` ("Monthly economic activity index (IGAE / IMACEC / IBC-Br), SA") carries `economic_activity`, `igae`, `imacec` and `ibc_br`. So `variables=["selic"]` and `variables=["policy_rate"]` are the same request.

Through the registry, the live calls are `vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")`, `vintage_panel(["BRA"], providers="bcb", variables=["gdp_real"], freq="Q")` and `vintage_panel(["CHL"], providers="bcch", variables=["activity"], freq="M")`; Section 4.4 shows what the first one does offline. Things to know about that route:

- `providers="auto"` never reaches these connectors: `DEFAULT_PROVIDER_ORDER` is `oecd_stes, alfred, bundesbank, ons, statcan, ecb_rtd`. Name them explicitly, as a string or in a list.
- **`freq` is enforced, one frequency per call.** `SUPPORTED_FREQUENCIES` is `{"Q", "M", "D"}` and anything else raises `ValueError`. A catalogue entry that declares a different frequency is *not served*: it is reported in `metadata["missing_pairs"]` / `metadata["failed"]` with the reason, e.g. `catalogue series SF61745 is daily (freq='D'); requested freq='Q'`. Entries that declare no frequency — every quarterly vintage archive, which predates the field — are served at `"Q"` only. Ask for each series at its own frequency, and pass the matching `periods_per_year` (4, 12, 252…) to `triangle()` / `revisions()` / `news_or_noise()`; keep daily policy rates out of revision tests altogether.
- The default `series="gdp_real"` resolves to nothing for `banxico` (its catalogue has no GDP), so `vintage_panel(["MEX"], providers="banxico")` reports `(MEX, gdp_real)` as missing rather than failing.
- `catalog=` overrides are consulted for servability only; the four connectors look series up in their own tables. For an identifier that is not catalogued, call `fetch_*_vintages(series_id)` directly and route the frame through `normalize_vintage_frame`.
- `fetch_*_panel` expects canonical variable names (`vintage_panel` canonicalises before dispatch) and silently skips countries and variables outside its table. `store=` and `refresh=` are accepted and ignored.
- Combining providers, e.g. `providers=["banxico", "bcb", "bcch"]`, triggers the usual mixed-provider warning. It is harmless here, since all four share the same snapshot-date semantics, and `warn_on_mixed_providers=False` silences it.

---

## 3. Credentials & Configuration

Resolution follows `puremacro.credentials` ([CREDENTIALS.md](CREDENTIALS.md)): explicit kwarg, then the environment variables in registry order, then the TOML file (`$PUREMACRO_CREDENTIALS_FILE` if set, else `$XDG_CONFIG_HOME/puremacro/credentials.toml` if `XDG_CONFIG_HOME` is set, else `~/.puremacro/credentials.toml`), else nothing. `credentials.get_credential` and `credentials.require_credential` are aliases of `get` and `require`; `credentials.status()` shows which services are configured without printing any value.

| Service | Environment variables (in order) | TOML keys | Sign-up |
|---|---|---|---|
| `banxico` | `BANXICO_API_KEY`, `BMX_TOKEN`, `PUREMACRO_BANXICO_API_KEY` | `[banxico] api_key` | https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html |
| `inegi` | `INEGI_API_KEY`, `PUREMACRO_INEGI_API_KEY` | `[inegi] api_key` | https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/ |
| `bcch` (user) | `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` | `[bcch] user` (or `api_key`) | https://si3.bcentral.cl/estadisticas/principal1/registro/index.html |
| `bcch` (password) | `BCCH_API_PASS`, `PUREMACRO_BCCH_API_PASS` | `[bcch] password` | (same registration) |

BCCh is the odd one out: SIETE authenticates with the registered e-mail **and** a password, and the registry keeps the two halves apart. `PASSWORD_ENV_VARS` names the services whose credential is a pair — `{"bcch": ("BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS")}` — and `credentials.get("bcch")` resolves only the *user*, scanning `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` and then `[bcch].user` / `[bcch].api_key`, while `credentials.get_password("bcch")` resolves only the password, from the password variables and then `[bcch].password`. A lone password can therefore no longer be picked up and sent as the user, and the password *is* readable from the TOML file. `credentials.require("bcch")` demands both halves and, when either is missing, raises `MissingCredentialError` whose message reads "needs a user and a password" and lists both variable groups; `credentials.status()` flags a service whose user is configured but whose password is not. `fetch_bcch_vintages(user=..., password=...)` overrides either half explicitly.

Both halves travel in the query string of the GET request (`user=`, `pass=`, URL-encoded by `_build_bcch_url`), so treat the request URL as a secret: do not paste it into an issue, and remember that proxies and server logs see it. The connector scrubs the user and the password — raw and URL-encoded — from the warning text and from the `url` / `filename` attributes urllib attaches to its exceptions, so a fallback message is safe to paste; the URL you build yourself is not. Banxico's token goes in the `Bmx-Token` header; INEGI's is a path segment of the URL.

---

## 4. Runnable Worked Examples

Every script below runs offline and prints exactly the output shown; the bodies take a fraction of a second, the `puremacro` import a few seconds. The scripts that touch the database first point it at a temporary directory (`PUREMACRO_HTTP_CACHE_DIR`) so that they leave `~/.cache/puremacro/cache.db` untouched; drop that line to work against your real cache.

### 4.1 Fixture to panel to cartridge and back

A payload shaped like the BCB SGS response goes through the canary, the parser, `normalize_vintage_frame` and a `VintagePanel`, is packed into a `.pmz` in a temporary directory and comes back verified:

```python
import json, os, tempfile, zipfile
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()   # keep the example out of ~/.cache

from puremacro import pocket
from puremacro.fetch.realtime import (
    VintagePanel, normalize_vintage_frame, validate_payload,
    pack_realtime_cartridge, load_realtime_cartridge,
)
from puremacro.fetch.realtime.bcb import parse_bcb_json

# A fixture shaped exactly like the SGS endpoint's JSON: a list of {data, valor}
raw = json.dumps([{"data": "01/01/2020", "valor": "4,50"},
                  {"data": "01/04/2020", "valor": "3,75"}])
print(validate_payload("bcb", raw))

df = parse_bcb_json(raw, series_id="432", vintage_date="2024-01-15")
print(df)

long = normalize_vintage_frame(df, country="BRA", variable="policy_rate",
                               provider="bcb", series_id="432", units="rate")
panel = VintagePanel(df=long, metadata={"provider": "bcb"})
print(panel.coverage()[["country", "variable", "n_obs", "n_vintages", "n_periods"]])

with tempfile.TemporaryDirectory() as tmp:
    path = pack_realtime_cartridge(panel, f"{tmp}/bra.pmz", vintage="2024-01-15")
    print(zipfile.ZipFile(path).namelist())
    manifest = pocket.inspect_cartridge(path)
    print(manifest["format"], manifest["version"], manifest["frames"][0]["name"], manifest["frames"][0]["n_rows"])
    back = load_realtime_cartridge(path)          # verify=True: SHA-256 per frame
    print(back.metadata["provenance_source"], "|", back.metadata["provenance_vintage"])
    print(back.df.equals(panel.df))
```

```text
(True, '')
        date    vintage  value
0 2020-01-01 2024-01-15   4.50
1 2020-04-01 2024-01-15   3.75
  country     variable  n_obs  n_vintages  n_periods
0     BRA  policy_rate      2           1          2
['manifest.json', 'frames/data.npz', 'frames/vintage_metadata.npz']
puremacro-cartridge 1 data 2
LatAm Regional Real-Time Ecosystem | 2024-01-15
True
```

The comma-decimal `"4,50"` became `4.50`, the archive holds `manifest.json`, `frames/data.npz` and the `frames/vintage_metadata.npz` that carries `panel.metadata` across, and the data frame survives the round trip unchanged (`equals` is `True`, dtypes included). Note `n_vintages = 1`: one snapshot, one edition.

### 4.2 The canary contract

```python
import os, tempfile, warnings
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()

from puremacro.fetch.realtime import (
    SchemaCanary, SchemaDriftError, SchemaDriftWarning, validate_payload,
)

good = {"bmx": {"series": [{"idSerie": "SF61745",
        "datos": [{"fecha": "15/01/2026", "dato": "10.75"}]}]}}
drifted = {"bmx": {"series": [{"idSerie": "SF61745",
        "datos": [{"fecha": "15/01/2026", "value": "10.75"}]}]}}   # 'dato' renamed

print(SchemaCanary.validate_banxico(good))
print(SchemaCanary.validate_banxico(drifted))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    print(validate_payload("banxico", drifted))                 # warns, returns
print(caught[0].category.__name__, "->", caught[0].message)

try:
    validate_payload("banxico", drifted, raise_on_drift=True)
except SchemaDriftError as exc:
    print("raised", type(exc).__name__)

print(SchemaCanary.validate_bcb([{"data": "2026-01-01", "valor": "12.0"}]))   # ISO date, not DD/MM/YYYY
print(SchemaCanary.validate_bcch({"Codigo": 99, "Descripcion": "Servicio no disponible"}))
print(validate_payload("alfred", {"anything": 1}))              # unknown provider: passes through
```

```text
(True, '')
(False, "datos[0] missing 'fecha' or 'dato' key")
(False, "datos[0] missing 'fecha' or 'dato' key")
SchemaDriftWarning -> Schema drift detected for banxico: datos[0] missing 'fecha' or 'dato' key
raised SchemaDriftError
(False, "element[0] date '2026-01-01' does not match DD/MM/YYYY")
(False, 'BCCh API returned error code 99: Servicio no disponible')
(True, '')
```

The `validate_*` statics only report; `validate_payload` is where the policy lives. Here it defaults to `on_drift="warn"` (no `raise_on_drift`), so it emits `SchemaDriftWarning`, records `('schema_drift', 'warning_emitted')` and returns; `raise_on_drift=True` — or `on_drift="raise"` — raises `SchemaDriftError` and records `('schema_drift', 'none')`, and `on_drift="ignore"` returns silently but still records `('schema_drift', 'ignored')`. The parsers take the same keyword and default to `"raise"`: `parse_banxico_json(drifted, on_drift="warn")` warns and returns the rows it could salvage, while the default raises.

### 4.3 Two captures make a vintage

The same BCB series captured on two different days, the second capture revising 2023Q4 and adding 2024Q1. Both go into the snapshot store; the as-of query and the revision triangle then fall out of `query_realtime_vintages` and `VintagePanel`:

```python
import os, tempfile
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()

from puremacro._cache_db import store_realtime_vintages, query_realtime_vintages
from puremacro.fetch.realtime import VintagePanel, normalize_vintage_frame
from puremacro.fetch.realtime.bcb import parse_bcb_json
from puremacro.reports import df_to_markdown

# Two captures of the same series on two different days. The second
# edition revises 2023Q4 and adds 2024Q1 -- that is what a vintage is here.
capture_1 = parse_bcb_json(
    [{"data": "01/07/2023", "valor": "100.0"}, {"data": "01/10/2023", "valor": "101.0"}],
    series_id="22099", vintage_date="2024-03-01")
capture_2 = parse_bcb_json(
    [{"data": "01/07/2023", "valor": "100.0"}, {"data": "01/10/2023", "valor": "101.6"},
     {"data": "01/01/2024", "valor": "102.1"}],
    series_id="22099", vintage_date="2024-06-01")
for cap in (capture_1, capture_2):
    n = store_realtime_vintages(cap.assign(provider="bcb", country="BRA", series_id="22099"))
    print("rows stored:", n)

cached = query_realtime_vintages("bcb", "BRA", "22099")
print(cached[["date", "vintage", "value"]])
as_of = query_realtime_vintages("bcb", "BRA", "22099", vintage_date="2024-03-31")  # editions up to that day
print(as_of[["date", "vintage", "value"]])

long = normalize_vintage_frame(cached, country="BRA", variable="gdp_real",
                               provider="bcb", series_id="22099", units="index")
panel = VintagePanel(long)
tri = panel.triangle("BRA", "gdp_real")
print(tri)
print(len(panel.revisions("BRA", "gdp_real")))   # every reference period predates the first capture
tri.index, tri.columns = tri.index.strftime("%Y-%m-%d"), tri.columns.strftime("%Y-%m-%d")
print(df_to_markdown(tri))                     # or df_to_latex / df_to_typst
print(panel.vintage_semantics()["bcb"])
```

```text
rows stored: 2
rows stored: 3
        date    vintage  value
0 2023-07-01 2024-03-01  100.0
1 2023-07-01 2024-06-01  100.0
2 2023-10-01 2024-03-01  101.0
3 2023-10-01 2024-06-01  101.6
4 2024-01-01 2024-06-01  102.1
        date    vintage  value
0 2023-07-01 2024-03-01  100.0
1 2023-10-01 2024-03-01  101.0
vintage     2024-03-01  2024-06-01
date                              
2023-07-01       100.0       100.0
2023-10-01       101.0       101.6
2024-01-01         NaN       102.1
0
|       date | 2024-03-01 | 2024-06-01 |
|------------|------------|------------|
| 2023-07-01 |        100 |        100 |
| 2023-10-01 |        101 |      101.6 |
| 2024-01-01 |            |      102.1 |
Banco Central do Brasil SGS API time series. Open public endpoint that overwrites in place, so the vintage date is the local SNAPSHOT date: the day this machine fetched the series and stored it in the SQLite realtime_vintages cache. One vintage per stored snapshot; history begins with the first local fetch. Not a publication date.
```

Five rows in the store, two of them for 2023Q4 (`101.0` on 2024-03-01, `101.6` on 2024-06-01), and the as-of query on 2024-03-31 returns only the March capture. The triangle is $T[t, v]$ of Section 1.2, `NaN` where 2024Q1 did not yet exist. Yet `revisions()` has zero rows: all three reference periods ended before the first capture (2024-03-01), so none has an observable first release and the censoring rule of Section 1.2 drops them all. Two captures, two columns, no usable revision pair: the revision sample of a snapshot archive begins with the reference periods that end after capturing started, and `news_or_noise()` needs many distinct capture days beyond that before it says anything.

### 4.4 The registry call, offline

`vintage_panel` against `banxico` with no token and a cold store, then with a snapshot in the store, then a keyless provider during a simulated outage (the network call is replaced by a `URLError`, so nothing leaves the machine). The Banxico policy rate is catalogued as daily, so the call asks for `freq="D"`; `freq="Q"` would be refused with `catalogue series SF61745 is daily (freq='D'); requested freq='Q'` before any request is built.

```python
import os, tempfile, warnings, urllib.error
from unittest.mock import patch
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()
os.environ["PUREMACRO_CREDENTIALS_FILE"] = "/nonexistent/credentials.toml"
for var in ("BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY"):
    os.environ.pop(var, None)

import pandas as pd
from puremacro._cache_db import store_realtime_vintages
from puremacro.fetch.realtime import vintage_panel
from puremacro.fetch.realtime.bcb import fetch_bcb_vintages
from puremacro.narrative.sources._telemetry import connector_health

# 1. No token and a cold store: the failure is recorded, not raised
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    panel = vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")
print(panel.is_empty(), "|", panel.metadata["failed"]["MEX:policy_rate:banxico"].split(" needs")[0])

# 2. A snapshot captured on an earlier day makes the same call work offline
store_realtime_vintages(pd.DataFrame({
    "provider": "banxico", "country": "MEX", "series_id": "SF61745",
    "date": ["2026-01-15", "2026-02-15"], "vintage": "2026-03-01", "value": [10.50, 10.25]}))
panel = vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")
print(panel.coverage()[["country", "variable", "provider", "n_obs", "n_vintages"]])
print(panel.metadata["provider_used"], panel.metadata["failed"])

# 3. A keyless provider during a simulated outage: warm cache -> warning + fallback event
store_realtime_vintages(pd.DataFrame({
    "provider": "bcb", "country": "BRA", "series_id": "432",
    "date": ["2026-01-01"], "vintage": "2026-03-01", "value": [12.25]}))
with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("simulated outage")):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = fetch_bcb_vintages("432")
print(len(df), "|", str(caught[0].message).split("; ")[1])
print(connector_health()[["source", "n_total", "n_success", "n_fallback"]])
```

```text
True | MissingCredentialError: Banco de México SIE API
  country     variable provider  n_obs  n_vintages
0     MEX  policy_rate  banxico      2           1
{'MEX': 'banxico'} {}
1 | falling back to cached vintages.
  source  n_total  n_success  n_fallback
0    bcb        1          0           1
```

Step 1 returns an empty panel and files the `MissingCredentialError` under `metadata["failed"]["MEX:policy_rate:banxico"]` (the suppressed warning is `vintage_panel`'s usual "returned nothing for 1 of 1 requested countries"). Step 2 is the same call served from the store: no token, no request, `provider_used == {'MEX': 'banxico'}` and nothing failed. Step 3 shows the fallback branch of Section 1.3 for the keyless connector — the stored row comes back with the `falling back to cached vintages` warning and a `('fallback', 'sqlite_cache')` event, which `connector_health()` reports as one fallback for `bcb`. Neither of the first two steps logged an event; only a live attempt does.

### 4.5 What the catalogue and the credentials registry say

```python
from puremacro.fetch.realtime import vintage_catalog, providers_for, VINTAGE_SEMANTICS
from puremacro import credentials

cat = vintage_catalog()
latam = cat[cat["provider"].isin(["banxico", "inegi", "bcb", "bcch"])]
print(latam[["provider", "country", "variable", "series_id", "units", "freq"]].to_string(index=False))
print(providers_for("MEX", "cpi"), providers_for("MEX", "tasa_objetivo"))
for svc in ("banxico", "inegi", "bcch"):
    print(svc, credentials.SERVICES[svc].env_vars)
print("bcch password vars:", credentials.PASSWORD_ENV_VARS["bcch"])
print(VINTAGE_SEMANTICS["banxico"])
```

```text
provider country    variable                       series_id      units freq
 banxico     MEX    activity                         SR17631      index    M
 banxico     MEX         cpi                             SP1      index    M
 banxico     MEX policy_rate                         SF61745       rate    D
     bcb     BRA    activity                           24363      index    M
     bcb     BRA         cpi                             433 growth_mom    M
     bcb     BRA    gdp_real                           22099      index    Q
     bcb     BRA policy_rate                             432       rate    D
    bcch     CHL    activity   F032.IMC.IND.Z.Z.EP18.Z.Z.0.M      index    M
    bcch     CHL         cpi            F074.IPC.IND.Z.Z.C.M      index    M
    bcch     CHL    gdp_real F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T      level    Q
    bcch     CHL policy_rate        F022.TPM.TPO.D001.NO.Z.D       rate    D
   inegi     MEX    activity                          736184      index    M
   inegi     MEX         cpi                          628197      index    M
   inegi     MEX    gdp_real                          735848      level    Q
['banxico', 'inegi'] ['banxico']
banxico ('BANXICO_API_KEY', 'BMX_TOKEN', 'PUREMACRO_BANXICO_API_KEY')
inegi ('INEGI_API_KEY', 'PUREMACRO_INEGI_API_KEY')
bcch ('BCCH_API_USER', 'PUREMACRO_BCCH_API_USER')
bcch password vars: ('BCCH_API_PASS', 'PUREMACRO_BCCH_API_PASS')
Banco de México SIE API time series. Upstream overwrites series in place, so the vintage date is the local SNAPSHOT date: the day this machine fetched the series and stored it in the SQLite realtime_vintages cache. One vintage per stored snapshot; history begins with the first local fetch. Not a publication date.
```

Notebook 58 (`notebooks/58_latin_america_realtime_macro.py`, Spanish edition `_es.py`) walks the same objects end to end — panel, cartridge, coverage, as-of cross-section, triangle, Mankiw-Shapiro. Its panel is **synthetic**: it is generated with `numpy.random.default_rng(42)` around linear trends, so it exercises the machinery but its numbers are not Banxico, BCB or BCCh data.

---

## 5. Full API Specification

The four connector modules share one shape; the differences are the credential arguments and the endpoint constant.

```text
puremacro.fetch.realtime.banxico
BANXICO_SERIES_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"
parse_banxico_json(raw: bytes | str | dict, *, series_id: str = "",
                   vintage_date: str | pd.Timestamp | None = None,
                   on_drift: str = "raise") -> pd.DataFrame
fetch_banxico_vintages(series_id: str, *, token: str | None = None, vintage_date: str | None = None,
                       timeout: float = 60.0, use_cache: bool = True, history: bool = True,
                       on_drift: str = "raise") -> pd.DataFrame
fetch_banxico_panel(countries, variables, *, token: str | None = None, timeout: float = 60.0,
                    use_cache: bool = True, history: bool = True, on_drift: str = "raise",
                    **_ignored) -> VintagePanel
MEXICO_VINTAGE_NOTE, SPAIN_VINTAGE_NOTE, BANXICO_SIE_BASE, INEGI_BIE_BASE   # module constants

puremacro.fetch.realtime.inegi
INEGI_SERIES_URL = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
                   "INDICATOR/{series_id}/es/00/false/BIE/2.0/{token}?type=json"
parse_inegi_json(raw: bytes | str | dict, *, series_id: str = "",
                 vintage_date: str | pd.Timestamp | None = None, freq: str | None = None,
                 on_drift: str = "raise") -> pd.DataFrame
fetch_inegi_vintages(series_id: str, *, token: str | None = None, vintage_date: str | None = None,
                     timeout: float = 60.0, use_cache: bool = True, history: bool = True,
                     on_drift: str = "raise") -> pd.DataFrame
fetch_inegi_panel(countries, variables, *, token: str | None = None, timeout: float = 60.0,
                  use_cache: bool = True, history: bool = True, on_drift: str = "raise",
                  **_ignored) -> VintagePanel

puremacro.fetch.realtime.bcb
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
parse_bcb_json(raw: bytes | str | list | dict, *, series_id: str = "",
               vintage_date: str | pd.Timestamp | None = None,
               on_drift: str = "raise") -> pd.DataFrame
fetch_bcb_vintages(series_id: str, *, vintage_date: str | None = None, timeout: float = 60.0,
                   use_cache: bool = True, history: bool = True,
                   on_drift: str = "raise") -> pd.DataFrame
fetch_bcb_panel(countries, variables, *, timeout: float = 60.0, use_cache: bool = True,
                history: bool = True, on_drift: str = "raise", **_ignored) -> VintagePanel

puremacro.fetch.realtime.bcch
BCCH_SIETE_ENDPOINT = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"
BCCH_SIETE_URL = BCCH_SIETE_ENDPOINT + "?user={user}&pass={password}&function=GetSeries"
                 "&timeseries={series_id}&firstdate={firstdate}&lastdate={lastdate}"
parse_bcch_json(raw: bytes | str | dict, *, series_id: str = "",
                vintage_date: str | pd.Timestamp | None = None,
                on_drift: str = "raise") -> pd.DataFrame
fetch_bcch_vintages(series_id: str, *, user: str | None = None, password: str | None = None,
                    firstdate: str | None = None, lastdate: str | None = None,
                    vintage_date: str | None = None, timeout: float = 60.0,
                    use_cache: bool = True, history: bool = True,
                    on_drift: str = "raise") -> pd.DataFrame
fetch_bcch_panel(countries, variables, *, user: str | None = None, password: str | None = None,
                 timeout: float = 60.0, use_cache: bool = True, history: bool = True,
                 on_drift: str = "raise", **_ignored) -> VintagePanel
```

| Parameter | Meaning |
|---|---|
| `raw` | The HTTP body (`bytes`), its text, or the decoded JSON. Empty input returns an empty `[date, vintage, value]` frame; a body that is not valid JSON raises `json.JSONDecodeError`, which the fetch layer treats as a failed fetch rather than as an empty series; a payload that fails the canary follows `on_drift`. |
| `series_id` | The provider's identifier. Not written into the output frame. |
| `vintage_date` | The snapshot date to stamp on this fetch. Default: today at local midnight. |
| `freq` (INEGI) | A hint used only when the payload is too short for the structural rule below to decide. |
| `firstdate`, `lastdate` (BCCh) | Optional `yyyy-mm-dd` window; empty means the whole series. |
| `token`, `user`, `password` | Explicit credentials; when omitted they are resolved as in Section 3. |
| `timeout` | Seconds passed to `urllib.request.urlopen`. |
| `use_cache` | `True` stores the parsed rows in `realtime_vintages` and falls back to the stored snapshots on failure. `False` neither reads nor writes the store: the live snapshot is returned as parsed and any failure is raised. |
| `history` | `True` (default) returns every snapshot stored locally for the series, today's included, one vintage per snapshot date; `False` returns only the snapshot just fetched. |
| `on_drift` | `"raise"` (default), `"warn"` or `"ignore"`; the canary policy of Section 1.4. Any other value raises `ValueError`. |
| `countries`, `variables` | ISO3 codes and **canonical** variable names. Entries outside the provider's own country or table are skipped silently. |

**INEGI period parsing.** `TIME_PERIOD` arrives as `YYYY/NN`, which is a month for a monthly indicator and a quarter for a quarterly one, and nothing in the payload says which reliably. `parse_inegi_json` decides structurally, with no hard-coded series id: a period number above 4 rules out quarters; period numbers that never exceed 4 across two or more years, or across five or more observations, rule out months, because a monthly series is contiguous and five consecutive months always reach May. Only when the payload is too short to tell does the caller's `freq` decide, then the `FREQ` field (Spanish words such as "trimestral" / "mensual", or the numeric `CL_FREQ` codes); failing all of those the series is treated as monthly. Quarterly periods are dated to the first day of the quarter, monthly ones to the first day of the month.

```text
puremacro.fetch.realtime.catalog
BANXICO_SERIES, INEGI_SERIES, BCB_SERIES, BCCH_SERIES : dict[str, SeriesSpec]
SeriesSpec(series_id: str, units: str = "level", source: str = "", note: str = "", freq: str = "")
SeriesSpec.default_transform() -> str
VERIFY_ONLINE = "VERIFY ONLINE"
CANONICAL_VARIABLES["policy_rate"], CANONICAL_VARIABLES["activity"]
VARIABLE_ALIASES: tpm, selic, tasa_objetivo, interest_rate, mp_rate, central_bank_rate,
    target_rate, overnight_rate -> "policy_rate"; economic_activity, igae, imacec,
    ibc_br -> "activity"
resolve_spec(provider, country, variable, *, catalog=None) -> SeriesSpec | None
resolve_series(provider, country, variable, *, catalog=None) -> str | None
providers_for(country, variable="gdp_real") -> list[str]
vintage_catalog(provider=None) -> pd.DataFrame   # + a freq column

puremacro.fetch.realtime.canary
DRIFT_POLICIES = ("raise", "warn", "ignore")
class SchemaDriftError(ValueError)
class SchemaDriftWarning(UserWarning)
handle_drift(provider: str, reason: str, on_drift: str = "raise", *, stacklevel: int = 2) -> None
class SchemaCanary:
    validate_banxico(payload: Any) -> tuple[bool, str]          # staticmethod
    validate_inegi(payload: Any) -> tuple[bool, str]            # staticmethod
    validate_bcb(payload: Any) -> tuple[bool, str]              # staticmethod
    validate_bcch(payload: Any) -> tuple[bool, str]             # staticmethod
    check(provider: str, payload: Any, *, raise_on_drift: bool = False,
          on_drift: str | None = None) -> tuple[bool, str]      # classmethod
validate_payload(provider: str, payload: Any, *, raise_on_drift: bool = False,
                 on_drift: str | None = None) -> tuple[bool, str]

puremacro.fetch.realtime.cartridge
pack_realtime_cartridge(panel_or_df: VintagePanel | pd.DataFrame, path: str | Path, *,
                        source: str = "LatAm Regional Real-Time Ecosystem",
                        vintage: str | None = None,
                        notes: str = "Latin America central bank real-time macroeconomic vintage cartridge") -> Path
load_realtime_cartridge(path: str | Path, *, verify: bool = True) -> VintagePanel

puremacro._cache_db
store_realtime_vintages(df, *, conn: sqlite3.Connection | None = None) -> int
query_realtime_vintages(provider: str, country: str, series_id: str, *,
                        vintage_date: str | None = None,
                        conn: sqlite3.Connection | None = None) -> pd.DataFrame
record_connector_event(source: str, outcome: str, fallback_used: str, *,
                       conn: sqlite3.Connection | None = None) -> None

puremacro.credentials
SERVICES["banxico" | "inegi" | "bcch"] : ServiceCredentialSpec(name, env_vars, signup_url, description)
PASSWORD_ENV_VARS = {"bcch": ("BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS")}
get(service: str, *, explicit: str | None = None) -> str | None          # alias: get_credential
get_password(service: str, *, explicit: str | None = None) -> str | None
require(service: str, *, explicit: str | None = None) -> str            # alias: require_credential
status() -> pd.DataFrame
class MissingCredentialError(RuntimeError)

puremacro.fetch.realtime
SUPPORTED_FREQUENCIES = {"Q", "M", "D"}
vintage_panel(countries=None, *, series="gdp_real", variables=None, freq="Q",
              providers="oecd_stes", catalog=None, store=None, refresh=False, use_cache=True,
              timeout=None, warn_on_mixed_providers=True, drop_unadjusted=True) -> VintagePanel
```

- `store_realtime_vintages` accepts `date` or `observation_date` and `vintage` or `vintage_date` as column names, requires `provider`, `country`, `series_id` and `value`, upper-cases the country, stores `NaN` values as `NULL`, skips rows whose dates do not parse, and returns the number of rows written (`0` for an empty frame).
- `query_realtime_vintages` returns `[date, vintage, value, provider, country, series_id]` with `date` and `vintage` as datetimes, ordered by observation then vintage; `vintage_date=` keeps editions with `vintage_date <= vintage_date`.
- `record_connector_event` writes `(ts, source, outcome, fallback_used)` with `ts` the current Unix time; the values are free strings here, unlike the validated vocabulary of `narrative.sources._telemetry.log_event`.
- `pack_realtime_cartridge` packs `panel.df` (or the frame you pass) as the frame `data`, plus `vintage_metadata` when a `VintagePanel` with metadata is handed to it; `path` is used verbatim, so the `.pmz` suffix is a convention.
- `get` resolves the token, or the *user* half of a two-part service; `get_password` resolves the password half and returns `None` for every service outside `PASSWORD_ENV_VARS`. `require` demands both halves of a pair before it returns the user.

---

## 6. Result Interface & Manuscript Export

**Parser and fetch frames.** `parse_*_json` and `fetch_*_vintages` return a `pd.DataFrame` with columns `date` (datetime), `vintage` (datetime) and `value` (float), sorted and deduplicated on `(date, vintage)`; the cache fallback returns the same three columns from `query_realtime_vintages`.

**`VintagePanel`** (`puremacro.fetch.realtime._base`), a dataclass with two fields:

- `df`: the tidy frame with `VINTAGE_COLUMNS = [country, variable, date, vintage, value, provider, series_id, units]`, one row per `(country, variable, date, vintage)`.
- `metadata`: a dict. From `fetch_*_panel`: `provider`, `failed`. From `vintage_panel`: `requested_countries`, `variables`, `providers_tried`, `provider_used`, `missing`, `missing_pairs`, `failed`, `freq` (and `unadjusted_dropped` when the seasonal screen removed editions). From `load_realtime_cartridge`: `cartridge_path`, `provenance_source`, `provenance_vintage`, `provenance_notes`.

Its methods are the operations of Section 1.2: `countries` / `variables` / `providers` (properties), `len(panel)`, `is_empty()`, `vintage_semantics()`, `coverage()` (`n_obs`, `n_vintages`, `n_periods`, first and last date and vintage per series), `as_of(vintage_date)` (a `(country, date)`-indexed wide frame), `long(country, variable)`, `triangle(country, variable, *, transform="level", periods_per_year=4)`, `revisions(country, variable, *, transform=None, periods_per_year=4, release=0)` (`transform=None` picks the transform the catalogued `units` imply), `news_or_noise(...)` returning a `MankiwShapiroResult`, `news_or_noise_panel(...)` returning one row per `(country, variable)`, `filter(...)` and `VintagePanel.concat(...)`.

**`MankiwShapiroResult`** (`puremacro.vintages`), a dataclass with `n_obs`, `transform`, `hac_lags`, `significance`, `mean_revision`, `se_mean_revision`, `t_mean_revision`, `p_mean_revision`, `std_revision`, `alpha_on_preliminary`, `beta_on_preliminary`, `se_beta_on_preliminary`, `t_beta_on_preliminary`, `p_beta_on_preliminary`, `alpha_on_final`, `beta_on_final`, `se_beta_on_final`, `t_beta_on_final`, `p_beta_on_final`, `noise_share`, `rejects_news`, `rejects_noise` and `verdict`.

**Canary and cartridge results.** The validators return `(ok, reason)` tuples; `load_realtime_cartridge` returns a `VintagePanel`, and `pocket.load(path)` returns the underlying `Cartridge` with `frames`, `provenance` (`created`, `puremacro_version`, `source`, `vintage`, `notes`, `call`, `host`), `records`, `verify()` and `summary()`.

**Manuscript export.** None of these objects carries `summary()`, `plot()`, `to_frame()`, `to_markdown()`, `to_latex()` or `to_typst()` methods of its own; their tables — `coverage()`, `as_of()`, `triangle()`, `revisions()`, `news_or_noise_panel()`, `vintage_catalog()`, `credentials.status()`, `connector_health()` — are plain DataFrames, and `puremacro.reports.df_to_markdown`, `df_to_latex` and `df_to_typst` render them, as Section 4.3 does for the triangle. `Cartridge.summary()` is the one text summary in this stack.

---

## 7. Caveats & Limitations

- **Snapshot semantics.** One fetch is one edition; revisions exist only across captures on distinct days; a same-day re-fetch overwrites; reference periods that ended before the first capture have no observable first release and are dropped from `revisions()` (Section 1.2); the vintage stamp is local midnight while `pack_realtime_cartridge`'s default `vintage` is the UTC date. For archived history of Mexican quarterly GDP use `oecd_stes`.
- **No shared HTTP layer.** The four connectors call `urllib` directly and bypass `puremacro._http` and `fetch_with_backoff`: no HTTP body cache, no SSL fallback, no retry on HTTP 429. A rate limit surfaces as an `HTTPError`, hence as a store fallback or an exception.
- **`use_cache=False` is a full opt-out.** It neither reads nor writes `realtime_vintages`, so there is no fallback at all: the live snapshot is returned as parsed and every failure propagates. That is the flag to use when you want to know the request failed; leave it `True` when you want the panel to survive an outage.
- **Telemetry.** Events use outcomes `success`, `fallback` and `schema_drift`, with `fallback_used` values `none`, `sqlite_cache`, `warning_emitted` and `ignored`. No event is written for the no-credential store path or for a cold-store raise. The connectors write through `_cache_db.record_connector_event`, so the `PUREMACRO_NARRATIVE_TELEMETRY=0` kill-switch of [CONNECTOR_HEALTH.md](CONNECTOR_HEALTH.md) does **not** silence them. `connector_health()` counts every row with `fallback_used != 'live'` as a fallback, so a successful real-time fetch (`'none'`) is counted in both `n_success` and `n_fallback`; query `connector_events` directly for these sources.
- **One frequency per `vintage_panel` call.** `freq` is now enforced against the catalogue's `freq` field, not against the data: a series that declares another frequency is refused with a reason rather than served at the wrong label, and an entry that declares none is served at `"Q"` only. Nothing checks that the *payload* really is at the declared frequency, so a provider that changes a series' frequency upstream will still be labelled by the catalogue. Pass the matching `periods_per_year` to the revision helpers, and keep the daily policy rates out of revision tests.
- **BCCh credentials** travel in the query string of a plain GET. They are scrubbed from the connector's own warnings and exception attributes, but a URL you build yourself with `BCCH_SIETE_URL.format(...)` carries them in clear text.
- **Catalogue.** Entries marked `VERIFY ONLINE` in their note were written without a live check of the identifier; `pytest -m network` is the audit that clears them. `catalog=` overrides do not redirect these connectors (Section 2).
- **Cartridges** are a transport format: the SHA-256 is unsigned, `verify=False` skips the re-encode check but not the on-read digest comparison, and the archive can only hold what the npz codec accepts (no arbitrary Python objects). As the `pocket` module notes, the fetchers need sockets, so in a Pyodide session these connectors can only serve the cache or a cartridge — the cartridge is the intended way in ([CACHE_DB.md](CACHE_DB.md) for the IDBFS persistence caveat).
- **Notebook 58** uses a synthetic panel (Section 4.5).

---

## References

- Croushore, D., & Stark, T. (2001). "A Real-Time Data Set for Macroeconomists." *Journal of Econometrics*, 105(1), 111–130.
- Mankiw, N. G., & Shapiro, M. D. (1986). "News or Noise: An Analysis of GNP Revisions." *Survey of Current Business*, 66(5), 20–25.
- Banco de México. *SIE API REST*, https://www.banxico.org.mx/SieAPIRest/service/v1/ (token request: https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html).
- INEGI. *API de Indicadores (Banco de Indicadores / BIE)*, https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/.
- Banco Central do Brasil. *Sistema Gerenciador de Séries Temporais (SGS)*, endpoint https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json.
- Banco Central de Chile. *Base de Datos Estadísticos, SIETE REST*, https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx (registration: https://si3.bcentral.cl/estadisticas/principal1/registro/index.html).
