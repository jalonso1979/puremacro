# `puremacro.instruments` — Unified Instrument Protocol + Discovery Registry

**Status:** Approved 2026-05-03 (Trim A scope — protocol + adapters + registry + catalog of already-shipped things; defers 4 new literature loaders to Phase 2 / 0.5.1).
**Target release:** 0.5.0 (minor — new public subpackage, no breaking changes).
**Driving lenses:** D ergonomics (uniform downstream API) + research-infrastructure value (discoverable shock catalogue).

## Motivation

The package today exposes two unrelated shock-source abstractions:

- `puremacro.narrative.NarrativeInstrument` — fiscal IV from event lists. Mutable, owns one `quarterly: pd.Series` (date-indexed), already has `.to_proxy_svar()` / `.to_lp_iv()`.
- `puremacro.hfi.JKResult` — monetary HFI decomposition. Frozen, owns **two** candidate proxy series (`mp_shock`, `info_shock`) as plain ndarrays (no date index), no convenience methods.

Downstream code (`var.identify.proxy.proxy_svar`, `lp.iv.lp_iv`) accepts a numpy proxy series. Users today write different unwrap patterns depending on the source, and there is no programmatic way to ask "what identified-shock series do I have available in this package?". Both are real friction points in the MAV research workflow.

This design introduces a thin shared abstraction (`Instrument` wrapper + `InstrumentLike` protocol) plus a self-describing registry of available instruments. Existing classes get a single `as_instrument()` adapter; downstream signatures are unchanged.

## Non-goals

- **No** `Instrument.compose()` operator (combining e.g. narrative + HFI series). YAGNI.
- **No** automatic frequency conversion inside `Instrument` (caller's responsibility — too many policy choices).
- **No** new external loaders (FRED, BIS, IMF) in Phase 1. Stub registry slots only.
- **No** new literature shock loaders (Romer-Romer 2004, BBD EPU, Caldara-Iacoviello GPR, Bloom 2009) in Phase 1. Deferred to Phase 2 / 0.5.1.
- **No** breaking changes to `proxy_svar` / `lp_iv` signatures. Polymorphism lives on `Instrument`, not the consumers.
- **No** changes to `JKResult` frozenness or `NarrativeInstrument` mutability.

## Architecture

New top-level subpackage `puremacro/instruments/` with three responsibilities:

```
puremacro/instruments/
├── __init__.py        # re-exports public surface
├── _core.py           # Instrument dataclass + InstrumentLike Protocol
├── _registry.py       # InstrumentSpec + list_available + load + describe
├── _catalog.py        # populates registry with Phase-1 entries
└── _results.py        # (unused in Phase 1 — Instrument lives in _core.py since
                       #  it's the protocol's canonical type, not a return type)
```

`Instrument` lives in `_core.py` (not `_results.py`) because it is the canonical *interface* type, not the result of any specific function. Functions that return identified shock series (`gk2015_surprise`, `events_to_quarterly`) continue to return raw `pd.Series` / `np.ndarray`; the wrapping happens at the `as_instrument()` adapter layer.

## Public API

### `Instrument` — canonical wrapper

```python
@dataclass(frozen=True)
class Instrument:
    """A single identified shock or instrument series with provenance.

    Constructed via `as_instrument()` adapters on existing classes
    (`NarrativeInstrument`, `JKResult`) or via the registry's `load(key)`.
    Downstream consumers (`proxy_svar`, `lp_iv`, future SVAR-IV variants)
    accept this wrapper and dispatch uniformly.
    """
    series: pd.Series           # date-indexed, any frequency
    name: str                   # short identifier matching registry key when loaded from registry
    source: str                 # human-readable provenance (e.g., "Ramey 2011 defense buildup events")
    category: str               # one of: narrative_replication | narrative_connector |
                                #         monetary_hfi | literature | external_csv
    frequency: str              # "M" | "Q" | "A" — pandas-style frequency code
    metadata: dict[str, Any]    # free-form (e.g., country, target, reference, n_events)

    # ---- ergonomic adapters into existing puremacro pipelines ----
    def to_proxy_svar(self, Y, *, p, horizon,
                      n_boot=500, ci=0.9, seed=0): ...
    def to_lp_iv(self, df, *, y, x, **kwargs): ...

    # ---- diagnostics ----
    def diagnostics(self) -> dict: ...   # n_obs, mean, std, first/last date, gap stats
    def validate_against(self, benchmark: pd.Series) -> dict:
        """Correlation, lead-lag CCF, overlap with a benchmark series."""
        ...
    def summary(self) -> str: ...        # human-readable one-paragraph summary
```

### `InstrumentLike` — runtime-checkable protocol

```python
@runtime_checkable
class InstrumentLike(Protocol):
    def as_instrument(self) -> Instrument: ...
```

Single-method protocol. Lets `isinstance(obj, InstrumentLike)` work for type-narrowing, but the canonical use is to call `obj.as_instrument()` and work with the returned `Instrument`. Forward-compatible with future shock types.

### Registry primitives

```python
@dataclass(frozen=True)
class InstrumentSpec:
    key: str                     # unique snake_case identifier, e.g. "ramey_2011_defense"
    name: str                    # human-readable display name
    category: str                # same enum as Instrument.category
    description: str             # one-paragraph, what the series represents
    reference: str               # full citation (Author Year, journal, vol)
    loader: Callable[..., Instrument]   # construct the Instrument; may take kwargs
    country: str | None          # ISO3 or None for cross-country
    frequency: str               # "M" | "Q" | "A"
    requires_network: bool       # True if loader needs HTTP
    requires_fixture: bool       # True if loader needs a user-supplied CSV (e.g. AB 1991, RR 2004)
```

### Public registry functions

```python
def list_available(
    *,
    category: str | None = None,
    country: str | None = None,
    include_unavailable: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of catalogued instruments, one row per spec.

    Columns: key, name, category, country, frequency, reference,
             available, requires_network, requires_fixture.

    By default filters to entries whose `available` flag is True
    (i.e., requires_network=False AND (requires_fixture=False OR
    fixture file is present in tests/fixtures/)). Pass
    include_unavailable=True to see the full catalogue including
    stub entries for unwrapped connectors.
    """
    ...

def load(key: str, **kwargs) -> Instrument:
    """Construct an Instrument by registry key. Forwards kwargs to the loader."""
    ...

def describe(key: str) -> str:
    """Return a multi-line human-readable description of the spec at `key`."""
    ...
```

## Adapter additions to existing classes (no breaking changes)

### `NarrativeInstrument.as_instrument()`

```python
def as_instrument(self) -> "Instrument":
    from ..instruments import Instrument
    return Instrument(
        series=self.quarterly,
        name=self.metadata.get("registry_key", "narrative_instrument"),
        source=self.metadata.get("source", "narrative aggregation"),
        category="narrative_replication" if self.metadata.get("is_replication")
                 else "narrative_connector",
        frequency="Q",
        metadata={
            "n_events": len(self.events),
            "target": self.target,
            "aggregation": self.aggregation,
            **self.metadata,
        },
    )
```

The existing `NarrativeInstrument.to_proxy_svar()` / `to_lp_iv()` keep their signatures verbatim; their bodies become 1-liner delegations to `self.as_instrument().to_proxy_svar(...)` so `Instrument` is the single source of truth for the call shape.

### `JKResult.as_instrument()`

```python
def as_instrument(
    self,
    *,
    component: str = "mp",       # "mp" | "info"
    index: pd.DatetimeIndex,     # required: JKResult carries no datetime info
) -> "Instrument":
    from ..instruments import Instrument
    if component not in ("mp", "info"):
        raise ValueError(f"component must be 'mp' or 'info', got {component!r}")
    arr = self.mp_shock if component == "mp" else self.info_shock
    if len(index) != len(arr):
        raise ValueError(
            f"index length {len(index)} != shock array length {len(arr)}"
        )
    return Instrument(
        series=pd.Series(arr, index=index, name=f"jk_{component}_shock"),
        name=f"jk2020_{component}_shock",
        source=f"Jarociński-Karadi 2020 {component} component ({self.method})",
        category="monetary_hfi",
        frequency="M",
        metadata={
            "method": self.method,
            "n_admissible": self.n_admissible,
            "rotation": self.rotation,
        },
    )
```

The required `index` parameter is the only ergonomic friction here, and it is necessary because `JKResult` deliberately does not carry datetime info (the surprises are computed from event-window FFR-futures changes, not a regular calendar). Callers typically have the index in hand from the surprise-construction step.

## Phase-1 catalog

Total entries: ~13 immediately-loadable + ~12 stubs = ~25 rows in `list_available(include_unavailable=True)`.

### Narrative replications (6 entries — `available=True`, no network/fixture needed)

| key | source |
|-----|--------|
| `ramey_2011_defense` | `narrative.replication.load_ramey_2011_defense` |
| `romer_romer_2010_fiscal` | `narrative.replication.load_romer_romer_2010` |
| `mertens_ravn_2013_tax` | `narrative.replication.load_mertens_ravn_2013` |
| `cloyne_2013_uk_tax` | `narrative.replication.load_cloyne_2013_uk` |
| `romer_romer_2017_fiscal` | `narrative.replication.load_romer_romer_2017` |
| `dglp_2011_consolidations` | `narrative.replication.load_dglp_2011` |

Each loader returns a `NarrativeInstrument` already; the registry adapter calls `.as_instrument()` on it.

### Narrative connectors (6 entries — `requires_network=True` OR `requires_fixture=True`)

| key | connector module | available |
|-----|------------------|-----------|
| `us_treasury_press` | `narrative.sources.us_treasury` | requires fixture (post-Cluster F) |
| `us_federal_register` | `narrative.sources.us_federal_register` | available with recorded fixture |
| `us_dod_contracts` | `narrative.sources.us_dod_contracts` | requires fixture (post-Cluster D UA) |
| `oecd_surveys` | `narrative.sources.oecd_surveys` | available with recorded fixture |
| `imf_articleiv` | `narrative.sources.imf_articleiv` | available with recorded fixture |
| `news_api` | `narrative.sources.news_api` | requires network + API key |

The loader for these wraps the connector's iterator into a list of `NarrativeEvent`s via `score_keyword` (default), aggregates via `events_to_quarterly`, and returns `Instrument`. They are listed even when unavailable (with `requires_network=True` or `requires_fixture=True`) so `list_available(include_unavailable=True)` exposes them as discoverable.

### Monetary HFI (1 entry — `requires_fixture=True`)

| key | source |
|-----|--------|
| `gk2015_ffr_surprise` | `puremacro.hfi.gk2015_surprise` — Gertler-Karadi 2015 FFR-futures month-end-adjusted surprise |

Loader takes `(announcement_dates, ffr_futures_changes)` kwargs forwarded to `gk2015_surprise`, then wraps the resulting Series as `Instrument`. The user must provide the underlying high-frequency data — same `requires_fixture=True` discipline as AB 1991. (The `JKResult.as_instrument()` adapter lives separately and is exercised by the round-trip tests rather than from the registry.)

### Stubs for the other 12 narrative connectors

`uk_obr`, `uk_hmt`, `de_bmf`, `fr_tresor`, `it_mef`, `jp_mof`, `ca_dof`, `ecb_press`, `eu_ecfin`, `imf_news`, `google_news`, `local_csv` — each gets an `InstrumentSpec` entry with `available=False` (no fixture, no offline test), so users browsing `list_available(include_unavailable=True)` see them and know they exist.

## Integration with existing pipelines

- `proxy_svar(Y, p, horizon, instrument_series, ...)` — **unchanged**. `Instrument.to_proxy_svar(Y, p=, horizon=)` builds the call as `proxy_svar(Y, p=p, horizon=horizon, instrument_series=self.series.values, ...)`.
- `lp_iv(df, y=, x=, z=, ...)` — **unchanged**. `Instrument.to_lp_iv(df, y=, x=)` reindexes `self.series` onto `df.index`, adds it as a column, and dispatches to `lp_iv` with `z=<that column name>`.
- `puremacro/instruments/__init__.py` re-exports: `Instrument`, `InstrumentLike`, `InstrumentSpec`, `list_available`, `load`, `describe`.
- `puremacro/__init__.py` is **not** modified — `__version__` stays the only top-level export per package convention.

## Testing strategy

`tests/test_instruments/` (new directory):

1. **`test_protocol.py`** (~10 tests)
   - `Instrument` is `@dataclass(frozen=True)` (assignment raises `FrozenInstanceError`)
   - `Instrument` constructor validates `category` is in the allowed enum
   - `InstrumentLike` is `@runtime_checkable`
   - `isinstance(NarrativeInstrument(...), InstrumentLike)` is True
   - `isinstance(JKResult(...), InstrumentLike)` is True
   - `Instrument.summary()` returns a non-empty string with `name`, `source`, frequency, n_obs
   - `Instrument.diagnostics()` returns the spec'd dict shape

2. **`test_adapters.py`** (~10 tests)
   - `NarrativeInstrument.as_instrument()` round-trip: identical proxy-SVAR output between `narr.to_proxy_svar(Y, p=, horizon=)` and `narr.as_instrument().to_proxy_svar(Y, p=, horizon=)`
   - `JKResult.as_instrument(component="mp", index=...)` returns Series with right shape, name, frequency
   - `JKResult.as_instrument(component="info", index=...)` returns the info component
   - `JKResult.as_instrument(component="bogus", ...)` raises ValueError
   - `JKResult.as_instrument(component="mp", index=<wrong-length>)` raises ValueError
   - `as_instrument()` results have correct category strings

3. **`test_registry.py`** (~10 tests)
   - `list_available()` returns a non-empty DataFrame with documented columns
   - `list_available(category="narrative_replication")` returns exactly 6 rows
   - `list_available(country="USA")` filters correctly
   - `list_available(include_unavailable=False)` excludes stub entries
   - `load("ramey_2011_defense")` returns an `Instrument` of the right category
   - `load("nonexistent_key")` raises `KeyError` with a helpful message
   - `describe("ramey_2011_defense")` returns a non-empty string with `reference`
   - Every catalog entry's `loader` is callable
   - Every catalog entry has a non-empty `reference` (citation discipline)
   - Every catalog entry's `country` is `None` or a valid ISO3

Total: ~30 tests. All deterministic (loaders for the 6 replications already work without network in the existing test suite).

## Pyodide compatibility

- `instruments/_core.py`, `_registry.py`, `_catalog.py` use only `dataclasses`, `typing`, `pandas`, `numpy` — no new deps.
- Network-needing loaders go through `narrative.sources._http.safe_get_*` (UA-override + SSL-fallback, hardened in 0.4.1).
- Public-API freeze test (`tests/test_public_api.py`) picks up `puremacro.instruments` automatically and snapshots `__all__` + `Instrument` field names.

## Versioning

- `pyproject.toml` and `puremacro/__init__.py` bump `0.4.1 → 0.5.0`.
- `tests/test_import.py` bump expected version.
- `CHANGELOG.md` adds `## 0.5.0 — YYYY-MM-DD` block: new `puremacro.instruments` subpackage; `as_instrument()` adapters on `NarrativeInstrument` + `JKResult`; ~25 catalogued instruments; 30 new tests.

## Risks and what could go wrong

1. **Registry coupling drift.** If a future refactor renames a connector function, the registry's `loader` reference breaks. Mitigated by `test_registry.py::test_every_loader_is_callable`, which would fail in CI.
2. **`as_instrument()` proliferation.** Once the protocol exists, every new shock class might be expected to ship one. Acceptable — it is a single 8-line method, and the protocol's value is precisely that uniform expectation.
3. **`requires_fixture=True` UX confusion.** A user who calls `load("us_treasury_press")` without a recorded fixture gets a runtime error from `safe_get_text` (deferred to live HTTP). Mitigated by `list_available()` defaulting to `available=True` filter, and by the spec's `requires_fixture` flag being human-discoverable.
4. **Phase-2 scope creep.** The registry will tempt incremental "let me just add one more loader" PRs. Discipline: anything beyond the Phase-1 catalog must land as a separate PR with its own fixture / network strategy.

## Out of scope (deferred to Phase 2 / 0.5.1)

- 4 new literature loaders (Romer-Romer 2004 monetary, BBD EPU, Caldara-Iacoviello GPR, Bloom 2009 stock-vol uncertainty).
- FRED, BIS, IMF Articleiv-as-CSV external loaders.
- `Instrument.compose()` operator for combining sources.
- Country-aware automatic filtering inside `proxy_svar`.
- Sphinx documentation page for the catalogue.
