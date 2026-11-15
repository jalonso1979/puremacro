# Automated LatAm & Macro Freshness Monitor

> 🇬🇧 English · 🇪🇸 [Español](es/macro_monitor.md)

`puremacro.macro_monitor` composes PureMacro's data-provenance, preflight,
nowcasting, and local-projection diagnostics into a deterministic, offline-first,
fail-closed monitoring dossier and bilingual (`en` / `es`) artifact bundle.

## Workflow Stages

1. **Connector Freshness & Provenance Audit**: calls
   `puremacro.fetch.freshness.connector_freshness_report` (`mode="recorded"`
   by default for deterministic offline runs, `"cache"`, or rate-budgeted
   `"live"`) cross-referenced with `puremacro.fetch.registry`. Live failures,
   empty payloads, stale data, and fallback paths (`failed`, `empty`,
   `schema_drift`, `degraded_fallback`, `stale`, `not_cached`) are surfaced
   explicitly in `overall_status`, `headline()`, and `warnings` with zero
   silent synthetic fallback.
2. **Panel Ingestion & Structural Preflight**: ingests a caller `DataFrame`, a
   CSV/JSON file, a registered `panel_builder` from `fetch.registry`, or the
   bundled offline OECD QNA growth panel (`MEX`, `USA`, `DEU`, `ESP`, `ITA`,
   `JPN`). Enforces explicit column checks, unparseable/corrupted date
   detection, and strict per-unit chronological ordering before running
   `puremacro.data.preflight_panel`, recording the input SHA-256 digest and
   sample span.
3. **Country Indicators, AR(1) Nowcast & LP Sensitivity Grid**: computes each
   country's latest observation date, QoQ and YoY growth, rolling volatility,
   rolling z-score anomaly flags (`OK`, `Z_ALERT_HIGH`, `Z_ALERT_LOW`,
   `SHORT_SAMPLE`, `MISSING`), pure-NumPy rolling AR(1) 1-step nowcast
   diagnostics, and a bounded `puremacro.lp.sensitivity.lp_sensitivity_grid`
   transmission check (`panel_lp` / `panel_lp_dk` for panels, or `lp_hac` /
   `la_lp` for single-country series) with explicit short-sample and
   instability notices.
4. **Bilingual UTF-8 Bundle Export**: `report.write_bundle(output_dir)` writes
   six self-contained UTF-8 files (`monitor.md`, `monitor.html`,
   `monitor.json`, `freshness.csv`, `latest_indicators.csv`, and
   `manifest.json`) with verified SHA-256 checksums, RFC 8259 strict JSON
   (`NaN`/`Infinity` serialized as `null`), and zero host filesystem paths.

## Quick Start (Python)

```python
from puremacro.macro_monitor import MacroMonitorConfig, run_macro_monitor

# Deterministic offline monitoring run
report = run_macro_monitor(mode="recorded", lang="en")
print(report.headline())
print(report.latest_indicators_df.head())

# Export bilingual dossier + SHA-256 manifest
files = report.write_bundle("artifacts/macro_monitor_en", lang="en")
report.write_bundle("artifacts/macro_monitor_es", lang="es")
```

Running against a custom LatAm panel with explicit transmission controls:

```python
config = MacroMonitorConfig(
    mode="recorded",
    sources=("vintage.banxico", "vintage.bcb", "oecd.qna_panel"),
    y="gdp_growth",
    x="policy_rate",
    controls=("inflation",),
    horizons=(0, 1, 2, 3, 4),
    lp_lags=(1, 2),
    lang="en",
    fail_on_outage=True,
)
report = run_macro_monitor("latam_panel.csv", config=config)
```

## CLI Usage

```bash
python -m puremacro.macro_monitor \
  --mode recorded \
  --lang en \
  --output-dir artifacts/macro_monitor \
  --format text
```

Exit codes:

- `0`: workflow completed with `overall_status` in `{"OK", "WARN"}`.
- `1`: `overall_status == "FAILED"` (preflight/schema/date corruption or connector
  failure), `--fail-on-outage` triggered on `DEGRADED`/outage connectors, or
  invalid/unreadable input arguments.

## Configuration & Report Reference

| Parameter / Option | Default | Meaning |
| --- | --- | --- |
| `mode` | `"recorded"` | Connector audit mode: `"recorded"`, `"cache"`, or `"live"`. |
| `sources` | `None` | Subset of cohort connector IDs (defaults to full 9-connector cohort). |
| `as_of` | `None` | ISO date override for freshness and staleness checks. |
| `db_path` | `None` | SQLite cache/telemetry database path (never written into exported files). |
| `max_requests` / `min_interval_seconds` / `timeout_seconds` | `5` / `0.5` / `20.0` | `RateBudget` bounds forwarded to live probes. |
| `panel_builder` | `None` | Optional `fetch.registry` source ID validated before execution. |
| `unit_col` / `time_col` | `None` | Panel identifier column names (auto-detected if omitted). |
| `y` / `x` / `controls` | Auto / `()` | Target, impulse, and control variables for the LP sensitivity grid. |
| `horizons` / `lp_lags` | `(0, 1, 2, 3, 4)` / `(1, 2)` | Horizons and lag grid passed to `lp_sensitivity_grid`. |
| `vol_window` / `zscore_threshold` | `8` / `2.0` | Rolling window and anomaly flag cutoff for latest indicators. |
| `short_sample_min_obs` | `24` | Threshold below which short-sample flags and LP warnings are raised. |
| `run_lp_sensitivity` | `True` | Disable with `False` (`--no-lp`) to skip the LP sensitivity step. |
| `lang` | `"en"` | Report language (`"en"` or `"es"`). |
| `fail_on_outage` | `False` | Exit CLI with code `1` whenever any connector outage or `DEGRADED` state occurs. |

## Fail-Closed Status Semantics

| `overall_status` | `passed` | Trigger Condition |
| --- | --- | --- |
| `OK` | `True` | All connectors healthy, preflight passes cleanly, and no warnings or errors recorded. |
| `WARN` | `True` | Non-fatal structural/LP warnings (such as unbalanced fuzzy edge or short sample), zero errors or outages. |
| `DEGRADED` | `False` | Connector outage/fallback (`degraded_fallback`), stale series (`stale`), or empty cache (`connectors_ok == 0`). |
| `FAILED` | `False` | Connector failure (`failed`, `empty`, `schema_drift`), missing required columns, corrupted dates, or unsorted dates. |

## Pyodide & Runtime Portability

Top-level imports in `puremacro.macro_monitor` are restricted to Python's
standard library, `numpy`, and `pandas`. `puremacro.fetch.*`, `puremacro.data`,
`puremacro.lp.*`, `puremacro.reports`, and `puremacro.workbench` are imported
lazily inside workflow functions so importing `puremacro.macro_monitor` never
loads `requests`, `statsmodels`, `linearmodels`, `arch`, `sqlite3`, or `ssl`.
All exported text, CSV, Markdown, HTML, and JSON files use explicit UTF-8
encoding and redact local host filesystem prefixes.
