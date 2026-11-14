# PureMacro MCP Server

> 🇬🇧 English · 🇪🇸 [Español](es/mcp_server.md)

`puremacro.mcp_server` is a zero-extra-dependency **Model Context Protocol (MCP)** stdio server exposing bounded, read-only computational workflows across econometrics, DSGE/HANK validation, multi-country tariff analysis, connector freshness audits, and library verification.

## Features

- **Zero extra dependencies**: Runs on the Python standard library JSON-RPC 2.0 transport with lazy `puremacro.*` imports inside tool handlers (preserving PureMacro's Pyodide top-level import invariant).
- **Dual stdio framing**: Auto-detects and preserves both newline-delimited JSON and `Content-Length` headers over `stdin`/`stdout`.
- **MCP Methods**: `initialize`, `notifications/initialized`, `ping`, `tools/list`, and `tools/call`.
- **Read-only sandboxing**: Accepts inline JSON tables (`rows` / `columns`) or read-only `.csv` / `.json` files (and `.db` / `.sqlite` / `.sqlite3` caches) strictly inside an explicit `--allowed-root` (or `PUREMACRO_MCP_ALLOWED_ROOT`). Path traversal, symlinks escaping the root, oversized files (> 16 MiB), and code evaluation (`eval`/`exec`) are disabled.
- **Strict JSON output**: Sanitizes `NaN`, `Inf`, NumPy arrays, Pandas DataFrames, and dataclasses with `allow_nan=False`.

## Quick Start

Verify installation and print tool schemas:

```bash
python -m puremacro.mcp_server --list-tools
```

Start the stdio server with read-only file access restricted to a data directory:

```bash
python -m puremacro.mcp_server --allowed-root /path/to/workspace/data
```

You can also pass `PUREMACRO_MCP_ALLOWED_ROOT=/path/to/workspace/data` in the environment. If neither `--allowed-root` nor `PUREMACRO_MCP_ALLOWED_ROOT` is provided, tools accept inline JSON payloads only and reject local file paths.

---

## Client Configuration

Replace `/path/to/puremacro/.venv/bin/python` (or `python`) and `/path/to/data` with your environment paths.

### Claude Code

Register via the CLI or add `.mcp.json` to your project root:

```bash
claude mcp add puremacro -- /path/to/puremacro/.venv/bin/python -m puremacro.mcp_server --allowed-root /path/to/data
```

```json
{
  "mcpServers": {
    "puremacro": {
      "command": "/path/to/puremacro/.venv/bin/python",
      "args": ["-m", "puremacro.mcp_server", "--allowed-root", "/path/to/data"],
      "env": {
        "PYTHONPATH": "/path/to/puremacro"
      }
    }
  }
}
```

### Antigravity

Add a server entry under `mcpServers` in your Antigravity MCP configuration (`mcp_config.json`):

```json
{
  "mcpServers": {
    "puremacro": {
      "command": "/path/to/puremacro/.venv/bin/python",
      "args": ["-m", "puremacro.mcp_server", "--allowed-root", "/path/to/data"],
      "env": {
        "PYTHONPATH": "/path/to/puremacro"
      }
    }
  }
}
```

### Cursor

Add `.cursor/mcp.json` in your workspace (or global Cursor MCP settings):

```json
{
  "mcpServers": {
    "puremacro": {
      "command": "/path/to/puremacro/.venv/bin/python",
      "args": ["-m", "puremacro.mcp_server", "--allowed-root", "/path/to/data"],
      "env": {
        "PYTHONPATH": "/path/to/puremacro"
      }
    }
  }
}
```

---

## Tool Reference

| Tool | PureMacro API | Purpose | Key Parameters |
|---|---|---|---|
| `preflight_panel` | `puremacro.data.preflight_panel` | Audit panel/time-series schema, duplicates, gaps, missingness, stationarity, and seasonality | `rows` / `columns` / `data_path`, `unit_col`, `time_col`, `variables`, `freq`, `min_obs`, `include_markdown` |
| `lp_hac` | `puremacro.lp.lp_hac` | Jordà (2005) local projections with Newey–West Bartlett HAC standard errors (`h + 1` lags) | `rows` / `columns` / `data_path`, `y`, `x`, `controls`, `time_col`, `horizons` / `horizon`, `n_lags` / `lags`, `ci` / `alpha` |
| `lp_sensitivity_grid` | `puremacro.lp.sensitivity.lp_sensitivity_grid` | Bounded LP specification sensitivity across lags, horizons, CI levels, or allowlisted estimators (`lp_hac`, `lp_iv`, `la_lp`, `smooth_lp`, `panel_lp`, `panel_lp_dk`, `lp_did`) | `rows` / `columns` / `data_path`, `y`, `x`, `grid` (max 64 specs), `estimator`, `baseline_spec`, `base_kwargs`, `unit_col`, `time_col`, `on_error` |
| `compare_irfs` | `puremacro.reports.compare_irfs` | Compare IRFs across specifications (point paths, bands, peaks, cumulative sums, RMSE, sign agreement) | `results` / `data_path`, `baseline`, `horizon_align`, `ci`, `cumulative`, `include_markdown` |
| `score_irf_against_targets` | `puremacro.dsge.validation_scorecard.score_irf_against_targets` | Score model IRFs vs empirical targets with Wald $\chi^2$ (`covariance`/`diagonal`), RMSE, MAE, sign concordance, band coverage | `model_irf` / `model_irf_path`, `targets` / `targets_path`, `shock`, `variables`, `horizons`, `weight_mode`, `hank_unit_mode`, `include_markdown` |
| `tariff_scenario_batch` | `puremacro.trade.scenarios.CompositeTariffScenario` & `summarize_scenario_batch` | Compose single/nested tariffs, audit overlaps (`on_conflict`), solve general equilibrium, and report Hicksian EV/CV | `scenarios` (max 16), `model_source` (`oecd_icio` / `synthetic` / `file`), `data_path`, `sigma`, `method`, `resolve_only`, `include_markdown` |
| `connector_freshness_report` | `puremacro.fetch.freshness.connector_freshness_report` | Audit connector cohort provenance, domain identities, revisions, and staleness (`openWorldHint: true` in `live` mode) | `mode` (`cache` / `recorded` / `live`), `sources`, `as_of`, `db_path`, `max_requests` (1–5), `min_interval_seconds`, `timeout_seconds`, `lang`, `include_markdown` |
| `validation_scorecard` | `puremacro.validation.scorecard` | List or run self-contained mathematical validation gallery cases by subsystem or case ID | `subsystem`, `case_ids`, `list_only` |

### Resource Caps

To keep interactive tool calls bounded and responsive:

- **Maximum table input**: 50,000 rows × 200 columns; file reads capped at 16 MiB.
- **Local Projections**: horizons in `[0, 120]`, lag order in `[0, 40]`, sensitivity grids capped at 64 combinations.
- **Trade equilibrium batches**: up to 16 scenarios per call (composite nesting depth up to 5).
- **Live connector probes**: capped at 5 requests per call (`max_requests` in `[1, 5]`); use `mode="recorded"` or `mode="cache"` for offline checks.

## Python API

The registry and dispatcher can also be embedded directly in Python:

```python
from puremacro.mcp_server import PUREMACRO_MCP_TOOLS, call_tool, list_tools, serve_stdio

specs = list_tools()
score = call_tool("validation_scorecard", {"subsystem": "unit_root", "list_only": True})
```
