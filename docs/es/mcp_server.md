# Servidor MCP de PureMacro

> 🇬🇧 [English](../mcp_server.md) · 🇪🇸 Español

`puremacro.mcp_server` es un servidor **Model Context Protocol (MCP)** por `stdio` sin dependencias adicionales que expone flujos computacionales acotados y de solo lectura para econometría, validación DSGE/HANK, escenarios arancelarios multipaís, auditoría de frescura de conectores y verificación matemática.

## Características

- **Cero dependencias extra**: Implementado sobre JSON-RPC 2.0 con biblioteca estándar de Python e importaciones perezosas (`lazy imports`) de `puremacro.*` dentro de cada manejador (respetando el invariante de importación Pyodide).
- **Doble entramado `stdio`**: Detecta y preserva automáticamente JSON delimitado por saltos de línea y cabeceras `Content-Length` en `stdin`/`stdout`.
- **Métodos MCP**: `initialize`, `notifications/initialized`, `ping`, `tools/list` y `tools/call`.
- **Aislamiento de solo lectura**: Acepta tablas inline en JSON (`rows` / `columns`) o archivos `.csv` / `.json` (y cachés `.db` / `.sqlite` / `.sqlite3`) estrictamente dentro de `--allowed-root` (o `PUREMACRO_MCP_ALLOWED_ROOT`). Bloquea recorridos `../`, enlaces simbólicos fuera de la raíz, archivos mayores a 16 MiB y cualquier uso de `eval`/`exec`.
- **Serialización JSON estricta**: Convierte `NaN`, `Inf`, arreglos NumPy, DataFrames de Pandas y dataclasses con `allow_nan=False`.

## Inicio rápido

Verifica la instalación e imprime los esquemas de herramientas disponibles:

```bash
python -m puremacro.mcp_server --list-tools
```

Inicia el servidor `stdio` restringiendo la lectura de archivos a un directorio de datos:

```bash
python -m puremacro.mcp_server --allowed-root /path/to/workspace/data
```

También puedes definir `PUREMACRO_MCP_ALLOWED_ROOT=/path/to/workspace/data` como variable de entorno. Si no se define ninguna raíz permitida, las herramientas aceptan únicamente datos inline en JSON y rechazan rutas locales.

---

## Configuración en clientes MCP

Sustituye `/path/to/puremacro/.venv/bin/python` (o `python`) y `/path/to/data` por las rutas reales de tu entorno.

### Claude Code

Regístralo desde la CLI o añade `.mcp.json` en la raíz de tu proyecto:

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

Agrega la entrada bajo `mcpServers` en tu archivo de configuración MCP de Antigravity (`mcp_config.json`):

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

Agrega `.cursor/mcp.json` en tu espacio de trabajo (o en la configuración global de MCP de Cursor):

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

## Referencia de herramientas

| Herramienta | API de PureMacro | Propósito | Parámetros principales |
|---|---|---|---|
| `preflight_panel` | `puremacro.data.preflight_panel` | Audita esquema de panel/series, duplicados, huecos, valores faltantes, estacionariedad y estacionalidad | `rows` / `columns` / `data_path`, `unit_col`, `time_col`, `variables`, `freq`, `min_obs`, `include_markdown` |
| `lp_hac` | `puremacro.lp.lp_hac` | Proyecciones locales de Jordà (2005) con errores estándar HAC Newey–West Bartlett (`h + 1` rezagos) | `rows` / `columns` / `data_path`, `y`, `x`, `controls`, `time_col`, `horizons` / `horizon`, `n_lags` / `lags`, `ci` / `alpha` |
| `lp_sensitivity_grid` | `puremacro.lp.sensitivity.lp_sensitivity_grid` | Rejilla acotada de sensibilidad LP entre rezagos, horizontes, niveles IC o estimadores permitidos (`lp_hac`, `lp_iv`, `la_lp`, `smooth_lp`, `panel_lp`, `panel_lp_dk`, `lp_did`) | `rows` / `columns` / `data_path`, `y`, `x`, `grid` (máx. 64 especificaciones), `estimator`, `baseline_spec`, `base_kwargs`, `unit_col`, `time_col`, `on_error` |
| `compare_irfs` | `puremacro.reports.compare_irfs` | Compara IRF entre especificaciones (estimaciones puntuales, bandas, picos, acumulados, RMSE y concordancia de signo) | `results` / `data_path`, `baseline`, `horizon_align`, `ci`, `cumulative`, `include_markdown` |
| `score_irf_against_targets` | `puremacro.dsge.validation_scorecard.score_irf_against_targets` | Evalúa IRF del modelo frente a momentos empíricos con $\chi^2$ de Wald (`covariance`/`diagonal`), RMSE, MAE, concordancia de signo y cobertura | `model_irf` / `model_irf_path`, `targets` / `targets_path`, `shock`, `variables`, `horizons`, `weight_mode`, `hank_unit_mode`, `include_markdown` |
| `tariff_scenario_batch` | `puremacro.trade.scenarios.CompositeTariffScenario` y `summarize_scenario_batch` | Compone aranceles simples o compuestos, audita solapamientos (`on_conflict`), resuelve equilibrio general y calcula EV/CV hicksiano | `scenarios` (máx. 16), `model_source` (`oecd_icio` / `synthetic` / `file`), `data_path`, `sigma`, `method`, `resolve_only`, `include_markdown` |
| `connector_freshness_report` | `puremacro.fetch.freshness.connector_freshness_report` | Audita procedencia, identidades de dominio, revisiones y frescura de conectores (`openWorldHint: true` en modo `live`) | `mode` (`cache` / `recorded` / `live`), `sources`, `as_of`, `db_path`, `max_requests` (1–5), `min_interval_seconds`, `timeout_seconds`, `lang`, `include_markdown` |
| `validation_scorecard` | `puremacro.validation.scorecard` | Lista o ejecuta casos matemáticos autocontenidos de la galería de validación por subsistema o identificador | `subsystem`, `case_ids`, `list_only` |
| `hank_fiscal_closures` | `puremacro.models.hank_fiscal.compare_fiscal_closures` | Compara transiciones fiscales HANK no lineales en espacio de secuencias bajo presupuesto equilibrado y regla de deuda (`proportional` vs `lump_sum`) | `horizon` (1–120), `n_a` (20–150), `a_max`, `phi_B`, `phi_G`, `g_ss`, `tol`, `include_unfinanced`, `shock_seq`, `include_markdown` |
| `cce_mg_panel_lp` | `puremacro.lp.cce.cce_panel_lp` | Estima proyecciones locales de panel con Efectos Correlacionados Comunes (CCE Mean Group o Pooled) controlando factores no observados | `rows` / `columns` / `data_path`, `y`, `x`, `unit_col`, `time_col`, `controls`, `estimator` (`mean_group` / `pooled`), `horizons` / `horizon`, `n_lags` / `lags`, `min_obs`, `ci` / `alpha`, `include_markdown` |
| `identification_robust_set` | `puremacro.structural.robust` | Evalúa pruebas puntuales de restricción de momentos, inversión exacta 1D para curvas cuadráticas o mallas muestreadas de momentos | `targets` (`values`, `covariance`, `labels`, `units`), `model_moments`, `covariance`, `curve`, `domain`, `grid`, `grid_moments`, `alpha`, `calibration`, `include_markdown` |
| `run_workbench_dossier` | `puremacro.workbench.run_empirical_structural_workbench` | Ejecuta el flujo unificado de preflight, LP/sensibilidad, comparación IRF, scorecard e inferencia robusta devolviendo el expediente | `rows` / `columns` / `data_path`, `y`, `x`, `unit_col`, `time_col`, `controls`, `horizons`, `lags`, `ci`, `estimator`, `model_irf`, `targets`, `moment_test_targets`, `model_moments`, `moment_curve`, `moment_domain`, `moment_grid`, `lang`, `include_markdown` |

### Límites operativos

Para garantizar respuestas rápidas y acotadas en sesiones interactivas:

- **Tablas de entrada**: máximo 50,000 filas × 200 columnas; lectura de archivos limitada a 16 MiB.
- **Proyecciones locales y CCE**: horizontes en `[0, 120]`, rezagos en `[0, 40]` y hasta 64 combinaciones por rejilla de sensibilidad.
- **Transiciones fiscales HANK**: horizonte acotado a `120` (`[1, 120]`) y tamaño de malla de activos `n_a` acotado a `150` (`[20, 150]`).
- **Momentos robustos a la identificación**: dimensión máxima del vector de momentos igual a `64` y mallas muestreadas limitadas a `250` nodos.
- **Lotes arancelarios**: hasta 16 escenarios por llamada (profundidad máxima de composición igual a 5).
- **Sondeos `live` de conectores**: máximo 5 solicitudes por llamada (`max_requests` en `[1, 5]`); usa `mode="recorded"` o `mode="cache"` para auditorías sin red.

## API en Python

El registro y despachador también pueden usarse directamente desde Python:

```python
from puremacro.mcp_server import PUREMACRO_MCP_TOOLS, call_tool, list_tools, serve_stdio

specs = list_tools()
score = call_tool("validation_scorecard", {"subsystem": "unit_root", "list_only": True})
```
