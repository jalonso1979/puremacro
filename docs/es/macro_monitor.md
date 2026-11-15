# Monitor Automatizado de Actualidad Macro y LatAm

> 🇬🇧 [English](../macro_monitor.md) · 🇪🇸 Español

`puremacro.macro_monitor` integra los diagnósticos de procedencia de datos,
verificación estructural (`preflight`), *nowcasting* y proyecciones locales de
PureMacro en un flujo reproducible, determinista fuera de línea (*offline-first*),
con fallo seguro (*fail-closed*) y exportación bilingüe (`en` / `es`).

## Etapas del Flujo

1. **Auditoría de Actualidad y Procedencia de Conectores**: ejecuta
   `puremacro.fetch.freshness.connector_freshness_report` (`mode="recorded"` por
   defecto para ejecución determinista sin red, `"cache"` o `"live"` con control
   de cuota `RateBudget`) y verifica los metadatos en
   `puremacro.fetch.registry`. Cualquier falla, respuesta vacía, rezago o uso de
   respaldo (`failed`, `empty`, `schema_drift`, `degraded_fallback`, `stale`,
   `not_cached`) se expone explícitamente en `overall_status`, `headline()` y
   `warnings`, sin reemplazo sintético silencioso.
2. **Ingesta del Panel y Verificación Estructural (`Preflight`)**: carga un
   `DataFrame` del usuario, un archivo CSV/JSON, un constructor registrado en
   `fetch.registry` o el panel incluido OECD QNA (`MEX`, `USA`, `DEU`, `ESP`,
   `ITA`, `JPN`). Valida columnas requeridas, fechas inválidas/corruptas y orden
   cronológico estricto por país antes de llamar a `puremacro.data.preflight_panel`,
   registrando el resumen SHA-256 de la entrada y el rango muestral.
3. **Indicadores por País, Nowcast AR(1) y Sensibilidad LP**: calcula la fecha
   más reciente por país, crecimiento trimestral (QoQ) e interanual (YoY),
   volatilidad móvil, banderas *z-score* (`OK`, `Z_ALERT_HIGH`, `Z_ALERT_LOW`,
   `SHORT_SAMPLE`, `MISSING`), un diagnóstico de *nowcast* AR(1) en NumPy puro y
   una malla acotada `puremacro.lp.sensitivity.lp_sensitivity_grid` (`panel_lp`
   / `panel_lp_dk` para paneles o `lp_hac` / `la_lp` para series individuales)
   con avisos explícitos ante muestras cortas o inestabilidad numérica.
4. **Exportación de Paquete UTF-8 Bilingüe**: `report.write_bundle(output_dir)`
   genera seis archivos autocontenidos en UTF-8 (`monitor.md`, `monitor.html`,
   `monitor.json`, `freshness.csv`, `latest_indicators.csv` y `manifest.json`)
   con sumas de verificación SHA-256, JSON estricto RFC 8259 (`NaN`/`Infinity`
   convertidos a `null`) y cero filtración de rutas locales del equipo.

## Inicio Rápido (Python)

```python
from puremacro.macro_monitor import MacroMonitorConfig, run_macro_monitor

# Ejecución determinista fuera de línea
report = run_macro_monitor(mode="recorded", lang="es")
print(report.headline())
print(report.latest_indicators_df.head())

# Exportar expediente bilingüe + manifiesto SHA-256
archivos = report.write_bundle("artifacts/macro_monitor_es", lang="es")
report.write_bundle("artifacts/macro_monitor_en", lang="en")
```

Ejecución con un panel LatAm propio y variables de transmisión explícitas:

```python
config = MacroMonitorConfig(
    mode="recorded",
    sources=("vintage.banxico", "vintage.bcb", "oecd.qna_panel"),
    y="gdp_growth",
    x="policy_rate",
    controls=("inflation",),
    horizons=(0, 1, 2, 3, 4),
    lp_lags=(1, 2),
    lang="es",
    fail_on_outage=True,
)
report = run_macro_monitor("latam_panel.csv", config=config)
```

## Uso desde CLI

```bash
python -m puremacro.macro_monitor \
  --mode recorded \
  --lang es \
  --output-dir artifacts/macro_monitor_es \
  --format text
```

Códigos de salida:

- `0`: flujo completado con `overall_status` en `{"OK", "WARN"}`.
- `1`: `overall_status == "FAILED"` (errores estructurales, fechas corruptas o
  desordenadas, o fallo de conectores), bandera `--fail-on-outage` ante estados
  `DEGRADED`, o argumentos inválidos.

## Referencia de Configuración

| Parámetro / Opción | Valor por defecto | Descripción |
| --- | --- | --- |
| `mode` | `"recorded"` | Modo de conectores: `"recorded"`, `"cache"` o `"live"`. |
| `sources` | `None` | Subconjunto de conectores (por defecto la cohorte de 9 fuentes). |
| `as_of` | `None` | Fecha de corte ISO para auditoría de rezago (*staleness*). |
| `db_path` | `None` | Ruta a base SQLite de caché/telemetría (nunca se exporta al reporte). |
| `max_requests` / `min_interval_seconds` / `timeout_seconds` | `5` / `0.5` / `20.0` | Límites de `RateBudget` aplicados a consultas en vivo. |
| `panel_builder` | `None` | Identificador opcional de fuente en `fetch.registry`. |
| `unit_col` / `time_col` | `None` | Nombres de columnas de unidad/país y fecha (autodetección si se omiten). |
| `y` / `x` / `controls` | Auto / `()` | Variables dependiente, choque y controles para la malla LP. |
| `horizons` / `lp_lags` | `(0, 1, 2, 3, 4)` / `(1, 2)` | Horizontes y rezagos evaluados en `lp_sensitivity_grid`. |
| `vol_window` / `zscore_threshold` | `8` / `2.0` | Ventana móvil y umbral *z-score* para banderas recientes. |
| `short_sample_min_obs` | `24` | Longitud mínima antes de emitir avisos de muestra corta. |
| `run_lp_sensitivity` | `True` | Use `False` (`--no-lp`) para omitir la malla LP. |
| `lang` | `"en"` | Idioma del reporte (`"en"` o `"es"`). |
| `fail_on_outage` | `False` | Devuelve código `1` en CLI si existe caída de conectores o estado `DEGRADED`. |

## Semántica Fail-Closed de `overall_status`

| `overall_status` | `passed` | Condición |
| --- | --- | --- |
| `OK` | `True` | Todos los conectores sanos, `preflight` limpio y sin advertencias ni errores. |
| `WARN` | `True` | Avisos no fatales (ej. panel desbalanceado o muestra corta), cero errores o caídas. |
| `DEGRADED` | `False` | Respaldo degradado (`degraded_fallback`), fuente rezagada (`stale`) o caché vacío (`connectors_ok == 0`). |
| `FAILED` | `False` | Falla de conector (`failed`, `empty`, `schema_drift`), columnas faltantes o fechas corruptas/desordenadas. |

## Portabilidad Pyodide y UTF-8

Las importaciones de nivel superior en `puremacro.macro_monitor` se limitan a la
biblioteca estándar, `numpy` y `pandas`. Los submódulos `puremacro.fetch.*`,
`puremacro.data`, `puremacro.lp.*`, `puremacro.reports` y `puremacro.workbench`
se importan de forma diferida dentro de funciones para que `import
puremacro.macro_monitor` jamás cargue `requests`, `statsmodels`, `linearmodels`,
`arch`, `sqlite3` ni `ssl`. Todos los archivos generados emplean codificación
UTF-8 explícita y sanean rutas locales del anfitrión.
