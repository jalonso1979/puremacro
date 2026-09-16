> 🇬🇧 [English](../CONNECTOR_HEALTH.md) · 🇪🇸 Español

# Estado de conectores

> Disponible a partir de puremacro **0.67.0**.

`puremacro.narrative.sources._telemetry.connector_health()` agrega
eventos por descarga registrados en la tabla SQLite `connector_events` (una fila
por intento de descarga en cada conector participante) y devuelve un DataFrame
que indica qué conectores están operativos, degradados o caídos.

## Inicio rápido

```python
from puremacro.narrative.sources._telemetry import connector_health
import pandas as pd

connector_health(window=pd.Timedelta(days=7))
#         source  n_total  n_success  success_rate  n_fallback  fallback_rate           last_seen
# 0    eu_eurlex      142        119         0.838          23          0.162  2026-05-25 14:22:03
# 1 eu_parliament       89         72         0.809          17          0.191  2026-05-24 09:11:47
# 2       us_cbo       45         45         1.000           0          0.000  2026-05-23 18:05:12
# 3          rba       28         26         0.929           2          0.071  2026-05-26 02:33:01
# 4          bok       31         28         0.903           3          0.097  2026-05-26 03:14:55
```

Un `fallback_rate > 0` indica que el endpoint primario falló y la solicitud fue
atendida mediante un mecanismo de respaldo (Wayback / Playwright). Un
`success_rate < 1` significa que algunas solicitudes fallaron completamente —
las filas de `connector_events` muestran el resultado exacto de cada una
(timeout, 404, ssl_fail, wayback_no_snapshot,
playwright_unavailable, parser_schema_mismatch, other_network_error).

## Filtrado

```python
connector_health(window=pd.Timedelta(days=30))                   # last 30 days
connector_health(sources=["eu_eurlex", "rba"])                   # subset of connectors
connector_health(window=pd.Timedelta(hours=1), sources=["bok"])  # combined
```

## Esquema de eventos

```sql
CREATE TABLE connector_events (
    ts             INTEGER NOT NULL,    -- unix epoch seconds
    source         TEXT NOT NULL,       -- 'eu_eurlex', 'rba', 'banxico', ...
    outcome        TEXT NOT NULL,       -- success / fallback / schema_drift /
                                        -- 404 / timeout / ssl_fail /
                                        -- server_5xx / wayback_no_snapshot /
                                        -- playwright_unavailable /
                                        -- parser_schema_mismatch / other_network_error
    fallback_used  TEXT NOT NULL        -- live / wayback / playwright / none /
                                        -- sqlite_cache / warning_emitted / ignored
);
CREATE INDEX connector_events_ts_source_idx
    ON connector_events(ts, source);
```

Reside en `~/.cache/puremacro/cache.db` (o `$PUREMACRO_HTTP_CACHE_DIR`).

## Qué se registra (alcance del Slice B)

- **`fetch_with_fallback`** (los 7 conectores con respaldo activo): un
  evento por cada intento de etapa. Conectores:
  `eu_eurlex`, `eu_parliament`, `us_cbo`, `rba`, `bok`, `riksbank`, `sarb`.
- **Envoltorios `iter_<source>` para los 8 conectores del Slice-A con
  verificación de esquema**: un evento por cada captura de `ParserSchemaMismatchError`
  (`outcome="parser_schema_mismatch"`, `fallback_used="none"`).
- **Otros conectores**: sin eventos en el Slice B. No aparecerán en
  `connector_health()` hasta que se incorporen (típicamente adoptando
  `fetch_with_fallback(policy=("live",))` o invocando `log_event(...)`
  directamente).

## Conectores de instantánea de tiempo real (3.4.0)

Los cuatro conectores latinoamericanos de instantánea — `banxico`, `inegi`,
`bcb` y `bcch`, documentados en
[`docs/es/real_time_data.md`](real_time_data.md) — registran eventos bajo su
propio nombre de proveedor e incorporan tres resultados al vocabulario
anterior:

| `outcome` | `fallback_used` | Qué ocurrió |
|---|---|---|
| `success` | `none` | la respuesta en vivo se interpretó correctamente y la instantánea se devolvió (y se almacenó, salvo que la escritura fallara) |
| `fallback` | `sqlite_cache` | la descarga, el análisis o el canario de esquema fallaron, o la respuesta venía vacía, y se sirvieron las instantáneas almacenadas localmente |
| `schema_drift` | `none` | deriva bajo `on_drift="raise"`: se lanzó un `SchemaDriftError`, que la capa de descarga convierte después en una fila `fallback` si dispone de caché |
| `schema_drift` | `warning_emitted` | deriva bajo `on_drift="warn"`: se emitió un `SchemaDriftWarning` y el analizador continuó |
| `schema_drift` | `ignored` | deriva bajo `on_drift="ignore"` |

Se escribe una fila `schema_drift` **sea cual sea** la política `on_drift`, de
modo que `connector_health()` detecta la deriva del servicio de origen incluso
en las ejecuciones donde un respaldo o una política permisiva la ocultaron al
invocador. Ese es precisamente el propósito del canario: un proveedor que
renombró silenciosamente una clave JSON aparece como un recuento de derivas en
aumento y no como una serie que se acortó de forma inexplicable.

La telemetría nunca interrumpe la descarga que la emitió: una escritura fallida
se reporta como `UserWarning` y los datos se devuelven igualmente.

## Interruptor de emergencia

```bash
export PUREMACRO_NARRATIVE_TELEMETRY=0    # disable all event logging
```

`log_event` pasa a ser una operación nula. `connector_health` sigue leyendo
las filas insertadas antes de que se estableciera la variable de entorno. Útil
para ejecuciones sin acceso a red, entornos de integración continua con
reproducibilidad estricta, o usuarios que prefieren no permitir escrituras en
la base de datos desde el código de descarga.

## Semántica de fallos

La telemetría nunca interrumpe una descarga. Si la base de datos está bloqueada,
inaccesible o el disco está lleno, `log_event` emite un `UserWarning` y retorna
silenciosamente. `connector_health` hace lo propio y devuelve un DataFrame vacío
con las columnas esperadas.

## Pyodide

`sqlite3` forma parte de la biblioteca estándar de Python — disponible en todos
los entornos donde se ejecuta puremacro.
El registro de eventos reside en el sistema de archivos virtual de Pyodide; la
persistencia entre recargas de página requiere un montaje IDBFS (misma
consideración que el resto de la base de datos de caché — véase [docs/es/CACHE_DB.md](CACHE_DB.md)).
