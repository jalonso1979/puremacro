> 🇬🇧 [English](../CACHE_DB.md) · 🇪🇸 Español

# Base de datos de caché

> Disponible a partir de puremacro **0.66.0**. Reemplaza la caché de archivos
> planos `~/.cache/puremacro/http/*.bin + *.json` de las versiones 0.65.0 e
> inferiores.

## Ubicación

Archivo SQLite único en `~/.cache/puremacro/cache.db` (modificable mediante
`$PUREMACRO_HTTP_CACHE_DIR`). Si la variable de entorno termina en `.db`, se
utiliza esa ruta de forma literal; en caso contrario se interpreta como
directorio y la base de datos queda en `<dir>/cache.db`.

## Esquema

```sql
CREATE TABLE http_cache (
    key            TEXT PRIMARY KEY,    -- sha256(url) hex
    url            TEXT NOT NULL,
    fetched_at     INTEGER NOT NULL,    -- unix epoch seconds
    content_type   TEXT,
    body           BLOB NOT NULL
);

CREATE TABLE alfred_vintages (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD
    value            REAL,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);

CREATE TABLE connector_events (
    ts             INTEGER NOT NULL,    -- unix epoch seconds
    source         TEXT NOT NULL,       -- connector name
    outcome        TEXT NOT NULL,
    fallback_used  TEXT NOT NULL
);

CREATE TABLE realtime_vintages (
    provider         TEXT NOT NULL,     -- 'banxico' / 'inegi' / 'bcb' / 'bcch'
    country          TEXT NOT NULL,     -- ISO-3, upper case
    series_id        TEXT NOT NULL,     -- the provider's own identifier
    observation_date TEXT NOT NULL,     -- ISO YYYY-MM-DD, the reference period
    vintage_date     TEXT NOT NULL,     -- ISO YYYY-MM-DD, the SNAPSHOT date
    value            REAL,
    PRIMARY KEY (provider, country, series_id, observation_date, vintage_date)
);

CREATE TABLE schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
```

El modo de diario WAL (`PRAGMA journal_mode=WAL`) está habilitado para que
varios cuadernos que accedan a la misma base de datos no se bloqueen
mutuamente en las operaciones de escritura.

La inicialización del esquema es aditiva (`CREATE TABLE IF NOT EXISTS` +
`INSERT OR IGNORE`), de modo que un archivo de caché escrito por una versión
anterior incorpora las tablas nuevas al abrirse por primera vez y conserva sus
filas. La tabla `schema_version` se inicializa con una fila por tabla
(`http_cache`, `alfred_vintages`, `connector_events`, `realtime_vintages`,
todas en la versión 1).

## Instantáneas de tiempo real (`realtime_vintages`)

> Incorporada en puremacro **3.4.0**.

Banxico, INEGI, BCB y BCCh publican únicamente la edición vigente en el momento
de la consulta — sobrescriben en el sitio y no incluyen campo de añada —, de
manera que el único historial de revisiones disponible para ellos es el que
esta tabla acumula localmente. Cada descarga se almacena con la marca de su
**fecha de instantánea**, y una llamada posterior devuelve cada instantánea
almacenada como una añada. Dos capturas del mismo día son, por tanto, una sola
añada y no dos: la clave primaria termina en `vintage_date` y la escritura es
`INSERT OR REPLACE`, de modo que una nueva descarga el mismo día sobrescribe en
lugar de duplicar.

```python
import pandas as pd
import puremacro._cache_db as db

snap = pd.DataFrame({
    "provider": ["bcb"] * 3,
    "country": ["BRA"] * 3,
    "series_id": ["432"] * 3,
    "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
    "vintage": pd.to_datetime(["2026-03-15"] * 3),
    "value": [11.25, 11.25, 11.00],
})

db.store_realtime_vintages(snap)                       # -> 3 rows written
db.query_realtime_vintages("bcb", "BRA", "432")        # every stored snapshot
db.query_realtime_vintages("bcb", "BRA", "432",        # as-of cutoff:
                           vintage_date="2026-03-01")  # keeps vintage_date <= cutoff
db.record_connector_event("bcb", "success", "none")    # fetch telemetry
```

- `store_realtime_vintages(df)` admite los nombres de columna `date` /
  `vintage` o bien `observation_date` / `vintage_date`, convierte el país a
  mayúsculas, normaliza ambas fechas y el valor, descarta las filas que no
  puede interpretar y devuelve el número de filas escritas. La ausencia de una
  columna obligatoria lanza `ValueError`.
- `query_realtime_vintages(provider, country, series_id, *, vintage_date=None)`
  devuelve `[date, vintage, value, provider, country, series_id]` ordenado por
  `(observation_date, vintage_date)`, o un marco vacío con esas columnas. El
  argumento `vintage_date` es un **corte «a fecha de»**, no una coincidencia
  exacta: conserva las instantáneas tomadas en esa fecha o antes, que es lo que
  reconstruye el conjunto de información disponible para un responsable de
  política económica.
- `record_connector_event(source, outcome, fallback_used)` escribe una fila de
  telemetría de descarga; véase
  [`docs/es/CONNECTOR_HEALTH.md`](CONNECTOR_HEALTH.md).

El almacenamiento es una comodidad, nunca el objetivo: una instantánea que no
puede escribirse se devuelve igualmente al invocador con un `UserWarning`, y una
caché que no puede leerse se reporta y se trata como vacía.

## Migración desde 0.65.0

El módulo de caché HTTP ejecuta la migración de forma diferida en la primera
operación de lectura o escritura tras la actualización: si existen archivos
`cache_dir/*.bin` y la tabla `http_cache` está vacía, las entradas se insertan
en la base de datos (sin eliminar los originales) y un `UserWarning` remite a
la herramienta de línea de comandos:

```bash
python tools/cache_migrate.py              # dry-run; report count
python tools/cache_migrate.py --apply      # migrate
python tools/cache_migrate.py --apply --rm # migrate + delete originals
```

La migración es idempotente: volver a ejecutarla no tiene ningún efecto.

## Introspección

```python
import puremacro.cache as C
import pandas as pd

C.http_list_urls()                                 # sorted list of cached URLs
C.http_cache_size_bytes()                          # total body bytes
C.http_cache_clear()                               # clear ALL entries; returns count
C.http_cache_clear(older_than=pd.Timedelta(days=30))  # clear stale only
```

Las eliminaciones masivas (más de 1000 filas) emiten automáticamente un
`VACUUM` para que el archivo en disco se reduzca de tamaño efectivamente.

## Semántica de fallos

`cache_read` / `cache_write` no deben nunca propagar excepciones al
invocador (este es un contrato esencial desde la versión 0.65.0). Los fallos
en la base de datos emiten un `UserWarning` y degradan el comportamiento de
forma controlada: `cache_read` devuelve `None` y `cache_write` no ejecuta
ninguna acción. Un cuaderno de investigación que consolide 30 fuentes
simplemente operará más lento (sin caché); nunca se interrumpirá con un error.

## Pyodide

`sqlite3` es parte de la biblioteca estándar de Python y está disponible en
todos los entornos de ejecución admitidos, incluido Pyodide. El archivo de
caché reside en el sistema de archivos virtual de Pyodide; para persistir la
caché entre recargas de página, el usuario debe montar IDBFS o un mecanismo
equivalente.
