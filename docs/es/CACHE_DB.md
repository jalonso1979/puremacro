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

CREATE TABLE schema_version (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);
```

El modo de diario WAL (`PRAGMA journal_mode=WAL`) está habilitado para que
varios cuadernos que accedan a la misma base de datos no se bloqueen
mutuamente en las operaciones de escritura.

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
