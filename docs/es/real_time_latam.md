> 🇬🇧 [English](../real_time_latam.md) · 🇪🇸 Español

# Conectores en tiempo real para América Latina, canarios de esquema y cartuchos .pmz

`puremacro.fetch.realtime` incorpora cuatro conectores para bancos centrales y agencias estadísticas de América Latina — `banxico` (Banco de México, SIE), `inegi` (INEGI, Banco de Indicadores / BIE), `bcb` (Banco Central do Brasil, SGS) y `bcch` (Banco Central de Chile, SIETE) — junto con tres piezas de infraestructura de las que dependen: un **canario de esquema** (`puremacro.fetch.realtime.canary`) que comprueba el envoltorio JSON de cada proveedor antes de analizarlo, un **almacén de instantáneas** (la tabla `realtime_vintages` de la base de datos de caché SQLite, `puremacro._cache_db`) que convierte capturas repetidas en añadas (*vintages*, es decir, ediciones), y un **cartucho portátil `.pmz`** (`puremacro.fetch.realtime.cartridge`, construido sobre `puremacro.pocket`) que traslada un `VintagePanel` a una máquina sin red y sin credenciales. Las credenciales se resuelven mediante `puremacro.credentials` (servicios `banxico`, `inegi` y `bcch`; `bcb` no requiere clave). Los conectores se inscriben en el registro de proveedores, de modo que el punto de entrada general `vintage_panel(...)` de [real_time_data.md](real_time_data.md) los alcanza a través de `providers=`.

Un hecho condiciona todo lo que sigue en esta página, así que va primero y no en una nota al pie: **ninguna de estas cuatro fuentes publica un archivo histórico de ediciones pasadas**. puremacro fabrica las añadas mediante instantáneas.

---

## 1. Marco teórico y algorítmico

### 1.1 Añadas por instantánea, no por archivo histórico

ALFRED, la base de datos en tiempo real del Bundesbank y el archivo de revisiones STES de la OCDE conservan cada edición que un instituto de estadística haya publicado alguna vez. El SIE de Banxico, el BIE del INEGI, el SGS del BCB y SIETE del BCCh no lo hacen: cada solicitud devuelve la serie **vigente**, sobrescrita in situ, y la respuesta no lleva ningún campo de añada. `puremacro.fetch.realtime.banxico.MEXICO_VINTAGE_NOTE` deja constancia de esa determinación para México, y lo mismo vale para Brasil y Chile.

Por tanto, los cuatro conectores construyen las añadas de la única manera posible. Cada descarga exitosa se analiza en una tabla ordenada (*tidy*) `[date, vintage, value]` cuya columna `vintage` es la **fecha de instantánea**, el día en que *esta máquina* capturó la serie — `pd.Timestamp.now().normalize()`, es decir, la medianoche local de hoy, salvo que se pase `vintage_date=` — y las filas se escriben en la tabla `realtime_vintages` de la caché SQLite local, con clave `(provider, country, series_id, observation_date, vintage_date)`. Escribiendo $y_t^{(v)}$ para el valor del periodo de referencia $t$ que lleva la instantánea capturada el día $v$, una descarga aporta el conjunto

$$S_v = \{(t, v, y_t^{(v)}) : t \in \text{payload}_v\},$$

y el archivo histórico local tras capturas en los días $v_1 < v_2 < \dots < v_K$ es $\bigcup_k S_{v_k}$. Las consecuencias son inmediatas y el resto de la página vuelve a ellas una y otra vez:

- Una sola descarga produce exactamente **una** edición. `coverage()` informa `n_vintages == 1` y ningún contraste de revisiones es posible.
- El historial de revisiones se acumula **solo entre capturas en días distintos**. Una nueva descarga el mismo día coincide con la misma clave primaria y la sobrescribe (`INSERT OR REPLACE`).
- La fecha de la añada es cuándo capturó *usted*, no cuándo publicó la institución. `VINTAGE_SEMANTICS["banxico"]` y sus tres hermanas dicen exactamente eso, y `VintagePanel.vintage_semantics()` devuelve el texto de cada proveedor presente.
- Para las ediciones históricas del PIB trimestral de México use el proveedor `oecd_stes`, que archiva la serie del INEGI con 329 ediciones mensuales desde 1999-02 (véase [real_time_data.md](real_time_data.md)).

### 1.2 De las instantáneas a las revisiones

Dado el archivo histórico, los objetos con los que trabaja el resto de `puremacro.fetch.realtime` son las construcciones estándar de las bases de datos en tiempo real de Croushore y Stark (2001). El **triángulo de revisiones** coloca las fechas de referencia en las filas y las fechas de captura en las columnas, $T[t, v] = y_t^{(v)}$, con `NaN` allí donde el periodo $t$ aún no figuraba en la respuesta del día $v$. El **corte transversal *as-of*** (a una fecha dada) del día $v^*$ conserva, para cada $t$, la última captura no posterior a $v^*$:

$$y_t^{\,\text{as of } v^*} = y_t^{(v^\dagger)}, \qquad v^\dagger = \max\{v \le v^* : t \in \text{payload}_v\}.$$

Para una serie catalogada en **niveles** o como **índice**, la transformación aplicada dentro de cada columna de añada antes de cualquier contraste es la diferencia logarítmica en porcentaje, $x_t^{(v)} = 100\,[\ln y_t^{(v)} - \ln y_{t-1}^{(v)}]$; las series catalogadas como **tasa** o **tasa de crecimiento** se usan tal cual (`UNITS_TRANSFORM`). Nunca se mezclan numeradores y denominadores de añadas distintas. La estimación preliminar de $t$ es su primera captura, la final es la más reciente, y la revisión es $r_t = x_t^{f} - x_t^{p}$ — siempre que la primera captura sea observable. `revision_frame` descarta todo periodo de referencia que terminó antes de la primera columna de añada ($t < v_1$), porque la edición capturada en $v_1$ ya lleva una cifra revisada para él (`require_observable_first=True`, que `VintagePanel.revisions` y `news_or_noise` no exponen; llame a `puremacro.vintages.revision_frame(panel.long(...), require_observable_first=False)` para la lectura ingenua). Con añadas por instantánea esto significa que la muestra de revisiones comienza en la fecha de la primera captura, no al inicio de la serie: todo lo capturado el primer día es historia que llegó ya revisada.

El par de contrastes de Mankiw y Shapiro (1986) son las dos regresiones

$$r_t = \alpha_p + \beta_p\, x_t^{p} + \varepsilon_{p,t}, \qquad r_t = \alpha_f + \beta_f\, x_t^{f} + \varepsilon_{f,t}.$$

Bajo **noticias** la estimación preliminar es un pronóstico eficiente y la revisión es ortogonal a ella, $\beta_p = 0$; bajo **ruido** la estimación preliminar es la verdad más un error de medición, $x_t^p = x_t^f + u_t$, de modo que $r_t = -u_t$ es ortogonal a la final, $\beta_f = 0$, y está negativamente relacionada con la preliminar. El `noise_share` reportado es $\max(0, -\hat\beta_p)$, la fracción de la varianza de la estimación preliminar atribuible a $u_t$ bajo el modelo de ruido puro. Los errores estándar son robustos a heterocedasticidad por defecto (`hac_lags=0`) y HAC de Newey-West con `hac_lags="auto"` o un ancho de banda entero. Con añadas por instantánea esta maquinaria es tan informativa como el calendario de capturas que la sustenta: $K$ capturas en $K$ días distintos dan a lo sumo $K$ columnas.

### 1.3 El almacén de instantáneas

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

La tabla vive en el mismo archivo que la caché HTTP y el almacén de ALFRED, `~/.cache/puremacro/cache.db` o `$PUREMACRO_HTTP_CACHE_DIR` ([CACHE_DB.md](CACHE_DB.md)), y `bootstrap_schema` la crea junto a `connector_events` (componente `realtime_vintages` de `schema_version`, versión 1). Tres funciones auxiliares de `puremacro._cache_db` la envuelven: `store_realtime_vintages` (inserta o reemplaza; devuelve el número de filas escritas), `query_realtime_vintages` (opcionalmente restringida a las ediciones con `vintage_date <= ...`, la materia prima del corte transversal *as-of*; `VintagePanel.as_of` conserva después la edición más reciente de cada periodo) y `record_connector_event` (una fila por evento de telemetría).

Los cuatro conectores solo se diferencian en cómo construyen la solicitud y analizan el cuerpo; todo lo demás —política del canario, escritura en la caché, relectura del historial, respaldo, telemetría— vive en una única función compartida (`puremacro.fetch.realtime._snapshot.fetch_snapshot_vintages`), de modo que cada llamada a `fetch_*_vintages` sigue el mismo árbol de decisión:

1. Resolver la credencial (Banxico, INEGI, BCCh; el BCCh necesita ambas mitades). Si no se resuelve y `use_cache=True`, devolver las instantáneas almacenadas cuando existan — sin solicitud, sin advertencia, sin evento — y, en caso contrario, lanzar `MissingCredentialError`.
2. Solicitar la serie en vivo con `urllib` (`User-Agent: puremacro (real-time vintage reader)`, `timeout=60.0` por defecto) y ejecutar el analizador (*parser*), que a su vez ejecuta el canario bajo la política `on_drift` de la Sección 1.4.
3. Con `use_cache=True`, almacenar las filas analizadas y registrar el evento `('success', 'none')`. Después, con `history=True` (valor por defecto), **devolver todas las instantáneas almacenadas localmente para esa serie** —la de hoy incluida, una añada por fecha de instantánea—, que es lo que una prueba de revisiones necesita; `history=False` devuelve solo la instantánea recién descargada.
4. Si la respuesta se analizó como una tabla vacía y el almacén contiene filas, emitir la advertencia `... received empty observations; falling back to cached vintages.` y registrar `('fallback', 'sqlite_cache')`.
5. Si algo lanzó una excepción — un error HTTP, un tiempo de espera agotado, un `SchemaDriftError` — y el almacén contiene filas, emitir la advertencia `... failed (<exc>); falling back to cached vintages.` (`SchemaDriftWarning` para la deriva, `UserWarning` en los demás casos) y registrar `('fallback', 'sqlite_cache')`. Con el almacén frío la excepción se propaga.

`use_cache=False` renuncia por completo al almacén —no lo lee ni lo escribe—, de modo que la instantánea en vivo se devuelve tal como se analizó y cualquier fallo se lanza en lugar de disimularse. Una instantánea que no se puede *almacenar* se devuelve de todos modos, y una escritura de telemetría fallida nunca rompe la descarga que la emitió. En los dos proveedores con credenciales en la URL, el mensaje de fallo y los atributos `url` / `filename` que urllib adjunta a sus excepciones se depuran antes de mostrarse.

Los puntos de entrada de panel `fetch_*_panel` capturan lo que se propague y lo registran en `metadata["failed"]` bajo `"<ISO3>:<variable>"`; `vintage_panel` lo reindexa como `"<ISO3>:<variable>:<provider>"` y nunca lanza una excepción por una serie fallida.

### 1.4 Canarios de esquema

Un servicio REST que renombra una clave o cambia un formato de fecha sigue respondiendo `200 OK`. `SchemaCanary` detecta esa clase de deriva antes de que se convierta en un panel silenciosamente vacío. Cada validador es una función pura que devuelve `(ok, reason)`; los analizadores los invocan y aplican una política, de modo que la deriva se trata como cualquier otro fallo de descarga en el árbol anterior. Las cargas (*payloads*) pueden ser `bytes`, `str` o JSON ya decodificado. Las listas de observaciones de más de 500 elementos se muestrean: se inspeccionan los primeros 20 y los últimos 20 elementos.

| Proveedor | Raíz | Envoltorio | Por observación | Regla de fechas |
|---|---|---|---|---|
| `banxico` | dict | `bmx.series[0].datos` (una lista, posiblemente vacía) | claves `fecha`, `dato` | `DD/MM/YYYY` o `YYYY-MM-DD`, analizadas de forma estricta |
| `inegi` | dict | `Series[0].OBSERVATIONS` (una lista) | claves `TIME_PERIOD`, `OBS_VALUE` | `YYYY/NN` con $1900 \le YYYY \le 2100$ y $1 \le NN \le 12$, un periodo del tipo `YYYYQn`, o cualquier cosa que `pd.to_datetime` acepte |
| `bcb` | list | la propia lista | claves `data`, `valor` | exactamente `DD/MM/YYYY` |
| `bcch` | dict | `Codigo` ausente o `0`; la lista de observaciones bajo `Series.Obs` —la grafía que emite SIETE—, aceptándose todavía la minúscula `obs`, bajo `Series` o en la raíz | claves `indexDateString`, `value` | `DD-MM-YYYY`, o cualquier cosa que `pd.to_datetime` acepte |

**La política `on_drift`.** Todos los analizadores y rutas de descarga latinoamericanos aceptan `on_drift`, uno de `DRIFT_POLICIES = ("raise", "warn", "ignore")`, y `canary.handle_drift(provider, reason, on_drift)` la aplica:

- `"raise"` (valor por defecto): lanza `SchemaDriftError`, un `ValueError`. La capa de descarga lo convierte en un respaldo desde la caché cuando el almacén contiene algo, y lo vuelve a lanzar en caso contrario.
- `"warn"`: emite `SchemaDriftWarning` (un `UserWarning`) y deja que el analizador lo intente de todos modos — sus propios recursos de fecha, una fecha ISO donde se esperaba `DD/MM/YYYY`, por ejemplo, a menudo recuperan la respuesta.
- `"ignore"`: analiza en silencio.

Sea cual sea la política, se registra un evento `('schema_drift', ...)` en `connector_events` —con `fallback_used` igual a `'none'`, `'warning_emitted'` o `'ignored'`, respectivamente—, de modo que `connector_health()` ve la deriva incluso cuando un respaldo se la ocultó a quien llamó. Una cadena de política fuera de `DRIFT_POLICIES` lanza `ValueError` en la primera llamada, antes de mirar la carga, y no meses después dentro de una ruta de respaldo.

`SchemaCanary.check(provider, payload, *, raise_on_drift=False, on_drift=None)` y su gemela funcional `validate_payload` admiten ambas grafías: `on_drift` manda, y cuando vale `None` la bandera antigua `raise_on_drift` selecciona `"raise"` (`True`) o `"warn"` (`False`). Ambas devuelven `(ok, reason)` siempre que devuelvan algo, y un nombre de proveedor desconocido pasa como `(True, "")`.

### 1.5 El contenedor de cartuchos `.pmz`

`pack_realtime_cartridge` y `load_realtime_cartridge` son envolturas ligeras sobre `puremacro.pocket`, así que un cartucho en tiempo real *es* un cartucho de pocket:

- un **zip plano** (`ZIP_STORED`) con dos tipos de miembro: `manifest.json` y un `frames/<name>.npz` por tabla. Un cartucho en tiempo real contiene `data`, el `VintagePanel.df` ordenado con las ocho `VINTAGE_COLUMNS`, y —cuando se empaqueta un `VintagePanel` que lleva metadatos— una segunda tabla `vintage_metadata`, un par de columnas clave / valor JSON que `load_realtime_cartridge` reconstruye en `panel.metadata`, de modo que `failed`, `missing`, `freq`, `provider_used` y compañía sobreviven al viaje de ida y vuelta. Los valores que no son nativos de JSON (un `DataFrame` de ediciones descartadas, un `Timestamp`) se almacenan mediante `str` en lugar de descartarse, para que el cartucho siga diciendo qué había;
- cada tabla es un **archivo npz** — un array de NumPy por columna más un esquema JSON que registra los dtypes, el tipo de índice y los nombres de columna (`puremacro.runtime.store`). Sin parquet, sin pyarrow y **sin pickle**: las columnas de objetos que contengan algo distinto de cadenas se rechazan al empaquetar, y en la carga se usa `np.load(..., allow_pickle=False)`;
- el manifiesto registra `format = "puremacro-cartridge"`, `version = 1`, un bloque `provenance` (marca temporal UTC `created`, `puremacro_version`, `source`, `vintage`, `notes`, `call`, `host`) y, por tabla, `name`, `n_rows`, `n_cols`, `columns`, `index`, `n_bytes` y el **SHA-256** de la carga npz almacenada.

`load_realtime_cartridge(path, verify=True)` lee el zip mediante `pocket.load`, que siempre compara el resumen (*digest*) de cada carga con el manifiesto (una discrepancia lanza `pocket.CartridgeError`); con `verify=True` además vuelve a codificar la tabla cargada y compara de nuevo (`Cartridge.verify()`), y `verify=False` omite únicamente esa segunda comprobación. Devuelve un `VintagePanel` cuya `metadata` lleva `cartridge_path`, `provenance_source`, `provenance_vintage` y `provenance_notes`. El resumen depende solo de los datos, no del reloj, y eso es lo que hace posible la comprobación por recodificación.

¿Es seguro con un archivo que usted no produjo? Frente a la **ejecución de código**, sí: el formato no contiene pickle, los miembros se leen en memoria por nombre y nunca se extraen al disco, y el manifiesto se analiza con `json`. Frente a la **manipulación**, no: el SHA-256 es una suma de comprobación sin firmar, así que cualquiera puede reempaquetar una tabla modificada con un resumen coincidente. El módulo `pocket` lo enuncia con precisión — los cartuchos son un formato de transporte, no una frontera de confianza — y un archivo hostil todavía puede consumir memoria al descomprimirse. La relación con `puremacro.pocket` es de inclusión: `pocket.load(path)` abre un cartucho en tiempo real como un `Cartridge` genérico, `pocket.inspect_cartridge(path)` lee su manifiesto sin decodificar dato alguno, y `pocket.to_base64` / `pocket.from_base64` lo trasladan por el portapapeles. `pack_realtime_cartridge` solo fija los valores por defecto (`source="LatAm Regional Real-Time Ecosystem"`, `vintage=` la fecha UTC de hoy, una nota estándar) y `load_realtime_cartridge` solo añade la envoltura `VintagePanel`.

---

## 2. Opciones metodológicas y de proveedor

| | `banxico` | `inegi` | `bcb` | `bcch` |
|---|---|---|---|---|
| Institución | Banco de México, SIE | INEGI, Banco de Indicadores / BIE | Banco Central do Brasil, SGS | Banco Central de Chile, SIETE |
| País atendido | `MEX` | `MEX` | `BRA` | `CHL` |
| Credencial | token | token | ninguna | usuario + contraseña |
| Dónde viaja | cabecera de la solicitud `Bmx-Token` | segmento de ruta de la URL de la solicitud | — | cadena de consulta (`user=`, `pass=`), codificada en URL |
| Constante del endpoint | `BANXICO_SERIES_URL` | `INEGI_SERIES_URL` | `BCB_SGS_URL` | `BCCH_SIETE_URL` (construida por `_build_bcch_url`) |
| Coerción de valores | descarta `N/E`, `NaN`, `null`; elimina las comas de miles | elimina las comas de miles | la coma es el separador decimal | la coma es el separador decimal |
| Semántica de la añada | fecha de instantánea | fecha de instantánea | fecha de instantánea | fecha de instantánea |

Las entradas de catálogo `BANXICO_SERIES`, `INEGI_SERIES`, `BCB_SERIES` y `BCCH_SERIES` de `puremacro.fetch.realtime.catalog` se registran al importar, de modo que `vintage_catalog()` las enumera y `providers_for("MEX", "cpi")` devuelve `['banxico', 'inegi']`. La columna `units` decide la transformación de revisiones por defecto de la sección 1.2, y la columna `freq` la hace cumplir `vintage_panel` (más abajo). Las entradas cuyo identificador no se pudo confirmar contra el servicio en vivo llevan el marcador `VERIFY ONLINE` en su `note`; `pytest -m network` es la auditoría que las despeja.

| Proveedor | Variable | Id. de serie | Unidades | Frec. | Nota del catálogo |
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

Dos de estas entradas conviene leerlas dos veces. El `433` del BCB es la **variación porcentual mensual** del IPCA, no un nivel de índice, así que sus unidades lo dicen y ningún auxiliar de revisiones aplicará una diferencia logarítmica a una serie ya diferenciada. El `22099` del BCB es el índice encadenado de volumen trimestral desestacionalizado del IBGE (el `22109` es el no desestacionalizado); el `4380`, que parece una serie de PIB, es «PIB mensal - valores correntes», mensual y nominal. Del lado chileno, los códigos de SIETE terminan en la letra española de la frecuencia (`D`, `M`, `T`, `A`) —ninguno termina en `.Q`— y el IPC vive en el capítulo `F074`, no en el `F073`, que son tipos de cambio.

`policy_rate` y `activity` son las dos variables canónicas que añade esta página. `policy_rate` («Central bank policy rate / target interest rate») lleva los alias `tpm`, `selic`, `tasa_objetivo`, `interest_rate`, `mp_rate`, `central_bank_rate`, `target_rate` y `overnight_rate`; `activity` («Monthly economic activity index (IGAE / IMACEC / IBC-Br), SA») lleva `economic_activity`, `igae`, `imacec` e `ibc_br`. Así, `variables=["selic"]` y `variables=["policy_rate"]` son la misma solicitud.

A través del registro, las llamadas en vivo son `vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")`, `vintage_panel(["BRA"], providers="bcb", variables=["gdp_real"], freq="Q")` y `vintage_panel(["CHL"], providers="bcch", variables=["activity"], freq="M")`; la sección 4.4 muestra qué hace la primera sin conexión. Cosas que conviene saber sobre esa ruta:

- `providers="auto"` nunca llega a estos conectores: `DEFAULT_PROVIDER_ORDER` es `oecd_stes, alfred, bundesbank, ons, statcan, ecb_rtd`. Nómbrelos explícitamente, como cadena o en una lista.
- **`freq` se hace cumplir, una frecuencia por llamada.** `SUPPORTED_FREQUENCIES` es `{"Q", "M", "D"}` y cualquier otra cosa lanza `ValueError`. Una entrada de catálogo que declara otra frecuencia *no se sirve*: se informa en `metadata["missing_pairs"]` / `metadata["failed"]` con la razón, p. ej. `catalogue series SF61745 is daily (freq='D'); requested freq='Q'`. Las entradas que no declaran frecuencia —todos los archivos trimestrales de añadas, anteriores al campo— se sirven solo a `"Q"`. Pida cada serie a su propia frecuencia y pase el `periods_per_year` correspondiente (4, 12, 252…) a `triangle()` / `revisions()` / `news_or_noise()`; mantenga las tasas de política monetaria diarias fuera de los contrastes de revisiones.
- El valor por defecto `series="gdp_real"` no resuelve a nada para `banxico` (su catálogo no tiene PIB), de modo que `vintage_panel(["MEX"], providers="banxico")` informa `(MEX, gdp_real)` como faltante en lugar de fallar.
- Un catálogo personalizado (`catalog=`) se consulta solo para decidir si una serie es servible; los cuatro conectores buscan las series en sus propias tablas. Para un identificador no catalogado, llame directamente a `fetch_*_vintages(series_id)` y pase la tabla por `normalize_vintage_frame`.
- `fetch_*_panel` espera nombres canónicos de variables (`vintage_panel` canoniza antes de despachar) y omite silenciosamente los países y variables ajenos a su tabla. `store=` y `refresh=` se aceptan y se ignoran.
- Combinar proveedores, p. ej. `providers=["banxico", "bcb", "bcch"]`, dispara la advertencia habitual de proveedores mixtos. Aquí es inocua, pues los cuatro comparten la misma semántica de fecha de instantánea, y `warn_on_mixed_providers=False` la silencia.

---

## 3. Credenciales y configuración

La resolución sigue `puremacro.credentials` ([CREDENTIALS.md](CREDENTIALS.md)): argumento explícito, luego las variables de entorno en el orden del registro, luego el archivo TOML (`$PUREMACRO_CREDENTIALS_FILE` si está definida; si no, `$XDG_CONFIG_HOME/puremacro/credentials.toml` si `XDG_CONFIG_HOME` está definida; si no, `~/.puremacro/credentials.toml`), y en último caso nada. `credentials.get_credential` y `credentials.require_credential` son alias de `get` y `require`; `credentials.status()` muestra qué servicios están configurados sin imprimir ningún valor.

| Servicio | Variables de entorno (en orden) | Claves TOML | Registro |
|---|---|---|---|
| `banxico` | `BANXICO_API_KEY`, `BMX_TOKEN`, `PUREMACRO_BANXICO_API_KEY` | `[banxico] api_key` | https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html |
| `inegi` | `INEGI_API_KEY`, `PUREMACRO_INEGI_API_KEY` | `[inegi] api_key` | https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/ |
| `bcch` (usuario) | `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` | `[bcch] user` (o `api_key`) | https://si3.bcentral.cl/estadisticas/principal1/registro/index.html |
| `bcch` (contraseña) | `BCCH_API_PASS`, `PUREMACRO_BCCH_API_PASS` | `[bcch] password` | (mismo registro) |

El BCCh es la excepción: SIETE autentica con el correo electrónico registrado **y** una contraseña, y el registro mantiene separadas ambas mitades. `PASSWORD_ENV_VARS` nombra los servicios cuya credencial es un par —`{"bcch": ("BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS")}`— y `credentials.get("bcch")` resuelve únicamente el *usuario*, recorriendo `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` y después `[bcch].user` / `[bcch].api_key`, mientras que `credentials.get_password("bcch")` resuelve únicamente la contraseña, a partir de las variables de contraseña y después de `[bcch].password`. Así, una contraseña suelta ya no puede recogerse y enviarse como usuario, y la contraseña *sí* se lee del archivo TOML. `credentials.require("bcch")` exige ambas mitades y, cuando falta alguna, lanza `MissingCredentialError` con un mensaje que dice «needs a user and a password» y enumera ambos grupos de variables; `credentials.status()` señala un servicio cuyo usuario está configurado pero cuya contraseña no. `fetch_bcch_vintages(user=..., password=...)` sustituye cualquiera de las dos mitades de forma explícita.

Ambas mitades viajan en la cadena de consulta de la solicitud GET (`user=`, `pass=`, codificadas en URL por `_build_bcch_url`), así que trate la URL de la solicitud como un secreto: no la pegue en una incidencia (*issue*) y recuerde que los proxies y los registros de los servidores la ven. El conector depura el usuario y la contraseña —en bruto y codificados en URL— del texto de la advertencia y de los atributos `url` / `filename` que urllib adjunta a sus excepciones, de modo que un mensaje de respaldo se puede pegar sin riesgo; una URL que construya usted mismo, no. El token de Banxico va en la cabecera `Bmx-Token`; el del INEGI es un segmento de ruta de la URL.

---

## 4. Ejemplos prácticos ejecutables

Todos los scripts siguientes se ejecutan sin conexión e imprimen exactamente la salida mostrada; los cuerpos tardan una fracción de segundo y la importación de `puremacro` unos pocos segundos. Los scripts que tocan la base de datos la apuntan primero a un directorio temporal (`PUREMACRO_HTTP_CACHE_DIR`) para dejar intacto `~/.cache/puremacro/cache.db`; elimine esa línea para trabajar contra su caché real.

### 4.1 Del fixture al panel, al cartucho y de vuelta

Una carga con la forma de la respuesta del SGS del BCB pasa por el canario, el analizador, `normalize_vintage_frame` y un `VintagePanel`, se empaqueta en un `.pmz` en un directorio temporal y regresa verificada:

```python
import json, os, tempfile, zipfile
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()   # mantener el ejemplo fuera de ~/.cache

from puremacro import pocket
from puremacro.fetch.realtime import (
    VintagePanel, normalize_vintage_frame, validate_payload,
    pack_realtime_cartridge, load_realtime_cartridge,
)
from puremacro.fetch.realtime.bcb import parse_bcb_json

# Un fixture con la forma exacta del JSON del endpoint SGS: una lista de {data, valor}
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
    back = load_realtime_cartridge(path)          # verify=True: SHA-256 por tabla
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

La coma decimal de `"4,50"` se convirtió en `4.50`, el archivo zip contiene `manifest.json`, `frames/data.npz` y el `frames/vintage_metadata.npz` que traslada `panel.metadata`, y la tabla de datos sobrevive el viaje de ida y vuelta sin cambios (`equals` es `True`, dtypes incluidos). Nótese `n_vintages = 1`: una instantánea, una edición.

### 4.2 El contrato del canario

```python
import os, tempfile, warnings
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()

from puremacro.fetch.realtime import (
    SchemaCanary, SchemaDriftError, SchemaDriftWarning, validate_payload,
)

good = {"bmx": {"series": [{"idSerie": "SF61745",
        "datos": [{"fecha": "15/01/2026", "dato": "10.75"}]}]}}
drifted = {"bmx": {"series": [{"idSerie": "SF61745",
        "datos": [{"fecha": "15/01/2026", "value": "10.75"}]}]}}   # 'dato' renombrado

print(SchemaCanary.validate_banxico(good))
print(SchemaCanary.validate_banxico(drifted))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    print(validate_payload("banxico", drifted))                 # advierte y devuelve
print(caught[0].category.__name__, "->", caught[0].message)

try:
    validate_payload("banxico", drifted, raise_on_drift=True)
except SchemaDriftError as exc:
    print("raised", type(exc).__name__)

print(SchemaCanary.validate_bcb([{"data": "2026-01-01", "valor": "12.0"}]))   # fecha ISO, no DD/MM/YYYY
print(SchemaCanary.validate_bcch({"Codigo": 99, "Descripcion": "Servicio no disponible"}))
print(validate_payload("alfred", {"anything": 1}))              # proveedor desconocido: pasa sin más
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

Los métodos estáticos `validate_*` solo informan; `validate_payload` es donde vive la política. Aquí toma por defecto `on_drift="warn"` (sin `raise_on_drift`), de modo que emite `SchemaDriftWarning`, registra `('schema_drift', 'warning_emitted')` y devuelve; `raise_on_drift=True` —u `on_drift="raise"`— lanza `SchemaDriftError` y registra `('schema_drift', 'none')`, y `on_drift="ignore"` devuelve en silencio pero registra igualmente `('schema_drift', 'ignored')`. Los analizadores aceptan el mismo argumento y su valor por defecto es `"raise"`: `parse_banxico_json(drifted, on_drift="warn")` advierte y devuelve las filas que pudo rescatar, mientras que el valor por defecto lanza la excepción.

### 4.3 Dos capturas hacen una añada

La misma serie del BCB capturada en dos días distintos; la segunda captura revisa 2023Q4 y añade 2024Q1. Ambas entran en el almacén de instantáneas; la consulta *as-of* y el triángulo de revisiones se obtienen entonces de `query_realtime_vintages` y `VintagePanel`:

```python
import os, tempfile
os.environ["PUREMACRO_HTTP_CACHE_DIR"] = tempfile.mkdtemp()

from puremacro._cache_db import store_realtime_vintages, query_realtime_vintages
from puremacro.fetch.realtime import VintagePanel, normalize_vintage_frame
from puremacro.fetch.realtime.bcb import parse_bcb_json
from puremacro.reports import df_to_markdown

# Dos capturas de la misma serie en dos días distintos. La segunda
# edición revisa 2023Q4 y añade 2024Q1: eso es lo que aquí constituye una añada.
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
as_of = query_realtime_vintages("bcb", "BRA", "22099", vintage_date="2024-03-31")   # ediciones hasta ese día
print(as_of[["date", "vintage", "value"]])

long = normalize_vintage_frame(cached, country="BRA", variable="gdp_real",
                               provider="bcb", series_id="22099", units="index")
panel = VintagePanel(long)
tri = panel.triangle("BRA", "gdp_real")
print(tri)
print(len(panel.revisions("BRA", "gdp_real")))   # todo periodo de referencia es anterior a la primera captura
tri.index, tri.columns = tri.index.strftime("%Y-%m-%d"), tri.columns.strftime("%Y-%m-%d")
print(df_to_markdown(tri))                     # o df_to_latex / df_to_typst
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

Cinco filas en el almacén, dos de ellas para 2023Q4 (`101.0` el 2024-03-01, `101.6` el 2024-06-01), y la consulta *as-of* al 2024-03-31 devuelve solo la captura de marzo. El triángulo es el $T[t, v]$ de la sección 1.2, con `NaN` donde 2024Q1 aún no existía. Sin embargo, `revisions()` tiene cero filas: los tres periodos de referencia terminaron antes de la primera captura (2024-03-01), de modo que ninguno tiene una primera publicación observable y la regla de censura de la sección 1.2 los descarta todos. Dos capturas, dos columnas, ningún par de revisión utilizable: la muestra de revisiones de un archivo de instantáneas comienza con los periodos de referencia que terminan después de iniciada la captura, y `news_or_noise()` necesita muchos días de captura distintos más allá de eso antes de poder decir algo.

### 4.4 La llamada a través del registro, sin conexión

`vintage_panel` contra `banxico` sin token y con el almacén frío, luego con una instantánea en el almacén, y luego un proveedor sin clave durante una caída simulada (la llamada de red se sustituye por un `URLError`, así que nada sale de la máquina). La tasa de política monetaria de Banxico está catalogada como diaria, de modo que la llamada pide `freq="D"`; con `freq="Q"` se rechazaría con `catalogue series SF61745 is daily (freq='D'); requested freq='Q'` antes de construir ninguna solicitud:

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

# 1. Sin token y con el almacén frío: el fallo se registra, no se lanza
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    panel = vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")
print(panel.is_empty(), "|", panel.metadata["failed"]["MEX:policy_rate:banxico"].split(" needs")[0])

# 2. Una instantánea capturada un día anterior hace que la misma llamada funcione sin conexión
store_realtime_vintages(pd.DataFrame({
    "provider": "banxico", "country": "MEX", "series_id": "SF61745",
    "date": ["2026-01-15", "2026-02-15"], "vintage": "2026-03-01", "value": [10.50, 10.25]}))
panel = vintage_panel(["MEX"], providers="banxico", variables=["policy_rate"], freq="D")
print(panel.coverage()[["country", "variable", "provider", "n_obs", "n_vintages"]])
print(panel.metadata["provider_used"], panel.metadata["failed"])

# 3. Un proveedor sin clave durante una caída simulada: caché caliente -> advertencia + evento de respaldo
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

El paso 1 devuelve un panel vacío y registra el `MissingCredentialError` bajo `metadata["failed"]["MEX:policy_rate:banxico"]` (la advertencia suprimida es la habitual de `vintage_panel`, «returned nothing for 1 of 1 requested countries»). El paso 2 es la misma llamada servida desde el almacén: sin token, sin solicitud, `provider_used == {'MEX': 'banxico'}` y nada falló. El paso 3 muestra la rama de respaldo de la sección 1.3 para el conector sin clave — la fila almacenada regresa con la advertencia `falling back to cached vintages` y un evento `('fallback', 'sqlite_cache')`, que `connector_health()` reporta como un respaldo para `bcb`. Ninguno de los dos primeros pasos registró un evento; solo un intento en vivo lo hace.

### 4.5 Lo que dicen el catálogo y el registro de credenciales

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

El cuaderno 58 (`notebooks/58_latin_america_realtime_macro.py`, edición en español `_es.py`) recorre los mismos objetos de principio a fin — panel, cartucho, cobertura, corte transversal *as-of*, triángulo, Mankiw-Shapiro. Su panel es **sintético**: se genera con `numpy.random.default_rng(42)` en torno a tendencias lineales, de modo que ejercita la maquinaria pero sus cifras no son datos de Banxico, del BCB ni del BCCh.

---

## 5. Especificación completa de la API

Los cuatro módulos de conectores comparten una misma forma; las diferencias son los argumentos de credencial y la constante del endpoint.

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

| Parámetro | Significado |
|---|---|
| `raw` | El cuerpo HTTP (`bytes`), su texto, o el JSON decodificado. Una entrada vacía devuelve una tabla `[date, vintage, value]` vacía; un cuerpo que no es JSON válido lanza `json.JSONDecodeError`, que la capa de descarga trata como descarga fallida y no como serie vacía; una carga que no supera el canario sigue la política `on_drift`. |
| `series_id` | El identificador del proveedor. No se escribe en la tabla de salida. |
| `vintage_date` | La fecha de instantánea que se estampa en esta descarga. Por defecto: hoy a medianoche local. |
| `freq` (INEGI) | Una pista que solo se usa cuando la respuesta es demasiado corta para que la regla estructural de más abajo decida. |
| `firstdate`, `lastdate` (BCCh) | Ventana opcional `yyyy-mm-dd`; vacía significa la serie completa. |
| `token`, `user`, `password` | Credenciales explícitas; si se omiten se resuelven como en la sección 3. |
| `timeout` | Segundos que se pasan a `urllib.request.urlopen`. |
| `use_cache` | `True` almacena las filas analizadas en `realtime_vintages` y recurre a las instantáneas almacenadas ante un fallo. `False` no lee ni escribe el almacén: la instantánea en vivo se devuelve tal como se analizó y cualquier fallo se lanza. |
| `history` | `True` (valor por defecto) devuelve todas las instantáneas almacenadas localmente para la serie, la de hoy incluida, una añada por fecha de instantánea; `False` devuelve solo la instantánea recién descargada. |
| `on_drift` | `"raise"` (valor por defecto), `"warn"` o `"ignore"`; la política del canario de la sección 1.4. Cualquier otro valor lanza `ValueError`. |
| `countries`, `variables` | Códigos ISO3 y nombres **canónicos** de variables. Las entradas ajenas al país o a la tabla del proveedor se omiten silenciosamente. |

**Análisis de periodos del INEGI.** `TIME_PERIOD` llega como `YYYY/NN`, que es un mes para un indicador mensual y un trimestre para uno trimestral, y nada en la respuesta lo dice de forma fiable. `parse_inegi_json` lo decide estructuralmente, sin ningún identificador de serie codificado: un número de periodo mayor que 4 descarta los trimestres; números de periodo que nunca superan 4 a lo largo de dos o más años, o de cinco o más observaciones, descartan los meses, porque una serie mensual es contigua y cinco meses consecutivos siempre llegan a mayo. Solo cuando la respuesta es demasiado corta para decidirlo manda el `freq` de quien llama, y después el campo `FREQ` (palabras españolas como «trimestral» / «mensual», o los códigos numéricos `CL_FREQ`); si nada de eso resuelve, la serie se trata como mensual. Los periodos trimestrales se fechan en el primer día del trimestre y los mensuales en el primer día del mes.

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
vintage_catalog(provider=None) -> pd.DataFrame   # + una columna freq

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

- `store_realtime_vintages` acepta `date` u `observation_date` y `vintage` o `vintage_date` como nombres de columna, exige `provider`, `country`, `series_id` y `value`, pasa el país a mayúsculas, almacena los valores `NaN` como `NULL`, omite las filas cuyas fechas no se pueden analizar y devuelve el número de filas escritas (`0` para una tabla vacía).
- `query_realtime_vintages` devuelve `[date, vintage, value, provider, country, series_id]` con `date` y `vintage` como datetimes, ordenadas por observación y luego por añada; `vintage_date=` conserva las ediciones con `vintage_date <= vintage_date`.
- `record_connector_event` escribe `(ts, source, outcome, fallback_used)` con `ts` el tiempo Unix actual; aquí los valores son cadenas libres, a diferencia del vocabulario validado de `narrative.sources._telemetry.log_event`.
- `pack_realtime_cartridge` empaqueta `panel.df` (o la tabla que usted pase) como la tabla `data`, más `vintage_metadata` cuando se le entrega un `VintagePanel` con metadatos; `path` se usa tal cual, así que el sufijo `.pmz` es una convención.
- `get` resuelve el token, o la mitad *usuario* de un servicio de dos partes; `get_password` resuelve la mitad de la contraseña y devuelve `None` para todo servicio fuera de `PASSWORD_ENV_VARS`. `require` exige ambas mitades de un par antes de devolver el usuario.

---

## 6. Interfaz de resultados y exportación a manuscritos

**Tablas de los analizadores y de las descargas.** `parse_*_json` y `fetch_*_vintages` devuelven un `pd.DataFrame` con las columnas `date` (datetime), `vintage` (datetime) y `value` (float), ordenado y sin duplicados en `(date, vintage)`; el respaldo desde la caché devuelve las mismas tres columnas de `query_realtime_vintages`.

**`VintagePanel`** (`puremacro.fetch.realtime._base`), una dataclass con dos campos:

- `df`: la tabla ordenada con `VINTAGE_COLUMNS = [country, variable, date, vintage, value, provider, series_id, units]`, una fila por `(country, variable, date, vintage)`.
- `metadata`: un dict. Desde `fetch_*_panel`: `provider`, `failed`. Desde `vintage_panel`: `requested_countries`, `variables`, `providers_tried`, `provider_used`, `missing`, `missing_pairs`, `failed`, `freq` (y `unadjusted_dropped` cuando el filtro estacional eliminó ediciones). Desde `load_realtime_cartridge`: `cartridge_path`, `provenance_source`, `provenance_vintage`, `provenance_notes`.

Sus métodos son las operaciones de la sección 1.2: `countries` / `variables` / `providers` (propiedades), `len(panel)`, `is_empty()`, `vintage_semantics()`, `coverage()` (`n_obs`, `n_vintages`, `n_periods`, primera y última fecha y añada por serie), `as_of(vintage_date)` (una tabla ancha indexada por `(country, date)`), `long(country, variable)`, `triangle(country, variable, *, transform="level", periods_per_year=4)`, `revisions(country, variable, *, transform=None, periods_per_year=4, release=0)` (`transform=None` elige la transformación que implican las `units` catalogadas), `news_or_noise(...)` que devuelve un `MankiwShapiroResult`, `news_or_noise_panel(...)` que devuelve una fila por `(country, variable)`, `filter(...)` y `VintagePanel.concat(...)`.

**`MankiwShapiroResult`** (`puremacro.vintages`), una dataclass con `n_obs`, `transform`, `hac_lags`, `significance`, `mean_revision`, `se_mean_revision`, `t_mean_revision`, `p_mean_revision`, `std_revision`, `alpha_on_preliminary`, `beta_on_preliminary`, `se_beta_on_preliminary`, `t_beta_on_preliminary`, `p_beta_on_preliminary`, `alpha_on_final`, `beta_on_final`, `se_beta_on_final`, `t_beta_on_final`, `p_beta_on_final`, `noise_share`, `rejects_news`, `rejects_noise` y `verdict`.

**Resultados del canario y del cartucho.** Los validadores devuelven tuplas `(ok, reason)`; `load_realtime_cartridge` devuelve un `VintagePanel`, y `pocket.load(path)` devuelve el `Cartridge` subyacente con `frames`, `provenance` (`created`, `puremacro_version`, `source`, `vintage`, `notes`, `call`, `host`), `records`, `verify()` y `summary()`.

**Exportación a manuscritos.** Ninguno de estos objetos lleva métodos propios `summary()`, `plot()`, `to_frame()`, `to_markdown()`, `to_latex()` ni `to_typst()`; sus tablas — `coverage()`, `as_of()`, `triangle()`, `revisions()`, `news_or_noise_panel()`, `vintage_catalog()`, `credentials.status()`, `connector_health()` — son DataFrames ordinarios, y `puremacro.reports.df_to_markdown`, `df_to_latex` y `df_to_typst` las renderizan, como hace la sección 4.3 con el triángulo. `Cartridge.summary()` es el único resumen textual de esta pila.

---

## 7. Advertencias y limitaciones

- **Semántica de instantáneas.** Una descarga es una edición; las revisiones existen solo entre capturas en días distintos; una nueva descarga el mismo día sobrescribe; los periodos de referencia que terminaron antes de la primera captura no tienen primera publicación observable y se descartan de `revisions()` (sección 1.2); el sello de la añada es la medianoche local mientras que el `vintage` por defecto de `pack_realtime_cartridge` es la fecha UTC. Para el historial archivado del PIB trimestral de México use `oecd_stes`.
- **Sin capa HTTP compartida.** Los cuatro conectores llaman a `urllib` directamente y eluden `puremacro._http` y `fetch_with_backoff`: sin caché del cuerpo HTTP, sin respaldo SSL, sin reintento ante HTTP 429. Un límite de tasa de solicitudes aflora como `HTTPError`, y por tanto como respaldo desde el almacén o como excepción.
- **`use_cache=False` es una renuncia completa.** No lee ni escribe `realtime_vintages`, de modo que no hay respaldo alguno: la instantánea en vivo se devuelve tal como se analizó y todo fallo se propaga. Esa es la bandera que hay que usar cuando se quiere saber que la solicitud falló; déjela en `True` cuando se quiera que el panel sobreviva a una caída.
- **Telemetría.** Los eventos usan los resultados `success`, `fallback` y `schema_drift`, con los valores de `fallback_used` `none`, `sqlite_cache`, `warning_emitted` e `ignored`. No se escribe ningún evento para la ruta del almacén sin credenciales ni para una excepción con el almacén frío. Los conectores escriben a través de `_cache_db.record_connector_event`, de modo que el interruptor de emergencia `PUREMACRO_NARRATIVE_TELEMETRY=0` de [CONNECTOR_HEALTH.md](CONNECTOR_HEALTH.md) **no** los silencia. `connector_health()` cuenta como respaldo toda fila con `fallback_used != 'live'`, de modo que una descarga en tiempo real exitosa (`'none'`) se cuenta tanto en `n_success` como en `n_fallback`; consulte `connector_events` directamente para estas fuentes.
- **Una frecuencia por llamada a `vintage_panel`.** Ahora `freq` se coteja contra el campo `freq` del catálogo, no contra los datos: una serie que declara otra frecuencia se rechaza con una razón en lugar de servirse con la etiqueta equivocada, y una entrada que no declara ninguna se sirve solo a `"Q"`. Nada comprueba que la *respuesta* esté realmente en la frecuencia declarada, de modo que un proveedor que cambie la frecuencia de una serie aguas arriba seguirá etiquetado por el catálogo. Pase el `periods_per_year` correspondiente a los auxiliares de revisiones y mantenga las tasas de política monetaria diarias fuera de esos contrastes.
- **Las credenciales del BCCh** viajan en la cadena de consulta de un GET simple. Se depuran de las advertencias del propio conector y de los atributos de sus excepciones, pero una URL que construya usted mismo con `BCCH_SIETE_URL.format(...)` las lleva en texto claro.
- **Catálogo.** Las entradas marcadas con `VERIFY ONLINE` en su nota se escribieron sin comprobar el identificador en vivo; `pytest -m network` es la auditoría que las despeja. Un catálogo personalizado (`catalog=`) no redirige estos conectores (sección 2).
- **Los cartuchos** son un formato de transporte: el SHA-256 no está firmado, `verify=False` omite la comprobación por recodificación pero no la comparación de resúmenes al leer, y el archivo solo puede contener lo que el códec npz acepta (ningún objeto arbitrario de Python). Como señala el módulo `pocket`, las funciones de descarga necesitan sockets, así que en una sesión de Pyodide estos conectores solo pueden servir la caché o un cartucho — el cartucho es la vía de entrada prevista ([CACHE_DB.md](CACHE_DB.md) para la advertencia sobre la persistencia en IDBFS).
- **El cuaderno 58** usa un panel sintético (sección 4.5).

---

## Referencias

- Croushore, D., & Stark, T. (2001). "A Real-Time Data Set for Macroeconomists." *Journal of Econometrics*, 105(1), 111–130.
- Mankiw, N. G., & Shapiro, M. D. (1986). "News or Noise: An Analysis of GNP Revisions." *Survey of Current Business*, 66(5), 20–25.
- Banco de México. *SIE API REST*, https://www.banxico.org.mx/SieAPIRest/service/v1/ (solicitud de token: https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html).
- INEGI. *API de Indicadores (Banco de Indicadores / BIE)*, https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/.
- Banco Central do Brasil. *Sistema Gerenciador de Séries Temporais (SGS)*, endpoint https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json.
- Banco Central de Chile. *Base de Datos Estadísticos, SIETE REST*, https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx (registro: https://si3.bcentral.cl/estadisticas/principal1/registro/index.html).
