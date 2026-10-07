> 🇬🇧 [English](../trade_mrio.md) · 🇪🇸 Español

# Tablas MRIO nativas: procedencia, lectores, agregación y aranceles gruesos

`puremacro.trade.mrio` lee bases de datos insumo-producto multirregionales
publicadas a su resolución nativa, registra de dónde sale cada número, las
equilibra bajo un contrato explícito, las agrega de forma exacta y agrupa
esquemas arancelarios con las dos reglas usadas en el espacio de trabajo de
investigación IO. Es la capa de ingesta sobre la que se apoyan los
solucionadores de comercio: una tabla se convierte en la matriz de
calibración heredada con `to_calibration_matrix`, y una calibración vuelve a
ser tabla con `MRIOTable.from_trade_calibration`.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, load_icio_data
from puremacro.trade.mrio import (
    Concordance, MRIOTable, aggregate_mrio, coarse_tariff_rates,
    regularize_table, to_calibration_matrix,
)

calib = calibrate_trade_model(load_icio_data(source="legacy"))
table = MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
print(table.summary())
print(table.accounting_report().to_markdown())

blocs = Concordance.from_mapping(
    "blocs", {c: c if c in ("USA", "CHN") else "REST" for c in table.country_codes})
coarse = aggregate_mrio(table, Concordance.identity(table.sector_codes), region_concordance=blocs)
print(coarse.to_markdown("countries"))

three = Concordance.from_mapping(
    "three", {c: "GOODS" if k < 2 else "SERVICES" for k, c in enumerate(table.sector_codes)})
rates = np.zeros((77, 11))
rates[:, 1] = 0.25
rates[table.country_index("USA")] = 0.0
wedges = coarse_tariff_rates(rates, table, three, "bilateral")
print(wedges.summary())

matrix = to_calibration_matrix(table)          # (850, 1078): calibrate_trade_model(matrix) accepts it
```

La tabla incluida tiene tres celdas de inversión negativas (reducciones de
existencias que el formato heredado fundió con la formación bruta de
capital), y por eso el puente exige una política `negative_investment`
explícita. Los ficheros nativos se leen de la misma forma:

```python
# requires: the native OECD ICIO 2023, Eurostat FIGARO 2026 and EXIOBASE 3.8.2 files (not shipped)
from puremacro.trade.mrio import read_exiobase_native, read_figaro_native, read_oecd_native

oecd = read_oecd_native("2019_SML.csv", 2019)          # refuses the known-corrupted digest
figaro = read_figaro_native("matrix_eu-ic-io_ind-by-ind_26ed_2019.csv")
exio = read_exiobase_native("IOT_2019_ixi.zip")         # sparse Z, members checked against the archive CRCs
core = read_exiobase_native("exiobase_core", archive="IOT_2019_ixi.zip")  # extracted members checked too
balanced, report = regularize_table(oecd)
eleven = aggregate_mrio(balanced, "agregar")
print(report.to_markdown("gates"))
```

Bajo `tests/fixtures/trade/mrio/` se incluyen un fichero FIGARO de dos países
y un directorio EXIOBASE de dos regiones en los formatos oficiales, de modo
que `read_figaro_native("tests/fixtures/trade/mrio/figaro_2c2s_26ed.csv")`
funciona desde una copia del repositorio.

## El contrato de la tabla

Una `MRIOTable` es una dataclass congelada cuyos arrays son copias de solo
lectura. Las celdas van por país, `i = country_index * S + sector_index`.

| atributo | forma | significado |
|---|---|---|
| `Z` | `(M, M)` densa o `scipy.sparse.csr_matrix` | entregas intermedias a precios básicos; las filas venden, las columnas compran |
| `F` | `(M, N, K)` | entregas finales por celda vendedora, país de destino y categoría |
| `fd_codes` | `K` cadenas | subconjunto de `("C", "G", "X", "V", "VAL")` en ese orden |
| `VA`, `TLS`, `output` | `(M,)` | valor añadido tal como se registra, impuestos netos sobre las compras intermedias del comprador, producción bruta |
| `TFD` | `(N, K)` | impuestos netos sobre las compras finales |
| `production_taxes`, `labor_compensation`, `operating_surplus` | `(M,)` o `None` | detalle factorial cuando la fuente lo tiene |
| `merchandise_mask` | `(S,)` bool o `None` | sectores de bienes (CIIU A-C) |
| `metadata` | dict | `sources` (tupla de `SourceRecord`), `transformations` (tupla de cadenas), convenciones |

C es el consumo de hogares, ISFLSH y gobierno; G es la formación bruta de
capital fijo; X son las compras de residentes en el extranjero (la cesta
DPABR de la OCDE); V es la variación de existencias con signo; VAL son los
objetos valiosos. El constructor que valida es `MRIOTable.from_arrays`: `Z`,
G y X deben ser no negativos, todo debe ser finito, y el trabajo y el
excedente de explotación se dan juntos. El consumo, las existencias, los
objetos valiosos, el valor añadido y los impuestos negativos son datos reales
de la fuente y se conservan. `accounting_report()` recalcula de forma
independiente las identidades de fila y columna y cuenta las entradas
negativas; ahí no se reequilibra nada. `net_exports()` mide los saldos fob
sobre todos los usos y asigna el redondeo de coma flotante de su suma mundial
a `reference_country`. `to_raw("cix")` devuelve un `RawIOData` con las
columnas heredadas C, I = G + V + VAL, Cx = X.

## Procedencia

`identify_source(path, expected_sha256=..., expected_md5=...)` procesa el
fichero en bloques de 4 MiB y rechaza un fichero ausente o modificado.
`verify_zip_member(archive, member, extracted_path=None)` comprueba el tamaño
y el CRC-32 de un miembro contra el directorio del archivo. Cada lector
guarda los `SourceRecord` resultantes en `metadata["sources"]`, con un
`status` de `"verified"`, `"whitelisted"`, `"hashed"` o `"unknown_edition"`.

`check_oecd_source(path, year=...)` compara un CSV de la OCDE (o el miembro
del año del ZIP 2016-2020) con dos registros:

| registro | digest | significado |
|---|---|---|
| `OECD_ICIO_MD5[2019]` | `28cba31491177955445051d459053744` | `2019_SML.csv` aceptado, OCDE ICIO edición 2023 |
| `OECD_ICIO_MD5[2020]` | `d3e0f4979d85d6c0bb7cf4c43e324287` | `2020_SML.csv` aceptado |
| `OECD_KNOWN_CORRUPTED_MD5` | `d1b887aaafa54ab3f28fde78fcd21cdf` | `data_2020_SML.csv` con puntos decimales perdidos: los valores con 3-4 decimales perdieron el punto decimal y los inferiores a 0.001 pasaron a 0; se rechaza |

Un digest corrupto lanza `MRIOIntegrityError`; un digest de la lista blanca
se registra; cualquier otro se lee con un `RuntimeWarning` que lo declara
edición no autenticada. El lector FIGARO compara el CSV con
`FIGARO_2026ED_2019_SHA256` y el lector EXIOBASE compara el archivo con
`EXIOBASE_382_2019_IXI_SHA256` de la misma forma; ambos exigen un
`expected_sha256` cuando se pasa, y un fichero registrado leído para un año
distinto de 2019 lanza un error. Los miembros EXIOBASE leídos del archivo se
comprueban frente a sus entradas CRC-32. Un directorio EXIOBASE extraído solo
queda autenticado si se pasa su archivo (`archive=`): se calcula el hash del
archivo y cada miembro extraído se compara con el tamaño y el CRC-32 de su
entrada, como hace el cargador IO, de modo que un solo byte cambiado lanza un
error. Leído solo, los miembros del directorio se hashean, se registran como
`"unknown_edition"` y se señalan con un `RuntimeWarning`;
`metadata["source_authenticated"]` indica qué caso se aplicó. No se descarga
nada ni se genera un sustituto.

## Lectores y convenciones de las fuentes

| | OCDE ICIO edición 2023 | Eurostat FIGARO edición 2026 | EXIOBASE 3.8.2 ixi |
|---|---|---|---|
| lector | `read_oecd_native(path, year)` | `read_figaro_native(path, year=2019)` | `read_exiobase_native(path_or_zip, year=2019, archive=None)` |
| formato | CSV SML 77 x 45 con filas TLS/VA/OUT (lo analiza `puremacro.trade._oecd_icio.read_native`) | cabecera `rowLabels`, celdas `{region}_{industry}`, cinco columnas de uso final por región, seis filas finales `W2_*`; dimensiones derivadas de las etiquetas | `Z.txt`, `Y.txt`, `x.txt`, `industries.txt`, `satellite/F.txt`, `satellite/F_Y.txt` con cabeceras región/sector de dos niveles |
| unidades | millones de USD, precios corrientes | millones de EUR, precios básicos corrientes | millones de EUR, precios básicos corrientes |
| usos finales | C = HFCE + NPISH + GGFC, G = GFCF, X = DPABR, V = INVNT | C = P3_S13 + P3_S14 + P3_S15, G = P51G, X = 0, V = P5M | C = y01 + y02.a + y02.b, G = y04, X = 0, V = y05.a, VAL = y05.b |
| consumo | consumo residente con cesta DPABR separada; incluye gobierno | consumo territorial; incluye gobierno; los totales `W2_OP_RES` y `W2_OP_NRES` se guardan en metadata | consumo publicado; incluye gobierno; sin cesta DPABR |
| existencias | INVNT es la variación con signo; no se observa ningún stock | P5M funde variación de existencias y objetos valiosos; la fuente no los separa | existencias y objetos valiosos son flujos separados |
| detalle factorial | ninguno: la fila VA no tiene reparto trabajo/capital observado | `W2_D1` trabajo, `W2_D29X39` otros impuestos netos sobre la producción, `W2_B2A3G` excedente bruto de explotación y renta mixta | tres filas de trabajo por cualificación, cuatro filas de excedente (incluidas rentas de tierra y recursos), otros impuestos netos sobre la producción |
| producción | columna OUT del proveedor | total de fila de las transacciones publicadas | `x.txt` |
| año | 1995-2020 | 2019 autenticado | 2019, una estimación anticipada (nowcast) |
| almacenamiento | densa | densa (`sparse_matrix=True` opcional) | CSR en bloques de 128 filas |

Los códigos de región se convierten a ISO-3 con `FIGARO_ISO3` (`FIGW1` pasa
a `ROW`) y `EXIOBASE_ISO3` (las cinco regiones resto del mundo conservan
nombres distintos). Las máscaras de bienes siguen `goods_mask`: secciones
CIIU A-C para las industrias OCDE, NACE A-C salvo C33 (reparación e
instalación, que "no es un envío de mercancías") para FIGARO, industrias
i01-i37 salvo i01.w para EXIOBASE.

El lector FIGARO aplica, y registra en `metadata["rounding_adjustments"]`,
las dos reparaciones acotadas del lector IO: una celda vacía cuya producción
es negativa en una unidad de redondeo (`-0.001 <= y < 0` con masa absoluta de
insumos como máximo `0.002`) se pone a cero, y una entrada de consumo en
`[-0.001, 0)` se traslada a V de modo que los totales de fila y el gasto del
país no cambian. Cualquier cosa mayor lanza un error. `rounding_repair=False`
conserva el redondeo publicado. El lector EXIOBASE rechaza una columna
`Exports: Total (fob)` distinta de cero y filas factoriales de demanda final
distintas de cero, porque exigirían un modelo de destino explícito.

## Contrato de regularización

`regularize_table(table)` porta la construcción en ocho pasos de
`table_from_arrays` de IO a cualquier tabla con categorías nombradas. Con
`out` la columna de producción, `row` los totales de fila y `active = out > 0`:

1. Guarda de identidad: `median_{active} |row_i - out_i| / out_i <= 1e-5`,
   sin producción negativa, y cada celda de producción cero completamente vacía.
2. Fantasmas: cada celda de producción cero recibe `F[i, own country, C] = VA_i = 1e-6`
   para que su fila y su columna coincidan exactamente (`inactive="mask"` las deja vacías).
3. Suelo de valor añadido: `VA_i = 1e-3 * out_i` solo donde la celda está activa y `VA_i <= 0`.
4. Los impuestos netos sobre productos se recalculan como el residuo que cierra cada columna,
   `tls_j = row_j - sum_i Z_ij - VA_j`. En las celdas sin suelo el cambio debe cumplir
   `sum_j |tls_j - TLS_j| <= 1e-4 * sum_j |TLS_j|` y
   `max_j |tls_j - TLS_j| / max(out_j, 1) <= 1e-2` (con la holgura relativa `1e-9`
   de IO; la tabla OCDE de 2019 queda exactamente en 0.0100).
5. Los saldos comerciales se miden fob sobre todos los usos; `|sum_o xn_o| <= 1e-9 * world GDP`
   con `world GDP = sum VA + sum tls + sum TFD`.

La tabla devuelta tiene `output` igual a los totales de fila tras los
fantasmas, el `TLS` residual y el paso añadido a
`metadata["transformations"]`; el `MRIOBuildReport` recoge los fantasmas, los
suelos, ambos totales del residuo impositivo, el redondeo mundial de los
saldos, el PIB mundial y cada puerta con su valor, su tolerancia y su
resultado, y nombra las celdas infractoras cuando la puerta trata de celdas
(`"cells"`, `"n_cells"` y `"worst_cell"` en la puerta por celda). Cada
tolerancia es un argumento; `strict=False` devuelve el informe con
`passed=False` en lugar de lanzar un error (la tabla OCDE de 2020 de la lista
blanca falla la puerta por celda en el 1.06%). Esa tabla queda marcada con
`metadata["regularized"] = False`, las puertas fallidas en
`metadata["failed_gates"]` y en su cadena de transformación, y
`aggregate_mrio` y `to_calibration_matrix` emiten un aviso al recibirla. Con
`strict=False` solo reciben fantasma las celdas vacías con producción
exactamente cero; las celdas con producción negativa o entradas sueltas
conservan sus datos y se listan en `report.skipped_phantoms`. El fantasma
`1e-6` y la escala `max(out, 1)` de la puerta por celda son cantidades
absolutas: como el código IO, suponen una tabla en millones de unidades
monetarias, así que conviene reescalar `phantom` para otras unidades.
`spectral=True` añade las
cotas de Collatz-Wielandt de `Z / output` con
`puremacro.trade.regularize.compute_spectral_radius`. Es un contrato distinto
de `regularize_mrio_table`, que pone suelo al valor añadido de toda celda
activa en `max(1e-3 Y, min(1, 0.5 Y))` (4.3.0: `max(1e-3 Y, 1)`, que elevaba
el valor añadido por encima de la producción en celdas de menos de 1 M USD) y
recalcula la producción a partir de las ventas por fila antes del residuo salvo
que se pase `Y`; no son intercambiables.

## Agregación

`aggregate_mrio(table, concordance)` calcula la agregación exacta con
`P = I_N kron C^T`, donde `C` es la matriz 0/1 `(S, G)` de una `Concordance`:

```
Z_G = P Z P^T,   F_G[:, n, k] = P F[:, n, k],   VA_G = P VA,   TLS_G = P TLS,   output_G = P output
```

`TFD` no cambia, el detalle factorial se suma igual, y una
`region_concordance` funde además orígenes y destinos. Ambas identidades
contables son lineales, así que la agregación conmuta con el cierre residual
de impuestos siempre que no se aplique ningún suelo, y todos los totales se
conservan hasta el redondeo de coma flotante. Si la tabla fina está
equilibrada, la gruesa debe estarlo también
(`max_j |row_j - col_j| / max(|col_j|, 1) <= 1e-8`); el residuo se registra en
`metadata["aggregation"]` en cualquier caso. Un grupo grueso es de bienes solo
si todos sus miembros lo son; los grupos mixtos se listan.

En `CONCORDANCES` hay dos mapas con nombre de las 45 industrias OCDE:

| nombre | grupos | grupos de bienes |
|---|---|---|
| `"agregar"` | `AGG11_SECTOR_CODES`: AGRI_MIN_FOOD (A01_02 a C10T12), MANUF_EXFOOD (C13T15 a C31T33), UTILITIES (D, E), CONSTRUCTION, TRADE, TRANSPORT (H49-H53), ACCOM_ICT (I, J58T60-J62_63), FIN_REALESTATE (K, L), PROF_ADMIN_PUBADM (M, N, O), EDU_HEALTH (P, Q), ARTS_OTHER (R, S, T) | AGRI_MIN_FOOD, MANUF_EXFOOD |
| `"isic_section"` | `ISIC_SECTION_11_CODES`: A, B, C, DE, F, G, H, J, KL, OPQ, REST (I, M, N, R, S, T) | A, B, C |

El mapa Agregar es el que se usó para construir la tabla 77 x 11 incluida.
Su primer grupo contiene la industria alimentaria y su tercer grupo son los
suministros, de modo que los `CANONICAL_SECTOR_CODES` de puremacro (AGRI,
MINQ, MANU, ENEG, ...) etiquetan esas posiciones pero no describen su
contenido; `goods_mask` rechaza esas etiquetas salvo que se nombre la
clasificación (`"agregar11"` para los grupos reales, `"canonical11"` para
leer las etiquetas literalmente). La máscara se decide etiqueta a etiqueta:
con `"agregar11"` cada etiqueta canónica representa el grupo Agregar de su
posición en `CANONICAL_SECTOR_CODES`, de modo que cualquier orden o
subconjunto da la misma respuesta por código, y un código fuera del registro
nombrado lanza `ValueError`.

## Reglas de aranceles gruesos

`coarse_tariff_rates(fine_rates, table, concordance, rule)` toma un esquema
`(N, S)` de derechos ad valorem `r_{os} >= 0` (multiplicador `tau = 1 + r`)
que el importador aplica al origen `o` y sector `s`, y las compras de
referencia del importador a precios básicos
`m_{os} = sum_{j in importer} Z_{(o,s),j} + sum_{f in fd_tariffed} F_{(o,s),importer,f}`
con `fd_tariffed = ("C", "G", "V", "VAL")` por defecto, de las que se usan
las categorías que lleva la tabla. Los objetos valiosos se gravan con las
existencias, como en la construcción EXIOBASE de IO (que los fundía en V) y
en FIGARO (cuyo P5M los contiene). Las tasas deben ser no negativas
(`ValueError` en otro caso), X nunca se grava y la fila propia del
importador es cero.

* `rule="output"` (el `coarse_11o` de IO): `r_{oG} = sum_{s in G} m_{os} r_{os} / sum_{s in G} m_{os}`,
  cero cuando la suma de pesos no es positiva. Con pesos no negativos la tasa
  gruesa queda entre la menor y la mayor tasa fina del grupo.
* `rule="bilateral"` (el `coarse_11b` de IO): equivalentes arancelarios por comprador
  `tau_{(o,G),(imp,J)} = 1 + sum_{s in G, j in J} Z_{(o,s),j} r_{os} / sum_{s in G, j in J} Z_{(o,s),j}`
  y `tau^f_{(o,G),imp} = 1 + sum_{s in G} F^f_{(o,s),imp} r_{os} / sum_{s in G} F^f_{(o,s),imp}`
  por categoría gravada, unos donde el denominador no es positivo. Estos
  preservan la recaudación de referencia bloque a bloque allí donde las compras
  de referencia del bloque son positivas; un bloque con compras no positivas
  (una reducción de existencias con `negative_weights="keep"`) conserva un
  multiplicador unitario y pierde su derecho del esquema fino, exactamente como
  la guarda `tot > 0` de la regla IO. La regla output preserva también el total de
  cada grupo de origen, porque sus pesos son exactamente las compras gravadas; lo
  que pierde es el reparto entre grupos compradores y categorías.

Las existencias con signo pueden hacer negativo un peso. Las reglas IO usan
las sumas brutas (`negative_weights="keep"`, el valor por defecto) y solo
protegen los denominadores; `negative_weights="clip"` recorta en cero cada
contribución de uso final, porque una reducción de existencias no es una
importación. El `CoarseTariffResult` lleva las tasas, los pesos, las cuñas
`tau` `(N G, N G)` y `tau_fd` `(N G, N, K)`, y la recaudación de referencia
con el esquema fino y con el grueso.

## Puentes con la calibración heredada

`to_calibration_matrix(table)` devuelve la matriz `(M + 3, M + 3N)` para
`calibrate_trade_model`: las filas `0..M-1` son `[Z | F]` con C, I = G + V +
VAL y Cx = X por destino, la fila `M` es TLS con los impuestos finales
condensados igual, y las filas `M + 1` y `M + 2` son trabajo y capital.
`calibrate_trade_model` reconstruye cada columna con un reparto Cobb-Douglas
y solo la acepta si ambos factores son estrictamente positivos (una
participación del capital estrictamente entre 0 y 1) o si ambos son cero, así
que las filas factoriales se construyen con ese criterio:

* Los otros impuestos netos sobre la producción pasan a la fila TLS por
  defecto (`production_taxes="to_tls"`), que la calibración heredada modela
  como un impuesto sobre la producción, salvo en las celdas donde son al
  menos tan grandes como el valor añadido. Allí se quedan en el valor
  añadido y un `RuntimeWarning` nombra las celdas: 3 celdas de la tabla
  FIGARO 2019 regularizada (ARG_A01, CYP_H51, JPN_M73) y 30 de EXIOBASE
  2019, 9 de las cuales una resta directa habría dejado con renta factorial
  negativa. `"to_factors"` deja todos los impuestos sobre la producción
  dentro del valor añadido.
* Trabajo y capital reparten el valor añadido factorial restante en las
  proporciones observadas donde ambos son estrictamente positivos, y
  2/3-1/3 en el resto: celdas fantasma, celdas con un factor negativo y
  celdas con un factor exactamente cero (la tabla FIGARO 2019 tiene 30
  celdas activas con excedente de explotación cero y 7 con trabajo cero).
  Sin detalle, cada celda usa 2/3-1/3.
* Una celda con valor añadido factorial negativo lanza `MRIOIntegrityError`;
  `regularize_table` pone suelo al valor añadido donde no es positivo.

`return_details=True` devuelve además las celdas de cada caso por defecto.
Una tabla sin regularizar dispara un `RuntimeWarning` que recomienda
`regularize_table` primero, y lo mismo una tabla cuyas puertas de
regularización fallaron. `MRIOTable.from_trade_calibration(calib)` es el
puente inverso desde `calib.data_calibra`. Lee la columna Cx de la calibración
como X (compras de los residentes en el exterior, exentas de aranceles) solo
para la disposición incluida y el mapeo `DPABR` de la OCDE
(`calib.metadata["final_use_mapping"]`); el Cx de los cargadores FIGARO, WIOD
y Eora (existencias con signo y objetos valiosos) pasa a V, y el Cx de
EXIOBASE, que incluye su columna de exportaciones, lanza `ValueError` salvo
que `cx_category` nombre la categoría. Una celda de inversión negativa que
`negative_investment="to_inventory"` traslada se suma a V. Una tabla obtenida
de una calibración de FIGARO, WIOD o Eora no tiene por tanto X, y
`to_calibration_matrix` la devuelve con la columna Cx vacía: resuelva esa
calibración con `accounting="consistent"`, porque la demanda heredada
`theta * Y / ppfd` es 0/0 para una categoría vacía (el solver heredado
devuelve NaN con `converged=False`).

## Procedencia de la tabla 77 x 11 incluida en puremacro

El primer hecho se identificó en el directorio de la cadena MATLAB
`computation/7_TIO_77c_vf` a partir del fichero y su MD5 (este módulo se
niega a leer ese fichero, así que no puede volver a derivarse aquí); las
comparaciones que siguen se midieron el 2026-09-22 con este módulo.

* El fichero sin agregar que la cadena MATLAB original agregó a
  `puremacro/trade/_datafiles/icio_77c_11s.npz` es `data_2020_SML.csv` con
  MD5 `d1b887aaafa54ab3f28fde78fcd21cdf`, el digest registrado en
  `OECD_KNOWN_CORRUPTED_MD5`: los valores con tres o cuatro decimales
  perdieron el punto decimal y los inferiores a 0.001 pasaron a cero.
  `load_raw_45sector_icio` comprueba ahora el MD5 del fichero que resuelve y
  rechaza este digest con `MRIOIntegrityError`; su búsqueda por defecto prueba
  los ficheros limpios `ICIOextended/2020_SML.csv` y `2019_SML.csv` antes de
  esa ruta heredada.
* El valor añadido mundial de la tabla incluida es 7.046e11 millones de USD;
  la edición OCDE 2020 de la lista blanca agregada con el mismo mapa Agregar
  da 7.971e7, un factor de 8,840. Su mayor celda intermedia es 5.83e10 (la
  manufactura de CHN consigo misma) frente a 6.79e6 en la edición limpia, y
  962 celdas superan 1e8 frente a ninguna. De las 717,409 celdas
  intermedias, el 35.5% son iguales en ambas tablas (el 25.5% de las 621,488
  celdas no nulas en al menos una: los valores con dos decimales o menos
  sobreviven intactos a la corrupción); la fila de valor añadido coincide en
  1 de 847 celdas (0.12%) y el bloque de demanda final en el 37.7% de sus
  celdas.
* Las soluciones de referencia MATLAB en `trade_reference_solutions.npz` se
  calcularon con la misma tabla corrupta, así que las baterías de paridad que
  comparan puremacro con ellas siguen siendo regresiones de software
  internamente consistentes: comprueban que el solucionador reproduce un
  cálculo fijo, no que el cálculo describa la economía mundial de 2020.
* El trabajo empírico nuevo debe leer los ficheros nativos limpios con
  `read_oecd_native`, `regularize_table` y `aggregate_mrio`. Regenerar la
  tabla incluida es una decisión del autor que rompería el contrato de
  paridad externa, y este módulo no lo hace.

## Qué está validado y qué no

`tests/test_trade_mrio.py` comprueba, sobre fixtures diseñados a mano con
arrays esperados derivados a mano: los auxiliares de procedencia frente a
`hashlib` y frente a miembros corruptos o discordantes; el álgebra de
concordancias (la proyección coincide con el producto de Kronecker explícito,
composición, máscaras gruesas); el contrato de la tabla portado del fixture
unitario de IO; los lectores FIGARO y EXIOBASE, incluidas las reparaciones de
redondeo, las rutas zip, de directorio comprobado contra su archivo y por
bloques, un byte cambiado en un miembro extraído, y cada rechazo; el lector
OCDE sobre un fichero diminuto en formato SML; cada puerta de regularización
en modo estricto y de informe, y que una puerta fallida nunca produce una
tabla marcada como regularizada; conservación, composición, agregación
regional y la coherencia de bienes 45/11; convexidad y preservación bloque a
bloque de la recaudación de las reglas gruesas; la calibración del fixture
OCDE agregado con una solución de referencia a precios uno; la calibración de
tablas con celdas de factor cero y con impuestos sobre la producción mayores
que el valor añadido; y el viaje de ida y vuelta del puente con la tabla
incluida.

La paridad con la implementación IO se ejercita en un subproceso que importa
el paquete `puremacro.trade.corrected` vendorizado (se omite si el volumen no
está montado): `regularize_table` frente a `table_from_arrays`,
`aggregate_mrio` frente a `aggregate_icio` para ambos mapas, y ambas reglas
gruesas frente a `coarse_11o` y `coarse_11b`, con diferencias observadas de 0
(2.8e-17 en las tasas 11o). Las pruebas lentas leen los ficheros reales de
2019 y reproducen las auditorías IO registradas: OCDE 3,440 celdas activas, 25
fantasmas, ningún suelo, identidad de fila mediana 3.99e-7, PIB mundial
87,854,087; FIGARO 50 x 64, 3,133 celdas activas, 20,342 entradas de
existencias negativas, la única reparación de redondeo AL_T, 67 fantasmas, 9
suelos, PIB mundial 79,054,914; EXIOBASE 49 x 163, 1,084 celdas de producción
cero, 73 celdas factoriales no positivas, existencias 1,100,405.96, objetos
valiosos 62,203.98, 1,084 fantasmas, 89 suelos, PIB mundial 78,110,997, con
el núcleo extraído autenticado frente al archivo registrado. Los lectores
coinciden con `dynamic_model.native_data` y
`calibrate_databases_2019.figaro` hasta el orden de suma en coma flotante: 0
en todos los arrays OCDE y EXIOBASE y frente a
`calibrate_databases_2019.figaro`, como mucho 9.3e-10 (la producción FIGARO
frente a `native_data.load_figaro`). La tabla FIGARO 2019 regularizada
(50 x 64) pasa `calibrate_trade_model`.

No validado, y no afirmado:

* Equilibrio general a tamaño completo sobre FIGARO o EXIOBASE. La tabla
  FIGARO 2019 regularizada se calibra, pero aquí no se resuelve ningún
  equilibrio sobre ella; EXIOBASE no se calibra en absoluto, porque
  `to_calibration_matrix` densifica `Z` (7,987 al cuadrado). Este módulo
  promete tablas a resolución nativa, no equilibrios a resolución nativa.
* Comparabilidad del bienestar entre bases de datos. Las calibraciones IO
  registran `residence_welfare_comparable: False`; FIGARO: "Territorial
  consumption is retained; a residence bridge has not been validated"; "This
  2019 release is a nowcast; no stronger measurement claim is implied" para
  EXIOBASE; y "active_mask means positive reported output; it is not an
  economic feasibility certificate."
* Otras ediciones y años. Los digests identifican los ficheros de 2019 que se
  auditaron; una edición distinta se lee con un aviso y sin ninguna afirmación.
* Un lector GTAP. El único analizador que decodifica ficheros HAR reales de
  GTAP en el espacio de trabajo IO es el paquete HARPY vendorizado, con
  licencia GPL-3 y por tanto no redistribuible bajo la licencia MIT de
  puremacro; el lector en Python puro de `headlinePaper/load_gtap11.py` solo
  hace el viaje de ida y vuelta con su propio escritor; y los propios datos
  GTAP tienen licencia y no pueden incluirse ni como fixtures.
* La puerta IO del residuo impositivo por celda es frágil por construcción:
  2019 queda exactamente en el umbral 1e-2 y el fichero limpio de 2020 la
  falla en el 1.06%. Las tolerancias son argumentos y el informe se devuelve
  en ambos casos.

## Documentos relacionados

* [Contabilidad de comercio consistente](trade_accounting.md) para el
  solucionador que alimenta la matriz de calibración.
* [Bienestar hicksiano del consumo](trade_welfare.md): el C nativo de la
  OCDE incluye el consumo del gobierno.
* [Economía espacial cuantitativa y equilibrio general de comercio](spatial_and_trade_ge.md)
  para los cargadores heredados y los adaptadores armonizados.
* [Estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
