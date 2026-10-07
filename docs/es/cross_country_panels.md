> 🇪🇸 Español · 🇬🇧 [English](../cross_country_panels.md)

# Paneles macroeconómicos entre países

Diez módulos de `puremacro.fetch` convierten toda la sección cruzada de un proveedor en un solo DataFrame con una sola llamada. Comparten un contrato:

- **Forma.** Un marco ancho indexado por `(code, date)`: códigos ISO3 y `date` al inicio del periodo (1 de enero, primer día del trimestre o del mes). Una columna por variable, con los nombres que se usan en el resto de puremacro (`gdp`, `cons_hh`, `inv`, `urate`, `cpi`, ...).
- **Metadatos en `attrs`.** `meta` (por columna, o por país y variable: flujo, unidades, cobertura), `source`, `fetched_at` (UTC) y `missing` (lo que se pidió y no se obtuvo, con el motivo). Cada módulo tiene un auxiliar `*_meta(panel)` que muestra `meta` como tabla.
- **Nunca lanza una excepción por una falla del proveedor.** Un HTTP 429, un tiempo de espera agotado, un cuerpo truncado o una API caída producen una advertencia y un marco vacío o parcial; el motivo queda en `attrs["missing"]`.
- **Solo urllib.** Ningún módulo importa `requests`, así que funcionan donde no está instalado y se importan sin problema en un iPad o bajo Pyodide.
- **Caché en disco** bajo `~/.cache/puremacro` (se cambia con `PUREMACRO_HTTP_CACHE_DIR`), nunca dentro del repositorio. Una nueva ejecución solo reenvía lo que falló. Pase `refresh=True` para obtener la publicación más reciente.
- **Ritmo de la OCDE.** Toda solicitud a la OCDE pasa por `puremacro.fetch._oecd_sdmx.oecd_csv`, que espacia las llamadas (5 s por omisión), reintenta ante HTTP 429 y guarda en caché la respuesta `csvfile`.

Las cifras de cobertura de esta página se midieron en vivo el 7 de octubre de 2026.

## WDI del Banco Mundial: `wdi_panel` (`puremacro.fetch.wdi`)

`wdi_panel(indicators=None, codes=None, *, start=1960, end=None, real=True, long=False, refresh=False, pause=1.0, timeout=120.0)` es el panel anual gratuito más amplio y más largo: los Indicadores del Desarrollo Mundial para las 217 economías que lista el Banco Mundial (territorios incluidos), desde 1960.

- **Unidades.** Niveles monetarios en millones de moneda local corriente, personas en miles, tasas e índices tal como se publican.
- **Términos reales y deflactores.** `real=True` agrega `<name>_real` (moneda local constante al año base propio de cada país, véase `wdi_meta`) y `<name>_defl = 100 * nominal / real`, que se conserva solo donde ambos niveles son positivos (las celdas enmascaradas se cuentan en `n_masked`). Las partidas que cambian de signo (`inventories`, `stat_disc`, `taxes_prod`) no reciben deflactor. El deflactor del PIB reproduce el del propio WDI con una diferencia de 1e-15. Los componentes `_real` encadenados no suman.
- **Indicadores.** `None` descarga `WDI_CORE` (cuentas nacionales, población, el bloque laboral de la OIT, PPA, tipo de cambio, IPC y crédito privado: 35 nombres, 51 solicitudes, unos 2,5 minutos sin caché). `"all"` descarga todo el registro `WDI_INDICATORS` (52 nombres). Un diccionario `{"name": "WDI.CODE"}` agrega cualquier código crudo, sin reescalar.
- **attrs.** `meta` tiene un diccionario por columna (indicator, units, unit_mult, first, last, n, n_codes, `lastupdated`). Los cuerpos de error que llegan con HTTP 200 se vuelven a pedir una vez, para que una respuesta defectuosa no se sirva desde la caché de 30 días.
- **Auxiliares.** `wdi_countries()` (región, grupo de ingreso) y `wdi_meta()` (moneda, año base, versión del SCN, notas sobre años fiscales).
- **Lo que no está en WDI.** Bienes durables, FBCF pública (solo `inv_priv`), acervos de capital, horas, un nivel de empleo (use `lf * (1 - urate/100)`), remuneraciones de toda la economía y nada subanual.

```python
from puremacro.fetch.wdi import wdi_panel
p = wdi_panel(["gdp", "pop"], ["MEX", "USA"], start=1960)
```

Fuente: WDI del Banco Mundial, API v2, fuente 2 (CC BY 4.0).

## Cuentas nacionales anuales de la OCDE: `ana_panel` (`puremacro.fetch.oecd_ana_panel`)

`ana_panel(codes=None, *, start="1950", assets=False, durability=False, income=False, output=False, labor=False, sectors=False, stocks=False, real=True, long=False, refresh=False)` es el gemelo anual de `qna_panel`.

```python
from puremacro.fetch import ana_panel, ana_countries, ana_panel_meta
p = ana_panel(["USA", "FRA", "MEX"], start="1950", income=True, labor=True)
p.loc["FRA", ["gdp", "gdp_defl", "gdp_real", "mixed_income", "hours"]]
```

- **Dinero** en millones de moneda nacional a precios corrientes.
- **`<x>_defl`** es `100 * V / LR` (2020 = 100), exactamente el deflactor que publica la OCDE. **`<x>_real`** es el volumen encadenado a precios de 2020: `LR` donde se publica; si no, `L` rebasado con `V/L` de 2020. No es aditivo fuera de 2020.
- **Una tabla por serie.** Cada (país, variable, base de precios) se lee completa de T0102, si no de T0101, si no de T0103; `meta["tables_not_t0102"]` lista las excepciones (hoy solo el volumen del PIB de Rusia).

Con `codes=None` el panel tiene 64 países: 6 empiezan antes de 1960, 12 antes de 1970, 34 antes de 1980 y 46 antes de 1995.

| Interruptor | Columnas | Fuente |
|---|---|---|
| (siempre) | gdp, cons_hh, cons_gov, inv, capform, exports, imports, discrepancy_exp | DSD_NAMAIN10 DF_TABLE1 (T0102) |
| `income` | comp_emp, surplus_mixed, taxes_prod_imp, subsidies, taxes_prod_imp_net, discrepancy_inc, gdp_income; surplus_gross, mixed_income, cfc, vat | DF_TABLE1 (T0103); DSD_NASEC10 DF_TABLE14 |
| `output` | gdp_output, va_total, va_agri, va_ind, va_mfg (partida informativa), va_constr, va_trade, va_ict, va_fin, va_realest, va_prof, va_public, va_other, va_services, taxes_prod, discrepancy_out | DF_TABLE1 (T0101) |
| `durability` | cons_dur, cons_semidur, cons_nondur, cons_serv (hogares, S14) | DF_TABLE5A_T117, si no DF_TABLE5_T117 |
| `assets` | inv_dwell, inv_struct, inv_equip, inv_transp, inv_ict, inv_othmach, inv_bio, inv_ipp | DF_TABLE1 (T0102) |
| `sectors` | inv_gov, inv_corp, inv_fin, inv_hh (S14, si no S1M) | DSD_NASEC10 DF_TABLE14_GFCF |
| `stocks` | k_net, k_gross (con volúmenes), k_net_gov | DF_TABLE9A; DSD_NASEC10 DF_TABLE9B |
| `labor` | emp, emp_employees, emp_selfemp, hours, hours_employees, hours_selfemp (cada una también `_agri`, `_public`); pop | DF_TABLE3_EMPDC; DF_TABLE3_POP_EMPNC |

Ingreso, sectores, `k_net_gov`, trabajo y población son bloques a precios corrientes o de conteo y no reciben deflactor.

`attrs`: `meta` (un diccionario por país, que se lee con `ana_panel_meta`: moneda, años, base de volumen, tabla de durabilidad, sector de `inv_hh`, escala de horas, columnas ausentes), `variables` (un diccionario por columna: flujo, unidades, descripción, cobertura, claves SDMX), `missing` (fallas que una nueva ejecución puede corregir: 429, tiempos agotados, cuerpos que no son SDMX, países sin PIB), `chunks_empty` (404 o respuestas vacías que una nueva ejecución no cambia) y `requests` (cada solicitud con su estado y número de filas). Las solicitudes van en bloques de a lo sumo 10 países; el principal, activos, ingreso y producción comparten una sola solicitud a DF_TABLE1.

Fuente: OECD SDMX, agencia OECD.SDD.NAD.

## Bloques trimestrales de la OCDE junto a `qna_panel` (`puremacro.fetch.oecd_qna_extras`)

Cuatro funciones, cada una una sola llamada, y todas se unen con `qna_panel` por el índice.

| función | columnas | unidades | cobertura |
|---|---|---|---|
| `qna_sector_gfcf(codes=None, *, start="1947", sa="x13")` | `inv_gov`, `inv_priv`, `inv_gov_real`, `inv_priv_real` | millones de moneda nacional | 34 países, EUA desde 1947T1; volúmenes para 11 |
| `qna_population(codes=None, *, start="1947")` | `pop`, `emp_nc`, `emp_employees_nc`, `emp_selfemp_nc` | miles de personas | pop 43 (EUA desde 1947), empleo 40 (desde 1980) |
| `oecd_lfs_panel(codes=None, *, freq="Q", start="1950")` | `urate`, `urate_1564`, `emp_lfs`, `lf`, `wap`, `wap_1564`, `prate_1564`, `epop_1564` | miles / porcentaje | 45 países desde 1955; `freq="M"` para mensual |
| `oecd_vacancies(codes=None, *, freq="Q", start="1950")` | `vacancies`, `vacancies_new`, `reg_unemp` | miles de personas | 18 con vacantes (DEU desde 1955), 29 con desempleo registrado |

- **Fuentes.** La inversión pública viene de `DF_QNA_EXPENDITURE_GFCF_SECTOR` (S13, S1W) donde se publica; si no, de las cuentas sectoriales trimestrales `DF_QSA` (privada = S1 menos S13); `attrs["meta"]["source"]` indica cuál. `urate` es la tasa armonizada (`UNE_LF_M`), o la tasa de la encuesta (`UNE_LF`) para BRA, RUS, ZAF y algunos otros. Las vacantes son solo niveles (`TRANSFORMATION=_Z`).
- **Ajuste estacional.** Se toman las series ajustadas de la OCDE donde se publican. `qna_sector_gfcf(sa="x13")` ajusta el resto con la ruta X-13 de `qna_panel` (`sa="prefer"` las deja sin ajustar); los demás bloques recurren a la serie sin ajustar. `attrs["meta"]["sa"]` registra `oecd`, `puremacro`, `none` o `mixed` por país y variable.
- **Fallas.** `attrs["missing"]` lista cada par (país, variable) no obtenido, con `"not published"` o, por ejemplo, `"HTTP 429 on <flow>"`; `attrs["requests"]` registra cada solicitud, incluida la de disponibilidad. `qna_extras_meta(panel)` tabula `meta`. Todas las llamadas aceptan `refresh=True`.

```python
from puremacro.fetch import qna_panel, qna_countries
from puremacro.fetch.oecd_qna_extras import (qna_sector_gfcf, qna_population,
                                             oecd_lfs_panel, oecd_vacancies)
codes = qna_countries()
quarterly = qna_panel(codes).join([qna_sector_gfcf(codes), qna_population(codes),
                                   oecd_lfs_panel(codes), oecd_vacancies(codes)], how="outer")
```

## Indicadores mensuales de corto plazo de la OCDE: `stes_panel` (`puremacro.fetch.oecd_stes_panel`)

`stes_panel(codes=None, *, concepts=None, start="1950", refresh=False, pause=5.0, retries=3, retry_sleep=90.0)` devuelve 31 conceptos mensuales en niveles, tal como los publica la OCDE: índices (2015 = 100; las encuestas y el CLI, promedio de largo plazo = 100), porcentajes, moneda local por USD y personas en miles. Los logaritmos, las tasas de crecimiento y el ajuste estacional de las series sin ajustar (IPC, dinero, `urate_nsa`) quedan a cargo del usuario. El valor por omisión `start="1950"` recorta la producción industrial de EUA (desde 1919) y el IPC de Canadá (desde 1914); pase `start="1900"` para la historia completa.

```python
from puremacro.fetch import stes_panel, stes_meta
m = stes_panel(start="1900", pause=8.0)            # every economy, all 31 concepts, full history
m = stes_panel(["USA", "MEX"], concepts=["ip", "cpi", "rate_3m", "urate"])
stes_meta(m)[["concept", "unit", "n_countries", "first", "last", "stale_countries"]]
```

| grupo | conceptos | flujo | cobertura |
|---|---|---|---|
| producción | ip, ip_mfg, ip_constr, retail_vol (desestacionalizados) | STES `DF_INDSERV` | ip 41 economías, EUA desde 1919 |
| mercados | rate_on, rate_3m, rate_10y (% anual), share_price, reer_cpi | STES `DF_FINMARK` | 43-47 |
| tipo de cambio | fx_usd (moneda local/USD; miembros del euro empalmados al EUR) | STES `DF_KEI` | 46, sin EUA |
| encuestas | bci, cci, cli (ajustado por amplitud) | STES `DF_CLI` | 46 / 41 / 17 |
| dinero | m1, m3 (índice sin ajustar) | STES `DF_MONAGG` | 29; ningún miembro individual del euro (pida `"EA20"`) |
| trabajo | urate, urate_nsa; emp, lf, prate, epop | TPS `DF_IALFS_UNE_M`, `DF_IALFS_INDIC` | 39; unos 10 para niveles mensuales de la encuesta |
| registros | vacancies, reg_unemp (miles, desestacionalizados) | TPS `DF_OIALAB_INDIC` | 17 / 27 |
| precios | cpi, cpi_food, cpi_energy, cpi_core; hicp, hicp_food, hicp_energy, hicp_core (sin ajustar) | TPS COICOP 2018, luego COICOP 1999 | 46/45/39/40; 30 cada uno |

- **Precios al consumidor.** Los países de la UE/EEE pasaron a COICOP 2018 en enero de 2026 y sus series COICOP 1999 terminan en 2025-12. Cada columna de precios toma primero COICOP 2018 y completa con COICOP 1999 mes a mes, empalmando por razón los meses anteriores sobre el nivel de 2018. Las brechas solo son grandes donde se redefinió un agregado (energía de BGR 13%, subyacente de NLD 5%, subyacente de BGR 4%). El IAPC toma el flujo 1999 solo para la UE, CHE, GBR, ISL, NOR y TUR. `stes_meta(m)` lista `countries_fallback`, `splice_factors` y `stale_countries` (series que terminan más de seis meses antes: IPC nacional de AUT, POL y SVK en 2025-12 y de RUS en 2022-03, subyacente de ZAF en 2024-12, IAPC de ISL).
- **Fallas.** Los conceptos que fallan van a `attrs["missing"]`; un país pedido sin datos aparece como `"concept:CODE"`. Si falla la solicitud COICOP 2018 de una columna de precios, la columna se descarta en lugar de construirse solo con 1999; si solo falla la de respaldo, se conserva y se marca (`"cpi@DF_PRICES_ALL"`). Mientras la OCDE limita el tráfico, una llamada completa puede bloquearse entre 30 y 45 minutos; `retries=1` falla rápido.

Fuente: API SDMX de la OCDE (`sdmx.oecd.org`).

## Eurostat: `eurostat_na_panel` y `eurostat_monthly_panel` (`puremacro.fetch.eurostat`)

Las cuentas nacionales europeas más largas (Francia y Noruega desde 1975, cuentas sectoriales francesas desde 1949), los informantes de los Balcanes y de la Asociación Oriental, e indicadores mensuales que faltan en los flujos de la OCDE.

- `eurostat_na_panel(codes=None, *, freq="A", start=None, durability=False, assets=False, output=False, income=False, labor=False, sectors=False, capital=False, real=True, sa="prefer", refresh=False, timeout=120.0)`
- `eurostat_monthly_panel(codes=None, *, start=None, variables=None, refresh=False, timeout=120.0)`
- Piezas básicas: `eurostat_get(flow, key, ...)`, `eurostat_codes(flow)`, `eurostat_meta(panel)`.

```python
from puremacro.fetch.eurostat import (eurostat_na_panel, eurostat_monthly_panel,
                                      eurostat_meta, eurostat_get, eurostat_codes)

a = eurostat_na_panel()                     # 41 countries, 1975-2025, gdp ... imports + _real + _defl
q = eurostat_na_panel(freq="Q")             # 39 countries, SCA > SA > NSA per country
full = eurostat_na_panel(["DEU", "FRA"], durability=True, assets=True, output=True,
                         income=True, labor=True, sectors=True, capital=True)
m = eurostat_monthly_panel(variables=["cpi", "cpi_core", "urate"])
eurostat_meta(full)                         # one row per country and variable
```

| bandera | columnas | flujo (anual / trimestral) |
|---|---|---|
| (siempre) | gdp cons_hh cons_gov inv capform exports imports | nama_10_gdp / namq_10_gdp |
| income | comp_emp surplus_mixed taxes_prod_imp subsidies taxes_products subsidies_prod taxes_prod; anual: mixed_income surplus_gross cfc | nama_10_gdp + nasa_10_nf_tr / namq_10_gdp |
| durability | cons_dur cons_semidur cons_nondur cons_serv | nama_10_fcs / namq_10_fcs |
| assets | inv_dwell inv_struct inv_constr inv_equip inv_ipp | nama_10_an6 / namq_10_an6 |
| output | va_total va_services va_agri va_ind va_mfg va_constr va_trade va_info va_fin va_realestate va_business va_public va_other | nama_10_a10 / namq_10_a10 |
| labor | emp emp_employees emp_selfemp (miles) hours (millones) pop | nama_10_a10_e + nama_10_pe / gemelos namq_ |
| sectors | inv_corp inv_fin inv_hh inv_gov | nasa_10_nf_tr + gov_10a_main / nasq_10_nf_tr + gov_10q_ggnfa |
| capital (anual) | k_net k_dwell k_struct k_equip k_ipp k_gross | nama_10_nfa_st |

- **Unidades.** Millones de moneda nacional; para los miembros del euro, euros en toda la historia, de modo que no hay quiebre en la adopción. `_real` son volúmenes encadenados con año de referencia 2020 (2015 donde 2020 no se publica, p. ej. el Reino Unido); `_defl` vale 100 en el año de referencia.
- **Columnas siempre presentes.** El marco trae cada columna pedida, toda NaN si no llegó nada, así que `df["gdp"]` nunca lanza una excepción.
- **Fallas** (tiempo agotado, 5xx, conexión cortada, descarga truncada, trabajo asíncrono vencido) producen un marco vacío, una advertencia y el motivo en `missing`. Cada solicitud deja abierta la dimensión de país, así que cualquier subconjunto de países reutiliza la misma descarga en caché.

## FMI: `puremacro.fetch.imf`

El FMI sirve sus datos en `api.imf.org/external/sdmx/2.1`. El antiguo servidor de IFS (`dataservices.imf.org`) ya no existe y `sdmxcentral.imf.org` responde 501 a las consultas de datos. Todas las funciones de abajo aceptan además `refresh=False`.

| función | devuelve | cobertura |
|---|---|---|
| `imf_nea_panel(codes=None, *, freq="A"\|"Q", sa="SA", real=True, start=None)` | `gdp, cons, cons_gov, cons_hh, cons_priv, inv, capform, inventories, exports, imports` + `_real`, `_defl` | A: 188 economías, 1950-2026; Q SA: 66 |
| `imf_monthly_panel(codes=None, *, variables=None, start=None)` | `cpi, cpi_food, cpi_housing, cpi_transport, ip, ip_sa, ip_mfg, urate, emp, lf, policy_rate, mm_rate, tbill_rate, bond_yield, discount_rate, deposit_rate, lending_rate, neer, reer, fx_usd, fx_usd_eop` | IPC: 191 economías desde 1914; fx: 222 desde 1940 |
| `imf_labour_panel(codes=None, *, freq="A"\|"Q"\|"M", start=None)` | `urate, emp, lf, unemp` | A: emp 190, lf 189 |
| `imf_pcps(*, freq="M", start=None)` | `pcom_all, pcom_energy, poil, pcopper, ...` + `<code>_idx` / `<code>_usd` | 188 series desde 1992, código `WLD` |
| `imf_weo(codes=None, *, indicators=None, start=None, vintage=None, history_only=False)` | 17 series nombradas del WEO + indicador `forecast` | 197 economías, 1980-2031 |
| `imf_icsd(codes=None, *, start=None)` | `inv_gov, inv_priv, k_gov, k_priv, k_pubpriv, gdp` (precios corrientes) + variantes `_ppp`, `_gdp` | 194 economías, 1960-2019 |
| `imf_get(flow, key="all", ...)`, `imf_dataflows()`, `imf_meta(panel)` | SDMX-CSV crudo; catálogo de flujos; tabla de metadatos | |

- **Unidades.** Niveles en moneda nacional en millones (los miles de millones de ICSD se multiplican por 1000), personas en miles, tasas, índices y precios tal como se publican. Cada fila de `attrs["meta"]` da `unit_mult` y `scale` (el factor aplicado).
- **Ceros de relleno.** Algunos flujos publican ceros exactos donde no hay datos (todo el lado del gasto de Burkina Faso en 1952-1998, los acervos de capital de las economías que ICSD no cubre). Se convierten en faltantes y se cuentan en `attrs["zeros_dropped"]`. Los ceros que pueden ser verdaderos (variación de existencias, el acervo de capital PPA, desempleo, tasas de interés) se conservan.
- **Precios y volúmenes.** `_real` está a precios constantes del año de referencia propio de cada país, encadenado en la mayoría, así que los componentes no suman. `_defl = 100 * nominal / real` tiene esa base, que no siempre cae dentro de la muestra (BRA, ECU y AUS nunca se acercan a 100): rebásela antes de comparar niveles entre países. `cons_hh` cubre solo hogares (67 economías); para hogares más ISFLSH use `cons_priv = cons - cons_gov`.
- **Códigos.** Los códigos de grupo del FMI (`G001`, `G163`, ...) se eliminan con `IMF_AGGREGATES`; `KOS` pasa a `XKX` y `WBG` a `PSE`; las economías históricas (`SUN`, `YUG`, ...) se conservan.
- **WEO.** La historia termina por separado para cada país e indicador (`LATEST_ACTUAL_ANNUAL_DATA`); la población de Argentina es proyección después de 2010, mientras que su PIB es observado hasta 2025. `forecast` sigue al PIB real. `history_only=True` corta cada columna en su propia frontera, que es lo que conviene para congelar la historia. Los años fiscales siguen la correspondencia del FMI (el AF 2024/25 de India es 2024; el de Pakistán, 2025).
- **Presupuesto de tiempo.** Cada solicitud se reintenta 3 veces (hasta 180 s cada una). `imf_monthly_panel` se detiene tras dos flujos fallidos seguidos, así que una API caída cuesta a lo sumo unos 25 minutos.
- **Lo que no está en los flujos del FMI.** El lado del ingreso, durables, FBCF por activo, horas, IPC subyacente o de energía, plazos del mercado de dinero: use los recolectores de la OCDE, Eurostat o la OIT.

```python
from puremacro.fetch.imf import imf_nea_panel, imf_monthly_panel, imf_weo
a = imf_nea_panel()                                   # 188 economies, annual
q = imf_nea_panel(["USA", "MEX", "KOR"], freq="Q")
m = imf_monthly_panel(start=1990)
hist = imf_weo(history_only=True)                     # WEO history only, per-column boundaries
```

## BPI: `bis_panel` (`puremacro.fetch.bis`)

`bis_panel(codes=None, *, variables=None, freq="M", start=None, aggregates=False, refresh=False)`. Cada variable es una sola solicitud comprimida con gzip que cubre todas las economías. Además: `bis_eer(kind="real", basket="broad", *, freq="M", ...)`, `bis_get(flow, key, ...)`, `bis_countries(flow)`, `bis_meta(panel)`.

```python
from puremacro.fetch.bis import bis_panel, bis_eer, bis_countries, bis_meta
m = bis_panel()                                    # monthly, every economy, full history
q = bis_panel(freq="Q", start="1960")
a = bis_panel(["GBR", "USA"], variables=["cpi_a", "policy_rate"], freq="A")
```

| variable | flujo / clave BPI | frecuencia nativa | unidades | economías | desde |
|---|---|---|---|---|---|
| neer, reer | WS_EER `M.N.B` / `M.R.B` | M | índice 2020=100 | 63 | 1994-01 |
| neer_narrow, reer_narrow | WS_EER `M.N.N` / `M.R.N` | M | índice 2020=100 | 25 / 26 | 1964-01 |
| policy_rate | WS_CBPOL `M` | M | % anual, fin de periodo | 48 | 1945-01 |
| cpi | WS_LONG_CPI `M.*.628` | M | índice 2010=100 | 62 | 1913-01 |
| xr_usd, xr_usd_eop | WS_XRU `M.*..A` / `..E` | M | moneda nacional por USD | 189 | 1791 / 1900 |
| credit_gdp, credit_bank_gdp, credit_hh_gdp, credit_nfc_gdp, credit_gov_gdp | WS_TC `Q.*.{P,P,H,N,G}.{A,B,A,A,A}.{M,M,M,M,N}.770.A` | Q | % del PIB | 43 (gov 42) | 1947T4 |
| credit, credit_usd | WS_TC `Q.*.P.A.M.XDC.A` / `USD` | Q | millones de moneda nacional / USD | 43 | 1940T2 |
| credit_gap, credit_trend | WS_CREDIT_GAP `Q.*.P.A.C` / `.B` | Q | pp del PIB / % del PIB | 43 | 1957T4 |
| house_price, house_price_real | WS_SPP `Q.*.N.628` / `R.628` | Q | índice 2010=100 | unas 57 | 1927T1 |
| dsr, dsr_hh, dsr_nfc | WS_DSR `Q.*.P` / `H` / `N` | Q | % del ingreso | unas 31 / 17 | 1999T1 |
| cpi_a | WS_LONG_CPI `A.*.628` | A | índice 2010=100 | 62 | 1661 |

- **Frecuencia.** El BPI publica solo las frecuencias nativas de arriba; una `freq` más gruesa se construye aquí. La tasa de política, el tipo de cambio de fin de periodo y los saldos y razones de crédito toman el último subperiodo; los índices, el tipo de cambio promedio, los precios de vivienda y el DSR toman el promedio de un periodo completo. Un trimestre o año en curso incompleto se omite, no se promedia sobre los meses disponibles.
- **Agregados.** `EA` y los demás agregados del BPI (EME, ADV, ALL, G20, WLD, WAEMU) se eliminan por omisión, para que los promedios entre países nunca cuenten dos veces a los miembros del euro; uno nombrado en `codes` se conserva, y `aggregates=True` los conserva todos. Los conteos de economías de arriba excluyen `EA`.
- **Tasa de política de la zona euro.** Las series nacionales terminan en 1998-12 (Grecia en 2000-12) y `EA` empieza en 1999-01; empálmelas usted, p. ej. `pd.concat([p.loc["DEU"][:"1998-12"], p.loc["EA"]["1999-01":]])`.
- **Fechas.** El nivel de fecha es `datetime64[ns]`, salvo en paneles con `cpi_a` anterior a 1678 (el Reino Unido desde 1661), que son `datetime64[us]`.
- **Metadatos.** `attrs["meta"]` da flujo, clave, unidades, unit_mult, sa, first, last y n por código y variable. `attrs["missing"]` lista también los códigos pedidos que un flujo no cubre (México no tiene NEER nominal estrecho).

Fuente: API SDMX v2 de estadísticas del BPI, https://stats.bis.org/api/v2.

## Tasas, rendimientos, acciones y materias primas (`puremacro.fetch.rates`)

- `rates_panel(codes=None, *, start="1950", variables=("rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"), refresh=False)`
- `stock_index_monthly(codes=None, *, start=None, refresh=False)`
- `commodity_prices_monthly(*, start=None, refresh=False)`
- Piezas básicas: `fetch_fred_many(ids)` (una solicitud por id) y `ecb_get(flow, key, start=, end=, last_n=)` (una consulta al ECB Data Portal).

```python
from puremacro.fetch.rates import rates_panel, stock_index_monthly, commodity_prices_monthly

r   = rates_panel(start="1900")                      # 47 countries, (code, date) monthly
r10 = rates_panel(None, variables=("yield_10y",))    # 42 countries
px  = stock_index_monthly()                          # 34 countries, stock_idx + stock_idx_avg
cm  = commodity_prices_monthly()                     # WLD, 48 commodity_* columns, 1960-01..
```

Las columnas de `rates_panel` están en por ciento anual, promedios mensuales, sin ajuste estacional. Los nombres coinciden con los de `stes_panel` y `eurostat` (`rate_on`, `rate_3m`), así que los paneles se empalman por columna.

| columna | fuente (en orden) | cobertura |
|---|---|---|
| `rate_on` | FRED `IRSTCI01{CC}M156N`; miembros del euro desde su ingreso: EONIA, luego €STR desde 2019-10 | 46 países; FRA, SWE desde 1955 |
| `rate_3m` | FRED `IR3TIB01{CC}M156N`; miembros del euro desde su ingreso: EURIBOR a 3 meses; CZE, HUN, POL, ROU completados con ECB IRS | 45; CAN desde 1956 |
| `yield_10y` | FRED `IRLTLT01{CC}M156N`; ECB IRS a 10 años (28 países de la UE) llena huecos; historias retrospectivas ECB FM para EUA (1900) y JPN (1972) | 42 |
| `yield_corp_aaa`, `yield_corp_baa` | FRED `AAA`, `BAA` (Moody's) | EUA desde 1919 |
| `yield_corp_de` (opcional) | Bundesbank BBSIS X2000 | DEU desde 1957 |

- **Los empalmes son visibles.** Cada tramo de fuente usado aparece en `attrs["meta"]`. `attrs["complete"]` es `False` cuando una serie que algún proveedor sirve falló en esta llamada (429, tiempo agotado); el tramo del BCE pudo haberla sustituido con una historia más corta, así que revise `complete` antes de congelar un CSV.
- **Acciones.** `stock_idx` es el cierre de fin de mes de Yahoo en moneda local (desde 1985 como muy pronto); `stock_idx_avg` es el promedio mensual del BCE (EUA desde 1964, JPN desde 1972, `"EMU"` a pedido desde 1986). Ninguno es el `share_price` de STES, un índice de promedios con 2015 = 100.
- **Materias primas.** La Pink Sheet del Banco Mundial, leída solo con la biblioteca estándar; las columnas llevan el prefijo `commodity_` y los índices son `commodity_index_*` (2010 = 100).
- **Caché.** Vigencia de 30 días; un cuerpo en caché que ya no se puede interpretar se vuelve a descargar una vez automáticamente.

## ILOSTAT: `ilostat_panel` (`puremacro.fetch.ilostat`)

`ilostat_panel(codes=None, *, freq="A", variables=None, start=None, end=None, modelled=False, refresh=False, timeout=300.0, pause=3.0)` devuelve las series armonizadas de encuestas de fuerza laboral de ILOSTAT para todas las economías informantes.

| columna | flujo / MEASURE de ILOSTAT | unidades | frecuencia |
|---|---|---|---|
| emp, une, lf, wap | DF_EMP_TEMP / DF_UNE_TUNE / DF_EAP_TEAP / DF_POP_XWAP `_SEX_AGE_NB` | miles de personas | A Q M |
| urate, prate, epop | DF_UNE_DEAP / DF_EAP_DWAP / DF_EMP_DWAP `_SEX_AGE_RT` | porcentaje | A Q M |
| emp_ees, emp_self | DF_EMP_TEMP_SEX_STE_NB, STE_AGGREGATE_EES / _SLF | miles de personas | A Q M |
| hours | DF_HOW_TEMP_SEX_NB | horas semanales por ocupado | A Q M |
| informal_rate | DF_EMP_NIFL_SEX_RT | porcentaje del empleo | A Q (M escasa) |
| labour_share | DF_LAP_2GDP_NOC_RT (modelado por la OIT) | porcentaje del PIB | A |
| *_model (`modelled=True`) | DF_EMP_2EMP, DF_UNE_2EAP, DF_EAP_2EAP, DF_POP_2WAP, DF_EMP_2EMP_SEX_STE, DF_HOW_2EMP | como arriba | A |

- **Definiciones.** Cada serie es el total de 15 años y más, ambos sexos, sin ajuste estacional salvo las tasas mensuales de la UE que ILOSTAT retransmite de Eurostat desestacionalizadas (`meta[i]["sa"]`). Los quiebres (`OBS_STATUS == "B"`) permanecen en los datos y se listan en `meta["breaks"]`.
- **Las proyecciones modeladas se recortan.** Los flujos modelados publican años proyectados sin marca (dos para emp/urate/lf/hours_model, cinco para wap_model, uno para labour_share). Salvo que se dé `end`, se recortan hacia atrás desde el último año de cada flujo, para que un panel congelado no dependa de la fecha de construcción; el recorte queda en `meta["trimmed_after"]`. `modelled=True` es solo anual y lanza `ValueError` con `freq="Q"`/`"M"` (la única excepción deliberada).
- **Cobertura.** emp anual: 226 economías (17 desde antes de 1970, 165 desde antes de 1995); urate 224; asalariados/independientes 214; labour_share 189 (2004-2025). urate trimestral: 122 economías desde 1948T1. EUA anual desde 1947; MEX anual desde 1988 (lf/wap desde 1960), trimestral desde 1995T2; ESP anual desde 1969, trimestral desde 1986T2.

```python
from puremacro.fetch.ilostat import ilostat_panel, ilostat_meta
a = ilostat_panel()                                   # 12 survey variables, ~12 requests
q = ilostat_panel(freq="Q")
m = ilostat_panel(freq="M", variables=("emp", "une", "urate", "prate"))
x = ilostat_panel(["MEX", "USA", "ESP"], freq="A", modelled=True)
self_share = a["emp_self"] / (a["emp_ees"] + a["emp_self"])
ilostat_meta(a).loc["urate"]["breaks"]
```

## Penn World Table 11.0 y Maddison 2023 (`puremacro.fetch.pwt`)

`fetch_pwt(table="main", *, version="11.0", codes=None, start=None, refresh=False, timeout=180.0)`, `fetch_maddison(codes=None, *, start=1950, refresh=False, timeout=180.0)` y `pwt_variables(table="main")` descargan los archivos Stata de DataverseNL. Cada id de archivo es una publicación inmutable, así que los archivos se guardan en caché por diez años.

```python
from puremacro.fetch.pwt import fetch_pwt, fetch_maddison, pwt_variables

main = fetch_pwt("main")      # 185 economies, 1950-2023: rgdpe, rgdpna, pop, emp, hc, cn, rnna, labsh, ctfp, csh_*, pl_*
na   = fetch_pwt("na")        # 212 economies with v_gdp: v_c v_i v_g v_x v_m v_gdp v_gfcf (millions of current national currency), q_* (constant 2021 prices)
cap  = fetch_pwt("capital")   # Ic_/Ip_/Nc_/Np_/Dc_/Kc_/Kp_/Ksh_ by asset (Struc, Mach, TraEq, Other), 180 economies
lab  = fetch_pwt("labor")     # comp_sh, lab_sh1..4 (Gollin adjustments), labsh, yr_sch, i_* source flags
trd  = fetch_pwt("trade")     # pl_x1..6, pl_m1..6, csh_x1..6, csh_m1..6 (BEC categories)
mpd  = fetch_maddison()       # 169 economies, gdppc (2011 int. $), pop (thousands), annual from 1950; start=None goes back to 1 AD
pwt_variables("na")           # published label | true units | note on stale labels
```

| tabla | archivo (id) | economías | años |
|---|---|---|---|
| main | pwt110.dta (554030) | 185 | 1950-2023 |
| na | pwt110_na_data.dta (554024) | 212 con v_gdp (216 códigos) | 1950-2023 (avh hasta 2025) |
| capital | pwt110_capital_detail.dta (554026) | 180 | 1950-2023 |
| labor | pwt110_labor_detail.dta (554028) | 211 códigos | 1950-2023 (avh hasta 2025) |
| trade | pwt110_trade_detail.dta (554023) | 185 | 1950-2023 |
| Maddison | maddison2023_web.dta (421303) | 169 | 1-2022 (anual desde 1950) |

- **Nombres de columna publicados**, valores float64; las banderas `i_*` son códigos enteros explicados en `attrs["value_labels"]`.
- **La base de precios es 2021 en todas partes**, aunque algunas etiquetas aún dicen "2017 prices", "2017=1" o "USA GDPo in 2011=1"; `attrs["units"]` declara la base verdadera. El `gdppc` de Maddison está en dólares internacionales de 2011.
- **C + I + G + X - M no es el PIB** en alrededor de un tercio de las filas de cuentas nacionales: use `v_gdp` como denominador y nunca calcule un componente como residuo. `v_gfcf` es en la práctica una serie desde 1970.
- **El detalle de capital suma la FBCF** (los `Ic_*` suman `v_gfcf`) en 179 de 180 economías; en Taiwán la suma es 6-8% mayor.
- **El empleo está en millones en todas las tablas** (las personas del archivo labor se reescalan).
- **Códigos.** `CH2` (la serie alternativa de China en PWT) se descarta salvo que se pida. Las economías desaparecidas sin PIB (`ANT`, `CSK`, `SUN`, `YUG` en `na`; `ANT` en `labor`) se quedan y se listan en `attrs["historical_entities"]`, así que cuente economías con `v_gdp`, no con códigos. Kosovo es `RKS` en PWT (`XKX` en el Banco Mundial y el FMI), marcado en `attrs["nonstandard_codes"]`: renómbrelo antes de unir.
- **Licencia.** Ambas bases son CC BY 4.0, así que los CSV congelados se pueden compartir; cite `attrs["citation"]` y, para Maddison, siga `attrs["citation_policy"]` (cite las fuentes originales al graficar o al usar menos de 12 países).

Fuentes: Feenstra, Inklaar & Timmer (2015), PWT 11.0, doi:10.34894/FABVLR; Bolt & van Zanden (2024), MPD 2023, doi:10.34894/INZBF2.

## Qué llamada para cada concepto

| frecuencia | concepto | llamada |
|---|---|---|
| anual | PIB y componentes del gasto | `wdi_panel`, `ana_panel`, `imf_nea_panel`, `eurostat_na_panel`, `fetch_pwt("na")` |
| anual | producto de largo plazo, productividad, capital humano | `fetch_pwt("main")`, `fetch_maddison` |
| anual | inversión pública y privada, acervos de capital | `ana_panel(sectors=True, stocks=True)`, `imf_icsd`, `eurostat_na_panel(sectors=True, capital=True)`, `fetch_pwt("capital")` |
| anual | empleo, horas, desempleo, participación del trabajo | `ana_panel(labor=True)`, `ilostat_panel`, `imf_labour_panel`, `fetch_pwt("labor")` |
| anual | IPC largo, crédito, precios de vivienda | `bis_panel(freq="A")` |
| anual | pronósticos e historia congelada del WEO | `imf_weo` |
| trimestral | PIB y componentes del gasto | `qna_panel`, `imf_nea_panel(freq="Q")`, `eurostat_na_panel(freq="Q")` |
| trimestral | inversión pública y privada | `qna_sector_gfcf`, `eurostat_na_panel(freq="Q", sectors=True)` |
| trimestral | población, empleo, desempleo, vacantes | `qna_population`, `oecd_lfs_panel`, `oecd_vacancies`, `ilostat_panel(freq="Q")` |
| trimestral | crédito, brecha de crédito, precios de vivienda, servicio de la deuda | `bis_panel(freq="Q")` |
| mensual | producción, precios, confianza, dinero, trabajo | `stes_panel`, `imf_monthly_panel`, `eurostat_monthly_panel`, `ilostat_panel(freq="M")` |
| mensual | tasas de política y de corto plazo, rendimientos largos y corporativos | `rates_panel`, `bis_panel`, `stes_panel` |
| mensual | tipos de cambio (bilaterales, efectivos) | `bis_panel`, `bis_eer`, `imf_monthly_panel` |
| mensual | precios de acciones y de materias primas | `stock_index_monthly`, `commodity_prices_monthly`, `imf_pcps` |
