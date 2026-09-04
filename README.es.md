> 🇬🇧 [English](README.md) · 🇪🇸 Español

# puremacro

Una **caja de herramientas de macroeconomía empírica compatible con Pyodide**: el código de los estimadores corre sobre numpy + scipy + pandas + matplotlib puros, de modo que el núcleo numérico sigue siendo importable bajo Pyodide (iPad / juno.sh, en la medida de lo posible — véase «juno.sh / iPad» más abajo). El destino soportado es una **instalación local** en una estación de trabajo convencional.

## Contenido

**Econometría central**

- **VAR** — mínimos cuadrados ordinarios en forma reducida, BVAR (Minnesota), VECM (Engle-Granger / Johansen), TVP-VAR, VAR de panel; FRI / FEVD / GFEVD; bandas de confianza mediante bootstrap de residuos, por bloques, por bloques móviles y wild bootstrap.
- **Identificación de SVAR** (`var.identify.*`) — Cholesky, Blanchard-Quah, restricciones de signo (Rubio-Ramirez-Waggoner-Zha), restricciones de signo y cero (Arias-Rubio Ramirez-Waggoner), bandas robustas a las restricciones de signo (Giacomini-Kitagawa), variables proxy / instrumentos externos, máxima participación espectral / noticias, heterocedasticidad (Rigobon), no gaussiano (Lanne-Meitz-Saikkonen). Todos los estimadores públicos devuelven objetos `…Result` de tipo dataclass congelado.
- **Proyecciones locales** (`lp.*`) — LP-HAC para un solo país, LP-IV, LP con retardos aumentados (Plagborg-Møller-Wolf), LP de panel con errores estándar agrupados / Driscoll-Kraay, LP dependiente del estado, LP suavizada (B-splines de Barnichon-Brownlees), LP asimétrica (Tenreyro-Thwaites), LP-GARCH en estado, LP-GARCH en media, grupo medio, CCE, LP cuantílica.
- **Inferencia** (`inference.*`) — MCO con HAC central, Newey-West, Kiefer-Vogelsang de b fijo, Driscoll-Kraay; diagnósticos de instrumentos débiles (Cragg-Donald, Kleibergen-Paap, Anderson-Rubin, Montiel Olea-Pflueger); Hansen-J / Stock-Yogo para sobreidentificación; CD de Pesaran, homogeneidad de pendientes de Swamy, quiebres estructurales de Quandt-Andrews, curvas de especificación.
- **Otros estimadores** — índice de derrame de Diebold-Yilmaz; comparación de pronósticos Diebold-Mariano / Giacomini-White y evaluación de pronósticos en densidad (CRPS, log-score); quiebres de Bai-Perron; pruebas de raíz unitaria (ADF, KPSS, PP, Zivot-Andrews); solver QZ de Klein para DSGE lineales (condición de Blanchard-Kahn verificada).

**Extensiones de macroeconomía moderna**

- **DiD escalonado** (`did.*`) — Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, DiD sintético; bootstrap de errores estándar en todos los métodos.
- **GMM de panel dinámico** (`dynpanel.*`) — Arellano-Bond, Blundell-Bond de dos etapas con corrección de Windmeijer + Hansen-J + AR(1)/AR(2) + colapso de Roodman.
- **Sorpresas monetarias de alta frecuencia** (`hfi.*`) — Gertler-Karadi 2015, Nakamura-Steinsson 2018, Jarociński-Karadi 2020.
- **Volatilidad** (`volatility.*`) — `SigmaObject` (traducción 1:1 de la clase MATLAB de MAV con API de descomposición extendida), BEKK, CCC, HAR-RV, basado en rango, diagnósticos ARCH-LM / Ljung-Box.
- **Nowcasting** (`nowcast.*`) — DFM-Kalman (Doz-Giannone-Reichlin) con manejo de bordes irregulares, MF-VAR de Mariano-Murasawa, combinaciones de pronósticos, reglas de puntuación probabilísticas.
- **Crecimiento en riesgo** (`gar.*`) — AR cuantílico, ajuste skew-t de ABG 2019, FCI al estilo NFCI.
- **Ciclos / cointegración / factores** — filtro de tendencia-ciclo de Hamilton 2018 (`cycles`), FM-OLS de Phillips-Hansen / DOLS de Stock-Watson / Phillips-Ouliaris (`cointegration_modern`), factores PCA + criterio IC de Bai-Ng (`factor`), MIDAS (`midas`), GMM de sistema CES de KORV (2000) (`korv_gmm`), control sintético + inferencia placebo (`synthetic_control`).
- **Espectral / wavelet** (`spectral`, `wavelet`) — PSD de Welch / espectro cruzado / coherencia (solo numpy.fft); descomposición de varianza wavelet MODWT-Haar.
- **Volatilidad realizada** (`realized_vol`) — varianza realizada, variación bipotencial, HAR-RV de Corsi.
- **Agentes heterogéneos / iteración sobre función de valor** (`vfi.*`) — iteración sobre la función de valor con EGM, ciclo de vida de horizonte finito, OLG, choques agregados de Krusell-Smith, entrada/salida de empresas de Hopenhayn, Epstein-Zin, tipos permanentes, trayectorias de transición y estimación por método de momentos; backend de referencia en numpy con aceleración opcional mediante numba / mlx / cupy. Véanse los cuadernos en `notebooks/` para una galería de ejemplos.

**Econometría narrativa** (`narrative.*`)

Pipeline de variables instrumentales narrativas para política fiscal, mercado laboral e incertidumbre: esquemas canónicos `NarrativeEvent` / `NarrativeInstrument`, deduplicación, clasificadores por palabras clave y puntuación manual, construcción de panel, cargadores de replicación para conjuntos de datos canónicos (Romer-Romer, Mertens-Ravn). El clasificador basado en LLM (`narrative.scoring.llm`) y los módulos de fuentes HTTP (`narrative.sources.*`) operan fuera de Pyodide como canales laterales.
Las fuentes disponibles incluyen:
- **Libro Beige** — corpus del Beige Book de la Fed a partir de las páginas modernas de federalreserve.gov y las páginas históricas del FOMC, con análisis por sección canónica y por distrito (`puremacro.narrative.sources.iter_beige_book`, `puremacro.narrative.indices.bbui`).
- **Narrativa ejecutiva estadounidense** — Economic Report of the President (`iter_erp`), State of the Union (`iter_sotu`) e informes del CBO (`iter_cbo`); tres índices asociados `erpui`, `sotuui`, `cboui`. Las solicitudes de cuerpo al CBO recurren de forma transparente a la Wayback Machine cuando cbo.gov devuelve un desafío DataDome.
- **Narrativa legislativa de la UE** — actos vinculantes de EUR-Lex (`iter_eurlex`) y debates plenarios verbatim del Parlamento Europeo (`iter_ep_debates`); dos índices trilingües EN/DE/FR `eurlex_ui` y `ep_ui`. Enumeración de EUR-Lex mediante el endpoint público SPARQL de Cellar (obtención por acto enrutada por Wayback debido a la protección AWS-WAF del sitio en vivo); PE mediante CDX de Wayback con cobertura desde la Legislatura 7 (2009-07-14).
- **Archivo de Bluesky** — gobernadores de bancos centrales y ministros de finanzas mediante AT Protocol (`iter_bluesky_posts`, `bluesky_ui`). Lista semilla de 29 identificadores seleccionados manualmente (`BLUESKY_KNOWN_HANDLES`); 12 resueltos al 2026-05-25. Soporte multilingüe mediante el argumento `languages=...` del conector; el índice utiliza por defecto agregación mensual a nivel de actor (`aggregate_to="actor_month"`) para mitigar la degradación del LUI con textos cortos.
- **Desacuerdo entre fuentes** — `consensus_disagreement` calcula la media y desviación estándar transversal sobre cualquier subconjunto de índices narrativos; `CROSS_SOURCE_GROUPS` documenta los subconjuntos temáticos.

Los conectores bloqueados por WAF / protección anti-bot (EUR-Lex, Parlamento Europeo, CBO) recurren a la Wayback Machine mediante el helper compartido `puremacro.narrative.sources._wayback`. La cobertura está limitada por lo que Wayback haya archivado.

**Pipelines de datos** (incorporados recientemente; véase `ARCHITECTURE.md`)

- **Captadores** (`fetch.*`) — FRED / ALFRED, SDMX-CSV (OCDE, Eurostat, BCE, SDMX-Central del FMI), EPU / GPR / WUI / JLN / Fernald, OCDE-MEI / QNA / Energía / Tipos de cambio, ILOSTAT, Yahoo, hoja rosa del Banco Mundial, además de cargadores FRED por estado para el seguimiento subnacional de EE. UU.
- **Cuentas nacionales largas** (`fetch.qna_long_panel`) — la serie de la OCDE
  extendida hacia atrás, país por país, empalmando por ratios las añadas
  nacionales archivadas: **España hasta 1970T1** (+100 trimestres) y **Japón
  hasta 1955T2** (+155), con la procedencia de cada serie y trimestre. El
  empalme conserva las tasas de crecimiento de la añada antigua e informa de
  cuán estable es el ratio de cada juntura: si el ratio deriva a lo largo del
  solape, las dos añadas discrepan sobre el crecimiento y el nivel empalmado
  depende del trimestre de anclaje. Otras siete fuentes candidatas no aportan
  ni un trimestre, y el motivo queda registrado en `LONG_PANEL_KNOWN_GAPS`.
  Véase `docs/long_panel.md`.
- **Datos en tiempo real** (`fetch.vintage_panel`, `fetch.realtime.*`) — las *ediciones* publicadas de una serie, con seis proveedores tras una sola llamada: el archivo de revisiones OCDE-STES (42 economías, ediciones mensuales desde 1999), ALFRED, la base Gerda del Bundesbank, el libro de tiempo real del ONS (746 ediciones desde 1961), las tablas de vintages de Statistics Canada y la base del BCE/EABCN. Incluye el instrumental de revisiones: triángulos de revisión, primera y última estimación, `r_t = y_f - y_p`, y el contraste de noticia frente a ruido de Mankiw-Shapiro (`vintages.mankiw_shapiro`). Cada proveedor documenta qué significa exactamente su fecha de edición, porque no coinciden entre sí. Véase `docs/real_time_data.md`.
- **Constructores de panel** (`build_panel`, `build_subnational_panel`) — puntos de entrada únicos que materializan paneles trimestrales y mensuales de países y estados de EE. UU. a partir de los captadores, con etiquetado de regímenes, ajuste estacional (X-13 / STL como alternativa) y una pipeline de σ-GARCH derivada.
- **Instrumentos** (`instruments.*`) — registro de instrumentos, composición y cargadores externos (ruta de clave API de FRED); columna vertebral de la maquinaria LP-IV.
- **Bartik / shift-share** (`bartik.*`) — participaciones, sensibilidades, pesos de Rotemberg, exposición EPU a nivel de condado.
- **Utilidades de datos misceláneas** — cargador EU-KLEMS 2023 (`klems`), agregador NEER del BIS (`bis_neer`), empalme homogéneo de vintage G9 (`long_panel`), participación laboral de Gollin (`labor_share`), series en tiempo real por vintage (`vintages`), ajuste estacional (`sa`).
- **Flujos laborales** — transiciones E/U/N de tres estados a partir de los agregados CPS del BLS (`labor_flows`) y transiciones F/I/U/N de cuatro estados a partir de los microdatos ENOE para México (`labor_flows_enoe`).

**Trabajar fuera de una estación de trabajo** (`runtime.*`, `pocket.*`, `longrun.*`)

La promesa central del paquete es que el núcleo de estimadores corre en un iPad. Estos tres módulos hacen que esa promesa sea utilizable, y no sólo cierta: `runtime` informa de lo que la máquina puede hacer realmente (¿sockets? ¿parquet? ¿hilos?) y encamina el HTTP por el navegador cuando no hay sockets; `pocket` empaqueta datos en cartuchos `.pmz` portátiles y autoverificables, de modo que un panel construido en línea se abre sin conexión; `longrun` ejecuta bootstraps y cadenas en trozos reanudables que sobreviven a la suspensión de la aplicación por parte del sistema, con resultados invariantes al modo en que se troceó el trabajo. Véase «juno.sh / iPad» más abajo.

**Cuaderno de bocetos DSGE** (`dsge.build`)

Escriba las condiciones de equilibrio como una función de Python y obtenga a cambio una aproximación de primer orden resuelta — estado estacionario, reglas de decisión, IRFs —, con los jacobianos obtenidos por diferenciación de paso complejo. Sin matrices derivadas a mano, sin Dynare y sin compilador: justo lo que una tableta no puede ofrecer.

**Artefactos docentes**

`teaching.*` es un canal lateral de investigación y docencia que envuelve intencionadamente `statsmodels` / `linearmodels` / `arch` para que los cuadernos puedan comparar los estimadores puros en numpy de puremacro con los paquetes canónicos. **No está cubierto por la promesa de compatibilidad con Pyodide.**

## Instalación

### Desde PyPI (usuarios)

```bash
pip install puremacro
```

Esto instala las **siete dependencias base** (numpy, scipy, pandas, matplotlib, requests, pyarrow, openpyxl) — todo lo que necesitan los estimadores, la capa `fetch` y las rutas de código en parquet. Los extras cubren únicamente las funciones opcionales que se listan más abajo.

### Local (desarrollo)

Desde el directorio del paquete `puremacro/` (el que contiene este `README.md` y `pyproject.toml`):

```bash
pip install -e .
```

Para ejecutar las pruebas de paridad del entorno de desarrollo, instale también las dependencias opcionales:

```bash
pip install -e '.[dev]'
```

Para utilizar el extractor de cuerpo PDF de `narrative.sources`:

```bash
pip install -e '.[narrative]'
```

Otros extras opcionales: `[backend]` (numba + Apple-Silicon mlx), `[cuda]` (NVIDIA cupy), `[data]` (captadores yfinance / fredapi / xlrd), `[llm]` (puntuación narrativa respaldada por Anthropic), `[embeddings]` (puntuación narrativa con sentence-transformers), `[notebooks]` (construcción de cuadernos con jupytext).

Para los conectores que requieren caché en disco bajo demanda y regulación por host, las variantes `safe_get_bytes_cached` y `safe_get_text_cached` aplican una caché indexada por SHA-256 en `~/.cache/puremacro/http/`. Defina `PUREMACRO_HTTP_NO_CACHE=1` para omitirla.

### juno.sh / iPad (no soportado; en la medida de lo posible)

Suba el directorio `puremacro/` a su espacio de trabajo en juno.sh y luego, en una celda de cuaderno:

```python
%pip install ./puremacro
```

**Advertencia desde que `pyarrow` es dependencia base:** ese comando resuelve el conjunto completo de dependencias y `pyarrow` no tiene rueda para Pyodide, así que bajo un núcleo Pyodide falla. Instale el núcleo de estimadores sin resolución de dependencias y añada a mano sólo lo que necesite:

```python
import micropip
await micropip.install("puremacro", deps=False)
await micropip.install(["numpy", "scipy", "pandas", "matplotlib", "requests"])
```

Las rutas de código en parquet (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`) quedan indisponibles en el navegador. El navegador **no** es un destino de despliegue soportado: el material docente presupone una instalación local.

#### Averiguar qué puede hacer realmente la tableta

`puremacro.runtime` lo responde en tiempo de ejecución, en lugar de dejar que se descubra un *traceback* cada vez:

```python
from puremacro import runtime
print(runtime.report())
#  host       : pyodide 3.12.7 (wasm32)
#  device     : tablet
#  network    : js-fetch (call runtime.enable_browser_network())
#  parquet    : unavailable -> use puremacro.runtime.store / pocket
#  threads    : no (1 cpu, unknown)
#  backends   : numpy
```

La detección es heurística — ninguna API dice «esto es Juno» —, así que cada campo puede fijarse con `PUREMACRO_HOST`, `PUREMACRO_DEVICE`, `PUREMACRO_SOCKETS` o `PUREMACRO_PARQUET`.

#### Las tres cosas que fallan, y qué hacer al respecto

**Sin sockets.** Bajo Pyodide ni `requests` ni `urllib` pueden abrir una conexión, de modo que toda llamada a `fetch.*` falla aunque el núcleo de estimadores se importe sin problema. Una sola llamada encamina toda la capa de descarga existente por la red del propio navegador:

```python
from puremacro import runtime
from puremacro.fetch import fetch_xrate_monthly

runtime.enable_browser_network()
fx = fetch_xrate_monthly(["MEX"])
```

Los puntos de acceso deben enviar `Access-Control-Allow-Origin` — algunas API estadísticas públicas lo hacen; muchos sitios gubernamentales tras un WAF, no. Una petición bloqueada lo indica y nombra a CORS; `proxy=` permite encaminar por un proxy CORS propio.

**Sin pyarrow.** Empaquete los datos donde estén la red y pyarrow, y ábralos donde no estén. Un cartucho es un único archivo autoverificable que lleva consigo su propia procedencia:

```python
from puremacro import pocket

# estación de trabajo
pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

# iPad, en modo avión
cart = pocket.load("g7.pmz")
panel = cart.frame()          # verificado por sha256 al leer
cart.provenance.vintage       # '2026-08-19'
```

Llevar un *archivo* a un iPad suele costar más que el propio análisis, así que un cartucho también viaja como texto: `pocket.to_base64("g7.pmz")` en una máquina y `pocket.from_base64(blob, "g7.pmz")` en la otra.

**La aplicación se suspende.** iPadOS detiene una aplicación en segundo plano, y un bootstrap de cuatro minutos no sobrevive a que alguien conteste un mensaje. `puremacro.longrun` calcula por trozos, guarda tras cada uno y se reanuda en una sesión posterior:

```python
import numpy as np
from puremacro import longrun

job = longrun.bootstrap(one_draw, 2000, checkpoint="irf.ckpt")
job.run(seconds=30)     # 240/2000 · 12% · ~220s of compute left
job.run(seconds=30)     # ... y de nuevo tras la suspensión de la app
bands = np.percentile(job.result(), [5, 95], axis=0)
```

La extracción *i* siempre usa `default_rng([seed, i])`, así que un trabajo reanudado a lo largo de cinco sesiones da resultados idénticos bit a bit a uno que corrió de un tirón — que es lo que hace publicable una ejecución reanudada.

**Ajustar el trabajo a la máquina.** `runtime.fit(n_boot=2000)` devuelve lo que esta máquina debería intentar de verdad, y `runtime.budgeted(estimator)` acota los argumentos de coste de una llamada. Ambos son opcionales: ningún valor por defecto de ningún estimador ha cambiado, de modo que un script que corre en su portátil produce los mismos números de siempre. Sólo se acotan los parámetros de coste — `horizon` cambia *qué* se estima, así que se deja intacto.

### Ejecutar las funciones LLM de forma gratuita (modelos locales)

Las funciones LLM narrativas (`score_llm`, `llm_prob_kernel`) se ejecutan sobre un **modelo local** — sin clave de API, sin API de pago, $0. Todo lo demás en puremacro ya es gratuito; esto cierra el único componente que antes requería pago.

Instale un motor una sola vez (cualquiera de los siguientes):

```bash
pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp (cualquier SO)
# o instale Ollama (https://ollama.com) — sin dependencias Python — y luego:  ollama pull qwen2.5:3b
```

Luego utilice un backend local (mismas firmas que los backends de pago):

```python
from puremacro.narrative.scoring import score_llm, LocalBackend
events = score_llm(records, backend=LocalBackend("qwen2.5-3b-instruct", engine="auto"))

from puremacro.narrative.indices import llm_prob_kernel, LocalProvider
idx = llm_prob_kernel(records, provider=LocalProvider("qwen2.5-3b-instruct"),
                      category="economic uncertainty")
```

`engine="auto"` selecciona el mejor motor instalado (GPU de Apple via MLX → llama.cpp → un servidor Ollama en ejecución; para LM Studio / vLLM / cualquier servidor compatible con OpenAI, pase `engine="openai"` con `base_url=`). Modelos disponibles: `qwen2.5-3b-instruct` (por defecto), `gemma2-2b` (Google), `llama3.2-3b` (Meta), `phi3.5` (Microsoft), o cualquier identificador de modelo del motor. Véanse `puremacro/examples/narrative_local_llm.py` y el cuaderno `local_llm_uncertainty`. (La inferencia local es solo para escritorio: requiere una instalación local de Python y no funciona bajo Pyodide.)

## Compatibilidad con Pyodide

La promesa de compatibilidad en tiempo de ejecución es: únicamente `numpy + scipy + pandas + matplotlib` serán importados por el código de los *estimadores* que se distribuye en la rueda. `statsmodels`, `linearmodels`, `arch` y `pypdf` son todos exclusivos del entorno de desarrollo, están limitados a extras o se importan de forma diferida tras una verificación.

Otros tres paquetes están declarados como dependencias base en `pyproject.toml` —**siete en total**— porque la rueda no puede funcionar sin ellos, aunque ninguno toque la ruta de los estimadores:

- `requests` — importado a nivel de módulo por `puremacro.fetch.*` y por las fuentes narrativas. Python puro; se instala bajo Pyodide.
- `pyarrow` — el motor parquet que necesita `pandas.read_parquet` (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel` y los conjuntos de datos en parquet que usa el material docente). pandas lo importa de forma diferida, así que nunca aparece en `sys.modules` en un barrido de importaciones. No tiene rueda para Pyodide: en el navegador use `micropip.install("puremacro", deps=False)`.
- `openpyxl` — el motor `.xlsx` que necesita `pandas.read_excel`. Dieciocho módulos distribuidos leen Excel: entre ellos los descargadores de EPU, WUI, JLN, LMN, Fernald, GPR y del Pink Sheet del Banco Mundial. Antes vivía en el extra `dev`, así que un `pip install puremacro` sin más no podía producir ninguna de esas series — y `build_all` convierte cada fallo en un `print`, de modo que el panel volvía sin la mayor parte de sus indicadores de incertidumbre y sin decirlo. Python puro; se instala bajo Pyodide.

Véase `ARCHITECTURE.md` → «contrato de compatibilidad con Pyodide» para la justificación completa.

La prueba de regresión correspondiente es `tests/test_pyodide_compat.py` — recorre cada submódulo distribuible y verifica que ningún módulo prohibido figure en `sys.modules`. Si agrega una nueva dependencia opcional, siga el patrón de importación diferida existente (véanse `narrative.scoring.llm` o `fetch._seasonal._x13_arima_analysis` como ejemplos canónicos).

## Inicio rápido

**Primeros 5 minutos — sin conexión, sin archivos de datos, sin clave de API.** La verificación más rápida de que la instalación funciona (un SVAR con restricciones de signo sobre un proceso generador de datos sintético de 3 variables; sin red, sin datos, semilla fija):

```bash
python -m puremacro.examples.sign_restrictions_uhlig
```

O bien, en Python, sobre un sistema sintético construido en tres líneas:

```python
import numpy as np
import pandas as pd
import puremacro as pm

# Un pequeño sistema sintético de 3 variables (sin archivos de datos, sin clave de API).
rng = np.random.default_rng(0)
T = 200
Y = rng.standard_normal((T, 3)).cumsum(0)          # ndarray, shape (T, 3)

# SVAR identificado por Cholesky con bandas de confianza al 90% mediante bootstrap de residuos.
from puremacro.var.identify import cholesky_svar
res = cholesky_svar(Y, p=2, horizon=20, n_boot=500, ci=0.90)
print(res.summary())
res.plot(target_idx=0, shock_idx=0)        # Gráfico de FRI en 1 línea con bandas
print(res.to_latex(target_idx=0, shock_idx=0))  # Tabla lista para LaTeX

# LP-HAC para un solo país: respuesta de y a un choque sintético.
panel = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(T)})
from puremacro.lp import lp_hac
irf = lp_hac(panel, y="y", x="shock", horizon=20, lags=2, ci=0.90)
print(irf.summary())
irf.plot(title="Respuesta de y al choque estructural")
print(irf.to_latex())                      # Tabla lista para LaTeX
```

Las claves de API opcionales se resuelven de forma centralizada (no se necesita ninguna para los ejemplos sintéticos anteriores):

```python
from puremacro import credentials
credentials.status()                  # see what's configured (no values leaked)
# credentials.require("fred")         # raises with a signup URL if the key is missing
```

## Piedra de Rosetta — Guía de equivalencias para macroeconomistas

Si proviene de Stata, MATLAB/Dynare o statsmodels:

| Tarea / Estimador | Stata | MATLAB / Dynare | statsmodels / linearmodels | **`puremacro 2.0`** |
|---|---|---|---|---|
| **SVAR de Cholesky** | `var y1 y2, lags(1/4)` + `irf create` | `varm` / VAR Toolbox | `VAR(Y).fit(4).irf(20)` | `var.identify.cholesky_svar(Y, p=4, horizon=20)` |
| **SVAR de Blanchard–Quah** | `svar y1 y2, lreq(...)` | VAR Toolbox `bq_svar` | `SVAR(..., svar_type='B')` | `var.identify.bq_svar(Y, p=4, horizon=20)` |
| **Restricciones de signo** | Plugin de usuario | Rubio-Ramírez / VAR Toolbox | — | `var.identify.sign_restrictions(Y, signs, p=4)` |
| **SVAR con Proxy / IV externo** | `svariv` | Mertens & Ravn SVAR-IV | — | `var.identify.proxy_svar(Y, p=4, instrument_series=z)` |
| **Proyecciones locales (HAC)** | `jorda` / MCO manual | Código de Jordà (2005) | `OLS(y_h, X).fit(cov_type='HAC')` | `lp.lp_hac(df, y="y", x="shock", horizon=20, lags=4)` |
| **LP-IV dependiente de estado** | 2SLS con interacción manual | — | — | `lp.lp_state_dep_iv(df, y="y", x="g", z="news", state="u")` |
| **PL de panel (Driscoll–Kraay)** | `xtscc` | Panel LP toolbox | `PanelOLS(..., cov_type='driscoll-kraay')` | `lp.panel_lp_dk(df, y="y", x="z", unit_col="id", time_col="t")` |
| **GMM de panel dinámico** | `xtabond2 y L.y, gmm(y) two robust` | Arellano–Bond MATLAB | — | `dynpanel.ab_gmm(y, panel_id, time_id, two_step=True, windmeijer=True)` |
| **DiD escalonado** | `csdid y, ivar(id) time(t) gvar(g)` | — | — | `did.callaway_santanna(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **DiD sintético** | `sdid y id t d` | synthdid paquete R | — | `did.synthetic_did(df, unit="id", time="t", outcome="y", treatment="d")` |
| **VAR aumentado con factores (FAVAR)** | — | BBE (2005) MATLAB | — | `var.favar(panel_df, policy_series, n_factors=3, horizon=20)` |
| **Iteración de función de valor** | — | VFIToolkit `ValueFnIter_Case1` | — | `vfi.VFIProblem(a_grid, z_grid, P_z, return_fn, beta).solve()` |
| **DSGE lineal (QZ / BK)** | — | Dynare `stoch_simul` / Klein `solab` | — | `dsge.klein.klein_solve(A, B, C, n_pre=...)` |
| **DSGE a partir de ecuaciones** | — | Dynare `.mod` file | — | `dsge.build(equations, variables=..., states=..., shocks=...)` |
| **Raíz unitaria GLS (DF-GLS)** | `dfgls y, maxlag(4)` | Código ERS (1996) | `adfuller` | `unit_root.dfgls_test(y, regression="ct")` |
| **Ajuste estacional** | `x13 y` | Wrapper X-13 | `STL` / `x13` | `sa.stl_sa(y)` / `sa.x11_sa(y)` |

Las replicaciones de extremo a extremo de artículos canónicos se encuentran en `puremacro/examples/` — Bloom 2009 (`bloom2009.py`), SVAR narrativo de Mertens-Ravn (`svariv_mertens_ravn.py`), narrativa monetaria de Romer-Romer (`romer_romer_*.py`) y aproximadamente 60 más. La mayoría (como el ejemplo de Uhlig anterior) son completamente sintéticos y no requieren datos ni claves; algunos leen datos incluidos en el paquete o descargados en línea.

## Documentación

- **`docs/lp.md`** — Guía de proyecciones locales (LP-HAC, LP-IV, LP-IV dependiente de estado, LP de panel, `LPResult`).
- **`docs/did.md`** — Diferencias en diferencias modernas (Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, DiD sintético).
- **`docs/reporting.md`** — Exportación para publicaciones (LaTeX, Typst, Markdown, estrellas de significancia).
- **`docs/var.md`** — VAR en forma reducida, identificación de SVAR, FAVAR, bandas bootstrap.
- **`ARCHITECTURE.md`** — mapa de módulos, niveles de estabilidad, contrato con Pyodide, estándar de objetos de resultado. Léalo antes si va a contribuir o busca dónde vive algo.
- **`CHANGELOG.md`** — diferencias por versión, incluidas las refactorizaciones internas.
- **`docs/es/ADVISORY.md`** — avisos de corrección: versiones publicadas que devolvieron un número equivocado, y la condición exacta bajo la cual cada error se anula.
- **Docstrings por función** como referencia canónica; el docstring de módulo de cada subpaquete explica su alcance.

## Convenciones

- **API pública por subpaquete** curada mediante `__init__.py::__all__`; el paquete de nivel superior `puremacro` solo reexporta `__version__`.
- **Objetos de resultado como dataclass congelado** para cualquier estimador que devuelva 3 o más campos o diagnósticos no triviales (véase `ARCHITECTURE.md` § Result-object standard). Los DataFrames con columnas nombradas quedan exentos.
- **Errores de diagnóstico en lugar de resultados silenciosos incorrectos** — `X'X` singular, Σ no definida positiva, violaciones de la condición de Blanchard-Kahn y replicaciones bootstrap mal condicionadas generan excepciones o advertencias que identifican la función invocante y la causa probable.

## Estado

Versión de producción, distribuyendo **2.0.0**. Cubierto por el gate 3 de publicación en `docs/1.0_path.md` § 5 enumera qué subpaquetes están dentro de esa promesa y cuáles son experimentales.

La CI está activa y corre en cada push: la suite sobre tres sistemas operativos y tres versiones de Python, el contrato con Pyodide, mypy, la guardia de deriva contra referencias, `mkdocs build --strict`, el despliegue del playground y una publicación en PyPI disparada por etiqueta mediante trusted publishing. Véase `.github/workflows/`. Aun así ejecute `python tools/release_check.py` localmente antes de etiquetar: los gates 5 y 6 son opcionales y la CI no los corre.

Cuando una versión publicada devolvió un número equivocado, queda registrado en **[`docs/es/ADVISORY.md`](docs/es/ADVISORY.md)**, junto con la condición bajo la cual el error se anula, para que pueda descartar su propia estimación.
