> 🇬🇧 [English](README.md) · 🇪🇸 Español

# puremacro

Una **caja de herramientas de macroeconomía empírica compatible con Pyodide**: el código de los estimadores corre sobre numpy + scipy + pandas + matplotlib puros, de modo que el núcleo numérico sigue siendo importable bajo Pyodide (iPad / juno.sh, en la medida de lo posible — véase «juno.sh / iPad» más abajo). El destino soportado es una **instalación local** en una estación de trabajo convencional.

## Inicio rápido en 5 minutos (API unificada 2.0)

`puremacro 2.0` estandariza la API econométrica con convenciones de parámetros comunes (`lags`, `horizon`, `ci`), objetos de resultado dataclass congelados, visualización interactiva (`.plot()`) y exportación directa para publicación (`.to_latex()`, `.to_typst()`, `.to_markdown()`):

### 1. Proyecciones Locales (LP) y Exportación para Publicación
```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# Serie de tiempo macroeconómica sintética
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
gdp = np.cumsum(0.7 * shock + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "shock": shock})

# API unificada: horizonte, retardos, nivel de confianza
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# Visualización instantánea
res.plot(title="Respuesta del PIB ante Choque de Política")

# Exportación directa a LaTeX o Typst para tu artículo
print(res.to_latex())
print(res.to_typst())
```

### 2. DSGE de Segundo Orden y Paridad con Dynare
Resuelve modelos DSGE no lineales hasta segundo orden con poda (*pruning*) de Kim, Kim, Schaumburg y Sims (2008), términos cruzados ($g_{xu}, g_{uu}$), corrección por riesgo ($g_{\sigma\sigma}$) y reglas de decisión compatibles con `oo_.dr`:
```python
from puremacro.dsge import load_mod

# 1. Un archivo .mod de Dynare: una ruta o, como aquí, el propio texto
rbc_mod = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;

alpha = 0.30;
beta  = 0.99;
delta = 0.025;
gamma = 1.0;
rho   = 0.80;

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;

initval;
  k = 38.0;
  a = 0.0;
  c = 2.0;
end;

shocks;
  var eps; stderr 0.01;
end;
"""
model = load_mod(rbc_mod)   # LinearModel de primer orden (load_mod(rbc_mod, order=2) va directo al segundo orden)

# 2. Resolver la perturbación de segundo orden con poda
sol = model.solve(order=2)

# 3. Reglas de decisión al estilo Dynare (oo_.dr)
print(sol.oo_dr["ghx"])   # transición de estados de primer orden
print(sol.oo_dr["ghxx"])  # curvatura de segundo orden
print(sol.oo_dr.summary())

# 4. Momentos teóricos analíticos y descomposición de varianza
mom = sol.theoretical_moments()
print(mom.summary())
print(mom.to_latex())
```

### 3. Descarga de Cómputo desde Juno / iPad hacia Google Colab
Cuando trabajes en un iPad o en sesiones cliente de Pyodide con límites de memoria o CPU, descarga tareas intensivas a Google Colab sin fricción:
```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

# 1. Empaqueta la tarea pesada en un cuaderno autocontenido (con celdas de autenticación
#    y montaje de Drive). La variable `result` de la tarea se exporta como cartucho .pmz.
nb = generate_colab_notebook(
    """
import puremacro as pm
result = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
""",
    mount_drive=True,
    save_path="sw07_offload.ipynb",
    output_filename="sw07_posterior.pmz",
)

# 2. Muestra las instrucciones (tarjeta HTML en Juno / Jupyter, texto en la terminal)
show_colab_offload_dialog("sw07_offload.ipynb")

# 3. Cuando el cartucho vuelva desde Google Drive, cárgalo en la sesión local:
#    posterior = load_colab_result("sw07_posterior.pmz")
```

---

## Contenido

**Econometría central**

- **VAR** — mínimos cuadrados ordinarios en forma reducida, BVAR (Minnesota), VECM (Engle-Granger / Johansen), TVP-VAR, VAR de panel; FRI / FEVD / GFEVD; bandas de confianza mediante bootstrap de residuos, por bloques, por bloques móviles y wild bootstrap.
- **Identificación de SVAR** (`var.identify.*`) — Cholesky, Blanchard-Quah, restricciones de signo (Rubio-Ramirez-Waggoner-Zha), restricciones de signo y cero (Arias-Rubio Ramirez-Waggoner), bandas robustas a las restricciones de signo (Giacomini-Kitagawa), variables proxy / instrumentos externos, máxima participación espectral / noticias, heterocedasticidad (Rigobon), no gaussiano (Lanne-Meitz-Saikkonen). Todos los estimadores públicos devuelven objetos `…Result` de tipo dataclass congelado.
- **Proyecciones locales** (`lp.*`) — LP-HAC para un solo país, LP-IV, LP con retardos aumentados (Plagborg-Møller-Wolf), LP de panel con errores estándar agrupados / Driscoll-Kraay, LP dependiente del estado, LP suavizada (B-splines de Barnichon-Brownlees), LP asimétrica (Tenreyro-Thwaites), LP-GARCH en estado, LP-GARCH en media, grupo medio, CCE, LP cuantílica.
- **Inferencia** (`inference.*`) — MCO con HAC central, Newey-West, Kiefer-Vogelsang de b fijo, Driscoll-Kraay; diagnósticos de instrumentos débiles (Cragg-Donald, Kleibergen-Paap, Anderson-Rubin, Montiel Olea-Pflueger); Hansen-J / Stock-Yogo para sobreidentificación; CD de Pesaran, homogeneidad de pendientes de Swamy, quiebres estructurales de Quandt-Andrews, curvas de especificación.
- **Otros estimadores** — índice de derrame de Diebold-Yilmaz; comparación de pronósticos Diebold-Mariano / Giacomini-White y evaluación de pronósticos en densidad (CRPS, log-score); quiebres de Bai-Perron; pruebas de raíz unitaria (ADF, KPSS, PP, Zivot-Andrews); solver QZ de Klein para DSGE lineales y perturbación de segundo orden con poda de Kim et al. (2008), términos cruzados y paridad con Dynare `oo_.dr` (`dsge.dynare`).

**Métodos de frontera (2.3)**

- **Restricciones narrativas de signo** (`var.identify.identify_narrative_sign`) — Antolín-Díaz y Rubio-Ramírez (2018): restricciones de signo del choque y de contribución histórica con pesos de importancia; véase `docs/es/narrative_sign_svar.md`.
- **DiD honesto** (`did.honest_did`) — Rambachan y Roth (2023): conjuntos de sensibilidad de suavidad y magnitud relativa con intervalos de longitud fija y valores de quiebre; véase `docs/es/honest_did.md`.
- **Proyecciones locales suavizadas** (`lp.smooth_lp`) — Barnichon y Brownlees (2019): LP penalizadas con B-splines y selección GCV/AIC/BIC/CV; véase `docs/es/smooth_lp.md`.
- **HANK no lineal en el espacio de secuencias** (`models.hank_sequence_space.solve_nonlinear_transition`) — transiciones de Broyden para choques MIT grandes sobre jacobianos genuinos de los hogares; véase `docs/es/hank_nonlinear.md`.
- **Gertler-Karadi (2011)** (`dsge.gertler_karadi`) — DSGE con fricciones bancarias bajo Klein y OccBin; véase `docs/es/gertler_karadi.md`.
- **BVAR con volatilidad estocástica** (`var.bvar_sv`) — muestreador de Gibbs KSC + FFBS, R̂ dividido, FRI condicionadas a la volatilidad, puntuaciones logarítmicas fuera de muestra y gráficos de abanico; véase `docs/es/bvar_sv.md`.

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
- **Agentes heterogéneos / VFI / HANK en el espacio de secuencias** (`vfi.*`, `models.hank_sequence_space`) — iteración sobre la función de valor con EGM, ciclo de vida de horizonte finito, OLG, choques agregados de Krusell-Smith, entrada/salida de empresas de Hopenhayn, Epstein-Zin, tipos permanentes, trayectorias de transición y estimación por método de momentos. Además, HANK completo en el espacio de secuencias (Auclert et al. 2021) con el algoritmo Fake News en $\mathcal{O}(T^2)$ (`fake_news_algorithm`, `FakeNewsResult`) y simulaciones de transferencias fiscales focalizadas entre deciles de riqueza (`simulate_targeted_transfer`, `FiscalTransferResult`); backend de referencia en numpy con aceleración opcional mediante numba / mlx / cupy. Véanse los cuadernos en `notebooks/` para una galería de ejemplos.

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

**Trabajar fuera de una estación de trabajo** (`runtime.*`, `runtime.colab`, `pocket.*`, `longrun.*`)

La promesa central del paquete es que el núcleo de estimadores corre en un iPad. Estos módulos hacen que esa promesa sea utilizable, y no sólo cierta: `runtime` informa de lo que la máquina puede hacer realmente (¿sockets? ¿parquet? ¿hilos?) y encamina el HTTP por el navegador cuando no hay sockets; `runtime.colab` proporciona descarga transparente de tareas pesadas a Google Colab con sincronización y persistencia en `.pmz`; `pocket` empaqueta datos en cartuchos `.pmz` portátiles y autoverificables, de modo que un panel construido en línea se abre sin conexión; `longrun` ejecuta bootstraps y cadenas en trozos reanudables que sobreviven a la suspensión de la aplicación por parte del sistema, con resultados invariantes al modo en que se troceó el trabajo. Véase «juno.sh / iPad» más abajo.

**Cuaderno de bocetos DSGE, paridad con Dynare y motores de frontera** (`dsge.build`, `dsge.dynare`, `dsge.cli`, `dsge.occbin`, `dsge.bayesian`, `dsge.perfect_foresight`)

Escriba las condiciones de equilibrio como una función de Python o cargue archivos `.mod` estándar de Dynare (`load_mod`, `parse_mod`). Resuelve aproximaciones de 1er y 2do orden con diferenciación de paso complejo, poda (*pruning*) de Kim, Kim, Schaumburg y Sims (2008), derivadas cruzadas ($g_{xu}, g_{uu}$), correcciones por riesgo ($g_{\sigma\sigma}$), reglas de decisión de Dynare `oo_.dr` y momentos teóricos analíticos (`stoch_simul`). Incluye la herramienta de línea de comandos `puremacro-dynare`, el algoritmo lineal por tramos OccBin de Guerrieri e Iacoviello (2015) para cotas de tasa cero (ZLB), relajación no lineal de Newton-Raphson de Boucekkine-Juillard para previsión perfecta y estimación bayesiana completa por MCMC (Laplace + Metropolis-Hastings adaptativo). Sin matrices derivadas a mano, sin compiladores Fortran/C++, 100% compatible con Pyodide.

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
# requiere: un kernel de Jupyter (magia de IPython)
%pip install ./puremacro
```

**Advertencia desde que `pyarrow` es dependencia base:** ese comando resuelve el conjunto completo de dependencias y `pyarrow` no tiene rueda para Pyodide, así que bajo un núcleo Pyodide falla. Instale el núcleo de estimadores sin resolución de dependencias y añada a mano sólo lo que necesite:

```python
# requiere: Pyodide (JupyterLite / juno.sh); `await` solo es válido allí
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
import numpy as np
import pandas as pd
from puremacro import pocket

# Un DataFrame para transportar (en la práctica: el panel recién descargado)
panel = pd.DataFrame(
    {"USA": [1.0, 1.2, 1.1], "DEU": [0.8, 0.9, 1.0]},
    index=pd.period_range("2025Q1", periods=3, freq="Q"),
)

# estación de trabajo
pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

# iPad, modo avión
cart = pocket.load("g7.pmz")
panel = cart.frame()          # verificado por SHA-256 al leer
cart.provenance.vintage       # '2026-08-19'
print(cart.summary())
```

Llevar un *archivo* a un iPad suele costar más que el propio análisis, así que un cartucho también viaja como texto: `pocket.to_base64("g7.pmz")` en una máquina y `pocket.from_base64(blob, "g7.pmz")` en la otra.

**La aplicación se suspende.** iPadOS detiene una aplicación en segundo plano, y un bootstrap de cuatro minutos no sobrevive a que alguien conteste un mensaje. `puremacro.longrun` calcula por trozos, guarda tras cada uno y se reanuda en una sesión posterior:

```python
import numpy as np
from puremacro import longrun

rng = np.random.default_rng(0)
Y = rng.standard_normal((120, 2))          # sus datos del VAR

def one_draw(i):
    """Una replicación bootstrap: devuelve aquello de lo que quiere las bandas."""
    idx = np.random.default_rng(i).integers(0, len(Y), len(Y))
    return Y[idx].mean(axis=0)

job = longrun.bootstrap(one_draw, 2000, checkpoint="irf.ckpt")
job.run(seconds=30)     # 240/2000 · 12% · ~220 s de cómputo restantes
job.run(seconds=30)     # ... y otra vez tras la suspensión de la aplicación
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
# requiere: un motor LLM local (llama-cpp-python o mlx-lm) y un corpus `records`
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
| **Restricciones de signo** | Plugin de usuario | Rubio-Ramírez / VAR Toolbox | — | `var.identify.sign_restrictions(Y, restrictions={0: [+1, -1]}, p=4)` |
| **SVAR con Proxy / IV externo** | `svariv` | Mertens & Ravn SVAR-IV | — | `var.identify.proxy_svar(Y, p=4, instrument_series=z)` |
| **Proyecciones locales (HAC)** | `jorda` / MCO manual | Código de Jordà (2005) | `OLS(y_h, X).fit(cov_type='HAC')` | `lp.lp_hac(df, y="y", x="shock", horizon=20, lags=4)` |
| **LP-IV dependiente de estado** | 2SLS con interacción manual | — | — | `lp.lp_state_dep_iv(df, y="y", x="g", z="news", state="u")` |
| **PL de panel (Driscoll–Kraay)** | `xtscc` | Panel LP toolbox | `PanelOLS(..., cov_type='driscoll-kraay')` | `lp.panel_lp_dk(df, y="y", x="z", unit_col="id", time_col="t")` |
| **GMM de panel dinámico** | `xtabond2 y L.y, gmm(y) two robust` | Arellano–Bond MATLAB | — | `dynpanel.ab_gmm(y, panel_id, time_id, two_step=True, windmeijer=True)` |
| **DiD escalonado** | `csdid y, ivar(id) time(t) gvar(g)` | — | — | `did.callaway_santanna(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **DiD sintético** | `sdid y id t d` | synthdid paquete R | — | `did.synthetic_did(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **VAR aumentado con factores (FAVAR)** | — | BBE (2005) MATLAB | — | `var.favar(panel_df, policy_series, n_factors=3, horizon=20)` |
| **Iteración de función de valor** | — | VFIToolkit `ValueFnIter_Case1` | — | `vfi.VFIProblem(a_grid, z_grid, P_z, return_fn, beta).solve()` |
| **DSGE lineal (QZ / BK)** | — | Dynare `stoch_simul` / Klein `solab` | — | `dsge.klein.klein_solve(A, B, C, n_pre=...)` |
| **DSGE a partir de ecuaciones / .mod** | — | Dynare `.mod` file | — | `dsge.load_mod("rbc.mod")` / `dsge.build_dynare(eqs)` |
| **DSGE con poda de 2do orden** | — | Dynare `stoch_simul(order=2, pruning)` | — | `dsge.build_dynare(eqs, order=2)` / `m.solve_second_order()` |
| **Línea de comandos `puremacro-dynare`** | — | `dynare model.mod` en terminal | — | `puremacro-dynare model.mod --order 2 --fevd --plot` |
| **OccBin (ZLB / lineal por tramos)** | — | Dynare `occbin_solver` / Guerrieri & Iacoviello | — | `dsge.solve_occbin(m_normal, m_zlb, constraint, shocks)` |
| **Previsión perfecta no lineal** | — | Dynare `simul` (Boucekkine-Juillard) | — | `dsge.solve_perfect_foresight(m, shocks, T=100)` |
| **DSGE Bayesiano (MCMC)** | — | Dynare `estimation(...)` (Metropolis-Hastings) | — | `dsge.estimate_dsge_bayesian(m, data, priors, n_draws=10000)` |
| **Fake News en espacio de secuencias** | — | SSJ (Auclert et al. 2021) Python/Julia | — | `models.fake_news_algorithm(T=40)` / `models.simulate_targeted_transfer(...)` |
| **Raíz unitaria GLS (DF-GLS)** | `dfgls y, maxlag(4)` | Código ERS (1996) | `adfuller` | `unit_root.dfgls_test(y, regression="ct")` |
| **Ajuste estacional** | `x13 y` | Wrapper X-13 | `STL` / `x13` | `sa.stl_sa(y)` / `sa.x11_sa(y)` |

Las replicaciones de extremo a extremo de artículos canónicos se encuentran en `puremacro/examples/` — Bloom 2009 (`bloom2009.py`), SVAR narrativo de Mertens-Ravn (`svariv_mertens_ravn.py`), narrativa monetaria de Romer-Romer (`romer_romer_*.py`), escaparate de frontera de Smets-Wouters 2007 (`41_dynare_frontier_showcase.py`) y aproximadamente 75 más. La mayoría (como el ejemplo de Uhlig anterior) son completamente sintéticos y no requieren datos ni claves; algunos leen datos incluidos en el paquete o descargados en línea.

## Documentación

- **`docs/es/quickstart.md`** — Guía de inicio rápido en 2 minutos cubriendo estimadores principales y exportación para publicaciones.
- **`docs/es/dsge_build.md`** — Modelos DSGE desde ecuaciones, cargador de archivos `.mod`, poda de 2do orden, CLI `puremacro-dynare`, OccBin ZLB, relajación no lineal y MCMC bayesiano.
- **`docs/es/models.md`** — Modelos estructurales: HANK en el espacio de secuencias, algoritmo Fake News, transferencias focalizadas y búsqueda y emparejamiento DMP.
- **`docs/es/narrative_sign_svar.md`**, **`docs/es/honest_did.md`**, **`docs/es/smooth_lp.md`**, **`docs/es/hank_nonlinear.md`**, **`docs/es/gertler_karadi.md`**, **`docs/es/bvar_sv.md`** — las seis guías de las funciones 2.3.
- **`docs/es/var.md`** — VAR en forma reducida, identificación de SVAR (Cholesky, signos, narrativa, proxy/IV), FAVAR y bandas bootstrap.
- **`docs/es/lp.md`** — Guía de proyecciones locales (LP-HAC, LP-IV, LP dependiente de estado, LP de panel, `LPResult`).
- **`docs/es/did.md`** — Diferencias en diferencias modernas (Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, DiD sintético).
- **`docs/es/nowcast.md`** — Nowcasting del PIB (modelos de factores dinámicos de frecuencias mixtas, bordes irregulares, descomposición de noticias).
- **`docs/es/climate.md`** — Macroeconomía del clima: simulador hacia adelante del modelo DICE de Nordhaus y contabilidad del coste social del carbono.
- **`docs/es/forecast.md`** — Pronóstico macroeconómico penalizado: Elastic Net y Lasso Adaptativo mediante descenso por coordenadas.
- **`docs/es/reporting.md`** — Exportación para publicaciones (tablas LaTeX, Typst, Markdown, estrellas de significancia).
- **`docs/es/tablet.md`** — Ejecución en iPad, Juno y WebAssembly, con descarga de cómputo a Google Colab.
- **`docs/es/benchmarks.md`** — Rendimiento y pruebas comparativas de velocidad computacional entre motores econométricos.
- **`docs/es/national_accounts.md`** — Extracción de cuentas nacionales trimestrales de la OCDE, deflactores e identidades contables.
- **`docs/es/real_time_data.md`** — Añadas de datos en tiempo real, triángulos de revisión y contraste de noticia frente a ruido de Mankiw-Shapiro.
- **`docs/es/long_panel.md`** — Panel largo histórico de cuentas nacionales (series empalmadas por ratio de España y Japón).
- **`docs/es/examples_gallery.md`** — Galería exhaustiva con el estado de ejecución y gráficos de los ejemplos del paquete.
- **`docs/es/ADVISORY.md`** — Avisos de corrección: versiones publicadas que devolvieron un número equivocado y condición de anulación.
- **`ARCHITECTURE.md`** — Mapa de módulos, niveles de estabilidad, contrato con Pyodide, estándar de objetos de resultado.
- **`CHANGELOG.md`** — Diferencias por versión, incluidas las refactorizaciones internas.
- **Docstrings por función** como referencia canónica; el docstring de módulo de cada subpaquete explica su alcance.

## Convenciones

- **API pública por subpaquete** curada mediante `__init__.py::__all__`; el paquete de nivel superior `puremacro` solo reexporta `__version__`.
- **Objetos de resultado como dataclass congelado** para cualquier estimador que devuelva 3 o más campos o diagnósticos no triviales (véase `ARCHITECTURE.md` § Result-object standard). Los DataFrames con columnas nombradas quedan exentos.
- **Errores de diagnóstico en lugar de resultados silenciosos incorrectos** — `X'X` singular, Σ no definida positiva, violaciones de la condición de Blanchard-Kahn y replicaciones bootstrap mal condicionadas generan excepciones o advertencias que identifican la función invocante y la causa probable.

## Estado

Versión de producción, distribuyendo **2.3.1**. `docs/1.0_path.md` § 5 enumera qué subpaquetes están dentro de la promesa del gate de publicación y cuáles son experimentales.

La CI está activa y corre en cada push: la suite sobre tres sistemas operativos y tres versiones de Python, el contrato con Pyodide, mypy, la guardia de deriva contra referencias, `mkdocs build --strict`, el despliegue del playground y una publicación en PyPI disparada por etiqueta mediante trusted publishing. Véase `.github/workflows/`. Aun así ejecute `python tools/release_check.py` localmente antes de etiquetar: los gates 5 y 6 son opcionales y la CI no los corre.

Cuando una versión publicada devolvió un número equivocado, queda registrado en **[`docs/es/ADVISORY.md`](docs/es/ADVISORY.md)**, junto con la condición bajo la cual el error se anula, para que pueda descartar su propia estimación.
