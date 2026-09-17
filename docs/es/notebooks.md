> 🇬🇧 [English](../notebooks.md) · 🇪🇸 Español

# Cuadernos de Demostración

`puremacro` incluye cuadernos de demostración interactivos con calidad de publicación que cubren los principales paradigmas de macroeconomía con agentes heterogéneos, econometría empírica, estimación bayesiana, aplicaciones especializadas de clima e historia macroeconómica y suites de frontera para política macroeconómica aplicada.

Cada cuaderno está escrito en formato Jupytext percent (`.py`), se ejecuta de forma determinista con semillas fijas, cumple con el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`) y cuenta con una edición gemela en español (`_es.py`). Los artefactos `.ipynb` precompilados con salidas y figuras prerrenderizadas se generan mediante `tools/build_notebooks.py`.

---

## La Arquitectura Pedagógica de 7 Secciones

Siguiendo `notebooks/_TEMPLATE.md`, los cuadernos profundizados y de frontera siguen un flujo estructural uniforme de 7 celdas:

1. **Pregunta Motivadora**: 1–2 oraciones que definen el problema económico y el dilema de política.
2. **El Método en Matemáticas**: Ecuaciones estructurales y econométricas en LaTeX riguroso y compacto ($...$ / $$...$$).
3. **Intuición**: Sección explícita `**Intuición.**` que traduce las ecuaciones algebraicas a mecanismos económicos intuitivos y lógica de identificación.
4. **Código Desarrollado**: Bloques ejecutables en NumPy puro, autocontenidos y comentados explicando el *porqué* de cada decisión.
5. **Lectura de Resultados**: Análisis narrativo en markdown que interpreta directamente los valores numéricos principales, estimaciones y figuras.
6. **Tu Turno**: Ejercicio exploratorio interactivo con controles `# ← modifica esto`, valores por defecto funcionales con aserciones y retos graduados.
7. **¿Qué tan Exhaustivo es Esto?**: Referencias contextuales que vinculan la demostración con otros módulos de `puremacro` y la literatura.

## Nowcasting de América Latina, DML Interactivo y Simuladores Cuantitativos de Política (puremacro 3.5)

Las demostraciones `59` a `61` presentan las capacidades de vanguardia de `puremacro` en nowcasting en tiempo real para América Latina con atribución de noticias de Bańbura-Modugno (2014) y evaluación de densidades de Berkowitz (2001), inferencia causal en alta dimensión mediante Aprendizaje Automático Doble Interactivo (IRM y DML-IV) con descenso por coordenadas logístico regularizado y diagnósticos de $F$ efectiva de Montiel Olea y Pflueger, y simuladores cuantitativos de política macroeconómica en equilibrio general comercial con la matriz ICIO de la OCDE (77 países, 11 sectores) y transmisión monetaria en HANK en el espacio de secuencias.

### `59_latam_realtime_nowcast_and_news_es`
- **Fuente**: `notebooks/59_latam_realtime_nowcast_and_news_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo pueden los bancos centrales y departamentos de estudios económicos de América Latina realizar el nowcasting del crecimiento trimestral del PIB en tiempo real a partir de paneles de indicadores mensuales con extremos irregulares (*ragged edges*), descomponer las actualizaciones del pronóstico en sorpresas de publicación y revisiones estadísticas usando la identidad exacta de Bańbura y Modugno (2014), y validar la calibración de la densidad predictiva mediante pruebas de uniformidad PIT de Berkowitz (2001)?
- **Matemáticas y Algoritmos Rectores**:
  - **Modelo de Factores Dinámicos en Forma Estado-Espacio**: Indicadores estandarizados $X_t = \Lambda F_t + \xi_t$ explicados por factores latentes $F_t = \sum_{l=1}^p A_l F_{t-l} + u_t$. Estimación en dos etapas mediante componentes principales y filtro/suavizador de Kalman (Doz, Giannone y Reichlin 2011) sobre paneles desbalanceados.
  - **Atribución Exacta de Noticias de Bańbura y Modugno (2014)**:
    $$\Delta \hat{y}_{t^*|v} \equiv \hat{y}_{t^*|v} - \hat{y}_{t^*|v-1} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot I_{j, v} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot R_{k, v}$$
    Las ponderaciones $\omega$ provienen de la ganancia de Kalman y autocovarianzas de estado, satisfaciendo la identidad matemática exacta con residuo cero ($|\Delta \hat{y}_{t^*|v} - \sum \text{Impacto}| < 10^{-10}$).
  - **Evaluación de Densidad Predictiva con Prueba de Razón de Verosimilitud de Berkowitz (2001)**: Transformada Integral de Probabilidad $p_t = \Phi((y_t - \mu_t)/\sigma_t)$ transformada a errores normales $z_t = \Phi^{-1}(p_t)$ con autorregresión $(z_t - \mu) = \rho (z_{t-1} - \mu) + \varepsilon_t$ evaluada bajo $H_0: \mu = 0, \sigma_\varepsilon^2 = 1, \rho = 0$ con estadístico $\text{LR} \sim \chi^2(3)$.
  - **Cartuchos de Datos para América Latina**: Conectores en tiempo real para México (INEGI, Banxico) y Brasil (BCB) empaquetados en contenedores portátiles fuera de línea `.pmz` con verificación criptográfica SHA-256.
- **Intuición Económica**: Los desfases asíncronos en los calendarios estadísticos generan un extremo irregular. El Modelo de Factores Dinámicos extrae las fluctuaciones comunes pese a los valores faltantes. Al publicarse nuevos datos, la descomposición de Bańbura-Modugno discrimina si la revisión del pronóstico obedece a sorpresas genuinas respecto a las expectativas del modelo o a ajustes retrospectivos de los institutos de estadística. Los abanicos de densidad calibrados ofrecen intervalos creíbles para las decisiones de política.
- **Código Desarrollado**:
  ```python
  import numpy as np
  from puremacro.fetch.realtime import VintagePanel, pack_realtime_cartridge, load_realtime_cartridge
  from puremacro.nowcast import (
      DynamicFactorModel, realtime_nowcast, banbura_modugno_news,
      fan_chart, pit_uniformity_test,
  )

  # Nowcasting con DFM bajo paneles irregulares
  dfm = DynamicFactorModel(n_factors=2, factor_lags=1)
  dfm.fit(X_train)
  nowcast_res = dfm.nowcast(target_series=y_gdp, X_eval=X_ragged)

  # Descomposición exacta de noticias Bańbura-Modugno
  news_res = banbura_modugno_news(dfm, y_target=y_gdp, vintage_old=p_old, vintage_new=p_new)
  news_res.total_revision, news_res.decomposition_error  # error < 1e-10

  # Calibración de densidades mediante prueba de Berkowitz
  pit_res = pit_uniformity_test(realizations=y_realized, means=mu_seq, stds=sigma_seq)
  pit_res.lr_stat, pit_res.p_value
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Trayectoria del nowcast del PIB en tiempo real con abanicos de incertidumbre; (2) Cascada de atribución de noticias de Bańbura-Modugno; (3) Contribuciones de sorpresas por indicador y sector; (4) Curva empírica de PIT de Berkowitz frente a la diagonal uniforme con bandas al 95 % de Kolmogorov-Smirnov.
- **Tu Turno y Aserciones Interactivas**: Parámetros de dimensión de factores, orden de rezagos y subconjuntos de indicadores con aserciones de convergencia e identidad exacta de descomposición.
- **Literatura y Referencias Cruzadas**: Bańbura y Modugno (2014), Giannone, Reichlin y Small (2008), Berkowitz (2001). Guías de usuario: [`docs/es/nowcast_latam_news.md`](nowcast_latam_news.md) y [`docs/es/real_time_latam.md`](real_time_latam.md).

### `60_interactive_dml_irm_and_iv_es`
- **Fuente**: `notebooks/60_interactive_dml_irm_and_iv_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo estiman los economistas el impacto causal de programas de ahorro voluntario —como la elegibilidad y participación en planes de pensiones 401(k)— sobre la riqueza financiera neta cuando la asignación depende de decenas de variables de confusión sociodemográficas no lineales, y cómo puede el Aprendizaje Automático Doble recuperar estimaciones no sesgadas del Efecto Medio del Tratamiento (ATE), del Efecto en los Tratados (ATT) y con Variables Instrumentales (DML-IV) sin sesgo de regularización?
- **Matemáticas y Algoritmos Rectores**:
  - **Modelo de Regresión Interactiva (IRM)**: Resultados potenciales $(Y(1), Y(0)) \perp D \mid X$ con funciones estructurales $Y = g_0(D, X) + U$ y propensión $D = m_0(X) + V$.
  - **Scores Ortogonales de Neyman Doblemente Robustos para ATE y ATT**:
    $$\psi_{\text{ATE}}(W; \theta, \eta) = g(1, X) - g(0, X) + \frac{D (Y - g(1, X))}{m(X)} - \frac{(1 - D) (Y - g(0, X))}{1 - m(X)} - \theta$$
    $$\psi_{\text{ATT}}(W; \theta, \eta) = \frac{D (Y - g(0, X))}{\mathbb{P}(D = 1)} - \frac{m(X)(1 - D)(Y - g(0, X))}{\mathbb{P}(D = 1)(1 - m(X))} - \theta$$
  - **Descenso por Coordenadas Logístico Regularizado en NumPy Puro**: Actualizaciones por umbralización suave $\beta_j^{\text{nueva}} = S(c_j \beta_j - g_j, \lambda)/c_j$ con cota superior de curvatura subrogada $c_j = \frac{1}{4N} \sum_i X_{ij}^2$, selección de penalización por BIC y truncamiento de solapamiento en $[\varepsilon, 1-\varepsilon]$.
  - **Variables Instrumentales con Doble ML (DML-IV)**: MC2E con ajuste cruzado sobre residuos ortogonales y diagnósticos de instrumento débil mediante el estadístico $F_{\text{eff}}$ de Montiel Olea y Pflueger (2013).
- **Intuición Económica**: La heterogeneidad en las respuestas y la presencia de variables de confusión invalidan los modelos lineales de efecto constante, mientras que un Lasso ingenuo introduce atenuación por encogimiento. DML-IRM combina modelos no lineales regularizados con scores ortogonales de Neyman y ajuste cruzado en $K$ particiones, preservando la normalidad asintótica y consistencia $\sqrt{N}$. Cuando la participación en el programa es endógena, DML-IV emplea la elegibilidad como instrumento excluido, evaluando formalmente la potencia del instrumento con el estadístico $F$ efectivo.
- **Código Desarrollado**:
  ```python
  import numpy as np
  from puremacro.causal import (
      DoubleMLIRM, DoubleMLIV, dml_irm, dml_iv,
      LogisticCoordinateDescent,
  )

  # DML-IRM: Efecto Medio del Tratamiento y sobre los Tratados
  irm_ate = DoubleMLIRM(ml_g="lasso", ml_m="logistic", n_folds=5, score="ATE", trimming_threshold=0.01)
  res_ate = irm_ate.fit(Y=activos_netos, D=e401, X=df_controles)
  res_ate.theta, res_ate.se, res_ate.n_trimmed

  # DML-IV: Tratamiento endógeno con F de Montiel Olea-Pflueger
  res_iv = dml_iv(Y=activos_netos, D=p401, Z=e401, X=df_controles, n_folds=5)
  res_iv.theta, res_iv.first_stage_effective_f, res_iv.weak_instrument
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Distribución de solapamiento de puntajes de propensión y cotas de truncamiento; (2) Comparación de estimadores causales (MCO ingenuo vs Lasso ingenuo vs DML-ATE vs DML-ATT vs DML-IV); (3) Diagrama de dispersión de la primera etapa residual de Montiel Olea-Pflueger; (4) Curva de calibración de la penalización $\ell_1$ de la propensión.
- **Tu Turno y Aserciones Interactivas**: Parámetros de truncamiento $\varepsilon$, número de particiones $K$ y penalizaciones de regularización con aserciones sobre cotas e intervalos de confianza.
- **Literatura y Referencias Cruzadas**: Chernozhukov et al. (2018), Belloni, Chernozhukov y Hansen (2014), Montiel Olea y Pflueger (2013). Guía de usuario: [`docs/es/dml_irm_iv.md`](dml_irm_iv.md).

### `61_quantitative_policy_simulators_es`
- **Fuente**: `notebooks/61_quantitative_policy_simulators_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo se propagan las disputas comerciales bilaterales y las escaladas arancelarias a través de los encadenamientos insumo-producto globales para alterar los términos de intercambio, la asignación sectorial y los salarios reales en equilibrio general, y cómo gobierna la heterogeneidad de riqueza e ingreso la transmisión de la política monetaria entre consumidores restringidos y tenedores de activos?
- **Matemáticas y Algoritmos Rectores**:
  - **Equilibrio General de Política Comercial Ricardiana (Caliendo y Parro 2015)**: Álgebra de cambios proporcionales exactos (*exact hat algebra*) sobre matrices insumo-producto internacionales:
    $$\hat{\pi}_{ni}^j = \left( \frac{\hat{\kappa}_{ni}^j \hat{c}_i^j}{\hat{P}_n^j} \right)^{-\theta_j}, \quad \hat{c}_i^j = \hat{w}_i^{\gamma_i^j} \prod_{k=1}^J (\hat{P}_i^k)^{\gamma_i^{j, k}}$$
    Resuelve el sistema de vaciado de mercados de bienes y factores determinando $\{\hat{w}_i\}$, descomponiendo el bienestar nacional en términos de intercambio, encadenamientos insumo-producto e ingresos arancelarios.
  - **Transmisión Monetaria en Espacio de Secuencias y Descomposición KMV (Kaplan et al. 2018; Auclert et al. 2021)**: Respuesta dinámica linealizada del consumo $d\mathbf{C} = \mathbf{J}^{C, r} d\mathbf{r} + \mathbf{J}^{C, Y} d\mathbf{Y}$. En RANK, $\mathbf{J}^{C, Y} = 0$ (100 % sustitución intertemporal directa). En HANK, los hogares mano a la boca con elevadas propensiones marginales a consumir generan amplios multiplicadores indirectos de ingreso laboral:
    $$\text{Proporción Indirecta} = \frac{(\mathbf{J}^{C, Y} d\mathbf{Y})_0}{dC_0} \times 100\%$$
- **Intuición Económica**: Los aranceles estatutarios producen desvío de comercio y cascadas de costos a lo largo de las cadenas de valor intermedias. El solucionador de Caliendo-Parro computa estos ajustes sin requerir la estimación de parámetros estructurales no observables. En el ámbito monetario, la concentración de riqueza líquida implica que los hogares con restricciones de liquidez transmiten la política a través de contracciones indirectas en el ingreso del trabajo y no mediante la suavización intertemporal de tasas de interés.
- **Código Desarrollado**:
  ```python
  import numpy as np
  from puremacro.trade.data import load_icio_data
  from puremacro.models import (
      TradePolicySimulator, MonetaryTransmissionSimulator,
  )

  # 1. Simulación de EG comercial sobre matriz ICIO OCDE (77 países, 11 sectores)
  trade_sim = TradePolicySimulator(load_icio_data(return_structured=True))
  trade_res = trade_sim.simulate_tariff_shock(
      tariffs={"USA": {"CHN": {"MANU": 0.25}}, "CHN": {"USA": {"MANU": 0.25}}},
  )
  trade_res.welfare_change["USA"], trade_res.welfare_change["MEX"]

  # 2. Transmisión monetaria en HANK vs RANK en espacio de secuencias
  mon_sim = MonetaryTransmissionSimulator()
  mon_res = mon_sim.simulate_monetary_shock(r_path=dr_seq)
  mon_res.direct_effect, mon_res.indirect_effect, mon_res.indirect_share
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Impacto en bienestar en equilibrio general por país; (2) Desvío sectorial de comercio bilateral y términos de intercambio; (3) Trayectorias de impulso-respuesta del consumo en HANK frente a RANK; (4) Descomposición directa e indirecta de Kaplan-Moll-Violante por deciles de riqueza.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos de aranceles de represalia, elasticidades comerciales y participación de hogares mano a la boca con aserciones de vaciado de mercados y descomposición del consumo.
- **Literatura y Referencias Cruzadas**: Caliendo y Parro (2015), Kaplan, Moll y Violante (2018), Auclert et al. (2021). Guía de usuario: [`docs/es/policy_simulators.md`](policy_simulators.md).

---

## Demostraciones de HJB/KFE Continuo, OccBin Multirrestricción y Macro en Tiempo Real de América Latina (puremacro 3.4)

Las demostraciones `56` a `58` presentan los motores de vanguardia de `puremacro` en resolución de Hamilton-Jacobi-Bellman (HJB) en tiempo continuo mediante esquema implícito upwind con ecuaciones de Kolmogorov hacia adelante (KFE) adjuntas, perturbación DSGE lineal a tramos multirrestricción (OccBin $M \ge 2$) combinada con Aprendizaje Automático Doble / Desesgado (DML-PLR), y cartuchos criptográficos portátiles `.pmz` de datos en tiempo real de América Latina con econometría de noticias frente a ruido de Mankiw-Shapiro (1986).

### `56_implicit_hjb_and_continuous_kfe_es`
- **Fuente**: `notebooks/56_implicit_hjb_and_continuous_kfe_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo resuelven los macroeconomistas cuantitativos modelos de agentes heterogéneos en tiempo continuo con riesgo de ingreso no asegurable y límites de endeudamiento sin incurrir en errores de inversión de la ecuación de Euler en tiempo discreto, y cómo asegura la ecuación adjunta de Kolmogorov hacia adelante (KFE) la conservación exacta de la masa de probabilidad a nivel de máquina para distribuciones estacionarias de riqueza?
- **Matemáticas y Algoritmos Rectores**:
  - **HJB en Tiempo Continuo con Esquema Upwind Implícito**: Los hogares resuelven:
    $$\rho v_j(a) = u(c_j(a)) + s_j(a) v_j'(a) + \sum_{k \ne j} \lambda_{jk} [v_k(a) - v_j(a)]$$
    Sujeto a $s_j(a) = r a + z_j - c_j(a)$ y $a \ge \underline{a}$. Las diferencias finitas upwind evalúan derivas hacia adelante ($v_{j, i}'^F$) y hacia atrás ($v_{j, i}'^B$), seleccionando hacia adelante si $s_F > 0$, hacia atrás si $s_B < 0$, y deriva nula $s_0 = 0$ en puntos de inflexión estacionarios.
  - **Estructura de M-Matriz del Generador Infinitesimal $A$**:
    $$\rho v^{n+1} = u(c^n) + A(v^n) v^{n+1} + \frac{1}{\Delta} (v^{n+1} - v^n)$$
    La matriz $(\rho + 1/\Delta) I - A(v^n)$ es estrictamente diagonal dominante con elementos fuera de la diagonal negativos, garantizando existencia, unicidad y estabilidad incondicional para cualquier $\Delta > 0$.
  - **Ecuación Adjunta de Kolmogorov hacia Adelante (KFE)**:
    $$A(v^*)^\top g = 0, \quad \text{s.a.} \quad \sum_{j=1}^J \sum_{i=1}^I g_{j, i} \Delta a_i = 1.0$$
    Sustituir una fila por la condición de normalización asegura la conservación de masa hasta el umbral del cuaderno ($|\sum g_i \Delta a_i - 1| \le 10^{-12}$; la ejecución anterior alcanza $1{,}1 \times 10^{-16}$). En una malla no uniforme el vector nulo es la *masa* por nodo, y `solve_kfe_achdou` la divide entre los pesos de cuadratura del ancho de celda, de modo que lo que se obtiene es una *densidad*.
  - **Equilibrio General Continuo de Aiyagari**:
    $$K^s(r) = \sum_{j, i} a_i g_{j, i} \Delta a_i = K^d(r) = \left( \frac{r + \delta}{\alpha \bar{Z}} \right)^{\frac{1}{\alpha - 1}}$$
- **Intuición Económica**: Los modelos de tiempo discreto requieren una agregación temporal fina y sufren de fluctuaciones en la frontera del endeudamiento. La formulación en tiempo continuo reemplaza las elecciones discretas con una función de deriva suave $s(a, z)$. Las diferencias finitas upwind reflejan la dirección física de los flujos de activos: los hogares que acumulan activos miran hacia adelante ($v_{i+1} - v_i$), mientras que aquellos que desacumulan activos miran hacia atrás ($v_i - v_{i-1}$), eliminando oscilaciones no físicas. Dado que el generador de transición $A$ es un generador de Markov infinitesimal, su adjunto $A^\top$ entrega la densidad estacionaria exacta de riqueza $g(a, z)$ en una única resolución lineal, preservando la conservación de masa sin ruido de simulación de Monte Carlo.
- **Código Desarrollado**:
  ```python
  import numpy as np
  from puremacro.vfi import solve_hjb_achdou, solve_aiyagari_continuous_hjb

  # Solucionador implícito HJB en tiempo continuo (atención a los nombres de
  # los parámetros: r_rate / w_rate / rho_val / gamma_r / Na, no r / w / rho /
  # gamma / n_a)
  sol = solve_hjb_achdou(
      r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0,
      Na=100, a_min=0.0, a_max=30.0, tol=1e-8, max_iter=100,
  )
  sol.converged, sol.n_iter          # True, 9
  sol.V.shape, sol.c_policy.shape    # (100, 2), (100, 2)
  sol.mass_residual                  # 1.11e-16  (umbral de la KFE: <= 1e-12)

  da = sol.a_grid[1] - sol.a_grid[0]
  mass = float(np.sum(sol.g_dist * da))   # 1.0 — la densidad es `g_dist`

  # Equilibrio general continuo de Aiyagari: bisección en r hasta Ks(r) = Kd(r)
  ge = solve_aiyagari_continuous_hjb(
      alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0,
      Na=40, a_max=25.0, tol_ge=1e-4, max_iter_ge=30,
  )
  ge.converged, ge.r_star, ge.K_star      # True, 0.018879, 6.2190
  ```
  Desde la 3.4.0, `solve_hjb_achdou` incorpora el término de conmutación de
  ingresos de Poisson entre dos estados que el solucionador explícito de la
  3.3.0 omitía (generador simétrico, $\lambda = 0{,}1$ por omisión), de modo que
  los estados de productividad ya no son problemas independientes de ingreso
  determinista; `tol_ge` gobierna la bisección de equilibrio general y
  `ge.converged` reporta $|K^s - K^d| \le$ `tol_ge`.
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Políticas óptimas de consumo $c_j(a)$ por estado de ingreso, con quiebres en la propensión marginal a consumir cerca de $\underline{a}$; (2) Deriva del ahorro $s(a, z) = r a + w z - c(a, z)$, con desacumulación para ingresos bajos y acumulación para ingresos altos; (3) Distribución estacionaria de riqueza obtenida de la KFE adjunta, con el pico característico de masa precautoria en el límite de crédito; (4) Vaciado del mercado de activos de Aiyagari, $K^s(r)$ frente a $K^d(r)$ en torno a $r^*$.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `rho_custom = 0.05`, `gamma_custom = 2.0`, `Na_custom = 50`, `a_max_custom = 30.0`, con una nueva resolución vía `solve_hjb_achdou`; aserciones `sol_custom.converged`, `sol_custom.n_iter <= 25`, `sol_custom.mass_residual <= 1e-12` y `abs(mass_custom - 1.0) <= 1e-12`, donde `mass_custom = np.sum(sol_custom.g_dist * da_custom)`.
- **Literatura y Referencias Cruzadas**: Achdou, Han, Lasry, Lions y Moll (2022), Aiyagari (1994), Huggett (1993). Guías de usuario: [`docs/es/vfi_hjb_continuous.md`](vfi_hjb_continuous.md) (el solucionador en tiempo continuo) y [`docs/es/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md) (la vía del histograma en tiempo discreto).

### `57_multiconstraint_occbin_and_dml_es`
- **Fuente**: `notebooks/57_multiconstraint_occbin_and_dml_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo responden las economías cuando múltiples restricciones vinculantes ocasionales (como el límite inferior cero en tasas de interés y los límites de endeudamiento colateral) se vuelven activas de forma simultánea, y cómo pueden los macroeconomistas obtener estimaciones causales no sesgadas de políticas macroeconómicas utilizando Aprendizaje Automático Doble / Desesgado (DML-PLR) en alta dimensión?
- **Matemáticas y Algoritmos Rectores**:
  - **Perturbación Lineal a Tramos con OccBin Multirrestricción**: Para $M$ restricciones, $2^M$ regímenes discretos:
    $$A_{r_t} x_t = B_{r_t} x_{t-1} + C_{r_t} \mathbb{E}_t[x_{t+1}] + D_{r_t} + E_{r_t} \varepsilon_t$$
    Las transiciones de régimen iteran hacia atrás desde el horizonte $T$ para determinar la secuencia de regímenes $\{r_1, r_2, \dots, r_T\}$ que satisface las condiciones de holgura complementaria para la ZLB ($i_t \ge 0$) y el crédito ($b_t \le \bar{b}$).
  - **Aprendizaje Automático Doble / Desesgado (DML-PLR)**: Modelo de regresión parcialmente lineal:
    $$Y = D \theta_0 + g_0(X) + U, \quad \mathbb{E}[U | D, X] = 0$$
    $$D = m_0(X) + V, \quad \mathbb{E}[V | X] = 0$$
    El score ortogonal de Neyman elimina el sesgo de regularización de los estimadores de aprendizaje automático $\hat{\ell}(X)$ y $\hat{m}(X)$:
    $$\psi(W; \theta, \eta) = (Y - \ell(X)) - \theta (D - m(X))$$
    El ajuste cruzado de $K$ particiones elimina el sesgo de sobreajuste, produciendo estimaciones $\sqrt{N}$-consistentes y asintóticamente normales $\hat{\theta} \sim \mathcal{N}(\theta_0, \sigma^2 / N)$.
- **Intuición Económica**: Ante una contracción profunda, los hogares reducen su endeudamiento contra su límite crediticio mientras el banco central reduce la tasa de interés a cero. Cuando ambas restricciones operan simultáneamente (Régimen 3), la economía sufre una amplificación no lineal severa: la política monetaria no puede acomodar la caída mientras los hogares no pueden endeudarse para suavizar consumo. En la etapa empírica el cuaderno contrasta tres estimadores sobre un panel sintético con $N = 500$, $p = 30$ controles y un efecto verdadero $\theta_0 = 1{,}75$: MCO sobre todos los controles ($\hat\theta = 1{,}686$, IC al 95 % $[1{,}570, 1{,}801]$), un Lasso ingenuo que penaliza la variable de política $D$ junto con $X$ ($\hat\theta = 1{,}582$ — atenuación por encogimiento, sin un intervalo de confianza que merezca citarse) y DML ($\hat\theta = 1{,}699$, IC al 95 % $[1{,}581, 1{,}817]$). El estimador que fracasa es el Lasso ingenuo; lo que aporta DML es el derecho a emplear un ajuste penalizado de los parámetros de nuisance *y* aun así reportar un intervalo válido, gracias al ajuste cruzado y al score ortogonal de Neyman. Conviene precisar el alcance de esa garantía: los aprendices de nuisance incluidos (`lasso`, `ridge` — no hay red elástica) son **lineales en las columnas de $X$ que usted suministra**, de modo que DML elimina la confusión que sea lineal en esas columnas. La confusión no lineal exige añadir una expansión de base a $X$, o pasar un aprendiz propio mediante `learner=`.
- **Código Desarrollado**:
  ```python
  import numpy as np
  from puremacro.dsge import (
      build_dynare, OccBinConstraint, solve_multiconstraint_occbin, OccBinResult,
  )
  from puremacro.causal import dml_plr, DoubleMLPLR, LassoCoordinateDescent, RidgeGCV

  # 1. OccBin multirrestricción: un modelo restringido por restricción,
  #    emparejados por clave
  res_occ = solve_multiconstraint_occbin(
      m_unconstrained=m_ref,
      m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
      shock_seq=shocks_mat,                      # (horizon, n_shocks)
      constraints={"zlb": c_zlb, "borrowing": c_borr},
      horizon=40,
  )
  regimes = np.asarray(res_occ.regimes)          # máscara: 1 = ZLB, 2 = tope, 3 = ambas
  res_occ.converged, regimes[:8]                 # True, [3 1 1 0 0 0 0 0]

  # 2. Regresión parcialmente lineal con DML (N = 500, p = 30, theta_0 = 1.75)
  res_dml_lasso = dml_plr(Y_outcome, D_treat, X_mat, n_folds=5,
                          learner="lasso", random_state=42)
  res_dml_ridge = dml_plr(Y_outcome, D_treat, X_mat, n_folds=5,
                          learner="ridge", random_state=42)
  res_dml_lasso.theta, res_dml_lasso.ci_lower, res_dml_lasso.ci_upper
  #                                              # 1.6989, 1.5813, 1.8165
  ```
  `DoubleMLPLR(n_folds=5, learner="lasso").fit(Y, D, X)` es la forma de clase del
  mismo estimador: los datos se pasan a `.fit()`, no al constructor, y el
  argumento es `learner=`, no `estimator=`. No existe ninguna clase
  `OccBinMultiConstraint` ni módulo `puremacro.dml`; el solucionador
  multirrestricción es `puremacro.dsge.solve_multiconstraint_occbin` y DML reside
  en `puremacro.causal`.
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Trayectoria de la tasa de interés, OccBin frente a la simulación lineal sin restricciones; (2) Secuencia de regímenes estructurales activos a lo largo del tiempo (la máscara de bits); (3) Comparación de estimadores causales con bandas de confianza: efecto verdadero, MCO, Lasso ingenuo, DML-Lasso y DML-Ridge; (4) Residuos ortogonalizados de Neyman con la pendiente de política ajustada.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `shock_g_custom = -0.06`, `shock_b_custom = 0.05`, `b_bar_custom = 0.020`, `n_folds_custom = 5`, `learner_custom = "lasso"` con aserciones `res_custom_occ.converged` y `res_custom_dml.ci_lower <= theta_true <= res_custom_dml.ci_upper`.
- **Literatura y Referencias Cruzadas**: Guerrieri e Iacoviello (2015), Chernozhukov et al. (2018), Belloni, Chernozhukov y Hansen (2014). Guías de usuario: [`docs/es/dsge_higher_order.md`](dsge_higher_order.md), [`docs/es/forecast.md`](forecast.md).

### `58_latin_america_realtime_macro_es`
- **Fuente**: `notebooks/58_latin_america_realtime_macro_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo evolucionan las publicaciones preliminares del PIB y variables macroeconómicas de los institutos de estadística y bancos centrales de América Latina a través de sucesivas versiones históricas (vintages), y están estas revisiones guiadas por actualizaciones racionales con nueva información ("noticias") o por errores y ruido en la medición inicial ("ruido")?
- **Matemáticas y Algoritmos Rectores**:
  - **Triángulos de Revisiones $(T \times V)$**: Matriz de datos en tiempo real donde las filas denotan trimestres de referencia $t = 1, \dots, T$ y las columnas denotan trimestres de publicación $v = 1, \dots, V$ ($v \ge t$):
    $$R_{t, k} = y_{t, t+k+1} - y_{t, t+k}, \quad R_{t, \text{final}} = y_{t, \text{final}} - y_{t, \text{primero}}$$
  - **Econometría de Noticias frente a Ruido de Mankiw-Shapiro (1986)**:
    $$\text{Modelo 1 (Noticias / Pronóstico Racional)}: \quad y_{t, \text{final}} - y_{t, \text{primero}} = \alpha + \beta y_{t, \text{primero}} + \varepsilon_t$$
    Bajo pronóstico estadístico racional, las publicaciones iniciales usan toda la información disponible; las revisiones futuras son impredecibles: $H_0: \alpha = 0, \beta = 0$.
    $$\text{Modelo 2 (Ruido / Error de Medición)}: \quad y_{t, \text{primero}} = \alpha + \beta y_{t, \text{final}} + u_t$$
    Bajo ruido clásico de medición, la publicación preliminar es una aproximación ruidosa del PIB verdadero: $H_0: \alpha = 0, \beta = 1$.
  - **Inferencia HAC Newey-West**: Estimación robusta de covarianza para heterocedasticidad y autocorrelación en los residuos de revisión.
  - **Cartuchos Criptográficos `.pmz` Fuera de Línea**: Archivos `.pmz` portátiles y autocontenidos con verificación de integridad SHA-256, validación de esquema y desempaquetado instantáneo sin conexión a red.
  - **Aquí las añadas son fechas de instantánea.** Banxico, INEGI, BCB y BCCh publican únicamente la edición vigente en el momento de la consulta; sobrescriben en el sitio. El panel del cuaderno es, por tanto, sintético, y frente a los conectores en vivo una fecha de añada es el día en que se capturó la serie: el historial de revisiones se acumula solo a lo largo de capturas repetidas en días distintos.
- **Intuición Económica**: Las cifras preliminares publicadas a los 30–45 días del cierre del trimestre se basan en muestras incompletas y modelos de nowcasting. Conforme arriba información dura (declaraciones tributarias, balances contables), las agencias revisan los datos. En economías emergentes de América Latina (México, Brasil, Chile), discernir si las revisiones son noticias o ruido determina si los formuladores de política deben reaccionar de inmediato a las señales preliminares o descontarlas como volatilidad transitoria. Los resultados empíricos confirman que las revisiones están dominadas por noticias, ratificando que los bancos centrales entregan estimaciones racionales en tiempo real.
- **Código Desarrollado**:
  ```python
  from puremacro.fetch.realtime import (
      VintagePanel,
      pack_realtime_cartridge,
      load_realtime_cartridge,
  )

  # `df_raw` es el marco largo ordenado: country, variable, date, vintage,
  # value, provider, series_id, units
  panel_raw = VintagePanel(df_raw)

  # Empaquetado en un cartucho .pmz offline y recarga con verificación SHA-256
  pack_realtime_cartridge(
      panel_raw, cartridge_file,
      source="Banxico, INEGI, BCB, BCCh Regional Real-Time Ecosystem",
      vintage="2026-04-01",
  )
  panel = load_realtime_cartridge(cartridge_file, verify=True)

  # Los triángulos de revisiones y el par de Mankiw-Shapiro son métodos del panel
  panel.coverage()                            # qué se recuperó, por (país, variable)
  panel.as_of("2025-06-01")                   # el conjunto de información en esa fecha
  panel.triangle("MEX", "gdp_real")           # períodos de referencia x añadas
  panel.revisions("MEX", "gdp_real")          # preliminar / final / revisión
  ms = panel.news_or_noise("MEX", "gdp_real") # MankiwShapiroResult
  ms.verdict, ms.beta_on_preliminary, ms.p_beta_on_final
  panel.news_or_noise_panel()                 # todas las series, una tabla ordenada
  ```
  No existe un módulo `puremacro.realtime` ni funciones `build_revision_triangle`,
  `mankiw_shapiro_test` o `export_realtime_cartridge`: los auxiliares de cartucho
  son `pack_realtime_cartridge` / `load_realtime_cartridge` en
  `puremacro.fetch.realtime`, y la maquinaria de revisiones vive en `VintagePanel`
  (con el contraste subyacente en `puremacro.vintages.mankiw_shapiro`).
- **Lectura de Resultados y Visualizaciones**: Tablero hero de 4 paneles: (1) Tasas de política monetaria de los bancos centrales latinoamericanos; (2) Mapa de calor triangular del triángulo de revisiones $\mathbf{T}[t, v]$ para el PIB real de México; (3) Estimaciones preliminares frente a definitivas del PIB a lo largo de los períodos de referencia; (4) Diagrama de dispersión de Mankiw-Shapiro de las revisiones sobre la publicación preliminar, anotado con el veredicto.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `country_custom = "MEX"`, `var_custom = "gdp_real"`, `as_of_custom = "2025-06-01"`, `signif_custom = 0.05` con aserciones `country_custom in panel.countries`, `var_custom in panel.variables`, `not custom_asof.empty`, `len(custom_rev) > 0` y `hasattr(custom_ms, "verdict")`.
- **Literatura y Referencias Cruzadas**: Mankiw y Shapiro (1986), Croushore y Stark (2001), Faust, Rogers y Wright (2005). Guías de usuario: [`docs/es/real_time_latam.md`](real_time_latam.md) (los cuatro conectores) y [`docs/es/real_time_data.md`](real_time_data.md).

---

## Demostraciones de Programación Dinámica Continua, Deep Macro y GE Espacial (puremacro 3.3)

Las demostraciones `51` a `55` presentan los motores de vanguardia de `puremacro` en programación dinámica sobre espacios de estados continuos, transiciones no lineales de distribuciones de riqueza, gradientes analíticos de sensibilidad adjunta de precisión de máquina, redes neuronales para aproximación de políticas en alta dimensión y equilibrio general espacial cuantitativo de comercio internacional.

### `51_continuous_projection_collocation_and_fem_es`
- **Fuente**: `notebooks/51_continuous_projection_collocation_and_fem_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo resuelven los macroeconomistas cuantitativos modelos de equilibrio general dinámico sin sufrir la maldición de la dimensionalidad ni las distorsiones de discretización de la iteración sobre funciones de valor en grillas, y cuándo conviene implementar polinomios ortogonales globales de Chebyshev frente a proyecciones locales de Galerkin con elementos finitos?
- **Matemáticas y Algoritmos Rectores**:
  - **Colocación Ortogonal de Chebyshev**: Mapeo de $k \in [a, b]$ al dominio canónico $x \in [-1, 1]$ mediante $x = \frac{2k - (a+b)}{b - a}$. Los polinomios base satisfacen la recurrencia de 3 términos $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$ con nodos de extremos de Chebyshev-Gauss-Lobatto $x_i = -\cos(i\pi/N)$. Se anula el operador de residuo continuo de Euler:
    $$\mathcal{R}_{\text{Euler}}(\theta; k_i) \equiv 1 - \beta \frac{u'(c(g(k_i), g(g(k_i))))}{u'(c(k_i, g(k_i)))} f'(g(k_i)) = 0$$
  - **Proyección de Galerkin con Elementos Finitos (FEM)**: Partición de $[a, b]$ en $E$ elementos con funciones base lineales a tramos en sombrero $\phi_j(k)$ con soporte compacto $[k_{j-1}, k_{j+1}]$. Los residuos se proyectan contra funciones de prueba usando cuadratura de Gauss-Legendre de $Q$ puntos:
    $$\int_a^b \mathcal{R}(k) \phi_i(k) \, dk = 0, \quad \forall i = 1, \dots, N_{\text{nodos}}$$
  - **Operador NCP de Fischer-Burmeister para Restricciones de Endeudamiento**: Complementariedad suavizada y perturbada ($\epsilon = 10^{-12}$) para resolver $k' \ge \bar{k}$ y $\mathcal{R}(k) \ge 0$:
    $$\Psi_{\text{FB}}^\epsilon(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon} = 0, \quad a = k' - \bar{k}, \quad b = \mathcal{R}(k)$$
- **Intuición Económica**: Los polinomios globales de Chebyshev logran convergencia espectral exponencial ($\mathcal{O}(c^{-N})$) para reglas de decisión analíticas reales (crecimiento de Brock-Mirman), pero sufren oscilaciones catastróficas de Gibbs y violaciones del límite inferior ante esquinas angulosas. FEM renuncia al soporte global a favor de funciones locales en sombrero (convergencia algebraica $\mathcal{O}(h^2)$), suprimiendo por completo el fenómeno de Gibbs cuando se fija un nodo del mallado en el umbral de endeudamiento $k^*$.
- **Código Desarrollado**:
  ```text
  import numpy as np
  from puremacro.vfi import CollocationProblem, solve_collocation, FEMProblem, solve_fem

  # Modelo analítico de referencia de Brock-Mirman (1972): u(c)=ln(c), f(k)=k^alpha, delta=1.0
  alpha, beta = 0.36, 0.96
  k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
  domain = (0.5 * k_ss, 1.5 * k_ss)

  def euler_res(k, kp, kpp, s, sp):
      c = k**alpha - kp
      c_next = kp**alpha - kpp
      return 1.0 / c - beta * (1.0 / c_next) * (alpha * kp**(alpha - 1.0))

  prob_col = CollocationProblem(domain=domain, deg=7, beta=beta, euler_residual_fn=euler_res)
  sol_col = solve_collocation(prob_col)

  prob_fem = FEMProblem(
      domain=domain, n_elements=25, deg=1, beta=beta,
      return_fn=lambda k, kp, s: np.log(np.maximum(k**alpha - kp, 1e-8)),
      transition_fn=lambda k, kp, s: kp,
      euler_residual_fn=euler_res,
  )
  sol_fem = solve_fem(prob_fem)
  ```
- **Lectura de Resultados y Visualizaciones**: Cuatro figuras principales: (1) Convergencia semilogarítmica del error de política demostrando decaimiento espectral exponencial ($L^\infty < 10^{-6}$ con $N=8$) frente a decaimiento algebraico de FEM; (2) Detalle en la vecindad del quiebre que evidencia el timbre de Chebyshev frente a la perfecta conformidad de frontera de FEM; (3) Curvas estocásticas de política $g(k, z)$ preservando monotonicidad; (4) Comparativa de tiempo de cómputo en NumPy, Numba y MLX.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `order_custom = 8`, `elements_custom = 50`, `borrow_ratio_custom = 0.50` con aserciones automáticas `sol_custom_c.converged`, `sol_custom_f.converged`, `err_custom_c < 1e-3`, `viol_custom_f == 0.0`.
- **Literatura y Referencias Cruzadas**: Brock y Mirman (1972), Judd (1992, 1998), McGrattan (1996), Fischer (1992). Guía de usuario: [`docs/es/vfi_continuous_projection.md`](vfi_continuous_projection.md).

### `52_continuous_transition_mit_shocks_es`
- **Fuente**: `notebooks/52_continuous_transition_mit_shocks_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo evolucionan de forma no lineal las distribuciones continuas de riqueza de los hogares $\mu_t(k, z)$ y los precios factoriales de equilibrio $\{r_t, w_t\}$ tras un choque macroeconómico agregado imprevisto en una economía de mercados incompletos, y cómo influye el ahorro precautorio endógeno en la persistencia macroeconómica y la desigualdad de riqueza?
- **Matemáticas y Algoritmos Rectores**:
  - **EGM Continuo hacia Atrás**: Inversión de la utilidad marginal a lo largo de activos continuos $a' \ge 0$ dada la trayectoria de precios $\{r_t, w_t\}$:
    $$c_t^{\text{endo}}(a', z_i) = \left( \beta (1 + r_{t+1}) \sum_j P_z(z_i, z_j) [c_{t+1}(a', z_j)]^{-\gamma} \right)^{-1/\gamma}$$
    $$a_t^{\text{endo}}(a', z_i) = \frac{c_t^{\text{endo}}(a', z_i) + a' - w_t z_i}{1 + r_t}$$
  - **Operador de Loterías de Young (2010) Temporal hacia Adelante**: Avance de densidad con conservación exacta de masa a nivel de máquina ($\sum \mu_t = 1.0 \pm 10^{-15}$):
    $$\mu_{t+1}(k_m, z_l) = \sum_j P_z(z_j, z_l) \sum_i \mu_t(k_i, z_j) [w_{\text{lo}} \mathbf{1}_{\{m=j_{\text{lo}}\}} + w_{\text{hi}} \mathbf{1}_{\{m=j_{\text{hi}}\}}]$$
  - **Vaciado de Mercado en el Espacio de Secuencias**:
    $$H_t(\mathbf{r}) \equiv \sum_{i, j} k_i \mu_t(k_i, z_j) - L \left( \frac{r_t + \delta}{\alpha Z_t} \right)^{\frac{1}{\alpha - 1}} = 0, \quad \forall t \in [0, T-1]$$
  - **Solucionador Cuasi-Newton de Broyden**: Trata la trayectoria completa de tasas $\mathbf{r} \in \mathbb{R}^T$ como un sistema unificado con actualizaciones de rango 1 de Sherman-Morrison para la matriz jacobiana inversa aproximada.
- **Intuición Económica**: La oferta agregada de capital no puede ajustarse de manera instantánea en $t=0$ debido a que la distribución de riqueza $\mu_0(k, z)$ es físicamente predeterminada. Para vaciar los mercados factoriales tras un choque positivo imprevisto del $+5\%$ en la PTF, la tasa de interés real salta de inmediato ($+49.9$ pb). El incremento salarial beneficia desproporcionadamente a los trabajadores con restricciones de liquidez y baja riqueza, contrayendo transitoriamente el índice de Gini ($0.5269 \to 0.5212$). A medida que los hogares acumulan ahorro precautorio, el stock agregado de capital alcanza su cúspide en $t=7$ ($K^* = 8.792 \to K_{\text{pico}} = 9.039$), generando una persistencia endógena que supera con creces la autocorrelación exógena del choque ($0.80^7 \approx 0.21$).
- **Código Desarrollado**:
  ```text
  import numpy as np
  from puremacro.vfi import solve_continuous_transition, TransitionShock

  # Choque imprevisto de PTF del +5% con rho=0.80 en un horizonte T=40 trimestres
  shock_path = 0.05 * (0.80 ** np.arange(40))
  shock = TransitionShock(variable="TFP", path=shock_path)
  res = solve_continuous_transition(model, shock=shock, T=40, damping=0.40)
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero central de 4 paneles (Senda de la Tasa Real, Oferta vs Demanda de Capital, Evolución del Gini de Riqueza, Desplazamiento de Curvas de Lorenz) y superficie en perspectiva 3D $\mu_t(k)$ que ilustra la dispersión dinámica de los activos precautorios.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `user_shock_size = 0.05`, `user_persistence = 0.80`, `user_damping = 0.40` con aserciones `user_res.converged`, `user_res.max_residual < 1e-4`, `user_res.mass_conservation_error < 1e-12`.
- **Literatura y Referencias Cruzadas**: Aiyagari (1994), Young (2010), Boppart, Krusell y Mitman (2018), Auclert et al. (2021). Guías de usuario: [`docs/es/vfi_continuous_transition.md`](vfi_continuous_transition.md), [`docs/es/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md).

### `53_exact_analytic_ift_gradients_es`
- **Fuente**: `notebooks/53_exact_analytic_ift_gradients_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo pueden los macroeconomistas cuantitativos efectuar estimaciones estructurales basadas en gradientes (SMM/GMM) de modelos dinámicos continuos sin incurrir en el ruido numérico y el costo combinatorio de las aproximaciones por diferencias finitas, y cómo entrega el Teorema de la Función Implícita sensibilidades exactas de política en una sola resolución lineal?
- **Matemáticas y Algoritmos Rectores**:
  - **Teorema de la Función Implícita en Sistemas de Proyección Continua**: Diferenciando la identidad de residuo nulo $\mathcal{R}(c^*(\theta); \theta) \equiv \mathbf{0} \in \mathbb{R}^N$ con respecto a los parámetros estructurales $\theta \in \mathbb{R}^p$:
    $$\nabla_\theta c^*(\theta) = - \left[ \nabla_c \mathcal{R}(c^*; \theta) \right]^{-1} \nabla_\theta \mathcal{R}(c^*; \theta)$$
  - **Sensibilidades de Políticas Continuas y Agregados Macroeconómicos**:
    $$\nabla_\theta g(k; \theta) = \Phi(k) \, \nabla_\theta c^*(\theta), \qquad \frac{d K^*}{d \theta} = \frac{\nabla_\theta g(k^*; \theta)}{1 - g'(k^*; \theta)}$$
  - **Función Objetivo y Gradiente Exacto de GMM Estructural**:
    $$Q(\theta) = (m(\theta) - \hat{m})^\top W (m(\theta) - \hat{m}), \qquad \nabla_\theta Q(\theta) = 2 \, G(\theta)^\top W (m(\theta) - \hat{m})$$
    donde $G(\theta) \equiv \nabla_\theta m(\theta) = \nabla_c m \, \nabla_\theta c^*(\theta)$ se calcula analíticamente sin diferencias finitas.
- **Intuición Económica**: Las diferencias finitas enfrentan un dilema de paso numérico: pasos grandes generan sesgos de truncamiento ($\mathcal{O}(h^2)$), mientras que pasos pequeños desatan errores de cancelación por redondeo y oscilaciones de parada que deforman la superficie de optimización. El TFI reutiliza la factorización LU del jacobiano residual convergente $J_c = \nabla_c \mathcal{R}$, evaluando las sensibilidades exactas por sustitución hacia atrás con una **aceleración empírica de 68.8x** y cero ruido numérico, logrando superficies parabólicas suaves que permiten al optimizador BFGS converger cuadráticamente en 17 iteraciones.
- **Código Desarrollado**:
  ```text
  from puremacro.vfi import (
      CollocationProblem, solve_collocation,
      compute_ift_gradients, policy_parameter_jacobian,
  )

  sol = solve_collocation(problem)
  ift_res = compute_ift_gradients(sol, problem, params=["beta", "sigma", "alpha", "delta"])
  d_policy_fn = policy_parameter_jacobian(sol, problem, params=["beta"])
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero de 4 paneles (Sensibilidades de la Política a lo largo de $k$, Gráfico de Barras de Rendimiento demostrando 68.8x de aceleración, Superficie de Contorno de Pérdida 2D $\log_{10} Q(\beta, \sigma)$ con la trayectoria BFGS, y Gráfico de Discrepancia Modal de la Base).
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `user_k1 = 0.10`, `user_k2 = 0.28`, `user_init_beta = 0.93`, `user_init_sigma = 1.30` con puertas de validación `user_opt.success`, `np.isclose(user_beta_hat, 0.96, atol=1e-4)`, `np.isclose(user_sigma_hat, 1.50, atol=1e-4)`.
- **Literatura y Referencias Cruzadas**: Judd (1998), Su y Judd (2012), Rust (1994). Guía de usuario: [`docs/es/vfi_analytic_gradients.md`](vfi_analytic_gradients.md).

### `54_deep_macro_pinns_high_dim_es`
- **Fuente**: `notebooks/54_deep_macro_pinns_high_dim_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo pueden los macroeconomistas cuantitativos quebrar la maldición exponencial de la dimensionalidad para resolver modelos de equilibrio general dinámico con más de 10 variables de estado continuo en Python puro sin librerías externas de deep learning, y cómo imponen las redes neuronales informadas por la física la viabilidad física exacta a lo largo de trayectorias ergódicas?
- **Matemáticas y Algoritmos Rectores**:
  - **Acumulación de Capital Multipaís en Alta Dimensión**: Vector de estado $\mathbf{s}_t = (k_{1, t}, \dots, k_{N, t}) \in \mathbb{R}_{++}^N$; recursos disponibles (*cash-on-hand*) $W_i(\mathbf{s}_t) = A_i k_{i, t}^\alpha + (1 - \delta) k_{i, t}$; ley del capital $k_{i, t+1} = W_i(\mathbf{s}_t) - c_{i, t}$.
  - **Operador de Residuo Continuo Adimensional de Euler**:
    $$\mathcal{R}_i(\mathbf{s}_t, \mathbf{c}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}) \equiv 1 - \beta \left(\frac{c_{i, t+1}}{c_{i, t}}\right)^{-\gamma} (\alpha A_i k_{i, t+1}^{\alpha-1} + 1 - \delta) = 0, \quad \forall i = 1, \dots, N$$
  - **Parametrización Neuronal de Políticas y Viabilidad Física**: Perceptrón multicapa entrenable $\mathbf{z}(\mathbf{s}; \Theta) \in \mathbb{R}^N$ con activación sigmoide acotada:
    $$\phi_i(\mathbf{s}; \Theta) = \epsilon_{\text{bound}} + (1 - 2\epsilon_{\text{bound}}) \sigma(z_i(\ln(\mathbf{s}/\mathbf{s}_{\text{ss}}))), \qquad c_i(\mathbf{s}; \Theta) = \phi_i(\mathbf{s}; \Theta) W_i(\mathbf{s})$$
    donde $\epsilon_{\text{bound}} = 10^{-4}$ garantiza estrictamente $c_i > 0, k'_i > 0, c_i < W_i$ en todo el dominio.
  - **Muestreo por Trayectorias Ergódicas (Maliar et al. 2021)**: La simulación de trayectorias de equilibrio de longitud $T = 1200$ concentra el cómputo sobre la variedad compacta del atractor estocástico, reduciendo $10^{10}$ puntos de grilla a $\mathcal{O}(T \cdot D)$. La función de pérdida $\mathcal{L}(\Theta) = \frac{1}{B \cdot N} \sum_{b, i} (\phi_{i, b} - \phi_{i, b}^{\text{target}})^2$ se minimiza mediante Adam con gradientes analíticos vectorizados en NumPy puro.
- **Intuición Económica**: Discretizar un vector continuo de 10 dimensiones con apenas 10 puntos por dimensión requeriría $10^{10}$ nodos. Los sistemas económicos, no obstante, jamás exploran hipercubos de manera uniforme; los rendimientos marginales decrecientes y la depreciación del capital ejercen fuerzas gravitacionales de reversión a la media que restringen la dinámica a un atractor estocástico de dimensión baja. Al entrenar perceptrones con activaciones suaves (SiLU) exclusivamente sobre las trayectorias ergódicas visitadas, las PINNs resuelven modelos de 10 países en $<0.5$ segundos con cero dependencias externas de deep learning (PyTorch/TensorFlow).
- **Código Desarrollado**:
  ```text
  from puremacro.vfi import DeepMacroModel, solve_deep_macro

  model = DeepMacroModel(d_in=10, d_out=10, hidden_layers=[64, 64], activation="silu", seed=42)
  res = solve_deep_macro(
      model, euler_loss_fn=euler_loss, state_sampler=sampler,
      n_epochs=120, lr=1e-3, optimizer="adam", batch_size=256,
  )
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero de 4 paneles: (1) Convergencia de la pérdida de entrenamiento ($9.58 \times 10^{-10}$); (2) Gráficos de caja de residuos de Euler fuera de muestra para los 10 países ($\text{MSE} = 3.35 \times 10^{-6} < 10^{-3}$); (3) Senda de profundización del capital en los 10 países partiendo de una severa depresión; (4) Sección unidimensional de la función de política de consumo continuo frente al capital disponible.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `hidden_dims_custom = (32, 32)`, `lr_custom = 2e-3`, `n_epochs_custom = 80`, `perturb_custom = 0.60` con aserciones `sol_custom.converged`, `sol_custom.test_euler_mse < 1e-3`, `sim_custom["physically_viable"]`.
- **Literatura y Referencias Cruzadas**: Maliar, Maliar y Winant (2021), Raissi, Perdikaris y Karniadakis (2019), Judd (1998). Guía de usuario: [`docs/es/deep_macro.md`](deep_macro.md).

### `55_quantitative_spatial_and_trade_ge_es`
- **Fuente**: `notebooks/55_quantitative_spatial_and_trade_ge_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo se propagan los choques arancelarios internacionales, las disrupciones de cadenas de suministro y las inversiones en infraestructura regional de transporte a través de los encadenamientos insumo-producto y la movilidad espacial laboral para transformar los flujos de comercio, las poblaciones regionales y el bienestar agregado?
- **Matemáticas y Algoritmos Rectores**:
  - **Álgebra Exacta de Sombreros de Caliendo-Parro (2015)**: $N$ países, $J$ sectores con participación de valor agregado $\gamma_n^j$ y participaciones de costo de insumos intermedios $\gamma_n^{j, k}$. Variaciones relativas $\hat{x} \equiv x'/x$:
    $$\ln \hat{c}_i^j = \gamma_i^j \ln \hat{w}_i + \sum_k \gamma_i^{j, k} \ln \hat{P}_i^k$$
    $$\hat{P}_n^j = \left[ \sum_i \pi_{ni}^j (\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j)^{-\theta_j} \right]^{-1/\theta_j}, \qquad \pi_{ni}'^j = \pi_{ni}^j \left(\frac{\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j}{\hat{P}_n^j}\right)^{-\theta_j}$$
    Vaciado de mercado para el ingreso factorial $w_n' L_n$ y variación de bienestar nacional $\hat{W}_n = \hat{I}_n / \prod_j (\hat{P}_n^j)^{\alpha_n^j}$.
  - **Equilibrio General Espacial Cuantitativo de Allen-Arkolakis (2014)**: Geografía continua con $N$ localidades, costos bilaterales tipo iceberg $\tau_{ij} \ge 1$, aglomeración marshalliana $\alpha > 0$ ($A_i = \bar{A}_i L_i^\alpha$) y congestión residencial $\beta < 0$ ($a_i = \bar{a}_i L_i^\beta$). La igualación de utilidad real $u_i = a_i \frac{w_i}{P_i} = \bar{u}$ determina:
    $$L_i = \bar{L} \frac{(\bar{a}_i w_i / P_i)^{-1/\beta}}{\sum_k (\bar{a}_k w_k / P_k)^{-1/\beta}}, \qquad P_i^{-\theta} = \sum_j \tau_{ji}^{-\theta} \left(\frac{w_j}{\bar{A}_j L_j^\alpha}\right)^{-\theta}$$
    Equilibrio espacial único garantizado cuando $\alpha + \beta \le \frac{\theta}{1+\theta}$ y $\alpha \le 1/\theta$.
- **Intuición Económica**: En el comercio internacional, los aranceles unilaterales generan mejoras en los términos de intercambio (+0.33% bienestar en EE.UU.) pero desvían comercio hacia terceros países. Las guerras comerciales bilaterales destruyen estas ventajas e imponen pérdidas netas de bienestar (-1.11% EE.UU., -0.69% China). Los insumos intermedios actúan como multiplicadores de la cadena de suministro: los aranceles se transmiten en cascada a través de las matrices insumo-producto, amplificando las mermas de bienestar en relación con los modelos ricardianos tradicionales. En geografía espacial, los corredores de transporte (-20% de fricción) amplían el acceso a mercados, atrayendo mano de obra móvil (+0.35% población regional) hasta ser contrarrestados por la congestión de amenidades, elevando el bienestar agregado (+0.0627%) bajo estricta conservación poblacional.
- **Código Desarrollado**:
  ```text
  from puremacro.trade import CaliendoParroModel
  from puremacro.spatial import AllenArkolakisModel

  # 1. Simulación de choque arancelario de Caliendo-Parro
  cp_res = cp_model.solve_counterfactual(tariff_hat=tariff_hat)

  # 2. Simulación de corredor de infraestructura de transporte de Allen-Arkolakis
  aa_res = aa_model.simulate_transport_shock(origin=0, destination=1, cost_reduction=0.20)
  ```
- **Lectura de Resultados y Visualizaciones**: Figuras de calidad editorial: (1) Gráfico de 3 paneles de comercio (Matriz inicial de participaciones, Matriz post-guerra arancelaria, Impacto en bienestar por país); (2) Red geográfica de 2 paneles con corredor modernizado y variaciones de salarios/población; (3) Comparativa del impacto en bienestar con y sin encadenamientos insumo-producto.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `tariff_rate_custom = 0.15`, `cost_reduction_custom = 0.20` con aserciones `res_cp_custom.converged`, `res_aa_custom.converged`, `res_cp_custom.market_clearing_residual < 1e-6`, `res_aa_custom.labor_conservation_residual < 1e-12`, `res_aa_custom.welfare_pct > 0.0`.
- **Literatura y Referencias Cruzadas**: Eaton y Kortum (2002), Caliendo y Parro (2015), Allen y Arkolakis (2014), Costinot y Rodríguez-Clare (2014). Guía de usuario: [`docs/es/spatial_and_trade_ge.md`](spatial_and_trade_ge.md).

---

## Cuadernos de Frontera en Política Macroeconómica Aplicada (puremacro 3.2)

Las demostraciones `47` a `50` conectan la macroeconometría teórica con la práctica de política en bancos centrales, ministerios de hacienda y mercados financieros.

### `47_applied_central_bank_policy_suite_es`
- **Fuente**: `notebooks/47_applied_central_bank_policy_suite_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Evaluación en tiempo real de la postura monetaria frente a reglas de Taylor (1993, 1999) y especificaciones con inercia.
  - Simulación de políticas contrafactuales: trayectorias macroeconómicas bajo sendas alternativas de tasas de interés.
  - Bandas de proyección en gráficos de abanico (fan charts) incorporando incertidumbre de choques y dispersión posterior de parámetros.
  - Extracción de tono y sentimiento en comunicados de bancos centrales (FOMC, BCE) mediante calificación léxica.

### `48_applied_realtime_nowcasting_and_news_es`
- **Fuente**: `notebooks/48_applied_realtime_nowcasting_and_news_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Nowcasting del PIB trimestral con modelos de factores dinámicos (DFM) siguiendo a Giannone, Reichlin y Small (2008).
  - Ingesta de series en tiempo real con extremos irregulares (ragged edges) combinando frecuencias mensuales y trimestrales.
  - **Descomposición de noticias**: atribución de las revisiones del pronóstico a sorpresas en publicaciones de datos (empleo, producción, encuestas).
  - Pruebas de ortogonalidad de Mankiw-Shapiro (1986) para evaluar si las revisiones representan noticias o ruido.

### `49_applied_macroprudential_gar_and_stress_es`
- **Fuente**: `notebooks/49_applied_macroprudential_gar_and_stress_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Regresión cuantil condicional para Growth-at-Risk (GaR) siguiendo a Adrian, Boyarchenko y Giannone (2019).
  - Ajuste de densidades paramétricas skew-$t$ de Azzalini para capturar el riesgo asimétrico a la baja en horizontes predictivos.
  - Redes de desbordamiento de riesgo sistémico mediante descomposición generalizada de varianza del error de pronóstico (GFEVD) (Diebold y Yílmaz 2014).
  - Escenarios de estrés para colchones de capital macroprudencial y probabilidades de recesión en gráficos de abanico.

### `50_applied_fiscal_multipliers_and_debt_sustainability_es`
- **Fuente**: `notebooks/50_applied_fiscal_multipliers_and_debt_sustainability_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Estimación trimétodo de multiplicadores fiscales: SVAR de Blanchard-Perotti, proyecciones locales narrativas de Romer-Romer y LP-IV de Mertens-Ravn con estadístico $F$ efectivo de primera etapa.
  - Análisis estocástico de sostenibilidad de la deuda soberana (DSA) con gráficos de abanico ante choques conjuntos de crecimiento, inflación y diferencial de tasas $(r - g)$.
  - Pruebas de estrés de la trayectoria de deuda ante escenarios de riesgo climático físico y de transición.

---

## Demostraciones de Frontera en DSGE Bayesiano y HANK (puremacro 3.0 y 3.1)

### `42_dsge_bayesian_estimation_and_diagnostics_es` (Insignia)
- **Fuente**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics_es.py` (Inglés: `.py`)
- **Modelo**: Economía estadounidense de Smets & Wouters (2007, AER) con 7 variables y 7 choques.
- **Capacidades Clave**: Búsqueda de moda con múltiples algoritmos (`lbfgs`, `csminwel`, `cmaes`), diagnósticos de curvatura `mode_check`, muestreo MCMC con diagnósticos $\hat{R}$ de Gelman-Rubin, suavizador de Kalman para descomposición de choques, pronósticos dinámicos y comparación de modelos por densidad marginal de Laplace y Geweke.

### `43_dsge_nuts_and_analytic_gradients_es`
- **Fuente**: `notebooks/43_dsge_nuts_and_analytic_gradients_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Recursión analítica exacta del gradiente de verosimilitud de Kalman ($\nabla_\theta \ln L$) con solucionadores de Sylvester generalizados, muestreador No-U-Turn (NUTS) con adaptación dual averaging y diagnósticos energéticos BFMI.

### `44_hank_sequence_space_bridge_es`
- **Fuente**: `notebooks/44_hank_sequence_space_bridge_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Puente de agentes heterogéneos en el espacio de secuencias desde archivos `.mod` de Dynare (`hetagent_block`), distribución estacionaria de riqueza $\mathcal{D}^*(a)$, jacobianos Fake-News ($J_{C,r}, J_{C,Y}$) y transiciones no lineales de Broyden.

### `45_dsge_discretion_dsge_var_and_news_shocks_es`
- **Fuente**: `notebooks/45_dsge_discretion_dsge_var_and_news_shocks_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Política óptima discrecional frente a compromiso (descomposición del sesgo de inflación y estabilización), optimización de previas DSGE-VAR($\lambda$) de Del Negro y Schorfheide (2004) y estado-espacio aumentado para choques anticipados de noticias.

### `46_dsge_particle_filtering_and_markov_switching_es`
- **Fuente**: `notebooks/46_dsge_particle_filtering_and_markov_switching_es.py` (Inglés: `.py`)
- **Capacidades Clave**: OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov de Foerster et al. (2016) con GIRFs analíticas cerradas y filtrado de partículas secuencial Monte Carlo con volatilidad estocástica.

---

## Catálogo Completo de Demostraciones

| Cuaderno | Tema y Metodología | Gemelo en Inglés |
|---|---|---|
| `00_whats_new_in_puremacro_3_0_es` | Hito 3.0: Gradientes analíticos de Kalman, NUTS HMC y puente HANK en espacio de secuencias | `00_whats_new_in_puremacro_3_0` |
| `00_whats_new_in_puremacro_2_0_es` | Hito 2.0: API unificada (`lags`, `horizon`, `ci`), objetos de resultado y exportadores | `00_whats_new_in_puremacro_2_0` |
| `01_wealth_inequality_es` | Mercados incompletos de Aiyagari y Huggett, heterogeneidad permanente en $\beta$, Lorenz y Gini | `01_wealth_inequality` |
| `02_aggregate_shocks_es` | Agregación aproximada de Krusell–Smith, dinámica de transición, agente representativo | `02_aggregate_shocks` |
| `03_life_cycle_and_demographics_es` | Consumo y ahorro en ciclo de vida finito, riqueza de cohortes por edad, mortalidad | `03_life_cycle_and_demographics` |
| `04_firm_dynamics_es` | Equilibrio de industria de Hopenhayn con entrada/salida endógena y selección | `04_firm_dynamics` |
| `05_portfolios_and_preferences_es` | Elección de cartera con dos activos, utilidad recursiva de Epstein–Zin, EGM frente a VFI | `05_portfolios_and_preferences` |
| `06_svar_identification_es` | Identificación en VAR estructural: ordenamiento de Cholesky frente a restricciones de signo | `06_svar_identification` |
| `07_local_projections_es` | Proyecciones locales de Jordà con inferencia HAC Newey-West y respuestas estado-dependientes | `07_local_projections` |
| `08_garch_volatility_es` | GARCH(1,1) por máxima verosimilitud y correlación condicional dinámica (DCC) de Engle | `08_garch_volatility` |
| `09_growth_at_risk_es` | Growth-at-Risk de Adrian-Boyarchenko-Giannone mediante regresión cuantil y densidad skew-$t$ | `09_growth_at_risk` |
| `10_staggered_did_es` | Diferencias en diferencias modernas: estimadores de Callaway-Sant'Anna y Sun-Abraham | `10_staggered_did` |
| `11_narrative_uncertainty_es` | Construcción de índices textuales de Incertidumbre de Política Económica (EPU) | `11_narrative_uncertainty` |
| `12_validation_gallery_es` | Cuadro de mando de validación comparando puremacro contra referencias independientes | `12_validation_gallery` |
| `13_build_your_own_index_es` | Construcción multirreceta de índices: Texto $\to$ EPU, Panel $\to$ JLN, Financiero $\to$ FCI | `13_build_your_own_index` |
| `14_tax_multiplier_three_ways_es` | Multiplicador fiscal: SVAR Blanchard-Perotti, LP narrativa Romer-Romer, LP-IV Mertens-Ravn | `14_tax_multiplier_three_ways` |
| `15_lp_did_es` | Proyecciones locales con diferencias en diferencias (LP-DiD) bajo adopción escalonada | `15_lp_did` |
| `16_regime_girf_es` | Respuestas a impulsos generalizadas (GIRF) para VAR de umbral y cambio de régimen de Markov | `16_regime_girf` |
| `17_identification_spec_curve_es` | Curvas de especificación de identificación para análisis de sensibilidad | `17_identification_spec_curve` |
| `18_beveridge_curve_es` | Curva de Beveridge, eficiencia del emparejamiento y desplazamientos vacantes-desempleo | `18_beveridge_curve` |
| `19_model_confidence_set_es` | Conjunto de confianza de modelos (MCS) de Hansen-Lunde-Nason para pronósticos | `19_model_confidence_set` |
| `20_unit_roots_with_power_es` | Pruebas de raíz unitaria de alta potencia DF-GLS de Elliott-Rothenberg-Stock y Ng-Perron | `20_unit_roots_with_power` |
| `21_dynare_vfi_dsl_es` | Especificación declarativa en DSL para VFI con aceleración de Howard | `21_dynare_vfi_dsl` |
| `22_continuous_time_hjb_es` | Esquema en diferencias finitas para Hamilton-Jacobi-Bellman en tiempo continuo | `22_continuous_time_hjb` |
| `23_aiyagari_endogenous_labor_es` | Equilibrio general de mercados incompletos con oferta laboral endógena | `23_aiyagari_endogenous_labor` |
| `24_synthetic_control_es` | Control sintético de Abadie et al. con pruebas de placebo espaciales y temporales | `24_synthetic_control` |
| `25_frequency_connectedness_es` | Redes de desbordamiento en el dominio de la frecuencia de Baruník y Křehlík | `25_frequency_connectedness` |
| `26_cycles_and_bandpass_es` | Filtros de ciclo de Baxter-King, Christiano-Fitzgerald y Beveridge-Nelson | `26_cycles_and_bandpass` |
| `27_garch_midas_macro_risk_es` | Modelado de volatilidad de frecuencias mixtas GARCH-MIDAS de Engle-Ghysels-Sohn | `27_garch_midas_macro_risk` |
| `28_weak_iv_anderson_rubin_es` | Conjuntos de confianza Anderson-Rubin robustos a instrumentos débiles para LP-IV | `28_weak_iv_anderson_rubin` |
| `29_synthetic_did_es` | Diferencias en diferencias sintéticas (SDID) de Arkhangelsky et al. | `29_synthetic_did` |
| `30_narrative_bursts_and_transcripts_es` | Detección de ráfagas de Kleinberg en transcripciones de discursos de bancos centrales | `30_narrative_bursts_and_transcripts` |
| `31_sequence_space_hank_es` | Método jacobiano en el espacio de secuencias de Auclert et al. (2021) para HANK | `31_sequence_space_hank` |
| `32_climate_macro_dice_es` | Modelo DICE de Nordhaus e impuestos óptimos al carbono | `32_climate_macro_dice` |
| `33_gdp_nowcasting_news_es` | Nowcasting del PIB con modelo de factores dinámicos y descomposición de noticias | `33_gdp_nowcasting_news` |
| `34_penalized_macro_forecasting_es` | Elastic Net y Lasso adaptativo para pronósticos macroeconómicos en alta dimensión | `34_penalized_macro_forecasting` |
| `35_empirical_benchmark_replications_es` | Cuadro de mando de replicación en literaturas de referencia de SVAR, LP y DiD | `35_empirical_benchmark_replications` |
| `36_climate_sovereign_debt_risk_es` | Transmisión del riesgo climático físico y de transición a la sostenibilidad de deuda | `36_climate_sovereign_debt_risk` |
| `37_central_bank_narrative_sentiment_es` | Sentimiento en comunicación de bancos centrales y proyecciones locales de alta frecuencia | `37_central_bank_narrative_sentiment` |
| `38_real_time_vintages_and_revisions_es` | Cuentas nacionales en tiempo real en más de 45 países y pruebas noticia vs. ruido | `38_real_time_vintages_and_revisions` |
| `39_multilingual_narrative_harvesting_es` | Recolección narrativa multifuente (50+ conectores) y puntuación macro en 8 idiomas | `39_multilingual_narrative_harvesting` |
| `40_quarterly_national_accounts_es` | Tres enfoques para cuentas nacionales: cambio de año base, identidades y contribuciones | `40_quarterly_national_accounts` |
| `41_dynare_frontier_showcase_es` | Smets-Wouters (2007) con funciones nativas de suavizado y estimación, ZLB y Ramsey | `41_dynare_frontier_showcase` |
| `42_dsge_bayesian_estimation_and_diagnostics_es` | Estimación bayesiana insignia de Smets-Wouters, mode_check, MCMC, fan charts y MDD | `42_dsge_bayesian_estimation_and_diagnostics` |
| `43_dsge_nuts_and_analytic_gradients_es` | Estimación DSGE bayesiana con NUTS y recursión exacta del score de Kalman | `43_dsge_nuts_and_analytic_gradients` |
| `44_hank_sequence_space_bridge_es` | Puente HANK en espacio de secuencias desde archivos `.mod`, jacobianos Fake-News | `44_hank_sequence_space_bridge` |
| `45_dsge_discretion_dsge_var_and_news_shocks_es` | Política discrecional vs compromiso, optimización de previas DSGE-VAR, choques de noticias | `45_dsge_discretion_dsge_var_and_news_shocks` |
| `46_dsge_particle_filtering_and_markov_switching_es` | OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov y partículas | `46_dsge_particle_filtering_and_markov_switching` |
| `47_applied_central_bank_policy_suite_es` | Postura monetaria, reglas de Taylor contrafactuales, abanicos de proyección, sentimiento | `47_applied_central_bank_policy_suite` |
| `48_applied_realtime_nowcasting_and_news_es` | Nowcasting DFM con series irregulares en tiempo real, descomposición de noticias | `48_applied_realtime_nowcasting_and_news` |
| `49_applied_macroprudential_gar_and_stress_es` | Densidades de Growth-at-Risk (skew-$t$), redes de interconexión financiera y estrés | `49_applied_macroprudential_gar_and_stress` |
| `50_applied_fiscal_multipliers_and_debt_sustainability_es` | Multiplicadores fiscales trimétodo, DSA estocástico soberano bajo estrés climático | `50_applied_fiscal_multipliers_and_debt_sustainability` |
| `51_continuous_projection_collocation_and_fem_es` | PD de estados continuos: colocación ortogonal de Chebyshev vs FEM Galerkin con restricciones de endeudamiento | `51_continuous_projection_collocation_and_fem` |
| `52_continuous_transition_mit_shocks_es` | Dinámica de transición continua no lineal y choques MIT acoplando EGM y operador de Young | `52_continuous_transition_mit_shocks` |
| `53_exact_analytic_ift_gradients_es` | Jacobianos analíticos exactos por el TFI, sensibilidades adjuntas y estimación estructural GMM | `53_exact_analytic_ift_gradients` |
| `54_deep_macro_pinns_high_dim_es` | Modelos dinámicos en alta dimensión (10+ estados) mediante Redes Neuronales Informadas por la Física | `54_deep_macro_pinns_high_dim` |
| `55_quantitative_spatial_and_trade_ge_es` | Equilibrio general espacial y comercial cuantitativo: aranceles Caliendo-Parro y geografía Allen-Arkolakis | `55_quantitative_spatial_and_trade_ge` |
| `56_implicit_hjb_and_continuous_kfe_es` | Solucionador implícito HJB en tiempo continuo, densidad estacionaria KFE adjunta y GE de Aiyagari | `56_implicit_hjb_and_continuous_kfe` |
| `57_multiconstraint_occbin_and_dml_es` | OccBin multirrestricción ($M \ge 2$: ZLB + límites de crédito) y Aprendizaje Automático Doble (DML-PLR) | `57_multiconstraint_occbin_and_dml` |
| `58_latin_america_realtime_macro_es` | Vintages del PIB en tiempo real de América Latina, cartuchos `.pmz` y noticias vs ruido Mankiw-Shapiro | `58_latin_america_realtime_macro` |
| `59_latam_realtime_nowcast_and_news_es` | Nowcasting DFM del PIB para América Latina, descomposición de noticias Bańbura-Modugno y PIT Berkowitz | `59_latam_realtime_nowcast_and_news` |
| `60_interactive_dml_irm_and_iv_es` | Aprendizaje Automático Doble interactivo: IRM (ATE/ATT) con DC logístico, solapamiento y DML-IV | `60_interactive_dml_irm_and_iv` |
| `61_quantitative_policy_simulators_es` | Simuladores cuantitativos: EG comercial ICIO OCDE (77 países) y transmisión monetaria HANK en secuencias | `61_quantitative_policy_simulators` |

---

## Ejecución y Construcción de Artefactos

Los cuadernos de demostración se pueden ejecutar y compilar en artefactos `.ipynb` prerrenderizados mediante la herramienta de línea de comandos:

```bash
# Ejecutar y compilar todos los cuadernos en .ipynb
python tools/build_notebooks.py

# Compilar un solo cuaderno
python tools/build_notebooks.py 51_continuous_projection_collocation_and_fem_es

# Ejecutar sin modificar archivos (falla si algún cuaderno genera excepción)
python tools/build_notebooks.py --check
```
