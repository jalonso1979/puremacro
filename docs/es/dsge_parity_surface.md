> 🇬🇧 [English](../dsge_parity_surface.md) · 🇪🇸 Español

# Superficie de paridad DSGE, simulación avanzada y panel de verificación

Puremacro 2.9.0 incorpora el módulo **Tier 3: Paridad y superficie operativa**, completando la superficie funcional integral de Dynare y la verificación automatizada de paridad bajo el estricto **contrato de cuatro paquetes de Pyodide** (`numpy`, `scipy`, `pandas`, `matplotlib`).

Esta versión introduce cuatro capacidades macroeconómicas fundamentales:
1. **Superficie de filtrado y momentos de `stoch_simul`**: Integración de la densidad espectral mediante cuadratura de Gauss-Legendre para filtrado teórico HP y paso de banda, filtro HP uniselectivo (causal) recursivo de Kalman, matrices completas de autocorrelación cruzada, correlaciones contemporáneas y momentos empíricos de simulación Monte Carlo (`simul_replic`).
2. **Simulación estocástica no lineal por sendero extendido (Fair y Taylor 1983)**: Simulación dinámica global sin truncamientos de Taylor por perturbación local, impulsada por un motor de Newton apilado disperso SuperLU con invariancia exacta para modelos lineales.
3. **Pronóstico condicional y descomposición avanzada de perturbaciones**: Pronóstico condicional según Waggoner y Zha (1999) mediante inversión de perturbaciones estructurales, bloques de sintaxis `.mod` `shock_groups;` con balance contable exacto a precisión de máquina, gráficos de abanico (*fan charts*) para funciones de impulso-respuesta (IRF) bayesianas, análisis predictivo a priori y criterio `qz_criterium` ajustable para sistemas cointegrados o con raíces unitarias.
4. **Panel de paridad con Dynare e interfaz de línea de comandos (CLI)**: Lectura nativa de archivos `.mat` de Dynare (`oo_.dr`, `oo_.mean`, `oo_.var`, `oo_.autocorr`), cuadro de mando automatizado de discrepancias frente a salidas oficiales de referencia y la herramienta de evaluación en consola `puremacro-dynare parity`.

---

## 1. Visión general y arquitectura

El motor DSGE de puremacro unifica soluciones analíticas exactas, perturbaciones de orden superior con poda de estados (*pruning*) y transiciones no lineales en el espacio de secuencias. El módulo Tier 3 consolida la superficie operativa requerida por bancos centrales, organismos internacionales e investigadores académicos para migrar modelos directamente desde entornos clásicos de Dynare a Python puro.

| Módulo | Capacidad central | Algoritmo principal | Referencia canónica |
| :--- | :--- | :--- | :--- |
| `puremacro.dsge._moments` | Filtrado espectral y momentos de simulación | Cuadratura de Gauss-Legendre y filtro de Kalman progresivo | Hodrick y Prescott (1997); Stock y Watson (1999) |
| `puremacro.dsge.extended_path` | Simulación no lineal por sendero extendido | Motor Newton SuperLU apilado con expectativas nulas de shocks futuros | Fair y Taylor (1983); Adjemian y Juillard (2014) |
| `puremacro.dsge.conditional` | Pronóstico condicional | Inversión de perturbaciones estructurales ($U = R^{-1} d$) | Waggoner y Zha (1999) |
| `puremacro.dsge.shock_groups` | Descomposición agrupada de perturbaciones | Atribución histórica exacta con balance contable riguroso | Especificación Dynare 4.6+ / 5.x / 6.x |
| `puremacro.dsge.bayesian` | IRF bayesianas y predictivo a priori | Gráficos de abanico posteriores y simulación a priori | Geweke (1999); Herbst y Schorfheide (2015) |
| `puremacro.dsge.parity` | Panel de paridad y verificación | Comparación automatizada frente a valores oficiales de referencia | Pfeifer (2014); Suite de referencia de Dynare |
| `puremacro.dsge.cli` | Evaluador de paridad en consola | CLI `puremacro-dynare parity` | Interfaz CLI de puremacro |

---

## 2. Superficie de filtrado y momentos de `stoch_simul`

### 2.1 Filtrado teórico de densidad espectral

En modelos DSGE lineales o linealizados, la representación en el espacio de estados vincula las desviaciones de los estados $x_t$ y las variables observadas o de interés $v_t$:

$$x_{t+1} = G x_t + N u_t, \quad v_t = M_x x_t + M_u u_t, \quad u_t \sim \mathcal{N}(0, \Sigma_u)$$

La matriz continua de densidad espectral de $v_t$ a la frecuencia $\omega \in [-\pi, \pi]$ viene dada por:

$$S_v(\omega) = \frac{1}{2\pi} H(e^{-i\omega}) \Sigma_u H(e^{i\omega})^\top$$

donde la función de transferencia es $H(e^{i\omega}) = M_x (e^{i\omega} I_{n_x} - G)^{-1} N + M_u$.

Al aplicar un filtro lineal con ganancia cuadrática de respuesta en frecuencia $|\Phi(\omega)|^2$, la autocovarianza filtrada teórica en el rezago $k$ se calcula integrando sobre el espectro:

$$\Gamma_k^{filtrada} = \int_{-\pi}^\pi |\Phi(\omega)|^2 S_v(\omega) e^{i\omega k} \, d\omega = 2 \int_0^\pi |\Phi(\omega)|^2 \operatorname{Re}\left[ S_v(\omega) e^{i\omega k} \right] \, d\omega$$

Puremacro aproxima esta integración espectral empleando $n_{quad} = 256$ nodos y ponderaciones de Gauss-Legendre proyectados sobre el intervalo $[0, \pi]$.

#### Filtros espectrales soportados

1. **Filtro de Hodrick-Prescott (`hp_filter = lambda`)**:
   $$|\Phi_{HP}(\omega)|^2 = \frac{4\lambda (1 - \cos\omega)^2}{1 + 4\lambda (1 - \cos\omega)^2}$$
   donde habitualmente $\lambda = 1600$ para frecuencia trimestral, $\lambda = 6.25$ para anual y $\lambda = 129600$ para mensual.
2. **Filtro de paso de banda o Baxter-King (`bandpass_filter = [low, high]`)**:
   Filtro ideal de paso de banda que preserva oscilaciones entre periodicidades $p_{high} = 2\pi/\omega_{low}$ y $p_{low} = 2\pi/\omega_{high}$ (por ejemplo, $[6, 32]$ trimestres para ciclos económicos):
   $$|\Phi_{BP}(\omega)|^2 = \begin{cases} 1 & \text{si } \omega_{low} \le \omega \le \omega_{high} \\ 0 & \text{en otro caso} \end{cases}$$

### 2.2 Filtro HP uniselectivo causal de Kalman

Dado que el filtro HP estándar de dos lados utiliza información futura y no puede aplicarse en tiempo real de forma causal, el **filtro HP uniselectivo** (`one_sided_hp_filter`) resuelve recursivamente el modelo de espacio de estados:

$$\Delta^2 \tau_t = \eta_t, \quad y_t = \tau_t + \varepsilon_t, \quad \frac{\sigma_\varepsilon^2}{\sigma_\eta^2} = \lambda$$

Definiendo el vector de estado $s_t = [\tau_t, \tau_{t-1}]^\top$:

$$s_t = \begin{bmatrix} 2 & -1 \\ 1 & 0 \end{bmatrix} s_{t-1} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t, \quad y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} s_t + \varepsilon_t$$

Puremacro estima la tendencia causal $\tau_{t|t}$ de manera progresiva mediante el filtro de Kalman discreto con inicialización por extrapolación lineal hacia atrás idéntica a Dynare.

### 2.3 Autocorrelaciones cruzadas y correlaciones contemporáneas

A partir de las autocovarianzas teóricas o filtradas $\Gamma_k$, la matriz de correlaciones contemporáneas $R(0)$ y las matrices de autocorrelación cruzada $R(k)$ para rezagos $k = 1, \dots, n$ se obtienen como:

$$D = \operatorname{diag}(\Gamma_0), \quad R(0) = D^{-1/2} \Gamma_0 D^{-1/2}, \quad R(k) = D^{-1/2} \Gamma_k D^{-1/2}$$

Todos los elementos diagonales de $R(0)$ son estrictamente 1.0 y sus valores quedan delimitados al intervalo cerrado $[-1, 1]$.

### 2.4 Momentos empíricos de simulación (`simul_replic = M`)

Al especificar `simul_replic = M` en `stoch_simul`:
1. El motor simula $M$ trayectorias muestrales independientes de longitud `periods`.
2. Aplica el filtro de series temporales solicitado (por ejemplo, HP bidireccional o causal) sobre cada trayectoria simulada.
3. Calcula medias, desviaciones estándar, varianzas y correlaciones muestrales agregadas.
4. Reporta el error estándar de Monte Carlo $\sigma / \sqrt{M}$ junto con cada estimador puntual.

```python
# requires: standalone snippet
from puremacro.dsge import load_mod

# Carga y resolución del modelo
m = load_mod("rbc.mod")
m.solve()

# Momentos teóricos con filtro HP(1600) y matriz de autocorrelaciones hasta orden 4
res = m.stoch_simul(hp_filter=1600.0, ar=4, contemporaneous_correlation=True)
print(res.summary())

# Simulación empírica de Monte Carlo con M=500 réplicas
res_mc = m.stoch_simul(periods=200, simul_replic=500, hp_filter=1600.0)
print(res_mc.simulated_moments)
```

---

## 3. Simulación estocástica no lineal por sendero extendido

### 3.1 El algoritmo de Fair y Taylor (1983)

El método de **sendero extendido** (*Extended Path*, `extended_path`) permite simular modelos macroeconómicos dinámicos no lineales bajo expectativas racionales sin recurrir a truncamientos en series de Taylor ni a la discretización en mallas de la programación dinámica.

En cada periodo temporal de la simulación $t = 0, \dots, T-1$:
1. Se extrae o proporciona el vector de innovaciones estructurales $u_t$.
2. Los agentes económicos operan bajo la expectativa de que todas las innovaciones futuras serán nulas:
   $$\mathbb{E}_t[u_{t+s}] = 0 \quad \text{para } s \ge 1$$
3. Se plantea el problema determinista apilado de valores en la frontera sobre el horizonte de anticipación $T_H$:
   $$f(y_{t+1|t}, y_{t|t}, y_{t-1}, u_t) = 0$$
   $$f(y_{t+s+1|t}, y_{t+s|t}, y_{t+s-1|t}, 0) = 0 \quad \text{para } s = 1, \dots, T_H - 1$$
   con condición terminal de estado estacionario $y_{t+T_H|t} = y_{ss}$.
4. El sistema global apilado se resuelve de manera simultánea mediante el algoritmo de Newton-Raphson con factorización dispersa SuperLU de `puremacro.dsge.perfect_foresight`.
5. Se extrae la realización contemporánea $y_t = y_{t|t}$, se actualiza el estado rezagado ($y_{t-1} \leftarrow y_t$) y se avanza al instante $t+1$ reutilizando la trayectoria previa como punto inicial de iteración (*warm-start*).

### 3.2 Invariancia para modelos lineales

Una propiedad formal esencial del algoritmo de Fair y Taylor es su **invariancia lineal**: cuando las ecuaciones del modelo $f$ son lineales en $(y_{t+1}, y_t, y_{t-1}, u_t)$, la trayectoria producida por el sendero extendido coincide exactamente con la simulación del espacio de estados lineal de primer orden:

$$\| y_t^{EP} - y_t^{Lineal} \|_\infty \le 10^{-10}$$

Puremacro valida de forma estricta esta invariancia matemática en sus pruebas de integración continua.

```python
# requires: standalone snippet
from puremacro.dsge.extended_path import extended_path

# Simulación estocástica no lineal sobre 150 periodos con horizonte 80
ep_res = extended_path(
    model_or_equations=m,
    periods=150,
    horizon=80,
    seed=42,
    tol=1e-8,
)

# Consulta de resultados y generación de gráficos
df_sim = ep_res.to_frame()
fig = ep_res.plot(variables=["c", "k", "y", "r"])
```

---

## 4. Pronóstico condicional y descomposición de perturbaciones

### 4.1 Pronóstico condicional (Waggoner y Zha 1999)

En el análisis de política económica, las instituciones formuladoras de política suelen proyectar escenarios condicionados a trayectorias prefijadas de determinadas variables objetivo (por ejemplo, mantener la tasa de interés de política fija durante cuatro trimestres).

Siguiendo a **Waggoner y Zha (1999, *Journal of Economic Dynamics and Control*)**, el pronóstico condicional determina la secuencia de innovaciones estructurales requerida $U = [u_1^\top, \dots, u_H^\top]^\top$ que satisface:

$$R \, U = d$$

donde $R$ es la matriz apilada de multiplicadores de impulso-respuesta que conecta las perturbaciones con las variables condicionadas, y $d$ es la discrepancia entre la proyección incondicional base y la trayectoria objetivo estipulada.

#### Métodos de resolución:
- **Ponderado por covarianza (`method="covariance_weighted"`)**: Minimiza la distancia de Mahalanobis $U^\top \Sigma_U^{-1} U$, ponderando las perturbaciones según su probabilidad estructural:
  $$U = \Sigma_U R^\top \left( R \Sigma_U R^\top \right)^{-1} d$$
- **Mínima energía (`method="minimum_energy"`)**: Minimiza la norma euclidiana $\|U\|_2$:
  $$U = R^\top \left( R R^\top \right)^{-1} d$$

```python
# requires: standalone snippet
from puremacro.dsge.conditional import conditional_forecast

# Fijar la tasa de interés 'r' en 0.015 durante los primeros 4 trimestres
cf_res = conditional_forecast(
    model=m,
    conditions={"r": [0.015, 0.015, 0.015, 0.015]},
    horizon=12,
    controlled_shocks=["eps_r"],
    method="covariance_weighted",
)

# Inspección de shocks requeridos y gráfico de abanico
print(cf_res.shocks)
fig = cf_res.plot(variables=["y", "pi", "r"])
```

### 4.2 Descomposiciones históricas por grupos de shocks (`shock_groups;`)

En los archivos `.mod`, es posible declarar agrupaciones lógicas de perturbaciones:

```dynare
shock_groups;
  'Oferta' = ea, eb;
  'Demanda' = eq, ec;
  'Politica Monetaria' = em;
end;
```

El módulo `puremacro.dsge.shock_groups` descompone las series observadas históricas en las contribuciones de estos grupos económicos más el efecto de las condiciones iniciales:

$$y_t^{obs} = \sum_{g \in \text{grupos}} y_t^{(g)} + y_t^{(init)}$$

Esta descomposición satisface el **balance contable exacto** a precisión de máquina ($\le 10^{-12}$).

```python
# requires: standalone snippet
from puremacro.dsge.shock_groups import shock_groups_decomposition

decomp = shock_groups_decomposition(
    model=m,
    data=observed_df,
    groups={
        "Oferta": ["ea", "eb"],
        "Demanda": ["eq", "ec"],
        "Politica Monetaria": ["em"],
    }
)
fig = decomp.plot(variable="y")
```

### 4.3 Gráficos de abanico para IRF bayesianas y predictivo a priori

- **`bayesian_irf`**: Evalúa las respuestas al impulso a lo largo de las extracciones de parámetros de cadenas MCMC o SMC, calculando intervalos creíbles puntuales (mediana, 68%, 90%, 95%) y generando gráficos de abanico (`BayesianIRFResult`).
- **`prior_predictive`**: Muestrea parámetros desde las distribuciones a priori y calcula momentos incondicionales e IRFs antes de incorporar observaciones empíricas, facilitando el diagnóstico de coherencia previa del modelo (`PriorPredictiveResult`).
- **`qz_criterium` configurable**: En modelos con raíces unitarias o relaciones de cointegración (como modelos de crecimiento o economías abiertas sin ancla nominal), es posible ajustar el umbral de separación de autovalores generalizados de Schur ($\alpha = 1.0 + \varepsilon$) evitando clasificaciones espurias de inestabilidad.

---

## 5. Panel de paridad con Dynare y CLI

### 5.1 Motor de verificación y umbrales de tolerancia

La infraestructura de paridad (`puremacro.dsge.parity`) contrasta directamente las soluciones obtenidas en puremacro frente a las salidas oficiales generadas por Dynare (archivos `.mat` con estructuras `oo_.dr`, `oo_.mean`, `oo_.var`, `oo_.autocorr`).

El motor compara las discrepancias absolutas máximas:

$$\Delta_{max}(A, B) = \max_{i,j} |A_{ij} - B_{ij}|$$

| Métrica | Matriz objetivo | Umbral de tolerancia |
| :--- | :--- | :--- |
| Estados de primer orden ($ghx$) | `oo_.dr.ghx` | $\le 10^{-6}$ (típicamente $\le 10^{-12}$) |
| Perturbaciones de primer orden ($ghu$) | `oo_.dr.ghu` | $\le 10^{-6}$ (típicamente $\le 10^{-12}$) |
| Estados de segundo orden ($ghxx$) | `oo_.dr.ghxx` | $\le 10^{-4}$ (típicamente $\le 10^{-10}$) |
| Corrección de riesgo ($ghs2$) | `oo_.dr.ghs2` | $\le 10^{-4}$ (típicamente $\le 10^{-10}$) |
| Medias ergódicas | `oo_.mean` | $\le 10^{-5}$ |
| Varianzas ergódicas | `oo_.var` | $\le 10^{-5}$ |
| Autocorrelaciones | `oo_.autocorr` | $\le 10^{-5}$ |

### 5.2 Uso desde Python

```python
# requires: standalone snippet
from puremacro.dsge.parity import verify_dynare_parity, run_parity_suite

# Verificación de un modelo individual
report = verify_dynare_parity(
    puremacro_model="sw07.mod",
    dynare_output="sw07_results.mat",
    order=2,
)
print(report.to_markdown())

# Verificación masiva de un directorio completo de modelos
suite_report = run_parity_suite("tests/fixtures/dynare_benchmarks/")
assert suite_report.passed
```

### 5.3 Interfaz de línea de comandos (`puremacro-dynare parity`)

El paquete proporciona el comando CLI nativo `puremacro-dynare parity`:

```bash
# Verificar un archivo .mod individual frente a su archivo .mat correspondiente
puremacro-dynare parity models/sw07.mod --order 2

# Evaluar todos los modelos de un directorio y exportar la tarjeta de evaluación en LaTeX
puremacro-dynare parity benchmarks/ --format latex --outdir reports/

# Modo estricto para CI con tolerancia personalizada
puremacro-dynare parity rbc.mod --tol 1e-8 --quiet
```

Ejemplo de salida en consola:

```
================================================================================
Dynare Parity Scorecard: sw07.mod vs sw07_results.mat
================================================================================
Component        Max Dev        Tolerance     Status
--------------------------------------------------------------------------------
dr.ghx           4.12e-13       1.00e-06      PASSED
dr.ghu           2.88e-13       1.00e-06      PASSED
dr.ghxx          8.45e-11       1.00e-04      PASSED
dr.ghs2          3.20e-11       1.00e-04      PASSED
moments.var      5.14e-07       1.00e-05      PASSED
moments.corr     2.31e-07       1.00e-05      PASSED
--------------------------------------------------------------------------------
Overall Status: PASSED (6/6 checks clean)
================================================================================
```

---

## 6. Conformidad con Pyodide y garantía de cero dependencias externas

Todos los algoritmos de Tier 3 cumplen estrictamente el **contrato de cuatro paquetes de Pyodide**:
- **NumPy**: Álgebra lineal, vectorización, manipulación matricial, autovalores y generación pseudoaleatoria.
- **SciPy**: Sistemas lineales dispersos (`scipy.sparse.linalg.splu`), funciones especiales (`erf`, `normcdf`) y factorizaciones matriciales (`scipy.linalg.ordqz`).
- **Pandas**: Estructuras Series y DataFrames etiquetadas para variables, fechas y paneles de control.
- **Matplotlib**: Visualización de gráficos de abanico, descomposiciones históricas en barras e impulso-respuesta.

No se requiere ningún software o solucionador externo adicional (sin SymPy, sin CasADi, sin JAX), generadores de analizadores sintácticos (sin PLY, ni Lark, ni ANTLR) ni extensiones binarias compiladas en C o Fortran. La totalidad del paquete se ejecuta fluidamente en entornos web vía WebAssembly (Pyodide), estaciones de trabajo y centros de computación de alto rendimiento.
