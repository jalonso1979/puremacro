> 🇬🇧 [English](../vfi_continuous_equilibrium.md) · 🇪🇸 Español

# Distribuciones continuas estacionarias y equilibrio general

`puremacro.vfi.continuous_distribution` implementa el método de simulación continua no estocástica desarrollado por **Young (2010, *Journal of Economic Dynamics and Control*)** y el marco acelerado en dos etapas de **Tan (2020, *Computational Economics*)**. Proporciona proyecciones de histograma con conservación de masa a precisión de máquina, operadores de transición de Markov dispersos (CSR, *Compressed Sparse Row*), solucionadores de distribución invariante y algoritmos de vaciado de mercado en equilibrio general para modelos macroeconómicos continuos con agentes heterogéneos.

En modelos macroeconómicos con riesgo no asegurable en los ingresos laborales (p. ej., Aiyagari 1994, Huggett 1993), los hogares optimizan sobre un continuo de activos $k \in [\underline{k}, \bar{k}]$. Mientras que la simulación de Monte Carlo con un panel finito de hogares introduce ruido estocástico y una lenta convergencia $O(1/\sqrt{N})$, el método no estocástico de Young proyecta las reglas de decisión continuas sobre una representación fina de densidad lineal a trozos. Esto garantiza la conservación exacta de la masa de probabilidad ($\sum \mu^* = 1.0 \pm 10^{-12}$) y de la tenencia media local de activos, permitiendo determinar los precios de equilibrio general con extrema rapidez mediante algoritmos de búsqueda de raíces sin derivadas.

> **Esta página describe la vía en tiempo discreto.** El tiempo es discreto:
> primero se resuelve una función de política $g(k, z)$, y este módulo la
> proyecta sobre un histograma fino para obtener la *masa de probabilidad*
> invariante $\mu^*$. puremacro incluye además una vía en **tiempo continuo** —
> `solve_hjb_achdou`, `solve_kfe_achdou` y `solve_aiyagari_continuous_hjb` —,
> donde un esquema implícito a contracorriente resuelve la ecuación de
> Hamilton-Jacobi-Bellman y la ecuación adjunta de Kolmogorov hacia adelante
> $A^\top g = 0$ devuelve una *densidad* $g$ en una única resolución lineal, sin
> iteración de la función de política ni pesos de lotería. Son dos
> discretizaciones distintas de la misma economía y no son intercambiables: aquí
> $\mu^*$ es masa por nodo y suma 1, mientras que allí $g$ es una densidad e
> integra 1 (en una malla no uniforme, masa del nodo dividida por el ancho de la
> celda y normalizada con pesos de cuadratura trapezoidal). Véase
> [`docs/es/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## 1. Marco teórico y computacional

### 1.1 El problema de la distribución invariante transversal

Sea la productividad laboral idiosincrática $z_t \in \mathcal{Z} = \{z_1, \dots, z_{n_z}\}$ gobernada por una cadena de Markov discreta con matriz de transición $\mathbf{P}_z$, donde $\Pr(z_{t+1} = z_m \mid z_t = z_j) = P_z(j, m)$. Las tenencias de activos $k_t \in [\underline{k}, \bar{k}]$ siguen una política óptima continua $g(k, z): [\underline{k}, \bar{k}] \times \mathcal{Z} \to [\underline{k}, \bar{k}]$ obtenida mediante programación dinámica (como colocación de Chebyshev, MEF Galerkin o EGM).

La distribución transversal conjunta de activos y productividad en la fecha $t$ se representa mediante la densidad de probabilidad $\mu_t(k, z)$. Dicha distribución evoluciona según el operador adjunto de Markov $\mathcal{T}^*$:

$$\mu_{t+1}(A, z_m) = \sum_{j=1}^{n_z} P_z(j, m) \int_{k \in g^{-1}(A, z_j)} \mu_t(dk, z_j)$$

para cualquier subconjunto de Borel $A \subseteq [\underline{k}, \bar{k}]$. Una distribución invariante o estacionaria $\mu^*(k, z)$ constituye un punto fijo de este operador:

$$\mu^* = \mathcal{T}^* \mu^*, \quad \text{con} \quad \sum_{m=1}^{n_z} \int_{\underline{k}}^{\bar{k}} \mu^*(dk, z_m) = 1.0$$

### 1.2 Proyección de masa por loterías lineales de Young (2010)

Para evaluar $\mathcal{T}^*$ numéricamente sin ruido estocástico, Young (2010) discretiza el espacio de estados continuo de activos en una rejilla de histograma fina $\mathbf{k} = \{k_1, k_2, \dots, k_{N_k}\}$ con $k_1 = \underline{k}$ y $k_{N_k} = \bar{k}$, habitualmente configurada con $N_k \in [500, 2000]$ nodos.

Para cualquier elección continua de ahorro $k' = g(k_i, z_j)$, el valor óptimo rara vez coincide de forma exacta con un nodo de la rejilla. El método de Young acota $k'$ entre dos nodos adyacentes $k_l \le k' \le k_{l+1}$ y asigna la masa de probabilidad mediante una lotería lineal:

$$\omega_l(k') = \frac{k_{l+1} - k'}{k_{l+1} - k_l}, \quad \omega_{l+1}(k') = 1 - \omega_l(k') = \frac{k' - k_l}{k_{l+1} - k_l}$$

Las elecciones situadas fuera de los límites ($k' < k_1$ o $k' > k_{N_k}$) se truncan a los nodos frontera con peso unitario $1.0$. Esta formulación de lotería satisface dos propiedades matemáticas cruciales:

1. **Conservación estricta de la masa de probabilidad**: $\omega_l(k') + \omega_{l+1}(k') = 1.0$ de manera exacta.
2. **Preservación del primer momento**: $\omega_l(k') k_l + \omega_{l+1}(k') k_{l+1} = k'$, lo que evita cualquier sesgo numérico en la acumulación agregada de riqueza.

### 1.3 Operadores dispersos de Markov y conservación de masa

Al combinar los pesos de la lotería de activos con las probabilidades de transición de la productividad exógena, se obtiene un operador lineal $\mathbf{T}^*$ que proyecta el vector de probabilidad discreta $\boldsymbol{\mu}_t \in \mathbb{R}^{N_k \cdot n_z}$:

$$\mu_{t+1}(k_l, z_m) = \sum_{j=1}^{n_z} P_z(j, m) \sum_{i=1}^{N_k} \mathbf{1}_{\{g(k_i, z_j) \in [k_l, k_{l+1}]\}} \omega_l(g(k_i, z_j)) \mu_t(k_i, z_j)$$

Dado que cada estado $(k_i, z_j)$ distribuye masa a lo sumo a dos nodos de activos por cada estado de productividad de destino $z_m$, la matriz del operador es extremadamente dispersa. En `puremacro`, este operador se ensambla en formato CSR (*Compressed Sparse Row*) con no más de $2 \cdot n_z \cdot N_k$ elementos distintos de cero (tasa de dispersión superior al $99.5\%$).

La distribución invariante resuelve el sistema lineal de autovalores:

$$(\mathbf{T}^* - \mathbf{I}) \boldsymbol{\mu}^* = \mathbf{0}, \quad \text{sujeto a} \quad \mathbf{1}^\top \boldsymbol{\mu}^* = 1.0$$

`puremacro` resuelve este sistema mediante cuatro métodos intercambiables:
- **`sparse_direct`**: Factorización dispersa directa SuperLU, sustituyendo una ecuación redundante por la restricción de suma unitaria.
- **`power`**: Iteración hacia adelante acelerada en dos etapas de Tan (2020): $\boldsymbol{\mu}^{(n+1)} = \mathbf{T}^* \boldsymbol{\mu}^{(n)}$.
- **`arnoldi`**: Iteración de Arnoldi con reinicio implícito para calcular el autovector dominante asociado al autovalor unitario $\lambda_1 = 1$.
- **`auto`**: Factorización directa dispersa con conmutación automática al método de potencias si se superan los umbrales de condicionamiento.

### 1.4 Vaciado de mercado en equilibrio general en espacios continuos

En la economía continua canónica de Aiyagari (1994), la oferta agregada de trabajo efectivo se determina mediante la media ergódica del proceso estocástico de productividad:

$$\bar{L} = \sum_{m=1}^{n_z} \pi_z(m) z_m$$

Dado un tipo de interés real tentativo $r$, la maximización de beneficios de las empresas competitivas determina la demanda agregada de capital $K^d(r)$ y el salario de equilibrio $w(r)$:

$$K^d(r) = \bar{L} \left( \frac{r + \delta}{\alpha A} \right)^{\frac{1}{\alpha - 1}}, \quad w(r) = (1 - \alpha) A \left( \frac{K^d(r)}{\bar{L}} \right)^\alpha$$

Los hogares resuelven su programa de optimización intertemporal continuo dados los precios factoriales $(r, w(r))$ para obtener la política de ahorro $g(k, z; r)$. La distribución invariante $\mu^*(k, z; r)$ define la oferta agregada de capital:

$$K^s(r) = \sum_{m=1}^{n_z} \sum_{i=1}^{N_k} k_i \mu^*(k_i, z_m; r)$$

El equilibrio general macroeconómico se alcanza en el tipo de interés que vacía el mercado $r^*$:

$$\Phi(r) \equiv K^s(r) - K^d(r) = 0$$

Dado que $K^s(r) \to \infty$ cuando $r \to 1/\beta - 1$ y $K^s(r) \to 0$ cuando $r \to -\delta$, existe un único tipo de interés de equilibrio $r^* \in (-\delta, 1/\beta - 1)$, el cual se calcula mediante el algoritmo de bisección de Brent.

---

## 2. Opciones metodológicas y del solucionador

| Característica | `sparse_direct` | `power` (Tan 2020) | `arnoldi` | `auto` |
|---|---|---|---|---|
| **Base matemática** | SuperLU $(\mathbf{I} - \mathbf{T}^*)^{-1}$ | Iteración de punto fijo $\boldsymbol{\mu}^{(n+1)} = \mathbf{T}^* \boldsymbol{\mu}^{(n)}$ | Solucionador en subespacio de Krylov | Híbrido disperso directo + potencias |
| **Iteraciones** | 1 pase de factorización | 500–5,000 iteraciones | 20–80 vectores de Krylov | 1 pase (o iteración de respaldo) |
| **Conservación de masa** | Normalizada a $1.0 \pm 10^{-14}$ | Precisión de máquina en cada paso | Normalizada en convergencia | $1.0 \pm 10^{-12}$ garantizada |
| **Memoria** | $O(N_k \cdot n_z)$ elementos CSR | Mínima (2 vectores de estado) | Moderada (base de Krylov) | Asignación dinámica |
| **Ámbito recomendado** | $N_k \le 2000$, equilibrio general | Rejillas ultra-finas ($N_k > 5000$) | Operadores no estándar | Solucionador predeterminado |

---

## 3. Calibración canónica y especificación empírica

La economía de referencia con mercados incompletos adopta los siguientes parámetros trimestrales:

| Parámetro | Símbolo | Valor de referencia | Justificación económica |
|---|---|---|---|
| Factor de descuento subjetivo | $\beta$ | $0.960$ | Ajusta la ratio capital/PIB anualizada $\approx 3.0$ |
| Aversión relativa al riesgo | $\gamma$ | $2.000$ | Elasticidad de sustitución intertemporal empírica $1/\gamma = 0.5$ |
| Participación del capital en el producto | $\alpha$ | $0.360$ | Participación del capital en el ingreso nacional |
| Tasa de depreciación del capital | $\delta$ | $0.080$ | Tasa anual de depreciación física $\approx 8\%$ |
| Persistencia de la productividad | $\rho_z$ | $0.900$ | Persistencia autorregresiva de los ingresos individuales |
| Desviación típica del choque de productividad | $\sigma_z$ | $0.200$ | Dispersión salarial transversal |
| Nodos de productividad | $n_z$ | $5$ | Discretización mediante Tauchen o Rouwenhorst |
| Tamaño de la rejilla de activos | $N_k$ | $1000$ | Representación de alta resolución de la densidad |
| Límite de endeudamiento | $\underline{k}$ | $0.000$ | Restricción financiera de no endeudamiento |

---

## 4. Ejemplos prácticos ejecutables

El siguiente script autocontenido ilustra:
1. La verificación de los pesos de la lotería de Young (2010) y la preservación del primer momento.
2. El cálculo de la distribución invariante $\mu^*(k)$ para una política continua.
3. La resolución del equilibrio general macroeconómico $(r^*, K^*, w^*)$ en la economía de Aiyagari.

```python
import numpy as np
from puremacro.vfi import (
    continuous_stationary_distribution,
    young_lottery_weights,
    solve_aiyagari_continuous,
)

# 1. Verificación de los pesos de la lotería lineal de Young (2010)
k_grid = np.linspace(0.0, 10.0, 100)
policy_k = np.clip(0.85 * k_grid + 0.5, 0.0, 10.0)

j_lo, w_lo, w_hi = young_lottery_weights(policy_k, k_grid)
assert np.allclose(w_lo + w_hi, 1.0)
assert np.all(w_lo >= 0.0) and np.all(w_hi >= 0.0)

# 2. Distribución estacionaria invariante (política AR(1) continua)
dist_1d = continuous_stationary_distribution(
    policy_k, k_grid, method="sparse_direct"
)
assert dist_1d.converged
assert dist_1d.mass_error <= 1e-12
mean_assets = dist_1d.mean()
assert 3.0 < mean_assets < 3.5

# 3. Equilibrio general continuo (economía de Aiyagari)
ge_result = solve_aiyagari_continuous(
    beta=0.96,
    gamma=2.0,
    alpha=0.36,
    delta=0.08,
    N_k=150,
    n_z=3,
    max_evals=25,
)
# converged: punto fijo del EGM del hogar, distribución estacionaria y
# |K^s - K^d| < tol_ge (1e-4), todo comprobado en r*
assert ge_result.converged
assert ge_result.metadata["egm_converged"]
assert abs(ge_result.capital_market_clearing_error) < 1e-4

summary_df = ge_result.summary()
print(f"Equilibrio r* = {ge_result.r:.4f}, K* = {ge_result.K:.4f}, w* = {ge_result.w:.4f}")
print(f"Iteraciones del EGM en r*: {ge_result.metadata['egm_iterations']}")
```

`ge_result.converged` solo es True cuando se cumplen tres condiciones en $r^*$: la política de consumo del hogar es un punto fijo del operador del EGM (de Coleman) con tolerancia `egm_tol` como máximo en `egm_max_iter` iteraciones, el solucionador de la distribución estacionaria convergió y $|K^s - K^d| <$ `tol_ge`. En caso contrario es False y un `RuntimeWarning` nombra el criterio que falló; `metadata` conserva `egm_iterations`, `egm_residual`, `dist_converged` y `clearing_ok`. Este ejemplo necesita unas 2600 iteraciones del EGM en $r^*$, porque $\beta(1+r^*)$ está cerca de uno. Hasta la versión 4.3.0 el bucle se detenía en silencio a las 500 iteraciones y `converged` era siempre True: la política quedaba hasta 0.09 lejos del punto fijo, el exceso de demanda verdadero en el $r^*$ reportado era $1.6\cdot10^{-2}$ frente a un valor reportado de $-2.5\cdot10^{-6}$, y $r^*$ salía 0.039421 en lugar de 0.039416 ($K^*$ 6.6252 en lugar de 6.6257).

---

## 5. Especificación completa de la API

```text
young_lottery_weights(
    kp_eval: np.ndarray,
    k_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]

continuous_push_distribution(
    pdf: np.ndarray,
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
) -> np.ndarray

build_continuous_transition_matrix(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
) -> scipy.sparse.csr_matrix

continuous_stationary_distribution(
    policy_fn: Any,
    asset_grid: np.ndarray,
    shock_transition: np.ndarray | None = None,
    shock_grid: np.ndarray | None = None,
    *,
    method: str = "auto",
    tol: float = 1e-12,
    max_iter: int = 50000,
    backend: str = "numpy",
) -> ContinuousStationaryDistribution

solve_aiyagari_continuous(
    beta: float = 0.96,
    gamma: float = 2.0,
    alpha: float = 0.36,
    delta: float = 0.08,
    rho_z: float = 0.90,
    sigma_z: float = 0.20,
    n_z: int = 5,
    a_max: float = 30.0,
    N_k: int = 1000,
    r_bracket: tuple[float, float] | None = None,
    solver: str = "auto",
    backend: str = "numpy",
    xtol: float = 1e-8,
    max_evals: int = 100,
    dist_options: dict | None = None,
    P_z: np.ndarray | None = None,
    z_grid: np.ndarray | None = None,
    n_a: int = 150,
    egm_tol: float = 1e-8,
    egm_max_iter: int = 10000,
    tol_ge: float = 1e-4,
    **kwargs: Any,
) -> AiyagariContinuousEquilibrium
```

#### Parámetros:
- `policy_fn`: Fuente de la regla de decisión continua. Admite un array NumPy 1D/2D, una función invocable `g(k)` o `g(k, z)`, o una solución convergida (`CollocationSolution`, `FEMSolution`, `SplineCollocationSolution`).
- `asset_grid`: Rejilla de activos 1D estrictamente creciente $k_1 < k_2 < \dots < k_{N_k}$ con $N_k \ge 2$.
- `shock_transition`: Matriz estocástica por filas de probabilidades de transición $\mathbf{P}_z \in \mathbb{R}^{n_z \times n_z}$.
- `shock_grid`: Rejilla de valores discretos de productividad $z_1, \dots, z_{n_z}$.
- `method`: Algoritmo de resolución: `"auto"`, `"sparse_direct"`, `"power"` o `"arnoldi"`.
- `tol`: Tolerancia absoluta de convergencia para la medida invariante (por defecto $10^{-12}$).
- `r_bracket`: Intervalo $(r_{\min}, r_{\max})$ que acota el tipo de interés para el método de Brent, con $-\delta < r_{\min}$ y $r_{\max} < 1/\beta - 1 - 10^{-4}$. Si se omite, la búsqueda empieza en $(10^{-3}, 1/\beta - 1 - 5\cdot10^{-4})$; si el ahorro ya supera la demanda de capital de la empresa en $10^{-3}$, el extremo inferior pasa a $-\delta + 10^{-3}$, de modo que se encuentran equilibrios con tipos negativos.
- `solver`: `"auto"` o `"egm"` (sin distinguir mayúsculas; `None` se lee como `"auto"`); ambos eligen el método de la rejilla endógena, el único solucionador del hogar implementado aquí. Cualquier otro valor, incluidos `"collocation"` y `"fem"`, lanza `ValueError` (hasta la versión 4.3.0 todos los valores se resolvían en silencio con EGM). Para un hogar resuelto por colocación o elementos finitos, pase su propio problema a `continuous_stationary_equilibrium`.
- `xtol`: Tolerancia de Brent sobre el propio tipo de interés $r$; `tol_ge` comprueba el residuo de vaciado del mercado, que es aproximadamente `xtol` por la pendiente de $K^s - K^d$ en $r$ (varios cientos cerca de $r^*$ con `n_z=3`). `AiyagariContinuousModel.solve` y `AiyagariContinuousEquilibrium.solve` usan el mismo valor por defecto, $10^{-8}$; hasta la versión 4.3.0 usaban $10^{-6}$, que deja $|K^s - K^d| \approx 6\cdot10^{-4}$ en el ejemplo anterior, de modo que con ese valor `converged` es False.
- `P_z`, `z_grid`: Cadena de Markov de productividad que sustituye a la discretización de Tauchen de `rho_z`, `sigma_z`, `n_z`.
- `n_a`: Nodos de la rejilla de activos del hogar para el EGM, $a_{\max} \cdot \operatorname{linspace}(0, 1, n_a)^{1.5}$. Súbalo junto con `a_max`: el espaciado cerca de la restricción de endeudamiento determina el error en $r$. En la economía de Aiyagari con $\sigma = 0.4$, $\rho = 0.9$, $\mu = 5$, `a_max=200` con los 150 nodos por defecto desplaza el equilibrio ($-0.09\%$) en 0.03 puntos porcentuales, y `n_a=800` lo deja a menos de 0.003.
- `egm_tol`: Regla de parada del bucle EGM del hogar: la variación en norma del supremo de la política de consumo entre dos iteraciones (en unidades de consumo; por defecto $10^{-8}$). Cuando la iteración contrae despacio, la política está más lejos del punto fijo que este paso: unos $2\cdot10^{-7}$ con los valores por defecto.
- `egm_max_iter`: Tope de iteraciones del EGM por tipo de prueba (por defecto 10000; hasta la versión 4.3.0 era un 500 fijo y oculto). El número crece a medida que $r$ se acerca a $1/\beta - 1$ y depende de la calibración: el ejemplo anterior necesita unas 2600 en $r^*$ y unas 3700 en el extremo superior del intervalo por defecto; las calibraciones con más riesgo necesitan más en ese extremo (unas 13300 con `sigma_z=0.35`, `n_z=3` y el valor por defecto `a_max=30`), de modo que el tope puede alcanzarse en el extremo del intervalo; en esa economía $r^*$ coincide hasta $10^{-10}$ con `egm_max_iter=20000`. Alcanzar el tope en un tipo de prueba distinto de $r^*$ (en la práctica el extremo superior del intervalo, donde Brent solo necesita el signo) solo puede desviar la búsqueda y se cuenta en `metadata["egm_cap_hits"]`; alcanzarlo en $r^*$ hace `converged` False y emite un aviso.
- `tol_ge`: Tolerancia de vaciado del mercado sobre $|K^s - K^d|$, en unidades de capital (por defecto $10^{-4}$), necesaria para `converged`.
- `**kwargs`: No se usa. Una palabra clave desconocida (por ejemplo `tol` o `max_iter`) emite un `FutureWarning` que la nombra y se ignora; en una versión futura lanzará `TypeError`. Hasta la versión 4.3.0 las palabras clave desconocidas se descartaban en silencio.

---

## 6. Interfaz de resultados y exportación a manuscritos

Tanto `ContinuousStationaryDistribution` como `AiyagariContinuousEquilibrium` proporcionan interfaces integrales para el análisis estadístico y la exportación de resultados:

### Métodos de `ContinuousStationaryDistribution`

- `dist.mean() -> float`: Media agregada de tenencias de activos $\mathbb{E}[k]$.
- `dist.std() -> float`: Desviación típica transversal de los activos.
- `dist.percentile(q) -> float`: Nivel de riqueza en el percentil $q \in [0, 100]$.
- `dist.gini() -> float`: Coeficiente de Gini de desigualdad de la riqueza.
- `dist.lorenz_curve(n_points=100) -> tuple[np.ndarray, np.ndarray]`: Coordenadas de población y riqueza acumulada $(p, L(p))$ para la curva de Lorenz.
- `dist.marginal_assets() -> np.ndarray`: Densidad marginal de probabilidad sobre los activos $\mu(k) = \sum_z \mu(k, z)$.
- `dist.marginal_shocks() -> np.ndarray`: Distribución marginal sobre los choques idiosincráticos.

### Métodos de exportación para publicaciones

`AiyagariContinuousEquilibrium` implementa los métodos de serialización estándar de `puremacro`:
- `res.summary() -> pd.DataFrame`: Cuadro resumen con precios factoriales, stock de capital, oferta de trabajo y discrepancias de vaciado.
- `res.to_frame() -> pd.DataFrame`: Alias que devuelve el DataFrame del cuadro resumen.
- `res.to_markdown(digits=4) -> str`: Tabla en Markdown compatible con GitHub para documentación y control de versiones.
- `res.to_latex(digits=4) -> str`: Entorno tabular en LaTeX de calidad para publicaciones académicas.
- `res.to_typst(digits=4) -> str`: Bloque de código Typst formateado para informes científicos.
- `res.plot(figsize=(10, 4.5)) -> matplotlib.figure.Figure`: Gráfico diagnóstico multipanel con la densidad invariante $\mu^*(k)$, la curva de Lorenz y el vaciado del mercado de capitales.

---

## Referencias

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Tan, C. (2020). "A Fast and Accurate Method for Solving Incomplete Markets Models." *Computational Economics*, 55, 347–362.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
