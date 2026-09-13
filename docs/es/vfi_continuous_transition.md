> 🇬🇧 [English](../vfi_continuous_transition.md) · 🇪🇸 Español

# Dinámica de transición continua y choques MIT

`puremacro.vfi.continuous_transition` implementa la dinámica de transición no lineal en equilibrio general para modelos macroeconómicos continuos con agentes heterogéneos tras perturbaciones agregadas imprevistas (**choques MIT**). El motor acopla la recursión hacia atrás de las reglas de decisión mediante el Método de la Malla Endógena (EGM continuo) con la propagación hacia adelante de la distribución transversal mediante el método no estocástico de **Young (2010)**, vaciando simultáneamente los mercados en el espacio de secuencias mediante un algoritmo especializado de Broyden Cuasi-Newton (**Auclert et al. 2021**).

Si bien las perturbaciones linealizadas de primer orden proporcionan aproximaciones locales precisas cerca del estado estacionario determinista, resultan insuficientes para capturar las no linealidades inducidas por choques de política de gran magnitud (como subidas abruptas de tipos de interés soberanos, paquetes de estímulo fiscal extraordinarios o perturbaciones estructurales de productividad). Ante dichos choques, las restricciones de endeudamiento asimétricas se activan, los motivos precautorios se intensifican y la distribución transversal de la riqueza $\mu_t(k, z)$ experimenta reasignaciones cuantitativamente relevantes. `puremacro` calcula la **trayectoria exacta de transición de equilibrio no lineal** sobre horizontes arbitrarios $T$ garantizando la conservación estricta de la masa de probabilidad ($\sum \mu_t = 1.0 \pm 10^{-12}$) en cada fecha.

---

## 1. Marco teórico y computacional

### 1.1 El problema de transición en el espacio de secuencias

Considérese una economía inicializada en una distribución de estado estacionario $\mu_0(k, z)$ en la fecha $t = 0$. Una secuencia determinista e imprevista de choques agregados $\mathbf{Z} = (Z_0, Z_1, \dots, Z_{T-1})$ impacta la economía en $t = 0$, donde el horizonte $T$ se elige suficientemente amplio para que el sistema converja al estado estacionario final $\mu_T(k, z)$ en el período $T$.

El equilibrio general se define como las secuencias de precios factoriales $\{r_t, w_t\}_{t=0}^{T-1}$, reglas de decisión de los hogares $\{c_t(k, z), a'_t(k, z)\}_{t=0}^{T-1}$ y distribuciones continuas de riqueza $\{\mu_t(k, z)\}_{t=0}^{T}$ tales que:
1. Los hogares optimizan hacia atrás tomando la trayectoria de tipos de interés y salarios como dada.
2. La distribución transversal evoluciona hacia adelante bajo las reglas de decisión variantes en el tiempo.
3. Las empresas competitivas maximizan beneficios y el mercado de capitales se vacía en cada fecha $t \in [0, T-1]$:

$$H_t(\mathbf{r}) \equiv K_t^s(\mathbf{r}) - K_t^d(r_t; Z_t) = 0$$

donde la oferta agregada de capital viene dada por la integral continua transversal:

$$K_t^s = \sum_{m=1}^{n_z} \int_{\underline{k}}^{\bar{k}} k \, \mu_t(dk, z_m)$$

### 1.2 Recursión hacia atrás de políticas continuas vía EGM

Dada una trayectoria tentativa de tipos de interés $\mathbf{r} = (r_0, r_1, \dots, r_{T-1})$ y los correspondientes salarios competitivos $w_t = (1 - \alpha) Z_t (K_t^d / \bar{L})^\alpha$, el consumo y el ahorro óptimos se resuelven **hacia atrás en el tiempo** desde $t = T-1$ hasta $t = 0$ aplicando el Método de la Malla Endógena (EGM) sin sesgo de discretización.

En la fecha $t$, conocida la política de consumo de continuación $c_{t+1}(a', z')$, la ecuación de Euler define la utilidad marginal esperada:

$$\mathcal{EMU}_t(a', z_j) = \beta (1 + r_{t+1}) \sum_{m=1}^{n_z} P_z(j, m) \cdot u'\left( c_{t+1}(a', z_m) \right)$$

Invirtiendo la función de utilidad marginal se obtiene el consumo endógeno $c_t^{\text{endo}}(a', z_j) = (u')^{-1}(\mathcal{EMU}_t(a', z_j))$. El estado de riqueza previo a la decisión se deduce analíticamente de la restricción presupuestaria:

$$k_t^{\text{endo}}(a', z_j) = \frac{c_t^{\text{endo}}(a', z_j) + a' - w_t z_j}{1 + r_t}$$

Las funciones de política $a'_t(k, z)$ y $c_t(k, z)$ se interpolan sobre la rejilla densa y la de histograma, imponiendo la restricción de endeudamiento $a'_t \ge \underline{k}$ mediante truncamiento en la frontera inferior.

### 1.3 Propagación de densidad hacia adelante de Young (2010)

Partiendo de la distribución inicial predeterminada $\mu_0 = \mu_{\text{init}}^*$, la distribución transversal avanza período a período para $t = 0, \dots, T-1$ mediante los operadores lineales de lotería de Young (2010):

$$\mu_{t+1} = \mathcal{T}_t^* \mu_t = \text{continuous\_push\_distribution}(\mu_t, a'_t, \mathbf{k}, \mathbf{P}_z, \mathbf{z})$$

Para cada estado $(k_i, z_j)$, la elección continua $a'_t(k_i, z_j)$ se acota entre nodos adyacentes $k_l \le a'_t \le k_{l+1}$ asignando pesos de lotería lineal:

$$\omega_l = \frac{k_{l+1} - a'_t}{k_{l+1} - k_l}, \quad \omega_{l+1} = 1 - \omega_l$$

Esto garantiza:
- **Conservación exacta de la masa**: $\sum_{i, m} \mu_t(k_i, z_m) = 1.0 \pm 10^{-12}$ para todo $t \in [0, T]$.
- **Cero difusión numérica**: La oferta de capital $K_t^s = \sum_{i, m} k_i \mu_t(k_i, z_m)$ coincide con la agregación exacta de las decisiones individuales.

### 1.4 Vaciado de mercado en el espacio de secuencias vía Broyden

La trayectoria de tipos de interés de equilibrio $\mathbf{r}^* \in \mathbb{R}^T$ resuelve el sistema no lineal apilado:

$$\mathbf{H}(\mathbf{r}) = \begin{bmatrix} K_0^s(\mathbf{r}) - K_0^d(r_0; Z_0) \\ \vdots \\ K_{T-1}^s(\mathbf{r}) - K_{T-1}^d(r_{T-1}; Z_{T-1}) \end{bmatrix} = \mathbf{0}$$

Para resolver $\mathbf{H}(\mathbf{r}) = \mathbf{0}$ de forma eficiente sin reevaluar repetidamente matrices jacobianas numéricas, `puremacro` utiliza el **método Cuasi-Newton de Broyden con actualizaciones de Sherman-Morrison de rango 1**:

1. **Inicialización analítica**: La aproximación inicial del jacobiano inverso $\mathbf{B}_0 \approx \mathbf{J}^{-1}$ se inicializa mediante la derivada analítica de la demanda de capital:
   $$J_{t, t}^{\text{init}} \approx -\frac{\partial K_t^d}{\partial r_t} = \frac{1}{1 - \alpha} \frac{K_t^d}{r_t + \delta} > 0, \quad \mathbf{B}_0 = \text{diag}\left( \frac{1}{J_{t, t}^{\text{init}}} \right)$$
2. **Paso Cuasi-Newton**: En la iteración $k$, la actualización propuesta es:
   $$\Delta \mathbf{r}^{(k)} = - \mathbf{B}_k \mathbf{H}(\mathbf{r}^{(k)})$$
3. **Búsqueda lineal con retroceso monótono**: Se determina el paso $\lambda \in (0, 1]$ asegurando la contracción estricta de la norma del supremo:
   $$\|\mathbf{H}(\mathbf{r}^{(k)} + \lambda \Delta \mathbf{r}^{(k)})\|_\infty < \|\mathbf{H}(\mathbf{r}^{(k)})\|_\infty$$
4. **Actualización de Sherman-Morrison**: Definiendo $\Delta \mathbf{H} = \mathbf{H}^{(k+1)} - \mathbf{H}^{(k)}$ y $\Delta \mathbf{r} = \mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}$:
   $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\Delta \mathbf{r} - \mathbf{B}_k \Delta \mathbf{H}) (\Delta \mathbf{r}^\top \mathbf{B}_k)}{\Delta \mathbf{r}^\top \mathbf{B}_k \Delta \mathbf{H}}$$
5. **Criterio de convergencia**: El proceso finaliza cuando $\|\mathbf{H}(\mathbf{r})\|_\infty < \text{tol}$ (típicamente $10^{-4}$).

---

## 2. Opciones metodológicas y del solucionador

| Característica | Broyden Cuasi-Newton (`solver="broyden"`) | Relajación de disparo (*shooting*) (`solver="shooting"`) |
|---|---|---|
| **Tasa de convergencia** | Superlineal (típicamente 4–8 iteraciones) | Lineal (20–60 iteraciones) |
| **Mecanismo de paso** | Actualización completa de Sherman-Morrison en rango 1 | Relajación amortiguada: $\mathbf{r}^{(n+1)} = (1 - \omega)\mathbf{r}^{(n)} + \omega \mathbf{r}^{\text{implied}}$ |
| **Búsqueda lineal** | Búsqueda retrógrada monótona de tipo Armijo | Parámetro fijo de amortiguación $\omega \in (0, 1]$ |
| **Robustez** | Excepcional ante choques persistentes y de gran tamaño | Contracción garantizada para innovaciones marginales |
| **Tiempo de cálculo** | $< 0.5$ segundos para $T = 150$ | $1.0$–$2.5$ segundos para $T = 150$ |

---

## 3. Calibración canónica y especificación de choques

La dinámica de transición admite tres clases fundamentales de perturbaciones macroeconómicas imprevistas:

1. **Choque de Productividad Total de los Factores (`shock_var="z"`)**:
   $$Z_t = 1.0 + \Delta Z \cdot \rho_Z^t$$
   Permite modelar perturbaciones transitorias ($\rho_Z < 1$) o permanentes ($\rho_Z = 1.0$).
2. **Choque monetario / cuña de tipo de interés (`shock_var="r"`)**:
   $$r_t^{\text{eff}} = r_t + \Delta r \cdot \rho_r^t$$
   Simula endurecimiento de la política monetaria o ensanchamiento de diferenciales de crédito.
3. **Choque en el factor de descuento / paciencia (`shock_var="beta"`)**:
   $$\beta_t = \beta_{\text{ss}} + \Delta \beta \cdot \rho_\beta^t$$
   Modela episodios de preferencia por la liquidez y aumentos repentinos del ahorro precautorio.

---

## 4. Ejemplos prácticos ejecutables

El siguiente script calcula la trayectoria exacta de transición no lineal tras una innovación imprevista del $+5\%$ en la PTF en una economía de Aiyagari:

```python
import numpy as np
from puremacro.vfi import (
    solve_aiyagari_continuous,
    solve_continuous_transition,
    continuous_mit_shock,
    TransitionShock,
)
from puremacro.vfi.aggregate import lorenz_and_gini

# 1. Cálculo del equilibrio general estacionario inicial
init_ss = solve_aiyagari_continuous(
    beta=0.96,
    gamma=2.0,
    alpha=0.36,
    delta=0.08,
    N_k=100,
    n_z=3,
    max_evals=20,
)

# 2. Simulación del choque MIT no lineal (+5% PTF con persistencia 0.75)
T_sim = 40
res_trans = continuous_mit_shock(
    steady_state=init_ss,
    shock_type="tfp",
    shock_size=0.05,
    persistence=0.75,
    horizon=T_sim,
    solver="broyden",
    max_iter=30,
)

assert res_trans.converged
assert len(res_trans.r_path) == T_sim
assert res_trans.mass_conservation_error < 1e-12

# 3. Métricas dinámicas de desigualdad a lo largo de la transición
k_grid = init_ss.distribution.asset_grid
mu_0 = np.sum(res_trans.distributions[0], axis=1)
mu_T = np.sum(res_trans.distributions[-1], axis=1)

_, _, gini_0 = lorenz_and_gini(k_grid, mu_0)
_, _, gini_T = lorenz_and_gini(k_grid, mu_T)

summary_df = res_trans.summary()
print(f"Transición convergida en {res_trans.iterations} iters, residuo máx: {res_trans.max_residual:.2e}")
print(f"Gini de riqueza: t=0 -> {gini_0:.4f}, t={T_sim} -> {gini_T:.4f}")
```

---

## 5. Especificación completa de la API

```text
TransitionShock(
    path: np.ndarray,
    var: str = "z",
)

solve_continuous_transition(
    initial_steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    terminal_steady_state: AiyagariContinuousEquilibrium | dict[str, Any] | None = None,
    shock_path: np.ndarray | Sequence[float] | None = None,
    shock_var: str = "z",
    horizon: int = 150,
    solver: str = "broyden",
    damping: float = 0.3,
    tol: float = 1e-4,
    max_iter: int = 100,
    backtracking: bool = True,
    backend: str = "numpy",
    r_init_path: np.ndarray | Sequence[float] | None = None,
    **kwargs: Any,
) -> ContinuousTransitionResult

continuous_mit_shock(
    steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    shock_type: str = "tfp",
    shock_size: float = 0.05,
    persistence: float = 0.8,
    horizon: int = 150,
    solver: str = "broyden",
    **kwargs: Any,
) -> ContinuousTransitionResult
```

#### Parámetros:
- `initial_steady_state`: Objeto de equilibrio general precalculado `AiyagariContinuousEquilibrium` o diccionario de configuración.
- `terminal_steady_state`: Equilibrio final de destino. Para choques transitorios se asume idéntico al inicial. Para choques permanentes se calcula automáticamente en los valores finales del choque.
- `shock_path`: Vector unidimensional con la trayectoria del choque de longitud $T$.
- `shock_var` / `shock_type`: Variable objetivo: `'tfp'` / `'z'`, `'rate'` / `'r'`, o `'beta'` / `'discount'`.
- `horizon`: Longitud temporal del horizonte de simulación $T$ (por defecto $150$).
- `solver`: Solucionador de vaciado: `'broyden'` (Cuasi-Newton) o `'shooting'` (relajación con amortiguación).
- `damping`: Factor de amortiguación para disparo o retroceso en búsqueda lineal.
- `tol`: Tolerancia de vaciado en el mercado de capitales $\|K^s - K^d\|_\infty$ (por defecto $10^{-4}$).

---

## 6. Interfaz de resultados y métricas dinámicas de desigualdad

`ContinuousTransitionResult` almacena la trayectoria dinámica completa y las métricas de desigualdad asociadas:

### Atributos del contenedor
- `r_path`: Trayectoria del tipo de interés real $r_0, \dots, r_{T-1}$.
- `w_path`: Trayectoria del salario real competitivo $w_0, \dots, w_{T-1}$.
- `K_s_path`: Secuencia de oferta agregada de capital $K_t^s = \int k \, d\mu_t$.
- `K_d_path`: Secuencia de demanda agregada de capital $K_t^d(r_t)$.
- `C_path`: Secuencia de consumo agregado $C_t = \int c_t \, d\mu_t$.
- `distributions`: Lista de $T+1$ distribuciones transversales conjuntas $\mu_0, \mu_1, \dots, \mu_T$.
- `residuals`: Trayectoria de excesos de demanda de capital $H_t = K_t^s - K_t^d$.
- `max_residual`: Residuo máximo de vaciado $\|H\|_\infty$.
- `converged`: Indicador booleano de convergencia exitosa.
- `mass_conservation_error`: Desviación absoluta máxima de masa a lo largo de los $T+1$ períodos.

### Métodos de serialización y gráficos
- `res.summary() -> pd.DataFrame`: Resumen de diagnósticos de convergencia, tiempo de cálculo y agregados iniciales/finales.
- `res.to_frame() -> pd.DataFrame`: DataFrame con las series temporales completas del espacio de secuencias ($r_t, w_t, K_t^s, K_t^d, C_t, H_t$).
- `res.to_markdown(digits=4) -> str`: Tabla en formato Markdown.
- `res.to_latex(digits=4) -> str`: Entorno tabular en LaTeX de calidad para publicaciones académicas.
- `res.to_typst(digits=4) -> str`: Código Typst formateado para informes científicos.
- `res.plot(figsize=(14, 8)) -> matplotlib.figure.Figure`: Gráfico diagnóstico multipanel que visualiza tipos de interés, salarios, vaciado de capital, consumo agregado y la evolución de la distribución de riqueza $\mu_t(k)$.

---

## Referencias

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica*, 89(6), 3115–3148.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
