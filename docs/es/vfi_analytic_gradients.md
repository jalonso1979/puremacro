> 🇬🇧 [English](../vfi_analytic_gradients.md) · 🇪🇸 Español

# Gradientes paramétricos vía el Teorema de la Función Implícita

`puremacro.vfi.analytic_gradients` proporciona jacobianos de soluciones de proyección continua con agente representativo, y de los agregados de estado estacionario determinista que implican, respecto a los parámetros estructurales $\theta = (\beta, \alpha, \delta, \sigma, \dots)$ mediante el **Teorema de la Función Implícita (TFI)**. Opera de manera uniforme sobre colocación polinómica ortogonal de Chebyshev, proyecciones de Galerkin del Método de Elementos Finitos (MEF) y sistemas de splines cúbicos o de Schumaker.

**Alcance (4.3.0 y posteriores).** El paso TFI es una única resolución lineal con el jacobiano de residuos convergido (sin re-resolver el modelo); los jacobianos de residuos $\mathbf{J}_c$ y $\mathbf{J}_\theta$ que consume se construyen por diferencias finitas centrales con pasos `step_c` y `h` (una `residual_fn` propia sustituye al evaluador de residuos incorporado y se diferencia del mismo modo). Solo se admiten soluciones con agente representativo: `compute_ift_gradients` y `equilibrium_parameter_jacobian` lanzan `NotImplementedError` para cualquier solución que tenga un atributo `distribution` (por ejemplo `AiyagariContinuousEquilibrium`). La sensibilidad adjunta de la distribución estacionaria descrita en la sección 1.3 no está implementada.

En la estimación econométrica estructural (GMM, SMM) y el muestreo bayesiano posterior (HMC, NUTS), el cálculo de gradientes en modelos económicos dinámicos ha dependido históricamente de diferencias finitas numéricas. Sin embargo, las diferencias finitas sufren de un severo dilema de tamaño de paso ($\epsilon \sim 10^{-5}$ para equilibrar el error de truncamiento y el de redondeo), exhiben ruido numérico espurio cerca de restricciones financieras no lineales y requieren resolver de nuevo el modelo no lineal completo $2 \times \dim(\theta)$ veces. El motor de gradientes analíticos de `puremacro` elimina las $2 \times \dim(\theta)$ re-resoluciones calculando las sensibilidades de los coeficientes a partir de una **única factorización LU** del jacobiano de residuos convergido. La batería de pruebas del repositorio exige una aceleración de al menos 5x frente a diferencias finitas centrales en el ejemplo de la sección 4; en bases mayores son habituales cocientes más altos, pero los tiempos citados en la sección 3 son indicativos, no un contrato.

---

## 1. Fundamentos teóricos y matemáticos

### 1.1 El Teorema de la Función Implícita en sistemas de proyección continua

Considérese un problema de programación dinámica continua aproximado mediante una base $N$-dimensional (polinomios de Chebyshev, elementos finitos lineales a trozos o B-splines cúbicos). La regla de política de equilibrio queda parametrizada por un vector de coeficientes $c^* \in \mathbb{R}^N$ que satisface un sistema de ecuaciones residuales no lineales evaluado en los nodos de colocación o cuadratura:

$$\mathbf{R}(c^*(\theta); \theta) = \mathbf{0} \in \mathbb{R}^N$$

donde $\theta \in \mathbb{R}^p$ es el vector de parámetros estructurales (tales como el factor de descuento $\beta$, la participación del capital $\alpha$ o la tasa de depreciación $\delta$).

Asumiendo que el operador residual $\mathbf{R}$ es continuamente diferenciable respecto a $c$ y $\theta$, la derivada total de la identidad de equilibrio respecto a $\theta$ es:

$$\frac{d \mathbf{R}(c^*(\theta); \theta)}{d \theta} = \nabla_c \mathbf{R}(c^*; \theta) \cdot \nabla_\theta c^*(\theta) + \nabla_\theta \mathbf{R}(c^*; \theta) = \mathbf{0}$$

Siempre que el jacobiano respecto a los coeficientes $\mathbf{J}_c \equiv \nabla_c \mathbf{R}(c^*; \theta) \in \mathbb{R}^{N \times N}$ sea no singular en la solución convergida $c^*$, el **Teorema de la Función Implícita** garantiza que $c^*(\theta)$ es localmente diferenciable, con matriz jacobiana respecto a los parámetros:

$$\nabla_\theta c^*(\theta) = - \left[ \nabla_c \mathbf{R}(c^*; \theta) \right]^{-1} \nabla_\theta \mathbf{R}(c^*; \theta)$$

donde:
- $\mathbf{J}_c = \nabla_c \mathbf{R} \in \mathbb{R}^{N \times N}$ es el jacobiano del sistema residual respecto a los coeficientes de la base.
- $\mathbf{J}_\theta = \nabla_\theta \mathbf{R} \in \mathbb{R}^{N \times p}$ es la sensibilidad directa de las ecuaciones residuales respecto a los parámetros.

### 1.2 Factorización LU única frente a diferencias finitas

En lugar de resolver de nuevo el problema de programación dinámica no lineal $2p$ veces (como exigen las diferencias finitas centrales), el TFI calcula la matriz completa de sensibilidades de coeficientes $N \times p$ resolviendo un sistema lineal con $p$ términos independientes:

$$\mathbf{J}_c \cdot \mathbf{X} = - \mathbf{J}_\theta$$

Mediante la factorización $\mathbf{J}_c = \mathbf{P} \mathbf{L} \mathbf{U}$:
1. **Factorización**: La matriz $\mathbf{J}_c$ de dimensión $N \times N$ se factoriza una sola vez en $\frac{2}{3} N^3$ operaciones de punto flotante.
2. **Sustitución**: Para cada parámetro $\theta_j$ ($j = 1, \dots, p$), la columna de sensibilidades $\nabla_{\theta_j} c^*$ se obtiene mediante sustitución progresiva y regresiva en $2 N^2$ operaciones.

Esto reduce la complejidad computacional de $O(2p \cdot N_{\text{iter}} \cdot N^3)$ requerida por re-resolución no lineal a $O(N^3 + p N^2)$, acelerando el cálculo del gradiente en 1 o 2 órdenes de magnitud.

### 1.3 Sensibilidad de reglas de política y jacobianos agregados

Dado que las aproximaciones de proyección continua expresan las funciones de política como combinaciones lineales de funciones de base $g(k; c) = \sum_{j=1}^N c_j \phi_j(k) = \mathbf{\Phi}(k) c$, el gradiente continuo de la política en cualquier estado $k$ se obtiene analíticamente:

$$\nabla_\theta g(k; \theta) = \mathbf{\Phi}(k) \cdot \nabla_\theta c^*(\theta) \in \mathbb{R}^{1 \times p}$$

#### Agregados macroeconómicos de equilibrio general

En una economía de agente representativo con proyección continua, el stock de capital de estado estacionario determinista $k^*$ satisface la condición de punto fijo $g(k^*; \theta) = k^*$. Diferenciando implícitamente:

$$\nabla_\theta k^* = \left( 1 - \frac{\partial g(k^*)}{\partial k} \right)^{-1} \nabla_\theta g(k^*)$$

Los precios factoriales competitivos de equilibrio $r^* = \alpha z (k^*)^{\alpha - 1} - \delta$ y $w^* = (1 - \alpha) z (k^*)^\alpha$ tienen sensibilidades:

$$\nabla_\theta r^* = \frac{\partial r^*}{\partial k} \nabla_\theta k^* + \left. \nabla_\theta r^* \right|_{\text{direct}}, \quad \nabla_\theta w^* = \frac{\partial w^*}{\partial k} \nabla_\theta k^* + \left. \nabla_\theta w^* \right|_{\text{direct}}$$

Estas sensibilidades de estado estacionario con agente representativo son las que devuelve `grad_aggregates` (claves `K`, `C`, `r`, `w`). Son semianalíticas: el punto fijo del estado estacionario se localiza numéricamente y su derivada se propaga después de forma analítica mediante las fórmulas anteriores.

**No implementado: economías con agentes heterogéneos.** En economías de mercados incompletos (Aiyagari 1994) la oferta agregada de capital es $K^* = \int k \, d\mu^*(k; \theta)$ y su sensibilidad tendría que propagarse a través del adjunto del operador de la distribución estacionaria (lotería),

$$\nabla_\theta K^* = \sum_{i, m} k_i \nabla_\theta \mu^*(k_i, z_m) = \mathbf{k}^\top (\mathbf{I} - \mathbf{T}^*)^{-1} \nabla_\theta \mathbf{T}^* \boldsymbol{\mu}^*$$

junto con la respuesta de los precios de equilibrio general. puremacro 4.3.0 retiró el atajo anterior de distribución fija porque no era una sensibilidad de equilibrio general; desde entonces `compute_ift_gradients`, `equilibrium_parameter_jacobian` y la rutina interna de agregados lanzan `NotImplementedError` para cualquier solución con atributo `distribution`. La fórmula se recoge aquí como objetivo de trabajo futuro, no como una funcionalidad disponible.

### 1.4 Estimación estructural acelerada (GMM y SMM)

La estimación estructural busca el vector de parámetros $\hat{\theta}$ que minimiza la distancia ponderada entre los momentos simulados del modelo $m(\theta)$ y los momentos empíricos muestrales $\hat{m}$:

$$Q(\theta) = \left( m(\theta) - \hat{m} \right)^\top \mathbf{W} \left( m(\theta) - \hat{m} \right)$$

Aplicando la regla de la cadena, el gradiente de la función objetivo de GMM es:

$$\nabla_\theta Q(\theta) = 2 \left[ \nabla_\theta m(\theta) \right]^\top \mathbf{W} \left( m(\theta) - \hat{m} \right)$$

donde $\nabla_\theta m(\theta) = \nabla_c m(c^*) \nabla_\theta c^* + \partial_\theta m$. Dado que $\nabla_\theta c^*$ procede de una única resolución lineal del sistema convergido y no de $2p$ re-resoluciones ruidosas, los optimizadores cuasi-Newton basados en gradiente (L-BFGS-B, SLSQP) convergen de forma robusta en una fracción del tiempo requerido bajo diferencias finitas.

---

## 2. Opciones metodológicas y del solucionador

| Característica | LU única (`solver="lu"`) | Regularización de Tikhonov (`solver="tikhonov"`) | SVD truncada (`solver="svd"`) |
|---|---|---|---|
| **Base matemática** | $\mathbf{P} \mathbf{L} \mathbf{U} = \mathbf{J}_c$ | $(\mathbf{J}_c^\top \mathbf{J}_c + \lambda \mathbf{I})^{-1} \mathbf{J}_c^\top$ | $\mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^\top$ con umbral $\sigma_i > \epsilon \sigma_1$ |
| **Umbral de condición** | $\text{cond}(\mathbf{J}_c) \le 10^{12}$ | $10^{12} < \text{cond}(\mathbf{J}_c) \le 10^{15}$ | Sistemas singulares o mal condicionados |
| **Precisión** | Resolución LU estable hacia atrás (error del orden de $\text{cond}(\mathbf{J}_c) \cdot \epsilon_{\text{máq}}$) sumada al error de las diferencias centrales de $\mathbf{J}_c$ y $\mathbf{J}_\theta$ | Gradiente amortiguado regularizado | Gradiente proyectado en subespacio filtrado |
| **Costo** | $\frac{2}{3} N^3$ operaciones | $O(N^3)$ operaciones | $O(N^3)$ (espectro singular completo) |
| **Selección** | Predeterminado automático (`"auto"`) | Activado en bases mal condicionadas | Polinomios colineales de alto grado |

---

## 3. Comparativa de rendimiento y análisis de precisión

| Dimensión | Gradientes TFI (`puremacro`) | Diferencias finitas centrales |
|---|---|---|
| **Precisión del gradiente** | Resolución TFI estable hacia atrás dados $\mathbf{J}_c$, $\mathbf{J}_\theta$ (a su vez diferencias centrales con pasos `step_c`, `h`); la prueba incluida comprueba la coincidencia con una re-resolución por diferencias finitas a $10^{-5}$ relativo | Error de discretización $O(\epsilon^2) \approx 10^{-5}$ más el ruido de la tolerancia de re-resolución |
| **Dilema del tamaño de paso** | Reducido: los pasos `step_c`, `h` diferencian el residuo en los coeficientes convergidos fijos, sin re-resolver dentro de la diferencia | Requiere calibración sensible ($\epsilon = 10^{-4}$ frente a $10^{-6}$) |
| **Evaluaciones del modelo** | $1$ (estado estacionario convergido) | $2 \times p$ re-resoluciones no lineales completas |
| **Tiempo de cálculo ($p = 4$)** | una construcción del jacobiano y una resolución LU (indicativo: milisegundos en el ejemplo de la sección 4) | $2p$ re-resoluciones no lineales (la prueba incluida exige que la vía TFI sea al menos 5x más rápida) |
| **Ruido numérico** | Sin ruido de la tolerancia de re-resolución (el residuo se diferencia con coeficientes fijos); persiste el redondeo de las diferencias del residuo | Severo ruido de redondeo cerca de restricciones |
| **Estabilidad del optimizador** | Actualizaciones Hessian robustas (BFGS) | Inversiones espurias del gradiente provocan parada prematura |

---

## 4. Ejemplos prácticos ejecutables

El siguiente script resuelve un modelo de crecimiento neoclásico mediante colocación de Chebyshev, calcula las sensibilidades de los parámetros mediante el TFI y evalúa las derivadas de los agregados macroeconómicos:

```python
import numpy as np
from puremacro.vfi.collocation import CollocationProblem
from puremacro.vfi.analytic_gradients import (
    compute_ift_gradients,
    policy_parameter_jacobian,
    equilibrium_parameter_jacobian,
)

# 1. Resolución del problema de proyección continua (Colocación de Chebyshev)
alpha, beta, delta, sigma = 0.36, 0.96, 1.0, 1.0
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))

def euler_res(k, kp, kpp, s, sp, beta=beta):
    c = k**alpha - kp
    c_next = kp**alpha - kpp
    R = alpha * kp**(alpha - 1.0)
    return 1.0 / c - beta * (1.0 / c_next) * R

prob = CollocationProblem(
    domain=(0.5 * k_ss, 1.5 * k_ss),
    orders=6,
    method="euler",
    params={"alpha": alpha, "delta": delta, "sigma": sigma},
    beta=beta,
    options={"tol": 1e-12, "max_iter": 500},
)
sol = prob.solve(backend="numpy")

# 2. Jacobianos TFI (J_c, J_theta por diferencias centrales; una resolución LU)
ift_res = compute_ift_gradients(sol, prob, params=["alpha", "beta", "delta"])
assert ift_res.condition_number < 1e6
assert ift_res.grad_coefficients.shape == (7, 3)

# 3. Gradiente continuo de la política en el estado estacionario
d_pol = ift_res.policy_gradient(k_ss)
assert d_pol[1] > 0.0  # d(k') / d(beta) > 0: hogares más pacientes acumulan más capital

# 4. Sensibilidades de agregados macroeconómicos
grad_aggs = ift_res.grad_aggregates
print(f"Número de condición cond(J_c): {ift_res.condition_number:.2e}")
print(f"d(K*)/d(beta): {grad_aggs['K'][1]:.4f}, d(r*)/d(beta): {grad_aggs['r'][1]:.4f}")
```

---

## 5. Especificación completa de la API

```text
compute_ift_gradients(
    solution: Any,
    problem: Any,
    params: Sequence[str] | None = None,
    h: float = 1e-5,
    step_c: float = 1e-6,
    residual_fn: Callable | None = None,
    cond_max: float = 1e12,
    backend: str = "numpy",
    **kwargs: Any,
) -> AnalyticGradientResult

policy_parameter_jacobian(
    solution: Any,
    problem: Any,
    s: float | Sequence[float] | np.ndarray,
    params: Sequence[str] | None = None,
    h: float = 1e-5,
    ift_result: AnalyticGradientResult | None = None,
    **kwargs: Any,
) -> np.ndarray

equilibrium_parameter_jacobian(
    solution: Any,
    problem: Any | None = None,
    params: Sequence[str] | None = None,
    h: float = 1e-5,
    ift_result: AnalyticGradientResult | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]

gmm_objective_and_gradient(
    theta_vals: Sequence[float],
    problem: Any,
    empirical_moments: Sequence[float],
    moment_fn: Callable | None = None,
    weighting_matrix: np.ndarray | None = None,
    param_names: Sequence[str] | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> tuple[float, np.ndarray]
```

#### Parámetros:
- `solution`: Contenedor de solución convergida con agente representativo procedente de `CollocationProblem.solve()`, `FEMProblem.solve()` o `SplineCollocationProblem.solve()`. Las soluciones con agentes heterogéneos (cualquier objeto con atributo `distribution`, como `AiyagariContinuousEquilibrium`) lanzan `NotImplementedError`.
- `problem`: Definición del modelo económico asociado.
- `params`: Secuencia de nombres de parámetros a diferenciar (p. ej. `['alpha', 'beta', 'delta']`). Si es `None`, inspecciona automáticamente los parámetros del modelo.
- `h`: Tamaño de paso para la diferenciación paramétrica $\nabla_\theta \mathbf{R}$ (por defecto $10^{-5}$).
- `step_c`: Tamaño de paso para el jacobiano de coeficientes $\mathbf{J}_c = \nabla_c \mathbf{R}$ (por defecto $10^{-6}$).
- `cond_max`: Umbral del número de condición para activar solucionadores regularizados (por defecto $10^{12}$).
- `s`: Coordenadas continuas de estado donde evaluar $\nabla_\theta g(s)$.
- `theta_vals`: Vector de parámetros candidatos en estimación estructural.
- `empirical_moments`: Vector de momentos empíricos observados $\hat{m}$.

---

## 6. Interfaz de resultados y exportación a publicaciones

`AnalyticGradientResult` almacena las matrices completas de sensibilidad y los diagnósticos numéricos:

### Atributos
- `grad_coefficients`: Matriz jacobiana $N \times p$ de coeficientes de la base $\nabla_\theta c^*$.
- `grad_aggregates`: Diccionario de derivadas de agregados de estado estacionario con agente representativo `{"K": (p,), "C": (p,), "r": (p,), "w": (p,), "mu": None}` (semianalíticas, sección 1.3). La entrada `"mu"` es siempre `None`: no se calcula ninguna sensibilidad de la distribución estacionaria.
- `param_names`: Lista ordenada de los nombres de los parámetros diferenciados.
- `jacobian_resid_c`: Jacobiano respecto a los coeficientes $\mathbf{J}_c = \nabla_c \mathbf{R}$ de forma $(N, N)$.
- `jacobian_resid_theta`: Jacobiano respecto a los parámetros $\mathbf{J}_\theta = \nabla_\theta \mathbf{R}$ de forma $(N, p)$.
- `condition_number`: Número de condición en norma 2 de $\mathbf{J}_c$.
- `elapsed_time`: Tiempo de ejecución en segundos.

### Métodos
- `res.policy_gradient(s) -> np.ndarray`: Evalúa la sensibilidad continua de la política $\nabla_\theta g(s)$ en coordenadas escalares o vectoriales $s$.
- `res.summary() -> pd.DataFrame`: Cuadro resumen con las sensibilidades paramétricas de capital, consumo, tipo de interés y salarios.
- `res.to_frame() -> pd.DataFrame`: Alias que devuelve el DataFrame del cuadro resumen.
- `res.to_markdown(digits=4) -> str`: Tabla en formato Markdown compatible con GitHub.
- `res.to_latex(digits=4) -> str`: Entorno tabular en LaTeX apto para publicaciones académicas.
- `res.to_typst(digits=4) -> str`: Tabla en formato Typst para informes científicos modernos.
- `res.plot(figsize=(10, 4.5)) -> matplotlib.figure.Figure`: Gráfico multipanel con el espectro de valores singulares de $\mathbf{J}_c$, los gradientes continuos de política a lo largo del espacio de estados y las sensibilidades agregadas.

---

## Referencias

- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.
- Maliar, L., & Maliar, S. (2014). "Numerical methods for large-scale dynamic economic models." En *Handbook of Computational Economics* (Vol. 3, pp. 325–477). Elsevier.
- Rust, J. (1994). "Structural estimation of Markov decision processes." En *Handbook of Econometrics* (Vol. 4, pp. 3081–3143). North-Holland.
