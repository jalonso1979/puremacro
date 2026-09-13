> 🇬🇧 [English](../vfi_splines_and_sparse_grids.md) · 🇪🇸 Español

# Splines con preservación de forma y mallas dispersas de Smolyak

Los módulos `puremacro.vfi.splines` y `puremacro.vfi.smolyak` implementan métodos de frontera para la programación dinámica continua aplicados a la modelización de curvaturas locales y modelos multidimensionales:

- **B-splines cúbicos** (**De Boor 1978**) proporcionan aproximaciones suaves de clase $C^2$ con condiciones de frontera flexibles sobre los nudos (anclados, naturales o *not-a-knot*).
- **Splines cuadráticos con preservación de forma de Schumaker (1983)** garantizan estrictamente la monotonicidad ($\partial g / \partial k \ge 0$) y la concavidad ($\partial^2 g / \partial k^2 \le 0$) mediante la inserción adaptativa de nudos secundarios, erradicando oscilaciones espurias y comportamientos anómalos cerca de restricciones de endeudamiento o cambios bruscos de curvatura.
- **Colocación sobre mallas dispersas de Smolyak (1963)** (**Krueger y Kubler 2004; Malin, Krueger y Kubler 2011**) resuelve modelos macroeconómicos dinámicos con múltiples variables de estado continuas ($d \in [2, 6]$, tales como modelos con múltiples bienes de capital, carteras de activos o economía abierta). Al combinar nodos extremos anidados de Clenshaw-Curtis con polinomios de base de Chebyshev, las mallas de Smolyak escalan con una complejidad de orden $\mathcal{O}(N (\log N)^{d-1})$ frente a la explosión exponencial del producto tensorial completo $\mathcal{O}(N^d)$, reduciendo el número de nodos hasta en dos órdenes de magnitud sin comprometer la precisión.

---

## 1. Fundamentos teóricos y matemáticos

### 1.1 B-splines cúbicos univariados

Los B-splines (splines de base) representan funciones continuas sobre un intervalo $[a, b]$ como combinaciones lineales de funciones base polinómicas por tramos con soporte local. Dado un vector de nudos no decreciente:

$$\tau_0 \le \tau_1 \le \dots \le \tau_{K+3}$$

la $i$-ésima función base B-spline $B_{i, p}(x)$ de grado $p$ se define de forma recursiva mediante la **fórmula de recurrencia de Cox-De Boor**:

$$B_{i, 0}(x) = \begin{cases} 1 & \text{si } \tau_i \le x < \tau_{i+1} \\ 0 & \text{en otro caso} \end{cases}$$

$$B_{i, p}(x) = \frac{x - \tau_i}{\tau_{i+p} - \tau_i} B_{i, p-1}(x) + \frac{\tau_{i+p+1} - x}{\tau_{i+p+1} - \tau_{i+1}} B_{i+1, p-1}(x)$$

Para los B-splines cúbicos ($p = 3$):
- Cada función base $B_{i, 3}(x)$ es dos veces continuamente diferenciable ($C^2$) a través de los nudos interiores.
- Posee soporte compacto local que abarca únicamente cuatro intervalos de nudos $[\tau_i, \tau_{i+4}]$, asegurando que las perturbaciones locales no generen ondulaciones globales.
- Constituyen una partición de la unidad: $\sum_{i=0}^{K-1} B_{i, 3}(x) = 1.0$ para todo $x \in [a, b]$.

#### Condiciones de frontera y nodos de colocación

`CubicBSplineBasis` (y su alias `SplineBasis`) admite tres configuraciones de nudos en las fronteras:
1. **Anclados (`bc_type="clamped"`)**: Nudos extremos con multiplicidad 4 ($\tau_0 = \tau_1 = \tau_2 = \tau_3 = a$ y $\tau_K = \dots = \tau_{K+3} = b$). Las funciones primera y última alcanzan valores unitarios exactos $B_0(a) = 1$ y $B_{K-1}(b) = 1$, idóneo para condiciones de frontera fijas.
2. **Naturales (`bc_type="natural"`)**: Impone segundas derivadas nulas en los extremos: $S''(a) = S''(b) = 0$.
3. ***Not-a-knot* (`bc_type="not-a-knot"`)**: Exige continuidad de la tercera derivada $S'''$ a través del primer y último nudo interior.

Las ecuaciones de colocación se evalúan en las **abscisas de Greville** (medias locales de nudos consecutivos):

$$\xi_i^* = \frac{\tau_{i+1} + \tau_{i+2} + \tau_{i+3}}{3}, \quad i = 0, \dots, K-1$$

---

### 1.2 Splines cuadráticos con preservación de forma de Schumaker (1983)

Los splines cúbicos convencionales pueden inducir oscilaciones espurias no monótonas al interpolar datos con cambios abruptos en la curvatura, tales como funciones de valor en las proximidades de límites de endeudamiento. La teoría económica exige reglas de decisión estrictamente monótonas ($\partial k' / \partial k \ge 0$) y funciones de valor estrictamente cóncavas ($\partial^2 V / \partial k^2 \le 0$).

El algoritmo de **Schumaker (1983)** construye un spline cuadrático de clase $C^1$ que preserva simultáneamente la monotonicidad y la concavidad de los datos.

#### Mecanismo algorítmico

Dados los nodos $x_1 < x_2 < \dots < x_N$ y valores $y_1, y_2, \dots, y_N$:
1. **Estimación de pendientes**: Calcula las pendientes secantes de cada intervalo $\delta_i = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}$. Asigna derivadas nodales $d_i$ consistentes con la monotonicidad ($d_i \ge 0$ si $\delta_{i-1} \ge 0$ y $\delta_i \ge 0$).
2. **Verificación de forma**: En cada intervalo $[x_i, x_{i+1}]$, evalúa si un polinomio cuadrático simple con derivadas $d_i, d_{i+1}$ violaría la monotonicidad o convexidad:
   - Si $(d_i + d_{i+1})\delta_i \ge 0$ y $|d_i - \delta_i| \le |d_{i+1} - \delta_i|$ (o viceversa), un único tramo cuadrático no puede respetar la forma y satisfacer las derivadas simultáneamente.
3. **Inserción adaptativa de nudos secundarios**: En los intervalos en conflicto, Schumaker introduce un nudo interior $\xi_i \in (x_i, x_{i+1})$:
   $$\xi_i = x_i + \frac{2(y_{i+1} - y_i) - 2 d_i (x_{i+1} - x_i)}{d_{i+1} - d_i}$$
   y ajusta dos parábolas conectadas con suavidad $C^1$ en $[x_i, \xi_i]$ y $[\xi_i, x_{i+1}]$.
4. **Propiedades demostradas**:
   - **Monotonicidad**: Si $y_{i+1} \ge y_i$ para todo $i$, entonces $s'(x) \ge 0$ para todo $x \in [x_1, x_N]$.
   - **Concavidad**: Si las pendientes secantes $\delta_i$ son no crecientes, entonces $s''(x) \le 0$ para todo $x \in [x_1, x_N]$.
   - **Cero oscilaciones espurias**: El interpolante queda acotado estrictamente por los extremos locales: $\min(y_i, y_{i+1}) \le s(x) \le \max(y_i, y_{i+1})$.

---

### 1.3 Colocación sobre mallas dispersas de Smolyak (1963)

En modelos dinámicos con $d$ variables de estado continuas (como capital físico $k_t$, capital humano $h_t$ y deuda externa $b_t$), las mallas basadas en productos tensoriales sufren la maldición de la dimensionalidad: una malla de solo 20 puntos por dimensión requiere $20^d$ nodos ($8.000$ para $d=3$, $6.4 \times 10^7$ para $d=6$).

El **método de mallas dispersas de Smolyak** resuelve este cuello de botella interpolando sobre subconjuntos optimizados de productos tensoriales unidimensionales.

#### 1. Nodos extremos anidados de Clenshaw-Curtis

En cada dimensión, la interpolación unidimensional utiliza nodos extremos de Clenshaw-Curtis en $[-1, 1]$:

$$x_j^i = -\cos\left(\frac{(j - 1)\pi}{m(i) - 1}\right), \quad j = 1, \dots, m(i)$$

con la regla exponencial de crecimiento de nodos:

$$m(1) = 1, \qquad m(i) = 2^{i-1} + 1 \quad \text{para } i \ge 2$$

Al duplicarse $m(i)$ en cada nivel, los conjuntos de nodos están estrictamente anidados: $X^{(1)} \subset X^{(2)} \subset X^{(3)} \subset \dots$.

#### 2. Fórmula de combinación de Smolyak

Sea $\mathcal{U}^i$ el operador de interpolación 1D de nivel $i$. Para una dimensión $d$ y nivel de aproximación $\mu \in [1, 5]$, el operador de malla dispersa $\mathcal{A}(d, \mu)$ se define como:

$$\mathcal{A}(d, \mu) = \sum_{d \le |\mathbf{i}|_1 \le d + \mu} (-1)^{d + \mu - |\mathbf{i}|_1} \binom{d - 1}{d + \mu - |\mathbf{i}|_1} \left( \mathcal{U}^{i_1} \otimes \dots \otimes \mathcal{U}^{i_d} \right)$$

donde $\mathbf{i} = (i_1, \dots, i_d)$ es un multi-índice con $i_k \ge 1$ y $|\mathbf{i}|_1 = \sum_{k=1}^d i_k$.

#### 3. Reducción en la escala de nodos

En lugar de evaluar todos los $m(\mu + 1)^d$ puntos del producto tensorial completo, la malla de Smolyak conserva únicamente aquellos puntos cuyo multi-índice satisface $|\mathbf{i}|_1 \le d + \mu$:

| Dimensión $d$ | Nivel $\mu$ | Nodos de Smolyak $N(d, \mu)$ | Nodos tensoriales $m(\mu+1)^d$ | Factor de reducción |
|---|---|---|---|---|
| **$d = 2$** | $\mu = 2$ | **13** | 25 | $1.9\times$ |
| **$d = 2$** | $\mu = 3$ | **29** | 81 | $2.8\times$ |
| **$d = 3$** | $\mu = 2$ | **25** | 125 | **$5.0\times$** |
| **$d = 3$** | $\mu = 3$ | **69** | 729 | **$10.6\times$** |
| **$d = 4$** | $\mu = 2$ | **41** | 625 | **$15.2\times$** |
| **$d = 5$** | $\mu = 3$ | **241** | 59.049 | **$245\times$** |
| **$d = 6$** | $\mu = 3$ | **389** | 531.441 | **$1.366\times$** |

---

## 2. Opciones metodológicas y de modelos

| Configuración | `SplineCollocationProblem` | `SmolyakProblem` |
|---|---|---|
| **Base de aproximación** | B-splines cúbicos (`"cubic"`) o Schumaker (`"schumaker"`) | Polinomios de Chebyshev en malla dispersa multidimensional |
| **Dimensión continua** | $d = 1$ (univariado) | $d \in [2, 6]$ (multivariado) |
| **Preservación de forma** | Garantizada con `spline_type="schumaker"` | Alta precisión polinómica en hipercubos continuos |
| **Parámetros de malla** | `n_knots` (def. 20), `bc_type="clamped"` | `mu` en $[1, 4]$ (def. 2) |
| **Método de solución** | `"euler"` (política) o `"bellman"` (valor) | `"euler"` (política) o `"bellman"` (valor) |
| **Aceleración disponible** | NumPy, Numba, Apple MLX, CuPy | NumPy, Numba, Apple MLX, CuPy |

---

## 3. Calibración canónica y especificaciones económicas

### 3.1 Modelo neoclásico 1D (Splines)
- $u(c) = \ln(c)$, $f(k) = k^\alpha$, $\alpha = 0.36$, $\beta = 0.96$, $\delta = 1.0$.
- Estado estacionario: $k_{\text{ss}} = (\alpha \beta)^{\frac{1}{1-\alpha}} \approx 0.1782$.
- Dominio: $[0.5 k_{\text{ss}}, 1.5 k_{\text{ss}}]$.

### 3.2 Modelo neoclásico con dos capitales 2D (Smolyak)
- Dos tipos de capital $(k_1, k_2)$ (por ejemplo, equipo y estructuras):
  $$Y = z k_1^{\alpha_1} k_2^{\alpha_2}, \qquad u(c) = \ln(c)$$
  con $\alpha_1 = 0.18, \alpha_2 = 0.18, \beta = 0.96$.
- Política analítica exacta: $k'_m = \alpha_m \beta Y$ para $m \in \{1, 2\}$.

---

## 4. Ejemplos de uso ejecutables

### 4.1 Verificación de la preservación de forma del spline de Schumaker

Este ejemplo comprueba el spline de Schumaker sobre una función cóncava no diferenciable $f(x) = \min(2x, 1 + 0.5x)$ en la que los splines cúbicos convencionales producen sobreoscilaciones:

```python
import numpy as np
from puremacro.vfi import SchumakerSpline

# Nodos de una función cóncava con codo
x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
y_nodes = np.minimum(2.0 * x_nodes, 1.0 + 0.5 * x_nodes)

# Ajuste del spline de Schumaker con preservación de forma
schumaker = SchumakerSpline(x_nodes, y_nodes)

# Evaluación sobre malla densa
x_dense = np.linspace(0.0, 4.0, 200)
y_interp = schumaker(x_dense)
slopes = schumaker.derivative(x_dense)

# Verificación de monotonicidad estricta: ninguna pendiente negativa
assert np.all(slopes >= -1e-12), "¡Monotonicidad violada!"
print("Spline de Schumaker verificado: pendientes estrictamente no negativas.")
```

### 4.2 Programación dinámica continua mediante colocación con B-splines cúbicos

Resolución del modelo de Brock-Mirman empleando B-splines cúbicos con nudos anclados en los extremos:

```python
import numpy as np
from puremacro.vfi import (
    CubicBSplineBasis,
    SplineCollocationProblem,
    solve_spline_collocation,
)

alpha = 0.36
beta = 0.96
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
domain = (0.5 * k_ss, 1.5 * k_ss)

# Definición del problema con 15 nudos interiores
prob_spline = SplineCollocationProblem(
    domain=domain,
    n_knots=15,
    spline_type="cubic",
    bc_type="clamped",
    method="euler",
    return_fn=lambda s, sp, params: np.log(np.maximum(s**alpha - sp, 1e-12)),
    transition_fn=lambda s: s**alpha,
    beta=beta,
    params={"alpha": alpha, "delta": 1.0},
)
sol_spline = solve_spline_collocation(prob_spline)
assert sol_spline.converged

# Evaluación frente a la solución analítica exacta
eval_k = np.linspace(domain[0], domain[1], 500)
g_true = alpha * beta * (eval_k**alpha)
g_spline = sol_spline.policy(eval_k)

max_spline_err = np.max(np.abs(g_spline - g_true) / g_true)
print(f"Error relativo máx. B-spline cúbico: {max_spline_err:.2e}")
assert max_spline_err < 1e-4
```

### 4.3 Modelo multidimensional con dos capitales en mallas dispersas de Smolyak

Resolución de un modelo con dos bienes de capital ($k_1, k_2$) sobre una malla dispersa de Smolyak de nivel $\mu = 2$ (únicamente 13 nodos frente a los 25 del producto tensorial completo):

```python
import numpy as np
from puremacro.vfi import SmolyakGrid, SmolyakProblem, solve_smolyak

alphas = [0.18, 0.18]
beta = 0.96
z = 1.0 / (alphas[0] * beta)
domain_2d = ((0.8, 1.2), (0.8, 1.2))

# 1. Inspección del factor de reducción de nodos
grid = SmolyakGrid(d=2, mu=2, domain=domain_2d)
print(f"Nodos en malla 2D Smolyak (mu=2): {grid.n_nodes} (frente a 25 tensoriales)")

# 2. Resolución del modelo continuo 2D
sol_smol = solve_smolyak(
    domain=domain_2d,
    mu=2,
    params={"alphas": alphas, "z": z, "beta": beta},
)
assert sol_smol.converged
assert sol_smol.residual_norm < 1e-4

# 3. Evaluación fuera de muestra en 500 puntos continuos
np.random.seed(42)
test_k = np.random.uniform(0.8, 1.2, (500, 2))
kp_approx = sol_smol.policy(test_k)

Y_test = z * (test_k[:, 0] ** alphas[0]) * (test_k[:, 1] ** alphas[1])
kp_true = np.column_stack([alphas[0] * beta * Y_test, alphas[1] * beta * Y_test])

max_smol_err = np.max(np.abs(kp_approx - kp_true) / kp_true)
print(f"Error relativo máx. Smolyak 2D: {max_smol_err:.2e}")
assert max_smol_err < 1e-4
```

---

## 5. Especificación completa de la API

### `CubicBSplineBasis` / `SplineBasis`

```text
CubicBSplineBasis(
    knots: Sequence[float] | np.ndarray | None = None,
    domain: tuple[float, float] | None = None,
    n_knots: int = 20,
    degree: int = 3,
    bc_type: str = "clamped",
    knot_spacing: str = "uniform",
)
```

#### Parámetros:
- `knots`: Vector explícito de nudos 1D (si `domain` es None).
- `domain`: Intervalo continuo `(k_min, k_max)` para generar nudos automáticos.
- `n_knots`: Número de nudos interiores (por defecto 20).
- `degree`: Grado del spline (debe ser 3 para splines cúbicos).
- `bc_type`: Tipo de condición de frontera: `"clamped"`, `"natural"` o `"not-a-knot"`.
- `knot_spacing`: Distribución de los nudos: `"uniform"` o `"geometric"`.

---

### `SchumakerSpline`

```text
SchumakerSpline(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray | None = None,
    extrapolate: bool = True,
)
```

#### Parámetros:
- `x`: Array 1D estrictamente creciente de coordenadas.
- `y`: Array 1D de valores de la función evaluados en `x`.
- `s`: Estimaciones opcionales de derivadas nodales.
- `extrapolate`: Booleano para extrapolar linealmente fuera de `[x[0], x[-1]]`.

---

### `SplineCollocationProblem` y `solve_spline_collocation`

```text
SplineCollocationProblem(
    domain: tuple[float, float] | list,
    n_knots: int = 20,
    spline_type: str = "cubic",
    bc_type: str = "clamped",
    knot_spacing: str = "uniform",
    method: str = "euler",
    euler_residual_fn: Callable | None = None,
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    borrowing_constraint: float | None = None,
    options: dict = {},
)

solve_spline_collocation(
    problem_or_domain: SplineCollocationProblem | tuple[float, float],
    backend: str = "numpy",
    **kwargs: Any,
) -> SplineCollocationSolution
```

---

### `SmolyakGrid`

```text
SmolyakGrid(
    d: int,
    mu: int,
    domain: Sequence[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
)
```

#### Parámetros:
- `d`: Dimensión continua del espacio de estados $d \in [2, 6]$.
- `mu`: Nivel de aproximación de Smolyak $\mu \in [1, 5]$ (por defecto 2).
- `domain`: Límites por dimensión `[(a_1, b_1), ..., (a_d, b_d)]`.

---

### `SmolyakProblem` y `solve_smolyak`

```text
SmolyakProblem(
    domain: Sequence[tuple[float, float]] | tuple[tuple[float, float], ...],
    mu: int = 2,
    method: str = "euler",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    options: dict = {},
)

solve_smolyak(
    problem: SmolyakProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> SmolyakSolution
```

---

## 6. Interfaz de resultados y exportación a manuscritos

Tanto `SplineCollocationSolution` como `SmolyakSolution` implementan el contrato estándar de resultados de `puremacro`:

- **Atributos**:
  - `.c` o `.coefficients`: Vector de coeficientes ajustados.
  - `.converged`: Booleano de convergencia del algoritmo.
  - `.residual_norm`: Norma del supremo final de los residuos $\|\mathcal{R}\|_\infty$.
  - `.n_iter`: Número de iteraciones evaluadas.
- **Métodos de evaluación continua**:
  - `.policy(s)`: Evaluación de la política en cualquier punto continuo $s$.
  - `.value(s)`: Evaluación de la función de valor continua.
  - `.euler_residual(s)`: Evaluación de residuos de la ecuación de Euler.
- **Exportación académica**:
  - `.summary()`: Resumen diagnóstico estructurado.
  - `.plot()`: Gráfico multipanel en Matplotlib.
  - `.to_frame()`: Exportación a `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas científicas formateadas.

---

## Referencias bibliográficas

- **De Boor, C. (1978)**. *A Practical Guide to Splines*. Springer-Verlag.
- **Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014)**. Smolyak method for solving dynamic economic models: Lagrange interpolation, anisotropic grid and adaptive domain. *Journal of Economic Dynamics and Control*, 44, 92–123.
- **Krueger, D., & Kubler, F. (2004)**. Computing equilibrium in OLG models with stochastic production and incomplete markets. *Journal of Economic Dynamics and Control*, 28(7), 1411–1436.
- **Malin, B. A., Krueger, D., & Kubler, F. (2011)**. Solving the multi-country real business cycle model using a Smolyak-style collocation method. *Journal of Economic Dynamics and Control*, 35(2), 229–239.
- **Schumaker, L. L. (1983)**. On shape preserving quadratic spline interpolation. *SIAM Journal on Numerical Analysis*, 20(4), 854–864.
- **Smolyak, S. A. (1963)**. Quadrature and interpolation formulas for tensor products of certain classes of functions. *Soviet Mathematics Doklady*, 4, 240–243.
