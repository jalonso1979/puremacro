> 🇬🇧 [English](../vfi_continuous_projection.md) · 🇪🇸 Español

# Solucionadores de proyección continua: Colocación de Chebyshev y Elementos Finitos de Galerkin

Los módulos `puremacro.vfi.collocation` y `puremacro.vfi.fem` implementan métodos de proyección y programación dinámica continua desarrollados por **Judd (1992, 1998)** y **McGrattan (1996, 1999)**, proporcionando aproximaciones de alta precisión para funciones de política y de valor sin los errores de discretización propios de la iteración de la función de valor (VFI) sobre mallas discretas.

La programación dinámica tradicional sobre mallas discretas restringe las decisiones de acumulación de capital y riqueza a una retícula artificial $\{k_1, \dots, k_M\}$. Esta discretización introduce un error de truncamiento de orden $\mathcal{O}(M^{-1})$, destruye la diferenciabilidad y sufre una severa maldición de la dimensionalidad. Los métodos de proyección continua resuelven directamente reglas de decisión continuas $g(k)$ y funciones de valor $V(k)$ definidas sobre variedades continuas del espacio de estados:

- **La colocación ortogonal de Chebyshev** emplea polinomios ortogonales de Chebyshev de primera especie evaluados en nodos de Gauss-Lobatto o Gauss-Chebyshev. En modelos económicos analíticos y suaves (como el modelo neoclásico de crecimiento), el teorema de aproximación de Jackson garantiza una **convergencia espectral o exponencial** ($\mathcal{O}(c^{-N})$), reduciendo los residuos de la ecuación de Euler por debajo de $10^{-8}$ con apenas 6 a 10 coeficientes polinómicos.
- **El método de elementos finitos con proyección de Galerkin (MEF)** descompone el dominio continuo en elementos locales empleando funciones base lineales o cuadráticas de Lagrange en forma de sombrero con soporte compacto. Al confinar los errores de aproximación al elemento local, el MEF resuelve de manera única **puntos angulosos (*kinks*), restricciones de endeudamiento ocasionalmente activas y no diferenciabilidades** mediante complementariedad de Fischer-Burmeister con **cero oscilaciones espurias de Gibbs**.

---

## 1. Fundamentos teóricos y matemáticos

### 1.1 Formulación económica dinámica continua

Considérese un problema macroeconómico de decisión intertemporal en tiempo discreto y estado continuo donde un agente elige el capital futuro $k' \in [\underline{k}, \bar{k}]$ dados el capital actual $k$ y un choque estocástico de productividad markoviano $z$:

$$V(k, z) = \max_{k' \ge \underline{k}} \left\{ u(c) + \beta \sum_{z'} \Pi(z, z') V(k', z') \right\}$$

sujeto a la restricción agregada de recursos:

$$c + k' \le f(k, z) + (1 - \delta)k$$

En soluciones interiores, la política óptima $k' = g(k, z)$ satisface el residuo funcional continuo de la ecuación de Euler $\mathcal{R}(k, z) = 0$:

$$\mathcal{R}(k, z; g) \equiv u'\left(f(k, z) + (1-\delta)k - g(k, z)\right) - \beta \sum_{z'} \Pi(z, z') u'\left(c(g(k, z), z')\right) \left[ f_k\left(g(k, z), z'\right) + 1 - \delta \right] = 0$$

### 1.2 Colocación ortogonal mediante polinomios de Chebyshev

El método de colocación aproxima la función de política $g(k)$ (o la función de valor $V(k)$) como una combinación lineal de $N+1$ polinomios ortogonales de Chebyshev:

$$g_N(k; \mathbf{c}) = \sum_{j=0}^N c_j T_j(x(k))$$

#### 1. Transformación afín del dominio de estados

Los polinomios de Chebyshev se definen sobre el intervalo canónico $x \in [-1, 1]$. Cualquier dominio acotado $k \in [\underline{k}, \bar{k}]$ se transforma biyectivamente a $[-1, 1]$ mediante la aplicación afín:

$$x(k) = \frac{2k - (\bar{k} + \underline{k})}{\bar{k} - \underline{k}}, \qquad k(x) = \frac{(x + 1)(\bar{k} - \underline{k})}{2} + \underline{k}$$

#### 2. Nodos de colocación: Gauss-Lobatto frente a Gauss-Chebyshev

Para prevenir el fenómeno de Runge propio de las mallas equiespaciadas, los residuos de proyección se anulan en los nodos de cuadratura de Chebyshev:

- **Nodos de Gauss-Lobatto** (extremos locales más los límites del dominio):
  $$x_i = -\cos\left(\frac{i \pi}{N}\right), \quad i = 0, 1, \dots, N$$
  Incluyen exactamente las fronteras $x_0 = -1$ y $x_N = 1$, siendo óptimos para modelos con condiciones de frontera o límites fijos de endeudamiento.
- **Nodos de Gauss-Chebyshev** (raíces o ceros de $T_{N+1}(x)$):
  $$x_i = -\cos\left(\frac{2i + 1}{2(N + 1)} \pi\right), \quad i = 0, 1, \dots, N$$
  Se ubican estrictamente en el interior abierto $(-1, 1)$, resultando idóneos cuando las funciones de utilidad presentan derivadas singulares en la frontera ($u'(0) \to \infty$).

#### 3. Relaciones de recurrencia de tres términos

Los polinomios $T_j(x)$ y sus derivadas $T'_j(x)$ se calculan con máxima estabilidad numérica mediante relaciones de recurrencia:

$$T_0(x) = 1, \qquad T_1(x) = x, \qquad T_{j+1}(x) = 2x T_j(x) - T_{j-1}(x), \quad j \ge 1$$

$$T'_0(x) = 0, \qquad T'_1(x) = 1, \qquad T'_{j+1}(x) = 2 T_j(x) + 2x T'_j(x) - T'_{j-1}(x), \quad j \ge 1$$

#### 4. Condiciones de colocación

El vector de coeficientes $\mathbf{c}^* \in \mathbb{R}^{N+1}$ se obtiene imponiendo que el residuo de Euler (o de Bellman) se anule exactamente en los $N+1$ nodos de colocación:

$$\mathcal{R}\left(k(x_i); \mathbf{c}^*\right) = 0, \quad \forall i = 0, 1, \dots, N$$

Este sistema no lineal de dimensión $N+1$ se resuelve numéricamente mediante el método híbrido de Powell (`solver_method="hybr"`) o Levenberg-Marquardt.

---

### 1.3 Método de Elementos Finitos con Proyección de Galerkin (MEF)

Los métodos de polinomios globales pueden presentar dificultades cuando las funciones de política exhiben discontinuidades en sus derivadas (*kinks*), causadas por restricciones de endeudamiento ($k' \ge \bar{k}$) o límites de capacidad. Aproximar estas discontinuidades con polinomios globales genera el **fenómeno de Gibbs**, caracterizado por oscilaciones espurias de alta frecuencia que se propagan a lo largo de todo el espacio de estados.

El Método de Elementos Finitos supera este obstáculo dividiendo el dominio continuo en $E$ elementos locales discretos con funciones base de soporte compacto.

#### 1. Partición de la malla y base de sombrero lineal por tramos

Sea $[\underline{k}, \bar{k}]$ particionado en $E$ elementos por los nodos $\underline{k} = k_0 < k_1 < \dots < k_E = \bar{k}$ con longitud de elemento $h_e = k_{e+1} - k_e$.

En cada nodo interior $j$, la función base de sombrero de Lagrange $\phi_j(k)$ posee soporte compacto en $[k_{j-1}, k_{j+1}]$:

$$\phi_j(k) = \begin{cases}
\frac{k - k_{j-1}}{k_j - k_{j-1}} & \text{si } k \in [k_{j-1}, k_j] \\
\frac{k_{j+1} - k}{k_{j+1} - k_j} & \text{si } k \in [k_j, k_{j+1}] \\
0 & \text{en otro caso}
\end{cases}$$

La función de política continua se expresa directamente en función de los valores nodales $\mathbf{y} = (y_0, y_1, \dots, y_E)^\top$:

$$g_E(k; \mathbf{y}) = \sum_{j=0}^E y_j \phi_j(k)$$

Dado que $\phi_j(k_i) = \delta_{ij}$ (la delta de Kronecker), los coeficientes nodales $y_j = g_E(k_j)$ coinciden con el valor efectivo de la política en los nodos, eliminando la necesidad de invertir matrices de cambio de base.

#### 2. Formulación débil y ortogonalización de Galerkin

En la proyección de Galerkin, el residuo continuo $\mathcal{R}(k; \mathbf{y})$ se obliga a ser ortogonal a cada función base $\phi_i(k)$ en el producto interno de Hilbert $L^2$:

$$\langle \mathcal{R}, \phi_i \rangle \equiv \int_{\underline{k}}^{\bar{k}} \mathcal{R}(k; \mathbf{y}) \phi_i(k) \, dk = 0, \quad i = 0, 1, \dots, E$$

Debido al soporte compacto de $\phi_i(k)$ en $[k_{i-1}, k_{i+1}]$, la integral se descompone en los elementos contiguos:

$$\int_{k_{i-1}}^{k_i} \mathcal{R}(k) \frac{k - k_{i-1}}{h_{i-1}} \, dk + \int_{k_i}^{k_{i+1}} \mathcal{R}(k) \frac{k_{i+1} - k}{h_i} \, dk = 0$$

Cada integral de elemento se evalúa numéricamente mediante cuadratura de Gauss-Legendre de $Q$ puntos ($Q = 3$ o $Q = 5$):

$$\int_{k_e}^{k_{e+1}} F(k) \, dk \approx \frac{h_e}{2} \sum_{q=1}^Q \omega_q F\left( \frac{k_e + k_{e+1}}{2} + \frac{h_e}{2} \xi_q \right)$$

donde $\xi_q \in [-1, 1]$ y $\omega_q$ son los nodos y pesos estándar de la cuadratura de Gauss-Legendre.

---

### 1.4 Complementariedad de Fischer-Burmeister para restricciones de endeudamiento

Cuando la elección de ahorro enfrenta una restricción de desigualdad $k' \ge \bar{k}$, las condiciones de optimalidad de Karush-Kuhn-Tucker (KKT) exigen:

$$k' - \bar{k} \ge 0, \qquad \mathcal{R}(k) \ge 0, \qquad (k' - \bar{k}) \cdot \mathcal{R}(k) = 0$$

Los algoritmos convencionales de búsqueda de raíces no pueden gestionar directamente desigualdades. `puremacro.vfi` reformula estas condiciones de complementariedad empleando el **operador regularizado de Fischer-Burmeister** ($\epsilon = 10^{-12}$):

$$\Psi_{\text{FB}}^\epsilon(a, b) \equiv a + b - \sqrt{a^2 + b^2 + \epsilon} = 0$$

donde:

$$a = k' - \bar{k}, \qquad b = \mathcal{R}(k)$$

Este operador satisface:
- $\Psi_{\text{FB}}^\epsilon(a, b) = 0 \iff a \ge 0, b \ge 0, a \cdot b \approx 0$.
- Es continuamente diferenciable en todo su dominio para $\epsilon > 0$, permitiendo convergencia rápida mediante Newton-Raphson.
- Al alinear explícitamente un nodo de la malla en el codo de endeudamiento $k^*$ mediante `FEMMesh.from_kinks(domain, kinks=[k_kink])`, el MEF aísla la no diferenciabilidad en la frontera del elemento, erradicando por completo el fenómeno de Gibbs.

---

### 1.5 Modelo de referencia analítico de Brock-Mirman: Convergencia espectral vs. polinómica

El banco de pruebas estándar para validar solucionadores continuos es el modelo de crecimiento neoclásico de Brock y Mirman (1972) con utilidad logarítmica $u(c) = \ln(c)$, función de producción Cobb-Douglas $f(k) = k^\alpha$ y depreciación total ($\delta = 1.0$):

$$V(k) = \max_{k'} \left\{ \ln(k^\alpha - k') + \beta V(k') \right\}$$

Este modelo admite soluciones analíticas exactas en forma cerrada:

$$g^*(k) = \alpha \beta k^\alpha$$

$$V^*(k) = \frac{\alpha}{1 - \alpha \beta} \ln(k) + \frac{\ln(1 - \alpha \beta) + \frac{\alpha \beta \ln(\alpha \beta)}{1 - \alpha \beta}}{1 - \beta}$$

#### Comparativa de convergencia del error

| Métrica | Colocación de Chebyshev ($N$ órdenes) | Elementos Finitos Galerkin ($E$ elementos) |
|---|---|---|
| **Tasa de convergencia** | **Espectral / Exponencial**: $\mathcal{O}(c^{-N})$ | **Algebraica / Polinómica**: $\mathcal{O}(h^2) = \mathcal{O}(E^{-2})$ |
| **Error Euler máx. ($N=6$ vs $E=30$)** | $< 10^{-6}$ | $\approx 10^{-4}$ |
| **Error Euler máx. ($N=12$ vs $E=100$)** | $< 10^{-9}$ | $\approx 10^{-5}$ |
| **Robustez en codos ($k' \ge \bar{k}$)** | Oscilaciones de Gibbs en todo el dominio | Cero oscilaciones con `FEMMesh.from_kinks` |
| **Estructura del jacobiano** | Matriz densa | Matriz tridiagonal / dispersa en banda |

---

## 2. Opciones metodológicas y de modelos

Los solucionadores continuos de `puremacro.vfi` ofrecen múltiples formulaciones y motores de cómputo:

| Configuración / Característica | `CollocationProblem` | `FEMProblem` |
|---|---|---|
| **Dominio de aplicación** | Variedades suaves y analíticas | Modelos con codos, fricciones o límites |
| **Método de solución (`method`)** | `"euler"` (política) o `"bellman"` (valor) | `"euler"` (política) o `"bellman"` (valor) |
| **Operador de proyección** | Colocación nodal en ceros/extremos | `"galerkin"` (cuadratura) o `"collocation"` |
| **Estructura de la malla** | Nodos de Gauss-Lobatto o Gauss-Chebyshev | Uniforme, acumulada por potencias o alineada a codos |
| **Límite de endeudamiento** | Vía penalización de Fischer-Burmeister | `borrowing_constraint` nativo con FB |
| **Aceleración por hardware** | NumPy, Numba, Apple MLX, CuPy | NumPy, Numba, Apple MLX, CuPy |
| **Tolerancia por defecto** | `tol=1e-8`, `maxiter=500` | `tol=1e-8`, `maxiter=500` |

---

## 3. Calibración canónica: Crecimiento neoclásico de Brock-Mirman

Parámetros de calibración para pruebas y replicación:

| Parámetro | Símbolo | Valor de referencia | Descripción económica |
|---|---|---|---|
| Elasticidad del capital | $\alpha$ | `0.36` | Participación del capital en el producto |
| Factor de descuento | $\beta$ | `0.96` | Descuento intertemporal trimestral |
| Tasa de depreciación | $\delta$ | `1.00` | Depreciación total (para tractabilidad analítica) |
| Capital de estado estacionario | $k_{\text{ss}}$ | $(\alpha \beta)^{\frac{1}{1-\alpha}} \approx 0.1782$ | Nivel de capital a largo plazo |
| Dominio del espacio de estados | $[\underline{k}, \bar{k}]$ | $[0.5 k_{\text{ss}}, 1.5 k_{\text{ss}}] \approx [0.089, 0.267]$ | Intervalo continuo evaluado |

---

## 4. Ejemplos de uso ejecutables

### 4.1 Crecimiento neoclásico: Colocación de Chebyshev vs. MEF Galerkin

Este ejemplo resuelve el modelo de Brock-Mirman mediante colocación de Chebyshev y elementos finitos de Galerkin, contrastando ambas soluciones frente a la solución analítica exacta:

```python
import numpy as np
from puremacro.vfi import (
    CollocationProblem,
    solve_collocation,
    FEMProblem,
    solve_fem,
)

# 1. Calibración de parámetros
alpha = 0.36
beta = 0.96
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
domain = (0.5 * k_ss, 1.5 * k_ss)

# 2. Resolución mediante colocación de Chebyshev (Orden N = 6)
prob_col = CollocationProblem(
    domain=domain,
    orders=6,
    method="euler",
    params={"alpha": alpha, "delta": 1.0},
    beta=beta,
)
sol_col = solve_collocation(prob_col)
assert sol_col.converged

# 3. Resolución mediante MEF Galerkin (30 elementos lineales)
prob_fem = FEMProblem(
    domain=domain,
    elements=30,
    method="euler",
    projection="galerkin",
    return_fn=lambda c: np.log(np.maximum(c, 1e-12)),
    transition_fn=lambda k: k**alpha,
    beta=beta,
    params={"alpha": alpha, "gamma": 1.0},
)
sol_fem = solve_fem(prob_fem)
assert sol_fem.converged

# 4. Evaluación fuera de muestra en 1.000 puntos continuos
eval_k = np.linspace(domain[0], domain[1], 1000)
g_true = alpha * beta * (eval_k**alpha)
g_col = sol_col.policy(eval_k)
g_fem = sol_fem.policy(eval_k)

max_col_err = np.max(np.abs(g_col - g_true) / g_true)
max_fem_err = np.max(np.abs(g_fem - g_true) / g_true)

print(f"Error relativo máx. Colocación Chebyshev: {max_col_err:.2e}")
print(f"Error relativo máx. MEF Galerkin:         {max_fem_err:.2e}")

assert max_col_err < 1e-4
assert max_fem_err < 1e-3
```

### 4.2 Restricciones de endeudamiento y resolución de codos mediante MEF

Cuando la acumulación de capital enfrenta un límite de endeudamiento $k' \ge \bar{k}$, los polinomios globales sufren oscilaciones de Gibbs. El MEF alinea la frontera de un elemento en el codo y activa la complementariedad de Fischer-Burmeister:

```python
import numpy as np
from puremacro.vfi import FEMMesh, FEMProblem, solve_fem

# Límite inferior de endeudamiento
b_const = 0.16

# Malla adaptada con un nodo exactamente en la frontera de la restricción
mesh = FEMMesh.from_kinks(domain=domain, n_elements=40, kinks=[0.17])

prob_constrained = FEMProblem(
    domain=domain,
    elements=40,
    method="euler",
    projection="galerkin",
    return_fn=lambda c: np.log(np.maximum(c, 1e-12)),
    transition_fn=lambda k: k**alpha,
    beta=beta,
    borrowing_constraint=b_const,
    params={"alpha": alpha, "gamma": 1.0},
    options={"mesh": mesh},
)
sol_constrained = solve_fem(prob_constrained)
assert sol_constrained.converged

# Comprobación estricta de la restricción en 500 puntos densos
dense_k = np.linspace(domain[0], domain[1], 500)
pol_k = sol_constrained.policy(dense_k)
assert np.all(pol_k >= b_const - 1e-10), "¡Restricción de endeudamiento violada!"
print("La política MEF respetó k' >= bar{k} sin oscilaciones de Gibbs.")
```

### 4.3 Aceleración computacional multiplataforma

Los solucionadores de `puremacro.vfi` se integran con `puremacro._backend`, permitiendo aceleración transparente en CPU y GPU:

```python
from puremacro import _backend as bk

# Detección de aceleradores disponibles
print("Numba JIT disponible:   ", bk.backend_available("numba"))
print("Apple MLX disponible:   ", bk.backend_available("mlx"))
print("NVIDIA CuPy disponible: ", bk.backend_available("cupy"))

# Ejecución seleccionando el backend de cómputo
sol_fast = solve_collocation(prob_col, backend="numpy")
print(f"Solución convergida: {sol_fast.converged} | Norma residual: {sol_fast.residual_norm:.2e}")
```

---

## 5. Especificación completa de la API

### `CollocationProblem`

```text
CollocationProblem(
    domain: tuple[float, float] | tuple[tuple[float, float], ...],
    orders: int | tuple[int, ...],
    method: str = "euler",
    node_type: str = "lobatto",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    options: dict = {},
)
```

#### Parámetros:
- `domain`: Intervalo continuo del espacio de estados `(k_min, k_max)` en 1D o tupla de intervalos en dimensiones múltiples.
- `orders`: Grado del polinomio de aproximación $N \ge 1$ (o tupla de grados por dimensión).
- `method`: Objetivo de aproximación:
  - `"euler"`: Proyección de la política minimizando los residuos de la ecuación de Euler.
  - `"bellman"`: Iteración continua de la función de valor sobre nodos de Chebyshev.
- `node_type`: Distribución de nodos de Chebyshev:
  - `"lobatto"`: Nodos extremos de Gauss-Lobatto (incluyen las fronteras $\pm 1$).
  - `"gauss"`: Raíces interiores de Gauss-Chebyshev (estrictamente en $(-1, 1)$).
- `return_fn`: Función de utilidad de un período $u(c)$ o $u(k, k', \mathbf{p})$.
- `transition_fn`: Ecuación de transición del capital $f(k)$ o presupuesto agregado.
- `euler_residual_fn`: Función personalizada del residuo de Euler $R(k, k', k'', \mathbf{p})$.
- `beta`: Factor de descuento del agente $\beta \in (0, 1)$.
- `params`: Diccionario de parámetros estructurales (por ejemplo, `{"alpha": 0.36, "delta": 1.0}`).
- `options`: Opciones del algoritmo de optimización (`tol`, `max_iter`, `solver_method`).

---

### `solve_collocation`

```text
solve_collocation(
    problem: CollocationProblem | None = None,
    backend: str = "numpy",
    **kwargs: Any,
) -> CollocationSolution
```

Resuelve el problema continuo de colocación de Chebyshev mediante búsqueda de raíces no lineales. Acepta una instancia de `CollocationProblem` o parámetros por clave.

---

### `FEMProblem`

```text
FEMProblem(
    domain: tuple[float, float] | tuple[tuple[float, float], ...],
    elements: int | tuple[int, ...] = 50,
    method: str = "euler",
    projection: str = "galerkin",
    return_fn: Callable | None = None,
    transition_fn: Callable | None = None,
    euler_residual_fn: Callable | None = None,
    beta: float = 0.96,
    params: dict = {},
    borrowing_constraint: float | None = None,
    options: dict = {},
)
```

#### Parámetros:
- `domain`: Límites del espacio de estados continuo `(s_min, s_max)`.
- `elements`: Número de elementos finitos de la malla $E \ge 2$.
- `method`: Método de solución: `"euler"` o `"bellman"`.
- `projection`: Esquema de proyección: `"galerkin"` (integración ponderada de residuos vía cuadratura de Gauss-Legendre) o `"collocation"` (colocación nodal).
- `return_fn`: Función de utilidad $u(c)$ o $u(s, s')$.
- `transition_fn`: Regla de transición del capital $f(s)$.
- `euler_residual_fn`: Función explícita del residuo de Euler continuo.
- `beta`: Factor de descuento intertemporal $\beta \in (0, 1)$.
- `params`: Diccionario de parámetros del modelo económico.
- `borrowing_constraint`: Límite inferior opcional de endeudamiento $s' \ge \underline{s}$.
- `options`: Diccionario de opciones avanzadas:
  - `"mesh"`: Instancia preconfigurada de `FEMMesh` (por ejemplo, vía `FEMMesh.from_kinks`).
  - `"complementarity"`: Operador de complementariedad (`"fb"` para Fischer-Burmeister, `"min"` para mínimo).
  - `"quad_order"`: Puntos de cuadratura de Gauss-Legendre por elemento (por defecto 5).

---

### `solve_fem`

```text
solve_fem(
    problem_or_domain: FEMProblem | tuple[float, float],
    *args: Any,
    backend: str = "numpy",
    **kwargs: Any,
) -> FEMSolution
```

Resuelve el modelo continuo de elementos finitos mediante búsqueda de raíces no lineales.

---

## 6. Interfaz de resultados y exportación a manuscritos

Tanto `CollocationSolution` como `FEMSolution` cumplen íntegramente el contrato de presentación estándar de `puremacro`:

- **Atributos de solución**:
  - `.c`: Vector de coeficientes polinómicos ajustados o valores nodales del MEF.
  - `.converged`: Booleano que certifica la convergencia del algoritmo.
  - `.residual_norm`: Norma del supremo final de los residuos $\|\mathcal{R}\|_\infty$.
  - `.n_iter`: Número total de iteraciones evaluadas por el algoritmo.
- **Métodos de evaluación continua**:
  - `.policy(k)`: Evalúa la regla de decisión continua $g(k)$ en cualquier valor $k$.
  - `.value(k)`: Evalúa la función de valor continua $V(k)$.
  - `.euler_residuals(k_eval=None)`: Calcula los residuos de Euler sobre una malla densa de prueba fuera de muestra.
- **Visualización y exportación académica**:
  - `.summary()`: Resumen estructurado de convergencia y diagnósticos de residuos.
  - `.plot()`: Gráfico multipanel en Matplotlib con funciones de política, valor y residuos de Euler.
  - `.to_frame()`: Exporta los valores nodales y evaluaciones a un `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas formateadas listas para su inclusión en manuscritos y artículos científicos.

---

## Referencias bibliográficas

- **Brock, W. A., & Mirman, L. J. (1972)**. Optimal economic growth and uncertainty: The discounted case. *Journal of Economic Theory*, 4(3), 479–513.
- **Fischer, A. (1992)**. A special Newton-type optimization method. *Optimization*, 24(3-4), 269–284.
- **Judd, K. L. (1992)**. Projection methods for solving aggregate growth models. *Journal of Economic Theory*, 58(2), 410–452.
- **Judd, K. L. (1998)**. *Numerical Methods in Economics*. MIT Press.
- **McGrattan, E. R. (1996)**. Solving the stochastic growth model with a finite element method. *Journal of Economic Dynamics and Control*, 20(1-3), 19–42.
- **McGrattan, E. R. (1999)**. Application of weighted residual methods to dynamic economic models. En *Computational Methods for the Study of Dynamic Economies* (pp. 114–142). Oxford University Press.
- **Miranda, M. J., & Fackler, P. L. (2002)**. *Applied Computational Economics and Finance*. MIT Press.
