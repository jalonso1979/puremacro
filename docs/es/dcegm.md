> 🇬🇧 [English](../dcegm.md) · 🇪🇸 Español

# Método de la malla endógena con elección discreta (DC-EGM) y programación dinámica no convexa

El módulo `puremacro.vfi.dcegm` implementa el Método de la Malla Endógena con Elección Discreta (DC-EGM) desarrollado por **Iskhakov, Jørgensen, Rust y Schjerning (2017, *Quantitative Economics*)**, extendiendo el método de mallas endógenas (**Carroll 2006**) a modelos estructurales dinámicos que combinan **decisiones discretas** (jubilación, participación laboral, impago hipotecario, renovación de bienes duraderos) y **decisiones continuas** (consumo, ahorro líquido, acumulación de riqueza).

En la programación dinámica estándar, la combinación de decisiones discretas y continuas destruye la concavidad de la función de valor. Las elecciones discretas introducen codos (*kinks*) y regiones no cóncavas en $V(M)$, lo que provoca que la aplicación de la ecuación de Euler $a' \mapsto M(a')$ se pliegue hacia atrás y se vuelva multivaluada. Los métodos tradicionales de búsqueda en malla requieren densas retículas y optimizaciones globales no lineales en cada punto del espacio de estados. El algoritmo DC-EGM resuelve este problema con máxima eficiencia numérica:

- **Inversión analítica tipo EGM**: Invierte la condición de primer orden sobre una malla exógena de ahorro posterior a la decisión $a'$, obteniendo el consumo candidato $c(a')$ y los recursos disponibles $M(a') = a' + c(a')$ sin requerir búsqueda de raíces no lineales.
- **Filtrado por envolvente superior (*Upper Envelope*)**: Detecta sistemáticamente los pliegues y ramas que se cruzan en el espacio $(M, v_d(M))$, descartando las ramas subóptimas que satisfacen la ecuación de Euler localmente pero que no alcanzan el óptimo global.
- **Perturbaciones de preferencia de valores extremos**: Incorpora choques aditivos de valor extremo tipo I (Gumbel) (**Rust 1987**), proporcionando probabilidades de elección logit suaves en forma cerrada $P(d \mid M)$ y valores inclusivos analíticos (operador *Log-Sum-Exp*).
- **Esperanzas exactas del valor marginal vía el teorema de la envolvente**: Calcula el valor marginal futuro esperado $\mathbb{E}[V'(M')]$ directamente a partir de las probabilidades de elección, sin recurrir a diferenciación numérica.

---

## 1. Fundamentos teóricos y econométricos

### 1.1 Formulación y sincronización temporal del modelo

Considérese un agente en el período $t$ con recursos líquidos disponibles (*cash-on-hand*) $M_t$. El agente toma una decisión discreta $d_t \in \{0, 1, \dots, D-1\}$ (por ejemplo, $d=0$ para continuar trabajando y $d=1$ para jubilarse) y elige el consumo continuo $c_t \in (0, M_t]$, dejando un ahorro al final del período $a_{t+1} = M_t - c_t \ge a_{\min}$.

La utilidad del agente incorpora choques de preferencia aditivos específicos de cada opción $\boldsymbol{\epsilon}_t = (\epsilon_{0, t}, \dots, \epsilon_{D-1, t})$, distribuidos independientemente según una ley de Valores Extremos Tipo I (Gumbel) con parámetro de dispersión $\sigma_\epsilon \ge 0$:

$$U(c_t, d_t, \boldsymbol{\epsilon}_t) = u(c_t, d_t) + \epsilon_{d, t}$$

La ecuación de Bellman para la función de valor ex-ante $V_t(M_t)$ es:

$$V_t(M_t) = \max_{d \in \{0, \dots, D-1\}} \left\{ v_t(M_t, d) + \epsilon_{d, t} \right\}$$

donde la función de valor condicional a la elección $v_t(M_t, d)$ satisface:

$$v_t(M_t, d) = \max_{a_{t+1} \ge a_{\min}} \left\{ u(M_t - a_{t+1}, d) + \beta \mathbb{E}\left[ V_{t+1}\left( (1+r)a_{t+1} + y_{t+1}(d_{t+1}, z_{t+1}) \right) \;\middle|\; d_t = d, a_{t+1} \right] \right\}$$

con tipo de interés real $r$ y rendimiento bruto $R = 1 + r$.

---

### 1.2 Choques de valor extremo y valor inclusivo

Al integrar analíticamente sobre la distribución conjunta de las perturbaciones de preferencia $\boldsymbol{\epsilon}_t$, se obtiene la función de valor esperada ex-ante (**valor inclusivo** o excedente social):

$$\mathcal{V}_t(M_t) \equiv \mathbb{E}_{\boldsymbol{\epsilon}}\left[ \max_{d \in \{0, \dots, D-1\}} \left\{ v_t(M_t, d) + \epsilon_{d, t} \right\} \right] = \sigma_\epsilon \ln \left( \sum_{d=0}^{D-1} \exp\left(\frac{v_t(M_t, d)}{\sigma_\epsilon}\right) \right) + \gamma_{\text{Euler}} \sigma_\epsilon$$

donde $\gamma_{\text{Euler}} \approx 0.5772$ es la constante de Euler-Mascheroni.

#### Probabilidades de elección suaves

La probabilidad condicional de elegir la opción discreta $d$ adopta la estructura multinomial logit estándar:

$$P_t(d \mid M_t) = \frac{\exp\left( \frac{v_t(M_t, d)}{\sigma_\epsilon} \right)}{\sum_{d'=0}^{D-1} \exp\left( \frac{v_t(M_t, d')}{\sigma_\epsilon} \right)} = \frac{1}{\sum_{d'=0}^{D-1} \exp\left( \frac{v_t(M_t, d') - v_t(M_t, d)}{\sigma_\epsilon} \right)}$$

En el límite determinista cuando $\sigma_\epsilon \to 0$, la probabilidad logit converge a la función indicatriz $\mathbf{1}\{d = \arg\max_{d'} v_t(M_t, d')\}$, y el valor inclusivo converge al máximo determinista $\max_d v_t(M_t, d)$.

---

### 1.3 Valor marginal esperado mediante el teorema de la envolvente

Para aplicar el método de mallas endógenas se requiere el valor marginal esperado respecto al ahorro $a_{t+1}$:

$$\mathfrak{w}_t(a_{t+1}) \equiv \beta R \, \mathbb{E}\left[ \mathcal{V}'_{t+1}(M_{t+1}) \;\middle|\; a_{t+1} \right]$$

La programación dinámica tradicional aproxima $\mathcal{V}'_{t+1}$ mediante diferencias finitas numéricas, lo cual introduce ruido y elevado coste computacional. Aplicando el **teorema de la envolvente** y la regla de la cadena sobre el valor inclusivo:

$$\mathcal{V}'_{t+1}(M) = \sum_{d=0}^{D-1} \frac{\partial \mathcal{V}_{t+1}}{\partial v_{t+1}(M, d)} \frac{\partial v_{t+1}(M, d)}{\partial M} = \sum_{d=0}^{D-1} P_{t+1}(d \mid M) \, u'\left(c_{t+1}^*(M, d), d\right)$$

Este resultado fundamental permite evaluar el valor marginal esperado de forma continua y exacta como una combinación convexa de utilidades marginales ponderada por las probabilidades de elección, **con cero error de diferenciación numérica**:

$$\mathfrak{w}_t(a_{t+1}) = \beta R \sum_{z'} \Pi(z, z') \sum_{d'=0}^{D-1} P_{t+1}\left(d' \;\middle|\; R a_{t+1} + y'(d', z')\right) u'\left(c_{t+1}^*\left(R a_{t+1} + y'(d', z'), d'\right), d'\right)$$

Dada una malla exógena de activos de ahorro $a' \in \{a_1', \dots, a_K'\}$, la ecuación de Euler se invierte directamente para cada opción discreta $d$:

$$c_t(a'; d) = (u')^{-1}\left( \mathfrak{w}_t(a') \right), \qquad M_t(a'; d) = a' + c_t(a'; d)$$

---

### 1.4 El algoritmo de la envolvente superior (*Upper Envelope*)

Dado que la función de valor $V_{t+1}$ presenta codos derivados de las elecciones discretas, $v_t(M, d)$ no es necesariamente cóncava en $M$. En consecuencia:
1. La función $a' \mapsto M(a'; d)$ no tiene garantizada la monotonicidad estricta: puede plegarse hacia atrás ($M(a'_{k+1}) < M(a'_k)$).
2. Para un nivel dado de recursos $M$, pueden existir múltiples valores de $a'$ que satisfacen la ecuación de Euler con igualdad. Solo uno de ellos representa el óptimo global; los restantes son mínimos locales o máximos locales inferiores.

El **algoritmo de la envolvente superior** poda estas ramas subóptimas:

```text
Malla endógena con pliegue no convexo:
  v(M)
   ^               Rama 1 (máx. local)           Rama 2 (máx. global)
   |                     /---\                     /----
   |                    /     \                   /
   |                   /       \                 /
   |                  /         \    Cruce      /
   |                 /           \     *       /
   |                /             \---/ \     /
   |               /                     \---/  <-- Rama subóptima descartada
   +----------------------------------------------------> M
                                       M*
```

#### Procedimiento de filtrado paso a paso

1. **Segmentación monótona**: Descompone la secuencia de puntos generada $(M_k, c_k, v_k)_{k=1}^K$ en segmentos conexos donde $M_k$ es estrictamente creciente.
2. **Evaluación del valor**: Para cada punto de la malla endógena, calcula el valor asociado a la opción:
   $$v_k = u(c_k, d) + \beta \, \mathfrak{W}(a'_k)$$
3. **Detección de intersecciones**: En los intervalos $[\underline{M}, \bar{M}]$ donde coexisten dos ramas, halla el punto de cruce $M^*$ donde $v_1(M^*) = v_2(M^*)$ mediante interpolación lineal.
4. **Eliminación por dominancia**: Por debajo del cruce $M^*$, descarta la rama de menor valor. Por encima de $M^*$, descarta la otra rama dominada.
5. **Interpolación continua**: Interpola la política limpia y monótona $c^*(M)$ sobre la malla exógena de evaluación continua de recursos $M \in \mathbf{M}_{\text{eval}}$.

---

## 2. Opciones metodológicas y de modelos

`DCEGMProblem` admite tanto problemas estacionarios de horizonte infinito como resolución hacia atrás en ciclo de vida finito:

| Parámetro / Característica | Descripción económica | Valor por defecto |
|---|---|---|
| `horizon` | Períodos de inducción hacia atrás $T$ (o `None` para horizonte infinito) | `None` (infinito) |
| `n_choices` | Número de alternativas discretas $D \ge 2$ | `2` (ej. trabajar vs jubilarse) |
| `sigma_eps` | Escala de los choques de valor extremo $\sigma_\epsilon \ge 0$ | `0.20` |
| `r` | Tipo de interés real (rendimiento bruto $R = 1 + r$) | `0.04` |
| `beta` | Factor de descuento subjetivo $\beta \in (0, 1)$ | `0.96` |
| `income` | Vector de ingresos salariales / pensiones $y(d)$ o $y(d, z)$ | `[1.0, 0.4]` |
| `discrete_transitions` | Matriz de factibilidad de transiciones discretas | `None` (todas factibles) |
| `a_grid` | Malla exógena de activos de ahorro $a' \ge a_{\min}$ | Array 1D obligatorio |

---

## 3. Calibración canónica: El modelo de decisión de jubilación

Especificación de referencia (**Iskhakov et al. 2017**):
- **Opción 0 (Trabajar)**: Salario $w = 1.0$, desutilidad del trabajo $\psi = 0.5$.
- **Opción 1 (Jubilarse)**: Pensión $p = 0.4$, desutilidad nula del trabajo ($\psi = 0$).
- **Preferencias**: Utilidad logarítmica CRRA $u(c, d) = \ln(c) - \psi \cdot \mathbf{1}\{d = 0\}$.
- **Invariantes de comportamiento económico**:
  - Los hogares con escasa riqueza no pueden permitirse jubilarse ($P(\text{trabajar} \mid M) \to 1$), ya que la pensión es insuficiente para suavizar el consumo.
  - Los hogares ricos se jubilan ($P(\text{jubilarse} \mid M) > 0.75$) para disfrutar del ocio sin mermar su nivel de vida.
  - La función de consumo presenta un salto o cambio de pendiente en el umbral crítico de jubilación $M^*$.

---

## 4. Ejemplos de uso ejecutables

### 4.1 Filtrado de la envolvente superior sobre una malla con pliegues

Este ejemplo demuestra el funcionamiento del algoritmo `upper_envelope` sobre una malla endógena sintética con dos ramas que se intersecan y un pliegue no monótono:

```python
import numpy as np
from puremacro.vfi import upper_envelope, UpperEnvelopeResult

# Rama 1 (valor empinado, óptimo con alta riqueza): v1(M) = -2.0 + 1.5 * M
M1 = np.linspace(1.5, 4.0, 30)
c1 = 0.6 * M1
v1 = -2.0 + 1.5 * M1

# Rama 2 (valor plano, óptimo con baja riqueza): v2(M) = 0.0 + 0.5 * M
M2 = np.linspace(0.5, 2.5, 30)
c2 = 0.3 * M2
v2 = 0.0 + 0.5 * M2

# Concatenación invertida para simular el pliegue no monótono del EGM
M_raw = np.concatenate([M2, M1])
c_raw = np.concatenate([c2, c1])
v_raw = np.concatenate([v2, v1])

# Malla exógena objetivo de recursos M
exog_grid = np.linspace(0.6, 3.5, 40)

# Filtrado mediante envolvente superior
res = upper_envelope(M_raw, c_raw, v_raw, exog_grid)
assert isinstance(res, UpperEnvelopeResult)
c_clean, v_clean = res

# Por debajo de la intersección (M < 2.0), la política sigue la Rama 2 (c = 0.3 * M)
below_kink = exog_grid < 1.9
assert np.allclose(c_clean[below_kink], 0.3 * exog_grid[below_kink], atol=1e-3)

# Por encima de la intersección (M > 2.1), la política sigue la Rama 1 (c = 0.6 * M)
above_kink = exog_grid > 2.1
assert np.allclose(c_clean[above_kink], 0.6 * exog_grid[above_kink], atol=1e-3)

print("¡La envolvente superior podó exitosamente la rama subóptima!")
```

### 4.2 Modelo de jubilación dinámico con choques de valor extremo

Resolución completa del modelo discreto-continuo de jubilación en horizonte infinito:

```python
import numpy as np
from puremacro.vfi import DCEGMProblem, solve_dcegm

# 1. Definición de la malla de ahorro y parámetros estructurales
a_grid = np.linspace(0.0, 12.0, 40)

prob = DCEGMProblem(
    a_grid=a_grid,
    n_choices=2,
    beta=0.96,
    r=0.04,
    sigma_eps=0.25,
    options={"tol": 1e-5, "max_iter": 500},
)

# 2. Resolución del problema dinámico mediante DC-EGM
sol = solve_dcegm(prob)
assert sol.converged
assert sol.sup_norm < 1e-4

# 3. Comprobación del comportamiento económico
# Opción 0 = Trabajar, Opción 1 = Jubilarse
p_work = sol.choice_probabilities[0]
p_retire = sol.choice_probabilities[1]

print(f"Iteraciones: {sol.n_iter} | Residuo final sup-norm: {sol.sup_norm:.2e}")
print(f"P(trabajar | riqueza mínima):  {p_work[0]:.3f}")
print(f"P(jubilarse | riqueza máxima): {p_retire[-1]:.3f}")

# Verificación de invariantes económicos
assert p_work[0] > 0.85, "Los hogares pobres deben trabajar"
assert p_retire[-1] > 0.70, "Los hogares ricos prefieren jubilarse"
assert np.all(sol.c > 0.0), "El consumo debe ser estrictamente positivo"
```

### 4.3 Evaluación continua y consulta de políticas

Evaluación continua de la política de consumo, función de valor y probabilidades logit para cualquier nivel de riqueza:

```python
# Consulta en un nivel continuo específico de riqueza
m_query = 3.5
c_work = sol.policy(m_query, choice=0)
c_retire = sol.policy(m_query, choice=1)
c_expected = sol.policy(m_query, choice=None)
v_inclusive = sol.value(m_query, choice=None)
p_retire_val = sol.choice_prob(m_query, choice=1)

print(f"En M = {m_query:.2f}:")
print(f"  Consumo si trabaja:       c_0 = {c_work:.3f}")
print(f"  Consumo si está jubilado: c_1 = {c_retire:.3f}")
print(f"  Consumo esperado:         E[c] = {c_expected:.3f}")
print(f"  Probabilidad de jubilarse: P(d=1) = {p_retire_val:.1%}")

# Evaluación continua vectorizada
m_vec = np.linspace(1.0, 8.0, 10)
p_work_vec = sol.choice_prob(m_vec, choice=0)
assert len(p_work_vec) == 10
assert np.all((p_work_vec >= 0.0) & (p_work_vec <= 1.0))
```

---

## 5. Especificación completa de la API

### `upper_envelope`

```text
upper_envelope(
    m_raw: np.ndarray,
    c_raw: np.ndarray,
    v_raw: np.ndarray,
    m_grid: np.ndarray,
) -> UpperEnvelopeResult
```

#### Parámetros:
- `m_raw`: Array 1D de recursos endógenos procedentes de la inversión de Euler.
- `c_raw`: Array 1D de consumos candidatos correspondientes a `m_raw`.
- `v_raw`: Array 1D de valores candidatos correspondientes a `m_raw`.
- `m_grid`: Malla objetivo 1D estrictamente creciente de evaluación.

#### Retorna:
- `UpperEnvelopeResult`: 2-tupla `(c_clean, v_clean)` con la política limpia monótona y la envolvente superior de la función de valor. Admite acceso por atributos `.c` y `.v`.

---

### `DCEGMProblem`

```text
DCEGMProblem(
    a_grid: Sequence[float] | np.ndarray,
    m_grid: Sequence[float] | np.ndarray | None = None,
    n_choices: int = 2,
    u_fn: Callable | Sequence[Callable] | None = None,
    u_prime: Callable | Sequence[Callable] | None = None,
    u_prime_inv: Callable | Sequence[Callable] | None = None,
    beta: float = 0.96,
    r: float = 0.04,
    income: Any = None,
    P_z: np.ndarray | None = None,
    z_grid: Sequence[float] | np.ndarray | None = None,
    sigma_eps: float = 0.2,
    a_min: float = 0.0,
    horizon: int | None = None,
    discrete_transitions: np.ndarray | None = None,
    options: dict | None = None,
    metadata: dict | None = None,
)
```

#### Parámetros:
- `a_grid`: Malla exógena 1D estrictamente creciente de activos de ahorro $a' \ge a_{\min}$.
- `m_grid`: Malla objetivo de evaluación de recursos líquidos $M$. Si es `None`, se genera automáticamente cubriendo el rango factible de riqueza.
- `n_choices`: Número de elecciones discretas alternativas $D \ge 2$.
- `u_fn`: Función de utilidad $u(c, d)$. Por defecto utilidad logarítmica con desutilidad laboral.
- `u_prime`: Función de utilidad marginal $u'(c, d)$.
- `u_prime_inv`: Inversa de la utilidad marginal $(u')^{-1}(w, d)$.
- `beta`: Factor de descuento intertemporal $\beta \in (0, 1)$.
- `r`: Tipo de interés real $r > -1$.
- `income`: Esquema de ingresos $y(d)$ o $y(d, z)$ (por defecto `[1.0, 0.4]`).
- `P_z`: Matriz de probabilidades de transición markoviana para perturbaciones exógenas de ingresos.
- `z_grid`: Soporte discreto de las perturbaciones de ingresos.
- `sigma_eps`: Parámetro de escala de los choques de Gumbel $\sigma_\epsilon \ge 0$.
- `a_min`: Límite inferior de endeudamiento sobre los activos.
- `horizon`: Número de períodos $T$ para inducción finita hacia atrás, o `None` para modelo estacionario infinito.
- `discrete_transitions`: Matriz de transiciones admisibles entre opciones discretas.
- `options`: Opciones del algoritmo (`tol`, `max_iter`).

---

### `solve_dcegm`

```text
solve_dcegm(
    problem_or_a_grid: DCEGMProblem | Sequence[float] | np.ndarray,
    m_grid: Sequence[float] | np.ndarray | None = None,
    *args: Any,
    backend: str = "numpy",
    **kwargs: Any,
) -> DCEGMSolution
```

Resuelve el problema continuo-discreto mediante DC-EGM y filtrado por envolvente superior.

---

## 6. Interfaz de resultados y exportación a manuscritos

`DCEGMSolution` proporciona métodos exhaustivos de inspección, evaluación y presentación:

- **Atributos de solución**:
  - `.c`: Diccionario `ChoiceMapping` con las funciones de política de consumo podadas $c(M, d)$.
  - `.choice_values`: Funciones de valor condicionales $v_d(M)$.
  - `.choice_probabilities`: Probabilidades logit de elección $P(d \mid M)$.
  - `.integrated_value`: Array 1D con el valor inclusivo $\mathcal{V}(M)$.
  - `.aprime`: Políticas de ahorro $a'(M, d) = M - c(M, d)$.
  - `.converged`: Booleano que certifica la convergencia.
  - `.n_iter`: Número de iteraciones evaluadas.
  - `.sup_norm`: Residuo final del supremo de Bellman.
- **Evaluación continua**:
  - `.policy(m, choice=None)`: Evalúa el consumo. Devuelve la política específica $c(m, d)$ si `choice` es un entero, o el consumo esperado ponderado $\mathbb{E}[c(m)]$ si `choice=None`.
  - `.value(m, choice=None)`: Evalúa el valor condicional $v_d(m)$ o el valor inclusivo $\mathcal{V}(m)$.
  - `.choice_prob(m, choice=None)`: Evalúa la probabilidad de elección $P(d \mid m)$ o el vector de probabilidades si `choice=None`.
- **Exportación académica**:
  - `.summary()`: Resumen textual estructurado de elecciones, convergencia y estadísticas de política.
  - `.plot()`: Gráfico multipanel en Matplotlib con políticas de consumo, probabilidades de elección y funciones de valor.
  - `.to_frame()`: Exporta las políticas, valores y probabilidades a un `pandas.DataFrame`.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas preparadas para su inclusión directa en artículos y publicaciones científicas.

---

## Referencias bibliográficas

- **Carroll, C. D. (2006)**. The method of endogenous gridpoints for solving dynamic stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- **Clausen, A., & Strub, C. (2020)**. A continuous method for discrete choice dynamic programming. *Quantitative Economics*, 11(3), 859–894.
- **Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017)**. The endogenous grid method for discrete-continuous dynamic choice models with taste shocks. *Quantitative Economics*, 8(2), 317–365.
- **Rust, J. (1987)**. Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. *Econometrica*, 55(5), 999–1033.
