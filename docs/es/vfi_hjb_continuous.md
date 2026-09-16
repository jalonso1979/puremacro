> 🇬🇧 [English](../vfi_hjb_continuous.md) · 🇪🇸 Español

# HJB en tiempo continuo, KFE adjunta y equilibrio general de Aiyagari continuo

`puremacro.vfi.hjb_achdou` (reexportado desde `puremacro.vfi`) implementa el conjunto de herramientas para agentes heterogéneos en tiempo continuo de **Achdou, Han, Lasry, Lions y Moll (2022, *Review of Economic Studies*)**: un solucionador implícito de diferencias finitas contra el viento (*upwind*) para la ecuación estacionaria de Hamilton-Jacobi-Bellman (HJB) del problema de fluctuaciones del ingreso (`solve_hjb_achdou`), la ecuación de Kolmogorov hacia adelante (KFE, *Kolmogorov Forward Equation*) adjunta que entrega la distribución estacionaria de la riqueza sin simulación (`solve_kfe_achdou`) y un algoritmo de búsqueda de raíces del equilibrio general para la economía de **Aiyagari (1994)** / **Huggett (1993)** en tiempo continuo (`solve_aiyagari_continuous_hjb`). Los resultados se devuelven como las `dataclass` inmutables `HJBSolution` y `AiyagariContinuousHJBResult`, que conservan el acceso al estilo de diccionario para el código escrito contra la interfaz anterior a 3.4.0.

En tiempo continuo, el problema del hogar es una ecuación en derivadas parciales de primer orden y no un operador de Bellman sobre una malla de elecciones del periodo siguiente. Al discretizarla con un esquema upwind, cada paso de mejora de la política se convierte en una única resolución de un sistema lineal disperso, y el mismo generador infinitesimal $\mathbf{A}$ que empuja la función de valor hacia atrás en el tiempo empuja la distribución transversal hacia adelante. La densidad estacionaria es, por tanto, un vector nulo de $\mathbf{A}^\top$, que se obtiene con precisión de máquina y sin el ruido de Monte Carlo. Esta página es la contraparte en tiempo continuo de [`vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md), que cubre la simulación no estocástica de Young (2010) en tiempo discreto y `solve_aiyagari_continuous`; la Sección 2.2 detalla cómo se relacionan ambas y dónde encajan los cuadernos.

---

## 1. Marco teórico y algorítmico

### 1.1 La ecuación HJB estacionaria

Un hogar descuenta a la tasa $\rho > 0$, tiene utilidad CRRA $u(c) = c^{1-\gamma}/(1-\gamma)$ (con $u(c) = \log c$ cuando $\gamma = 1$), mantiene activos $a \in [a_{\min}, a_{\max}]$ que rinden la tasa de interés $r$ y recibe ingreso laboral $w e_j$, donde el estado de productividad $e_j \in \{e_1, \dots, e_{N_e}\}$ sigue una cadena de Markov en tiempo continuo con generador $\boldsymbol{\Lambda} = (\lambda_{jk})$: $\lambda_{jk} \ge 0$ para $k \ne j$ y cada fila suma cero. La función de valor $v_j(a) = v(a, e_j)$ resuelve

$$\rho\, v_j(a) = \max_{c \ge 0} \Big\{ u(c) + v_j'(a)\, s_j(a) \Big\} + \sum_{k \ne j} \lambda_{jk} \big[ v_k(a) - v_j(a) \big], \qquad s_j(a) = r a + w e_j - c .$$

La condición de primer orden entrega la regla de consumo $c_j(a) = \big(v_j'(a)\big)^{-1/\gamma}$ y la deriva del ahorro $s_j(a) = r a + w e_j - c_j(a)$. El límite de endeudamiento $a \ge a_{\min}$ entra como una *condición de frontera de restricción de estado* (Achdou et al. 2022): la deriva no puede apuntar fuera del espacio de estados en $a_{\min}$, lo que equivale a $v_j'(a_{\min}) \ge u'(r a_{\min} + w e_j)$. La condición especular $v_j'(a_{\max}) \le u'(r a_{\max} + w e_j)$ (ausencia de deriva hacia afuera en el extremo superior) se impone en la frontera superior artificial $a_{\max}$.

### 1.2 Diferencias finitas upwind y la frontera de restricción de estado

Sobre una malla de activos $a_1 < a_2 < \dots < a_{N_a}$ con espaciamientos locales $\Delta a_i^{+} = a_{i+1} - a_i$ y $\Delta a_i^{-} = a_i - a_{i-1}$, la derivada se aproxima de forma unilateral en la dirección en que se mueve el estado («contra el viento» o *upwind*):

$$v'_{i,j,F} = \frac{v_{i+1,j} - v_{i,j}}{\Delta a_i^{+}}, \qquad v'_{i,j,B} = \frac{v_{i,j} - v_{i-1,j}}{\Delta a_i^{-}} .$$

Cada derivada candidata implica un nivel de consumo $c = (v')^{-1/\gamma}$ y una deriva $s_F$ o $s_B$. El esquema usa la diferencia hacia adelante donde $s_{F} > 0$, la diferencia hacia atrás donde $s_{B} < 0$ y, en caso contrario, fija la deriva en cero con $c_{i,j} = r a_i + w e_j$ (el hogar consume su ingreso corriente y permanece donde está). En los dos extremos de la malla, la derivada unilateral que falta se reemplaza por $u'(r a_i + w e_j)$, lo que hace que la deriva correspondiente sea exactamente cero, de modo que ningún flujo de probabilidad abandona jamás $[a_{\min}, a_{\max}]$. `solve_hjb_achdou` además enmascara la rama hacia adelante en $a_{\max}$ y la rama hacia atrás en $a_{\min}$, de modo que las filas de frontera solo pueden llevar una deriva que apunte hacia adentro o se anule. Por eso, en el Ejemplo 4.1, el hogar de ingreso bajo en $a = 0$ tiene $c = 0.2 = w e_1$ y $s = 0$ exactamente.

Se dispone de dos alternativas al tratamiento de restricción de estado:

- **Derivadas de frontera prescritas** (`v_prime_boundary=(v'_{\min}, v'_{\max})`): condiciones de Neumann. Las máscaras se desactivan, se usa la derivada dada en la fila de frontera y cualquier flujo hacia afuera $s \cdot v'_{\text{frontera}}$ se traslada al lado derecho del sistema lineal como un término inhomogéneo. Una entrada dejada en `None` recurre a la derivada de restricción de estado en ese lado.
- **Referencia del problema del pastel (*cake-eating*)** (`w_rate=0.0`): sin ingreso laboral el problema tiene la forma cerrada $c(a) = \mu a$, $\mu = (\rho - (1-\gamma) r)/\gamma$. El solucionador inicializa entonces $v$ a partir de esa solución, desactiva las máscaras de frontera (como en el caso de Neumann) y usa las utilidades marginales analíticas $(\mu a_{\min})^{-\gamma}$ y $(\mu a_{\max})^{-\gamma}$ en las filas de frontera, que es como `tests/test_hjb_implicit.py` valida el esquema contra una solución exacta. Como $(\mu a)^{-\gamma}$ es singular en el origen, este modo exige una malla que empiece estrictamente por encima de cero: `w_rate=0.0` con `a_grid[0] <= 0` lanza `ValueError` indicando que pase `a_min > 0`.

### 1.3 El esquema implícito y la propiedad de M-matriz

Al reunir las derivas upwind en el generador infinitesimal disperso $\mathbf{A}^n$ (una matriz de $N_a N_e \times N_a N_e$ indexada en orden por columnas, estado $(i, j) \mapsto j N_a + i$), cada fila del bloque de activos tiene entradas

$$x_{i,j} = \frac{\max(-s_{i,j}, 0)}{\Delta a_i^{-}}, \qquad z_{i,j} = \frac{\max(s_{i,j}, 0)}{\Delta a_i^{+}}, \qquad y_{i,j} = -x_{i,j} - z_{i,j},$$

colocadas en la subdiagonal, la superdiagonal y la diagonal, respectivamente, y el proceso de ingreso aporta $\boldsymbol{\Lambda} \otimes \mathbf{I}_{N_a}$. Las entradas fuera de la diagonal son no negativas y cada fila de $\mathbf{A}^n$ suma cero (las sumas por fila del generador devuelto en el Ejemplo 4.1 son cero hasta $2.2 \times 10^{-16}$). En lugar de avanzar la función de valor de forma explícita, lo que está sujeto a una restricción de Courant-Friedrichs-Lewy $\Delta t = O(\Delta a)$, la actualización es implícita en $v^{n+1}$:

$$\Big[ \Big(\rho + \frac{1}{\Delta}\Big) \mathbf{I} - \mathbf{A}^n \Big] v^{n+1} = u(c^n) + \frac{1}{\Delta} v^n .$$

Con derivadas de frontera prescritas, el término de flujo hacia afuera de la Sección 1.2 se añade al lado derecho; bajo la restricción de estado es idénticamente cero. La matriz $\mathbf{B}^n = (\rho + 1/\Delta)\mathbf{I} - \mathbf{A}^n$ tiene diagonal positiva, entradas fuera de la diagonal no positivas y sumas por fila iguales a $\rho + 1/\Delta > 0$, de modo que es estrictamente diagonalmente dominante y, por tanto, una **M-matriz** no singular con $(\mathbf{B}^n)^{-1} \ge 0$ elemento a elemento. Una inversa no negativa hace que el esquema discreto sea *monótono*, y el teorema de Barles-Souganidis (1991) garantiza entonces que un esquema monótono, estable y consistente converge a la solución de viscosidad de la ecuación HJB a medida que se refina la malla. `solve_hjb_achdou` factoriza $\mathbf{B}^n$ con `scipy.sparse.linalg.spsolve` (almacenamiento CSC) en cada iteración y se detiene cuando $\| v^{n+1} - v^n \|_\infty <$ `tol`.

**El papel de $\Delta$.** Cuando $\Delta \to \infty$ la actualización se convierte en la iteración de políticas de Howard, es decir, el método de Newton sobre la ecuación HJB, razón por la cual el valor por defecto `Delta=1e4` converge en un puñado de iteraciones; un $\Delta$ pequeño convierte el mismo sistema en una marcha amortiguada en pseudotiempo que necesita muchos más barridos. Medido con los valores por defecto (`Na=100`, `tol=1e-8`, `max_iter=100`; Ejemplo 4.5):

| `Delta` | Iteraciones | Convergió | $\max \lvert v - v_{\Delta = 10^4} \rvert$ |
|---|---|---|---|
| $10^4$ | 9 | sí | referencia |
| $10^2$ | 14 | sí | $9.8 \times 10^{-10}$ |
| $10$ | 50 | sí | $1.8 \times 10^{-8}$ |
| $1$ | 100 | **no** (necesita 360 con `max_iter=1000`) | $6.7 \times 10^{-2}$ al alcanzar el tope |

El esquema es *incondicionalmente estable* para todo $\Delta > 0$; lo que depende de $\Delta$ es el *número* de iteraciones, no la estabilidad.

### 1.4 La KFE adjunta y la normalización de la masa

La distribución conjunta estacionaria $g(a, e)$ del proceso controlado es el vector nulo del generador adjunto,

$$\mathbf{A}^\top g = 0, \qquad \sum_{i=1}^{N_a} \sum_{j=1}^{N_e} g_{i,j}\, w_i = 1, \qquad g_{i,j} \ge 0,$$

donde los $w_i$ son los pesos de cuadratura por ancho de celda de la malla de activos,

$$w_i = \tfrac{1}{2}\big(\Delta a_{i-1} + \Delta a_i\big), \qquad \Delta a_{-1} \equiv \Delta a_0, \quad \Delta a_{N_a} \equiv \Delta a_{N_a - 1},$$

de modo que $w_i = \Delta a$ en todos los nodos de una malla uniforme. El generador actúa sobre los *nodos* de la malla, así que el vector nulo de $\mathbf{A}^\top$ es el vector de masas nodales estacionarias $\pi_{i,j}$, no una densidad. Por eso `solve_kfe_achdou` toma el generador de la última iteración de la HJB, reemplaza la primera fila de $\mathbf{A}^\top$ por una fila de unos y la primera entrada del lado derecho por $1$ (el reemplazo estándar de una ecuación redundante por la normalización $\sum \pi = 1$), resuelve con `spsolve`, recorta a cero cualquier negativo de redondeo, reescala $\pi$ para que sume exactamente uno y solo entonces convierte a **densidad** dividiendo por el ancho local de la celda,

$$g_{i,j} = \pi_{i,j} / w_i .$$

Así $\sum_{i,j} g_{i,j} w_i = 1$ y $\sum_i g_{i,j} w_i = \pi_j$ (la distribución estacionaria del generador del ingreso) en cualquier malla, uniforme o no, y los agregados de la forma $\sum_{i,j} a_i g_{i,j} w_i$ son la aproximación por cuadratura correcta de $\int a\, g(a, e)\, da$ — los mismos pesos que `solve_aiyagari_continuous_hjb` usa para la oferta de capital. En una malla uniforme ambas lecturas coinciden: $w_i = \Delta a$, la masa de probabilidad en el nodo $i$ es $g_{i,j} \Delta a$, y $\sum_{i,j} g_{i,j} \Delta a = 1$ (el Ejemplo 4.2 imprime `1.000000000000`).

El `mass_residual` reportado es $\lvert \sum_{i,j} g_{i,j} w_i - 1 \rvert$ medido *después* de esa normalización, por lo que es del orden de $10^{-16}$ por construcción (`1.11e-16` en el Ejemplo 4.1); certifica la normalización, no la exactitud de la resolución lineal. Antes de resolver, `solve_kfe_achdou` comprueba además que el grafo de transiciones del generador tenga exactamente una clase comunicante cerrada, que es la condición exacta para que el espacio nulo sea unidimensional, y lanza `ValueError` en caso contrario: un generador de ingreso reducible como `A_z=np.zeros((2, 2))` se rechaza con un mensaje que indica el número de clases cerradas, en lugar de devolver un vector nulo arbitrario.

### 1.5 Equilibrio general: búsqueda de raíces para $r$ con la condición de primer orden de la empresa

Las empresas producen $Y = K^\alpha L^{1-\alpha}$ y alquilan capital a $r + \delta$, de modo que, dada una tasa de interés tentativa, la razón capital-trabajo y el salario se siguen de las condiciones de primer orden de la empresa:

$$\frac{K^d(r)}{L} = \Big( \frac{\alpha}{r + \delta} \Big)^{\frac{1}{1-\alpha}}, \qquad w(r) = (1-\alpha) \Big( \frac{K^d(r)}{L} \Big)^{\alpha}, \qquad L = \sum_{j} e_j\, \pi_j ,$$

donde $\boldsymbol{\pi}$ es la distribución estacionaria del generador del ingreso ($\boldsymbol{\pi}^\top \boldsymbol{\Lambda} = 0$; con el proceso simétrico de dos estados por defecto, $\pi = (0.5, 0.5)$ y $L = 0.6$). Para cada $r$ tentativo, `solve_aiyagari_continuous_hjb` calcula $w(r)$, llama a `solve_hjb_achdou` (con la KFE), agrega la riqueza de los hogares $K^s(r) = \sum_{i,j} a_i\, g_{i,j}(r)\, w_i$ con los pesos de cuadratura de la Sección 1.4 y evalúa el exceso de oferta $\Phi(r) = K^s(r) - K^d(r)$. El método de Brent (`scipy.optimize.root_scalar`, `method="brentq"`, a lo sumo `max_iter_ge` iteraciones por pasada) localiza $r^\ast$ con $\Phi(r^\ast) = 0$ en el intervalo $[r_{\min}, r_{\max}]$, cuyo extremo superior por defecto es $\rho - 0.002$: con una restricción de endeudamiento, la distribución estacionaria solo existe para $r < \rho$, porque con $r \ge \rho$ los hogares acumularían sin límite. Antes de llamar a Brent, el solucionador comprueba el signo de $\Phi$ en ambos extremos; si $\Phi(r_{\min}) > 0$ divide $r_{\min}$ a la mitad (sin bajar de $0.001$, a lo sumo cinco veces) y si $\Phi(r_{\max}) < 0$ desplaza $r_{\max}$ hasta la mitad del camino hacia $\rho$ (a lo sumo cinco veces). Un intervalo inválido ($r_{\min} \ge r_{\max}$, $r_{\max} \ge \rho$ o $r_{\min} \le -\delta$) lanza `ValueError`. Si no se encuentra ningún cambio de signo, devuelve la tasa tentativa almacenada en caché con el menor $\lvert \Phi \rvert$. Cada solución HJB se almacena en caché por $r$, de modo que la ejecución del Ejemplo 4.3 necesitó siete resoluciones del problema del hogar para seis iteraciones de Brent.

**`tol_ge` gobierna el residuo de vaciado del mercado.** No es un argumento decorativo: la tolerancia en $x$ que se entrega a Brent se deriva de `tol_ge` y de la pendiente secante del intervalo, `xtol = min(1e-5, max(0.5 * tol_ge / slope, 1e-12))`, y tras la primera pasada el solucionador vuelve a acotar el intervalo con sus evaluaciones en caché y reejecuta Brent —hasta tres pasadas más— hasta que $\lvert K^s(r^\ast) - K^d(r^\ast) \rvert <$ `tol_ge`. El `converged` devuelto es `True` solo cuando Brent convergió **y** se cumple esa desigualdad, de modo que significa lo que dice: el mercado de capital se vacía con tolerancia `tol_ge`. Ajustar la tolerancia ajusta el residuo: con la calibración por defecto, `tol_ge=1e-4` se detiene en $\Phi = -1.76 \times 10^{-6}$ tras 6 iteraciones de Brent, mientras que `tol_ge=1e-8` alcanza $\Phi = -7.4 \times 10^{-13}$ en 7.

---

## 2. Opciones metodológicas y del modelo

### 2.1 Tabla de opciones

| Característica | Argumento(s) | Valor por defecto | Notas |
|---|---|---|---|
| **Proceso de ingreso** | `e_grid`, `A_z` | `[0.2, 1.0]`, conmutación de Poisson simétrica con intensidad $0.1$ | Cualquier $N_e$. `A_z` se valida: forma $(N_e, N_e)$, entradas finitas, entradas fuera de la diagonal no negativas y sumas por fila iguales a cero (`ValueError` en caso contrario, con una pista si pasó una matriz de probabilidades de transición). Con `A_z=None` y $N_e > 2$ el valor por defecto es $\lambda_{jk} = 0.2/(N_e-1)$ fuera de la diagonal y $-0.2$ en ella. |
| **Malla de activos** | `Na`, `a_min`, `a_max`, `a_grid` | 100 nodos en $[0, 30]$, uniforme | `a_grid` anula los tres escalares y puede ser no uniforme; la KFE devuelve una densidad en ambos casos (Sección 1.4). Debe ser estrictamente creciente y tener al menos dos puntos. |
| **Tratamiento de la frontera** | `v_prime_boundary`, `w_rate` | restricción de estado en ambos extremos | Una tupla de derivadas cambia a Neumann; `w_rate=0.0` selecciona las fronteras analíticas del problema del pastel y entonces exige `a_grid[0] > 0`. |
| **Paso implícito** | `Delta` | $10^4$ | Cuanto mayor, más cerca de la iteración de políticas (menos barridos); véase la tabla de la Sección 1.3. |
| **Regla de parada** | `max_iter`, `tol` | 100, $10^{-8}$ | Norma del supremo sobre funciones de valor sucesivas; `max_iter` debe ser un entero $\ge 1$. En 3.3.0 los valores por defecto eran 500 y $10^{-7}$. |
| **Distribución** | `compute_kfe` | `True` | `False` deja `g_dist=None` y `mass_residual=0.0`; `A_generator` se devuelve de todos modos, de modo que `solve_kfe_achdou` puede llamarse después. |
| **Intervalo del equilibrio general** | `r_min`, `r_max`, `tol_ge`, `max_iter_ge` | $0.005$, $\rho - 0.002$, $10^{-4}$, 40 | `tol_ge` es la tolerancia de vaciado del mercado: fija la tolerancia en $x$ de Brent y `converged` significa $\lvert K^s - K^d \rvert <$ `tol_ge` (Sección 1.5). |
| **Resolución del hogar dentro del equilibrio general** | `max_iter_hjb`, `tol_hjb`, `Delta` | 100, $10^{-8}$, $10^4$ | Se pasan a `solve_hjb_achdou` en cada tasa tentativa. |

### 2.2 Relación con las páginas en tiempo discreto y con los cuadernos

- [`vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md) resuelve la misma economía de Aiyagari en **tiempo discreto**: los hogares descuentan con $\beta = 0.96$, la política continua de ahorro se proyecta sobre un histograma con las loterías lineales de Young (2010) y `solve_aiyagari_continuous` vacía el mercado de capital con Brent. Ese módulo también incluye `ContinuousStationaryDistribution.gini()` y `.lorenz()`. La página que está leyendo reemplaza la proyección por loterías con la KFE adjunta y el operador de Bellman con el generador de la HJB; aquí no hay una clase auxiliar de distribución, de modo que los estadísticos de desigualdad se calculan directamente a partir de `g_dist`, como en el Ejemplo 4.2. Las calibraciones difieren ($\rho = 0.05$ frente a $\beta = 0.96$, dos estados de ingreso de Poisson frente a una cadena de Tauchen o Rouwenhorst), por lo que las cifras de equilibrio no son comparables entre las dos páginas.
- `notebooks/22_continuous_time_hjb.py` (y su gemelo `_es`) introduce el esquema upwind en el problema de fluctuaciones del ingreso. Se escribió contra la interfaz 3.3.0: lee el resultado con `results["V"]` y pasa explícitamente `max_iter=500, tol=1e-7`, que son los valores por defecto antiguos. Ambas cosas siguen funcionando porque `HJBSolution` implementa el protocolo `Mapping` (Sección 6). Sus *cifras*, en cambio, no son las de 3.3.0: el solucionador explícito de 3.3.0 no tenía término de conmutación del ingreso, de modo que cada estado de productividad era un problema separado de ingreso determinista, mientras que desde 3.4.0 los estados están ligados por el generador de Poisson `A_z` (por defecto, simétrico con $\lambda = 0.1$). Pasar `A_z=np.zeros((Ne, Ne))` junto con `compute_kfe=False` reproduce la economía de 3.3.0; hay que desactivar la KFE porque un generador sin conmutación no tiene una distribución estacionaria única (Sección 1.4).
- `notebooks/56_implicit_hjb_and_continuous_kfe.py` (en inglés y en español) es el cuaderno de demostración de este módulo: iteración implícita de la HJB, distribución de la riqueza por KFE con curva de Lorenz y coeficiente de Gini, el problema del pastel de referencia frente a la forma cerrada y el equilibrio general de Aiyagari continuo con curvas de oferta y demanda. La entrada del catálogo está en [`notebooks.md`](notebooks.md).

---

## 3. Calibración canónica

Los valores por defecto configuran un problema de fluctuaciones del ingreso con dos estados, en el espíritu de Achdou et al. (2022), con parámetros de frecuencia anual:

| Parámetro | Símbolo | Valor por defecto | Argumento |
|---|---|---|---|
| Tasa de descuento | $\rho$ | $0.05$ | `rho_val` |
| Aversión relativa al riesgo | $\gamma$ | $2.0$ | `gamma_r` |
| Tasa de interés (equilibrio parcial) | $r$ | $0.03$ | `r_rate` |
| Salario (equilibrio parcial) | $w$ | $1.0$ | `w_rate` |
| Estados de productividad | $e$ | $\{0.2, 1.0\}$ | `e_grid` |
| Intensidades de conmutación | $\lambda_{12} = \lambda_{21}$ | $0.1$ | `A_z` |
| Malla de activos | $[a_{\min}, a_{\max}]$, $N_a$ | $[0, 30]$, 100 nodos uniformes | `a_min`, `a_max`, `Na` |
| Paso implícito | $\Delta$ | $10^4$ | `Delta` |
| Participación del capital (equilibrio general) | $\alpha$ | $0.33$ | `alpha` |
| Depreciación (equilibrio general) | $\delta$ | $0.05$ | `delta` |
| Intervalo de la tasa de interés (equilibrio general) | $[r_{\min}, r_{\max}]$ | $[0.005, 0.048]$ | `r_min`, `r_max` |

---

## 4. Ejemplos prácticos ejecutables

Todos los ejemplos son deterministas (no hay generación de números aleatorios en ningún punto del módulo), no necesitan acceso a la red y terminan en bastante menos de una décima de segundo de tiempo de solucionador en un portátil; las cifras citadas a continuación son las que imprimió cada script.

### 4.1 Resolución en equilibrio parcial y `summary()`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
print(sol.n_iter, sol.converged, f"{sol.mass_residual:.1e}")
print(sol.V.shape, sol.c_policy.shape, sol.g_dist.shape, sol.A_generator.shape)
print(sol.summary())

# El acceso heredado al estilo de diccionario sigue funcionando (12 claves); los campos nuevos son atributos.
print(sol["n_iter"] == sol.n_iter, list(sol.keys())[:4], "g_dist" in sol, "g_dist" in sol.keys())
print(f"c(0, e_low) = {sol.c_policy[0, 0]:.4f}   c(0, e_high) = {sol.c_policy[0, 1]:.4f}")
print(f"s(0, e_low) = {sol.s_drift[0, 0]:.4f}   s(0, e_high) = {sol.s_drift[0, 1]:.4f}")
```

El solucionador converge en 9 iteraciones a `tol=1e-8` con un residuo de masa de `1.1e-16`; los arrays son `(100, 2)` y el generador es una matriz dispersa `(200, 200)`. `summary()` devuelve un `DataFrame` indexado por métrica: 200 estados, tiempo transcurrido 0.0144 s (dependiente de la máquina), valor medio $-22.000181$, consumo entre $0.200000$ y $1.972175$. Las dos últimas líneas muestran la restricción de estado en acción: el hogar de ingreso bajo en $a = 0$ consume exactamente su ingreso laboral $0.2000$ con deriva cero, mientras que el hogar de ingreso alto consume $0.4896$ y ahorra a la tasa $0.5104$. `"g_dist" in sol` es `True` pero `"g_dist" in sol.keys()` es `False`, porque `keys()` enumera solo las doce claves heredadas, mientras que la pertenencia y el subíndice cubren todos los campos de la dataclass (Sección 6).

### 4.2 Curva de Lorenz y coeficiente de Gini a partir de `g_dist`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
a, g = sol.a_grid, sol.g_dist
da = a[1] - a[0]                      # malla uniforme: masa nodal = g * da
mass_a = g.sum(axis=1) * da           # distribución marginal de la riqueza, suma uno
mean_a = float(np.sum(a * mass_a))

# Curva de Lorenz: participación acumulada de la población frente a participación acumulada de la riqueza
pop = np.concatenate([[0.0], np.cumsum(mass_a)])
wealth = np.concatenate([[0.0], np.cumsum(a * mass_a) / mean_a])
gini = float(1.0 - np.sum((wealth[1:] + wealth[:-1]) * np.diff(pop)))

top10 = 1.0 - float(np.interp(0.90, pop, wealth))
print(f"mass total = {mass_a.sum():.12f}")
print(f"mean wealth (capital supply) = {mean_a:.4f}")
print(f"share of households at a = 0 : {mass_a[0]:.4f}")
print(f"Gini coefficient of wealth   : {gini:.4f}")
print(f"wealth share of the top 10%  : {top10:.4f}")
print(f"stationary income shares     : {g.sum(axis=0) * da}")
```

Las masas nodales suman `1.000000000000`. Con $r = 0.03$ y $w = 1$, la riqueza media (la oferta de capital en equilibrio parcial) es $6.2150$, el $3.82\%$ de los hogares se sitúa exactamente en el límite de endeudamiento, el coeficiente de Gini de la riqueza es $0.3611$ y el decil superior posee el $21.99\%$ de la riqueza. La marginal del ingreso es `[0.5 0.5]`, la distribución estacionaria de la cadena simétrica de dos estados. La fórmula del Gini es el área trapezoidal entre la curva de Lorenz $(p, L(p))$ y la diagonal, la misma expresión que usa el cuaderno 56.

### 4.3 Equilibrio general: $r^\ast$, $w^\ast$, $K^\ast$

```python
from puremacro.vfi import solve_aiyagari_continuous_hjb

ge = solve_aiyagari_continuous_hjb(alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0, Na=100, a_max=30.0)
print(f"r* = {ge.r_star:.5f}  w* = {ge.w_star:.4f}  K* = {ge.K_star:.4f}  L* = {ge.L_star:.3f}  Y* = {ge.Y_star:.4f}")
print(f"excess capital = {ge.excess_capital:.2e}  converged = {ge.converged}  Brent iterations = {ge.n_iter_ge}")
print(f"K/Y = {ge.K_star / ge.Y_star:.3f}   household HJB iterations at r*: {ge.hjb_solution.n_iter}")
print(ge.to_markdown())
```

El mercado se vacía en $r^\ast = 0.02004$, $w^\ast = 1.4376$, $K^\ast = 6.0656$ con $L^\ast = 0.600$ e $Y^\ast = 1.2874$ (razón capital-producto $4.712$); el exceso de oferta en la raíz es $-1.76 \times 10^{-6}$, muy dentro del `tol_ge=1e-4` por defecto, de modo que `converged` es `True` tras 6 iteraciones de Brent, y el problema del hogar en $r^\ast$ requirió 10 iteraciones de la HJB. Pasar `tol_ge=1e-8` lleva la misma ejecución a $\Phi = -7.4 \times 10^{-13}$ en 7 iteraciones. `to_markdown()` genera la tabla resumen (la fila del tiempo transcurrido depende de la máquina):

```text
|                         Metric |     Value |
|--------------------------------|-----------|
| Equilibrium Interest Rate (r*) |  0.020041 |
|          Equilibrium Wage (w*) |  1.437585 |
|         Aggregate Capital (K*) |  6.065605 |
|            Capital Supply (Ks) |  6.065605 |
|            Capital Demand (Kd) |  6.065607 |
|  Capital Market Clearing Error | -1.76e-06 |
|           Aggregate Labor (L*) |  0.600000 |
|          Aggregate Output (Y*) |  1.287389 |
|      GE Root-finding Converged |      True |
|                  GE Iterations |         6 |
|               Elapsed Time (s) |    0.0421 |
```

### 4.4 Un proceso de ingreso personalizado: pasar `e_grid` y `A_z`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

e_grid = np.array([0.5, 1.0, 1.5])
A_z = np.array([[-0.20, 0.20, 0.00],
                [ 0.10,-0.20, 0.10],
                [ 0.00, 0.20,-0.20]])          # generador: fuera de la diagonal >= 0, las filas suman cero
assert np.allclose(A_z.sum(axis=1), 0.0)

sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, Na=100, a_max=30.0, e_grid=e_grid, A_z=A_z)
da = sol.a_grid[1] - sol.a_grid[0]
print(sol.n_iter, sol.converged, sol.V.shape)
print("stationary income shares:", np.round(sol.g_dist.sum(axis=0) * da, 4))
print("mean wealth by income state:", np.round((sol.a_grid[:, None] * sol.g_dist * da).sum(axis=0) / (sol.g_dist.sum(axis=0) * da), 3))
```

`A_z` es un generador, no una matriz de probabilidades de transición: la entrada $(j, k)$ es la intensidad de Poisson de saltar del estado $j$ al estado $k$, la diagonal es menos la tasa total de salida y las filas suman cero. El solucionador lo comprueba por usted —una matriz cuyas filas suman uno se rechaza con `ValueError: A_z rows must sum to zero (continuous-time generator) ... A_z looks like a transition-probability matrix P; the solver expects a continuous-time generator with zero row sums, e.g. (P - I) / dt`—, de modo que el `assert` anterior es una precaución y no una necesidad. La cadena de nacimiento y muerte de tres estados converge en 8 iteraciones; la KFE recupera la distribución estacionaria de la cadena `[0.25 0.5 0.25]` como marginal del ingreso, y la riqueza media aumenta con la productividad, `[1.686 2.837 4.227]`.

### 4.5 Número de iteraciones en función de `Delta`

```python
import numpy as np
from puremacro.vfi import solve_hjb_achdou

base = solve_hjb_achdou(Delta=1e4)
for Delta in (1e4, 1e2, 10.0, 1.0):
    s = solve_hjb_achdou(Delta=Delta)                # max_iter=100, tol=1e-8
    print(f"Delta = {Delta:>8.0f}: n_iter = {s.n_iter:3d}  converged = {s.converged!s:5}  "
          f"max|V - V_ref| = {np.max(np.abs(s.V - base.V)):.1e}")
print("Delta = 1, max_iter = 1000:", solve_hjb_achdou(Delta=1.0, max_iter=1000).n_iter)
```

Esta es la fuente de la tabla de la Sección 1.3: 9, 14, 50 y 100 iteraciones para $\Delta = 10^4, 10^2, 10, 1$, con `converged=False` en $\Delta = 1$ bajo el `max_iter=100` por defecto (la función de valor aún está a $6.7 \times 10^{-2}$ de la referencia) y 360 iteraciones cuando el tope se eleva a 1000. Compruebe siempre `converged` después de reducir `Delta`.

---

## 5. Especificación completa de la API

```text
solve_hjb_achdou(
    r_rate: float = 0.03,
    w_rate: float = 1.0,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    max_iter: int = 100,
    tol: float = 1e-8,
    *,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    Delta: float = 1e4,
    compute_kfe: bool = True,
    v_prime_boundary: tuple[float | None, float | None] | None = None,
) -> HJBSolution

solve_kfe_achdou(
    A: scipy.sparse.spmatrix,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]

solve_aiyagari_continuous_hjb(
    alpha: float = 0.33,
    delta: float = 0.05,
    rho_val: float = 0.05,
    gamma_r: float = 2.0,
    Na: int = 100,
    a_min: float = 0.0,
    a_max: float = 30.0,
    a_grid: np.ndarray | None = None,
    e_grid: np.ndarray | None = None,
    A_z: np.ndarray | None = None,
    r_min: float = 0.005,
    r_max: float | None = None,
    tol_ge: float = 1e-4,
    max_iter_ge: int = 40,
    max_iter_hjb: int = 100,
    tol_hjb: float = 1e-8,
    Delta: float = 1e4,
) -> AiyagariContinuousHJBResult
```

#### Parámetros de `solve_hjb_achdou`

| Parámetro | Significado |
|---|---|
| `r_rate`, `w_rate` | Tasa de interés sobre los activos y salario por unidad de eficiencia. `w_rate=0.0` cambia al tratamiento de frontera del problema del pastel (Sección 1.2) y entonces exige `a_grid[0] > 0`, porque la forma cerrada $(\mu a)^{-\gamma}$ es singular en $a = 0$; `ValueError` en caso contrario. |
| `rho_val`, `gamma_r` | Tasa de descuento $\rho$ y coeficiente CRRA $\gamma$; $\gamma = 1$ selecciona utilidad logarítmica. |
| `Na`, `a_min`, `a_max` | Malla uniforme de activos `np.linspace(a_min, a_max, Na)`; `a_min` es el límite de endeudamiento. |
| `a_grid` | Malla explícita; anula `Na`, `a_min`, `a_max`. Puede ser no uniforme; la KFE pondera cada nodo por el ancho de su celda (Sección 1.4). Debe ser unidimensional, finita y estrictamente creciente, con al menos dos puntos (`ValueError` en caso contrario). |
| `e_grid` | Estados de productividad, forma `(Ne,)`; por defecto `[0.2, 1.0]`. |
| `A_z` | Generador del ingreso, forma `(Ne, Ne)`, entradas finitas, fuera de la diagonal no negativas y sumas por fila iguales a cero — todo ello validado, con `ValueError` ante cualquier incumplimiento. Por defecto: intensidad simétrica $0.1$ para dos estados, $0.2/(N_e - 1)$ fuera de la diagonal con $-0.2$ en ella en otro caso, `[[0.]]` para un solo estado. |
| `max_iter`, `tol` | Tope de iteraciones (entero $\ge 1$, `ValueError` en caso contrario) y tolerancia en norma del supremo sobre $v^{n+1} - v^n$. `converged` es `False` cuando se alcanza el tope. |
| `Delta` | Paso implícito $\Delta$ (Sección 1.3). |
| `compute_kfe` | Resolver la KFE adjunta tras la convergencia y rellenar `g_dist` y `mass_residual`. |
| `v_prime_boundary` | Derivadas de Neumann `(v'_min, v'_max)`; cualquiera de las dos entradas puede ser `None`. |

#### Parámetros de `solve_kfe_achdou`

| Parámetro | Significado |
|---|---|
| `A` | Generador de `HJBSolution.A_generator` (o cualquier matriz dispersa con la misma estructura), forma `(Na*Ne, Na*Ne)` en orden de estados por columnas. Una forma incompatible, o un generador con más de una clase comunicante cerrada, lanza `ValueError`. |
| `a_grid`, `e_grid` | Las mallas sobre las que se construyó el generador; solo se usan sus longitudes y los espaciamientos de activos. `a_grid` debe ser estrictamente creciente. |
| `return_residual` | `True` devuelve `(g, mass_residual)`; `False` devuelve solo `g`. `g` es una densidad: masa por unidad de $a$, con $\sum_{i,j} g_{i,j} w_i = 1$. |

#### Parámetros de `solve_aiyagari_continuous_hjb`

| Parámetro | Significado |
|---|---|
| `alpha`, `delta` | Participación del capital en la Cobb-Douglas y tasa de depreciación. |
| `rho_val`, `gamma_r`, `Na`, `a_min`, `a_max`, `a_grid`, `e_grid`, `A_z`, `Delta` | Como en `solve_hjb_achdou`; se pasan a cada resolución del problema del hogar. |
| `r_min`, `r_max` | Intervalo de Brent; `r_max=None` significa $\rho - 0.002$. El intervalo se amplía automáticamente como se describe en la Sección 1.5. `r_min >= r_max`, `r_max >= rho_val` o `r_min <= -delta` lanza `ValueError`. |
| `tol_ge` | Tolerancia de vaciado del mercado sobre $\lvert K^s - K^d \rvert$. Fija la tolerancia en $x$ de Brent, impulsa las pasadas de reacotamiento y decide `converged` (Sección 1.5). |
| `max_iter_ge` | Número máximo de iteraciones de Brent por pasada de acotamiento. |
| `max_iter_hjb`, `tol_hjb` | Pasados a `solve_hjb_achdou` como `max_iter` y `tol`. |

---

## 6. Interfaz de resultados y exportación a manuscritos

### `HJBSolution`

Una `@dataclass(frozen=True)`: asignar a un campo lanza `FrozenInstanceError`. Campos:

| Campo | Tipo / forma | Contenido |
|---|---|---|
| `V` | `(Na, Ne)` | Función de valor convergida $v_j(a_i)$. |
| `c_policy`, `s_drift` | `(Na, Ne)` | Regla de consumo y deriva del ahorro $s = r a + w e - c$. |
| `a_grid`, `e_grid` | `(Na,)`, `(Ne,)` | Mallas efectivamente utilizadas. |
| `n_iter`, `elapsed`, `converged` | `int`, `float`, `bool` | Iteraciones realizadas, segundos transcurridos, si se alcanzó `tol`. |
| `r_rate`, `w_rate`, `rho_val`, `gamma_r` | `float` | Copia de los parámetros de calibración. |
| `g_dist` | `(Na, Ne)` o `None` | **Densidad** estacionaria de la KFE — masa por unidad de $a$, de modo que la masa nodal es $g_{i,j} w_i$ (`None` cuando `compute_kfe=False`). |
| `A_generator` | `scipy.sparse.csr_matrix` | Generador $\mathbf{A}$ de la última iteración, `(Na*Ne, Na*Ne)`. |
| `mass_residual` | `float` | $\lvert \sum g_{i,j} w_i - 1 \rvert$ tras la renormalización. |

**Compatibilidad hacia atrás con `Mapping`.** `HJBSolution` hereda de `collections.abc.Mapping`. `sol["V"]`, `dict(sol)`, `list(sol)`, `len(sol)` y `sol.get(key, default)` funcionan todos. `keys()` (y por tanto `items()`, `values()`, la iteración, `dict(sol)` y `**sol`) devuelve exactamente doce claves: las siete que llevaba el diccionario de 3.3.0, `V`, `c_policy`, `s_drift`, `a_grid`, `e_grid`, `n_iter`, `elapsed`, más las copias de los parámetros `r_rate`, `w_rate`, `rho_val`, `gamma_r` y la bandera `converged`. El subíndice, `in` y `get` están restringidos a los **campos de la dataclass**, es decir, esos doce más `g_dist`, `A_generator` y `mass_residual`: `"g_dist" in sol` es `True`, mientras que `"summary" in sol` es `False` y `sol["summary"]` lanza `KeyError`, de modo que ningún método ni atributo privado se filtra por la interfaz de mapeo. `==` compara los campos uno a uno y es consciente de arrays y matrices dispersas (tenga en cuenta que `elapsed` es tiempo de reloj, así que dos resoluciones distintas del mismo problema *no* resultan iguales); `__hash__` es `None`, de modo que las instancias no son hashables, como un `dict`, y la dataclass es inmutable, de modo que asignar a un atributo lanza `FrozenInstanceError` y asignar a un elemento lanza `TypeError`.

Métodos:

- `summary() -> pd.DataFrame`: catorce filas indexadas por `Metric` (tamaños de malla, iteraciones, convergencia, tiempo transcurrido, los cuatro parámetros, valor medio, consumo mínimo y máximo, error de masa de la KFE), como se imprime en el Ejemplo 4.1.
- `to_frame() -> pd.DataFrame`: una fila por estado, con columnas `a_idx`, `e_idx`, `asset_a`, `prod_e`, `value_V`, `consumption_c`, `savings_drift_s` y, cuando se resolvió la KFE, `density_g` (la densidad, no la masa nodal). Este es el `DataFrame` que conviene usar para gráficos personalizados de la distribución.
- `to_markdown(**kwargs)`, `to_latex(**kwargs)`, `to_typst(**kwargs) -> str`: la tabla resumen generada mediante `puremacro.reports.df_to_markdown`, `df_to_latex` y `df_to_typst` (que aceptan sus argumentos `index` y `digits`).
- `plot(*, ax=None, figsize=None, show=False)`: tres paneles, función de valor, regla de consumo y deriva del ahorro, una línea por estado de ingreso. Devuelve la `Figure` cuando la crea, o los `Axes` que usted pasó; una secuencia de al menos tres ejes rellena todos los paneles, un único eje dibuja solo la función de valor. Apto para entornos sin pantalla (*headless*): no se muestra nada salvo que `show=True`. La distribución estacionaria no se grafica; constrúyala a partir de `g_dist` o `to_frame()`.

### `AiyagariContinuousHJBResult`

También una `dataclass` inmutable con el mismo protocolo `Mapping`: aquí `keys()` enumera los trece campos, el subíndice, `in` y `get` están restringidos a esos campos, `==` es la igualdad campo a campo consciente de arrays y las instancias no son hashables.

| Campo | Contenido |
|---|---|
| `r_star`, `w_star` | Tasa de interés y salario que vacían el mercado. |
| `K_star`, `Ks_star`, `Kd_star` | Capital de equilibrio ($K^\ast = K^s(r^\ast)$), oferta de los hogares y demanda de las empresas. |
| `L_star`, `Y_star` | Trabajo efectivo $\sum_j e_j \pi_j$ y producto $Y = (K^d)^\alpha L^{1-\alpha}$. |
| `excess_capital` | $K^s(r^\ast) - K^d(r^\ast)$. |
| `hjb_solution` | La `HJBSolution` en $r^\ast$, $w^\ast$. |
| `g_dist` | Alias de `hjb_solution.g_dist` (una densidad). |
| `converged`, `n_iter_ge`, `elapsed` | `True` solo cuando Brent convergió **y** $\lvert K^s - K^d \rvert <$ `tol_ge`; iteraciones de Brent sumadas sobre las pasadas de acotamiento; segundos transcurridos en toda la resolución del equilibrio general. |

Métodos: `summary()` (once filas: precios, oferta y demanda de capital, error de vaciado, trabajo, producto, convergencia, iteraciones, tiempo transcurrido), `to_frame()` (delega en `hjb_solution.to_frame()`, de modo que tabula la solución del hogar en el equilibrio), `to_markdown()`, `to_latex()`, `to_typst()` (la tabla resumen, como en el Ejemplo 4.3) y `plot()` (delega en `hjb_solution.plot()`, de modo que muestra los tres paneles del hogar, no la distribución ni el diagrama del mercado de capital).

---

## 7. Advertencias y limitaciones

- **`g_dist` es una densidad, no un vector de masas nodales.** Multiplique por los pesos de ancho de celda antes de sumar: la masa nodal es $g_{i,j} w_i$, y $w_i = \Delta a$ solo en una malla uniforme. Los ejemplos de esta página hacen exactamente eso. Esto importa sobre todo en mallas no uniformes: en la malla cuadrática `np.linspace(0, 1, 100)**2 * 30` la oferta de capital ponderada es $5.9155$, y leer `g_dist` como un vector de masas daría un resultado equivocado por el ancho local de la celda en cada nodo.
- **El número de iteraciones depende de `Delta`.** Con `Delta=1` el `max_iter=100` por defecto no basta (Sección 4.5). `converged` es el campo que hay que comprobar; el solucionador no emite advertencias.
- **El presupuesto de iteraciones de Brent es por pasada.** `max_iter_ge` acota cada pasada de acotamiento, y `n_iter_ge` reporta la suma sobre las pasadas, de modo que el número total de resoluciones del hogar puede superar `max_iter_ge` cuando `tol_ge` fuerza un reacotamiento. Las soluciones HJB se almacenan en caché por $r$, así que repetir una tasa tentativa no cuesta nada.
- **`mass_residual` se mide tras la renormalización** y por ello es del orden de $10^{-16}$ con independencia de la exactitud de la resolución lineal; certifica la normalización, no el residuo de la KFE. Calcule `A_generator.T @ (g_dist * w[:, None]).ravel(order="F")` —el generador actúa sobre masas nodales— si desea este último.
- **El orden de los estados es por columnas.** `A_generator` indexa el estado $(i, j)$ como $j N_a + i$; reorganice los vectores con `order="F"`, como hace `solve_kfe_achdou`.
- **`keys()` es más estrecho que `in`.** La pertenencia y el subíndice cubren los quince campos de la dataclass, pero `keys()` (y por tanto `dict(sol)`, `**sol` y la iteración) enumera solo las doce claves heredadas, de modo que `dict(sol)` omite `g_dist`, `A_generator` y `mass_residual`. Léalos por atributo.
- **Truncamiento en `a_max`.** La frontera superior es una restricción de estado artificial. Si una masa visible de `g_dist` se acumula en los últimos nodos, aumente `a_max` (el valor por defecto 30 es holgado para la calibración por defecto: la riqueza media es de alrededor de 6).
- **La interfaz y la economía cambiaron en 3.4.0.** `solve_hjb_achdou` devolvía un diccionario plano con siete claves y usaba `max_iter=500`, `tol=1e-7` y un esquema explícito; 3.4.0 devuelve `HJBSolution`, usa por defecto `max_iter=100`, `tol=1e-8` y el esquema implícito, y hace que todos los argumentos posteriores a `tol` sean exclusivamente por nombre (*keyword-only*). También añade el término de conmutación de Poisson del ingreso que el solucionador explícito omitía, de modo que el valor y el consumo para argumentos idénticos difieren de forma sustancial respecto a 3.3.0 (Sección 2.2). El código que usaba el diccionario sigue funcionando; el código que dependía de los valores por defecto antiguos debe pasarlos explícitamente, y el código que necesita las *cifras* antiguas debe pasar `A_z=np.zeros((Ne, Ne))` con `compute_kfe=False`.
- **NumPy/SciPy puros, float64, sin aceleradores, sin red.** El generador se ensambla en bucles de Python sobre `Na * Ne` estados y se factoriza con SuperLU; unos pocos miles de estados se resuelven en milisegundos, pero el costo de ensamblaje crece linealmente con `Na * Ne**2`. El módulo importa solo NumPy, SciPy, pandas y `puremacro.reports` —matplotlib se importa de forma diferida dentro de `plot`, así que `import puremacro.vfi` deja `matplotlib` fuera de `sys.modules`— y todo se ejecuta sin conexión. Esa primera importación sigue tardando segundos (2.4 s en la ejecución en frío en la que se basa esta página), mucho más que cualquier resolución de esta página.

---

## Referencias

- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022). "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *The Review of Economic Studies*, 89(1), 45–86.
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Barles, G., & Souganidis, P. E. (1991). "Convergence of approximation schemes for fully nonlinear second order equations." *Asymptotic Analysis*, 4(3), 271–283.
- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies." *Journal of Economic Dynamics and Control*, 17(5–6), 953–969.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
