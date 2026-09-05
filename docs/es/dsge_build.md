> 🇬🇧 [English](../dsge_build.md) · 🇪🇸 Español

# Cuaderno de bocetos DSGE: modelos desde ecuaciones, no matrices

`puremacro.dsge.klein_solve` siempre ha resuelto modelos lineales de expectativas racionales sin necesidad de Dynare ni de un compilador externo — pero requiere las matrices `A`, `B` y `C`, y obtenerlas implica diferenciar las condiciones de equilibrio a mano. Esa derivación algebraica es precisamente el paso donde los investigadores cometen errores y el paso en el que una tableta ofrece menos facilidades: sin sistema algebraico computacional, sin MATLAB y sin margen para el error.

`dsge.build` elimina esa barrera. Escriba las condiciones de equilibrio tal como aparecen en el artículo:

```python
import numpy as np
from puremacro import dsge

def eqs(xp, x, e, p):
    # xp = t+1 (adelantos), x = t (actuales), e = choques, p = parámetros
    return [
        1 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]

m = dsge.build(
    eqs,
    variables=["c", "k", "z"],
    states=["k", "z"],
    shocks=["eps"],
    params=dict(alpha=0.33, beta=0.98, rho=0.9),
    guess=dict(c=0.5, k=0.1, z=1.0),
)
print(m.summary())
m.irf("eps", horizon=20)     # DataFrame: horizontes × variables
```

`xp`, `x`, `e` y `p` admiten acceso por atributos (`x.k`), indexación por cadenas (`x["k"]`), indexación posicional y desempaquetado de tuplas. De este modo, el modelo se lee en el código exactamente igual que en el papel. Un nombre mal escrito genera inmediatamente un error informativo listando las variables declaradas.

## Qué devuelve el resolvedor

`LinearModel` es una dataclass congelada que almacena el estado estacionario, las unidades, las matrices en forma de Klein y la `KleinSolution` subyacente:

| Miembro | Significado |
|---|---|
| `.steady_state` | `pandas.Series`, verificado contra las ecuaciones en lugar de asumido |
| `.units` | `"log"` o `"level"` por cada variable |
| `.irf(shock, horizon, size)` | Funciones de respuesta al impulso (horizontes × variables) |
| `.simulate(periods, sigma, seed)` | Simulación estocástica lineal |
| `.policy()` | Reglas de decisión en una tabla etiquetada |
| `.decision_rules()` / `.dynare_dr` | Objeto `DynareDR` idéntico a `oo_.dr` de Dynare (`ghx`, `ghu`, `ys`) |
| `.theoretical_moments()` | Momentos analíticos, correlaciones cruzadas, autocorrelaciones y FEVD (`stoch_simul`) |
| `.fevd(horizons)` | Descomposición analítica de la varianza del error de pronóstico |
| `.solution` | La `KleinSolution` resuelta — `G`, `F`, `N`, `L` |
| `.A`, `.B`, `.C` | Las matrices producidas por la diferenciación de paso complejo |

Los estados estacionarios se calculan numéricamente a partir de `guess=`, o bien se suministran mediante `steady_state=` y son **verificados rigurosamente** contra las condiciones de equilibrio (`max|f(ss, ss, 0)| <= tol`). La log-linealización opera variable por variable, recurriendo a desviaciones en niveles cuando el estado estacionario no es estrictamente positivo.

## Diferenciación por paso complejo y sus restricciones

Los jacobianos se calculan mediante $\operatorname{Im} f(x + ih) / h$ con $h = 10^{-20}$ — derivadas con precisión de máquina en una única evaluación por argumento, sin compensaciones de tamaño de paso y sin errores de cancelación numérica.

La única restricción matemática es que este método es exacto si y sólo si la función residual es **analítica**, y falla silenciosamente si no lo es: $\operatorname{Im}$ evaluada a través de `abs()` es idénticamente cero, devolviendo derivadas nulas sin generar excepciones. Cuatro construcciones rompen la analiticidad: `abs`, `min`/`max`, comparaciones condicionales que ramifican según valores perturbados y conversiones explícitas a `float()` o `np.real()` que descartan la parte imaginaria.

Por ello, `build` verifica de forma cruzada cada jacobiano de paso complejo contra diferencias finitas en una dirección aleatoria, generando un `ModelError` si discrepan. Para modelos con restricciones ocasionalmente activas, pase `method="central"` con precisión de $\approx 10^{-8}$.

## Temporización y orden temporal

Las ecuaciones se plantean como $\mathbb{E}_t f(z_{t+1}, z_t, u_t) = 0$, respetando la convención de `klein_solve`. Un proceso exógeno estándar $z_{t+1} = \rho z_t + \varepsilon_{t+1}$ traslada el *estado* al período siguiente, por lo que todo estado es cero en la fila $h=0$ de una FRI e impacta en $h=1$.

Por el contrario, los **controles** con visión de futuro (*forward-looking*) habitualmente sí reaccionan en $h=0$, ya que la innovación entra en el conjunto de información de los agentes y estos descuentan el valor futuro en $t+1$. Ese salto contemporáneo corresponde a la matriz de carga $L$ de Klein.

## Condiciones de Blanchard-Kahn

La determinación no es una premisa impuesta sino una propiedad comprobada durante la resolución. Si se viola el principio de Taylor en un modelo nuevo keynesiano, el resolvedor levanta una excepción explícita:

```python
solve(phi_pi=0.9)
# BlanchardKahnError: Blanchard-Kahn indeterminacy: 2 unstable generalised
# eigenvalues vs 3 forward-looking variables
```

Pase `strict=False` si prefiere el comportamiento permisivo de Sims/gensys (matrices nulas con bandera de error `eu`).

## Validación analítica

El modelo neoclásico de crecimiento con depreciación completa y utilidad logarítmica admite solución analítica exacta: $k' = \alpha\beta z k^\alpha$, $c = (1-\alpha\beta) z k^\alpha$, con $G = [[\alpha, 1], [0, \rho]]$ y $F = [\alpha, 1]$. `build` reproduce cada celda de $G$, $F$, $N$ y $L$ con un error inferior a $10^{-9}$ y el estado estacionario hasta $10^{-12}$.

---

## Compatibilidad con Dynare: Reglas de Decisión y Momentos

Para reproducir fielmente la salida de `stoch_simul(order=1)` de Dynare, `LinearModel` proporciona:

### 1. Reglas de decisión (`m.decision_rules()` / `m.dynare_dr`)
Mapea la solución de Klein a la representación canónica de Dynare:
$$y_t = y^* + g_x (x_{t-1} - x^*) + g_u u_t$$

```python
dr = m.decision_rules()
print(dr.summary())
```
Imprime la tabla exacta `POLICY AND TRANSITION FUNCTIONS` con la constante, los estados rezagados (`k(-1)`, `z(-1)`) y los choques estructurales.

### 2. Momentos teóricos analíticos (`m.theoretical_moments()`)
Resuelve la ecuación discreta de Lyapunov $\Sigma_x = G \Sigma_x G' + N \Sigma_u N'$ para obtener los momentos no condicionados analíticos exactos sin recurrir a simulaciones de Monte Carlo:

```python
mom = m.theoretical_moments(lags=5)
print(mom.summary())
```
Genera los 4 bloques estándar de Dynare:
- **THEORETICAL MOMENTS**: Media, desviación estándar y varianza de cada variable.
- **MATRIX OF CORRELATIONS**: Matriz completa de correlaciones cruzadas.
- **COEFFICIENTS OF AUTOCORRELATION**: Autocorrelaciones teóricas exactas para los retardos 1 a 5.
- **VARIANCE DECOMPOSITION**: Participaciones porcentuales de la descomposición de varianza del error de pronóstico (FEVD) en horizontes finitos (1, 4, 8, 16, 32) y asintótico.

---

### 3. Interfaz canónica de adelantos y retardos (`dsge.build_dynare`)

Permite especificar modelos en la representación dinámica canónica de Dynare:
$$\mathbb{E}_t [ f(y_{t+1}, y_t, y_{t-1}, u_t; \theta) ] = 0$$

```python
from puremacro import dsge

def rbc(lead, curr, lag, shocks, p):
    return [
        curr.c**(-p.gamma) - p.beta * lead.c**(-p.gamma) * (p.alpha * np.exp(lead.a) * curr.k**(p.alpha - 1) + 1 - p.delta),
        curr.k - (np.exp(curr.a) * lag.k**p.alpha - curr.c + (1 - p.delta) * lag.k),
        curr.a - (p.rho * lag.a + shocks.eps),
    ]

m = dsge.build_dynare(
    rbc,
    variables=["k", "a", "c"],
    shocks=["eps"],
    params=dict(alpha=0.3, beta=0.99, delta=0.025, gamma=1.0, rho=0.8),
    guess=dict(k=38.0, a=0.0, c=2.0),
)
# ¡Las variables de estado ('k', 'a') y control ('c') se clasifican automáticamente!
```

---

### 4. Analizador nativo de archivos `.mod` de Dynare (`dsge.load_mod`)

Ejecute archivos `.mod` directamente en Python puro sin dependencias de MATLAB u Octave:

```python
from puremacro import dsge

m = dsge.load_mod("rbc.mod")
print(m.decision_rules().summary())
print(m.theoretical_moments().summary())
```

---

### 5. Perturbación de segundo orden con poda (*Pruning*)

Resuelva aproximaciones cuadráticas estables siguiendo el algoritmo de Schmitt-Grohé y Uribe (2004) y Kim, Kim, Schaumburg y Sims (2008):

```python
sol_2nd = m.solve_second_order()
# o directamente al construir el modelo:
sol_2nd = dsge.build_dynare(rbc, ..., order=2)
```

Descompone el espacio de estados en componentes de primer y segundo orden:
$$x_t^{(1)} = G x_{t-1}^{(1)} + N u_t$$
$$x_t^{(2)} = G x_{t-1}^{(2)} + \frac{1}{2} H_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + \frac{1}{2} H_{\sigma\sigma} \sigma^2$$
$$y_t^{(1)} = F x_t^{(1)} + L u_t$$
$$y_t^{(2)} = F x_t^{(2)} + \frac{1}{2} G_{xx} (x_t^{(1)} \otimes x_t^{(1)}) + \frac{1}{2} G_{\sigma\sigma} \sigma^2$$

Capacidades destacadas:
- **Simulación incondicionalmente estable**: `sol_2nd.simulate(periods=200, sigma=0.01)` evita las trayectorias explosivas características de las aproximaciones cuadráticas sin poda.
- **Respuestas al impulso generalizadas (GIRF)**: `sol_2nd.girf(shock="eps", size=0.01, horizon=20)` evalúa impactos dependientes del estado.
- **Estado estacionario estocástico corregido por riesgo**: `sol_2nd.stochastic_steady_state(sigma=0.01)` captura el ahorro precautorio originado por la volatilidad.

---

### 6. Herramienta de línea de comandos `puremacro-dynare` CLI

`puremacro 2.2.0` incluye el comando de terminal `puremacro-dynare` (`puremacro.dsge.cli`), permitiendo resolver modelos desde scripts de consola o terminales:

```bash
# Resolución básica y visualización de reglas de política
puremacro-dynare rbc.mod

# Resolver a 2do orden con poda y generar tabla FEVD
puremacro-dynare rbc.mod --order 2 --fevd

# Exportar tablas listas para publicación (Markdown, LaTeX, Typst) y guardar gráficos FRI
puremacro-dynare rbc.mod --irf 20 --format all --plot

# Descomposición histórica de choques a partir de un archivo CSV de datos observados
puremacro-dynare rbc.mod --shock-decomp datos_macro.csv --plot
```

---

### 7. OccBin: Restricciones Ocasionalmente Activas (ZLB)

`puremacro.dsge.occbin` implementa el algoritmo lineal por tramos de Guerrieri e Iacoviello (2015) para modelos con cotas inferiores de tasa de interés cero (Zero Lower Bound, ZLB) o restricciones de colateral.

Calcula la trayectoria de alternancia de regímenes entre el régimen de referencia $M_1$ y el régimen restringido $M_2$ mediante recursión hacia atrás:

```python
from puremacro.dsge import solve_occbin, OccBinConstraint

m_ref = dsge.load_mod("nk_taylor.mod")
m_zlb = dsge.load_mod("nk_zlb.mod")

# Definir la restricción: activa cuando la tasa nominal sombra r <= 0
constraint = OccBinConstraint(
    variable="r",
    threshold=0.0,
    direction="below",
)

res_occbin = solve_occbin(
    m_reference=m_ref,
    m_constrained=m_zlb,
    constraint=constraint,
    shocks={"eps_demand": -0.04},
    horizon=30,
)

print(res_occbin.summary())
print("Períodos en ZLB:", res_occbin.regime_history)
res_occbin.plot(title="Dinámica Nuevo Keynesiana con OccBin ZLB")
```

---

### 8. Simulación no lineal y previsión perfecta

`puremacro.dsge.perfect_foresight` implementa el método de relajación apilada de Newton-Raphson de Boucekkine (1995) y Juillard (1996) para resolver transiciones deterministas no lineales exactas:

```python
from puremacro.dsge import solve_perfect_foresight

# Transición determinista no lineal ante choque permanente de PTF del 5%
pf_res = solve_perfect_foresight(
    m,
    shocks={"eps_a": [0.05] + [0.0] * 99},
    T=100,
    max_iter=50,
    tol=1e-8,
)

print(pf_res.summary())
pf_res.plot()
```

---

### 9. Estimación Bayesiana de DSGE por MCMC

`puremacro.dsge.bayesian` proporciona un flujo bayesiano completo sin dependencias compiladas externas:
1. Distribuciones a priori (`BetaPrior`, `GammaPrior`, `InvGammaPrior`, `NormalPrior`, `UniformPrior`).
2. Búsqueda de moda mediante L-BFGS-B o Nelder-Mead.
3. Inversión del hessiano numérico en la moda para la covarianza de propuesta de Laplace.
4. Muestreador adaptativo Random-Walk Metropolis-Hastings (RWMH).
5. Diagnósticos de convergencia $\hat{R}$ dividida (Gelman-Rubin) y prueba espectral de Geweke.

```python
import pandas as pd
from puremacro.dsge import estimate_dsge_bayesian
from puremacro.dsge.priors import BetaPrior, GammaPrior, InvGammaPrior

data = pd.read_csv("datos_macro.csv")

priors = {
    "alpha": BetaPrior(mean=0.33, std=0.05),
    "beta":  BetaPrior(mean=0.99, std=0.005),
    "rho":   BetaPrior(mean=0.80, std=0.10),
    "sigma": InvGammaPrior(mean=0.01, std=0.005),
}

bayes_res = estimate_dsge_bayesian(
    model=m,
    data=data,
    priors=priors,
    n_draws=10000,
    burn_in=2000,
    n_chains=2,
    seed=42,
)

print(bayes_res.summary())
bayes_res.plot_priors_posteriors()
print(bayes_res.to_latex())
```

---

### 10. Descomposición Histórica de Choques y FEVD

```python
# Descomposición analítica de varianza del error de pronóstico (FEVD)
fevd_res = m.fevd_result(horizons=[1, 4, 8, 16, 40])
print(fevd_res.summary())

# Descomposición histórica de choques suavizada por Kalman
decomp_res = m.shock_decomposition(data_obs=data)
print(decomp_res.summary())
decomp_res.plot(variable="output", title="Descomposición Histórica del Producto")
```
