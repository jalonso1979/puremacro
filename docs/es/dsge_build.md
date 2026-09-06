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

```text
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

mod_text = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;

alpha = 0.30;
beta  = 0.99;
delta = 0.025;
gamma = 1.0;
rho   = 0.80;

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;

initval;
  k = 38.0;
  a = 0.0;
  c = 2.0;
end;
"""

# Cargar desde texto o desde una ruta de archivo
m = dsge.load_mod(mod_text)
print(m.decision_rules().summary())
print(m.theoretical_moments().summary())
```

---

### 5. Perturbación de segundo orden con poda (*Pruning*)

Resuelva aproximaciones cuadráticas estables siguiendo el algoritmo de Schmitt-Grohé y Uribe (2004) y Kim, Kim, Schaumburg y Sims (2008):

```python
# Segundo orden: pase order=2 a load_mod / build_dynare, o vuelva a resolver un modelo existente
sol_2nd = m.solve(order=2)              # PrunedDSGESolution (Kim-Kim-Schaumburg-Sims pruning)
print(sol_2nd.oo_dr.summary())          # ghx, ghu, ghxx, ghxu, ghuu, ghs2 en formato Dynare
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
import numpy as np
from puremacro.dsge import build_dynare, solve_occbin, OccBinConstraint

# Modelo neokeynesiano de tres ecuaciones. El régimen restringido fija la tasa
# nominal en el límite inferior cero (r = -r_ss en desviaciones del estado estacionario).
params = {"beta": 0.99, "sigma": 1.0, "kappa": 0.1, "phi_pi": 1.5, "phi_y": 0.125, "rho_g": 0.8, "r_ss": 0.01}
variables = ["y", "pi", "r", "g"]
shocks = ["eps_r", "eps_g"]
steady_state = {v: 0.0 for v in variables}

def nk_taylor(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,          # dynamic IS
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,                     # NK Phillips curve
        curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,   # Taylor rule
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,                         # demand shock process
    ]

def nk_zlb(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),                                                # rate pegged at the ZLB
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

m_ref = build_dynare(nk_taylor, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
# Un régimen con tasa fija es indeterminado por sí solo; OccBin solo necesita sus
# jacobianos, por eso el régimen alternativo se construye con strict=False.
m_zlb = build_dynare(nk_zlb, variables=variables, shocks=shocks, params=params,
                     steady_state=steady_state, check_steady_state=False, strict=False)

# La restricción se activa cuando la tasa sombra cae por debajo del límite
constraint = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")

horizon = 30
shock_seq = np.zeros((horizon, len(shocks)))
shock_seq[0, shocks.index("eps_g")] = -0.04        # choque de demanda contractivo en t=0

res_occbin = solve_occbin(m_ref, m_zlb, constraint, shock_sequence=shock_seq, horizon=horizon)
print(res_occbin.summary())
print("Períodos en el límite inferior cero:", res_occbin.binding_periods)
print("Régimen por período (1 = restringido):", res_occbin.regimes)
res_occbin.plot()
```

---

### 8. Simulación no lineal y previsión perfecta

`puremacro.dsge.perfect_foresight` implementa el método de relajación apilada de Newton-Raphson de Boucekkine (1995) y Juillard (1996) para resolver transiciones deterministas no lineales exactas:

```python
import numpy as np
from puremacro.dsge import solve_perfect_foresight

# Modelo de Ramsey determinista en niveles:
#   1/c_t = beta / c_{t+1} * (alpha A_t k_t^(alpha-1) + 1 - delta)
#   k_t   = A_t k_{t-1}^alpha + (1 - delta) k_{t-1} - c_t
alpha, beta, delta = 0.33, 0.96, 0.10
r_ss = 1.0 / beta - (1.0 - delta)
k_ss = (alpha / r_ss) ** (1.0 / (1.0 - alpha))
c_ss = k_ss ** alpha - delta * k_ss

def ramsey(y_plus, y_curr, y_lag, exo):
    c_p, k_p = y_plus
    c, k = y_curr
    c_m, k_m = y_lag
    A = float(np.ravel(exo)[0])
    return [
        1.0 / c - beta / c_p * (alpha * A * k ** (alpha - 1.0) + 1.0 - delta),
        k - (A * k_m ** alpha + (1.0 - delta) * k_m - c),
    ]

# Un alza de +5% en la PTF en el período 5, anunciada en t=0: el consumo salta antes de que llegue el choque
n_periods = 100
tfp_path = np.ones(n_periods)
tfp_path[4] = 1.05

pf_res = solve_perfect_foresight(
    ramsey,
    y_init=np.array([c_ss, k_ss]),
    y_ss=np.array([c_ss, k_ss]),
    exogenous_path=tfp_path,
    n_periods=n_periods,
    variable_names=["c", "k"],
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
import numpy as np
from puremacro.dsge import estimate_dsge_bayesian
from puremacro.dsge.priors import BetaPrior, InvGammaPrior
from puremacro.state_space import StateSpaceModel, kalman_filter

# Observable: y_t = rho y_{t-1} + sigma eps_t, T = 300 (sustituto de la verosimilitud de su modelo)
rng = np.random.default_rng(42)
y = np.zeros(300)
for t in range(1, 300):
    y[t] = 0.7 * y[t - 1] + 0.4 * rng.standard_normal()
y = y[:, None]

def log_likelihood(params):
    rho, sigma = (params["rho"], params["sigma"]) if isinstance(params, dict) else (params[0], params[1])
    ssm = StateSpaceModel(T=np.array([[rho]]), Z=np.array([[1.0]]), R=np.array([[1.0]]),
                          Q=np.array([[sigma ** 2]]), H=np.array([[1e-6]]))
    return kalman_filter(y, ssm)["loglik"]

priors = {
    "rho": BetaPrior(mean=0.6, std=0.15, lb=0.01, ub=0.99),
    "sigma": InvGammaPrior(mean=0.3, std=2.0, lb=0.01, ub=3.0),
}

bayes_res = estimate_dsge_bayesian(
    log_likelihood, priors,
    initial_params=np.array([0.6, 0.3]),
    n_draws=1000, n_burn=200, n_chains=2, seed=42,
)
print(bayes_res.summary())

# Densidades a priori y a posteriori, y una tabla lista para publicación
bayes_res.plot_priors_posteriors()
print(bayes_res.to_latex())
```

---

### 10. Descomposición Histórica de Choques y FEVD

```python
# 1. FEVD analítica en los horizontes [1, 4, 8, 16, 40]
fevd_res = m.fevd_result(horizons=[1, 4, 8, 16, 40])
print(fevd_res.summary())
print(fevd_res.to_latex())

# 2. Descomposición histórica de choques de una muestra (aquí simulada del modelo;
#    pase sus datos observados con una columna por variable)
data = m.simulate(periods=80, seed=1)
decomp_res = m.shock_decomposition(data)
print(decomp_res.summary())

# Gráfico de barras apiladas de las contribuciones históricas de los choques
decomp_res.plot(variable="c")
```
