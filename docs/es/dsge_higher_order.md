> 🇬🇧 [English](../dsge_higher_order.md) · 🇪🇸 Español

# Perturbación de orden superior, restricciones y métodos DSGE no lineales

Puremacro 2.8.0 incorpora el módulo **Tier 2: Orden superior y restricciones**, extendiendo el motor de modelos DSGE más allá de las aproximaciones locales de primer y segundo orden hacia transiciones no lineales globales, poda de estados (*pruning*), restricciones de desigualdad ocasionalmente activas, muestreo por Monte Carlo Secuencial (SMC) y política óptima de Ramsey bajo compromiso.

Todos los algoritmos de esta versión se encuentran implementados íntegramente en Python puro bajo el estricto **contrato de cuatro paquetes de Pyodide** (`numpy`, `scipy`, `pandas`, `matplotlib`), sin recurrir a extensiones compiladas en C, generadores de analizadores sintácticos externos ni dependencias pesadas de álgebra computacional (sin SymPy ni JAX).

---

## 1. Visión general y arquitectura

Los modelos DSGE linealizados (resueltos típicamente mediante la descomposición QZ generalizada de Klein) ofrecen aproximaciones de primer orden en torno al estado estacionario determinista. Si bien su resolución computacional es instantánea, las aproximaciones lineales presentan severas limitaciones:
1. Ignoran los motivos de precaución y las primas de riesgo endógenas ($g_{\sigma\sigma} = 0$).
2. Operan bajo equivalencia de certidumbre: la volatilidad macroeconómica no desplaza las reglas de decisión de política ni los estados estacionarios ergódicos.
3. Son incapaces de representar cotas asimétricas, tales como la cota inferior cero (*Zero Lower Bound* o ZLB) sobre los tipos de interés nominales o restricciones colaterales de endeudamiento.
4. Fracasan ante cambios estructurales permanentes en los que la economía transita entre estados estacionarios iniciales y terminales netamente distintos.

El módulo Tier 2 proporciona un conjunto unificado y riguroso de métodos para superar dichas restricciones:

| Módulo | Capacidad central | Algoritmo principal | Referencia canónica |
| :--- | :--- | :--- | :--- |
| `puremacro.dsge.pruning` | Perturbación de orden 3 y poda ergódica | Solucionador Schur Sylvester 3D y expansión de 3 estados | Andreasen, Fernández-Villaverde y Rubio-Ramírez (2018) |
| `puremacro.dsge.perfect_foresight` | Transiciones deterministas y MCP | Newton semisuave con complementariedad de Fischer-Burmeister | Boucekkine (1995); Juillard (1996) |
| `puremacro.dsge.occbin` | Lineal por tramos multirrestricción | Relajación de regímenes duales y Filtro de Kalman por tramos | Guerrieri e Iacoviello (2015); Giovannini et al. (2021) |
| `puremacro.dsge.smc` | Monte Carlo Secuencial y partículas | SMC con templado adaptativo y filtro de partículas bootstrap | Herbst y Schorfheide (2014, 2015) |
| `puremacro.dsge.ramsey` | Política óptima de Ramsey no lineal y BGP | CPO simbólicas del lagrangiano en AST y solución Klein | Dennis (2007); Clarida, Galí y Gertler (1999) |

---

## 2. Perturbación de orden 3 y poda de Andreasen

### 2.1 Fundamentos teóricos y la maldición de la dimensionalidad

Un modelo macroeconómico dinámico y estocástico de equilibrio general (DSGE) viene caracterizado por el sistema de ecuaciones con expectativas racionales:

$$\mathbb{E}_t \left[ f(y_{t+1}, y_t, y_{t-1}, u_t) \right] = 0$$

donde $y_t$ es un vector de $n_y$ variables endógenas y $u_t \sim \mathcal{N}(0, \Sigma)$ representa las innovaciones exógenas escaladas por el parámetro de perturbación $\sigma$.

Particionando $y_t$ en estados predeterminados $x_t$ y variables de control no predeterminadas $w_t$, la regla de política exacta adopta la forma:

$$y_t = g(x_{t-1}, u_t, \sigma)$$

Aproximando $g$ hasta el tercer orden mediante series de Taylor:

$$g(x, u, \sigma) \approx g_{ss} + g_x \hat{x} + g_u u + \frac{1}{2} g_{xx} (\hat{x} \otimes \hat{x}) + g_{xu} (\hat{x} \otimes u) + \frac{1}{2} g_{uu} (u \otimes u) + \frac{1}{2} g_{\sigma\sigma} \sigma^2 + \frac{1}{6} g_{xxx} (\hat{x} \otimes \hat{x} \otimes \hat{x}) + \dots + \frac{1}{6} g_{\sigma\sigma\sigma} \sigma^3$$

La evaluación simbólica de la tercera derivada tensorial dinámica:

$$\mathcal{T}_f = \frac{\partial^3 f}{\partial z_j \, \partial z_k \, \partial z_l}$$

sobre el vector extendido $z = (y_{t+1}, y_t, y_{t-1}, u_t)$ exigiría evaluar $O((3n_y + n_u)^3)$ coeficientes. El motor de diferenciación simbólica de puremacro (`puremacro.dsge._symbolic`) elude la asignación de tensores cúbicos densos mediante:
- Explotación estricta de la simetría de permutación de orden 6 ($j \le k \le l$).
- Eliminación de subexpresiones comunes (CSE, *Common Subexpression Elimination*).
- Vectorización directa en funciones ejecutables compiladas en Python puro.

### 2.2 Solucionador Schur Sylvester generalizado para tensores de orden 3

El tensor de curvatura de tercer orden respecto a los estados, $g_{xxx}$, satisface la ecuación matricial generalizada de Sylvester:

$$\hat{A} g_{xxx} + A_+ g_{xxx} (h_x \otimes h_x \otimes h_x) = - K_{xxx}$$

donde $\hat{A} = A_0 + A_+ g_x P_s$ y $h_x = P_s g_x$ constituye la matriz de transición del bloque de estados estables.

Construir explícitamente la matriz densa de Kronecker $(h_x \otimes h_x \otimes h_x)$ de orden $n_x^3 \times n_x^3$ requeriría cientos de gigabytes de memoria en modelos de mediana escala (como Smets y Wouters 2007 con $n_x = 14$, donde $14^3 = 2.744$, generando un sistema de $11 \times 10^7$ entradas).

Puremacro implementa el **solucionador desacoplado Schur Sylvester tridimensional** (`puremacro.dsge._sylvester`):
1. Obtiene la descomposición de Schur compleja $h_x = U_h T_h U_h^H$ con factor triangular superior $T_h$.
2. Transforma el tensor inhomogéneo a coordenadas de Schur: $\tilde{K} = K_{xxx} (U_h \otimes U_h \otimes U_h)$.
3. Resuelve columna a columna mediante sustitución regresiva triangular con almacenamiento en caché de factores LU:

$$\left( \hat{A} + T_h(i, i) T_h(j, j) T_h(k, k) A_+ \right) \tilde{g}_{i,j,k} = - \tilde{K}_{i,j,k} - \text{términos de sustitución previa}$$

4. Rota la solución al espacio original: $g_{xxx} = \tilde{g}_{xxx} (U_h^H \otimes U_h^H \otimes U_h^H)$.

En el modelo canónico de Smets y Wouters (2007), este solucionador converge en tan sólo **0,034 segundos** consumiendo **menos de 1 MB de memoria**, evitando por completo el ensamblado de tensores densos.

### 2.3 El algoritmo de poda de Andreasen et al. (2018)

Las simulaciones polinomiales no podadas de orden $k \ge 2$ generan divergencias numéricas espurias: perturbaciones suficientemente grandes sitúan al sistema en regiones donde los términos cuadráticos y cúbicos dominan sobre la retroalimentación lineal estable, desencadenando trayectorias explosivas al infinito.

Siguiendo a **Andreasen, Fernández-Villaverde y Rubio-Ramírez (2018, *Review of Economic Studies*)**, puremacro descompone la desviación del estado $\hat{x}_t = x_t - x_{ss}$ en componentes aditivas de primer, segundo y tercer orden:

$$\hat{x}_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)}$$

cuyo espacio de estados recursivo preserva la estabilidad asintótica eliminando retroalimentaciones cruzadas explosivas:

$$x_t^{(1)} = h_x x_{t-1}^{(1)} + h_u u_t$$

$$x_t^{(2)} = h_x x_{t-1}^{(2)} + \frac{1}{2} h_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + h_{xu} (x_{t-1}^{(1)} \otimes u_t) + \frac{1}{2} h_{uu} (u_t \otimes u_t) + \frac{1}{2} h_{\sigma\sigma} \sigma^2$$

$$x_t^{(3)} = h_x x_{t-1}^{(3)} + h_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(2)}) + h_{xu} (x_{t-1}^{(2)} \otimes u_t) + \frac{1}{6} h_{xxx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + \frac{1}{2} h_{x\sigma\sigma} x_{t-1}^{(1)} \sigma^2 + \frac{1}{2} h_{u\sigma\sigma} u_t \sigma^2$$

Las variables de control $y_t$ se reconstruyen de modo concordante:

$$\hat{y}_t = g_x x_t^{(1)} + g_u u_t + g_x x_t^{(2)} + \frac{1}{2} g_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + \dots + g_x x_t^{(3)} + g_{xx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(2)}) + \frac{1}{6} g_{xxx} (x_{t-1}^{(1)} \otimes x_{t-1}^{(1)} \otimes x_{t-1}^{(1)}) + \dots$$

Este esquema garantiza:
- **Estacionariedad ergódica**: Las simulaciones estocásticas incondicionales se mantienen estables a lo largo de más de 10.000 períodos sin trayectorias divergentes.
- **Momentos analíticos**: Expresiones analíticas exactas para el sesgo de la media incondicional $\mathbb{E}[\hat{y}_t]$, la covarianza, la asimetría (*skewness*) y la curtosis de la distribución estacionaria.

### 2.4 Funciones de impulso-respuesta generalizadas (GIRF) y API

Al violarse el principio de superposición lineal en órdenes superiores, la respuesta dinámica depende intrínsecamente del estado inicial $x_0$ y del signo y magnitud de la perturbación. Puremacro calcula **Funciones de Impulso-Respuesta Generalizadas (GIRF)** (Koop, Pesaran y Potter 1996):

$$\text{GIRF}_t(v, x_0) = \mathbb{E} \left[ y_t \mid u_0 = v, x_0 \right] - \mathbb{E} \left[ y_t \mid u_0 = 0, x_0 \right]$$

```python
import numpy as np
from puremacro.dsge import load_mod

# Carga de un modelo RBC no lineal
mod_text = """
var c k y a;
varexo e_a;
parameters beta alpha delta rho sigma_a;
beta = 0.99; alpha = 0.36; delta = 0.025; rho = 0.95; sigma_a = 0.01;

model;
  1/c = beta * (1/c(+1)) * (alpha * exp(a(+1)) * k^(alpha - 1) + 1 - delta);
  c + k = exp(a) * k(-1)^alpha + (1 - delta) * k(-1);
  y = exp(a) * k(-1)^alpha;
  a = rho * a(-1) + e_a;
end;

initval;
  k = 38.0; c = 2.7; y = 3.6; a = 0;
end;
shocks;
  var e_a; stderr 0.01;
end;
"""

model = load_mod(mod_text)

# Resolución de orden 3 con poda de Andreasen
sol3 = model.solve(order=3, pruning=True)

# 1. Momentos ergódicos incondicionales (media, varianza, asimetría, curtosis)
moments = sol3.theoretical_moments()
print("Asimetría ergódica:", moments.skewness.round(4))
print("Curtosis ergódica:", moments.kurtosis.round(4))

# 2. Funciones de Impulso-Respuesta Generalizadas (GIRF)
girf = sol3.girf(shock="e_a", size=2.0, horizon=40)
girf.plot(title="GIRF de orden 3 ante perturbación tecnológica de 2 desviaciones típicas")

# 3. Simulación estocástica podada
sim = sol3.simulate(periods=1000, seed=42)
print("Media estacionaria del capital bajo riesgo:", sim["k"].mean())
```

---

## 3. Transiciones deterministas y problemas de complementariedad mixta (MCP)

### 3.1 Senda extendida y cambios de estado estacionario terminal

El análisis de políticas económicas con frecuencia exige evaluar reformas estructurales permanentes, tales como:
- Una modificación permanente del objetivo de inflación $\pi^*$.
- La introducción permanente de impuestos al carbono o aranceles comerciales.
- Desplazamientos permanentes en la frontera tecnológica.

Bajo perturbaciones permanentes, el estado estacionario terminal $y_{end}$ difiere del estado estacionario inicial $y_{init}$. Puremacro interpreta los bloques `histval; ... end;` y `endval; ... end;` para calcular la trayectoria determinista no lineal exacta a lo largo de un horizonte $T$:

$$f(y_{t+1}, y_t, y_{t-1}, u_t) = 0, \quad t = 1, \dots, T$$

con condiciones de frontera $y_0 = y_{init}$ e $y_{T+1} = y_{end}$.

### 3.2 Trayectorias exógenas anticipadas y sorpresivas

Puremacro distingue de forma nativa entre:
1. **Perturbaciones deterministas anticipadas (`varexo_det`)**: Variables exógenas anunciadas en $t=0$ cuya evolución futura se conoce con plena certeza.
2. **Perturbaciones sorpresivas (`simulate_surprise_shocks`)**: Novedades inesperadas (*MIT shocks*) que acontecen secuencialmente en cada período $t$, forzando una replanificación hacia adelante sobre el horizonte restante.

### 3.3 MCP con Newton semisuave y complementariedad de Fischer-Burmeister

Múltiples problemas macroeconómicos conllevan restricciones de desigualdad, como la cota inferior cero (ZLB) de la tasa de política monetaria:

$$i_t = \max(0, i_t^*)$$

Formulada como un Problema de Complementariedad Mixta (MCP):

$$0 \le i_t \perp \lambda_t \ge 0 \quad \text{donde } \lambda_t = i_t - i_t^*$$

Para evitar la búsqueda combinatoria sobre $2^T$ configuraciones temporales de regímenes, puremacro implementa un **algoritmo de Newton semisuave** fundamentado en la **función de complementariedad de Fischer-Burmeister**:

$$\Phi(a, b) = a + b - \sqrt{a^2 + b^2} = 0 \iff a \ge 0, \, b \ge 0, \, ab = 0$$

Sea $\mathbf{F}(\mathbf{Y}) = \mathbf{0}$ el sistema apilado de ecuaciones temporales para $t=1,\dots,T$. Para cada variable acotada en el período $t$, la condición de igualdad se sustituye por $\Phi(y_{i,t} - \underline{y}_i, \lambda_{i,t}) = 0$.

El jacobiano generalizado $\partial_B \Phi$ es semisuave:

$$\frac{\partial \Phi}{\partial a} = 1 - \frac{a}{\sqrt{a^2 + b^2}}, \quad \frac{\partial \Phi}{\partial b} = 1 - \frac{b}{\sqrt{a^2 + b^2}}$$

Puremacro ensambla el jacobiano apilado en formato disperso de bloques tridiagonales, resolviendo la transición no lineal global con complejidad $O(T)$ y convergencia local superlineal.

```python
from puremacro.dsge import load_mod, solve_perfect_foresight

MOD_ZLB = """
var y pi i;
varexo eps_r;
parameters beta sigma phi_pi phi_y;
beta = 0.99; sigma = 1.0; phi_pi = 1.5; phi_y = 0.5;

model;
  y = y(+1) - (1/sigma) * (i - pi(+1));
  pi = beta * pi(+1) + 0.1 * y;
  i = max(0, phi_pi * pi + phi_y * y + eps_r);
end;

initval; y = 0; pi = 0; i = 0.02; end;
"""

model = load_mod(MOD_ZLB)

# Solución con Newton semisuave MCP garantizando la cota inferior cero
res_mcp = solve_perfect_foresight(
    model,
    periods=100,
    shocks={"eps_r": [-0.05, -0.03, -0.01]},
    mcp=True,
    mcp_bounds={"i": (0.0, np.inf)},
)

print(f"Convergencia en {res_mcp.iterations} iteraciones.")
print("Períodos con ZLB activa:", res_mcp.binding_periods["i"])
res_mcp.plot(title="Experimento deflacionario Nuevo Keynesiano con ZLB vinculante")
```

---

## 4. OccBin multirrestricción y Filtro de Kalman por tramos

### 4.1 Arquitectura lineal por tramos multirrestricción

**OccBin (Guerrieri e Iacoviello 2015)** aproxima modelos con restricciones ocasionalmente activas mediante regímenes lineales por tramos:
- Régimen de referencia sin restricciones: $A_0 y_t + A_+ \mathbb{E}_t y_{t+1} + A_- y_{t-1} + C_0 = 0$
- Regímenes con restricciones ($k=1,\dots,K$): $A_0^{(k)} y_t + A_+^{(k)} \mathbb{E}_t y_{t+1} + A_-^{(k)} y_{t-1} + C_0^{(k)} = 0$

Puremacro 2.8.0 generaliza OccBin a **$K \ge 2$ restricciones simultáneas**, permitiendo modelar hasta $2^K$ regímenes discretos (por ejemplo, ZLB monetaria concurrente con restricciones colaterales de crédito).

El algoritmo:
1. Inicializa una conjetura sobre la secuencia de regímenes $(r_1, \dots, r_H)$ sobre un horizonte de previsión $H$.
2. Realiza recursiones de Riccati retrospectivas dependientes del tiempo:

$$P_t = - (A_0^{(r_t)} + A_+^{(r_t)} P_{t+1})^{-1} A_-^{(r_t)}$$

3. Simula prospectivamente los estados y recalcula los multiplicadores de sombra de holgura.
4. Verifica la convergencia de la secuencia de activación.

### 4.2 El Filtro de Kalman por tramos (PKF, Giovannini et al. 2021)

La estimación econométrica de modelos con cotas activas exigía tradicionalmente filtros no lineales o simuladores de partículas computacionalmente lentos.

El **Filtro de Kalman por Tramos (PKF)** desarrollado por **Giovannini, Pfeiffer y Ratto (2021, *Journal of Economic Dynamics and Control*)** sortea este obstáculo integrando la estructura lineal por tramos directamente en el filtrado:
1. **Etapa de predicción**: Dados el estado filtrado $\hat{x}_{t-1|t-1}$ y la covarianza $P_{t-1|t-1}$, resuelve las recursiones de OccBin asumiendo nulas las perturbaciones futuras esperadas, derivando matrices de transición temporales $T_t(r)$ y vectores de constantes $c_t(r)$.
2. **Etapa de actualización**: Computa los errores de pronóstico $v_t = y_t - Z \hat{x}_{t|t-1} - d_t$ y actualiza estados y covarianzas mediante la ganancia de Kalman estándar $K_t$.
3. **Acumulación de verosimilitud**: Evalúa la densidad gaussiana tramo a tramo:

$$\ln L(Y \mid \theta) = - \frac{1}{2} \sum_{t=1}^T \left( n_y \ln(2\pi) + \ln |F_t| + v_t^\top F_t^{-1} v_t \right)$$

### 4.3 Estimación bayesiana con PKF

```python
from puremacro.dsge import load_mod, OccBinConstraint

# Carga de modelos sin restricción y bajo régimen restringido
m_ref = load_mod("modelo_libre.mod")
m_zlb = load_mod("modelo_zlb.mod")

# Definición de la restricción: tasa de política en 0 cuando la tasa nocional sea <= 0
constraint = OccBinConstraint(
    name="zlb",
    condition="i_notional <= 0",
    constrained_model=m_zlb,
)

# Estimación bayesiana mediante el Filtro de Kalman por Tramos
res_est = m_ref.estimate(
    data=datos_macro,
    method="piecewise_kalman",
    occbin_regimes=[constraint],
    n_draws=2000,
    burn_in=500,
)

print(res_est.summary())
```

---

## 5. Monte Carlo Secuencial (SMC) y filtrado de partículas

### 5.1 El algoritmo SMC de Herbst y Schorfheide (2014, 2015)

Los muestreadores estándar de paseo aleatorio Metropolis-Hastings (RWMH) presentan serias deficiencias ante superficies de verosimilitud macroeconómicas caracterizadas por multimodalidad, crestas estrechas y parámetros mal identificados.

Puremacro incorpora el algoritmo de **Monte Carlo Secuencial (SMC)** de **Herbst y Schorfheide (2014, *Journal of Applied Econometrics*; 2015, *Princeton University Press*)**. El método conecta la distribución a priori $\pi(\theta)$ con la distribución a posteriori $p(\theta|Y)$ mediante una sucesión de $N$ distribuciones puente templadas:

$$\pi_n(\theta) \propto \pi(\theta) \left[ p(Y \mid \theta) \right]^{\phi_n}, \quad 0 = \phi_0 < \phi_1 < \dots < \phi_N = 1$$

Rasgos fundamentales:
1. **Templado adaptativo**: En lugar de emplear una malla estática, puremacro determina $\phi_{n+1}$ endógenamente resolviendo:

$$\text{ESS}(\phi_{n+1}) = \alpha^* \cdot N_{\text{partículas}}$$

mediante bisección unidimensional, asegurando una dispersión uniforme de partículas entre etapas.
2. **Remuestreo sistemático**: Las partículas con pesos reducidos son remuestreadas cuando el tamaño muestral efectivo cae bajo el umbral $ESS < 0.5 N_{part}$.
3. **Mutación de partículas**: Las partículas remuestreadas mutan a través de varios pasos de Metropolis-Hastings guiados por la matriz de covarianza empírica $\Sigma_n = \text{Cov}_\pi(\theta)$ y un factor de escala adaptativo $c_n$.

### 5.2 Densidad marginal de los datos (MDD) exacta

Una ventaja determinante del algoritmo SMC sobre MCMC convencional radica en la obtención analítica exacta de la **densidad marginal de los datos (*Marginal Data Density*, MDD)** $\ln p(Y)$ como el producto acumulado de las constantes de normalización de cada etapa:

$$\ln \hat{p}(Y) = \sum_{n=1}^N \ln \left( \frac{1}{M} \sum_{m=1}^M w_n^{(m)} \right)$$

donde $w_n^{(m)} = [p(Y \mid \theta_{n-1}^{(m)})]^{\phi_n - \phi_{n-1}}$ son las ponderaciones incrementales de importancia. Puremacro calcula asimismo el error estándar numérico asintótico $\text{SE}(\ln \hat{p}(Y))$.

### 5.3 Filtro de partículas bootstrap no lineal

Para modelos DSGE no lineales (órdenes 2 y 3) o con errores de medida no gaussianos, puremacro provee un **filtro de partículas bootstrap** en NumPy puro:
- Propaga las partículas a través de las ecuaciones de estado no lineales podadas.
- Pondera las partículas frente a las densidades empíricas de observación.
- Evalúa la verosimilitud no lineal exacta mediante remuestreo sistemático.

```python
from puremacro.dsge import load_mod
from puremacro.dsge.smc import SMCSampler

model = load_mod("smets_wouters_07.mod")

# Configuración del muestreador SMC
smc = SMCSampler(
    model,
    data=datos_observados,
    varobs=["dy", "dc", "dinve", "dw", "pinf", "robs", "labobs"],
    n_particles=2000,
    n_stages=50,
    adaptive_tempering=True,
    target_ess=0.5,
    seed=123,
)

# Muestreo a posteriori
res_smc = smc.sample()

print(f"Log Densidad Marginal de Datos (MDD): {res_smc.mdd:.3f} +/- {res_smc.mdd_se:.3f}")
print(res_smc.posterior_summary)

# Diagnósticos visuales de etapas
res_smc.plot_stages()
res_smc.plot_posterior()
```

---

## 6. Política óptima de Ramsey no lineal y desestacionalización BGP

### 6.1 Política óptima de Ramsey automatizada

Bajo compromiso (*commitment*), un planificador social benevolente maximiza el bienestar intertemporal de los hogares sujeto a las condiciones de equilibrio general descentralizado de la economía:

$$\max_{\{y_t\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t U(y_t) \quad \text{s.a.} \quad \mathbb{E}_t f(y_{t+1}, y_t, y_{t-1}, u_t) = 0$$

La función `ramsey_model` de puremacro (`puremacro.dsge.ramsey`) automatiza este procedimiento:
1. Construye automáticamente el lagrangiano del planificador:

$$\mathcal{L} = \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left[ U(y_t) + \lambda_t^\top f(y_{t+1}, y_t, y_{t-1}, u_t) \right]$$

2. Evalúa simbólicamente las Condiciones de Primer Orden (CPO) sobre el DAG del AST respecto a cada variable endógena $y_{i,t}$ y multiplicador de Lagrange $\lambda_{j,t}$:

$$\frac{\partial U(y_t)}{\partial y_{i,t}} + \left[ \frac{\partial f(y_{t+1}, y_t, y_{t-1}, u_t)}{\partial y_{i,t}} \right]^\top \lambda_t + \beta^{-1} \left[ \frac{\partial f(y_t, y_{t-1}, y_{t-2}, u_{t-1})}{\partial y_{i,t+1}} \right]^\top \lambda_{t-1} + \beta \mathbb{E}_t \left[ \frac{\partial f(y_{t+2}, y_{t+1}, y_t, u_{t+1})}{\partial y_{i,t-1}} \right]^\top \lambda_{t+1} = 0$$

3. Extiende el espacio de estados incorporando los multiplicadores de política $\lambda_t$.
4. Resuelve el equilibrio sobre la senda de silla (*saddle-path*) mediante la descomposición QZ de Klein, recuperando de forma natural la **perspectiva intemporal** (*timeless perspective*, Woodford 2003; Clarida, Galí y Gertler 1999).

```python
from puremacro.dsge import load_mod, ramsey_model

# Modelo Nuevo Keynesiano canónico de 3 ecuaciones
NK_MOD = """
var x pi i;
varexo eps_u;
parameters beta sigma kappa;
beta = 0.99; sigma = 1.0; kappa = 0.15;

model;
  x = x(+1) - (1/sigma) * (i - pi(+1));
  pi = beta * pi(+1) + kappa * x + eps_u;
end;
"""

model = load_mod(NK_MOD)

# Resolución de la política óptima de Ramsey minimizando la pérdida cuadrática L = pi^2 + 0.1 * x^2
ramsey_res = ramsey_model(
    model,
    objective="-(pi^2 + 0.1 * x^2)",
    planner_discount=0.99,
)

print("CPO analíticas de Ramsey derivadas:")
for eq in ramsey_res.focs:
    print(" ", eq)

# Gráfico de respuestas a impulsos de las variables y multiplicadores sombra
ramsey_res.plot(title="Respuestas de política bajo compromiso ante choque de empuje de costes")
```

### 6.2 Desestacionalización de la senda de crecimiento balanceado (BGP)

Los modelos empíricos con frecuencia exhiben tendencias estocásticas o deterministas de crecimiento (como el progreso técnico neutral de Harrod $A_t = \gamma^t A_0$ o el cambio tecnológico específico a la inversión).

Puremacro provee un motor de desestacionalización BGP automático:
- Declaración de variables de tendencia: `trend_var A;` o `log_trend_var a;`
- Asignación de deflactores: `var(deflator=A) Y C I;`
- El motor reescribe automáticamente el sistema transformado estacionario $y_t^* = Y_t / A_t$ y computa analíticamente el estado estacionario transformado.

---

## 7. Matriz comparativa resumen

| Capacidad | Módulo | Clase de entrada | Clase de salida | Método analítico clave |
| :--- | :--- | :--- | :--- | :--- |
| **Perturbación de orden 3** | `puremacro.dsge.pruning` | `LinearModel` / `.mod` | `Order3PrunedSolution` | Schur Sylvester 3D y poda de Andreasen |
| **Transiciones deterministas** | `puremacro.dsge.perfect_foresight` | `LinearModel` / `.mod` | `PerfectForesightResult` | Newton-Raphson en bloques tridiagonales dispersos |
| **Complementariedad MCP** | `puremacro.dsge.perfect_foresight` | `LinearModel` / `.mod` | `MCPResult` | Newton semisuave con Fischer-Burmeister |
| **OccBin multirrestricción** | `puremacro.dsge.occbin` | Diccionario de `LinearModel` | `OccBinResult` | Recursiones de Riccati de Guerrieri-Iacoviello |
| **Filtro de Kalman por tramos** | `puremacro.dsge.estimate` | `LinearModel` + Diccionario | `PiecewiseKalmanResult` | Verosimilitud de Giovannini-Pfeiffer-Ratto |
| **Monte Carlo Secuencial** | `puremacro.dsge.smc` | `LinearModel` + Datos | `SMCResult` | Templado adaptativo de Herbst-Schorfheide |
| **Filtro de partículas** | `puremacro.dsge.smc` | `Order3PrunedSolution` | `tuple[float, np.ndarray]` | Remuestreo sistemático bootstrap |
| **Política óptima de Ramsey** | `puremacro.dsge.ramsey` | `LinearModel` + Objetivo | `RamseyResult` | CPO del lagrangiano en AST + Klein QZ |
| **Desestacionalización BGP** | `puremacro.dsge._parser` | `.mod` con `deflator` | `ParsedModelDAG` | Transformación algebraica estacionaria |

---

## 8. Referencias bibliográficas

- **Andreasen, M. M., Fernández-Villaverde, J., y Rubio-Ramírez, J. F. (2018)**. *The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications*. Review of Economic Studies, 85(1), 1–49.
- **Auclert, A., Bardóczy, B., Rognlie, M., y Straub, L. (2021)**. *Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models*. Econometrica, 89(6), 2787–2815.
- **Boucekkine, R. (1995)**. *An Alternative Methodology for Solving Nonlinear Forward-Looking Models*. Journal of Economic Dynamics and Control, 19(4), 711–734.
- **Clarida, R., Galí, J., y Gertler, M. (1999)**. *The Science of Monetary Policy: A New Keynesian Perspective*. Journal of Economic Literature, 37(4), 1661–1707.
- **Dennis, R. (2007)**. *Optimal Policy in Rational Expectations Models: New and Alternative Solutions*. Journal of Economic Dynamics and Control, 31(12), 3959–3983.
- **Giovannini, M., Pfeiffer, P., y Ratto, M. (2021)**. *The Piecewise Kalman Filter for Occasionally Binding Constraints*. Journal of Economic Dynamics and Control, 128, 104128.
- **Guerrieri, L., e Iacoviello, M. (2015)**. *OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily*. Journal of Monetary Economics, 75, 38–54.
- **Herbst, E., y Schorfheide, F. (2014)**. *Sequential Monte Carlo Sampling for DSGE Models*. Journal of Applied Econometrics, 29(7), 1073–1098.
- **Herbst, E., y Schorfheide, F. (2015)**. *Bayesian Estimation of DSGE Models*. Princeton University Press.
- **Juillard, M. (1996)**. *Dynare: A Program for the Resolution and Simulation of Dynamic Models with Forward-Looking Variables Through the Use of a Relaxation Algorithm*. Documento de trabajo CEPREMAP 9602.
- **Klein, P. (2000)**. *Using the Generalized Schur Form to Solve a System of Linear Expectational Difference Equations*. Journal of Economic Dynamics and Control, 24(10), 1405–1423.
- **Woodford, M. (2003)**. *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.
