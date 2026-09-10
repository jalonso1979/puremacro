**Español** · [English](../dsge_phase_d.md)

# Frontera DSGE: Filtrado de Partículas No Lineal con Volatilidad Estocástica y Modelos DSGE con Cambio de Régimen de Markov

`puremacro` introduce dos arquitecturas computacionales de vanguardia para el análisis macroeconómico no lineal y de regímenes cambiantes:

1. **Filtrado de Partículas Vectorizado en Python Puro (Gordon et al. 1993; Pitt & Shephard 1999; Fernández-Villaverde & Rubio-Ramírez 2007)**:
   Un motor de Monte Carlo Secuencial (SMC) de alto rendimiento para la evaluación exacta de la verosimilitud no lineal en soluciones de perturbación podadas de segundo y tercer orden (`PrunedDSGESolution`, `Order3PrunedSolution`). Incorpora **Volatilidad Estocástica (SV)** autorregresiva $\sigma_{j,t} = \bar{\sigma}_j \exp(h_{j,t})$ con aumento del espacio de estados capaz de capturar desplazamientos por ahorro precautorio y asimetría, distribuciones de innovaciones de colas pesadas ($t$ de Student, mixturas de Gaussianas) y algoritmos de remuestreo vectorizados en $O(N)$ (sistemático, estratificado, residual y multinomial) ejecutados sin ningún bucle en Python sobre las partículas.

2. **Modelos DSGE con Cambio de Régimen de Markov (MS-DSGE) (Foerster, Rubio-Ramírez, Waggoner & Zha 2016; Farmer, Waggoner & Zha 2011)**:
   Un marco de perturbación integral para modelos de expectativas racionales sujetos a cambios discretos de régimen markoviano en parámetros de política monetaria y fiscal. Resuelve las ecuaciones cuadráticas matriciales acopladas mediante Newton-Raphson por bloques analítico e iteración funcional amortiguada, verifica estabilidad en media cuadrática (MSS) $\rho(M_2) < 1$ y estabilidad de primer momento $\rho(M_1) < 1$, calcula distribuciones estacionarias ergódicas y covarianzas de Lyapunov discretas incondicionales, y evalúa funciones de impulso-respuesta generalizadas (GIRF) analíticas en forma cerrada $(1_S^\top \otimes I_n) M_1^h z_0$ a precisión de máquina en fracciones de milisegundo.

Ambos módulos cumplen estrictamente con el contrato Pyodide de cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`), sin requerir extensiones en C, compiladores de Fortran ni solucionadores propietarios externos.

---

## Comparativa Metodológica

| Dimensión | Filtro de Partículas Bootstrap (BPF) | Filtro de Partículas Auxiliar (APF) | DSGE con Cambio de Régimen (MS-DSGE) | Filtro de Kalman Lineal Estándar |
|:---|:---|:---|:---|:---|
| **Referencia Fundamental** | Gordon et al. (1993); FV-RR (2007) | Pitt & Shephard (1999) | Foerster et al. (2016); FWZ (2011) | Kalman (1960); Hamilton (1994) |
| **Clase de Modelo** | DSGE podado 2º/3º orden + SV | DSGE podado 2º/3º orden + SV | Perturbación con regímenes de Markov | DSGE lineal gaussiano |
| **Espacio de Estados** | Partículas $x_t^i$ + log-vol $h_t^i$ | Partículas $x_t^i$ + índices primera etapa | Reglas acopladas $(T(s), R(s), c(s))$ | Estado medio $\hat{x}_{t\|t}$ + Covarianza $P_{t\|t}$ |
| **Evaluación de Verosimilitud** | Promedio SMC $\frac{1}{N} \sum w_t^i$ | Propuesta SMC reponderada | Filtro con ponderación de regímenes / Hamilton | Error de predicción gaussiano exacto |
| **Núcleo Computacional** | Propagación tensorial con einsum | Covarianza predictiva $\Sigma_\mu = Z R Q R^\top Z^\top + H$ | Cuadráticas acopladas (Newton por bloques) | Actualización discreta de Riccati |
| **Aporte Económico Principal** | Ahorro precautorio, asimetría, SV | Robustez de propuesta ante atípicos | Cambios de política, determinabilidad MSS | Dinámica lineal gaussiana |
| **Compatible con Pyodide** | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) |

---

## 1. Filtrado de Partículas No Lineal y Volatilidad Estocástica

### 1.1 Representación en Espacio de Estados con Perturbación Podada

Los modelos de perturbación lineal no pueden capturar el ahorro precautorio, las primas de riesgo ni la asimetría, dado que la equivalencia de certidumbre rige a primer orden. Las expansiones de orden superior resuelven esta limitación, pero las expansiones estándar de Taylor exhiben trayectorias explosivas artificiales. Para garantizar estabilidad, `puremacro` utiliza las soluciones de **perturbación podada** de Kim, Kim, Schaumburg & Sims (2008).

A segundo orden, el vector de estados se descompone en componentes de primer y segundo orden $x_t = x_t^{\text{I}} + x_t^{\text{II}}$:
$$x_t^{\text{I}} = h_x x_{t-1}^{\text{I}} + h_u \varepsilon_t$$
$$x_t^{\text{II}} = h_x x_{t-1}^{\text{II}} + \frac{1}{2} H_{xx} (x_{t-1}^{\text{I}} \otimes x_{t-1}^{\text{I}}) + H_{xu} (x_{t-1}^{\text{I}} \otimes \varepsilon_t) + \frac{1}{2} H_{uu} (\varepsilon_t \otimes \varepsilon_t) + \frac{1}{2} h_{\sigma\sigma} \sigma^2$$
$$y_t = g_x x_t + \frac{1}{2} G_{xx} (x_t^{\text{I}} \otimes x_t^{\text{I}}) + \frac{1}{2} g_{\sigma\sigma} \sigma^2$$

`puremacro` vectoriza esta evaluación a lo largo de $N$ partículas empleando contracciones tensoriales (`np.einsum("mi,mj->mij", x, x)`), procesando 10,000 partículas en 50 periodos en aproximadamente 50 milisegundos sin bucles en Python.

### 1.2 Dinámica de Volatilidad Estocástica

La incertidumbre macroeconómica varía sustancialmente a lo largo del tiempo. `puremacro` amplía el espacio de estados estructural con un proceso autorregresivo de Volatilidad Estocástica (SV) para las perturbaciones estructurales $\varepsilon_t$:
$$\varepsilon_{j, t} = \sigma_{j, t} \cdot \zeta_{j, t}, \quad \zeta_{j, t} \sim \mathcal{N}(0, 1)$$
$$\sigma_{j, t} = \bar{\sigma}_j \exp(h_{j, t})$$
$$h_{j, t} = \rho_{h, j} h_{j, t-1} + \sigma_{\eta, j} \eta_{j, t}, \quad \eta_{j, t} \sim \mathcal{N}(0, 1)$$
donde $\bar{\sigma}_j$ es la escala base de la perturbación, $\rho_{h, j} \in [0, 1)$ es la persistencia de la volatilidad y $\sigma_{\eta, j} > 0$ es la volatilidad de la volatilidad.

Cada partícula rastrea tanto los estados físicos del DSGE $(x_t^{\text{I}}, x_t^{\text{II}})$ como el vector latente de log-volatilidad $h_t$, permitiendo inferencia conjunta sobre los estados macroeconómicos y la incertidumbre económica.

### 1.3 Filtro de Partículas Bootstrap Vectorizado (BPF)

El Filtro de Partículas Bootstrap propaga una distribución empírica discreta $\{x_t^i, w_t^i\}_{i=1}^N$ que aproxima la densidad de filtrado $p(x_t \mid y_{1:t})$:

1. **Inicialización**: Muestreo $x_0^i \sim p(x_0)$ y $h_0^i \sim \mathcal{N}\left(0, \frac{\sigma_\eta^2}{1 - \rho_h^2}\right)$ con ponderaciones uniformes $w_0^i = 1/N$.
2. **Propagación**: Muestreo de innovaciones estructurales $\zeta_t^i \sim \mathcal{N}(0, I)$ e innovaciones de volatilidad $\eta_t^i \sim \mathcal{N}(0, I)$. Cálculo de las escalas temporales $\sigma_t^i$ y evaluación de transiciones podadas:
   $$x_t^i = \mathcal{T}(x_{t-1}^i, \sigma_t^i \zeta_t^i)$$
3. **Ponderación de Medida**: Evaluación de las ponderaciones de importancia con la densidad de observación:
   $$\tilde{w}_t^i = p(y_t \mid x_t^i) = (2\pi)^{-d_y/2} |H|^{-1/2} \exp\left( -\frac{1}{2} (y_t - g(x_t^i))^\top H^{-1} (y_t - g(x_t^i)) \right)$$
   Normalización: $w_t^i = \frac{\tilde{w}_t^i}{\sum_{j=1}^N \tilde{w}_t^j}$.
4. **Contribución a la Verosimilitud**:
   $$\ln p(y_t \mid y_{1:t-1}) = \ln \left( \frac{1}{N} \sum_{i=1}^N \tilde{w}_t^i \right)$$
5. **Tamaño Muestral Efectivo (ESS) y Remuestreo**:
   $$ESS_t = \frac{1}{\sum_{i=1}^N (w_t^i)^2}$$
   Cuando $ESS_t < \tau_{\text{resample}} \cdot N$ (típicamente $\tau = 0.5$), se remuestrean las partículas utilizando esquemas de $O(N)$: sistemático, estratificado o residual.

### 1.4 Filtro de Partículas Auxiliar (APF)

Cuando el error de medición es pequeño o las observaciones se sitúan en las colas de la distribución, la densidad de propuesta del BPF $p(x_t \mid x_{t-1}^i)$ se aparta del verdadero posterior, provocando empobrecimiento de partículas. El Filtro de Partículas Auxiliar (Pitt & Shephard 1999) incorpora la observación contemporánea $y_t$ en las ponderaciones de selección de primera etapa:
$$\alpha_t^i \propto w_{t-1}^i \cdot p(y_t \mid \mu_t^i)$$
donde $\mu_t^i = \mathbb{E}[x_t \mid x_{t-1}^i]$ es la media predictiva determinista de la partícula $i$.

`puremacro` perfecciona el APF estándar calculando la covarianza predictiva exacta de observación:
$$\Sigma_\mu = Z (R Q R^\top) Z^\top + H$$
Esto asegura densidades predictivas no singulares $p(y_t \mid \mu_t^i) = \mathcal{N}(y_t \mid g(\mu_t^i), \Sigma_\mu)$, evitando el colapso de las ponderaciones auxiliares y estabilizando el filtrado.

---

## 2. Modelos DSGE con Cambio de Régimen de Markov (MS-DSGE)

### 2.1 Sistema de Perturbación Estructural

Siguiendo a Foerster, Rubio-Ramírez, Waggoner & Zha (FRWZ 2016), considérese un sistema macroeconómico con expectativas racionales donde los parámetros estructurales alternan entre $S$ regímenes discretos $s_t \in \{1, \dots, S\}$:
$$A(s_t) \mathbb{E}_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + K(s_t) + D(s_t) \varepsilon_t = 0$$
donde $y_t \in \mathbb{R}^n$, $\varepsilon_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma_\varepsilon)$, y las transiciones de régimen siguen una cadena de Markov ergódica exógena:
$$P = [p_{ij}], \quad p_{ij} = \Pr(s_{t+1} = j \mid s_t = i)$$

### 2.2 Solución de Estado Mínimo de Variables (MSV)

La regla de equilibrio MSV lineal por perturbación adopta la forma:
$$y_t = c(s_t) + T(s_t) y_{t-1} + R(s_t) \varepsilon_t$$
Sustituyendo en el sistema estructural y tomando expectativas condicionadas a la información en la fecha $t$ y régimen $s_t = i$:
$$\mathbb{E}_t [y_{t+1}] = \sum_{j=1}^S p_{ij} \left( c_j + T_j y_t \right) = \sum_{j=1}^S p_{ij} c_j + \left( \sum_{j=1}^S p_{ij} T_j \right) \left( c_i + T_i y_{t-1} + R_i \varepsilon_t \right)$$

Igualando coeficientes para cada variable de estado se deducen las **ecuaciones cuadráticas matriciales acopladas**:
$$A_i \left( \sum_{j=1}^S p_{ij} T_j \right) T_i + B_i T_i + C_i = 0, \quad \forall i \in \{1, \dots, S\}$$
Obtenidas las matrices solución $T_1, \dots, T_S$, las matrices de impacto de perturbaciones $R_i$ y constantes $c_i$ quedan determinadas de manera unívoca:
$$R_i = - \left[ A_i \left( \sum_{j=1}^S p_{ij} T_j \right) + B_i \right]^{-1} D_i$$
$$\left[ A_i \left( \sum_{j=1}^S p_{ij} T_j \right) + B_i \right] c_i + A_i \sum_{j=1}^S p_{ij} c_j + K_i = 0$$

### 2.3 Solucionadores: Newton-Raphson por Bloques e Iteración Funcional

`puremacro` implementa dos algoritmos complementarios para resolver el sistema cuadrático acoplado:

1. **Newton-Raphson por Bloques Analítico**:
   Vectorizando el sistema en $F(\mathbf{T}) = 0$ con $\mathbf{T} = [\operatorname{vec}(T_1)^\top, \dots, \operatorname{vec}(T_S)^\top]^\top$. El jacobiano exacto por bloques $J \in \mathbb{R}^{Sn^2 \times Sn^2}$ se evalúa analíticamente:
   $$J_{ii} = I_n \otimes \left( A_i \sum_{j=1}^S p_{ij} T_j + B_i \right) + T_i^\top \otimes (p_{ii} A_i)$$
   $$J_{ij} = T_i^\top \otimes (p_{ij} A_i) \quad (i \neq j)$$
   Los pasos de Newton $\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} - J(\mathbf{T}^{(k)})^{-1} F(\mathbf{T}^{(k)})$ logran convergencia cuadrática en 5–10 iteraciones a precisión de máquina ($10^{-16}$).

2. **Iteración Funcional Amortiguada**:
   $$T_i^{(k+1)} = (1 - \alpha) T_i^{(k)} - \alpha \left[ A_i \sum_{j=1}^S p_{ij} T_j^{(k)} + B_i \right]^{-1} C_i$$
   Alternativa robusta cuando el punto inicial se encuentra alejado del dominio de atracción.

### 2.4 Diagnóstico de Estabilidad: Media y Media Cuadrática (MSS)

Los autovalores individuales de las matrices $T_i$ no determinan la estabilidad del equilibrio, ya que una economía puede alternar entre regímenes localmente explosivos y estables. Siguiendo a Costa, Fragoso & Marques (2005) y Farmer et al. (2011), `puremacro` calcula los operadores exactos de estabilidad:

1. **Estabilidad de Primer Momento (Estabilidad en Media)**:
   $$M_1 = (P^\top \otimes I_n) \operatorname{diag}(T_1, \dots, T_S)$$
   El sistema es estable en media si y solo si $\rho(M_1) < 1$.

2. **Estabilidad de Segundo Momento (Estabilidad en Media Cuadrática - MSS)**:
   $$M_2 = (P^\top \otimes I_{n^2}) \operatorname{diag}(T_1 \otimes T_1, \dots, T_S \otimes T_S)$$
   El sistema es estable en media cuadrática si y solo si $\rho(M_2) < 1$. La estabilidad en media cuadrática garantiza varianzas incondicionales asintóticas finitas.

### 2.5 Momentos Ergódicos y Covarianza Discreta de Lyapunov

Sea $\pi_\infty$ la distribución estacionaria unívoca de la cadena de Markov:
$$\pi_\infty P = \pi_\infty, \quad \sum_{i=1}^S \pi_{\infty, i} = 1$$
La media ergódica incondicional es:
$$\bar{y} = \sum_{i=1}^S \pi_{\infty, i} c_i$$
La matriz de covarianza incondicional $\operatorname{Var}(y) = \sum_{i=1}^S \pi_{\infty, i} \Sigma_{y, i}$ satisface la ecuación discreta acoplada de Lyapunov:
$$\Sigma_{y, i} = T_i \left( \sum_{j=1}^S p_{ji} \frac{\pi_{\infty, j}}{\pi_{\infty, i}} \Sigma_{y, j} \right) T_i^\top + R_i \Sigma_\varepsilon R_i^\top$$
resuelta de forma vectorizada mediante $(I_{Sn^2} - M_2)^{-1}$.

### 2.6 Función de Impulso-Respuesta Generalizada (GIRF) Analítica Cerrada

Las funciones de impulso-respuesta tradicionales condicionales al régimen asumen de forma contrafáctica que el régimen permanece inmutable para siempre. En contraste, los agentes económicos comprenden que el régimen cambiará estocásticamente en el futuro. La Función de Impulso-Respuesta Generalizada (GIRF; Koop, Pesaran & Potter 1996) integra sobre todas las posibles trayectorias futuras de regímenes:
$$\operatorname{GIRF}_h(s_0, \varepsilon_0) = \mathbb{E}[y_{t+h} \mid s_t = s_0, \varepsilon_t = \varepsilon_0] - \mathbb{E}[y_{t+h} \mid s_t = s_0]$$

`puremacro` calcula la GIRF analítica exacta en forma cerrada:
$$\operatorname{GIRF}_h(s_0, \varepsilon_0) = (1_S^\top \otimes I_n) M_1^h z_0$$
donde $z_0 = e_{s_0} \otimes (R(s_0) \varepsilon_0) \in \mathbb{R}^{Sn}$, evaluándose en menos de 1 milisegundo a precisión de máquina sin ruido de simulación de Monte Carlo.

---

## 3. API en Python y Ejemplos de Código

### 3.1 Filtrado de Partículas No Lineal con Volatilidad Estocástica

```python
import numpy as np
import pandas as pd
from puremacro.dsge.dynare import load_mod
from puremacro.dsge.particle_filter import particle_filter, StochasticVolatilitySpec

# 1. Compilar modelo DSGE y resolver perturbación podada de 2º orden
mod_code = """
var c k z;
varexo eps;
parameters beta alpha delta rho sigma_pref sigma_eps;
beta = 0.99; alpha = 0.33; delta = 0.025; rho = 0.95; sigma_pref = 1.0; sigma_eps = 0.01;
model;
  exp(-sigma_pref*c) - beta*exp(-sigma_pref*c(+1))*(alpha*exp(z(+1))*exp((alpha-1)*k) + 1 - delta);
  exp(c) + exp(k) - exp(z)*exp(alpha*k(-1)) - (1 - delta)*exp(k(-1));
  z - rho*z(-1) - sigma_eps*eps;
end;
initval; k = 3.8; c = 0.8; z = 0.0; end;
steady;
"""
model = load_mod(mod_code)
solution = model.solve(order=2)

# 2. Datos observables simulados
data = pd.DataFrame({"c": [0.80, 0.81, 0.79], "k": [3.31, 3.32, 3.30]})

# 3. Especificar volatilidad estocástica: sigma_t = bar{sigma} * exp(h_t)
sv = StochasticVolatilitySpec(rho=0.85, sigma_eta=0.20, base_scale=0.01)

# 4. Ejecutar el Filtro de Partículas Bootstrap (BPF)
result = particle_filter(
    solution,
    data=data,
    observed_vars=["c", "k"],
    n_particles=10_000,
    method="bootstrap",
    resampling_method="systematic",
    stochastic_volatility=sv,
    seed=42,
)

print(result.summary())
print(f"Log-Verosimilitud: {result.log_likelihood:.4f}")
print(f"Frecuencia de Remuestreo: {result.resampling_frequency:.1%}")
```

### 3.2 Solución de DSGE con Cambio de Régimen de Markov

```python
import numpy as np
from puremacro.dsge.markov_switching import solve_ms_dsge

# 1. Parámetros estructurales: modelo neokeynesiano de 3 ecuaciones
beta, sigma, kappa, rho_i, phi_x = 0.99, 1.0, 0.1, 0.8, 0.1
regime_names = ["Hawkish", "Dovish"]
phi_pi = [1.8, 0.8]  # Respuesta activa vs pasiva en la regla de Taylor

# Matriz de probabilidad de transición: 90% persistencia hawkish, 80% dovish
P = np.array([
    [0.90, 0.10],
    [0.20, 0.80],
])

# 2. Matrices estructurales específicas de régimen A(s), B(s), C(s), D(s)
A, B, C, D = [], [], [], []
for s in range(2):
    As = np.array([[1.0, 1.0 / sigma, 0.0], [0.0, beta, 0.0], [0.0, 0.0, 0.0]])
    Bs = np.array([[-1.0, 0.0, -1.0 / sigma], [kappa, -1.0, 0.0], [(1 - rho_i) * phi_x, (1 - rho_i) * phi_pi[s], -1.0]])
    Cs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, rho_i]])
    Ds = np.eye(3)
    A.append(As); B.append(Bs); C.append(Cs); D.append(Ds)

# 3. Resolver MS-DSGE con Newton-Raphson por bloques
ms_res = solve_ms_dsge(
    A, B, C, D, P,
    regime_names=regime_names,
    variable_names=["output_gap", "inflation", "interest_rate"],
    shock_names=["demand", "cost_push", "monetary_policy"],
    method="newton",
)

# 4. Verificar estabilidad y momentos ergódicos
print(f"Convergencia: {ms_res.converged} en {ms_res.iterations} iteraciones")
print(f"Estable en Media Cuadrática (MSS): {ms_res.mean_square_stable} (rho(M2) = {ms_res.spectral_radius_mss:.4f})")
print("Distribución Ergódica:", ms_res.ergodic_distribution.to_dict())

# 5. Evaluar GIRF analítica exacta
girf = ms_res.girf("Hawkish", shock="monetary_policy", horizon=12)
print("GIRF en Impacto:\n", girf.head(2))
```

---

## 4. Contrato de Presentación de Resultados

En conformidad con la arquitectura de presentación unificada de puremacro, tanto `ParticleFilterResult` como `MSDSGEResult` implementan el contrato estándar de seis métodos:

| Método | Tipo Retornado | Descripción |
|:---|:---|:---|
| `.summary()` | `str` o `pd.DataFrame` | Tabla estadística de resumen con formato editorial |
| `.plot()` | `Figure` o `tuple[Figure, Any]` | Gráficos diagnósticos multipanel de calidad de publicación |
| `.to_frame()` | `pd.DataFrame` | Representación tabular estructurada en DataFrame |
| `.to_markdown(**kwargs)` | `str` | Tabla en formato GitHub-flavored Markdown |
| `.to_latex(**kwargs)` | `str` | Tabla en formato LaTeX con estilo booktabs |
| `.to_typst(**kwargs)` | `str` | Tabla en formato moderno Typst para documentos científicos |

---

## Referencias Académicas

- **Costa, O. L., Fragoso, M. D., & Marques, R. P. (2005).** *Discrete-Time Markov Jump Linear Systems*. Springer Science & Business Media.
- **Davig, T., & Leeper, E. M. (2007).** Generalizing the Taylor principle. *American Economic Review*, 97(3), 607–635.
- **Farmer, R. E., Waggoner, D. F., & Zha, T. (2011).** Minimal state variable solutions to Markov-switching rational expectations models. *Journal of Economic Dynamics and Control*, 35(12), 2150–2166.
- **Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007).** Estimating macroeconomics models: A likelihood approach. *Review of Economic Studies*, 74(4), 1059–1087.
- **Foerster, A. T., Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2016).** Perturbation methods for Markov-switching dynamic stochastic general equilibrium models. *Econometrica*, 84(6), 2219–2270.
- **Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993).** Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F (Radar and Signal Processing)*, 140(2), 107–113.
- **Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008).** Calculating and using second-order accurate solutions of discrete time dynamic equilibrium models. *Journal of Economic Dynamics and Control*, 32(11), 3397–3414.
- **Koop, G., Pesaran, M. H., & Potter, S. M. (1996).** Impulse response analysis in nonlinear multivariate models. *Journal of Econometrics*, 74(1), 119–147.
- **Leeper, E. M. (1991).** Equilibria under 'active' and 'passive' monetary and fiscal policies. *Journal of Monetary Economics*, 27(1), 129–147.
- **Pitt, M. K., & Shephard, N. (1999).** Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590–599.
