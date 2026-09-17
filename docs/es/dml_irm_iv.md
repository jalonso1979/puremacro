> 🇬🇧 [English](../dml_irm_iv.md) · 🇪🇸 Español

# Aprendizaje Automático Doble para Efectos de Tratamiento e Instrumentos (IRM y DML-IV)

Si bien el modelo de Regresión Parcialmente Lineal (PLR) asume un efecto de tratamiento constante y aditivo ($Y = D \theta_0 + g_0(X) + U$), las aplicaciones macroeconómicas y microeconométricas aplicadas a menudo involucran:
1. **Heterogeneidad en el Efecto del Tratamiento**: El impacto causal de una intervención de política $D \in \{0, 1\}$ varía según características individuales o regionales $X$. Esto se aborda mediante el **Modelo de Regresión Interactiva (IRM)** para el **Efecto Medio del Tratamiento (ATE)** y el **Efecto Medio en los Tratados (ATT)**.
2. **Políticas Endógenas e Instrumentos Excluidos**: El tratamiento $D$ es endógeno ($\mathbb{E}[U \mid D, X] \neq 0$), pero se dispone de instrumentos excluidos exógenos $Z$ junto con controles confusores de alta dimensión $X$. Esto se resuelve mediante **Variables Instrumentales con Doble ML (DML-IV)**.

`puremacro.causal` implementa ambas metodologías con rigor matemático institucional:

- **`DoubleMLIRM` y `dml_irm`**: *Scores* doblemente robustos y ortogonales de Neyman, con descenso por coordenadas en NumPy puro para clasificación logística regularizada de propensiones, recorte (*trimming*) automático de soporte común y gráficos de diagnóstico (`.plot_overlap()`, `.plot_coefficients()`, `.plot_tuning()`).
- **`DoubleMLIV` y `dml_iv`**: Residualización ortogonal mediante ajuste cruzado con estimación MC2E (*2SLS*) y diagnósticos de instrumentos débiles según el estadístico $F$ efectivo de Montiel Olea y Pflueger (2013).
- **`LogisticCoordinateDescent`**: Regresión logística regularizada ($\ell_1$/Lasso, $\ell_2$/Ridge y ElasticNet) en NumPy puro, empleando funciones subrogadas cuadráticas cóncavas y selección de penalización por AIC/BIC.

No se requieren compiladores de C++ ni paquetes externos de aprendizaje automático (`scikit-learn`, `torch`). Todos los algoritmos cumplen estrictamente el contrato Pyodide de 4 paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`).

---

## 1. Marco Teórico y Algorítmico

### 1.1 El Modelo de Regresión Interactiva (IRM)

Sea $Y \in \mathbb{R}$ la variable de resultado, $D \in \{0, 1\}$ una intervención de política y $X \in \mathbb{R}^p$ un vector de controles confusores de alta dimensión. Bajo el supuesto de no confusión condicionada y solapamiento:

$$(Y(1), Y(0)) \perp D \mid X, \qquad \varepsilon \le m_0(X) \le 1 - \varepsilon$$

El modelo estructural se define como:

$$Y = g_0(D, X) + U, \qquad \mathbb{E}[U \mid D, X] = 0$$

$$D = m_0(X) + V, \qquad \mathbb{E}[V \mid X] = 0, \quad m_0(X) \equiv \mathbb{P}(D = 1 \mid X)$$

donde $g_0(d, X) \equiv \mathbb{E}[Y \mid D = d, X]$ modela la función de respuesta condicional en el estado $d \in \{0, 1\}$, y $m_0(X)$ es la puntuación de propensión (*propensity score*).

#### Scores Ortogonales de Neyman para ATE y ATT

El *score* ortogonal de Neyman para el **Efecto Medio del Tratamiento (ATE)** $\theta_0 = \mathbb{E}[Y(1) - Y(0)]$ es la función de momentos doblemente robusta (Chernozhukov et al. 2018):

$$\psi_{\text{ATE}}(W; \theta, \eta) = g(1, X) - g(0, X) + \frac{D \big(Y - g(1, X)\big)}{m(X)} - \frac{(1 - D) \big(Y - g(0, X)\big)}{1 - m(X)} - \theta$$

Para el **Efecto Medio en los Tratados (ATT)** $\theta_0 = \mathbb{E}[Y(1) - Y(0) \mid D = 1]$:

$$\psi_{\text{ATT}}(W; \theta, \eta) = \frac{D \big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)} - \frac{m(X)(1 - D)\big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)\big(1 - m(X)\big)} - \theta$$

Ambos *scores* satisfacen $\mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0$ y poseen derivada de Gateaux nula respecto de las funciones de molestia $\eta = (g(0, \cdot), g(1, \cdot), m(\cdot))$ en el valor verdadero. En consecuencia, los errores de aproximación de primer orden introducidos por los modelos de aprendizaje automático regularizados no sesgan la estimación causal $\hat{\theta}$.

### 1.2 Descenso por Coordenadas Logístico Regularizado en NumPy Puro

Para estimar la propensión $m_0(X) = \sigma(X \beta) = \frac{1}{1 + e^{-X \beta}}$ sin dependencias externas, `puremacro` implementa `LogisticCoordinateDescent`. La log-verosimilitud negativa con penalización ElasticNet es:

$$\min_{\beta_0, \beta} -\frac{1}{N} \sum_{i=1}^N \left[ D_i \ln p_i + (1 - D_i) \ln(1 - p_i) \right] + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right]$$

Dado que la matriz hessiana logística $p_i(1 - p_i)$ está acotada superiormente por $1/4$, las actualizaciones por coordenadas utilizan la cota cuadrática global:

$$c_j = \frac{1}{4N} \sum_{i=1}^N X_{i, j}^2, \qquad c_0 = \frac{1}{4}$$

En cada barrido por coordenadas, el gradiente es $g_j = \frac{1}{N} X_j^\top (p - D)$ y la actualización es:

$$\rho_j = c_j \beta_j - g_j$$

$$\beta_j^{\text{nuevo}} = \frac{S\big(\rho_j, \, \lambda \alpha\big)}{c_j + \lambda(1 - \alpha)}$$

donde $S(z, \tau) \equiv \operatorname{sign}(z) \max(0, |z| - \tau)$ es el operador de umbral suave (*soft-thresholding*). La penalización óptima $\lambda^\star$ se selecciona a lo largo de una trayectoria geométrica mediante el Criterio de Información Bayesiano (BIC) o de Akaike (AIC).

### 1.3 Recorte de Soporte Común (*Overlap Trimming*)

Cuando las propensiones se aproximan a $0$ o $1$, los ponderadores inversos $1/m(X)$ y $1/(1 - m(X))$ divergen, desestabilizando las varianzas en muestras finitas. `DoubleMLIRM` aplica dos reglas de recorte:

- **`trimming_rule="clip"`**: Proyecta las propensiones al intervalo cerrado:
  $$\hat{m}(X) \leftarrow \operatorname{clip}\big(\hat{m}(X), \, \varepsilon, \, 1 - \varepsilon\big), \qquad \varepsilon = \text{trimming\_threshold}$$
- **`trimming_rule="drop"`**: Descarta de la función de momentos las observaciones con $\hat{m}(X_i) < \varepsilon$ o $\hat{m}(X_i) > 1 - \varepsilon$.

### 1.4 Variables Instrumentales con Doble ML (DML-IV)

Cuando el tratamiento $D$ es endógeno, sean $Z \in \mathbb{R}^{k_z}$ los instrumentos excluidos:

$$Y = D^\top \theta_0 + g_0(X) + U, \qquad \mathbb{E}[U \mid X, Z] = 0$$

$$D = r_0(X, Z) + V$$

El estimador DML-IV residualiza tres esperanzas condicionales mediante ajuste cruzado en $K$ pliegues:
1. $\ell_0(X) \equiv \mathbb{E}[Y \mid X] \implies \tilde{Y} = Y - \hat{\ell}^{(-k)}(X)$
2. $m_0(X) \equiv \mathbb{E}[D \mid X] \implies \tilde{D} = D - \hat{m}^{(-k)}(X)$
3. $r_0(X) \equiv \mathbb{E}[Z \mid X] \implies \tilde{Z} = Z - \hat{r}^{(-k)}(X)$

El parámetro causal $\theta_0$ se estima mediante **Mínimos Cuadrados en Dos Etapas (MC2E / 2SLS)** sobre los residuos fuera del pliegue:

$$\hat{\theta} = \left( \tilde{Z}^\top \tilde{D} \right)^{-1} \tilde{Z}^\top \tilde{Y}$$

#### Diagnóstico de Instrumentos Débiles (Montiel Olea y Pflueger 2013)

Para verificar la robustez de la inferencia, `DoubleMLIV` calcula el estadístico $F$ convencional de primera etapa y el **estadístico $F$ efectivo de Montiel Olea y Pflueger (2013)**, robusto a heterocedasticidad arbitraria:

$$F_{\text{eff}} = \frac{\tilde{D}^\top \tilde{Z} (\tilde{Z}^\top \tilde{Z})^{-1} \tilde{Z}^\top \tilde{D}}{\operatorname{tr}(\hat{W})}$$

`DMLIVResult` reporta `first_stage_f`, `first_stage_effective_f` y activa `weak_instrument = True` cuando $F_{\text{eff}} < 10.0$ (o por debajo del umbral crítico para un sesgo de Nagar máximo del 10%).

---

## 2. Clases y Métodos de la API

| Objeto | Tipo | Descripción |
|---|---|---|
| `DoubleMLIRM` | Clase | Estimador del Modelo de Regresión Interactiva para ATE y ATT. |
| `dml_irm` | Función | Interfaz funcional de una llamada que retorna `DMLIRMResult`. |
| `DMLIRMResult` | Dataclass congelada | Estimaciones, errores estándar y diagnósticos de soporte común. |
| `DoubleMLIV` | Clase | Estimador de Variables Instrumentales con Doble ML. |
| `dml_iv` | Función | Interfaz funcional de una llamada que retorna `DMLIVResult`. |
| `DMLIVResult` | Dataclass congelada | Estimaciones IV y diagnósticos de instrumentos débiles. |
| `LogisticCoordinateDescent` | Clase | Clasificador logístico regularizado en NumPy puro con búsqueda por AIC/BIC. |

---

## 3. Ejemplo Práctico de Estimación

### 3.1 Estimación de Efectos Heterogéneos (DML-IRM)

```python
import numpy as np
import pandas as pd
from puremacro.causal import DoubleMLIRM, LogisticCoordinateDescent

# 1. Simulación de datos con confusión y efectos heterogéneos
rng = np.random.default_rng(2026)
N, p = 1000, 20
X = rng.normal(size=(N, p))

# Propensión condicional a las primeras variables
logit_m = 0.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2]
m_prob = 1.0 / (1.0 + np.exp(-logit_m))
D = rng.binomial(1, m_prob)

# Resultado con efecto medio ATE = 2.5
true_ate = 2.5
Y = 1.2 * X[:, 0] + 0.8 * X[:, 1]**2 + D * (true_ate + 0.5 * X[:, 0]) + rng.normal(scale=1.0, size=N)

# 2. Estimación con DoubleMLIRM
irm = DoubleMLIRM(
    ml_g="ridge",
    ml_m="logistic",
    n_folds=5,
    score="ATE",
    trimming_threshold=0.02,
    trimming_rule="clip",
    random_state=42,
)
res_irm = irm.fit(Y, D, X)

print(res_irm.summary())
print(f"ATE Estimado: {res_irm.theta:.4f} ± {1.96 * res_irm.se:.4f} (Verdadero: {true_ate:.2f})")
print(f"Observaciones recortadas: {res_irm.n_trimmed}")
```

### 3.2 Visualización de Soporte Común y Regularización

```python
# 1. Histograma de solapamiento de propensiones por grupo de tratamiento
fig1 = res_irm.plot_overlap()

# 2. Coeficientes regularizados de los modelos de molestia
fig2 = res_irm.plot_coefficients()

# 3. Curva de penalización e información AIC/BIC
fig3 = res_irm.plot_tuning()
```

### 3.3 Variables Instrumentales en Alta Dimensión (DML-IV)

```python
from puremacro.causal import DoubleMLIV

# Generación de datos con tratamiento endógeno D e instrumento Z
U = rng.normal(size=N)
Z = rng.normal(size=N)
D_endo = 1.5 * Z + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.2 * U + rng.normal(scale=0.5, size=N)

true_iv_theta = 1.75
Y_iv = D_endo * true_iv_theta + 2.0 * X[:, 0] + 1.5 * U + rng.normal(scale=1.0, size=N)

# Ajuste con DoubleMLIV
iv_model = DoubleMLIV(
    ml_l="lasso",
    ml_m="lasso",
    ml_r="lasso",
    n_folds=5,
    random_state=42,
)
res_iv = iv_model.fit(Y_iv, D_endo, Z, X)

print(res_iv.summary())
print(f"Parámetro IV : {res_iv.theta:.4f} (Verdadero: {true_iv_theta:.2f})")
print(f"Estadístico F de 1ra etapa : {res_iv.first_stage_f:.2f}")
print(f"Estadístico F efectivo     : {res_iv.first_stage_effective_f:.2f}")
print(f"¿Instrumento Débil?        : {res_iv.weak_instrument}")
```

---

## 4. Exportación a Tablas de Publicación

```python
# Tabla en Markdown para GitHub
print(res_irm.to_markdown())

# Tabla en LaTeX para artículos científicos
print(res_irm.to_latex())

# Tabla en Typst
print(res_irm.to_typst())
```

---

## Referencias

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. y Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68.
2. Montiel Olea, J. L. y Pflueger, C. (2013). "A robust test for weak instruments." *Journal of Business & Economic Statistics*, 31(3), 358–369.
3. Robinson, P. M. (1988). "Root-N-consistent semiparametric regression." *Econometrica*, 56(4), 931–954.
