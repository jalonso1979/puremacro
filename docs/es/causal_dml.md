> 🇬🇧 [English](../causal_dml.md) · 🇪🇸 Español

# Aprendizaje Automático Doble / Desesgado (DML-PLR)

`puremacro.causal.dml` (reexportado desde `puremacro.causal`) implementa el estimador de **Aprendizaje Automático Doble / Desesgado** (*Double / Debiased Machine Learning*, DML) de **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey y Robins (2018, *The Econometrics Journal*)** para el modelo de **regresión parcialmente lineal (PLR)**, construido sobre la idea de parcialización (*partialling-out*) de **Robinson (1988, *Econometrica*)** y el programa de controles de alta dimensión de **Belloni, Chernozhukov y Hansen (2014, *Review of Economic Studies*)**. El módulo expone cinco objetos públicos:

1. `DoubleMLPLR` — la clase estimadora, `DoubleMLPLR(...).fit(Y, D, X) -> DMLResult`.
2. `dml_plr` — una interfaz funcional de una sola llamada con los mismos argumentos.
3. `DMLResult` — una dataclass congelada que contiene las estimaciones, los residuos fuera del pliegue y la suite de presentación `summary` / `plot` / `to_markdown` / `to_latex` / `to_typst`.
4. `LassoCoordinateDescent` — un aprendiz (*learner*) $\ell_1$ (descenso cíclico por coordenadas a lo largo de una trayectoria de regularización, con el modelo seleccionado por BIC o AIC).
5. `RidgeGCV` — un aprendiz $\ell_2$ (forma cerrada vía SVD, con la penalización seleccionada por validación cruzada generalizada).

Todo es NumPy / SciPy / pandas / Matplotlib puro en float64, no requiere acceso a la red y se ejecuta sin cambios en Pyodide. Complementa las demás páginas causales, [Diferencias en Diferencias Modernas](did.md) y [Análisis de sensibilidad en diferencias en diferencias honesto](honest_did.md), y se muestra de principio a fin en el cuaderno 57 (`notebooks/57_multiconstraint_occbin_and_dml.py`, con versión en español en `notebooks/57_multiconstraint_occbin_and_dml_es.py`), donde se estima un multiplicador de política con controles macroeconómicos de alta dimensión y se compara con MCO ingenuo y con lasso ingenuo.

---

## 1. Marco teórico y algorítmico

### 1.1 El modelo parcialmente lineal

Sea $Y$ la variable de resultado, $D \in \mathbb{R}^{k_d}$ la(s) variable(s) de tratamiento o de política cuyo efecto interesa, y $X \in \mathbb{R}^p$ un vector (posiblemente de alta dimensión) de controles. El modelo PLR es

$$Y = D^\top \theta_0 + g_0(X) + U, \qquad \mathbb{E}[U \mid X, D] = 0,$$

$$D = m_0(X) + V, \qquad \mathbb{E}[V \mid X] = 0,$$

donde $\theta_0$ es el parámetro estructural de baja dimensión y $g_0$ y $m_0$ son funciones de molestia (*nuisance functions*) desconocidas y potencialmente no lineales. Tomar esperanzas condicionales en la primera ecuación define una tercera función de molestia, $\ell_0(X) \equiv \mathbb{E}[Y \mid X] = m_0(X)^\top \theta_0 + g_0(X)$.

### 1.2 Parcialización de Robinson y el score ortogonal de Neyman

Restar $\ell_0(X)$ de $Y$ y $m_0(X)$ de $D$ elimina por completo las funciones de molestia (Robinson 1988):

$$Y - \ell_0(X) = \big(D - m_0(X)\big)^\top \theta_0 + U .$$

La condición de momentos implícita es el **score ortogonal de Robinson / Neyman**

$$\psi(W; \theta, \eta) = \Big(Y - \ell(X) - \big(D - m(X)\big)^\top \theta\Big)\big(D - m(X)\big), \qquad \eta = (\ell, m),$$

con $\mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0$. El score es ortogonal porque su derivada de Gateaux respecto de las funciones de molestia se anula en el valor verdadero: perturbar $\ell$ en la dirección $h(X)$ da $-\mathbb{E}[h(X) V] = 0$, y perturbar $m$ en la dirección $h(X)$ da $\mathbb{E}[h(X)(\theta_0 V - U)] = 0$, ambas por $\mathbb{E}[V \mid X] = 0$ y $\mathbb{E}[U \mid X, D] = 0$. Los errores de estimación de primer orden en $\hat\ell$ y $\hat m$ no se transmiten, por tanto, a $\hat\theta$; solo importa su producto, que es $o_p(N^{-1/2})$ siempre que cada aprendiz converja a tasa $o_p(N^{-1/4})$. Esto es lo que elimina el sesgo de regularización de un lasso de una sola ecuación de $Y$ sobre $(D, X)$.

Escribiendo $\tilde Y_i = Y_i - \hat\ell(X_i)$ y $\tilde D_i = D_i - \hat m(X_i)$, la solución del score empírico es el coeficiente MCO de $\tilde Y$ sobre $\tilde D$ sin intercepto:

$$\hat\theta = \Big( \sum_{i=1}^N \tilde D_i \tilde D_i^\top \Big)^{-1} \sum_{i=1}^N \tilde D_i \tilde Y_i .$$

`puremacro` resuelve este sistema $k_d \times k_d$ con `np.linalg.solve` y recurre a la pseudoinversa de Moore-Penrose si los tratamientos residualizados son colineales.

### 1.3 Ajuste cruzado en $K$ pliegues (DML2: una única resolución agrupada)

Usar las mismas observaciones para ajustar $\hat\ell, \hat m$ y para evaluar el score reintroduce un sesgo de sobreajuste. El ajuste cruzado (*cross-fitting*) lo elimina:

1. Se extrae una permutación aleatoria de los $N$ índices con `np.random.default_rng(random_state)` y se divide en $K$ pliegues (*folds*) $I_1, \dots, I_K$ de tamaño (casi) igual con `np.array_split`.
2. Para cada pliegue $k$, se ajusta un aprendiz para $\ell$ y un aprendiz **por columna de tratamiento** para $m$ sobre el complemento $I_k^c$, y a continuación se residualiza el pliegue retenido: $\tilde Y_i = Y_i - \hat\ell^{(-k)}(X_i)$ y $\tilde D_i = D_i - \hat m^{(-k)}(X_i)$ para $i \in I_k$. Esto cuesta $K (1 + k_d)$ ajustes de aprendiz en total.
3. Se apilan los residuos fuera del pliegue de las $N$ observaciones y se resuelven **una sola vez** las ecuaciones normales de la Sección 1.2.

El paso 3 es la variante agrupada "DML2" de Chernozhukov et al. (2018, Definición 3.2): la estimación puntual es la solución del score sumado sobre todos los pliegues, no el promedio de $K$ estimaciones específicas de cada pliegue (DML1). El encabezado de `summary()` lo reporta como `Score: Robinson Orthogonal (DML2)`, y la estimación puede reproducirse exactamente a partir de los residuos almacenados en el resultado, $\hat\theta = (\tilde D^\top \tilde D)^{-1} \tilde D^\top \tilde Y$. Como la partición es una única permutación con semilla, la estimación depende de `random_state` (véase la Sección 6).

### 1.4 Varianza sándwich e inferencia

Con $\hat U_i = \tilde Y_i - \tilde D_i^\top \hat\theta$, el estimador sándwich por sustitución (*plug-in*) es

$$\hat J = \frac{1}{N} \sum_i \tilde D_i \tilde D_i^\top, \qquad \hat\Omega = \frac{1}{N} \sum_i \hat U_i^2 \, \tilde D_i \tilde D_i^\top, \qquad \hat V = \frac{1}{N} \hat J^{-1} \hat\Omega \hat J^{-1},$$

es decir, una varianza robusta a heterocedasticidad (HC0) para la regresión de residuos sobre residuos, que es la varianza $\sqrt{N}$-asintótica del Teorema 3.1 de Chernozhukov et al. (2018). Los errores estándar son $\sqrt{\operatorname{diag} \hat V}$, `t_stat` es $\hat\theta / \widehat{se}$ (`nan` si un error estándar es cero), `p_value` es bilateral bajo la normal estándar, y el intervalo de confianza es $\hat\theta \pm z_{1 - \alpha / 2} \, \widehat{se}$ con el cuantil normal de `scipy.stats.norm`. No se aplica corrección por grados de libertad, agrupamiento por conglomerados (*clustering*) ni ajuste HAC.

### 1.5 Aprendices para las funciones de molestia

**Lasso (`LassoCoordinateDescent`).** Con `fit_intercept=True` (valor por defecto) las columnas de $X$ se estandarizan a media cero y varianza unitaria y $y$ se centra; con `fit_intercept=False` la penalización actúa sobre las columnas originales, y la actualización por coordenadas divide por el segundo momento de cada columna $c_j = \tfrac{1}{N} x_j^\top x_j$, de modo que sigue siendo el minimizador exacto por coordenada sea cual sea la escala de la columna. El aprendiz minimiza

$$\frac{1}{2N} \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1$$

mediante descenso cíclico por coordenadas con la actualización de umbralización suave (*soft-thresholding*) $\beta_j \leftarrow S(\rho_j, \alpha) / c_j$, $\rho_j = \tfrac{1}{N} x_j^\top r + c_j \beta_j$, deteniéndose cuando el mayor cambio de un coeficiente en un barrido es inferior a `tol` o tras `max_iter` barridos. Cuando `alpha=None` (valor por defecto) el aprendiz recorre una trayectoria geométrica de `n_alphas` penalizaciones desde $\alpha_{\max} = \max_j |x_j^\top y_c| / N$ (todos los coeficientes nulos; sustituido por $10^{-3}$ si cae por debajo de $10^{-12}$) hasta $\alpha_{\max} \cdot$ `eps` (con un piso de $10^{-7}$), con arranque en caliente (*warm start*) de cada paso desde la solución anterior, y conserva la penalización que minimiza

$$\text{BIC}(\alpha) = N \log \hat\sigma^2_\alpha + \mathrm{df}_\alpha \log N \qquad \text{o} \qquad \text{AIC}(\alpha) = N \log \hat\sigma^2_\alpha + 2 \, \mathrm{df}_\alpha ,$$

donde $\hat\sigma^2_\alpha$ es el residuo cuadrático medio dentro de muestra y $\mathrm{df}_\alpha$ el número de coeficientes no nulos (`criterion="bic"` o `"aic"`). La penalización elegida se almacena en `alpha_`, y `coef_` / `intercept_` se reportan en la escala original. Se trata de una vía por criterio de información, y no de validación cruzada ni de la penalización teóricamente calibrada de Belloni, Chernozhukov y Hansen (2014); el bucle interno sobre las $p$ coordenadas es a nivel de Python, de modo que el costo crece linealmente en $p$ por barrido.

**Ridge (`RidgeGCV`).** Tras la misma estandarización, el aprendiz calcula una única SVD reducida (*economy SVD*) $X_s = U S V^\top$ y evalúa, para cada penalización $a$ de la malla (`alphas`, por defecto `np.logspace(-4, 6, 100)`), el criterio de validación cruzada generalizada

$$\text{GCV}(a) = \frac{\operatorname{RSS}(a) / N}{\big(1 - \operatorname{tr} H(a) / N\big)^2}, \qquad \operatorname{tr} H(a) = \sum_i \frac{s_i^2}{s_i^2 + a},$$

en $O(p)$ por punto de la malla, y luego fija $\hat\beta = V \operatorname{diag}\!\big(s_i / (s_i^2 + a^\star)\big) U^\top y_c$ en el minimizador de GCV $a^\star$ (almacenado en `alpha_`). La penalización actúa sobre la función objetivo sin normalizar $\lVert y_c - X_s \beta \rVert_2^2 + a \lVert \beta \rVert_2^2$ con columnas estandarizadas (de modo que $s_i^2 \approx N$ para regresores ortogonales), que es la escala que debe usarse al suministrar una malla `alphas` propia.

**Aprendices suministrados por el usuario.** Cualquier objeto que exponga `fit(X, y)` y `predict(X)` puede pasarse como `learner`. Si es una clase (o cualquier invocable), se llama con `learner_kwargs` para construir un aprendiz nuevo en cada regresión auxiliar; si es una instancia, se copia con `copy.deepcopy` en cada regresión auxiliar, de modo que cada uno de los $K(1 + k_d)$ ajustes recibe su propio objeto y la instancia que usted pasó nunca se ajusta (su `coef_` sigue siendo `None` al terminar). Combinar una instancia con `learner_kwargs` lanza `ValueError` en lugar de descartar los argumentos en silencio: configure la propia instancia, o pase la clase.

---

## 2. Opciones metodológicas y del modelo

| Dimensión | `learner="lasso"` (`"l1"`) | `learner="ridge"` (`"l2"`) | Objeto `fit`/`predict` propio |
|---|---|---|---|
| **Modelo de las funciones de molestia** | Lineal y disperso en las variables suministradas | Lineal y denso, contrae todos los coeficientes | Cualquiera (árboles, núcleos, una base propia) |
| **Selección de la penalización** | Trayectoria de `n_alphas` penalizaciones, BIC (por defecto) o AIC | GCV sobre la malla `alphas` (100 puntos por defecto) | Dentro del objeto |
| **Penalización fija** | `learner_kwargs={"alpha": a}` o `LassoCoordinateDescent(alpha=a)` | `RidgeGCV(alphas=[a])` | n/a |
| **Costo por ajuste** | Bucle en Python sobre $p$ coordenadas $\times$ barridos $\times$ longitud de la trayectoria | Una SVD del bloque de entrenamiento $(N_{\text{train}} \times p)$ | El suyo |
| **Más indicado para** | Muchos controles candidatos, pocos relevantes ($p$ hasta unos cientos) | Controles correlacionados, $p \ll N$, velocidad | Funciones de molestia no lineales sin diccionarios construidos a mano |
| **Campo `learner` reportado** | La cadena de alias que se pasó | La cadena de alias que se pasó | El nombre de la clase (`'RidgeGCV'`) tanto para una clase como para una instancia; el `__name__` de la función para una función fábrica |

Común a todas las opciones: `n_folds` $K$ con $2 \le K \le N$ (por defecto 5), `alpha` como nivel de significancia del IC (por defecto 0.05, es decir, intervalos al 95%), `random_state` para la permutación de los pliegues (por defecto 42), y entradas como arrays de NumPy, `Series` o `DataFrame` de pandas. Ambos aprendices incorporados son lineales en las columnas de `X`; capturar la confusión no lineal es tarea del usuario mediante un diccionario de transformaciones (cuadrados, interacciones, splines), como en el Ejemplo 1.

---

## 3. Ejemplos prácticos ejecutables

Todos los scripts siguientes se ejecutan sin conexión en bastante menos de un segundo de cómputo (más la importación del paquete) y llevan semilla.

### 3.1 Efecto verdadero conocido con un factor de confusión no lineal: MCO ingenuo frente a DML

$X_1$ entra en la ecuación del tratamiento a través de $X_1 + X_1^2 - 1$ y en la de resultado a través de $X_1 + 1.5 (X_1^2 - 1)$. La regresión MCO de $Y$ solo sobre $D$ absorbe el factor de confusión; añadir $X$ linealmente no ayuda porque $X_1^2 - 1$ no está correlacionado con $X_1$. DML con un diccionario cuadrático permite que el lasso seleccione $X_1^2$ en ambas regresiones auxiliares.

```python
import numpy as np
from puremacro.causal import dml_plr

rng = np.random.default_rng(0)
N, p, theta_0 = 1000, 10, 0.5
X = rng.standard_normal((N, p))
q = X[:, 0] ** 2 - 1.0                                   # factor de confusión no lineal
D = X[:, 0] + q + rng.standard_normal(N)                 # m_0(X) = X_1 + X_1^2 - 1
Y = theta_0 * D + X[:, 0] + 1.5 * q + 0.5 * X[:, 1] + rng.standard_normal(N)

def ols_slope(y, regressors):
    Z = np.column_stack([np.ones(len(y)), regressors])
    return np.linalg.lstsq(Z, y, rcond=None)[0][1]

theta_ols_d = ols_slope(Y, D[:, None])                   # Y sobre D únicamente
theta_ols_lin = ols_slope(Y, np.column_stack([D, X]))     # Y sobre D y X lineal

dictionary = np.column_stack([X, X ** 2])                # 2p = 20 controles
res = dml_plr(Y, D, dictionary, n_folds=5, learner="lasso", random_state=0)

print(f"true theta            : {theta_0:.3f}")
print(f"OLS, D only           : {theta_ols_d:.3f}")
print(f"OLS, D + linear X     : {theta_ols_lin:.3f}")
print(f"DML-PLR (lasso, K=5)  : {res.theta:.3f}  se={res.se:.3f}  95% CI=[{res.ci_lower:.3f}, {res.ci_upper:.3f}]")
print(res.summary())
```

Salida:

```text
true theta            : 0.500
OLS, D only           : 1.535
OLS, D + linear X     : 1.546
DML-PLR (lasso, K=5)  : 0.533  se=0.033  95% CI=[0.469, 0.597]
==============================================================================
Double / Debiased Machine Learning (DML-PLR)
Model: Partially Linear Regression (Robinson 1988 / Chernozhukov et al. 2018)
==============================================================================
Outcome: Y                  Observations: 1000         Folds: 5
Learner: lasso              Score: Robinson Orthogonal (DML2)
------------------------------------------------------------------------------
Variable              Coef.   Std.Err.        z    P>|z| [95% Conf. Interval]
------------------------------------------------------------------------------
D                    0.5331     0.0328   16.259   <0.001     0.4688     0.5973
==============================================================================
```

Ambas especificaciones MCO triplican el efecto verdadero (1.535 y 1.546 frente a 0.5); DML devuelve 0.533 con un intervalo al 95% $[0.469, 0.597]$ que cubre el valor verdadero. En este caso deliberadamente de baja dimensión, MCO sobre el diccionario completo de 20 columnas también sería insesgado (0.524 con estos datos); la ventaja de DML es que el diccionario puede crecer con $N$ (todas las interacciones por pares, bases de splines) mientras la inferencia sobre $\theta$ sigue siendo válida a tasa $\sqrt{N}$.

### 3.2 Lasso frente a ridge, argumentos con nombre del aprendiz y un aprendiz suministrado por el usuario

El proceso generador de datos es deliberadamente disperso: solo $X_1$ determina $D$ y solo $X_2$ determina $Y$ más allá de $D$, con valor verdadero $\theta_0 = 0.5$.

```python
import numpy as np
from puremacro.causal import dml_plr, LassoCoordinateDescent, RidgeGCV

rng = np.random.default_rng(0)
N, p = 300, 20
X = rng.standard_normal((N, p))
D = X[:, 0] + rng.standard_normal(N)
Y = 0.5 * D + X[:, 1] + rng.standard_normal(N)

res_lasso = dml_plr(Y, D, X, n_folds=3, learner="lasso")                  # trayectoria BIC (por defecto)
res_aic = dml_plr(Y, D, X, n_folds=3, learner="lasso", criterion="aic")   # argumento con nombre del aprendiz
res_ridge = dml_plr(Y, D, X, n_folds=3, learner="ridge")                  # SVD-GCV

class OLSLearner:
    """Any object with fit(X, y) and predict(X) is a valid nuisance learner."""
    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(y)), X])
        self.beta_ = np.linalg.lstsq(Z, y, rcond=None)[0]
        return self
    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta_

res_ols = dml_plr(Y, D, X, n_folds=3, learner=OLSLearner())

for r in (res_lasso, res_aic, res_ridge, res_ols):
    print(f"{r.learner:<12} theta={r.theta:.4f}  se={r.se:.4f}  z={r.t_stat:.2f}")

# Los aprendices independientes pueden usarse por sí solos
lasso = LassoCoordinateDescent(criterion="bic").fit(X, D)
ridge = RidgeGCV().fit(X, D)
print("lasso: alpha_=%.4f  nonzero coef=%d" % (lasso.alpha_, np.count_nonzero(lasso.coef_)))
print("ridge: alpha_=%.4f  coef_[0]=%.3f" % (ridge.alpha_, ridge.coef_[0]))
```

Salida:

```text
lasso        theta=0.4770  se=0.0522  z=9.13
lasso        theta=0.4624  se=0.0516  z=8.96
ridge        theta=0.4645  se=0.0514  z=9.04
OLSLearner   theta=0.4660  se=0.0518  z=8.99
lasso: alpha_=0.1102  nonzero coef=1
ridge: alpha_=17.8865  coef_[0]=1.004
```

Los cuatro aprendices quedan a menos de un error estándar entre sí (0.462 a 0.477) porque $N = 300$ supera holgadamente a $p = 20$; el lasso BIC usado por sí solo selecciona exactamente el único regresor verdadero de $D$, y ridge-GCV contrae ese mismo coeficiente a 1.004. El argumento `criterion="aic"` se reenvía a `LassoCoordinateDescent` a través de `**learner_kwargs`; el campo `learner` del resultado sigue siendo `"lasso"` porque registra el alias, no la configuración.

### 3.3 Tratamientos múltiples, entradas pandas y exportación a manuscritos

Con un `DataFrame` de tratamientos las estimaciones se devuelven como arrays, se ajusta una regresión auxiliar por columna de tratamiento, y los nombres de las columnas y de la `Series` fluyen a las tablas. `alpha=0.10` solicita intervalos al 90%.

```python
import numpy as np
import pandas as pd
from puremacro.causal import DoubleMLPLR

rng = np.random.default_rng(3)
N, p = 400, 8
X = pd.DataFrame(rng.standard_normal((N, p)), columns=[f"x{j}" for j in range(p)])
D = pd.DataFrame({
    "policy_rate": 0.8 * X["x0"] + rng.standard_normal(N),
    "credit_spread": -0.6 * X["x1"] + rng.standard_normal(N),
})
Y = pd.Series(
    -0.4 * D["policy_rate"] + 0.9 * D["credit_spread"] + X["x0"] + X["x1"] + rng.standard_normal(N),
    name="gdp_growth",
)

res = DoubleMLPLR(n_folds=4, learner="ridge", alpha=0.10, random_state=7).fit(Y, D, X)
print(res.treatment_names, res.outcome_name)
print("theta:", np.round(res.theta, 4), " se:", np.round(res.se, 4))
print("residuals_d shape:", res.residuals_d.shape, " ci_level:", res.ci_level)
print(res.to_markdown())
print(res.to_typst())

# No hay to_frame(): construya un DataFrame a partir de los campos cuando lo necesite
table = pd.DataFrame(
    {"theta": res.theta, "se": res.se, "ci_lower": res.ci_lower, "ci_upper": res.ci_upper},
    index=res.treatment_names,
)
print(table.round(4))
```

Salida:

```text
('policy_rate', 'credit_spread') gdp_growth
theta: [-0.394   0.8464]  se: [0.0479 0.0488]
residuals_d shape: (400, 2)  ci_level: 0.9
### DML-PLR: gdp_growth
*Learner: ridge | N: 400 | Folds: 4*

| Variable | Coef. | Std.Err. | z | P>\|z\| | [90% Conf. Interval] |
|:---|---:|---:|---:|---:|:---:|
| policy_rate | -0.3940 | 0.0479 | -8.229 | <0.001 | [-0.4728, -0.3152] |
| credit_spread | 0.8464 | 0.0488 | 17.342 | <0.001 | [0.7662, 0.9267] |
#figure(
  table(
    columns: (2fr, 1.2fr, 1.2fr, 1fr, 1fr, 2fr),
    align: (left, right, right, right, right, center),
    stroke: none,
    table.hline(),
    [*Variable*], [*Coef.*], [*Std.Err.*], [*z*], [*P>|z|*], [*90% CI*],
    table.hline(stroke: 0.5pt),
    [policy\_rate], [-0.3940], [0.0479], [-8.229], [\<0.001], [[-0.4728, -0.3152]],
    [credit\_spread], [0.8464], [0.0488], [17.342], [\<0.001], [[0.7662, 0.9267]],
    table.hline(),
  ),
  caption: [Double Machine Learning Estimates (ridge, N=400)],
)
                theta      se  ci_lower  ci_upper
policy_rate   -0.3940  0.0479   -0.4728   -0.3152
credit_spread  0.8464  0.0488    0.7662    0.9267
```

Los efectos verdaderos eran $-0.4$ y $0.9$; los intervalos al 90% $[-0.473, -0.315]$ y $[0.766, 0.927]$ cubren ambos. `to_markdown()` escapa las barras verticales del encabezado `P>|z|` como `P>\|z\|`, de modo que las filas de encabezado y de delimitadores coinciden en seis celdas y la tabla se renderiza en GFM estricto y en mkdocs; `to_typst()` escapa los guiones bajos de los nombres de los tratamientos y el `<` de la celda del valor p. `to_latex()` produce la tabla booktabs correspondiente, escapando también `gdp_growth` en `\caption{}` (véase la Sección 5).

---

## 4. Especificación completa de la API

```text
DoubleMLPLR(
    n_folds: int = 5,
    learner: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    learner_kwargs: dict[str, Any] | None = None,
)

DoubleMLPLR.fit(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
) -> DMLResult

dml_plr(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    learner: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    **learner_kwargs: Any,
) -> DMLResult

LassoCoordinateDescent(
    alpha: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-5,
    n_alphas: int = 50,
    eps: float = 1e-3,
    criterion: str = "bic",
    fit_intercept: bool = True,
)
LassoCoordinateDescent.fit(X, y) -> LassoCoordinateDescent
LassoCoordinateDescent.predict(X) -> np.ndarray

RidgeGCV(
    alphas: Sequence[float] | np.ndarray | None = None,
    fit_intercept: bool = True,
)
RidgeGCV.fit(X, y) -> RidgeGCV
RidgeGCV.predict(X) -> np.ndarray
```

#### Parámetros de `DoubleMLPLR` / `dml_plr`

| Parámetro | Significado |
|---|---|
| `Y` | Variable de resultado, de forma $(N,)$. El nombre de una `Series` de pandas se convierte en `outcome_name` (por defecto `"Y"`). |
| `D` | Tratamiento(s), de forma $(N,)$ o $(N, k_d)$. El nombre de una `Series` o las columnas de un `DataFrame` se convierten en `treatment_names`; los arrays sin nombre reciben `"D"` o `"D_1", "D_2", ...`. |
| `X` | Controles, de forma $(N, p)$, array o `DataFrame` (convertido con `to_numpy(dtype=float)`). Una `X` unidimensional se lee como una única columna; tres o más dimensiones lanzan `ValueError`, y el número de filas debe coincidir o se lanza un `ValueError`. |
| `n_folds` | Número de pliegues de ajuste cruzado $K$; debe ser un entero $\ge 2$ y como máximo $N$ (`ValueError` en caso contrario). |
| `learner` | `"lasso"` / `"l1"`, `"ridge"` / `"l2"`, una clase de aprendiz o un invocable (instanciado en cada regresión auxiliar con `learner_kwargs`), o una instancia de aprendiz (copiada en profundidad en cada regresión auxiliar). Cualquier otra cadena lanza `ValueError`; un objeto no invocable sin `fit`/`predict` lanza `TypeError`. |
| `alpha` | Nivel de significancia de los intervalos de confianza; `ci_level = 1 - alpha`. Debe estar estrictamente entre 0 y 1 (`ValueError` en caso contrario). **No** es la penalización del lasso. |
| `random_state` | Semilla de la permutación de los pliegues. `None` extrae una permutación nueva en cada llamada, de modo que los resultados no son reproducibles. |
| `learner_kwargs` / `**learner_kwargs` | Argumentos con nombre reenviados al constructor del aprendiz, p. ej. `criterion="aic"`, `n_alphas=100`, `alphas=np.logspace(-2, 4, 50)`. Como `alpha` está reservado para el nivel del IC, una penalización fija del lasso debe pasar por `DoubleMLPLR(learner_kwargs={"alpha": ...})` o por una instancia `LassoCoordinateDescent(alpha=...)`. Pasar argumentos con nombre junto con una *instancia* de aprendiz lanza `ValueError`. |

`fit` valida sus entradas antes de construir ningún aprendiz: `Y`, `D` o `X` con `NaN` o `inf` lanza `ValueError` nombrando el array culpable, `D` con tres o más dimensiones o con cero columnas lanza `ValueError`, y un pliegue de entrenamiento con $n_{\text{train}} \le p + 1$ observaciones emite un `UserWarning` (Sección 6).

#### Parámetros de los aprendices

| Parámetro | Aprendiz | Significado |
|---|---|---|
| `alpha` | lasso | Penalización $\ell_1$ fija; `None` la selecciona sobre la trayectoria. |
| `n_alphas`, `eps` | lasso | Longitud de la trayectoria geométrica y cociente $\alpha_{\min} / \alpha_{\max}$. |
| `max_iter`, `tol` | lasso | Número máximo de barridos de descenso por coordenadas por penalización y umbral de convergencia sobre el mayor cambio de un coeficiente. |
| `criterion` | lasso | `"bic"` (por defecto) o `"aic"`, sin distinción entre mayúsculas y minúsculas. |
| `alphas` | ridge | Malla de penalizaciones $\ell_2$ para GCV; por defecto `np.logspace(-4, 6, 100)`. |
| `fit_intercept` | ambos | Centra $y$, estandariza las columnas de $X$ y reporta un intercepto (por defecto `True`). |

Atributos ajustados en ambos aprendices: `coef_` (escala original), `intercept_`, `alpha_` (penalización seleccionada). Llamar a `predict` antes de `fit` lanza `RuntimeError`; `fit` lanza `ValueError` si el diseño no es finito, está vacío o tiene longitudes que no coinciden, y ambos constructores rechazan una penalización negativa o no finita.

---

## 5. Interfaz de resultados y exportación a manuscritos

`DMLResult` es una dataclass congelada; sus atributos son de solo lectura. Para un único tratamiento los estadísticos son `float` de Python y `residuals_d` es unidimensional; para $k_d > 1$ son arrays de longitud $k_d$ y `residuals_d` tiene forma $(N, k_d)$.

| Campo | Contenido |
|---|---|
| `theta` | Estimación(es) DML con ajuste cruzado $\hat\theta$. |
| `se` | Error(es) estándar sándwich. |
| `t_stat`, `p_value` | $\hat\theta / \widehat{se}$ y el valor p normal bilateral. |
| `ci_lower`, `ci_upper` | Límites del intervalo $(1 - \alpha)$. |
| `n_obs`, `n_folds` | $N$ y $K$. |
| `learner` | Cadena de alias que se pasó, o nombre de la clase de una clase o una instancia de aprendiz (el `__name__` de una función fábrica). |
| `residuals_y`, `residuals_d` | $\tilde Y$ y $\tilde D$ fuera del pliegue; suficientes para recomputar $\hat\theta$ y la varianza. |
| `ci_level` | $1 - \alpha$. |
| `treatment_names`, `outcome_name` | Etiquetas usadas por todas las exportaciones. |

Métodos:

- `res.summary() -> str`: informe de texto de ancho fijo (encabezado con variable de resultado, $N$, $K$, aprendiz y tipo de score, y una fila por tratamiento).
- `res.plot(kind="forest", ax=None, **kwargs) -> matplotlib.axes.Axes`: `"forest"` dibuja las estimaciones puntuales con sus intervalos de confianza y una línea en cero; `"residuals"` traza la dispersión de $\tilde D$ frente a $\tilde Y$ con la pendiente DML ajustada (solo el primer tratamiento cuando $k_d > 1$). Acepta `figsize` y `alpha` (transparencia de los marcadores) a través de `kwargs`; devuelve los ejes, no la figura.
- `res.to_markdown() -> str`: encabezado de nivel 3, una línea en cursiva con aprendiz / $N$ / $K$, y una tabla de seis columnas. Las barras verticales dentro de las celdas se escapan como `\|` — el encabezado `P>|z|` y cualquier `|` en un nombre de tratamiento —, de modo que las filas de encabezado y de delimitadores tienen el mismo número de celdas y los renderizadores estrictos de GFM (mkdocs incluido) reconocen la tabla.
- `res.to_latex() -> str`: un entorno `table` con un tabular `booktabs` y nada más; `booktabs` es el único paquete que el documento debe cargar. La nota con el tamaño muestral es una fila `\multicolumn` dentro del tabular, y no un `\subcaption{}`; la celda del valor p se compone en modo matemático (`$<0.001$`) porque un `<` suelto en modo texto OT1 se imprime como un signo de exclamación invertido, y `puremacro.reports.latex_escape` se aplica a los nombres de los tratamientos, a `outcome_name` en `\caption{}` y al nombre del aprendiz.
- `res.to_typst() -> str`: un bloque `#figure(table(...))` con líneas `hline` y un pie de tabla que nombra el aprendiz y $N$; los nombres de los tratamientos, el nombre del aprendiz y la celda `<0.001` pasan por `puremacro.reports.typst_escape`.
- **No** existe `to_frame()`; el Ejemplo 3.3 muestra la construcción en dos líneas de un `pd.DataFrame` a partir de los campos.

---

## 6. Advertencias y limitaciones

- **Solo PLR.** Las variantes interactiva (IRM / ATE), instrumental (PLIV) y de panel de Chernozhukov et al. (2018) no están implementadas; el parámetro es el coeficiente parcialmente lineal, que coincide con el ATE solo bajo efectos constantes.
- **Una sola partición, sin repetición.** Una única permutación con semilla define los pliegues; no hay particiones muestrales repetidas con agregación por mediana o por media. Las estimaciones se mueven con `random_state` (el ajuste ridge del Ejemplo 3.2 da 0.524, 0.506 y 0.465 para `random_state=0`, `1` y `42`), así que reporte la semilla o promedie usted mismo sobre varias.
- **Aprendices lineales.** Lasso y ridge son lineales en las columnas que se pasan; la confusión no lineal debe capturarse con un diccionario explícito de transformaciones (Ejemplo 3.1) o con un aprendiz propio. El bucle de coordenadas del lasso es Python puro sobre $p$, de modo que los diccionarios con miles de columnas son lentos; ridge es una SVD por ajuste.
- **Selección de la penalización.** La trayectoria del lasso usa BIC / AIC con el número de coeficientes no nulos como grados de libertad, no validación cruzada; ridge usa GCV sobre una malla fija. Ninguna de las dos es la penalización teóricamente calibrada de Belloni, Chernozhukov y Hansen (2014).
- **La inferencia es i.i.d. y marginal.** La varianza sándwich es HC0 con valores críticos normales; sin agrupamiento por conglomerados (*clustering*), sin corrección HAC para $U$ serialmente correlacionados (relevante en series temporales macroeconómicas), sin ajuste de muestra pequeña. Con varios tratamientos los intervalos son marginales, y no se devuelve la matriz de covarianzas completa.
- **`alpha` es el nivel del IC**, tanto en `DoubleMLPLR` como en `dml_plr`; la penalización del lasso viaja por `learner_kwargs` o por una instancia de aprendiz.
- **Los valores faltantes debe eliminarlos usted.** `fit` rechaza las entradas no finitas en lugar de propagarlas: un `NaN` o un `inf` en cualquier lugar de `Y`, `D` o `X` lanza `ValueError: X contains NaN or inf; drop or impute missing observations before calling fit()` (nombrando el array culpable) antes de construir ningún aprendiz. Elimine o impute antes; el estimador no hace ninguna de las dos cosas.
- **Los pliegues de entrenamiento escasos solo se advierten.** Cuando el pliegue de entrenamiento más pequeño tiene $n_{\text{train}} \le p + 1$ observaciones, `fit` emite un `UserWarning` y continúa. En ese régimen `RidgeGCV` interpola el pliegue de entrenamiento y ambos aprendices incorporados devuelven estimaciones sesgadas cuyos intervalos nominales al 95% cubren por debajo de lo declarado; aumente $N$, reduzca `n_folds` o recorte el diccionario.
- **Contrato de ejecución.** Sin acceso a la red, sin rutas de código torch / MLX / CuPy, float64 en todo momento; la misma ruta de código NumPy / SciPy se ejecuta sin cambios en Pyodide.

---

## Referencias

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). "Inference on Treatment Effects after Selection among High-Dimensional Controls." *The Review of Economic Studies*, 81(2), 608–650.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68.
- Robinson, P. M. (1988). "Root-N-Consistent Semiparametric Regression." *Econometrica*, 56(4), 931–954.
