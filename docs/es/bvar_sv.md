> 🇬🇧 [English](../bvar_sv.md) · 🇪🇸 Español

# VAR bayesiano con volatilidad estocástica (BVAR-SV)

El módulo `puremacro.var.bvar_sv` implementa modelos de vectores autorregresivos bayesianos con volatilidad residual variable en el tiempo (BVAR-SV) en NumPy/SciPy puro, siguiendo las metodologías de espacio de estados y muestreo MCMC desarrolladas por **Carriero, Clark y Marcellino (2016, 2019)**, **Kim, Shephard y Chib (1998)**, y **Carter y Kohn (1994)**.

Las series macroeconómicas exhiben marcadas alternancias de régimen en su volatilidad —como la Gran Moderación (1984–2007), la conmoción de la Crisis Financiera Global de 2007–2008 y las disrupciones asociadas a la pandemia de 2020—. Los modelos VAR homocedásticos convencionales distorsionan las estimaciones de los parámetros dinámicos y generan densidades predictivas descalibradas al confundir variaciones en la magnitud de los choques con modificaciones en los mecanismos de transmisión estructural.

El estimador BVAR-SV descompone la dinámica macroeconómica en coeficientes autorregresivos constantes y **log-volatilidades estocásticas latentes**, y proporciona respuestas al impulso condicionadas a la volatilidad, gráficos de abanico predictivos y una evaluación honesta de la densidad predictiva.

---

## 1. Marco econométrico y representación en espacio de estados

### 1.1 Modelo estructural

Considérese un modelo $\text{VAR}(p)$ con $n$ variables endógenas y $p$ retardos:

$$y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t, \quad t = 1, \dots, T$$

Las perturbaciones de forma reducida $u_t$ presentan una estructura de covarianza variable en el tiempo:

$$u_t = A^{-1} D_t^{1/2} \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I_n)$$

donde:
1. $A$ es una matriz triangular inferior con unos en su diagonal principal que modeliza las relaciones estructurales contemporáneas:
   $$A = \begin{pmatrix} 1 & 0 & \dots & 0 \\ a_{2,1} & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & 0 \\ a_{n,1} & a_{n,2} & \dots & 1 \end{pmatrix}$$
2. $D_t = \text{diag}\left( \exp(h_{1,t}), \dots, \exp(h_{n,t}) \right)$ contiene las varianzas residuales instantáneas.
3. La matriz de covarianza residual en la fecha $t$ adopta la forma:
   $$\Sigma_t = A^{-1} D_t (A^{-1})'$$
4. Los estados latentes de log-volatilidad $h_{i,t}$ siguen procesos autorregresivos estacionarios $\text{AR}(1)$ independientes para cada variable $i = 1, \dots, n$:
   $$h_{i,t} - \mu_i = \phi_i (h_{i,t-1} - \mu_i) + \sigma_{h,i} \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, 1), \quad |\phi_i| < 1$$
   con la condición inicial estacionaria $h_{i,1} \sim \mathcal{N}\left(\mu_i, \, \frac{\sigma_{h,i}^2}{1 - \phi_i^2}\right)$.

### 1.2 Distribuciones a priori

- Coeficientes del VAR $\beta = \text{vec}(B')$: a priori normal independiente. Con `minnesota_prior=True` (por defecto) se usa la a priori de Minnesota del paquete (`lambda1`, `lambda2`, `lambda3`; primer retardo propio centrado en 1); con `minnesota_prior=False` cada coeficiente de retardo recibe una a priori difusa $\mathcal{N}(0, 100^2)$. En ambos modos las constantes reciben $\mathcal{N}(0, \texttt{intercept\_prior\_std}^2)$.
- Coeficientes contemporáneos: $a_{i,j} \sim \mathcal{N}(0, 100)$.
- Hiperparámetros de la volatilidad (hiper-a priori fijas): $\mu_i \sim \mathcal{N}(0, 10)$, $\phi_i \sim \mathcal{N}(0.85, 0.1)$ truncada a $|\phi_i| < 0.999$, $\sigma_{h,i}^2 \sim \text{IG}(2, 0.05)$. El desplazamiento en $\log(\nu_{i,t}^2 + c)$ es $c = 10^{-6}$.

---

## 2. El algoritmo de muestreo de Gibbs MCMC

El modelo se estima mediante un muestreador de Gibbs estructurado en bloques condicionales:

1. **Paso 1: Coeficientes dinámicos del VAR ($\beta$)**
   Condicionado a la matriz contemporánea $A$ y a la senda de volatilidades $D_{1:T}$, $\beta$ se extrae **conjuntamente** de su distribución condicional exacta
   $$\beta \mid A, h, Y \sim \mathcal{N}(\bar{\beta}, \bar{V}_\beta), \qquad \bar{V}_\beta^{-1} = V_0^{-1} + \sum_t \Sigma_t^{-1} \otimes x_t x_t'$$
   usando la precisión GLS completa de dimensión $(nk \times nk)$ ($k = 1 + np$). Es exacto pero de coste $O((nk)^3)$ por barrido, pensado para sistemas pequeños y medianos; *no* es el algoritmo triangular ecuación por ecuación de Carriero, Clark y Marcellino (2019). Solo se aceptan extracciones con matriz compañera estable (a posteriori truncada a la región estacionaria). Tras 50 rechazos consecutivos se conserva la $B$ anterior; esos barridos se contabilizan en `n_stuck_iterations` (con `n_unstable_rejections` candidatos rechazados) y se emite un `UserWarning`.
2. **Paso 2: Coeficientes estructurales contemporáneos ($A$)**
   Cada fila $i$ de $A$ se extrae ecuación por ecuación (Cogley y Sargent 2005; Primiceri 2005) mediante una regresión lineal del residuo $\hat{u}_{i,t}$ sobre los residuos previos $\hat{u}_{1:i-1,t}$, ponderada por $\exp(-h_{i,t}/2)$.
3. **Paso 3: Mixtura gaussiana de 7 componentes de Kim, Shephard y Chib (1998)**
   La transformación logarítmica de los residuos estandarizados al cuadrado produce un sistema de medida lineal no gaussiano:
   $$y_{i,t}^* \equiv \log\left( (A \hat{u}_t)_i^2 + c \right) = h_{i,t} + \log(\varepsilon_{i,t}^2)$$
   donde la distribución del error $\log(\varepsilon_{i,t}^2) \sim \log(\chi_1^2)$ se aproxima mediante una mixtura de 7 densidades normales con ponderaciones, medias y varianzas $(q_k, m_k, v_k^2)$ tabuladas en Kim, Shephard y Chib (1998, Tabla 4). Los indicadores de mixtura $s_{i,t} \in \{1, \dots, 7\}$ se muestrean condicionados a $h_{i,t}$.
4. **Paso 4: Filtrado hacia adelante y muestreo hacia atrás (FFBS) de Carter y Kohn (1994)**
   Condicionado a los indicadores de la mixtura, el modelo se torna condicionalmente lineal y gaussiano. El algoritmo FFBS de Carter y Kohn extrae la trayectoria completa de estados $h_{i, 1:T}$ de forma conjunta en un único barrido retrógrado.
5. **Paso 5: Parámetros autorregresivos de la volatilidad $(\mu_i, \phi_i, \sigma_{h,i}^2)$**
   $\mu_i$ se extrae de su condicional normal conjugada y $\sigma_{h,i}^2$ de su condicional inversa-gamma conjugada. Para $\phi_i$ se usa el paso Metropolis-Hastings de independencia de Kim, Shephard y Chib (1998, §3.3): la propuesta es la condicional gaussiana de la regresión AR(1) sobre $h_{i,2:T}$ y la probabilidad de aceptación es el cociente de densidades del estado inicial estacionario
   $$\log \frac{p(h_{i,1} \mid \phi')}{p(h_{i,1} \mid \phi)} = \tfrac{1}{2}\log(1-\phi'^2) - \tfrac{1}{2}\log(1-\phi^2) + \tfrac{1}{2}(\phi'^2 - \phi^2)\frac{(h_{i,1}-\mu_i)^2}{\sigma_{h,i}^2}.$$
6. **Paso 6: Diagnósticos de convergencia multicadena de Gelman-Rubin ($\hat{R}$)**
   Se ejecutan `n_chains` cadenas (secuencialmente); las cadenas posteriores a la primera parten de valores iniciales perturbados (nivel de log-volatilidad, $\phi_i$, $\sigma_{h,i}^2$). Cada cadena se divide en dos mitades y se calcula el $\hat{R}$ con división de cadenas para cada coeficiente del VAR, cada $a_{i,j}$, la log-volatilidad media de cada variable y cada $(\mu_i, \phi_i, \sigma_{h,i})$. Valores de $\hat{R} < 1.1$ indican una mezcla adecuada; $\hat{R}$ es un diagnóstico estocástico, así que aumente `n_draws` / `n_burn` si no se cumple.

---

## 3. Respuestas al impulso condicionadas al régimen de volatilidad

Dado que la matriz de impacto estructural $B_t = A^{-1} D_t^{1/2}$ varía temporalmente, la magnitud de la respuesta macroeconómica difiere en función del régimen de volatilidad.

`BVAR_SVResult.irf(horizon=20, t_idx=..., ci=0.9)` condiciona las respuestas al estado de volatilidad en una fecha concreta $t^*$ (índice dentro de la muestra efectiva, es decir, la observación $t^* + p$):
- **Régimen de alta volatilidad (ej. 2008Q4 o 2020Q2)**: Cuantifica la propagación del choque bajo estrés financiero o incertidumbre aguda.
- **Régimen de calma (ej. Gran Moderación)**: Evalúa la propagación en épocas macroeconómicas de baja volatilidad.

El resultado es un array `BVAR_SV_IRF` de forma $(H+1, n, n)$ (`[horizonte, respuesta, choque]`, ordenación de Cholesky) con la mediana a posteriori, más `.lower`, `.upper` y las extracciones completas `.draws` $(D, H+1, n, n)$. `to_frame(target_idx=None, shock_idx=None, names=None)` devuelve una tabla ordenada; cada índice actúa como filtro independiente (puede pasarse uno, ambos o ninguno).

---

## 4. Pronósticos, gráficos de abanico y puntuaciones predictivas

- `forecast(horizon, ci=0.9, seed=None)` simula una trayectoria predictiva a posteriori por cada extracción retenida: las log-volatilidades se propagan con la ley de movimiento AR(1), se extraen los choques estructurales y se itera el VAR hacia adelante, de modo que se integra la incertidumbre paramétrica, de volatilidad y de los choques. El objeto `BVAR_SVForecast` devuelto expone `paths` $(D, H, n)$, `h_paths`, `median`, `mean`, `lower`, `upper`, `quantile(q)`, un `index` con fechas (cuando los datos tienen un `DatetimeIndex` regular), `to_frame()` y gráficos de abanico con `plot(levels=(0.5, 0.8, 0.95))`.
- `log_scores` / `predictive_log_score()` es la densidad predictiva puntual logarítmica **dentro de muestra** (lppd): $\sum_t \log \frac{1}{D}\sum_d p(y_t \mid \beta^{(d)}, A^{(d)}, h_t^{(d)})$ evaluada con las extracciones *suavizadas* de la volatilidad, que condicionan a toda la muestra de estimación. Es una medida de ajuste, no una evaluación de pronósticos.
- `log_score(holdout)` es la puntuación predictiva logarítmica **fuera de muestra** de observaciones posteriores a la muestra de estimación: para cada extracción la media condicional usa los retardos realizados de $y_{T+j}$ y la volatilidad se proyecta desde el estado de fin de muestra con $h_{T+j} = \mu + \phi (h_{T+j-1} - \mu) + \sigma_h \eta$. No se reestiman los parámetros ni se refiltra la volatilidad con los datos reservados. `log_score()` sin argumentos devuelve la lppd dentro de muestra.

---

## 5. Ejemplo de uso y código ejecutable

### Estimación del modelo BVAR-SV

```python
import numpy as np
import pandas as pd
from puremacro.var.bvar_sv import bvar_sv

# 1. Series trimestrales sintéticas (PIB, Inflación, Tipo) con un repunte de volatilidad
rng = np.random.default_rng(42)
T = 180
fechas = pd.date_range("1975-01-01", periods=T, freq="QE")

datos = np.zeros((T, 3))
for t in range(1, T):
    escala_vol = 2.5 if 130 <= t <= 145 else 1.0  # Incremento de volatilidad
    datos[t] = 0.6 * datos[t-1] + escala_vol * 0.5 * rng.standard_normal(3)

df = pd.DataFrame(datos, index=fechas, columns=["PIB", "Inflacion", "Tipo"])

# Se reservan los últimos 12 trimestres para la evaluación fuera de muestra
entrenamiento, prueba = df.iloc[:168], df.iloc[168:]

# 2. Estimación con el muestreador de Gibbs (n_draws es por cadena: 2 x 2000 extracciones)
res_sv = bvar_sv(
    data=entrenamiento,
    lags=2,
    n_draws=2000,
    n_burn=1000,
    n_chains=2,
    minnesota_prior=True,
    lambda1=0.2,
    lambda2=0.5,
    seed=123,
)

# 3. Diagnósticos de convergencia e informe resumen
print(res_sv.summary())
print(f"Máximo R-hat con división de cadenas: {res_sv.max_rhat:.3f}")
assert res_sv.max_rhat < 1.1, "Las cadenas deben converger formalmente (< 1.1)"

# 4. Respuestas al impulso condicionadas: crisis (t_idx=135) frente a calma (t_idx=50)
irf_crisis = res_sv.irf(horizon=16, t_idx=135, ci=0.90)
irf_calma = res_sv.irf(horizon=16, t_idx=50, ci=0.90)

print(f"Respuesta máxima del PIB (crisis): {irf_crisis.median[:, 0, 0].max():.3f}")
print(f"Respuesta máxima del PIB (calma) : {irf_calma.median[:, 0, 0].max():.3f}")
tabla_irf = irf_crisis.to_frame(shock_idx=0, names=res_sv.names)  # todas las respuestas al choque del PIB

# 5. Densidades predictivas: lppd dentro de muestra frente a puntuación fuera de muestra
print(f"lppd dentro de muestra (volatilidades suavizadas): {res_sv.predictive_log_score():.2f}")
print(f"Puntuación logarítmica fuera de muestra, 12 trimestres: {res_sv.log_score(prueba, seed=0):.2f}")

# 6. Pronósticos predictivos a posteriori con gráficos de abanico
fc = res_sv.forecast(horizon=12, ci=0.90, seed=1)
fig_abanico = fc.plot(levels=(0.5, 0.8, 0.95))
cobertura = np.mean((prueba.values >= fc.lower) & (prueba.values <= fc.upper))
print(f"Proporción de observaciones reservadas dentro de la banda predictiva del 90%: {cobertura:.2f}")

# 7. Sendas de volatilidad, desviaciones típicas condicionales e IRF; exportación de tablas
fig = res_sv.plot(t_idx=135, shock_idx=0, target_idx=0)
tabla_latex = res_sv.to_latex()
tabla_md = res_sv.to_markdown()
```

---

## 6. Especificación completa de la API

### `bvar_sv`

```text
bvar_sv(
    data: pd.DataFrame | np.ndarray,
    lags: int = 4,
    n_draws: int = 2000,
    n_burn: int = 1000,
    minnesota_prior: bool = True,
    seed: int | None = None,
    *,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    thin: int = 1,
    n_chains: int = 2,
    p: int | None = None,
) -> BVAR_SVResult
```

#### Parámetros:
- `data`: Panel temporal $(T, n)$ con las variables endógenas; debe ser finito (NaN / inf lanzan `ValueError`).
- `lags` / `p`: Número de retardos del VAR (por defecto `4`). La muestra efectiva $T - p$ debe superar el número de regresores por ecuación $k = 1 + np$.
- `n_draws`: Extracciones MCMC conservadas tras el calentamiento **por cada cadena** (por defecto `2000`, mínimo `2`); el resultado agrupa `n_chains * n_draws` extracciones.
- `n_burn`: Iteraciones iniciales de descarte por cadena (*burn-in*, por defecto `1000`).
- `minnesota_prior`: Aplica la a priori de Minnesota (por defecto `True`); en caso contrario, a priori difusa $\mathcal{N}(0, 100^2)$ sobre los coeficientes de retardo.
- `seed`: Semilla para asegurar reproducibilidad exacta.
- `lambda1`: Parámetro global de contracción de Minnesota (por defecto `0.2`).
- `lambda2`: Contracción relativa entre variables cruzadas (por defecto `0.5`).
- `lambda3`: Decaimiento en los retardos (por defecto `1.0`).
- `intercept_prior_std`: Desviación típica a priori de las constantes (por defecto `1e3`), respetada en ambos modos de a priori.
- `thin`: Intervalo de adelgazamiento (*thinning*, por defecto `1`, mínimo `1`): se conserva una extracción cada `thin` barridos hasta reunir `n_draws`.
- `n_chains`: Número de cadenas MCMC ejecutadas secuencialmente para los diagnósticos de Gelman-Rubin (por defecto `2`, mínimo `1`).

Se emite un `UserWarning` cuando algún barrido agota el presupuesto de reintentos de estabilidad (véase el Paso 1).

---

## 7. Interfaz de resultados

El objeto `BVAR_SVResult` proporciona:

- **Atributos** (`D = n_chains * n_draws` extracciones agrupadas):
  - `beta_draws`: Extracciones de los coeficientes del VAR $(D, 1+np, n)$; `A_draws` $(D, p, n, n)$ e `intercept_draws` $(D, n)$ son vistas derivadas.
  - `h_draws`: Trayectorias de las log-volatilidades latentes $(D, T_{eff}, n)$.
  - `a_draws`: Extracciones de la matriz contemporánea $A$ $(D, n, n)$.
  - `mu_draws`, `phi_draws`, `sigma_h_draws`: Extracciones de los parámetros AR(1) de la volatilidad $(D, n)$.
  - `r_hat` / `rhat`: Diccionario de estadísticos de Gelman-Rubin con división de cadenas (`beta_max`, `beta_mean`, `a_max`, `h_max`, `h_mean`, `h_<nombre>_mean`, `mu_<nombre>`, `phi_<nombre>`, `sigma_h_<nombre>`, `max`).
  - `max_rhat`: Valor máximo de $\hat{R}$ registrado (`r_hat['max']`).
  - `log_scores`: Densidad predictiva puntual logarítmica dentro de muestra por observación $(T_{eff},)$ (volatilidades suavizadas).
  - `n_draws`, `n_burn`, `n_chains`, `n_total_draws`: Configuración del muestreador y número total de extracciones agrupadas.
  - `n_unstable_rejections`, `n_stuck_iterations`: Contabilidad de los rechazos por estabilidad (Paso 1).
- **Métodos**:
  - `.irf(horizon=20, t_idx=-1, ci=0.9)`: Devuelve un `BVAR_SV_IRF` con `.median`, `.lower`, `.upper`, `.draws` y `.to_frame(target_idx=None, shock_idx=None, names=None)`.
  - `.forecast(horizon=8, ci=0.9, seed=None)`: Devuelve un `BVAR_SVForecast` con trayectorias predictivas `paths`, bandas, `to_frame()` y gráficos de abanico con `.plot()`.
  - `.log_score(holdout=None, point_by_point=False, seed=None)`: Puntuación predictiva logarítmica fuera de muestra sobre una muestra reservada; sin `holdout`, la lppd dentro de muestra.
  - `.predictive_log_score(point_by_point=False)`: lppd dentro de muestra (total o puntual).
  - `.gelman_rubin()`: El diccionario `r_hat`.
  - `.summary(ci=0.9)`: Informe de texto con la configuración del muestreador, el estado de convergencia, la lppd dentro de muestra, los rechazos por estabilidad (si los hay) y, para cada variable, la mediana a posteriori, el intervalo de credibilidad central y el $\hat{R}$ de $\mu_i$, $\phi_i$, $\sigma_{h,i}$.
  - `.plot(t_idx=-1, horizon=20, ci=0.9, shock_idx=0, target_idx=None, ax=None)`: Tres paneles — sendas de log-volatilidad y desviaciones típicas condicionales con bandas de credibilidad para todas las variables (o solo `target_idx`), y la IRF condicionada a la volatilidad de `target_idx` (la primera variable por defecto) ante `shock_idx`.
  - `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tabla con media / desviación típica a posteriori de los parámetros de volatilidad por variable y el $\hat{R}$ de su log-volatilidad media.

### Limitaciones

- Los coeficientes del VAR se extraen conjuntamente con una factorización de Cholesky $(nk \times nk)$ por barrido; los sistemas muy grandes (digamos $nk > 300$) serán lentos. No está implementado el muestreador triangular ecuación por ecuación de CCM (2019).
- El `log_score` fuera de muestra proyecta la volatilidad desde el final de la muestra de estimación y no reestima ni refiltra con los datos reservados; para una evaluación totalmente recursiva, reestime el modelo en cada origen.
- Las hiper-a priori de los procesos de volatilidad son fijas (Sección 1.2) y no configurables por el usuario.
