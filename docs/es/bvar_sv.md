> 🇬🇧 [English](../bvar_sv.md) · 🇪🇸 Español

# VAR bayesiano con volatilidad estocástica (BVAR-SV)

El módulo `puremacro.var.bvar_sv` implementa modelos de vectores autorregresivos bayesianos con volatilidad residual variable en el tiempo (BVAR-SV), siguiendo las metodologías de espacio de estados y muestreo MCMC desarrolladas por **Carriero, Clark y Marcellino (2016, 2019)**, **Kim, Shephard y Chib (1998)**, y **Carter y Kohn (1994)**.

Las series macroeconómicas exhiben marcadas alternancias de régimen en su volatilidad —como la Gran Moderación (1984–2007), la conmoción de la Crisis Financiera Global de 2007–2008 y las disrupciones asociadas a la pandemia de 2020—. Los modelos VAR homocedásticos convencionales distorsionan las estimaciones de los parámetros dinámicos y generan densidades predictivas descalibradas al confundir variaciones en la magnitud de los choques con modificaciones en los mecanismos de transmisión estructural.

El estimador BVAR-SV descompone la dinámica macroeconómica en coeficientes autorregresivos estables y **log-volatilidades estocásticas latentes**, permitiendo la estimación precisa de densidades de pronóstico y de funciones de respuesta al impulso condicionadas al estado de volatilidad.

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
   con distribución inicial incondicional dada por $h_{i,0} \sim \mathcal{N}\left(\mu_i, \, \frac{\sigma_{h,i}^2}{1 - \phi_i^2}\right)$.

---

## 2. El algoritmo de muestreo de Gibbs MCMC

El modelo se estima en puro NumPy/SciPy mediante un muestreador de Gibbs estructurado en bloques condicionales:

1. **Paso 1: Coeficientes dinámicos del VAR ($\beta$) con a priori de Minnesota**  
   Condicionado a la matriz contemporánea $A$ y a la senda de volatilidades $D_{1:T}$, el sistema se transforma mediante Mínimos Cuadrados Generalizados (GLS) ponderados por la precisión instantánea. La distribución a priori de Minnesota contrae los coeficientes de retardos hacia paseos aleatorios o ruido blanco:
   $$\beta \mid A, h, Y \sim \mathcal{N}(\bar{\beta}, \bar{V}_\beta)$$
2. **Paso 2: Coeficientes estructurales contemporáneos ($A$)**  
   Aplicando la descomposición triangular de Carriero, Clark y Marcellino (2016, 2019), cada fila $i$ de $A$ se extrae ecuación por ecuación mediante una regresión lineal del residuo $\hat{u}_{i,t}$ sobre los residuos previos $\hat{u}_{1:i-1,t}$, ponderada por $\exp(-h_{i,t})$.
3. **Paso 3: Mixtura gaussiana de 7 componentes de Kim, Shephard y Chib (1998)**  
   La transformación logarítmica de los residuos estandarizados al cuadrado produce un sistema de medida lineal no gaussiano:
   $$y_{i,t}^* \equiv \log\left( (A \hat{u}_t)_i^2 + c \right) = h_{i,t} + \log(\varepsilon_{i,t}^2)$$
   donde la distribución del error $\log(\varepsilon_{i,t}^2) \sim \log(\chi_1^2)$ se aproxima analíticamente mediante una mixtura de 7 densidades normales con ponderaciones, medias y varianzas $(q_k, m_k, v_k^2)$ tabuladas en Kim, Shephard y Chib (1998). Los indicadores de mixtura $s_{i,t} \in \{1, \dots, 7\}$ se muestrean condicionados a $h_{i,t}$.
4. **Paso 4: Filtrado hacia adelante y muestreo hacia atrás (FFBS) de Carter y Kohn (1994)**  
   Condicionado a los indicadores de la mixtura, el modelo se torna condicionalmente lineal y gaussiano. El algoritmo FFBS de Carter y Kohn extrae la trayectoria completa de estados $h_{i, 1:T}$ de forma conjunta en un único barrido estocástico retrógrado.
5. **Paso 5: Parámetros autorregresivos de la volatilidad $(\mu_i, \phi_i, \sigma_{h,i}^2)$**  
   Se muestrean de distribuciones conjugadas Normal-Inversa-Gamma condicionales a la serie latente $h_{i, 1:T}$.
6. **Paso 6: Diagnósticos de convergencia multicanal de Gelman-Rubin ($\hat{R}$)**  
   Se calcula el estadístico $\hat{R}$ con división de cadenas para todos los parámetros. Valores de $\hat{R} < 1.1$ garantizan la adecuada mezcla de las cadenas y la convergencia de la distribución a posteriori.

---

## 3. Respuestas al impulso condicionadas al régimen de volatilidad

Dado que la matriz de impacto estructural $B_t = A^{-1} D_t^{1/2}$ varía temporalmente, la magnitud de la respuesta macroeconómica difiere sustancialmente en función del contexto de incertidumbre.

El método `BVAR_SVResult.irf(horizon=20, t_idx=...)` permite condicionar las funciones de respuesta sobre fechas históricas concretas $t^*$:
- **Régimen de alta volatilidad (ej. 2008Q4 o 2020Q2)**: Cuantifica la propagación del choque bajo estrés financiero o incertidumbre aguda.
- **Régimen de calma (ej. Gran Moderación)**: Evalúa la propagación en épocas macroeconómicas de baja volatilidad.

---

## 4. Ejemplo de uso y código ejecutable

### Estimación del modelo BVAR-SV

```python
import numpy as np
import pandas as pd
from puremacro.var.bvar_sv import bvar_sv

# 1. Preparación de series macroeconómicas trimestrales (PIB, Inflación, Tipo de Interés)
rng = np.random.default_rng(42)
T = 180
fechas = pd.date_range("1975-01-01", periods=T, freq="QE")

# Generación de datos sintéticos con volatilidad variable
datos = np.zeros((T, 3))
for t in range(1, T):
    escala_vol = 2.5 if 130 <= t <= 145 else 1.0  # Incremento de volatilidad
    datos[t] = 0.6 * datos[t-1] + escala_vol * 0.5 * rng.standard_normal(3)

df = pd.DataFrame(datos, index=fechas, columns=["PIB", "Inflacion", "Tipo"])

# 2. Estimación del BVAR-SV con el muestreador de Gibbs
res_sv = bvar_sv(
    data=df,
    lags=2,
    n_draws=1500,
    n_burn=500,
    n_chains=2,
    minnesota_prior=True,
    lambda1=0.2,
    lambda2=0.5,
    seed=123,
)

# 3. Diagnósticos de convergencia e informe resumen
print(res_sv.summary())
print(f"Estadístico máximo de Gelman-Rubin R-hat: {res_sv.max_rhat:.3f}")
assert res_sv.max_rhat < 1.1, "Las cadenas deben converger formalmente (< 1.1)"

# 4. Cálculo de respuestas al impulso condicionadas a la volatilidad
# Comparación de la respuesta en crisis (t_idx=135) frente a un período de calma (t_idx=50)
irf_crisis = res_sv.irf(horizon=16, t_idx=135, ci=0.90)
irf_calma = res_sv.irf(horizon=16, t_idx=50, ci=0.90)

print(f"Respuesta máxima del PIB (Crisis): {irf_crisis.median[:, 0, 0].max():.3f}")
print(f"Respuesta máxima del PIB (Calma) : {irf_calma.median[:, 0, 0].max():.3f}")

# 5. Visualización gráfica de las trayectorias de volatilidad
fig = res_sv.plot()

# 6. Exportación académica
tabla_latex = res_sv.to_latex()
tabla_md = res_sv.to_markdown()
```

---

## 5. Especificación completa de la API

### `bvar_sv`

```python
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
- `data`: Panel temporal $(T, n)$ con las variables endógenas.
- `lags` / `p`: Número de retardos del VAR (por defecto `4`).
- `n_draws`: Extracciones MCMC conservadas tras el período de calentamiento por cada cadena (por defecto `2000`).
- `n_burn`: Iteraciones iniciales de descarte (*burn-in*, por defecto `1000`).
- `minnesota_prior`: Aplica regularización mediante la distribución a priori de Minnesota (por defecto `True`).
- `seed`: Semilla generadora para asegurar reproducibilidad exacta.
- `lambda1`: Parámetro global de contracción de Minnesota (por defecto `0.2`).
- `lambda2`: Contracción relativa entre variables cruzadas (por defecto `0.5`).
- `lambda3`: Decaimiento armónico en los retardos (por defecto `1.0`).
- `n_chains`: Número de cadenas independientes de MCMC para diagnósticos de Gelman-Rubin (por defecto `2`).
- `thin`: Intervalo de adelgazamiento (*thinning*) para reducir la autocorrelación de la muestra.

---

## 6. Interfaz de resultados

El objeto `BVAR_SVResult` almacena las extracciones a posteriori y proporciona herramientas de análisis:

- **Atributos numéricos**:
  - `beta_draws`: Muestras a posteriori de los coeficientes autorregresivos $(D, 1+np, n)$.
  - `h_draws`: Trayectorias de las log-volatilidades estocásticas latentes $(D, T_{eff}, n)$.
  - `a_draws`: Muestras de la matriz contemporánea $A$ $(D, n, n)$.
  - `phi_draws`: Persistencia de las log-volatilidades $(D, n)$.
  - `sigma_h_draws`: Volatilidad de las log-volatilidades $(D, n)$.
  - `rhat`: Vector de estadísticos de convergencia de Gelman-Rubin.
  - `max_rhat`: Valor máximo de $\hat{R}$ registrado.
  - `log_score`: Puntuaciones de densidad predictiva fuera de muestra.
- **Métodos disponibles**:
  - `.irf(horizon=20, t_idx=-1, ci=0.9)`: Devuelve un contenedor `BVAR_SV_IRF` con `.median`, `.lower` y `.upper`.
  - `.plot()`: Gráfico en Matplotlib con las sendas de volatilidad posterior y las bandas de credibilidad.
  - `.summary()`: Informe estructurado con medianas posteriores, intervalos de credibilidad y estadísticos $\hat{R}$.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas estructuradas listas para su inclusión en manuscritos.
