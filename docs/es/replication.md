> 🇪🇸 Español · 🇬🇧 [English](../replication.md)

# Galería de Replicación

> A diferencia de `puremacro.validation` (que previene desviaciones algorítmicas frente a paquetes de software de referencia), `puremacro.replication` está diseñado específicamente para reproducir de forma científica los principales resultados empíricos y estructurales publicados en la literatura académica revisada por pares.

Cada caso de replicación se declara como una instancia autocontenida e inmutable de `ReplicationCase` que se ejecuta de manera determinista en el navegador bajo el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`), sin requerir acceso a la red y ejecutándose en fracciones de segundo.

## Ejecución de la Suite de Replicación

```python
from puremacro.replication import run_all, scorecard

# Ejecuta todos los casos de replicación fuera de línea y muestra el cuadro de mando
df = scorecard()
print(df[["id", "family", "paper", "target_kind", "tol", "passed", "margin"]])

# Comprueba el 100 % de éxito en la reproducción de hallazgos publicados
assert df["passed"].all()
```

---

## Familias de Replicación

### 1. Estimación Bayesiana de DSGE (`dsge_estimation`)

Replica los resultados empíricos fundamentales del modelo DSGE de escala media para la economía de EE.UU. de Smets & Wouters (2007, AER 97(3):586–606):

- **`dsge_estimation.sw07_log_posterior_at_mode`**
  - **Objetivo**: Valor exacto del log-posteriori negativo de `-1673.72` en la moda posterior sobre los datos de EE.UU. 1966Q1–2004Q4 (155 trimestres, 7 variables observables).
  - **Metodología**: Evalúa la verosimilitud completa de Kalman inicializada a partir de la covarianza estacionaria incondicional de Lyapunov ($P_0 = T P_0 T' + R Q R'$) con 36 distribuciones a priori informativas.
  - **Tolerancia**: `Tol.TIGHT` ($\text{rtol} \le 0.02$).

- **`dsge_estimation.sw07_laplace_marginal_data_density`**
  - **Objetivo**: Densidad marginal de los datos asintótica de Laplace de `-1686.09`.
  - **Metodología**: Evalúa $\log p(Y) \approx \log p(Y, \theta^*) + \frac{d}{2}\log(2\pi) + \frac{1}{2}\log|\Sigma^*|$ utilizando el hessiano inverso en la moda.
  - **Tolerancia**: `Tol.TIGHT`.

- **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`**
  - **Objetivo**: Promedio de densidad marginal de los datos con media armónica modificada de Geweke (1999) de `-2524.36` con una dispersión entre niveles de truncamiento $\le 2.20$ puntos logarítmicos.
  - **Metodología**: Evalúa las extracciones de parámetros reponderadas por núcleos gaussianos truncados en niveles de truncamiento $p \in [0.1, 0.3, 0.5, 0.7, 0.9]$, comprobando la invariancia ante el truncamiento.
  - **Tolerancia**: `Tol.TIGHT`.

- **`dsge_estimation.sw07_structural_parameters_mode`**
  - **Objetivo**: Parámetros clave de la moda publicados en la Tabla 1: persistencia de hábitos $\lambda = 0.71$, elasticidad intertemporal $\sigma_c = 1.38$, coste de ajuste de la inversión $S'' = 5.74$, rigidez de precios de Calvo $\xi_p = 0.66$, rigidez salarial $\xi_w = 0.70$, costes fijos $\Phi = 1.60$, regla de Taylor sobre inflación $\phi_\pi = 2.04$, suavizado de tasas de interés $\rho = 0.81$ y crecimiento tendencial $\bar{\gamma} = 0.43$.
  - **Tolerancia**: `Tol.COARSE` ($\text{rtol} \le 0.25$, reflejando la estabilidad numérica del entorno de la moda).

---

### 2. Regresiones Econométricas en NumPy Puro (`regression`)

Replica resultados fundamentales de micro y macroeconometría utilizando el subsistema de regresión de puremacro sin dependencias externas (`puremacro.regress`):

- **`regression.card1995_iv_vs_ols`**
  - **Artículo**: Card, D. (1995), *"Using Geographic Variation in College Proximity to Estimate the Return to Schooling"*.
  - **Resultados**: Replica el retorno clásico a la educación: coeficiente de MCO $\beta_{\text{educ}} = 0.0740$ frente a la estimación por Mínimos Cuadrados en Dos Etapas (MC2E) con variables instrumentales $\beta_{\text{educ}} = 0.1323$ instrumentado por la proximidad a una universidad de 4 años en el condado (`nearc4`).
  - **Tolerancia**: `Tol.TIGHT`.

- **`regression.long_ervin2000_hc_hierarchy`**
  - **Artículo**: Long, J. S. y Ervin, L. H. (2000), *"Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model"*, The American Statistician 54(3):217–224.
  - **Resultados**: Comprueba la jerarquía monótona estricta de errores estándar bajo heterocedasticidad y puntos con alto apalancamiento:
    $$\mathrm{SE}_{\text{OLS}} < \mathrm{SE}_{\text{HC0}} < \mathrm{SE}_{\text{HC1}} < \mathrm{SE}_{\text{HC2}} < \mathrm{SE}_{\text{HC3}}$$
  - **Tipo de Objetivo**: `TargetKind.SIGN` (diferencias positivas en cada escalón).

- **`regression.mroz1987_logit_participation`**
  - **Artículo**: Mroz, T. A. (1987), *"The Sensitivity of an Empirical Model of Married Women's Hours of Work"*, Econometrica 55(4):765–799.
  - **Resultados**: Modelo Logit binario de participación laboral femenina (`inlf`): log-verosimilitud de `-401.77`, constante `0.4255`, efecto de hijos pequeños (`kidslt6`) `-1.4434` e ingresos no laborales del cónyuge (`nwifeinc`) `-0.0213`.
  - **Tolerancia**: `Tol.TIGHT`.

- **`regression.romer_romer_tax_multiplier_ols`**
  - **Artículo**: Romer, C. D. y Romer, D. H. (2010), *"The Macroeconomic Effects of Tax Changes"*, AER 100(3):763–801.
  - **Resultados**: Respuesta acumulada del PIB a 8 trimestres ante choques tributarios narrativos mediante MCO con retardos distribuidos y covarianza HAC de Newey-West: multiplicador de `-2.9004` (referencia publicada $\approx -3.0$).
  - **Tolerancia**: `Tol.MEDIUM` ($\text{rtol} \le 0.10$).

---

### 3. Modelos Estructurales y Agentes Heterogéneos (`ha_dsge_causal`)

- **`ha_dsge_causal.aiyagari_precautionary_wedge`**: Replica la brecha de ahorro precautorio de Aiyagari (1994) que sitúa la tasa de interés de equilibrio por debajo de la tasa de descuento subjetiva ($r^* < \rho$).
- **`ha_dsge_causal.aiyagari_r_decreasing_in_sigma`**: Valida la estática comparativa: $r^*$ decrece a medida que aumenta el riesgo de ingresos idiosincrásico $\sigma$.
- **`ha_dsge_causal.aiyagari_r_decreasing_in_rho`**: Valida la estática comparativa: $r^*$ decrece a medida que aumenta la persistencia de ingresos $\rho$.
- **`ha_dsge_causal.huggett_precautionary_riskfree_rate`**: Replica la depresión de la tasa libre de riesgo en una economía de crédito puro de Huggett (1993) debido a restricciones de endeudamiento.
- **`ha_dsge_causal.sw07_seven_shocks_seven_observables`**: Solución estacionaria de covarianza de Smets-Wouters (2007) que reproduce las varianzas observadas del ciclo económico.

---

### 4. Hechos Estilizados Macroeconómicos (`stylized_facts`)

- **`stylized_facts.okun_law_fred`**: Replica la comovilidad cíclica negativa entre el crecimiento del producto y la variación en la tasa de desempleo (Ley de Okun, correlación $\le -0.60$).
