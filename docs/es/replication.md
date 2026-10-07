> 🇪🇸 Español · 🇬🇧 [English](../replication.md)

# Galería de Replicación

> A diferencia de `puremacro.validation` (que previene desviaciones algorítmicas frente a paquetes de software de referencia), `puremacro.replication` está diseñado específicamente para reproducir de forma científica los principales resultados empíricos y estructurales publicados en la literatura académica revisada por pares. Cuando un caso fija una cifra calculada por puremacro en lugar de una publicada, su `citation` lo indica ("puremacro regression value, not published").

Cada caso de replicación se declara como una instancia autocontenida e inmutable de `ReplicationCase` que se ejecuta de manera determinista en el navegador bajo el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`), sin requerir acceso a la red y ejecutándose en fracciones de segundo.

## Ejecución de la Suite de Replicación

```python
from puremacro.replication import run_all, scorecard

# Ejecuta todos los casos de replicación fuera de línea y muestra el cuadro de mando
df = scorecard()
print(df[["id", "family", "paper", "target_kind", "tol", "passed", "margin"]])

# Todos los casos fuera de línea pasan (objetivos publicados o valores de regresión etiquetados)
assert df["passed"].all()
```

---

## Familias de Replicación

### 1. Estimación Bayesiana de DSGE (`dsge_estimation`)

Smets & Wouters (2007), *"Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach"*, AER 97(3):586–606. Las tablas y páginas citadas corresponden a la versión ECB Working Paper 722 (febrero de 2007, <https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf>).

- **Modelo**: `puremacro.dsge.smets_wouters`. Sus respuestas al impulso ante los siete choques coinciden con las del `.mod` de Pfeifer incluido (`puremacro/dsge/_references/sw07_pfeifer.mod`) resuelto por puremacro con los mismos parámetros (diferencia máxima inferior a 1e-12). Las perturbaciones de margen de precios y salarios son ARMA(1,1) con innovación contemporánea, $\varepsilon^p_t = \rho_p \varepsilon^p_{t-1} + \eta^p_t - \mu_p \eta^p_{t-1}$, y análogamente para salarios (WP 722, pp. 14–15 impresas).
- **Datos**: `puremacro/dsge/_sw07_data.csv`, 1966T1–2004T4 (156 trimestres), reconstruido desde FRED por `tools/build_sw07_data.py` con las definiciones del apéndice de datos de SW07 (p. 47 impresa). PIB, consumo e inversión van en términos per cápita; el consumo nominal y la inversión fija nominal se deflactan con el deflactor del PIB. El salario real es la remuneración por hora del sector NFB dividida por el deflactor del PIB. Las horas son la media de horas NFB × empleo civil ÷ población de 16 años o más. Todas estas series van en 100 × log. La inflación y el tipo de los fondos federales van en porcentaje trimestral. Son las versiones actuales de FRED, no los ficheros de SW de 2006.
- **Fixture**: `puremacro/replication/data/sw07_parity_seed0_200draws.npz`, distribuido como datos del paquete para que los casos funcionen también desde un wheel instalado, y regenerable con `python tools/build_sw07_data.py fixture`. Contiene la moda posterior optimizada, la inversa del hessiano en ella y 200 extracciones submuestreadas de Metropolis de paseo aleatorio. Registra además el SHA-256 del CSV con el que se construyó, y si no coincide los casos fallan.

- **`dsge_estimation.sw07_structural_parameters_mode`** (objetivo publicado)
  - **Objetivo**: la columna **Mode** (moda posterior) de la Tabla 1a (p. 35 del PDF) y de la Tabla 1b (p. 36 del PDF). La columna Mean (media) no es el objetivo: por ejemplo da $\varphi = 5.74$ y $\sigma_c = 1.38$, frente a las modas 5.48 y 1.39.
  - **Qué se compara**: la moda posterior del modelo de puremacro sobre los datos incluidos. Se obtuvo con L-BFGS-B desde dos puntos de partida, refinada con pasos de Newton, y es un punto estacionario: ningún paso por coordenadas aumenta el log-posteriori (comprobado en los tests).

| Parámetro | Símbolo SW07 | Tabla | Moda SW07 | Moda puremacro | Diferencia |
|---|---|---|---:|---:|---:|
| `csadjcost` | $\varphi$ | 1a | 5.48 | 5.694 | +3.9 % |
| `csigma` | $\sigma_c$ | 1a | 1.39 | 1.405 | +1.1 % |
| `chabb` | $h$ (hábitos) | 1a | 0.71 | 0.700 | -1.4 % |
| `csigl` | $\sigma_L$ | 1a | 1.92 | 2.035 | +6.0 % |
| `cprobp` | $\xi_p$ | 1a | 0.65 | 0.690 | +6.2 % |
| `cfc` | $\Phi$ | 1a | 1.61 | 1.617 | +0.4 % |
| `crr` | $\rho$ | 1a | 0.81 | 0.809 | -0.1 % |
| `crdy` | $r_{\Delta y}$ | 1a | 0.22 | 0.226 | +2.6 % |
| `ctrend` | $\bar\gamma$ | 1a | 0.43 | 0.416 | -3.2 % |
| `crhopinf` | $\rho_p$ | 1b | 0.90 | 0.879 | -2.3 % |
| `cmap` | $\mu_p$ | 1b | 0.74 | 0.703 | -5.0 % |
| `crhow` | $\rho_w$ | 1b | 0.97 | 0.972 | +0.2 % |
| `cmaw` | $\mu_w$ | 1b | 0.88 | 0.886 | +0.6 % |

  - **Tolerancia**: `Tol.COARSE` ($\text{rtol} \le 0.25$), porque la versión de los datos difiere de la de SW. Entre las demás modas publicadas, 21 de las 22 restantes (incluidas las siete desviaciones típicas de los choques) también quedan dentro del 25 %; la excepción es la indexación de precios $\iota_p$ (`cindp`), 0.32 frente a 0.22. La constante de horas $\bar l$ (`constelab`) no se compara, porque a la serie de horas incluida se le ha restado su media.

- **`dsge_estimation.sw07_log_posterior_at_mode`**, **`dsge_estimation.sw07_laplace_marginal_data_density`**, **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`** (valores de regresión de puremacro, **no son cifras publicadas**)
  - **Objetivos**:
    - log-posteriori (log-verosimilitud + log-priori) en la moda: `-822.04`, recalculado en vivo;
    - log densidad marginal de Laplace: `-902.83`, a partir del log-posteriori en vivo y de la inversa del hessiano almacenada;
    - media armónica modificada de Geweke (1999) sobre las 200 extracciones almacenadas: `-908.03`, con dispersión `2.66` entre niveles de truncamiento 0.1–0.9.

    Los tres usan `Tol.TIGHT`.
  - **Por qué no son objetivos publicados**: SW07 no publican el log-posteriori en la moda, y su verosimilitud marginal (Tabla 2, p. 37 del PDF: −905.8) se calcula sobre 1966:1–2004:4 "usando el periodo 1956:1–1965:4 como muestra de entrenamiento" con la aproximación de Laplace. Estos casos protegen el modelo, los datos, las distribuciones a priori y los estimadores de verosimilitud marginal de puremacro frente a cambios no intencionados.
  - **El cálculo de la Tabla 2 en sí (4.6.0)**: `sw07_laplace_mdd` (`puremacro.dsge.sw07_marginal`) implementa la muestra de entrenamiento como el `presample` de Dynare con inicialización difusa, sobre las series de los propios autores (`load_sw07_data("authors")`, los datos de replicación de la AER). El 3 de octubre de 2026 dio una log-densidad de Laplace de **−932.3** (log-posteriori en la moda −849.6) para la configuración del artículo (inicio 1956Q1, presample de 40 trimestres), **−921.4** (−840.0) para la configuración del `.mod` de replicación de los autores (inicio 1965Q1, presample de 4 trimestres, `lik_init=2`), **−926.2** (−845.1) para 1966–2004 con inicialización incondicional, y **−902.8** (−822.0) sobre la reconstrucción FRED incluida, que reproduce el caso de regresión anterior. En la moda a posteriori de los propios autores (`usmodel_mode.mat`) el log-posteriori de puremacro es −840.81 frente a **−841.46 impreso por Dynare 8** sobre los mismos archivos y opciones (Laplace −922.4 frente a −923.1 con el hessiano de Dynare; con inicio en 1956Q1 y presample de 40, −888.27 frente a −887.13). El caso de replicación `dsge_estimation.sw07_log_posterior_at_authors_mode_vs_dynare` fija esa coincidencia. El −905.8 impreso no lo reproduce, por tanto, ni puremacro ni Dynare 8 sobre los archivos públicos de replicación con ninguna de las elecciones de muestra e inicialización anteriores; la brecha está en los datos o en el cálculo detrás de la tabla publicada, no en el modelo de puremacro.
  - Hasta la versión 4.3.0 esta página presentaba −1673.72, −1686.09 y −2524.36 como resultados de SW07. Eran salidas de puremacro con un modelo cuyos choques de margen carecían de sus términos MA y entraban con un trimestre de retraso, y con datos en los que a las horas les faltaba el factor 100.

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
  - **Alcance**: Proyección local modificada a horizonte ocho, con HAC de Newey-West, datos revisados del producto y muestra 1950–2006. Su respuesta aproximada de `-2.9004` se compara de forma gruesa con la magnitud central del artículo cercana a `-3`; no es el estimador original de retardos distribuidos ni un coeficiente LP publicado a horizonte ocho.
  - **Tolerancia**: `Tol.MEDIUM` ($\text{rtol} \le 0.10$), solo para esa comparación gruesa. La [replicación con datos originales](empirical_research.md) reproduce por separado la especificación de los autores y la compara con resultados de software independiente.

---

### 3. Modelos Estructurales y Agentes Heterogéneos (`ha_dsge_causal`)

- **`ha_dsge_causal.aiyagari_precautionary_wedge`**: Replica la brecha de ahorro precautorio de Aiyagari (1994) que sitúa la tasa de interés de equilibrio por debajo de la tasa de descuento subjetiva ($r^* < \rho$).
- **`ha_dsge_causal.aiyagari_r_decreasing_in_sigma`**: Valida la estática comparativa: $r^*$ decrece a medida que aumenta el riesgo de ingresos idiosincrásico $\sigma$.
- **`ha_dsge_causal.aiyagari_r_decreasing_in_rho`**: Valida la estática comparativa: $r^*$ decrece a medida que aumenta la persistencia de ingresos $\rho$.
- **`ha_dsge_causal.huggett_precautionary_riskfree_rate`**: Replica la depresión de la tasa libre de riesgo en una economía de crédito puro de Huggett (1993) debido a restricciones de endeudamiento.
- **`ha_dsge_causal.sw07_seven_shocks_seven_observables`**: Comprobación de la especificación de Smets-Wouters (2007): el modelo tiene los siete choques estructurales publicados asociados a siete observables (un recuento, no una comparación de momentos).

---

### 4. Hechos Estilizados Macroeconómicos (`stylized_facts`)

- **`stylized_facts.okun_law_fred`**: Replica la comovilidad cíclica negativa entre el crecimiento del producto y la variación en la tasa de desempleo (Ley de Okun, correlación $\le -0.60$).
