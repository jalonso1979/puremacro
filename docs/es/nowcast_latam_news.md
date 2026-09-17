> 🇬🇧 [English](../nowcast_latam_news.md) · 🇪🇸 Español

# Nowcasting en Tiempo Real de América Latina y Descomposición de Noticias

El seguimiento macroeconómico en economías de mercados emergentes—particularmente en América Latina—enfrenta desafíos empíricos singulares: las series estadísticas se publican con rezagos heterogéneos (*ragged edge* o borde irregular), las cifras históricas experimentan frecuentes revisiones estadísticas y las publicaciones de bancos centrales e institutos de estadística operan en frecuencias mixtas (producción industrial, ventas minoristas y encuestas mensuales frente al PIB trimestral).

`puremacro.nowcast` proporciona una infraestructura institucional de nowcasting en tiempo real y evaluación probabilística de pronósticos diseñada para bancos centrales, ministerios de hacienda y departamentos de análisis macroeconómico. La arquitectura integra:

1. **Orquestador de Nowcast en Tiempo Real (`realtime_nowcast`)**: Estimación unificada que integra conectores de datos de bancos centrales e institutos estadísticos de América Latina (Banxico, INEGI, BCB, BCCh y ALFRED/St. Louis Fed) a través de paneles de ediciones históricas (*vintage panels*).
2. **Modelo de Factores Dinámicos con Filtro de Kalman (`DynamicFactorModel`)**: Algoritmo Esperanza-Maximización y suavizamiento exacto de Kalman para paneles desbalanceados con patrones arbitrarios de datos faltantes (Doz, Giannone y Reichlin 2011; Bańbura y Modugno 2014).
3. **Descomposición Analítica de Noticias frente a Ruido (`banbura_modugno_news`)**: Atribución matemáticamente exacta de las revisiones del pronóstico a sorpresas en indicadores específicos y revisiones de datos pasados, con garantía de cierre numérico ($|\Delta \hat{y} - \sum \text{impacto}| < 10^{-10}$).
4. **Evaluación de Densidades y Gráficos de Abanico (*Fan Charts*) (`pit_uniformity_test`, `fan_chart`)**: Calibración probabilística mediante la prueba de Razón de Verosimilitud de Berkowitz (2001), prueba de Kolmogorov-Smirnov y gráficos de abanico por cuantiles con paletas de bancos centrales.

Todos los algoritmos están implementados en NumPy, SciPy y pandas puros en float64, compatibles con Pyodide en el navegador y sin dependencias externas compiladas ni requerimientos de red.

---

## 1. Fundamentos Teóricos y Matemáticos

### 1.1 Modelo de Factores Dinámicos en Forma de Espacio de Estados

Sea $X_t \in \mathbb{R}^N$ un panel mensual estandarizado de $N$ indicadores macroeconómicos en el período $t = 1, \dots, T$. Se asume que la dinámica común está impulsada por un vector de baja dimensión $r \ll N$ de factores latentes $F_t \in \mathbb{R}^r$ y un vector de perturbaciones idiosincrásicas $\xi_t \in \mathbb{R}^N$:

$$X_t = \Lambda F_t + \xi_t, \qquad \xi_t \sim \text{i.i.d.} \, \mathcal{N}(0, R)$$

donde $\Lambda \in \mathbb{R}^{N \times r}$ es la matriz de cargas factoriales (*factor loadings*), y $R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_N^2)$ es diagonal. Los factores comunes siguen un proceso VAR($p$):

$$F_t = A_1 F_{t-1} + A_2 F_{t-2} + \dots + A_p F_{t-p} + u_t, \qquad u_t \sim \text{i.i.d.} \, \mathcal{N}(0, Q)$$

En la forma compañera de espacio de estados con vector de estado $\alpha_t = [F_t^\top, F_{t-1}^\top, \dots, F_{t-p+1}^\top]^\top \in \mathbb{R}^{rp}$:

$$\alpha_t = T \alpha_{t-1} + R_{\eta} \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)$$

$$X_t = Z_t \alpha_t + \xi_t$$

donde $Z_t = W_t \begin{bmatrix} \Lambda & 0_{N \times r(p-1)} \end{bmatrix}$, y $W_t$ es una matriz de selección diagonal cuyo elemento $i$-ésimo es $1$ si el indicador $i$ fue observado en el mes $t$, y $0$ si falta (gestionando de forma nativa publicaciones desfasadas y bordes rasgados).

### 1.2 Descomposición Analítica de Noticias de Bańbura y Modugno (2014)

Cuando el analista actualiza el nowcast desde la edición de datos $v-1$ (conjunto de información $\Omega_{v-1}$) hasta la edición $v$ ($\Omega_v$), la proyección de la variable objetivo $y_{t^*}$ cambia en:

$$\Delta \hat{y}_{t^*|v} = \mathbb{E}[y_{t^*} \mid \Omega_v] - \mathbb{E}[y_{t^*} \mid \Omega_{v-1}]$$

Bańbura y Modugno (2014) demuestran que, dado que el filtro y suavizador de Kalman constituyen proyecciones lineales en modelos lineales gaussianos, esta actualización se descompone exactamente en la suma de contribuciones de nuevas publicaciones (**innovaciones o noticias**) y revisiones de datos pasados:

$$\Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot \underbrace{\left( x_{j, t_j} - \mathbb{E}[x_{j, t_j} \mid \Omega_{v-1}] \right)}_{\text{innovación / noticia } I_{j, v}} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot \underbrace{\left( x_{k, t_k}^{(v)} - x_{k, t_k}^{(v-1)} \right)}_{\text{revisión histórica } R_{k, v}}$$

$$\Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \text{impacto}_j + \sum_{k \in \mathcal{I}_{\text{rev}}} \text{impacto}_k$$

El ponderador $\omega_j$ refleja tanto la correlación estructural de la variable con los factores latentes como la ganancia de Kalman asignada a la nueva observación. En `puremacro`, esta descomposición se satisface con precisión numérica:

$$|\Delta \hat{y}_{t^*|v} - \text{impacto\_total}| < 10^{-10}$$

### 1.3 Hipótesis de Noticias frente a Ruido de Mankiw-Shapiro (1986)

Las revisiones de datos $R_{t, v} = y_{t}^{(v)} - y_{t}^{(v-1)}$ se analizan econométricamente bajo dos hipótesis fundamentales:
- **Noticia (*News*)**: Las estimaciones preliminares son proyecciones eficientes basadas en la información disponible; las revisiones posteriores son ortogonales a las publicaciones tempranas ($\operatorname{Cov}(y_t^{(v-1)}, R_{t, v}) = 0$).
- **Ruido (*Noise*)**: Las cifras tempranas contienen errores de medición clásicos; las revisiones están correlacionadas con las cifras preliminares ($\operatorname{Cov}(y_t^{(v)}, R_{t, v}) = 0$).

`realtime_nowcast` ejecuta la regresión de Mankiw y Shapiro $R_{t, v} = \alpha + \beta y_t^{(v-1)} + \varepsilon_t$ para diagnosticar el comportamiento de las revisiones de las agencias estadísticas.

### 1.4 Transformación Integral de Probabilidad (PIT) y Prueba de Berkowitz

Para densidades proyectadas $\hat{f}_{t|t-h}(y_t)$, los valores empíricos PIT se definen como:

$$p_t = \int_{-\infty}^{y_t} \hat{f}_{t|t-h}(u) \, du = \Phi\left( \frac{y_t - \hat{\mu}_t}{\hat{\sigma}_t} \right)$$

Bajo calibración correcta e independencia temporal, $p_t \sim \text{i.i.d.} \, \mathcal{U}(0, 1)$. Siguiendo a Berkowitz (2001), los valores PIT se transforman a la escala normal estándar:

$$z_t = \Phi^{-1}(p_t)$$

Bajo la hipótesis nula de calibración perfecta, $z_t \sim \text{i.i.d.} \, \mathcal{N}(0, 1)$. El modelo de diagnóstico autorregresivo AR(1) es:

$$z_t - \mu = \rho (z_{t-1} - \mu) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

La hipótesis nula conjunta $H_0: \mu = 0, \sigma^2 = 1, \rho = 0$ se evalúa mediante el estadístico de Razón de Verosimilitud:

$$\text{LR} = -2 \left[ \ln L(\mu=0, \sigma^2=1, \rho=0) - \ln L(\hat{\mu}, \hat{\sigma}^2, \hat{\rho}) \right] \sim \chi^2(3)$$

Simultáneamente, la prueba de Kolmogorov-Smirnov verifica si la distribución empírica de $p_t$ difiere significativamente de una uniforme estándar $\mathcal{U}(0, 1)$.

---

## 2. Estructura de la API y Clases de Resultados

| Objeto | Tipo | Descripción |
|---|---|---|
| `DynamicFactorModel` | Clase | Estimador DFM con algoritmo EM, suavizamiento de Kalman e imputación de datos faltantes. |
| `DynamicFactorModelResult` | Dataclass congelada | Resultados (`factors`, `loadings`, `A`, `H`, `Q`, `loglik`, `X_filled`, `.plot()`). |
| `banbura_modugno_news` | Función | Calcula la descomposición analítica de noticias entre dos ediciones de datos. |
| `NewsDecompositionResult` | Dataclass congelada | Almacena `forecast_old`, `forecast_new`, `revision`, `impact_releases`, `impact_revisions` y `news_table`. |
| `realtime_nowcast` | Función | Orquestador en tiempo real para América Latina (MEX, BRA, CHL, USA). |
| `RealtimeNowcastResult` | Dataclass congelada | Nowcast puntual, desviaciones estándar, factores, descomposición de noticias y abanicos. |
| `pit_uniformity_test` | Función | Calcula prueba LR de Berkowitz (2001), KS e histograma PIT. |
| `PITUniformityResult` | Dataclass congelada | Estadísticos (`lr_stat`, `lr_pvalue`, `ks_stat`, `ks_pvalue`), `is_uniform` y diagnósticos gráficos. |
| `fan_chart` | Función | Genera abanicos de proyección por cuantiles con estilos institucionales. |
| `FanChartResult` | Dataclass congelada | Series de cuantiles, intervalos y exportaciones (`.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`). |

### Especificaciones de Países (`COUNTRY_SPECS`)

`puremacro.nowcast.realtime_nowcast.COUNTRY_SPECS` contiene las configuraciones institucionales de las principales economías:

```python
from puremacro.nowcast.realtime_nowcast import COUNTRY_SPECS

# México: Banco de México e INEGI
COUNTRY_SPECS["MEX"]
# {'name': 'Mexico', 'central_bank': 'Banco de México (Banxico) / INEGI',
#  'default_target': 'gdp', 'palette': 'banxico'}

# Brasil: Banco Central do Brasil
COUNTRY_SPECS["BRA"]

# Chile: Banco Central de Chile
COUNTRY_SPECS["CHL"]

# Estados Unidos: Reserva Federal (ALFRED)
COUNTRY_SPECS["USA"]
```

---

## 3. Ejemplo Integral de Uso

### 3.1 Estimación de Factores Dinámicos con Borde Irregular

```python
import numpy as np
import pandas as pd
from puremacro.nowcast import DynamicFactorModel

# 1. Panel mensual sintético con borde irregular (T=120, N=8)
rng = np.random.default_rng(42)
dates = pd.date_range("2014-01-01", periods=120, freq="MS")
f_latent = np.zeros(120)
for t in range(1, 120):
    f_latent[t] = 0.75 * f_latent[t - 1] + rng.normal(scale=0.5)

loadings = rng.uniform(0.6, 1.4, size=8)
X_raw = f_latent[:, None] @ loadings[None, :] + rng.normal(scale=0.3, size=(120, 8))
df_panel = pd.DataFrame(X_raw, index=dates, columns=[f"ind_{i+1}" for i in range(8)])

# Borde irregular: último mes ausente para indicadores 4 a 8
df_panel.iloc[-1, 3:] = np.nan
# Penúltimo mes ausente para indicadores 7 y 8
df_panel.iloc[-2, 6:] = np.nan

# 2. Ajuste del Modelo de Factores Dinámicos
dfm = DynamicFactorModel(n_factors=1, p=1)
dfm.fit(df_panel)
res = dfm.result_

print(f"Log-Verosimilitud : {res.loglik:.2f}")
print(f"Factores comunes   : {res.factors.shape}")
print(f"Panel imputado     : {res.X_filled.shape}")

# Inspección de cargas factoriales
print(res.loadings)
```

### 3.2 Descomposición de Noticias de Bańbura y Modugno

```python
from puremacro.nowcast import banbura_modugno_news

# Creación de edición v-1 y edición actualizada v
panel_v0 = df_panel.copy()
panel_v1 = df_panel.copy()

# Edición v publica nuevos datos para los indicadores 4 y 5
panel_v1.iloc[-1, 3] = panel_v0.iloc[-2, 3] + 0.45
panel_v1.iloc[-1, 4] = panel_v0.iloc[-2, 4] - 0.20
# Edición v revisa el indicador 1 en la fecha t-1
panel_v1.iloc[-2, 0] += 0.15

# Objetivo: indicador 1 en el período más reciente
news_res = banbura_modugno_news(
    model=dfm,
    old_vintage=panel_v0,
    new_vintage=panel_v1,
    target_series="ind_1",
    target_period=dates[-1],
)

print(news_res.summary())
print(f"Error numérico de identidad: {news_res.decomposition_error:.2e}")
assert news_res.decomposition_error < 1e-10
```

### 3.3 Nowcasting en Tiempo Real (`realtime_nowcast`)

```python
from puremacro.fetch.realtime import VintagePanel
from puremacro.nowcast import realtime_nowcast

# Integración de ediciones históricas en un panel en tiempo real (o carga vía load_realtime_cartridge)
rows = []
for d in dates:
    for col in df_panel.columns:
        val0 = panel_v0.loc[d, col]
        if not np.isnan(val0):
            rows.append({
                "country": "MEX", "variable": col, "date": d,
                "vintage": pd.Timestamp("2023-11-01"), "value": float(val0),
                "provider": "banxico", "series_id": col, "units": "rate",
            })
        val1 = panel_v1.loc[d, col]
        if not np.isnan(val1):
            rows.append({
                "country": "MEX", "variable": col, "date": d,
                "vintage": pd.Timestamp("2023-12-01"), "value": float(val1),
                "provider": "banxico", "series_id": col, "units": "rate",
            })

vp = VintagePanel(pd.DataFrame(rows))

resultado = realtime_nowcast(
    country="MEX",
    panel=vp,
    target_variable="ind_1",
    target_period=dates[-1],
    method="dfm",
    n_factors=1,
)

print(f"País     : {resultado.country}")
print(f"Nowcast  : {resultado.nowcast:.4f} (±{1.96 * resultado.forecast_sd:.4f})")
print(f"Paleta   : {resultado.palette}")

if resultado.news_decomposition:
    print(resultado.news_decomposition.news_table)
```

### 3.4 Evaluación de Calibración de Pronósticos: PIT y Prueba de Berkowitz

```python
from puremacro.nowcast import pit_uniformity_test

T_eval = 80
y_real = rng.normal(loc=1.0, scale=0.5, size=T_eval)
mu_forecast = np.full(T_eval, 1.0)
sigma_forecast = np.full(T_eval, 0.5)

pit_res = pit_uniformity_test(
    realised=y_real,
    mu=mu_forecast,
    sigma=sigma_forecast,
    n_bins=10,
)

print(pit_res.summary())
print(f"P-valor LR Berkowitz : {pit_res.lr_pvalue:.4f}")
print(f"P-valor KS           : {pit_res.ks_pvalue:.4f}")
print(f"¿Bien calibrado?     : {pit_res.is_uniform}")
```

### 3.5 Gráficos de Abanico de Bancos Centrales (*Fan Charts*)

```python
from puremacro.nowcast import fan_chart

historia = pd.Series([1.2, 1.5, 1.8, 1.6, 2.0, 2.3], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"])
media_pronostico = pd.Series([2.2, 2.1, 2.0, 1.9], index=["2025Q3", "2025Q4", "2026Q1", "2026Q2"])
sd_pronostico = [0.25, 0.35, 0.45, 0.55]

fc = fan_chart(
    history=historia,
    forecast_mean=media_pronostico,
    forecast_sd=sd_pronostico,
    levels=(0.50, 0.70, 0.90),
    palette="banxico",
)

# Exportación a tablas
print(fc.to_markdown())
```

---

## 4. Exportación para Publicaciones e Informes

Todos los resultados de nowcasting integran exportadores directos a tablas en Markdown, $\LaTeX$ (usando el paquete `booktabs`) y Typst:

```python
md_table = fc.to_markdown()
tex_table = fc.to_latex()
typst_table = fc.to_typst()
```

---

## Referencias

1. Bańbura, M. y Modugno, M. (2014). "Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data." *Journal of Applied Econometrics*, 29(1), 133–160.
2. Berkowitz, J. (2001). "Testing density forecasts, with applications to risk management." *Journal of Business & Economic Statistics*, 19(4), 465–474.
3. Doz, C., Giannone, D. y Reichlin, L. (2011). "A two-step estimator for large approximate dynamic factor models based on Kalman filtering." *Journal of Econometrics*, 164(1), 188–205.
4. Mankiw, N. G. y Shapiro, M. D. (1986). "News or noise: An analysis of GNP revisions." *Survey of Current Business*, 66(5), 20–25.
