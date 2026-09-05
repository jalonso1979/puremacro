> 🇬🇧 [English](../lp.md) · 🇪🇸 Español

# Proyecciones Locales (LP)

Las Proyecciones Locales (Jordà 2005) estiman funciones de respuesta al impulso mediante una secuencia de regresiones directas e independientes para cada horizonte $h = 0, 1, \dots, H$:

$$y_{t+h} - y_{t-1} = \alpha_h + \beta_h x_t + \sum_{l=1}^p \Gamma_{h,l} w_{t-l} + \varepsilon_{t+h}$$

donde $x_t$ representa el choque estructural o la variable de política, $w_t$ denota el vector de controles y $\beta_h$ traza directamente el valor de la función de respuesta al impulso en el horizonte $h$.

A diferencia de los modelos VAR estructurales, que iteran hacia adelante un modelo lineal a un paso:
- **No acumulan errores de especificación** a través de los horizontes sucesivos.
- **Incorporan de forma natural no linealidades**, asimetrías y efectos dependientes del estado económico.
- **Permiten inferencia robusta y flexible** mediante errores estándar HAC de Newey-West o de Driscoll-Kraay sin requerir estacionariedad estricta de matrices compañeras.

En `puremacro 2.0`, todos los estimadores de proyecciones locales devuelven un objeto unificado **`LPResult`**, que opera como un DataFrame de Pandas a la vez que proporciona métodos integrados de graficación (`.plot()`) y exportación para publicaciones (`.to_latex()`, `.to_typst()`, `.to_markdown()`).

---

## 1. Inicio rápido: LP-HAC para series individuales

Para estimar respuestas al impulso con errores estándar de Newey-West consistentes ante heterocedasticidad y autocorrelación:

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# Conjunto de datos sintético: respuesta del producto ante un choque
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
output = np.cumsum(0.6 * shock + 0.4 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": output, "shock": shock})

# Estimar proyección local hasta el horizonte 12 con 4 retardos de control
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# Resumen formateado
print(res.summary())

# Inspeccionar estimaciones y errores estándar
print(res[["h", "beta", "se", "lo", "hi"]].head())

# Graficar la FRI con bandas de confianza en una línea
res.plot(title="Respuesta del PIB ante Choque Estructural")
```

### El objeto `LPResult`

| Propiedad / Método | Descripción |
|---|---|
| `res.point` | Estimaciones puntuales $\hat{\beta}_h$ como vector NumPy de forma `(H+1,)` |
| `res.se` | Errores estándar HAC $\hat{\text{se}}(\hat{\beta}_h)$ |
| `res.ci_lower` | Cota inferior del intervalo de confianza al nivel especificado |
| `res.ci_upper` | Cota superior del intervalo de confianza |
| `res.horizons` | Arreglo de horizontes evaluados `[0, 1, ..., H]` |
| `res.plot()` | Genera el gráfico de la FRI con bandas de confianza sombreadas |
| `res.summary()` | Resumen tabular en texto plano con indicadores de significancia |
| `res.to_markdown()` | Renderiza la tabla formateada para Quarto, GitHub o notas de investigación |
| `res.to_latex()` | Genera un entorno `\begin{tabular}` limpio para LaTeX |
| `res.to_typst()` | Genera un bloque `#table(...)` nativo para Typst |

---

## 2. Proyecciones locales con variables instrumentales (`lp_iv`)

Cuando la variable de intervención $x_t$ es endógena (por ejemplo, la tasa de interés de política monetaria reaccionando contemporáneamente al estado de la economía), el estimador de MCO sufre sesgo por variables omitidas.

`lp_iv` implementa mínimos cuadrados en dos etapas (MC2E / 2SLS) utilizando un instrumento externo $z_t$ (como sorpresas de alta frecuencia o registros narrativos):

$$\text{Etapa 1: } x_t = \pi_{0,h} + \pi_{1,h} z_t + \text{controles} + v_{t,h}$$
$$\text{Etapa 2: } y_{t+h} - y_{t-1} = \alpha_h + \beta_h \hat{x}_t + \text{controles} + \varepsilon_{t+h}$$

```python
from puremacro.lp import lp_iv

res_iv = lp_iv(
    df,
    y="gdp",
    x="fedfunds",
    z="hf_monetary_surprise",
    controls=["inflation", "commodity_prices"],
    horizon=16,
    lags=4,
    ci=0.90,
)

print("Estadísticos F de primera etapa:", res_iv["first_stage_f"].values)
res_iv.plot(title="LP-IV: Transmisión Monetaria vía Instrumento Externo")
```

---

## 3. Proyecciones locales dependientes del estado (`lp_state_dep`)

La transmisión de la política económica puede variar según el régimen del ciclo (recesión frente a expansión):

$$y_{t+h} - y_{t-1} = \alpha_h + F(s_t) \beta_h^H x_t + (1 - F(s_t)) \beta_h^L x_t + \text{controles} + \varepsilon_{t+h}$$

```python
from puremacro.lp import lp_state_dep

res_regime = lp_state_dep(
    df,
    y="gdp",
    x="gov_spending",
    state="unemployment_rate",
    threshold=6.5,
    horizon=12,
    lags=4,
)
res_regime.plot(title="Multiplicadores Fiscales según el Nivel de Desempleo")
```

---

## 4. Proyecciones locales para paneles (`panel_lp`)

Estime proyecciones locales sobre paneles de datos con efectos fijos y errores estándar robustos a correlación espacial y temporal mediante Driscoll y Kraay (1998):

```python
from puremacro.lp import panel_lp

res_panel = panel_lp(
    df,
    y="investment",
    x="monetary_shock",
    unit_col="country",
    time_col="quarter",
    horizon=12,
    cov_type="driscoll-kraay",
)
print(res_panel.summary())
```
