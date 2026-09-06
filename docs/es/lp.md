> 🇬🇧 [English](../lp.md) · 🇪🇸 Español

# Proyecciones Locales (LP)

Las Proyecciones Locales (Jordà 2005) estiman funciones de respuesta al impulso mediante una secuencia de regresiones directas e independientes para cada horizonte $h = 0, 1, \dots, H$:

$$y_{t+h} - y_{t-1} = \alpha_h + \beta_h x_t + \gamma_h' w_t + \sum_{l=1}^p \Gamma_{h,l} w_{t-l} + \varepsilon_{t+h}$$

donde $x_t$ representa el choque estructural o la variable de política, $w_t$ denota el vector de controles (que entran contemporáneamente y con $p$ retardos, junto con $p$ retardos de $y_t$ y $x_t$) y $\beta_h$ traza directamente el valor de la función de respuesta al impulso en el horizonte $h$.

A diferencia de los modelos VAR estructurales, que iteran hacia adelante un modelo lineal a un paso:
- **No acumulan errores de especificación** a través de los horizontes sucesivos.
- **Incorporan de forma natural no linealidades**, asimetrías y efectos dependientes del estado económico.
- **Permiten inferencia robusta y flexible** mediante errores estándar HAC de Newey-West o de Driscoll-Kraay sin requerir estacionariedad estricta de matrices compañeras.

En `puremacro 2.0`, todos los estimadores de proyecciones locales devuelven un objeto unificado **`LPResult`**, que opera como un DataFrame de Pandas a la vez que proporciona métodos integrados de graficación (`.plot()`) y exportación para publicaciones (`.to_latex()`, `.to_typst()`, `.to_markdown()`).

Los bloques de código de esta página se construyen unos sobre otros: ejecutados en orden, la página completa corre tal cual.

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

# Resumen formateado (con indicadores de significancia)
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
| `res.se` | Errores estándar HAC $\hat{\text{se}}(\hat{\beta}_h)$ (todo NaN en estimadores que solo reportan bandas, como `lp_quantile`) |
| `res.ci_lower` | Cota inferior del intervalo de confianza al nivel especificado |
| `res.ci_upper` | Cota superior del intervalo de confianza |
| `res.t_stat` | $\hat{\beta}_h / \hat{\text{se}}$ (NaN donde no hay error estándar) |
| `res.horizons` | Arreglo de horizontes evaluados `[0, 1, ..., H]` |
| `res.labels` | `[]` para resultados de un solo coeficiente; `['H', 'L']` / `['pos', 'neg']` en resultados por régimen o signo |
| `res.plot()` | Genera el gráfico de la FRI con bandas de confianza sombreadas (una línea y banda por régimen en resultados por régimen; una línea por cuantil en `lp_quantile`) |
| `res.summary()` | Resumen tabular en texto plano con indicadores de significancia: `***` p<0,01, `**` p<0,05, `*` p<0,10 (contraste z normal bilateral); en estimadores que solo reportan bandas, un `*` marca las bandas que excluyen el cero |
| `res.to_markdown()` | Renderiza la tabla formateada para Quarto, GitHub o notas de investigación |
| `res.to_latex()` | Genera un entorno `\begin{tabular}` limpio para LaTeX |
| `res.to_typst()` | Genera un bloque `#table(...)` nativo para Typst |

En los resultados por régimen o signo (`lp_state_dep`, `lp_state_dep_iv`, `lp_asymmetric`, `lp_garch_state`) la tabla lleva un coeficiente por etiqueta (`beta_H`/`beta_L`, `beta_pos`/`beta_neg`) y `res.point`, `res.se`, `res.ci_lower`, `res.ci_upper` y `res.t_stat` devuelven un DataFrame indexado por `h` con una columna por etiqueta.

---

## 2. Proyecciones locales con variables instrumentales (`lp_iv`)

Cuando la variable de intervención $x_t$ es endógena (por ejemplo, la tasa de interés de política monetaria reaccionando contemporáneamente al estado de la economía), el estimador de MCO sufre sesgo por variables omitidas.

`lp_iv` implementa mínimos cuadrados en dos etapas (MC2E / 2SLS) utilizando un instrumento externo $z_t$ (como sorpresas de alta frecuencia o registros narrativos):

$$\text{Etapa 1: } x_t = \pi_{0,h} + \pi_{1,h} z_t + \text{controles} + v_{t,h}$$
$$\text{Etapa 2: } y_{t+h} - y_{t-1} = \alpha_h + \beta_h \hat{x}_t + \text{controles} + \varepsilon_{t+h}$$

El bloque siguiente amplía primero el conjunto sintético con las columnas ilustrativas que usa el resto de la página (una tasa de política endógena movida por una sorpresa de alta frecuencia, dos controles, gasto público movido por un instrumento de noticias militares y una tasa de desempleo en porcentaje):

```python
from puremacro.lp import lp_iv

# Columnas ilustrativas para los ejemplos restantes (sintéticas, con la semilla anterior)
df["hf_monetary_surprise"] = rng.standard_normal(T)
df["fedfunds"] = 0.7 * df["hf_monetary_surprise"] + 0.5 * rng.standard_normal(T)
df["inflation"] = rng.standard_normal(T)
df["commodity_prices"] = rng.standard_normal(T)
df["military_news"] = rng.standard_normal(T)
df["gov_spending"] = 0.6 * df["military_news"] + 0.5 * rng.standard_normal(T)
df["unemployment_rate"] = 6.0 + 1.5 * rng.standard_normal(T)   # porcentaje

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

donde $F(s_t) \in [0, 1]$ es el peso del régimen alto:

- **Umbral discreto (`transition="threshold"`)**: $F(s_t) = \mathbb{I}\{s_t > c\}$.
- **Transición suave (`transition="logistic"`, por defecto)**: $F(s_t) = \frac{1}{1 + \exp(-\gamma (s_t - c) / \sigma_s)}$, con $\sigma_s$ la desviación típica muestral del estado, de modo que $\gamma$ es la velocidad de transición en desviaciones típicas de $s_t$.

El corte $c$ es el argumento `threshold` **en la escala original de la variable de estado**: `threshold=6.5` sobre una tasa de desempleo en porcentaje significa 6,5 %. Por defecto (`threshold=None`) la partición se hace en la media muestral del estado. Un corte que deja todas las observaciones en un solo régimen lanza un `ValueError` que indica el rango del estado, en lugar de una regresión singular.

```python
from puremacro.lp import lp_state_dep

res_regime = lp_state_dep(
    df,
    y="gdp",
    x="gov_spending",
    state="unemployment_rate",   # state_var= es un alias
    threshold=6.5,               # 6,5 % de desempleo, escala original
    horizon=12,
    lags=4,
)
print(res_regime.summary())          # una fila por horizonte y régimen, con indicadores
print(res_regime.point.head())       # DataFrame con columnas H y L
res_regime.plot(title="Multiplicadores Fiscales según el Nivel de Desempleo")
```

---

## 4. Proyecciones locales para paneles (`panel_lp` y `panel_lp_dk`)

Estime proyecciones locales sobre paneles de datos con efectos fijos bidireccionales y errores estándar agrupados por unidad (`panel_lp`) o robustos a correlación espacial y temporal mediante Driscoll y Kraay (1998) (`panel_lp_dk`, o `panel_lp(..., cov_type="driscoll-kraay")`).

El panel puede ser un DataFrame **largo** cuyas columnas identifican la unidad y el periodo (`unit_col=` / `time_col=`), o un DataFrame ya indexado por un `MultiIndex` `(unidad, tiempo)` cuyos niveles se nombran con `entity_level=` / `time_level=`:

```python
from puremacro.lp import panel_lp, panel_lp_dk

# Panel largo: una fila por (country, quarter)
countries = [f"C{i}" for i in range(8)]
quarters = list(pd.period_range("2000Q1", periods=60, freq="Q"))
panel_df = pd.DataFrame({
    "country": np.repeat(countries, len(quarters)),
    "quarter": quarters * len(countries),
})
panel_df["monetary_shock"] = rng.standard_normal(len(panel_df))
panel_df["investment"] = -0.4 * panel_df["monetary_shock"] + rng.standard_normal(len(panel_df))

res_panel = panel_lp(
    panel_df,
    y="investment",
    x="monetary_shock",
    unit_col="country",
    time_col="quarter",
    horizon=12,
    cov_type="driscoll-kraay",
)
print(res_panel.summary())

# Mismas estimaciones con panel_lp_dk sobre un panel ya indexado por (country, quarter)
res_dk = panel_lp_dk(
    panel_df.set_index(["country", "quarter"]),
    y="investment", x="monetary_shock",
    entity_level="country", time_level="quarter",
    horizon=12,
)
assert np.allclose(res_panel.point, res_dk.point)
assert np.allclose(res_panel.se, res_dk.se)
```

---

## 5. Firmas estandarizadas y alias de parámetros

Todos los estimadores exportados por `puremacro.lp` (incluidos `lp_state_dep` y `lp_did`) aceptan los argumentos modernos siguientes, solo por palabra clave, además de los nombres heredados:

| Parámetro moderno | Alias heredado | Valor por defecto | Descripción |
|---|---|---|---|
| `lags` | `n_lags` | `2` — salvo `la_lp` y `smooth_lp` (`4`) y `lp_did` (`0`) | Número de retardos de control incluidos en la proyección (en `lp_did`: variaciones retardadas del resultado $\Delta y_{i,t-k}$ añadidas como controles) |
| `horizon` | `horizons` | `20`, es decir $h = 0 \dots 20$ — salvo `lp_iv_lewbel` y `lp_did` (`12`) | Horizonte máximo de proyección (calcula $h = 0 \dots H$) |
| `ci` | `alpha` | `0.90` — salvo `smooth_lp` (`0.95`) | Cobertura del intervalo de confianza (p. ej. `ci=0.95` $\leftrightarrow$ `alpha=0.05`) |

`ci` y `alpha` deben ser probabilidades en $(0, 1)$: pasar un porcentaje por error (`ci=90`) o `alpha=1.5` lanza un `ValueError` en lugar de producir bandas NaN en silencio. `lp_state_dep` acepta además `state_var=` como alias de `state=`.
