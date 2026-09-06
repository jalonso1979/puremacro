> 🇬🇧 [English](../nowcast.md) · 🇪🇸 Español

# Nowcasting del PIB

El PIB es una magnitud trimestral que se publica con retraso considerable — entre cuatro y ocho semanas tras el cierre del trimestre de referencia, según el instituto de estadística. Hasta que dicha cifra se difunde, toda la información disponible sobre el trimestre en curso es de periodicidad mensual: producción industrial, empleo, ventas minoristas y encuestas de opinión empresarial, cada una con su propio calendario de publicación. El *nowcasting* responde a la pregunta: «¿cuál es la tasa de crecimiento del trimestre actual *a fecha de hoy*?». Los dos obstáculos principales para lograrlo son estructurales y no meramente estadísticos.

**Frecuencias mixtas.** La variable objetivo se observa cuatro veces al año; los predictores, doce. Cualquier método debe definir cómo se agrega la serie mensual en términos trimestrales — mediante promedios, sumas o una trayectoria mensual latente restringida a coincidir con el total trimestral.

**Bordes irregulares (*ragged edge*).** En cualquier día hábil, el panel de datos es un rectángulo con el extremo inferior deshilachado. Las encuestas de confianza del mes $m$ se publican a los pocos días de su finalización; los índices de producción industrial tardan semanas; otras series arrastran dos meses de desfase. Eliminar filas incompletas (*listwise deletion*) descartaría precisamente los períodos más recientes objeto del pronóstico.

`puremacro.nowcast` proporciona tres estimadores que resuelven estos desafíos:

| Función | Modelo | Entradas requeridas | Tratamiento del borde irregular |
|---|---|---|---|
| `nowcast_gdp` | Factores EM-PCA + VAR de factores + regresión puente trimestral | Panel mensual **y** serie trimestral de PIB | Imputación iterativa por PCA; los meses sin ninguna observación y el resto del trimestre objetivo, con el VAR de factores |
| `kalman_dfm` | DFM en dos etapas (Doz-Giannone-Reichlin 2011) | Un panel en una sola frecuencia | Suavizador exacto de Kalman |
| `mf_var` | VAR de frecuencias mixtas (Mariano-Murasawa 2003) | Panel mensual con el dato trimestral estampado una vez por trimestre | Suavizador exacto de Kalman |

Todo el código se ejecuta sin dependencias de red. Los bloques de esta página usan un panel sintético con un factor, diez indicadores, un último mes incompleto y 39 trimestres de PIB determinados por el promedio trimestral del factor:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
T_m, N = 119, 10                                   # 39 trimestres completos + 2 meses de 2024Q4
F = np.zeros(T_m)
for t in range(1, T_m):
    F[t] = 0.85 * F[t - 1] + rng.normal(scale=0.5)
lam = rng.uniform(0.5, 1.5, N)
X = F[:, None] * lam[None, :] + rng.normal(scale=0.3, size=(T_m, N))
monthly_df = pd.DataFrame(X, index=pd.date_range("2015-01-01", periods=T_m, freq="MS"),
                          columns=[f"ind_{i}" for i in range(N)])
monthly_df.iloc[-1, [3, 4, 8, 9]] = np.nan         # cuatro series aún no publicadas
gdp_series = pd.Series(1.0 + 2.0 * F[:117].reshape(39, 3).mean(axis=1) + rng.normal(scale=0.1, size=39),
                       index=pd.period_range("2015Q1", periods=39, freq="Q"), name="gdp")
```

---

## 1. `nowcast_gdp`: factores, VAR de factores y ecuación puente

Es el método estándar de trabajo y el que genera cifras publicables directamente. Opera en cinco fases integradas:

1. Estandariza el panel mensual, interpola inicialmente los datos faltantes y extrae factores mediante componentes principales (PCA), re-imputando los valores omitidos a partir de la reconstrucción de rango $k$, $F\Lambda' = U_k S_k V_k'$ (con $F = \sqrt{T}\,U_k$ y $\Lambda = V_k S_k/\sqrt{T}$), hasta la convergencia del algoritmo EM (`max_em_iter=50`, `em_tol=1e-4`). Un mes sin ninguna observación no aporta información y se mantiene en la media del panel durante el EM.
2. Estima un VAR(`p_factor_lags`) sobre los factores (MCO, sin constante) y lo usa para sustituir el factor de los meses completamente vacíos por su pronóstico y para pronosticar los meses del trimestre objetivo que no están en el panel.
3. Promedia los factores mensuales — observados y pronosticados — dentro de cada trimestre natural.
4. Estima una regresión puente por MCO del PIB trimestral histórico sobre los factores trimestralizados, alineando ambas series por **etiqueta de trimestre**.
5. Aplica los coeficientes estimados al promedio de factores del trimestre objetivo.

```python
from puremacro.nowcast import nowcast_gdp

res = nowcast_gdp(monthly_df, gdp_series, n_factors=2)
print("Nowcast puntual:", res.nowcast)
print("Trimestre objetivo:", res.target_quarter)
print("R² de la ecuación puente:", res.model_r2)
print("Meses del trimestre objetivo pronosticados por el VAR:", len(res.factor_forecast))
print(res.summary())
```

### Índices de entrada

| Argumento | Tipo | Índice necesario | Valores faltantes |
|---|---|---|---|
| `monthly_data` | `DataFrame` (T_meses × N) | `DatetimeIndex` mensual | NaN en cualquier posición; los NaN de las últimas filas *son* el borde irregular |
| `quarterly_gdp` | `Series` | `PeriodIndex` o `DatetimeIndex` trimestral, o cadenas `'2024Q3'` | los trimestres NaN se excluyen del puente |

Los factores mensuales se agrupan por trimestre natural con etiquetas del tipo `'2015Q1'`, y el índice de `quarterly_gdp` se convierte del mismo modo (`to_period("Q")` para `PeriodIndex` y `DatetimeIndex`; `str` para cualquier otro). Un `PeriodIndex`, un `DatetimeIndex` o cadenas dan exactamente el mismo resultado, y una serie de PIB que empieza cuatro trimestres después del panel se alinea sobre los trimestres que cubre. **Si coinciden menos de cuatro etiquetas se lanza `ValueError`** (con una muestra de ambos conjuntos de etiquetas en el mensaje): ya no existe la alineación posicional silenciosa. Un `monthly_data` sin `DatetimeIndex` se agrupa posicionalmente en bloques de tres filas etiquetados `'Q1', 'Q2', …`, y `quarterly_gdp` debe llevar esas mismas etiquetas.

```python
gdp_str = gdp_series.copy(); gdp_str.index = gdp_series.index.astype(str)
assert nowcast_gdp(monthly_df, gdp_str, n_factors=2).nowcast == res.nowcast
```

### Dónde debe terminar el panel

Termine el panel mensual en el último mes con al menos una observación publicada. Añadir filas completamente NaN hasta el final del trimestre es **equivalente**, no perjudicial: un mes sin observaciones recibe el pronóstico del VAR de factores, que es exactamente lo que produce el paso de compleción para un mes ausente del panel (las dos rutas coinciden a 5e-15 en el panel sintético). Lo que sí cuesta es información: los meses que faltan son pronósticos de un proceso AR con raíz 0,85, que revierten a la media. Sobre 200 réplicas del panel anterior, el RMSE frente al PIB realizado es 0,153 con los tres meses observados, 0,409 con dos, 0,735 con uno (y 0,735 con uno más dos filas NaN añadidas), frente a 1,758 para la media histórica. Una última fila totalmente NaN deja vacía `news_decomposition`.

### Resultado

`NowcastResult` es un `dataclass` inmutable con `nowcast` (en las unidades de `quarterly_gdp`; no se anualiza nada), `target_quarter` (solo una etiqueta: el nowcast es siempre para el último trimestre del panel), `factors` (T_meses × K, indexado como `monthly_data`), `factor_forecast` (0–2 filas: los meses del trimestre objetivo posteriores al panel), `loadings` (N × K, escalados de modo que `factors @ loadings.T` es la reconstrucción de rango K del panel estandarizado), `bridge_coefficients` (`const, Factor_1 … Factor_K`), `factor_var` (matriz $B$ de dimensión $Kp \times K$, o `None` si hubo demasiado pocos meses), `news_decomposition` y `model_r2` (recortado a $[0, 1]$; `0.0` si `quarterly_gdp` no tiene varianza). Métodos de presentación: `summary()`, `to_frame()` (la tabla de noticias), `to_markdown()`, `to_latex()`, `to_typst()` y `plot()` (factores, con los meses pronosticados en trazo discontinuo; devuelve la figura).

`p_factor_lags` (por defecto 1) es el orden del VAR de factores de la fase 2: afecta a la compleción del trimestre objetivo y al relleno de los meses vacíos, y a nada más.

---

## 2. Modelo de factores dinámicos con filtro de Kalman (`kalman_dfm`)

Implementa la metodología en dos etapas de Doz, Giannone y Reichlin (2011): PCA sobre las filas completas como valor inicial, un VAR(p) sobre esos factores como ecuación de transición, cargas por MCO y una pasada del suavizador de Kalman sobre el panel completo con sus huecos. Devuelve un `KalmanDFMResult` — una subclase de `dict`, de modo que `out["factors"]` sigue funcionando — con `summary()`, `to_frame()` (cargas), `to_markdown()` / `to_latex()` / `to_typst()` y `plot()`:

```python
from puremacro.nowcast import kalman_dfm

dfm_res = kalman_dfm(monthly_df, n_factors=2, p=1)
print(dfm_res.summary())
dfm_res["X_filled_df"].tail(3)     # las filas incompletas, rellenadas: este es el nowcast
```

Claves: `factors`, `loadings`, `A`, `Q`, `H`, `X_filled`, `means`, `stds`, `loglik`, `n_missing` y, solo cuando `X` es un DataFrame, `factors_df` y `X_filled_df`. La trampa está en los casos completos: las fases 1–3 se estiman únicamente sobre las filas sin ningún NaN, de modo que una serie que empieza tarde reduce drásticamente la muestra de estimación sin aviso (el único guardarraíl es `ValueError: too few complete-cases rows …`). Recorte el inicio del panel antes de llamar.

---

## 3. VAR de frecuencias mixtas (`mf_var`)

Mariano-Murasawa: la variable trimestral tiene una contrapartida mensual latente $m^*_t$ dentro de un VAR en forma compañera, con la restricción de agregación $y^Q_t = (m^*_t + m^*_{t-1} + m^*_{t-2})/3$ como ecuación de observación que solo actúa en los meses con dato trimestral.

```python
from puremacro.nowcast import mf_var

panel = monthly_df[["ind_0", "ind_1"]].copy()
latent = pd.Series(0.5 + 0.8 * F + rng.normal(scale=0.2, size=T_m), index=monthly_df.index)
panel["gdp"] = latent.rolling(3).mean().where(panel.index.month % 3 == 0)   # estampado a fin de trimestre

out = mf_var(panel, quarterly_col="gdp", p=3)
out["df_filled"]["gdp_monthly"]   # trayectoria mensual latente de la variable trimestral
```

- `p` debe ser ≥ 3 (lanza `ValueError` en caso contrario).
- `quarter_end_offset` indica en qué mes del trimestre está estampado el dato: `2` (por defecto, fin de trimestre), `1` (mes central) o `0` (primer mes); otros valores lanzan `ValueError`. Internamente el dato estampado en el mes $t$ con desfase $o$ se traslada a $t + (2 - o)$, de modo que la restricción retrospectiva siempre vincula los tres meses del trimestre al que pertenece el valor: los mismos datos estampados en el mes 0, 1 o 2 dan la misma trayectoria mensual (a 1e-6) y la restricción se cumple a 1,9e-06 con cualquier desfase. Declare el desfase que usó: estampar en el primer mes y dejar el valor por defecto aplica la restricción a los meses del trimestre *anterior*.
- `gdp_monthly` está en la escala de la variable trimestral (es un promedio, no una suma).
- `df_monthly` rellena los huecos de las columnas mensuales; `df_filled` conserva sus NaN.

---

## 4. Descomposición de noticias (*news*)

`res.news_decomposition` contiene una fila por cada serie observada en la **última** fila del panel: `actual`, `forecast` (el valor implícito por los factores para ese mismo mes, en las unidades de la serie: $\bar x_i + s_i F_T \Lambda_i$), `surprise` (= `actual − forecast`), `weight` ($\beta'(\Lambda'\Lambda)^{-1}\Lambda_i / (3\sigma_i)$: el efecto de una sorpresa unitaria en la serie $i$ sobre el nowcast, a través de la proyección factorial y la pendiente del puente) y `contribution` (= `weight × surprise`, en las unidades de `quarterly_gdp`).

```python
print(res.news_decomposition)
print(res.to_markdown())
```

Léala como una ordenación de qué series publicadas empujan el trimestre al alza o a la baja. **No** es la descomposición de noticias de Bańbura-Modugno: `nowcast_gdp` se llama una sola vez, con un único conjunto de información, de modo que no hay revisión entre dos nowcasts que descomponer, y `forecast` es un valor ajustado calculado con esa observación ya en el panel, no una expectativa previa a la publicación.

---

## 5. Combinación y evaluación

Los combinadores de `puremacro.nowcast` (`equal_weight`, `inverse_mse`, `bates_granger`, `rank_weight`, `model_confidence_set`) calculan los pesos *dentro de muestra*, sobre las mismas filas que después combinan. Para pesos que solo vean errores pasados use `puremacro.forecast.combine_forecasts(..., rolling=n)`: los pesos aplicados a `forecasts[t]` se construyen con `errors[t-n:t]`, los últimos `n` errores *estrictamente anteriores* a `t` (pesos iguales mientras haya menos de dos). Las reglas de puntuación (`crps_gaussian`, `crps_ensemble`, `log_score_gaussian`, `brier_score`, `pit_histogram`) devuelven vectores por observación. Consulte la versión inglesa para las colisiones de nombres con `puremacro.forecast` (`model_confidence_set` y `crps_ensemble` calculan cosas distintas en cada módulo).
