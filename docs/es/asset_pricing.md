> 🇬🇧 [English](../asset_pricing.md) · 🇪🇸 Español

Para evaluación histórica de carteras, asignación, análisis técnico/fundamental
y dinámica de chartistas y fundamentalistas, consulte [investigación de inversiones](investment_research.md).

# Rendimientos, riesgo y pruebas de valoración de activos

`puremacro.asset_pricing` mide el riesgo de los rendimientos y contrasta
modelos de valoración de activos, incluidos modelos cuyos factores proceden
de datos macroeconómicos. Cubre rendimientos y estadísticos de riesgo, valor
en riesgo y sus pruebas retrospectivas, las pruebas de factores de serie
temporal (GRS) y de Fama-MacBeth, el GMM del factor de descuento estocástico
(SDF) lineal, la distancia de Hansen-Jagannathan y el CAPM de consumo de
Hansen-Singleton. Los estimadores basados en momentos usan el motor GMM
público `puremacro.inference.gmm`. La batería de pruebas contrasta cada
estimador con una implementación independiente (statsmodels, linearmodels,
fórmulas cerradas o una minimización directa). En tiempo de ejecución solo
se necesitan NumPy, SciPy y pandas.

Todas las entradas son rendimientos por periodo en **decimales**. Todos los
estimadores rechazan valores NaN o infinitos y nombran las columnas
afectadas; no se descarta nada en silencio. Se rechaza una columna de
rendimientos que parezca expresada en porcentaje (mediana de `|r|` mayor que
0,2, o más del 5% de sus valores mayores que 1 en valor absoluto); use
`check_scale=False` para rendimientos decimales genuinos de ese tamaño. Las series con `DatetimeIndex` se emparejan por fecha, y las
frecuencias mezcladas producen un error.

Los ejemplos siguientes se ejecutan sin conexión con datos sintéticos de
semilla fija.

## Rendimientos y riesgo

```python
import numpy as np
import pandas as pd
from puremacro import asset_pricing as ap

rng = np.random.default_rng(1)
idx = pd.date_range("2000-01-01", periods=240, freq="MS")
factors = pd.DataFrame(0.005 + 0.04 * rng.standard_normal((240, 2)),
                       index=idx, columns=["mkt", "hml"])
loadings = rng.uniform(0.5, 1.5, (2, 6))
excess = pd.DataFrame(factors.to_numpy() @ loadings + 0.02 * rng.standard_normal((240, 6)),
                      index=idx, columns=[f"p{i}" for i in range(6)])
rf = pd.Series(0.002, index=idx)

prices = 100 * (1 + excess["p0"] + rf).cumprod()
r = ap.to_returns(prices)                       # simple returns, first row dropped
quarterly = ap.compound(excess, to="Q")         # prod(1 + r) - 1 within quarters
print(round(ap.annualize_return(excess["p0"], 12), 4),
      round(ap.annualize_volatility(excess["p0"], 12), 4))

risk = ap.risk_summary(excess, periods_per_year=12)
print(risk.summary())
risk.to_dataframe()[["name", "sharpe", "sharpe_se", "max_drawdown"]]
```

`sharpe_se` es el error estándar de Lo (2002) por el método delta, con una
covarianza de largo plazo de Bartlett (`hac_lags=None` usa
`floor(4 (T/100)^(2/9))`; `hac_lags=0` da la fórmula iid). Use `rf=` para
restar una tasa libre de riesgo antes de cada estadístico.

## Valor en riesgo y pruebas retrospectivas

```python
loss = excess["p0"]
print(ap.value_at_risk(loss, alpha=0.05),                     # historical
      ap.value_at_risk(loss, alpha=0.05, method="normal"),
      ap.expected_shortfall(loss, alpha=0.05, method="garch"))  # filtered historical simulation

bt = ap.var_backtest(loss, ap.value_at_risk(loss, alpha=0.05), alpha=0.05)
print(bt.summary())
```

El VaR y el ES son pérdidas positivas para el periodo siguiente. `"garch"`
filtra la serie con `puremacro.garch.garch11_fit` y escala el cuantil de
los residuos estandarizados por la volatilidad prevista a un paso (solo
horizonte 1). `var_backtest` da la prueba de cobertura incondicional de
Kupiec y las pruebas de independencia y de cobertura condicional de
Christoffersen. Sin violaciones, los estadísticos siguen siendo finitos
(`0 log 0 = 0`). El VaR dentro de muestra usado aquí solo ilustra la
llamada: una prueba retrospectiva real usa pronósticos formados antes de
cada periodo.

## Pruebas de factores

```python
ts = ap.time_series_test(excess, factors, hac_lags=4)
print(ts.summary())          # alphas, GRS F(N, T-N-K), HAC Wald chi2(N)

fm = ap.fama_macbeth(excess, factors)
print(fm.summary())          # lambdas with Fama-MacBeth, Shanken and Newey-West se
```

`time_series_test` regresa cada activo sobre `[1, f]`. El estadístico GRS
es `F(N, T-N-K)` exacto solo con errores normales iid y con factores que
sean excesos de rendimiento negociables. Para factores macroeconómicos use
`traded=False`: el GRS no se reporta, pero la prueba de Wald HAC sí.
`fama_macbeth` acepta `factors` (betas de regresiones de serie temporal con
la muestra completa) o `betas=`.

## Factor de descuento estocástico y distancia HJ

```python
sdf = ap.sdf_gmm(excess, factors)            # m = 1 - b'(f - mu), two-step GMM
print(sdf.summary())

hj = ap.hj_distance(excess, factors, n_sim=20_000, seed=0)
print(hj.summary())
```

`sdf_gmm` estima `b` (y `mu`, con `demeaned=True`) mediante
`inference.gmm` y reporta las primas implícitas `lambdas = Sigma_f b`, los
errores de valoración y la prueba J de Hansen. `hj_distance` usa la matriz
de ponderación de segundos momentos y el valor p de chi-cuadrado ponderada
de Jagannathan-Wang (1996). Simula el valor p con la semilla `seed`, que
queda registrada.

La distancia HJ rechaza activos redundantes y ambos estimadores rechazan
cargas factoriales no identificadas con `ValueError`. El VaR y el ES históricos
componen rendimientos simples en ventanas superpuestas de varios períodos; el
método normal conserva su aproximación aditiva iid.

## CAPM de consumo con datos macroeconómicos

```python
T = 5_000
g = pd.Series(np.exp(0.005 + 0.02 * rng.standard_normal(T)))   # gross consumption growth
beta, gamma = 0.98, 4.0
def moment(a):                                                # E[g^a], lognormal growth
    return np.exp(a * 0.005 + 0.5 * a ** 2 * 0.02 ** 2)

gross = pd.DataFrame({"equity": g / (beta * moment(1 - gamma)),   # Lucas-tree claim on C
                      "levered": g ** 3 / (beta * moment(3 - gamma))})  # claim on C^3
rf_gross = pd.Series(np.full(T, 1 / (beta * moment(-gamma))))
cc = ap.ccapm(gross, g, rf=rf_gross, instruments=np.ones((T, 1)))
print(cc.summary())
```

`ccapm` estima las ecuaciones de Euler con utilidad potencial
`E[(beta g^-gamma R - 1) z_t] = 0`. Necesita rendimientos y crecimiento
**brutos**. Los instrumentos por defecto son una constante, el crecimiento
del consumo rezagado y los rendimientos rezagados. Con datos reales,
construya las entradas a partir de FRED:

```python
# requires: network
g_q = ap.fred_consumption_growth(frequency="Q", start="1959")   # per-capita real ND + services
rf_q = ap.fred_risk_free(frequency="Q", start="1959")           # TB3MS / 100 / 4, quarter average
```

## Fuentes de datos

```python
# requires: network
from puremacro.fetch.kenfrench_factors import fetch

ff3 = fetch("ff3")                  # Mkt-RF, SMB, HML, RF in decimals
p25 = fetch("25_size_bm")           # 25 value-weighted Size-B/M portfolios plus RF
sample = slice("1963-07-01", "2013-12-01")
ex25 = p25.loc[sample].drop(columns="RF").sub(p25.loc[sample, "RF"], axis=0)
grs = ap.time_series_test(ex25, ff3.loc[sample, ["Mkt-RF", "SMB", "HML"]], hac_lags=0)
grs.grs_stat     # about 3.57 on the 2026 library; Fama and French (2015) report 3.62
```

- **Biblioteca de datos de Ken French.** Los archivos indican "Copyright
  Eugene F. Fama and Kenneth R. French" y no conceden licencia de
  redistribución. puremacro no incluye ninguno. `fetch.kenfrench_factors.fetch`
  los descarga bajo demanda en su caché HTTP de usuario
  (`~/.cache/puremacro/http`, o `$PUREMACRO_HTTP_CACHE_DIR`). El registro de
  fuentes marca `kenfrench.factors` como no redistribuible. La biblioteca se
  revisa después de la publicación (correcciones de CRSP, nuevos puntos de
  corte). Por eso los estadísticos publicados solo se reproducen de forma
  aproximada. La prueba de red admite un 15% y mide 3,567 frente al 3,62
  publicado.
- **FRED** (sin clave). `fred_consumption_growth` suma el gasto nominal
  mensual en bienes no duraderos (`PCEND`) y en servicios (`PCES`), cada uno
  deflactado por su propio índice de precios encadenado (`DNDGRG3M086SBEA`,
  `DSERRG3M086SBEA`), y divide por la población (`POPTHM`). Con `"Q"`
  promedia dentro de cada trimestre. Sumar dos series en dólares
  encadenados solo aproxima un agregado encadenado. Si falta cualquier
  serie, se lanza `RuntimeError` en lugar de calcular el crecimiento con una
  suma parcial. `fred_risk_free` convierte `TB3MS` (porcentaje anual, base de
  descuento) con una división simple, que es una aproximación.

## GMM genérico

```python
from puremacro.inference import gmm

zi = rng.standard_normal((2_000, 3))
x = zi @ np.array([0.8, 0.5, 0.3]) + rng.standard_normal(2_000)
y = 1.0 + 2.0 * x + rng.standard_normal(2_000)
Z = np.column_stack([np.ones(2_000), zi])
X = np.column_stack([np.ones(2_000), x])

fit = gmm(lambda th, d: d[2] * (d[0] - d[1] @ th)[:, None], np.zeros(2), (y, X, Z),
          names=("const", "x"))
print(fit.summary())
fit.wald(np.eye(2)[[1]], [2.0])
```

`gmm(moments, theta0, data)` recibe las contribuciones de momentos por
periodo `(T, q)`. Admite `weighting="two_step"` (por defecto),
`"iterated"`, `"identity"` o una matriz, un `jacobian` analítico opcional y
una covarianza de largo plazo de Bartlett. La covarianza es el sándwich
`B G'W S W G B / T`, `B = (G'WG)^-1`, que es la forma eficiente cuando
`W = S^-1`. La `J = T gbar'W gbar` de Hansen solo se reporta con las
ponderaciones eficientes. En VI lineal, las estimaciones, los errores
estándar y J coinciden con `IV2SLS`/`IVGMM` de linearmodels y con
`inference.hansen_j` a 1e-8.

## Supuestos y límites

- **GRS** es exacto solo con errores normales iid y requiere factores que
  sean excesos de rendimiento negociables. En otro caso, use la prueba de
  Wald HAC, pero tenga en cuenta que rechaza en exceso en muestras cortas
  (alrededor del 23% con un nivel nominal del 5%, T = 60 y N = 10): con
  factores negociables y pocos periodos, prefiera el GRS.
- **Ancho de banda HAC.** El valor por defecto de toda covarianza de largo
  plazo de Bartlett es `floor(4 (T/100)^(2/9))`, sin centrar. `hac_lags=0`
  es el caso iid (robusto a heterocedasticidad).
- **Anualización del Sharpe.** Multiplica por `sqrt(periods_per_year)` e
  ignora el factor de correlación serial eta(q) de Lo.
- **Distancia HJ con excesos de rendimiento.** La media del SDF no está
  identificada, por lo que la distancia depende de la normalización
  `m = 1 - b'f` (Kan y Robotti 2008).
- **Errores estándar de Shanken.** Necesitan los residuos de la primera
  etapa. Si se dan `betas=`, son `NaN` y `se` vuelve a Fama-MacBeth.
  `LinearFactorModel` de linearmodels reporta un error estándar robusto de
  GMM. Es asintóticamente igual al de Shanken con errores iid, pero es una
  definición distinta: la prueba mide una diferencia de alrededor del 0,2%.
- **Lambdas del SDF.** Tratan la covarianza de los factores como conocida.
  `LinearFactorModelGMM` de linearmodels estima la representación en betas,
  un estimador distinto, así que las dos primas solo coinciden de forma
  aproximada.
- **La gamma del CCAPM está débilmente identificada** con datos de consumo
  (Stock y Wright 2000). Reporte la prueba J y los intervalos del 95% que
  imprime `summary()`, no solo la estimación puntual.
- **Protección contra porcentajes.** Es una heurística aplicada a cada
  columna de rendimientos: mediana de `|r|` mayor que 0,2 o más del 5% de
  valores con `|r| > 1`. Los rendimientos aislados superiores al 100% (el
  archivo de las 25 carteras Size-B/M en 1932, acciones individuales,
  criptoactivos) se aceptan. Las series volátiles que la activen pueden
  pasarse con `check_scale=False`. Los factores solo se comprueban por
  finitud, salvo los factores negociables en `time_series_test(traded=True)`.
- **Convenciones de GMM.** La covarianza sándwich no lleva corrección de
  muestra pequeña. J es `NaN` con ponderación identidad o de usuario, y
  exactamente cero cuando el modelo está exactamente identificado.
