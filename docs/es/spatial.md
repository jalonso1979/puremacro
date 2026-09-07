> 🇬🇧 [English](../spatial.md) · 🇪🇸 Español

# Econometría espacial para macro regional

`puremacro.spatial` incorpora las herramientas espaciales que más necesitan la macro regional y el comercio aplicado: matrices de pesos espaciales, los diagnósticos de autocorrelación I de Moran y C de Geary, y errores estándar HAC espaciales de Conley para cortes transversales y para las proyecciones locales de panel con efectos fijos bidireccionales de `puremacro.lp`. Su complemento `puremacro.bartik.shift_share_iv` estima regresiones de variables instrumentales shift-share (Bartik) con los errores estándar a nivel de shock de Adão, Kolesár y Morales (2019).

Todo funciona sobre el núcleo Pyodide de cuatro paquetes (numpy, scipy, pandas, matplotlib). No hace falta ninguna pila GIS: las coordenadas son columnas de latitud y longitud, los vecinos son diccionarios y la proximidad económica es una matriz de flujos.

---

## 1. Pesos espaciales

Un objeto `SpatialWeights` guarda una matriz dispersa `W` de dimensión `n × n` con diagonal nula y las etiquetas de las unidades `ids`. Cuatro constructores cubren los casos habituales:

| Constructor | Entrada | Peso |
| --- | --- | --- |
| `contiguity_weights(vecinos)` | `{unidad: [vecino, ...]}` | 1 si comparten frontera (simetrizado por defecto) |
| `knn_weights(coords, k)` | coordenadas latitud/longitud (o planas) | 1 para las `k` unidades más cercanas |
| `distance_weights(coords, cutoff, decay=...)` | coordenadas y un radio en km | decaimiento inverso, uniforme o gaussiano dentro del radio |
| `economic_weights(flujos)` | matriz de flujos origen × destino (comercio, migración, input-output) | participación del flujo en la fila, sin flujos propios |

Todos los constructores estandarizan por filas por defecto, de modo que `W.lag(x)` es el promedio de `x` entre los vecinos. Las unidades sin vecinos se reportan como islas (`W.n_islands`, `W.islands`) y `distance_weights` avisa cuando el radio deja alguna.

```python
import pandas as pd
from puremacro.spatial import distance_weights, knn_weights

capitales = pd.DataFrame(
    {
        "lat": [40.4168, 41.3874, 39.4699, 37.3891, 41.6488, 36.7213, 43.2630, 43.3623],
        "lon": [-3.7038, 2.1686, -0.3763, -5.9845, -0.8891, -4.4214, -2.9350, -8.4115],
    },
    index=["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Malaga", "Bilbao", "A Coruna"],
)
W = distance_weights(capitales, cutoff=450.0, decay="inverse")   # km haversine por defecto
print(W.summary())
print(W.neighbors("Madrid"))                                      # {etiqueta: peso}
Wk = knn_weights(capitales, k=3)
print(Wk.to_frame().head())                                       # lista de aristas: source, target, weight
```

`W.lag(serie)` alinea una Series o un DataFrame de pandas por etiqueta, así que el orden de sus datos nunca tiene que coincidir con el de los pesos. `W.to_dense()` devuelve la matriz numpy cuando la necesite dentro de un solucionador.

---

## 2. Diagnósticos de autocorrelación espacial

La I de Moran y la C de Geary resumen si una variable está agrupada (vecinos parecidos), dispersa (vecinos distintos) o distribuida al azar sobre el mapa. Con `z = x − media(x)` y `S₀ = Σᵢⱼ wᵢⱼ`:

$$I = \frac{n}{S_0}\,\frac{z' W z}{z' z}, \qquad C = \frac{(n-1)\sum_{ij} w_{ij}(x_i - x_j)^2}{2 S_0\, z'z}.$$

Sin autocorrelación espacial, `E[I] = −1/(n−1)` y `E[C] = 1`. La autocorrelación positiva empuja `I` por encima de su esperanza y `C` por debajo de uno. Ambas funciones reportan las varianzas de Cliff y Ord bajo normalidad y bajo aleatorización, con sus estadísticos z y valores p, además de un valor p de permutación que reordena `x` entre unidades (`n_perm`, por defecto 999).

```python
import numpy as np
from puremacro.spatial import contiguity_weights, gearys_c, morans_i

rng = np.random.default_rng(0)
lado = 8
vecinos = {}
for i in range(lado):
    for j in range(lado):
        u = i * lado + j
        vecinos[u] = [v for v in (u - lado, u + lado, u - 1, u + 1)
                      if 0 <= v < lado * lado and abs((v % lado) - j) <= 1]
W = contiguity_weights(vecinos)                 # contigüidad tipo torre en una retícula 8 x 8
rho = 0.6
x = np.linalg.solve(np.eye(W.n) - rho * W.to_dense(), rng.standard_normal(W.n))   # campo SAR
print(morans_i(x, W, n_perm=499).summary())
print(gearys_c(x, W, n_perm=499).summary())
```

`MoranResult.plot()` dibuja el diagrama de dispersión de Moran (`z` frente a su rezago espacial); la pendiente de la recta ajustada es la I de Moran. Los estadísticos están contrastados frente a `esda` (PySAL) con precisión 1e-10.

---

## 3. Errores estándar HAC espaciales de Conley

Los shocks regionales están correlacionados entre unidades cercanas. Agrupar por región administrativa supone que la correlación se detiene en la frontera; Conley (1999) deja en cambio que la covarianza de los scores decaiga con la distancia:

$$\hat V = (X'X)^{-1}\Big[\sum_i \sum_j K(d_{ij})\, u_i u_j\, x_i x_j'\Big](X'X)^{-1}, \qquad K(d) = \max\{0,\, 1 - d/\text{radio}\}\ \text{(Bartlett) o } \mathbf{1}\{d \le \text{radio}\}\ \text{(uniforme)}.$$

El radio es el ancho de banda: con `cutoff=0` el estimador es exactamente HC0, y con núcleo uniforme y unidades agrupadas en bloques muy alejados es exactamente la covarianza robusta por conglomerados. Reporte varios radios; los errores deberían estabilizarse en cuanto el radio supere el alcance de la correlación espacial.

```python
import numpy as np
from puremacro.spatial import conley_se, pairwise_distances

rng = np.random.default_rng(1)
n = 200
coords = rng.uniform([36.0, -9.0], [43.5, 3.0], size=(n, 2))      # lat, lon sobre España
D = pairwise_distances(coords, "haversine")
comun = np.exp(-D / 150.0) @ rng.standard_normal(n)               # shock correlacionado dentro de ~150 km
x = rng.standard_normal(n) + 0.5 * comun
y = 1.0 + 0.5 * x + 0.5 * rng.standard_normal(n) + comun
X = np.column_stack([np.ones(n), x])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
for cutoff in (0.0, 100.0, 300.0):
    print(f"radio {cutoff:5.0f} km  se(beta) = {conley_se(X, resid, coords, cutoff)[1]:.4f}")
```

### 3.1 Proyecciones locales de panel con HAC espacial

`panel_lp` acepta `cov_type="conley"`. La covarianza es el HAC espacio-temporal de Hsiang (2010): un núcleo de Conley entre unidades dentro de cada periodo y un núcleo de Bartlett entre periodos hasta `time_lags` (por defecto, la regla de ancho de banda de Driscoll-Kraay que ya usa `panel_lp_dk`). Con un radio mayor que todas las distancias por pares colapsa exactamente en Driscoll-Kraay; con `time_lags=0` es una covarianza de Conley periodo a periodo.

```python
import numpy as np
import pandas as pd
from puremacro.lp import panel_lp

rng = np.random.default_rng(2)
regiones = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Malaga", "Bilbao", "A Coruna"]
coords = pd.DataFrame(
    {
        "lat": [40.4168, 41.3874, 39.4699, 37.3891, 41.6488, 36.7213, 43.2630, 43.3623],
        "lon": [-3.7038, 2.1686, -0.3763, -5.9845, -0.8891, -4.4214, -2.9350, -8.4115],
    },
    index=regiones,
)
T = 80
comun = rng.standard_normal(T)
filas = []
for r in regiones:
    shock = rng.standard_normal(T)
    y = np.cumsum(0.3 * shock + 0.5 * comun + rng.standard_normal(T))
    filas += [{"code": r, "date": t, "y": y[t], "shock": shock[t]} for t in range(T)]
panel = pd.DataFrame(filas).set_index(["code", "date"])
irf = panel_lp(panel, "y", "shock", horizons=range(0, 9), n_lags=2,
               cov_type="conley", coords=coords, cutoff_km=400.0)
print(irf.round(3))
```

`coords` debe estar indexado por las etiquetas de entidad del panel (las entidades ausentes lanzan `KeyError`). Use `kernel="uniform"` para un corte duro y `metric="euclidean"` cuando las coordenadas ya sean planas (kilómetros sobre una malla proyectada).

---

## 4. VI shift-share con errores estándar a nivel de shock

Un instrumento shift-share combina participaciones de exposición del periodo base `sᵢₖ` con shocks sectoriales `gₖ`: `zᵢ = Σₖ sᵢₖ gₖ`. Adão, Kolesár y Morales (2019) muestran que las unidades con vectores de participaciones parecidos tienen residuos correlacionados aunque estén lejos, así que los errores robustos a heterocedasticidad o agrupados geográficamente cubren de menos. Su estimador agrega los residuos al nivel sectorial:

$$\widehat{\text{se}}_{\text{AKM}}(\hat\beta) = \frac{\sqrt{\sum_k \tilde g_k^2 \Big(\sum_i w_i s_{ik} \hat\varepsilon_i\Big)^2}}{\big|\sum_i w_i \tilde z_i \tilde x_i\big|},$$

donde las tildes denotan residuos respecto de los controles (y, para los shocks, respecto de los `shock_controls` ponderados por participaciones). `shift_share_iv` devuelve la estimación 2SLS, ambos errores estándar, el F robusto de primera etapa y los pesos de Rotemberg de Goldsmith-Pinkham, Sorkin y Swift (2020), que indican qué sectores mueven la estimación.

```python
import numpy as np
import pandas as pd
from puremacro.bartik import shift_share_iv

rng = np.random.default_rng(3)
n_regiones, n_industrias = 300, 25
participaciones = pd.DataFrame(rng.dirichlet(np.full(n_industrias, 0.5), size=n_regiones),
                               columns=[f"ind{k:02d}" for k in range(n_industrias)])
shocks = pd.Series(rng.standard_normal(n_industrias), index=participaciones.columns)   # shocks nacionales por industria
exposicion = participaciones.to_numpy() @ shocks.to_numpy()
crecimiento_empleo = exposicion + rng.standard_normal(n_regiones)
confusor_industrial = participaciones.to_numpy() @ rng.standard_normal(n_industrias)  # lo que rompe los errores robustos
crecimiento_salarial = 0.8 * crecimiento_empleo + confusor_industrial + 0.5 * rng.standard_normal(n_regiones)
df = pd.DataFrame({"crecimiento_salarial": crecimiento_salarial, "crecimiento_empleo": crecimiento_empleo})
res = shift_share_iv(df, "crecimiento_salarial", "crecimiento_empleo", participaciones, shocks)
print(res.summary())
print(res.rotemberg_weights.sort_values(ascending=False).head())
```

Pase `se="robust"` para que el error convencional sea el principal, `weights=` para ponderaciones poblacionales y `controls=` para covariables a nivel de unidad. Las participaciones deben ser no negativas y se alinean con `df` por índice cuando se dan como DataFrame.

---

## 5. Elegir la especificación

Cuatro especificaciones se apoyan en la misma matriz de pesos, y no están anidadas de una forma que permita a una sola regresión escoger entre ellas:

| modelo | ecuación | estimador |
| --- | --- | --- |
| SLX | `y = X β + W X_d θ + ε` | MCO |
| SAR | `y = ρ W y + X β + ε` | MV concentrada o GMM de Kelejian y Prucha |
| SEM | `y = X β + u`, `u = λ W u + ε` | MV concentrada o GMM de Kelejian y Prucha |
| SDM | `y = ρ W y + X β + W X_d θ + ε` | MV concentrada o GMM de Kelejian y Prucha |

`ols_spatial` es el punto de entrada que hay que llamar primero. Es mínimos cuadrados ordinarios con la batería de multiplicadores de Lagrange de Anselin, Bera, Florax y Yoon (1996) incorporada: cinco estadísticos calculados enteramente **bajo la hipótesis nula de ausencia de dependencia espacial**, de modo que no hay que estimar nada espacial para obtenerlos. Con `e` el residuo MCO, `σ̂² = e'e/n`, `T = tr(WW + W'W)`, `M = I − X(X'X)⁻¹X'` y `nJ = [(WXb)'M(WXb) + T σ̂²]/σ̂²`:

$$\text{LM}_{\text{lag}} = \frac{(e'Wy/\hat\sigma^2)^2}{nJ}, \qquad \text{LM}_{\text{err}} = \frac{(e'We/\hat\sigma^2)^2}{T},$$

$$\text{RLM}_{\text{lag}} = \frac{(d_\rho - d_\lambda)^2}{nJ - T}, \qquad \text{RLM}_{\text{err}} = \frac{(d_\lambda - T d_\rho/nJ)^2}{T\,(1 - T/nJ)},$$

con `d_ρ = e'Wy/σ̂²` y `d_λ = e'We/σ̂²`. Las versiones robustas son lo esencial: en un SAR genuino el estadístico *de error* también rechaza, y viceversa, así que el par simple casi siempre rechaza dos veces. `RLM_lag` es robusto a un proceso de error localmente mal especificado y `RLM_err` a un rezago localmente mal especificado, y ese par es el que los separa.

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import contiguity_weights, ols_spatial

estados = load_us_state_centroids()
contiguos48 = estados.drop(index=["AK", "HI", "PR"])              # 48 estados + DC
vecinos = {s: [v for v in fila.split() if v in contiguos48.index]
           for s, fila in contiguos48["neighbors"].items()}
W = contiguity_weights(vecinos)                                   # el grafo real de fronteras
ids = list(W.ids)
coords = contiguos48.loc[ids, ["lat", "lon"]]
print(W.summary())

rng = np.random.default_rng(11)
n = W.n
area_log = np.log(contiguos48.loc[ids, "land_sqmi"].to_numpy())
area_log = (area_log - area_log.mean()) / area_log.std()
X = pd.DataFrame({"area_log": area_log, "costero": rng.standard_normal(n)}, index=ids)
A_inv = np.linalg.inv(np.eye(n) - 0.5 * W.to_dense())             # campo SAR real, rho = 0.5
y = pd.Series(A_inv @ (1.0 + 0.8 * X["area_log"] - 0.4 * X["costero"]
                       + 0.5 * rng.standard_normal(n)), index=ids)

ajuste_ols = ols_spatial(y, X, W)
print(ajuste_ols.summary())
print(ajuste_ols.lm.recommendation)
```

`recommendation` es una cadena corta que contiene toda la regla de decisión. Si ninguno de los dos estadísticos rechaza → `ols`. Si rechaza sólo uno → el modelo correspondiente. Si rechazan ambos → decide el par robusto, y si rechazan *los dos* robustos la cadena es `sdm or sarar (both robust LM reject)`, porque dos rechazos apuntan a un modelo con dos procesos espaciales, no a uno de los simples. Si ningún robusto rechaza después de que ambos simples lo hicieran, la respuesta es `inconclusive`, y ése es el informe honesto.

La batería viene con tres advertencias. `summary()` imprime la primera; las tres quedan detalladas aquí:

- Es una prueba **local**. Dice a qué alternativa apuntan los residuos MCO, no que el modelo indicado sea el correcto.
- Todos los estadísticos usan la varianza de máxima verosimilitud `e'e/n`, tal como exige la derivación de la verosimilitud gaussiana. Sustituirla por la corregida por grados de libertad, `e'e/(n−k)`, reescalaría `RLM_lag` por `(n−k)/n` y `LM_err` por `((n−k)/n)²`, y no reescalaría `LM_lag`, `RLM_err` ni `LM_SARMA` por ninguna constante, porque `nJ` es **afín**, no lineal, en `1/σ̂²`. (Medido con `n = 60` y `k = 3`, donde `(n−k)/n = 0,95`, los cinco cocientes son 0,902812, 0,902500, 0,950000, 0,934909 y 0,909873.) El `sigma2` que `ols_spatial` reporta para sus propios errores estándar **sí** está corregido. Las dos escalas conviven a propósito.
- La I de Moran de los residuos es aquí el estadístico de Cliff y Ord (1972) con los momentos que dependen de `X`, y difiere de `morans_i` de §2, que usa los momentos de la variable en bruto. Los residuos MCO no son la variable en bruto.

`ols_spatial` admite `cov_type='nonrobust'` o `'hc1'` y deliberadamente **no** ofrece la opción de Conley: la batería se deriva bajo residuos MCO gaussianos homocedásticos, y combinarla con una covarianza HAC espacial imprimiría dos supuestos incompatibles en la misma tabla. Llame directamente a `conley_cov` de §3 si eso es lo que busca. El LM-lag robusto tampoco está definido cuando `nJ − T` se anula —`W X b` en el espacio columna de `X`, por ejemplo un único regresor que sea autovector de `W`— y `lm_spatial_tests` lanza un error con ese diagnóstico en lugar de dividir por cero.

---

## 6. Modelos espaciales de corte transversal

El estimador por defecto es la máxima verosimilitud gaussiana a través de la log-verosimilitud *concentrada* de Ord (1975): `β` y `σ²` se perfilan fuera y queda un problema unidimensional suave en el parámetro espacial,

$$\ln L_c(\rho) = -\tfrac{n}{2}\left(\ln 2\pi + 1\right) + \ln\lvert I - \rho W\rvert - \tfrac{n}{2}\ln\frac{\text{SSR}(\rho)}{n},$$

maximizado por una búsqueda escalar acotada sobre el intervalo admisible y contrastado con la raíz del score analítico. El jacobiano `ln|I − ρW|` es exacto —autovalores densos para `n ≤ 1000`, una factorización LU dispersa por encima— y no hay ningún log-determinante estocástico en el módulo, así que los mismos datos siempre dan el mismo `ρ`.

```python
from puremacro.spatial import sar, sdm, sem, slx

ajuste_sar = sar(y, X, W)          # MV gaussiana concentrada
ajuste_sem = sem(y, X, W)
ajuste_sdm = sdm(y, X, W)          # SAR ampliado con W X, más la prueba de factor común
ajuste_slx = slx(y, X, W)          # MCO sobre [X, W X]

print(pd.DataFrame(
    {"espacial": [ajuste_sar.rho, ajuste_sem.lam, ajuste_sdm.rho],
     "area_log": [ajuste_sar.params["area_log"], ajuste_sem.params["area_log"],
                  ajuste_sdm.params["area_log"]],
     "loglik": [ajuste_sar.loglik, ajuste_sem.loglik, ajuste_sdm.loglik],
     "aic": [ajuste_sar.aic, ajuste_sem.aic, ajuste_sdm.aic]},
    index=["sar", "sem", "sdm"]).round(3))
print(ajuste_sar.summary())
```

**SDM frente a SEM: la prueba de factor común.** Un SDM se reduce a un SEM exactamente cuando `θ = −ρ β_d`; escriba el SEM en forma reducida y la restricción aparece sola. `sdm` reporta la prueba de Wald de esa restricción en `common_factor = (estadístico, gl, valor p)`; el no rechazo dice que los términos Durbin adicionales son el proceso de error espacial disfrazado, y el rechazo dice que no lo son.

```python
est, gl, pval = ajuste_sdm.common_factor
print(f"factor común: W = {est:.4f}, gl = {gl:.0f}, p = {pval:.4f}")
print("SEM se rechaza" if pval < 0.05 else "SEM no se rechaza")
print(ajuste_slx.params.round(3))
```

La prueba necesita el bloque Durbin **completo**: con `durbin=` seleccionando un subconjunto estricto la restricción ya no anida al SEM, y `common_factor=True` lanza un error en vez de imprimir una afirmación sobre un modelo fuera del espacio paramétrico que se está contrastando. La constante queda siempre excluida del bloque Durbin: con una `W` estandarizada por filas y sin islas, `W1 = 1` hace la constante rezagada exactamente colineal con la constante. Y el estadístico sólo es *interpretable* donde `signo(θ_r) = −signo(ρ β_r)` para todo `r`; donde eso falla, `summary()` añade la advertencia en lugar de ocultar el número.

**Cuándo preferir `method='gmm'`.** MV es la opción por defecto y es eficiente bajo sus supuestos, pero esos supuestos son exigentes. Bajo heterocedasticidad de forma desconocida el estimador MV gaussiano de `ρ` es él mismo **inconsistente** (Lin y Lee 2010) —ningún sándwich rescata un error estándar del número equivocado—, y por eso `vcov='robust'` con `method='ml'` lanza un error en lugar de calcularlo en silencio. El GMM de Kelejian y Prucha son mínimos cuadrados en dos etapas espaciales con `[X, WX, …, W^q X]` como instrumentos; admite errores robustos a heterocedasticidad, no necesita log-determinante y es la vía a tomar cuando los residuos son claramente heterocedásticos o `n` es lo bastante grande como para que el jacobiano domine.

```python
ajuste_gmm = sar(y, X, W, method="gmm", vcov="robust")
print(ajuste_gmm.params.round(4))
print("rho admisible:", ajuste_gmm.rho_admissible, " intervalo:",
      tuple(round(b, 4) for b in ajuste_gmm.rho_bounds))
```

Cuatro rasgos de la vía GMM son límites deliberados, no descuidos:

- `vcov='robust'` es un sándwich de White alrededor de las condiciones de momento **homocedásticas** de Kelejian y Prucha. Corrige los errores estándar, no los momentos, y no vuelve el `λ` de GMM consistente bajo heterocedasticidad. El GMM robusto a heterocedasticidad de Kelejian y Prucha (2010) no está implementado.
- `sem(method='gmm')` reporta `λ` con `bse['lambda'] = NaN`, porque un error estándar honesto necesita esas mismas matrices `Ψ`. Los errores estándar de `β` siguen siendo asintóticamente válidos: el estimador espacial generalizado factible tiene la distribución límite del que conoce `λ`.
- Los MC2E de Kelejian y Prucha no están restringidos, así que en un diseño débilmente identificado pueden devolver un `ρ` **fuera** del intervalo admisible. Cuando ocurre, salta un `RuntimeWarning`, `rho_admissible` es `False`, `fitted` / `resid_reduced` / `pseudo_r2` son `NaN` en lugar de pasar por un multiplicador explosivo, y `.effects()` lanza un error.
- **No hay SARAR/SAC** en ningún punto del módulo: no existe el argumento `error_weights=`. Use `sdm`, que anida la restricción de factor común del SEM y es contrastable frente a él.

Dos advertencias más que conviene tener presentes. Un `ρ` estimado sobre una `W` binaria no es comparable con uno estimado sobre su versión estandarizada por filas: la normalización cambia la escala del parámetro (en una retícula de torre 6×6 una `W` binaria restringe `ρ` a unos `(−0,277, 0,277)`). Y `logdet='chebyshev'` es una *aproximación*, no una ruta exacta: en una retícula de torre 20×20 con orden 5 su error es ~2e-4 en `ρ = 0,3` pero ~0,66 en `ρ = 0,9`. Úselo sólo con `W` muy grandes y simetrizables, y nunca para una cifra principal.

---

## 7. Impactos, no coeficientes

Ésta es la magnitud peor leída de toda la econometría espacial aplicada. **Un `β` de un SAR no es un efecto marginal.** En `y = ρWy + Xβ + ε` la variable dependiente entra en su propio lado derecho, así que subir `x_r` en una unidad sube su `y`, lo cual sube la `y` de sus vecinos, lo cual retroalimenta. La matriz de derivadas del regresor `r`-ésimo es

$$S_r(W) = (I - \rho W)^{-1}\left(I \beta_r + W \theta_r\right),$$

y `β_r` no es ni su diagonal ni la suma de sus filas. LeSage y Pace (2009, sec. 2.7) la resumen en tres escalares: el impacto **directo** `tr(S_r)/n` (la propia unidad, bucles de retroalimentación incluidos), el impacto **total** `1'S_r 1/n` (la suma media por filas) y el impacto **indirecto** como su diferencia: la parte que se filtra a todos los demás.

```python
imp = ajuste_sar.effects(n_sim=1000)
print(imp.summary())
print(imp.to_frame()[["variable", "direct", "indirect", "total",
                      "total_lo", "total_hi"]].round(3))
print("beta real 0.800 | MCO", round(float(ajuste_ols.params["area_log"]), 3),
      "| beta SAR", round(float(ajuste_sar.params["area_log"]), 3),
      "| directo", round(float(imp.direct[0]), 3),
      "| total", round(float(imp.total[0]), 3))
```

Lea esa línea. La verdad es `0,8`; MCO, que absorbe todo el multiplicador espacial dentro de la pendiente, reporta alrededor de `1,31`; el `β` del SAR vuelve a estar cerca de `0,84`; el impacto *directo* queda algo por encima porque la retroalimentación que sale hacia los vecinos y vuelve forma parte de la respuesta de la propia unidad; y el impacto *total* —lo que en promedio movería una política aplicada en todas partes— ronda `1,93`. Reporte los tres, o reporte el total y dígalo. Presentar `β` a secas como «el efecto» es el error.

`n_sim` extrae `[β, θ, ρ]` de su distribución normal estimada y pasa cada extracción por la fórmula de impactos. Los errores estándar son, por tanto, una aproximación normal a la distribución de los **parámetros** propagada por una transformación no lineal —no un bootstrap de los datos— y los valores p reportados son valores p de simulación, `2·mín(proporción de extracciones > 0, proporción < 0)`, deliberadamente no `2Φ(−|efecto|/ee)`: con `ρ` cerca de un límite la distribución simulada es asimétrica y un valor p normal contradiría el intervalo impreso a su lado. Las extracciones cuyo `ρ` cae fuera del intervalo admisible se **descartan, no se truncan** (por encima del 5% descartado avisa, por encima del 50% lanza error). `n_sim=0`, el valor por defecto, devuelve estimaciones puntuales con errores estándar `NaN`, de modo que la llamada más simple siempre funciona.

Cuatro advertencias:

- **Un efecto indirecto es la *suma* de las `n(n−1)` derivadas cruzadas fuera de la diagonal dividida entre `n`**: el desbordamiento total medio que una unidad envía —equivalentemente, que recibe—. No es el efecto sobre un vecino concreto, ni el efecto sobre el primer anillo, ni la media de las derivadas cruzadas individuales: esa media es `n−1` veces más pequeña (48 en el grafo de los 48 estados contiguos, donde un impacto indirecto de `0,737456` corresponde a una derivada cruzada media de `0,015364`). No lo narre como «el desbordamiento sobre los estados adyacentes».
- **El impacto total medio no es `1/(1−ρ)`.** Ese atajo coincide con `1'A⁻¹1/n` sólo para una `W` estandarizada por filas y sin islas: es el primer par de números de abajo, donde ambos valores coinciden hasta la última cifra. En cuanto falla cualquiera de las dos condiciones, el atajo se rompe. Tome la versión *binaria* de ese mismo grafo de fronteras: su mayor autovalor es `5,4154`, de modo que `ρ` debe quedarse por debajo de `0,1847`, y con `ρ = 0,15` el valor exacto de `1'A⁻¹1/n` ronda `4,13` frente al `1,18` del atajo —el atajo se equivoca por un factor de tres y medio, y además se queda corto, no largo—. Una sola isla también lo rompe. Este módulo siempre calcula `1'A⁻¹1`.

```python
print("W estandarizada por filas y sin islas -- el atajo sí vale:")
print("  1'A^-1 1/n =", round(imp.s / imp.n, 6),
      " frente al atajo 1/(1-rho) =", round(1.0 / (1.0 - imp.rho), 6))

Wb = W.binary()                                   # mismo grafo, sin estandarizar por filas
rho_b = 0.15                                      # admisible: 1 / max autovalor(Wb) = 0.1847
Ab = np.eye(Wb.n) - rho_b * Wb.to_dense()
exacto_b = float(np.ones(Wb.n) @ np.linalg.solve(Ab, np.ones(Wb.n)) / Wb.n)
print("W binaria con rho = 0.15 -- el atajo no vale:")
print("  1'A^-1 1/n =", round(exacto_b, 6),
      " frente al atajo 1/(1-rho) =", round(1.0 / (1.0 - rho_b), 6))
print("Los impactos del SLX son exactos y locales:")
print(ajuste_slx.effects(n_sim=500).to_frame()[["variable", "direct", "indirect", "total"]].round(3))
```

- **El SEM no tiene descomposición de impactos**, y `.effects()` lanza un error para él: en `y = Xβ + u` el efecto marginal de `x_r` es `β_r` en la diagonal y cero fuera de ella, sea cual sea `λ`. Eso es una propiedad del modelo, no una carencia del software.
- **Los impactos del SLX son exactos y locales**: `directo = β`, `indirecto = θ·S₀/n`, sin multiplicador espacial y sin simulación para las estimaciones puntuales. Es un argumento real a favor del SLX cuando la teoría dice que los desbordamientos llegan a un solo anillo.

Una divergencia respecto de `spreg`, deliberada y documentada: `spat_impacts='full'` asigna todos los términos SLX a la columna *indirecta* y quita `θ_r tr(A⁻¹W)/n` del efecto directo; este módulo sigue el libro de texto y lo mantiene ahí. El **total** es idéntico en ambos casos, y para un SAR (`θ = 0`) los dos coinciden exactamente.

---

## 8. Paneles espaciales

`spatial_panel` estima SAR, SDM y SEM sobre un panel `(entidad, tiempo)` **balanceado** por cuasi-máxima verosimilitud, con cualquier combinación de efectos fijos individuales y temporales:

$$y_{it} = \rho \sum_j w_{ij} y_{jt} + x_{it}'\beta + (Wx)_{it}'\theta + \mu_i + \xi_t + \varepsilon_{it}.$$

Los efectos fijos son el problema. Los `μ_i` son `n` parámetros incidentales estimados cada uno con `T` observaciones y los `ξ_t` son `T` estimados cada uno con `n`; concentrarlos fuera de la verosimilitud gaussiana deja un score que no está centrado en el verdadero valor. Con `G = W(I − ρW)⁻¹` y efectos bidireccionales, `E[∂L/∂ρ] = −tr(G) − (T−1)/(1−ρ)` y `E[∂L/∂σ²] = −(n+T−1)/(2σ²)`, lo que sobre la información de orden `O(nT)` es un sesgo `O(1/T)` por los efectos individuales más un sesgo `O(1/n)` por los temporales. `β` no se ve afectado; `ρ` y `σ²` sí.

Lee y Yu (2010) los eliminan exactamente. `method='transformation'` (el valor por defecto) evalúa la verosimilitud sobre una transformación ortonormal cuyo score está centrado **por construcción**, de modo que `bias_correction` reporta `'none'` porque la corrección es demostrablemente nula, no porque se haya omitido. `method='direct'` concentra los efectos y aplica en su lugar la corrección analítica `θ̂ − I⁻¹a`; se conserva por comparabilidad con el software que lo hace así, y porque su jacobiano no exige ningún supuesto sobre las sumas por filas de `W`.

```python
from puremacro.spatial import spatial_panel

rng = np.random.default_rng(4)
T = 24
S_inv = np.linalg.inv(np.eye(n) - 0.4 * W.to_dense())             # rho real = 0.4
xs = rng.standard_normal((n, T))
ys = S_inv @ (0.9 * xs + rng.standard_normal((n, 1))              # mu_i
              + rng.standard_normal((1, T))                       # xi_t
              + 0.4 * rng.standard_normal((n, T)))
idx = pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"])
panel = pd.DataFrame({"pib": ys.ravel(), "gasto": xs.ravel()}, index=idx)

ajuste_tr = spatial_panel(panel, "pib", "gasto", W, model="sar", effects="two-way")
print(ajuste_tr.summary())
```

La tabla de impactos que imprime `spatial_panel` es la misma descomposición de LeSage y Pace de §7 (sus estimaciones puntuales están fijadas frente a `spatial_effects` hasta 1e-12), con una diferencia de convención: el panel reporta intervalos `punto ± z·ee` en vez de cuantiles empíricos de simulación. `impacts=None` —el valor por defecto— significa «sí para SAR/SDM, no para SEM»; pedir `impacts=True` con `model='sem'` lanza un error en lugar de devolver `None` en silencio.

Pasar los mismos datos por `method='direct'` muestra la corrección que el enfoque de transformación nunca necesita:

```python
ajuste_dir = spatial_panel(panel, "pib", "gasto", W, model="sar", effects="two-way",
                           method="direct", impacts=False)
print(ajuste_dir.bias.round(4))                 # lo que restó Lee-Yu
print("rho transformación:", round(ajuste_tr.rho, 4),
      " rho directo + Lee-Yu:", round(ajuste_dir.rho, 4))
```

**Para qué son válidos los errores estándar.** `vcov='oim'` (por defecto) es la inversa de la información esperada: válida bajo `ε` iid gaussianos, extendida por Lee (2004) a errores no normales pero todavía homocedásticos. `vcov='dk'` (Driscoll y Kraay) y `vcov='conley'` sustituyen **únicamente el bloque de `β`**: las filas y columnas de `ρ`/`λ` y `σ²` proceden siempre de la información esperada.

```python
ajuste_conley = spatial_panel(panel, "pib", "gasto", W, model="sar", effects="two-way",
                              vcov="conley", coords=coords, cutoff_km=800.0, impacts=False)
print(ajuste_conley.vcov_note)
print(pd.DataFrame({"oim": ajuste_tr.se, "conley": ajuste_conley.se}).round(4))
```

Esa restricción es deliberada, y aquí no existen `vcov='qml'` ni `vcov='cluster'` en absoluto: no es que no estén disponibles, es que son *incorrectos* para este score. Un producto exterior por celda desvirtúa gravemente el parámetro espacial: `s_it^ρ` lleva `(Wy)_it e_it`, y `Wy` carga sobre el `ε` de todos los vecinos, así que los términos cruzados en `i` que se omiten son grandes. En una retícula de torre 6×6 estandarizada por filas con `ρ = 0,5` con `T = 8` la «carne» del producto exterior vale 163,9 frente a una varianza verdadera del score de 303,5 —el 54% de la verdad, subestimando el error estándar de `ρ` en un 27%.

Y ninguna covarianza es robusta a una `W` mal especificada: si `W` está mal, `ρ̂` es inconsistente. Bajo `ε` heterocedásticos la **estimación puntual** del QMLE del SAR es inconsistente (Lin y Lee 2010), así que tampoco ahí hay sándwich que la rescate.

Restricciones con las que hay que contar:

- **Sólo paneles balanceados.** La transformación necesita un `T` común; un panel no balanceado lanza un error que nombra `puremacro.inference.balanced_panel.balanced_subpanel`.
- **Sólo estático.** Los paneles espaciales dinámicos (una `y` rezagada o `W y_{t−1}`) no están implementados; Yu, de Jong y Lee (2008) se cita por las asintóticas `n/T → c` que hay detrás del término de sesgo, no por estar implementado.
- **Sin efectos aleatorios**, sin prueba de Hausman, **sin SARAR/SAC** y **sin SDEM** (`model='sem'` con `durbin=` lanza un error en vez de estimar una ecuación que el módulo no define).
- **Espectro denso.** La matriz de información, el vector de sesgo y los impactos necesitan `tr(G)`, `tr(G²)`, `tr(G'G)` y `diag(G)`, que una factorización LU no proporciona y que una sonda estocástica volvería irreproducibles; por eso `logdet='lu'` lanza un error, salta un `RuntimeWarning` por encima de 2000 unidades y `dense_max = 5000` es un techo duro. Un panel de 3143 condados está cerca del techo; las secciones censales lo superan.
- Con un componente de **efectos temporales** y `method='transformation'`, todas las filas de `W` deben sumar uno: eso es lo que hace de `1` un autovector y permite que los efectos temporales desaparezcan en un SAR exacto de `n−1` unidades. El mensaje de error nombra `standardize()`.

---

## 9. Proyecciones locales espaciales

`spatial_lp` es una proyección local de panel con efectos fijos bidireccionales (Jordà 2005) ampliada con el shock rezagado espacialmente, de modo que una respuesta al impulso regional se separa en la respuesta al shock **propio** de la unidad y la respuesta al shock que reciben sus **vecinos**. Para cada horizonte `h`,

$$y_{i,t+h} - y_{i,t-1} = \mu_i^h + \tau_t^h + \beta_h x_{it} + \sum_p \gamma_{p,h}\,(W^{(p)}x)_{it} + (\text{rezagos}) + \varepsilon_{i,t+h},$$

donde `W^(p)` es la matriz de vecinos de orden `p` en el sentido de Anselin (1988): los caminos de exactamente `p` pasos, menos la diagonal y menos todos los órdenes inferiores, para que `γ₂` mida un segundo anillo en vez de contar dos veces el primero.

```python
from puremacro.spatial import spatial_lp

rng = np.random.default_rng(7)
T = 70
shock = rng.standard_normal((n, T))
dy = 0.6 * shock + 0.35 * (W.to_dense() @ shock) + 0.3 * rng.standard_normal((n, T))
idx = pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"])
panel_pl = pd.DataFrame({"empleo": np.cumsum(dy, axis=1).ravel(),
                         "gasto": shock.ravel()}, index=idx)

irf = spatial_lp(panel_pl, "empleo", "gasto", W, horizons=range(0, 7), n_lags=2,
                 cov_type="conley", coords=coords, cutoff_km=800.0)
print(irf.summary())
```

### Qué es `total`, y qué no es

`β_h` y `γ_{p,h}` se estiman **después** de una transformación intragrupo bidireccional, así que son respuestas *relativas a la media del periodo*. El efecto temporal `τ_t^h` absorbe, por construcción, todo componente de la respuesta que sea común a todas las unidades en un periodo, incluida la respuesta a la parte agregada del shock. Por tanto:

- `direct` = `β_h`: la respuesta adicional de una unidad cuyo shock propio está una unidad por encima de la media del periodo, con su vecindario fijo;
- `indirect` = `c₁ γ_{1,h}`: la respuesta adicional de una unidad cuyos vecinos del primer anillo reciben shocks `c₁` por encima de la media del periodo (una columna por orden —`indirect2`, `indirect3`, …— más `indirect_all` cuando hay más de uno);
- `total` = `direct + Σ_p c_p γ_{p,h}`: un **contraste transversal**, la respuesta de una unidad cuyo shock propio y cuyos shocks vecinos están ambos por encima de la media del periodo, frente a una unidad situada en esa media.

`c_p` es el tamaño del contraste de vecindad, y lo fija `total_scale=`. Con el valor por defecto `total_scale='row_sum'` es la suma media por filas de `W^(p)` —exactamente `1` para una `W` estandarizada por filas y sin islas, de modo que en ese caso `indirect = γ_{1,h}`—; `total_scale='unit'` fuerza `c_p = 1` en todos los órdenes, y un número o un número por orden los fija a mano. El vector realmente empleado, `c = (1, c₁, …, c_P)`, queda en `.total_scale`.

**`total` no es la respuesta a un shock unitario uniforme que golpea a todas las unidades, y este módulo nunca afirma que lo sea.** Un shock uniforme es exactamente la variación que `τ_t` elimina: tras la transformación intragrupo, la respuesta prevista a un `+1` uniforme es `γ(s_i − s̄)`, cuya media entre unidades es cero. Añadir un componente agregado arbitrario `κ·x̄_t` al proceso generador deja las estimaciones intactas hasta 1e-16, así que aquí no se identifica ningún multiplicador agregado. `total` coincide con la respuesta agregada sólo bajo el supuesto adicional, y no contrastable, de que la respuesta común es nula, que es justamente el argumento de Chodorow-Reich (2019). Escríbalo así en el artículo.

Además, éstos son coeficientes de forma reducida sobre **shocks** propios y vecinos. *No* son las derivadas parciales de LeSage y Pace de §7; en este estimador no hay ningún multiplicador `(I − ρW)⁻¹`.

### Inferencia, con honestidad

```python
print(irf.spillover[["h", "order", "gamma", "se", "corr_own"]].round(4).to_string(index=False))
print(irf.cumulative[["h", "cum_direct", "cum_indirect", "cum_total", "se_total"]].round(3).to_string(index=False))
```

La covarianza entre horizontes se conserva completa (`.vcov`), así que el error estándar de `direct + indirect` incorpora el término fuera de la diagonal y las bandas acumuladas son honestas en vez de una suma de varianzas por horizonte.

**La prueba conjunta de desbordamiento entre horizontes deliberadamente no se distribuye.** Una referencia χ² para la restricción apilada `γ_{p,h} = 0 en todo h` está gravemente sobredimensionada precisamente en los paneles a los que apunta este módulo: un Monte Carlo de 150 réplicas con `N = 20`, `T = 50`, `H = 6` y un `γ = 0` verdadero rechazó al **0,313** frente a un nominal de 0,10 (0,187 con nominal 0,05). Un Wald con muchas restricciones y una covarianza HAC estimada sobre `T ≈ 50` periodos no tiene su tamaño nominal, así que no se distribuye. Contraste un horizonte cada vez, o construya su propia referencia *fixed-b* a partir de `.vcov`.

La prueba z por horizonte que *sí* se distribuye tampoco es exacta, y el módulo reporta su tamaño en vez de una única cifra favorable. Sobre el mismo diseño (`N = 20`, `T = 50`, `h = 0…5`, `γ = 0` verdadero, 400 réplicas, error de Monte Carlo de 0,011 con nominal 0,05):

```text
cov_type           nominal 0.05                nominal 0.10
                  h = 0   peor sobre h      h = 0   peor sobre h
driscoll-kraay    0.083   0.110 (h = 3)     0.138   0.172 (h = 5)
conley 1000 km    0.068   0.100 (h = 3)     0.138   0.152 (h = 2)
cluster           0.090   0.090 (h = 0)     0.160   0.160 (h = 0)
```

Así que el contraste honesto con el 0,313 de la prueba conjunta es 0,08–0,11 con nominal 0,05, y el tamaño se desvía al alza con el horizonte a medida que crece la dependencia MA(h) inducida por las ventanas solapadas. Duplicar la sección cruzada a `N = 40` no arregla Driscoll y Kraay (es consistente cuando `T → ∞`, no en `N`); Conley sí mejora. No se aplica ninguna corrección de muestra finita en ningún punto —que es lo que hace que `spillover_orders=()` reproduzca `panel_lp` bit a bit—, así que todas las bandas aquí son asintóticas y medidamente demasiado estrechas en un panel regional. **Lea un `γ` cuyo valor p quede cerca del umbral como no concluyente.**

El `cov_type` por defecto es `'driscoll-kraay'` y no el `'cluster'` de `panel_lp`, deliberadamente: el conglomerado por entidad supone independencia entre entidades, algo que esta regresión niega por construcción, ya que `(Wx)_i` es función de los shocks de otras unidades. Para los paneles regionales con `T ≈ 40–100` a los que apunta el módulo, `'conley'` es la elección recomendada.

Dos notas más de identificación. `γ` se identifica exclusivamente a partir de la variación transversal de `Wx` dada `x`, así que un shock casi uniforme, una `W` densa o una `W` elegida después de mirar los resultados producen estimaciones débilmente identificadas o fruto de la búsqueda de especificación; `spillover['corr_own']` reporta la correlación muestral de las columnas propia y de desbordamiento tras la doble demediación, y por encima de 0,999 salta un `RuntimeWarning`. Y `x` entra como **exógena** condicional a los efectos fijos y a los rezagos: no se distribuye ninguna vía de variables instrumentales, aunque los diseños que este módulo respalda (Chodorow-Reich 2019; Auerbach, Gorodnichenko y Murphy 2020; Dupor et al. 2023) instrumenten el gasto regional con contratación militar o con exposición shift-share. Construya ese instrumento con `shift_share_iv` de §4 y trátelo como el shock.

---

## 10. Lista de comprobación práctica

- **Ancho de banda.** Reporte los errores de Conley con dos o tres radios. Si siguen creciendo con el radio, la correlación espacial no es local y la opción honesta es Driscoll-Kraay (`panel_lp_dk`) o un radio mayor.
- **Islas.** Una unidad sin vecinos tiene rezago espacial nulo; compruebe `W.n_islands` antes de la I de Moran y use `distance_weights` con un radio mayor o `knn_weights` si aparecen islas. Una isla también invalida el atajo `1/(1−ρ)` para el impacto total medio, por lo que §7 nunca lo usa.
- **Coordenadas.** Primero la latitud, después la longitud, en grados. `metric="euclidean"` trata las columnas como distancias planas en la misma unidad que el radio.
- **Shift-share.** El error AKM es válido cuando los shocks son tan buenos como aleatorios entre sectores; cuando la identificación proviene de las participaciones, siga a Goldsmith-Pinkham, Sorkin y Swift e inspeccione los pesos de Rotemberg.
- **Especifique antes de estimar.** Ejecute `ols_spatial` primero y lea `lm.recommendation`. Es una prueba local bajo la nula de ausencia de dependencia: dice a qué alternativa apuntan los residuos, no que el modelo indicado sea correcto. Dos rechazos robustos significan SDM o SARAR, no «escoja el estadístico mayor».
- **Nunca reporte un `β` de rezago espacial como un efecto.** Reporte `.effects()` —directo, indirecto y total— y diga de cuál está hablando. El efecto indirecto es la *suma* de las `n(n−1)` derivadas cruzadas fuera de la diagonal dividida entre `n` —el desbordamiento total medio que una unidad envía, equivalentemente recibe—, no el efecto sobre un vecino ni la media de las derivadas cruzadas individuales (que es `n−1` veces más pequeña). El SEM no admite esa descomposición, por construcción.
- **Diga qué `W` usa, y enseñe una segunda.** El `ρ` de una `W` binaria no es comparable con el de su versión estandarizada por filas, y ningún error estándar es robusto a una `W` mal especificada. Reporte pesos de contigüidad, de k vecinos más cercanos y económicos en paralelo.
- **La heterocedasticidad no es aquí un problema de errores estándar.** Bajo heterocedasticidad de forma desconocida la estimación MV gaussiana de `ρ` es ella misma inconsistente, y por eso `vcov='robust'` con `method='ml'` lanza un error. Reestime con `method='gmm'`.
- **Paneles: compruebe qué corrigió realmente.** `spatial_panel` usa por defecto la transformación de Lee y Yu, cuya corrección de sesgo es demostrablemente nula: `bias_correction='none'` ahí es una afirmación, no una omisión. Los paneles deben ser balanceados y estáticos, y caber por debajo de `dense_max = 5000` unidades. `vcov='dk'`/`'conley'` sustituyen sólo el bloque de `β`.
- **Proyecciones locales espaciales: `total` es un contraste, no un multiplicador.** El efecto temporal bidireccional absorbe exactamente la variación del shock uniforme, así que no se identifica ningún multiplicador agregado. Cite `total` como contraste transversal, use las pruebas por horizonte de una en una (la prueba conjunta entre horizontes no se distribuye porque su tamaño medido fue 0,31 frente a un nominal de 0,10) y trate un `γ` marginal como no concluyente.

## Referencias

- Adão, R., Kolesár, M. y Morales, E. (2019). Shift-share designs: theory and inference. *Quarterly Journal of Economics* 134(4), 1949–2010.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L., Bera, A. K., Florax, R. y Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics* 26(1), 77–104.
- Auerbach, A. J., Gorodnichenko, Y. y Murphy, D. (2020). Local fiscal multipliers and fiscal spillovers in the USA. *IMF Economic Review* 68, 195–229.
- Chodorow-Reich, G. (2019). Geographic cross-sectional fiscal spending multipliers: what have we learned? *American Economic Journal: Economic Policy* 11(2), 1–34.
- Cliff, A. D. y Ord, J. K. (1972). Testing for spatial autocorrelation among regression residuals. *Geographical Analysis* 4(3), 267–284.
- Cliff, A. D. y Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- Driscoll, J. C. y Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics* 80(4), 549–560.
- Dupor, B., Karabarbounis, M., Kudlyak, M. y Mehkari, M. S. (2023). Regional consumption responses and the aggregate fiscal multiplier. *Review of Economic Studies* 90(6), 2982–3021.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
- Goldsmith-Pinkham, P., Sorkin, I. y Swift, H. (2020). Bartik instruments: what, when, why, and how. *American Economic Review* 110(8), 2586–2624.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
- Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1), 161–182.
- Kelejian, H. H. y Prucha, I. R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics* 17(1), 99–121.
- Kelejian, H. H. y Prucha, I. R. (1999). A generalized moments estimator for the autoregressive parameter in a spatial model. *International Economic Review* 40(2), 509–533.
- Kelejian, H. H. y Prucha, I. R. (2010). Specification and estimation of spatial autoregressive models with autoregressive and heteroskedastic disturbances. *Journal of Econometrics* 157(1), 53–67.
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood estimators for spatial autoregressive models. *Econometrica* 72(6), 1899–1925.
- Lee, L.-F. y Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics* 154(2), 165–185.
- LeSage, J. P. y Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Lin, X. y Lee, L.-F. (2010). GMM estimation of spatial autoregressive models with unknown heteroskedasticity. *Journal of Econometrics* 157(1), 34–52.
- Ord, J. K. (1975). Estimation methods for models of spatial interaction. *Journal of the American Statistical Association* 70(349), 120–126.
- Pace, R. K. y LeSage, J. P. (2004). Chebyshev approximation of log-determinants of spatial weight matrices. *Computational Statistics & Data Analysis* 45(2), 179–196.
- Yu, J., de Jong, R. y Lee, L.-F. (2008). Quasi-maximum likelihood estimators for spatial dynamic panel data models with fixed effects when both n and T are large. *Journal of Econometrics* 146(1), 118–134.
