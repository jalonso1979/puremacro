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

## 5. Lista de comprobación práctica

- **Ancho de banda.** Reporte los errores de Conley con dos o tres radios. Si siguen creciendo con el radio, la correlación espacial no es local y la opción honesta es Driscoll-Kraay (`panel_lp_dk`) o un radio mayor.
- **Islas.** Una unidad sin vecinos tiene rezago espacial nulo; compruebe `W.n_islands` antes de la I de Moran y use `distance_weights` con un radio mayor o `knn_weights` si aparecen islas.
- **Coordenadas.** Primero la latitud, después la longitud, en grados. `metric="euclidean"` trata las columnas como distancias planas en la misma unidad que el radio.
- **Shift-share.** El error AKM es válido cuando los shocks son tan buenos como aleatorios entre sectores; cuando la identificación proviene de las participaciones, siga a Goldsmith-Pinkham, Sorkin y Swift e inspeccione los pesos de Rotemberg.

## Referencias

- Adão, R., Kolesár, M. y Morales, E. (2019). Shift-share designs: theory and inference. *Quarterly Journal of Economics* 134(4), 1949–2010.
- Cliff, A. D. y Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- Goldsmith-Pinkham, P., Sorkin, I. y Swift, H. (2020). Bartik instruments: what, when, why, and how. *American Economic Review* 110(8), 2586–2624.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
