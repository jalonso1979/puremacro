> 🇬🇧 [English](../gvar.md) · 🇪🇸 Español

# VAR global (GVAR)

Treinta países con cuatro variables cada uno forman un sistema de 120 variables. Un VAR(2) sobre él tiene 28.800 coeficientes autorregresivos y quizá 150 observaciones trimestrales para estimarlos: el VAR global apilado no está mal estimado, es que no se puede estimar. El GVAR de Pesaran, Schuermann y Weiner (2004) recupera la interdependencia sin estimar nunca ese sistema. Cada país es un pequeño VARX\* en sus propias variables más un agregado *extranjero* construido con pesos de comercio, cada país se estima por separado, y el modelo global se **ensambla** después a partir de los bloques nacionales mediante álgebra matricial exacta, en lugar de ajustarse.

`puremacro.var.gvar` implementa esa secuencia: `star_variables` construye el bloque extranjero, `gvar` estima los modelos nacionales y resuelve el sistema de enlace en una sola llamada, y `solve_gvar` expone el paso puramente algebraico por separado, para poder leerlo y probarlo sin estimación alguna. Solo numpy/scipy/pandas, así que corre bajo Pyodide.

```text
from puremacro.var import gvar

res = gvar(panel, W, p=2, q=1)          # panel: frame (country, date) o {país: frame}
res.summary()                           # bloques nacionales, cond(G), estabilidad, exogeneidad débil
res.girf("US", "r", horizon=24, n_boot=500)     # respuestas generalizadas, bandas puntuales
res.gfevd(horizon=24); res.pp(b); res.forecast(8)
```

---

## 1. El modelo de cada país

El país `i` tiene `k_i` variables `x_it` y el agregado estrella `x*_it = Σ_j w_ij x_jt`, donde `w_ij` es la participación comercial de `i` con `j` (diagonal nula, filas que suman uno). Su VARX\*(p_i, q_i) es

$$x_{it} = a_{i0} + a_{i1}t + \sum_{l=1}^{p_i}\Phi_{il}x_{i,t-l} + \Lambda_{i0}x^{*}_{it} + \sum_{l=1}^{q_i}\Lambda_{il}x^{*}_{i,t-l} + \sum_{l=0}^{q_d}\Psi_{il}d_{t-l} + u_{it},$$

con `d_t` un bloque global estrictamente exógeno opcional (el caso canónico es el precio del petróleo). Todas las ecuaciones del país `i` comparten una misma matriz de regresores, de modo que la regresión aparentemente no relacionada colapsa a MCO ecuación por ecuación, y así lo hace `gvar`.

Dos supuestos distintos autorizan ese MCO, y el módulo los mantiene separados porque fallan por separado:

- **Exogeneidad débil** de `x*_it` para los parámetros del país `i` (Pesaran, Shin y Smith 2000): las innovaciones propias del país no deben retroalimentar su bloque extranjero dentro del periodo. `gvar` reporta un contraste aproximado para ello (§6).
- **Granularidad**, `max_j w_ij → 0` cuando crece `N`. Con pocos países un solo socio concentra un peso grande, `x*_it` está correlacionado contemporáneamente con `u_it` y el MCO nacional es *inconsistente*. Nada en el código puede reparar esto, y todos los ejemplos pequeños de esta página —seis unidades, pesos de hasta 0,69— son econométricamente incorrectos exactamente por ese motivo. Son pedagogía, no una estimación.

### 1.1 Construir la matriz de pesos y las variables estrella

Los pesos de comercio no son más que una matriz de flujos estandarizada por filas con diagonal nula, que es justo lo que produce `puremacro.spatial.economic_weights`; `gvar` acepta un `SpatialWeights`, un `DataFrame` etiquetado o un array `(N, N)` (con `country_order`). Aquí los «países» son seis estados de EE. UU. y los flujos son una aproximación gravitacional construida con la geografía del Censo que se distribuye con el paquete: superficie terrestre dividida por la distancia de círculo máximo entre los puntos internos de cada estado.

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import economic_weights, pairwise_distances

states = ["CA", "TX", "NY", "FL", "IL", "PA"]
geo = load_us_state_centroids().loc[states]
D = pairwise_distances(geo[["lat", "lon"]].to_numpy(), "haversine")     # km
size = geo["land_sqmi"].to_numpy()
flows = np.outer(size, size) / np.where(D > 0.0, D, 1.0)                # flujos gravitacionales
np.fill_diagonal(flows, 0.0)
W = economic_weights(pd.DataFrame(flows, index=states, columns=states)) # estandarizada por filas
Wf = pd.DataFrame(W.to_dense(), index=states, columns=states)
print(Wf.round(3))
print("mayor peso individual por fila:", Wf.max(axis=1).round(2).to_dict())
```

Esos pesos máximos van de 0,38 a 0,69. En un GVAR real estarían por debajo de 0,2; imprímalos siempre, porque son la condición de granularidad hecha visible.

`star_variables` construye el panel extranjero por sí solo, de modo que se puede graficar o alimentar otro estimador sin ajustar nada. El peso estrella del país `i` para la variable `v` se renormaliza sobre los países que realmente tienen `v`,

$$\tilde w_{ij}^{(v)} = \frac{w_{ij}\,\mathbf 1\{v \in K_j\}}{\sum_{m \neq i,\ v \in K_m} w_{im}},$$

de manera que la fila sigue sumando uno aunque algunos socios no tengan la variable.

```python
from puremacro.var.gvar import star_variables

rng = np.random.default_rng(11)
T, N = 160, len(states)
Wd = W.to_dense()
Y, R = np.zeros((T, N)), np.zeros((T, N))
for t in range(1, T):                       # un sistema VARX* estatal, por construcción
    ystar, rstar = Wd @ Y[t - 1], Wd @ R[t - 1]
    common = rng.standard_normal()          # un factor nacional en las innovaciones
    Y[t] = 0.55 * Y[t-1] + 0.25 * ystar - 0.08 * R[t-1] + 0.5 * common + rng.standard_normal(N)
    R[t] = 0.50 * R[t-1] + 0.20 * rstar + 0.10 * Y[t-1] + 0.3 * common + 0.7 * rng.standard_normal(N)
dates = pd.RangeIndex(T, name="date")
panel = {s: pd.DataFrame({"y": Y[:, i], "r": R[:, i]}, index=dates) for i, s in enumerate(states)}

xstar = star_variables(panel, W)             # frame (country, date), columnas y_star, r_star
print(xstar.loc["CA"].head(3).round(3))
```

El panel puede ser un frame con MultiIndex `(country, date)` o un diccionario `país -> frame`. Debe estar balanceado, sin huecos y ser numérico: un NaN interior siempre lanza error, y también lo hace una celda `(país, variable)` enteramente NaN salvo que se opte explícitamente por lo contrario (§8).

---

## 2. Estimar los bloques nacionales

```python
from puremacro.var import gvar

res = gvar(panel, W, p=1, q=1)
print(res.summary())
print(res.country_models["CA"].summary())
```

`p` y `q` aceptan un entero o un diccionario por país; `ic="aic"`/`"bic"` selecciona `(p_i, q_i)` para cada país sobre una muestra de rejilla común. Todos los países se ajustan sobre la *misma* muestra efectiva `t = t0 … T-1`, con `t0 = max(s, q_d)` y `s = max_i max(p_i, q_i)`. Eso no es configurable a propósito: es lo que alinea por fecha los vectores de residuos, hace de `Sigma_eps` una covarianza contemporánea genuina y vuelve exacta en cada fecha la identidad de enlace de más abajo.

`res.country_models[c]` es un `CountryVARX` con `coef`, `se`, `tstat`, `pvalue` (indexados por regresor, con las variables propias del país en columnas), las matrices `Phi`, `Lambda`, `Psi` ya troceadas, `resid`, `fitted`, `r2`, `aic`/`bic` y `condition_number`. `res.coef_frame()` aplana todos los países en una única tabla larga.

Dos advertencias que el resumen imprime y que no conviene saltarse. Los estadísticos `t` son condicionales a la exogeneidad débil y, con datos en niveles, **no son asintóticamente normales** (Sims, Stock y Watson 1990): con variables I(1) el coeficiente de un nivel rezagado tiene un límite no estándar. Y `trend='ct'` introduce una tendencia lineal irrestricta en un VARX\* en niveles, lo que implica una tendencia *cuadrática* en los niveles del sistema global resuelto; los GVAR de tipo VECX\* restringen la tendencia al espacio de cointegración, y este módulo no tiene espacio de cointegración al que restringirla.

---

## 3. Resolver el sistema global

Este es el paso que lo convierte en un modelo *global*, y es álgebra exacta, no estimación. Ordene el vector global `x_t = (x'_1t, …, x'_Nt)'`, `k = Σ_i k_i`. Sea `S_i` (`k_i × k`) la matriz que selecciona las variables propias del país `i` y `W*_i` (`k*_i × k`) la que construye sus variables estrella, de modo que

$$z_{it} = \begin{pmatrix}x_{it}\\ x^{*}_{it}\end{pmatrix} = W_i x_t, \qquad W_i = \begin{bmatrix}S_i\\ W^{*}_i\end{bmatrix}.$$

Escribiendo `A_i0 = [I, −Λ_i0]` y `A_il = [Φ_il, Λ_il]`, rellenados con ceros hasta `s`, cada ecuación nacional es `A_i0 W_i x_t = a_i0 + a_i1 t + Σ_l A_il W_i x_{t−l} + u_it`, y apilando sobre `i` se obtiene un único sistema cuadrado

$$G x_t = a_0 + a_1 t + \sum_{l=1}^{s} H_l x_{t-l} + \sum_l \Upsilon_l d_{t-l} + \varepsilon_t,$$

cuyo bloque de filas `i` es `G_i = S_i − Λ_i0 W*_i` y `H_{l,i} = Φ_il S_i + Λ_il W*_i`. Cuando `G` es no singular, `x_t = G^{-1}(...)` es un VAR(s) ordinario con matrices `F_l = G^{-1} H_l` e innovaciones en forma reducida `G^{-1} ε_t`. `res.G`, `res.H`, `res.F`, `res.G_inv`, `res.eigenvalues`, `res.max_eigenvalue` y `res.stable` están todos en el resultado.

`solve_gvar` expone el paso por separado. Dos países de una sola variable hacen visible todo el mecanismo, y también cómo falla:

```python
from puremacro.var.gvar import solve_gvar

S = {"A": np.array([[1.0, 0.0]]), "B": np.array([[0.0, 1.0]])}
Wstar = {"A": np.array([[0.0, 1.0]]), "B": np.array([[1.0, 0.0]])}
link = {c: np.vstack([S[c], Wstar[c]]) for c in ("A", "B")}
A0 = {"A": np.array([[1.0, -0.3]]), "B": np.array([[1.0, -0.2]])}       # [I, -Lambda_0]
A_lags = {"A": [np.array([[0.5, 0.1]])], "B": [np.array([[0.4, 0.0]])]} # [Phi_1, Lambda_1]
sol = solve_gvar(link, A0, A_lags, country_order=["A", "B"], block_sizes={"A": 1, "B": 1})
print(sol.G.round(3), sol.F[0].round(3), sep="\n")
print("max |autovalor| =", round(float(np.abs(np.linalg.eigvals(sol.F[0])).max()), 4))

A0_bad = {"A": np.array([[1.0, -2.0]]), "B": np.array([[1.0, -0.5]])}   # l1 * l2 = 1
try:
    solve_gvar(link, A0_bad, A_lags, country_order=["A", "B"], block_sizes={"A": 1, "B": 1})
except np.linalg.LinAlgError as exc:
    print(str(exc)[:118], "...")
```

Con una variable por país, `G = [[1, −λ₁], [−λ₂, 1]]` es singular exactamente cuando `λ₁λ₂ = 1`: el sistema contemporáneo entre países `x_t = Λ_0 W* x_t + …` no tiene entonces solución única. El error nombra los bloques de filas y de columnas que más cargan sobre el espacio nulo, que es como se localiza al culpable en un modelo de 30 países.

**Lea `condition_number`, nunca `condition_number_raw`.** Bajo un reescalado diagonal de los datos, `G → D G D^{-1}`, una transformación de semejanza que deja intactos los autovalores y el determinante pero puede mover el número de condición bruto once órdenes de magnitud: exprese una serie en millones y un modelo bien planteado parecerá singular. Por eso `G` se equilibra (balanceo de LAPACK más barridos de Ruiz) antes de la comprobación de condicionamiento, `res.condition_number` es ese número invariante a las unidades y `condition_number_raw` se conserva solo como referencia. Las matrices de diseño nacionales se equilibran por columnas antes de su propia comprobación de singularidad, por la misma razón, y el escalado se deshace exactamente en los coeficientes, los errores estándar y `(Z'Z)^{-1}`.

**La inestabilidad es un aviso, no un error.** Un GVAR en niveles I(1) lleva raíces unitarias por construcción, así que `max |autovalor| ≥ 1` es lo esperado y `gvar` solo advierte (`stability_warn=False` lo silencia). Ahora bien, las bandas bootstrap extraídas de un sistema resuelto explosivo no son de fiar.

---

## 4. Respuestas al impulso generalizadas

Con `Ψ_h` los coeficientes MA del VAR(s) resuelto y `R_h = Ψ_h G^{-1}`, la respuesta a una innovación de una desviación típica en la combinación `d` de las ecuaciones nacionales es

$$\mathrm{GIRF}_h(d) = \frac{R_h \Sigma_\varepsilon d}{\sqrt{d'\Sigma_\varepsilon d}}.$$

*Generalizadas*, no ortogonalizadas, y eso es una decisión, no una comodidad. Una factorización de Cholesky de `Σ_ε` exige un orden de las `k` variables país-variable; nadie puede defender uno, y la respuesta cambia con él. La respuesta generalizada es invariante al orden porque nunca impone ninguno, pero el precio es que se trata de una esperanza condicional dada una innovación con las demás en su media condicional, **no** del efecto de un shock estructural independiente. Las respuestas a distintos pares `(país, variable)` no descomponen nada. Este módulo no ofrece otra vía: ninguna identificación del sistema global por Cholesky, signos o restricciones narrativas.

```python
g = res.girf("TX", "y", horizon=16, n_boot=200, ci=0.90, seed=0)
print(g.summary())

gc = res.girf_combination({("TX", "r"): 1.0, ("CA", "r"): 1.0},
                          label="innovación conjunta en la tasa de TX y CA", horizon=12)
print(gc.to_frame().query("h == 0").head(4).round(4))
```

`girf` golpea una variable de un país; `girf_combination` recibe un diccionario de `(país, variable)` (o `"país:variable"`) con pesos. Por la normalización `sqrt(d' Σ_ε d)`, la trayectoria es invariante a la escala positiva de `d`: `1.0` y `3.0` dan respuestas idénticas. `GVARGIRFResult` lleva `irf` `(H+1, k)`, `lower`/`upper`/`median`, `names`, `shock_sd`, y `plot(by_variable=True)` dibuja la figura estándar del GVAR: un panel por variable, una línea por país.

Las bandas son **bandas percentiles puntuales de un bootstrap recursivo de residuos**: nunca conjuntas a lo largo de los horizontes, y sin corrección de sesgo, razón por la cual `median` (la mediana bootstrap) difiere de `irf` (la estimación puntual) en el sesgo de MCO. Cada réplica remuestrea filas completas de la matriz de residuos apilada (`bootstrap='iid'`) o les cambia el signo (`'wild'`), simula hacia adelante el sistema resuelto, reconstruye el panel estrella con los *mismos* pesos, reestima cada bloque nacional con los órdenes de rezago **fijos** y vuelve a resolver. Las réplicas que explotan o no se pueden estimar se descartan y se cuentan en `n_boot_failed`; si se pierde más del 10 % se emite una advertencia y la banda superviviente está sesgada hacia dentro.

Una trampa de escala silenciosa: `sigma_ddof` (por defecto 0, el divisor MLE que usa la literatura GVAR) no es cosmético. La GIRF es homogénea de grado ½ en `Σ_ε`, así que cambiar el divisor multiplica todas las respuestas por `sqrt(T_eff / (T_eff − sigma_ddof))`.

---

## 5. GFEVD, perfiles de persistencia y pronósticos

```python
theta = res.gfevd(horizon=8)                       # (H+1, k, k), filas que suman 1
raw = res.gfevd(horizon=8, normalize=False)
print("sumas de fila sin normalizar, h = 0:", raw[0].sum(axis=1).round(3))
print("sumas de fila sin normalizar, h = 8:", raw[8].sum(axis=1).round(3))

pp = res.pp({("TX", "y"): 1.0, ("CA", "y"): -1.0}, horizon=20)
print("perfil de persistencia:", pp[:5].round(3), "->", round(float(pp[-1]), 4))

print(res.forecast(4).round(3).iloc[:, :4])
```

**La FEVD generalizada.** `gfevd` es Pesaran-Shin (1998) con `Ψ_l` sustituido por `R_l` y `Σ` por `Σ_ε`. Las participaciones son no negativas y, con el valor por defecto `normalize=True` (convención de Diebold-Yılmaz), cada fila suma exactamente uno. Las filas sin normalizar son harina de otro costal, y el bloque anterior imprime valores por debajo de uno ya en el impacto:

> **La cota `≥ 1` de Pesaran-Shin sobre la suma de fila sin normalizar no se cumple en un GVAR, ni siquiera en `h = 0`.** Su demostración necesita `Ψ_0 = I`, para que la fila `i` en el impacto sea `Σ_j r_ij² ≥ r_ii² = 1`. Un GVAR tiene en cambio `R_0 = G^{-1}`, y la suma de fila en el impacto es un cociente de Rayleigh de `C C'`, con `C` la matriz `Σ_ε^{1/2}` normalizada por columnas, de modo que lo único que puede afirmarse es
>
> $$\lambda_{\min}(\mathrm{corr}(\Sigma_\varepsilon)) \le \mathrm{suma\ fila}_i(h=0) \le \lambda_{\max}(\mathrm{corr}(\Sigma_\varepsilon)).$$
>
> La cota `≥ 1` solo reaparece cuando `G = I`, es decir, sin bloque estrella contemporáneo en ningún país. No es un caso raro: en el propio DGP de prueba de tres países del módulo, variando únicamente la semilla, 50 de 59 ajustes tienen una suma de fila en el impacto inferior a uno, y para `h ≥ 1` aparecen valores cercanos a 0,28 en sistemas corrientes de tres variables. **Nunca lea una suma de fila sin normalizar como una comprobación de completitud.**

**Perfiles de persistencia.** `pp(b, horizon=…)` es Pesaran y Shin (1996),

$$\mathrm{PP}(b, h) = \frac{b' R_h \Sigma_\varepsilon R_h' b}{b' R_0 \Sigma_\varepsilon R_0' b},$$

exactamente 1 en `h = 0` por construcción. Un perfil que decae a cero señala una combinación asintóticamente estacionaria —cointegrante—; uno que converge a una constante positiva señala una combinación I(1). `b` puede darse como un array de longitud `k` o como un diccionario `(país, variable) -> peso`.

**Pronóstico.** `forecast(steps)` itera hacia adelante la media condicional del sistema resuelto desde el final de la muestra; es una trayectoria puntual, no una densidad, y devuelve un frame indexado `1..steps`. Si el modelo se ajustó con `exog`, hace falta un `exog_future` acorde: el sistema resuelto *condiciona* sobre `d_t` y nunca lo golpea.

**Vigile `T_eff` frente a `k`.** `Σ_ε` es una matriz `k × k` estimada con `T_eff` observaciones, y el GVAR canónico tiene `k ≫ T_eff`, en cuyo caso es singular. `gvar` advierte en `T_eff < 2k` y de nuevo en `T_eff < k` en vez de regularizar en silencio: aquí no hay ningún estimador de encogimiento. Las GIRF se siguen evaluando con una `Σ_ε` singular, pero la GFEVD y las bandas bootstrap no significan gran cosa. El ejemplo de seis estados de arriba tiene `k = 12` y `T_eff = 159`, holgadamente a salvo.

---

## 6. El contraste de exogeneidad débil: qué es y qué no

```python
we = res.weak_exogeneity
print(we.summary())
print(we.table.sort_values("p_value").head(4).round(4))
```

Para cada variable estrella del país `i` se estima la regresión auxiliar (en primeras diferencias por defecto, porque las entradas canónicas de un GVAR son I(1) y el estadístico F necesita regresores estacionarios)

$$\Delta x^{*}_{il,t} = \mu + \sum_a \psi_a' \Delta x^{*}_{i,t-a} + \sum_a \phi_a' \Delta x_{i,t-a} + \gamma' \hat u_{i,t-1} + \eta_t$$

y se contrasta `γ = 0` con `F(k_i, T_e − m)`. Entra el bloque estrella rezagado completo, como en Dees et al. (2007, ec. 15). Ahora las salvedades, que son el fondo de esta sección:

- Es un contraste **aproximado en forma reducida**, **no** el contraste de corrección de error de Dees et al. (2007, ec. 15). Aquel usa los términos cointegrantes estimados de un VECX\*; este módulo no tiene capa de cointegración, de modo que el residuo rezagado del propio modelo nacional, `û_{i,t−1}`, hace las veces de ellos. La hipótesis nula aquí es la condición necesaria de que las innovaciones del modelo del país `i` no causen a la Granger su propio bloque extranjero.
- `û_{i,t−1}` es un **regresor generado y no se aplica ninguna corrección por regresor generado**, así que la distribución F es aproximada en un segundo sentido, independiente del primero.
- **Tamaño medido, revelado en vez de omitido.** Bajo la nula (tres países independientes, `p = q = 1`, 1000 réplicas, 6000 contrastes por celda) la tasa de rechazo no es la nominal. En niveles I(1) —la entrada canónica de un GVAR— es 0,027/0,109/0,178 a niveles nominales 0,01/0,05/0,10 con `T = 200`, y 0,035/0,114/0,182 con `T = 300`. La distorsión es *estable en T*, lo que la identifica como el problema no tratado del regresor generado y no como un artefacto de muestra pequeña. Con datos estacionarios (AR(1), ρ = 0,5, `T = 200`) resulta en cambio conservador: 0,002 a nivel nominal 0,05.

Por eso el resultado se publica como una **ordenación diagnóstica de p-valores, no como una regla de decisión calibrada**. No hay atributo `expected_rejections`, y la columna booleana se llama `reject_nominal` para dejar claro que `alpha` es un umbral nominal y no el tamaño del contraste. Ordene por `p_value`, o por `p_holm` para una ordenación por familia, y tome un p-valor pequeño como motivo para mirar el bloque estrella de ese país. No cuente rechazos ni compare `n_reject_nominal` con `alpha × n_tests`.

---

## 7. Lista de comprobación práctica

- **Construya los pesos a partir de flujos, no de la geografía, cuando pueda.** `economic_weights(matriz_comercio)` estandariza por filas y anula la diagonal, que es exactamente lo que `gvar` valida: no negativos, diagonal nula, filas que suman uno con tolerancia 1e-8. Solo pesos fijos: un diccionario de frames o un array tridimensional lanzan error en vez de promediar en silencio.
- **Imprima el mayor peso de cada fila.** Ese número *es* la condición de granularidad. Por encima de 0,2 aproximadamente, el MCO nacional es materialmente inconsistente y los errores estándar subestiman el daño.
- **¿Cuánto `T` hace falta?** El país `i` necesita `T_eff − m_i ≥ k_i + 1` solo para tener una `Σ_i` no singular (por debajo, `gvar` lanza error nombrando al país); el resumen marca los casos con menos de `k_i + 5` grados de libertad residuales. Para los objetos *globales* la restricción que ata es `Σ_ε`: apunte a `T_eff > 2k`, y sepa que el GVAR canónico con `k ≫ T_eff` reporta GFEVD y bandas a partir de una covarianza singular.
- **Cuando `G` está mal condicionada.** Mire primero `condition_number` (equilibrado), no el bruto. Después restrinja el bloque estrella contemporáneo —`contemporaneous_star={"US": ("po",)}` es como el modelo canónico excluye la `y*` extranjera contemporánea de las ecuaciones de EE. UU.— y revise si alguna fila de pesos se concentra en un solo socio. `cond_tol` (por defecto 1e12) lanza error en vez de devolver un sistema resuelto inservible.
- **Órdenes de rezago.** El valor por defecto `p = 2, q = 1` es la elección canónica del GVAR. `ic=` selecciona por país, pero los criterios son comparables **solo dentro de una misma rejilla**: cambiar `max_p` cambia la muestra de la rejilla y con ella la verosimilitud de cada candidato.
- **Nada se ignora en silencio.** `max_p`/`max_q` sin `ic`, `exog_lags` sin `exog`, `we_lags`/`alpha` con `weak_exogeneity=False`, `ci`/`seed`/`bootstrap` con `n_boot=0`: cada uno lanza error en lugar de no hacer nada.

```python
res_ic = gvar(panel, W, ic="bic", max_p=2, max_q=1,
              contemporaneous_star={"CA": ("r",), "TX": ("r",)})
print(res_ic.lag_orders)
print(res_ic.country_models["CA"].contemporaneous_star,
      res_ic.country_models["NY"].contemporaneous_star)
print(f"cond(G) = {res_ic.condition_number:.3f}, max |autovalor| = {res_ic.max_eigenvalue:.3f}")
```

---

## 8. Deliberadamente fuera de alcance

Todo lo que sigue es una decisión, no una carencia. Busque otra herramienta en vez de esperar una versión futura.

- **Sin capa VECX\*/cointegración.** Los modelos nacionales son VARX\* irrestrictos en niveles: sin contraste de rango, sin tendencia restringida, sin términos de corrección de error; y, en consecuencia, el contraste de exogeneidad débil de la §6 no es el de Dees et al. (2007, ec. 15).
- **Sin conjuntos de variables por país, por defecto.** Todos los países deben llevar todas las columnas de `data`; una celda `(país, variable)` enteramente NaN lanza error nombrando ambos, igual que un NaN interior, porque la causa más frecuente es una serie mal nombrada o eliminada por accidente, que si no estimaría en silencio un modelo distinto. La asimetría país petrolero / EE. UU. requiere optar explícitamente: `allow_missing_variables=True` restaura la lectura permisiva y emite un `RuntimeWarning` nombrando cada celda descartada. Un bloque *estrella* sí puede nombrar cualquier variable que tenga otro país, incluso una que el país `i` no tenga. Las *definiciones* genuinamente distintas por país quedan fuera de alcance.
- **Una variable endógena en un país no puede ser exógena en otro.** Un nombre que aparezca en `data` y en `exog` lanza error: el sistema resuelto condicionaría sobre una variable endógena a uno de sus propios bloques.
- **Sin identificación estructural del sistema global** más allá de la vía generalizada: ni Cholesky, ni restricciones de signo, ni narrativas sobre el GVAR.
- **Sin pesos de comercio variables en el tiempo (móviles) ni estimación con ventana móvil.**
- **Sin shocks al bloque exógeno global `d_t`;** el sistema resuelto condiciona sobre él.
- **Sin GVAR con unidad dominante o aumentado por factores** (Chudik y Pesaran 2011).
- **Sin contrastes de cambio estructural** en las ecuaciones nacionales.
- **Sin encogimiento para `Σ_ε`.** El módulo advierte en `T_eff < 2k` y `T_eff < k` en vez de regularizar en silencio una covarianza singular.
- **Las bandas bootstrap son bandas percentiles puntuales**, nunca conjuntas a lo largo de los horizontes, y no llevan corrección de sesgo.

## Referencias

- Chudik, A. y Pesaran, M. H. (2011). Infinite-dimensional VARs and factor models. *Journal of Econometrics* 163(1), 4–22.
- Chudik, A. y Pesaran, M. H. (2016). Theory and practice of GVAR modelling. *Journal of Economic Surveys* 30(1), 165–197.
- Dees, S., di Mauro, F., Pesaran, M. H. y Smith, L. V. (2007). Exploring the international linkages of the euro area: a global VAR analysis. *Journal of Applied Econometrics* 22(1), 1–38.
- Pesaran, M. H., Schuermann, T. y Weiner, S. M. (2004). Modeling regional interdependencies using a global error-correcting macroeconometric model. *Journal of Business & Economic Statistics* 22(2), 129–162.
- Pesaran, M. H. y Shin, Y. (1996). Cointegration and speed of convergence to equilibrium. *Journal of Econometrics* 71(1–2), 117–143.
- Pesaran, M. H. y Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters* 58(1), 17–29.
- Pesaran, M. H., Shin, Y. y Smith, R. J. (2000). Structural analysis of vector error correction models with exogenous I(1) variables. *Journal of Econometrics* 97(2), 293–343.
- Sims, C. A., Stock, J. H. y Watson, M. W. (1990). Inference in linear time series models with some unit roots. *Econometrica* 58(1), 113–144.
