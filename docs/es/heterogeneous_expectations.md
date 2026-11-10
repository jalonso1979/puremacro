> 🇬🇧 [English](../heterogeneous_expectations.md) · 🇪🇸 Español

# Información heterogénea y pronósticos neokeynesianos

`puremacro.expectations` representa **conjuntos de información gaussianos con
fechas explícitas**. Distintos grupos reciben señales diferentes, conservan
observaciones anteriores, actualizan creencias y pronostican resultados
económicos lineales. `models.information_nk` aplica estos operadores a un
sistema neokeynesiano finito de valores presentes.

La estructura informativa es una **restricción de procesamiento** explícita:
los agentes utilizan únicamente las señales declaradas. Los precios de
mercado, el ingreso y las observaciones de otros grupos no entran automáticamente
en su información. La implementación resuelve el sistema de pronósticos
condicionales bajo esa restricción. No resuelve aprendizaje endógeno de mercado,
atención óptima ni un equilibrio microeconómico completo de información.
Sus trayectorias de primer orden no identifican bienestar. Estas diferencias
importan al interpretar los resultados de política monetaria.

## Un experimento con información fechada

El estado primitivo contiene la noticia de política y dos ruidos grupales
independientes. El grupo A recibe una señal más precisa que B en la fecha cero.
Ambos observan exactamente la noticia cuando cambia la tasa en la fecha 8.

```python
import numpy as np
from puremacro.expectations import (
    GaussianSignal, build_gaussian_information, audit_gaussian_information,
)
from puremacro.models import solve_information_nk, audit_information_nk

signals = (
    (GaussianSignal(0, [1., .5, 0.], "initial"),
     GaussianSignal(8, [1., 0., 0.], "reveal")),
    (GaussianSignal(0, [1., 0., 2.], "initial"),
     GaussianSignal(8, [1., 0., 0.], "reveal")),
)
info = build_gaussian_information(
    np.eye(3), signals, horizon=12, weights=[.5, .5],
    group_names=("A", "B"), state_names=("news", "noise_A", "noise_B"),
)
rate_loadings = np.zeros((12, 3))
rate_loadings[8, 0] = -.0025
solution = solve_information_nk(info, rate_loadings)
assert audit_gaussian_information(info)["accepted"]
assert solution.converged and audit_information_nk(solution)["accepted"]
path = solution.evaluate([1., 0., 0.])
print(path.to_frame().head())
print(path.forecast_frame().head())
print(info.belief_frame([1., 0., 0.]).head())
```

Las tasas y la inflación son desviaciones por período. Una variación trimestral
de −.0025 significa −25 puntos básicos trimestrales. El producto es una
desviación logarítmica; multiplicar por 100 aproxima localmente un porcentaje,
no una variación porcentual exacta del nivel. La realización anterior fija los
ruidos grupales en cero. En este ejemplo lineal con ruidos independientes,
también coincide con la respuesta media condicional a una noticia de política
igual a uno. No es la media incondicional del experimento gaussiano de media cero.

## Qué significa una señal

Sea $z\sim N(0,\Sigma)$ el vector de **todos** los choques primitivos, incluido
el ruido de las señales. `GaussianSignal(date, loading, label)` revela $hz$
a su grupo en la fecha indicada. La función no agrega otro ruido aleatorio.
Un componente primitivo separado representa una nueva medición independiente;
reutilizar la misma carga representa la misma observación, aunque cambien su
etiqueta o fecha.

Los grupos comparten las observaciones internamente. El ruido específico de
un grupo es común dentro de él y no tiene por qué promediar cero en una
realización. La covarianza admite correlación entre componentes. Su dimensión
debe coincidir con cada vector de cargas. Los pesos grupales son no negativos
y suman uno; se permiten pesos cero. Las fechas son enteros entre cero y
`horizon-1`.

Apile en $H_{gt}$ las filas de las señales ya recibidas por cada grupo. El
mapa de medias condicionales y la covarianza posterior son

$$
P_{gt}=\Sigma H_{gt}'(H_{gt}\Sigma H_{gt}')^+H_{gt},\qquad
E[z\mid\mathcal I_{gt}]=P_{gt}z,\qquad
V_{gt}=\Sigma-P_{gt}\Sigma.
$$

Para un resultado $az$, `info.forecast(a, z, date=t, group=g)` devuelve
$aP_{gt}z$. `beliefs` devuelve medias condicionales con ejes fecha × grupo ×
componente. `belief_frame` informa medias y desviaciones estándar posteriores.
Sin observaciones, la media es cero y la covarianza es la previa; observar
todo el estado produce $P=I$ y varianza condicional cero.

La covarianza previa debe ser definida positiva y tener número de condición
no mayor que $10^{12}$. El constructor usa transformación de Cholesky,
normalización de filas observadas y SVD. Registra la redundancia exacta en
el rango. Rechaza valores singulares relativos en $[10^{-13},10^{-11}]$ por
ambigüedad alrededor del umbral $10^{-12}$. No agrega regularización diagonal.
Los arreglos densos de proyecciones y covarianzas tienen un límite de dos
millones de entradas cada uno, y el total de señales no puede superar 4,096.

La auditoría independiente comprueba rangos, señales observadas, idempotencia,
ortogonalidad de covarianzas, positividad posterior e información anidada.
Las expectativas condicionales individuales cumplen la ley de expectativas
iteradas. El mapa promedio $\bar P_t=\sum_gw_gP_{gt}$ generalmente **no**:
$\bar P_t\bar P_s$ puede diferir de $\bar P_t$, aun con información individual
anidada.

## La aplicación monetaria finita

Los pesos de consumidores provienen de `info.weights`; los `firm_weights`
opcionales pueden ser distintos. Hay una trayectoria exógena de tasa nominal
$i_t=R_tz$, desviaciones de tasa natural nulas y continuación exactamente
cero para producto, inflación y tasas desde $T$. El horizonte de entrada debe
incluir todo el soporte temporal de la política. El modelo no tiene regla
de Taylor ni elección endógena de política.

Los valores predeterminados son descuento $\beta=.99$, elasticidad intertemporal
$s=1$, persistencia de Calvo $\theta=.75$ e inversa de Frisch $\epsilon=1$.
La pendiente de Phillips implícita es
$\kappa=(1-\theta)(1-\beta\theta)(\epsilon+1/s)/\theta$.
Todos estos valores son supuestos ilustrativos.

Los valores presentes de demanda y precios siguen la distinción de
[Angeletos y Lian (2018), ecuaciones 5–6](https://economics.mit.edu/sites/default/files/publications/Angeletos%20Lian%20(AER%202018).pdf).
La frontera finita y las señales mantenidas son restricciones explícitas de
esta implementación; no se reproduce el equilibrio de señales endógenas de
mercado del artículo. Con promedios de consumidores y empresas
$\bar E^c_t$ y $\bar E^f_t$:

$$
y_t=-s\sum_{j=t}^{T-1}\beta^{j-t}\bar E^c_t[i_j-\pi_{j+1}]
 +(1-\beta)\sum_{j=t+1}^{T-1}\beta^{j-t-1}\bar E^c_t[y_j],
$$

$$
\pi_t=\kappa y_t+\sum_{j=t+1}^{T-1}(\beta\theta)^{j-t}
 \left(\kappa\bar E^f_t[y_j]+\frac{1-\theta}{\theta}\bar E^f_t[\pi_j]\right).
$$

El código resuelve las cargas de resultados hacia atrás mediante acumuladores
de sumas descontadas. `audit_information_nk` reconstruye las sumas explícitas
para cada dirección primitiva de forma independiente. La aceptación exige
errores informativos y de coeficientes no mayores que $10^{-9}$ por defecto.
El resultado conserva coeficientes, parámetros, identificadores de información
y modelo, y diagnósticos. `evaluate` devuelve trayectorias realizadas y
pronósticos grupales a una fecha del producto, la inflación y la tasa nominal;
los pronósticos posteriores a la última fecha son cero. Para otro horizonte,
aplique `info.forecast` a la carga del resultado en la fecha deseada.
Las tablas admiten Markdown, LaTeX y Typst.

Con información completa, las ecuaciones se reducen a las recursiones usuales
$y_t=y_{t+1}-s(i_t-\pi_{t+1})$ y
$\pi_t=\kappa y_t+\beta\pi_{t+1}$. Las pruebas las verifican directamente.
Con información dispersa, sustituir la expectativa de la Euler usual por
un promedio de pronósticos generalmente produce otro modelo. Por eso se
conservan los valores presentes. Los pronósticos individuales son bayesianos
bajo las señales declaradas; los errores condicionales de una realización
particular no tienen por qué ser cero.

## Reproducir e interpretar la ilustración

```bash
python -m puremacro.examples.heterogeneous_expectations --output expectations-example --lang es
```

El ejemplo compara información completa, ausencia de noticias anticipadas,
señales iniciales heterogéneas y una segunda señal independiente en la fecha 4.
Todos conocen la noticia al implementarse la política. Exporta coeficientes,
trayectorias, creencias, pronósticos, auditorías, arreglos completos, parámetros,
hashes del código y figuras PNG/PDF. Un experimento de 50,000 simulaciones con
semilla fija contrasta varianzas posteriores analíticas, errores incondicionales
medios nulos y ortogonalidad respecto a las señales observadas. Informa la
incertidumbre Monte Carlo y aplica criterios de aceptación fijos.

Para la reducción de tasa en la fecha 8, las respuestas iniciales del producto
(desviación logarítmica × 100) son aproximadamente 3.9586 con información
completa, 0 sin noticias anticipadas, 0.8028 con señales heterogéneas y 1.1838
con la segunda actualización. En los dos últimos casos, el pronóstico inicial
promedio de política es la mitad del pronóstico con información completa.
La figura también divide la razón de respuestas por la razón de pronósticos
de política, separando la atenuación de noticias de primer orden de la
retroalimentación posterior. Esa razón no está definida sin noticias anticipadas.

El barrido de horizontes conserva las respuestas grandes: en la fecha 16,
la respuesta lineal del producto bajo información completa supera una unidad
logarítmica. Son sensibilidades del sistema lineal, no pronósticos cuantitativos
creíbles con ese tamaño de choque. El CSV marca desviaciones absolutas de
producto/inflación superiores a .05; la marca es descriptiva y no acota el
error de aproximación. Agregar fechas con política cero preserva los
coeficientes anteriores hasta la precisión numérica, pero no demuestra
determinación de horizonte infinito. Esta primera implementación no establece
calibración empírica, recomendaciones de política, bienestar ni integración HANK.
