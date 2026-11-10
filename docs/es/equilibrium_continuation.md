> 🇬🇧 [English](../equilibrium_continuation.md) · 🇪🇸 Español

# Continuación de ramas de equilibrio

`puremacro.core.continue_equilibrium` sigue una rama local de
\(F(x,\theta)=0\), variando un parámetro y manteniendo fijos los demás.
Usa el mismo [contrato de residuos](equilibrium_linearization.md) y los
jacobianos comprobados de la interfaz común de derivadas.

El predictor avanza sobre una tangente en coordenadas escaladas; un corrector
Newton amortiguado resuelve las ecuaciones y un hiperplano tangente.
Esta es la construcción de pseudo-longitud de arco descrita en la
[documentación de BifurcationKit](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/PALC/).
El sistema aumentado puede seguir siendo regular en un pliegue simple donde
falla la derivada ordinaria respecto al parámetro; véase
[Dickson et al.](https://arxiv.org/abs/math/0603716) para su condicionamiento.

## Trazar e inspeccionar una rama

```python
import numpy as np
from puremacro.core import EquilibriumSystem, continue_equilibrium

system = EquilibriumSystem(
    lambda x, p: x**3 - x - p,
    ("capital",), ("policy",),
)
branch = continue_equilibrium(
    system, [-1.5], [-1.875],
    step=.08, max_step=.12, max_steps=100,
    parameter_bounds=(-2., 2.), branch_id="cubic-example",
)
assert len(branch.folds) == 2
assert all(f.status == "simple_fold" for f in branch.folds)
print(branch.summary())
ax = branch.plot("capital")
```

El punto inicial debe satisfacer las ecuaciones. `direction=1` o `-1`
selecciona la dirección inicial del parámetro, que puede invertirse en un
pliegue. Con varios parámetros se especifica `parameter="name"`. Semillas y
opciones inválidas, y excepciones de las funciones o del dominio del modelo,
generan errores directamente.

`states` y `parameters` contienen coordenadas físicas. `tangents` contiene
vectores unitarios en coordenadas escaladas de estado/parámetro seleccionado,
con el parámetro al final. `arclength` acumula longitudes de cuerda en esas
coordenadas; `step_sizes` contiene distancias solicitadas al hiperplano
tangente. Los arreglos son copias de solo lectura. Las funciones genéricas
siguen bajo control del usuario y deben ser puras; los adaptadores incorporados
conservan copias de sus entradas numéricas.

`to_frame`/`to_dataframe`, `summary`, `to_markdown`, `to_latex`,
`to_typst` y `plot` muestran puntos aceptados. Las columnas usan prefijos
`state:` y `parameter:`. `attempts_frame()` incluye intentos aceptados y
rechazados, iteraciones, residuos y motivos. Los diagnósticos no disponibles
en predictores rechazados son NaN. Los gráficos no llaman a `show`.

## Reiniciar o invertir el recorrido

```python
reverse = continue_equilibrium(
    branch.system, branch.states[-1], branch.parameters[-1],
    parameter=branch.parameter_name,
    initial_tangent=-branch.tangents[-1],
    step=.03, max_step=.05, max_steps=8,
)
assert reverse.states[-1, 0] < reverse.states[0, 0]
```

`initial_tangent` determina la orientación y tiene prioridad sobre `direction`;
debe pertenecer al espacio nulo aumentado en coordenadas escaladas. Comenzar
en un punto de giro del parámetro exige esta orientación explícita.
Los pasos de hiperplano inversos no tienen que visitar puntos idénticos.

## Reutilizar las ecuaciones de un modelo

```python
from puremacro.vfi import CollocationProblem, linearize_projection

kss = (.36*.96)**(1/(1-.36))
problem = CollocationProblem(
    domain=(.5*kss, 1.5*kss), orders=6, method="euler", beta=.96,
    params={"alpha": .36, "delta": 1., "sigma": 1.},
    options={"tol": 1e-12, "max_iter": 500},
)
solution = problem.solve()
lin = linearize_projection(solution, problem)
growth = continue_equilibrium(
    lin.system, lin.state, lin.parameters, parameter="beta",
    step=.002, max_step=.002, max_steps=8,
)
assert growth.status == "max_steps"
```

La misma llamada acepta el sistema conservado por `linearize_aiyagari`.
Consumo, probabilidades estacionarias y tasas de interés cambian conjuntamente.
Las mallas y la cadena de ingreso permanecen fijas. La implementación se
detiene ante cambios declarados de régimen de restricción, interpolación o
lotería. Las ramas suaves y pequeñas de Aiyagari se comparan con soluciones
independientes; esto no establece continuación global entre regímenes.

## Pasos, terminación y pliegues

Los valores iniciales son `step=.05`, `min_step=1e-5`, `max_step=.2`,
`max_steps=100`, `max_newton=12` y `max_retries=8`. Los intentos fallidos
reducen el paso a la mitad; los exitosos lo adaptan según el esfuerzo del
corrector y la alineación de tangentes. Correcciones grandes y giros bruscos
provocan reintentos. Los límites finitos del parámetro detienen el recorrido,
posiblemente antes de alcanzarlos exactamente. Restringen la trayectoria,
no las sondas de derivación: las ecuaciones necesitan un entorno bilateral.

La terminación por presupuesto o límite usa `max_steps` o `parameter_bound`.
Los demás estados conservan la trayectoria parcial aceptada e identifican el
último fallo diagnosticado, como `regime_change`, `newton_failed`,
`rank_deficient`, `rank_unresolved`, `possible_branch_point`,
`ill_conditioned` o `step_not_representable`. Revise estado e intentos antes
de usar una rama. Una prueba de signos detecta algunos puntos de ramificación
intermedios; no puede detectar todas las singularidades entre muestras.

Las tolerancias iniciales del equilibrio escalado y del hiperplano son
`1e-9`. `derivative_step=1e-5` y `derivative_tol=1e-3` controlan la
comprobación de jacobianos. La derivada aumentada debe cumplir
`rank_tol=1e-10` y `condition_limit=1e10`. Cerca de una pérdida de rango,
un paso de derivación reducido a la mitad comprueba que el menor valor
singular supere la discrepancia observada entre derivadas.

Una inversión de la tangente del parámetro delimita un candidato a pliegue.
El refinamiento usa hasta `fold_max_iter=40` correcciones y tolerancia
escalada del intervalo `fold_tol=1e-7`. Los registros incluyen corango de
estado, condicionamiento aumentado, transversalidad del vector nulo izquierdo
y un coeficiente direccional cuadrático. La curvatura usa
`curvature_step=1e-4` y la mitad de ese paso; `degeneracy_tol=1e-5` controla
las pruebas de no degeneración. Se distingue entre `simple_fold`,
`degenerate_or_unresolved` y `unresolved`. Son clasificaciones numéricas
según escalas y tolerancias declaradas, no certificados matemáticos. Los
pliegues refinados se registran aparte de la trayectoria. Una resolución
finita puede omitir pliegues cercanos o giros muy planos.

## Alcance y ejemplo reproducible

El método es denso, con `max_states=1500` por defecto. Conserva arreglos de
la trayectoria, no todos los jacobianos densos. Las escalas afectan la
geometría y el condicionamiento. No hay cambio automático de rama, enumeración
de raíces desconectadas, afirmación de exhaustividad global, clasificación de
estabilidad, análisis de Hopf ni inferencia estadística. `branch_id` es una
etiqueta del usuario.

Ejecute `python -m puremacro.examples.equilibrium_branches --output /tmp/equilibrium-branches`.
El modelo ilustrativo de ahorro fijo tiene producción
`f(k)=.05+k**2/(1+k**2)` y equilibrio `.1*k=s*f(k)`.
Los archivos CSV/JSON y PNG/PDF comparan trayectoria y pliegues con una curva
analítica independiente. En `s=.20`, soluciones escalares independientes
recuperan tres equilibrios y coinciden con el polinomio cúbico asociado.
La ilustración no es una economía estimada ni un modelo de ahorro óptimo.

Las pruebas también cubren una curva cúbica en S, ecuaciones acopladas,
transformaciones de unidades, pliegues degenerados, semillas de ramificación,
reintentos y fallos, parámetros fijos no diferenciables y soluciones
independientes de RBC/Aiyagari.
