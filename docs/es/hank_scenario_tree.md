> 🇬🇧 [English](../hank_scenario_tree.md) · 🇪🇸 Español

# HANK con riesgo agregado en un árbol finito

Este solucionador experimental extiende las [ecuaciones publicadas del modelo
de un activo](hank_equilibrium.md) a historias monetarias en un árbol finito.
Los hogares eligen ahorro y horas antes de conocer los choques futuros.
EGM utiliza el valor marginal esperado de los sucesores, condicionado a la
historia actual. Cada historia conserva su distribución de hogares. La
transición idiosincrática es independiente de la rama agregada.

La malla pequeña siguiente demuestra la API; no satisface el protocolo de
refinamiento que se presenta más adelante.

```python
from puremacro.models import (
    calibrate_one_asset_hank, monetary_scenario_tree, solve_hank_scenario_tree,
)

ss = calibrate_one_asset_hank(n_a=120, n_e=5)
tree = monetary_scenario_tree(horizon=12, branch_date=1,
                             innovations=(-.005, .005), probabilities=(.5, .5))
result = solve_hank_scenario_tree(ss, tree)
print(result.converged, result.diagnostics)
print(result.expected_paths())
```

Las probabilidades son condicionales. El choque de la fecha uno todavía no se
conoce en cero y después decae según la persistencia declarada. Un choque de
media cero puede cambiar las decisiones iniciales por la no linealidad del
valor marginal. Promediar trayectorias resueltas con previsión perfecta usaría
un conjunto de información diferente.

## Cierre económico explícito

Inflación, salarios y producción satisfacen los mercados de activos y trabajo
y la curva de Phillips esperada en cada nodo. La tasa nominal se determina
en el padre; la inflación del hijo determina el rendimiento real realizado.
La deuda pública real es constante y los impuestos pagan sus intereses
realizados. Impuestos y dividendos son proporcionales a la habilidad.

Se **supone** el factor de valoración empresarial `M_hijo=1/(1+r_hijo)`,
equivalente a valoración nominal neutral al riesgo. La curva de Phillips es

$$0=\kappa(w_v-1/\mu)-\log(1+\pi_v)+
\sum_{c\in hijos(v)}p(c\mid v)\frac{Y_c}{Y_v}
\frac{\log(1+\pi_c)}{1+r_c}.$$

El factor no se estima a partir de la utilidad marginal de los hogares. En
una cadena determinista se recuperan las ecuaciones existentes del modelo
publicado; ello no valida empíricamente el nuevo supuesto estocástico.

En los nodos terminales se usan valores marginales estacionarios e inflación
futura cero. No se resuelve una continuación estocástica. Se admiten árboles
sin recombinación de 2 a 256 nodos, ordenados por dependencia, con una raíz
y una fecha terminal común. Historias distintas no comparten silenciosamente
una distribución.

El sistema cuadrado de equilibrio usa el método híbrido de Powell, que
actualiza su Jacobiano entre barridos numéricos completos. Se compara con una
optimización independiente por mínimos cuadrados de las mismas ecuaciones.
`max_evaluations=None` usa el presupuesto de SciPy según la dimensión; un entero
explícito cuenta evaluaciones residuales, incluidas las del Jacobiano numérico.
Se mantienen todas las verificaciones económicas y numéricas de aceptación.

## Evidencia y límites

Las pruebas comparan ambas señales del choque determinista con el solucionador
publicado independiente. Duplicar historias reproduce la misma trayectoria
anunciada. También se reconstruyen presupuestos, distribuciones heredadas,
rendimientos de deuda nominal, financiamiento fiscal y expectativas de Phillips.

```sh
python -m pytest tests/test_models/test_hank_scenario_tree.py -q
python tools/reference_validation/check_hank_scenario_tree.py --output evidencia-arbol
```

El estudio guarda trayectorias, políticas, distribuciones, huellas de código y
comparaciones fallidas. Compara los primeros seis periodos de consumo,
producción e inflación condicionales con tolerancia de 1% del máximo de la
respuesta fina y piso `1e-6`. Examina horizontes de 12/24 periodos, 120/240
nodos de activos y cinco/siete estados de ingreso.
Cada malla recalibra el descuento y la desutilidad laboral a los mismos
objetivos de bonos y trabajo efectivo, conservando los coeficientes obtenidos.
Se comparan calibraciones discretizadas, no coeficientes estructurales fijos.

El registro completo de `tests/fixtures/hank_scenario_tree/` extiende el estudio
a 31 estados de ingreso, 960 nodos de activos, límite de activos 600 y 96
trimestres. La diferencia con el solucionador determinista es `7.10e-10`,
inferior a la tolerancia `3e-8`. Las comparaciones finales pasan para C, Y e
inflación:

| Refinamiento | Dimensiones fijas | Mayor cambio absoluto entre C/Y/pi |
|---|---|---:|
| Límite de activos 300 → 600 | 480 activos, 15 ingresos, 48 trimestres | 3.76e-7 |
| Estados de ingreso 21 → 31 | 480 activos, límite 600, 48 trimestres | 3.76e-5 |
| Nodos de activos 480 → 960 | 31 ingresos, límite 600, 48 trimestres | 4.67e-7 |
| Horizonte 48 → 96 | 480 activos, 31 ingresos, límite 600 | 1.16e-7 |

La comparación de ingreso alcanza el 98,3% de su tolerancia de consumo. Se
conservan los fallos anteriores de horizonte e ingreso y los de malla estrecha
por salidas o distribución estacionaria inválida. Por eso
`all_refinements_passed` sigue siendo falso. Las comparaciones aceptadas se
refieren a los primeros seis periodos condicionales bajo la innovación
declarada; no establecen precisión de malla infinita ni de pequeñas correcciones
por incertidumbre. El README del registro contiene el comando completo.

Se conservan todas las trayectorias condicionales y esperadas. El registro
compacto omite explícitamente los grandes archivos de hogares, conserva sus
huellas originales y permite regenerarlos con los programas del estudio.

`converged` exige las verificaciones de mercados, presupuestos y masa, y menos
de `1e-10` de masa con elecciones fuera de la malla de activos. Los errores
Euler/KKT y la masa en la frontera son diagnósticos adicionales. Un residuo
de mercado pequeño no establece precisión respecto al horizonte o la malla.
No existe aquí una referencia estocástica independiente de los autores, un
equilibrio estocástico estacionario general, un certificado de bienestar
infinito ni un modelo de dos activos. La demostración reducida anterior de
riesgo agregado permanece sin cambios.

Para soluciones repetidas o árboles mayores, el
[espacio de trabajo optimizado](hank_optimized.md) usa tangentes analíticas
del hogar y de la distribución con precondicionamiento disperso. Resuelve las
mismas ecuaciones y conserva los supuestos terminales y de valoración; este
solver denso sigue disponible como comparación numérica.
