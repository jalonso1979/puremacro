# Arellano: código original y sensibilidad a las mallas

**Se reproduce el algoritmo archivado. Para la calibración pública de
PureMacro, la convergencia de mallas queda establecida con 6.401 nodos de deuda
y 641 estados de ingreso bajo tolerancias prerregistradas. Los momentos
publicados no convergen con la malla: en mallas finas el mismo modelo produce
cerca de la mitad de la volatilidad publicada del diferencial.** La comparación conserva las convenciones de cada
modelo y no recalibra parámetros para acercarse a los objetivos publicados.

## Ejecutar localmente

Desde el código de desarrollo, con Python 3.11+ y las dependencias habituales:

```sh
python -m pytest tests/test_arellano_2008_study.py -q
MPLBACKEND=Agg python -m tools.reference_validation.run_arellano_study --output /tmp/arellano-study
```

Se generan figuras PNG/PDF, `moments.csv` y `results.json`, con semillas,
intervalos, episodios, residuos y hashes del código. La ejecución habitual usa
referencias congeladas y no requiere red, gfortran ni Octave. El estudio tarda
unos minutos en la máquina probada.

## Verificación independiente

El Fortran del autor converge tras 49 iteraciones de precios. La implementación
independiente reproduce las 4.200 elecciones de deuda y default. Los precios
coinciden exactamente y los valores difieren como máximo en 5,3e-11. Además,
el bloque MATLAB original de agregación, ejecutado en Octave con las mismas
series y ventanas, coincide en los 12 momentos dentro de 1,1e-14. Esto verifica
código y medición; no certifica una replicación cuantitativa completa.

Fuentes: [artículo AER](https://doi.org/10.1257/aer.98.3.690),
[borrador público de julio de 2007](https://mpra.ub.uni-muenchen.de/7867/1/MPRA_paper_7867.pdf)
y [archivo de replicación](https://doi.org/10.3886/E113248V1). El borrador también
contiene los objetivos de la Tabla 4. Los archivos originales, licencias,
hashes y procedimientos de regeneración están en el directorio local
`tests/fixtures/arellano_2008/`.

## Objetivos publicados y malla original

Se usan 16 semillas independientes, 100.000 trimestres por semilla y 1.000 de
descarte. Cada réplica promedia 100 ventanas de 74 trimestres anteriores al
default, sin interrupciones del acceso al mercado. Los intervalos miden error
Monte Carlo de la **media entre réplicas**, no incertidumbre paramétrica ni la
dispersión de una sola réplica de 100 ventanas; esta última está en el JSON.

| Momento | Modelo publicado | Malla original 200 × 21 | Intervalo MC 95% |
|---|---:|---:|---:|
| Deuda/producto (%) | 5,95 | 6,283 | [6,083; 6,482] |
| Diferencial anual medio (%) | 3,58 | 3,405 | [3,362; 3,448] |
| Frecuencia anual de default (%) | 3,00 | 2,822 | [2,756; 2,887] |
| DE del diferencial (puntos porcentuales) | 6,36 | 6,009 | [5,891; 6,126] |
| DE balanza comercial/producto (pp) | 1,50 | 1,485 | [1,463; 1,508] |
| DE log consumo (%) | 6,38 | 6,391 | [6,323; 6,459] |
| DE log producto (%) | 5,81 | 5,826 | [5,754; 5,898] |
| Correlación diferencial, producto | -0,29 | -0,251 | [-0,265; -0,236] |
| Correlación balanza comercial, producto | -0,25 | -0,245 | [-0,252; -0,239] |
| Correlación consumo, producto | 0,97 | 0,971 | [0,970; 0,971] |
| Correlación balanza comercial, diferencial | 0,43 | 0,432 | [0,423; 0,441] |
| Correlación consumo, diferencial | -0,36 | -0,331 | [-0,347; -0,316] |

Persisten diferencias respecto a la media estimada. No están disponibles los
sorteos originales y el script archivado examina candidatos a default 2:300,
mientras el artículo describe 100 ventanas. No se afirma una reproducción
exacta de la tabla publicada.

## La calibración archivada no converge con la malla

Las mallas de deuda son anidadas, con extremos fijos y cero exacto; las mallas
de ingreso mantienen sus extremos logarítmicos. Los residuos independientes de
Bellman son inferiores a 4e-12 y los de precios a 1e-8. Ninguna elección simulada
de deuda con repago alcanza los extremos. Aun así, los momentos cambian:

| Activos × estados de ingreso | Deuda/producto (%) | Diferencial medio (%) | Default anual (%) | DE diferencial (pp) |
|---|---:|---:|---:|---:|
| 200 × 21, Fortran original | 6,283 | 3,405 | 2,822 | 6,009 |
| 399 × 21 | 6,322 | 3,248 | 2,663 | 5,402 |
| 797 × 21 | 6,347 | 3,320 | 2,740 | 5,348 |
| 797 × 41 | 6,010 | 4,043 | 3,558 | 4,442 |
| 797 × 81 | 5,491 | 3,504 | 3,080 | 3,477 |
| 797 × 161 | 5,246 | 3,370 | 2,957 | 3,230 |
| 797 × 321 | 5,144 | 3,321 | 2,942 | 3,098 |

Las filas de 161 y 321 estados provienen de
`run_arellano_study --income-states 41 81 161 321`, con las mismas semillas,
ventanas, calibración y reingreso archivados. Residuos pequeños y ausencia de
masa en los extremos no establecen convergencia económica. De 81 a 321 estados
la desviación del diferencial sigue cayendo, la deuda/producto baja y la
correlación entre balanza comercial y diferencial se estabiliza cerca de 0,77
frente al 0,43 publicado. La volatilidad publicada del diferencial, 6,36 puntos
porcentuales, es por tanto la que produce la malla archivada de 200 × 21 con
esta calibración; en mallas finas el mismo modelo genera cerca de la mitad. Es
el patrón documentado por Hatchondo, Martinez y Sapriza (2010): la técnica de
espacio de estados discreto con mallas equiespaciadas necesita muchos puntos
para evitar movimientos espurios de la tasa de interés. La escalera de la
calibración pública, más abajo, muestra que los nodos de deuda importan igual:
de 801 a 6.401 nodos la volatilidad del diferencial cae otros 0,44 puntos
porcentuales. Acercarse a un objetivo
aislado no demuestra convergencia conjunta, y aquí no se reajusta ningún
parámetro.

Hatchondo, J. C., L. Martinez y H. Sapriza (2010), "Quantitative properties of
sovereign default models: solution methods matter", *Review of Economic
Dynamics* 13(4), 919–933, [doi:10.1016/j.red.2010.03.001](https://doi.org/10.1016/j.red.2010.03.001).

## Convenciones del modelo y de medición

El archivo limita el ingreso en default a `10*.969`, aunque el ingreso medio
es aproximadamente 10,0307. PureMacro normaliza la media estacionaria y aplica
el límite a esa media. Los datos archivados coinciden con Tauchen a la precisión
impresa, aunque el borrador menciona Tauchen–Hussey. La malla de activos llega
a 1,53502537, no a su extremo nominal de 1,5.

El archivo impone un período adicional de exclusión después de entrar en
default. Cambiar únicamente esa convención por la ecuación canónica reduce
deuda/producto de 6,283% a 5,440% y eleva el diferencial medio de 3,405% a 4,184%.
La calibración separada de la API pública, con 321 × 21 puntos, produce 5,055%
y 4,329%, respectivamente. También cambia normalización, malla y precisión;
por tanto no aísla solamente el efecto del momento de reingreso.

El estudio usa diferenciales compuestos `q**-4 - Rf**4`, desviaciones muestrales
por ventana y correlaciones sin ajustes. La frecuencia comparable al archivo
usa períodos de repago como denominador; el JSON también presenta el riesgo
con períodos de entrada al default incluidos en la exposición. Numerador y
denominador usan la misma muestra posterior al descarte. No se usan los
diagnósticos simulados breves de la API pública. Su antiguo bucle conservaba
un período adicional respecto a la ecuación canónica de Bellman. Ese error ya
está corregido: el reingreso permite decidir en el mismo trimestre. Las dos
interfaces públicas restablecen deuda cero y coinciden con la misma semilla.
El simulador común del estudio respeta la ecuación de Bellman de cada caso.

## Comprobar las mallas de la calibración pública

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids
```

Se resuelven ecuaciones independientes, contrastadas con la API pública en una
malla común de 401 × 9. Calibración, extremos, momento de reingreso y medición
permanecen fijos. La ejecución predeterminada compara 401/801 activos y 321/641
estados de ingreso, con 16 semillas emparejadas por caso. Esa escalera corta no
aprueba; la sección siguiente presenta la que sí. Las convenciones públicas de
reingreso y medición están documentadas en la [guía soberana](sovereign_default.md).

El cambio de cada momento porcentual debe ser como máximo 0,10 puntos
porcentuales; el de cada correlación, 0,02. Se suma el cambio medio absoluto a
la semiamplitud del intervalo Student-t de 95% de las diferencias emparejadas.
Deben aprobar tanto el refinamiento final de deuda como el de ingreso, con al
menos 797 activos y 321 estados de ingreso en la malla fina. Son tolerancias
numéricas, no ajustes a objetivos publicados. Un código de salida distinto de
cero distingue cambios grandes de precisión Monte Carlo insuficiente. Aprobar
la base no valida nuevos parámetros, políticas ni deuda a largo plazo.

El informe registra semillas, hashes, residuos y comparaciones. Las soluciones
verificadas se reutilizan con `--periods 500000 --windows 1000` para aumentar
precisión manteniendo semillas. Los resultados archivados anteriores conservan
su significado histórico para otra calibración y convención de reingreso.

## Calibración pública: convergencia de mallas establecida

```sh
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids \
  --refinements 1 2 3 4 --income-states 321 641
MPLBACKEND=Agg python -m tools.reference_validation.check_arellano_grids --output /tmp/arellano-grids \
  --refinements 1 2 3 4 --income-states 321 641 --periods 500000 --windows 1000
```

La primera orden resuelve mallas de deuda de 801, 1.601, 3.201 y 6.401 nodos con
321 y 641 estados de ingreso usando la búsqueda monótona certificada. Las ocho
soluciones tardaron cerca de dos horas en la máquina probada, la mayor 35
minutos. La segunda orden reutiliza las soluciones verificadas por hash y solo
vuelve a simular.

Con el diseño estándar aprobaron 23 de las 24 comparaciones finales. La
correlación entre balanza comercial y diferencial cambió 0,017 de 3.201 a 6.401
nodos, dentro de su tolerancia de 0,02, pero su cota fue 0,0204, de modo que la
precisión muestral no permitía clasificarla. La regla prerregistrada aplica
entonces las mismas semillas con 500.000 trimestres y 1.000 ventanas por
semilla. Con ese diseño aprueban todas las comparaciones:

| Nodos de deuda × estados de ingreso | Deuda/producto (%) | Diferencial medio (%) | Default anual (%) | DE diferencial (pp) | Correlación balanza comercial, diferencial |
|---|---:|---:|---:|---:|---:|
| 801 × 321 | 4,042 | 3,479 | 3,105 | 3,274 | 0,701 |
| 1.601 × 321 | 4,044 | 3,461 | 3,086 | 3,034 | 0,753 |
| 3.201 × 321 | 4,043 | 3,425 | 3,048 | 2,895 | 0,787 |
| 6.401 × 321 | 4,052 | 3,414 | 3,037 | 2,844 | 0,809 |
| 801 × 641 | 3,984 | 3,453 | 3,084 | 3,215 | 0,704 |
| 1.601 × 641 | 3,993 | 3,423 | 3,064 | 2,959 | 0,760 |
| 3.201 × 641 | 4,002 | 3,378 | 3,012 | 2,813 | 0,798 |
| 6.401 × 641 | 4,003 | 3,362 | 2,997 | 2,774 | 0,813 |

En el refinamiento final de deuda, de 3.201 a 6.401 nodos con 641 estados, las
mayores cotas son 0,043 pp para la volatilidad del diferencial, frente a 0,10,
y 0,017 para la correlación entre balanza comercial y diferencial, frente a
0,02. En el refinamiento final de ingreso, de 321 a 641 estados con 6.401 nodos,
la mayor es 0,076 pp para la volatilidad del diferencial. Cada solución tiene
residuo de Bellman inferior a 2e-12 y residuo de precios inferior a 9e-9. La
comprobación exhaustiva de las políticas aceleradas alcanza el mismo residuo de
Bellman. El registro congelado,
`tests/fixtures/arellano_2008/public_grid_convergence.json`, conserva cada
semilla, momento, hash, residuo y comparación. Una prueba recalcula ambas
comparaciones a partir de él y verifica esta tabla.

**Lo que el criterio no acota.** Limita el último cambio, no la distancia al
límite. La volatilidad del diferencial y la correlación entre balanza comercial
y diferencial se han movido en una sola dirección en cada refinamiento. En la
escalera de 641 estados sus cambios sucesivos se reducen por factores de 0,26 a
0,57 y de 0,41 a 0,68 por duplicación. La extrapolación geométrica con esos
factores sitúa el límite de la volatilidad 0,01 a 0,05 pp por debajo de 2,774,
dentro de su tolerancia, y el de la correlación 0,01 a 0,03 por encima de 0,813,
posiblemente fuera de ella. Reporte esa correlación con dos decimales. La
extrapolación es un diagnóstico, no parte del criterio registrado.

## Resolver el diseño convergido con el solucionador público

```python
from puremacro.sovereign import solve_sovereign_default

converged = solve_sovereign_default(
    beta=0.95282, r_star=0.017, theta=0.282, rho_y=0.945, sigma_y=0.025,
    b_min=-0.15, b_max=0.33, n_b=6401, n_y=641, sigma_eps=0.0,
    y_penalty_type="arellano", enforce_monotonicity=False,
    tol=1e-10, max_iter=10000, policy_search="monotone",
)
```

La búsqueda densa necesitaría arreglos de nodos de deuda × nodos de deuda ×
estados de ingreso, cerca de 210 GB cada uno con esta malla.
`policy_search="monotone"` busca la deuda de corto plazo en O(n_b log n_b) por
estado de ingreso y devuelve los mismos valores que la búsqueda densa, y las
mismas políticas salvo empates exactos de valor entre opciones con distinto
ingreso por bonos. Tras la
convergencia, una búsqueda exhaustiva por bloques sobre todos los candidatos la
certifica; una diferencia de valor de reembolso superior a 1e-11 produce un
error. En la máquina probada esta solución tardó 28 minutos, 138 iteraciones y
2,4 GB de memoria, con diferencia de certificación cero. Sus precios coinciden
con la solución de referencia de 6.401 × 641 dentro de 5,4e-9, con políticas y
conjuntos de impago idénticos; una prueba repite la comparación con 401 × 21, y
con 1.601 × 21 bajo `pytest -m slow`.

Los momentos de la tabla anterior provienen del simulador del estudio: 16
semillas emparejadas, 500.000 trimestres y ventanas anteriores al impago. Se
trasladan a esta solución porque sus arreglos coinciden con los de referencia.
`converged.simulate()` devuelve una trayectoria con otras convenciones de
medición, de modo que sus estadísticas no son las mismas cifras.

### Mallas de deuda explícitas

El solucionador de desarrollo 6.1 también acepta `debt_grid=coordinates`, un
vector finito y estrictamente creciente que incluya un cero exacto. Copia y
conserva esas coordenadas, que sustituyen `n_b`, `b_min` y `b_max`; el
comportamiento de la malla generada existente no cambia. La reestructuración
de vencimientos conserva la malla suministrada. Las tangentes de precios en
mallas no uniformes usan pendientes PCHIP ponderadas por los intervalos.

Concentrar nodos donde el endeudamiento y los precios de los bonos son sensibles
puede mejorar la eficiencia; véase [Hatchondo, Martinez y Sapriza (2010)](https://www.imf.org/-/media/websites/imf/imported-full-text-pdf/external/pubs/ft/wp/2010/_wp10100.pdf).
Esto motiva un experimento separado de reingreso alto con el soporte original
`[-0.15, 0.33]`, nodos de ahorro lineales y nodos de endeudamiento cuadráticos.
Sus cuatro casos cruzan 6.401/12.801 nodos de deuda y 641/1.281 estados de
ingreso; se conservan el diseño de simulación y las cuatro tolerancias de 0,10.
Ambos refinamientos finales deben pasar. El [informe soberano](brief_sovereign_risk.md)
conserva los resultados fallidos de las mallas uniformes originales.

```bash
python -m tools.reference_validation.check_sovereign_curved_grids --output grids_curved
```

El comando escribe el protocolo antes de resolver y guarda cada coordenada,
semilla, hash de fuente, residuo de certificación y comparación. Una malla no
uniforme por sí sola no demuestra convergencia; evalúe el registro completo.

## Próxima decisión

La calibración pública ya tiene un diseño numérico fijo: 6.401 nodos de deuda,
641 estados de ingreso y el diseño de precisión para los momentos simulados.
Los experimentos de reingreso, volatilidad y persistencia del ingreso pueden
partir de ese diseño, pero cada escenario requiere su propia comprobación de
malla. Mantenga separadas la calibración
archivada y la pública; la convergencia de la archivada no está establecida más
allá de 797 puntos de deuda. Vencimientos y reestructuración requieren su propia
validación externa. Cualquier recalibración sería un resultado nuevo, no parte
de esta validación.
