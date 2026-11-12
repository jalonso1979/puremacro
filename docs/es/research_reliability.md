# Fiabilidad de investigación y próximos hitos de PureMacro

El orden acordado es **fiabilidad y replicaciones**, después **modelos y
aplicaciones de política**, y finalmente **docencia y facilidad de uso**.

## Correcciones publicadas y versión actual

La versión **6.3.0** sucede a las versiones 6.2.0 y 6.1.0 (publicada en PyPI el
2026-11-03 tras la 6.0.1 del 2026-10-29), incorporando las correcciones siguientes
y las registradas en el [registro de cambios](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md). Las etiquetas 5.8.0, 5.9.0 y 6.0.0
no se publicaron. Consulte la [hoja de ruta y política de API](1.0_path.md).

Los adaptadores conservan ecuaciones y cargas del modelo suministrado.
DSGE usa sensibilidades de Kalman; DAG usa diferencias centrales de Whittle
nativo manteniendo fijo el estado estacionario. Las operaciones no compatibles
producen errores; `model=None` selecciona demostraciones con advertencia.
Consulte [el alcance del protocolo](autodiff_and_protocol.md).

El DAG HANK predeterminado prescribe respuestas de hogares; el ejemplo de riesgo
agregado prescribe momentos y matrices. Son demostraciones, no distribuciones
de hogares resueltas. `asymmetry_ratio` mide el pico no lineal respecto al lineal,
no dos soluciones con choques opuestos. Consulte [el alcance HANK](hank_ssj_and_aggregate_risk.md).

Las correcciones soberanas incluyen default determinista, costos basados en
ingreso medio estacionario, reingreso con deuda cero y estadísticas simuladas
sin ajustes de signo. La correlación de un diferencial constante es `NaN`, con
metadatos que explican por qué no está definida.

## Primeras pruebas

Desde la raíz del repositorio, con Python 3.11+ y `.[dev,notebooks]`:

```sh
python -m pytest tests/test_protocol_model_fidelity.py tests/test_research_reliability_references.py tests/test_sovereign/test_reference_reliability.py -q
python tools/release_check.py --no-tests
```

El cuaderno **72** ofrece una visita interactiva bilingüe. **73** explora deuda
soberana; **74** combina LP distributivas sintéticas, inferencia gaussiana y
transiciones reducidas. Sus cuadros verifican propiedades concretas, no toda
la validez económica del modelo.

## Evidencia independiente disponible

| Comparación | Resultado en las mallas probadas | Pendiente |
|---|---|---|
| Ecuaciones Arellano de deuda corta y default determinista | Ecuaciones y archivos Fortran/MATLAB contrastados; reingreso e inicio con deuda cero corregidos; mallas de la calibración pública convergidas con 6.401 × 641, y el solucionador público alcanza esa malla con una búsqueda monótona certificada | Diferencias de momentos publicados y convergencia económica de extensiones |
| Bloque de hogares y equilibrio HANK publicado de un activo | Políticas y consumo coinciden con los autores SSJ dentro de 5e-11; jacobianos de rendimientos dentro de 4e-4. Se contrastan independientemente el equilibrio publicado completo, ambos signos no lineales y una malla/horizonte ampliados; véase la [evidencia HANK](hank_equilibrium.md) | Incidencia sobre hogares con datos observados y riesgo agregado general de orden superior |

Las referencias originales de ecuaciones y bloques, hashes y procedencia están
en `tests/fixtures/research_reliability/`; las guías enlazadas indican dónde
consultar los registros adicionales de equilibrio y malla. Los generadores están en
`tools/reference_validation/`. SSJ externo no es dependencia de ejecución.
Coincidir en una malla común no demuestra precisión de discretización.

## Estudio soberano: base completada, mallas públicas convergidas

La [comparación con código y momentos originales](arellano_replication.md)
incluye 16 simulaciones independientes por caso, mallas anidadas de deuda y de
21 a 321 estados de ingreso. Con la calibración archivada el refinamiento
cambia los momentos mucho más que la incertidumbre Monte Carlo, y la volatilidad
del diferencial se reduce a la mitad. Para la calibración pública, un criterio
prerregistrado con semillas emparejadas establece la convergencia de mallas con
6.401 nodos de deuda y 641 estados de ingreso; el mismo criterio establece la
convergencia de la calibración archivada con 6.369 × 641, donde la volatilidad
del diferencial es de cerca de 2,9 puntos porcentuales frente a los 6,36
publicados. Un residuo pequeño de Bellman no
habría bastado para esa conclusión.

1. Reingreso, inicio con deuda cero y medición públicos están documentados y
   probados. Mantener separados el archivo y la calibración pública porque
   usan convenciones económicas diferentes.
2. Hecho para la calibración pública: las mallas convergen con 6.401 × 641 bajo
   el criterio registrado, separando incertidumbre muestral de error de
   discretización y conservando todos los momentos.
   `solve_sovereign_default(policy_search="monotone")` resuelve esa malla,
   certificada por una comprobación exhaustiva, y coincide con los arreglos de
   referencia.
3. Estudiar volatilidad, persistencia del ingreso y probabilidad de reingreso
   en el modelo validado de deuda corta. Reportar bienestar e incertidumbre
   numérica antes de agregar vencimientos y reestructuración no contrastados.

Así se obtiene un resultado revisable desde ecuaciones hasta tablas de política.

Las [referencias de extensiones soberanas](sovereign_extensions_validation.md)
contrastan independientemente el operador suavizado de deuda larga, estados tras
quitas/vencimientos y políticas del planificador con colateral. Simulación y
valoración usan la misma distribución de decisiones de deuda. Los diagnósticos
se miden sobre trayectorias, con amplificación indefinida explícita y sin valores
sustitutos impuestos. El residuo temporal es un ajuste de capitalización. Son
contrastes de ecuaciones en mallas pequeñas, no replicaciones de momentos publicados.

## Después: incidencia sobre hogares y docencia

El [equilibrio HANK publicado de un activo](hank_equilibrium.md) ya cuenta con
reglas fiscales, fechas de rendimientos y mercados comparables, trayectorias
no lineales independientes para ambos signos y una malla/horizonte ampliados
que pasan el criterio. Sigue vincular la incidencia sobre hogares con LP de
datos observados. El riesgo agregado general de orden superior sigue siendo
una tarea posterior separada.

Convertir el estudio validado en una ruta bilingüe breve: instalar un wheel
verificado, reproducir una base, cambiar un supuesto y revisar diagnósticos.
Mantener explícita la distinción entre demostraciones y resultados empíricos.
