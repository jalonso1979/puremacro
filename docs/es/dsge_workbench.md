> 🇬🇧 [English](../dsge_workbench.md) · 🇪🇸 Español

# Un espacio de trabajo para experimentos DSGE

`DSGEWorkbench` reúne un modelo, su calibración editable y sus ejecuciones
guardadas. Utiliza los motores existentes de IRF de primer orden y previsión
perfecta determinista. Las vistas del cuaderno muestran el estado, los insumos
exactos y comprobaciones separadas; no eligen otro problema matemático.

## Resolver, guardar y comparar

```python
import numpy as np
from puremacro.dsge import DSGEWorkbench, DSGERun

source = """
var x; varexo e; parameters rho; rho=.8;
model; [name='Persistence'] x=rho*x(-1)+e; end;
initval; x=0; end;
"""
bench = DSGEWorkbench.from_mod(source, name="Experimento de persistencia")
baseline = bench.irf("e", horizon=20, size=.01, label="Base")
alternative = bench.irf("e", horizon=20, size=.01,
                         params={"rho": .5}, label="Menor persistencia")
assert baseline.status == alternative.status == "succeeded"
np.testing.assert_allclose(baseline.table().x, .01 * .8 ** np.arange(21))
comparison = baseline.compare(alternative)
print(comparison.to_frame())
print(comparison.settings().query("changed"))
```

Muestre `bench`, `baseline`, `baseline.accuracy` o `comparison` en el cuaderno
para ver tablas HTML con evidencia desplegable. `comparison.plot()` superpone
las trayectorias guardadas. `baseline.plot()` y `baseline.accuracy.plot()` crean
figuras Matplotlib. Cambiar parámetros después no modifica estas capturas.
`to_dict()`, `table()` y los accesores de evidencia devuelven copias.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    directory = Path(directory)
    path = baseline.save(directory / "baseline.json")
    saved = DSGERun.load(path)  # comprueba integridad; no ejecuta el modelo
    replayed = saved.replay()  # resuelve explícitamente la receta guardada
    np.testing.assert_allclose(replayed.table(), baseline.table())
    (directory / "baseline.py").write_text(saved.to_python(), encoding="utf-8")
    (directory / "baseline.html").write_text(saved.to_html(), encoding="utf-8")
    (directory / "comparison.html").write_text(comparison.to_html(), encoding="utf-8")
```

Los archivos conservan el texto `.mod` expandido, hashes de los archivos fuente
originales, calibración, choques, opciones, unidades, convención temporal,
identificador, entorno, tablas y evidencia. Modificar archivos incluidos después
no cambia la receta guardada. Sobrescribir requiere `overwrite=True`; la carga
rechaza hashes dañados y archivos mayores de 32 MiB. Los hashes detectan corrupción
accidental, no certifican la identidad del autor.

La repetición verifica versiones de Python y paquetes y una huella del código de
PureMacro. Cambiar el entorno requiere `allow_environment_change=True` y produce
una ejecución nueva. El Python exportado es una receta editable con comentarios
del entorno; ejecutarlo **no** aplica esa comprobación. Cargar nunca ejecuta la
receta. Los modelos con funciones Python y las capturas externas son instantáneas
sin repetición portátil ni exportación de código.

Comparar requiere ejecuciones exitosas con las mismas variables ordenadas,
unidades declaradas, convención temporal y fechas. Fechas distintas requieren
`alignment="inner"`, que informa la ventana común y ambas longitudes originales.
Choques o condiciones de frontera diferentes requieren
`allow_experiment_change=True`. `settings()` muestra cambios de calibración y
método. Una diferencia entre ejecuciones no prueba convergencia ni bienestar.

## Aplicar cambios explícitamente

```python
widget = bench.interactive_irf(parameters={"rho": (.1, .95)}, shocks="e",
                               horizon=20, size=.01)
widget.update({"rho": .65})
assert widget.run_state == "edited"
assert widget.to_frame().isna().all().all()
widget.apply()
assert widget.is_current
snapshot = widget.capture_run(label="Experimento seleccionado")
np.testing.assert_allclose(snapshot.table().x, .01 * .65 ** np.arange(21))
widget.disconnect()
```

El espacio de trabajo abre la interfaz Matplotlib existente en modo Apply.
Editar marca el estado como `edited` y borra la curva activa hasta resolver.
`auto_solve=True` resuelve cada cambio; `model.interactive_irf(...)` mantiene ese
valor predeterminado por compatibilidad. `update({...})` agrupa cambios en una
sola resolución. Reset restaura los valores iniciales y respeta el modo Apply.

La curva base sigue identificada como referencia. Un fallo borra las curvas y
exportaciones activas; `last_success_model` conserva el último éxito explícitamente.
Los mensajes distinguen estado estacionario, determinación, insumos/dominio y
fallos numéricos, con un siguiente diagnóstico. También se muestra el fallo de
una calibración inicial alternativa. `capture_run()` guarda los resultados
seleccionados como instantáneas; no agrega cada movimiento al historial ni inventa
una receta portátil.

Arrastrar controles y pulsar botones requiere un backend interactivo de Matplotlib,
por ejemplo un backend compatible de cuaderno o de escritorio. Inline/Agg genera
imágenes estáticas; los métodos Python siguen funcionando. Los informes HTML son
capturas inspeccionables, no un servicio de ejecución en segundo plano.

## Examinar cada dimensión de precisión

`run.status == "succeeded"` indica que el motor terminó con su criterio de
convergencia. `run.accuracy` separa ecuaciones, determinación, factibilidad,
malla de hogares, horizonte, cuadratura, compresión, ramificación y derivadas.
No hay una bandera global que certifique toda la precisión.

| Estado de evidencia | Significado |
|---|---|
| `passed` / `failed` | Una comprobación se evaluó con el criterio mostrado |
| `reported` | Se guardó un valor o diagnóstico sin una nueva prueba de aceptación |
| `not_checked` | No se aportó ni evaluó evidencia para esa dimensión |

El panel reutiliza `audit_result` cuando está soportado. Las ecuaciones de primer
orden no certifican precisión estocástica global. Una captura de orden superior
solo audita el modelo de primer orden incorporado. La distancia terminal se
separa de la comparación de horizontes. Las políticas globales conservan su
auditoría puntual; converger en iteración no certifica refinamiento de malla.

En HANK, los diagnósticos existentes se identifican como evidencia registrada
del solver. Euler, compresión, masa y presupuesto requieren tolerancias explícitas
para generar una aprobación numérica. La evidencia ausente queda sin comprobar.
Al capturar un resultado externo, el usuario debe aportar sus metadatos; no se
pueden recuperar automáticamente funciones Python o calibraciones omitidas.

```python
from puremacro.dsge import solution_accuracy

checks = solution_accuracy(None)
assert set(checks.to_frame().status) == {"not_checked"}
```

Para un resultado HANK existente, use
`solution_accuracy(result, tolerances={"euler": 1e-3, "compression": 1e-4})`
con criterios justificados para el experimento. Capture con
`DSGERun.capture(result, parameters=..., configuration=..., units=..., timing=...)`,
o agregue al historial con `bench.record(...)`. Las tolerancias anteriores ilustran
la API; no son requisitos universales recomendados.

## Progreso, cancelación y recuperación

```python
from puremacro.dsge import DSGERunControl

events = []
control = DSGERunControl(max_seconds=30, on_progress=events.append)
path = bench.perfect_foresight(periods=5, y_init=[0.], y_end=[0.],
                               shocks=[[.01], [0.], [0.], [0.], [0.]],
                               linear_solver="krylov", control=control)
assert path.status == "succeeded"
assert len(events) > 0
```

`control.cancel()` solicita cancelación cooperativa. Krylov comprueba en pasos de
Newton aceptados y en los puntos de progreso internos existentes; otras operaciones
comprueban entre etapas. Las llamadas siguen siendo síncronas. `max_seconds` es
un presupuesto cooperativo del solver, no un plazo estricto; capturar la evidencia
final puede requerir tiempo adicional. Una interfaz que ocupa el hilo principal
no procesa un botón de cancelación hasta ceder ese hilo; use otro contexto de
ejecución o una función de progreso.

Cancelación y presupuesto agotado producen estados distintos (`cancelled` y
`budget_exhausted`). Tras un paso Krylov aceptado, el archivo conserva la trayectoria
y su huella de insumos. `stopped_run.replay(resume=True)` utiliza esa trayectoria
como valor inicial y repite todas las comprobaciones. No restaura los detalles
internos de Krylov ni certifica una trayectoria interrumpida. Un control nuevo
puede establecer otro presupuesto. Sin un punto de recuperación compatible y
aceptado, se rechaza la reanudación.

`python -m puremacro.examples.dsge_workbench` compara dos respuestas AR(1) con
solución analítica. Agregue `--out DIRECTORIO` para exportar archivos, receta Python,
informes HTML y figuras. Consulte [Flujos DSGE](dsge_workflows.md) para el alcance
matemático de los demás solvers.
