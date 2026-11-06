> 🇬🇧 [English](../investment_research_validation.md) · 🇪🇸 Español

# Registro de validación de investigación de inversiones

Fecha de validación: 2026-11-01. Rama: `feat/investment-research`, basada en main
`0b3caf3ec44dfbf94b4db47ea0c5b18b16ce6672`. Es código aún no publicado.

| Capa | Resultado verificado |
|---|---|
| Integración numérica/API local | 128 pruebas de activos y ambas guías ejecutables: **130 aprobadas**; 86 casos ajenos u opcionales excluidos |
| Nuevos núcleos en WebAssembly | **54 aprobadas** en Pyodide real 0.28.3 / Python 3.13.2, con el código fuente y paquetes de su distribución |
| Cuadernos | Ambas lenguas ejecutadas: siete celdas y dos figuras opacas cada una, sin errores; huellas del código y paridad computacional comprobadas |
| Contratos rápidos requeridos | **5 aprobados**: importaciones, API, versiones, sintaxis Python 3.11 y mypy nativo/con destino Windows |
| Documentación | `mkdocs build --strict` aprobado |
| Wheel candidato instalado | Ambas guías fuera del árbol importado de desarrollo: **12 bloques aprobados**, resultados EN/ES idénticos y siete huellas de módulos iguales al código fuente |
| Comprobaciones corregidas de la batería general | **13 aprobadas**: tres pruebas CLI, ocho de cuadernos y dos de artefactos |

## Batería general y correcciones

<!-- BROAD_SUITE_RESULT -->
La selección predeterminada completa de 911 archivos se ejecutó en cuatro
subprocesos disjuntos: **22.462 aprobadas, 4 fallidas, 89 omitidas, 258 excluidas**.
Los cuatro fallos iniciales aprobaron tras las correcciones siguientes. Cinco
casos añadidos después están incluidos en la ejecución dirigida final. El
subproceso más largo tardó unos 17 minutos.

Al principio fallaron tres subprocesos CLI porque el árbol de trabajo no estaba
instalado en el intérprete compartido. Al proporcionar su ruta mediante
`PYTHONPATH`, aprobaron sin cambiar esas pruebas ni el código de caché. La batería
también detectó `plt.show()` en los cuadernos nuevos, contrario a la convención
existente. Ahora se muestran y cierran explícitamente las figuras; se volvieron
a ejecutar ambas lenguas y aprobaron las ocho comprobaciones del archivo de
cuadernos y las dos de artefactos. Se conservan los fallos iniciales; las
repeticiones dirigidas no se presentan como una segunda batería completa sin fallos.

La revisión también corrigió repetición con zonas horarias/horario de verano,
calendarios festivos personalizados, columnas fundamentales duplicadas,
desbordamiento de razones pese a entradas finitas y covarianzas casi indefinidas
que producían riesgo negativo. Las pruebas numéricas finales locales y en
WebAssembly incluyen estas correcciones.

## Evidencia y reproducción

- El [registro del wheel instalado](../assets/investment_research/installed_wheel.json)
  contiene huellas de módulos/guías, intérprete, versiones, huella de repetición,
  escenarios y huella del wheel candidato.
- El [registro Pyodide](../assets/investment_research/pyodide.json) contiene el
  entorno real y su resultado pytest.
- El [registro conjunto](../assets/investment_research/validation.json) conserva
  la batería general, correcciones y límites de validación.

Entorno local: CPython 3.11.17 en macOS 27.2 arm64, NumPy 2.4.6, SciPy 1.17.1 y
pandas 3.1.0. Los registros muestran advertencias existentes de frecuencias pandas
y de gráficos sin interfaz; no son fallos de las pruebas.

```bash
PYTHONPATH=. python -m pytest -q tests/test_asset_pricing
python -m pytest -q tests/test_docs_code_blocks.py -k investment_research
python tools/release_check.py --no-tests --mypy-floor
python -m mkdocs build --strict --site-dir /tmp/investment-docs
node tools/reference_validation/investment_research_pyodide.cjs
```

El comando Pyodide utiliza `tools/pyodide/node_modules/pyodide`, instalado con el
manifiesto existente de ese directorio, o una ruta explícita como argumento.
Ejecuta los seis archivos numéricos nuevos. Cuatro pruebas de flujo también
están marcadas para la selección Pyodide existente de publicación.

Para verificar el wheel, constrúyalo e instálelo sin dependencias en un destino
aislado usando el entorno numérico anterior. Desde otro directorio:

```bash
PYTHONPATH=/path/to/installed python /checkout/tools/reference_validation/check_investment_research.py \
  --expected-package-root /path/to/installed \
  --wheel /path/to/puremacro-6.1.0-py3-none-any.whl \
  --output /tmp/investment-research.json
```

El verificador comprueba la ruta del paquete candidato. Ejecuta ejemplos confiables
del repositorio; no aísla código no confiable. Los cuadernos se reconstruyen con
`tools/build_notebooks.py investment_research investment_research_es`; aísle la
configuración Jupyter/IPython para conservar la del usuario.

## Alcance de la evidencia

La contabilidad usa cálculos manuales y posiciones monetarias independientes.
La asignación usa soluciones analíticas, malla independiente, productos exteriores
para covarianza y diagnósticos de optimización. Indicadores y creencias usan
recurrencias escalares y cambios de entradas futuras. Los fundamentos comprueban
disponibilidad y revisiones; la valoración usa identidades independientes de valor
presente. Son comprobaciones numéricas y temporales con entradas sintéticas.

No se acredita rentabilidad empírica, normalización de proveedores en vivo,
universos sin sesgo de supervivencia, creencias estimadas ni una trayectoria
caótica digitalizada. WebAssembly se ejecutó en Node, no en una interfaz iPad o
navegador. El tipado para Windows no equivale a ejecución Windows. El wheel local
se verificó, no se publicó; la matriz CI de plataformas y la aceptación de una
publicación siguen siendo requisitos separados.
