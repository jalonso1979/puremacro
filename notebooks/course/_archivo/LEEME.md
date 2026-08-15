# Archivo — material que ya no es del curso vigente

Estos archivos **no** forman parte del curso de Macroeconomía Avanzada (ITAM, Otoño 2026).
Se movieron aquí el 8 de agosto de 2026 para que no los recojan ni el descubrimiento de
lecciones de `tools/build_course_site.py` (`notebooks/course/*_es.py`) ni la construcción
de diapositivas. No se borraron: quedan como referencia histórica.

| Archivo | Por qué salió |
|---|---|
| `20_heterogeneidad_hank_es 2.py` | Duplicado huérfano de `20_heterogeneidad_hank_es.py` (el nombre lleva un espacio y un `2` de una copia del Finder). Difiere del vigente y no está enlazado desde ninguna parte. |
| `00_syllabus.py` / `00_syllabus.ipynb` | Programa en inglés del 31-may-2026, reemplazado por `00_syllabus_es.py`. |
| `01_business_cycle_facts.py` / `01_business_cycle_facts.ipynb` | Lección 01 en inglés del 31-may-2026, reemplazada por `01_business_cycle_facts_es.py`. |

Verificado antes de mover: ninguna lección vigente (`*_es.py`) los importa ni los cita.

**Pendiente para quien mantenga el paquete:** `tests/test_course.py::test_syllabus_lists_module_one`
lee `notebooks/course/00_syllabus.py` y hay que apuntarlo a `00_syllabus_es.py` (y a las
palabras en español). Con el movimiento, en cambio, dejan de fallar
`test_every_course_notebook_has_es_sibling` y `test_lessons_have_required_sections`,
que se rompían por el duplicado ` 2.py`.
