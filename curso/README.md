# Macroeconomía Avanzada (MAV) · ITAM

Materiales públicos del curso *Macroeconomía Avanzada* de Jorge Alonso-Ortiz (ITAM, otoño 2026), construidos sobre [puremacro](https://jorgealonsoortiz.work/puremacro/).

**Edición del curso:** Python 3.12 o posterior y `puremacro==5.5.0`, con auxiliares, datos, modelos y caché
fijados por el tag `course-2026A-5.5.0`. Las secciones `puremacro.dp` y
`puremacro.text` son obligatorias. Los 72 cuadernos se ejecutan con datos
congelados y respuestas del modelo de referencia en caché; los modelos vivos y
las actualizaciones de datos son opcionales. Consulte las
[notas de instalación](notebooks/README.md). Los laboratorios HTML/JavaScript
siguen siendo implementaciones separadas para la enseñanza.

| | |
|---|---|
| **Página del curso** | <https://jorgealonsoortiz.work/macro-avanzada/> |
| **Laboratorios interactivos** | <https://jalonso1979.github.io/puremacro/curso/> |
| **Diapositivas (PDF)** | <https://jalonso1979.github.io/puremacro/curso/diapositivas/> |
| **Cuadernos (Colab)** | [`notebooks/`](notebooks/) |

**Actualización de diapositivas del 3 de noviembre de 2026.** Los ocho PDF de [diapositivas](https://jalonso1979.github.io/puremacro/curso/diapositivas/) se publican en su versión actual, con las correcciones de las revisiones del 25 y 28 de octubre. Las [diapositivas 5](https://jalonso1979.github.io/puremacro/curso/diapositivas/MAV_05_identificacion_empirica.pdf) (93 láminas) añaden un bloque sobre multiplicadores fiscales dependientes del régimen —una réplica de Ramey–Zubairy con todo interactuado y diagnósticos de instrumento por régimen— y un cierre sobre qué choques mueven el ciclo. Los cuadernos no cambian: siguen en la edición `course-2026A-5.5.0` (T02_A–D en `v5.7.1`), y el código del laboratorio nuevo corre con `puremacro==5.5.0`.

**Actualización DSGE del 23 de octubre de 2026.** Los ocho cuadernos T02_A–D en español e inglés fijan su código, auxiliares y modelos a `v5.7.1`. Comparten ecuaciones entre aproximaciones locales, previsión perfecta y políticas globales; incluyen ejecución ordenada de `.mod`, continuación, anuncios y auditorías independientes. Use la [instalación para T02_A–D](notebooks/README.md) y la [guía de métodos](../docs/es/dsge_workflows.md). La edición completa original de 5.5.0 conserva su tag y su evidencia.

**Antecedente: 22 de octubre de 2026.** [T02_D](notebooks/T02_D_occbin_zlb_regimes_es.ipynb) añade un RBC con inversión irreversible: §8 aplica OccBin a un choque de productividad realizado y §9 resuelve un choque de volatilidad estocástica con `puremacro.vfi.Case2Problem`. [Las diapositivas 4](https://jalonso1979.github.io/puremacro/curso/diapositivas/MAV_04_ciclos_economicos_reales_y_contabilidad_del_ciclo.pdf#page=43) desarrollan el álgebra, el método y las respuestas. Esa ejecución histórica usó puremacro 5.5.0 y auxiliares congelados; la revisión actual usa 5.7.1; el tag `course-2026A-5.5.0` conserva su contenido original. Véase la [validación adicional](validation/README.md).

## Qué hay aquí

- `site/`: el sitio estático de laboratorios que publica GitHub Pages en `/puremacro/curso/`. HTML, CSS y JavaScript sin paso de compilación. Los cálculos corren en el navegador; son implementaciones y datos del curso.
- `site/diapositivas/`: los mazos del curso en PDF.
- `notebooks/`: los cuadernos T00–T09 en español (`*_es.ipynb`) y en inglés, con sus salidas. Cada uno abre en Google Colab y descarga solo esta carpeta.
- `notebooks/data_curso/`: instantáneas congeladas de datos públicos. Las fuentes y cómo citarlas están en [`FUENTES.md`](notebooks/data_curso/FUENTES.md).
- `tools/`: scripts que generan los datos de los laboratorios (`datos_*.py`) y arman el sitio (`construir_sitio.py`).

Los cuadernos conservan sus esquemas de solución para el trabajo autónomo; no incluyen exámenes ni claves internas de calificación.

## Licencia

El código es MIT, como puremacro. Las diapositivas y el texto de los cuadernos están bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es): puedes reutilizarlos y adaptarlos citando «Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM». Los datos pertenecen a sus publicadores originales: cita la fuente original, no este repositorio.

---

# Advanced Macroeconomics (MAV) · ITAM

Public materials for Jorge Alonso-Ortiz's *Advanced Macroeconomics* course (ITAM, fall 2026), built on [puremacro](https://jorgealonsoortiz.work/puremacro/). The course is taught in Spanish; the labs and notebooks also come in English.

**3 November 2026 slides update.** The eight [slide PDFs](https://jalonso1979.github.io/puremacro/curso/diapositivas/) are published in their current version, with the corrections from the 25 and 28 October reviews. [Slides 5](https://jalonso1979.github.io/puremacro/curso/diapositivas/MAV_05_identificacion_empirica.pdf) (93 slides) add a block on regime-dependent fiscal multipliers — a fully interacted Ramey–Zubairy replication with regime-specific instrument diagnostics — and a closing section on which shocks drive the business cycle. The notebooks are unchanged: they remain the `course-2026A-5.5.0` edition (T02_A–D on `v5.7.1`), and the new lab code runs with `puremacro==5.5.0`.

**23 October 2026 DSGE update.** The eight English/Spanish T02_A–D notebooks pin code, helpers and models to `v5.7.1`. They share equations across local approximations, perfect foresight and global policies, with ordered `.mod` execution, continuation, announcements and independent audits. Use the [T02_A–D setup](notebooks/README.md) and [method guide](../docs/dsge_workflows.md). The complete original 5.5.0 edition retains its tag and evidence.

**Earlier update: 22 October 2026.** [T02_D](notebooks/T02_D_occbin_zlb_regimes.ipynb) adds an RBC model with irreversible investment: §8 applies OccBin to a realized productivity shock and §9 solves a stochastic-volatility shock with `puremacro.vfi.Case2Problem`. [Slides 4](https://jalonso1979.github.io/puremacro/curso/diapositivas/MAV_04_ciclos_economicos_reales_y_contabilidad_del_ciclo.pdf#page=43) develop the algebra, method and responses. That historical execution used puremacro 5.5.0 and frozen helpers; the current revision uses 5.7.1; tag `course-2026A-5.5.0` retains its original contents. See the [additional validation](validation/README.md).

**Course edition:** Python 3.12 or later and `puremacro==5.5.0`, with helpers, data, models and cache pinned
by tag `course-2026A-5.5.0`. The `puremacro.dp` and `puremacro.text` sections are
required. All 72 notebooks execute with frozen data and cached reference-model
responses; live models and data refreshes are optional. See the
[setup notes](notebooks/README.md). The HTML/JavaScript labs remain separate
teaching implementations.

- `site/`: the interactive labs published at <https://jalonso1979.github.io/puremacro/curso/> (static HTML/JS with course implementations and data), plus the slide PDFs.
- `notebooks/`: notebooks T00–T09 in Spanish and English, runnable in Google Colab, with frozen public data in `data_curso/`. See `FUENTES.md` for sources and citation.
- `tools/`: lab data builders and the site assembler.

The notebooks retain their solution outlines for independent study; exams and internal grading keys are not included. The code is MIT-licensed; the slides and notebook text are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); data belong to their original publishers.


## Reproduce the execution audit

Install the exact wheel and notebook tools in a dedicated Python 3.12+ environment,
then run from that environment with `PYTHONPATH` unset:

```bash
python -m pip install /path/puremacro-5.5.0-py3-none-any.whl
python -m pip install nbclient nbformat ipykernel statsmodels arch scikit-learn openpyxl pyarrow
python curso/scripts/ejecutar_edicion.py --wheel /path/puremacro-5.5.0-py3-none-any.whl --work-dir /tmp/course-5.5-audit --jobs 2
```

The work directory must be new and outside the repository. The runner copies
all course inputs there, records the wheel hash, checks installed files against
the wheel byte for byte, and verifies the import location. It
executes all 72 notebooks with the environment's kernel, and records per-cell
execution plus required DP/text coverage in `summary.json` and `evidence/`.
`inputs.json` records package versions and hashes of all copied inputs;
`snapshot_integrity.json` checks frozen inputs afterwards. Network access is
blocked and missing cached reference responses fail the audit. Add `--write-back`
to retain successful executed outputs in the source notebooks. `--notebook NAME`
can repeat a particular notebook in a new directory after a correction.
The historical T01_B lessons export reconstructed national accounts to
`generated/qna/` and check their round trip there, keeping the frozen data intact.
The [edition execution manifest](validation/README.md) records the exact candidate
wheel, dependency versions, input and output hashes, and all 72 notebook outcomes.
