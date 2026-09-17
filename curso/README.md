# Macroeconomía Avanzada (MAV) · ITAM

Materiales públicos del curso *Macroeconomía Avanzada* de Jorge Alonso-Ortiz (ITAM, otoño 2026), construidos sobre [puremacro](https://jorgealonsoortiz.work/puremacro/).

| | |
|---|---|
| **Página del curso** | <https://jorgealonsoortiz.work/macro-avanzada/> |
| **Laboratorios interactivos** | <https://jalonso1979.github.io/puremacro/curso/> |
| **Diapositivas (PDF)** | <https://jalonso1979.github.io/puremacro/curso/diapositivas/> |
| **Cuadernos (Colab)** | [`notebooks/`](notebooks/) |

## Qué hay aquí

- `site/`: el sitio estático de laboratorios que publica GitHub Pages en `/puremacro/curso/`. HTML, CSS y JavaScript sin paso de compilación. Los cálculos corren en el navegador y se validaron contra puremacro.
- `site/diapositivas/`: los mazos del curso en PDF.
- `notebooks/`: los cuadernos T00–T09 en español (`*_es.ipynb`) y en inglés, con sus salidas. Cada uno abre en Google Colab y descarga solo esta carpeta.
- `notebooks/data_curso/`: instantáneas congeladas de datos públicos. Las fuentes y cómo citarlas están en [`FUENTES.md`](notebooks/data_curso/FUENTES.md).
- `tools/`: scripts que generan los datos de los laboratorios (`datos_*.py`) y arman el sitio (`construir_sitio.py`).

No hay exámenes, soluciones ni material interno del curso en este repositorio.

## Licencia

El código es MIT, como puremacro. Las diapositivas y el texto de los cuadernos están bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es): puedes reutilizarlos y adaptarlos citando «Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM». Los datos pertenecen a sus publicadores originales: cita la fuente original, no este repositorio.

---

# Advanced Macroeconomics (MAV) · ITAM

Public materials for Jorge Alonso-Ortiz's *Advanced Macroeconomics* course (ITAM, fall 2026), built on [puremacro](https://jorgealonsoortiz.work/puremacro/). The course is taught in Spanish; the labs and notebooks also come in English.

- `site/`: the interactive labs published at <https://jalonso1979.github.io/puremacro/curso/> (static HTML/JS, validated against puremacro), plus the slide PDFs.
- `notebooks/`: notebooks T00–T09 in Spanish and English, runnable in Google Colab, with frozen public data in `data_curso/`. See `FUENTES.md` for sources and citation.
- `tools/`: lab data builders and the site assembler.

No exams, solutions or internal course material are included. The code is MIT-licensed; the slides and notebook text are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); data belong to their original publishers.
