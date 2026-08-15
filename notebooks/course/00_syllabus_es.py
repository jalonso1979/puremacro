# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Curso complementario de puremacro — programa
#
# **Macroeconomía Avanzada · ITAM · Otoño 2026**
#
# Un curso complementario construido sobre **puremacro**: Python puro, ejecutable en tu
# **instalación local** (`pip install puremacro`), $0. Cada lección es *una sola fuente*
# que puedes **leer como diapositivas**, **ejecutar como cuaderno** y —de forma opcional—
# **explorar con un tutor de IA sin conexión**, si instalas un LLM local.
#
# Las lecciones **acompañan** a los ocho mazos de transparencias del curso
# (`Slides01mav` … `Slides08mav`): no los sustituyen.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Cómo usar una lección
# - **Instálala una vez:** `pip install puremacro` en tu Python local, sobre **Python 3.11
#   o posterior** (el curso no usa entornos en la nube; MATLAB y Dynare también se
#   instalan localmente).
# - **Ejecútala:** arranca Jupyter **desde la carpeta `notebooks/`** del material del curso
#   (`cd notebooks && jupyter lab`), abre `course/<lección>.ipynb` y córrelo de arriba a
#   abajo. No muevas los cuadernos de sitio: localizan sus datos por ruta relativa
#   (`notebooks/course/data/`), y así corren sin conexión.
# - **Preséntala:** las celdas están etiquetadas como diapositivas y de ahí salen las
#   presentaciones reveal.js del curso, generadas desde estos mismos cuadernos (las figuras
#   provienen del código en vivo). El generador (`tools/build_course_slides.py`) vive en el
#   repositorio de `puremacro`, no en el paquete de materiales que recibes.
# - **Pregunta al tutor:** cada lección importa `tutor` en su celda de arranque, así que
#   basta con `tutor("…")`. Responde sin conexión con un LLM local si hay uno instalado
#   (`pip install puremacro[local-llm]`); si no, te remite a las indicaciones de
#   exploración con IA.
#
# **Plantilla de la lección:** objetivos de aprendizaje → teoría concisa → uno o varios
# ejemplos trabajados con puremacro sobre datos reales → ejercicios → una sección **«Explora
# con IA»** (la traen todas las lecciones menos la 25). Varias intercalan además un
# "laboratorio" práctico.
#
# La mayoría de las lecciones cierra los ejercicios con una celda de notas del profesor
# —titulada **«Notas para las preguntas»** o **«Soluciones (esquema)»**— marcada `skip`, así
# que no aparece en las diapositivas pero sí en el cuaderno. Cinco lecciones (05, 06, 08,
# 10c y 23) todavía no la tienen: ahí las preguntas son deliberadamente abiertas y se
# discuten en clase.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## El calendario manda: 16 semanas, ocho mazos
#
# El curso avanza por bloques de una o dos semanas; cada bloque tiene un mazo de
# transparencias y una o más lecciones-cuaderno que lo acompañan. El semestre va del
# **10 de agosto al 2 de diciembre de 2026**; los exámenes finales son del 7 al 19 de
# diciembre.
#
# Fuera del itinerario quedan la **semana de fiestas patrias** (14–18 de septiembre: el
# miércoles 16 no hay sesión), los lunes inhábiles **2** y **16 de noviembre**, y la
# suspensión de labores del ITAM del **3 al 5 de diciembre**. Por eso la semana 5 cierra el
# 11 de septiembre y la semana 6 —la del primer parcial— no empieza hasta el 21.

# %% slideshow={"slide_type": "slide"}
import pandas as pd

calendario = pd.DataFrame(
    [
        ("1–2",   "10–21 ago",
                  "Medición del ciclo: contabilidad nacional, filtros y hechos estilizados",
                  "Slides01mav", "01, 01b"),
        ("3–4",   "24 ago–4 sep",
                  "El modelo neoclásico de crecimiento",
                  "Slides02mav", "02"),
        ("5",     "7–11 sep",
                  "Riesgo e incertidumbre",
                  "Slides03mav", "03"),
        ("6",     "21–25 sep",
                  "Primer examen parcial (miércoles 23 de septiembre)",
                  "—", "—"),
        ("7–8",   "28 sep–9 oct",
                  "Ciclos económicos reales y contabilidad del ciclo",
                  "Slides04mav", "04, 04b"),
        ("9–10",  "12–23 oct",
                  "Identificación empírica: SVAR, evidencia narrativa y proyecciones locales",
                  "Slides05mav", "05, 06, 09"),
        ("11",    "26–30 oct",
                  "Mercados incompletos, heterogeneidad y HANK",
                  "Slides06mav", "20"),
        ("12",    "2–6 nov",
                  "Segundo examen parcial (miércoles 4 de noviembre)",
                  "—", "—"),
        ("13–14", "9–20 nov",
                  "Mecanismos: trabajo, capital y economía pequeña y abierta",
                  "Slides07mav", "07, 10, 10b"),
        ("15–16", "23 nov–2 dic",
                  "Mercados no competitivos: precios rígidos y fricciones laborales",
                  "Slides08mav", "08, 10c, 11, 24"),
        ("finales", "7–19 dic",
                  "Examen final teórico, en clase (B1–B2); examen final computacional, "
                  "para llevar (A6 y la parte empírica)",
                  "—", "—"),
    ],
    columns=["semana", "fechas", "tema", "mazo", "lecciones"],
).set_index("semana")
calendario

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Entregas y fechas clave
#
# | Cuándo | Qué |
# |---|---|
# | Semana 2 | Sale la Tarea 1 |
# | Semana 5 | Entrega de la Tarea 1 |
# | Semana 6 | **Primer examen parcial** (miércoles 23 de septiembre) |
# | Semana 8 | Sale la Tarea 2 |
# | Semana 10 | Entrega de la Tarea 2 |
# | Semana 11 | Sale la Tarea 3 y **sale la auditoría adversarial con IA** |
# | Semana 12 | **Segundo examen parcial** (miércoles 4 de noviembre) |
# | Semana 13 | Entrega de la Tarea 3; sale el proyecto final |
# | Semana 14 | Sale la Tarea 4; **entrega de la auditoría IA (viernes 20 de noviembre)** |
# | Semana 16 | Entrega de la Tarea 4 y del optativo **«Para subir nota»** (miércoles 2 de diciembre); presentación del proyecto final |
# | Finales (7–19 dic) | Examen final teórico (en clase) y examen final computacional (para llevar, con una ventana de entre cuatro y seis días); entrega escrita del proyecto final |
#
# Las fechas exactas del periodo de finales, y la apertura y el cierre de la ventana del
# examen final computacional, se anuncian en Canvas.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Evaluación
#
# | Componente | Ponderación |
# |---|---|
# | Tareas | 20 % |
# | Auditoría adversarial con IA | 10 % |
# | Proyecto | 10 % |
# | Primer examen parcial (semana 6) | 15 % |
# | Segundo examen parcial (semana 12) | 15 % |
# | Examen final teórico | 15 % |
# | Examen final computacional | 15 % |
# | **Total** | **100 %** |
#
# Hay **dos parciales**, no uno.
#
# **Cobertura de cada examen.** El **primer parcial** cubre las unidades A1–A3; el
# **segundo parcial** cubre A4–A5 (la unidad A6 —mercados incompletos y HANK— *no* entra
# en el segundo parcial); el **examen final teórico** cubre B1–B2; y el **examen final
# computacional** cubre A6 junto con la parte empírica del curso.
#
# La clave de las unidades es la de los mazos: **A1–A6 = `Slides01mav` … `Slides06mav`**
# (semanas 1–2, 3–4, 5, 7–8, 9–10 y 11) y **B1–B2 = `Slides07mav`–`Slides08mav`**
# (semanas 13–14 y 15–16).
#
# **Ejercicio optativo «Para subir nota».** Aparte del 100 % hay un ejercicio **optativo**,
# «Para subir nota», que se entrega junto con la Tarea 4 el **miércoles 2 de diciembre**.
# No forma parte de los siete rubros de la tabla: no entregarlo no cuesta puntos, y
# entregarlo bien ajusta al alza la calificación final del curso, hasta el tope que se
# anuncia en Canvas. No debe confundirse con las secciones tituladas «Para subir nota» que
# trae dentro cada tarea y el proyecto final, que suben la nota de esa tarea o de ese
# proyecto y sí viven dentro del 100 %.
#
# El documento que manda es el **programa oficial del curso**; ahí vive además la sección de
# logística y políticas (sesiones, asesoría, entregas, exámenes y uso de IA), y Canvas es el
# canal oficial por donde se anuncian todos los cambios.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Módulos electivos
#
# Además del itinerario semanal hay cuatro lecciones de profundización que **no** están
# asignadas a ninguna semana y se pueden cursar en cualquier orden: **21** (IA y KORV),
# **22** (riesgo de orden 3 y desastres), **23** (Growth-at-Risk) y **25** (el benchmark de
# programación dinámica, VFI).
#
# Que no estén asignadas no significa que sus temas queden fuera del temario: el programa
# oficial ve *Growth-at-Risk* en la **semana 5** —con la lección 03— y la *iteración de la
# función de valor* (VFI) en la **semana 4** —con la lección 02—. Las lecciones 23 y 25
# profundizan en ellos.
#
# La lección **24 — mercado laboral de cuatro estados (matriz 4×4 F/I/U/N de la ENOE)**
# **sí** forma parte del calendario: es el contenido B2(ii) de la semana 16 en el programa
# oficial y el cierre metodológico del curso.
#
# **Empieza con la lección 01 → `01_business_cycle_facts_es`.**
