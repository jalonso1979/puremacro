#!/usr/bin/env python3
"""Arma el sitio de laboratorios de MAV: datos, enlaces a diapositivas e índices.

1. Corre cada tools/datos_*.py (escriben site/data/<lab>.json).
2. Reescribe los enlaces <a href="../diapositivas/" data-deck="N"> de cada laboratorio
   al PDF real que lista site/diapositivas/index.json.
3. Escribe site/index.html (portada de laboratorios) y site/diapositivas/index.html.
4. Revisa que cada laboratorio tenga su .js y sus datos, y que no queden enlaces sin resolver.

    python tools/construir_sitio.py [--datos curso/notebooks/data_curso] [--sin-datos]
"""
import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SITIO = AQUI.parent / "site"
CURSO_WP = "https://jorgealonsoortiz.work/macro-avanzada/"
# Mientras la página del curso no esté publicada en WordPress, los enlaces «El curso» apuntan a la
# portada de laboratorios. Al publicarla, cambia esto a True y vuelve a correr el script.
CURSO_PUBLICADO = False
NOTEBOOKS_GH = "https://github.com/jalonso1979/puremacro/tree/main/curso/notebooks"

# Orden y resumen de cada laboratorio en la portada. El título y la unidad se leen del propio HTML.
LABS = [
    ("filtros", "M3 26c6 0 7-16 13-16s7 16 13 16 7-12 13-12 6 8 12 8",
     {"es": "Hodrick–Prescott frente a Hamilton en 11 países, y cuánto se revisa el ciclo cuando llegan datos nuevos.",
      "en": "Hodrick–Prescott versus Hamilton in 11 countries, and how much the cycle is revised as new data arrive."}),
    ("neoclasico", "M3 34C14 30 22 20 30 16s18-8 30-9",
     {"es": "Calcula la senda de silla del modelo de Ramsey sin linealizar y mira dónde falla la aproximación de primer orden y cuánto tarda la economía en converger.",
      "en": "Compute the Ramsey model's saddle path without linearizing, and see where the first-order approximation fails and how long convergence takes."}),
    ("incertidumbre", "M3 34h8V26h8V12h8V6h8v8h8v12h8v8h11",
     {"es": "Discretiza un AR(1) con Tauchen y con Rouwenhorst y descubre por qué el primero exagera la volatilidad y se atora cuando la persistencia es alta.",
      "en": "Discretize an AR(1) with Tauchen and with Rouwenhorst, and see why the former overstates volatility and gets stuck when persistence is high."}),
    ("rbc", "M3 8c10 4 16 18 26 22s20 4 34 4",
     {"es": "Resuelve el RBC canónico, compara trabajo divisible e indivisible y pon sus momentos del ciclo junto a los de EE. UU.: la correlación horas–productividad es la gran falla.",
      "en": "Solve the canonical RBC model, compare divisible and indivisible labor, and set its cycle moments next to US data: the hours–productivity correlation is the big miss."}),
    ("identificacion", "M3 20c8-14 16-14 24-4s16 14 36 4",
     {"es": "Con una verdad conocida, mira cómo el orden de Cholesky puede invertir el signo de un choque monetario y cómo una proyección local instrumentada cambia sesgo por varianza.",
      "en": "With a known truth, watch a Cholesky ordering flip the sign of a monetary shock and an instrumented local projection trade bias for variance."}),
    ("desigualdad", "M3 34C24 34 40 30 52 20S60 6 63 4",
     {"es": "Resuelve el modelo de Aiyagari en tu navegador y mira cómo el riesgo que no se puede asegurar baja la tasa de interés, concentra la riqueza y hace que la PMC agregada dependa de la distribución.",
      "en": "Solve the Aiyagari model in your browser and watch uninsurable risk lower the interest rate, concentrate wealth, and make the aggregate MPC depend on the distribution."}),
    ("nuevo-keynesiano", "M3 20h6c6 0 8 12 16 12s14-8 38-8",
     {"es": "Mueve la rigidez de Calvo y la regla de Taylor, compara choques monetarios, de demanda y de costos, y mira cómo la cota cero convierte el mismo choque en una recesión profunda.",
      "en": "Move Calvo stickiness and the Taylor rule, compare monetary, demand and cost-push shocks, and watch the zero bound turn the same shock into a deep slump."}),
    ("mercado-laboral", "M5 10c6 10 14 16 22 18s16 2 24 0 10-6 12-10",
     {"es": "Recorre la curva de Beveridge de EE. UU. y Europa, compara el desempleo con el que dictan los flujos y cambia un flujo de la ENOE para ver la informalidad mexicana como un equilibrio.",
      "en": "Trace the Beveridge curve for the US and Europe, compare unemployment with the level the flows imply, and change one ENOE flow to see Mexico's informality as an equilibrium."}),
]

HEAD = """<!doctype html>
<html lang="es" data-lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title data-title-es="{title_es}" data-title-en="{title_en}">{title_es}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23111'/%3E%3Cpath d='M5 21c4 0 5-10 9-10s5 10 9 10 3-5 4-5' fill='none' stroke='%23fff' stroke-width='2.5' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{root}assets/mav.css">
</head>
<body>
<header class="topbar">
  <div class="wrap">
    <a class="brand" href="{root}index.html">MAV <span class="es">· Laboratorios</span><span class="en">· Labs</span></a>
    <nav class="topnav" aria-label="Principal">
      <a href="{root}diapositivas/index.html" class="hide-sm"><span class="es">Diapositivas</span><span class="en">Slides</span></a>
      <a href="{notebooks}" class="hide-sm"><span class="es">Cuadernos</span><span class="en">Notebooks</span></a>
      <a href="{curso}" class="hide-sm"><span class="es">El curso</span><span class="en">The course</span></a>
      <span class="lang-toggle" role="group" aria-label="Idioma / Language">
        <button type="button" data-lang="es">ES</button><button type="button" data-lang="en">EN</button>
      </span>
    </nav>
  </div>
</header>
<main>
"""

FOOT = """</main>
<footer class="footer">
  <div class="wrap">
    <span class="es">Macroeconomía Avanzada · ITAM · Jorge Alonso-Ortiz. Código MIT; textos y diapositivas <a href="https://creativecommons.org/licenses/by/4.0/deed.es">CC BY 4.0</a>. Fuente en</span><span class="en">Advanced Macroeconomics · ITAM · Jorge Alonso-Ortiz. Code MIT; text and slides <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Source at</span>
    <a href="https://github.com/jalonso1979/puremacro/tree/main/curso">github.com/jalonso1979/puremacro</a>.
  </div>
</footer>
<script defer src="{root}assets/mav.js"></script>
</body>
</html>
"""


def bi(es, en, tag="span"):
    return f'<{tag} class="es">{es}</{tag}><{tag} class="en">{en}</{tag}>'


def spans(fragment, cls):
    m = re.search(rf'<span class="{cls}">(.*?)</span>', fragment, re.S)
    return m.group(1).strip() if m else ""


def correr_datos(datos):
    for script in sorted(AQUI.glob("datos_*.py")):
        args = [sys.executable, str(script)] + (["--datos", str(datos)] if datos else [])
        print(f"· {script.name}")
        subprocess.run(args, check=True)


def mazos():
    idx = json.loads((SITIO / "diapositivas" / "index.json").read_text())
    return {m["numero"]: m for m in idx["mazos"]}


def enlazar_diapositivas(M):
    patron = re.compile(r'href="\.\./diapositivas/[^"]*"(\s+data-deck="(\d+)")')
    for f in sorted((SITIO / "labs").glob("*.html")):
        s = f.read_text()
        nuevo = patron.sub(lambda m: f'href="../diapositivas/{M[int(m.group(2))]["archivo"]}"{m.group(1)}', s)
        if nuevo != s:
            f.write_text(nuevo)
            print(f"· diapositivas enlazadas en {f.name}")


def leer_lab(slug):
    s = (SITIO / "labs" / f"{slug}.html").read_text()
    kicker = re.search(r'<p class="kicker">(.*?)</p>', s, re.S).group(1)
    h1 = re.search(r"<h1>(.*?)</h1>", s, re.S).group(1)
    return {"unidad_es": spans(kicker, "es"), "unidad_en": spans(kicker, "en"),
            "titulo_es": spans(h1, "es"), "titulo_en": spans(h1, "en")}


def portada(M):
    tarjetas = []
    for slug, trazo, resumen in LABS:
        if not (SITIO / "labs" / f"{slug}.html").exists():
            print(f"  (falta labs/{slug}.html: se omite de la portada)")
            continue
        d = leer_lab(slug)
        tarjetas.append(f"""      <a class="card" href="labs/{slug}.html">
        <span class="unit">{bi(d['unidad_es'], d['unidad_en'])}</span>
        <svg viewBox="0 0 66 40" preserveAspectRatio="none" aria-hidden="true"><path d="{trazo}"/></svg>
        <h3>{bi(d['titulo_es'], d['titulo_en'])}</h3>
        <p>{bi(html.escape(resumen['es']), html.escape(resumen['en']))}</p>
        <span class="go">{bi('Abrir →', 'Open →')}</span>
      </a>""")
    cuerpo = f"""  <section class="hero wrap">
    <p class="kicker">{bi('ITAM · Otoño 2026', 'ITAM · Fall 2026')}</p>
    <h1>{bi('Laboratorios de Macroeconomía Avanzada', 'Advanced Macroeconomics labs')}</h1>
    <p class="lead">{bi('Modelos y datos del curso que se resuelven en tu navegador. Mueve un parámetro, mira qué cambia y sigue los experimentos guiados. Nada que instalar.',
                        'The course’s models and data, solved in your browser. Move a parameter, watch what changes and follow the guided experiments. Nothing to install.')}</p>
    <div class="btn-row">
      <a class="btn btn-primary" href="{CURSO_WP}">{bi('El curso completo →', 'The full course →')}</a>
      <a class="btn" href="diapositivas/index.html">{bi('Diapositivas', 'Slides')}</a>
      <a class="btn" href="{NOTEBOOKS_GH}">{bi('Cuadernos en Colab', 'Notebooks in Colab')}</a>
      <a class="btn" href="https://jorgealonsoortiz.work/puremacro/">puremacro</a>
    </div>
  </section>
  <section class="section band rule-top">
    <div class="wrap grid-cards">
{chr(10).join(tarjetas)}
    </div>
  </section>
  <section class="section wrap two">
    <div>
      <h2>{bi('Cómo usarlos', 'How to use them')}</h2>
      <ul class="try">
        <li>{bi('Empieza por la pregunta del título y haz una predicción antes de mover nada.', 'Start with the question in the title and make a prediction before moving anything.')}</li>
        <li>{bi('Sigue los experimentos de «Qué probar»: cada uno aísla una idea de la unidad.', 'Follow the "Things to try" experiments: each one isolates one idea from the unit.')}</li>
        <li>{bi('Cada gráfica tiene tabla y CSV. La matemática está al pie, y el cuaderno de Colab reproduce el cálculo en Python.', 'Every chart has a table and a CSV. The math is at the bottom, and the Colab notebook reproduces the computation in Python.')}</li>
      </ul>
    </div>
    <div>
      <h2>{bi('Con qué están hechos', 'What they are built with')}</h2>
      <p class="muted">{bi('Los cálculos corren en JavaScript dentro de la página y se validaron contra <a href="https://jorgealonsoortiz.work/puremacro/">puremacro</a>, la biblioteca de Python del curso. Los datos son instantáneas congeladas de fuentes públicas; cada laboratorio cita la suya.',
                           'The computations run in JavaScript inside the page and were validated against <a href="https://jorgealonsoortiz.work/puremacro/">puremacro</a>, the course’s Python library. The data are frozen snapshots from public sources; each lab cites its own.')}</p>
    </div>
  </section>
"""
    head = HEAD.format(title_es="Laboratorios · Macroeconomía Avanzada", title_en="Labs · Advanced Macroeconomics",
                       desc="Laboratorios interactivos del curso Macroeconomía Avanzada (ITAM): filtros, crecimiento, RBC, identificación, agentes heterogéneos, nuevo keynesiano y mercado laboral.",
                       root="", notebooks=NOTEBOOKS_GH, curso=CURSO_WP)
    (SITIO / "index.html").write_text(head + cuerpo + FOOT.format(root=""))
    print(f"· index.html con {len(tarjetas)} laboratorios")


def pagina_diapositivas(M):
    filas = []
    for n in sorted(M):
        m = M[n]
        filas.append(f"""      <a class="card" href="{m['archivo']}">
        <span class="unit">{bi('Mazo', 'Deck')} {n} · {bi('semanas', 'weeks')} {m.get('semanas', '')}</span>
        <h3>{bi(html.escape(m['titulo_es']), html.escape(m['titulo_en']))}</h3>
        <p>PDF · {m['paginas']} {bi('láminas', 'slides')} · {m['bytes'] / 1e6:.1f} MB</p>
        <span class="go">{bi('Descargar →', 'Download →')}</span>
      </a>""")
    cuerpo = f"""  <section class="hero wrap">
    <p class="kicker">{bi('Macroeconomía Avanzada · ITAM', 'Advanced Macroeconomics · ITAM')}</p>
    <h1>{bi('Diapositivas', 'Slides')}</h1>
    <p class="lead">{bi('Los ocho mazos del curso, en PDF. Se actualizan durante el semestre.', 'The course’s eight slide decks, as PDFs, in Spanish. They are updated during the semester.')}</p>
  </section>
  <section class="section band rule-top">
    <div class="wrap grid-cards">
{chr(10).join(filas)}
    </div>
  </section>
"""
    head = HEAD.format(title_es="Diapositivas · Macroeconomía Avanzada", title_en="Slides · Advanced Macroeconomics",
                       desc="Diapositivas del curso Macroeconomía Avanzada (ITAM), en PDF.",
                       root="../", notebooks=NOTEBOOKS_GH, curso=CURSO_WP)
    (SITIO / "diapositivas" / "index.html").write_text(head + cuerpo + FOOT.format(root="../"))
    print(f"· diapositivas/index.html con {len(filas)} mazos")


def versionar_recursos():
    """Añade ?v=<hash del contenido> a los .js y .css locales, para que GitHub Pages no sirva versiones viejas."""
    import hashlib
    patron = re.compile(r'((?:src|href)=")((?:\.\./)?(?:assets/)?[\w.-]+\.(?:js|css))(?:\?v=[0-9a-f]+)?(")')
    for f in [SITIO / "index.html", SITIO / "diapositivas" / "index.html", *sorted((SITIO / "labs").glob("*.html"))]:
        s = f.read_text()

        def firma(m):
            ruta = (f.parent / m.group(2)).resolve()
            if not ruta.exists():
                return m.group(0)
            return f"{m.group(1)}{m.group(2)}?v={hashlib.sha1(ruta.read_bytes()).hexdigest()[:8]}{m.group(3)}"

        nuevo = patron.sub(firma, s)
        if nuevo != s:
            f.write_text(nuevo)


def enlazar_curso():
    """Apunta los enlaces al curso a WordPress, o a la portada de laboratorios si aún no está publicado.
    El destino real queda en data-curso, así que el cambio es reversible en ambos sentidos."""
    patron = re.compile(r'href="(?:https://jorgealonsoortiz\.work/macro-avanzada/|(?:\.\./)?index\.html)"'
                        r'(?: data-curso="[^"]*")?(?=[^>]*>(?:\s*<span class="es">(?:El curso|El curso completo →)</span>))')
    for f in [SITIO / "index.html", SITIO / "diapositivas" / "index.html", *sorted((SITIO / "labs").glob("*.html"))]:
        s = f.read_text()
        raiz = "" if f.parent == SITIO else "../"
        destino = CURSO_WP if CURSO_PUBLICADO else f"{raiz}index.html"
        nuevo = patron.sub(f'href="{destino}" data-curso="{CURSO_WP}"', s)
        if nuevo != s:
            f.write_text(nuevo)


def revisar():
    problemas = []
    for f in sorted((SITIO / "labs").glob("*.html")):
        s = f.read_text()
        for src in re.findall(r'<script[^>]+src="([^"?]+)', s):
            if not src.startswith("http") and not (f.parent / src).exists():
                problemas.append(f"{f.name}: falta {src}")
        for data in re.findall(r'MAV\.json\("([^"]+)"\)', (f.with_suffix(".js")).read_text() if f.with_suffix(".js").exists() else ""):
            if not (f.parent / data).exists():
                problemas.append(f"{f.stem}.js: falta {data}")
        if re.search(r'href="\.\./diapositivas/"', s):
            problemas.append(f"{f.name}: enlace a diapositivas sin resolver")
    for p in problemas:
        print("  ✗", p)
    return not problemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=None)
    ap.add_argument("--sin-datos", action="store_true")
    a = ap.parse_args()
    if not a.sin_datos:
        correr_datos(a.datos)
    M = mazos()
    enlazar_diapositivas(M)
    portada(M)
    pagina_diapositivas(M)
    enlazar_curso()
    versionar_recursos()
    sys.exit(0 if revisar() else 1)


if __name__ == "__main__":
    main()
