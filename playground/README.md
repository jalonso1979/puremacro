# puremacro browser playground (JupyterLite)

A zero-install, zero-key, zero-cloud sandbox: `puremacro`'s pure-compute core
running entirely in your browser via JupyterLite + Pyodide (WebAssembly).

## Build

```bash
cd ..                      # the puremacro project root
pip install -e ".[playground,notebooks]"
bash playground/build_playground.sh
python -m http.server -d playground/dist 8000   # then open http://localhost:8000/lab
```

`build_playground.sh` builds the `puremacro` wheel, pins it via
`PipliteAddon.piplite_urls` (so `%pip install puremacro` resolves offline in the
browser), copies the showcase notebooks (`notebooks/00..19_*.ipynb`) **and** the
course lessons (`notebooks/course/*.ipynb`) plus `_nbstyle.py` / `_tutor.py`
into `content/`, injects a `%pip install puremacro` bootstrap cell into **every**
`NN…`-prefixed notebook it copied, and runs `jupyter lite build` → `dist/`.

## Deploy — NOT DEPLOYED (this is a course blocker, not a nicety)

**There is no public URL today.** `dist/` is built locally and nothing publishes
it: the repo has no remote and no commits, so the Pages workflow has never run.
Anything that promises students a browser-only, zero-install path ("Forma A" in
the course's Software page) is promising this playground, and it does not exist
at any address yet.

To publish on GitHub Pages: push the branch, enable Pages, and add a workflow at
the **repo root** that runs `build_playground.sh` and uploads `playground/dist/`.

TODO(profesor): dos decisiones que sólo tú puedes tomar, y hasta que se tomen la
"Forma A" del curso no puede anunciarse.
  1. Camino de publicación del repo (repo aparte / PyPI desde subdirectorio /
     sólo GitHub) y cuenta bajo la que se publica.
  2. Si el playground NO se despliega antes del inicio del curso: hay que
     retirar la "Forma A" de la página de Software y de los enunciados, y dejar
     la instalación local (Forma B) como único camino.

The showcase notebooks are fully synthetic and offline (Pyodide blocks network
sockets), so they run without any data files or API keys. The **course lessons**
do read small local files, so the build copies `notebooks/course/data/` into
`content/course/data/` — the layout their `DATA = _nb/"course"/"data"` expects.
Either way no network and no API key is ever needed at run time.
