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
browser), copies the showcase notebooks (`notebooks/NN_*.ipynb`) **and** the
course lessons (`notebooks/course/*.ipynb`) plus `_nbstyle.py` / `_tutor.py`
into `content/`, injects a `%pip install puremacro` bootstrap cell into **every**
`NN…`-prefixed notebook it copied (`%pip install puremacro pyarrow` in the ones
whose code reads parquet), and runs `jupyter lite build` → `dist/`.

## Deploy

`.github/workflows/pages.yml` runs `build_playground.sh` on every push to `main`
and publishes `playground/dist/` to GitHub Pages:
<https://jalonso1979.github.io/puremacro/lab/index.html>.

The browser install works only because every base dependency of `puremacro`
ships with the Pyodide distribution: the playground disables PyPI fallback
(`content_static/jupyter-lite.json`), so a base dependency that Pyodide lacks
makes the first cell of every notebook fail. Through 3.2.1 that is exactly what
happened (the wheel required `openpyxl` and `pyarrow`).
`tests/test_pyodide_compat.py::test_runtime_deps_ship_with_pyodide` now guards
it, and the opt-in Pyodide gate (`tools/pyodide/runner.js`) performs the same
dependency-resolving install. After a deploy, run the first cells of one
showcase notebook and of one parquet lesson (08 or 24) to confirm.

Whether the course offers the browser path ("Forma A" in the course's Software
page) is a teaching decision; the build no longer blocks it.

The showcase notebooks are fully synthetic and offline (Pyodide blocks network
sockets), so they run without any data files or API keys. The **course lessons**
do read small local files, so the build copies `notebooks/course/data/` into
`content/course/data/` — the layout their `DATA = _nb/"course"/"data"` expects.
Either way no network and no API key is ever needed at run time.
