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
browser), copies the 10 showcase notebooks + `_nbstyle.py` into `content/`,
injects a `%pip install puremacro` bootstrap cell into each, and runs
`jupyter lite build` → `dist/`.

## Deploy (deferred — needs the release-path decision)

To publish on GitHub Pages: push the branch, enable Pages, and add a workflow at
the **repo root** that runs `build_playground.sh` and uploads `playground/dist/`.
Gated on the puremacro release decision (split repo / PyPI-from-subdir /
GitHub-only); not part of this build.

The notebooks are fully synthetic and offline (Pyodide blocks network sockets),
so the playground runs without any data files or API keys.
