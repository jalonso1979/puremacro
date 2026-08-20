#!/usr/bin/env bash
# Build the puremacro JupyterLite browser playground into playground/dist/.
# Run from the puremacro project root:  bash playground/build_playground.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # puremacro project root
PG="$ROOT/playground"
cd "$ROOT"

echo "==> 1/5 build the wheel"
# Build into wheels/ (NOT pypi/): JupyterLite auto-indexes <lite_dir>/pypi/, so
# placing the wheel there AND listing it in PipliteAddon.piplite_urls would
# register it twice and abort the build ("two tasks, common target"). wheels/ is
# referenced only by the explicit piplite_urls in jupyter_lite_config.json.
rm -rf "$PG/wheels"; mkdir -p "$PG/wheels"
pip wheel . -w "$PG/wheels" --no-deps
WHEEL="$(ls "$PG"/wheels/puremacro-*.whl | head -1)"
echo "    wheel: $WHEEL"

# Re-pin PipliteAddon on the wheel we just built. The version used to be
# hard-coded in jupyter_lite_config.json and drifted (it still said 0.92.0 while
# pyproject was at 0.93.0): piplite then points at a file that `rm -rf wheels`
# just deleted, so `%pip install puremacro` in the browser either fails or
# silently falls back to PyPI over the network — exactly what the offline
# playground promises not to need. Rewriting the pin here keeps them in sync
# forever and preserves any other key in the config.
python - "$PG/jupyter_lite_config.json" "$(basename "$WHEEL")" <<'PY'
import json, pathlib, sys
cfg_path, wheel_name = pathlib.Path(sys.argv[1]), sys.argv[2]
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
cfg.setdefault("PipliteAddon", {})["piplite_urls"] = [f"./wheels/{wheel_name}"]
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"    piplite pin: ./wheels/{wheel_name}")
PY

echo "==> 2/5 assemble content/ (showcase notebooks + style + offline config)"
rm -rf "$PG/content"; mkdir -p "$PG/content"
cp "$ROOT"/notebooks/[0-9][0-9]_*.ipynb "$PG/content/"
cp "$ROOT"/notebooks/course/[0-9]*.ipynb "$PG/content/" 2>/dev/null || true   # course companion lessons (EN/ES)
cp "$ROOT"/notebooks/_nbstyle.py "$PG/content/"
cp "$ROOT"/notebooks/course/_tutor.py "$PG/content/" 2>/dev/null || true       # offline AI-tutor helper (graceful in-browser)
# Course lessons resolve their data as `_nb/"course"/"data"`, and in the flat
# content/ dir `_nb` is content/ itself (it holds _nbstyle.py). So the CSV/parquet
# files must land in content/course/data/ or lessons 01b..24 die on the first
# read — Pyodide has no sockets, the files have to travel with the build.
if [ -d "$ROOT/notebooks/course/data" ]; then
  mkdir -p "$PG/content/course"
  cp -R "$ROOT/notebooks/course/data" "$PG/content/course/"
fi
# The offline pin (disablePyPIFallback) belongs at the lite dir ROOT, not inside
# content/. Inside the contents dir jupyterlite treats it as a config to patch at
# <dist>/content/jupyter-lite.json -- a directory that is never created -- which is
# how the Pages build died under core 0.2.3; under 0.8.x the file is ignored
# outright instead, so the setting silently never applied in either. At $PG/ it
# merges into dist/jupyter-lite.json, which is the file the browser reads.
cp "$PG"/content_static/jupyter-lite.json "$PG/jupyter-lite.json"
python -m jupytext --to ipynb --output "$PG/content/00_start_here.ipynb" "$PG/00_start_here.py"

echo "==> 3/5 inject the %pip bootstrap cell into every notebook in content/"
# The glob must cover EVERY notebook copied above, not just 00..19_: the old
# `[0-1][0-9]_*` missed both the lettered stems (01b_, 04b_, 10b_, 10c_) and the
# course lessons 20..25 (HANK is the week-11 lesson, not an elective), which then
# hit ModuleNotFoundError on their first import cell in JupyterLite.
# `[0-9][0-9]*.ipynb` matches every NN-prefixed notebook and leaves the .py
# helpers (_nbstyle.py, _tutor.py) alone. inject_pip_cell.py is idempotent.
python "$PG/inject_pip_cell.py" "$PG"/content/[0-9][0-9]*.ipynb

echo "==> 4/5 jupyter lite build"
cd "$PG"
# Wipe dist/ first: jupyterlite builds are INCREMENTAL, and a stale dist
# once shipped an old wheel in the piplite index alongside the new one
# (0.91.0 vs 0.92.0, 2026-07-20). A clean build is cheap; a stale index
# is silent.
rm -rf ./dist
# jupyter lite reads jupyter_lite_config.json (PipliteAddon) from this dir.
jupyter lite build --contents ./content --output-dir ./dist

echo "==> 5/5 done"
echo "    serve locally:  python -m http.server -d \"$PG/dist\" 8000"
echo "    then open:      http://localhost:8000/lab/index.html"
