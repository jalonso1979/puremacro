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
python -m build --wheel -o "$PG/wheels"
WHEEL="$(ls "$PG"/wheels/puremacro-*.whl | head -1)"
echo "    wheel: $WHEEL"

echo "==> 2/5 assemble content/ (showcase notebooks + style + offline config)"
rm -rf "$PG/content"; mkdir -p "$PG/content"
cp "$ROOT"/notebooks/[0-1][0-9]_*.ipynb "$PG/content/"
cp "$ROOT"/notebooks/course/[0-9]*.ipynb "$PG/content/" 2>/dev/null || true   # course companion lessons (EN/ES)
cp "$ROOT"/notebooks/_nbstyle.py "$PG/content/"
cp "$ROOT"/notebooks/course/_tutor.py "$PG/content/" 2>/dev/null || true       # offline AI-tutor helper (graceful in-browser)
cp "$PG"/content_static/jupyter-lite.json "$PG/content/"
python -m jupytext --to ipynb --output "$PG/content/00_start_here.ipynb" "$PG/00_start_here.py"

echo "==> 3/5 inject the %pip bootstrap cell into each showcase notebook"
python "$PG/inject_pip_cell.py" "$PG"/content/[0-1][0-9]_*.ipynb

echo "==> 4/5 jupyter lite build"
cd "$PG"
# jupyter lite reads jupyter_lite_config.json (PipliteAddon) from this dir.
jupyter lite build --contents ./content --output-dir ./dist

echo "==> 5/5 done"
echo "    serve locally:  python -m http.server -d \"$PG/dist\" 8000"
echo "    then open:      http://localhost:8000/lab/index.html"
