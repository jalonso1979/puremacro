# Release verification commands

Run from the repository root unless stated otherwise. Dependency resolution and
notebook kernels require network/local-port permissions in the execution sandbox.

```sh
uv venv --python 3.12 /tmp/puremacro-release-4.3.0-venv
uv pip install --python /tmp/puremacro-release-4.3.0-venv/bin/python -e '.[dev,notebooks]' build twine mkdocs-material
JUPYTER_DATA_DIR=/tmp/puremacro-release-jupyter IPYTHONDIR=/tmp/puremacro-release-ipython MPLBACKEND=Agg MPLCONFIGDIR=/tmp/puremacro-release-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /tmp/puremacro-release-4.3.0-venv/bin/python -u tools/release_check.py
/tmp/puremacro-release-4.3.0-venv/bin/python -m mkdocs build --strict --site-dir /tmp/puremacro-release-4.3.0-docs
```

The clean build used `git archive` of the implementation commit, extracted into
an empty directory, followed there by `python -m build` and `python -m twine check`
on each artifact. A separate environment installed the wheel by absolute path;
`python -I` smoke checks ran from `/tmp`, preventing imports from the checkout.
The same installed-wheel interpreter ran
`tools/reference_validation/validate_dynare.py --output <report-path>` from the
clean source export against its frozen fixtures.

The final tag is created only after the full baseline gate completes. Publication
uses only `git push origin HEAD:main` and `git push origin refs/tags/v4.3.0`.
`release.yml` publishes via PyPI Trusted Publishing; `pages.yml` deploys the docs
and a playground containing the wheel built from that commit. Post-publication
verification must reinstall `puremacro==4.3.0` from PyPI with cache disabled and
confirm the deployed docs and Piplite wheel index.

Additional WebAssembly checks, using `tools/pyodide` dependencies:

```sh
node tools/pyodide/runner.js --wheel /absolute/path/to/puremacro-4.3.0-py3-none-any.whl
node reviews/2026-09-20-release-4.3.0/pyodide_policy_check.cjs /absolute/path/to/puremacro-4.3.0-py3-none-any.whl
node reviews/2026-09-20-release-4.3.0/pyodide_dynare_check.cjs /absolute/path/to/puremacro-4.3.0-py3-none-any.whl
python -I reviews/2026-09-20-release-4.3.0/installed_wheel_smoke.py
```

The final command needs an environment containing the installed wheel. The Node
checks use the repository harness's Pyodide 0.28.3, not the newer playground
kernel distribution. Their scope is the listed cases.
