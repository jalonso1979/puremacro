# Pyodide harness for puremacro Gate 6

This directory holds the headless Pyodide runner used by
`tools/release_check.py --pyodide` (Gate 6).

## One-time setup

```bash
cd tools/pyodide
npm install
```

This downloads the pinned Pyodide build (~150 MB into `node_modules/`,
gitignored). Subsequent gate runs reuse the cache.

## Requirements

- Node.js ≥ 18 (Pyodide's requirement).
- npm (bundled with Node).

## Pyodide version

Pinned exactly in `package.json` (no `^` range). Pyodide releases tie to
specific numpy / scipy / pandas / matplotlib versions, so a floating
range would invite hidden drift. Bumping is a deliberate maintainer
act; re-run Gate 6 immediately after bumping.

## Contract

`node runner.js --wheel <absolute-path-to-puremacro-*.whl>` emits one
JSON document to stdout:

```json
{
  "schema_version": 1,
  "pyodide_version": "0.28.3",
  "loaded_at": "2026-05-22T15:00:00Z",
  "wheel_installed": true,
  "wheel_path": "/tmp/.../puremacro-0.49.0-py3-none-any.whl",
  "pytest_returncode": 0,
  "passed": 8,
  "failed": 0,
  "skipped": 0,
  "runtime_s": 84.2,
  "stdout_tail": "============ 8 passed in 12.4s ============"
}
```

Exit 0 if the JSON envelope was emitted (regardless of
`pytest_returncode`); non-zero only if Pyodide failed to boot or
`runner.js` itself crashed before producing JSON.
