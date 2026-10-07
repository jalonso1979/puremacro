# The validation gallery inside Pyodide — 3 October 2026

The 110-case validation gallery of puremacro 4.5.0 was executed inside real
Pyodide, headless under Node.js, with the new `tools/pyodide_gallery.py`
(`tools/pyodide/gallery_runner.js` + `gallery_cases.py`). The wheel was
installed the way the JupyterLite playground installs it, through micropip with
every dependency resolved from the Pyodide distribution (a dependency coming
from PyPI fails the run), and the cases ran one at a time, each timed with
`time.perf_counter()` inside Python. The same loop was then run on the desktop
with the conda interpreter for comparison.

**Result.** On Pyodide 314.0.5, the version the playground's kernel pins, all
110 cases pass in 34.6 s. On Pyodide 0.28.3, the version pinned by the release
gate harness, 109 pass in 31.5 s; the one failure is an import error, not a
numerical one: `puremacro.narrative` imports `sqlite3` and `ssl` at module
level and Pyodide 0.28 ships both as unvendored standard-library modules that
must be loaded explicitly. With them preloaded, 0.28.3 also passes 110. The
desktop passes 110 in 22.0 s. Every pass flag agrees between Pyodide 314.0.5 and
the desktop; four `max_margin` values differ by more than 1e-9 and all four
pass on both platforms.

## What ran

| | Pyodide 0.28.3 | Pyodide 314.0.5 | Desktop |
|---|---|---|---|
| Runtime | Pyodide 0.28.3 under Node v26.10.0, wasm32 | Pyodide 314.0.5 under Node v26.10.0, wasm32 | CPython 3.13.12 (conda `work`) |
| Python | 3.13.2 | 3.14.2 | 3.13.12 |
| Why this version | `tools/pyodide/package.json` pin (Gate 6) | `jupyterlite-pyodide-kernel` 0.8.4, which `.github/workflows/pages.yml` pins, hard-codes `PYODIDE_VERSION = "314.0.5"` | the project's test interpreter |
| numpy / scipy / pandas / matplotlib / requests | 2.2.5 / 1.14.1 / 2.3.1 / 3.8.4 / 2.32.4 | 2.4.6 / 1.18.0 / 3.0.2 / 3.10.8 / 2.33.1 | 2.4.6 / 1.18.1 / 3.0.6 / 3.11.1 / 2.34.2 |
| Dependency source | Pyodide distribution bundled in the npm package | Pyodide distribution, fetched by Node from `cdn.jsdelivr.net/pyodide/v314.0.5/full` | conda environment |
| puremacro | wheel `puremacro-4.5.0-py3-none-any.whl`, 22,845,050 bytes, sha256 `34db88cec2e1ae3d2413570db9568d056c7dc1cf69e0f84ee78ada0d963436f5`, installed with `micropip.install("emfs:/tmp/<wheel>")` | same wheel | the checkout the wheel was built from (`PYTHONPATH=.`) |
| Host | Apple M3 Pro, 12 cores, 36 GiB, macOS 27.2 | same | same |
| Envelope | `results.json` | `results-pyodide-314.0.5.json` | `results-desktop.json` |

The wheel was built with
`/opt/homebrew/Caskroom/miniconda/base/envs/work/bin/python -m pip wheel --no-deps --no-build-isolation -w <dir> .`
on commit `86b3ac99`; every library file in the wheel is byte-identical to that
commit (a parallel session began editing `puremacro/dsge` and `puremacro/trade`
in the working tree at 21:00 local time, after the build at 20:46 and the
desktop run at 20:52, and none of those edits or new files is in the wheel; no
gallery case touches `puremacro.trade`, and the `dsge` margins agree exactly
across platforms). The three runs were sequential, between 02:52 and 02:59 UTC
on 4 October (the evening of 3 October local time). A fourth run on 314.0.5
later in the session, made to exercise the runner's final code, passed 110 again
in 37.4 s, so single-run gallery times vary by several per cent. Nothing in the Pyodide runs
resolved from PyPI: `micropip.list()` reports `source == "pyodide"` for every
dependency and the `emfs:` wheel for puremacro.

The playground's Pyodide version was determined from the 0.8.4 kernel wheel on
PyPI (`jupyterlite_pyodide_kernel/constants.py`: `PYODIDE_VERSION = "314.0.5"`,
`PYODIDE_CDN_URL = https://cdn.jsdelivr.net/pyodide/v314.0.5/full`). The kernel
installed locally is 0.8.3, which pins 314.0.4, and that is what the local
`playground/dist` build references; the deployed site is built by `pages.yml`
with 0.8.4.

## Totals

| | Cases | Passed | Failed | Gallery time, s | Boot, s | Install, s | Import + discovery, s | Runner wall time, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pyodide 0.28.3 | 110 | 109 | 1 | 31.55 | 0.97 | 2.12 | 1.33 | 36.02 |
| Pyodide 0.28.3, `sqlite3` + `ssl` preloaded (diagnostic) | 110 | 110 | 0 | 32.33 | 1.07 | 2.22 | 1.33 | 37.00 |
| Pyodide 314.0.5 | 110 | 110 | 0 | 34.63 | 1.25 | 3.51 | 1.58 | 41.02 |
| Desktop (CPython 3.13.12) | 110 | 110 | 0 | 22.04 | — | — | 0.64 | 22.75 |

"Boot" is `loadPyodide()` plus loading micropip; "Install" is the
dependency-resolving `micropip.install` of the wheel (on 314.0.5 it includes
downloading the dependency wheels from the CDN, which Node does once and
caches); "Gallery time" is the sum of the per-case timings. The gallery takes
1.43x (0.28.3) and 1.57x (314.0.5) the desktop time overall; the median
per-case ratio is 1.6x and 1.7x, with a range from 0.2x to 12x on 0.28.3
(0.7x to 9x on 314.0.5).

### By subsystem

| Subsystem | Cases | Passed (0.28.3) | s (0.28.3) | Passed (314.0.5) | s (314.0.5) | Passed (desktop) | s (desktop) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cointegration | 4 | 4 | 1.10 | 4 | 1.90 | 4 | 0.95 |
| did | 11 | 11 | 1.41 | 11 | 1.64 | 11 | 1.13 |
| dsge | 6 | 6 | 0.60 | 6 | 0.68 | 6 | 0.46 |
| dynpanel | 7 | 7 | 9.65 | 7 | 9.68 | 7 | 3.43 |
| forecast | 6 | 6 | 2.46 | 6 | 2.63 | 6 | 3.70 |
| garch | 7 | 7 | 0.68 | 7 | 0.53 | 7 | 0.39 |
| inference | 9 | 9 | 0.04 | 9 | 0.04 | 9 | 0.02 |
| lp | 6 | 6 | 0.11 | 6 | 0.15 | 6 | 0.07 |
| narrative | 7 | 6 | 0.05 | 7 | 0.25 | 7 | 0.18 |
| spatial | 12 | 12 | 0.12 | 12 | 0.13 | 12 | 0.08 |
| spectral | 6 | 6 | 0.01 | 6 | 0.01 | 6 | 0.01 |
| state_space | 6 | 6 | 0.03 | 6 | 0.03 | 6 | 0.02 |
| unit_root | 3 | 3 | 0.01 | 3 | 0.01 | 3 | 0.01 |
| var | 15 | 15 | 0.10 | 15 | 0.11 | 15 | 0.08 |
| vfi | 5 | 5 | 15.17 | 5 | 16.84 | 5 | 11.52 |
| **All** | **110** | **109** | **31.55** | **110** | **34.63** | **110** | **22.04** |

### By mechanism

| Mechanism | Cases | Passed (0.28.3) | s (0.28.3) | Passed (314.0.5) | s (314.0.5) | Passed (desktop) | s (desktop) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytical | 31 | 30 | 7.13 | 31 | 8.17 | 31 | 5.38 |
| internal | 59 | 59 | 24.22 | 59 | 26.23 | 59 | 16.53 |
| package | 13 | 13 | 0.18 | 13 | 0.21 | 13 | 0.13 |
| published | 2 | 2 | 0.00 | 2 | 0.00 | 2 | 0.00 |
| scipy | 5 | 5 | 0.01 | 5 | 0.02 | 5 | 0.01 |

## Pyodide vs desktop agreement

Each case was matched by id; a discrepancy is a different `passed` flag or a
`max_margin` differing by more than 1e-9 (`max_margin` is `max(|pm - ref| -
allowed)` for numeric tiers and the largest shortfall for the qualitative tier;
negative means pass with that much headroom).

Pyodide 314.0.5 vs desktop: 110 of 110 pass flags agree; 106 of 110 margins
agree to 1e-9, 63 of them exactly. Pyodide 0.28.3 vs desktop: 109 of 110 pass
flags agree (the `sqlite3` case); 105 of 109 margins agree to 1e-9, 59 exactly.
The cases beyond 1e-9 are the same four against both Pyodide builds:

| Case | Mechanism / tier | Desktop | Pyodide 0.28.3 | Pyodide 314.0.5 |
|---|---|---:|---:|---:|
| `garch.params_vs_arch` | package / numeric | -4.56460e-4 | -4.56473e-4 (Δ 1.3e-8) | -4.56488e-4 (Δ 2.8e-8) |
| `garch.cond_vol_vs_arch` | package / numeric | -9.10756e-3 | -9.10753e-3 (Δ 3.3e-8) | -9.10750e-3 (Δ 6.5e-8) |
| `garch.simulate_then_recover` | internal / coarse | -1.72212e-3 | -1.72214e-3 (Δ 1.3e-8) | -1.72215e-3 (Δ 2.8e-8) |
| `inference.supt_monotonic_over_pointwise` | internal / qualitative | -0.722381 | -0.721911 (Δ 4.7e-4) | -0.721911 (Δ 4.7e-4) |
| `narrative.count_keywords_known_counts` | analytical / exact | -1e-12, pass | **failed**: `ModuleNotFoundError: No module named 'sqlite3'` | -1e-12, pass |

The three GARCH cases are L-BFGS-B fits (their tiers were chosen because
distinct optimizer runs agree only to about 1e-3; the differences here are
1e-8). The sup-t case is a Monte Carlo critical value; both Pyodide builds give
the same value, 0.72191, against 0.72238 on the desktop. Pyodide 314.0.5 ships
the desktop's numpy 2.4.6 and scipy 1.18.0 against 1.18.1, so these differences
are consistent with WebAssembly (wasm32, Pyodide's OpenBLAS) against native
arm64 (Accelerate) floating point rather than with dependency versions, but this
was not isolated further. None of them comes close to a tolerance: the tightest
headroom among the four is 4.6e-4 on a tier that allows 1e-2 relative error.

## The failure on Pyodide 0.28.3

`narrative.count_keywords_known_counts` calls
`puremacro.narrative.indices._kernels.count_keywords`. Importing that module
imports the `puremacro.narrative` package, whose `__init__` reaches, at module
level,

- `import sqlite3` via `narrative/__init__.py:39 -> narrative/indices/__init__.py:30 -> narrative/indices/_llm_kernel.py:29`, and
- `import ssl` via `narrative/__init__.py:45 -> narrative/replication/__init__.py:13 -> narrative/replication/ramey_2011_defense.py:30 -> narrative/sources/_http.py:9 -> puremacro/_http.py:19`

(chains reproduced on the desktop with an import hook that blocks the two
modules). Pyodide 0.28.3's `pyodide-lock.json` lists `sqlite3` and `ssl` as
`cpython_module` packages, that is, standard-library modules unvendored from the
interpreter and loaded only on request (`pyodide.loadPackage(["sqlite3", "ssl"])`
or `micropip.install`); Pyodide 314.0.5 ships both inside `python_stdlib.zip`,
which is why the same wheel passes there. The two diagnostic runs confirm the
chain: preloading `sqlite3` alone moves the error to `No module named 'ssl'`
(`results-pyodide-0.28.3-preload-sqlite3.json`); preloading both gives 110 of
110 (`results-pyodide-0.28.3-preload-sqlite3-ssl.json`).

The other module-level import sites are `puremacro/_http_cache.py:29` and
`puremacro/_cache_db.py:33` (`sqlite3`), `puremacro/narrative/harvest.py:11`
(`sqlite3`), `puremacro/narrative/scoring/llm.py:25` and
`puremacro/narrative/sources/_fallback.py:21` (`ssl`). `tests/test_pyodide_compat.py`
cannot see this: its blocker covers the browser-absent third-party packages, and
`sqlite3` and `ssl` are always present on CPython. No library file was changed
for this report; the playground kernel's Pyodide (314.x) is not affected, and a
kernel pinned to a 0.2x Pyodide would be.

## Slowest cases

Pyodide 0.28.3 (desktop time and ratio alongside):

| Case | Subsystem | 0.28.3, s | Desktop, s | Ratio |
|---|---|---:|---:|---:|
| `vfi.egm_matches_vfi_policy` | vfi | 10.426 | 7.969 | 1.3 |
| `vfi.neoclassical_brock_mirman_policy` | vfi | 4.549 | 3.389 | 1.3 |
| `dynpanel.ab_recovers_rho` | dynpanel | 4.412 | 1.008 | 4.4 |
| `forecast.crps_ensemble_vs_gaussian` | forecast | 2.393 | 3.660 | 0.7 |
| `dynpanel.ab_within_sampling_band` | dynpanel | 2.089 | 0.782 | 2.7 |
| `dynpanel.hansen_j_does_not_reject` | dynpanel | 1.220 | 0.589 | 2.1 |
| `dynpanel.ar_test_ar1_present_ar2_absent` | dynpanel | 1.178 | 0.527 | 2.2 |
| `cointegration.fmols_planted_beta` | cointegration | 1.088 | 0.945 | 1.2 |
| `did.callaway_santanna_recovers_canonical_2x2` | did | 0.619 | 0.549 | 1.1 |
| `dynpanel.bb_recovers_rho` | dynpanel | 0.611 | 0.406 | 1.5 |

Pyodide 314.0.5:

| Case | Subsystem | 314.0.5, s | Desktop, s | Ratio |
|---|---|---:|---:|---:|
| `vfi.egm_matches_vfi_policy` | vfi | 11.325 | 7.969 | 1.4 |
| `vfi.neoclassical_brock_mirman_policy` | vfi | 5.289 | 3.389 | 1.6 |
| `dynpanel.ab_recovers_rho` | dynpanel | 4.421 | 1.008 | 4.4 |
| `forecast.crps_ensemble_vs_gaussian` | forecast | 2.550 | 3.660 | 0.7 |
| `dynpanel.ab_within_sampling_band` | dynpanel | 2.078 | 0.782 | 2.7 |
| `cointegration.fmols_planted_beta` | cointegration | 1.891 | 0.945 | 2.0 |
| `dynpanel.hansen_j_does_not_reject` | dynpanel | 1.226 | 0.589 | 2.1 |
| `dynpanel.ar_test_ar1_present_ar2_absent` | dynpanel | 1.219 | 0.527 | 2.3 |
| `did.callaway_santanna_recovers_canonical_2x2` | did | 0.788 | 0.549 | 1.4 |
| `dsge.klein_forward_policy_loading` | dsge | 0.662 | 0.453 | 1.5 |

Desktop:

| Case | Subsystem | Desktop, s | 0.28.3, s | 314.0.5, s |
|---|---|---:|---:|---:|
| `vfi.egm_matches_vfi_policy` | vfi | 7.969 | 10.426 | 11.325 |
| `forecast.crps_ensemble_vs_gaussian` | forecast | 3.660 | 2.393 | 2.550 |
| `vfi.neoclassical_brock_mirman_policy` | vfi | 3.389 | 4.549 | 5.289 |
| `dynpanel.ab_recovers_rho` | dynpanel | 1.008 | 4.412 | 4.421 |
| `cointegration.fmols_planted_beta` | cointegration | 0.945 | 1.088 | 1.891 |
| `dynpanel.ab_within_sampling_band` | dynpanel | 0.782 | 2.089 | 2.078 |
| `dynpanel.hansen_j_does_not_reject` | dynpanel | 0.589 | 1.220 | 1.226 |
| `did.callaway_santanna_recovers_canonical_2x2` | did | 0.549 | 0.619 | 0.788 |
| `dynpanel.ar_test_ar1_present_ar2_absent` | dynpanel | 0.527 | 1.178 | 1.219 |
| `dsge.klein_forward_policy_loading` | dsge | 0.453 | 0.581 | 0.662 |

Two cases dominate the gallery on every platform (`vfi.egm_matches_vfi_policy`
and `vfi.neoclassical_brock_mirman_policy`, about half the total). The
dynamic-panel GMM cases, which are many small linear-algebra calls, slow down
most in WebAssembly (2x to 4.4x); the CRPS ensemble case is faster in Pyodide
than on the desktop, presumably because Pyodide's single-threaded BLAS avoids
thread start-up that the desktop pays on many small products. These are
single runs; no repetition was done, so differences of a few tenths of a second
are not meaningful.

## Files

- `REPORT.md`: this document.
- `results.json`: the Pyodide 0.28.3 envelope (schema in the header of `tools/pyodide/gallery_runner.js`; per case: id, subsystem, mechanism, tol, passed, max_margin, error, seconds). The `preloaded` and `preload_error` fields were added to the runner after these envelopes were written, so they carry neither.
- `results-pyodide-314.0.5.json`: the Pyodide 314.0.5 envelope.
- `results-desktop.json`: the desktop envelope from `tools/pyodide/gallery_cases.py`.
- `results-pyodide-0.28.3-preload-sqlite3.json`, `results-pyodide-0.28.3-preload-sqlite3-ssl.json`: the two diagnostic runs.
- `summary-0.28.3-vs-desktop.txt`, `summary-314.0.5.txt`, `summary-0.28.3-preload-sqlite3-ssl.txt`: the wrapper's printed summaries.

## Reproduce

```bash
cd tools/pyodide && npm install && cd ../..          # once; installs the pinned Pyodide 0.28.3
PY=/opt/homebrew/Caskroom/miniconda/base/envs/work/bin/python
$PY -m pip wheel --no-deps --no-build-isolation -w /tmp/pm-wheel .
WHEEL=/tmp/pm-wheel/puremacro-4.5.0-py3-none-any.whl

# Pyodide 0.28.3 (harness pin) + desktop baseline in the same interpreter, with comparison
$PY tools/pyodide_gallery.py --wheel $WHEEL --json results.json --desktop-json results-desktop.json

# the playground's Pyodide generation (npm needs network; Node fetches the dependency wheels from the CDN)
npm install --prefix /tmp/pyodide-314 pyodide@314.0.5
$PY tools/pyodide_gallery.py --wheel $WHEEL --json results-pyodide-314.0.5.json \
    --pyodide /tmp/pyodide-314/node_modules/pyodide

# diagnostics for the 0.28.3 failure
$PY tools/pyodide_gallery.py --wheel $WHEEL --preload sqlite3 --json results-pyodide-0.28.3-preload-sqlite3.json
$PY tools/pyodide_gallery.py --wheel $WHEEL --preload sqlite3,ssl --json results-pyodide-0.28.3-preload-sqlite3-ssl.json
```

`tools/pyodide_gallery.py` without `--wheel` builds the wheel itself. The
runner alone is `node tools/pyodide/gallery_runner.js --wheel <whl> [--out
<json>] [--pyodide <dir>] [--preload a,b]`; the desktop loop alone is
`PYTHONPATH=. MPLBACKEND=Agg python tools/pyodide/gallery_cases.py --json <path>`.

## What this does and does not establish

It establishes that the 4.5.0 wheel installs into Pyodide with its dependencies
resolved from the Pyodide distribution alone, that all 110 gallery cases pass
inside the Pyodide WebAssembly runtime at the version the playground kernel
pins (and 109 of 110 at the release-gate pin, for the stated import reason),
that the pass flags agree with the desktop, and how long the gallery takes
there. Node-hosted Pyodide is the same WebAssembly runtime family as the
JupyterLite playground but not a browser tab: there is no DOM, no browser
memory cap or tab scheduling, the file system and network are Node's, and the
dependency wheels were fetched by Node rather than by the kernel. The deployed
playground builds with `jupyterlite-pyodide-kernel` 0.8.4 and therefore Pyodide
314.0.5, the version run here; the Gate 6 harness still pins 0.28.3, where the
`sqlite3`/`ssl` import matters. Margins are not bitwise identical across
platforms: four of 110 differ at the 1e-8 to 1e-4 level in optimizer- and
Monte-Carlo-dependent cases. The gallery is a set of small seeded problems that
completes in about half a minute; it says nothing about long MCMC chains, large
bootstraps, the course notebooks or any other realistic heavy workload in a
browser, and nothing about GPU execution. Those remain unverified.

## Addendum, same day: after the fix and with the four trade cases

The import defect above was fixed the same afternoon (the nine module-level
`import sqlite3` / `import ssl` statements now go through
`puremacro._optional_stdlib`, which degrades to a placeholder that fails at
first use; `tests/test_pyodide_compat.py` guards against new ones), and four
`Trade` cases on the clean OECD 2020 table were added to the gallery (114 cases,
16 subsystems). The same harness was then re-run on the rebuilt wheel
(`results-rerun-114-*.json`):

| | Pyodide 0.28.3 | Pyodide 314.0.5 | Desktop (conda CPython 3.13) |
|---|---|---|---|
| Cases / passed | 114 / 114 | 114 / 114 | 114 / 114 |
| Gallery time | 35.8 s | 38.4 s | 21.0 s |
| Boot + install + import | 1.1 + 3.2 + 1.7 s | 1.3 + 2.8 + 1.8 s | — |
| Runner wall time | 41.8 s | 44.2 s | 21.5 s |

Nothing was preloaded; the dependency-resolving install again took every
dependency from the Pyodide distribution. The four trade cases run in 0.02 to
0.09 s each on every platform. The same four `max_margin` values as before
(three GARCH L-BFGS-B fits, one Monte Carlo sup-t) differ between WebAssembly and
the desktop by 1e-8 to 5e-4 and pass on all three. The machine was under heavy
load from parallel MATLAB and optimisation jobs during this re-run, so the
times are upper bounds.

