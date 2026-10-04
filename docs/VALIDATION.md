> 🇬🇧 English · 🇪🇸 [Español](es/VALIDATION.md)

# Validation

> Available from puremacro **0.92.0** onwards.

`puremacro` reimplements many estimators in pure numpy/scipy so they run in the
browser. To show those reimplementations are correct, the package ships a
**validation gallery**: a declarative registry of cases, each comparing a
puremacro estimator to an *independent* reference. Run it yourself:

```python
from puremacro.validation import run_all, scorecard

scorecard()                 # one row per case: puremacro vs reference, pass/fail, margin
assert all(r.passed for r in run_all())
```

`scorecard()` and `run_all()` are pyodide-safe — they need only the four core
dependencies — so the gallery runs unchanged in the browser playground (notebook
`12_validation_gallery`, also available in Spanish as `12_validation_gallery_es`).

### Pyodide run

On 3 October 2026 the gallery was executed inside real Pyodide, headless under
Node.js v26.10.0, with `tools/pyodide_gallery.py`: the wheel installed through micropip
with every dependency resolved from the Pyodide distribution (nothing from PyPI, as in
the playground) and the cases ran one at a time. The first run, on the 110-case gallery
of 4.5.0, passed 110 of 110 on Pyodide 314.0.5 (Python 3.14.2, the version pinned by the
playground's `jupyterlite-pyodide-kernel` 0.8.4) and 109 of 110 on Pyodide 0.28.3
(Python 3.13.2, the Gate 6 harness pin): `puremacro.narrative` imported `sqlite3` and `ssl`
at module level and Pyodide 0.28 ships both as unvendored standard-library modules. Those
imports now go through `puremacro._optional_stdlib`, and the re-run the same day on the
114-case gallery (with the four `Trade` cases) passed **114 of 114 on both Pyodide
versions** (35.8 s on 0.28.3, 38.4 s on 314.0.5; about 42 to 44 s with boot, install and
import) and 114 on the desktop (CPython 3.13, 21.0 s). Every pass flag agrees across the
three platforms; four `max_margin` values (three L-BFGS-B GARCH fits and one Monte Carlo
sup-t value) differ between WebAssembly and the desktop by 1e-8 to 5e-4 and pass on all.
This is Pyodide under Node, not a browser tab, and it measures the gallery, not heavy
workloads. Details, per-case timings and the commands are in
`reviews/2026-10-03-pyodide-gallery/REPORT.md`.

## How a case is validated

Each case declares a **mechanism** (how its reference is sourced) and a
**tolerance tier**. The mechanism is the heart of the trust argument — the
reference must be genuinely independent of the puremacro code under test.

| Mechanism | Reference | Runs live in the browser? |
|---|---|---|
| `package` | statsmodels / linearmodels / arch, captured once as a frozen *golden* value | yes — compares to the golden (no heavy dependency at run time) |
| `scipy` | `scipy` / `numpy` (a runtime dependency), recomputed live | yes |
| `analytical` | a closed-form solution | yes |
| `published` | a number from a paper, with citation | yes (frozen constant) |
| `internal` | a cross-method identity or simulate-then-recover check | yes |

The `package` mechanism keeps the shipped code pure: statsmodels/linearmodels/arch
are **never imported by puremacro**. Their outputs are frozen as golden JSON, and
a continuous-integration drift-guard (`pytest -m reference`) recomputes the live
package and asserts the golden is still faithful. So `run_all()` reads goldens for
`package` cases and recomputes everything else live — and never needs the heavy
packages.

Tolerance tiers: `EXACT` (rtol 1e-10) · `TIGHT` (1e-6) · `NUMERIC` (1e-2) ·
`COARSE` (10%) · `QUALITATIVE` (sign / ordering / a lower-bound threshold).

## Coverage

**114 cases across 16 subsystems — all passing.** By mechanism: internal 62,
analytical 31, package 14, scipy 5, published 2.

| Subsystem | Cases | Reference(s) |
|---|---|---|
| `var` | 15 | Cholesky IRF vs statsmodels `orth_irfs`; FEVD-sums-to-1 and stability ⇔ companion spectral radius < 1 (identities); narrative sign restrictions; Minnesota BVAR posterior means vs hand-built Theil (λ₂ = 0.5) and Bańbura–Giannone–Reichlin eq. (5) Normal-inverse-Wishart closed forms; GVAR identities — zero trade weights decouple the country blocks, the solved global system reproduces them, the persistence profile is 1 at impact, normalised GFEVD rows sum to 1, and a one-country GVAR's GIRFs equal the `var` generalised IRFs |
| `lp` | 6 | Jordà LP coefficients/HAC SE vs statsmodels OLS-HAC; LP-IV vs linearmodels `IV2SLS`; two-way FE vs `PanelOLS`; IV-reduces-to-OLS identity |
| `garch` | 7 | GARCH(1,1) params/vols vs `arch`; simulate-then-recover; GARCH-MIDAS variance decomposition identity |
| `inference` | 9 | Newey–West / OLS-HAC SE vs statsmodels HAC; Stock–Yogo critical-value table (published); Kiefer–Vogelsang (2005) Table I Bartlett fixed-b critical values (published); sup-t plug-in critical value vs its i.i.d. closed form; sup-t monotonicity over pointwise critical values |
| `state_space` | 6 | Kalman filter/smoother states + log-likelihood vs statsmodels state space; smoother-variance identities |
| `dynpanel` | 7 | Arellano–Bond / Blundell–Bond GMM recover known ρ; exact-identification J = 0; Windmeijer WC-robust SEs equal an independent finite-difference evaluation (uncollapsed `lags(2)` layout) |
| `did` | 11 | Callaway–Sant'Anna recovers 2x2 DiD analytically; Sun–Abraham equals Callaway–Sant'Anna on 2x2; on a staggered design with unequal cohort sizes the Callaway–Sant'Anna event study and overall ATTs equal CS (2021) eqs. 3.4/3.7/3.10–3.12 and the Sun–Abraham IW event study equals SA eq. 27; Borusyak–Jaravel–Spiess imputation recovers 2x2; Synthetic DiD recovers treatment effect; spatial DiD with every unit beyond the outermost ring reduces exactly to two-way FE, and the ring adjustment beats the contaminated naive estimate on a planted-spillover DGP |
| `unit_root` | 3 | ERS GLS detrending recovers deterministic trend & constant mean analytically; Ng–Perron MZt = MZa · MSB cross-statistic identity |
| `spectral` | 6 | Welch PSD / cross-spectrum / coherence vs `scipy.signal`; band-power partition-of-unity; coherence ∈ [0,1] |
| `forecast` | 6 | Gaussian CRPS closed form (Gneiting–Raftery); fair-ensemble convergence; PIT calibration; Diebold–Mariano sign/tie; MCS retention |
| `vfi` | 5 | Tauchen/Rouwenhorst reproduce AR(1) moments; Brock–Mirman closed-form policy; Markov stationary vs scipy left eigenvector; EGM = VFI |
| `dsge` | 6 | Klein = gensys on a known model; closed-form forward-looking solution; Kalman log-likelihood vs the AR(1) analytical likelihood |
| `narrative` | 7 | Known-value lexicon scoring on crafted text; index monotonicity / standardization identities |
| `cointegration` | 4 | FM-OLS and DOLS recover a planted cointegrating β; the two agree; DOLS mitigates endogeneity bias |
| `spatial` | 12 | Moran's I / Geary's C with their Cliff-Ord moments vs `esda` (PySAL); Conley HAC at cutoff 0 = HC0 and = an explicit Bartlett double loop; flat-kernel space-time HAC = Driscoll-Kraay; the SAR/SEM concentrated log-likelihood at ρ=0 = the OLS Gaussian log-likelihood; SDM = SAR on the augmented design and SLX = OLS on it; LeSage-Pace impacts = a brute-force dense `(I−ρW)⁻¹(Iβ+Wθ)`; the spatial panel at ρ=0 = two-way FE; the Lee-Yu correction rescales σ² by exactly T/(T−1); `spatial_lp` without a spillover = `panel_lp` |
| `Trade` | 4 | Clean OECD 2020 77x11 table (`load_icio_data(source="oecd2020")`, built by `tools/build_icio_77c_11s.py`): the aggregation conserves the native file's world value added and gross output (frozen in `MANIFEST_OECD2020.json`); every industry column balances with positive value added; the calibrated model's zero-tariff equilibrium is the base year (unit prices, wages and rentals); the base equilibrium vector agrees with the legacy MATLAB model's solution on the same table (`trade_reference_solutions_oecd2020.npz`, an external reference copied verbatim) |

Each case carries its full citation in the code (`ValidationCase.citation`), shown
in the `citation` column of `scorecard()`. Key references include Lütkepohl (2005),
Newey & West (1987), Stock & Yogo (2005), Gneiting & Raftery (2007), Diebold &
Mariano (1995), Brock & Mirman (1972), Rouwenhorst (1995), Tauchen (1986), Engle
(2002), Arellano & Bond (1991), Blundell & Bond (1998), Windmeijer (2005),
Kiefer & Vogelsang (2005), Callaway & Sant'Anna (2021), Sun & Abraham (2021) and
Bańbura, Giannone & Reichlin (2010).

## Native X-11/ARIMA vs the real X-13ARIMA-SEATS binary

`puremacro.sa.x11` (native, pure-Python, Pyodide-safe) is validated
against the genuine Census X-13ARIMA-SEATS binary (v1.1.57) via frozen
goldens: `tools/gen_validation_goldens_sa.py` runs the binary with the
spec pinned to the native v1 configuration (log/level by AICC, airline
model, no outlier regressors, 60-period ARIMA extension, X-11 with
MSR-selected seasonal filter) on nine real series — six county LAUS
unemployment levels spanning large to tiny (Keweenaw MI, ~2k
population), NSA industrial production, and two quarterly aggregates —
and `tests/test_sa_x11_native.py` asserts agreement within tolerances
that were **measured, then frozen with headroom** (re-measured
2026-07-22 after the genuine B17/B20 weight cascade landed — extreme
weights from each stage's irregular forming weight-modified originals
that feed the next stage's trends, the structure the v1 chain had
condensed away): interior medians 0.08–0.83%, interior maxima under
0.7% on smooth macro series and up to ~7% at extreme-replaced outlier
months of wild small-count series (roughly HALF the v1 gaps),
seasonal-filter selection matching the binary on every monthly series.
The remaining divergence sources are documented rather than hidden:
the binary applies asymmetric filters at the series start (it does not
use backcasts inside X-11) where the native engine uses
symmetric-over-backcasts, and residual replacement-value details at
flagged outliers. CI never runs the
binary — only the frozen goldens. This is what the naming
"X-11/ARIMA-style, X-13-validated" promises: exactly what the frozen
tolerances measured, no more.

## Honest scope

Where no sound *independent* reference exists, a case is **skipped with a stated
reason** rather than rubber-stamped with a circular check. Documented examples:
`inference.kleibergen_paap_f` (the rk Wald F has no independent closed form for
two or more endogenous regressors; the `k = 1` case, where it equals the
HC0-robust first-stage F, is pinned in `tests/test_inference/test_weak_iv_values.py`
rather than in the gallery), the heavy `dsge` estimation routines, the external
Arellano–Bond check against Stata `xtabond` Examples 1, 2 and 4 (it lives in
`tests/test_dynpanel/` and runs only when a copy of `abdata` is available,
through `PUREMACRO_ABDATA` or as `tests/fixtures/abdata.dta` or `.csv`; the file
is third-party data and not bundled), and the narrative LLM
and live-fetch paths. The gallery validates what can be
validated independently, and says so when it cannot.

## Re-verifying

```bash
# fast, pure: puremacro vs the contracted references
python -m pytest tests/validation/ -q

# CI drift-guard: recompute the LIVE statsmodels/linearmodels/arch references
# and assert the frozen goldens are still faithful (needs the dev extra)
python -m pytest -m reference -q
```

To extend the gallery, drop a `cases_<subsystem>.py` module exposing
`CASES: list[ValidationCase]` into `puremacro/validation/` — `run_all()` discovers
it automatically.
