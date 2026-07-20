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

**62 cases across 11 subsystems — all passing.** By mechanism: internal 29,
analytical 16, package 11, scipy 5, published 1. By tier: tight 26, exact 15,
numeric 11, qualitative 7, coarse 3.

| Subsystem | Cases | Reference(s) |
|---|---|---|
| `var` | 3 | Cholesky IRF vs statsmodels `orth_irfs`; FEVD-sums-to-1 and stability ⇔ companion spectral radius < 1 (identities) |
| `lp` | 5 | Jordà LP coefficients/HAC SE vs statsmodels OLS-HAC; LP-IV vs linearmodels `IV2SLS`; two-way FE vs `PanelOLS`; IV-reduces-to-OLS identity |
| `garch` | 6 | GARCH(1,1) params/vols vs `arch`; simulate-then-recover; stationarity identities |
| `inference` | 7 | Newey–West / OLS-HAC SE vs statsmodels HAC; Stock–Yogo critical-value table (published); sup-t plug-in critical value vs its i.i.d. closed form; analytical + internal anchors |
| `state_space` | 6 | Kalman filter/smoother states + log-likelihood vs statsmodels state space; smoother-variance identities |
| `dynpanel` | 6 | Arellano–Bond / Blundell–Bond GMM recover a known ρ on a simulated panel; exact-identification J = 0 |
| `spectral` | 6 | Welch PSD / cross-spectrum / coherence vs `scipy.signal`; band-power partition-of-unity; coherence ∈ [0,1] |
| `forecast` | 5 | Gaussian CRPS closed form (Gneiting–Raftery); fair-ensemble convergence; PIT calibration; Diebold–Mariano sign/tie |
| `vfi` | 5 | Tauchen/Rouwenhorst reproduce AR(1) moments; Brock–Mirman closed-form policy; Markov stationary vs scipy left eigenvector; EGM = VFI |
| `dsge` | 6 | Klein = gensys on a known model; closed-form forward-looking solution; Kalman log-likelihood vs the AR(1) analytical likelihood |
| `narrative` | 7 | Known-value lexicon scoring on crafted text; index monotonicity / standardization identities |

Each case carries its full citation in the code (`ValidationCase.citation`), shown
in the `citation` column of `scorecard()`. Key references include Lütkepohl (2005),
Newey & West (1987), Stock & Yogo (2005), Gneiting & Raftery (2007), Diebold &
Mariano (1995), Brock & Mirman (1972), Rouwenhorst (1995), Tauchen (1986), Engle
(2002), Arellano & Bond (1991), and Blundell & Bond (1998).

## Honest scope

Where no sound *independent* reference exists, a case is **skipped with a stated
reason** rather than rubber-stamped with a circular check. Documented examples:
`inference.kleibergen_paap_f` (the implementation returns a non-standard statistic
with no matching closed form), the heavy `dsge` estimation routines, an external
Arellano–Bond coefficient cross-check (no offline dataset / Python GMM package),
and the narrative LLM and live-fetch paths. The gallery validates what can be
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
