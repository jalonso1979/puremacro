# puremacro 0.51.0 — identification innovations (Magnusson-Mavroeidis + non-Gaussian extensions + Lewbel-IV)

**Status:** draft 2026-05-23. Target release: **0.51.0**.

## Why

The 0.50.0 release closed the last of the 2026-05-22 maturity-arc pitches (P4/P5/P3/P1). The follow-on "research directions" brainstorm picked four candidates (R1, R2, R3, R4); this spec covers **R4 expanded**: three identification innovations bundled under one release.

Each addresses a gap in puremacro's current identification surface:

1. **Continuous-heteroskedasticity SVAR.** `puremacro.var.identify.hetero.rigobon_svar` requires the user to pre-specify regime dates. Magnusson-Mavroeidis (2014) generalises this to **continuous** time-varying variance with endogenous break-date detection — the data tells you where the variance changes are.

2. **Non-Gaussian SVAR identification fragility.** `puremacro.var.identify.non_gaussian_svar` produces a B matrix identified only up to permutation + sign. Downstream IRFs are noisy across runs because column ordering is non-deterministic. Plus the existing module doesn't tell the user whether non-Gaussianity is statistically informative for identification on their data — Gaussian-residual datasets get silent garbage results.

3. **Cross-section / panel IV without external instruments.** `puremacro.lp.iv` and `puremacro.inference.weak_iv` assume the user has an external instrument. Lewbel (2012) constructs IVs from heteroskedasticity in an auxiliary regression — useful in cross-section / panel contexts (and applicable to your subnational-labor research) where external IVs are weak.

All three are pure-numpy + scipy, Pyodide-compatible by construction. None overlap each other; each can be deferred or scoped down independently.

## Scope

One release. Three components:

- **Component A** — `puremacro/var/identify/magmav.py` (new file) implements `magmav_svar` + `MagMavSVARResult`.
- **Component B** — extends `puremacro/var/identify/non_gaussian.py` with `_sign_lock_by_kurtosis`, `gaussian_lr_test`, `variance_decomposition_consistency`; augments the existing result with `lr_test`, `consistency_check`, `kurtoses` fields.
- **Component C** — `puremacro/inference/lewbel_iv.py` (new file) + `puremacro/lp/iv_lewbel.py` (thin LP wrapper) implement `lewbel_iv` + `LewbelIVResult`.

Plus tests, version bump, CHANGELOG, public-API snapshot regeneration.

Out of scope: HANK-lite (R5); Fertility DSGE estimation (R1a/R1b); Climate × fertility (R2); Paleoclimate VARX (R3); sign-restricted + non-Gaussian combined identification (Lanne-Liu-Luoto 2023); ICA-with-mixed-Gaussian shocks.

## Pre-conditions

- 0.50.0 shipped at tag `v0.50.0` (commit `d4014f5`), pushed to `origin/feature/subnational-labor-uncertainty-us`.
- 6 release-gate gates green.
- `puremacro.var.estimate.estimate_var(Y, p) -> (A_list, c, Sigma, residuals, _)` available.
- `puremacro.var.identify.non_gaussian.non_gaussian_svar(...)` exists with the FastICA implementation from 0.45.0 work.
- `puremacro.var.identify.hetero.rigobon_svar(...)` exists for comparison.
- `puremacro.inference.weak_iv` has the weak-IV diagnostics suite.
- `puremacro.var.identify._results.py` has 9 existing `*Result` frozen dataclasses with the canonical `(H+1, n, n)` axis convention.

## Architecture

Three independent components under one release banner. Each lives in its existing puremacro home; no cross-component dependencies.

```
0.51.0
   │
   ├── Component A — puremacro/var/identify/magmav.py (new file)
   │     └── magmav_svar(Y, p, horizon, k_breaks=None, n_boot, ci, seed) → MagMavSVARResult
   │
   ├── Component B — puremacro/var/identify/non_gaussian.py (modified)
   │     ├── new private: _sign_lock_by_kurtosis, gaussian_lr_test, variance_decomposition_consistency
   │     └── existing non_gaussian_svar augmented (new fields, no removed fields)
   │
   └── Component C — puremacro/inference/lewbel_iv.py (new) + puremacro/lp/iv_lewbel.py (new)
         ├── lewbel_iv(y, X_endog, X_exog, heterosk_source) → LewbelIVResult
         └── lp_iv_lewbel(panel, ...) → long-form DataFrame
```

All public symbols added to the relevant `__init__.py::__all__` lists. Frozen-dataclass result objects per the project standard.

## Component A — Magnusson-Mavroeidis SVAR

### Reference

Magnusson & Mavroeidis (2014), "Identification Using Stability Restrictions" (*Econometrica*). The identifying moment conditions use the assumption that **structural-shock variances change differently over time across shocks** — without requiring the user to specify regime boundaries.

### Public API

```python
def magmav_svar(
    Y: np.ndarray,                   # (T, n) reduced-form data
    *,
    p: int,                          # VAR lag order
    horizon: int = 20,
    k_breaks: int | None = None,     # None → BIC selection over {1, 2, 3, 4}
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> MagMavSVARResult:
    ...
```

Result class (in `var/identify/_results.py`):

```python
@dataclass(frozen=True)
class MagMavSVARResult:
    """Result of magmav_svar.

    Attributes
    ----------
    irf_point : (H+1, n, n)  point IRFs
    irf_lower : (H+1, n, n)  bootstrap lower band
    irf_upper : (H+1, n, n)  bootstrap upper band
    B : (n, n)               identified structural impact matrix
    variance_change_dates : tuple[int, ...]
        Break dates (T-indices) identified by BIC + sup-Wald sweep.
    k_breaks : int           number of breaks selected/specified
    n_boot : int
    ci : float
    eu : tuple[int, int]     existence/uniqueness flags (1,1) = OK
    n_fail : int             count of failed bootstrap draws (warning if > 5%)
    """
```

### Algorithm

1. Reduced-form VAR(p) via `puremacro.var.estimate.estimate_var(Y, p)` → `A_list, c, Σ, residuals`.
2. **Detect break dates.** If `k_breaks is None`, sweep `k ∈ {1, 2, 3, 4}` and select via BIC. For each candidate k:
   - Use a sup-Wald scan: for each candidate break date τ ∈ [0.15T, 0.85T], compute the Wald statistic for "variance shift" in residual covariance. Pick the k highest non-overlapping (min separation 0.05T).
   - BIC: `BIC(k) = -2·log L_k + k · log(T)`, where `log L_k` is the multivariate-normal likelihood with k+1 regime-specific variances.
3. **Estimate B.** Minimise `Σ_g || Σ_g − B D_g B^T ||_F` over regimes `g = 1, …, k+1`, where `D_g` is a diagonal matrix of structural-shock variances in regime g. Use `scipy.optimize.minimize(method="BFGS")`, parameterising B's lower-triangular elements + diagonal. Normalisation: `B[0, 0] > 0`; columns ordered by **descending cross-regime variance ratio** `r_j = max_g(D_g[j,j]) / min_g(D_g[j,j])` — the shock whose variance changes most across regimes goes in column 0. This pins identification because Magnusson-Mavroeidis derives identification precisely from cross-regime variance differences.
4. **Residual bootstrap.** `n_boot` draws; resample residuals WITHIN regimes (preserves heteroskedasticity); refit B per draw; count failures; if > 5% fail, emit warning. Percentile bands at `(1−ci)/2` and `(1+ci)/2`.
5. **IRFs.** Point + bootstrap percentiles, axis order `(H+1, n, n)`.

### Failure handling

- BIC selects k=0 (homoskedastic data) → `eu=(0,0)`, B set to lower-Cholesky of Σ as a fallback, warning issued.
- B optimisation fails to converge → multi-start retry (3 starts); if all fail, `eu=(0,0)` + Cholesky fallback + warning.
- Bootstrap draw fails → drop, increment `n_fail`. If `n_fail/n_boot > 0.05`, warning. If `n_fail/n_boot == 1.0`, raise RuntimeError.

## Component B — Non-Gaussian SVAR extensions

### State of existing code (as of 0.50.0)

`puremacro.var.identify.non_gaussian.non_gaussian_svar` **already** orders columns by descending `|excess kurtosis|` and **already** locks diagonal signs of `B0` positive (lines 79-93 of `non_gaussian.py`). The result class is:

```python
@dataclass(frozen=True)
class NonGaussianSVARResult:
    B0: np.ndarray                # (n, n)  identified impact matrix
    Q: np.ndarray                 # (n, n)  orthogonal rotation, B0 = chol(Σ) @ Q
    kurtosis: np.ndarray          # (n,)    excess kurtosis of recovered shocks, sorted
    irf: np.ndarray               # (H+1, n, n)
    ordering_by_kurt: np.ndarray  # (n,)    permutation applied
```

There is no bootstrap, no `sign_lock` kwarg (the lock IS the default and only behaviour), and no Gaussianity diagnostic.

### What's added in 0.51.0

Three diagnostics + two new optional fields on the result. **Strictly additive — no breaking changes.**

1. New private helper `_tiebreak_kurtosis_order(kurt, src) -> ndarray` that fixes the sort when `|Δk| < 1e-3 · max_k` between adjacent columns. Tie-break: |skewness| then |5th central moment|; lexicographic if still tied. Emits `warnings.warn(...)` when invoked.
2. New public function `gaussian_lr_test(B0, residuals) -> dict`.
3. New public function `variance_decomposition_consistency(B0, sigma_u) -> dict`.
4. `non_gaussian_svar(...)` calls #2 and #3 internally and stores results in the augmented dataclass.

### `gaussian_lr_test(B, residuals) -> dict`

```python
{
    "stat": float,    # LR test statistic
    "df": int,        # degrees of freedom
    "p_value": float, # χ² p-value
}
```

Compares the log-likelihood under the non-Gaussian-shocks model (each shock's density via KDE) to the Gaussian baseline (multivariate normal at the reduced-form Σ_u). DOF: `n*(n-1)/2` (the additional identifying restrictions). If `LR < 0` (non-Gaussian fit worse than Gaussian), clamp to 0 and `p_value = 1.0` — a strong signal that non-Gaussian identification isn't informative.

### `variance_decomposition_consistency(B, sigma_u) -> dict`

```python
{
    "max_abs_diff": float,  # max |B @ B.T - sigma_u| element-wise
    "rms_diff": float,      # √(mean(...²))
    "passed": bool,         # max_abs_diff < 1e-6
}
```

### Augmented `NonGaussianSVARResult`

Two new fields are appended (with default `None` so existing callers that positionally construct the dataclass — none in the repo — still work):

```python
@dataclass(frozen=True)
class NonGaussianSVARResult:
    # existing — unchanged
    B0: np.ndarray
    Q: np.ndarray
    kurtosis: np.ndarray
    irf: np.ndarray
    ordering_by_kurt: np.ndarray
    # NEW in 0.51.0
    lr_test: Optional[dict] = None
    consistency_check: Optional[dict] = None
```

No fields are removed, no fields are renamed, no field semantics change. The CHANGELOG should describe these as additive enhancements, not breaking changes.

## Component C — Lewbel-IV for cross-section / panel LP-IV

### Reference

Lewbel (2012), "Using Heteroscedasticity to Identify and Estimate Mismeasured and Endogenous Regressor Models" (*JBES*).

### Public API

`puremacro/inference/lewbel_iv.py`:

```python
def lewbel_iv(
    y: np.ndarray,                # (n_obs,)
    X_endog: np.ndarray,          # (n_obs, k_endog) endogenous regressors
    X_exog: np.ndarray,           # (n_obs, k_exog) exogenous (incl. constant)
    heterosk_source: np.ndarray,  # (n_obs, k_z) exogenous heteroskedasticity drivers
) -> LewbelIVResult:
    ...
```

Result class (in a new or extended `puremacro/inference/_results.py`):

```python
@dataclass(frozen=True)
class LewbelIVResult:
    """Result of lewbel_iv.

    Attributes
    ----------
    beta : (k_endog + k_exog,)  2SLS coefficients
    se : (k_endog + k_exog,)    standard errors
    t : (k_endog + k_exog,)     t-statistics
    n_obs : int
    n_iv_constructed : int      number of Lewbel-constructed IVs
    first_stage_F : float       first-stage F statistic
    lewbel_diagnostic : dict    {stat, p_value} — Breusch-Pagan-style test of
                                 heteroskedasticity in X_endog conditional on
                                 heterosk_source. p_value > 0.10 → weak IV.
    """
```

### Algorithm

1. Residualise `X_endog` and `heterosk_source` against `X_exog` (Frisch-Waugh): `X_endog_res`, `Z_res`.
2. Construct Lewbel IVs: `Z_constructed = Z_res * (X_endog_res - X_endog_res.mean(axis=0))`. Shape `(n_obs, k_z * k_endog)`.
3. 2SLS: regress y on `X_exog ∪ X_endog`, using `X_exog ∪ Z_constructed` as the instrument set.
4. First-stage F: from the first-stage regression of each `X_endog` column on `X_exog ∪ Z_constructed`.
5. **Lewbel diagnostic** (the IDENTIFICATION strength): Breusch-Pagan-style test that `heterosk_source` actually drives heteroskedasticity in `X_endog`. Specifically, regress `(X_endog_res)²` on `heterosk_source` and test the joint significance of the heterosk_source coefficients. `p_value < 0.10` → IV is reasonably strong; `p_value > 0.10` → warning issued.

### LP-IV wrapper

`puremacro/lp/iv_lewbel.py`:

```python
def lp_iv_lewbel(
    panel: pd.DataFrame,
    *,
    y: str,
    x_endog: str,
    heterosk_source: str,
    controls: Sequence[str] = (),
    horizons: Iterable[int] = range(0, 13),
    n_lags: int = 2,
    entity_level: str = "code",
    time_level: str = "date",
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Local projection with Lewbel-constructed IVs.

    Returns a long-form DataFrame with columns (h, beta, se, t, lo, hi).
    """
```

Internally: for each horizon h, construct `y^{(c)}_{t+h} - y^{(c)}_{t-1}`, regress on `x_endog` with `lewbel_iv` providing the constructed instrument. Uses the within transformation per `lp/panel.py`'s convention.

## Testing

Total: ~24 new unit tests across three test files + 1 LP wrapper test = ~25 new tests for 0.51.0.

### Component A tests (`tests/test_var/test_magmav.py`, ~10 tests)

- `test_magmav_svar_returns_result_dataclass` — basic API.
- `test_magmav_svar_shapes` — IRF + B shapes correct.
- `test_magmav_svar_recovers_b_on_synthetic_data` — synthetic VAR with known B + 2 variance regimes, T=500 → recovered B within ±10% column-wise post sign-lock.
- `test_magmav_svar_detects_k_breaks_via_bic` — synthetic with 2 breaks → BIC picks k=2.
- `test_magmav_svar_no_breaks_falls_back_warning` — homoskedastic data → BIC picks k=0 → warning + `eu=(0,0)`.
- `test_magmav_svar_bootstrap_bands_cover_true_irf` — true IRF inside 90% bands ≥ 85% of horizons.
- `test_magmav_svar_handles_k_breaks_arg` — explicit `k_breaks=2` skips BIC.
- `test_magmav_svar_seed_reproducibility` — same seed → identical bands.
- `test_magmav_svar_failed_optimization_warns` — pathological input → warning + fallback, no crash.
- `test_magmav_svar_variance_change_dates_shape` — returned dates are a tuple of T-indices, length `k_breaks`.

### Component B tests (new file `tests/test_var/test_non_gaussian_extensions.py`, ~7 tests)

- `test_tiebreak_uses_skewness_when_kurtoses_near_equal` — synthetic shocks with `kurt = [5.0, 5.0005, 1.0]` (within tolerance) but skewness `[0.0, 1.5, 0.0]` → second shock comes before first after tie-break; warning issued.
- `test_tiebreak_warns_when_invoked` — assert `warnings.warn` fires on tied inputs.
- `test_gaussian_lr_test_rejects_non_gaussian_data` — t-distributed (df=4) shocks → LR rejects (p < 0.05).
- `test_gaussian_lr_test_does_not_reject_gaussian_data` — Gaussian shocks → LR does NOT reject (p > 0.5 majority of seeds).
- `test_gaussian_lr_test_clamps_negative_lr_to_zero` — pathological draw where KDE fit beats Gaussian → assert `stat == 0`, `p_value == 1.0`.
- `test_variance_decomposition_consistency_passes_at_truth` — `B0 @ B0.T == Σ_u` exactly → `max_abs_diff < 1e-6`, `passed=True`.
- `test_non_gaussian_svar_result_has_new_fields` — call existing `non_gaussian_svar`, assert `result.lr_test` and `result.consistency_check` are non-None dicts with the documented keys.

### Component C tests (`tests/test_inference/test_lewbel_iv.py`, ~6 tests)

- `test_lewbel_iv_recovers_known_beta_on_synthetic_data` — DGP with `β=1.5`, n=5000 → estimated β within ±0.1.
- `test_lewbel_iv_first_stage_F_finite` — well-conditioned DGP → finite, sensible F.
- `test_lewbel_iv_warns_on_weak_lewbel_diagnostic` — DGP with no heteroskedasticity → warning raised.
- `test_lewbel_iv_handles_multiple_endogenous_regressors` — `k_endog == 2` works.
- `test_lewbel_iv_returns_result_dataclass` — basic API.
- `test_lewbel_iv_handles_rank_deficient_raises_valuerror` — rank-deficient `Z' X` → ValueError.

Plus 1 wrapper test in `tests/test_lp/test_lp_iv_lewbel.py`.

### Markers

- No `@pytest.mark.slow` on any new test (all <5s individually).
- No new `@pytest.mark.pyodide_smoke` tags initially; the existing 8-test Gate 6 set unchanged.

## Acceptance criteria for 0.51.0

1. `puremacro.var.identify.magmav_svar` + `MagMavSVARResult` exported from `puremacro.var.identify.__init__`.
2. `NonGaussianSVARResult` gains two optional fields `lr_test` and `consistency_check`. `non_gaussian_svar` populates both on every call. The existing column-ordering and sign-lock behaviour is unchanged (kurtosis-descending ordering with positive diagonal). No field is renamed or removed. The internal tie-breaker fires only when adjacent kurtoses are within `1e-3 · max_k` and emits a warning.
3. `puremacro.inference.lewbel_iv` + `LewbelIVResult` exported.
4. `puremacro.lp.iv_lewbel.lp_iv_lewbel` exported.
5. All ~25 new unit tests green under CPython.
6. Public-API snapshot regenerated.
7. All 6 release-gate gates green at HEAD.
8. CHANGELOG 0.51.0 entry — additive only; no breaking-changes section needed.
9. Version bumped 0.50.0 → 0.51.0.

## Risks and mitigations

1. **Magnusson-Mavroeidis BIC + break detection is expensive.** For T=200 quarters, the sup-Wald scan plus the B optimisation across k∈{1,2,3,4} is ~30s. *Mitigation:* default budget acceptable; if too slow, users pass `k_breaks` explicitly to skip selection.

2. **B-matrix optimisation in Component A can stall.** Non-convex. *Mitigation:* multi-start (3 starts) + Cholesky fallback + warning. Tests cover the pathological-input path.

3. **Tiebreak-by-skewness is fragile on small T.** With T < 100, skewness estimates are noisy and could reorder columns differently from a hand-derived "correct" ordering. *Mitigation:* the tie-break is only invoked when adjacent kurtoses are within tolerance; in that regime, the ordering is genuinely indeterminate and the warning surfaces the ambiguity. Tests use T=2000 to keep moment estimates tight.

4. **Lewbel-IV is weak when `heterosk_source` is itself endogenous.** *Mitigation:* the Lewbel diagnostic test surfaces this. Docs explicitly warn against using Lewbel when external IVs exist.

5. **Kurtosis-based sign-locking is fragile when shocks have similar kurtoses.** *Mitigation:* multi-moment tie-breaking + warning. Tests cover the tie case.

6. **The non-Gaussian LR test uses KDE-based shock densities.** Small T or heavy-tailed shocks can produce unreliable LRs. *Mitigation:* document the asymptotic + small-sample caveats. Note in test comments that the LR test is informative-not-definitive.

## Out of scope (deferred to follow-on specs)

- **R1a/R1b** — fertility DSGE + generic Bayesian DSGE engine (queued after 0.51.0).
- **R2** — climate × fertility temperature pipeline.
- **R3** — paleoclimate VARX / long-run cliometrics.
- **R5** — HANK-lite (Reiter / sequence-space Jacobian).
- Sign-restricted + non-Gaussian combined identification (Lanne-Liu-Luoto 2023).
- ICA-with-mixed-Gaussian shocks.
- Pyodide-Gate-6 expansion — initial 8-test set unchanged.
- PyPI publishing (still on the 1.0 path).
