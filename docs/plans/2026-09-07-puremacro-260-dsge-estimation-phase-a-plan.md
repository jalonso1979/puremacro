# puremacro 2.6.0 Implementation Plan — DSGE Tier 1, Phase A: `.mod` to posterior

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax for tracking. Work task-by-task; each task ends in a green test run and a commit.

**Goal:** make a Dynare `.mod` file estimable end to end. Parse `estimated_params`, build the measurement equation automatically from `varobs`, and expose `LinearModel.estimate()`, `.smoother()` and `.forecast()` — plus the mode-search menu and the marginal-likelihood estimators that make the output reportable. Close the silent `@#` macro-directive hole on the way in.

**Architecture:** seven new modules under `puremacro/dsge/`, all additive. `estimate_dsge`'s contract is unchanged — it keeps taking `observation_eq: Callable[[dict], StateSpaceModel]`; this phase *builds* that callable. The audited numerical core (Klein QZ, Kalman recursion, KKSS pruning) is not touched.

**Tech Stack:** numpy, scipy, pandas, matplotlib. No new dependencies (Pyodide four-package import contract).

**Spec:** `docs/specs/2026-09-07-puremacro-dsge-tier1-design.md` (§ Phase A)
**Roadmap:** `docs/plans/2026-09-07-puremacro-dsge-dynare-parity-roadmap.md`

---

## File map

### New files
- `puremacro/dsge/_estimated_params.py` (~320 LOC) — `estimated_params` / `_init` / `_bounds` grammar → `EstimatedParamSpec`, `EstimatedParams`.
- `puremacro/dsge/observation.py` (~220 LOC) — `make_state_space_from_varobs`.
- `puremacro/dsge/mode.py` (~340 LOC) — `csminwel`, `cmaes`, `find_mode`, `mode_check`.
- `puremacro/dsge/marginal.py` (~180 LOC) — `laplace_mdd`, `harmonic_mean_mdd`, `model_comparison`.
- `puremacro/dsge/smoother.py` (~200 LOC) — `smoothed_states`, `forecast_states` behind the `LinearModel` methods.
- `tests/test_dsge/test_estimated_params.py` (~14 tests)
- `tests/test_dsge/test_observation.py` (~9 tests)
- `tests/test_dsge/test_mode_search.py` (~8 tests)
- `tests/test_dsge/test_marginal_likelihood.py` (~6 tests)
- `tests/test_dsge/test_smoother.py` (~6 tests)
- `tests/test_dsge/test_estimate_from_mod.py` (~7 tests)
- `tests/test_dsge/test_macro_directive_guard.py` (~4 tests)
- `docs/dsge_estimation.md`, `docs/es/dsge_estimation.md`

### Modified files
- `puremacro/dsge/dynare.py` — `@#` guard; `parse_mod` returns `estimated_params`, `estimated_params_init`, `estimated_params_bounds`, `observation_trends`, `estimation_options`; `load_mod` attaches them to the model.
- `puremacro/dsge/priors.py` — `shift`/`scale` on `BetaPrior`/`GammaPrior`, `WeibullPrior`, inverse-gamma type 2, `ensure_prior` branches.
- `puremacro/dsge/build.py` — `LinearModel` gains `_varobs`, `_estimated_params`, `_mod_options` fields and `.estimate()`, `.smoother()`, `.forecast()` methods.
- `puremacro/dsge/estimate.py` — record `log_post_mode`; accept `mode_compute=`; accept a per-parameter `jscale` vector.
- `puremacro/dsge/_results.py` — `SmootherResult`, `DSGEForecastResult`; `DSGEPosteriorResult` gains `log_post_mode`, `log_mdd_laplace`, `log_mdd_harmonic`, `mdd_harmonic_spread`, `mode_compute` (all optional, default `None`).
- `puremacro/dsge/__init__.py` — exports.
- `mkdocs.yml`, `tests/test_bilingual_docs.py` (`_USER_DOCS`), `CHANGELOG.md`, `pyproject.toml`, `puremacro/__init__.py`, `CITATION.cff`, `tests/fixtures/public_api_snapshot.json`.

---

## Verified API surfaces

Read from the tree on 2026-09-07; every line below was checked, not assumed.

- `puremacro.dsge.decomposition._extract_companion_matrices(model, sigma=None)` → `(A, B, C, D, ys, variables, shocks, states, sigma_u)` with `x_{t+1} = A x_t + B u_t` and `v_t = ys + C x_t + D u_t`. `variables` is declaration order; `states` is the Klein state list.
- `puremacro.state_space.StateSpaceModel(T, Z, Q, H, R=None, c=None, d=None)` — time-invariant only (`__post_init__` rejects 3-D arrays). `R` is `(m, r)`, `Q` is `(r, r)`, `H` is `(n, n)`, `d` is `(n,)`.
- `puremacro.state_space.kalman_filter(y, model, a0=None, P0=None, diffuse_scale=1e6, diffuse_states=None)` → dict with `a_pred, P_pred, a_filt, P_filt, innov, F, K, loglik`.
- `puremacro.state_space.kalman_smoother(...)` → the filter dict plus `a_smooth (T, m)`, `P_smooth (T, m, m)`.
- `puremacro.dsge.estimate.estimate_dsge(data, *, observation_eq, priors, observed_vars, initial_params, fixed_params=None, model_name="unknown", n_draws=10_000, n_chains=2, burn_in=2_000, seed=0)` → `DSGEPosteriorResult`.
- `puremacro.dsge.estimate._check_stochastic_singularity(y, observation_eq, params, *, caller=...)` — raises `ValueError`; silent on anything it cannot evaluate.
- `puremacro.mcmc.random_walk_metropolis(log_posterior_fn, init, proposal_cov, n_draws, *, seed=0, accept_target=0.25, adapt_burnin=0)` → dict with `chain, log_post, accept_rate, final_scale`. **Scalar-c adaptation only** — no covariance adaptation.
- `puremacro.numerics.numerical_hessian(f, x, h=1e-4)` → `(n, n)`, `O(n^2)` evaluations.
- `puremacro.dsge.priors.Prior(dist, mean, std, lb=-inf, ub=inf)`; subclasses `BetaPrior, InvGammaPrior, NormalPrior, GammaPrior, UniformPrior`; `ensure_prior(spec)`; `_validate_priors(priors, *, caller=...)`; `param_names/param_bounds/prior_means/prior_stds`.
- `puremacro.dsge.build.LinearModel` is `@dataclass(frozen=True)` — new fields need defaults, and `load_mod` must attach metadata with `dataclasses.replace`, never assignment.
- `LinearModel.simulate` runs exactly `out[t] = M_x @ x + M_u @ u[t]; x = G @ x + N @ u[t]`, which is the `(A, B, C, D)` recursion above. Task 4's reference loop mirrors it deliberately.
- `puremacro.dsge.dynare._remove_comments` strips `//`, `/* */` and `%`. `@#` directives survive it, so the guard goes immediately after the call in `parse_mod`.
- `puremacro.dsge.decomposition.compute_shock_decomposition(model, data, initial_state=None, *, sigma=None)` → `ShockDecompResult`.
- `puremacro.reports._df_to_markdown / _df_to_latex / _df_to_typst(df, index=None, ...)`.
- Version lives in **four** places, checked by `release_check` gate 4: `pyproject.toml`, `puremacro/__init__.py` (`__version__ = "2.5.0"`), the top `## X.Y.Z` heading in `CHANGELOG.md`, `version:` in `CITATION.cff`. `tests/test_import.py` only checks the *format*, so it needs no bump.
- `tests/test_bilingual_docs.py::_USER_DOCS` is an explicit list; a page added to it needs `docs/es/<name>` and the string `Español` in the first 600 characters of the English file (`English` in the Spanish one).
- `tests/test_docs_nav.py` asserts every `mkdocs.yml` nav target exists and none is duplicated.
- `tools/release_check.py` — gates 1 test baseline, 2 pyodide compat, 3 public API snapshot, 4 version sync, and opt-in 5 examples gallery, 6 pyodide smoke. Snapshot is regenerated from `tests/test_public_api.py::collect_current_api`.
- `sw07_pfeifer.mod` `estimated_params` block: **36 statements**. Parser fixture target.

---

## Task 1: Close the silent macro-directive hole — DONE (e6d83ad)

**Files:** Modify `puremacro/dsge/dynare.py`, `puremacro/dsge/__init__.py`. Create `tests/test_dsge/test_macro_directive_guard.py`.

Today `@#define N = 2` is ignored, the file loads clean, and a **different model is solved**. Every other unsupported construct fails loudly; this one does not. It is fixed first because it is four lines and because everything downstream would otherwise estimate the wrong model without saying so.

- [x] **Step 1: Write the failing tests**

```python
"""Dynare macro directives are not implemented — they must not be ignored."""
from __future__ import annotations

import pytest

from puremacro.dsge import DynareFeatureError, load_mod, parse_mod

_RBC_BODY = """
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
"""
_BASE = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;
alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
model;
""" + _RBC_BODY + """
end;
initval; k = 38.0; a = 0.0; c = 2.0; end;
shocks; var eps; stderr 0.01; end;
"""


@pytest.mark.parametrize("directive", [
    "@#define N = 2",
    "@#include \"common.mod\"",
    "@#if N == 2\n@#endif",
    "@#for i in 1:3\n@#endfor",
])
def test_macro_directive_raises(directive):
    with pytest.raises(DynareFeatureError) as exc:
        parse_mod(directive + "\n" + _BASE)
    assert "macro" in str(exc.value).lower()
    assert "2.7.0" in str(exc.value)


def test_macro_interpolation_raises():
    src = _BASE.replace("rho = 0.80;", "rho = @{rho_val};")
    with pytest.raises(DynareFeatureError):
        parse_mod(src)


def test_directive_inside_a_comment_is_still_ignored():
    """`% @#define N = 2` is a comment, not a directive."""
    parse_mod("% @#define N = 2\n" + _BASE)  # must not raise


def test_guard_is_not_a_valueerror():
    """A caller doing `except ValueError` around parse_mod must not swallow it:
    this failure is different in kind from a malformed declaration."""
    with pytest.raises(DynareFeatureError):
        try:
            load_mod("@#define N = 2\n" + _BASE)
        except ValueError:  # pragma: no cover - the point of the test
            pytest.fail("DynareFeatureError was caught as a ValueError")
```

- [x] **Step 2: Implement**

In `dynare.py`, above `parse_mod`:

```python
class DynareFeatureError(NotImplementedError):
    """A .mod construct puremacro recognises but does not yet implement.

    Deliberately not a ValueError: callers wrap parse failures in
    ``except ValueError`` and this must not be swallowed by them. Silently
    solving a different model is the failure this class exists to prevent.
    """


_MACRO_DIRECTIVE = re.compile(r"^\s*@#\s*(\w+)", re.M)
_MACRO_INTERP = re.compile(r"@\{")
```

In `parse_mod`, immediately after `clean_text = _remove_comments(mod_text)` (so a `%`-commented directive is already gone):

```python
    directive = _MACRO_DIRECTIVE.search(clean_text) or _MACRO_INTERP.search(clean_text)
    if directive is not None:
        raise DynareFeatureError(
            f"this .mod file uses the Dynare macro processor "
            f"({directive.group(0).strip()!r}); puremacro does not implement it yet "
            "(planned for 2.7.0). Parsing on would silently solve a *different* "
            "model from the one the file describes — expand the macros with "
            "Dynare's `dynare model.mod savemacro` and pass the expanded file."
        )
```

Export `DynareFeatureError` from `puremacro/dsge/__init__.py`.

- [x] **Step 3: Run**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_macro_directive_guard.py -q
```

Expected: 7 passed.

- [x] **Step 4: Confirm no existing .mod fixture regresses**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge_dynare_parser.py tests/test_dynare_advanced.py tests/test_dsge_dynare_moments.py -q
```

Expected: all pass (`sw07_pfeifer.mod` contains no `@#`; verified).

- [x] **Step 5: Commit**

```bash
git add puremacro/dsge/dynare.py puremacro/dsge/__init__.py tests/test_dsge/test_macro_directive_guard.py
git commit -m "fix(dsge): raise on Dynare macro directives instead of silently ignoring them"
```

---

## Task 2: Extend `priors.py` for the full `estimated_params` shape set — DONE

**Files:** Modify `puremacro/dsge/priors.py`. Modify `tests/test_dsge/test_priors.py`.

Dynare's `PRIOR_P3` / `PRIOR_P4` shift and scale the beta and gamma families; `weibull_pdf` and `inv_gamma2_pdf` have no class today. Without these, 36-line real blocks cannot round-trip.

- [x] **Step 1: Write failing tests** (append to `tests/test_dsge/test_priors.py`)

```python
def test_generalised_beta_matches_scipy_on_a_shifted_support():
    from scipy import stats
    from puremacro.dsge.priors import BetaPrior
    # mean/std describe the variable on [P3, P4] = [0.2, 0.8].
    p = BetaPrior(mean=0.5, std=0.1, shift=0.2, scale=0.6)
    m01 = (0.5 - 0.2) / 0.6
    s01 = 0.1 / 0.6
    a = m01 * (m01 * (1 - m01) / s01**2 - 1)
    b = a * (1 - m01) / m01
    x = 0.45
    expected = float(stats.beta.logpdf((x - 0.2) / 0.6, a, b)) - np.log(0.6)
    assert p.logpdf(x) == pytest.approx(expected, rel=1e-12)


def test_shifted_gamma_matches_scipy():
    from scipy import stats
    from puremacro.dsge.priors import GammaPrior
    p = GammaPrior(mean=1.0, std=0.5, shift=0.25)
    k = (1.0 / 0.5) ** 2
    theta = 0.5**2 / 1.0
    x = 1.4
    expected = float(stats.gamma.logpdf(x - 0.25, a=k, scale=theta))
    assert p.logpdf(x) == pytest.approx(expected, rel=1e-12)


def test_weibull_prior_matches_scipy():
    from scipy import stats
    from puremacro.dsge.priors import WeibullPrior
    p = WeibullPrior(shape=2.0, scale=1.5)
    x = 1.1
    assert p.logpdf(x) == pytest.approx(
        float(stats.weibull_min.logpdf(x, c=2.0, scale=1.5)), rel=1e-12)


def test_inv_gamma_type2_differs_from_type1_and_matches_scipy():
    from scipy import stats
    from puremacro.dsge.priors import InvGammaPrior
    p2 = InvGammaPrior(s=0.1, nu=4.0, kind="type2")
    x = 0.05
    assert p2.logpdf(x) == pytest.approx(
        float(stats.invgamma.logpdf(x, a=4.0 / 2.0, scale=0.1 / 2.0)), rel=1e-12)
    p1 = InvGammaPrior(s=0.1, nu=4.0)
    assert p1.logpdf(x) != pytest.approx(p2.logpdf(x), rel=1e-6)


@pytest.mark.parametrize("spec,cls", [
    ({"dist": "beta", "mean": 0.5, "std": 0.1, "shift": 0.2, "scale": 0.6}, "BetaPrior"),
    ({"dist": "weibull", "shape": 2.0, "scale": 1.5}, "WeibullPrior"),
    ({"dist": "invgamma", "s": 0.1, "nu": 4.0, "kind": "type2"}, "InvGammaPrior"),
])
def test_ensure_prior_round_trips_the_new_specs(spec, cls):
    from puremacro.dsge import priors as P
    assert type(P.ensure_prior(spec)).__name__ == cls
    assert P.ensure_prior(spec).to_dict()["dist"] == spec["dist"]
```

- [x] **Step 2: Implement**

- `BetaPrior(mean, std, lb=1e-4, ub=0.9999, *, shift=0.0, scale=1.0)`; `GammaPrior(..., *, shift=0.0)`. Both store the extra fields and include them in `to_dict()` / `__repr__`. `logpdf` maps `x → (x - shift) / scale` and subtracts `log(scale)` (Jacobian) for beta; gamma shifts only. `lb`/`ub` default to `shift` / `shift + scale` when `scale != 1`.
- `WeibullPrior(shape, scale, lb=0.0, ub=inf)` — parameterised by shape/scale (Dynare's P1/P2 for this family), with `mean`/`std` derived so the base-class interface stays coherent.
- `InvGammaPrior(..., kind="type1"|"type2")` — type 2 is `invgamma(a=nu/2, scale=s/2)` against type 1's `invgamma(a=nu/2, scale=s**2*nu/2)`.
- `_logpdf_for_spec` gains the branches; `ensure_prior` gains `weibull`, and reads `shift`, `scale`, `kind` when present.
- `_validate_priors` gains: `scale <= 0` for a generalised beta, `shape <= 0` for Weibull, and an unknown `kind`.

**Back-compat requirement:** every existing call site constructs these classes positionally with `(mean, std, lb, ub)`. All new arguments are keyword-only with defaults that reproduce today's behaviour exactly.

- [x] **Step 3: Run**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_priors.py tests/test_dsge/test_sw07_priors.py -q
```

Expected: all pass, including the 9 pre-existing prior tests unchanged.

- [x] **Step 4: Commit**

```bash
git add puremacro/dsge/priors.py tests/test_dsge/test_priors.py
git commit -m "feat(dsge): generalised beta/gamma, Weibull and inverse-gamma type-2 priors"
```

---

## Task 3: `_estimated_params.py` — the grammar — DONE

**Files:** Create `puremacro/dsge/_estimated_params.py`, `tests/test_dsge/test_estimated_params.py`. Modify `puremacro/dsge/dynare.py`, `puremacro/dsge/build.py`.

- [x] **Step 1: Write the module skeleton**

```python
@dataclass(frozen=True)
class EstimatedParamSpec:
    kind: str            # "param" | "stderr_shock" | "corr_shock" | "stderr_obs"
    target: tuple        # ("alpha",) | ("eps_a",) | ("eps_a", "eps_b")
    name: str            # "alpha" | "SE_eps_a" | "CORR_eps_a_eps_b" | "ME_dy"
    prior: Prior | None  # None => estimated without a prior (ML)
    init: float | None
    lb: float
    ub: float
    jscale: float = 1.0


@dataclass(frozen=True)
class EstimatedParams:
    specs: tuple[EstimatedParamSpec, ...]
    def priors(self) -> dict[str, Prior]: ...
    def initial_params(self) -> dict[str, float]: ...
    def by_kind(self, kind: str) -> tuple[EstimatedParamSpec, ...]: ...
    def jscale_vector(self, names: Sequence[str]) -> np.ndarray: ...


def parse_estimated_params(block_text, *, shocks, varobs, params) -> EstimatedParams: ...
def parse_estimated_params_init(block_text) -> dict[str, float]: ...
def parse_estimated_params_bounds(block_text) -> dict[str, tuple[float, float]]: ...
```

**Disambiguation algorithm** (implement exactly this; the grammar is positionally ambiguous):

1. Strip comments, split the block on `;`.
2. Per statement: if it starts with `stderr`, consume the keyword and one identifier; if `corr`, consume the keyword and two identifiers separated by a comma; else consume one identifier. This resolves the `corr a, b` comma before any comma splitting.
3. Split the remainder on commas into `rest`.
4. Find the index of the first element of `rest` that is a known `*_pdf` shape, **matched case-insensitively**. The real `sw07_pfeifer.mod` block writes `INV_GAMMA_PDF` in uppercase while the Dynare manual and the synthetic fixture use lowercase; a case-sensitive match silently reclassifies every statement as "no prior".
   - not found → no prior. `rest` is `[INITVAL]` or `[INITVAL, LB, UB]` by length.
   - found at `i` → `rest[:i]` is `[]`, `[INITVAL]` or `[INITVAL, LB, UB]` by length; `rest[i+1:]` is `P1, P2, P3, P4, JSCALE` by position.
5. Any other length is a `ValueError` quoting the statement verbatim.
6. `kind` is decided by the target: a `varexo` under `stderr` → `stderr_shock`; a `varobs` under `stderr` → `stderr_obs`; `corr` → `corr_shock`; otherwise `param`, and a name that is neither a declared parameter nor a variable is a `ValueError`.

- [x] **Step 2: Write the tests**

```python
_SYNTHETIC = """
estimated_params;
  alpha, 0.35, beta_pdf, 0.35, 0.02;
  rho, , 0, 1, beta_pdf, 0.5, 0.2;
  gam, normal_pdf, 1, 0.5;
  psi, 0.3, 0.0, 1.0, beta_pdf, 0.5, 0.15, 0.2, 0.8;
  omega, gamma_pdf, 1.0, 0.5, 0.25;
  kappa, uniform_pdf, , , 0.0, 4.0;
  lam, 0.9;
  stderr eps_a, inv_gamma_pdf, 0.1, 2;
  stderr eps_b, 0.05, 0.001, 3, inv_gamma_pdf, 0.1, 2, , , 0.5;
  stderr dy, inv_gamma_pdf, 0.01, 2;
  corr eps_a, eps_b, normal_pdf, 0, 0.2;
end;
"""

def test_every_statement_form_parses():
    ep = parse_estimated_params(_SYNTHETIC, shocks=["eps_a", "eps_b"],
                                varobs=["dy"], params=[...])
    assert len(ep.specs) == 11

def test_kinds_and_names():
    ...
    assert {s.name for s in ep.by_kind("stderr_shock")} == {"SE_eps_a", "SE_eps_b"}
    assert [s.name for s in ep.by_kind("stderr_obs")] == ["ME_dy"]
    assert [s.name for s in ep.by_kind("corr_shock")] == ["CORR_eps_a_eps_b"]

def test_corr_comma_does_not_confuse_the_field_split():
    spec = next(s for s in ep.specs if s.kind == "corr_shock")
    assert spec.target == ("eps_a", "eps_b")
    assert spec.prior.mean == 0.0 and spec.prior.std == 0.2

def test_generalised_beta_picks_up_p3_p4():
    spec = _by_name(ep, "psi")
    assert (spec.prior.shift, spec.prior.scale) == (0.2, 0.6)

def test_jscale_is_read_from_the_tenth_field():
    assert _by_name(ep, "SE_eps_b").jscale == 0.5

def test_initval_lb_ub_forms():
    assert _by_name(ep, "rho").init is None
    assert (_by_name(ep, "rho").lb, _by_name(ep, "rho").ub) == (0.0, 1.0)
    assert _by_name(ep, "alpha").init == 0.35

def test_no_prior_means_prior_is_none():
    assert _by_name(ep, "lam").prior is None and _by_name(ep, "lam").init == 0.9

def test_unknown_shape_raises_naming_it():
    with pytest.raises(ValueError, match="dirichlet_pdf"):
        parse_estimated_params("estimated_params; x, dirichlet_pdf, 1, 1; end;", ...)

def test_unknown_target_raises_naming_it():
    with pytest.raises(ValueError, match="nosuchparam"):
        parse_estimated_params("estimated_params; nosuchparam, normal_pdf, 0, 1; end;", ...)

def test_sw07_pfeifer_block_round_trips():
    """The real thing: 36 statements, all in the
    `NAME, INITVAL, LB, UB, PRIOR_SHAPE, P1, P2` form with UPPERCASE shape names
    (`stderr ea,0.4618,0.01,3,INV_GAMMA_PDF,0.1,2;`), and 7 of them `stderr`."""
    text = (Path(puremacro.dsge.__file__).parent / "_references" / "sw07_pfeifer.mod").read_text()
    parsed = parse_mod(text)
    ep = parsed["estimated_params"]
    assert len(ep.specs) == 36
    assert all(s.prior is not None for s in ep.specs)
    assert len(ep.by_kind("stderr_shock")) == 7          # ea eb eg eqs em epinf ew
    means = ep.priors()
    assert set(means) == {s.name for s in ep.specs}

def test_estimated_params_init_and_bounds_override():
    ...
```

- [x] **Step 3: Wire into `parse_mod`**

`parse_mod` currently strips `estimated_params` in `block_pattern` (`dynare.py:936`). Keep the strip (the block must not pollute the top-level parameter scan), and **additionally** capture it:

```python
    ep_match = re.search(r"\bestimated_params\s*;\s*(.*?)\bend\s*;", clean_text, re.DOTALL)
    estimated_params = (
        parse_estimated_params(ep_match.group(1), shocks=shocks, varobs=varobs,
                               params=sorted(declared_params))
        if ep_match else None
    )
```

New keys in the returned dict: `estimated_params`, `estimated_params_init`, `estimated_params_bounds`, `observation_trends`, `estimation_options`. Existing keys are untouched — the addition is purely additive, and `tests/test_dsge_dynare_parser.py` must pass unchanged.

- [x] **Step 4: Attach to the model**

`LinearModel` is `@dataclass(frozen=True)`: add three fields with `None` defaults —

```python
    _varobs: tuple | None = None
    _estimated_params: Any | None = None
    _mod_options: dict | None = None
```

and in `load_mod`, after the model is built, attach with `dataclasses.replace(model, _varobs=..., _estimated_params=...)`. **Never** by assignment — the dataclass is frozen.

- [x] **Step 5: Run**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_estimated_params.py tests/test_dsge_dynare_parser.py -q
```

- [x] **Step 6: Commit**

```bash
git add puremacro/dsge/_estimated_params.py puremacro/dsge/dynare.py puremacro/dsge/build.py tests/test_dsge/test_estimated_params.py
git commit -m "feat(dsge): parse the estimated_params block and attach it to the solved model"
```

---

## Task 4: `observation.py` — `varobs` to a state space — DONE

**Files:** Create `puremacro/dsge/observation.py`, `tests/test_dsge/test_observation.py`.

The algebra, from the spec. `_extract_companion_matrices` gives `x_{t+1} = A x_t + B u_t`, `v_t = ys + C x_t + D u_t`. Because `v_t` loads the contemporaneous innovation and `StateSpaceModel` assumes `η ⟂ ε`, the innovation is carried in the state:

```
alpha_t = [x_t ; u_t]
T = [[A, B], [0, 0]]      R = [[0], [I_ne]]      Q = Sigma_u
Z = [C_O, D_O]            d = ys_O               H = diag(sigma_ME^2)
```

Check: `T alpha_{t-1} + R u_t = [A x_{t-1} + B u_{t-1}; 0] + [0; u_t] = [x_t; u_t]`.

Two consequences to write into the docstring: growth-rate observables need no special casing (`dy = y - y(-1)` is a declared variable, so it is just a row of `(C, D)`), and **the smoothed structural shocks are the last `n_e` rows of `a_smooth`** — no separate disturbance smoother is needed. Task 5 depends on that.

- [x] **Step 1: Signature**

```python
def make_state_space_from_varobs(
    model, varobs, *, shock_cov=None, measurement_error=None,
    prefilter=False, ridge=0.0,
) -> StateSpaceModel
```

`measurement_error` is `{obs_name: std}`; `H = diag(std**2)` and defaults to **zero** (Dynare's default), not the `1e-8` ridge `sw07_observation.py` uses. `ridge` adds `ridge * I` on top when a caller wants conditioning. `prefilter=True` sets `d = 0`. An observable not among `model.variables` raises naming it and listing the nearest matches.

- [x] **Step 2: Write the decisive tests**

```python
"""make_state_space_from_varobs — the varobs -> Kalman connector."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from puremacro.dsge import load_mod
from puremacro.dsge.decomposition import _extract_companion_matrices
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter

_RBC = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;
alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;
initval; k = 38.0; a = 0.0; c = 2.0; end;
shocks; var eps; stderr 0.01; end;
"""


@pytest.fixture(scope="module")
def rbc():
    return load_mod(_RBC)


def test_augmented_state_space_reproduces_the_model_recursion(rbc):
    """The (T, R, Z, d) construction must trace the same path as the model's
    own companion recursion — the loop LinearModel.simulate runs."""
    A, B, C, D, ys, variables, shocks, states, _ = _extract_companion_matrices(rbc)
    obs = ["c", "k"]
    rows = [variables.index(v) for v in obs]
    ssm = make_state_space_from_varobs(rbc, obs)

    rng = np.random.default_rng(0)
    n_t = 25
    u = rng.standard_normal((n_t, len(shocks))) * 0.01

    ref = np.zeros((n_t, len(obs)))
    x = np.zeros(len(states))
    for t in range(n_t):
        ref[t] = (ys + C @ x + D @ u[t])[rows]
        x = A @ x + B @ u[t]

    got = np.zeros((n_t, len(obs)))
    alpha = np.zeros(ssm.T.shape[0])
    for t in range(n_t):
        alpha = ssm.T @ alpha + ssm.R @ u[t]
        got[t] = ssm.d + ssm.Z @ alpha

    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_shapes_and_blocks(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c"])
    n_s, n_e = rbc.n_states, len(rbc.shocks)
    assert ssm.T.shape == (n_s + n_e, n_s + n_e)
    assert ssm.R.shape == (n_s + n_e, n_e)
    assert ssm.Q.shape == (n_e, n_e)
    np.testing.assert_allclose(ssm.T[n_s:], 0.0)               # shock block is iid
    np.testing.assert_allclose(ssm.R[:n_s], 0.0)
    np.testing.assert_allclose(ssm.R[n_s:], np.eye(n_e))


def test_kalman_loglik_matches_a_stacked_multivariate_normal(rbc):
    """Independent check of T, R, Z, Q, H, d together: the filter's
    log-likelihood must equal the density of the stacked observations."""
    ssm = make_state_space_from_varobs(rbc, ["c"], ridge=1e-4)
    Tm, Rm, Qm, Zm, Hm, dv = ssm.T, ssm.R, ssm.Q, ssm.Z, ssm.H, ssm.d
    P = solve_discrete_lyapunov(Tm, Rm @ Qm @ Rm.T)

    n, n_t = Zm.shape[0], 8
    Sigma = np.zeros((n * n_t, n * n_t))
    for i in range(n_t):
        for j in range(n_t):
            h = i - j
            cross = (np.linalg.matrix_power(Tm, h) @ P) if h >= 0 \
                else P @ np.linalg.matrix_power(Tm.T, -h)
            block = Zm @ cross @ Zm.T + (Hm if h == 0 else 0.0)
            Sigma[i * n:(i + 1) * n, j * n:(j + 1) * n] = block
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = np.tile(dv, n_t)

    rng = np.random.default_rng(1)
    y = rng.multivariate_normal(mu, Sigma).reshape(n_t, n)
    ref = float(multivariate_normal(mean=mu, cov=Sigma).logpdf(y.ravel()))
    got = kalman_filter(y, ssm, a0=np.zeros(Tm.shape[0]), P0=P)["loglik"]
    assert got == pytest.approx(ref, rel=1e-8)


def test_observable_covariance_matches_theoretical_moments(rbc):
    """Z P Z' + H must equal the model's own analytical covariance block.

    This holds because the Lyapunov solution has Cov(x_t, u_t) = 0 (the [x, u]
    block of both T P T' and R Q R' is zero), so Z P Z' is exactly
    Var(C x_t + D u_t) -- the observables in the timing the model reports."""
    obs = ["c", "k"]
    ssm = make_state_space_from_varobs(rbc, obs)
    P = solve_discrete_lyapunov(ssm.T, ssm.R @ ssm.Q @ ssm.R.T)
    n_s = rbc.n_states
    np.testing.assert_allclose(P[:n_s, n_s:], 0.0, atol=1e-12)   # the premise
    got = ssm.Z @ P @ ssm.Z.T + ssm.H
    ref = rbc.theoretical_moments().covariance.loc[obs, obs].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_prefilter_zeroes_the_intercept(rbc):
    assert np.any(make_state_space_from_varobs(rbc, ["c"]).d != 0.0)
    np.testing.assert_allclose(
        make_state_space_from_varobs(rbc, ["c"], prefilter=True).d, 0.0)


def test_measurement_error_lands_on_the_diagonal_of_H(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c", "k"], measurement_error={"k": 0.02})
    np.testing.assert_allclose(np.diag(ssm.H), [0.0, 0.02 ** 2])


def test_unknown_observable_raises_naming_it(rbc):
    with pytest.raises(ValueError, match="output"):
        make_state_space_from_varobs(rbc, ["output"])


@pytest.mark.slow
def test_sw07_cross_check_against_the_hand_built_observation_equation():
    """Informational: the generic route on sw07_pfeifer.mod and the hand-built
    sw07_observation.make_state_space are two independent ports of the same
    model. A mismatch here is a finding about one of the ports, NOT
    automatically a defect in make_state_space_from_varobs — the tests above
    are the gate for this task."""
    ...
```

`TheoreticalMomentsResult` has **no** `.variance` attribute — checked. It carries `.moments` (columns `Mean`, `Std.Dev.`, `Variance`), `.covariance`, `.correlation`, `.autocorr`, `.fevd`. The test uses `.covariance` because it is unambiguous about timing; `load_mod` models are Dynare-timed, so `covariance` and the state space agree row for row.

- [x] **Step 3: Run**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_observation.py -q -m "not slow"
```

- [x] **Step 4: Commit**

```bash
git add puremacro/dsge/observation.py tests/test_dsge/test_observation.py
git commit -m "feat(dsge): build the Kalman measurement equation from varobs"
```

---

## Task 5: `LinearModel.smoother()` and `.forecast()` — Dynare's `calib_smoother` — DONE

**Files:** Create `puremacro/dsge/smoother.py`, `tests/test_dsge/test_smoother.py`. Modify `puremacro/dsge/build.py`, `puremacro/dsge/_results.py`.

Because the state is `[x_t; u_t]`, `kalman_smoother`'s `a_smooth` already contains the smoothed structural shocks in its last `n_e` columns. That is the whole implementation.

- [x] **Step 1: Result objects** in `_results.py`

```python
@dataclass(frozen=True)
class SmootherResult:
    states: pd.DataFrame        # (T, n_states) smoothed model states
    shocks: pd.DataFrame        # (T, n_shocks) smoothed structural innovations
    smoothed_obs: pd.DataFrame  # (T, n_obs) fitted observables
    filtered_states: pd.DataFrame
    loglik: float
    varobs: tuple
    def shock_decomposition(self) -> ShockDecompResult: ...
    # + summary / plot / to_markdown / to_latex / to_typst

@dataclass(frozen=True)
class DSGEForecastResult:
    mean: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    ci: float
    horizon: int
    # + the five presentation methods
```

- [x] **Step 2: Methods on `LinearModel`**

```python
def smoother(self, data, *, varobs=None, params=None, measurement_error=None,
             prefilter=False, observation_trends=None, a0=None, P0=None) -> SmootherResult
def forecast(self, horizon=8, *, data=None, ci=0.90, **kw) -> DSGEForecastResult
```

`varobs` defaults to `self._varobs`. `observation_trends` subtracts `a_i + b_i * t` from column `i` before filtering and adds it back to `smoothed_obs` and to the forecast — `StateSpaceModel` is time-invariant, so the trend cannot live inside the filter.

- [x] **Step 3: The decisive test**

```python
def test_smoother_recovers_the_simulated_shocks_exactly(rbc):
    """One observable, one shock, no measurement error, known x_0: the map
    from observables to shocks is bijective, so the smoother must invert it."""
    A, B, C, D, ys, variables, shocks, states, _ = _extract_companion_matrices(rbc)
    rng = np.random.default_rng(3)
    n_t = 200
    u_true = rng.standard_normal((n_t, 1)) * 0.01

    row = variables.index("c")
    y = np.zeros((n_t, 1))
    x = np.zeros(len(states))
    for t in range(n_t):
        y[t, 0] = (ys + C @ x + D @ u_true[t])[row]
        x = A @ x + B @ u_true[t]

    res = rbc.smoother(pd.DataFrame(y, columns=["c"]),
                       a0=np.zeros(rbc.n_states + 1), P0=1e-10 * np.eye(rbc.n_states + 1))
    got = res.shocks["eps"].to_numpy()
    np.testing.assert_allclose(got[10:], u_true[10:, 0], rtol=0, atol=1e-6)


def test_smoothed_observables_reproduce_the_data(rbc):
    """With no measurement error the fitted observables are the data."""
    ...

def test_forecast_mean_converges_to_the_steady_state(rbc):
    res = rbc.forecast(horizon=400, data=...)
    np.testing.assert_allclose(res.mean.iloc[-1].to_numpy(),
                               rbc.steady_state[list(res.mean.columns)].to_numpy(),
                               rtol=1e-4)

def test_forecast_bands_widen_monotonically(rbc): ...
def test_shock_decomposition_components_sum_to_the_series(rbc): ...
def test_observation_trends_round_trip(rbc): ...
```

- [x] **Step 4: Run and commit**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_smoother.py -q
git add puremacro/dsge/smoother.py puremacro/dsge/_results.py puremacro/dsge/build.py tests/test_dsge/test_smoother.py
git commit -m "feat(dsge): LinearModel.smoother and .forecast (Dynare calib_smoother)"
```

---

## Task 6: `mode.py` — the mode-search menu

**Files:** Create `puremacro/dsge/mode.py`, `tests/test_dsge/test_mode_search.py`. Modify `puremacro/dsge/estimate.py`.

The 2.5.0 audit found the single L-BFGS-B run returning `initial_params` bit-identically with `converged_mle=True`. The fix then was a finite penalty; the fix now is alternatives.

- [ ] **Step 1: Implement**

```python
def csminwel(f, x0, *, bounds=None, h0=1e-4, tol=1e-7, max_iter=500) -> OptimizeResult
def cmaes(f, x0, *, sigma0=0.3, bounds=None, popsize=None, max_iter=1000, seed=0) -> OptimizeResult
def find_mode(f, x0, *, method="csminwel", bounds=None, **kw) -> OptimizeResult
def mode_check(f, mode_vec, names, *, n_points=20, width=2.0, cov=None) -> ModeCheckResult
```

| `mode_compute` | algorithm |
|---|---|
| `"lbfgs"` | today's bounded L-BFGS-B (kept; still the compatibility setting) |
| `"simplex"` | Nelder-Mead with restarts |
| `"csminwel"` | Sims' quasi-Newton with the perturbed-Hessian retry; the new default |
| `"cmaes"` | covariance-matrix adaptation ES, pure numpy |
| `"none"` | skip; start MCMC at `init` |

`find_mode` always re-reports the numerical Hessian through the existing `_nearest_pd` eigenvalue floor and preserves the 2.5.0 warnings: a mode that did not move, and a Hessian that was not positive definite, each say so.

`estimate_dsge` gains `mode_compute="csminwel"` — but the **default must stay `"lbfgs"` for one release** so the frozen `tests/fixtures/sw07_parity_seed0_200draws.npz` parity test keeps passing bit-for-bit. Flip the default in 2.7.0 with a fixture refresh, and say so in the CHANGELOG.

- [ ] **Step 2: Tests**

```python
@pytest.mark.parametrize("method", ["lbfgs", "simplex", "csminwel", "cmaes"])
def test_each_method_finds_a_known_maximum(method):
    """-(x-3)^2 - 2(y+1)^2, maximum at (3, -1)."""
    f = lambda v: (v[0] - 3.0) ** 2 + 2.0 * (v[1] + 1.0) ** 2
    res = find_mode(f, np.array([0.0, 0.0]), method=method)
    np.testing.assert_allclose(res.x, [3.0, -1.0], atol=1e-3)

def test_csminwel_on_rosenbrock(): ...
def test_cmaes_escapes_a_local_optimum_that_lbfgs_does_not(): ...
def test_bounds_are_respected(): ...
def test_mode_that_does_not_move_warns(): ...
def test_mode_check_slices_peak_at_the_mode(): ...
def test_mode_check_result_plots(): ...
def test_unknown_method_raises_listing_the_menu(): ...
```

- [ ] **Step 3: Run and commit**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_mode_search.py tests/test_dsge/test_sw07_wrapper.py -q
git add puremacro/dsge/mode.py puremacro/dsge/estimate.py tests/test_dsge/test_mode_search.py
git commit -m "feat(dsge): mode_compute menu (csminwel, CMA-ES, simplex) and mode_check"
```

The `test_sw07_wrapper.py` parity test is in that run deliberately: it is the guard that the default path did not shift.

---

## Task 7: `marginal.py` — marginal likelihood and model comparison

**Files:** Create `puremacro/dsge/marginal.py`, `tests/test_dsge/test_marginal_likelihood.py`. Modify `puremacro/dsge/estimate.py`, `puremacro/dsge/_results.py`.

- [ ] **Step 1: Record the missing input**

`DSGEPosteriorResult` has no log posterior *at the mode*. Add `log_post_mode: float | None = None` and have `estimate_dsge` fill it. Optional with a `None` default, so existing pickles and callers are unaffected.

- [ ] **Step 2: Implement**

```python
def laplace_mdd(log_post_mode: float, hessian_inv: np.ndarray) -> float:
    """log p(y) ~= log p(y|th*) + log p(th*) + (d/2) log 2pi + 0.5 log|Sigma*|."""

def harmonic_mean_mdd(draws, log_post, *, p=(0.1, ..., 0.9)) -> HarmonicMeanResult:
    """Geweke (1999) modified harmonic mean, evaluated at every truncation
    level. The SPREAD across levels is returned, not hidden: a swing of more
    than ~1 log point means the estimate has not converged and .converged is
    False."""

def model_comparison(results, *, model_priors=None, method="laplace") -> pd.DataFrame
```

`model_comparison` refuses to mix estimators across models — a Laplace value and a harmonic-mean value are not comparable, and silently mixing them is how a Bayes factor becomes fiction. The module docstring records that marginal likelihoods computed before 2.5.0 are not comparable at all, because the Kalman recursion then started from a diffuse `P0` rather than the unconditional covariance.

- [ ] **Step 3: The analytic test**

```python
def test_laplace_is_exact_for_a_gaussian_posterior():
    """y_i ~ N(theta, s^2) iid, prior theta ~ N(m0, t0^2). The posterior is
    Gaussian, so Laplace is not an approximation — it is the answer."""
    rng = np.random.default_rng(0)
    n, s, m0, t0 = 50, 1.3, 0.2, 0.8
    y = rng.normal(1.0, s, size=n)

    Sigma = s**2 * np.eye(n) + t0**2 * np.ones((n, n))
    ref = float(multivariate_normal(mean=np.full(n, m0), cov=Sigma).logpdf(y))

    prec = n / s**2 + 1 / t0**2
    post_mean = (y.sum() / s**2 + m0 / t0**2) / prec
    log_post_mode = (
        -0.5 * n * np.log(2 * np.pi * s**2) - 0.5 * ((y - post_mean) ** 2).sum() / s**2
        - 0.5 * np.log(2 * np.pi * t0**2) - 0.5 * (post_mean - m0) ** 2 / t0**2
    )
    got = laplace_mdd(log_post_mode, np.array([[1.0 / prec]]))
    assert got == pytest.approx(ref, rel=1e-12)


def test_harmonic_mean_recovers_the_same_number_on_the_same_model():
    """Same conjugate model, 200k draws from the exact posterior."""
    ...  # tolerance 0.05 log points; assert .converged is True

def test_harmonic_mean_reports_a_wide_spread_when_it_has_not_converged():
    """40 draws: the estimate must report itself unreliable, not quietly return."""
    ...  # assert res.converged is False and res.spread > 1.0

def test_model_comparison_refuses_to_mix_methods(): ...
def test_model_comparison_posterior_odds_sum_to_one(): ...
def test_log_mdd_accessor_on_the_result_object(): ...
```

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_marginal_likelihood.py -q
git add puremacro/dsge/marginal.py puremacro/dsge/estimate.py puremacro/dsge/_results.py tests/test_dsge/test_marginal_likelihood.py
git commit -m "feat(dsge): Laplace and modified-harmonic-mean marginal likelihood, model_comparison"
```

---

## Task 8: `LinearModel.estimate()` — the connector

**Files:** Modify `puremacro/dsge/build.py`. Create `tests/test_dsge/test_estimate_from_mod.py`.

- [ ] **Step 1: Signature**

```python
def estimate(self, data, *, priors=None, varobs=None, mode_compute="lbfgs",
             n_draws=10_000, n_chains=2, burn_in=2_000, prefilter=False,
             observation_trends=None, measurement_error=None,
             diffuse_filter=False, seed=0) -> DSGEPosteriorResult
```

`priors` and `varobs` default to `self._estimated_params` and `self._varobs`; a model built without a `.mod` file and given neither raises saying which is missing.

- [ ] **Step 2: The observation closure**

```python
def _make_observation_eq(model, specs, varobs, **kw):
    """theta -> StateSpaceModel, re-solving only when structural params move."""
    structural = [s.name for s in specs if s.kind == "param"]
    cache = {}

    def observation_eq(params: dict) -> StateSpaceModel:
        key = tuple(round(float(params[n]), 15) for n in structural)
        if key not in cache:
            cache.clear()                       # depth 1: MCMC never revisits
            cache[key] = _resolve(model, params, warm_start=observation_eq._last_ss)
            observation_eq.n_solves += 1
        solved = cache[key]
        Sigma_u = _assemble_shock_cov(specs, params, solved.shocks)
        H = _assemble_measurement_cov(specs, params, varobs)
        return make_state_space_from_varobs(solved, varobs, shock_cov=Sigma_u,
                                            measurement_error=H, **kw)

    observation_eq.n_solves = 0
    observation_eq._last_ss = None
    return observation_eq
```

Two performance requirements, both measurable:

1. **Only `kind == "param"` triggers a re-solve.** `stderr` / `corr` / measurement-error draws write into `Q` and `H` directly.
2. **Warm start.** `_resolve` seeds `scipy.optimize.root` with the previous accepted draw's steady state instead of the static `initval` guess. At the measured 0.057 s per SW07 solve, 10 000 draws × 2 chains is ~19 minutes of solving before the filter runs at all.

`observation_eq.n_solves` is a documented test hook, not incidental state.

- [ ] **Step 3: Tests**

```python
def test_estimate_recovers_known_parameters_on_a_simulated_ar1_mod():
    """A .mod file whose data was simulated at known rho, sigma. The posterior
    mean must land inside a 3-posterior-sd band of the truth."""
    ...

def test_estimate_defaults_to_the_mod_files_own_varobs_and_estimated_params():
    m = load_mod(_AR1_MOD_WITH_BLOCKS)
    res = m.estimate(data, n_draws=200, n_chains=1, burn_in=50, seed=0)
    assert set(res.param_names) == {"rho", "SE_eps"}

def test_shock_std_draws_do_not_trigger_a_re_solve():
    eq = _make_observation_eq(...)
    eq({"rho": 0.8, "SE_eps": 0.01}); assert eq.n_solves == 1
    eq({"rho": 0.8, "SE_eps": 0.02}); assert eq.n_solves == 1   # Q only
    eq({"rho": 0.7, "SE_eps": 0.02}); assert eq.n_solves == 2   # structural

def test_warm_start_does_not_change_the_answer():
    """Cold and warm steady states must agree to solver tolerance — the
    speed-up must not buy a different model."""
    ...

def test_stochastic_singularity_is_reported_before_sampling():
    """3 observables, 1 shock, no measurement error."""
    with pytest.raises(ValueError, match="stochastically singular"):
        m.estimate(data_3col, ...)

def test_missing_varobs_and_priors_raises_naming_both(): ...

@pytest.mark.slow
def test_sw07_end_to_end_from_the_mod_file():
    """The headline. 200 draws, one chain — a smoke test of the whole path,
    not a posterior. Asserts: it runs, the log-likelihood at the mode is
    finite, and the parameter set matches the 36-entry block."""
    ...
```

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_estimate_from_mod.py -q -m "not slow"
PYTHONPATH=. python3 -m pytest tests/test_dsge/test_estimate_from_mod.py -q -m slow
git add puremacro/dsge/build.py tests/test_dsge/test_estimate_from_mod.py
git commit -m "feat(dsge): LinearModel.estimate — Bayesian estimation straight from a .mod file"
```

---

## Task 9: Exports and the public surface

- [ ] **Step 1** — add to `puremacro/dsge/__init__.py` and `__all__`: `DynareFeatureError`, `EstimatedParamSpec`, `EstimatedParams`, `parse_estimated_params`, `make_state_space_from_varobs`, `find_mode`, `mode_check`, `csminwel`, `cmaes`, `laplace_mdd`, `harmonic_mean_mdd`, `model_comparison`, `SmootherResult`, `DSGEForecastResult`, `WeibullPrior`, and the `marginal`, `mode`, `observation` modules.
- [ ] **Step 2** — confirm every new result class implements `.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`.

```bash
PYTHONPATH=. python3 -m pytest tests/test_reports_export.py tests/test_public_api.py -q
```

There is no single `test_result_contract.py` in the tree — checked. The presentation contract is
enforced per module; `tests/test_public_api.py::collect_current_api` records `result_classes` and
their fields, so a new result class shows up in the snapshot diff (gate 3) whether or not it has a
dedicated test. Add the five-method assertions to the new per-module test files.

- [ ] **Step 3: Commit**

---

## Task 10: Documentation

- [ ] **Step 1** — write `docs/dsge_estimation.md` with runnable blocks: load a `.mod`, `.check()` it, `.smoother()`, `.estimate()`, marginal likelihood, `mode_check`. First 600 characters **must** contain `Español` and a link to `es/dsge_estimation.md`; copy the switcher header from `docs/gvar.md`.
- [ ] **Step 2** — write `docs/es/dsge_estimation.md` (native academic Spanish, first 600 characters contain `English`).
- [ ] **Step 3** — add both to `mkdocs.yml` nav under "Econometric Engines"; add `"dsge_estimation.md"` to `_USER_DOCS` in `tests/test_bilingual_docs.py`.
- [ ] **Step 4** — if the English page's code blocks are to be executed, add it to the list in `tests/test_docs_code_blocks.py`; otherwise leave it out deliberately and say why in the page.

```bash
PYTHONPATH=. python3 -m pytest tests/test_docs_nav.py tests/test_bilingual_docs.py tests/test_docs_code_blocks.py -q
```

- [ ] **Step 5: Commit**

---

## Task 11: Version, changelog, API snapshot

- [ ] **Step 1** — bump to `2.6.0` in **all four**: `pyproject.toml`, `puremacro/__init__.py`, the top `## 2.6.0 (2026-XX-XX)` heading in `CHANGELOG.md`, `version:` in `CITATION.cff`.
- [ ] **Step 2** — write the CHANGELOG section in house style: what each new estimator does, and **what it deliberately does not do** (no AST, so `@#`/`STEADY_STATE()`/`#`-locals over endogenous variables still fail; no order 3; no particle filter; `observation_trends` handled by detrending, not inside the filter; `mode_compute` default still `"lbfgs"` this release). Lead with the one behaviour change: `@#` now raises.
- [ ] **Step 3** — regenerate the API snapshot **from a clean checkout**, not the live tree:

```bash
git worktree add /tmp/pm-clean HEAD
cd /tmp/pm-clean && PYTHONPATH=. python3 -c "
import json, sys; sys.path.insert(0, 'tests')
from test_public_api import collect_current_api
json.dump(collect_current_api(), open('tests/fixtures/public_api_snapshot.json','w'), indent=2, sort_keys=True)
"
cp /tmp/pm-clean/tests/fixtures/public_api_snapshot.json tests/fixtures/
git worktree remove /tmp/pm-clean
```

Regenerating in the working tree risks baking a file-sync artifact into the release fixture.

- [ ] **Step 4: Commit**

---

## Task 12: Release gates

- [ ] **Step 1**

```bash
PYTHONPATH=. python3 tools/release_check.py
```

Expected: `all 4 gates PASS`.

- [ ] **Step 2**

```bash
PYTHONPATH=. python3 tools/release_check.py --examples --pyodide
```

Expected: `all 6 gates PASS`. Diagnose any failure at its cause — no `--no-verify`, no hook skips, no whitelist edits to make a real failure disappear.

---

## Deferred findings (surfaced while executing this plan, not fixed here)

**F1 — `random_walk_metropolis` adaptation is a silent no-op for `burn_in < 100`.**
`puremacro/mcmc.py` adapts the scalar proposal scale only when `(it + 1) % adapt_window == 0` with
`adapt_window = 100`, so any `adapt_burnin` below 100 never adapts once. The proposal keeps its
initial scale — on SW07 the oversized `diag(prior_stds**2)` Hessian fallback — and the chain rejects
everything, with no warning. Measured on `main` at 2.5.0, `estimate_sw07(seed=0, n_chains=1,
n_draws=200, burn_in=B)`:

| `B` | adaptation windows | acceptance | distinct draws |
|---|---|---|---|
| 50 | 0 | **0.000** | 1 / 200 |
| 200 | 2 | 0.110 | 12 / 100 |

Not fixed here: every candidate fix (scale the window with `adapt_burnin`, adapt on a fractional
schedule, warn when `adapt_burnin < adapt_window`) changes the draws of every existing posterior, and
`estimate_dsge`'s sampler is Task 6's territory. `tests/test_dsge/test_sw07_wrapper.py` now runs at
`burn_in=500` so its sampler invariants are testable at all. Candidate home: 2.6.0 Task 6, or 2.8.0
alongside the SMC/slice samplers.

**F2 — slow-marked tests run in neither the default suite nor `release_check` gate 1.**
`pyproject.toml` `addopts` carries `-m "not slow and not network and not reference and not
replication"`, and `run_pytest_collect_failures` independently passes `-m "not network and not
slow"`. That is how F1 and the stale `sw07_parity_seed0_200draws.npz` both survived a release with
`known_failures.json` empty. Not fixed here (it is a CI-policy decision, not a Phase A change), but
every task in this plan that leans on a slow test must run it explicitly.

---

## Self-review checklist (run AFTER all 12 tasks)

1. **Spec coverage** — every Phase A component in `docs/specs/2026-09-07-puremacro-dsge-tier1-design.md` maps to a task: A1 grammar → Tasks 2+3; A2 observation equation → Task 4; A3 `estimate`/`smoother`/`forecast` → Tasks 5+8; A4 mode search → Task 6; A5 marginal likelihood → Task 7; the `@#` guard → Task 1; exports/docs/release → Tasks 9-12.
2. **Validation coverage** — spec validation items 1-5, 13 and 14 all have a named test. Item 1 (SW07 observation cross-check) is `@pytest.mark.slow` and **informational**, because it compares two independent ports of SW07; the gate for Task 4 is the recursion, stacked-MVN and theoretical-moment tests.
3. **Type consistency** —
   - `EstimatedParamSpec.kind` values `{"param", "stderr_shock", "corr_shock", "stderr_obs"}` used identically in Tasks 3, 8.
   - `make_state_space_from_varobs(model, varobs, *, shock_cov, measurement_error, prefilter, ridge)` — same signature in Tasks 4, 5, 8.
   - `_extract_companion_matrices` 9-tuple unpacked in the same order in Tasks 4, 5.
   - `StateSpaceModel` block shapes `T (m,m)`, `R (m,r)`, `Q (r,r)`, `Z (n,m)`, `d (n,)` with `m = n_states + n_shocks`, `r = n_shocks` — consistent in Tasks 4, 5, 8.
   - `find_mode(f, x0, *, method, bounds, **kw)` — same in Tasks 6 and 8.
4. **Frozen-dataclass discipline** — `LinearModel` is `frozen=True`; Task 3 attaches metadata with `dataclasses.replace`, never assignment. Grep for `object.__setattr__` on `LinearModel` before finishing.
5. **Back-compat** — every new `Prior` argument is keyword-only with a today-preserving default (Task 2); `parse_mod`'s existing keys are unchanged (Task 3); `estimate_dsge`'s `mode_compute` default stays `"lbfgs"` so `tests/fixtures/sw07_parity_seed0_200draws.npz` still passes bit-for-bit (Task 6); every new `DSGEPosteriorResult` field defaults to `None` (Task 7).
6. **Placeholder scan** — Tasks 3, 5, 6, 8 contain `...` in test bodies by design (the assertion is specified, the fixture plumbing is not). No `...` may survive in shipped `puremacro/` source.
7. **Honesty pass** — the CHANGELOG and each new module docstring state the limits: no AST, no order 3, RWMH still adapts only a scalar, `harmonic_mean_mdd` reports its own spread, and the SW07 cross-check is informational rather than a gate.
