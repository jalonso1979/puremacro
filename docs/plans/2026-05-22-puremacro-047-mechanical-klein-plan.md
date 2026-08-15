# puremacro 0.47.0 — garch_utils rename + ProxySVAR axis flip + Klein hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining items from the consolidate-and-finish spec that survived API verification: rename `lp/garch_utils.py` → `lp/_garch_utils.py` (A2), flip `ProxySVARResult.irf_point` from `(n, n, H+1)` to `(H+1, n, n)` to match the other six SVAR result dataclasses (A3), and lift `smets_wouters._solve_F_sylvester` into `dsge/klein.py` as a Z-partition-degenerate fallback so SW07 can stop carrying its local workaround (C). Tag as **0.47.0**, gated by `tools/release_check.py`.

**Architecture:** Three independent tracks land in one release. A2 is mechanical (one rename + three import updates, no behavior change, no notebook re-execution per `feedback_builder_clobbers_outputs`). A3 changes a public-API return shape; touches `_results.py`, `proxy.py`, five example/notebook call sites, R1_04 paired builder, and one new shape-locking unit test. C ports `_solve_F_sylvester` from SW07 into `klein.solve()` as a conditional fallback when the Z-partition is degenerate; SW07's local workaround is deleted; the existing 10 SW07 unit tests must remain byte-equal as the regression contract.

**Tech Stack:** Python ≥3.10, numpy/scipy/pandas/matplotlib, pytest, jupyter nbconvert.

**Source spec:** `docs/specs/2026-05-22-puremacro-046-047-consolidate-finish-design.md` § 0.47.0 (commit `a2065ff`). A1 (`regress/lp.py` retirement) is **dropped** from this release per the spec's own "pause and re-spec" clause — API verification confirmed `regress.lp_panel` and `lp.panel.panel_lp` differ in signature (`shock`/`unit`/`date` vs `x`/`entity_level`/`time_level`), return columns (`horizon`/`ci_lo`/`ci_hi` vs `h`/`lo`/`hi`), and SE method (Driscoll-Kraay vs cluster-by-entity), so the file is legitimately distinct, not a deferred shim.

**Pre-execution state (HEAD `f66c00d`, on `feature/subnational-labor-uncertainty-us`):**
- 0.46.0 shipped, `tools/release_check.py` is the pre-tag gate.
- Current versions: `pyproject.toml`, `puremacro/__init__.py::__version__`, `CHANGELOG.md` first heading all `0.46.0`.
- `tests/known_failures.json` entries: `[]` (empty).
- Full-suite test count: 1304 passed, 21 skipped, 31 deselected.
- Verified API surfaces (2026-05-22):
  - `puremacro.var.identify._results.ProxySVARResult.irf_point: shape (n, n, H+1)` — currently the lone outlier; other six `*Result` dataclasses use `(H+1, n, n)`.
  - `puremacro.var.identify._results.ProxySVARResult.__post_init__` reads `H = self.irf_point.shape[2] - 1`.
  - `puremacro.var.identify.proxy.proxy_svar` builds `point, lo, hi = wild_bootstrap_var(...)` (shape `(n, n, H+1)`) and passes them straight into `ProxySVARResult(...)`.
  - `wild_bootstrap_var` lives at `puremacro/inference/wild_bootstrap.py:59`. Used by other identifications too — do NOT change its return shape; transpose in `proxy.py` only.
  - `puremacro.lp.garch_utils` exports `fit_garch`, `make_regime_indicator`, `align_series_for_lp`.
  - Active callers of `garch_utils` (3 sites, repo-wide grep verified): `tests/test_garch_utils.py:10`, `notebooks/R1_methods/R1_02_lp_menu.ipynb` (one cell), `tools/make_notebook_R1_02.py:114`.
  - `puremacro.dsge.klein.klein_solve(A, B, n_pre, C=None, *, strict=False) -> KleinSolution(G, F, N, L, eu, eigenvalues)`. Z-partition logic at `puremacro/dsge/klein.py:165-200`. The `G = Z11 @ inv(S11) @ T11 @ inv(Z11)` formula is verified at machine precision for SW07; the corruption is in `F`, not `G`.
  - `puremacro.dsge.smets_wouters._solve_F_sylvester(G0, G1, G_x)` — 90 LOC private function at `puremacro/dsge/smets_wouters.py:683-770`, recovers `F` from the equilibrium Sylvester equation when Klein's closed-form F is corrupted by unit-eigenvalue lag states. Residual ~9e-15 per the existing docstring.
  - SW07 has 10 regression tests in `tests/test_dsge_smets_wouters.py` (consumption-Euler coefficients, BK condition, qualitative IRFs, growth-rate IRFs, unit-sd impact values). These are the byte-equal regression contract for the C track.

---

## File structure

**Modified (rename + content):**
- `puremacro/lp/garch_utils.py` → `puremacro/lp/_garch_utils.py` (rename, no content change — Task 1).
- `puremacro/var/identify/_results.py` — `ProxySVARResult` docstring shapes + `__post_init__` axis-extraction (Task 2).
- `puremacro/var/identify/proxy.py` — transpose `point`, `lo`, `hi` from `(n, n, H+1)` to `(H+1, n, n)` before constructing `ProxySVARResult` (Task 2).
- `puremacro/examples/narrative_ramey_2011.py` — 3 caller lines (Task 3).
- `puremacro/examples/hfi_gertler_karadi.py` — 1 caller line (Task 3).
- `puremacro/examples/svariv_mertens_ravn.py` — 2 caller lines (Task 3).
- `puremacro/dsge/klein.py` — add condition-number check + `_solve_F_sylvester_fallback` (Task 5).
- `puremacro/dsge/smets_wouters.py` — delete local `_solve_F_sylvester`; call `klein_solve` directly (Task 6).
- `tests/test_garch_utils.py` — import-path update (Task 1).
- `tools/make_notebook_R1_02.py` — import-path update (Task 1).
- `tools/make_notebook_R1_04.py` — proxy_svar caller updates if any, paired with R1_04 notebook edits (Task 4).
- `notebooks/R1_methods/R1_02_lp_menu.ipynb` — single import cell update via `NotebookEdit` (Task 1).
- `notebooks/R1_methods/R1_04_dsge_compare.ipynb` — 10 mentions of proxy_svar/irf_point/ProxySVAR; cell edits + re-execution via paired builder (Task 4).

**Created:**
- New test `tests/test_var/test_proxy_axis.py` — asserts `ProxySVARResult.irf_point.shape == (H+1, n, n)` (Task 2).
- New test `tests/test_dsge/test_klein_unit_eigenvalue_fallback.py` — synthetic system with multiple unit-eigenvalue lag states verifies the Sylvester branch triggers + equilibrium residual ≤ 1e-13 (Task 5).

**Untouched:**
- `puremacro/regress/lp.py` — A1 dropped from scope.
- `puremacro/lp/garch_utils.py` content (only the filename changes; symbols stay public-by-existence).
- Other VAR identification result classes (`CholeskySVARResult`, `BQSVARResult`, `SignSVARResult`, etc.) — already on the `(H+1, n, n)` convention.
- `puremacro/inference/wild_bootstrap.py::wild_bootstrap_var` — shared by other identifications; do not change its return shape, transpose locally in `proxy.py`.

---

## Working-directory convention

All paths below are relative to the **repo root**:

`/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/`

Within that root: `puremacro/` is the package, `tests/`, `tools/`, `notebooks/` are the outer-repo siblings. Some tasks (Task 1's test, the gate run) operate against `puremacro/tests/test_pyodide_compat.py`; others (Task 1's notebook cell, Task 4's R1_04) operate against the outer `tests/test_garch_utils.py` and `notebooks/R1_methods/`. **Pay attention to which `tests/` directory each step references.**

---

## Task 0: Pre-flight + branch creation

- [ ] **Step 1: Verify clean state (puremacro subtree only)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git status --short puremacro/ | head -10
```

Expected: empty (no modifications). The outer repo has many pre-existing `M` and `??` entries under `docs/paper/` and `notebooks/output_*/` — those are fine and pre-existing.

- [ ] **Step 2: Confirm HEAD on feature/subnational-labor-uncertainty-us**

```bash
git log --oneline -1
git branch --show-current
```

Expected: HEAD on `feature/subnational-labor-uncertainty-us`, top commit `f66c00d fix(0.46.0): CHANGELOG cosmetic — drop incidental test count`.

- [ ] **Step 3: Confirm release-gate is green at this baseline**

```bash
python puremacro/tools/release_check.py
```

Expected: exit 0, all 4 gates PASS. If not, **stop and fix** — 0.47.0 must start from a clean baseline.

- [ ] **Step 4: Create the release branch**

```bash
git checkout -b release/0.47.0
git branch --show-current
```

Expected: `release/0.47.0`.

- [ ] **Step 5: No commit yet** — Task 0 is verification only.

---

## Task 1: A2 — Rename `lp/garch_utils.py` → `lp/_garch_utils.py`

Mechanical rename. No content change. No notebook re-execution. Three import sites update in lockstep with the `git mv`.

- [ ] **Step 1: `git mv` the file**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git mv puremacro/puremacro/lp/garch_utils.py puremacro/puremacro/lp/_garch_utils.py
```

Verify:

```bash
ls puremacro/puremacro/lp/garch_utils.py 2>&1   # should NOT exist
ls puremacro/puremacro/lp/_garch_utils.py        # should exist
```

- [ ] **Step 2: Update `tests/test_garch_utils.py:10`**

Replace:

```python
from puremacro.lp.garch_utils import fit_garch, make_regime_indicator, align_series_for_lp
```

With:

```python
from puremacro.lp._garch_utils import fit_garch, make_regime_indicator, align_series_for_lp
```

Use `Edit` on `tests/test_garch_utils.py`. Do NOT rename the test file — the test name remains `test_garch_utils.py` (it tests the helpers; the underscore in the module is an internal-organization signal, not a test-name change).

- [ ] **Step 3: Update `tools/make_notebook_R1_02.py:114`**

Find the line:

```python
        "from puremacro.lp.garch_utils import fit_garch, make_regime_indicator        # §6/§7\n"
```

Replace with:

```python
        "from puremacro.lp._garch_utils import fit_garch, make_regime_indicator        # §6/§7\n"
```

- [ ] **Step 4: Update `notebooks/R1_methods/R1_02_lp_menu.ipynb` — single import cell**

There is exactly one cell whose source contains `from puremacro.lp.garch_utils import`. Use the `NotebookEdit` tool (deferred — load with `ToolSearch query="select:NotebookEdit"`). Patch that cell's source line to read `from puremacro.lp._garch_utils import …`. **Do NOT re-execute the notebook.** Per memory pin `feedback_builder_clobbers_outputs`: "Renames + path edits: do NOT run the builder; `git mv` is enough." The executed outputs in the `.ipynb` are preserved as-is; only the source-cell import line changes.

- [ ] **Step 5: Run the test that uses the renamed module**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest ../tests/test_garch_utils.py -v
```

Expected: all tests in `test_garch_utils.py` pass (the test exercises `fit_garch`, `make_regime_indicator`, `align_series_for_lp`; behavior is unchanged).

- [ ] **Step 6: Run the gate**

```bash
python tools/release_check.py --no-tests
```

Expected: exit 0. (Gate 3 will diff the public-API snapshot — `garch_utils` was not in `__all__` of `lp/__init__.py`, so the snapshot is unaffected by the rename. If the snapshot diff fires unexpectedly, stop and investigate.)

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/lp/_garch_utils.py tests/test_garch_utils.py tools/make_notebook_R1_02.py notebooks/R1_methods/R1_02_lp_menu.ipynb
# Note: the rename shows as a delete+add in `git status` until staged; `git mv` records it as a rename in `git diff --stat`.
git commit -m "$(cat <<'EOF'
refactor(0.47.0): rename lp/garch_utils.py → lp/_garch_utils.py

Signals "internal helper" via leading underscore. No content change.
3 import sites updated: tests/test_garch_utils.py, tools/make_notebook_R1_02.py,
notebooks/R1_methods/R1_02_lp_menu.ipynb (source cell, no re-execution per
feedback_builder_clobbers_outputs).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: A3 — `ProxySVARResult` axis flip in `_results.py` + `proxy.py`

These two files MUST land in one atomic commit. Changing `_results.py::__post_init__`'s `shape[2]` → `shape[0]` without simultaneously transposing in `proxy.py` would crash every `ProxySVARResult` construction.

- [ ] **Step 1: Write the failing shape-lock test**

Create `tests/test_var/test_proxy_axis.py`:

```python
"""Axis-convention regression test for ProxySVARResult.

Locks the contract: irf_point / irf_lower / irf_upper are shape (H+1, n, n),
matching the other six *Result dataclasses in var/identify/_results.py.
"""
import numpy as np
import pytest

from puremacro.var.identify.proxy import proxy_svar


@pytest.fixture
def small_svar_inputs():
    rng = np.random.default_rng(2026)
    n = 3
    T = 200
    # Generate a small VAR(1) with a clean shock.
    A = 0.5 * np.eye(n)
    eps = rng.standard_normal((T, n))
    Y = np.zeros((T, n))
    Y[0] = eps[0]
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + eps[t]
    # Proxy correlates with eps[:, 0].
    z = eps[:, 0] + 0.2 * rng.standard_normal(T)
    return Y, z


def test_proxy_svar_irf_point_shape(small_svar_inputs):
    Y, z = small_svar_inputs
    H = 8
    res = proxy_svar(Y, p=1, horizon=H, instrument_series=z, n_boot=20, ci=0.9, seed=0)
    assert res.irf_point.shape == (H + 1, Y.shape[1], Y.shape[1])
    assert res.irf_lower.shape == (H + 1, Y.shape[1], Y.shape[1])
    assert res.irf_upper.shape == (H + 1, Y.shape[1], Y.shape[1])
```

If the `tests/test_var/` directory does not yet exist, create it with an empty `__init__.py`:

```bash
mkdir -p puremacro/tests/test_var
touch puremacro/tests/test_var/__init__.py 2>/dev/null || true
```

(Check first: `ls puremacro/tests/test_var/` — if it exists already, skip the mkdir.)

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest puremacro/tests/test_var/test_proxy_axis.py -v
```

Expected: 1 FAILED with shape mismatch — current shape is `(n, n, H+1)` = `(3, 3, 9)`, not `(H+1, n, n)` = `(9, 3, 3)`.

- [ ] **Step 3: Update `puremacro/var/identify/_results.py::ProxySVARResult`**

Change the three shape docstrings:

```python
    irf_point : ndarray, shape (n, n, H+1)
    irf_lower : ndarray, shape (n, n, H+1)
    irf_upper : ndarray, shape (n, n, H+1)
```

To:

```python
    irf_point : ndarray, shape (H+1, n, n)
    irf_lower : ndarray, shape (H+1, n, n)
    irf_upper : ndarray, shape (H+1, n, n)
```

Update `__post_init__` shape extraction. Find:

```python
        n = self.B.shape[0]
        H = self.irf_point.shape[2] - 1
```

Replace with:

```python
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
```

(Note: `self.B.shape[0]` is still `n`, but using `irf_point.shape[1]` keeps the shape-extraction logic uniform with the other six result classes. Verify by reading `CholeskySVARResult.__post_init__` for the pattern.)

- [ ] **Step 4: Update `puremacro/var/identify/proxy.py` — transpose at the return**

In `proxy_svar`, after the `wild_bootstrap_var` call returns `point, lo, hi`, transpose each from `(n, n, H+1)` to `(H+1, n, n)`. The existing call:

```python
    point, lo, hi = wild_bootstrap_var(
        Y, p=p, horizon=horizon, impact_fn=impact_fn,
        n_boot=n_boot, ci=ci, seed=seed,
    )
    return ProxySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        ...
    )
```

Becomes:

```python
    point, lo, hi = wild_bootstrap_var(
        Y, p=p, horizon=horizon, impact_fn=impact_fn,
        n_boot=n_boot, ci=ci, seed=seed,
    )
    # wild_bootstrap_var returns (n, n, H+1); canonical *Result convention is (H+1, n, n).
    # See puremacro/var/identify/_results.py for the project-wide axis convention.
    point = np.transpose(point, (2, 0, 1))
    lo = np.transpose(lo, (2, 0, 1))
    hi = np.transpose(hi, (2, 0, 1))
    return ProxySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        ...
    )
```

`np` must be imported in `proxy.py` — it should already be (the file uses `np.asarray` and `np.zeros`). Verify with `grep "^import numpy" puremacro/puremacro/var/identify/proxy.py` first.

- [ ] **Step 5: Verify the shape test now passes**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest puremacro/tests/test_var/test_proxy_axis.py -v
```

Expected: 1 PASSED.

- [ ] **Step 6: Confirm no other test regressed**

```bash
python -m pytest puremacro/tests/test_var/ puremacro/tests/test_cholesky_shocks.py -v --tb=short -q 2>&1 | tail -10
```

Expected: all green. If anything regresses, root-cause it — likely an existing test indexed `irf_point[i, j, h]` and now needs `irf_point[h, i, j]`.

- [ ] **Step 7: Commit (Task 2 atomic — `_results.py` + `proxy.py` + new test)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/var/identify/_results.py puremacro/puremacro/var/identify/proxy.py puremacro/puremacro/tests/test_var/__init__.py puremacro/puremacro/tests/test_var/test_proxy_axis.py
git commit -m "$(cat <<'EOF'
breaking(0.47.0): ProxySVARResult axis flip — (n,n,H+1) → (H+1,n,n)

Aligns with the other six *Result dataclasses in var/identify/_results.py.
proxy.py transposes wild_bootstrap_var's output before construction.
_results.py docstrings + __post_init__ updated. New shape-lock test under
tests/test_var/test_proxy_axis.py.

Caller updates (5 example/notebook files) land in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: A3 — Update 3 example-script callers (`examples/narrative_ramey_2011`, `examples/hfi_gertler_karadi`, `examples/svariv_mertens_ravn`)

Three files, six lines total, all index `irf_point` with the old axis order.

- [ ] **Step 1: Update `puremacro/examples/narrative_ramey_2011.py:126`**

```python
    # proxy_svar returns shape (n, n, H+1): [response_var, shock_var, h]
```

Becomes:

```python
    # proxy_svar returns shape (H+1, n, n): [h, response_var, shock_var]
```

- [ ] **Step 2: Update `puremacro/examples/narrative_ramey_2011.py:133`**

```python
        print(f"   h={h:>2d}q : {irf_point[1, 0, h]:+.4f}")
```

Becomes:

```python
        print(f"   h={h:>2d}q : {irf_point[h, 1, 0]:+.4f}")
```

- [ ] **Step 3: Update `puremacro/examples/narrative_ramey_2011.py:143`**

```python
            ax.plot(h_arr, irf_point[i, 0, :], color="0.0", lw=1.0)
```

Becomes:

```python
            ax.plot(h_arr, irf_point[:, i, 0], color="0.0", lw=1.0)
```

- [ ] **Step 4: Update `puremacro/examples/hfi_gertler_karadi.py:59`**

```python
        ax.plot(horizons, res.irf_point[j, 0, :], "b-", label="point")
```

Becomes:

```python
        ax.plot(horizons, res.irf_point[:, j, 0], "b-", label="point")
```

- [ ] **Step 5: Update `puremacro/examples/svariv_mertens_ravn.py:113`**

```python
    - ``irf_point`` : ndarray ``(n, n, H+1)`` from ``proxy_svar``.
```

Becomes:

```python
    - ``irf_point`` : ndarray ``(H+1, n, n)`` from ``proxy_svar``.
```

- [ ] **Step 6: Update `puremacro/examples/svariv_mertens_ravn.py:154`**

```python
    recovered_path = res.irf_point[1, 0, :]
```

Becomes:

```python
    recovered_path = res.irf_point[:, 1, 0]
```

- [ ] **Step 7: Smoke-run one example to verify it doesn't blow up on import / docstring inconsistency**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -c "import puremacro.examples.narrative_ramey_2011; import puremacro.examples.hfi_gertler_karadi; import puremacro.examples.svariv_mertens_ravn; print('imports OK')"
```

Expected: `imports OK`. (We are NOT running the full examples — they fetch data and take minutes. Just verifying the modules import.)

- [ ] **Step 8: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/examples/narrative_ramey_2011.py puremacro/puremacro/examples/hfi_gertler_karadi.py puremacro/puremacro/examples/svariv_mertens_ravn.py
git commit -m "$(cat <<'EOF'
breaking(0.47.0): update 3 example callers for ProxySVARResult axis flip

narrative_ramey_2011 (3 sites), hfi_gertler_karadi (1), svariv_mertens_ravn
(2) — all index irf_point with the new (H+1, n, n) axis order.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: A3 — Update R1_04 notebook + paired builder + re-execute

R1_04_dsge_compare.ipynb has 10 mentions of proxy_svar/irf_point/ProxySVAR. The paired builder is `tools/make_notebook_R1_04.py`. Per memory pins `feedback_notebook_builders_paired` and `feedback_builder_clobbers_outputs`: source-cell changes that alter behavior require re-execution; both the `.ipynb` and the builder must be patched in the same commit.

- [ ] **Step 1: Audit which lines in the notebook use the old axis order**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
grep -n "irf_point\|proxy_svar\|ProxySVAR" notebooks/R1_methods/R1_04_dsge_compare.ipynb | head -20
```

Expected: 10 hits (verified pre-plan-write). Some are docstring/markdown mentions (no code change); some are slicing patterns like `[i, 0, :]` or `[:, j, k]` that need flipping.

For each hit:
- If it's a markdown / comment mentioning shape, update the shape annotation.
- If it's `irf_point[i, j, h]` (positional indexing), flip to `irf_point[h, i, j]`.
- If it's `irf_point[i, j, :]` (all horizons), flip to `irf_point[:, i, j]`.
- If it's `irf_point[:, i, j]` (sweep over horizons under the OLD convention — meaning "for each response var, take fixed shock j at fixed h=j"), the meaning is genuinely different under the new convention. **Pause and ask before changing.** This is a semantic ambiguity, not a mechanical rewrite.

Build a per-line rewrite map BEFORE editing.

- [ ] **Step 2: Apply the rewrite map to both `notebooks/R1_methods/R1_04_dsge_compare.ipynb` (via `NotebookEdit`) and `tools/make_notebook_R1_04.py` (via `Edit`)**

Both files must be updated in lockstep. The builder generates the notebook; if you patch only the `.ipynb`, the next builder run silently overwrites your edits (per `feedback_notebook_builders_paired`).

For each cell containing changed `irf_point` indexing:
- `NotebookEdit` on the live `.ipynb` to update the cell source.
- `Edit` on `tools/make_notebook_R1_04.py` to update the corresponding source string that builder emits.

- [ ] **Step 3: Re-execute R1_04 via the paired builder + nbconvert**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
# Per feedback_long_nbconvert_no_subagent: run in the controller's background, not a subagent.
jupyter nbconvert --to notebook --execute --inplace notebooks/R1_methods/R1_04_dsge_compare.ipynb 2>&1 | tail -20
```

Expected: nbconvert completes without errors. Time: depends on R1_04's content; if it does bootstraps, could be 5-20 minutes. If >5 min, run in background and wait for notification.

If a cell errors out during execution, the rewrite map missed a site. Stop, root-cause, fix, re-execute.

- [ ] **Step 4: Diff the executed notebook to confirm only intended cells changed**

```bash
git diff --stat notebooks/R1_methods/R1_04_dsge_compare.ipynb
```

Expected: source cell changes match the rewrite map; output cells changed only for cells that re-ran (numerical values may differ slightly within tolerance — pin seeds if the builder supports it).

- [ ] **Step 5: Commit (paired)**

```bash
git add notebooks/R1_methods/R1_04_dsge_compare.ipynb tools/make_notebook_R1_04.py
git commit -m "$(cat <<'EOF'
breaking(0.47.0): R1_04 + paired builder migrated to ProxySVARResult (H+1,n,n)

10 source-cell sites flipped from irf_point[i,j,h]-style to irf_point[h,i,j]-style.
Notebook re-executed via jupyter nbconvert --execute. Builder updated in
the same commit per feedback_notebook_builders_paired.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Optional — check `notebooks/T_fiscal_channels.ipynb:666`**

The repo-wide grep found one more potential caller: `notebooks/T_fiscal_channels.ipynb:666` uses `r.irf_point[:, var_idx, shock_idx]`. **Inspect whether `r` is a `ProxySVARResult` or a different result class.** If yes (proxy), the slicing pattern `[:, var_idx, shock_idx]` under the OLD convention meant "all responses, fixed shock=var_idx, fixed h=shock_idx" — which is semantically odd; the variable names suggest the slice index meanings drifted. Under the NEW convention `[:, var_idx, shock_idx]` means "all horizons, response=var_idx, shock=shock_idx" — which IS semantically clean. So this slice may **accidentally become correct** under the new convention, OR it may break a previously-working sweep. **Read the surrounding cell to disambiguate.**

If T_fiscal_channels.ipynb is in `notebooks/_archive/` already, defer per the pattern in 0.44.0 (don't touch archived notebooks). If it's live, decide based on the surrounding context whether to edit + re-execute, or leave with a comment explaining the deferral.

- [ ] **Step 7: Confirm gate is still green**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py
```

Expected: all 4 gates PASS, exit 0.

---

## Task 5: C — Port `_solve_F_sylvester` into `klein.py` as a Z-partition fallback

The most algorithmically substantive task. SW07's local workaround proves the math; this task lifts it into the canonical solver so SW07 can drop the local copy.

- [ ] **Step 1: Re-read `puremacro/dsge/smets_wouters.py::_solve_F_sylvester`**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
sed -n '683,770p' puremacro/dsge/smets_wouters.py
```

Read it end-to-end. It's ~90 LOC; understand the Sylvester equation it solves before porting. Key callers: only `solve_sw07` at line 769 calls it.

- [ ] **Step 2: Write the failing fallback-trigger test**

Create `tests/test_dsge/test_klein_unit_eigenvalue_fallback.py`:

```python
"""Klein-solver fallback test: when the Z-partition is degenerate due to
multiple unit-eigenvalue lag states, klein_solve should detect the condition
and return a corrected F via the Sylvester equilibrium equation.
"""
import numpy as np
import pytest

from puremacro.dsge.klein import klein_solve


def test_klein_unit_eigenvalue_triggers_sylvester_fallback():
    """Construct a small system with two unit-eigenvalue lag states; verify
    the fallback branch fires and the policy function F satisfies the
    equilibrium condition to ~1e-13.

    System:
        x_t = G x_{t-1}  with two unit eigenvalues + one stable eigenvalue
        y_t = F x_t      forward-looking control

    A z_{t+1} = B z_t  with z = [x, y]; expectation error eta on y.
    """
    n_pre = 3
    n_fwd = 1
    n = n_pre + n_fwd

    # G_x has two unit eigenvalues + one stable.
    G_x = np.array([
        [1.0, 0.0, 0.0],   # unit
        [0.0, 1.0, 0.0],   # unit
        [0.0, 0.0, 0.5],   # stable
    ])
    # F: control loads on the third state with coefficient 1.5.
    F_true = np.array([[0.0, 0.0, 1.5]])

    # Construct A, B for: y_{t+1} = F x_{t+1};  x_{t+1} = G_x x_t.
    A = np.eye(n)
    B = np.zeros((n, n))
    B[:n_pre, :n_pre] = G_x
    B[n_pre:, :n_pre] = F_true @ G_x

    sol = klein_solve(A, B, n_pre=n_pre, strict=False)

    # eu must indicate a valid solution.
    assert sol.eu == (1, 1), f"BK condition failed: eu={sol.eu}"

    # G recovers G_x.
    np.testing.assert_allclose(sol.G, G_x, atol=1e-12)

    # F recovers F_true via the fallback. The closed-form Klein F is
    # corrupted by the unit eigenvalues; the Sylvester fallback recovers
    # the true F.
    np.testing.assert_allclose(sol.F, F_true, atol=1e-10)


def test_klein_stable_system_uses_closed_form_path():
    """When the Z-partition is well-conditioned, klein_solve uses the
    closed-form F. Verify by constructing a well-conditioned system.
    """
    n_pre = 2
    n_fwd = 1
    G_x = np.array([
        [0.7, 0.1],
        [0.0, 0.5],
    ])
    F_true = np.array([[0.3, -0.2]])
    A = np.eye(n_pre + n_fwd)
    B = np.zeros((n_pre + n_fwd, n_pre + n_fwd))
    B[:n_pre, :n_pre] = G_x
    B[n_pre:, :n_pre] = F_true @ G_x

    sol = klein_solve(A, B, n_pre=n_pre, strict=False)
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, G_x, atol=1e-12)
    np.testing.assert_allclose(sol.F, F_true, atol=1e-10)
```

If `tests/test_dsge/` does not yet exist, create it with an `__init__.py`:

```bash
mkdir -p puremacro/tests/test_dsge
touch puremacro/tests/test_dsge/__init__.py 2>/dev/null || true
```

- [ ] **Step 3: Run the test to verify failure modes**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest puremacro/tests/test_dsge/test_klein_unit_eigenvalue_fallback.py -v
```

Expected: the unit-eigenvalue test FAILS (F mismatch — the closed-form formula is corrupted). The well-conditioned test PASSES (the closed-form is correct in that regime).

- [ ] **Step 4: Port `_solve_F_sylvester` into `klein.py` as a private helper**

In `puremacro/dsge/klein.py`, add (above `klein_solve`):

```python
def _solve_F_sylvester(A: np.ndarray, B: np.ndarray, G: np.ndarray,
                       n_pre: int) -> np.ndarray:
    """Recover the forward-looking policy function F from the equilibrium
    Sylvester equation, bypassing the closed-form Klein formula.

    Used as a fallback when the Z-partition is degenerate (multiple
    unit-eigenvalue lag states corrupting Z11). Solves the linear system
    derived from substituting x_{t+1} = G x_t and y_{t+1} = F x_{t+1} into
    A z_{t+1} = B z_t.

    Parameters
    ----------
    A, B : (n, n) ndarrays — original system coefficients.
    G    : (n_pre, n_pre) — recovered state transition.
    n_pre : int — number of predetermined variables.

    Returns
    -------
    F : (n_fwd, n_pre) ndarray
    """
    # [PORT THE BODY OF smets_wouters._solve_F_sylvester HERE]
    # The existing implementation at puremacro/dsge/smets_wouters.py:683-770
    # is the reference. Replace its smets_wouters-specific inputs (G0, G1, G_x)
    # with the generic (A, B, G) names. Preserve the vectorized Sylvester
    # solve and any numerical safeguards (rank checks, conditioning floors).
    # Document any constants that come from smets_wouters and re-derive
    # them generically here.
    raise NotImplementedError("Port _solve_F_sylvester body from smets_wouters.py")
```

Then modify `klein_solve` to detect the degenerate case and route to the fallback. After the existing computation of `F` (closed-form, around line 200), add:

```python
        # Z-partition degeneracy check: if Z11 is severely ill-conditioned
        # (typically due to multiple unit-eigenvalue lag states), the
        # closed-form F via Klein's formula is corrupted. Route through
        # the Sylvester equilibrium solve as a fallback.
        cond_Z11 = np.linalg.cond(Z11) if Z11.size > 0 else 1.0
        if cond_Z11 > 1e10:
            F = _solve_F_sylvester(A, B, G, n_pre=n_pre)
```

(The condition-number threshold `1e10` is heuristic; calibrate against the SW07 case in Task 6. If `1e10` is too tight or too loose, adjust to the smallest value that triggers for SW07 and not for well-conditioned systems.)

- [ ] **Step 5: Implement the body of `_solve_F_sylvester`**

Open `puremacro/dsge/smets_wouters.py:683-770` and port the function body. Translate:
- `G0` → `A`
- `G1` → `B`
- `G_x` → `G`
- Hard-coded `_N_PRE` (SW07's 20) → parameter `n_pre`
- Hard-coded `_N_FWD` (SW07's 24) → derived as `A.shape[0] - n_pre`

The Sylvester solve uses `scipy.linalg.solve_sylvester` or an explicit vec-Kronecker form — preserve whichever the SW07 version uses.

Replace the `raise NotImplementedError` with the ported body.

- [ ] **Step 6: Run the new tests to verify both pass**

```bash
python -m pytest puremacro/tests/test_dsge/test_klein_unit_eigenvalue_fallback.py -v
```

Expected: 2 PASSED.

- [ ] **Step 7: Run the existing SW07 tests — they MUST stay green**

```bash
python -m pytest puremacro/tests/test_dsge_smets_wouters.py -v 2>&1 | tail -15
```

Expected: all 10 PASSED (consumption-Euler coefs, BK condition, qualitative IRFs, growth-rate IRFs, unit-sd impacts). **Smets_wouters still calls its local `_solve_F_sylvester` at this point** — both implementations should yield byte-equal results.

If any SW07 test fails, the Sylvester port has a bug. Stop and root-cause before proceeding to Task 6.

- [ ] **Step 8: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/dsge/klein.py puremacro/puremacro/tests/test_dsge/__init__.py puremacro/puremacro/tests/test_dsge/test_klein_unit_eigenvalue_fallback.py
git commit -m "$(cat <<'EOF'
feat(0.47.0): klein_solve Sylvester fallback for degenerate Z-partition

When Z11 is severely ill-conditioned (typically due to multiple
unit-eigenvalue lag states), the closed-form Klein F is corrupted by
the QZ stable-block mixing. New _solve_F_sylvester helper recovers F
from the equilibrium Sylvester equation directly. SW07 tests stay
green; smets_wouters' local _solve_F_sylvester is retired in the
next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: C — `smets_wouters.py` retracts its local workaround

With klein.py's fallback in place, SW07 can call `klein_solve` directly and drop its local `_solve_F_sylvester`.

- [ ] **Step 1: Inspect the SW07 call site**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
sed -n '760,790p' puremacro/dsge/smets_wouters.py
```

Verify `solve_sw07` currently calls `_solve_F_sylvester(G0, G1, G_x)` after `klein_solve(G0, G1, n_pre=_N_PRE, C=Psi, strict=False)`.

- [ ] **Step 2: Replace the workaround with a `sol.F` read**

In `solve_sw07`, remove the line:

```python
    F    = _solve_F_sylvester(G0, G1, G_x)     # (24, 20) corrected policy function
```

Replace with:

```python
    F    = sol_klein.F                          # (24, 20) policy fn — Klein fallback handles unit eigenvalues
```

- [ ] **Step 3: Delete the local `_solve_F_sylvester` function**

In `puremacro/dsge/smets_wouters.py`, delete the `_solve_F_sylvester` function body (lines 683-770, or wherever it lives — locate via `grep -n "^def _solve_F_sylvester" puremacro/dsge/smets_wouters.py`).

- [ ] **Step 4: Run SW07 tests**

```bash
python -m pytest puremacro/tests/test_dsge_smets_wouters.py -v 2>&1 | tail -15
```

Expected: 10 PASSED — byte-equal to pre-Task-5 state. The Sylvester fallback in klein.py is the same arithmetic as the SW07 local copy, so the IRF / BK / Euler / impact tests should be unchanged within numerical tolerance.

If any test fails, the threshold `1e10` may be wrong, OR the port introduced a bug. Diagnose by:
1. Print `cond_Z11` for the SW07 system to verify the fallback is being triggered.
2. If not triggered, lower the threshold to match.
3. If triggered but results differ, compare `_solve_F_sylvester`'s output between klein.py's port and the deleted smets_wouters copy on the same inputs.

- [ ] **Step 5: Run the gate**

```bash
python tools/release_check.py
```

Expected: all 4 gates PASS, exit 0.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/dsge/smets_wouters.py
git commit -m "$(cat <<'EOF'
refactor(0.47.0): smets_wouters drops local _solve_F_sylvester workaround

klein.solve()'s Sylvester fallback (added in the previous commit)
replaces SW07's local workaround. solve_sw07 reads sol_klein.F
directly. The 10 SW07 regression tests pass byte-equal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Version bump + CHANGELOG + final gate verification

- [ ] **Step 1: Bump `pyproject.toml`**

Edit `puremacro/pyproject.toml` line `version = "0.46.0"` → `version = "0.47.0"`.

- [ ] **Step 2: Bump `puremacro/__init__.py`**

Edit `__version__ = "0.46.0"` → `__version__ = "0.47.0"`.

- [ ] **Step 3: Bump `tests/test_import.py`**

Edit `assert puremacro.__version__ == "0.46.0"` → `assert puremacro.__version__ == "0.47.0"`.

- [ ] **Step 4: Prepend CHANGELOG 0.47.0 entry**

Edit `puremacro/CHANGELOG.md`, inserting after the `# Changelog` preamble (before the `## 0.46.0` heading):

```markdown
## 0.47.0 — 2026-05-22

Mechanical cleanup + DSGE solver hardening.

Three deferred items from the 0.43-0.45 cycle that survived API
verification: `lp/garch_utils.py` renamed to `_garch_utils.py` (signal
"internal helper"); `ProxySVARResult.irf_point` axis flipped from
`(n, n, H+1)` to `(H+1, n, n)` to match the other six SVAR result
dataclasses; `klein.solve()` now detects degenerate Z-partitions and
routes F-recovery through a Sylvester equilibrium fallback, retiring
SW07's local workaround.

Not in scope: `regress/lp.py` retirement (A1 from the original spec) —
API verification confirmed the file is legitimately distinct
(incompatible signature, return columns, and SE method) and not a
deferred shim. The 0.43.0 deferral note was conditional on shipping a
canonical equivalent with the same signature, which has not happened.

### Breaking
- `puremacro.var.identify.ProxySVARResult.irf_point` / `.irf_lower` /
  `.irf_upper` are now shape `(H+1, n, n)`, not `(n, n, H+1)`. Callers
  must flip indexing from `[i, j, h]` to `[h, i, j]` (or `[:, i, j]` for
  horizon sweeps).

### Changed
- `puremacro.lp.garch_utils` → `puremacro.lp._garch_utils` (private name).
  Three import sites updated in lockstep: `tests/test_garch_utils.py`,
  `tools/make_notebook_R1_02.py`, `notebooks/R1_methods/R1_02_lp_menu.ipynb`.
- `puremacro.dsge.klein.klein_solve` now detects Z11 condition number
  > 1e10 and routes F-recovery through the new private
  `_solve_F_sylvester` helper. Well-conditioned systems are unchanged.

### Removed
- `puremacro.dsge.smets_wouters._solve_F_sylvester` — the canonical
  Sylvester fallback now lives in `klein.py`.

### Added
- `tests/test_var/test_proxy_axis.py` — locks the `(H+1, n, n)` shape contract.
- `tests/test_dsge/test_klein_unit_eigenvalue_fallback.py` — locks the
  fallback trigger and the well-conditioned-path bypass.

### Internal
- Five example/notebook callers updated for the ProxySVAR axis flip:
  `narrative_ramey_2011.py` (3 sites), `hfi_gertler_karadi.py` (1),
  `svariv_mertens_ravn.py` (2), `R1_04_dsge_compare.ipynb` (10 sites,
  re-executed via paired builder).
- SW07's 10 regression tests pass byte-equal through the klein.py
  refactor (`test_dsge_smets_wouters.py`).

---
```

- [ ] **Step 5: Run the gate end-to-end on staged state**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py
```

Expected: all 4 gates PASS, exit 0. Gate 3 (public API snapshot) will detect the `ProxySVARResult` docstring shape change indirectly only if the snapshot tracks docstrings — most likely it doesn't, but verify. The `garch_utils` rename does NOT show up in Gate 3 because the symbol isn't in `lp/__init__.py::__all__`.

If Gate 3 flags new symbols (e.g., the new `_solve_F_sylvester` private in `klein.py`), regenerate the snapshot deliberately and commit it in the same release commit:

```bash
python -c "
import sys; sys.path.insert(0, 'tests')
from test_public_api import collect_current_api
import json
with open('tests/fixtures/public_api_snapshot.json', 'w') as f:
    json.dump(collect_current_api(), f, indent=2, sort_keys=True)
"
git add puremacro/tests/fixtures/public_api_snapshot.json
```

- [ ] **Step 6: Commit the bump**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/tests/test_import.py puremacro/CHANGELOG.md
# Also stage the snapshot fixture if Step 5 regenerated it.
git commit -m "$(cat <<'EOF'
chore(puremacro): bump 0.46.0 → 0.47.0 (garch_utils rename + ProxySVAR axis + Klein hardening)

Three deferred-from-0.43 items closed out. Breaking change: ProxySVARResult
axis flip aligns the lone outlier with the other six SVAR result classes.
A1 (regress/lp.py retirement) explicitly dropped — file is legitimately
distinct per API verification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Final gate run on the bump commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py
```

Expected: all 4 gates PASS at 0.47.0, exit 0.

- [ ] **Step 8: Hand off to the controller for the merge decision**

Do NOT auto-merge. Report back the final HEAD SHA and the gate output; the controller asks the user about tag + merge.

---

## Self-review notes

**Spec coverage** (from `docs/specs/2026-05-22-puremacro-046-047-consolidate-finish-design.md` § 0.47.0):
- A1 (regress/lp.py retirement) — DROPPED per spec's pause-and-re-spec clause; API verification confirmed legitimately distinct ✓
- A2 (garch_utils rename) — Task 1 ✓
- A3 (ProxySVAR axis flip) — Tasks 2-4 ✓
- C (Klein many-unit-eigenvalue) — Tasks 5-6 ✓
- Version bump + CHANGELOG — Task 7 ✓
- Gate-gated throughout — Tasks 0/1/4/5/6/7 all run the gate ✓

**Placeholder scan:** Task 5 Step 4 has a single `[PORT THE BODY OF ...]` placeholder by design — the body must be transcribed from existing in-tree code, not invented. The instructions name the exact source file + line range, the renaming map (G0→A etc.), and the structural invariants. This is documenting where to copy from, not a TBD. Task 4 Step 6 has a "pause and ask" branch for the T_fiscal_channels disambiguation — explicit deferral on a real ambiguity, not a placeholder.

**Type consistency:**
- `_solve_F_sylvester(A, B, G, n_pre)` signature consistent across Tasks 5 (klein.py port) and 6 (smets_wouters deletion).
- `ProxySVARResult.irf_point.shape == (H+1, n, n)` used consistently in Tasks 2-4.
- `klein_solve(A, B, n_pre, C, *, strict)` signature unchanged across the file modifications.
- All gate runs use the same `python tools/release_check.py` command (no flag drift).

**Risks pulled forward from the spec:**
- **R3 (Klein refactor may shift SW07 IRFs):** Tasks 5-6 require byte-equal SW07 results; if Task 6 Step 4 fails, the threshold or port has a bug — root-cause before continuing.
- **R1 (R1_04 re-execution introduces numerical churn):** Task 4 Steps 3-4 explicitly check the diff before committing; pin seeds in the builder if churn is unrelated to the intentional axis flip.

**Out of scope (deferred):**
- 0.46.1 (whitelist drain) — N/A: baseline is empty.
- A1 (regress/lp.py retirement) — explicitly deferred; spec entry should be updated post-release to "kept by design."
- The notebook `T_fiscal_channels.ipynb:666` decision — Task 4 Step 6 leaves it as "inspect and decide; OK to defer if ambiguous."
