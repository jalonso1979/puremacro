# Notebook Pedagogy: Enriched Showcases + Build-Your-Own-Index Lab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen three showcase notebooks (`01_wealth_inequality`, `06_svar_identification`, `11_narrative_uncertainty`) with equations + intuition + fill-in exercises, and add a new `13_build_your_own_index` lab that builds four uncertainty indices from one toolkit — all bilingual, all executing green.

**Architecture:** Notebooks are jupytext *percent* `.py` sources (source of truth) compiled to executed `.ipynb` by `tools/build_notebooks.py`. We edit only the `.py`, rebuild, and the build executes inline `assert`s as the test. Every English `.py` has a code-byte-identical Spanish `_es.py` twin (prose translated). No library code changes.

**Tech Stack:** Python 3.11+, jupytext/nbconvert, pure-numpy puremacro (`vfi`, `var.identify`, `narrative.indices`, `factor`, `garch`, `gar`, `sigma`), matplotlib, pytest.

---

## Conventions & verified APIs (read once; every task relies on this)

### Build / verify commands
- Build + execute one notebook (writes the committed `.ipynb` WITH outputs):
  `python tools/build_notebooks.py <stem>` (e.g. `python tools/build_notebooks.py 11_narrative_uncertainty`). Exit code 0 ⇒ it executed top-to-bottom and **all inline `assert`s passed**. A failed assert ⇒ non-zero exit.
- Edit the **`.py` only**. Never hand-edit the `.ipynb` (the builder regenerates it and would clobber hand edits). Patch `.py` + rebuild together.
- Run from the package root: `cd puremacro` (the dir containing `notebooks/`, `tools/`, `pyproject.toml`).

### Bilingual rule (hard gate: `tests/test_bilingual_docs.py::test_notebooks_have_spanish_sibling`)
- Every `notebooks/<stem>.py` needs `notebooks/<stem>_es.py`. When editing an EN source, apply the **identical code** edits to its `_es` twin and translate only the prose (markdown + comments) to **native academic Spanish**.
- Register exemplar to match: `notebooks/11_narrative_uncertainty_es.py` (already exists). Read it before translating.
- Do not translate code identifiers, API names, or LaTeX math.

### Fill-in-the-blank pattern (must execute green)
A *Your turn* cell ships a **working default** the student overwrites. Mark the line with `# ← change this …`, and follow with an `assert` that holds for the default. The committed notebook runs; the "blank" is conceptual.

### Commit hygiene (Drive tree; avoid sibling churn — see memory)
Building one notebook can leave *other* notebooks' `.ipynb` with id/timestamp churn. Before each commit, `git add` only the specific paths you changed (the `.py` + its `.ipynb`, and the `_es` pair). If a sibling `.ipynb` shows spurious diffs, `git restore` it first.

### Verified API reference (dumped this session — write against these, not memory)
```text
# puremacro.vfi  (notebooks/01)
aiyagari_steady_state(*, alpha=0.36, delta=0.08, beta=0.96, gamma=1.0,
                      rho=0.9, sigma=0.2, n_z=5, n_a=150, a_max=80.0,
                      r_bracket=None, backend="numpy", solve_options=None)
   -> dict: 'equilibrium','r','w','K','L','Y','wealth_gini'
huggett_steady_state(n_z=7, n_a=150, ...) -> dict: 'equilibrium','r','mean_assets','frac_borrowing'
lorenz_and_gini(mu, wealth) -> (pop_share, value_share, gini)
solve_permanent_types(build_type_fn, weights) -> obj with .mixture_distribution()
VFIProblem(a_grid, z_grid, P_z, return_fn, beta, options)

# puremacro.var.identify  (notebooks/06)
cholesky_svar(Y, p, horizon, n_boot, ci, seed) -> .irf_point/.irf_lower/.irf_upper (H+1,n,n), .summary()
sign_restriction_svar(Y, p, horizon, restrictions={h:[+1/-1,...]}, n_draws, ci, seed)
   -> .irf_median/.irf_lower/.irf_upper, .n_accepted, .summary()
# irf arrays indexed [horizon, response_var, shock_var]

# puremacro.narrative.indices  (notebooks/11 and 13)
epu(text_iter, *, country, language="en", lexicon=None, normalize="bbd_100",
    base_period=None, agg="mean", with_quality=False) -> RiskIndex (has .series)
#   records are (date, text, source_url, metadata) 4-tuples
#   lexicon override = {"economy": frozenset, "policy": frozenset, "uncertainty": frozenset}
mpu(text_iter, *, country, language, normalize) -> RiskIndex
LEXICONS["epu"]["en"]["economy"|"policy"|"uncertainty"]; LEXICONS["mpu"]["en"]

# puremacro.factor / garch / gar / sigma  (notebooks/13)
factor.pca_factors(X, k=1, *, demean=True, standardize=True)
   -> dict: 'factors'(T,k),'loadings'(n,k),'eigvals','variance_share','cumulative_share'
garch.garch11_fit(returns, mean="zero") -> GARCH11Result(.sigma=pd.Series cond-vol, .loglik, .converged, .persistence)
gar.fci(df, *, tightening_columns=None, handle_nan="drop") -> dict: 'index'(T,),'loadings'(n,),'var_explained'
sigma.SigmaObject(sigma, R, labels, country="", period="")
   -> .var(w), .var_no_cov(w), .cov_premium_var(w), .mean_corr(), .diag_contrib(w)

# notebooks/_nbstyle.py
apply_style(); palette(n)->list[str]; styles(n)->list
```

### The enriched-showcase template (the 7 parts, in order)
1. Motivating question (keep existing opener) · 2. **The method in math** (LaTeX) · 3. `**Intuition.**` paragraph · 4. Worked code (existing + 1-2 "why" comments) · 5. **Read the output** · 6. **Your turn** (fill-in + assert + 2-3 prompts) · 7. **How comprehensive is this?** (cross-references).

---

## Task 0: Write the enriched-showcase template doc

**Files:**
- Create: `notebooks/_TEMPLATE.md`

- [ ] **Step 1: Create `notebooks/_TEMPLATE.md`** with this content:

```markdown
# Enriched showcase template

The structure every showcase notebook (`NN_topic.py`) follows. Edit the `.py`
(jupytext percent), then `python tools/build_notebooks.py NN_topic`. Keep the
Spanish twin `NN_topic_es.py` code-identical (translate prose only).

Cells, in order:

1. **Motivating question** — 1-2 sentences. What economic question does this answer?
2. **The method in math** — one `# %% [markdown]` cell, the governing equations in
   LaTeX ($...$ / $$...$$), compact (~4-8 lines).
3. **Intuition** — a `**Intuition.**` lead-in paragraph mapping the math to economics.
   No emoji; bold used sparingly.
4. **Worked code** — the runnable example; 1-2 inline comments per block say *why*.
   Pure-numpy, fixed seed, inline `assert`s on headline numbers.
5. **Read the output** — a `# %% [markdown]` cell interpreting the printed numbers and
   the hero figure explicitly.
6. **Your turn** — a fill-in cell: a working default marked `# ← change this`, a
   downstream `assert` that holds for the default, then 2-3 graded prompts
   (basic → stretch). The committed notebook must execute green.
7. **How comprehensive is this?** — 2-3 lines pointing to the other puremacro entry
   points that use the same machinery.

Constraints: numpy-only / Pyodide-safe (no statsmodels/linearmodels/arch/bs4), the
`_nbstyle` preamble (`apply_style()`, `palette(n)`), deterministic seeds.
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/_TEMPLATE.md
git commit -m "docs(notebooks): enriched-showcase template"
```

---

## Task 1: Deepen `11_narrative_uncertainty` (start here — seeds the lab)

**Files:**
- Modify: `notebooks/11_narrative_uncertainty.py`
- Modify: `notebooks/11_narrative_uncertainty_es.py`

- [ ] **Step 1: Insert the "method in math" cell.** In `11_narrative_uncertainty.py`, after the title/intro markdown cell (the one ending "...recover it with the Baker-Bloom-Davis **EPU** and the monetary-policy **MPU** indices.") and **before** the `## Setup — imports and style` cell, insert:

```python
# %% [markdown]
# ## The index in one equation
#
# Baker-Bloom-Davis EPU flags a document $d$ as *uncertain* only when it hits all
# **three** term groups at once:
#
# $$ \mathrm{flag}(d) = \mathbb{1}\!\left[(\exists\,t\in E:\,t\in d)\,\wedge\,(\exists\,t\in P:\,t\in d)\,\wedge\,(\exists\,t\in U:\,t\in d)\right], $$
#
# where $E,P,U$ are the **economy**, **policy**, and **uncertainty** lexicons. The raw
# index in period $\tau$ is the share of flagged documents,
# $\mathrm{EPU}^{\mathrm{raw}}_\tau = \frac{1}{N_\tau}\sum_{d\in\tau}\mathrm{flag}(d)$,
# and `normalize="bbd_100"` rescales it to the published units (sample mean 100, sd 50):
# $\mathrm{EPU}_\tau = 100 + 50\cdot(\mathrm{EPU}^{\mathrm{raw}}_\tau - \overline{\mathrm{EPU}^{\mathrm{raw}}})/\mathrm{sd}(\mathrm{EPU}^{\mathrm{raw}})$.
#
# **Intuition.** The *co-occurrence* of all three groups is what makes the index specific.
# A piece about "economic growth" (no policy, no uncertainty) or a sports story that
# happens to say "uncertain" never fires; only documents simultaneously about the economy,
# about policy, and about uncertainty count. MPU drops the co-occurrence requirement and
# just counts monetary-policy keywords, then z-scores — looser, but enough when the
# vocabulary is already narrow.
```

- [ ] **Step 2: Add the fill-in *Your turn* cell + comprehensive pointer.** Replace the final markdown cell (`# **What this shows / Try it** ...`, the last cell of the file) with the following three cells:

```python
# %% [markdown]
# ## Your turn — build a *climate*-policy uncertainty index
#
# `epu()` takes a `lexicon=` override of the form
# `{"economy": frozenset(...), "policy": frozenset(...), "uncertainty": frozenset(...)}`.
# Swap the "economy" group for a **climate** vocabulary and you have a climate-policy
# uncertainty index — with no new library code. Change the three frozensets (and the
# injected signal phrases) below to your own domain and re-run.

# %%
# ← Replace these three groups with your own domain vocabulary.
my_lexicon = {
    "economy": frozenset({"climate", "emissions", "carbon", "warming", "greenhouse"}),
    "policy": frozenset({"policy", "regulation", "treaty", "tax", "subsidy", "mandate"}),
    "uncertainty": frozenset({"uncertain", "uncertainty", "unclear", "ambiguous"}),
}
# ← And the phrases injected into the 2020 window (must hit all three groups).
CLIMATE_HIGH = [
    "Climate policy uncertainty is high as the carbon tax treaty remains unclear.",
    "Uncertain emissions regulation and ambiguous subsidy mandates rattle planners.",
    "Greenhouse policy is uncertain amid an ambiguous warming-target treaty.",
]

clim_records = []
for (date, text, url, meta) in records:
    in_w = HIGH_START <= date <= HIGH_END
    add = CLIMATE_HIGH[int(RNG.integers(0, len(CLIMATE_HIGH)))] if (in_w and RNG.random() < 0.70) else ""
    clim_records.append((date, (text + " " + add).strip(), url, meta))

clim_s = epu(clim_records, country="SYN", language="en",
             lexicon=my_lexicon, normalize="bbd_100").series.dropna()
in_w = (clim_s.index >= HIGH_START) & (clim_s.index <= HIGH_END)
clim_gap = clim_s[in_w].mean() - clim_s[~in_w].mean()
print(f"climate-EPU  in-window = {clim_s[in_w].mean():.1f}  "
      f"out-window = {clim_s[~in_w].mean():.1f}  gap = {clim_gap:.1f} pts")
assert clim_gap > 20, "your climate lexicon should fire in the injected window"

# %% [markdown]
# **Prompts.** (1) Add a fourth term group by *prepending* a sector word to every phrase and
# extending `my_lexicon` — does precision improve? (2) Set `normalize="zscore"` and compare
# the in/out gap in z-units. (3) Build the same index in another language by translating the
# three frozensets and passing `language="es"`.
#
# **How comprehensive is this?** The same `(date, text, url, metadata)` records feed every
# narrative index in puremacro: `mpu` (monetary-policy uncertainty), `lui`/`lwui`
# (labor-market and wage uncertainty), and `tone` (hawkish/dovish central-bank tone). The
# library ships ~70 free text connectors (central-bank speeches, Beige Book, EUR-Lex,
# Bluesky, …) and multilingual lexicons. **Notebook 13** generalizes the *idea* of an index
# beyond text — the same "define a kernel, apply it to data, normalize" recipe builds macro,
# financial, and cross-sectional uncertainty indices.
```

- [ ] **Step 3: Build EN and verify it executes + asserts pass**

Run: `python tools/build_notebooks.py 11_narrative_uncertainty`
Expected: exit code 0, prints the climate-EPU gap line, no assertion error.

- [ ] **Step 4: Mirror to the Spanish twin.** Open `notebooks/11_narrative_uncertainty_es.py`. Apply the **same two structural edits** (insert the math cell; replace the final cell with the three new cells). Keep all **code byte-identical**; translate every markdown line and comment to academic Spanish, matching the register already used in that file. (E.g. "## The index in one equation" → "## El índice en una ecuación"; "Your turn" → "Tu turno"; "How comprehensive is this?" → "¿Qué tan completa es la biblioteca?")

- [ ] **Step 5: Build ES and verify**

Run: `python tools/build_notebooks.py 11_narrative_uncertainty_es`
Expected: exit code 0, no assertion error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/11_narrative_uncertainty.py notebooks/11_narrative_uncertainty.ipynb \
        notebooks/11_narrative_uncertainty_es.py notebooks/11_narrative_uncertainty_es.ipynb
git commit -m "docs(nb11): equations, intuition, climate fill-in (EN+ES)"
```

---

## Task 2: Deepen `06_svar_identification`

**Files:**
- Modify: `notebooks/06_svar_identification.py`
- Modify: `notebooks/06_svar_identification_es.py`

- [ ] **Step 1: Insert the "method in math" cell.** In `06_svar_identification.py`, after the title/intro cell (ending "...Everything runs in the browser on synthetic data.") and before the imports `# %%` cell, insert:

```python
# %% [markdown]
# ## From reduced form to structure
#
# A reduced-form VAR($p$) projects each variable on the recent past of all variables:
# $$ y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t, \qquad \mathbb{E}[u_t u_t'] = \Sigma_u. $$
# The residuals $u_t$ are *forecast errors*, not economic shocks — they are correlated across
# equations ($\Sigma_u$ is not diagonal). Structural shocks $\varepsilon_t$ are the
# mutually-uncorrelated economic disturbances, related to the residuals by an impact matrix $B$:
# $$ u_t = B\,\varepsilon_t,\qquad \mathbb{E}[\varepsilon_t\varepsilon_t']=I \;\Rightarrow\; \Sigma_u = BB'. $$
#
# **The identification problem.** $\Sigma_u$ is symmetric, so it pins down only $n(n+1)/2$
# numbers, but $B$ has $n^2$ free elements. We are short $n(n-1)/2$ restrictions — for $n=3$,
# three of them. *How* you supply those restrictions is the identification scheme:
#
# - **Cholesky** sets $B=\operatorname{chol}(\Sigma_u)$, the unique lower-triangular factor.
#   Its $n(n-1)/2$ zeros above the diagonal *are* the restrictions: a recursive ordering in
#   which variable 1 reacts to no shock on impact, variable 2 only to shock 1, and so on.
# - **Sign restrictions** use that for *any* orthogonal $Q$ ($QQ'=I$), $\tilde B = BQ$ also
#   satisfies $\Sigma_u = \tilde B\tilde B'$. Instead of zeros we keep every rotation whose
#   impact responses match a sign prior. Many $B$'s qualify, so the shock is
#   **set-identified** — we report the median and a band across admissible rotations.
#
# **Intuition.** A recursive ordering is an *economic* assumption about what can move within
# the period; reorder the variables and the "shock" changes. Sign restrictions assume less,
# so they identify a *set* of responses — which is why their bands are wider.
```

- [ ] **Step 2: Add the fill-in *Your turn* + comprehensive pointer.** Replace the final markdown cell (`# **What this shows about `puremacro.var.identify`:** ...`) with these two cells:

```python
# %% [markdown]
# ## Your turn — the ordering is an assumption
#
# A Cholesky SVAR's recursive shock depends on the variable order: the contemporaneous
# response of an *earlier* variable to a *later* variable's shock is zero by construction.
# Re-order the columns of `Y` and watch a previously-zero impact response open up. Change
# `order` below.

# %%
# ← Change this ordering (a permutation of 0=output, 1=prices, 2=rate).
order = [2, 0, 1]                      # rate FIRST → output/prices may now react on impact
Y_re = Y[:, order]
chol_re = cholesky_svar(Y_re, p=2, horizon=H, n_boot=200, ci=0.9, seed=0)
pos = {v: order.index(v) for v in (0, 1, 2)}            # new position of each original var
# Output's contemporaneous response to the rate shock:
out_to_rate = chol_re.irf_point[0, pos[0], pos[RATE]]
print(f"order={order}: output's impact response to the rate shock = {out_to_rate:+.3f}")
# With the rate ordered before output, this is no longer forced to zero (it was, originally).
assert abs(out_to_rate) > 1e-8

# %% [markdown]
# **Prompts.** (1) Flip a sign in `restrictions` above (e.g. make the rate negative) and
# re-run the sign-restriction cell — the prior becomes unsatisfiable and you get an
# informative `RuntimeError`. (2) Put the rate *last* (`order=[0,1,2]`) and confirm
# `out_to_rate` returns to ≈0 (the original recursive zero). (3) Widen the bootstrap
# (`n_boot=800`) and see how little the point IRF moves.
#
# **How comprehensive is this?** `puremacro.var` / `puremacro.svar` go well beyond these two
# schemes: Blanchard-Quah long-run restrictions, proxy-SVAR / external-instrument
# identification (e.g. Mertens-Ravn narrative tax shocks), forecast-error variance
# decompositions, and historical decompositions. The `examples/` gallery (`bloom2009`,
# `sign_restrictions_uhlig`, `svariv_mertens_ravn`, `gk_robust_signs`) runs each end-to-end.
```

- [ ] **Step 3: Build EN and verify**

Run: `python tools/build_notebooks.py 06_svar_identification`
Expected: exit code 0, prints the reordered output-to-rate impact, no assertion error.

- [ ] **Step 4: Mirror to the Spanish twin.** Open `notebooks/06_svar_identification_es.py`; apply the same two edits, code byte-identical, prose in academic Spanish (e.g. "## From reduced form to structure" → "## De la forma reducida a la estructura").

- [ ] **Step 5: Build ES and verify**

Run: `python tools/build_notebooks.py 06_svar_identification_es`
Expected: exit code 0, no assertion error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/06_svar_identification.py notebooks/06_svar_identification.ipynb \
        notebooks/06_svar_identification_es.py notebooks/06_svar_identification_es.ipynb
git commit -m "docs(nb06): identification math, ordering fill-in (EN+ES)"
```

---

## Task 3: Deepen `01_wealth_inequality` (slowest build)

**Files:**
- Modify: `notebooks/01_wealth_inequality.py`
- Modify: `notebooks/01_wealth_inequality_es.py`

- [ ] **Step 1: Insert the "model in math" cell.** In `01_wealth_inequality.py`, after the title/intro cell (ending "...All with `puremacro.vfi`.") and before the imports `# %%` cell, insert:

```python
# %% [markdown]
# ## The model in three equations
#
# **Households.** A continuum of households face idiosyncratic labor-productivity risk $z$
# (an AR(1), discretized by Tauchen) and cannot borrow. They solve
# $$ V(a,z) = \max_{a'\ge 0}\; u(c) + \beta\,\mathbb{E}\!\left[V(a',z')\mid z\right]
# \quad\text{s.t.}\quad c = w\,e^{z} + (1+r)\,a - a', $$
# with $u(c)=\log c$ here ($\gamma=1$). The constraint $a'\ge 0$ is the engine of the model.
#
# **Firms.** A representative Cobb-Douglas firm sets factor prices from aggregates:
# $$ r = \alpha\,(K/L)^{\alpha-1} - \delta, \qquad w = (1-\alpha)\,(K/L)^{\alpha}. $$
#
# **Equilibrium.** Let $\mu$ be the stationary distribution induced by the saving policy.
# The interest rate clears the capital market, $K = \int a\, d\mu(a,z)$. In **Huggett** the
# asset is a bond in zero net supply, so clearing is $\int a\,d\mu = 0$ and the equilibrium
# rate sits strictly below $1/\beta-1$.
#
# **Intuition.** With no borrowing and uninsurable income risk, households self-insure by
# holding a precautionary buffer of assets. That buffer is why a non-degenerate wealth
# distribution emerges even though everyone is *ex-ante* identical — and why the clearing
# rate is pushed below the complete-markets benchmark $1/\beta-1$: the extra desire to save
# bids the return down. Permanent differences in patience $\beta$ then stretch the
# distribution further — patient households climb the asset grid, impatient ones pile near
# the constraint.
```

- [ ] **Step 2: Insert a "Read the output" cell** immediately after the Aiyagari solve cell (the one printing `Aiyagari:  r* = ...` and asserting the Gini), before the `## 2. Permanent β-heterogeneity` markdown:

```python
# %% [markdown]
# **Read the output.** The complete-markets benchmark is $1/\beta-1 = 1/0.96-1 \approx 0.042$.
# The equilibrium $r^\*$ printed above sits *below* it: that gap is the precautionary wedge —
# households' self-insurance demand for assets bids the return down. The wealth Gini near
# $0.6$ is produced entirely by idiosyncratic risk plus the borrowing constraint, with no
# ex-ante heterogeneity yet.
```

- [ ] **Step 3: Replace the final *Try it* line with a fill-in *Your turn* + comprehensive pointer.** Replace the last markdown cell (`# **What this shows about `puremacro.vfi`:** ... watch the Gini move.`) with these three cells:

```python
# %% [markdown]
# ## Your turn — how much does patience heterogeneity matter?
#
# The mixture above used `betas = [0.96, 0.93]`. Widen or narrow the spread — *keeping prices
# fixed*, so this is cheap (no GE re-solve) — and watch the Gini respond. Change `betas_you`.

# %%
# ← Change this β spread (patient first, impatient second). Keep both < 1.
betas_you, weights_you = [0.97, 0.91], [0.5, 0.5]

def build_type_you(t):
    b = betas_you[t]
    def rf(ap, a, z, xp=np):
        c = w_star * xp.exp(z) + (1.0 + r_star) * a - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
    return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P_z, return_fn=rf,
                      beta=b, options=dict(tol=1e-9, n_howard=40))

pt_you = solve_permanent_types(build_type_you, weights_you)
mu_you = pt_you.mixture_distribution()
_, _, gini_you = lorenz_and_gini(mu_you, np.broadcast_to(a_grid[:, None], mu_you.shape))
print(f"β spread {betas_you}: mixture Gini = {gini_you:.3f}  "
      f"(baseline [0.96, 0.93] → {gini_mix:.3f}; homogeneous → {gini_ai:.3f})")
assert gini_you > gini_ai      # any patience heterogeneity raises inequality vs the homogeneous economy

# %% [markdown]
# **Prompts.** (1) Shrink the spread to `[0.955, 0.945]` — does the Gini fall toward the
# homogeneous value? (2) Make the weights asymmetric (`[0.8, 0.2]`). (3) *Stretch* (slow,
# re-solves GE): call `aiyagari_steady_state(..., sigma=0.30)` to raise income risk and
# compare `wealth_gini` to the baseline `sigma=0.2`.
#
# **How comprehensive is this?** `puremacro.vfi` is a full heterogeneous-agent toolkit. The
# same `VFIProblem` → solve → stationary-distribution → market-clearing stack powers the
# other showcase notebooks: **Krusell-Smith** aggregate risk and transition paths (NB02),
# **life-cycle / OLG** with mortality (NB03), **Hopenhayn** firm entry/exit (NB04), and
# **two-asset** portfolios with EGM and Epstein-Zin preferences (NB05). Endogenous grids,
# permanent types, and GE transitions are all first-class.
```

- [ ] **Step 4: Build EN and verify** (this is the slowest notebook; if it exceeds ~5 min, run it in the controller's background, not a subagent — see memory)

Run: `python tools/build_notebooks.py 01_wealth_inequality`
Expected: exit code 0, prints the β-spread Gini line, no assertion error.

- [ ] **Step 5: Mirror to the Spanish twin.** Open `notebooks/01_wealth_inequality_es.py`; apply the same edits, code byte-identical, prose in academic Spanish.

- [ ] **Step 6: Build ES and verify**

Run: `python tools/build_notebooks.py 01_wealth_inequality_es`
Expected: exit code 0, no assertion error.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_wealth_inequality.py notebooks/01_wealth_inequality.ipynb \
        notebooks/01_wealth_inequality_es.py notebooks/01_wealth_inequality_es.ipynb
git commit -m "docs(nb01): model equations, precautionary intuition, β-spread fill-in (EN+ES)"
```

---

## Task 4: Prototype the four NB13 recipes to green (lock the assert thresholds)

**Files:**
- Create (temporary, NOT committed): `notebooks/_scratch_nb13.py`

- [ ] **Step 1: Write the prototype** combining all four recipes, with the exact code from Task 5 (preamble + recipes 1-4). Print every headline ratio/number that an `assert` will gate.

- [ ] **Step 2: Run it**

Run: `python notebooks/_scratch_nb13.py`
Expected: exit 0; prints the text-EPU gap, the JLN mid-vs-rest ratio, the FCI var-explained and corr-with-stress, and the covariance premium.

- [ ] **Step 3: Lock thresholds.** Read the printed numbers. In Task 5, set each `assert` to a margin comfortably inside the observed value (rule: threshold = 0.8 × observed for "greater-than" gates). Record the observed numbers here as a comment so Task 5 uses them. If recipe 2's mid/rest ratio is < 1.25, increase the factor-innovation volatility window contrast (`vol_hi`) until the ratio is ≥ 1.4, then set the assert to `> 1.2`.

- [ ] **Step 4: Delete the scratch file** (do NOT commit it)

```bash
rm notebooks/_scratch_nb13.py
```

---

## Task 5: Write the new `13_build_your_own_index` lab (EN + ES)

**Files:**
- Create: `notebooks/13_build_your_own_index.py`
- Create: `notebooks/13_build_your_own_index_es.py`

- [ ] **Step 1: Write `notebooks/13_build_your_own_index.py`** with exactly this content (adjust only the assert thresholds per Task 4):

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build your own uncertainty index
#
# An "uncertainty index" sounds like proprietary infrastructure — a Bloomberg terminal, a
# vendor feed. It is not. Almost every published uncertainty or financial-conditions index is
# one of **four** elementary operations on data you can assemble yourself. This lab builds one
# of each with `puremacro`, on synthetic data, entirely in the browser — and at each step you
# change the inputs to your own.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.narrative.indices import epu
from puremacro.factor import pca_factors
from puremacro.garch import garch11_fit
from puremacro.gar import fci
from puremacro.sigma import SigmaObject

# Pyodide-clean guard: none of these heavy/paid modules should have leaked.
_bad = [m for m in ("bs4", "requests", "anthropic", "sentence_transformers",
                    "statsmodels", "linearmodels", "arch", "numba") if m in sys.modules]
assert _bad == [], f"Unexpected module leak: {_bad}"

RNG = np.random.default_rng(13)

# %% [markdown]
# ## Recipe 1 — text → an EPU index
#
# **Kernel: three-group co-occurrence.** A document counts as "uncertain" when it mentions
# the economy *and* policy *and* uncertainty. We plant a 2020 signal in a synthetic corpus
# and recover it. `epu()` takes `(date, text, url, metadata)` records and `normalize="bbd_100"`
# (mean 100, sd 50). Change `NEUTRAL`/`EPU_HIGH` to your own corpus.

# %%
NEUTRAL = ["local sports results were mixed", "the weather stayed mild",
           "a new museum opened downtown", "rail service ran on schedule"]
ECON = ["the economy expanded", "gdp and output", "economic growth"]
EPU_HIGH = [   # ← swap in your own corpus; each phrase must hit economy ∧ policy ∧ uncertainty
    "economic policy uncertainty is high amid uncertain fiscal regulation",
    "uncertain trade policy is weighing on the economy",
    "regulatory uncertainty clouds the economic policy outlook",
]
HIGH_START, HIGH_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")

records = []
for month in pd.date_range("2018-01-01", "2022-12-01", freq="MS"):
    for _ in range(4):
        d = pd.Timestamp(month + pd.Timedelta(days=int(RNG.integers(0, 27))))
        in_w = HIGH_START <= d <= HIGH_END
        base = NEUTRAL[int(RNG.integers(0, len(NEUTRAL)))] + ". " + ECON[int(RNG.integers(0, len(ECON)))]
        sig = EPU_HIGH[int(RNG.integers(0, len(EPU_HIGH)))] if RNG.random() < (0.75 if in_w else 0.08) else ""
        records.append((d, (base + ". " + sig).strip(), "syn://news", {"language": "en"}))

epu_s = epu(records, country="SYN", language="en", normalize="bbd_100").series.dropna()
_w = (epu_s.index >= HIGH_START) & (epu_s.index <= HIGH_END)
epu_gap = epu_s[_w].mean() - epu_s[~_w].mean()
print(f"[1] text EPU: in-minus-out-window gap = {epu_gap:.1f} bbd points")
assert epu_gap > 25      # Task 4: set to 0.8 × observed gap

# %% [markdown]
# ## Recipe 2 — a macro panel → a JLN-*style* uncertainty index
#
# **Kernel: common factors + conditional volatility.** Macro uncertainty is the *common,
# unpredictable* volatility across many series. Recipe: (1) extract common factors with
# `pca_factors`; (2) forecast each series from the *lagged* factor; (3) fit GARCH(1,1) to each
# one-step forecast error for its conditional-volatility path; (4) average across series.
#
# This is a deliberately *simplified* Jurado-Ludvigson-Ng — one horizon, factor-projection
# residuals, GARCH volatility — **not** the full multi-horizon stochastic-volatility machinery.
# For the published series, see `puremacro.fetch.jln`.

# %%
T, n = 240, 8
t = np.arange(T)
vol_hi, win = 3.5, 30                                   # ← injected high-volatility window
sig_t = np.where((t >= T // 2) & (t < T // 2 + win), vol_hi, 1.0)
f_true = np.zeros(T)
for i in range(1, T):
    f_true[i] = 0.5 * f_true[i - 1] + sig_t[i] * RNG.standard_normal()
loadings = RNG.uniform(0.3, 1.0, size=n)
X = np.outer(f_true, loadings) + 0.5 * RNG.standard_normal((T, n))     # (T, n) macro panel

F = pca_factors(X, k=1)["factors"]                      # (T, 1) common factor
F_lag = np.vstack([np.zeros((1, 1)), F[:-1]])           # F_{t-1}
U = np.zeros(T)
for j in range(n):
    Z = np.column_stack([np.ones(T), F_lag[:, 0]])      # forecast from a constant + lagged factor
    coef, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)
    resid = X[:, j] - Z @ coef                          # one-step forecast error
    U += np.asarray(garch11_fit(resid).sigma)           # its conditional-volatility path
U = pd.Series(U / n, index=pd.RangeIndex(T))            # average across series

mid = U[T // 2:T // 2 + win].mean()
rest = pd.concat([U[:T // 2], U[T // 2 + win:]]).mean()
print(f"[2] JLN-style macro uncertainty: window mean {mid:.2f} vs rest {rest:.2f} (ratio {mid/rest:.2f})")
assert mid > 1.2 * rest      # Task 4: confirm ratio; raise vol_hi if < 1.25

# %% [markdown]
# ## Recipe 3 — financial indicators → a Financial Conditions Index
#
# **Kernel: first principal component, sign-normalized.** An FCI compresses many financial
# indicators into one "tightness" factor. `gar.fci` standardizes the panel, takes the first PC,
# and orients it so the columns you name in `tightening_columns` load positively. Add your own
# indicator columns below.

# %%
stress = np.zeros(T)
for i in range(1, T):
    stress[i] = 0.85 * stress[i - 1] + RNG.standard_normal()        # latent financial-stress factor
fin = pd.DataFrame({
    "credit_spread": 0.8 * stress + 0.4 * RNG.standard_normal(T),   # higher = tighter
    "equity_vol":    0.7 * stress + 0.5 * RNG.standard_normal(T),   # higher = tighter
    "term_spread":  -0.6 * stress + 0.5 * RNG.standard_normal(T),   # higher = looser
    "equity_price": -0.7 * stress + 0.5 * RNG.standard_normal(T),   # higher = looser
}, index=pd.RangeIndex(T))

out = fci(fin, tightening_columns=["credit_spread", "equity_vol"])  # ← name your "tighter when high" cols
fci_idx = pd.Series(out["index"], index=fin.index)
corr_stress = float(np.corrcoef(fci_idx, stress)[0, 1])
print(f"[3] FCI: var explained by PC1 = {out['var_explained']:.2f}; corr(FCI, latent stress) = {corr_stress:+.2f}")
assert out["var_explained"] > 0.4 and corr_stress > 0.7      # Task 4: confirm

# %% [markdown]
# ## Recipe 4 — a cross-section of volatilities → a comovement premium
#
# **Kernel: the $\sigma\!\cdot\! R$ quadratic form.** Aggregate volatility is
# $\mathrm{Var}(w'g) = w'\Sigma w$ with $\Sigma = \mathrm{diag}(\sigma)\,R\,\mathrm{diag}(\sigma)$.
# The *covariance premium* $(\mathrm{Var} - \mathrm{Var}_{\text{no-cov}})/\mathrm{Var}_{\text{no-cov}}$
# isolates how much cross-sector comovement inflates aggregate risk. Change `sig`, `rho`, or `w`.

# %%
labels = ["manuf", "services", "construction", "retail", "energy"]
sig = np.array([2.0, 1.2, 3.0, 1.5, 4.0])               # ← per-sector volatilities
rho = 0.6                                               # ← common pairwise correlation
R = (1 - rho) * np.eye(len(sig)) + rho * np.ones((len(sig), len(sig)))
w = np.full(len(sig), 1 / len(sig))                     # ← weights (equal here)

S = SigmaObject(sig, R, labels)
premium = S.cov_premium_var(w)
print(f"[4] comovement: mean corr = {S.mean_corr():.2f}; covariance premium = {premium:.1%}")
assert premium > 0      # positive correlation inflates aggregate variance above the diagonal sum

# %% [markdown]
# ### Hero figure — four indices, four data types, one library

# %%
cols = _nbstyle.palette(4)
def _z(s):
    s = np.asarray(s, dtype=float)
    return (s - np.nanmean(s)) / np.nanstd(s)

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
axes[0, 0].plot(epu_s.index, _z(epu_s.values), color=cols[0]); axes[0, 0].set_title("1 · text → EPU")
axes[0, 1].plot(U.index, _z(U.values), color=cols[1]); axes[0, 1].set_title("2 · macro panel → JLN-style")
axes[1, 0].plot(fci_idx.index, _z(fci_idx.values), color=cols[2]); axes[1, 0].set_title("3 · financial → FCI")
axes[1, 1].bar(range(len(labels)), S.diag_contrib(w), color=cols[3])
axes[1, 1].set_xticks(range(len(labels))); axes[1, 1].set_xticklabels(labels, rotation=30, ha="right")
axes[1, 1].set_title("4 · cross-section → variance shares")
for ax in axes.flat[:3]:
    ax.axhline(0, color="0.7", linewidth=0.6)
fig.suptitle("Four uncertainty indices from one toolkit (series standardized)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## One toolkit, four indices
#
# | You have… | The kernel | puremacro entry point |
# |---|---|---|
# | a text corpus | three-group co-occurrence | `narrative.indices.epu` (also `mpu`, `lui`, `tone`) |
# | a macro panel | common factors + conditional volatility | `factor.pca_factors` → `garch.garch11_fit` |
# | financial indicators | first principal component, sign-normalized | `gar.fci` |
# | a cross-section of volatilities | the $\sigma\!\cdot\! R$ quadratic form | `sigma.SigmaObject` |
#
# Every uncertainty index in the library is one of these moves. Pick the data you have, pick
# the matching kernel, normalize — and you have a research-grade index, in the browser, at \$0.
# To go deeper on the text kernel, see **Notebook 11**.
```

- [ ] **Step 2: Build EN and verify it executes + all four asserts pass**

Run: `python tools/build_notebooks.py 13_build_your_own_index`
Expected: exit 0; prints `[1]`…`[4]` lines; one 2×2 figure; no assertion error.

- [ ] **Step 3: Create the Spanish twin** `notebooks/13_build_your_own_index_es.py`: copy the EN file, keep **all code byte-identical**, translate every markdown cell and comment to academic Spanish (title → "# Construye tu propio índice de incertidumbre"; match the register of `11_narrative_uncertainty_es.py`).

- [ ] **Step 4: Build ES and verify**

Run: `python tools/build_notebooks.py 13_build_your_own_index_es`
Expected: exit 0, no assertion error.

- [ ] **Step 5: Commit**

```bash
git add notebooks/13_build_your_own_index.py notebooks/13_build_your_own_index.ipynb \
        notebooks/13_build_your_own_index_es.py notebooks/13_build_your_own_index_es.ipynb
git commit -m "feat(nb13): build-your-own-index lab — four kernels, one toolkit (EN+ES)"
```

---

## Task 6: Bump the gates, update the README, full verification

**Files:**
- Modify: `tests/test_notebooks/test_notebooks_execute.py:28`
- Modify: `notebooks/README.md`

- [ ] **Step 1: Bump the source-count gate.** In `tests/test_notebooks/test_notebooks_execute.py`, change line 28 from `assert len(srcs) == 30, ...` to:

```python
    assert len(srcs) == 32, [p.name for p in srcs]
```

- [ ] **Step 2: Add the NB13 row + template note to `notebooks/README.md`.** Add to the table (after the `11_narrative_uncertainty` row):

```markdown
| `13_build_your_own_index` | Build four uncertainty indices from one toolkit — text→EPU, macro panel→JLN-style, financial→FCI, cross-section→comovement premium |
```

And after the table, add:

```markdown
The deepened showcases (`01`, `06`, `11`, and new ones) follow the structure in
[`_TEMPLATE.md`](./_TEMPLATE.md): motivating question → the method in math → intuition →
worked code → read the output → a fill-in *Your turn* → "how comprehensive is this?".
```

- [ ] **Step 3: Run the bilingual parity test**

Run: `python -m pytest tests/test_bilingual_docs.py -q`
Expected: PASS (every EN notebook, including `13_build_your_own_index`, has an `_es` sibling).

- [ ] **Step 4: Run the execute-all notebook gate** (slow; runs all 32 sources). If the environment makes the full run too long for one call, run it in the controller's background.

Run: `python -m pytest -m slow tests/test_notebooks/test_notebooks_execute.py -q`
Expected: PASS (`len(srcs) == 32`; all sources execute without error).

- [ ] **Step 5: Confirm no public-API drift** (sanity — we changed no library code)

Run: `python -m pytest tests/test_public_api.py -q`
Expected: PASS (snapshot unchanged).

- [ ] **Step 6: Commit**

```bash
git add tests/test_notebooks/test_notebooks_execute.py notebooks/README.md
git commit -m "test(notebooks): source count 30->32; README row + template note for nb13"
```

---

## Self-review (completed by plan author)

**Spec coverage.** §3 template → Task 0. §4.1/4.2/4.3 flagship deepenings → Tasks 3/2/1 (each EN+ES, equations + intuition + read-output + fill-in + comprehensive pointer). §5 NB13 four recipes (incl. simplified-JLN honesty + recap table) → Tasks 4-5. §6 build/bilingual/fill-in guardrails → Conventions + per-task ES steps. §7 gate bump (+2 → 32), README, no API drift → Task 6. All sections mapped.

**Placeholder scan.** No TBD/TODO. Assert thresholds in NB13 are set to concrete starting values with an explicit Task-4 rule to confirm/tighten against observed prints (the TDD reality for numeric gates), not left blank. ES translation is a fully-specified task with a register exemplar, not a placeholder.

**Type/name consistency.** Function/field names match the verified API table: `aiyagari_steady_state(..., sigma=, rho=)`, `GARCH11Result.sigma`, `epu(..., lexicon=)`, `pca_factors(...)["factors"]`, `fci(...)["index"|"var_explained"]`, `SigmaObject(...).cov_premium_var/.mean_corr/.diag_contrib`, `cholesky_svar(...).irf_point[h, resp, shock]`. NB01 fill-in reuses variables defined earlier in that notebook (`w_star`, `r_star`, `a_grid`, `z_grid`, `P_z`, `gini_ai`, `gini_mix`). NB06 fill-in reuses `Y`, `H`, `RATE`. NB11 fill-in reuses `records`, `HIGH_START`, `HIGH_END`, `RNG`.
