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
# # Reproducible studies: from economic question to a portable record
#
# **How can we preserve the data, assumptions and diagnostics behind a result so another researcher can examine it?**
#
# We run three offline examples: a recursive monetary VAR with frozen Mexican observations, a calibrated household transfer written as DP `.mod` declarations, and an OECD production-network tariff scenario. They share a recording/export interface, not a common causal interpretation. Use Python 3.11 or newer and Puremacro 5.6.0: `python -m pip install "puremacro[notebooks]==5.6.0"`. Select that environment's kernel and keep `notebooks/_nbstyle.py` beside this notebook (or download the notebook source folder). A source checkout of version 5.6.0 can instead be installed with `python -m pip install -e ".[notebooks]"`.
#
# | Experiment | Evidence and shock | What it does not establish |
# |---|---|---|
# | Mexico | Frozen OECD/FRED monthly observations, January 2000–December 2023; 25 basis-point annualized rate innovation | An externally identified monetary policy surprise |
# | Household | Illustrative quarterly calibration; externally funded 0.05-unit transfer per household | Survey fit, fiscal financing or general equilibrium |
# | Tariff | OECD ICIO 2023 edition, reference year 2019; 5% US tariff on Mexican manufactures | A causal policy forecast, GDP or welfare effect |
#
# See the [workflow guide](https://jalonso1979.github.io/puremacro/docs/study_workflows/) and [Spanish notebook](69_reproducible_studies_es.ipynb).

# %% [markdown]
# ## The method in math
#
# The VAR estimates $x_t=c+\sum_{j=1}^{p}A_jx_{t-j}+u_t$, with $\Sigma_u=PP'$ and $P$ lower triangular. For rate coordinate $k$, its normalized level response is
# $$IRF_h=\Psi_h P e_k\,\frac{0.25}{P_{kk}}.$$
# The household comparison subtracts two paths with the same initial distribution and terminal values:
# $$\Delta C_t+\Delta A_{t+1}=(1+r)\Delta A_t+\tau_t,\qquad \Delta A_0=0.$$
# Tariff responses are $100(Y_c^{tariff}/Y_c^{base}-1)$, with $Y_c$ gross output valued at baseline prices. These are three different response conventions.
#
# ## Intuition
#
# **Intuition.** A saved number is incomplete evidence. The same coefficient can mean a monthly rate change, an annualized percentage-point innovation, or a relative change in a model quantity. We declare those choices before solving, inspect diagnostics after solving, and archive the inputs alongside the results. Checksums detect altered records; they do not certify identification or authenticate who produced the record.

# %% [markdown]
# ## Worked code
#
# All inputs ship with the package. No download, API key or model service is used. Plot labels and code are identical in both editions.

# %%
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _nbstyle
_nbstyle.apply_style()
from puremacro.examples.study_workflows import mexico_data, run_household, run_tariff
from puremacro.study import (
    StudySpec, VariableSpec, run_study, compare_studies, save_study, load_study,
)

SEED = 734

# %% [markdown]
# ### 1. Mexico: a declared recursive VAR
#
# The data loader verifies packaged file hashes. The two columns are already year-on-year CPI inflation and the annualized three-month interbank rate; we apply no logarithm, difference or additional annualization. The original retrieval date was not recorded, so these are not real-time vintages. Sources: [OECD/FRED inflation](https://fred.stlouisfed.org/series/CPALTT01MXM659N) and [interbank rate](https://fred.stlouisfed.org/series/IR3TIB01MXM156N).
#
# These frozen OECD observations distributed by FRED are third-party data, outside Puremacro's MIT code license. Cite OECD and retain attribution when sharing. The packaged manifest carries source URLs, hashes, citations and the [OECD reuse terms](https://www.oecd.org/en/about/terms-conditions.html); the original retrieval date remains unknown.
#
# Inflation precedes the rate in a two-variable recursive ordering. This imposes no contemporaneous inflation response to the identified rate innovation. Six monthly lags and a constant enter the VAR; horizons run from 0 through 18. The 90% bands use 199 seeded residual-bootstrap draws with the point-estimate shock normalization held fixed, and are pointwise, not simultaneous. A price puzzle is possible in this small information set.

# %%
data = mexico_data()
spec = StudySpec(
    name="Mexico recursive rate innovation", method="var_cholesky",
    variables=(VariableSpec("inflation_yoy", "percentage points"),
               VariableSpec("rate_3m", "percentage points per annum")),
    responses=("inflation_yoy", "rate_3m"), shock="rate_3m", frequency="M",
    identification="recursive", ordering=("inflation_yoy", "rate_3m"),
    lags=6, horizon=18, shock_size=0.25, ci=0.90, n_boot=199, seed=SEED,
)
assert data.index.equals(pd.period_range("2000-01", "2023-12", freq="M"))
assert data.shape == (288, 2) and np.isfinite(data.to_numpy()).all()
print(data.head().round(3).to_string())
print(json.dumps(asdict(spec), indent=2))
mexico = run_study(data, spec)
irf = mexico.to_frame()
impact = irf.loc[irf.horizon.eq(0)].set_index("response").estimate
np.testing.assert_allclose(impact.loc["rate_3m"], 0.25, atol=1e-13)
np.testing.assert_allclose(impact.loc["inflation_yoy"], 0.0, atol=1e-13)
print(mexico.audits.to_string(index=False))
print(pd.DataFrame(mexico.manifest["sample"]["effective_by_horizon"]).head().to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
for ax, response, label in zip(axes, spec.responses,
                              ("Inflation, pp (year-on-year)", "Rate, pp per annum")):
    row = irf.loc[irf.response.eq(response)]
    ax.fill_between(row.horizon, row.lower, row.upper, color=_nbstyle.BANDA,
                    label="90% pointwise bootstrap")
    ax.plot(row.horizon, row.estimate, label="25 bp rate innovation")
    ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.8)
    ax.set(xlabel="Months after innovation", ylabel=label)
axes[0].legend(fontsize=8)
fig.suptitle("Mexico: a recursive identification assumption")

# %% [markdown]
# **Read the output.** The impact checks verify the chosen 25-basis-point normalization and the zero restriction, not the truth of the recursive identification. Read the subsequent inflation response from the figure; do not impose a negative sign after seeing the data. `identification` remains `not_checked` in the audit. Changing the lag count uses the same input window but changes the fitted sample length, which the manifest records.

# %%
short_lag = run_study(data, replace(spec, name="Mexico, three lags", lags=3, n_boot=0))
comparison = compare_studies(mexico, short_lag)
print(comparison.to_frame().loc[lambda f: f.horizon.eq(6),
      ["study", "response", "estimate", "lags", "n_obs", "n_boot"]].round(4).to_string(index=False))
assert mexico.manifest["sample"]["raw_n_obs"] == short_lag.manifest["sample"]["raw_n_obs"]
assert irf.n_obs.iloc[0] == 282 and short_lag.to_frame().n_obs.iloc[0] == 285

# %% [markdown]
# ### 2. An externally funded household transfer
#
# The packaged template compiles the DP `.mod` objective, budget, state grid and income process to the existing EGM solver. This is a quarterly illustrative calibration, not an estimated household model. A transfer at date 0 changes consumption and saving at fixed prices. Both shocked and unshocked paths start with the same stationary distribution and use the same terminal values. `A` means assets entering the date: they cannot jump on impact.
#
# We print the retained model source and audit instead of treating solver convergence as a grid certificate. Failed boundary or Euler screens remain visible; the recorded audit concerns the stationary household, not a complete independent transition audit.

# %%
household = run_household(amount=0.05, horizon=40, n_a=100)
print(household.model_source)
print(json.dumps(asdict(household.spec), indent=2))
print(household.inputs.head().round(5).to_string(index=False))
print(household.audits[["check", "status"]].to_string(index=False))
print("Financing:", household.provenance["financing"])
hh = household.to_frame().pivot(index="horizon", columns="response", values="estimate")
transfer = household.evidence["shock_path"].transfer.to_numpy()
np.testing.assert_allclose(hh.A.iloc[0], 0, atol=1e-12)
# Independent resource identity: entering assets plus transfer fund consumption and next assets.
budget_gap = hh.C.iloc[:-1].to_numpy() + hh.A.iloc[1:].to_numpy() - (
    1.01 * hh.A.iloc[:-1].to_numpy() + transfer[:-1])
assert np.max(np.abs(budget_gap)) < 2e-10
print(f"Impact consumption: {hh.C.iloc[0]:.5f}; transfer: {transfer[0]:.5f}")
print(f"Largest aggregate budget gap on paired dates: {np.max(np.abs(budget_gap)):.2e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
for ax, response, label in zip(axes, ("C", "A"),
                              ("Consumption units per household", "Entering asset units per household")):
    ax.plot(hh.index, hh[response], label="Shocked minus zero-transfer path")
    ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.8)
    ax.set(xlabel="Quarters after transfer", ylabel=label)
fig.suptitle("Illustrative household transfer at fixed prices")

# %% [markdown]
# **Read the output.** Consumption rises immediately by less than the transfer; saving carries part of the resources to later dates. The resource identity is an algebraic check independent of the solver's convergence flag. It does not test household optimality, grid convergence or fiscal sustainability. There are no statistical confidence bands for this calibrated path.
#
# ### 3. A tariff through a production network
#
# The template conserves the bundled OECD table's flows while aggregating to USA/Mexico/rest and resources/manufacturing/other industries. It solves a 5% US tariff on Mexican manufactures with lump-sum duty rebates and fixed foreign saving in numeraire units. The USA resources price is one; labor receives two-thirds of value added by assumption. Gross output at baseline prices is not GDP or welfare.
#
# Cite OECD (2023), OECD Inter-Country Input-Output Tables, [ICIO source](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html). The [bundled attribution and adaptation terms](https://github.com/jalonso1979/puremacro/blob/main/puremacro/trade/_datafiles/SOURCES.md) apply to redistributed outputs.

# %%
tariff = run_tariff(rate=0.05)
print(json.dumps(asdict(tariff.spec), indent=2))
print(tariff.model_source)
print("Calibration matrix shape:", tariff.inputs.shape)
print(tariff.audits.to_string(index=False))
print(tariff.evidence["tariff_accounting_checks"].to_string(index=False))
trade_response = tariff.to_frame()
base_output = tariff.evidence["baseline_output"].sum(axis=0)
new_output = tariff.evidence["tariff_output"].sum(axis=0)
np.testing.assert_allclose(trade_response.estimate, 100 * (new_output / base_output - 1), atol=1e-10)
print(trade_response[["response", "estimate", "unit"]].round(6).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
bars = ax.bar(["USA", "Mexico", "Rest"], trade_response.estimate, color=_nbstyle.TINTA)
ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.8)
ax.bar_label(bars, fmt="%.5f", padding=4, color=_nbstyle.TEXTO)
ax.margins(y=0.25)
ax.set(ylabel="Gross output change at baseline prices (%)",
       title="Illustrative 5% US tariff on Mexican manufacturing")

# %% [markdown]
# **Read the output.** The signs and magnitudes describe this technology and fiscal closure. Physical-equation residuals mix native quantity, price, factor and budget units; their maximum is not a monetary loss. Accounting checks establish internal consistency, not the likely real-world effect of a tariff. No parameter uncertainty is estimated.
#
# ### 4. Export, reload, and reproduce the recorded experiment
#
# The archive holds inputs, specification, outputs, audits, hashes, versions and seed; scenarios also retain source text and evidence tables. Loading checks a bounded ZIP/JSON/NPZ record without executing the model or unpickling objects. The code below verifies every retained frame, then writes and reloads actual report files in a temporary directory that is cleaned automatically. To keep a report, run `mexico.write_report("my_mexico_report")` with a **new** directory name; existing directories are never overwritten. The source can be adapted without any machine-specific path.
#
# Exact hashes here describe this environment and stored record. Another BLAS/library version can change floating-point results. The rerun checks below establish repeatability here, not universal bitwise reproducibility.

# %%
records = {"mexico": mexico, "household": household, "tariff": tariff}
archives = {}
archive_rows = []
for name, original in records.items():
    payload = save_study(original)
    assert isinstance(payload, bytes) and payload == save_study(original)
    archives[name] = payload
    restored = load_study(payload)
    assert restored.spec == original.spec and restored.manifest == original.manifest
    for attr in ("inputs", "data", "audits"):
        pd.testing.assert_frame_equal(getattr(restored, attr), getattr(original, attr))
    pd.testing.assert_frame_equal(restored.to_frame(), original.to_frame())
    if name == "mexico":
        assert restored.inputs.attrs["source"] == original.inputs.attrs["source"]
    if name != "mexico":
        assert restored.model_source == original.model_source
        assert set(restored.evidence) == set(original.evidence)
        for key in original.evidence:
            pd.testing.assert_frame_equal(restored.evidence[key], original.evidence[key])
    archive_rows.append({"experiment": name, "bytes": len(payload),
                         "archive_sha256": hashlib.sha256(payload).hexdigest()})
print(pd.DataFrame(archive_rows).to_string(index=False))
print("Recorded versions:", json.dumps(mexico.manifest["versions"], sort_keys=True))

with TemporaryDirectory(prefix="puremacro-study-demo-") as temporary:
    report_inventory = {}
    for name, original in records.items():
        target = original.write_report(Path(temporary) / name)
        reread = load_study(target / "study.pmstudy")
        pd.testing.assert_frame_equal(reread.to_frame(), original.to_frame())
        if name == "mexico":
            assert reread.inputs.attrs["source"] == original.inputs.attrs["source"]
        report_inventory[name] = sorted(p.name for p in target.iterdir())
    print(json.dumps(report_inventory, indent=2))

repeat = run_study(data, spec)
pd.testing.assert_frame_equal(repeat.to_frame(), mexico.to_frame(), check_exact=True)
assert save_study(repeat) == archives["mexico"]
print("Same-environment seeded VAR rerun and all archive round-trips passed.")

# %% [markdown]
# We do not merge the three examples into one response comparison: monthly recursive innovations, quarterly transfers and static tariffs have different units, identification and timing. An explicit explanation cannot override an incompatible estimand. LP and VAR also differ here: the LP adapter reports $y_{t+h}-y_{t-1}$, while this VAR reports level IRFs.

# %%
try:
    compare_studies(mexico, household, comparison_note="These answer different questions.")
except ValueError as exc:
    print("Incompatible comparison rejected:", str(exc))
else:
    raise AssertionError("Unlike monthly VAR and quarterly model responses must not compare")

# %% [markdown]
# ## Your turn
#
# Predict how doubling the **same identified innovation** affects the linear VAR response. Change `shock_pp` between 0.10 and 0.50 percentage points per annum. The assertion checks the entire response against linear scaling while retaining the sample, ordering and estimated dynamics. It does not assert that the shock is exogenous in the real economy.
#
# 1. Basic: use 0.10 and 0.50. Explain why no extra factor of 12 belongs in these rate units.
# 2. Intermediate: compare three and six lags above. Which conclusions depend on dynamics, and why do their fitted observation counts differ?
# 3. Stretch: rerun the household model with a finer grid and longer horizon. Compare the first 20 quarters under the same transfer and initial-distribution convention; report changes in economic metrics and audit failures. Do not assert grid convergence from one successful solve.

# %%
shock_pp = 0.50  # ← change this within [0.10, 0.50], annualized percentage points
assert 0.10 <= shock_pp <= 0.50
scaled = run_study(data, replace(spec, name="Shock normalization exercise", shock_size=shock_pp, n_boot=0))
np.testing.assert_allclose(scaled.to_frame().estimate,
                           mexico.to_frame().estimate * (shock_pp / spec.shock_size),
                           rtol=1e-12, atol=1e-13)
print(f"Response scaling verified for a {100 * shock_pp:.0f} basis-point rate innovation.")

# %% [markdown]
# ## How comprehensive is this?
#
# These are complete, bounded applications of existing engines, not an automated empirical-to-structural identification system. `puremacro.study` also supports explicitly exogenous LPs and records custom model scenarios; `puremacro.validation` exposes scoped result audits and explicit refinement ladders. Narrative signal draws describe conditional measurement uncertainty, not additional confidence bands to add to this VAR.
#
# Continue with the [study contract](https://jalonso1979.github.io/puremacro/docs/studies/), [DP file guide](https://jalonso1979.github.io/puremacro/docs/dp_modfiles/), [extended audits](https://jalonso1979.github.io/puremacro/docs/result_audits_extended/) and [trade accounting](https://jalonso1979.github.io/puremacro/docs/trade_accounting/). Packaged versions of these three templates are available through `python -m puremacro.examples.study_workflows --output NEW_DIRECTORY`.
