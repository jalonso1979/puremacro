"""Authenticate and combine the four completed, independently saved runs.

Run from any directory with this script's absolute path. No estimation is
repeated and no unresolved draw is replaced by a later successful replay.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("baseline_fixed", "published_fixed", "baseline_fitted", "published_fitted")
summaries, moments, manifests, curves = [], [], {}, []
for name in SCENARIOS:
    directory = ROOT/name
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["replications_per_scenario"] == 399
    assert manifest["master_seed"] == 20261001 and manifest["bandwidth"] == 8
    assert list(manifest["scenarios"]) == [name]
    for filename, digest in manifest["artifacts"].items():
        assert hashlib.sha256((directory/filename).read_bytes()).hexdigest() == digest, filename
    summary = pd.read_csv(directory/"summary.csv")
    draws = pd.read_csv(directory/"draws.csv")
    assert len(draws) == 399 and np.array_equal(draws.replication, np.arange(399))
    with np.load(directory/"draw_evidence.npz", allow_pickle=False) as arrays:
        assert arrays["moments"].shape == (399, 15)
        assert arrays["covariances"].shape == (399, 15, 15)
        with np.load(directory/"_checkpoints"/f"{name}.npz", allow_pickle=False) as saved:
            checkpoint = json.loads(str(saved["document"]))
            assert len(checkpoint["rows"]) == 399
            for key in ("moments", "covariances"):
                np.testing.assert_array_equal(arrays[key], saved[key])
    design = checkpoint["design"]
    encoded = json.dumps(design, sort_keys=True, allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == checkpoint["fingerprint"]
    package = ROOT.parents[1]/"puremacro"
    for filename, digest in design["source_hashes"].items():
        assert hashlib.sha256((package/filename).read_bytes()).hexdigest() == digest, filename
    assert draws.unresolved.sum() == summary.unresolved_draws.iloc[0]
    summaries.append(summary)
    moments.append(pd.read_csv(directory/"moment_audit.csv"))
    curves.append(draws)
    manifests[name] = {"manifest_sha256": hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
                       "checkpoint": manifest["checkpointing"], "environment": manifest["environment"],
                       "checkpoint_fingerprint": checkpoint["fingerprint"],
                       "numerical_source_hashes": design["source_hashes"]}

summary = pd.concat(summaries, ignore_index=True)
moment_table = pd.concat(moments, ignore_index=True)
summary.to_csv(ROOT/"combined_summary.csv", index=False)
moment_table.to_csv(ROOT/"combined_moment_audit.csv", index=False)
details = {"replications_total": int(summary.replications.sum()),
           "unresolved_total": int(summary.unresolved_draws.sum()),
           "complete_dossier_hashes_verified": True,
           "checkpoint_arrays_match_final_evidence": True,
           "numerical_sources_match_checkout": True, "source_runs": manifests,
           "limitation": "Conditional Gaussian diagnostics, not composite-null p-values or parameter confidence intervals"}
(ROOT/"combined_run_validation.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for ax, draws, (_, row) in zip(axes.ravel(), curves, summary.iterrows()):
    values = np.sort(draws.loc[~draws.unresolved, "objective"].to_numpy())
    ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=f"Usable simulations: {len(values)}")
    ax.axvline(row.observed_objective, color="#b0472f", label="Observed criterion")
    ax.set(xscale="log", ylim=(0, 1.03), title=row.scenario,
           xlabel="Minimized criterion (log scale)", ylabel="Empirical cumulative fraction")
    ax.legend(fontsize=8)
fig.savefig(ROOT/"criterion_distributions.png", dpi=160)
plt.close(fig)
print(summary[["scenario", "usable_fits", "unresolved_draws", "boundary_fits",
               "criterion_median", "tail_fraction_lower", "tail_fraction_upper",
               "tail_mc_95_lower", "tail_mc_95_upper", "naive_chi2_rate_lower",
               "naive_chi2_rate_upper"]].to_string(index=False))
