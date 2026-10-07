"""Offline public-API validation from the isolated installation, not the checkout."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

BASE = Path(os.environ["PUREMACRO_VALIDATION_BASE"])
INSTALLED = BASE / "installed"
REPOSITORY = Path("/Users/jalonso/Documents/RESEARCH/puremacro")
assert str(REPOSITORY) not in sys.path
assert (BASE / "offline_controls" / f"{os.getpid()}.json").exists()

import numpy as np
import pandas as pd
import scipy
import puremacro
from puremacro.structural import (
    MomentTargets, SW07FiniteSampleStudy, run_sw07_finite_sample,
    simulate_sw07_sample, sw07_finite_sample_moments,
)

assert Path(puremacro.__file__).is_relative_to(INSTALLED)
exact = sw07_finite_sample_moments({}, nobs=156, common_max_lag=4)
known = sw07_finite_sample_moments({}, nobs=156, common_max_lag=4, demean=False)
assert isinstance(exact, MomentTargets)
assert exact.values.shape == (15,) and exact.covariance.shape == (15, 15)
assert np.isfinite(exact.values).all() and np.isfinite(exact.covariance).all()
np.testing.assert_allclose(exact.covariance, exact.covariance.T, rtol=0, atol=1e-12)
assert np.linalg.eigvalsh(exact.covariance).min() >= -1e-10
assert exact.metadata["effective_observations"] == 152
assert exact.metadata["common_max_lag"] == 4
assert exact.metadata["measurement_error_included"] is False
assert exact.metadata["transition_spectral_radius"] < 1
assert exact.metadata["lyapunov_relative_residual"] < 1e-9
assert not np.allclose(exact.values, known.values)

sample = simulate_sw07_sample({}, nobs=156, seed=20261001)
repeat = simulate_sw07_sample({}, nobs=156, seed=20261001)
different = simulate_sw07_sample({}, nobs=156, seed=20261002)
assert sample.shape == (156, 7) and np.isfinite(sample.to_numpy()).all()
pd.testing.assert_frame_equal(sample, repeat)
assert not np.array_equal(sample.to_numpy(), different.to_numpy())
assert sample.attrs["is_synthetic"] is True
assert sample.attrs["measurement_error_included"] is False
assert sample.attrs["observation_intercepts_included"] is True
assert "stationary" in sample.attrs["initialization"]
assert not np.allclose(sample.iloc[0], list(sample.attrs["population_means"].values()))
sample.to_csv(BASE / "stationary_sample.csv")
np.savez_compressed(BASE / "exact_moments.npz", means=exact.values,
                    covariance=exact.covariance, known_means=known.values,
                    known_covariance=known.covariance)

study = run_sw07_finite_sample(replications=2, scenarios=["published_fixed"],
    progress=lambda scenario, done, total: print(f"{scenario}: {done}/{total}", flush=True))
assert isinstance(study, SW07FiniteSampleStudy)
assert study.draw_moments.shape == (2, 15)
assert study.draw_covariances.shape == (2, 15, 15)
assert len(study.summary) == 1 and len(study.draws) == 2
assert len(study.moment_audit) == 15 and len(study.covariance_audit) == 5 * 225
assert len(study.observed_fits) == 1
assert study.metadata["small_replication_warning"] is True
assert study.metadata["parameter_inference_reported"] is False
assert study.metadata["scenarios"]["published_fixed"]["is_plugin"] is False
assert list(study.draws.array_row) == [0, 1]
usable = ~study.draws.unresolved & np.isfinite(study.draws.objective)
row = study.summary.iloc[0]
failures = int((~usable).sum())
tail = int((usable & (study.draws.objective >= row.observed_objective)).sum())
assert row.replications == 2
assert row.usable_fits == int(usable.sum()) and row.unresolved_draws == failures
assert row.usable_fits + row.unresolved_draws == 2
assert row.tail_exceedances == tail
assert row.tail_fraction_lower == tail / 2
assert row.tail_fraction_upper == (tail + failures) / 2
assert 0 <= row.tail_mc_95_lower <= row.tail_fraction_lower
assert row.tail_fraction_upper <= row.tail_mc_95_upper <= 1
assert not study.draws.loc[study.draws.unresolved, "numerically_regular"].any()
assert study.draws.loc[~study.draws.numerically_regular, "j_pvalue"].isna().all()
assert set(study.draws.status) <= {"failed", "optimization_unresolved", "boundary", "regular", "nonregular"}
evidence = BASE / "installed-api"
evidence.mkdir(exist_ok=True)
for name in ("summary", "draws", "moment_audit", "covariance_audit", "observed_fits"):
    getattr(study, name).to_csv(evidence / f"{name}.csv", index=False)
np.savez_compressed(evidence / "draw_evidence.npz", moments=study.draw_moments,
                    covariances=study.draw_covariances)

assert "statsmodels" not in sys.modules
for name, module in list(sys.modules.items()):
    if name == "puremacro" or name.startswith("puremacro."):
        file = getattr(module, "__file__", None)
        if file:
            assert Path(file).is_relative_to(INSTALLED), (name, file)

result = {
    "passed": True, "package_file": puremacro.__file__,
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__},
    "exact_moment_count": 15, "exact_covariance_shape": list(exact.covariance.shape),
    "stationary_sample_shape": list(sample.shape),
    "stationary_sample_sha256": hashlib.sha256(sample.to_numpy(dtype="<f8").tobytes()).hexdigest(),
    "scenario": "published_fixed", "replications": 2,
    "usable_fits": int(row.usable_fits), "unresolved_draws": failures,
    "statuses": study.draws.status.tolist(),
    "tail_fraction_bounds": [float(row.tail_fraction_lower), float(row.tail_fraction_upper)],
    "network_blocked_with_positive_control": True,
    "statsmodels_blocked_with_positive_control": True,
    "statsmodels_runtime_imported": False,
    "repository_source_absent_from_sys_path": True,
    "scope": "Installation and two-draw workflow verification, not scientific Monte Carlo evidence",
}
(BASE / "installed_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
