"""Validate full exported evidence and conservative failure accounting."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(os.environ["PUREMACRO_VALIDATION_BASE"])
OUT = BASE / "installed-cli"
assert (BASE / "offline_controls" / f"{os.getpid()}.json").exists()
manifest = json.loads((OUT / "manifest.json").read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))
for name, digest in manifest["artifacts"].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
assert manifest["array_shapes"] == {"moments": [2, 15], "covariances": [2, 15, 15]}
assert manifest["small_replication_warning"] is True
assert manifest["parameter_inference_reported"] is False
assert manifest["checkpointing"]["enabled"] is True
tables = {name: pd.read_csv(OUT / f"{name}.csv") for name in
          ("summary", "draws", "moment_audit", "covariance_audit", "observed_fits")}
assert {name: len(frame) for name, frame in tables.items()} == {
    "summary": 1, "draws": 2, "moment_audit": 15, "covariance_audit": 1125, "observed_fits": 1}
summary, draws = tables["summary"].iloc[0], tables["draws"]
usable = ~draws.unresolved & np.isfinite(draws.objective)
unresolved = int((~usable).sum())
assert summary.usable_fits + summary.unresolved_draws == 2
assert summary.usable_fits == int(usable.sum()) and summary.unresolved_draws == unresolved
tail = int((usable & (draws.objective >= summary.observed_objective)).sum())
assert summary.tail_fraction_lower == tail / 2
assert summary.tail_fraction_upper == (tail + unresolved) / 2
assert not draws.loc[draws.unresolved, "numerically_regular"].any()
assert draws.loc[~draws.numerically_regular, "j_pvalue"].isna().all()
status = json.loads((BASE / "cli_exit.json").read_text())
assert status["returncode"] == int(unresolved > 0)
text = (OUT / "report.md").read_text()
assert "Small simulation run" in text
assert ("computational outcomes unresolved" in text) == bool(unresolved)
arrays = np.load(OUT / "draw_evidence.npz", allow_pickle=False)
assert arrays.files == ["moments", "covariances"]
assert arrays["moments"].shape == (2, 15) and arrays["covariances"].shape == (2, 15, 15)
api = np.load(BASE / "installed-api" / "draw_evidence.npz", allow_pickle=False)
for key in arrays.files:
    np.testing.assert_array_equal(arrays[key], api[key])
api_draws = pd.read_csv(BASE / "installed-api" / "draws.csv")
pd.testing.assert_frame_equal(draws, api_draws, check_exact=True)
checkpoint = np.load(OUT / "_checkpoints" / "published_fixed.npz", allow_pickle=False)
checkpoint_document = json.loads(str(checkpoint["document"]))
assert len(checkpoint_document["rows"]) == 2
for key in arrays.files:
    np.testing.assert_array_equal(checkpoint[key], arrays[key])
for value in draws.start_diagnostics_json:
    starts = json.loads(value)
    assert len(starts) == 4
    assert all("start" in record and "success" in record for record in starts)
assert "statsmodels" not in sys.modules
result = {"passed": True, "cli_returncode": status["returncode"],
          "unresolved_draws": unresolved, "usable_fits": int(summary.usable_fits),
          "statuses": draws.status.tolist(), "all_artifact_hashes_valid": True,
          "api_cli_draws_and_arrays_identical": True,
          "all_five_csv_and_full_npz_verified": True,
          "checkpoint_and_four_start_records_verified": True,
          "resumed_draws": manifest["checkpointing"]["resumed_draws"],
          "conservative_failure_bounds_verified": True}
(BASE / "cli_status.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
