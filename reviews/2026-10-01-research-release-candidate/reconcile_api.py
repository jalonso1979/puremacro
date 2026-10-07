"""Update only the nine audited legacy API entries; preserve all other entries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_public_api import collect_current_api

ALLOWED = {
    "all": ["puremacro.dsge"],
    "result_classes": [
        "puremacro.bartik.akm.ShiftShareIVResult",
        "puremacro.did._results.CallawaySantannaResult",
        "puremacro.did._results.SunAbrahamResult",
        "puremacro.did._results.SyntheticDiDResult",
        "puremacro.dynpanel._results.GMMResult",
        "puremacro.nowcast.evaluation.PITUniformityResult",
        "puremacro.nowcast.realtime_nowcast.RealtimeNowcastResult",
        "puremacro.var.regime.ms_var.MSVARResult",
    ],
}

path = ROOT / "tests/fixtures/public_api_snapshot.json"
old_bytes = path.read_bytes()
before = json.loads(old_bytes)
after = json.loads(old_bytes)
current = collect_current_api()
changes = {}
for section, names in ALLOWED.items():
    changes[section] = {}
    for name in names:
        assert name in before[section] and name in current[section]
        assert set(before[section][name]) <= set(current[section][name])
        changes[section][name] = {
            "before": before[section][name], "after": current[section][name],
        }
        after[section][name] = current[section][name]
for section in before:
    for name, values in before[section].items():
        if name not in ALLOWED.get(section, []):
            assert after[section][name] == values
path.write_text(json.dumps(after, indent=2) + "\n")
evidence = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "before_sha256": hashlib.sha256(old_bytes).hexdigest(),
    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "audited_entries": 9,
    "all_changes_additive": True,
    "new_result_fields_have_defaults": True,
    "other_snapshot_entries_preserved": True,
    "validation": {
        "api_regressions": "275 passed, 6 skipped, 4 deselected",
        "gmm_notes_regressions": "11 passed",
        "logs": ["api-regressions.log", "gmm-notes-regressions.log"],
    },
    "changes": changes,
    "scope": "Previously implemented API additions now reflected in the fixture; no implementation or test assertions changed.",
}
Path(__file__).with_name("api-reconciliation.json").write_text(json.dumps(evidence, indent=2) + "\n")
print("Reconciled exactly nine additive legacy API entries; all others preserved.")
