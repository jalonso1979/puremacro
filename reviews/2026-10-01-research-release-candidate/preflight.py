"""Read-only preflight evidence for the current-worktree research candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
ENV = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp/puremacro-candidate-mpl",
    "XDG_CACHE_HOME": "/tmp/puremacro-candidate-cache",
    "JUPYTER_DATA_DIR": "/tmp/puremacro-candidate-jupyter",
    "IPYTHONDIR": "/tmp/puremacro-candidate-ipython",
    "PYTHONPATH": str(ROOT),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(ENV)


def write_json(name, data):
    (EVIDENCE / name).write_text(json.dumps(data, indent=2) + "\n")


def run_logged(command, name):
    with (EVIDENCE / name).open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=stream,
                                stderr=subprocess.STDOUT)
    write_json(name + ".json", {
        "command": command, "returncode": result.returncode,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "preflight; current checkout remains under active development",
    })
    print(f"{name}: exit {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inventory"
    if mode == "inventory":
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        (EVIDENCE / "initial-git-status.txt").write_text(status)
        sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
        from test_public_api import collect_current_api
        current = collect_current_api()
        fixture = ROOT / "tests/fixtures/public_api_snapshot.json"
        expected = json.loads(fixture.read_text())
        changes = {}
        for section in current:
            changed = {}
            for key in sorted(current[section].keys() | expected[section].keys()):
                before = expected[section].get(key, [])
                after = current[section].get(key, [])
                if before == after:
                    continue
                detail = {"added": sorted(set(after) - set(before)),
                          "removed": sorted(set(before) - set(after))}
                if section == "result_classes" and key in current[section]:
                    module, _, attr = key.rpartition(".")
                    cls = getattr(importlib.import_module(module), attr)
                    detail["added_fields_have_defaults"] = all(
                        field.default is not dataclasses.MISSING
                        or field.default_factory is not dataclasses.MISSING
                        for field in dataclasses.fields(cls)
                        if field.name in detail["added"]
                    )
                changed[key] = detail
            changes[section] = changed
        write_json("initial-inventory.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_entries": len(status.splitlines()),
            "api_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "api_differences": changes,
            "note": "Read-only inventory; no API snapshot, source, version, Git index, or known-failure whitelist changes.",
        })
        source_files = [*sorted((ROOT / "puremacro").rglob("*")),
                        ROOT / "pyproject.toml", ROOT / "README.md"]
        write_json("preflight-source-sha256.json", {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Active-worktree preflight inventory; final candidate uses a separate frozen manifest.",
            "sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files
                if path.is_file() and "__pycache__" not in path.parts
                and path.suffix not in (".pyc", ".pyo") and path.name != ".DS_Store"
            },
        })
        print(json.dumps(changes, indent=2))
    elif mode == "gates":
        sys.exit(run_logged([sys.executable, "tools/release_check.py", "--no-tests"],
                            "preflight-gates.log"))
    elif mode == "api-regressions":
        targets = [
            "tests/test_fix_I1_dsge_core.py",
            "tests/test_fix_STEADY_structural_singularity.py",
            "tests/test_fix_AKM_shift_share_projection.py",
            "tests/test_did/test_fix_did_aggregation.py",
            "tests/test_did/test_fix_SDID_inference_and_weights.py",
            "tests/test_dynpanel/test_fix_dynpanel_windmeijer.py",
            "tests/test_dynpanel/test_fix_dynpanel_instruments.py",
            "tests/test_fix_NOWCAST_plots_and_revision_stats.py",
            "tests/test_fix_RTNOW_realtime_asof.py",
            "tests/test_fix_MSAR_ms_var_em.py",
            "tests/test_release_check.py",
            "tests/test_release_check_34fixes.py",
        ]
        sys.exit(run_logged([sys.executable, "-m", "pytest", *targets, "-q", "--tb=short"],
                            "api-regressions.log"))
    else:
        raise ValueError(mode)
