"""Build and validate an unpublished candidate from a frozen working tree.

Stages are explicit so the full release gate can run alongside installed-wheel
checks. The package version stays 4.3.0; a unique directory and source hashes
identify this candidate. No Git state or publication endpoint is modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
PYTHON = "/tmp/puremacro-research-candidate-venv/bin/python"
POINTER = EVIDENCE / "candidate-directory.txt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def files_under(base, directories):
    for directory in directories:
        for path in sorted((base / directory).rglob("*")):
            if (path.is_file() and not {"__pycache__", ".ipynb_checkpoints", "node_modules", "build"}.intersection(path.parts)
                    and path.suffix not in (".pyc", ".pyo") and path.name != ".DS_Store"):
                yield path


def env(base, *, installed=False):
    paths = (base / "offline", base / "installed") if installed else (ROOT,)
    return {
        **os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths)),
        "PUREMACRO_VALIDATION_BASE": str(base), "PUREMACRO_REPOSITORY": str(ROOT),
        "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "UV_OFFLINE": "1",
        "UV_CACHE_DIR": str(base / "uv-cache"), "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(base / "mpl-cache"), "XDG_CACHE_HOME": str(base / "cache"),
        "JUPYTER_DATA_DIR": str(base / "jupyter"), "IPYTHONDIR": str(base / "ipython"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }


def run(base, command, name, *, cwd=None, installed=False, allowed=(0,)):
    start = datetime.now(timezone.utc)
    print(f"Running {name}", flush=True)
    with (EVIDENCE / name).open("w") as stream:
        proc = subprocess.run(command, cwd=cwd or base, env=env(base, installed=installed),
                              stdout=stream, stderr=subprocess.STDOUT)
    write_json(EVIDENCE / (name + ".json"), {
        "command": command, "cwd": str(cwd or base), "returncode": proc.returncode,
        "started_at_utc": start.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if proc.returncode not in allowed:
        raise RuntimeError(f"{name} exited {proc.returncode}; see the evidence log")
    return proc.returncode


def freeze():
    if POINTER.exists():
        raise RuntimeError("Candidate already frozen; retain its evidence and use a new run directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path("/tmp") / ("puremacro-research-candidate-" + stamp)
    base.mkdir()
    POINTER.write_text(str(base) + "\n")
    source = base / "source"
    source.mkdir()
    for name in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "puremacro", source / "puremacro",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"))
    sources = {str(path.relative_to(source)): digest(path)
               for path in sorted(source.rglob("*")) if path.is_file()}
    gate_files = list(files_under(ROOT, ("puremacro", "tools", "tests", "docs", "notebooks", "curso")))
    gate_files.extend(ROOT / name for name in ("pyproject.toml", "README.md", "README.es.md",
        "ARCHITECTURE.md", "CHANGELOG.md", "CITATION.cff", "mkdocs.yml", "playground/jupyter_lite_config.json"))
    provenance = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_directory": str(base),
        "package_version": "4.3.0", "candidate_kind": "unpublished current-worktree research candidate",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources,
        "gate_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in gate_files},
        "note": "Includes earlier worktree changes; not a feature-only diff or clean committed release.",
    }
    write_json(EVIDENCE / "candidate-provenance.json", provenance)
    (EVIDENCE / "frozen-git-status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    run(base, [shutil.which("uv"), "pip", "freeze", "--python", PYTHON], "environment.log")
    print(str(base), flush=True)


def main():
    mode = sys.argv[1]
    if mode == "freeze":
        freeze()
        return
    base = Path(POINTER.read_text().strip())
    if mode == "build":
        run(base, [PYTHON, "-m", "build", "--no-isolation", "--wheel", "--sdist",
                   "--outdir", str(base / "dist"), str(base / "source")], "build.log")
        distributions = sorted((base / "dist").iterdir())
        run(base, [PYTHON, "-m", "twine", "check", *map(str, distributions)], "twine-check.log")
        wheel, = (base / "dist").glob("*.whl")
        run(base, [shutil.which("uv"), "pip", "install", "--offline", "--no-deps",
                   "--python", PYTHON, "--target", str(base / "installed"), str(wheel)], "install.log")
        sources = json.loads((EVIDENCE / "candidate-provenance.json").read_text())["source_sha256"]
        with zipfile.ZipFile(wheel) as archive:
            members = set(archive.namelist())
            shipped = {name: digest for name, digest in sources.items() if name.startswith("puremacro/")
                       and name in members}
            for name, expected in shipped.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
            omitted_modules = [name for name in sources if name.startswith("puremacro/")
                               and name.endswith(".py") and name not in members]
            assert not omitted_modules, omitted_modules
            required_resources = ["puremacro/dsge/_sw07_data.csv", "puremacro/datasets/data/enigh2024_deciles.csv",
                "puremacro/datasets/data/enigh2024_deciles_metadata.json", "puremacro/replication/data/rr2010_original.csv",
                "puremacro/replication/data/rr2010_original_metadata.json", "puremacro/replication/data/rr2010_statsmodels_reference.json"]
            assert set(required_resources) <= members
        write_json(EVIDENCE / "artifacts.json", {
            "artifacts": [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                          for path in distributions],
            "wheel_members": len(members), "matched_shipped_source_files": len(shipped),
            "all_package_python_modules_shipped": True, "required_resources": required_resources,
        })
    elif mode == "release-gates":
        run(base, [PYTHON, "-u", "tools/release_check.py"], "release-gate.log", cwd=ROOT, allowed=(0, 1))
    elif mode == "release-gates-amended":
        assert (EVIDENCE / "amended-candidate-provenance.json").exists()
        run(base, [PYTHON, "-u", "tools/release_check.py"], "release-gate-after-notebook-repairs.log",
            cwd=ROOT, allowed=(0, 1))
    elif mode == "release-gates-third":
        assert (EVIDENCE / "third-candidate-provenance.json").exists()
        assert not (EVIDENCE / "release-gate-after-exact-parity.log").exists()
        run(base, [PYTHON, "-u", "tools/release_check.py"], "release-gate-after-exact-parity.log",
            cwd=ROOT, allowed=(0, 1))
    elif mode == "preserve-original":
        for source, target in (("summary.json", "original-validation-summary.json"),
                               ("final-source-match.json", "original-final-source-match.json")):
            if (EVIDENCE / target).exists():
                raise RuntimeError(f"Refusing to overwrite preserved original evidence: {target}")
            shutil.copy2(EVIDENCE / source, EVIDENCE / target)
    elif mode == "freeze-amended":
        original = json.loads((EVIDENCE / "candidate-provenance.json").read_text())
        allowed_notebooks = {
            f"notebooks/{stem}{suffix}"
            for stem in ("00_whats_new_in_puremacro_3_0_es", "10_staggered_did", "10_staggered_did_es",
                         "45_dsge_discretion_dsge_var_and_news_shocks", "45_dsge_discretion_dsge_var_and_news_shocks_es")
            for suffix in (".py", ".ipynb")
        }
        display_manifest = json.loads((EVIDENCE / "notebook-inline-display-repair.json").read_text())
        notebook_manifest_path = EVIDENCE / "notebook-repair/final_manifest.json"
        notebook_manifest = json.loads(notebook_manifest_path.read_text())
        for name, expected in notebook_manifest["files"].items():
            assert name in allowed_notebooks and digest(ROOT / name) == expected, name
        for repair in display_manifest["files"]:
            assert repair["non_display_ast_unchanged"] and repair["existing_outputs_unchanged"]
            for kind in ("source", "rendered"):
                name = repair[kind]
                assert repair[kind + "_before_sha256"] == original["gate_source_sha256"][name]
                assert digest(ROOT / name) == repair[kind + "_after_sha256"]
                allowed_notebooks.add(name)
        new_hashes = {name: digest(ROOT / name) for name in original["gate_source_sha256"]}
        changed = {name: {"before": expected, "after": new_hashes[name]}
                   for name, expected in original["gate_source_sha256"].items() if new_hashes[name] != expected}
        notebook_changes = {name: value for name, value in changed.items() if name in allowed_notebooks}
        unexpected = [name for name in changed if name not in allowed_notebooks
                      and not name.startswith("docs/") and name != "mkdocs.yml"
                      and not (name.startswith("puremacro/examples/output/") and name.endswith(".png"))]
        assert notebook_changes and not unexpected, unexpected
        wheel, = (base / "dist").glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            shipped = {name for name in original["source_sha256"] if name.startswith("puremacro/")
                       and name in archive.namelist()}
        assert all(digest(ROOT / name) == original["source_sha256"][name] for name in shipped)
        amended = {**original, "amended_at_utc": datetime.now(timezone.utc).isoformat(),
            "amended_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "notebook_repair_manifest_sha256": digest(notebook_manifest_path),
            "inline_display_manifest_sha256": digest(EVIDENCE / "notebook-inline-display-repair.json"),
            "gate_source_sha256": new_hashes, "notebook_repairs": notebook_changes,
            "all_changes_since_original_freeze": changed,
            "all_878_shipped_package_files_match_original": len(shipped) == 878,
            "package_artifacts_rebuilt": False,
            "scope": "Notebook-only repairs, final documentation and generated unshipped example figures; original full-suite failure evidence preserved.",
        }
        target = EVIDENCE / "amended-candidate-provenance.json"
        assert not target.exists(), "Amended candidate has already been frozen"
        write_json(target, amended)
        shutil.copy2(target, EVIDENCE / "artifacts" / base.name / target.name)
        print(json.dumps({"notebook_repairs": list(notebook_changes), "unchanged_shipped_files": len(shipped)}, indent=2))
    elif mode == "freeze-third":
        previous = json.loads((EVIDENCE / "amended-candidate-provenance.json").read_text())
        original = json.loads((EVIDENCE / "candidate-provenance.json").read_text())
        manifest_path = EVIDENCE / "notebook-exact-source-parity-repair.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["numerical_code_outputs_and_metadata_unchanged"]
        allowed = set()
        for repair in manifest["files"]:
            assert repair["only_trailing_newlines_changed"] and repair["all_other_fields_unchanged"]
            source, rendered = repair["source"], repair["rendered"]
            assert rendered.startswith("notebooks/") and rendered.endswith(".ipynb")
            assert digest(ROOT / source) == repair["source_sha256"] == previous["gate_source_sha256"][source]
            assert previous["gate_source_sha256"][rendered] == repair["rendered_before_sha256"]
            assert digest(ROOT / rendered) == repair["rendered_after_sha256"]
            allowed.add(rendered)
        hashes = {name: digest(ROOT / name) for name in previous["gate_source_sha256"]}
        changes = {name: {"before": expected, "after": hashes[name]}
                   for name, expected in previous["gate_source_sha256"].items() if hashes[name] != expected}
        assert set(changes) == allowed and len(allowed) == 30
        for record_name in ("original-evidence-sha256.json", "second-evidence-sha256.json"):
            evidence = json.loads((EVIDENCE / record_name).read_text())
            assert all(digest(EVIDENCE / name) == expected for name, expected in evidence.items())
        wheel, = (base / "dist").glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            shipped = {name for name in original["source_sha256"] if name.startswith("puremacro/")
                       and name in archive.namelist()}
        assert len(shipped) == 878
        assert all(digest(ROOT / name) == original["source_sha256"][name] for name in shipped)
        third = {**previous, "third_freeze_at_utc": datetime.now(timezone.utc).isoformat(),
            "third_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gate_source_sha256": hashes, "changes_since_second_freeze": changes,
            "exact_parity_repair_manifest_sha256": digest(manifest_path),
            "all_878_shipped_package_files_match_original": True,
            "package_artifacts_rebuilt": False,
            "scope": "Third validation after exact terminal-newline parity repairs to 30 rendered notebooks; both failed full-suite records preserved.",
        }
        third["notebook_repairs"] = {name: {"before": original["gate_source_sha256"][name], "after": hashes[name]}
                                    for name in previous["notebook_repairs"]}
        third["all_changes_since_original_freeze"] = {
            name: {"before": expected, "after": hashes[name]}
            for name, expected in original["gate_source_sha256"].items() if hashes[name] != expected}
        target = EVIDENCE / "third-candidate-provenance.json"
        assert not target.exists(), "Third candidate has already been frozen"
        write_json(target, third)
        shutil.copy2(target, EVIDENCE / "artifacts" / base.name / target.name)
        print(json.dumps({"exact_parity_repairs": len(changes), "unchanged_shipped_files": len(shipped)}, indent=2))
    elif mode == "retain-artifacts":
        retained = EVIDENCE / "artifacts" / base.name
        retained.mkdir(parents=True, exist_ok=True)
        record = json.loads((EVIDENCE / "artifacts.json").read_text())
        for item in record["artifacts"]:
            source = Path(item["path"])
            target = retained / source.name
            shutil.copy2(source, target)
            assert digest(target) == item["sha256"]
            item["retained_path"] = str(target.relative_to(ROOT))
        shutil.copy2(EVIDENCE / "candidate-provenance.json", retained / "candidate-provenance.json")
        write_json(EVIDENCE / "artifacts.json", record)
    elif mode == "audit-artifacts":
        sources = json.loads((EVIDENCE / "candidate-provenance.json").read_text())["source_sha256"]
        record = json.loads((EVIDENCE / "artifacts.json").read_text())
        wheel, = (base / "dist").glob("*.whl")
        sdist, = (base / "dist").glob("*.tar.gz")
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = set(archive.namelist())
            assert not any(name.endswith((".png", ".pyc", ".pyo", ".so", ".mat")) for name in wheel_names)
        with tarfile.open(sdist, "r:gz") as archive:
            members = {member.name.partition("/")[2]: member for member in archive.getmembers() if member.isfile()}
            assert {name for name in members if name.startswith("puremacro/")} == {
                name for name in wheel_names if name.startswith("puremacro/")}
            assert not any(name.startswith(("notebooks/", "docs/", "curso/")) or name.endswith(".png")
                           for name in members)
            expected_names = {name for name in sources if name.startswith("puremacro/") and name in wheel_names}
            expected_names |= set(record["required_resources"])
            assert expected_names <= members.keys()
            for name in expected_names:
                assert hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest() == sources[name], name
        record["wheel_contains_no_png_bytecode_shared_libraries_or_mat_files"] = True
        record.pop("sdist_authenticated_python_and_required_resource_files", None)
        record["sdist_authenticated_shipped_source_and_resource_files"] = len(expected_names)
        record["sdist_contains_all_package_python_modules_and_required_resources"] = True
        record["sdist_package_files_identical_to_wheel"] = True
        record["sdist_contains_no_notebooks_docs_curso_or_png_files"] = True
        write_json(EVIDENCE / "artifacts.json", record)
    elif mode == "docs":
        snapshot = Path(tempfile.mkdtemp(prefix="docs-source-", dir=base))
        shutil.copy2(ROOT / "mkdocs.yml", snapshot / "mkdocs.yml")
        shutil.copytree(ROOT / "docs", snapshot / "docs")
        run(base, [PYTHON, "-m", "mkdocs", "build", "--strict", "--site-dir", str(base / "site")],
            "mkdocs-strict.log", cwd=snapshot)
        write_json(EVIDENCE / "docs-provenance.json", {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(snapshot),
            "sha256": {str(path.relative_to(snapshot)): digest(path)
                       for path in snapshot.rglob("*") if path.is_file()},
        })
    elif mode == "smoke":
        offline = base / "offline"
        offline.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "reviews/2026-10-01-sw07-finite-sample/wheel-validation/offline_sitecustomize.py",
                     offline / "sitecustomize.py")
        shutil.copy2(EVIDENCE / "installed_smoke.py", base / "installed_smoke.py")
        run(base, [PYTHON, "-u", str(base / "installed_smoke.py")], "installed-smoke.log", installed=True)
        shutil.copy2(base / "installed-smoke.json", EVIDENCE / "installed-smoke.json")
        shutil.copytree(base / "installed-evidence", EVIDENCE / "installed-evidence", dirs_exist_ok=True)
    elif mode in ("match", "match-amended", "match-third"):
        third = mode == "match-third"
        amended = mode != "match"
        provenance_name = "third-candidate-provenance.json" if third else (
            "amended-candidate-provenance.json" if amended else "candidate-provenance.json")
        frozen = json.loads((EVIDENCE / provenance_name).read_text())
        changed = [name for name, expected in frozen["gate_source_sha256"].items()
                   if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
        current = {str(path.relative_to(ROOT)) for path in files_under(ROOT,
            ("puremacro", "tools", "tests", "docs", "notebooks", "curso"))}
        added = sorted(current - frozen["gate_source_sha256"].keys())
        docs_only = lambda name: name.startswith("docs/") or name == "mkdocs.yml"
        generated_output = lambda name: name.startswith("puremacro/examples/output/") and Path(name).suffix == ".png"
        generated_changes = [name for name in changed + added if generated_output(name)]
        if generated_changes:
            wheel, = (base / "dist").glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                assert not set(generated_changes) & set(archive.namelist())
        changed_code = [name for name in changed if not docs_only(name) and not generated_output(name)]
        added_code = [name for name in added if not docs_only(name) and not generated_output(name)]
        docs = json.loads((EVIDENCE / "docs-provenance.json").read_text())["sha256"]
        docs_changed = [name for name, expected in docs.items()
                        if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
        docs_added = sorted({str(path.relative_to(ROOT)) for path in (ROOT / "docs").rglob("*")
                            if path.is_file()} - docs.keys())
        match_name = "final-source-match-third.json" if third else (
            "final-source-match-amended.json" if amended else "final-source-match.json")
        write_json(EVIDENCE / match_name, {
            "all_frozen_gate_sources_match": not changed and not added,
            "changed_since_freeze": changed, "added_since_freeze": added,
            "all_frozen_package_test_tool_sources_match": not changed_code and not added_code,
            "changed_non_documentation_sources": changed_code, "added_non_documentation_sources": added_code,
            "final_strict_docs_match": not docs_changed and not docs_added,
            "docs_changed_after_strict_build": docs_changed, "docs_added_after_strict_build": docs_added,
            "generated_unshipped_example_output_changes": {
                name: {"frozen_sha256": frozen["gate_source_sha256"].get(name),
                       "current_sha256": digest(ROOT / name), "in_wheel": False}
                for name in generated_changes
            },
            "frozen_file_count": len(frozen["gate_source_sha256"]),
            "note": "Post-freeze prose changes are listed and separately authenticated against the final strict documentation build. Test-regenerated PNGs in examples/output are recorded separately and verified absent from the wheel.",
        })
        print(json.dumps({"changed": changed, "added": added}, indent=2))
        assert not changed_code and not added_code and not docs_changed and not docs_added
    elif mode in ("finalize", "finalize-amended", "finalize-third"):
        third = mode == "finalize-third"
        amended = mode != "finalize"
        gate_name = "release-gate-after-exact-parity.log" if third else (
            "release-gate-after-notebook-repairs.log" if amended else "release-gate.log")
        gate = json.loads((EVIDENCE / (gate_name + ".json")).read_text())
        log = (EVIDENCE / gate_name).read_text()
        match_name = "final-source-match-third.json" if third else (
            "final-source-match-amended.json" if amended else "final-source-match.json")
        match = json.loads((EVIDENCE / match_name).read_text())
        smoke = json.loads((EVIDENCE / "installed-smoke.json").read_text())
        docs = json.loads((EVIDENCE / "mkdocs-strict.log.json").read_text())
        twine = json.loads((EVIDENCE / "twine-check.log.json").read_text())
        artifacts = json.loads((EVIDENCE / "artifacts.json").read_text())
        frozen = json.loads((EVIDENCE / "candidate-provenance.json").read_text())
        original_evidence = json.loads((EVIDENCE / "original-evidence-sha256.json").read_text()) if amended else {}
        assert all(digest(EVIDENCE / name) == expected for name, expected in original_evidence.items())
        provenance_names = ["candidate-provenance.json", "docs-provenance.json", match_name]
        if amended:
            provenance_names.extend(["amended-candidate-provenance.json", "original-evidence-sha256.json"])
        if third:
            second_evidence = json.loads((EVIDENCE / "second-evidence-sha256.json").read_text())
            assert all(digest(EVIDENCE / name) == expected for name, expected in second_evidence.items())
            provenance_names.extend(["third-candidate-provenance.json", "second-evidence-sha256.json"])
        for item in artifacts["artifacts"]:
            assert digest(ROOT / item["retained_path"]) == item["sha256"]
        full_gate_passed = gate["returncode"] == 0 and "all 5 gates PASS" in log
        source_match = match["all_frozen_package_test_tool_sources_match"]
        docs_match = match["final_strict_docs_match"]
        summary_line = next((line.strip() for line in log.splitlines() if "pytest:" in line), None)
        counts = {} if summary_line is None else {
            key: int(value) for value, key in re.findall(r"(\d+) (passed|failed|skipped|deselected|xfailed|xpassed|errors?)", summary_line)
        }
        new_nodes = re.findall(r"^\s+NEW: (.+)$", log, flags=re.MULTILINE)
        recovered = re.findall(r"^\s+recovered: (.+)$", log, flags=re.MULTILINE)
        whitelist = json.loads((ROOT / "tests/known_failures.json").read_text())
        baseline_compared = summary_line is not None and (
            "known red" in log or "previously-red now green" in log or "new failure(s)" in log)
        remaining_known = (sorted({entry["nodeid"] for entry in whitelist["entries"]} - set(recovered))
                           if baseline_compared else None)
        summary = {
            "candidate_kind": "unpublished current-worktree research candidate",
            "validation_stage": "after exact rendered-source parity repair" if third else (
                "after scoped notebook repairs" if amended else "original frozen tree"),
            "release_gate_log": gate_name,
            "original_validation_evidence": "original-validation-summary.json" if amended else None,
            "original_gate_evidence_unchanged": True if amended else None,
            "second_gate_evidence_unchanged": True if third else None,
            "second_validation_evidence": "second-validation-summary.json" if third else None,
            "provenance_sha256": {name: digest(EVIDENCE / name) for name in provenance_names},
            "package_version": "4.3.0", "candidate_directory": str(base),
            "software_validation_passed": bool(full_gate_passed and source_match and docs_match
                and smoke["passed"] and docs["returncode"] == 0 and twine["returncode"] == 0),
            "full_release_gate_passed": full_gate_passed,
            "full_suite_summary": summary_line, "full_suite_counts": counts,
            "known_failure_baseline_compared": baseline_compared,
            "unexpected_failure_nodeids": new_nodes,
            "remaining_known_failure_nodeids": remaining_known,
            "recovered_known_failure_nodeids": recovered,
            "gate_reports": [line.strip() for line in log.splitlines() if re.match(r"\s+Gate \d", line)],
            "all_frozen_package_test_tool_sources_match": source_match,
            "final_strict_docs_match": docs_match,
            "installed_wheel_validation_passed": smoke["passed"],
            "strict_docs_passed": docs["returncode"] == 0,
            "wheel_and_sdist_twine_passed": twine["returncode"] == 0,
            "research_benchmarks_passed": 7, "research_benchmarks_total": 8,
            "research_benchmarks_overall_passed": False,
            "documented_scientific_discrepancy": "RR2010 printed horizon-10 t-statistic misses the unchanged rounding tolerance",
            "known_failure_whitelist_unchanged": digest(ROOT / "tests/known_failures.json")
                == frozen["gate_source_sha256"]["tests/known_failures.json"],
            "api_legacy_entries_reconciled": 9,
            "artifacts": artifacts["artifacts"],
            "release_commit_created_by_validation": False,
            "tagged": False, "published": False, "deployed": False,
            "original_frozen_git_head": frozen["git_head"],
            "current_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "limitations": ["Includes earlier worktree changes; not a feature-only release",
                "Fixed-DGP/oracle SW07 controls do not establish composite-null inference",
                "Synthetic trade technology/sourcing and assumed factor exposure are not an estimated Mexican economy",
                "Tiny installed simulations validate packaging, not scientific performance"],
        }
        write_json(EVIDENCE / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
