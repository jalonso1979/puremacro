"""Reproducible offline validation driver; run after the source freeze."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPOSITORY = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
BASE = Path("/tmp/puremacro-sw07-sampling-wheel")
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/work/bin/python"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command, name, *, cwd=BASE, allowed=(0,)):
    print(f"Running {name}", flush=True)
    with (BASE / name).open("w") as log:
        process = subprocess.run(command, cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    shutil.copy2(BASE / name, EVIDENCE / name)
    if process.returncode not in allowed:
        raise RuntimeError(f"{name} exited {process.returncode}; inspect {BASE / name}")
    return process.returncode


BASE.mkdir(exist_ok=True)
offline = BASE / "offline"
offline.mkdir(exist_ok=True)
shutil.copy2(EVIDENCE / "offline_sitecustomize.py", offline / "sitecustomize.py")
ENV = {**os.environ, "PUREMACRO_VALIDATION_BASE": str(BASE),
       "PYTHONPATH": os.pathsep.join((str(offline), str(BASE / "installed"))),
       "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "UV_OFFLINE": "1",
       "UV_CACHE_DIR": str(BASE / "uv-cache"),
       "XDG_CACHE_HOME": str(BASE / "xdg-cache"),
       "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(BASE / "mpl-cache"),
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

mode = sys.argv[1] if len(sys.argv) > 1 else "build"
if mode == "build":
    source = BASE / "source"
    if source.exists():
        raise RuntimeError("Fresh-source validation requires a new source directory")
    source.mkdir()
    for name in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(REPOSITORY / name, source / name)
    shutil.copytree(REPOSITORY / "puremacro", source / "puremacro",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"))
    hashes = {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob("*")) if path.is_file()}
    write_json(BASE / "build_provenance.json", {
        "source": str(REPOSITORY), "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes, "build_python": PYTHON,
        "note": "Fresh copy of final current worktree, including pre-existing changes; offline build; no publication",
    })
    run([PYTHON, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(BASE / "dist"), str(source)], "build.log")
    wheel, = (BASE / "dist").glob("*.whl")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Expected existing uv installer unavailable")
    run([uv, "pip", "install", "--offline", "--no-deps", "--python", PYTHON,
         "--target", str(BASE / "installed"), str(wheel)], "install.log")
    required = ["puremacro/structural/sw07_sampling.py", "puremacro/structural/sw07_finite_sample.py",
                "puremacro/examples/sw07_finite_sample.py", "puremacro/structural/__init__.py",
                "puremacro/dsge/_sw07_data.csv"]
    with zipfile.ZipFile(wheel) as archive:
        for name in required:
            assert name in archive.namelist(), name
            assert hashlib.sha256(archive.read(name)).hexdigest() == hashes[name], name
        assert not any(name.startswith("statsmodels/") for name in archive.namelist())
        write_json(BASE / "wheel_contents.json", {"wheel": str(wheel), "sha256": digest(wheel),
                   "members": len(archive.namelist()), "verified_source_and_resource_members": required})
    for name in ("wheel_smoke.py", "verify_cli.py"):
        shutil.copy2(EVIDENCE / name, BASE / name)
    for name in ("build_provenance.json", "wheel_contents.json"):
        shutil.copy2(BASE / name, EVIDENCE / name)
elif mode == "smoke":
    run([PYTHON, "-u", str(BASE / "wheel_smoke.py")], "installed_smoke.log")
    shutil.copy2(BASE / "installed_smoke.json", EVIDENCE / "installed_smoke.json")
elif mode in ("cli", "cli_resume"):
    if mode == "cli_resume":
        shutil.copy2(BASE / "installed-cli" / "manifest.json", BASE / "cli_initial_manifest.json")
    shutil.copy2(EVIDENCE / "verify_cli.py", BASE / "verify_cli.py")
    result = run([PYTHON, "-u", "-m", "puremacro.examples.sw07_finite_sample",
                  "--output", str(BASE / "installed-cli"), "--scenario", "published_fixed",
                  "--replications", "2", "--seed", "20261001"],
                 "sampling-cli-resume.log" if mode == "cli_resume" else "sampling-cli.log", allowed=(0, 1))
    write_json(BASE / "cli_exit.json", {"returncode": result})
    run([PYTHON, str(BASE / "verify_cli.py")], "cli_verification.log")
    shutil.copy2(BASE / "cli_status.json", EVIDENCE / "cli_status.json")
    if mode == "cli_resume":
        result = json.loads((BASE / "cli_status.json").read_text())
        assert result["resumed_draws"] == {"published_fixed": 2}
elif mode == "docs":
    docs_source = Path(tempfile.mkdtemp(prefix="docs-source-", dir=BASE))
    shutil.copy2(REPOSITORY / "mkdocs.yml", docs_source / "mkdocs.yml")
    shutil.copytree(REPOSITORY / "docs", docs_source / "docs")
    run([PYTHON, "-m", "mkdocs", "build", "--strict", "--site-dir", str(BASE / "site")],
        "mkdocs-strict.log", cwd=docs_source)
    docs = [docs_source / "mkdocs.yml", *sorted((docs_source / "docs").rglob("*"))]
    write_json(BASE / "docs_source_sha256.json", {
        str(path.relative_to(docs_source)): digest(path) for path in docs if path.is_file()})
    write_json(BASE / "docs_validation.json", {
        "passed": True, "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": str(docs_source),
        "note": "Fresh documentation copy after completed-simulation results were added to both language guides; existing wheel and installed smoke preserved",
    })
    shutil.copy2(BASE / "docs_source_sha256.json", EVIDENCE / "docs_source_sha256.json")
    shutil.copy2(BASE / "docs_validation.json", EVIDENCE / "docs_validation.json")
elif mode == "match":
    copied = json.loads((BASE / "build_provenance.json").read_text())["source_sha256"]
    mismatches = [name for name, expected in copied.items()
                  if not (REPOSITORY / name).is_file() or digest(REPOSITORY / name) != expected]
    current = {str(path.relative_to(REPOSITORY)) for path in (REPOSITORY / "puremacro").rglob("*")
               if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
               and path.name != ".DS_Store"} | {"LICENSE", "README.md", "pyproject.toml"}
    additions = sorted(current - copied.keys())
    docs = json.loads((BASE / "docs_source_sha256.json").read_text())
    docs_changed = [name for name, expected in docs.items()
                    if not (REPOSITORY / name).is_file() or digest(REPOSITORY / name) != expected]
    current_docs = {str(path.relative_to(REPOSITORY)) for path in (REPOSITORY / "docs").rglob("*")
                    if path.is_file()} | {"mkdocs.yml"}
    docs_additions = sorted(current_docs - docs.keys())
    result = {"source_file_count": len(copied), "all_source_files_match": not mismatches and not additions,
              "mismatches": mismatches, "new_uncopied_source_files": additions,
              "docs_file_count": len(docs), "all_docs_sources_match": not docs_changed and not docs_additions,
              "docs_changed_after_strict_build": docs_changed,
              "new_uncopied_docs_files": docs_additions}
    write_json(BASE / "source_match.json", result)
    shutil.copy2(BASE / "source_match.json", EVIDENCE / "source_match.json")
    print(json.dumps(result, indent=2))
    assert result["all_source_files_match"] and result["all_docs_sources_match"]
elif mode == "finalize":
    smoke = json.loads((BASE / "installed_smoke.json").read_text())
    cli = json.loads((BASE / "cli_status.json").read_text())
    match = json.loads((BASE / "source_match.json").read_text())
    wheel = json.loads((BASE / "wheel_contents.json").read_text())
    assert smoke["passed"] and cli["passed"]
    assert match["all_source_files_match"] and match["all_docs_sources_match"]
    for name in ("installed-api", "installed-cli", "offline_controls"):
        shutil.copytree(BASE / name, EVIDENCE / name, dirs_exist_ok=True)
    for name in ("stationary_sample.csv", "exact_moments.npz", "cli_initial_manifest.json"):
        shutil.copy2(BASE / name, EVIDENCE / name)
    result = {
        "validation_passed": True, "wheel_sha256": wheel["sha256"],
        "strict_docs_passed": True, "installed_smoke_passed": smoke["passed"],
        "docs_validation": json.loads((BASE / "docs_validation.json").read_text()),
        "installed_cli_passed": cli["passed"], "installed_cli_exit": cli["cli_returncode"],
        "all_copied_sources_match_final_checkout": True,
        "source_file_count": match["source_file_count"], "docs_file_count": match["docs_file_count"],
        "network_blocked_with_positive_control": True, "statsmodels_blocked_with_positive_control": True,
        "scenario": "published_fixed", "replications": 2,
        "usable_fits": cli["usable_fits"], "unresolved_draws": cli["unresolved_draws"],
        "completed_checkpoint_resumed_draws": cli["resumed_draws"],
        "full_npz_and_csv_evidence": True, "api_cli_draws_and_arrays_identical": True,
        "artifacts": str(BASE),
        "scope": "Offline installation and two-draw workflow verification; not Monte Carlo scientific evidence; no package published",
    }
    write_json(BASE / "summary.json", result)
    shutil.copy2(BASE / "summary.json", EVIDENCE / "summary.json")
    print(json.dumps(result, indent=2))
else:
    raise ValueError(f"Unknown stage: {mode}")
