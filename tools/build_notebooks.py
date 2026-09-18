"""Build and verify puremacro showcase and course notebooks.

Source of truth:
- Showcase suite: ``notebooks/NN_topic.py`` (percent format). Converted + executed
  to ``notebooks/NN_topic.ipynb`` (committed, WITH outputs).
- Course suite: ``curso/notebooks/TNN_*.ipynb`` (executed against ``data_curso/``).

Usage:
    python tools/build_notebooks.py                      # build all suites
    python tools/build_notebooks.py --suite showcase    # build showcase suite only
    python tools/build_notebooks.py --suite curso       # build course suite only
    python tools/build_notebooks.py 01_wealth_inequality # build one (by stem or filename)
    python tools/build_notebooks.py --list               # list discovered sources
    python tools/build_notebooks.py --check              # execute to temp, validate outputs, fail on error
    python tools/build_notebooks.py --validate-only      # validate existing .ipynb artifacts without running

Requires the ``notebooks`` extra: pip install -e ".[notebooks]"
Kernel cwd is resolved per-suite (``notebooks/`` or ``curso/notebooks/``)
so notebooks can ``import _nbstyle`` directly.
Execution is strictly headless via ``MPLBACKEND=Agg``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure headless execution backend
os.environ["MPLBACKEND"] = "Agg"

PROJ_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJ_ROOT / "notebooks"
CURSO_DIR = PROJ_ROOT / "curso" / "notebooks"
KERNEL_NAME = "python3"


def discover_sources(suite: str = "showcase") -> list[Path]:
    """Sorted notebook sources for the requested suite.

    Suites:
    - 'showcase': ``notebooks/*.py`` and ``notebooks/course/*.py`` (excluding ``_*.py``)
    - 'curso': ``curso/notebooks/*.ipynb`` (excluding ``_*`` and checkpoints)
    - 'all': both showcase and curso sources in deterministic order
    """
    suite = suite.lower()
    top = [p for p in NB_DIR.glob("*.py") if not p.name.startswith("_")]
    course = [p for p in (NB_DIR / "course").glob("*.py") if not p.name.startswith("_")]
    showcase_sources = sorted(top + course)

    curso_sources = (
        sorted(
            p for p in CURSO_DIR.glob("*.ipynb")
            if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
        )
        if CURSO_DIR.exists()
        else []
    )

    if suite == "showcase":
        return showcase_sources
    elif suite == "curso":
        return curso_sources
    elif suite == "all":
        return showcase_sources + curso_sources
    else:
        raise ValueError(f"unknown suite {suite!r}; choose from 'showcase', 'curso', 'all'")


def ensure_kernel() -> None:
    """Register a ``python3`` Jupyter kernelspec for the current interpreter.

    jupytext --execute needs a registered kernelspec; a bare ``ipykernel``
    install does not create one. Idempotent (re-registering overwrites). Points
    the standard ``python3`` kernel at ``sys.executable`` so the notebooks build
    with the same Python that has puremacro installed.
    """
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME],
        capture_output=True,
    )


def _jupytext(args: list[str], *, cwd: Path | None = None, env: dict | None = None) -> int:
    cmd = [sys.executable, "-m", "jupytext", *args]
    print("›", " ".join(cmd))
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "Agg"
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=cwd or NB_DIR, env=run_env).returncode


def validate_notebook_node(nb_path: Path) -> list[str]:
    """Validate notebook artifact for structural and visual integrity.

    Checks:
    1. Valid nbformat v4 JSON structure.
    2. Active cell execution counts (no None on code cells with executable statements).
    3. Zero cell error tracebacks (output_type != 'error').
    4. 100% opaque PNG alpha channels (min_alpha == 255).
    """
    errors: list[str] = []
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return [f"{nb_path}: invalid JSON: {e}"]

    if not isinstance(nb, dict) or "cells" not in nb:
        return [f"{nb_path}: missing 'cells' array"]
    if nb.get("nbformat", 0) < 4:
        return [f"{nb_path}: nbformat {nb.get('nbformat')} < 4"]

    try:
        from PIL import Image
        have_pil = True
    except ImportError:
        have_pil = False
        import numpy as np
        import matplotlib.image as mpimg

    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            has_code = any(
                line.strip() and not line.strip().startswith("#")
                for line in source.splitlines()
            )
            if has_code and cell.get("execution_count") is None:
                errors.append(f"{nb_path}: cell {idx} has unexecuted code (execution_count is None)")

            for out_idx, out in enumerate(cell.get("outputs", [])):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")
                    errors.append(f"{nb_path}: cell {idx} output {out_idx} error: {ename}: {evalue}")

                data = out.get("data", {})
                if "image/png" in data:
                    raw = data["image/png"]
                    if isinstance(raw, list):
                        raw = "".join(raw)
                    try:
                        im_bytes = base64.b64decode(raw)
                        if have_pil:
                            im = Image.open(io.BytesIO(im_bytes))
                            if im.mode == "RGBA":
                                import numpy as np
                                arr = np.array(im)
                                min_alpha = int(arr[:, :, 3].min())
                                if min_alpha < 255:
                                    errors.append(
                                        f"{nb_path}: cell {idx} output {out_idx} has transparent canvas "
                                        f"(min_alpha={min_alpha} < 255)"
                                    )
                        else:
                            arr = mpimg.imread(io.BytesIO(im_bytes), format="png")
                            if arr.ndim == 3 and arr.shape[2] == 4:
                                min_val = arr[:, :, 3].min()
                                min_alpha = int(min_val * 255) if arr.dtype != np.uint8 else int(min_val)
                                if min_alpha < 255:
                                    errors.append(
                                        f"{nb_path}: cell {idx} output {out_idx} has transparent canvas "
                                        f"(min_alpha={min_alpha} < 255)"
                                    )
                    except Exception as e:
                        errors.append(f"{nb_path}: cell {idx} output {out_idx} PNG decode failed: {e}")
    return errors


def build_one(src: Path, *, check: bool = False, theme: str | None = None) -> int:
    """Convert+execute one source. check=True writes to a temp file and validates outputs.

    - If src is a .py file (showcase): executes via jupytext --to ipynb --execute
      with --run-path notebooks/.
    - If src is a .ipynb file (curso): executes via jupytext --execute
      with --run-path curso/notebooks/.
    """
    env = {}
    if theme:
        env["MAV_NB_TEMA"] = theme

    if src.suffix == ".py":
        run_path = NB_DIR
        rel = str(src.relative_to(NB_DIR))
        base = ["--to", "ipynb", "--execute", "--set-kernel", KERNEL_NAME,
                "--run-path", str(run_path)]
        if check:
            out = Path(tempfile.gettempdir()) / f"{src.stem}.check.ipynb"
            rc = _jupytext([*base, "--output", str(out), rel], cwd=run_path, env=env)
            if rc == 0:
                errors = validate_notebook_node(out)
                if errors:
                    print(f"Validation failed for {src.name}:", file=sys.stderr)
                    for err in errors:
                        print(f"  {err}", file=sys.stderr)
                    rc = 1
            out.unlink(missing_ok=True)
            return rc
        rc = _jupytext([*base, rel], cwd=run_path, env=env)
        if rc == 0:
            out_file = src.parent / f"{src.stem}.ipynb"
            if out_file.exists():
                errors = validate_notebook_node(out_file)
                if errors:
                    print(f"Validation warning/error for {out_file.name}:", file=sys.stderr)
                    for err in errors:
                        print(f"  {err}", file=sys.stderr)
        return rc

    elif src.suffix == ".ipynb":
        run_path = CURSO_DIR if (CURSO_DIR in src.parents or src.parent == CURSO_DIR) else src.parent
        rel = str(src.relative_to(run_path))
        base = ["--execute", "--set-kernel", KERNEL_NAME,
                "--run-path", str(run_path)]
        if check:
            out = Path(tempfile.gettempdir()) / f"{src.stem}.check.ipynb"
            rc = _jupytext([*base, "--output", str(out), rel], cwd=run_path, env=env)
            if rc == 0:
                errors = validate_notebook_node(out)
                if errors:
                    print(f"Validation failed for {src.name}:", file=sys.stderr)
                    for err in errors:
                        print(f"  {err}", file=sys.stderr)
                    rc = 1
            out.unlink(missing_ok=True)
            return rc
        rc = _jupytext([*base, rel], cwd=run_path, env=env)
        if rc == 0:
            errors = validate_notebook_node(src)
            if errors:
                print(f"Validation warning/error for {src.name}:", file=sys.stderr)
                for err in errors:
                    print(f"  {err}", file=sys.stderr)
        return rc

    else:
        print(f"Unsupported notebook source format: {src}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build and verify puremacro showcase and course notebooks.")
    ap.add_argument("names", nargs="*", help="source stems/filenames to build (default: all)")
    ap.add_argument(
        "--suite",
        choices=["showcase", "curso", "all"],
        default="all",
        help="notebook suite to discover/build: showcase, curso, or all (default: all)",
    )
    ap.add_argument("--theme", choices=["grafito", "papel"], default=None, help="override figure theme via MAV_NB_TEMA")
    ap.add_argument("--list", action="store_true", help="list discovered sources and exit")
    ap.add_argument("--check", action="store_true", help="execute to temp, validate outputs, fail on error")
    ap.add_argument("--validate-only", action="store_true", help="validate committed .ipynb files without executing")
    ns = ap.parse_args(argv)

    srcs = discover_sources(ns.suite)
    if ns.names:
        wanted = set(ns.names)
        srcs = [s for s in srcs if s.stem in wanted or s.name in wanted]
        if not srcs:
            print(f"no sources match {sorted(wanted)} in suite {ns.suite!r}", file=sys.stderr)
            return 2

    if ns.list:
        for s in srcs:
            print(s.relative_to(PROJ_ROOT))
        return 0

    if ns.validate_only:
        total_errors = 0
        total_checked = 0
        for s in srcs:
            target_ipynb = s if s.suffix == ".ipynb" else s.parent / f"{s.stem}.ipynb"
            if not target_ipynb.exists():
                print(f"Missing artifact: {target_ipynb.relative_to(PROJ_ROOT)}", file=sys.stderr)
                total_errors += 1
                continue
            total_checked += 1
            errs = validate_notebook_node(target_ipynb)
            if errs:
                total_errors += len(errs)
                for err in errs:
                    print(f"  {err}", file=sys.stderr)
        print(f"Validated {total_checked} notebooks ({total_errors} errors found)")
        return 0 if total_errors == 0 else 1

    ensure_kernel()
    rc_total = 0
    for s in srcs:
        rc = build_one(s, check=ns.check, theme=ns.theme)
        rc_total = rc_total or rc
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
