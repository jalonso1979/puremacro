"""tools/pyodide_gallery.py — run the validation gallery inside headless Pyodide.

Builds the puremacro wheel, hands it to ``tools/pyodide/gallery_runner.js``
(Pyodide under Node.js, the harness Gate 6 uses), and prints a summary of the
110-case validation gallery as it ran inside WebAssembly: totals, totals by
subsystem and by mechanism, the ten slowest cases and every failure with its
error text. With ``--desktop-json`` it also runs the same gallery, with the same
per-case timing loop (``tools/pyodide/gallery_cases.py``), in a desktop CPython
subprocess of the interpreter running this script, and lists every case whose
pass flag or ``max_margin`` differs between the two platforms.

What this measures
    Whether the wheel installs into Pyodide with its dependencies resolved from
    the Pyodide distribution alone (PyPI fallback counts as a failure, as in the
    JupyterLite playground), whether each gallery case passes there against the
    same frozen goldens and tolerances as on the desktop, and how long each case
    takes in that runtime.

What this does not measure
    Pyodide under Node.js is the same WebAssembly runtime family as the
    JupyterLite playground but not a browser tab: no DOM, no browser memory cap,
    no main-thread scheduling, and Node's own file system and network. The
    Pyodide version is the one pinned in ``tools/pyodide/package.json``, which
    may differ from the version bundled by the playground's
    ``jupyterlite-pyodide-kernel``. The gallery is a set of small, seeded
    estimation problems; its run time says nothing about long MCMC chains, large
    bootstraps or other heavy workloads, and nothing about GPU execution.

Usage::

    python tools/pyodide_gallery.py                       # build the wheel, run, summarise
    python tools/pyodide_gallery.py --json results.json   # also keep the envelope
    python tools/pyodide_gallery.py --wheel dist/puremacro-4.5.0-py3-none-any.whl
    python tools/pyodide_gallery.py --json p.json --desktop-json d.json   # compare platforms

No dependencies beyond the standard library, Node.js and the Pyodide package
installed by ``cd tools/pyodide && npm install``. Exit status: 0 when every case
passed in Pyodide (and, if requested, agrees with the desktop), 1 when a case
failed or disagreed, 2 when the harness itself could not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "tools" / "pyodide" / "gallery_runner.js"
CASES_PY = REPO_ROOT / "tools" / "pyodide" / "gallery_cases.py"


def _check_node() -> bool:
    try:
        return subprocess.run(["node", "--version"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_wheel(repo_root: Path, out_dir: Path) -> Path:
    """Build a puremacro wheel into ``out_dir`` and return its path.

    Uses ``pip wheel --no-deps --no-build-isolation`` (needs setuptools in the
    current interpreter, no network) and falls back to ``python -m build``.
    """
    attempts = [
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(out_dir), "."],
        [sys.executable, "-m", "build", "--wheel", "-o", str(out_dir)],
    ]
    errors = []
    for cmd in attempts:
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        if proc.returncode == 0:
            wheels = sorted(out_dir.glob("puremacro-*.whl"))
            if wheels:
                return wheels[-1]
        errors.append(f"$ {' '.join(cmd)}\n" + "\n".join(proc.stderr.splitlines()[-10:]))
    raise RuntimeError("wheel build failed:\n" + "\n".join(errors))


def run_pyodide(wheel: Path, out_json: Path | None, timeout: int, pyodide_dir: Path | None,
                preload: str | None = None) -> dict:
    """Invoke the Node runner; stream its progress to stderr; return the envelope."""
    cmd = ["node", str(RUNNER), "--wheel", str(wheel)]
    if out_json is not None:
        cmd += ["--out", str(out_json)]
    if pyodide_dir is not None:
        cmd += ["--pyodide", str(pyodide_dir)]
    if preload:
        cmd += ["--preload", preload]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=None,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"gallery_runner.js exited {proc.returncode} without an envelope")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_desktop(out_json: Path, timeout: int) -> dict:
    """Run gallery_cases.py in a fresh subprocess of this interpreter on the checkout."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(CASES_PY), "--json", str(out_json)], cwd=str(REPO_ROOT),
                          env=env, stdout=subprocess.PIPE, stderr=None, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"desktop gallery exited {proc.returncode} without an envelope")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*[str(c) for c in r]) for r in rows]
    return "\n".join("    " + ln for ln in lines)


def summarize(env: dict, label: str) -> str:
    """Human-readable summary of one envelope (Pyodide or desktop)."""
    cases = env.get("cases", [])
    out = [f"== {label} =="]
    if env.get("runtime") == "pyodide":
        out.append(f"    Pyodide {env.get('pyodide_version')} on Node {env.get('node_version')}, "
                   f"Python {env.get('python_version')}; wheel {env.get('wheel')} "
                   f"(sha256 {str(env.get('wheel_sha256', ''))[:12]}...)")
        if env.get("preloaded"):
            out.append(f"    DIAGNOSTIC RUN: {', '.join(env['preloaded'])} preloaded before the install "
                       "(the playground preloads nothing)")
        if env.get("preload_error"):
            out.append(f"    PRELOAD FAILED (run continued without it): {env['preload_error']}")
        if not env.get("wheel_installed"):
            out.append(f"    WHEEL INSTALL FAILED: {env.get('install_error')}")
        out.append(f"    boot {env.get('boot_s')} s, micropip install {env.get('install_s')} s, "
                   f"import + discovery {env.get('import_s')} s")
    else:
        out.append(f"    CPython {env.get('python_version')} on {env.get('platform')}; "
                   f"puremacro from {env.get('puremacro_file')}")
        out.append(f"    import + discovery {env.get('import_s')} s")
    pk = env.get("packages") or {}
    pk_s = ", ".join(f"{k} {v['version'] if isinstance(v, dict) else v}" for k, v in sorted(pk.items()))
    if pk_s:
        out.append(f"    packages: {pk_s}")
    out.append(f"    cases {env.get('n_cases')}, passed {env.get('n_passed')}, failed {env.get('n_failed')}; "
               f"gallery {env.get('gallery_s')} s, total runner wall time {env.get('runtime_s')} s")
    if not cases:
        return "\n".join(out)

    def group(key: str) -> list[list[str]]:
        agg: dict = defaultdict(lambda: [0, 0, 0.0])
        for c in cases:
            a = agg[c[key]]
            a[0] += 1
            a[1] += int(c["passed"])
            a[2] += c["seconds"]
        return [[k, n, p, f"{s:.3f}"] for k, (n, p, s) in sorted(agg.items())]

    out.append("  by subsystem:")
    out.append(_table(["subsystem", "cases", "passed", "seconds"], group("subsystem")))
    out.append("  by mechanism:")
    out.append(_table(["mechanism", "cases", "passed", "seconds"], group("mechanism")))
    slowest = sorted(cases, key=lambda c: -c["seconds"])[:10]
    out.append("  10 slowest cases:")
    out.append(_table(["case", "subsystem", "seconds"], [[c["id"], c["subsystem"], f"{c['seconds']:.3f}"] for c in slowest]))
    failures = [c for c in cases if not c["passed"]]
    if failures:
        out.append("  failures:")
        for c in failures:
            margin = "n/a" if c["max_margin"] is None else f"{c['max_margin']:.3e}"
            out.append(f"    {c['id']} ({c['subsystem']}, {c['mechanism']}/{c['tol']}): max_margin {margin}"
                       + (f"; {c['error']}" if c["error"] else ""))
    else:
        out.append("  failures: none")
    return "\n".join(out)


def compare(pyodide_env: dict, desktop_env: dict, margin_tol: float = 1e-9) -> list[str]:
    """Cases whose pass flag or max_margin differ between the two platforms."""
    desk = {c["id"]: c for c in desktop_env.get("cases", [])}
    diffs = []
    for c in pyodide_env.get("cases", []):
        d = desk.pop(c["id"], None)
        if d is None:
            diffs.append(f"{c['id']}: present in Pyodide only")
            continue
        if c["passed"] != d["passed"]:
            diffs.append(f"{c['id']}: passed {c['passed']} in Pyodide vs {d['passed']} on desktop")
        a, b = c["max_margin"], d["max_margin"]
        if (a is None) != (b is None) or (a is not None and abs(a - b) > margin_tol):
            diffs.append(f"{c['id']}: max_margin {a!r} in Pyodide vs {b!r} on desktop")
    diffs += [f"{cid}: present on desktop only" for cid in desk]
    return diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, default=None, help="use this wheel instead of building one")
    parser.add_argument("--json", type=Path, default=None, help="write the Pyodide envelope here")
    parser.add_argument("--desktop-json", type=Path, default=None,
                        help="also run the gallery in this interpreter, write its envelope here and compare")
    parser.add_argument("--pyodide", type=Path, default=None,
                        help="alternative Pyodide npm package directory (default: tools/pyodide/node_modules/pyodide)")
    parser.add_argument("--preload", default=None, metavar="PKGS",
                        help="comma-separated Pyodide packages to load before the wheel, for diagnosis only "
                             "(e.g. sqlite3, which Pyodide unvendors from the stdlib); the playground preloads nothing")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed for each run (default 3600)")
    parser.add_argument("--margin-tol", type=float, default=1e-9,
                        help="max_margin difference tolerated between platforms (default 1e-9)")
    args = parser.parse_args(argv)

    if not _check_node():
        print("error: Node.js not found on PATH", file=sys.stderr)
        return 2
    if args.pyodide is None and not (REPO_ROOT / "tools" / "pyodide" / "node_modules" / "pyodide").is_dir():
        print("error: Pyodide not installed; run `cd tools/pyodide && npm install` once", file=sys.stderr)
        return 2

    tmp = None
    try:
        if args.wheel is None:
            tmp = tempfile.mkdtemp(prefix="puremacro_pyodide_gallery_")
            print("building wheel ...", file=sys.stderr)
            wheel = build_wheel(REPO_ROOT, Path(tmp))
        else:
            wheel = args.wheel.resolve()
            if not wheel.is_file():
                print(f"error: wheel not found at {wheel}", file=sys.stderr)
                return 2
        print(f"wheel {wheel.name}: {wheel.stat().st_size} bytes, sha256 {_sha256(wheel)}", file=sys.stderr)

        try:
            pyodide_env = run_pyodide(wheel, args.json, args.timeout, args.pyodide, args.preload)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(summarize(pyodide_env, "Pyodide (Node-hosted)"))
        if not pyodide_env.get("wheel_installed"):
            return 2

        status = 0 if pyodide_env.get("n_failed") == 0 and pyodide_env.get("n_cases") else 1
        if args.desktop_json is not None:
            print("\nrunning the desktop baseline ...", file=sys.stderr)
            try:
                desktop_env = run_desktop(args.desktop_json, args.timeout)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print()
            print(summarize(desktop_env, "Desktop (this interpreter)"))
            diffs = compare(pyodide_env, desktop_env, args.margin_tol)
            print()
            if diffs:
                print(f"== Pyodide vs desktop: {len(diffs)} discrepancies (margin tolerance {args.margin_tol:g}) ==")
                print("\n".join("    " + d for d in diffs))
                status = 1
            else:
                print(f"== Pyodide vs desktop: every case agrees on passed and on max_margin to {args.margin_tol:g} ==")
            ratio = (pyodide_env.get("gallery_s") or 0) / max(desktop_env.get("gallery_s") or 1e-9, 1e-9)
            print(f"    gallery time {pyodide_env.get('gallery_s')} s in Pyodide vs {desktop_env.get('gallery_s')} s "
                  f"on the desktop ({ratio:.2f}x)")
        return status
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
