"""tools/render_examples_gallery.py — render the puremacro examples gallery.

Discovers `puremacro/examples/*.py`, subprocess-runs each, classifies the
result as PASS / SKIP / FAIL, captures figures from `examples/output/`,
and emits `docs/examples_gallery.{md,json}`.

Run me explicitly. Gate 5 in `tools/release_check.py` reads my JSON; it
does not invoke me.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NETWORK_MARKERS = re.compile(
    r"(urllib\.error\.URLError|urllib\.error\.HTTPError|"
    r"http\.client\.HTTPException|requests\.exceptions\.\w+|"
    r"httpx\.\w+Error|aiohttp\.client_exceptions|"
    r"socket\.gaierror|ssl\.SSLError)"
)
DATA_MARKERS = re.compile(
    r"FileNotFoundError.*\.(parquet|csv|feather|hdf5|h5|json|npz|pkl|txt|npy)"
)
SKIP_COMMENT = re.compile(r"^\s*#\s*skip:\s*(.+?)\s*$")


def capture_figures(
    before_mtimes: dict[str, float],
    after_mtimes: dict[str, float],
    output_dir: Path,
) -> list[str]:
    """Return sorted list of figure basenames new or modified between snapshots.

    A file is "new or modified" if its post-run mtime is strictly greater than
    its pre-run mtime (or it didn't exist pre-run).

    The `output_dir` argument is currently unused but reserved for future
    extensions (e.g., filtering by glob pattern).
    """
    new_files = []
    for name, after_mt in after_mtimes.items():
        before_mt = before_mtimes.get(name)
        if before_mt is None or after_mt > before_mt:
            new_files.append(name)
    return sorted(new_files)


def classify(
    *,
    returncode: int | None,
    stderr: str,
    source_first_lines: list[str],
    timed_out: bool,
) -> dict:
    """Classify a subprocess result into PASS / SKIP / FAIL.

    Precedence:
        1. Explicit `# skip: <reason>` comment in source_first_lines (top-of-file marker wins).
        2. timed_out=True → FAIL with reason "timeout".
        3. returncode is None (signal-killed or never reaped) → FAIL with explicit reason.
        4. returncode == 0 → PASS.
        5. stderr matches network markers (dotted-name only) → SKIP.
        6. stderr matches data-file markers → SKIP.
        7. else → FAIL with the last stderr line truncated to 200 chars.

    Returns:
        dict with keys: status ("PASS"|"SKIP"|"FAIL"), reason (str|None).
    """
    # 1. Explicit skip comment wins over everything else.
    for line in source_first_lines[:10]:
        m = SKIP_COMMENT.match(line)
        if m:
            return {"status": "SKIP", "reason": m.group(1)}
    # 2. Timeout.
    if timed_out:
        return {"status": "FAIL", "reason": "timeout"}
    # 3. Process never returned a code (signal, kernel kill).
    if returncode is None:
        return {"status": "FAIL", "reason": "no return code (likely signal-killed)"}
    # 4. Clean exit.
    if returncode == 0:
        return {"status": "PASS", "reason": None}
    # 5/6. Non-zero — check stderr markers.
    if NETWORK_MARKERS.search(stderr):
        return {"status": "SKIP", "reason": "network unavailable"}
    if DATA_MARKERS.search(stderr):
        return {"status": "SKIP", "reason": "local data file missing"}
    # 7. Unknown error → FAIL with truncated stderr.
    reason = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
    if len(reason) > 200:
        reason = reason[:200]
    return {"status": "FAIL", "reason": reason}


def _read_source_first_lines(path: Path, n: int = 10) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()[:n]
    except (OSError, UnicodeDecodeError):
        return []


def _snapshot_mtimes(output_dir: Path) -> dict[str, float]:
    if not output_dir.exists():
        return {}
    return {p.name: p.stat().st_mtime for p in output_dir.iterdir() if p.is_file()}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_example(
    name: str,
    *,
    examples_dir: Path,
    output_dir: Path,
    timeout: int = 300,
) -> dict:
    """Run one example via `python -m puremacro.examples.<name>` and classify.

    Returns:
        dict with keys: name, status, reason, runtime_s, figures, last_run.
    """
    src_path = examples_dir / f"{name}.py"
    source_lines = _read_source_first_lines(src_path)
    # Explicit # skip: comment shortcuts subprocess entirely.
    for line in source_lines[:10]:
        m = SKIP_COMMENT.match(line)
        if m:
            return {
                "name": name,
                "status": "SKIP",
                "reason": m.group(1),
                "runtime_s": 0.0,
                "figures": [],
                "last_run": _now_iso(),
            }

    before = _snapshot_mtimes(output_dir)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"puremacro.examples.{name}"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        timed_out = False
        returncode = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = None
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if e.stderr else ""
    runtime = time.time() - t0
    after = _snapshot_mtimes(output_dir)
    figures = capture_figures(before, after, output_dir)

    cls = classify(
        returncode=returncode,
        stderr=stderr,
        source_first_lines=source_lines,
        timed_out=timed_out,
    )
    return {
        "name": name,
        "status": cls["status"],
        "reason": cls["reason"],
        "runtime_s": round(runtime, 3),
        # Path is relative to docs/examples_gallery.md location (docs/),
        # so we prepend ../ to reach puremacro/examples/output/ from there.
        "figures": [str(Path("../puremacro/examples/output") / f) for f in figures],
        "last_run": _now_iso(),
    }


def _discover_examples(examples_dir: Path) -> list[str]:
    names = []
    for p in sorted(examples_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        if p.name.startswith("_"):
            continue
        names.append(p.stem)
    return names


def render_all(
    *,
    examples_dir: Path,
    output_dir: Path,
    docs_dir: Path,
    timeout: int = 300,
    only: str | None = None,
    limit: int | None = None,
) -> dict:
    """Render every (or filtered) example, write JSON + Markdown to docs_dir."""
    names = _discover_examples(examples_dir)
    if only is not None:
        names = [n for n in names if n == only]
    if limit is not None:
        names = names[:limit]

    entries: dict[str, dict] = {}
    for n in names:
        print(f"  rendering {n} ...", flush=True)
        r = run_example(n, examples_dir=examples_dir, output_dir=output_dir, timeout=timeout)
        entries[n] = {
            "status": r["status"],
            "reason": r["reason"],
            "runtime_s": r["runtime_s"],
            "figures": r["figures"],
            "last_run": r["last_run"],
        }

    # If --only was set, preserve any pre-existing entries for other examples.
    json_path = docs_dir / "examples_gallery.json"
    if only is not None and json_path.exists():
        prior = json.loads(json_path.read_text(encoding="utf-8")).get("examples", {})
        for k, v in prior.items():
            if k not in entries:
                entries[k] = v

    data = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "examples": entries,
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _emit_markdown(data, docs_dir / "examples_gallery.md")
    return data


def _emit_markdown(data: dict, md_path: Path) -> None:
    examples = data["examples"]
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    for e in examples.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1

    lines = []
    lines.append("# puremacro examples gallery")
    lines.append("")
    lines.append(f"Generated: {data['generated_at']}")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    lines.append(f"| PASS | {counts['PASS']} |")
    lines.append(f"| SKIP | {counts['SKIP']} |")
    lines.append(f"| FAIL | {counts['FAIL']} |")
    lines.append("")

    for status in ("FAIL", "PASS", "SKIP"):
        section = [(name, e) for name, e in sorted(examples.items()) if e["status"] == status]
        if not section:
            continue
        lines.append(f"## {status} ({len(section)})")
        lines.append("")
        for name, e in section:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- **Status:** {e['status']}")
            if e["reason"] is not None:
                lines.append(f"- **Reason:** {e['reason']}")
            lines.append(f"- **Runtime:** {e['runtime_s']} s")
            lines.append(f"- **Last run:** {e['last_run']}")
            if e["figures"]:
                lines.append("- **Figures:**")
                for f in e["figures"]:
                    lines.append(f"  - `{f}`")
            else:
                lines.append("- **Figures:** none")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render_examples_gallery",
        description="Render the puremacro examples gallery + JSON status.",
    )
    parser.add_argument("--only", help="render only this example by name (no .py)")
    parser.add_argument("--timeout", type=int, default=300, help="per-example timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="process at most N examples")
    args = parser.parse_args(argv)

    examples_dir = REPO_ROOT / "puremacro" / "examples"
    output_dir = examples_dir / "output"
    docs_dir = REPO_ROOT / "docs"

    if not examples_dir.exists():
        print(f"error: {examples_dir} not found", file=sys.stderr)
        return 2

    output_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    print(f"render_examples_gallery — examples_dir={examples_dir}")
    print(f"  timeout={args.timeout} only={args.only} limit={args.limit}")
    data = render_all(
        examples_dir=examples_dir,
        output_dir=output_dir,
        docs_dir=docs_dir,
        timeout=args.timeout,
        only=args.only,
        limit=args.limit,
    )
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    for e in data["examples"].values():
        counts[e["status"]] += 1
    print(f"done: {counts['PASS']} PASS, {counts['SKIP']} SKIP, {counts['FAIL']} FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
