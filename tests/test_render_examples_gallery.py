"""Tests for tools/render_examples_gallery.py classifier logic."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "render_examples_gallery.py"

_spec = importlib.util.spec_from_file_location("render_examples_gallery", SCRIPT)
render_examples_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_examples_gallery)


def test_classify_pass():
    r = render_examples_gallery.classify(
        returncode=0,
        stderr="",
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "PASS"
    assert r["reason"] is None


def test_classify_fail_unknown_error():
    r = render_examples_gallery.classify(
        returncode=1,
        stderr="ValueError: bad input\n",
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "FAIL"
    assert "ValueError" in r["reason"]


def test_classify_skip_network_urllib():
    r = render_examples_gallery.classify(
        returncode=1,
        stderr='urllib.error.URLError: <urlopen error [Errno 8]>\n',
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "SKIP"
    assert "network" in r["reason"].lower()


def test_classify_skip_network_requests():
    r = render_examples_gallery.classify(
        returncode=1,
        stderr="requests.exceptions.ConnectionError: HTTPSConnectionPool...\n",
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "SKIP"
    assert "network" in r["reason"].lower()


def test_classify_skip_data_missing_parquet():
    r = render_examples_gallery.classify(
        returncode=1,
        stderr="FileNotFoundError: [Errno 2] No such file: 'data/processed/panel_Q.parquet'\n",
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "SKIP"
    assert "data" in r["reason"].lower()


def test_classify_skip_explicit_comment():
    r = render_examples_gallery.classify(
        returncode=99,
        stderr="this would normally fail",
        source_first_lines=["# skip: requires CUDA GPU", "import torch"],
        timed_out=False,
    )
    assert r["status"] == "SKIP"
    assert r["reason"] == "requires CUDA GPU"


def test_classify_explicit_skip_wins_over_stderr():
    """If both an explicit # skip: comment AND a network error are present, the comment wins."""
    r = render_examples_gallery.classify(
        returncode=1,
        stderr="urllib.error.URLError: ...",
        source_first_lines=["# skip: by design"],
        timed_out=False,
    )
    assert r["status"] == "SKIP"
    assert r["reason"] == "by design"


def test_classify_timeout():
    r = render_examples_gallery.classify(
        returncode=None,
        stderr="",
        source_first_lines=["import puremacro"],
        timed_out=True,
    )
    assert r["status"] == "FAIL"
    assert "timeout" in r["reason"].lower()


def test_classify_fail_truncates_long_stderr():
    long_err = "X" * 1000
    r = render_examples_gallery.classify(
        returncode=1,
        stderr=long_err,
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "FAIL"
    assert len(r["reason"]) <= 200


def test_classify_returncode_none_without_timeout():
    """returncode=None when timed_out=False is treated as a failure with explicit reason."""
    r = render_examples_gallery.classify(
        returncode=None,
        stderr="",
        source_first_lines=["import puremacro"],
        timed_out=False,
    )
    assert r["status"] == "FAIL"
    assert "no return code" in r["reason"].lower() or "signal" in r["reason"].lower()


def test_capture_figures_attributes_only_new(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Pre-existing file from another example.
    (output_dir / "a.png").write_bytes(b"old")
    before = {p.name: p.stat().st_mtime for p in output_dir.iterdir()}
    # Sleep briefly so mtime resolution can distinguish before/after.
    import time
    time.sleep(0.05)
    (output_dir / "b.png").write_bytes(b"new")
    after = {p.name: p.stat().st_mtime for p in output_dir.iterdir()}
    figures = render_examples_gallery.capture_figures(before, after, output_dir)
    assert figures == ["b.png"]


def test_capture_figures_includes_modified(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "a.png").write_bytes(b"v1")
    before = {p.name: p.stat().st_mtime for p in output_dir.iterdir()}
    import time
    time.sleep(0.05)
    # Modify the existing file.
    (output_dir / "a.png").write_bytes(b"v2")
    after = {p.name: p.stat().st_mtime for p in output_dir.iterdir()}
    figures = render_examples_gallery.capture_figures(before, after, output_dir)
    assert figures == ["a.png"]


def test_capture_figures_empty_when_no_change(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "a.png").write_bytes(b"unchanged")
    before = {p.name: p.stat().st_mtime for p in output_dir.iterdir()}
    after = before
    figures = render_examples_gallery.capture_figures(before, after, output_dir)
    assert figures == []


def test_run_example_pass(tmp_path, monkeypatch):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    output_dir = examples_dir / "output"
    output_dir.mkdir()
    # Write a tiny synthetic "example" that succeeds.
    src = examples_dir / "syn_pass.py"
    src.write_text("print('hello')\n")
    # Monkeypatch subprocess.run to avoid relying on a real Python module import path.
    class P:
        returncode = 0
        stdout = "hello\n"
        stderr = ""
    monkeypatch.setattr(render_examples_gallery.subprocess, "run", lambda *a, **kw: P())
    r = render_examples_gallery.run_example(
        "syn_pass", examples_dir=examples_dir, output_dir=output_dir, timeout=5,
    )
    assert r["name"] == "syn_pass"
    assert r["status"] == "PASS"
    assert r["reason"] is None
    assert r["runtime_s"] >= 0
    assert r["figures"] == []
    assert r["last_run"].endswith("Z")


def test_run_example_explicit_skip_no_subprocess(tmp_path, monkeypatch):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    output_dir = examples_dir / "output"
    output_dir.mkdir()
    src = examples_dir / "syn_skip.py"
    src.write_text("# skip: synthetic\nraise RuntimeError('should not run')\n")
    # Set subprocess.run to a sentinel that would mark the test as broken if invoked.
    def must_not_be_called(*a, **kw):
        raise AssertionError("subprocess.run should not be invoked for explicit-skip examples")
    monkeypatch.setattr(render_examples_gallery.subprocess, "run", must_not_be_called)
    r = render_examples_gallery.run_example(
        "syn_skip", examples_dir=examples_dir, output_dir=output_dir, timeout=5,
    )
    assert r["status"] == "SKIP"
    assert r["reason"] == "synthetic"


def test_render_emits_json_with_schema(tmp_path, monkeypatch):
    """End-to-end: render against a tmp directory of 2 examples, check JSON shape."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    output_dir = examples_dir / "output"
    output_dir.mkdir()
    (examples_dir / "a_pass.py").write_text("print('a')\n")
    (examples_dir / "b_skip.py").write_text("# skip: synthetic\n")

    class P:
        returncode = 0
        stdout = ""
        stderr = ""
    monkeypatch.setattr(render_examples_gallery.subprocess, "run", lambda *a, **kw: P())

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    render_examples_gallery.render_all(
        examples_dir=examples_dir,
        output_dir=output_dir,
        docs_dir=docs_dir,
        timeout=5,
    )
    import json
    data = json.loads((docs_dir / "examples_gallery.json").read_text())
    assert data["schema_version"] == 1
    assert "generated_at" in data
    assert set(data["examples"].keys()) == {"a_pass", "b_skip"}
    assert data["examples"]["a_pass"]["status"] == "PASS"
    assert data["examples"]["b_skip"]["status"] == "SKIP"


def test_render_emits_markdown_with_status_counts(tmp_path, monkeypatch):
    """Markdown contains a summary table with PASS/SKIP/FAIL counts."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    output_dir = examples_dir / "output"
    output_dir.mkdir()
    (examples_dir / "a_pass.py").write_text("print('a')\n")
    (examples_dir / "b_skip.py").write_text("# skip: synthetic\n")

    class P:
        returncode = 0
        stdout = ""
        stderr = ""
    monkeypatch.setattr(render_examples_gallery.subprocess, "run", lambda *a, **kw: P())

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    render_examples_gallery.render_all(
        examples_dir=examples_dir,
        output_dir=output_dir,
        docs_dir=docs_dir,
        timeout=5,
    )
    md = (docs_dir / "examples_gallery.md").read_text()
    assert "# puremacro examples gallery" in md
    assert "PASS" in md and "SKIP" in md
    # FAIL section appears even when zero, for predictability.
    assert "FAIL" in md
    assert "a_pass" in md
    assert "b_skip" in md
