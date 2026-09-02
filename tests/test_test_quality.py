"""Meta-tests: guard the suite against tests that cannot fail.

A green test that would still be green with the code deleted is worse than no
test, because it is read as coverage. Three of them turned up in one working
session, each from a different cause, and only two of the three are the kind of
thing mutation testing would ever find:

1. **The fixture could not produce the condition.** A guard filtering the
   labour rows down to countries the money block returned was deletable with
   the suite still green, because the fixture generated both from the same
   country list. Mutation testing finds this.
2. **The mock replaced the subject.** A migration's tests patched the very
   function whose behaviour they asserted, so the assertion was about a
   hand-written dict. :func:`test_named_tests_execute_what_they_name` finds
   this, in seconds, and points at the cause.
3. **The test's own mechanism was inert.** An import blocker written against
   ``find_module`` — removed in Python 3.12 — blocked nothing, so eight tests
   asserting "this imports without requests" passed without importing anything
   under the stated condition. *Mutation testing cannot find this even in
   principle*: the defect is in the scaffolding, not the subject, so mutating
   the subject changes nothing. Only a positive control finds it, which is what
   :func:`test_mechanism_tests_have_a_positive_control` requires.

``monkeypatch.setattr`` is deliberately not treated as a mechanism: it raises
when the target attribute does not exist, so it fails loudly on its own. The
risk is concentrated in hand-rolled mechanisms, which is a small set.
"""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

# --------------------------------------------------------------------------
# 1. positive controls
# --------------------------------------------------------------------------

#: Source markers of a hand-rolled mechanism — something the test installs that
#: could silently stop working without any test noticing.
_MECHANISMS = {
    "sys.meta_path": "import hook",
    "MetaPathFinder": "import hook",
    "builtins.__import__": "import interception",
    "socket.socket": "network blocker",
    "socket.create_connection": "network blocker",
}

#: A file using a mechanism must mark one test with this, asserting the
#: mechanism is actually live.
_CONTROL_MARK = "mechanism_control"


def _mechanism_files() -> dict[Path, list[str]]:
    found: dict[Path, list[str]] = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        hits = sorted({label for marker, label in _MECHANISMS.items()
                       if marker in src})
        if hits:
            found[path] = hits
    return found


def _has_control_mark(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if _CONTROL_MARK in ast.dump(dec):
                return True
    return False


def test_mechanism_tests_have_a_positive_control():
    """A test that installs a mechanism must prove the mechanism is live.

    The failure this prevents is silent and total: an inert import hook makes
    every test that depends on it pass while asserting nothing, and nothing else
    in the suite — coverage, mutation, the gates — can tell.
    """
    missing = {p.relative_to(_ROOT).as_posix(): kinds
               for p, kinds in _mechanism_files().items()
               if not _has_control_mark(p)}
    assert not missing, (
        "these files install a mechanism with no positive control asserting it "
        f"works: {json.dumps(missing, indent=2)}\n"
        f"Add a test marked @pytest.mark.{_CONTROL_MARK} that asserts the "
        "mechanism actually bites — e.g. that the blocked import really raises.")


def test_the_mechanism_scan_finds_something():
    """Guards the guard. If the markers stop matching anything, the check above
    passes vacuously — which is the exact failure mode this file exists for."""
    found = _mechanism_files()
    assert found, (
        "no test file matched any mechanism marker; either the markers in "
        "_MECHANISMS are stale or the scan is broken")


# --------------------------------------------------------------------------
# 2. coverage attribution
# --------------------------------------------------------------------------

#: ``test file -> the functions it must actually execute``. Not a coverage
#: percentage: the useful question is not "how much of the module ran" but "did
#: the code this test names ever run at all", which is what a mock replacing the
#: subject silently answers no to.
_MUST_EXECUTE: dict[str, tuple[str, ...]] = {
    "tests/test_capital.py": (
        "puremacro.capital:qna_capital",
        "puremacro.capital:qna_tfp",
        "puremacro.capital:_pim",
    ),
    "tests/test_build_panel_qna_labor.py": (
        # the migration whose first tests mocked this away
        "puremacro.build_panel:_fetch_qna_labor_logs",
        "puremacro.fetch.oecd_qna_panel:qna_labor",
        "puremacro.fetch.oecd_qna_panel:_labor_tidy",
    ),
    "tests/test_oecd_qna_panel.py": (
        "puremacro.fetch.oecd_qna_panel:qna_panel",
        "puremacro.fetch.oecd_qna_panel:_tidy",
        "puremacro.fetch.oecd_qna_panel:_rescale_hours",
        "puremacro.fetch.oecd_qna_panel:_labor_activity_lookup",
        "puremacro.fetch.oecd_qna_panel:_build_meta",
    ),
    # Annual national accounts by activity. These tests patch `get_sdmx_csv`,
    # so the registry is what proves they run the aggregation, the chaining
    # and the plausibility guard rather than asserting about the patch.
    "tests/test_oecd_ana_activity.py": (
        "puremacro.fetch.oecd_ana_activity:ana_by_activity",
        "puremacro.fetch.oecd_ana_activity:chain_volume",
        "puremacro.fetch.oecd_ana_activity:ana_hours_wedge",
        "puremacro.fetch.oecd_ana_activity:_rescale_hours",
        "puremacro.fetch.oecd_ana_activity:_nonneg",
    ),
    # Real-time / vintage layer (1.7.0). These tests patch the HTTP
    # boundary, so the registry is what proves they still run the
    # accessor and the parsers rather than asserting about the patch.
    "tests/test_realtime_providers/test_alfred.py": (
        "puremacro.fetch.realtime.alfred:alfred_vintages",
        "puremacro.fetch.realtime.alfred:parse_alfredgraph_csv",
        "puremacro.fetch.realtime.alfred:parse_alfred_api_observations",
        "puremacro.fetch.realtime.alfred:_fetch_via_graph_csv",
    ),
    "tests/test_realtime_providers/test_parsers.py": (
        "puremacro.fetch.realtime.statcan:parse_statcan_vintage_csv",
        "puremacro.fetch.realtime.bundesbank:parse_bbk_rtd_csv",
        "puremacro.fetch.realtime.oecd_stes:parse_stes_revisions_csv",
        "puremacro.fetch.realtime.ecb_rtd:parse_ecb_history_csv",
        "puremacro.fetch.realtime.ons:parse_ons_vintage_label",
    ),
    "tests/test_vintages_revisions.py": (
        "puremacro.vintages:mankiw_shapiro",
        "puremacro.vintages:revision_triangle",
        "puremacro.vintages:revision_frame",
        "puremacro.vintages:_apply_revision_transform",
        "puremacro.vintages:_first_last_from_triangle",
        "puremacro.vintages:_kth_edition",
    ),
    "tests/test_realtime_providers/test_seasonal.py": (
        "puremacro.fetch.realtime.seasonal:seasonal_signature",
        "puremacro.fetch.realtime.seasonal:drop_unadjusted_editions",
    ),
    "tests/test_longpanel/test_splice.py": (
        "puremacro.fetch.longpanel._splice:ratio_splice",
        "puremacro.fetch.longpanel._splice:overlap_ratio",
        "puremacro.fetch.longpanel._splice:splice_frame",
        "puremacro.fetch.longpanel._splice:expenditure_residual",
    ),
    "tests/test_longpanel/test_sources.py": (
        "puremacro.fetch.longpanel.ine_es:parse_ine_series",
        "puremacro.fetch.longpanel.ine_es:parse_cntrb86_workbook",
        "puremacro.fetch.longpanel.esri_jp:parse_esri_csv",
        "puremacro.fetch.longpanel.panel:qna_long_panel",
    ),
    "tests/test_realtime_providers/test_panel_plumbing.py": (
        # The registry resolves module-level functions only, so the
        # VintagePanel methods these tests drive are covered through the
        # entry point that calls them.
        "puremacro.fetch.realtime.panel:vintage_panel",
        "puremacro.fetch.realtime.catalog:resolve_spec",
    ),
}


def _line_range(target: str) -> tuple[str, int, int]:
    module_name, _, func_name = target.partition(":")
    module = __import__(module_name, fromlist=["_"])
    func = getattr(module, func_name)
    lines, start = inspect.getsourcelines(func)
    return inspect.getsourcefile(func), start, start + len(lines) - 1


@pytest.mark.parametrize("test_file", sorted(_MUST_EXECUTE))
def test_named_tests_execute_what_they_name(test_file, tmp_path):
    """The test file must actually run the functions it is about.

    A test that patches its own subject asserts only that the patch works. This
    runs the file under coverage in a subprocess and checks that at least one
    line inside each named function was executed.
    """
    targets = _MUST_EXECUTE[test_file]
    data_file = tmp_path / ".coverage"
    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "run",
         f"--data-file={data_file}", "--source=puremacro",
         "-m", "pytest", test_file, "-q", "-p", "no:randomly"],
        cwd=_ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, f"{test_file} did not pass:\n{proc.stdout[-3000:]}"

    from coverage import CoverageData
    data = CoverageData(basename=str(data_file))
    data.read()
    measured = {Path(f).resolve(): set(data.lines(f) or ()) for f in data.measured_files()}

    unexecuted = []
    for target in targets:
        src, lo, hi = _line_range(target)
        hit = measured.get(Path(src).resolve(), set())
        if not any(lo <= ln <= hi for ln in hit):
            unexecuted.append(target)
    assert not unexecuted, (
        f"{test_file} never executed {unexecuted}. Either it patches its own "
        "subject — in which case the assertions are about the patch, not the "
        "code — or the registry in _MUST_EXECUTE is stale.")


def test_the_attribution_registry_points_at_real_functions():
    """A typo'd target would make the check above skip silently."""
    for test_file, targets in _MUST_EXECUTE.items():
        assert (_ROOT / test_file).exists(), f"{test_file} does not exist"
        for target in targets:
            src, lo, hi = _line_range(target)          # raises if it is wrong
            assert hi > lo, f"{target} resolved to an empty range"
