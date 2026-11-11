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
import re
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
    # Latin America real-time connectors (3.4.0). The fetch tests patch
    # urllib.request.urlopen, so the registry is what proves the connectors'
    # own fetch paths, JSON parsers, schema canary and cartridge round-trip
    # run rather than the patch.
    "tests/test_realtime_latam.py": (
        "puremacro.fetch.realtime.banxico:fetch_banxico_vintages",
        "puremacro.fetch.realtime.banxico:parse_banxico_json",
        "puremacro.fetch.realtime.inegi:fetch_inegi_vintages",
        "puremacro.fetch.realtime.inegi:parse_inegi_json",
        "puremacro.fetch.realtime.bcb:fetch_bcb_vintages",
        "puremacro.fetch.realtime.bcb:parse_bcb_json",
        "puremacro.fetch.realtime.bcch:fetch_bcch_vintages",
        "puremacro.fetch.realtime.bcch:parse_bcch_json",
        "puremacro.fetch.realtime.canary:validate_payload",
        "puremacro.fetch.realtime.cartridge:pack_realtime_cartridge",
        "puremacro.fetch.realtime.cartridge:load_realtime_cartridge",
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


# ---------------------------------------------------------------------------
# Tail-accuracy guard for p-values.
# ---------------------------------------------------------------------------
_CDF_COMPLEMENT = re.compile(r"1(?:\.0)?\s*-\s*[\w.]*\bcdf\s*\(")

#: The only permitted `1 - cdf` in the package. It completes the top row of a
#: Tauchen transition matrix so the row sums to one; it is a probability mass
#: near one, not a p-value, and accuracy in the far tail is not what it needs.
_CDF_COMPLEMENT_ALLOWED = {"puremacro/vfi/discretize.py"}


def test_no_p_value_is_computed_as_one_minus_cdf():
    """p-values must use the survival function, which is accurate in the tail.

    ``1 - dist.cdf(x)`` loses every digit once ``cdf`` rounds to 1.0 in double
    precision, and then reports a p-value of **exactly zero** — a number no
    test statistic can produce and which reads as infinite significance. The
    crossover is not exotic: at a two-sided normal ``|z| = 9`` the correct
    p-value is 2.3e-19 and ``1 - cdf`` gives 0.0; a chi-square of 200 on 5
    degrees of freedom is 2.8e-41 and gives 0.0. Both are ordinary values for
    an over-identification test on a badly specified model, which is exactly
    when a reader looks at the p-value.

    ``dist.sf(x)`` is the same quantity evaluated without the cancellation, so
    this is a pure accuracy fix: no correctly-reported p-value changes.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "puremacro").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in _CDF_COMPLEMENT_ALLOWED:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _CDF_COMPLEMENT.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "p-values computed as `1 - cdf` collapse to exactly 0.0 in the tail; "
        "use `.sf(...)` instead:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# AST guards: Panel sorting before shift, kron orientation, and conditioning.
# ---------------------------------------------------------------------------

def test_panel_lp_sorts_before_positional_shift():
    """Panel estimators that compute lags/leads via positional shift must sort first.

    Shifting an unsorted panel silently crosses entity or temporal boundaries,
    producing corrupt design matrices without raising any error. Every panel
    helper in `puremacro/lp/` must call .sort_index() or .sort_values() before
    applying .shift().
    """
    root = Path(__file__).resolve().parents[1] / "puremacro" / "lp"
    for py_file in sorted(root.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_shift = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "shift"
                    for n in ast.walk(node)
                )
                args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
                is_panel = "entity_level" in args or "unit" in args or "panel" in args
                has_groupby = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "groupby"
                    for n in ast.walk(node)
                )
                if has_shift and (is_panel or has_groupby):
                    has_sort = any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("sort_index", "sort_values")
                        for n in ast.walk(node)
                    )
                    assert has_sort, (
                        f"{py_file.name}:{node.name} applies positional shift to "
                        "panel observations without explicitly calling sort_index or "
                        "sort_values first."
                    )


def test_weak_iv_vec_matches_kron_fortran_convention():
    """vec(Pi_hat) in weak_iv_rk_f must use order='F' to match np.kron."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "puremacro" / "inference" / "weak_iv.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "weak_iv_rk_f":
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "ravel":
                    kw = {k.arg: ast.literal_eval(k.value) for k in n.keywords if isinstance(k.value, ast.Constant)}
                    assert kw.get("order") == "F", (
                        "weak_iv_rk_f matrix vectorization must specify order='F' "
                        "to align with np.kron's column-major vec convention."
                    )


def test_linear_algebra_uses_relative_conditioning_thresholds():
    """Inversion and Cholesky singularity checks must be scale-invariant."""
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "puremacro" / "_linalg.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("inv_xtx", "safe_cholesky"):
            comparisons = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
            has_rel = any(
                isinstance(c.left, ast.Call)
                and getattr(c.left.func, "attr", "") == "min"
                and any(
                    isinstance(r, ast.BinOp) and isinstance(r.op, ast.Mult)
                    for r in c.comparators
                )
                for c in comparisons
            )
            assert has_rel, (
                f"_linalg.py:{node.name} must use relative conditioning "
                "(scaled by maximum pivot) rather than a scale-dependent constant."
            )


def test_no_vacuous_or_empty_tests():
    """Every test function under tests/ must execute meaningful logic.

    An empty test function, a test containing only a docstring, or a test whose
    body is solely `pass` or `...` provides illusory green coverage while asserting
    nothing. This AST walk proves zero test functions in the suite are vacuous.
    """
    tests_dir = _TESTS
    vacuous = []
    for p in sorted(tests_dir.rglob("test_*.py")):
        if p.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                if not body:
                    vacuous.append((p.relative_to(_ROOT).as_posix(), node.name, "empty-or-docstring-only"))
                elif len(body) == 1:
                    stmt = body[0]
                    if isinstance(stmt, ast.Pass):
                        vacuous.append((p.relative_to(_ROOT).as_posix(), node.name, "pass-only"))
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                        vacuous.append((p.relative_to(_ROOT).as_posix(), node.name, "ellipsis-only"))

    assert not vacuous, (
        f"Found {len(vacuous)} vacuous/empty test function(s):\n  "
        + "\n  ".join(f"{f}:{name} ({reason})" for f, name, reason in vacuous)
    )


def test_kron_vec_identities_in_inference_and_dsge():
    """Verify Kronecker product and vec operations follow matrix calculus conventions.

    1. vec(A X B) = (B.T kron A) vec(X) (klein.py).
    2. vec(Z_t V_t') = V_t kron Z_t in Kleibergen-Paap rk Wald covariance (weak_iv.py).
    """
    import numpy as np
    from puremacro.inference.weak_iv import kleibergen_paap_f

    root = Path(__file__).resolve().parents[1]
    weak_iv_src = (root / "puremacro" / "inference" / "weak_iv.py").read_text(encoding="utf-8")
    assert "np.kron(Z[t], V[t])" not in weak_iv_src, (
        "weak_iv psi must follow column-major vec(Z_t V_t') = V_t kron Z_t, never Z kron V"
    )

    # Numerical check on k=2, l=3 heteroskedastic design where V_t kron Z_t != Z_t kron V_t:
    rng = np.random.default_rng(20261111)
    n, k, l = 120, 2, 3
    Z = rng.normal(size=(n, l))
    Pi = np.array([[0.8, 0.2], [-0.4, 0.7], [0.3, -0.5]])
    het = 0.5 + Z[:, 0] ** 2 + 0.7 * np.abs(Z[:, 2])
    X = Z @ Pi + rng.normal(size=(n, k)) * het[:, None]
    Y = X @ np.array([1.0, -0.5]) + rng.normal(size=n)

    Q_Z, _ = np.linalg.qr(Z, mode="reduced")
    V_res = X - Q_Z @ (Q_Z.T @ X)
    Sigma_VV = (V_res.T @ V_res) / (n - l)
    ew, ev = np.linalg.eigh(Sigma_VV)
    F_scale = (ev * (ew ** -0.5)) @ ev.T
    V_std = V_res @ F_scale
    Theta_hat = (Q_Z.T @ X) @ F_scale
    U, _, Vt = np.linalg.svd(Theta_hat, full_matrices=True)
    U_2 = U[:, k - 1 :]
    v_k = Vt[k - 1, :]
    ell_q = U_2.T @ (Theta_hat @ v_k)
    K_proj = np.kron(v_k[:, None], U_2)

    psi_right = np.array([np.kron(V_std[t], Q_Z[t]) for t in range(n)])
    omega_right = K_proj.T @ (psi_right.T @ psi_right) @ K_proj
    kp_expected = float(ell_q @ np.linalg.solve(omega_right, ell_q)) / l

    psi_wrong = np.array([np.kron(Q_Z[t], V_std[t]) for t in range(n)])
    omega_wrong = K_proj.T @ (psi_wrong.T @ psi_wrong) @ K_proj
    kp_wrong = float(ell_q @ np.linalg.solve(omega_wrong, ell_q)) / l

    kp_actual = kleibergen_paap_f(Y, X, Z)
    assert np.isclose(kp_actual, kp_expected, rtol=1e-12, atol=1e-12)
    assert not np.isclose(kp_actual, kp_wrong, rtol=1e-3, atol=1e-3)

    klein_src = (root / "puremacro" / "dsge" / "klein.py").read_text(encoding="utf-8")
    assert "np.kron(G.T, A2)" in klein_src, "klein.py vectorization must use np.kron(G.T, A2)"


def test_bvar_uses_relative_cholesky_jitter():
    """BVAR posterior sampling must use scale-invariant jitter."""
    root = Path(__file__).resolve().parents[1]
    bvar_src = (root / "puremacro" / "var" / "bvar.py").read_text(encoding="utf-8")
    assert "safe_cholesky(XtX_inv" in bvar_src, (
        "bvar.py must use safe_cholesky for XtX_inv factor rather than unscaled np.linalg.cholesky"
    )
    assert "XtX_inv + 1e-12 * np.eye" not in bvar_src, (
        "bvar.py must not use hardcoded 1e-12 * eye without scaling by matrix magnitude"
    )



# ---------------------------------------------------------------------------
# Wall-clock guard: every absolute timing budget must scale.
# ---------------------------------------------------------------------------
#
# Shared CI runners are several times slower and noisier than a developer
# machine, so ``tests/_timing.py`` widens every absolute budget by
# ``PUREMACRO_TIMING_SCALE``. A timing measurement compared with a bare number
# (``elapsed < 0.2``, or a module constant such as ``CEILING_S = 1.0``) skips
# that scaling and turns into a flake the first time a runner is busy. This
# scan finds such comparisons by data flow: a name is a *timing* quantity when
# it is assigned from a clock call (``time.perf_counter()`` and friends), from a
# ``*latency*`` attribute, or from an expression over other timing names; the
# ratio of two timing quantities is dimensionless and deliberately not scaled.

_CLOCK_FUNCS = frozenset({"perf_counter", "perf_counter_ns", "monotonic",
                          "monotonic_ns", "time", "process_time"})

#: Files allowed to compare a timing with a literal, with the reason.
_UNSCALED_TIMING_ALLOWED: dict[str, str] = {
    # Opt-in ``slow`` smoke test; the 600 s ceiling is a hang watchdog two
    # orders of magnitude above the measured run time.
    "tests/test_dsge/test_sw07_estimate_smoke.py": "slow opt-in hang watchdog",
}


def _is_clock_call(node: ast.AST) -> bool:
    """A clock read, or ``best_of(...)`` (which returns a duration)."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if (isinstance(f, ast.Attribute) and f.attr in _CLOCK_FUNCS
            and isinstance(f.value, ast.Name) and f.value.id == "time"):
        return True
    if isinstance(f, ast.Attribute) and f.attr == "best_of":
        return True
    return isinstance(f, ast.Name) and f.id in ("perf_counter", "monotonic", "best_of")


def _is_timing_expr(node: ast.AST, names: set[str]) -> bool:
    if _is_clock_call(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return "latency" in node.attr.lower()
    if isinstance(node, ast.BinOp):
        left = _is_timing_expr(node.left, names)
        right = _is_timing_expr(node.right, names)
        if isinstance(node.op, ast.Div) and left and right:
            return False                        # a ratio: dimensionless
        return left or right
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Attribute)
                and _is_timing_expr(node.func.value, names)):
            return True                         # e.g. latencies.mean()
        return any(_is_timing_expr(a, names) for a in node.args)
    if isinstance(node, ast.Subscript):
        return _is_timing_expr(node.value, names)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return _is_timing_expr(node.elt, names)
    return False


def _timing_names(scope: ast.AST) -> set[str]:
    """Names in ``scope`` holding a timing quantity (iterated to a fixed point)."""
    names: set[str] = set()
    for _ in range(25):
        new: set[str] = set()
        for n in ast.walk(scope):
            if (isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and n.value is not None and _is_timing_expr(n.value, names)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    base = t.value if isinstance(t, ast.Subscript) else t
                    if isinstance(base, ast.Name):
                        new.add(base.id)
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "append" and isinstance(n.func.value, ast.Name)
                  and n.args and _is_timing_expr(n.args[0], names)):
                new.add(n.func.value.id)
        if new == names:
            break
        names = new
    return names


def _is_numeric_literal(node: ast.AST, consts: set[str]) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    if isinstance(node, ast.Name):
        return node.id in consts
    if isinstance(node, ast.Attribute):
        return node.attr in consts
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool))


def _module_numeric_constants(tree: ast.Module) -> set[str]:
    """Module- and class-level names bound to a bare number (``CEILING_S = 1.0``)."""
    consts: set[str] = set()
    bodies = [tree.body] + [c.body for c in tree.body if isinstance(c, ast.ClassDef)]
    for body in bodies:
        for st in body:
            if (isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None
                    and _is_numeric_literal(st.value, set())):
                for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                    if isinstance(t, ast.Name):
                        consts.add(t.id)
    return consts


def _unscaled_timing_comparisons(src: str) -> list[int]:
    """Line numbers where a timing quantity is compared with a bare number."""
    tree = ast.parse(src)
    consts = _module_numeric_constants(tree)
    hits: set[int] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = _timing_names(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare):
                operands = [n.left, *n.comparators]
                if (any(_is_timing_expr(o, names) for o in operands)
                        and any(_is_numeric_literal(o, consts) for o in operands)):
                    hits.add(n.lineno)
    return sorted(hits)


def test_wall_clock_budgets_go_through_the_timing_helper():
    """No test compares a wall-clock measurement with an unscaled number.

    Every absolute budget must go through ``budget()``/``budget_ms()`` from
    ``tests/_timing.py`` so ``PUREMACRO_TIMING_SCALE`` widens it on CI. The one
    CI flake that failed 72 runs in a month was a bare ``0.20`` here.
    """
    offenders = []
    for path in sorted(_TESTS.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        if path.name == Path(__file__).name or rel in _UNSCALED_TIMING_ALLOWED:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            lines = _unscaled_timing_comparisons(src)
        except SyntaxError as exc:              # never pass a file unread
            offenders.append(f"{rel}:{exc.lineno}: could not parse ({exc.msg})")
            continue
        offenders += [f"{rel}:{ln}" for ln in lines]
    assert not offenders, (
        "wall-clock measurements compared with an unscaled number; wrap the "
        "budget in budget()/budget_ms() from tests/_timing.py (ratios of two "
        "timings are exempt):\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("snippet", [
    "def test_x():\n    t0 = time.perf_counter()\n    f()\n"
    "    elapsed = time.perf_counter() - t0\n    assert elapsed < 0.5\n",
    "def test_x():\n    times = []\n    for _ in range(3):\n"
    "        t0 = time.perf_counter(); f()\n"
    "        times.append(time.perf_counter() - t0)\n    assert max(times) <= 0.2\n",
    "def test_x():\n    t0 = time.perf_counter(); f()\n"
    "    assert time.perf_counter() - t0 < 3.5\n",
    "CEILING_S = 1.0\ndef test_x():\n    t0 = time.time(); f()\n"
    "    dt = time.time() - t0\n    assert dt <= CEILING_S\n",
    "def test_x(res):\n    lat = np.array([res.last_latency_ms for _ in range(3)])\n"
    "    assert float(lat.mean()) < 15.0\n",
    "def test_x(res):\n    assert res.last_latency_ms < 15.0\n",
    "def test_x():\n    assert 1000.0 * best_of(f) < 10.0\n",
], ids=["delta", "max-of-list", "inline", "module-constant", "latency-mean",
        "latency-attr", "best-of"])
@pytest.mark.mechanism_control
def test_the_timing_guard_bites(snippet):
    """Negative control: the scan must flag each known unscaled shape."""
    assert _unscaled_timing_comparisons(snippet), snippet


@pytest.mark.parametrize("snippet", [
    "def test_x():\n    t0 = time.perf_counter(); f()\n"
    "    elapsed = time.perf_counter() - t0\n    assert elapsed <= budget(0.5)\n",
    "def test_x():\n    t_a = best_of(f)\n    t_b = best_of(g)\n"
    "    speedup = t_b / t_a\n    assert speedup >= 5.0\n",
    "def test_x(res):\n    assert res.iterations < 25\n",
], ids=["scaled", "ratio", "not-timing"])
def test_the_timing_guard_passes_scaled_budgets_and_ratios(snippet):
    assert not _unscaled_timing_comparisons(snippet), snippet


@pytest.mark.mechanism_control
def test_the_timing_guard_bites_on_a_real_file_with_its_budget_removed():
    """Stripping ``budget(...)`` from a repaired file must make the scan fire."""
    src = (_TESTS / "test_hank_regressions.py").read_text(encoding="utf-8")
    assert "budget(" in src and not _unscaled_timing_comparisons(src)
    stripped = re.sub(r"budget\(([^()]*)\)", r"\1", src)
    assert _unscaled_timing_comparisons(stripped)
