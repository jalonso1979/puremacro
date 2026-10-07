"""STEADY: structurally singular steady states are no longer solved silently.

Before the fix, ``steady(solve_algo="block")`` answered an incomplete
equation-variable matching with a full-system ``hybr`` solve and, when that
converged, returned its point with no warning. For an under-determined system
that point is arbitrary: in the reviewers' case A below, ``z`` appears in no
equation and came back as ``guess + 0.769133``. The documented contract
(``docs/dsge_estimation.md``, "Dulmage-Mendelsohn Singularity Decomposition") is a
``StructuralSingularityError``.

After the fix:
- general singular structure -> StructuralSingularityError unless
  ``allow_singular=True`` / ``allow_structural_singularity()``;
- identity rows only (the static ``0 = 0`` of a unit-root law of motion, the case
  seven existing tests build through ``build()``) -> the hybr point with a
  StructuralSingularityWarning, or an error under ``allow_singular=False``;
- every singular return is recorded in ``info``; the full-system polish is flagged.

All expected values are worked out by hand in the comments.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import puremacro.dsge as dsge
from puremacro.dsge.build import SteadyStateError, build
from puremacro.dsge.steady import (
    StructuralSingularityError,
    StructuralSingularityWarning,
    _Vec,
    allow_structural_singularity,
    steady,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def consistent_singular(xp, x, e, p):
    """x = 1 twice (consistent over-determination), y = -x, z free.

    Solution set: {(1, -1, z) : z real}. Matching size 2 / 3; DM sets:
    over-determined equations {e1, e2} on {x}, under-determined variable {z}.
    """
    return [x.x - 1.0, 2.0 * x.x - 2.0, x.x + x.y]


def inconsistent_singular(xp, x, e, p):
    """x = 1 and x = -2: no solution at all (the repo's Validation 8 system)."""
    return [x.x - 1.0, x.x + 2.0, x.x + x.y]


def random_walk(xp, x, e, p):
    """y' = y + eps, c = 2y: any (y, 2y) is a steady state."""
    return [xp.y - x.y - e.eps, x.c - 2.0 * x.y]


def growth(xp, x, e, p):
    """Neoclassical growth with log utility and full depreciation.

    With rho = 1 the TFP law of motion is z = z in steady state (a 0 = 0 row),
    so z is not pinned and c, k follow z:
    k = (alpha * beta * z)^(1/(1-alpha)), c = z k^alpha - k.
    """
    return [
        1.0 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1.0)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]


GROWTH_RW_PARAMS = dict(alpha=0.33, beta=0.98, rho=1.0)
GROWTH_GUESS = dict(c=0.5, k=0.1, z=1.0)


def kink(xp, x, e, p):
    """x = 1 + max(0, y - 5), y = 10 -> (6, 10). Flat in y near y = 1."""
    return [x.x - 1.0 - max(0.0, x.y - 5.0), x.y - 10.0]


def flat_near_guess(xp, x, e, p):
    """y = 2 and (y - 1)^+ (x - 3) = 0 -> unique root (3, 2).

    At the guess y = 0 the second equation is identically zero in a
    neighbourhood, so the numeric incidence has an empty row there (matching
    1 / 2); at the root both entries are present (matching 2 / 2).
    """
    return [x.y - 2.0, max(0.0, x.y - 1.0) * (x.x - 3.0)]


# ---------------------------------------------------------------------------
# Default: the documented contract (raise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z_guess", [1.0, 7.5, -123.0])
def test_consistent_underdetermined_system_raises_by_default(z_guess):
    """Pre-fix: returned (1, -1, z_guess + 0.769133) with no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning may escape either
        with pytest.raises(StructuralSingularityError) as exc_info:
            steady(
                consistent_singular, ["x", "y", "z"],
                {"x": 0.3, "y": 0.2, "z": z_guess}, None,
                equation_names=["e1", "e2", "e3"],
            )
    err = exc_info.value
    assert isinstance(err, SteadyStateError)
    assert err.matching_size == 2
    assert err.n_equations == 3 and err.n_variables == 3
    assert set(err.overdetermined_equations) == {"e1", "e2"}
    assert set(err.overdetermined_variables) == {"x"}
    assert set(err.underdetermined_variables) == {"z"}
    msg = str(err)
    assert "structurally singular in steady state" in msg
    assert "allow_singular=True" in msg
    assert "allow_structural_singularity" in msg


def test_explicit_allow_singular_false_overrides_context():
    with allow_structural_singularity():
        with pytest.raises(StructuralSingularityError):
            steady(consistent_singular, ["x", "y", "z"], [0.3, 0.2, 1.0], None,
                   allow_singular=False)


def test_build_random_walk_warns_by_default():
    """Pre-fix: build() returned y=1.5, c=3.0 from guess (4, 3) with no warning.

    The only over-determined equation is the 0 = 0 row of y' = y + eps, so the
    default accepts the hybr point but says so.
    """
    with pytest.warns(StructuralSingularityWarning, match="unit-root") as rec:
        m = build(random_walk, variables=["y", "c"], states=["y"], shocks=["eps"],
                  params={}, guess={"y": 4.0, "c": 3.0}, linearize="level")
    # Attributed to the build() call in this file.
    assert any(r.filename == __file__ for r in rec if r.category is StructuralSingularityWarning)
    ss = m.steady_state
    assert ss["c"] == pytest.approx(2.0 * ss["y"], abs=1e-10)
    # Same numbers as before the fix (hybr is unchanged): (1.5, 3.0).
    assert ss["y"] == pytest.approx(1.5, abs=1e-10)
    assert ss["c"] == pytest.approx(3.0, abs=1e-10)


def test_build_random_walk_strict_mode_raises():
    with allow_structural_singularity(False):
        with pytest.raises(StructuralSingularityError) as exc_info:
            build(random_walk, variables=["y", "c"], states=["y"], shocks=["eps"],
                  params={}, guess={"y": 4.0, "c": 3.0}, linearize="level")
    err = exc_info.value
    assert set(err.underdetermined_variables) == {"y", "c"}
    assert err.overdetermined_variables == ()          # the 0 = 0 row
    assert len(err.overdetermined_equations) == 1
    # Strictness ends with the block: the default accepts identity rows again.
    with pytest.warns(StructuralSingularityWarning):
        build(random_walk, variables=["y", "c"], states=["y"], shocks=["eps"],
              params={}, guess={"y": 4.0, "c": 3.0}, linearize="level")


def test_identity_rows_steady_records_info_and_strict_flag():
    with pytest.warns(StructuralSingularityWarning):
        x, info = steady(random_walk, ["y", "c"], {"y": 4.0, "c": 3.0}, None, shocks=["eps"])
    assert info["structurally_singular"] is True
    assert info["singularity_kind"] == "identity_rows"
    assert info["overdetermined_variables"] == []
    assert info["underdetermined_variables"] == ["y", "c"]
    with pytest.raises(StructuralSingularityError):
        steady(random_walk, ["y", "c"], {"y": 4.0, "c": 3.0}, None, shocks=["eps"],
               allow_singular=False)


def test_inconsistent_system_raises_even_with_opt_in():
    """No point satisfies x = 1 and x = -2, so the opt-in cannot help."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(StructuralSingularityError) as exc_info:
            steady(inconsistent_singular, ["x", "y", "z"], [1.0, 1.0, 1.0], None,
                   equation_names=["e1", "e2", "e3"], allow_singular=True)
    assert "did not converge either" in str(exc_info.value)
    assert set(exc_info.value.underdetermined_variables) == {"z"}


# ---------------------------------------------------------------------------
# Opt-in: warn and record
# ---------------------------------------------------------------------------

def test_allow_singular_true_warns_and_records_info():
    with pytest.warns(StructuralSingularityWarning, match="not unique") as rec:
        x, info = steady(
            consistent_singular, ["x", "y", "z"], {"x": 0.3, "y": 0.2, "z": 1.0}, None,
            equation_names=["e1", "e2", "e3"], allow_singular=True,
        )
    assert issubclass(StructuralSingularityWarning, RuntimeWarning)
    # Attributed to this file, not to puremacro internals.
    assert rec[0].filename == __file__
    # x and y are pinned; z is whatever hybr reached.
    assert x[0] == pytest.approx(1.0, abs=1e-10)
    assert x[1] == pytest.approx(-1.0, abs=1e-10)
    assert np.max(np.abs(consistent_singular(None, _Vec(["x", "y", "z"], x), None, None))) <= 1e-8
    assert info["structurally_singular"] is True
    assert info["fallback_from_block"] is True
    assert info["fallback_reason"] == "structural_singularity"
    assert info["solve_algo"] == "hybr"
    assert info["matching_size"] == 2
    assert info["underdetermined_variables"] == ["z"]
    assert info["overdetermined_equations"] == ["e1", "e2"]
    assert "z" in info["singularity_diagnosis"]


def statically_duplicated_model(xp, x, e, p):
    """k' = 0.5 k + eps; w' = w - k' + 0.5 k; c = 0.5 k.

    Dynamically regular (w is a unit-root state), but in the steady state the
    second equation reduces to 0.5 k = 0, a second equation in k alone, and w
    cancels: over-determined {eq_1, eq_2} on {k}, under-determined {w}. That is a
    general singular structure (not a pure 0 = 0 row). Steady states:
    k = c = 0, w free.
    """
    return [
        xp.k - 0.5 * x.k - e.eps,
        xp.w - x.w + xp.k - 0.5 * x.k,
        x.c - 0.5 * x.k,
    ]


def test_context_manager_reaches_build_and_scopes_the_opt_in():
    kw = dict(variables=["k", "w", "c"], states=["k", "w"], shocks=["eps"], params={},
              guess={"k": 0.3, "w": 1.0, "c": 0.2}, linearize="level")
    # General singularity: raises by default.
    with pytest.raises(StructuralSingularityError) as exc_info:
        build(statically_duplicated_model, **kw)
    assert set(exc_info.value.underdetermined_variables) == {"w"}
    assert set(exc_info.value.overdetermined_variables) == {"k"}
    with allow_structural_singularity():
        with pytest.warns(StructuralSingularityWarning) as rec:
            m = build(statically_duplicated_model, **kw)
    # Warning points at the build() call in this file.
    assert any(r.filename == __file__ for r in rec if r.category is StructuralSingularityWarning)
    ss = m.steady_state
    assert ss["k"] == pytest.approx(0.0, abs=1e-10)
    assert ss["c"] == pytest.approx(0.0, abs=1e-10)
    # The opt-in ended with the block.
    with pytest.raises(StructuralSingularityError):
        build(statically_duplicated_model, **kw)


def test_growth_unit_root_solves_and_check_reports_unit_root():
    """The case test_diagnostics.py::test_check_unit_roots relies on (default mode)."""
    with pytest.warns(StructuralSingularityWarning, match="z"):
        m = dsge.build(growth, variables=["c", "k", "z"], states=["k", "z"],
                       shocks=["eps"], params=GROWTH_RW_PARAMS, guess=GROWTH_GUESS)
    ss = m.steady_state
    a, b = GROWTH_RW_PARAMS["alpha"], GROWTH_RW_PARAMS["beta"]
    k = (a * b * ss["z"]) ** (1.0 / (1.0 - a))
    assert ss["k"] == pytest.approx(k, rel=1e-8)
    assert ss["c"] == pytest.approx(ss["z"] * k ** a - k, rel=1e-8)
    assert "Unit roots   : 1" in m.check().summary()


def test_supplying_the_steady_state_needs_no_opt_in():
    """The documented alternative for unit-root models: steady_state=."""
    with pytest.warns(StructuralSingularityWarning):
        x, _ = steady(growth, ["c", "k", "z"], GROWTH_GUESS, GROWTH_RW_PARAMS,
                      shocks=["eps"], allow_singular=True)
    ss = dict(zip(["c", "k", "z"], x))
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralSingularityWarning)
        m = dsge.build(growth, variables=["c", "k", "z"], states=["k", "z"],
                       shocks=["eps"], params=GROWTH_RW_PARAMS, steady_state=ss)
    assert m.steady_state["z"] == pytest.approx(ss["z"])


def test_homotopy_passes_structural_error_through_and_records_opt_in():
    def singular_with_param(xp, x, e, p):
        return [x.x - p.a, 2.0 * x.x - 2.0 * p.a, x.x + x.y]

    with pytest.raises(StructuralSingularityError):
        steady(singular_with_param, ["x", "y", "z"], [0.3, 0.2, 1.0], {"a": 1.0},
               homotopy={"a": (1.0, 2.0)}, homotopy_steps=2)

    with pytest.warns(StructuralSingularityWarning):
        x, info = steady(singular_with_param, ["x", "y", "z"], [0.3, 0.2, 1.0], {"a": 1.0},
                         homotopy={"a": (1.0, 2.0)}, homotopy_steps=2, allow_singular=True)
    assert x[0] == pytest.approx(2.0, abs=1e-8)
    assert x[1] == pytest.approx(-2.0, abs=1e-8)
    assert info["structurally_singular"] is True
    assert info["underdetermined_variables"] == ["z"]


# ---------------------------------------------------------------------------
# Regular models: unchanged answers, explicit flags, no warnings
# ---------------------------------------------------------------------------

def test_regular_model_flags_and_no_warning():
    params = dict(alpha=0.33, beta=0.98, rho=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, info = steady(growth, ["c", "k", "z"], GROWTH_GUESS, params, shocks=["eps"])
    a, b = params["alpha"], params["beta"]
    k = (a * b) ** (1.0 / (1.0 - a))
    np.testing.assert_allclose(x, [k ** a - k, k, 1.0], rtol=1e-10)
    assert info["solve_algo"] == "block"
    assert info["structurally_singular"] is False
    assert info["fallback_from_block"] is False
    assert info["full_system_polish"] is False
    assert info["matching_size"] == 3


def test_full_system_polish_is_recorded():
    """Pre-fix: (6, 10) with solve_algo='block' and no trace of the hybr polish."""
    x, info = steady(kink, ["x", "y"], {"x": 1.0, "y": 1.0}, None)
    np.testing.assert_allclose(x, [6.0, 10.0], atol=1e-10)
    assert info["solve_algo"] == "block"
    assert info["full_system_polish"] is True
    assert info["structurally_singular"] is False


def test_incomplete_matching_only_at_guess_is_not_an_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralSingularityWarning)
        x, info = steady(flat_near_guess, ["x", "y"], {"x": 0.0, "y": 0.0}, None)
    np.testing.assert_allclose(x, [3.0, 2.0], atol=1e-8)
    assert info["structurally_singular"] is False
    assert info["fallback_from_block"] is True
    assert info["fallback_reason"] == "incomplete_matching_at_guess"
    assert info["matching_size_at_guess"] == 1
    assert info["matching_size"] == 2


def test_nonfinite_guess_keeps_solver_error():
    def log_eq(xp, x, e, p):
        return [np.log(x.x) - 1.0, x.y - 2.0]

    with np.errstate(all="ignore"), pytest.raises(SteadyStateError) as exc_info:
        steady(log_eq, ["x", "y"], {"x": 0.0, "y": 0.0}, None)
    assert not isinstance(exc_info.value, StructuralSingularityError)


def test_direct_solvers_unchanged():
    """solve_algo='hybr' does no structural analysis (compatibility setting)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralSingularityWarning)
        x, info = steady(consistent_singular, ["x", "y", "z"], [0.3, 0.2, 1.0], None,
                         solve_algo="hybr")
    assert info["solve_algo"] == "hybr"
    assert "structurally_singular" not in info
    assert x[0] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Review round 1: the printed opt-in must work, homotopy keeps the diagnosis,
# the scope of the structural check is stated, the polish is logged.
# ---------------------------------------------------------------------------

import re  # noqa: E402

_DUP_KW = dict(variables=["k", "w", "c"], states=["k", "w"], shocks=["eps"], params={},
               guess={"k": 0.3, "w": 1.0, "c": 0.2}, linearize="level")


def _snippets(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def test_opt_in_hint_printed_in_the_error_runs_as_written():
    """Round-1 review: the hint said ``with puremacro.dsge.steady.allow_...():``,
    which raised AttributeError (``puremacro.dsge.steady`` is the function).

    Execute the backticked snippets of a real error message, in order, around a
    build() of a general singular model: it must build (with the warning).
    """
    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(consistent_singular, ["x", "y", "z"], [0.3, 0.2, 1.0], None)
    snippets = _snippets(str(exc_info.value))
    imports = [s for s in snippets if s.startswith("from ") or s.startswith("import ")]
    withs = [s for s in snippets if s.startswith("with ")]
    assert imports and withs, snippets
    ns: dict = {"build": build, "model": statically_duplicated_model, "kw": _DUP_KW}
    for s in imports:
        exec(s, ns)
    code = withs[0] + "\n    m = build(model, **kw)\n"
    with pytest.warns(StructuralSingularityWarning):
        exec(code, ns)
    assert ns["m"].steady_state["k"] == pytest.approx(0.0, abs=1e-10)


def test_strict_hint_printed_in_the_warning_runs_as_written():
    with pytest.warns(StructuralSingularityWarning) as rec:
        steady(random_walk, ["y", "c"], {"y": 4.0, "c": 3.0}, None, shocks=["eps"])
    msg = str(next(r.message for r in rec if r.category is StructuralSingularityWarning))
    withs = [s for s in _snippets(msg) if s.startswith("with ")]
    assert withs, msg
    ns: dict = {"build": build, "model": random_walk}
    exec("from puremacro.dsge.steady import allow_structural_singularity", ns)
    code = (withs[0] + "\n    build(model, variables=['y', 'c'], states=['y'], shocks=['eps'],"
            " params={}, guess={'y': 4.0, 'c': 3.0}, linearize='level')\n")
    with pytest.raises(StructuralSingularityError):
        exec(code, ns)


def test_dotted_path_through_the_package_works():
    import puremacro

    st = puremacro.dsge.steady          # the function, which shadows the submodule
    assert st.allow_structural_singularity is allow_structural_singularity
    assert st.StructuralSingularityWarning is StructuralSingularityWarning
    assert st.StructuralSingularityError is StructuralSingularityError
    with puremacro.dsge.steady.allow_structural_singularity():
        with pytest.warns(StructuralSingularityWarning):
            build(statically_duplicated_model, **_DUP_KW)


def test_error_carries_the_hybr_point():
    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(consistent_singular, ["x", "y", "z"], [0.3, 0.2, 1.0], None)
    xp = exc_info.value.hybr_point
    assert xp is not None and xp.shape == (3,)
    assert xp[0] == pytest.approx(1.0, abs=1e-10)
    assert xp[1] == pytest.approx(-1.0, abs=1e-10)
    # Inconsistent system: hybr did not converge, the diagnosis is at the guess.
    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(inconsistent_singular, ["x", "y", "z"], [1.0, 1.0, 1.0], None,
               allow_singular=True)
    assert exc_info.value.hybr_point is None


def singular_at_homotopy_target(xp, x, e, p):
    """x = 1, y = -x, (1 - a) z + a (x - 1) = 0.

    For a < 1: z = 0 (unique). At a = 1 the third equation is x - 1 again: z free,
    over-determined {eq_1, eq_3} on {x}.
    """
    return [x.x - 1.0, x.y + x.x, (1.0 - p.a) * x.z + p.a * (x.x - 1.0)]


def test_homotopy_to_a_singular_target_raises_the_structural_error():
    """Round-1 review: the loop caught the error as a SteadyStateError, bisected
    about 13 times and raised 'Homotopy continuation stalled at s=0.9999'."""
    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(singular_at_homotopy_target, ["x", "y", "z"], [0.3, 0.2, 0.5], {"a": 0.0},
               homotopy={"a": (0.0, 1.0)})
    err = exc_info.value
    assert "stalled" not in str(err)
    assert set(err.underdetermined_variables) == {"z"}
    assert set(err.overdetermined_variables) == {"x"}
    assert any("homotopy target s=1" in n for n in getattr(err, "__notes__", []))
    # With the opt-in, the pre-fix answer (1, -1, 0): z carried over from a < 1.
    with pytest.warns(StructuralSingularityWarning):
        x, info = steady(singular_at_homotopy_target, ["x", "y", "z"], [0.3, 0.2, 0.5],
                         {"a": 0.0}, homotopy={"a": (0.0, 1.0)}, allow_singular=True)
    np.testing.assert_allclose(x, [1.0, -1.0, 0.0], atol=1e-8)
    assert info["structurally_singular"] is True
    assert info["underdetermined_variables"] == ["z"]


def _gap(a):
    """Zero for 0.4 <= a <= 0.8, positive outside."""
    return max(0.0, 0.4 - a) + max(0.0, a - 0.8)


def duplicated_on_an_interval(xp, x, e, p):
    """x = 1, y = -x, g(a) z + (x - 1) = 0 with g = 0 on [0.4, 0.8]: there the third
    equation repeats x = 1 and z is free (a general singular structure)."""
    return [x.x - 1.0, x.y + x.x, _gap(p.a) * x.z + (x.x - 1.0)]


def test_homotopy_stall_on_a_singular_interval_keeps_the_diagnosis():
    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(duplicated_on_an_interval, ["x", "y", "z"], [1.0, -1.0, 0.0], {"a": 0.0},
               homotopy={"a": (0.0, 1.0)})
    err = exc_info.value
    assert "stalled" in str(err) and "structurally singular" in str(err)
    assert set(err.underdetermined_variables) == {"z"}
    assert isinstance(err.__cause__, StructuralSingularityError)


def identity_row_on_an_interval(xp, x, e, p):
    """x = 1, y = -x, g(a) (z - 1) = 0: a 0 = 0 row for a in [0.4, 0.8]; z = 1 at a = 1."""
    return [x.x - 1.0, x.y + x.x, _gap(p.a) * (x.z - 1.0)]


def test_homotopy_intermediate_singularity_is_recorded_apart():
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralSingularityWarning)
        x, info = steady(identity_row_on_an_interval, ["x", "y", "z"], [1.0, -1.0, 1.0],
                         {"a": 0.0}, homotopy={"a": (0.0, 1.0)}, homotopy_steps=10)
    np.testing.assert_allclose(x, [1.0, -1.0, 1.0], atol=1e-8)
    assert info["structurally_singular"] is False
    assert info["structurally_singular_on_path"] is True
    assert info["matching_size"] == 3                      # the returned point
    assert info["path_singularity"]["singularity_kind"] == "identity_rows"
    assert info["path_singularity"]["underdetermined_variables"] == ["z"]
    assert "underdetermined_variables" not in info


def collinear(xp, x, e, p):
    """x + y = 2 and 2x + 2y = 4: complete incidence, rank-1 Jacobian."""
    return [x.x + x.y - 2.0, 2.0 * x.x + 2.0 * x.y - 4.0]


def test_numerically_collinear_equations_are_outside_the_structural_check():
    """Documented scope: the check is structural, like Dynare's dmperm. The
    solution set is the line x + y = 2 and hybr returns a guess-dependent point."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralSingularityWarning)
        x1, info1 = steady(collinear, ["x", "y"], [0.3, 5.0], None)
        x2, _ = steady(collinear, ["x", "y"], [3.0, 1.0], None)
    assert info1["structurally_singular"] is False
    assert info1["matching_size"] == 2
    assert x1.sum() == pytest.approx(2.0, abs=1e-8)
    assert x2.sum() == pytest.approx(2.0, abs=1e-8)
    assert abs(x1[0] - x2[0]) > 1e-3                      # not locally unique


def test_full_system_polish_is_logged(caplog):
    with caplog.at_level("INFO", logger="puremacro.dsge.steady"):
        steady(kink, ["x", "y"], {"x": 1.0, "y": 1.0}, None)
    assert any("full nonlinear problem" in r.getMessage() for r in caplog.records)
