"""The QZ reordering must not depend on which LAPACK the machine happens to ship.

``ztgsen`` in the OpenBLAS that ships with the numpy 2.5 wheels refuses to
reorder the four-variable block model below -- generalised eigenvalues 0.5,
0.6, 1.11, 1.18, nothing degenerate about it -- while ``dtgsen`` handles the
same pencil and the same selection. Accelerate and the CI runners do both. So
these tests simulate the refusal rather than depending on the local build:
whichever LAPACK runs them, the fallback path is exercised.
"""
import numpy as np
import pytest
import scipy.linalg

from puremacro.dsge._qz import ordqz_sorted
from puremacro.dsge.gensys import gensys

REORDER_ERROR = (
    "Reordering of (A, B) failed because the transformed matrix pair (A, B) "
    "would be too far from generalized Schur form; the problem is very "
    "ill-conditioned."
)


def _block_model():
    """Two orthogonal 2-variable subsystems; the pencil ztgsen chokes on."""
    rho1, rho2, b1, b2 = 0.6, 0.5, 0.9, 0.85
    G0 = np.zeros((4, 4))
    G0[0, 0] = G0[1, 1] = 1.0
    G0[2, 2], G0[3, 3] = b1, b2
    G1 = np.zeros((4, 4))
    G1[0, 0], G1[1, 1] = rho1, rho2
    G1[2, 0], G1[2, 2] = -1.0, 1.0
    G1[3, 1], G1[3, 3] = -1.0, 1.0
    Psi = np.zeros((4, 2))
    Psi[0, 0] = Psi[1, 1] = 1.0
    Pi = np.zeros((4, 2))
    Pi[2, 0], Pi[3, 1] = b1, b2
    return G0, G1, Psi, Pi


def _break_complex(monkeypatch):
    """Make the complex reordering fail exactly as the broken build does."""
    real_ordqz = scipy.linalg.ordqz

    def fake(A, B, sort=None, output="real", **kw):
        if output == "complex":
            raise ValueError(REORDER_ERROR)
        return real_ordqz(A, B, sort=sort, output=output, **kw)

    monkeypatch.setattr(scipy.linalg, "ordqz", fake)


def test_falls_back_to_real_arithmetic(monkeypatch):
    G0, G1, _, _ = _block_model()
    _break_complex(monkeypatch)

    S, T, alpha, beta, Q, Z = ordqz_sorted(
        G0, G1, lambda a, b: abs(b) < abs(a), output="complex")

    assert not np.iscomplexobj(S)                       # the retry ran
    assert np.allclose(Q @ S @ Z.conj().T, G0, atol=1e-12)
    assert np.allclose(Q @ T @ Z.conj().T, G1, atol=1e-12)


def test_stable_eigenvalues_still_come_first(monkeypatch):
    G0, G1, _, _ = _block_model()
    _break_complex(monkeypatch)

    _, _, alpha, beta, _, _ = ordqz_sorted(
        G0, G1, lambda a, b: abs(b) < abs(a), output="complex")

    with np.errstate(divide="ignore", invalid="ignore"):
        mod = np.where(np.abs(alpha) > 1e-12, np.abs(beta / alpha), np.inf)
    assert np.all(mod[:2] < 1.0) and np.all(mod[2:] > 1.0)


def test_gensys_gives_the_same_answer_through_either_arithmetic(monkeypatch):
    """The point of the exercise: the model solves the same either way."""
    G0, G1, Psi, Pi = _block_model()
    healthy = gensys(G0, G1, Psi, Pi)

    _break_complex(monkeypatch)
    fallen_back = gensys(G0, G1, Psi, Pi)

    assert fallen_back.eu == healthy.eu == (1, 1)
    assert np.allclose(fallen_back.G, healthy.G, atol=1e-12)
    assert np.allclose(fallen_back.Impact, healthy.Impact, atol=1e-12)
    assert np.allclose(fallen_back.eigenvalues, healthy.eigenvalues, atol=1e-12)


def test_a_value_error_that_is_not_a_reordering_failure_propagates(monkeypatch):
    def fake(*a, **kw):
        raise ValueError("sort must be a callable or one of 'lhp', 'rhp', ...")

    monkeypatch.setattr(scipy.linalg, "ordqz", fake)
    G0, G1, _, _ = _block_model()
    with pytest.raises(ValueError, match="sort must be"):
        ordqz_sorted(G0, G1, lambda a, b: abs(b) < abs(a), output="complex")


def test_failure_in_both_arithmetics_raises_a_diagnosis(monkeypatch):
    def fake(*a, **kw):
        raise ValueError(REORDER_ERROR)

    monkeypatch.setattr(scipy.linalg, "ordqz", fake)
    G0, G1, _, _ = _block_model()
    with pytest.raises(np.linalg.LinAlgError, match="complex or in real"):
        ordqz_sorted(G0, G1, lambda a, b: abs(b) < abs(a), output="complex")


def test_the_diagnosis_names_a_genuinely_degenerate_pencil(monkeypatch):
    def fake(*a, **kw):
        raise ValueError(REORDER_ERROR)

    monkeypatch.setattr(scipy.linalg, "ordqz", fake)
    # A common null space makes alpha and beta vanish together: the
    # stable/unstable split really is undefined there.
    A = np.diag([1.0, 0.0])
    B = np.diag([0.5, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="genuinely undefined"):
        ordqz_sorted(A, B, lambda a, b: abs(b) < abs(a), output="complex")


def test_a_healthy_build_is_left_alone():
    """No monkeypatch: whatever this machine does, it must agree with ordqz."""
    G0, G1, _, _ = _block_model()
    sort = lambda a, b: abs(b) < abs(a)
    try:
        expected = scipy.linalg.ordqz(G0, G1, sort=sort, output="complex")
    except ValueError:
        pytest.skip("this build cannot reorder the pencil in complex arithmetic")
    got = ordqz_sorted(G0, G1, sort, output="complex")
    for a, b in zip(got, expected):
        assert np.allclose(a, b, atol=1e-14)
