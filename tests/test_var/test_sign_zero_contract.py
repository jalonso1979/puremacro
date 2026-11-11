"""Scientific invariants at the public sign/zero acceptance boundary."""
import numpy as np
import pytest

from puremacro.var.identify import sign_zero


@pytest.mark.parametrize("seed", [0, 3, 17])
def test_positive_orthogonal_columns_are_impossible(seed):
    # Two strictly positive vectors have positive inner product; they cannot
    # be two columns of an orthogonal matrix when Sigma = I.
    signs = {(i, j): 1 for i in range(2) for j in range(2)}
    out = sign_zero([], np.eye(2), sign_constraints=signs,
                    n_draws=25, rng=np.random.default_rng(seed))
    assert not out.success
    assert out.B0 is None and out.Q is None
    assert out.n_draws_used == 25


@pytest.mark.parametrize("seed", [0, 1, 2, 9])
def test_feasible_whole_column_signs_and_dictionary_order(seed):
    # A rotation [[cos(a), -sin(a)], [sin(a), cos(a)]] with 0<a<pi/2
    # proves this sign pattern is feasible, independently of the sampler.
    signs = {(0, 0): 1, (1, 0): 1, (0, 1): -1, (1, 1): 1}
    kwargs = dict(n_draws=100)
    out = sign_zero([], np.eye(2), sign_constraints=signs,
                    rng=np.random.default_rng(seed), **kwargs)
    reordered = sign_zero([], np.eye(2),
                          sign_constraints=dict(reversed(list(signs.items()))),
                          rng=np.random.default_rng(seed), **kwargs)
    assert out.success and reordered.success
    for ij, sign in signs.items():
        assert sign * out.B0[ij] > 0
    np.testing.assert_allclose(out.Q.T @ out.Q, np.eye(2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.B0 @ out.B0.T, np.eye(2), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out.B0, reordered.B0)
    assert out.n_draws_used == reordered.n_draws_used
    if seed == 2:
        # The first rotation has incompatible signs within a column. It
        # must be rejected, despite the existence of later feasible draws.
        assert out.n_draws_used > 1


def test_zero_and_strict_sign_at_same_entry_never_succeed():
    out = sign_zero([], np.eye(2), zero_constraints=[(0, 0)],
                    sign_constraints={(0, 0): 1}, n_draws=20,
                    rng=np.random.default_rng(1))
    assert not out.success
    assert out.Q is None and out.B0 is None


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_zero_and_sign_acceptance_reconstructs_correlated_covariance(scale):
    # Lower-triangular Cholesky with positive diagonal is a known admissible
    # impact matrix for this covariance and restriction pattern.
    sigma = scale * np.array([[1.0, 0.3], [0.3, 2.0]])
    out = sign_zero([], sigma, zero_constraints=[(0, 1)],
                    sign_constraints={(0, 0): 1, (1, 0): 1, (1, 1): 1},
                    n_draws=100, rng=np.random.default_rng(11))
    assert out.success
    assert abs(out.B0[0, 1]) / np.sqrt(sigma[0, 0]) < 1e-10
    for ij in [(0, 0), (1, 0), (1, 1)]:
        assert out.B0[ij] > 0
    np.testing.assert_allclose(out.Q.T @ out.Q, np.eye(2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.B0 @ out.B0.T / scale, sigma / scale,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("sign", [0, 2, -2, True, np.nan, np.inf, "1", None])
def test_malformed_sign_value_rejected(sign):
    with pytest.raises(ValueError, match="sign_constraints"):
        sign_zero([], np.eye(2), sign_constraints={(0, 0): sign}, n_draws=1)


@pytest.mark.parametrize("pair", [(-1, 0), (2, 0), (0, 2), (0.5, 0),
                                   (True, 0), (0,), (0, 0, 0), "00"])
@pytest.mark.parametrize("kind", ["zero_constraints", "sign_constraints"])
def test_malformed_constraint_coordinate_rejected(pair, kind):
    constraints = [pair] if kind == "zero_constraints" else {pair: 1}
    with pytest.raises(ValueError, match=kind):
        sign_zero([], np.eye(2), **{kind: constraints}, n_draws=1)


@pytest.mark.parametrize("n_draws", [-1, 1.5, True])
def test_invalid_draw_budget_rejected(n_draws):
    with pytest.raises(ValueError, match="n_draws"):
        sign_zero([], np.eye(2), n_draws=n_draws)


def test_zero_budget_is_a_valid_empty_search():
    out = sign_zero([], np.eye(2), n_draws=0)
    assert not out.success
    assert out.B0 is None and out.Q is None and out.n_draws_used == 0


@pytest.mark.parametrize("sigma", [np.empty((0, 0)), np.ones(2), np.ones((2, 3)),
                                    [[np.nan, 0.0], [0.0, 1.0]],
                                    [[np.inf, 0.0], [0.0, 1.0]],
                                    np.eye(2, dtype=complex)])
def test_malformed_covariance_rejected(sigma):
    with pytest.raises(ValueError, match="Sigma"):
        sign_zero([], sigma, n_draws=1)
