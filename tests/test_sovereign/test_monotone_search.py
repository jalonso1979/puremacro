"""Certified monotone policy search and vectorized price-schedule tangents."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro._autodiff import DualArray
from puremacro.sovereign.arellano import _compute_dq_db_dualarray


def _reference_dq_db(q: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
    """The per-node DualArray loop used before vectorization, kept as the oracle."""
    n_b, n_y = q.shape
    dq_db = np.zeros_like(q)
    h = np.diff(b_grid)
    for j in range(n_y):
        q_col = q[:, j]
        delta = np.diff(q_col) / h
        d = np.zeros(n_b, dtype=np.float64)
        for k in range(1, n_b - 1):
            if delta[k - 1] * delta[k] > 0.0:
                d[k] = 2.0 / (1.0 / delta[k - 1] + 1.0 / delta[k])
        d[0] = delta[0] if len(delta) > 0 and delta[0] <= 0.0 else 0.0
        d[-1] = delta[-1] if len(delta) > 0 and delta[-1] <= 0.0 else 0.0
        for k in list(range(n_b - 1)) + [n_b - 1]:
            left = min(k, n_b - 2)
            bk, hk = b_grid[left], h[left]
            t = (DualArray(b_grid[k], np.array([1.0])) - bk) / hk
            q_eval = ((2.0 * t ** 3 - 3.0 * t ** 2 + 1.0) * q_col[left]
                      + (t ** 3 - 2.0 * t ** 2 + t) * (hk * d[left])
                      + (-2.0 * t ** 3 + 3.0 * t ** 2) * q_col[left + 1]
                      + (t ** 3 - t ** 2) * (hk * d[left + 1]))
            dq_db[k, j] = q_eval.der[0]
    return np.minimum(dq_db, 0.0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape,kind", [((2, 3), "decreasing"), ((41, 7), "decreasing"),
                                        ((41, 7), "rough"), ((200, 5), "flat"),
                                        ((60, 4), "uneven")])
def test_dq_db_vectorized_matches_loop(seed, shape, kind):
    rng = np.random.default_rng(seed)
    n_b, n_y = shape
    b_grid = np.linspace(-0.3, 0.4, n_b)
    if kind == "uneven":
        b_grid = np.sort(rng.uniform(-0.3, 0.4, n_b))
    q = np.sort(rng.uniform(0.0, 0.99, shape), axis=0)[::-1]
    if kind == "rough":
        q = rng.uniform(0.0, 0.99, shape)
    if kind == "flat":
        q[n_b // 3: n_b // 2] = q[n_b // 3]
        q[-n_b // 4:] = 0.0
    assert np.allclose(_compute_dq_db_dualarray(q, b_grid), _reference_dq_db(q, b_grid),
                       atol=1e-12, rtol=0)


def _crra(gamma):
    if gamma == 1.0:
        return np.log
    return lambda c: (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _inputs(seed, n_b, n_y, saving=True):
    rng = np.random.default_rng(seed)
    b_grid = np.linspace(-0.3 if saving else 0.0, 0.4, n_b)
    y_grid = np.sort(rng.uniform(0.8, 1.2, n_y))
    q = np.sort(rng.uniform(0.0, 1 / 1.01, (n_b, n_y)), axis=0)[::-1].copy()
    ev = rng.normal(-30.0, 2.0, (n_b, n_y))
    return b_grid, y_grid, q, ev


def _dense(b_grid, y_grid, q, ev, beta, u):
    c = (y_grid[None, None, :] - 1.0 * b_grid[:, None, None]) + q[None, :, :] * b_grid[None, :, None]
    w = u(c) + beta * ev[None, :, :]
    return w.max(axis=1), w.argmax(axis=1), c.min()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_b,n_y", [(2, 2), (3, 5), (17, 2), (64, 5)])
@pytest.mark.parametrize("gamma", [2.0, 5.0])
@pytest.mark.parametrize("saving", [True, False])
def test_maximize_matches_brute_force(seed, n_b, n_y, gamma, saving):
    from puremacro.sovereign._monotone import maximize

    args = (*_inputs(seed, n_b, n_y, saving), 0.95, _crra(gamma))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value)
    assert np.array_equal(policy, dense_policy)


def test_ties_choose_lowest_debt_index():
    from puremacro.sovereign._monotone import maximize

    b_grid, y_grid, q, ev = _inputs(3, 41, 5)
    q[25:, :] = 0.0          # fully defaulting region: no proceeds ...
    ev[25:, :] = ev[25, :]   # ... and identical continuation (V = V_D)
    ev[:25, :] -= 50.0       # make the tied region optimal for every row
    args = (b_grid, y_grid, q, ev, 0.95, _crra(2.0))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(policy, dense_policy)
    assert np.all(policy == 25)
    assert np.array_equal(value, dense_value)


def test_minimal_grid():
    from puremacro.sovereign._monotone import maximize

    args = (*_inputs(0, 2, 2), 0.95, _crra(2.0))
    value, policy = maximize(*args)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value) and np.array_equal(policy, dense_policy)


@pytest.mark.parametrize("block_rows", [1, 7, 128])
def test_exhaustive_matches_dense(block_rows):
    from puremacro.sovereign._monotone import exhaustive_maximize

    args = (*_inputs(5, 50, 4), 0.95, _crra(2.0))
    value, policy = exhaustive_maximize(*args, block_rows=block_rows)
    dense_value, dense_policy, _ = _dense(*args)
    assert np.array_equal(value, dense_value) and np.array_equal(policy, dense_policy)


@pytest.mark.parametrize("seed", range(5))
def test_min_candidate_consumption_is_exact(seed):
    from puremacro.sovereign._monotone import min_candidate_consumption

    b_grid, y_grid, q, ev = _inputs(seed, 30, 6)
    assert min_candidate_consumption(b_grid, y_grid, q) == _dense(b_grid, y_grid, q, ev, 0.95, np.log)[2]


GRIDS = {"saving": dict(b_min=-0.3, b_max=0.3, n_b=41), "no_saving": dict(b_min=0.0, b_max=0.4, n_b=40)}


@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("sigma_eps", [0.0, 0.01])
@pytest.mark.parametrize("gamma", [1.0, 2.0, 5.0])
def test_monotone_equals_dense(gamma, sigma_eps, grid):
    from puremacro.sovereign import solve_sovereign_default

    kwargs = dict(gamma=gamma, sigma_eps=sigma_eps, n_y=7, y_penalty_type="arellano", **GRIDS[grid])
    dense = solve_sovereign_default(**kwargs)
    fast = solve_sovereign_default(policy_search="monotone", **kwargs)
    assert np.array_equal(fast.policy_b, dense.policy_b)
    assert (fast.iterations, fast.converged) == (dense.iterations, dense.converged)
    for name in ("v_repay", "v_default", "bond_price", "default_prob"):
        assert np.allclose(getattr(fast, name), getattr(dense, name), atol=1e-12, rtol=0), name


def test_monotone_metadata_reports_certification():
    from puremacro.sovereign import solve_sovereign_default

    kwargs = dict(n_y=7, y_penalty_type="arellano", **GRIDS["saving"])
    fast = solve_sovereign_default(policy_search="monotone", **kwargs)
    assert fast.metadata["policy_search"] == "monotone"
    assert fast.metadata["monotone_certification_gap"] <= 1e-11
    assert np.isfinite(fast.metadata["exhaustive_bellman_residual"])
    assert fast.metadata["exhaustive_bellman_residual"] <= 10 * fast.metadata["tol"]
    dense = solve_sovereign_default(**kwargs)
    assert dense.metadata["policy_search"] == "dense"
    assert dense.metadata["monotone_certification_gap"] is None
    assert dense.metadata["exhaustive_bellman_residual"] is None


def test_monotone_rejects_long_term_debt():
    from puremacro.sovereign import solve_sovereign_default

    with pytest.raises(ValueError, match="short-term"):
        solve_sovereign_default(delta=0.2, n_b=20, n_y=5, policy_search="monotone")


def test_unknown_policy_search():
    from puremacro.sovereign import solve_sovereign_default

    with pytest.raises(ValueError, match="policy_search"):
        solve_sovereign_default(n_b=20, n_y=5, policy_search="fast")


def test_monotone_rejects_barrier_region():
    from puremacro.sovereign import solve_sovereign_default

    with pytest.raises(ValueError, match='policy_search="dense"'):
        solve_sovereign_default(b_min=0.0, b_max=2.0, n_b=20, n_y=5, policy_search="monotone")


def test_monotone_rejects_convex_utility_before_solving():
    from puremacro.sovereign import solve_sovereign_default

    with pytest.raises(ValueError, match="gamma"):
        solve_sovereign_default(gamma=-1.0, n_b=20, n_y=5, max_iter=1, policy_search="monotone")


def _penalized_dense(b_grid, y_grid, q, ev, beta, u, column=None, transition=None):
    c = (y_grid[None, None, :] - 1.0 * b_grid[:, None, None]) + q[None, :, :] * b_grid[None, :, None]
    w = u(c) + beta * ev[None, :, :]
    if column is not None:
        w = w - column[None, :, :]
    if transition is not None:
        w = w - transition
    return w.max(axis=1), w.argmax(axis=1)


def _random_column_penalty(rng, n_b, n_y, hard):
    pen = rng.uniform(0.0, 2.0, (n_b, n_y))
    if hard:
        pen = np.where(rng.random((n_b, n_y)) < 0.3, np.inf, 0.0)
        pen[0] = 0.0  # keep one admissible node per income state
    return pen


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("hard", [False, True])
def test_column_penalty_matches_brute_force(seed, hard):
    from puremacro.sovereign._monotone import exhaustive_maximize, maximize

    rng = np.random.default_rng(100 + seed)
    b_grid, y_grid, q, ev = _inputs(seed, 33, 4)
    pen = _random_column_penalty(rng, 33, 4, hard)
    args = (b_grid, y_grid, q, ev, 0.95, _crra(2.0))
    dense_value, dense_policy = _penalized_dense(*args, column=pen)
    for value, policy in (maximize(*args, column_penalty=pen),
                          exhaustive_maximize(*args, column_penalty=pen)):
        assert np.max(np.abs(value - dense_value)) <= 1e-12
        assert np.array_equal(policy, dense_policy)


def test_hard_column_mask_equals_removed_columns():
    from puremacro.sovereign._monotone import maximize

    rng = np.random.default_rng(7)
    b_grid, y_grid, q, ev = _inputs(7, 40, 5)
    pen = _random_column_penalty(rng, 40, 5, hard=True)
    masked_ev = np.where(np.isinf(pen), -np.inf, ev)
    a = maximize(b_grid, y_grid, q, ev, 0.95, _crra(2.0), column_penalty=pen)
    b = maximize(b_grid, y_grid, q, masked_ev, 0.95, _crra(2.0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("block_rows", [1, 7, 128])
def test_transition_penalty_matches_brute_force(block_rows):
    from puremacro.sovereign._monotone import exhaustive_maximize

    b_grid, y_grid, q, ev = _inputs(11, 30, 4)
    limit = 0.05
    full = np.where(b_grid[None, :, None] - b_grid[:, None, None] > limit, 0.3, 0.0) * np.ones((1, 1, 4))
    full[:, :, 2] = np.where(full[:, :, 2] > 0, np.inf, 0.0)  # one income state hard

    def transition(first, last, j):
        return full[first:last, :, j]

    column = np.where(b_grid[:, None] > 0.3, 0.1, 0.0) * np.ones((1, 4))
    args = (b_grid, y_grid, q, ev, 0.95, _crra(2.0))
    dense_value, dense_policy = _penalized_dense(*args, column=column, transition=full)
    value, policy = exhaustive_maximize(*args, block_rows=block_rows, column_penalty=column,
                                        transition_penalty=transition)
    assert np.max(np.abs(value - dense_value)) <= 1e-12
    assert np.array_equal(policy, dense_policy)


def test_penalty_share_nonincreasing_at_fixed_prices():
    from puremacro.sovereign._monotone import maximize

    b_grid, y_grid, q, ev = _inputs(4, 60, 5)
    ev = ev + 5.0 * b_grid[:, None]  # make high debt attractive so the cap matters
    above = b_grid > 0.1
    shares = []
    for penalty in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e6):
        pen = np.where(above[:, None], penalty, 0.0) * np.ones((1, 5))
        _, policy = maximize(b_grid, y_grid, q, ev, 0.95, _crra(2.0), column_penalty=pen)
        shares.append(float(above[policy].mean()))
    assert shares[0] > 0.0
    assert all(later <= earlier for earlier, later in zip(shares, shares[1:]))
    assert shares[-1] == 0.0


def test_all_infeasible_income_column():
    import warnings

    from puremacro.sovereign._monotone import exhaustive_maximize, maximize

    b_grid, y_grid, q, ev = _inputs(2, 20, 3)
    pen = np.zeros((20, 3))
    pen[:, 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for value, _ in (maximize(b_grid, y_grid, q, ev, 0.95, _crra(2.0), column_penalty=pen),
                         exhaustive_maximize(b_grid, y_grid, q, ev, 0.95, _crra(2.0), column_penalty=pen)):
            assert np.all(np.isneginf(value[:, 1]))
            assert np.all(np.isfinite(value[:, [0, 2]]))
