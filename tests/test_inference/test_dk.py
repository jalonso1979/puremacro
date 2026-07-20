import numpy as np

from puremacro.inference.dk import driscoll_kraay


def test_dk_aggregates_by_time_then_nw():
    # 3 entities x 5 periods: score = 1 for entity A, 2 for B, 3 for C
    # so cross-section sum at every time = 6.
    rng = np.random.default_rng(0)
    T = 5
    score = np.repeat([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], T, axis=0)
    time_keys = np.tile(np.arange(T), 3)
    S = driscoll_kraay(score, time_keys, lags=2)
    # Hand-computed: by_t[t] = (6, 3) for all t. S_0 = 5 * (6, 3)' (6, 3)
    expected_S0 = 5 * np.outer([6.0, 3.0], [6.0, 3.0])
    # With Bartlett correction at lag 1: w=2/3, Gamma = 4 * outer; lag 2: w=1/3, 3*outer.
    expected_S = expected_S0 + (2/3) * 2 * 4 * np.outer([6, 3], [6, 3])
    expected_S += (1/3) * 2 * 3 * np.outer([6, 3], [6, 3])
    assert np.allclose(S, expected_S, atol=1e-8)


def test_dk_zero_lags_equals_outer_sum():
    rng = np.random.default_rng(7)
    T_panel = 50
    score = rng.standard_normal((T_panel, 3))
    time_keys = rng.integers(0, 10, size=T_panel)
    S = driscoll_kraay(score, time_keys, lags=0)
    # With lags=0, S = sum_t by_t outer by_t
    by_t = np.array([score[time_keys == t].sum(axis=0) for t in np.unique(time_keys)])
    expected = by_t.T @ by_t
    assert np.allclose(S, expected, atol=1e-10)
