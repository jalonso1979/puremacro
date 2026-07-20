"""Tests for var.identify.maxshare.identify_maxshare full pipeline + MaxShareResult."""
import numpy as np
import pytest


def _toy_var2(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    L = np.array([[1.0, 0.0], [0.3, 0.9]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def test_maxshare_result_is_frozen():
    from puremacro.var.identify._results import MaxShareResult

    H = 4; n = 2
    res = MaxShareResult(
        B=np.eye(n),
        q=np.array([1.0, 0.0]),
        fev_share_at_target=0.85,
        irfs=np.zeros((H + 1, n, n)),
        fevd=np.zeros((H + 1, n, n)),
        max_fev_at=1,
        irf_lower=None,
        irf_upper=None,
        ci=0.68,
    )
    assert res.B.shape == (n, n)
    with pytest.raises(Exception):
        res.ci = 0.95  # frozen


def test_identify_maxshare_full_pipeline_returns_dataclass():
    from puremacro.var.identify.maxshare import identify_maxshare
    from puremacro.var.identify._results import MaxShareResult

    Y = _toy_var2()
    res = identify_maxshare(
        Y, p=2, target_idx=0, max_fev_at=1, horizon=4,
        n_bootstrap=50, ci=0.68, seed=0,
    )
    assert isinstance(res, MaxShareResult)
    assert res.B.shape == (2, 2)
    assert res.q.shape == (2,)
    assert res.irfs.shape == (5, 2, 2)
    assert res.fevd.shape == (5, 2, 2)
    assert res.irf_lower.shape == (5, 2, 2)
    assert res.irf_upper.shape == (5, 2, 2)
    assert 0.0 <= res.fev_share_at_target <= 1.0 + 1e-9


def test_identify_maxshare_skips_bootstrap_when_n_bootstrap_is_zero():
    from puremacro.var.identify.maxshare import identify_maxshare

    Y = _toy_var2()
    res = identify_maxshare(Y, p=2, horizon=4, n_bootstrap=0, seed=0)
    assert res.irf_lower is None
    assert res.irf_upper is None


def test_identify_maxshare_summary_smoke():
    from puremacro.var.identify.maxshare import identify_maxshare

    Y = _toy_var2()
    res = identify_maxshare(Y, p=2, horizon=4, n_bootstrap=20, ci=0.68, seed=0)
    s = res.summary()
    assert "Max-share" in s
    assert "FEV" in s


def test_low_level_maxshare_still_works():
    """Backwards compat: maxshare(...) and news_maxshare(...) still return B0 ndarray."""
    from puremacro.var.identify.maxshare import maxshare, news_maxshare
    from puremacro.var.estimate import estimate_var

    Y = _toy_var2()
    A_list, _, Sigma, _, _ = estimate_var(Y, p=2)
    B0 = maxshare(A_list, Sigma, target_var=0, horizon=4)
    assert B0.shape == (2, 2)
    B0_news = news_maxshare(A_list, Sigma, target_var=0, horizon=40)
    assert B0_news.shape == (2, 2)
