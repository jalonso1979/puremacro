"""Tests for var.identify.panel.mean_group_svar + PanelSVARResult."""
import numpy as np
import pytest


def _toy_panel(n_countries: int = 3, T: int = 100, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    panel = {}
    for i in range(n_countries):
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + L @ rng.standard_normal(2)
        panel[f"C{i}"] = Y
    return panel


def test_panel_svar_result_is_frozen():
    from puremacro.var.identify._results import PanelSVARResult

    H = 4; n = 2; N = 3
    res = PanelSVARResult(
        irf_mean=np.zeros((H + 1, n, n)),
        irf_lower=np.zeros((H + 1, n, n)),
        irf_upper=np.zeros((H + 1, n, n)),
        country_irfs=np.zeros((N, H + 1, n, n)),
        country_ids=("C0", "C1", "C2"),
        identification="cholesky",
        p=1,
        horizon=H,
        ci=0.9,
    )
    assert res.irf_mean.shape == (5, 2, 2)
    with pytest.raises(Exception):
        res.ci = 0.95  # frozen


def test_mean_group_svar_returns_dataclass():
    from puremacro.var.identify.panel import mean_group_svar
    from puremacro.var.identify._results import PanelSVARResult

    panel = _toy_panel()
    res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky",
                          ci=0.9, seed=0)
    assert isinstance(res, PanelSVARResult)
    assert res.irf_mean.shape == (5, 2, 2)  # (H+1, n, n) canonical
    assert res.country_irfs.shape == (3, 5, 2, 2)
    assert set(res.country_ids) == {"C0", "C1", "C2"}
    assert res.identification == "cholesky"


def test_mean_group_svar_supports_bq():
    from puremacro.var.identify.panel import mean_group_svar

    panel = _toy_panel()
    res = mean_group_svar(panel, p=1, horizon=4, identification="bq",
                          ci=0.9, seed=0, permanent_var_idx=0)
    assert res.identification == "bq"
    assert res.irf_mean.shape == (5, 2, 2)


def test_mean_group_svar_rejects_unsupported_identification():
    from puremacro.var.identify.panel import mean_group_svar

    panel = _toy_panel()
    with pytest.raises(ValueError, match="Unsupported identification"):
        mean_group_svar(panel, p=1, horizon=4, identification="proxy",
                        ci=0.9, seed=0)


def test_mean_group_svar_summary_smoke():
    from puremacro.var.identify.panel import mean_group_svar

    panel = _toy_panel()
    res = mean_group_svar(panel, p=1, horizon=4, ci=0.9, seed=0)
    s = res.summary()
    assert "Panel SVAR" in s
    assert "cholesky" in s.lower()
