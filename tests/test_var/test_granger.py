import numpy as np
import pandas as pd

from puremacro.var.diagnostics import granger_causality, block_exogeneity


def test_granger_detects_true_causality():
    """In y_{t} = 0.6 y_{t-1} + 0.4 x_{t-1} + e, x Granger-causes y."""
    rng = np.random.default_rng(11)
    T = 300
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t-1] + rng.standard_normal()
        y[t] = 0.6 * y[t-1] + 0.4 * x[t-1] + rng.standard_normal()
    Y = pd.DataFrame({"y": y, "x": x})
    out = granger_causality(Y, source="x", target="y", p=2)
    assert out["p_value"] < 0.01
    assert out["stat"] > 0


def test_granger_no_causality_when_independent():
    """Independent series: p-value should not be small."""
    rng = np.random.default_rng(13)
    T = 300
    x = rng.standard_normal(T)
    y = rng.standard_normal(T)
    Y = pd.DataFrame({"y": y, "x": x})
    out = granger_causality(Y, source="x", target="y", p=2)
    assert out["p_value"] > 0.05


def test_block_exogeneity_returns_documented_keys():
    rng = np.random.default_rng(17)
    T, n = 300, 3
    A = np.array([[0.5, 0.0, 0.0], [0.3, 0.4, 0.0], [0.0, 0.2, 0.3]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + rng.standard_normal(n)
    Y_df = pd.DataFrame(Y, columns=["y0", "y1", "y2"])
    out = block_exogeneity(Y_df, source_block=["y0"], target_block=["y2"], p=2)
    for key in ("stat", "p_value", "df_num", "df_den"):
        assert key in out
