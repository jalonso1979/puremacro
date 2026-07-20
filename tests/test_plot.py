"""Smoke tests for puremacro.plot helpers (headless via Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from puremacro.plot import (  # noqa: E402
    plot_fan,
    plot_fevd,
    plot_irf_multi,
    plot_irf_single,
    plot_spillover,
)


def _make_irf_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = np.arange(0, 13)
    beta = np.exp(-0.2 * h) * rng.normal(size=h.size).cumsum()
    se = 0.3 + 0.05 * h
    return pd.DataFrame({"h": h, "beta": beta, "lo": beta - se, "hi": beta + se})


def test_plot_irf_single_dataframe():
    df = _make_irf_df()
    fig = plot_irf_single(df, title="Test IRF", ylabel="resp", scale=2.0)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_irf_single_run_experiment_dict():
    df = _make_irf_df()
    point = pd.Series(df["beta"].values, index=df["h"].values)
    lower = pd.Series(df["lo"].values, index=df["h"].values)
    upper = pd.Series(df["hi"].values, index=df["h"].values)
    irf = {"irf_point": point, "irf_lower": lower, "irf_upper": upper}
    fig = plot_irf_single(irf, title="dict path")
    assert isinstance(fig, Figure)
    # The plotted line + the band should populate the axes.
    ax = fig.axes[0]
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_irf_multi_three_curves():
    irf_by = {f"proxy_{i}": _make_irf_df(seed=i) for i in range(3)}
    fig = plot_irf_multi(irf_by, title="multi", show_bands=True)
    assert isinstance(fig, Figure)
    leg = fig.axes[0].get_legend()
    assert leg is not None
    assert len(leg.get_texts()) == 3
    plt.close(fig)


def test_plot_fevd_stacked():
    rng = np.random.default_rng(7)
    raw = rng.uniform(size=(11, 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    fig = plot_fevd(raw, var_names=["x", "y", "z"], title="fevd")
    assert isinstance(fig, Figure)
    leg = fig.axes[0].get_legend()
    assert leg is not None
    assert len(leg.get_texts()) == 3
    plt.close(fig)


def test_plot_fan():
    horizons = np.arange(1, 13)
    quantiles = ["mean", "q05", "q10", "q25", "q75", "q90", "q95"]
    rng = np.random.default_rng(11)
    base = np.linspace(0.0, 1.0, horizons.size) + rng.normal(scale=0.05, size=horizons.size)
    rows = {
        "mean": base,
        "q05": base - 0.6,
        "q10": base - 0.4,
        "q25": base - 0.2,
        "q75": base + 0.2,
        "q90": base + 0.4,
        "q95": base + 0.6,
    }
    df = pd.DataFrame(np.vstack([rows[q] for q in quantiles]),
                      index=quantiles, columns=horizons)
    fig = plot_fan(df, levels=(0.5, 0.8, 0.9), title="fan")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_spillover_heatmap():
    rng = np.random.default_rng(3)
    arr = rng.uniform(size=(5, 5))
    df = pd.DataFrame(arr, index=[f"v{i}" for i in range(5)],
                      columns=[f"v{i}" for i in range(5)])
    fig = plot_spillover(df, title="spill")
    assert isinstance(fig, Figure)
    # Should have at least one image (the heatmap).
    assert len(fig.axes[0].images) == 1
    plt.close(fig)


def test_plot_irf_single_with_existing_ax():
    fig, ax = plt.subplots()
    df = _make_irf_df()
    out = plot_irf_single(df, ax=ax)
    assert isinstance(out, Figure)
    assert out is fig
    plt.close(fig)
