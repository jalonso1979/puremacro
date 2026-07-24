"""Planted-truth + reduction tests for the Hansen-Lunde-Nason (2011) MCS."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.forecast import model_confidence_set, losses_from_forecasts


def _horse_race(rng, T=250, spread=None):
    """(T, M) losses where model m has mean loss level `spread[m]` + iid noise."""
    if spread is None:
        spread = [1.0, 1.4, 1.8, 2.4, 3.2]
    spread = np.asarray(spread, dtype=float)
    # Common noise (comovement) + idiosyncratic noise, so the bootstrap has real
    # cross-model dependence to preserve.
    common = 0.3 * rng.standard_normal((T, 1))
    idio = 0.4 * rng.standard_normal((T, len(spread)))
    return np.abs(spread[None, :] + common + idio)


def test_dominant_best_retained_and_dominant_worst_eliminated_first():
    rng = np.random.default_rng(0)
    losses = _horse_race(rng)
    res = model_confidence_set(losses, alpha=0.10, n_boot=800, seed=1)
    # Model 0 is uniformly the lowest-loss model => always in the set.
    assert 0 in res["included"].tolist()
    # Model 4 is uniformly the worst => first to be eliminated.
    assert res["elimination_order"][0] == 4


def test_identical_models_all_retained_pvalues_one():
    rng = np.random.default_rng(2)
    col = np.abs(rng.standard_normal((200, 1)) + 2.0)
    losses = np.repeat(col, 4, axis=1)  # four byte-identical models
    res = model_confidence_set(losses, alpha=0.10, n_boot=500, seed=3)
    assert res["included"].tolist() == [0, 1, 2, 3]
    # Equal predictive ability is never rejected => MCS p-values ~1.
    assert np.all(res["pvalues"] >= 0.99)


def test_reproducible_same_seed_same_pvalues():
    rng = np.random.default_rng(4)
    # A tight race (near-ties) so the bootstrap p-values are non-degenerate and
    # genuinely seed-dependent, not pinned at {0, 1}.
    losses = _horse_race(rng, spread=[1.00, 1.03, 1.06, 1.10])
    a = model_confidence_set(losses, n_boot=600, seed=42)
    b = model_confidence_set(losses, n_boot=600, seed=42)
    assert np.array_equal(a["pvalues"], b["pvalues"])
    assert a["included"].tolist() == b["included"].tolist()
    # A different seed shifts the bootstrap p-values.
    c = model_confidence_set(losses, n_boot=600, seed=43)
    assert not np.array_equal(a["pvalues"], c["pvalues"])


def test_alpha_monotonicity_larger_alpha_weakly_smaller_set():
    rng = np.random.default_rng(5)
    losses = _horse_race(rng, spread=[1.0, 1.2, 1.5, 1.9, 2.5, 3.3])
    prev = None
    for alpha in (0.01, 0.05, 0.10, 0.25, 0.50):
        res = model_confidence_set(losses, alpha=alpha, n_boot=700, seed=7)
        size = len(res["included"])
        if prev is not None:
            assert size <= prev, f"alpha={alpha}: set grew ({size} > {prev})"
        prev = size
    # The set is never empty.
    assert prev >= 1


def test_statistics_and_forecast_input_path_agree_on_best():
    rng = np.random.default_rng(8)
    # Feed forecasts + realized instead of a loss matrix.
    T = 200
    y = rng.standard_normal(T)
    # Model 0 sees the truth with tiny noise; others progressively noisier.
    fc = np.column_stack([
        y + s * rng.standard_normal(T) for s in (0.1, 0.5, 0.9, 1.4)
    ])
    for stat in ("tmax", "tr", "tsq"):
        res = model_confidence_set(
            forecasts=fc, realized=y, statistic=stat,
            alpha=0.10, n_boot=500, seed=11,
        )
        assert 0 in res["included"].tolist()
        assert res["elimination_order"][0] == 3  # noisiest goes first


def test_losses_from_forecasts_matches_manual_mse():
    rng = np.random.default_rng(9)
    fc = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    L = losses_from_forecasts(fc, y, loss="mse")
    assert np.allclose(L, (fc - y[:, None]) ** 2)


def test_input_validation():
    with pytest.raises(ValueError):
        model_confidence_set(np.zeros((10, 1)))  # need >= 2 models
    with pytest.raises(ValueError):
        model_confidence_set()  # nothing supplied
