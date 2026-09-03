import numpy as np
import pandas as pd

from puremacro.connectedness.diebold_yilmaz import spillover_index


def test_spillover_independent_series_low_total():
    """Two independent processes -> total spillover should be small."""
    rng = np.random.default_rng(11)
    T = 300
    df = pd.DataFrame({
        "y1": rng.standard_normal(T).cumsum(),
        "y2": rng.standard_normal(T).cumsum(),
    })
    out = spillover_index(df, var_lags=2, fevd_horizon=10)
    for k in ("total", "directional_to", "directional_from", "net", "pairwise_matrix"):
        assert k in out
    # Total spillover (in %) should be low for independent series
    assert out["total"] < 30


def test_spillover_correlated_series_high_total():
    """Two highly co-moving series (driven by a common factor) -> high spillover."""
    rng = np.random.default_rng(13)
    T = 300
    common = rng.standard_normal(T).cumsum()
    df = pd.DataFrame({
        "y1": common + 0.3 * rng.standard_normal(T),
        "y2": common + 0.3 * rng.standard_normal(T),
    })
    out = spillover_index(df, var_lags=2, fevd_horizon=10)
    assert out["total"] > 30  # most variance comes from "shared" identification


# ---------------------------------------------------------------------------
# Regression test for the gfevd units bug (fixed after 1.9.0).
# ---------------------------------------------------------------------------
def test_gfevd_is_invariant_to_the_units_of_a_variable():
    """Pesaran-Shin GFEVD does not depend on how a variable is scaled.

    The estimand is exactly invariant to y -> D y for diagonal positive D:
    under it A_i -> D A_i D^-1 and Sigma -> D Sigma D, so the numerator picks
    up d_i^2 d_j^2 / d_j^2 = d_i^2 and the denominator picks up the same
    d_i^2, term by term.

    `gfevd` used to divide by `np.maximum(sigma_jj, 1e-12)`. An ABSOLUTE floor
    cannot be invariant to a change of units, because whether a variance falls
    under it depends on the units the caller chose. On this fixture the total
    connectedness index moved from 13.43 to 39.48 as one variable was rescaled
    from 1e-3 to 1e-10 — a spillover index that depends on whether the series
    is measured in billions or trillions. It was silent, and every intermediate
    answer stayed plausible.
    """
    import numpy as np
    from puremacro.var.irf import gfevd

    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.55]])
    L = np.array([[1.0, 0.0, 0.0], [0.35, 1.4, 0.0], [0.2, -0.3, 0.9]])
    Sigma = L @ L.T
    H = 12

    def total_connectedness(theta):
        t = theta[-1]
        return 100.0 * (t.sum() - np.trace(t)) / t.sum()

    base = gfevd([A], Sigma, horizon=H)
    base_total = total_connectedness(base)
    for s in (1e3, 1e-3, 1e-6, 1e-8, 1e-10):
        D = np.diag([1.0, 1.0, s])
        Di = np.linalg.inv(D)
        theta = gfevd([D @ A @ Di], D @ Sigma @ D, horizon=H)
        assert np.allclose(theta, base, atol=1e-10), (
            f"GFEVD shares changed when variable 3 was rescaled by {s:g}: "
            f"max deviation {np.abs(theta - base).max():.3e}"
        )
        assert abs(total_connectedness(theta) - base_total) < 1e-8, (
            f"total connectedness moved from {base_total:.4f} to "
            f"{total_connectedness(theta):.4f} on a pure change of units"
        )


def test_gfevd_reports_a_degenerate_variance_rather_than_flooring_it():
    """A zero shock variance has no defined GFEVD column, so it must raise.

    It used to be floored to 1e-12 and the run continued, producing shares for
    a shock that has no variance.
    """
    import numpy as np
    import pytest
    from puremacro.var.irf import gfevd

    Sigma = np.diag([1.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="non-positive diagonal"):
        gfevd([np.array([[0.5, 0.0], [0.0, 0.5]])], Sigma, horizon=4)
