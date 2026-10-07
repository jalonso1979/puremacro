"""Joint LP covariance against a dense, independently assembled kernel oracle."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.lp import lp_hac
from puremacro.structural.lp import lp_moment_targets


def panel():
    rng = np.random.default_rng(87)
    x = rng.normal(size=180)
    z = rng.normal(size=180)
    y = np.zeros(180)
    for t in range(1, len(y)):
        y[t] = .6 * y[t-1] + .8 * x[t] + .2 * z[t] + rng.normal()
    return pd.DataFrame({"y": y, "w": y + rng.normal(size=180), "x": x, "z": z},
                        index=pd.period_range("1980Q1", periods=180, freq="Q"))


def targets(data=None, **kw):
    params = dict(responses=["y", "w"], shock="x", horizons=[0, 2, 4],
                  response_units={"y": "log points", "w": "level"},
                  shock_unit="percentage point", frequency="Q", lags=2,
                  controls=["z"], bandwidth=5)
    params.update(kw)
    return lp_moment_targets(panel() if data is None else data, **params)


def test_point_estimates_match_canonical_lp():
    d = panel()
    actual = targets(d)
    expected = np.concatenate([lp_hac(d, y=y, x="x", horizons=[0, 2, 4],
                                     lags=2, controls=["z"])["beta"].to_numpy()
                               for y in ["y", "w"]])
    assert_allclose(actual.values, expected, atol=1e-13)
    assert actual.metadata["n_observations"] == (178, 176, 174, 178, 176, 174)


def test_joint_covariance_matches_dense_bartlett_oracle():
    d = panel()
    n = len(d)
    # Explicit dense time kernel and QR least-squares coefficient loading,
    # independent of the production lag accumulation and normal-equation bread.
    kernel = np.maximum(0, 1 - np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) / 6)
    contributions = []
    for name in ["y", "w"]:
        for h in [0, 2, 4]:
            t = np.arange(2, n-h)
            y, x, z = (d[c].to_numpy() for c in [name, "x", "z"])
            design = np.column_stack([np.ones(len(t)), x[t], x[t-1], y[t-1], z[t-1],
                                      x[t-2], y[t-2], z[t-2], z[t]])
            response = y[t+h] - y[t-1]
            loading = np.linalg.pinv(design)
            errors = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
            score = np.zeros(n)
            score[t] = loading[1] * errors
            contributions.append(score)
    a = np.array(contributions)
    reference = a @ kernel @ a.T
    actual = targets(d)
    assert_allclose(actual.covariance, reference, rtol=2e-13, atol=1e-15)
    assert np.max(np.abs(actual.covariance - np.diag(np.diag(actual.covariance)))) > .001


def test_rescaling_and_response_permutation_preserve_joint_information():
    d = panel()
    initial = targets(d)
    changed = d.copy()
    changed["y"] *= 100
    rescaled = targets(changed)
    scaling = np.array([100, 100, 100, 1, 1, 1])
    assert_allclose(rescaled.values, initial.values * scaling, rtol=1e-10, atol=1e-10)
    assert_allclose(rescaled.covariance, scaling[:, None] * initial.covariance * scaling, rtol=1e-10)
    reordered = targets(d, responses=["w", "y"])
    ix = [3, 4, 5, 0, 1, 2]
    assert_allclose(reordered.covariance, initial.covariance[np.ix_(ix, ix)])
    assert_allclose(targets(d, shock_size=-.25).covariance, initial.covariance * .25**2)
    assert_allclose(targets(d, shock_size=-.25).values, -.25 * initial.values)


@pytest.mark.parametrize("change", [
    {"lags": -1}, {"horizons": [0, 0]}, {"bandwidth": 200},
    {"responses": ["y", "x"]}, {"shock_size": 0}, {"frequency": ""},
    {"response_units": {"y": "log"}}, {"horizons": [170]},
])
def test_invalid_specs_refused(change):
    with pytest.raises(ValueError):
        targets(**change)


def test_missing_and_irregular_periods_are_not_compressed():
    d = panel()
    with pytest.raises(ValueError, match="consecutive"):
        targets(d.drop(d.index[30]))
    d.iloc[30, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        targets(d)


def test_collinear_design_is_refused():
    d = panel()
    d["z"] = d["x"]
    with pytest.raises(np.linalg.LinAlgError):
        targets(d)
