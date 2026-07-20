"""CI-only drift-guard: recompute the LIVE statsmodels reference and assert the
frozen golden in goldens/state_space.json is still faithful. This is the single
place statsmodels is touched for the state_space validation cases.
Run via: pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

statsmodels = pytest.importorskip("statsmodels")


def _fit(spec):
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    class _UnivariateUC(MLEModel):
        def __init__(self, endog, phi, q, h, a0, P0):
            super().__init__(
                endog, k_states=1, k_posdef=1,
                initialization="known",
                constant=np.asarray(a0, dtype=float),
                stationary_cov=np.asarray(P0, dtype=float),
            )
            self["transition", 0, 0] = float(phi)
            self["design", 0, 0] = 1.0
            self["selection", 0, 0] = 1.0
            self["state_cov", 0, 0] = float(q)
            self["obs_cov", 0, 0] = float(h)

        @property
        def param_names(self):
            return []

        @property
        def start_params(self):
            return np.array([])

    return _UnivariateUC(
        spec["y"], phi=spec["phi"], q=spec["q"], h=spec["h"],
        a0=spec["a0"], P0=spec["P0"],
    ).smooth([])


def test_filter_local_level_golden_matches_live_statsmodels():
    from puremacro.validation._goldens import load_golden
    from puremacro.validation.cases_state_space import local_level_demo

    res = _fit(local_level_demo())
    live_a = np.asarray(res.filtered_state, dtype=float).T.ravel()
    live_ll = float(res.llf)

    g = load_golden("state_space:filter_local_level_vs_statsmodels")
    # If this fails, statsmodels changed its output — regenerate the golden:
    #   python tools/gen_validation_goldens_state_space.py
    np.testing.assert_allclose(live_a, np.asarray(g["a_filt"], dtype=float),
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(live_ll, float(g["loglik"]), rtol=1e-6, atol=1e-9)


def test_smoother_ar1_golden_matches_live_statsmodels():
    from puremacro.validation._goldens import load_golden
    from puremacro.validation.cases_state_space import ar1_noise_demo

    res = _fit(ar1_noise_demo())
    live_smooth = np.asarray(res.smoothed_state, dtype=float).T.ravel()
    live_cov = np.asarray(res.smoothed_state_cov, dtype=float).T.ravel()

    g = load_golden("state_space:smoother_ar1_vs_statsmodels")
    np.testing.assert_allclose(live_smooth, np.asarray(g["a_smooth"], dtype=float),
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(live_cov, np.asarray(g["P_smooth"], dtype=float),
                               rtol=1e-6, atol=1e-9)
