"""The 4.6.0 SW07 estimation options: presample, diffuse initialisation, datasets, starts.

The Table 2 marginal likelihood of Smets and Wouters (2007) conditions on a
1956Q1-1965Q4 training sample; ``estimate_dsge(presample=..., lik_init=...)``
and :mod:`puremacro.dsge.sw07_marginal` implement that. These tests pin the
semantics with cheap single evaluations (one SW07 likelihood costs ~0.03 s);
the mode search itself is exercised by the replication case built from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dsge import load_sw07_data, sw07_log_posterior
from puremacro.dsge.estimate import _make_neg_log_posterior
from puremacro.dsge.sw07_data import SW07_DATASETS
from puremacro.dsge.sw07_estimate import _FIXED_PARAMS, _start_values
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import PRIORS, SW07_MOD_INITIAL_VALUES, param_names
from puremacro.state_space import kalman_filter


def _vec():
    return np.array([SW07_MOD_INITIAL_VALUES[n] for n in param_names()])


def _y(dataset="authors", first_obs="1956Q1", last_obs="2004Q4"):
    return load_sw07_data(dataset, first_obs=first_obs, last_obs=last_obs)[list(OBSERVED_VARS)].to_numpy()


def _period_loglik(out) -> np.ndarray:
    """Per-period Gaussian log-likelihood contributions from the filter output."""
    innov, F = out["innov"], out["F"]
    n = innov.shape[1]
    ll = np.empty(len(innov))
    for t in range(len(innov)):
        sign, logdet = np.linalg.slogdet(F[t])
        ll[t] = -0.5 * (n * np.log(2 * np.pi) + logdet + innov[t] @ np.linalg.solve(F[t], innov[t]))
    return ll


class TestDatasets:
    def test_both_datasets_load_with_the_observable_columns(self):
        fred = load_sw07_data("fred")
        authors = load_sw07_data("authors")
        assert list(fred.columns) == list(authors.columns) == list(OBSERVED_VARS)
        assert (fred.index[0], fred.index[-1], len(fred)) == ("1966Q1", "2004Q4", 156)
        assert (authors.index[0], authors.index[-1], len(authors)) == ("1947Q3", "2004Q4", 230)
        assert set(SW07_DATASETS) == {"fred", "authors"}

    def test_slicing_is_inclusive_and_validated(self):
        df = load_sw07_data("authors", first_obs="1956Q1", last_obs="1965Q4")
        assert len(df) == 40 and df.index[0] == "1956Q1" and df.index[-1] == "1965Q4"
        with pytest.raises(ValueError, match="first_obs"):
            load_sw07_data("authors", first_obs="1940Q1")
        with pytest.raises(ValueError, match="unknown SW07 dataset"):
            load_sw07_data("vintage2006")

    def test_authors_and_fred_series_agree_over_the_estimation_sample(self):
        """Same definitions, different vintages: the federal funds rate is identical."""
        a = load_sw07_data("authors", first_obs="1966Q1")
        f = load_sw07_data("fred")
        # Same FRED series, quarterly average of monthly data; rounding differs at the 3rd decimal.
        assert np.corrcoef(a["ffr"], f["ffr"])[0, 1] > 0.9999
        assert np.max(np.abs(a["ffr"].to_numpy() - f["ffr"].to_numpy())) < 0.02
        for col in ("gdp_growth", "cons_growth", "inv_growth", "infl"):
            assert np.corrcoef(a[col], f[col])[0, 1] > 0.98, col


class TestPresample:
    def test_presample_drops_exactly_the_first_contributions(self):
        """lik(presample=k) equals the sum of the per-period contributions after k, from the same filter."""
        y = _y(first_obs="1960Q1")
        params = {**dict(zip(param_names(), _vec())), **_FIXED_PARAMS}
        ssm = make_state_space(params)
        m = ssm.T.shape[0]
        a0, P0 = np.zeros(m), 10.0 * np.eye(m)
        full = kalman_filter(y, ssm, a0=a0, P0=P0)
        tail = _period_loglik(full)[8:].sum()
        nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS,
                                      presample=8, lik_init="diffuse", diffuse_scale=10.0)
        from puremacro.dsge.priors import log_prior
        lp = log_prior(params, PRIORS)
        assert np.isclose(-nlp(_vec()) - lp, tail, rtol=0, atol=1e-6)

    def test_zero_presample_and_stationary_init_reproduce_the_previous_closure(self):
        y = _y(dataset="fred", first_obs="1966Q1")
        old = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS)
        new = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS,
                                      presample=0, lik_init="stationary")
        assert old(_vec()) == new(_vec())

    def test_diffuse_and_stationary_initialisations_differ(self):
        y = _y(first_obs="1966Q1")
        st = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS)
        df = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS, lik_init="diffuse")
        assert np.isfinite(st(_vec())) and np.isfinite(df(_vec()))
        assert abs(st(_vec()) - df(_vec())) > 1.0

    def test_invalid_options_are_rejected_early(self):
        y = _y(first_obs="1990Q1")
        with pytest.raises(ValueError, match="lik_init"):
            _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS, lik_init="exact")
        with pytest.raises(ValueError, match="presample"):
            _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS, presample=len(y))

    def test_sw07_log_posterior_matches_the_closure(self):
        y = _y()
        nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, param_names(), _FIXED_PARAMS,
                                      presample=40, lik_init="diffuse")
        direct = sw07_log_posterior(SW07_MOD_INITIAL_VALUES)
        assert np.isclose(direct, -nlp(_vec()), rtol=0, atol=1e-9)
        assert -1100 < direct < -900


class TestStarts:
    def test_mod_start_is_the_default_and_needs_no_clipping(self):
        from puremacro.dsge.estimate import _initial_vec_from_dict
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vec = _initial_vec_from_dict(_start_values("mod"), PRIORS)
        np.testing.assert_array_equal(vec, _vec())

    def test_table1_start_is_still_available_and_is_clipped_with_a_warning(self):
        from puremacro.dsge.estimate import _initial_vec_from_dict
        with pytest.warns(UserWarning, match="outside the prior support"):
            _initial_vec_from_dict({k: v for k, v in _start_values("table1").items() if k in PRIORS}, PRIORS)

    def test_unknown_start_is_rejected(self):
        with pytest.raises(ValueError, match="start must be"):
            _start_values("paper")


def test_estimate_dsge_rejects_presample_outside_the_kalman_path():
    from puremacro.dsge import estimate_sw07
    with pytest.raises(ValueError, match="presample"):
        estimate_sw07(presample=-1, n_draws=10, n_chains=1, burn_in=5)
