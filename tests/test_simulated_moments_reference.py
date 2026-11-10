"""Independent Gaussian identities and stationary-chain timing references."""
import numpy as np

from puremacro.structural import SimulationPlan, simulate_moments
from puremacro.examples.simulation_aware_smm import ar1_moments, markov_moments


def test_ar1_full_finite_sample_covariance_from_gaussian_product_identities():
    # Direct covariance sums, independent of sample covariances and simulator.
    mu, rho, sigma, periods = .4, .6, .5, 128
    plan = SimulationPlan(seed=9402, replications=2048, periods=periods, burn_in=17)
    batch = simulate_moments(ar1_moments, [mu, rho, sigma], plan,
                             labels=("mean", "second", "lag"), definition_id="stationary-ar1-pairs-v1")
    variance = sigma**2/(1-rho**2)
    i, j = np.meshgrid(np.arange(periods), np.arange(periods), indexing="ij")
    k = variance*rho**abs(i-j)
    plus, minus = variance*rho**abs(i-j+1), variance*rho**abs(i-j-1)
    cov11 = k.sum()
    cov12 = (2*mu*k).sum()
    cov13 = (mu*(k+plus)).sum()
    cov22 = (2*k*k+4*mu*mu*k).sum()
    cov23 = (2*k*plus+2*mu*mu*(k+plus)).sum()
    cov33 = (k*k+plus*minus+mu*mu*(2*k+plus+minus)).sum()
    exact = np.array([[cov11, cov12, cov13], [cov12, cov22, cov23],
                      [cov13, cov23, cov33]])/(periods**2*plan.replications)
    sd = np.sqrt(np.diag(exact))
    standardized_cov_error = (batch.covariance-exact)/sd[:, None]/sd[None, :]
    assert np.max(abs(standardized_cov_error)) < .12
    expected = np.array([mu, mu*mu+variance, mu*mu+rho*variance])
    assert np.max(abs((batch.mean-expected)/sd)) < 4
    # Independent scalar finite-sample formula for the serial mean.
    lag = np.arange(1, periods)
    mean_variance = variance*(periods+2*np.sum((periods-lag)*rho**lag))/periods**2/plan.replications
    np.testing.assert_allclose(exact[0, 0], mean_variance, rtol=1e-14)
    assert np.all(batch.counts == periods)


def test_two_state_markov_joint_moments_use_matched_lag_and_transition_timing():
    p01, p10 = .15, .2
    plan = SimulationPlan(seed=8802, replications=256, periods=2048, burn_in=20, role="validation")
    batch = simulate_moments(markov_moments, [p01, p10], plan,
                             labels=("mean", "lag_product", "switching"), definition_id="two-state-pairs-v1")
    stationary = p01/(p01+p10)
    expected = np.array([stationary, stationary*(1-p10), (1-stationary)*p01+stationary*p10])
    error = (batch.mean-expected)/np.sqrt(np.diag(batch.covariance))
    assert np.max(abs(error)) < 4
    assert np.all(batch.counts == plan.periods)
    # Changing the transition matrix must change every reference moment.
    changed = simulate_moments(markov_moments, [.4, .3], plan,
                               labels=batch.labels, definition_id="two-state-pairs-v1")
    assert np.all(abs(changed.mean-batch.mean) > .01)
