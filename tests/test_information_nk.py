"""Independent full-information NK and explicit present-value references."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.expectations import GaussianSignal, build_gaussian_information
from puremacro.models.information_nk import (
    InformationNKParameters, solve_information_nk, audit_information_nk,
)


def information(T=12, full=False):
    signals = ((GaussianSignal(0, [1, 1, 0], "a"), GaussianSignal(8, [1, 0, 0], "reveal")),
               (GaussianSignal(0, [1, 0, 2], "b"), GaussianSignal(8, [1, 0, 0], "reveal")))
    if full:
        signals = tuple(tuple(GaussianSignal(0, row, str(j)) for j, row in enumerate(np.eye(3))) for _ in range(2))
    return build_gaussian_information(np.eye(3), signals, horizon=T)


def rates(T=12):
    R = np.zeros((T, 3))
    R[8, 0] = -.0025
    return R


def test_full_information_matches_textbook_backward_recursion():
    model = InformationNKParameters()
    R = rates()
    solution = solve_information_nk(information(full=True), R, parameters=model)
    y, pi = np.zeros((13, 3)), np.zeros((13, 3))
    for t in range(11, -1, -1):
        y[t] = y[t+1]-model.intertemporal_elasticity*(R[t]-pi[t+1])
        pi[t] = model.kappa*y[t]+model.beta*pi[t+1]
    np.testing.assert_allclose(solution.output_loadings, y[:-1], atol=1e-12)
    np.testing.assert_allclose(solution.inflation_loadings, pi[:-1], atol=1e-12)
    assert solution.converged and audit_information_nk(solution)["accepted"]


def test_present_values_hold_for_every_primitive_direction():
    info, model = information(), InformationNKParameters()
    solution = solve_information_nk(info, rates(), parameters=model, firm_weights=[.7, .3])
    Y, Pi, R = solution.output_loadings, solution.inflation_loadings, solution.rate_loadings
    for t in range(12):
        Pc = np.einsum("g,gij->ij", info.weights, info.projectors[t])
        Pf = np.einsum("g,gij->ij", solution.firm_weights, info.projectors[t])
        demand = np.zeros(3)
        prices = model.kappa*Y[t]
        for j in range(t, 12):
            future_pi = Pi[j+1] if j+1 < 12 else np.zeros(3)
            demand -= model.intertemporal_elasticity*model.beta**(j-t)*(R[j]-future_pi)@Pc
            if j > t:
                demand += (1-model.beta)*model.beta**(j-t-1)*Y[j]@Pc
                prices += (model.beta*model.price_stickiness)**(j-t)*(
                    model.kappa*Y[j]+(1-model.price_stickiness)/model.price_stickiness*Pi[j])@Pf
        np.testing.assert_allclose(Y[t], demand, atol=1e-12)
        np.testing.assert_allclose(Pi[t], prices, atol=1e-12)


def test_zero_tail_padding_preserves_all_earlier_coefficients():
    short = solve_information_nk(information(12), rates(12))
    long = solve_information_nk(information(20), rates(20))
    for field in ("output_loadings", "inflation_loadings"):
        np.testing.assert_allclose(getattr(short, field), getattr(long, field)[:12], atol=1e-12)
        np.testing.assert_array_equal(getattr(long, field)[12:], 0.)


def test_no_news_before_revelation_and_zero_policy():
    info = build_gaussian_information(np.eye(1), ((GaussianSignal(8, [1], "reveal"),),), horizon=12)
    R = np.zeros((12, 1))
    R[8] = -.0025
    result = solve_information_nk(info, R)
    np.testing.assert_array_equal(result.output_loadings[:8], 0.)
    assert result.output_loadings[8, 0] > 0
    zero = solve_information_nk(info, np.zeros_like(R))
    np.testing.assert_array_equal(zero.output_loadings, 0.)


def test_realized_path_and_group_forecasts_use_the_same_information():
    solution = solve_information_nk(information(), rates())
    z = np.array([1., -.2, .4])
    path = solution.evaluate(z)
    np.testing.assert_allclose(path.output, solution.output_loadings@z)
    for t in range(11):
        for g in range(2):
            expected = solution.information.forecast(solution.inflation_loadings[t+1], z, date=t, group=g)
            assert path.next_inflation_forecasts[t, g] == pytest.approx(expected)
    assert np.max(abs(path.next_output_forecasts[:, 0]-path.next_output_forecasts[:, 1])) > 1e-5
    assert path.to_frame().shape[0] == 12
    assert "tabular" in path.to_latex() and "table" in path.to_typst()


def test_group_permutation_and_splitting_preserve_solution():
    original = information()
    flipped = build_gaussian_information(np.eye(3), original.signals[::-1], horizon=12)
    split = build_gaussian_information(np.eye(3), (original.signals[0], original.signals[0], original.signals[1]),
                                      horizon=12, weights=[.25, .25, .5])
    reference = solve_information_nk(original, rates())
    for info in (flipped, split):
        other = solve_information_nk(info, rates())
        np.testing.assert_allclose(other.output_loadings, reference.output_loadings, atol=1e-12)
        np.testing.assert_allclose(other.inflation_loadings, reference.inflation_loadings, atol=1e-12)


def test_audit_rejects_coefficient_tampering():
    result = solve_information_nk(information(), rates())
    changed = result.output_loadings.copy()
    changed[0, 0] += 1e-4
    assert not audit_information_nk(replace(result, output_loadings=changed))["accepted"]


@pytest.mark.parametrize("kwargs", [dict(beta=1), dict(price_stickiness=0), dict(price_stickiness=1),
    dict(intertemporal_elasticity=0), dict(inverse_frisch=-1), dict(beta=np.nan)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        InformationNKParameters(**kwargs)


def test_rate_shape_and_nonfinite_inputs_rejected():
    info = information()
    with pytest.raises(ValueError):
        solve_information_nk(info, np.zeros((12, 2)))
    with pytest.raises(ValueError):
        solve_information_nk(info, np.full((12, 3), np.nan))


def test_solver_rejects_tampered_information():
    info = information()
    valid = solve_information_nk(info, rates())
    bad = info.projectors.copy()
    bad[0] = np.eye(3)
    changed = replace(info, projectors=bad)
    with pytest.raises(ValueError, match="information set audit"):
        solve_information_nk(changed, rates())
    assert not audit_information_nk(replace(valid, information=changed))["accepted"]


def test_terminal_forecasts_and_evaluation_snapshot():
    solution = solve_information_nk(information(), rates())
    z = np.array([1., 0., 0.])
    path = solution.evaluate(z)
    z[0] = 9
    assert path.realization[0] == 1
    for name in ("next_output_forecasts", "next_inflation_forecasts", "next_rate_forecasts"):
        np.testing.assert_array_equal(getattr(path, name)[-1], 0.)
    frame = path.to_frame()
    frame.iloc[0, 0] = 1e9
    assert path.to_frame().iloc[0, 0] != 1e9
