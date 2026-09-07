"""The Dynare ``estimated_params`` grammar.

The block is comma-separated and positionally ambiguous: ``NAME, 0.35,
beta_pdf, 0.35, 0.02`` and ``NAME, beta_pdf, 0.35, 0.02`` differ only by a
leading field, ``corr a, b, normal_pdf, 0, 0.2`` carries a comma inside its
target, and the real SW07 file writes shape names in UPPERCASE while the manual
uses lowercase. Each of those is pinned below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge._estimated_params import (
    EstimatedParams,
    EstimatedParamSpec,
    parse_estimated_params,
    parse_estimated_params_bounds,
    parse_estimated_params_init,
)


_PARAMS = ["alpha", "rho", "gam", "psi", "omega", "kappa", "lam"]
_SHOCKS = ["eps_a", "eps_b"]
_VAROBS = ["dy"]

_SYNTHETIC = """
estimated_params;
  alpha, 0.35, beta_pdf, 0.35, 0.02;
  rho, , 0, 1, beta_pdf, 0.5, 0.2;
  gam, normal_pdf, 1, 0.5;
  psi, 0.3, 0.0, 1.0, beta_pdf, 0.5, 0.15, 0.2, 0.8;
  omega, gamma_pdf, 1.0, 0.5, 0.25;
  kappa, uniform_pdf, , , 0.0, 4.0;
  lam, 0.9;
  stderr eps_a, inv_gamma_pdf, 0.1, 2;
  stderr eps_b, 0.05, 0.001, 3, inv_gamma_pdf, 0.1, 2, , , 0.5;
  stderr dy, inv_gamma_pdf, 0.01, 2;
  corr eps_a, eps_b, normal_pdf, 0, 0.2;
end;
"""


@pytest.fixture(scope="module")
def ep() -> EstimatedParams:
    return parse_estimated_params(
        _SYNTHETIC, shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS
    )


def _by_name(ep: EstimatedParams, name: str) -> EstimatedParamSpec:
    return next(s for s in ep.specs if s.name == name)


def test_every_statement_form_parses(ep):
    assert len(ep.specs) == 11
    assert all(isinstance(s, EstimatedParamSpec) for s in ep.specs)


def test_kinds_and_names(ep):
    assert {s.name for s in ep.by_kind("stderr_shock")} == {"SE_eps_a", "SE_eps_b"}
    assert [s.name for s in ep.by_kind("stderr_obs")] == ["ME_dy"]
    assert [s.name for s in ep.by_kind("corr_shock")] == ["CORR_eps_a_eps_b"]
    assert {s.name for s in ep.by_kind("param")} == set(_PARAMS)


def test_corr_comma_does_not_confuse_the_field_split(ep):
    spec = _by_name(ep, "CORR_eps_a_eps_b")
    assert spec.target == ("eps_a", "eps_b")
    assert spec.prior.dist == "normal"
    assert (spec.prior.mean, spec.prior.std) == (0.0, 0.2)


def test_initval_only_form(ep):
    assert _by_name(ep, "alpha").init == 0.35
    assert _by_name(ep, "alpha").prior.dist == "beta"


def test_empty_initval_with_explicit_bounds(ep):
    spec = _by_name(ep, "rho")
    assert spec.init is None
    assert (spec.lb, spec.ub) == (0.0, 1.0)


def test_shape_first_form_has_no_initval(ep):
    assert _by_name(ep, "gam").init is None
    assert _by_name(ep, "gam").prior.dist == "normal"


def test_generalised_beta_picks_up_p3_p4(ep):
    prior = _by_name(ep, "psi").prior
    assert (prior.shift, prior.scale) == pytest.approx((0.2, 0.6))
    assert (prior.mean, prior.std) == (0.5, 0.15)


def test_gamma_picks_up_p3_as_a_shift(ep):
    assert _by_name(ep, "omega").prior.shift == 0.25


def test_uniform_reads_its_bounds_from_p3_p4(ep):
    prior = _by_name(ep, "kappa").prior
    assert (prior.lb, prior.ub) == (0.0, 4.0)


def test_jscale_is_read_from_the_tenth_field(ep):
    assert _by_name(ep, "SE_eps_b").jscale == 0.5
    assert _by_name(ep, "SE_eps_a").jscale == 1.0


def test_no_prior_means_prior_is_none(ep):
    spec = _by_name(ep, "lam")
    assert spec.prior is None
    assert spec.init == 0.9


def test_priors_and_initial_params_helpers(ep):
    priors = ep.priors()
    assert set(priors) == {s.name for s in ep.specs if s.prior is not None}
    init = ep.initial_params()
    assert init["alpha"] == 0.35
    assert init["SE_eps_b"] == 0.05
    # A spec with no INITVAL falls back to the prior mean.
    assert init["gam"] == pytest.approx(1.0)


def test_jscale_vector_follows_the_requested_order(ep):
    names = ["SE_eps_b", "alpha", "SE_eps_a"]
    np.testing.assert_allclose(ep.jscale_vector(names), [0.5, 1.0, 1.0])


def test_shape_names_are_matched_case_insensitively():
    """sw07_pfeifer.mod writes INV_GAMMA_PDF; the manual writes inv_gamma_pdf.
    A case-sensitive match silently reclassifies every statement as
    'estimated without a prior', which is why this is pinned."""
    upper = parse_estimated_params(
        "estimated_params; stderr eps_a,0.4618,0.01,3,INV_GAMMA_PDF,0.1,2; end;",
        shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
    )
    assert upper.specs[0].prior is not None
    assert upper.specs[0].prior.dist == "invgamma"
    assert upper.specs[0].init == 0.4618


def test_unknown_shape_raises_naming_it():
    with pytest.raises(ValueError, match="dirichlet_pdf"):
        parse_estimated_params(
            "estimated_params; alpha, dirichlet_pdf, 1, 1; end;",
            shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
        )


def test_unknown_target_raises_naming_it():
    with pytest.raises(ValueError, match="nosuchparam"):
        parse_estimated_params(
            "estimated_params; nosuchparam, normal_pdf, 0, 1; end;",
            shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
        )


def test_stderr_of_an_unknown_series_raises_naming_it():
    with pytest.raises(ValueError, match="eps_zz"):
        parse_estimated_params(
            "estimated_params; stderr eps_zz, inv_gamma_pdf, 0.1, 2; end;",
            shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
        )


def test_malformed_field_count_quotes_the_statement():
    with pytest.raises(ValueError, match="alpha, 1, 2, 3, 4, 5"):
        parse_estimated_params(
            "estimated_params; alpha, 1, 2, 3, 4, 5, normal_pdf, 0, 1; end;",
            shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
        )


def test_comments_inside_the_block_are_ignored():
    ep = parse_estimated_params(
        "estimated_params;\n"
        "// PARAM NAME, INITVAL, LB, UB, PRIOR_SHAPE, PRIOR_P1, PRIOR_P2\n"
        "  alpha, 0.35, beta_pdf, 0.35, 0.02;\n"
        "end;",
        shocks=_SHOCKS, varobs=_VAROBS, params=_PARAMS,
    )
    assert len(ep.specs) == 1


def test_estimated_params_init_and_bounds_parse():
    init = parse_estimated_params_init(
        "estimated_params_init;\n alpha, 0.4;\n stderr eps_a, 0.7;\nend;"
    )
    assert init == {"alpha": 0.4, "SE_eps_a": 0.7}
    bounds = parse_estimated_params_bounds(
        "estimated_params_bounds;\n alpha, 0.1, 0.9;\nend;"
    )
    assert bounds == {"alpha": (0.1, 0.9)}


def test_sw07_pfeifer_block_round_trips():
    """The real thing: 36 statements, all in the
    `NAME, INITVAL, LB, UB, PRIOR_SHAPE, P1, P2` form with UPPERCASE shape
    names (`stderr ea,0.4618,0.01,3,INV_GAMMA_PDF,0.1,2;`), 7 of them `stderr`.
    """
    import puremacro.dsge as D
    from puremacro.dsge import parse_mod

    text = (Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod").read_text()
    parsed = parse_mod(text)
    ep = parsed["estimated_params"]
    assert ep is not None
    assert len(ep.specs) == 36
    assert all(s.prior is not None for s in ep.specs)
    assert len(ep.by_kind("stderr_shock")) == 7
    assert {s.target[0] for s in ep.by_kind("stderr_shock")} == {
        "ea", "eb", "eg", "eqs", "em", "epinf", "ew"
    }
    assert set(ep.priors()) == {s.name for s in ep.specs}
    # Every spec carries a usable initial value and a finite truncation.
    init = ep.initial_params()
    for s in ep.specs:
        assert np.isfinite(init[s.name])
        assert s.lb < init[s.name] < s.ub


def test_parse_mod_exposes_the_new_keys_without_disturbing_the_old():
    from puremacro.dsge import parse_mod

    src = """
    var c k a; varexo eps;
    parameters alpha beta delta gamma rho;
    alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
    model;
      c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
      k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
      a = rho * a(-1) + eps;
    end;
    initval; k = 38.0; a = 0.0; c = 2.0; end;
    shocks; var eps; stderr 0.01; end;
    varobs c;
    estimated_params;
      rho, beta_pdf, 0.8, 0.1;
      stderr eps, inv_gamma_pdf, 0.01, 2;
    end;
    """
    parsed = parse_mod(src)
    assert parsed["variables"] == ["c", "k", "a"]        # unchanged
    assert parsed["shocks"] == ["eps"]                   # unchanged
    assert parsed["params"]["alpha"] == 0.30             # block did not pollute
    assert parsed["varobs"] == ["c"]
    ep = parsed["estimated_params"]
    assert [s.name for s in ep.specs] == ["rho", "SE_eps"]
    assert parsed["estimated_params_init"] is None
    assert parsed["estimated_params_bounds"] is None


def test_a_mod_file_without_the_block_returns_none():
    from puremacro.dsge import parse_mod

    src = """
    var c a; varexo eps;
    parameters rho; rho = 0.8;
    model; c = a; a = rho * a(-1) + eps; end;
    initval; a = 0.0; c = 0.0; end;
    shocks; var eps; stderr 0.01; end;
    """
    assert parse_mod(src)["estimated_params"] is None


def test_the_solved_model_carries_varobs_and_estimated_params():
    from puremacro.dsge import load_mod

    src = """
    var c k a; varexo eps;
    parameters alpha beta delta gamma rho;
    alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
    model;
      c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
      k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
      a = rho * a(-1) + eps;
    end;
    initval; k = 38.0; a = 0.0; c = 2.0; end;
    shocks; var eps; stderr 0.01; end;
    varobs c;
    estimated_params;
      rho, beta_pdf, 0.8, 0.1;
      stderr eps, inv_gamma_pdf, 0.01, 2;
    end;
    """
    model = load_mod(src)
    assert model._varobs == ("c",)
    assert [s.name for s in model._estimated_params.specs] == ["rho", "SE_eps"]


def test_a_model_built_without_a_mod_file_has_none_metadata():
    """build() models must be unaffected: the fields default to None."""
    from puremacro.dsge import build

    def eqs(xp, x, e, p):
        # build() uses Klein timing: (t+1, t, shocks, params), four arguments.
        return [xp.a - p.rho * x.a - e.eps]

    m = build(eqs, variables=["a"], states=["a"], shocks=["eps"],
              params={"rho": 0.8}, guess={"a": 0.0})
    assert m._varobs is None
    assert m._estimated_params is None
