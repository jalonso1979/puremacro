"""Unit tests for Dynare canonical lead-lag builder and pure-Python .mod parser."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dsge import (
    build_dynare,
    parse_mod,
    load_mod,
    LinearModel,
)


def test_build_dynare_auto_detect_states():
    # User writes in lead, curr, lag form without declaring states
    def rbc_lead_lag(lead, curr, lag, e, p):
        return [
            curr.c**-p.sigma - p.beta * lead.c**-p.sigma * (p.alpha * lead.z * curr.k**(p.alpha - 1) + 1 - p.delta),
            curr.c + curr.k - (1 - p.delta) * lag.k - curr.z * lag.k**p.alpha,
            curr.z - (1 - p.rho) - p.rho * lag.z - e.eps,
        ]

    m = build_dynare(
        rbc_lead_lag,
        variables=["c", "k", "z"],
        shocks=["eps"],
        params=dict(alpha=0.33, beta=0.99, delta=0.025, sigma=1.0, rho=0.95),
        guess=dict(c=2.0, k=25.0, z=1.0),
    )

    assert isinstance(m, LinearModel)
    # k and z appear with lag.k and lag.z -> automatically detected as states
    assert m.states == ("k", "z")
    # c only appears at t and t+1 -> automatically detected as control
    assert m.controls == ("c",)

    # Transition matrix properties
    assert m.solution.G[0, 0] == pytest.approx(0.962061, rel=1e-3)
    assert m.solution.G[1, 1] == pytest.approx(0.95, rel=1e-3)


def test_parse_mod_string():
    mod_text = """
    // Comments with double slashes
    % Comments with percent sign
    /* Block comment
       spanning multiple lines */
    var c, k, z;
    varexo eps;
    parameters alpha, beta, delta, rho, sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    log(z) = rho * log(z(-1)) + eps;
    end;

    initval;
    k = 25.0;
    c = 2.0;
    z = 1.0;
    end;
    """

    res = parse_mod(mod_text)
    assert res["variables"] == ["c", "k", "z"]
    assert res["shocks"] == ["eps"]
    assert res["params"]["alpha"] == pytest.approx(0.33)
    assert res["params"]["beta"] == pytest.approx(0.99)
    assert res["guess"] == {"k": 25.0, "c": 2.0, "z": 1.0}
    assert callable(res["equations"])


def test_load_mod_from_string_and_file(tmp_path):
    mod_text = """
    var c, k, z;
    varexo eps;
    parameters alpha, beta, delta, rho, sigma;
    alpha = 0.33;
    beta = 0.99;
    delta = 0.025;
    rho = 0.95;
    sigma = 1.0;

    model;
    c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * z(+1) * k^(alpha-1) + 1 - delta);
    c + k - (1-delta)*k(-1) = z * k(-1)^alpha;
    log(z) = (1-rho)*0 + rho * log(z(-1)) + eps;
    end;

    initval;
    k = 25.0;
    c = 2.0;
    z = 1.0;
    end;
    """

    # 1. From string
    m_str = load_mod(mod_text)
    assert isinstance(m_str, LinearModel)
    assert m_str.states == ("k", "z")
    assert m_str.controls == ("c",)

    # 2. From file
    mod_file = tmp_path / "rbc_test.mod"
    mod_file.write_text(mod_text, encoding="utf-8")

    m_file = load_mod(mod_file, params=dict(rho=0.90))
    assert isinstance(m_file, LinearModel)
    # Parameter override check
    assert m_file.solution.G[1, 1] == pytest.approx(0.90, rel=1e-3)

    # Check IRF and simulation work out of the box
    irf_df = m_file.irf("eps", horizon=10)
    assert len(irf_df) == 11
    assert "c" in irf_df.columns
    assert "k" in irf_df.columns

    sim_df = m_file.simulate(periods=50)
    assert len(sim_df) == 50
