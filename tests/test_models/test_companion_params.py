from __future__ import annotations

import dataclasses

import pytest

from puremacro.models.nested_dmp.params import NestedDMPParameters


def test_defaults_construct_and_are_frozen():
    p = NestedDMPParameters()
    # Hosios baseline + NB43-anchored beliefs (phi_D<0<phi_H), policy lever off.
    assert p.alpha == 0.5
    assert p.phi_D < 0.0 < p.phi_H
    assert p.phi_sigma == 0.0  # status-quo "no-response" rule
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.alpha = 0.6  # type: ignore[misc]


def test_validation_rejects_out_of_range():
    with pytest.raises(ValueError, match="beta"):
        NestedDMPParameters(beta=1.5)
    with pytest.raises(ValueError, match="alpha"):
        NestedDMPParameters(alpha=0.0)
    with pytest.raises(ValueError, match="n_x"):
        NestedDMPParameters(n_x=1)
    with pytest.raises(ValueError, match="phi_D"):
        NestedDMPParameters(phi_D=2.0)  # dove must cut: phi_D < 0
    with pytest.raises(ValueError, match="phi_H"):
        NestedDMPParameters(phi_H=0.0)  # hawk must hike: phi_H > 0


def test_phi_defaults_are_decimal_and_delta_present():
    p = NestedDMPParameters()
    # phi now stored in DECIMAL rate units (NB43 -320 bps = -0.0320), matching
    # calibrate_dmp_signal_extraction; pi* = phi_H/(phi_H-phi_D) = 0.319.
    assert p.phi_D == pytest.approx(-0.0320)
    assert p.phi_H == pytest.approx(0.015)
    assert (p.phi_H / (p.phi_H - p.phi_D)) == pytest.approx(0.319, abs=1e-3)
    assert p.delta == 0.0  # firm-exit hazard, off by default


def test_post_init_guard_rejects_rate_below_pole():
    # r* + (phi_D - phi_sigma)*sigma must stay > -1 on the intended sigma support
    # or beta(r)=1/(1+r) crosses the pole. Defaults are safe; an extreme phi_D is not.
    with pytest.raises(ValueError, match="pole|admissib"):
        NestedDMPParameters(phi_D=-30.0)  # decimal -30 => r far below -1 at sigma=1


def test_h_max_default_off_and_validated():
    p = NestedDMPParameters()
    assert p.h_max == 0.0  # participation off by default (2-state {E,U} core)
    with pytest.raises(ValueError, match="h_max"):
        NestedDMPParameters(h_max=-0.5)
