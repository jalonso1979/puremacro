"""puremacro.realized_vol.har_rv: the docstring must describe what is returned
(audit garch-vol-inference: 'realized_vol.har_rv docstring vs returned dict').

Before the fix the docstring promised a key ``beta_0`` (the dict has
``intercept``) and a default bandwidth ``4 (T/100)^(2/9)`` (the code uses
``floor(4 ((T-22)/100)^(2/9))``, the Newey-West rule on the number of
regression observations).
"""
from __future__ import annotations

import re

import numpy as np

from puremacro.realized_vol import har_rv


def _rv(T=400, seed=0):
    rng = np.random.default_rng(seed)
    log_s = np.zeros(T)
    for t in range(1, T):
        log_s[t] = 0.9 * log_s[t - 1] + 0.3 * rng.standard_normal()
    return np.exp(-1.0 + log_s) ** 2


def test_har_rv_docstring_keys_are_the_returned_keys():
    doc = har_rv.__doc__
    assert "beta_0" not in doc.replace("β_0", "")   # only as the formula's name
    out = har_rv(_rv())
    documented = {"intercept", "beta_d", "beta_w", "beta_m", "se_intercept",
                  "se_d", "se_w", "se_m", "R2", "fitted", "residuals",
                  "design", "log_transform", "hac_lags"}
    assert documented == set(out.keys())
    for key in documented:
        assert f"``{key}``" in doc, f"{key} not documented"


def test_har_rv_docstring_bandwidth_rule_matches_code():
    doc = har_rv.__doc__
    assert "n = T - 22" in doc
    assert re.search(r"floor\(4 \* \(n / 100\) \*\* \(2 / 9\)\)", doc)
    for T in (100, 400, 1234):
        out = har_rv(_rv(T))
        n = T - 22
        assert out["hac_lags"] == int(np.floor(4 * (n / 100) ** (2 / 9)))
        assert out["design"].shape == (n, 4)
    assert har_rv(_rv(), hac_lags=0)["hac_lags"] == 0
