"""puremacro.inference.newey_west must be an alias of puremacro.inference.hac
(audit garch-vol-inference: 'Duplicate inference/newey_west.py module').

Before the fix the module held an unreferenced duplicate implementation that
inverted ``X'X`` with a bare ``np.linalg.inv``; a rank-deficient design gave
garbage standard errors from one copy and a named ``LinAlgError`` from the
other.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

# ``from puremacro.inference import newey_west`` returns the FUNCTION alias
# (the package re-exports ``hac.newey_west_se`` under that name), so fetch
# the submodule explicitly.
nw_mod = importlib.import_module("puremacro.inference.newey_west")
hac = importlib.import_module("puremacro.inference.hac")


def test_newey_west_module_reexports_hac_implementation():
    assert nw_mod.newey_west_se is hac.newey_west_se
    from puremacro.inference import newey_west as top_level_alias
    assert top_level_alias is hac.newey_west_se


def test_rank_deficient_design_raises_named_error_from_both_paths():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x, 2.0 * x])   # collinear
    resid = rng.standard_normal(n)
    with pytest.raises(np.linalg.LinAlgError, match="newey_west_se"):
        nw_mod.newey_west_se(X, resid, bw=2)
    with pytest.raises(np.linalg.LinAlgError, match="newey_west_se"):
        hac.newey_west_se(X, resid, bw=2)
