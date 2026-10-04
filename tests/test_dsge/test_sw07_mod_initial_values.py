"""SW07_MOD_INITIAL_VALUES mirrors the estimated_params INITVAL column of sw07_pfeifer.mod."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import puremacro.dsge as D
from puremacro.dsge.sw07_priors import PRIORS, SW07_MOD_INITIAL_VALUES, param_names


def _parse_mod_initvals() -> dict[str, float]:
    text = (Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod").read_text(encoding="utf-8")
    block = re.search(r"estimated_params;(.*?)\nend;", text, re.S).group(1)
    out: dict[str, float] = {}
    for line in block.splitlines():
        line = line.split("//")[0].strip()
        if not line or "," not in line:
            continue
        head, initval = line.split(",")[:2]
        name = head.replace("stderr", "").strip()
        out[name] = float(initval)
    return out


def test_dict_matches_the_mod_file_and_the_prior_order():
    mod = _parse_mod_initvals()
    assert set(mod) == set(SW07_MOD_INITIAL_VALUES) == set(PRIORS)
    for name in param_names():
        assert SW07_MOD_INITIAL_VALUES[name] == mod[name], name


def test_mod_initial_values_lie_inside_the_prior_support():
    """Unlike the rounded Table 1a/1b Mode column, the .mod start needs no clipping."""
    for name, spec in PRIORS.items():
        v = SW07_MOD_INITIAL_VALUES[name]
        assert spec["lb"] < v < spec["ub"], (name, v, spec["lb"], spec["ub"])
    vec = np.array([SW07_MOD_INITIAL_VALUES[n] for n in param_names()])
    assert vec.shape == (36,) and np.isfinite(vec).all()
