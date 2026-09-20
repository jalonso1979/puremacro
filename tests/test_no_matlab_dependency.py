"""puremacro must reproduce its documented results with Python alone.

Before 4.0.0 three separate things tied the project to MATLAB:

* the trade model calibrated from ``data_77c_11s.mat`` in a directory *above*
  the checkout, so every trade parity suite silently skipped for anyone but
  the author, and CI reported errors rather than skips;
* ``puremacro.dsge`` read Dynare's ``*_results.mat`` through ``scipy.io``;
* a ``matlab/`` companion toolbox shipped a second, unaudited implementation.

All three are gone. These tests fail if any of them comes back, because the
failure mode is quiet: a module grows a ``scipy.io`` import, or a loader
reaches outside the distribution, and nobody notices until a stranger tries to
reproduce a published number.
"""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import puremacro

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Subpackages that are shipped samples rather than library code.
_NON_LIBRARY = (".tests", ".examples", ".teaching")


def test_importing_every_module_never_loads_scipy_io():
    """No library module may import ``scipy.io``, directly or transitively."""
    code = "\n".join([
        "import importlib, pkgutil, sys, warnings",
        "warnings.simplefilter('ignore')",
        "import puremacro",
        f"skip = {_NON_LIBRARY!r}",
        "for m in pkgutil.walk_packages(puremacro.__path__, 'puremacro.'):",
        "    if any(s in m.name for s in skip):",
        "        continue",
        "    try:",
        "        importlib.import_module(m.name)",
        "    except Exception:",
        "        pass",
        "print('scipy.io' in sys.modules)",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("False"), (
        "A puremacro module imports scipy.io again. puremacro reads no MATLAB "
        "files; a caller who has one loads it and passes the mapping.\n"
        f"stdout: {proc.stdout[-500:]}"
    )


def test_no_matlab_sources_are_distributed():
    """Runtime sources carry no MATLAB files; development references are explicit."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if tracked.returncode != 0:  # not a git checkout, e.g. an unpacked sdist
        pytest.skip("not a git checkout")
    # These files reproduce external evidence and are excluded from wheel/sdist.
    # Keep an exact allowlist: a new runtime dependency must still fail here.
    development_references = {
        "tools/reference_validation/export_dynare.m",
        "curso/notebooks/modelos/rbc_mexico_dual/Output/rbc_mexico_dual_results.mat",
    }
    offenders = [
        line for line in tracked.stdout.split("\n")
        if line.endswith((".m", ".mat", ".mlx"))
        and line not in development_references
    ]
    assert not offenders, (
        "MATLAB files are tracked again: " + ", ".join(sorted(offenders)[:10])
    )


def test_trade_calibration_data_ships_with_the_package():
    """The ICIO matrix loads from the installation, not from a sibling directory."""
    from puremacro.trade.data import bundled_icio_path, load_icio_data

    path = bundled_icio_path()
    assert path.is_file(), f"bundled ICIO dataset missing at {path}"
    assert path.suffix == ".npz"
    # It must live inside the installed package, not beside the checkout.
    assert Path(puremacro.__file__).resolve().parent in path.parents

    matrix = load_icio_data()
    assert matrix.shape == (850, 1078)
    assert matrix.dtype == np.float64


def test_reference_solutions_ship_and_stay_external_evidence():
    """The MATLAB-derived reference solutions are bundled and still verbatim."""
    from puremacro.trade.data import (
        available_reference_scenarios,
        load_reference_solution,
    )

    scenarios = available_reference_scenarios()
    assert "base" in scenarios and "t10" in scenarios

    base = load_reference_solution("base")
    # Shapes are MATLAB's, because these are unmodified copies: turning them
    # into puremacro's own output would make the parity suites tautological.
    assert base["xx_sol"].shape == (2001, 1)
    assert base["xx_sol"].dtype == np.float64

    # t10_54's source is a dataless placeholder that could not be read. It must
    # stay absent rather than be fabricated or regenerated from puremacro.
    assert "t10_54" not in scenarios
    with pytest.raises(KeyError, match="t10_54"):
        load_reference_solution("t10_54")


def test_mat_input_is_refused_with_a_migration_message():
    """A ``.mat`` path raises and names what to do instead."""
    from puremacro.trade.data import load_icio_data

    with pytest.raises(ValueError, match=r"(?i)no longer supported|savez_compressed"):
        load_icio_data(path="anything.mat")


def test_dynare_helpers_take_a_mapping_not_a_path():
    """The Dynare bridge consumes an ``oo_`` mapping the caller loaded."""
    from puremacro.dsge import load_dynare_dr, load_dynare_moments

    for fn in (load_dynare_dr, load_dynare_moments):
        with pytest.raises(TypeError, match=r"(?i)no longer reads MATLAB|loadmat"):
            fn("some_results.mat")

    # A mapping without oo_ is the caller's mistake, and still a KeyError.
    with pytest.raises(KeyError, match=r"(?i)oo_"):
        load_dynare_dr({"unrelated": 1})


def test_removed_module_name_is_gone():
    """``puremacro.dsge.load_dynare`` was renamed; the old path must not resolve."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("puremacro.dsge.load_dynare")
