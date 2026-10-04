"""puremacro.dp -- declarative front end for dynamic programs.

Declare states, shocks, choices, a reward, constraints and (optionally) an
equilibrium condition as symbolic equations; ``Model.solve`` compiles them to
the ``puremacro.vfi`` solvers. Equations use Dynare expression syntax and are
parsed by the ``.mod`` parser, so the model stays symbolic.

Phase 1 targets discrete VFI (infinite and finite horizon) and scalar-price
stationary general equilibrium. See ``puremacro.dp.model`` for an example.
"""
from __future__ import annotations

from puremacro.dp._expr import ModelSpecError, crra
from puremacro.dp._results import DPSolution
from puremacro.dp.model import Model
from puremacro.dp.processes import AR1, Markov

__all__ = ["AR1", "DPSolution", "Markov", "Model", "ModelSpecError", "crra"]
