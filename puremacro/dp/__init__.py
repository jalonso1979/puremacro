"""puremacro.dp -- declarative front end for dynamic programs.

Declare states, shocks, choices, a reward, constraints and (optionally) an
equilibrium condition as symbolic equations; ``Model.solve`` compiles them to
the ``puremacro.vfi`` solvers. Equations use Dynare expression syntax and are
parsed by the ``.mod`` parser, so the model stays symbolic.

Targets: discrete VFI (infinite and finite horizon), the endogenous grid
method with an optional discrete choice and taste shocks (``method="egm"``),
the continuous-time HJB upwind scheme for models declared with ``drift``
(``method="hjb"``), and scalar-price stationary general equilibrium for each.
See ``puremacro.dp.model`` for an example.
"""
from __future__ import annotations

from puremacro.dp._expr import ModelSpecError, crra
from puremacro.dp._results import DPSolution
from puremacro.dp.model import Model
from puremacro.dp.processes import AR1, Jump, Markov

__all__ = ["AR1", "DPSolution", "Jump", "Markov", "Model", "ModelSpecError", "crra"]
