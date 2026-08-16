"""Climate and environmental macroeconomics for puremacro.

Contains:
- Dynamic Integrated Climate-Economy (DICE) model (Nordhaus / Golosov et al. 2014).
- Optimal Social Cost of Carbon (SCC) and carbon tax path projections.
- Multi-reservoir carbon cycle and climate temperature anomaly dynamics.
"""
from puremacro.climate.dice import DICEResult, simulate_dice_model

__all__ = [
    "DICEResult",
    "simulate_dice_model",
]
