"""Flexible CGE engine extensions for puremacro.trade.

This module implements:
- Requirement R1: Flexible Nested CES Technology in Calibrated Share Form (CSF),
  including inner nest (value-added) CES unit costs, exact Cobb-Douglas gating,
  normalized factor demand derivatives, outer nest (gross output) CES unit costs,
  exact Leontief gating, intermediate composite pricing, two-tier intermediate demands,
  and zero-profit producer pricing.
- Declarative configuration dataclasses and stubs for general equilibrium integration.

Conforms strictly to the puremacro Pyodide runtime contract (NumPy, SciPy, Pandas only).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import sys
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult
from puremacro.trade.equilibrium import _get_ces_weights
from puremacro.trade.solver import solve_trade_equilibrium

# Ensure backward-compatible alias property calib.pfd -> calib.afd
if not hasattr(TradeCalibrationResult, "pfd"):
    setattr(TradeCalibrationResult, "pfd", property(lambda self: getattr(self, "afd", None)))


def _get_replace_caller() -> Any | None:
    """Inspect call stack to retrieve the original instance if called via dataclasses.replace."""
    for i in range(1, 6):
        try:
            f = sys._getframe(i)
            if f.f_code.co_name == "_replace" and "self" in f.f_locals:
                return f.f_locals["self"]
        except (AttributeError, ValueError):
            break
    return None


def _get_replace_explicit_keys() -> set[str] | None:
    """Retrieve explicit keys passed to dataclasses.replace, if available."""
    for i in range(1, 6):
        try:
            f = sys._getframe(i)
            if f.f_code.co_name == "_replace":
                f_parent = sys._getframe(i + 1)
                if f_parent.f_code.co_name == "replace" and "changes" in f_parent.f_locals:
                    return set(f_parent.f_locals["changes"].keys())
        except (AttributeError, ValueError):
            break
    return None


def _val_equal(a: Any, b: Any) -> bool:
    """Check equality between config parameter values, handling None, arrays, and scalars."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    return bool(a == b)


# =============================================================================
# R1. Flexible Nested CES Technology Configuration
# =============================================================================

@dataclass(frozen=True)
class FlexibleTechnologyConfig:
    """Configuration for flexible nested CES production technology.

    Parameters
    ----------
    rho_va : float, default 1.0
        Elasticity of substitution between capital and labor in the value-added nest.
        Aliased with `sigma_va`. Must be strictly positive (rho_va > 0).
    sigma_y : float, default 0.0
        Elasticity of substitution between value-added and intermediate composite
        in gross output. Must be non-negative (sigma_y >= 0).
    sigma_inter : float, default 0.0
        Elasticity of substitution in intermediate input sourcing across origins.
        Must be non-negative (sigma_inter >= 0).
    capacity_margins : Mapping[str | tuple[int, int], float] | np.ndarray | None, default None
        Sectoral capacity buffer limits for smooth barrier penalties.
    penalty_scale : float, default 0.05
        Scale parameter for capacity barrier penalty.
    penalty_exponent : float, default 2.0
        Exponent for capacity barrier penalty.
    replicate_matlab_precedence : bool, default False
        Whether to replicate legacy MATLAB operator precedence in Cobb-Douglas pricing.
    sigma_va : float | None, default None
        Alias for rho_va. If provided, sets rho_va.
    """

    rho_va: float = 1.0
    sigma_y: float = 0.0
    sigma_inter: float = 0.0
    capacity_margins: Any = None
    penalty_scale: float = 0.05
    penalty_exponent: float = 2.0
    replicate_matlab_precedence: bool = False
    sigma_va: float | None = None

    def __post_init__(self) -> None:
        # Type validation for rho_va and sigma_va
        if not isinstance(self.rho_va, (int, float)) or isinstance(self.rho_va, bool):
            raise TypeError(
                f"rho_va must be a numeric float, got {type(self.rho_va).__name__}."
            )
        if self.sigma_va is not None:
            if not isinstance(self.sigma_va, (int, float)) or isinstance(self.sigma_va, bool):
                raise TypeError(
                    f"sigma_va must be a numeric float, got {type(self.sigma_va).__name__}."
                )

        old_inst = _get_replace_caller()
        if old_inst is not None and isinstance(old_inst, FlexibleTechnologyConfig):
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                rho_changed = "rho_va" in exp_keys
                sigma_changed = "sigma_va" in exp_keys
            else:
                rho_changed = (self.rho_va != old_inst.rho_va)
                sigma_changed = (self.sigma_va is not None and self.sigma_va != old_inst.sigma_va)

            if rho_changed and not sigma_changed:
                target_rho = self.rho_va
            elif sigma_changed and not rho_changed:
                target_rho = self.sigma_va
            elif rho_changed and sigma_changed:
                if self.rho_va != self.sigma_va:
                    raise ValueError(
                        f"Conflicting values provided for rho_va ({self.rho_va}) "
                        f"and sigma_va ({self.sigma_va})."
                    )
                target_rho = self.rho_va
            else:
                target_rho = self.rho_va
        else:
            target_rho = self.rho_va
            if self.sigma_va is not None:
                if self.rho_va != 1.0 and self.rho_va != self.sigma_va:
                    raise ValueError(
                        f"Conflicting values provided for rho_va ({self.rho_va}) "
                        f"and sigma_va ({self.sigma_va})."
                    )
                target_rho = self.sigma_va

        if not isinstance(self.sigma_y, (int, float)) or isinstance(self.sigma_y, bool):
            raise TypeError(
                f"sigma_y must be a numeric float, got {type(self.sigma_y).__name__}."
            )
        if not isinstance(self.sigma_inter, (int, float)) or isinstance(self.sigma_inter, bool):
            raise TypeError(
                f"sigma_inter must be a numeric float, got {type(self.sigma_inter).__name__}."
            )

        float_rho = float(target_rho)
        float_sig_y = float(self.sigma_y)
        float_sig_inter = float(self.sigma_inter)

        object.__setattr__(self, "rho_va", float_rho)
        object.__setattr__(self, "sigma_va", float_rho)
        object.__setattr__(self, "sigma_y", float_sig_y)
        object.__setattr__(self, "sigma_inter", float_sig_inter)

        # Economic range validation with plain-English economic explanations
        if float_rho <= 0.0:
            raise ValueError(
                f"rho_va (value-added substitution elasticity) must be strictly positive (got {float_rho}). "
                "A non-positive elasticity of substitution violates strict quasi-concavity of the production "
                "function and results in undefined factor cost minimization."
            )
        if float_sig_y < 0.0:
            raise ValueError(
                f"sigma_y (gross output substitution elasticity) must be non-negative (got {float_sig_y}). "
                "A negative elasticity of substitution violates cost minimization and produces upward-sloping demand."
            )
        if float_sig_inter < 0.0:
            raise ValueError(
                f"sigma_inter (intermediate sourcing elasticity) must be non-negative (got {float_sig_inter}). "
                "Negative substitution elasticity implies complementary intermediate varieties with perverse cross-price responses."
            )


# =============================================================================
# Declarative Configs for Preferences & Market Structure (M2-M5 Stubs)
# =============================================================================

@dataclass(frozen=True)
class FlexiblePreferenceConfig:
    """Configuration for flexible non-homothetic preferences (Stone-Geary LES).

    Parameters
    ----------
    subsistence_shares : float | Mapping[str, float] | np.ndarray | None, default None
        Subsistence consumption expenditure shares mu_s in [0, 1) across sectors.
        Represents the committed subsistence expenditure as a fraction of baseline
        consumption. Default 0.0 for all sectors (standard homothetic Cobb-Douglas).
        Aliased with `mu_s` and `subsistence_ratio`.
    sigma_trade : float | np.ndarray, default 5.0
        Armington elasticity of substitution across source countries in international
        trade sourcing. Must be strictly greater than 1.0 (empirically in [4.0, 8.0]).
    mu_s : float | Mapping[str, float] | np.ndarray | None, default None
        Alias for `subsistence_shares`. If multiple subsistence parameters are specified
        and differ, raises ValueError.
    subsistence_ratio : float | Mapping[str, float] | np.ndarray | None, default None
        Alias for `subsistence_shares` and `mu_s`.
    """

    subsistence_shares: Any = None
    sigma_trade: float | np.ndarray = 5.0
    mu_s: Any = None
    subsistence_ratio: Any = None

    def __post_init__(self) -> None:
        # Reject booleans explicitly
        if isinstance(self.sigma_trade, bool):
            raise TypeError("sigma_trade cannot be a boolean.")
        if isinstance(self.subsistence_shares, bool):
            raise TypeError("subsistence_shares cannot be a boolean.")
        if isinstance(self.mu_s, bool):
            raise TypeError("mu_s cannot be a boolean.")
        if isinstance(self.subsistence_ratio, bool):
            raise TypeError("subsistence_ratio cannot be a boolean.")

        old_inst = _get_replace_caller()
        if old_inst is not None and isinstance(old_inst, FlexiblePreferenceConfig):
            sub_aliases = ("subsistence_shares", "mu_s", "subsistence_ratio")
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                changed_keys = [k for k in sub_aliases if k in exp_keys]
            else:
                old_vals = {
                    "subsistence_shares": old_inst.subsistence_shares,
                    "mu_s": old_inst.mu_s,
                    "subsistence_ratio": old_inst.subsistence_ratio,
                }
                vals = {
                    "subsistence_shares": self.subsistence_shares,
                    "mu_s": self.mu_s,
                    "subsistence_ratio": self.subsistence_ratio,
                }
                changed_keys = [k for k in sub_aliases if not _val_equal(vals[k], old_vals[k])]

            if changed_keys:
                vals_list = [getattr(self, k) for k in changed_keys]
                if any(not _val_equal(u, vals_list[0]) for u in vals_list[1:]):
                    raise ValueError("Conflicting values provided for subsistence parameters.")
                raw_mu = vals_list[0]
            else:
                raw_mu = self.subsistence_shares if self.subsistence_shares is not None else 0.0
        else:
            # Alias resolution between subsistence_shares, mu_s, and subsistence_ratio
            candidates = [
                ("subsistence_shares", self.subsistence_shares),
                ("mu_s", self.mu_s),
                ("subsistence_ratio", self.subsistence_ratio),
            ]
            non_none = [(name, val) for name, val in candidates if val is not None]
            if len(non_none) > 1:
                first_name, first_val = non_none[0]
                for other_name, other_val in non_none[1:]:
                    if not _val_equal(first_val, other_val):
                        raise ValueError(
                            f"Conflicting values provided for subsistence parameters: "
                            f"'{first_name}' and '{other_name}'."
                        )
                raw_mu = first_val
            elif len(non_none) == 1:
                raw_mu = non_none[0][1]
            else:
                raw_mu = 0.0

        # Validate and set sigma_trade
        if isinstance(self.sigma_trade, (int, float)):
            sig = float(self.sigma_trade)
            if sig <= 1.0:
                raise ValueError(
                    f"sigma_trade (Armington trade substitution elasticity) must be strictly greater than 1.0 (got {sig}). "
                    "In Armington international trade, varieties from different source countries must be gross substitutes "
                    "(sigma_trade > 1.0); an elasticity <= 1.0 violates gross substitutability, leading to perverse trade responses "
                    "or degenerate expenditure shares."
                )
            object.__setattr__(self, "sigma_trade", sig)
        elif isinstance(self.sigma_trade, np.ndarray):
            if np.any(self.sigma_trade <= 1.0):
                raise ValueError(
                    f"All elements of sigma_trade must be strictly greater than 1.0 (got min={np.min(self.sigma_trade)}). "
                    "In Armington international trade, varieties from different source countries must be gross substitutes."
                )
            object.__setattr__(self, "sigma_trade", self.sigma_trade)
        else:
            raise TypeError(f"sigma_trade must be numeric or ndarray, got {type(self.sigma_trade).__name__}.")

        # Validate and set subsistence shares mu
        if isinstance(raw_mu, (int, float)):
            mu_val = float(raw_mu)
            if mu_val < 0.0:
                raise ValueError(
                    f"mu_s / subsistence_shares must be non-negative (got {mu_val}). "
                    "Negative subsistence shares imply that the good is an economic bad whose subsistence requirement "
                    "increases disposable income, violating Stone-Geary preference axioms."
                )
            if mu_val >= 1.0:
                raise ValueError(
                    f"mu_s / subsistence_shares must be strictly less than 1.0 (got {mu_val}). "
                    "A subsistence share >= 1.0 absorbs 100% or more of baseline income into mandatory subsistence "
                    "consumption, leaving zero or negative supernumerary income and making consumer choice degenerate."
                )
            object.__setattr__(self, "subsistence_shares", mu_val)
            object.__setattr__(self, "mu_s", mu_val)
            object.__setattr__(self, "subsistence_ratio", mu_val)
        elif isinstance(raw_mu, (dict, Mapping)):
            resolved_dict: dict[str, float] = {}
            for k, v in raw_mu.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"Subsistence share for sector '{k}' must be numeric, got {type(v).__name__}.")
                v_f = float(v)
                if v_f < 0.0:
                    raise ValueError(
                        f"mu_s / subsistence_shares for sector '{k}' must be non-negative (got {v_f}). "
                        "Negative subsistence shares imply an economic bad, violating Stone-Geary axioms."
                    )
                if v_f >= 1.0:
                    raise ValueError(
                        f"mu_s / subsistence_shares for sector '{k}' must be strictly less than 1.0 (got {v_f}). "
                        "A subsistence share >= 1.0 absorbs 100% or more of baseline income."
                    )
                resolved_dict[str(k)] = v_f
            object.__setattr__(self, "subsistence_shares", resolved_dict)
            object.__setattr__(self, "mu_s", resolved_dict)
            object.__setattr__(self, "subsistence_ratio", resolved_dict)
        elif isinstance(raw_mu, np.ndarray):
            if np.any(raw_mu < 0.0):
                raise ValueError(
                    f"All subsistence shares must be non-negative (got min={np.min(raw_mu)})."
                )
            if np.any(raw_mu >= 1.0):
                raise ValueError(
                    f"All subsistence shares must be strictly less than 1.0 (got max={np.max(raw_mu)})."
                )
            object.__setattr__(self, "subsistence_shares", raw_mu)
            object.__setattr__(self, "mu_s", raw_mu)
            object.__setattr__(self, "subsistence_ratio", raw_mu)
        else:
            raise TypeError(f"subsistence_shares / mu_s must be numeric, dict, or ndarray, got {type(raw_mu).__name__}.")


@dataclass(frozen=True)
class FlexibleMarketStructureConfig:
    """Configuration for flexible market structure (Atkeson-Burstein markups).

    Parameters
    ----------
    variable_markups : bool, default False
        Whether endogenous Cournot-Armington oligopolistic markups are active.
        When False, competitive pricing with constant markups (1.0) is enforced.
    sigma_j : float | dict | np.ndarray, default 6.0
        Within-industry variety substitution elasticity across origin countries.
        Must be strictly greater than 1.0 (finite markups requirement).
    theta_j : float | dict | np.ndarray, default 2.0
        Across-industry sectoral substitution elasticity. Must be strictly
        greater than 1.0 and bounded by `sigma_j` (theta_j <= sigma_j).
    variety_condensation : bool, default False
        Whether Dixit-Stiglitz varieties are condensed out via zero-profit
        equilibrium conditions. Aliased with `condense_varieties` and
        `variety_expansion`.
    condense_varieties : bool, default False
        Alias for `variety_condensation`.
    variety_expansion : bool, default False
        Alias for `variety_condensation`.
    markup_min : float, default 1.0
        Minimum allowable markup bound. Must be >= 1.0.
    markup_max : float, default 5.0
        Maximum allowable markup bound. Must be strictly greater than markup_min.
    markup_bounds : tuple[float, float] | None, default None
        Tuple (markup_min, markup_max) synchronizing markup clipping bounds.
    cournot_weights : Any, default None
        Origin-specific Cournot conduct parameters or market share weights.
    clamping_threshold : float, default 1e-4
        Numerical floor for markup denominator to prevent division by zero.
    fl : float, default 0.0
        Labor fixed overhead requirement per firm. Must be non-negative.
    fk : float, default 0.0
        Capital fixed overhead requirement per firm. Must be non-negative.
    """

    variable_markups: bool = False
    sigma_j: float = 6.0
    theta_j: float = 2.0
    variety_condensation: bool = False
    condense_varieties: bool = False
    variety_expansion: bool = False
    markup_min: float = 1.0
    markup_max: float = 5.0
    markup_bounds: tuple[float, float] | None = None
    cournot_weights: Any = None
    clamping_threshold: float = 1e-4
    fl: float = 0.0
    fk: float = 0.0

    def __post_init__(self) -> None:
        old_inst = _get_replace_caller()
        if old_inst is not None and isinstance(old_inst, FlexibleMarketStructureConfig):
            cond_aliases = ["variety_condensation", "condense_varieties", "variety_expansion"]
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                exp_cond = [k for k in cond_aliases if k in exp_keys]
            else:
                exp_cond = [k for k in cond_aliases if getattr(self, k) != getattr(old_inst, k)]

            if exp_cond:
                vals = [getattr(self, k) for k in exp_cond]
                if any(c != vals[0] for c in vals[1:]):
                    raise ValueError("Conflicting values provided for variety condensation flags.")
                cond = bool(vals[0])
            else:
                cond = bool(self.condense_varieties or self.variety_condensation or self.variety_expansion)
        else:
            cond = bool(self.condense_varieties or self.variety_condensation or self.variety_expansion)
        object.__setattr__(self, "condense_varieties", cond)
        object.__setattr__(self, "variety_condensation", cond)
        object.__setattr__(self, "variety_expansion", cond)

        # Synchronize markup_bounds with (markup_min, markup_max)
        if old_inst is not None and isinstance(old_inst, FlexibleMarketStructureConfig):
            exp_keys = _get_replace_explicit_keys()
            if exp_keys is not None:
                b_changed = "markup_bounds" in exp_keys
                min_changed = "markup_min" in exp_keys
                max_changed = "markup_max" in exp_keys
            else:
                b_changed = (self.markup_bounds != old_inst.markup_bounds)
                min_changed = (self.markup_min != old_inst.markup_min)
                max_changed = (self.markup_max != old_inst.markup_max)
            if (min_changed or max_changed) and not b_changed:
                if not isinstance(self.markup_min, (int, float)) or isinstance(self.markup_min, bool):
                    raise TypeError("markup_min must be numeric.")
                if not isinstance(self.markup_max, (int, float)) or isinstance(self.markup_max, bool):
                    raise TypeError("markup_max must be numeric.")
                b_min = float(self.markup_min)
                b_max = float(self.markup_max)
                if b_min < 1.0:
                    raise ValueError(f"markup_min must be >= 1.0, got {b_min}.")
                if b_min >= b_max:
                    raise ValueError(f"markup_min ({b_min}) must be strictly less than markup_max ({b_max}).")
                object.__setattr__(self, "markup_min", b_min)
                object.__setattr__(self, "markup_max", b_max)
                object.__setattr__(self, "markup_bounds", (b_min, b_max))
            elif b_changed and not (min_changed or max_changed):
                if (
                    not isinstance(self.markup_bounds, (tuple, list))
                    or len(self.markup_bounds) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self.markup_bounds)
                ):
                    raise TypeError("markup_bounds must be a tuple of two numbers (markup_min, markup_max).")
                b_min, b_max = float(self.markup_bounds[0]), float(self.markup_bounds[1])
                if b_min < 1.0:
                    raise ValueError(f"markup_bounds lower bound must be >= 1.0, got {b_min}.")
                if b_min >= b_max:
                    raise ValueError(f"markup_bounds lower bound ({b_min}) must be strictly less than upper bound ({b_max}).")
                object.__setattr__(self, "markup_min", b_min)
                object.__setattr__(self, "markup_max", b_max)
                object.__setattr__(self, "markup_bounds", (b_min, b_max))
            elif b_changed and (min_changed or max_changed):
                if (
                    not isinstance(self.markup_bounds, (tuple, list))
                    or len(self.markup_bounds) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self.markup_bounds)
                ):
                    raise TypeError("markup_bounds must be a tuple of two numbers (markup_min, markup_max).")
                if not isinstance(self.markup_min, (int, float)) or isinstance(self.markup_min, bool):
                    raise TypeError("markup_min must be numeric.")
                if not isinstance(self.markup_max, (int, float)) or isinstance(self.markup_max, bool):
                    raise TypeError("markup_max must be numeric.")
                b_min, b_max = float(self.markup_bounds[0]), float(self.markup_bounds[1])
                if min_changed and float(self.markup_min) != b_min:
                    raise ValueError(
                        f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                        f"and markup_min ({self.markup_min})."
                    )
                if max_changed and float(self.markup_max) != b_max:
                    raise ValueError(
                        f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                        f"and markup_max ({self.markup_max})."
                    )
                if b_min < 1.0:
                    raise ValueError(f"markup_bounds lower bound must be >= 1.0, got {b_min}.")
                if b_min >= b_max:
                    raise ValueError(f"markup_bounds lower bound ({b_min}) must be strictly less than upper bound ({b_max}).")
                object.__setattr__(self, "markup_min", b_min)
                object.__setattr__(self, "markup_max", b_max)
                object.__setattr__(self, "markup_bounds", (b_min, b_max))
            else:
                b_min = float(self.markup_min)
                b_max = float(self.markup_max)
                object.__setattr__(self, "markup_bounds", (b_min, b_max))
        else:
            if self.markup_bounds is not None:
                if (
                    not isinstance(self.markup_bounds, (tuple, list))
                    or len(self.markup_bounds) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self.markup_bounds)
                ):
                    raise TypeError("markup_bounds must be a tuple of two numbers (markup_min, markup_max).")
                b_min, b_max = float(self.markup_bounds[0]), float(self.markup_bounds[1])
                if (self.markup_min != 1.0 and float(self.markup_min) != b_min) or (self.markup_max != 5.0 and float(self.markup_max) != b_max):
                    raise ValueError(
                        f"Conflicting values provided for markup_bounds {self.markup_bounds} "
                        f"and (markup_min={self.markup_min}, markup_max={self.markup_max})."
                    )
                if b_min < 1.0:
                    raise ValueError(f"markup_bounds lower bound must be >= 1.0, got {b_min}.")
                if b_min >= b_max:
                    raise ValueError(f"markup_bounds lower bound ({b_min}) must be strictly less than upper bound ({b_max}).")
                object.__setattr__(self, "markup_min", b_min)
                object.__setattr__(self, "markup_max", b_max)
                object.__setattr__(self, "markup_bounds", (b_min, b_max))
            else:
                if not isinstance(self.markup_min, (int, float)) or isinstance(self.markup_min, bool):
                    raise TypeError("markup_min must be numeric.")
                if not isinstance(self.markup_max, (int, float)) or isinstance(self.markup_max, bool):
                    raise TypeError("markup_max must be numeric.")
                b_min = float(self.markup_min)
                b_max = float(self.markup_max)
                if b_min < 1.0:
                    raise ValueError(f"markup_min must be >= 1.0, got {b_min}.")
                if b_min >= b_max:
                    raise ValueError(f"markup_min ({b_min}) must be strictly less than markup_max ({b_max}).")
                object.__setattr__(self, "markup_bounds", (b_min, b_max))

        if not isinstance(self.clamping_threshold, (int, float)) or isinstance(self.clamping_threshold, bool):
            raise TypeError("clamping_threshold must be numeric.")
        if self.clamping_threshold <= 0.0:
            raise ValueError(f"clamping_threshold must be strictly positive, got {self.clamping_threshold}.")

        if not isinstance(self.fl, (int, float)) or isinstance(self.fl, bool):
            raise TypeError("fl must be numeric.")
        if not isinstance(self.fk, (int, float)) or isinstance(self.fk, bool):
            raise TypeError("fk must be numeric.")
        if self.fl < 0.0 or self.fk < 0.0:
            raise ValueError(
                f"Fixed cost parameters fl and fk must be non-negative (got fl={self.fl}, fk={self.fk})."
            )

        if isinstance(self.sigma_j, (int, float)) and not isinstance(self.sigma_j, bool):
            if self.sigma_j <= 1.0:
                raise ValueError(
                    f"sigma_j must be strictly greater than 1.0, got {self.sigma_j}. "
                    "Finite Cournot markups require cross-sector elasticity sigma_j > 1."
                )
        elif not (isinstance(self.sigma_j, (dict, np.ndarray)) or hasattr(self.sigma_j, "__len__")):
            raise TypeError(f"sigma_j must be numeric, dict, or ndarray, got {type(self.sigma_j).__name__}.")

        if isinstance(self.theta_j, (int, float)) and not isinstance(self.theta_j, bool):
            if self.theta_j <= 1.0:
                raise ValueError(
                    f"theta_j must be strictly greater than 1.0, got {self.theta_j}. "
                    "Monopoly markup limit theta_j / (theta_j - 1) requires sectoral substitution elasticity theta_j > 1."
                )
        elif not (isinstance(self.theta_j, (dict, np.ndarray)) or hasattr(self.theta_j, "__len__")):
            raise TypeError(f"theta_j must be numeric, dict, or ndarray, got {type(self.theta_j).__name__}.")

        if (
            isinstance(self.sigma_j, (int, float))
            and not isinstance(self.sigma_j, bool)
            and isinstance(self.theta_j, (int, float))
            and not isinstance(self.theta_j, bool)
        ):
            if self.theta_j > self.sigma_j:
                raise ValueError(
                    f"theta_j ({self.theta_j}) cannot exceed sigma_j ({self.sigma_j}). "
                    "Within-industry elasticity theta_j must be bounded by across-industry elasticity sigma_j."
                )


@dataclass(frozen=True)
class FlexibleTradeModelConfig:
    """Composite declarative configuration for flexible general equilibrium trade model.

    Assembles nested CES technology, Stone-Geary LES preferences, Atkeson-Burstein
    market structure, and fixed-point solver parameters into a unified model
    specification.

    Parameters
    ----------
    technology : FlexibleTechnologyConfig, default FlexibleTechnologyConfig()
        Nested CES production technology configuration (inner value-added and
        outer gross output nests).
    preference : FlexiblePreferenceConfig, default FlexiblePreferenceConfig()
        Non-homothetic Stone-Geary Linear Expenditure System (LES) and
        Armington trade sourcing configuration.
    market_structure : FlexibleMarketStructureConfig, default FlexibleMarketStructureConfig()
        Atkeson-Burstein imperfect competition and variable markups configuration.
    max_inner_iter : int, default 10
        Maximum number of inner fixed-point iterations for the quasi-condensed
        solver loop. Enforces a strict latency cap of 10 to guarantee deterministic
        execution and Pyodide browser runtime compliance.
    """

    technology: FlexibleTechnologyConfig = field(default_factory=FlexibleTechnologyConfig)
    preference: FlexiblePreferenceConfig = field(default_factory=FlexiblePreferenceConfig)
    market_structure: FlexibleMarketStructureConfig = field(default_factory=FlexibleMarketStructureConfig)
    max_inner_iter: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.max_inner_iter, int) or isinstance(self.max_inner_iter, bool):
            raise TypeError(f"max_inner_iter must be an integer, got {type(self.max_inner_iter).__name__}.")
        if self.max_inner_iter < 1:
            raise ValueError(f"max_inner_iter must be >= 1, got {self.max_inner_iter}.")
        if self.max_inner_iter > 10:
            raise ValueError(
                f"max_inner_iter ({self.max_inner_iter}) exceeds the latency cap of 10. "
                "The fixed-point quasi-condensed solver enforces max_inner_iter <= 10 "
                "to guarantee deterministic execution time and strict Pyodide runtime compliance."
            )


# =============================================================================
# Helper Utilities
# =============================================================================

def _compute_benchmark_cva0(calib: TradeCalibrationResult) -> tuple[np.ndarray, np.ndarray]:
    """Compute benchmark value-added unit cost c_va_0 and validity mask.

    c_va,0 = 1 / (beta * alpha^alpha * (1 - alpha)^(1 - alpha))
    Identically equal to theta_va,0 = VA_0 / Y_0 to float64 machine precision.
    For single-factor sectors (alpha = 0.0 pure labor or alpha = 1.0 pure capital),
    uses the standard analytical limit lim_{x -> 0} x^x = 1.0.
    """
    alpha = calib.alpha
    beta = calib.beta
    mask_valid = (beta > 0.0) & (alpha >= 0.0) & (alpha <= 1.0)
    alpha_safe = np.where(mask_valid, alpha, 0.5)
    beta_safe = np.where(mask_valid, beta, 1.0)

    term_a = np.where(alpha_safe > 0.0, alpha_safe ** alpha_safe, 1.0)
    term_1ma = np.where(alpha_safe < 1.0, (1.0 - alpha_safe) ** (1.0 - alpha_safe), 1.0)
    term_alpha = term_a * term_1ma
    c_va_0 = np.where(mask_valid, 1.0 / (beta_safe * term_alpha), 0.0)
    return c_va_0, mask_valid


def compute_capacity_penalty(
    ytot: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    capacity_target_country: str = "USA",
) -> np.ndarray:
    """Compute smooth capacity barrier penalty array of shape (1, ns, nc)."""
    ns, nc = calib.n_sectors, calib.n_countries
    pen_arr = np.zeros((1, ns, nc), dtype=float)
    if tech_cfg.capacity_margins is None:
        return pen_arr

    margins = tech_cfg.capacity_margins
    scale = tech_cfg.penalty_scale
    exp = tech_cfg.penalty_exponent

    c_cap_idx = (
        calib.country_codes.index(capacity_target_country)
        if capacity_target_country in calib.country_codes
        else (calib.country_codes.index("USA") if "USA" in calib.country_codes else 0)
    )

    if isinstance(margins, (dict, Mapping)):
        for k_sec, margin in margins.items():
            if isinstance(k_sec, str) and k_sec in calib.sector_codes:
                s_i = calib.sector_codes.index(k_sec)
                c_i = c_cap_idx
            elif isinstance(k_sec, tuple) and len(k_sec) == 2:
                c_i, s_i = k_sec
            else:
                continue
            y0_val = float(calib.ytot[0, s_i, c_i])
            y_bar = (1.0 + float(margin)) * y0_val
            if y_bar > 1e-12:
                y_curr = float(ytot[0, s_i, c_i])
                ratio = max(y_curr / y_bar, 0.0)
                pen_arr[0, s_i, c_i] = float(scale * (ratio ** exp))
    elif isinstance(margins, np.ndarray):
        y_bar = (1.0 + margins) * calib.ytot
        y_bar_safe = np.maximum(y_bar, 1e-12)
        ratio = np.maximum(ytot / y_bar_safe, 0.0)
        pen_arr = scale * (ratio ** exp)

    return pen_arr


# =============================================================================
# Inner Nest (Value-Added) Cost & Derivatives
# =============================================================================

def compute_inner_ces_cost(
    r: np.ndarray,
    w: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    r0: float = 1.0,
    w0: float = 1.0,
) -> np.ndarray:
    """Compute inner nest (Value-Added) unit cost c_va in Calibrated Share Form.

    Parameters
    ----------
    r : np.ndarray
        Capital rental rate multipliers, broadcastable to (1, ns, nc).
    w : np.ndarray
        Wage rate multipliers, broadcastable to (1, ns, nc).
    calib : TradeCalibrationResult
        Calibrated model container.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.
    r0 : float, default 1.0
        Benchmark capital rental rate.
    w0 : float, default 1.0
        Benchmark wage rate.

    Returns
    -------
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost.
    """
    if np.any(r < 0.0):
        raise ValueError("Capital rental rates r must be non-negative.")
    if np.any(w < 0.0):
        raise ValueError("Wage rates w must be non-negative.")

    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    alpha = calib.alpha
    alpha_safe = np.where(mask_valid, alpha, 0.5)
    beta_safe = np.where(mask_valid, calib.beta, 1.0)

    r_use = np.maximum(r, 1e-12)
    w_use = np.maximum(w, 1e-12)

    rho_va = tech_cfg.rho_va
    if abs(rho_va - 1.0) < 1e-6:
        # Exact Cobb-Douglas gating branch
        if tech_cfg.replicate_matlab_precedence:
            term_r = np.where(alpha_safe > 0.0, (r_use / np.maximum(alpha_safe, 1e-12)) ** alpha_safe, 1.0)
            term_w_denom = np.where(alpha_safe < 1.0, (1.0 - alpha_safe) ** (1.0 - alpha_safe), 1.0)
            term_w = np.where(alpha_safe == 1.0, 1.0, w_use / term_w_denom)
            c_va = np.where(mask_valid, (1.0 / beta_safe) * (term_r * term_w), 0.0)
        else:
            r_rel = np.clip(r_use / r0, 1e-12, 1e12)
            w_rel = np.clip(w_use / w0, 1e-12, 1e12)
            term_r_cd = np.where(alpha_safe > 0.0, r_rel ** alpha_safe, 1.0)
            term_w_cd = np.where(alpha_safe < 1.0, w_rel ** (1.0 - alpha_safe), 1.0)
            c_va = np.where(
                mask_valid,
                c_va_0 * term_r_cd * term_w_cd,
                0.0,
            )
    else:
        # Calibrated share form CES branch with reference-price factorization (Log-Sum-Exp)
        e = 1.0 - rho_va
        r_rel = np.clip(r_use / r0, 1e-30, 1e30)
        w_rel = np.clip(w_use / w0, 1e-30, 1e30)

        # Factor out reference price: max when e > 0, min when e < 0
        # (Equivalent to factoring out exp(z_max / e) in Log-Sum-Exp)
        # Guarantees that u_r^e <= 1.0 and u_w^e <= 1.0 unconditionally.
        w_ref = np.where(e > 0, np.maximum(r_rel, w_rel), np.minimum(r_rel, w_rel))
        w_ref_safe = np.maximum(w_ref, 1e-12)

        u_r = r_rel / w_ref_safe
        u_w = w_rel / w_ref_safe

        term_r_ces = u_r ** e
        term_w_ces = u_w ** e

        bracket = alpha_safe * term_r_ces + (1.0 - alpha_safe) * term_w_ces
        bracket_safe = np.maximum(bracket, 1e-300)
        c_va_ratio = w_ref * (bracket_safe ** (1.0 / e))
        c_va = np.where(mask_valid, c_va_0 * c_va_ratio, 0.0)

    return c_va


def compute_inner_ces_derivatives(
    r: np.ndarray,
    w: np.ndarray,
    c_va: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    r0: float = 1.0,
    w0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized derivatives of the inner nest cost: [ (1/c_va,0) * d(c_va)/dw ] and [ (1/c_va,0) * d(c_va)/dr ].

    Normalizing by dividing by c_va,0 cancels the benchmark cost scale factor and
    guarantees exact baseline factor demand replication (xl0 == l0, xk0 == k0)
    without theta_va,0^2 double-counting.

    Parameters
    ----------
    r : np.ndarray
        Capital rental rate multipliers, broadcastable to (1, ns, nc).
    w : np.ndarray
        Wage rate multipliers, broadcastable to (1, ns, nc).
    c_va : np.ndarray
        Value-added unit cost, shape (1, ns, nc).
    calib : TradeCalibrationResult
        Calibrated model container.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.
    r0 : float, default 1.0
        Benchmark capital rental rate.
    w0 : float, default 1.0
        Benchmark wage rate.

    Returns
    -------
    norm_dc_dw : np.ndarray, shape (1, ns, nc)
        Normalized gradient [ (1/c_va,0) * d(c_va)/dw ].
    norm_dc_dr : np.ndarray, shape (1, ns, nc)
        Normalized gradient [ (1/c_va,0) * d(c_va)/dr ].
    """
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    alpha = calib.alpha
    alpha_safe = np.where(mask_valid, alpha, 0.5)

    r_use = np.maximum(r, 1e-12)
    w_use = np.maximum(w, 1e-12)
    r_rel = np.clip(r_use / r0, 1e-30, 1e30)
    w_rel = np.clip(w_use / w0, 1e-30, 1e30)

    c_va_safe = np.maximum(c_va, 1e-12)
    c_va_0_safe = np.where(mask_valid, np.maximum(c_va_0, 1e-12), 1.0)
    c_va_ratio = np.clip(c_va_safe / c_va_0_safe, 1e-30, 1e30)

    rho_va = tech_cfg.rho_va
    if abs(rho_va - 1.0) < 1e-6:
        # Cobb-Douglas normalized gradients
        ratio_w = c_va_ratio / w_rel
        ratio_r = c_va_ratio / r_rel
        norm_dc_dw = np.where(mask_valid & (alpha_safe < 1.0), ((1.0 - alpha_safe) / w0) * ratio_w, 0.0)
        norm_dc_dr = np.where(mask_valid & (alpha_safe > 0.0), (alpha_safe / r0) * ratio_r, 0.0)
    else:
        # Calibrated Share Form CES normalized gradients
        # Use widened clip [-300.0, 300.0] to support extreme substitution ratios without premature truncation
        ratio_w = c_va_ratio / w_rel
        ratio_r = c_va_ratio / r_rel
        term_w_pow = np.exp(np.clip(rho_va * np.log(np.maximum(ratio_w, 1e-300)), -300.0, 300.0))
        term_r_pow = np.exp(np.clip(rho_va * np.log(np.maximum(ratio_r, 1e-300)), -300.0, 300.0))
        norm_dc_dw = np.where(mask_valid & (alpha_safe < 1.0), ((1.0 - alpha_safe) / w0) * term_w_pow, 0.0)
        norm_dc_dr = np.where(mask_valid & (alpha_safe > 0.0), (alpha_safe / r0) * term_r_pow, 0.0)

    return norm_dc_dw, norm_dc_dr


# =============================================================================
# Intermediate Composite Price & Outer Nest Cost
# =============================================================================

def compute_intermediate_composite_price(
    p: np.ndarray,
    tau_a: np.ndarray,
    calib: TradeCalibrationResult,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute intermediate composite purchaser price index P_M across sectors and countries.

    Parameters
    ----------
    p : np.ndarray
        Gross output seller price vector or tensor.
    tau_a : np.ndarray, shape (ns*nc, ns, nc)
        Intermediate bilateral gross tariff multipliers.
    calib : TradeCalibrationResult
        Calibrated model container.
    sigma : float, default 0.0
        Intermediate variety sourcing substitution elasticity across origins.

    Returns
    -------
    P_M : np.ndarray, shape (1, ns, nc)
        Intermediate purchaser price index (normalized P_M,0 = 1.0).
    inter_cost : np.ndarray, shape (1, ns, nc)
        Unit intermediate input cost (A_mat * P_M).
    p_tau_safe : np.ndarray, shape (ns*nc, ns, nc)
        Clamped purchaser price matrix p * tau_a.
    """
    A_mat, omega = _get_ces_weights(calib)
    p_vec = p.flatten(order="F") if p.ndim == 3 else np.asarray(p, dtype=float).ravel()
    p_tau = p_vec[:, np.newaxis, np.newaxis] * tau_a
    p_tau_safe = np.maximum(p_tau, 1e-12)

    if sigma < 1e-6:
        # Exact Leontief sourcing branch
        P_M_2d = np.sum(np.where(omega > 0, omega * p_tau, 0.0), axis=0)
    elif abs(sigma - 1.0) < 1e-6:
        # Exact Cobb-Douglas sourcing branch
        ln_p_tau = np.where(omega > 0, np.log(p_tau_safe), 0.0)
        P_M_2d = np.exp(np.sum(np.where(omega > 0, omega * ln_p_tau, 0.0), axis=0))
    else:
        # General CES sourcing branch with reference price factorization
        e = 1.0 - sigma
        has_active = np.any(omega > 0, axis=0, keepdims=True)
        if e < 0:
            p_cand = np.where(omega > 0, p_tau_safe, np.inf)
            p_min = np.min(p_cand, axis=0, keepdims=True)
            p_ref = np.where(has_active, p_min, 1.0)
        else:
            p_cand = np.where(omega > 0, p_tau_safe, -np.inf)
            p_max = np.max(p_cand, axis=0, keepdims=True)
            p_ref = np.where(has_active, p_max, 1.0)

        p_ref_safe = np.maximum(p_ref, 1e-12)
        u = np.where(omega > 0, p_tau_safe / p_ref_safe, 1.0)
        term = np.where(omega > 0, omega * (u ** e), 0.0)
        inner = np.sum(term, axis=0)
        inner_safe = np.maximum(inner, 1e-300)
        P_M_2d = p_ref.squeeze(0) * (inner_safe ** (1.0 / e))
        P_M_2d = np.where(has_active.squeeze(0), P_M_2d, 1.0)

    P_M = P_M_2d[np.newaxis, :, :]
    inter_cost = (A_mat * P_M_2d)[np.newaxis, :, :]
    return P_M, inter_cost, p_tau_safe


def compute_outer_ces_cost(
    c_va: np.ndarray,
    P_M: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    normalized: bool = False,
) -> np.ndarray:
    """Compute outer nest gross output unit net cost c_y in Calibrated Share Form.

    Parameters
    ----------
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost.
    P_M : np.ndarray, shape (1, ns, nc)
        Intermediate composite purchaser price index (or unit intermediate cost).
    calib : TradeCalibrationResult
        Calibrated model container.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.
    normalized : bool, default False
        If True, returns the normalized unit cost index (baseline value 1.0).
        If False, returns unit net cost in levels: c_y0 * c_y_norm (baseline value 1.0 - tax_0).

    Returns
    -------
    c_y : np.ndarray, shape (1, ns, nc)
        Gross output unit net cost.
    """
    A_mat, _ = _get_ces_weights(calib)
    theta_m0 = A_mat[np.newaxis, :, :]
    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    theta_va0 = np.where(mask_valid, c_va_0, 1.0 - calib.tax - theta_m0)

    c_y0 = theta_va0 + theta_m0  # identically 1.0 - calib.tax
    c_y0_safe = np.where(c_y0 > 0, c_y0, 1.0)

    s_va0 = np.where(c_y0 > 0, theta_va0 / c_y0_safe, 0.5)
    s_m0 = np.where(c_y0 > 0, theta_m0 / c_y0_safe, 0.5)

    c_va0_safe = np.where(theta_va0 > 0, theta_va0, 1.0)
    P_M0_safe = np.ones_like(theta_m0)

    rel_c_va = np.maximum(c_va / c_va0_safe, 1e-12)
    rel_P_M = np.maximum(P_M / P_M0_safe, 1e-12)

    sigma_y = tech_cfg.sigma_y
    if sigma_y < 1e-6:
        # Exact Leontief outer nest branch
        c_y_norm = s_va0 * rel_c_va + s_m0 * rel_P_M
    elif abs(sigma_y - 1.0) < 1e-6:
        # Exact Cobb-Douglas outer nest branch
        c_y_norm = (rel_c_va ** s_va0) * (rel_P_M ** s_m0)
    else:
        # General CES outer nest branch with reference-price factorization (Log-Sum-Exp analog)
        e = 1.0 - sigma_y
        P_ref = np.minimum(rel_c_va, rel_P_M) if e < 0 else np.maximum(rel_c_va, rel_P_M)
        P_ref_safe = np.maximum(P_ref, 1e-12)
        r_c_va = rel_c_va / P_ref_safe
        r_P_M = rel_P_M / P_ref_safe
        bracket = s_va0 * (r_c_va ** e) + s_m0 * (r_P_M ** e)
        bracket_safe = np.maximum(bracket, 1e-300)
        c_y_norm = P_ref * (bracket_safe ** (1.0 / e))

    if normalized:
        return c_y_norm
    return c_y0 * c_y_norm


def compute_nested_ces_costs(
    r: np.ndarray,
    w: np.ndarray,
    P_M: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (c_va, c_y) tuple across sectors and countries.

    Parameters
    ----------
    r : np.ndarray
        Capital rental rate multipliers.
    w : np.ndarray
        Wage rate multipliers.
    P_M : np.ndarray
        Intermediate composite price index.
    calib : TradeCalibrationResult
        Calibrated model container.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.

    Returns
    -------
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost.
    c_y : np.ndarray, shape (1, ns, nc)
        Gross output unit cost (normalized to 1.0 at baseline).
    """
    c_va = compute_inner_ces_cost(r, w, calib, tech_cfg)
    c_y = compute_outer_ces_cost(c_va, P_M, calib, tech_cfg, normalized=True)
    return c_va, c_y


# =============================================================================
# Normalized Factor & Intermediate Demands
# =============================================================================

def compute_nested_factor_demands(
    ytot: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    P_M: np.ndarray,
    c_va: np.ndarray,
    c_y: np.ndarray,
    p: np.ndarray,
    tau: np.ndarray,
    calib: TradeCalibrationResult,
    tech_cfg: FlexibleTechnologyConfig,
    normalized: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized factor demands (xl, xk) and intermediate demands (x_mat).

    Formulas:
        xl = theta_va,0 * Y * ( (c_y / c_y,0) / (c_va / c_va,0) )^sigma_y * [ (1/c_va,0) * d(c_va)/dw ]
        xk = theta_va,0 * Y * ( (c_y / c_y,0) / (c_va / c_va,0) )^sigma_y * [ (1/c_va,0) * d(c_va)/dr ]
        x_mat = a * Y * ( (c_y / c_y,0) / (P_M / P_M,0) )^sigma_y * ( P_M / (p * tau) )^sigma

    Guarantees machine-precision baseline factor demand replication (xl0 == l0, xk0 == k0)
    without theta_va,0^2 double-counting.

    Parameters
    ----------
    ytot : np.ndarray, shape (1, ns, nc)
        Gross output quantities.
    r : np.ndarray
        Capital rental rate multipliers.
    w : np.ndarray
        Wage rate multipliers.
    P_M : np.ndarray, shape (1, ns, nc)
        Intermediate input composite price index.
    c_va : np.ndarray, shape (1, ns, nc)
        Value-added unit cost.
    c_y : np.ndarray, shape (1, ns, nc)
        Gross output unit cost.
    p : np.ndarray
        Gross output seller prices.
    tau : np.ndarray, shape (ns*nc, ns, nc)
        Bilateral tariff multiplier tensor.
    calib : TradeCalibrationResult
        Calibrated model container.
    tech_cfg : FlexibleTechnologyConfig
        Technology configuration.
    normalized : bool | None, default None
        Whether c_y is normalized index (baseline 1.0) or level (baseline 1 - tax).
        If None, scale-invariant dispersion check is used.

    Returns
    -------
    xl : np.ndarray, shape (1, ns, nc)
        Labor demand tensor.
    xk : np.ndarray, shape (1, ns, nc)
        Capital demand tensor.
    x_mat : np.ndarray, shape (ns*nc, ns, nc)
        Intermediate deliveries tensor.
    """
    if np.any(ytot < 0.0):
        raise ValueError("Gross output quantities ytot must be non-negative.")
    if np.any(r < 0.0):
        raise ValueError("Capital rental rates r must be non-negative.")
    if np.any(w < 0.0):
        raise ValueError("Wage rates w must be non-negative.")
    if np.any(P_M < 0.0):
        raise ValueError("Intermediate composite prices P_M must be non-negative.")
    if np.any(p < 0.0):
        raise ValueError("Prices p must be non-negative.")

    c_va_0, mask_valid = _compute_benchmark_cva0(calib)
    mask_active = mask_valid & (ytot > 0.0)

    norm_dc_dw, norm_dc_dr = compute_inner_ces_derivatives(r, w, c_va, calib, tech_cfg)

    # Reference benchmark c_y0 (scale-invariant CSF convention, Defect 1 fix)
    if normalized is None:
        tax_disp = float(np.std(calib.tax))
        if tax_disp > 1e-5:
            tax_denom = np.maximum(1.0 - calib.tax, 1e-12)
            std_scaled = float(np.std(c_y / tax_denom))
            std_raw = float(np.std(c_y))
            is_norm = not (std_scaled < 0.5 * std_raw)
        else:
            is_norm = True
    else:
        is_norm = normalized

    cy0_ref = 1.0 if is_norm else (1.0 - calib.tax)

    # Outer nest substitution factor
    sigma_y = tech_cfg.sigma_y
    if sigma_y < 1e-6:
        term_va_outer = 1.0
        term_mat_outer = 1.0
    else:
        cva0_safe = np.where(c_va_0 > 0, c_va_0, 1.0)
        cva_rel = np.clip(np.maximum(c_va, 1e-12) / cva0_safe, 1e-30, 1e30)
        cy_rel = np.clip(np.maximum(c_y, 1e-12) / cy0_ref, 1e-30, 1e30)
        pm_rel = np.clip(np.maximum(P_M, 1e-12) / 1.0, 1e-30, 1e30)

        term_va_outer = np.exp(np.clip(sigma_y * np.log(cy_rel / cva_rel), -300.0, 300.0))
        term_mat_outer = np.exp(np.clip(sigma_y * np.log(cy_rel / pm_rel), -300.0, 300.0))

    # Capacity barrier penalty factor
    pen_factor: float | np.ndarray = 1.0
    if tech_cfg.capacity_margins is not None:
        pen_arr = compute_capacity_penalty(ytot, calib, tech_cfg)
        pen_factor = 1.0 + pen_arr

    # Normalized factor demands (theta_va,0 * Y * outer_ratio^sigma_y * norm_grad * pen)
    xl = np.where(
        mask_active,
        c_va_0 * ytot * term_va_outer * norm_dc_dw * pen_factor,
        0.0,
    )
    xk = np.where(
        mask_active,
        c_va_0 * ytot * term_va_outer * norm_dc_dr * pen_factor,
        0.0,
    )

    # Intermediate input deliveries x_mat
    p_vec = p.flatten(order="F") if p.ndim == 3 else np.asarray(p, dtype=float).ravel()
    p_tau = p_vec[:, np.newaxis, np.newaxis] * tau
    p_tau_safe = np.maximum(p_tau, 1e-12)

    sigma_inter = tech_cfg.sigma_inter
    term_sourcing: float | np.ndarray = 1.0
    if sigma_inter > 1e-6:
        sourcing_ratio = np.clip(np.maximum(P_M, 1e-12) / p_tau_safe, 1e-12, 1e12)
        term_sourcing = np.exp(np.clip(sigma_inter * np.log(sourcing_ratio), -80.0, 80.0))

    x_mat = np.where(
        ytot > 0,
        calib.a * ytot * term_mat_outer * term_sourcing,
        0.0,
    )

    return xl, xk, x_mat


# =============================================================================
# Zero-Profit Unit Price
# =============================================================================

def compute_zero_profit_price(
    c_y: np.ndarray,
    calib: TradeCalibrationResult,
    sub_prod: np.ndarray | None = None,
    ytot: np.ndarray | None = None,
    normalized: bool | None = None,
) -> np.ndarray:
    """Compute producer zero-profit prices pp across sectors and countries.

    pp = c_y / (1 - tax) - sub_prod

    Parameters
    ----------
    c_y : np.ndarray, shape (1, ns, nc)
        Gross output unit net cost (in levels or normalized index).
    calib : TradeCalibrationResult
        Calibrated model container.
    sub_prod : np.ndarray | None, default None
        Production subsidies per unit of gross output.
    ytot : np.ndarray | None, default None
        Gross output levels for active sector masking.
    normalized : bool | None, default None
        Whether c_y is normalized index (baseline 1.0) or level (baseline 1 - tax).
        If None, scale-invariant dispersion check is used.

    Returns
    -------
    pp : np.ndarray, shape (1, ns, nc)
        Producer zero-profit prices.
    """
    if sub_prod is None:
        sub_prod = np.zeros_like(c_y)
    tax = calib.tax
    tax_denom = np.maximum(1.0 - tax, 1e-12)

    # Scale-invariant normalization handling (Defect 4 fix)
    if normalized is None:
        tax_disp = float(np.std(tax))
        if tax_disp > 1e-5:
            std_scaled = float(np.std(c_y / tax_denom))
            std_raw = float(np.std(c_y))
            is_norm = not (std_scaled < 0.5 * std_raw)
        else:
            is_norm = True
    else:
        is_norm = normalized

    c_y_level = (c_y * tax_denom) if is_norm else c_y
    pp = (c_y_level / tax_denom) - sub_prod

    y_mask = ytot if ytot is not None else getattr(calib, "ytot", None)
    if y_mask is not None:
        pp = np.where(y_mask > 0, pp, 1.0)

    return pp


def compute_zero_profit_prices(
    c_y: np.ndarray,
    tax: np.ndarray,
    sub_prod: np.ndarray | None = None,
    ytot: np.ndarray | None = None,
) -> np.ndarray:
    """Alias for compute_zero_profit_price with tax array."""
    if sub_prod is None:
        sub_prod = np.zeros_like(c_y)
    tax_denom = np.maximum(1.0 - tax, 1e-12)
    pp = (c_y / tax_denom) - sub_prod
    if ytot is not None:
        pp = np.where(ytot > 0, pp, 1.0)
    return pp


# =============================================================================
# R2. Stone-Geary LES Final Demand & Tier 2 Armington Trade Sourcing
# =============================================================================

def smooth_subsistence_scaling(u: np.ndarray | float) -> np.ndarray | float:
    """Smooth subsistence scaling function g(u) = tanh(3u) / tanh(3).

    Satisfies:
    - g(1.0) == 1.0 to machine precision (< 10^-16).
    - g(0.0) == 0.0 (smooth vanishing under income collapse).
    - g(u) -> 1 / tanh(3) ~ 1.00497 as u -> inf (asymptotic subsistence cap).
    - g'(u) = 3 * sech^2(3u) / tanh(3) > 0 for all u (strict monotonicity).
    - g''(u) < 0 for all u > 0 (strict concavity).

    Parameters
    ----------
    u : np.ndarray | float
        Income ratio Y^con / Y_0^con. Can be scalar, 1D, 2D, or 3D array.

    Returns
    -------
    g_val : np.ndarray | float
        Scaled multiplier matching input shape.
    """
    denom = np.tanh(3.0)
    if isinstance(u, (int, float, np.floating, np.integer)):
        return float(np.tanh(3.0 * u) / denom)

    arr = np.asarray(u, dtype=float)
    return np.tanh(3.0 * arr) / denom


def _extract_benchmark_household_data(
    calib: TradeCalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract benchmark household income, expenditure, and sectoral shares.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model container.

    Returns
    -------
    Ycon_0 : np.ndarray, shape (1, 1, nc)
        Benchmark total consumer income.
    E_C_0 : np.ndarray, shape (1, 1, nc)
        Benchmark total household expenditure.
    theta_sec_0 : np.ndarray, shape (1, ns, nc)
        Benchmark expenditure share of each composite sector in household consumption.
    E_Cs0 : np.ndarray, shape (1, ns, nc)
        Benchmark expenditure on each composite sector.
    c_s0 : np.ndarray, shape (1, ns, nc)
        Benchmark real consumption quantity of each composite sector.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    l_3d = calib.l_endow.reshape((1, 1, nc))
    k_3d = calib.k_endow.reshape((1, 1, nc))
    if calib.T is not None:
        T_base = calib.T.reshape((1, 1, nc))
    elif calib.TT is not None and calib.TTfd is not None:
        T_base = (calib.TT + calib.TTfd).reshape((1, 1, nc))
    else:
        T_base = np.zeros((1, 1, nc))

    Ycon_0 = l_3d + k_3d + T_base  # (1, 1, nc)

    theta_hh = (
        calib.theta[:, 0:1, :]
        if calib.theta is not None
        else np.ones((1, 1, nc)) / calib.n_final_demand
    )
    E_C_0 = theta_hh * Ycon_0  # (1, 1, nc)

    # Sectoral shares across origin countries for category 0 (Household)
    if getattr(calib, "afd_4d", None) is not None:
        theta_sec_0 = np.sum(calib.afd_4d[:, :, 0, :], axis=1, keepdims=True).transpose(1, 0, 2)
    elif getattr(calib, "afd", None) is not None:
        afd_reshaped = calib.afd[:, 0, :].reshape(nc, ns, nc)
        theta_sec_0 = np.sum(afd_reshaped, axis=0)[np.newaxis, :, :]
    else:
        theta_sec_0 = np.full((1, ns, nc), 1.0 / ns)

    sum_shares = np.sum(theta_sec_0, axis=1, keepdims=True)
    theta_sec_0 = np.where(sum_shares > 0, theta_sec_0 / sum_shares, 1.0 / ns)

    E_Cs0 = theta_sec_0 * E_C_0  # (1, ns, nc)
    c_s0 = E_Cs0.copy()          # At benchmark P_C,0 = 1.0
    return Ycon_0, E_C_0, theta_sec_0, E_Cs0, c_s0


def _resolve_subsistence_shares(
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Resolve subsistence shares into an array of shape (1, ns, nc).

    Handles scalar float, dict (with unmentioned sectors defaulting to 0.0),
    and ndarray representations.
    """
    ns, nc = calib.n_sectors, calib.n_countries
    mu_arr = np.zeros((1, ns, nc), dtype=float)
    if pref_cfg is None:
        return mu_arr

    mu_input = pref_cfg.subsistence_shares if pref_cfg.subsistence_shares is not None else pref_cfg.mu_s
    if mu_input is None:
        return mu_arr

    if isinstance(mu_input, (int, float)) and not isinstance(mu_input, bool):
        mu_arr[:] = float(mu_input)
    elif isinstance(mu_input, (dict, Mapping)):
        for k, v in mu_input.items():
            val = float(v)
            s_idx = None
            if k in calib.sector_codes:
                s_idx = calib.sector_codes.index(k)
            else:
                k_str = str(k).strip()
                for idx, code in enumerate(calib.sector_codes):
                    if k_str.lower() == code.lower():
                        s_idx = idx
                        break
                if s_idx is None:
                    for idx in range(ns):
                        if (
                            k_str.lower() == f"s{idx}"
                            or k_str.lower() == f"s{idx:02d}"
                            or k_str.lower() == f"sector_{idx}"
                            or k_str == str(idx)
                        ):
                            s_idx = idx
                            break
            if s_idx is not None and 0 <= s_idx < ns:
                mu_arr[0, s_idx, :] = val
    elif isinstance(mu_input, np.ndarray):
        if mu_input.ndim == 1:
            mu_arr[0, :, :] = mu_input[:, np.newaxis]
        elif mu_input.ndim == 2:
            mu_arr[0, :, :] = mu_input
        elif mu_input.ndim == 3:
            mu_arr = np.broadcast_to(mu_input, (1, ns, nc)).copy()
    return mu_arr


def compute_les_marginal_budget_shares(
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Compute calibrated marginal budget shares theta_s^LES across sectors and countries.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model container.
    pref_cfg : FlexiblePreferenceConfig | None, default None
        Preference configuration.

    Returns
    -------
    theta_LES : np.ndarray, shape (1, ns, nc)
        Marginal budget shares, strictly summing to 1.0 across axis 1.
    """
    _, E_C_0, theta_sec_0, E_Cs0, _ = _extract_benchmark_household_data(calib)
    if pref_cfg is None:
        return theta_sec_0

    mu_arr = _resolve_subsistence_shares(calib, pref_cfg)
    if np.all(mu_arr == 0.0):
        return theta_sec_0

    num = (1.0 - mu_arr) * E_Cs0
    sub_E0 = np.sum(mu_arr * E_Cs0, axis=1, keepdims=True)
    denom = E_C_0 - sub_E0
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    theta_LES = np.divide(num, denom_safe, out=theta_sec_0.copy(), where=(denom > 1e-12))
    sum_les = np.sum(theta_LES, axis=1, keepdims=True)
    theta_LES = np.where(sum_les > 0, theta_LES / sum_les, theta_sec_0)
    return theta_LES


def compute_stone_geary_final_demand(
    Y_con: np.ndarray,
    P_C: np.ndarray,
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> np.ndarray:
    """Compute household real consumption demand across sectors under Stone-Geary LES preferences.

    Formula:
        c_{C, s} = c_bar_s + (theta_s^LES / P_{C, s}) * (Y_C^con - sum_k P_{C, k} * c_bar_k)

    where:
        c_bar_s(Y) = mu_s * c_{s, 0} * g(Y_C^con / Y_{C, 0}^con)
        g(u) = tanh(3u) / tanh(3)

    Parameters
    ----------
    Y_con : np.ndarray
        Total consumer income tensor, broadcastable to (1, 1, nc).
    P_C : np.ndarray
        Composite purchaser price index tensor across sectors, shape (1, ns, nc) or (ns, nc).
    calib : TradeCalibrationResult
        Calibrated model container.
    pref_cfg : FlexiblePreferenceConfig | None, default None
        Preference configuration.

    Returns
    -------
    c_C : np.ndarray, shape (1, ns, nc)
        Real household consumption demand across composite sectors.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    Y_con_3d = Y_con.reshape((1, 1, nc)) if Y_con.ndim != 3 else Y_con
    if P_C.ndim == 2:
        P_C_3d = P_C[np.newaxis, :, :]
    elif P_C.ndim == 3 and P_C.shape == (ns, calib.n_final_demand, nc):
        P_C_3d = P_C[:, 0, :][np.newaxis, :, :]
    elif P_C.ndim == 3:
        P_C_3d = P_C
    else:
        P_C_3d = np.asarray(P_C, dtype=float).reshape((1, ns, nc))

    P_C_safe = np.maximum(P_C_3d, 1e-12)

    Ycon_0, E_C_0, theta_sec_0, E_Cs0, c_s0 = _extract_benchmark_household_data(calib)
    theta_hh = (
        calib.theta[:, 0:1, :]
        if calib.theta is not None
        else np.ones((1, 1, nc)) / calib.n_final_demand
    )
    Y_C_con = theta_hh * Y_con_3d

    if pref_cfg is None:
        return (theta_sec_0 * Y_C_con) / P_C_safe

    mu_arr = _resolve_subsistence_shares(calib, pref_cfg)
    if np.all(mu_arr == 0.0):
        return (theta_sec_0 * Y_C_con) / P_C_safe

    # Smooth subsistence scaling g(u)
    u = np.maximum(Y_con_3d / np.maximum(Ycon_0, 1e-12), 0.0)
    g_u = smooth_subsistence_scaling(u)
    c_bar = mu_arr * c_s0 * g_u

    # Calibrated marginal budget shares theta_LES
    num = (1.0 - mu_arr) * E_Cs0
    sub_E0 = np.sum(mu_arr * E_Cs0, axis=1, keepdims=True)
    denom = E_C_0 - sub_E0
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    theta_LES = np.divide(num, denom_safe, out=theta_sec_0.copy(), where=(denom > 1e-12))
    sum_les = np.sum(theta_LES, axis=1, keepdims=True)
    theta_LES = np.where(sum_les > 0, theta_LES / sum_les, theta_sec_0)

    # Supernumerary expenditure allocation
    sub_exp = np.sum(P_C_safe * c_bar, axis=1, keepdims=True)
    super_inc = np.maximum(Y_C_con - sub_exp, 0.0)

    # LES real demand
    c_C = c_bar + (theta_LES / P_C_safe) * super_inc
    return c_C


def compute_homothetic_final_demand(
    Y_con: np.ndarray,
    P_fd: np.ndarray,
    calib: TradeCalibrationResult,
    category: int = 1,
    invforT: np.ndarray | None = None,
) -> np.ndarray:
    """Compute linear homothetic Cobb-Douglas demand for GCF (cat=1) or Gov (cat=2).

    Parameters
    ----------
    Y_con : np.ndarray
        Total consumer income tensor, broadcastable to (1, 1, nc).
    P_fd : np.ndarray
        Composite purchaser price index across sectors, shape (1, ns, nc) or (ns, nc).
    calib : TradeCalibrationResult
        Calibrated model container.
    category : int, default 1
        Final demand category index (1 for Gross Capital Formation, 2 for Government).
    invforT : np.ndarray | None, default None
        Net foreign surplus for investment category adjustment.

    Returns
    -------
    c_fd : np.ndarray, shape (1, ns, nc)
        Real sectoral final demand.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    Y_con_3d = Y_con.reshape((1, 1, nc)) if Y_con.ndim != 3 else Y_con
    if P_fd.ndim == 2:
        P_fd_3d = P_fd[np.newaxis, :, :]
    elif P_fd.ndim == 3 and P_fd.shape == (ns, calib.n_final_demand, nc):
        P_fd_3d = P_fd[:, category, :][np.newaxis, :, :]
    elif P_fd.ndim == 3:
        P_fd_3d = P_fd
    else:
        P_fd_3d = np.asarray(P_fd, dtype=float).reshape((1, ns, nc))

    P_fd_safe = np.maximum(P_fd_3d, 1e-12)

    theta_cat = (
        calib.theta[:, category : category + 1, :]
        if calib.theta is not None
        else np.ones((1, 1, nc)) / calib.n_final_demand
    )
    exp_nom = theta_cat * Y_con_3d
    if category == 1:
        inv_val = invforT if invforT is not None else getattr(calib, "invforT", 0.0)
        inv_arr = np.asarray(inv_val, dtype=float).reshape((1, 1, nc))
        exp_nom = exp_nom - inv_arr

    if getattr(calib, "afd_4d", None) is not None:
        theta_sec = np.sum(calib.afd_4d[:, :, category, :], axis=1, keepdims=True).transpose(1, 0, 2)
    elif getattr(calib, "afd", None) is not None:
        afd_reshaped = calib.afd[:, category, :].reshape(nc, ns, nc)
        theta_sec = np.sum(afd_reshaped, axis=0)[np.newaxis, :, :]
    else:
        theta_sec = np.full((1, ns, nc), 1.0 / ns)

    sum_sec = np.sum(theta_sec, axis=1, keepdims=True)
    theta_sec = np.where(sum_sec > 0, theta_sec / sum_sec, 1.0 / ns)

    c_fd = (theta_sec * exp_nom) / P_fd_safe
    return c_fd


def _get_armington_weights(
    calib: TradeCalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute and cache Tier 2 Armington benchmark sourcing weights.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model parameters from calibrate_trade_model.

    Returns
    -------
    theta_sec : np.ndarray, shape (ns, 1, nfd, nc)
        Sectoral expenditure shares within each category and destination country.
    b_4d : np.ndarray, shape (ns, nc, nfd, nc)
        Calibrated benchmark origin expenditure shares (sum over axis 1 == 1.0).
    active_mask : np.ndarray, shape (ns, nfd, nc)
        Boolean mask of active consumption flows.
    """
    cached = getattr(calib, "_armington_weights_cache", None)
    if cached is not None:
        return cached

    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    if getattr(calib, "afd_4d", None) is not None:
        afd_4d = calib.afd_4d
    else:
        afd_4d = calib.afd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)

    theta_sec = np.sum(afd_4d, axis=1, keepdims=True)  # (ns, 1, nfd, nc)
    theta_sec_safe = np.where(theta_sec > 0.0, theta_sec, 1.0)
    b_4d = np.divide(afd_4d, theta_sec_safe, out=np.zeros_like(afd_4d), where=(theta_sec > 0.0))

    active_mask = (theta_sec[:, 0, :, :] > 0.0)  # (ns, nfd, nc)
    cached_val = (theta_sec, b_4d, active_mask)

    try:
        object.__setattr__(calib, "_armington_weights_cache", cached_val)
    except Exception:
        pass

    return cached_val


def compute_armington_purchaser_prices(
    p: np.ndarray,
    tau_fd: np.ndarray,
    calib: TradeCalibrationResult,
    sigma_trade: float | np.ndarray = 5.0,
) -> np.ndarray:
    """Compute Tier 2 Armington composite purchaser price index P_C across sectors and categories.

    Formula:
        P_{s, fd, n} = [ sum_i b_{s, i, fd, n} * (p_i^s * tau_{i, fd, n}^s)^(1 - sigma_trade) ]^(1 / (1 - sigma_trade))

    Parameters
    ----------
    p : np.ndarray, shape (1, ns, nc) or (ns*nc,)
        Gross output seller prices.
    tau_fd : np.ndarray, shape (ns*nc, nfd, nc) or (ns, nc, nfd, nc)
        Bilateral final demand tariff multipliers.
    calib : TradeCalibrationResult
        Calibrated model parameters.
    sigma_trade : float or np.ndarray, default 5.0
        Trade elasticity of substitution across origin countries.

    Returns
    -------
    P_C : np.ndarray, shape (ns, nfd, nc)
        Composite purchaser price index for each sector, category, and destination.
    """
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    theta_sec, b_4d, active_mask = _get_armington_weights(calib)

    # 4D tariff tensor reshaping
    if tau_fd.ndim == 3 and tau_fd.shape == (ns * nc, nfd, nc):
        taufd_4d = tau_fd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
    elif tau_fd.ndim == 4:
        taufd_4d = np.asarray(tau_fd, dtype=float)
    else:
        raise ValueError(f"Unexpected tau_fd shape: {tau_fd.shape}")

    # Seller price 4D broadcast: (ns, nc, 1, 1)
    if p.ndim == 3:
        p_4d = p.squeeze(0)[:, :, np.newaxis, np.newaxis]
    elif p.ndim == 1:
        p_4d = p.reshape(nc, ns).T[:, :, np.newaxis, np.newaxis]
    else:
        p_4d = np.asarray(p, dtype=float).reshape(ns, nc, 1, 1)

    p_tau = p_4d * taufd_4d
    p_tau_safe = np.maximum(p_tau, 1e-12)

    # Analytical branches
    if isinstance(sigma_trade, (int, float)):
        sig = float(sigma_trade)
        if sig < 1e-6:
            P = np.sum(np.where(b_4d > 0, b_4d * p_tau_safe, 0.0), axis=1)
        elif abs(sig - 1.0) < 1e-6:
            ln_pt = np.where(b_4d > 0, np.log(p_tau_safe), 0.0)
            P = np.exp(np.sum(np.where(b_4d > 0, b_4d * ln_pt, 0.0), axis=1))
        else:
            e = 1.0 - sig
            has_active = np.any(b_4d > 0, axis=1, keepdims=True)
            if e < 0:
                p_cand = np.where(b_4d > 0, p_tau_safe, np.inf)
                p_min = np.min(p_cand, axis=1, keepdims=True)
                p_ref = np.where(has_active, p_min, 1.0)
            else:
                p_cand = np.where(b_4d > 0, p_tau_safe, -np.inf)
                p_max = np.max(p_cand, axis=1, keepdims=True)
                p_ref = np.where(has_active, p_max, 1.0)

            p_ref_safe = np.maximum(p_ref, 1e-12)
            u_pt = np.where(b_4d > 0, p_tau_safe / p_ref_safe, 1.0)
            term = np.where(b_4d > 0, b_4d * (u_pt ** e), 0.0)
            inner = np.sum(term, axis=1)
            inner_safe = np.maximum(inner, 1e-300)
            P = p_ref.squeeze(1) * (inner_safe ** (1.0 / e))
    else:
        sig_arr = np.asarray(sigma_trade, dtype=float).reshape(ns, 1, 1)
        e = 1.0 - sig_arr
        is_cd = np.abs(sig_arr - 1.0) < 1e-6
        is_leo = sig_arr < 1e-6

        has_active = np.any(b_4d > 0, axis=1, keepdims=True)
        p_min = np.where(has_active, np.min(np.where(b_4d > 0, p_tau_safe, np.inf), axis=1, keepdims=True), 1.0)
        p_max = np.where(has_active, np.max(np.where(b_4d > 0, p_tau_safe, -np.inf), axis=1, keepdims=True), 1.0)
        e_4d = e[:, :, np.newaxis]
        p_ref = np.where(e_4d < 0, p_min, p_max)
        p_ref_safe = np.maximum(p_ref, 1e-12)
        u_pt = np.where(b_4d > 0, p_tau_safe / p_ref_safe, 1.0)
        term = np.where(b_4d > 0, b_4d * (u_pt ** e_4d), 0.0)
        inner = np.sum(term, axis=1)
        inner_safe = np.maximum(inner, 1e-300)
        P_ces = p_ref.squeeze(1) * (inner_safe ** (1.0 / e))

        ln_pt = np.where(b_4d > 0, np.log(p_tau_safe), 0.0)
        P_cd = np.exp(np.sum(np.where(b_4d > 0, b_4d * ln_pt, 0.0), axis=1))
        P_leo = np.sum(np.where(b_4d > 0, b_4d * p_tau_safe, 0.0), axis=1)
        P = np.where(is_leo, P_leo, np.where(is_cd, P_cd, P_ces))

    return np.where(active_mask, P, 1.0)


def compute_armington_final_demands(
    c_sec: np.ndarray,
    P_C: np.ndarray,
    p: np.ndarray,
    tau_fd: np.ndarray,
    calib: TradeCalibrationResult,
    sigma_trade: float | np.ndarray = 5.0,
    as_3d: bool = True,
) -> np.ndarray:
    """Compute bilateral deliveries x_{s, i, fd, n} across all origin countries.

    Formula:
        x_{s, i, fd, n} = b_{s, i, fd, n} * c_{s, fd, n} * ( P_{s, fd, n} / (p_i^s * tau_{i, fd, n}^s) )^sigma_trade

    Parameters
    ----------
    c_sec : np.ndarray, shape (ns, nfd, nc) or (1, ns, nc)
        Composite real absorption demands by sector, category, and destination.
    P_C : np.ndarray, shape (ns, nfd, nc) or (1, ns, nc)
        Composite purchaser price index.
    p : np.ndarray, shape (1, ns, nc) or (ns*nc,)
        Gross output seller prices.
    tau_fd : np.ndarray, shape (ns*nc, nfd, nc) or (ns, nc, nfd, nc)
        Bilateral final demand tariff multipliers.
    calib : TradeCalibrationResult
        Calibrated model parameters.
    sigma_trade : float or np.ndarray, default 5.0
        Trade elasticity of substitution.
    as_3d : bool, default True
        Whether to return flattened 3D array of shape (ns*nc, nfd, nc) or 4D (ns, nc, nfd, nc).

    Returns
    -------
    xc : np.ndarray, shape (ns*nc, nfd, nc) or (ns, nc, nfd, nc)
        Bilateral final deliveries tensor.
    """
    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    theta_sec, b_4d, active_mask = _get_armington_weights(calib)

    # 4D tariff tensor reshaping
    if tau_fd.ndim == 3 and tau_fd.shape == (ns * nc, nfd, nc):
        taufd_4d = tau_fd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
    elif tau_fd.ndim == 4:
        taufd_4d = np.asarray(tau_fd, dtype=float)
    else:
        raise ValueError(f"Unexpected tau_fd shape: {tau_fd.shape}")

    # Seller price 4D broadcast: (ns, nc, 1, 1)
    if p.ndim == 3:
        p_4d = p.squeeze(0)[:, :, np.newaxis, np.newaxis]
    elif p.ndim == 1:
        p_4d = p.reshape(nc, ns).T[:, :, np.newaxis, np.newaxis]
    else:
        p_4d = np.asarray(p, dtype=float).reshape(ns, nc, 1, 1)

    p_tau = p_4d * taufd_4d
    p_tau_safe = np.maximum(p_tau, 1e-12)

    # Shape alignment for c_sec and P_C
    if c_sec.ndim == 3 and c_sec.shape == (1, ns, nc):
        c_sec_use = np.repeat(c_sec.transpose(1, 0, 2), nfd, axis=1)
    else:
        c_sec_use = c_sec

    if P_C.ndim == 3 and P_C.shape == (1, ns, nc):
        P_C_use = np.repeat(P_C.transpose(1, 0, 2), nfd, axis=1)
    else:
        P_C_use = P_C

    rel_p = np.clip(P_C_use[:, np.newaxis, :, :] / p_tau_safe, 1e-30, 1e30)
    if isinstance(sigma_trade, (int, float)):
        term_sourcing = np.exp(np.clip(float(sigma_trade) * np.log(rel_p), -80.0, 80.0))
    else:
        sig_arr = np.asarray(sigma_trade, dtype=float).reshape(ns, 1, 1, 1)
        term_sourcing = np.exp(np.clip(sig_arr * np.log(rel_p), -80.0, 80.0))

    xfd_4d = b_4d * c_sec_use[:, np.newaxis, :, :] * term_sourcing

    if as_3d:
        return xfd_4d.transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    return xfd_4d


def compute_multi_category_final_demands(
    Y_con: np.ndarray,
    p: np.ndarray,
    tau_fd: np.ndarray,
    invforT: np.ndarray,
    calib: TradeCalibrationResult,
    pref_cfg: FlexiblePreferenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate expenditure and source bilateral deliveries across all 3 final demand categories.

    Integrates:
    - nfd=0 (Household): Stone-Geary LES across sectors + Tier 2 Armington sourcing.
    - nfd=1 (GCF): Foreign investment transfer adjustment + homothetic CD + Armington sourcing.
    - nfd=2 (Government): Public budget allocation + homothetic CD + Armington sourcing.

    Parameters
    ----------
    Y_con : np.ndarray, shape (1, 1, nc)
        Total national disposable income vector.
    p : np.ndarray, shape (1, ns, nc)
        Gross output seller prices.
    tau_fd : np.ndarray, shape (ns*nc, nfd, nc)
        Bilateral final demand gross tariff multipliers.
    invforT : np.ndarray, shape (1, nc) or (1, 1, nc)
        Net foreign transfers / trade surplus baseline.
    calib : TradeCalibrationResult
        Calibrated model container.
    pref_cfg : FlexiblePreferenceConfig | None, default None
        Flexible preference configuration (mu_s and sigma_trade).

    Returns
    -------
    xc : np.ndarray, shape (ns*nc, nfd, nc)
        Bilateral final demand delivery quantities.
    P_C : np.ndarray, shape (ns, nfd, nc)
        Tier 2 Armington composite purchaser price index.
    P_agg : np.ndarray, shape (1, nfd, nc)
        Aggregate category-level price indices.
    Tax_c : np.ndarray, shape (1, nfd, nc)
        Final demand consumption tax revenues.
    """
    if pref_cfg is None:
        pref_cfg = FlexiblePreferenceConfig()

    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
    theta_sec, b_4d, active_mask = _get_armington_weights(calib)

    # 1. Tier 2 Armington Purchaser Prices P_C (ns, nfd, nc)
    sigma_trade = pref_cfg.sigma_trade
    P_C = compute_armington_purchaser_prices(p, tau_fd, calib, sigma_trade=sigma_trade)

    # 2. Aggregate Category Prices P_agg (1, nfd, nc)
    P_agg = np.sum(theta_sec[:, 0, :, :] * P_C, axis=0, keepdims=True)


    # 3. Step-by-Step Category Allocation
    theta_arr = calib.theta if calib.theta is not None else np.ones((1, nfd, nc)) / nfd
    cd = theta_arr * Y_con / P_agg  # (1, nfd, nc)
    tax_fd_arr = calib.tax_fd if calib.tax_fd is not None else np.zeros((1, nfd, nc))
    Tax_c = tax_fd_arr * P_agg * cd

    c = cd.copy()
    invforT_3d = np.asarray(invforT, dtype=float).reshape((1, 1, nc))
    c[:, 1:2, :] -= invforT_3d / P_agg[:, 1:2, :]
    c_net = c - tax_fd_arr * cd  # (1, nfd, nc)

    # 4. Composite Sectoral Demands c_all_sec (ns, nfd, nc)
    c_all_sec = np.zeros((ns, nfd, nc), dtype=float)

    # Check for subsistence
    mu_arr_check = _resolve_subsistence_shares(calib, pref_cfg)
    has_subsistence = np.any(mu_arr_check > 0.0)

    if not has_subsistence:
        # Default zero subsistence: exact Cobb-Douglas baseline reduction
        c_all_sec[:, 0, :] = theta_sec[:, 0, 0, :] * c_net[0, 0, :]
    else:
        # Tier 1 Stone-Geary LES for Household (nfd=0)
        P_C_hh = P_C[:, 0, :][np.newaxis, :, :]  # (1, ns, nc)
        c_hh = compute_stone_geary_final_demand(Y_con, P_C_hh, calib, pref_cfg)  # (1, ns, nc)
        c_all_sec[:, 0, :] = c_hh[0, :, :] * (1.0 - tax_fd_arr[0, 0, :])

    # Category 1 (GCF) & Category 2 (Government): Linear homothetic Cobb-Douglas
    c_all_sec[:, 1, :] = theta_sec[:, 0, 1, :] * c_net[0, 1, :]
    c_all_sec[:, 2, :] = theta_sec[:, 0, 2, :] * c_net[0, 2, :]
    c_all_sec = np.maximum(c_all_sec, 0.0)

    # 5. Tier 2 Armington Sourcing
    xc = compute_armington_final_demands(
        c_sec=c_all_sec,
        P_C=P_C,
        p=p,
        tau_fd=tau_fd,
        calib=calib,
        sigma_trade=sigma_trade,
        as_3d=True,
    )

    return xc, P_C, P_agg, Tax_c



def compute_benchmark_market_shares(
    calib: TradeCalibrationResult,
) -> np.ndarray:
    """Compute benchmark destination market shares s_ni_0 of shape (ns, nc, nc).

    Index ordering: (sector j, origin country i, destination country n).
    Guarantees sum over origin axis (axis 1) == 1.0 for all active markets.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model containing baseline transaction flows or coefficients.

    Returns
    -------
    s_0 : np.ndarray, shape (ns, nc, nc)
        Benchmark destination market shares where s_0[j, i, n] is the share of
        origin i in total absorption of sector j in destination n.
    """
    cached = getattr(calib, "_benchmark_market_shares_cache", None)
    if cached is not None:
        return cached

    ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand

    data = getattr(calib, "data_calibra", None)
    if data is not None and data.shape[0] >= ns * nc and data.shape[1] >= ns * nc + nfd * nc:
        # 4D Intermediate flows: (origin_sector, origin_country, dest_sector, dest_country)
        x_4d = data[:ns * nc, :ns * nc].reshape(nc, ns, nc, ns).transpose(1, 0, 3, 2)
        # 4D Final demand flows: (origin_sector, origin_country, category, dest_country)
        xfd_4d = data[:ns * nc, ns * nc : ns * nc + nfd * nc].reshape(nc, ns, nc, nfd).transpose(1, 0, 3, 2)
        flow_interm = np.sum(x_4d, axis=2)  # (ns, nc, nc) -> sum over dest_sector
        flow_fd = np.sum(xfd_4d, axis=2)      # (ns, nc, nc) -> sum over category
        flow_tot = flow_interm + flow_fd      # (ns, nc, nc)
    else:
        # Fallback using a and afd
        ytot_arr = np.asarray(calib.ytot, dtype=float).reshape(1, ns, nc)
        a_arr = np.asarray(calib.a, dtype=float)
        if a_arr.ndim == 3 and a_arr.shape == (ns * nc, ns, nc):
            x_3d = a_arr * ytot_arr  # (ns * nc, ns, nc)
            flow_interm_2d = np.sum(x_3d, axis=1)  # (ns * nc, nc)
            flow_interm = flow_interm_2d.reshape(nc, ns, nc).transpose(1, 0, 2)  # (ns, nc, nc)
        else:
            flow_interm = np.zeros((ns, nc, nc), dtype=float)

        if getattr(calib, "afd_4d", None) is not None:
            flow_fd = np.sum(calib.afd_4d, axis=2)  # (ns, nc, nc)
        elif getattr(calib, "afd", None) is not None:
            afd_arr = np.asarray(calib.afd, dtype=float)
            if afd_arr.ndim == 3 and afd_arr.shape == (ns * nc, nfd, nc):
                flow_fd_2d = np.sum(afd_arr, axis=1)  # (ns * nc, nc)
                flow_fd = flow_fd_2d.reshape(nc, ns, nc).transpose(1, 0, 2)
            else:
                flow_fd = np.zeros((ns, nc, nc), dtype=float)
        else:
            flow_fd = np.zeros((ns, nc, nc), dtype=float)

        flow_tot = flow_interm + flow_fd

    abs_tot = np.sum(flow_tot, axis=1, keepdims=True)  # (ns, 1, nc)
    s_0 = np.divide(flow_tot, abs_tot, out=np.zeros_like(flow_tot), where=(abs_tot > 0))

    try:
        object.__setattr__(calib, "_benchmark_market_shares_cache", s_0)
    except Exception:
        pass

    return s_0


def compute_benchmark_markups(
    calib: TradeCalibrationResult,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray:
    """Compute and cache benchmark Atkeson-Burstein markups mu_ni_0 of shape (ns, nc, nc).

    Formula:
        mu_ni_0 = clip( sigma_j / max(sigma_j - 1 + (1 - sigma_j / theta_j) * s_0, clamping_threshold), markup_min, markup_max )

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated trade model.
    market_cfg : FlexibleMarketStructureConfig, optional
        Market structure configuration containing elasticities and markup bounds.

    Returns
    -------
    mu_0 : np.ndarray, shape (ns, nc, nc)
        Benchmark markups bounded in [markup_min, markup_max].
    """
    if market_cfg is None:
        market_cfg = FlexibleMarketStructureConfig()

    s_0 = compute_benchmark_market_shares(calib)
    sigma_j = market_cfg.sigma_j
    theta_j = market_cfg.theta_j
    clamping_threshold = market_cfg.clamping_threshold
    markup_min = market_cfg.markup_min
    markup_max = market_cfg.markup_max
    cournot_weights = market_cfg.cournot_weights

    s_eff = s_0 * cournot_weights if cournot_weights is not None else s_0
    denom = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_eff
    denom_clamped = np.maximum(denom, clamping_threshold)
    mu_0 = np.clip(sigma_j / denom_clamped, markup_min, markup_max)
    return mu_0


def compute_atkeson_burstein_markups(
    s_ni: np.ndarray,
    c_i: np.ndarray | None = None,
    market_cfg: FlexibleMarketStructureConfig | None = None,
    sigma_j: float = 6.0,
    theta_j: float = 2.0,
    mu_0: np.ndarray | float | None = None,
    s_0: np.ndarray | float | None = None,
    s0: np.ndarray | float | None = None,
    cournot_weights: Any = None,
    clamping_threshold: float = 1e-4,
    markup_min: float = 1.0,
    markup_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute Atkeson-Burstein Cournot markups and firm-level prices.

    Formulas:
        s_eff = s_ni * cournot_weights (if cournot_weights provided)
        denom = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_eff
        denom_clamped = np.maximum(denom, clamping_threshold)
        mu_ni = np.clip(sigma_j / denom_clamped, markup_min, markup_max)

    Relative pricing (when mu_0 or s_0 / s0 provided):
        p_ni = (mu_ni / mu_0) * c_i

    Level pricing (backward compatibility when c_i provided without mu_0 or s_0):
        p_ni = mu_ni * c_i

    Parameters
    ----------
    s_ni : np.ndarray
        Destination market shares of origin i in destination n.
    c_i : np.ndarray | None, default None
        Marginal cost of production in origin country.
    market_cfg : FlexibleMarketStructureConfig | None, default None
        Configuration container for market structure parameters.
    sigma_j : float, default 6.0
        Within-industry variety substitution elasticity.
    theta_j : float, default 2.0
        Across-industry sectoral substitution elasticity.
    mu_0 : np.ndarray | float | None, default None
        Benchmark markup for exact Calibrated Share Form relative pricing.
    s_0 : np.ndarray | float | None, default None
        Benchmark market share from which mu_0 is computed if mu_0 is None.
    s0 : np.ndarray | float | None, default None
        Alias for s_0.
    cournot_weights : Any, default None
        Conduct parameter / market share weighting.
    clamping_threshold : float, default 1e-4
        Lower bound for denominator to prevent division by zero.
    markup_min : float, default 1.0
        Lower bound on markups.
    markup_max : float, default 5.0
        Upper bound on markups.

    Returns
    -------
    mu_ni : np.ndarray
        Endogenous markups bounded in [markup_min, markup_max].
    p_ni : np.ndarray | None
        FOB seller prices to destination n (relative or level).
    """
    if s_0 is None and s0 is not None:
        s_0 = s0

    if market_cfg is not None:
        sigma_j = market_cfg.sigma_j
        theta_j = market_cfg.theta_j
        markup_min = market_cfg.markup_min
        markup_max = market_cfg.markup_max
        clamping_threshold = market_cfg.clamping_threshold
        if cournot_weights is None:
            cournot_weights = market_cfg.cournot_weights

    s_eff = s_ni * cournot_weights if cournot_weights is not None else s_ni
    denom = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_eff
    denom_clamped = np.maximum(denom, clamping_threshold)
    mu_ni = np.clip(sigma_j / denom_clamped, markup_min, markup_max)

    if c_i is None:
        return mu_ni, None

    if mu_0 is not None:
        mu_0_safe = np.maximum(np.asarray(mu_0, dtype=float), 1e-12)
        p_ni = (mu_ni / mu_0_safe) * c_i
    elif s_0 is not None:
        s_0_eff = s_0 * cournot_weights if cournot_weights is not None else s_0
        denom_0 = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_0_eff
        denom_0_clamped = np.maximum(denom_0, clamping_threshold)
        mu_0_calc = np.clip(sigma_j / denom_0_clamped, markup_min, markup_max)
        mu_0_safe = np.maximum(mu_0_calc, 1e-12)
        p_ni = (mu_ni / mu_0_safe) * c_i
    else:
        p_ni = mu_ni * c_i

    return mu_ni, p_ni


def compute_dixit_stiglitz_varieties(
    pi_op: np.ndarray | float,
    w: np.ndarray | float = 1.0,
    r: np.ndarray | float = 1.0,
    fl: float = 0.0,
    fk: float = 0.0,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray | float:
    """Condense firm variety count N via zero-profit entry condition.

    Formula:
        N = max(pi_op / (w * fl + r * fk), 0.0)

    When fl <= 0.0 and fk <= 0.0, falls back to constant baseline variety N = 1.0.

    Parameters
    ----------
    pi_op : np.ndarray or float
        Operating profit earned by firms.
    w : np.ndarray or float, default 1.0
        Domestic wage rate.
    r : np.ndarray or float, default 1.0
        Domestic capital rental rate.
    fl : float, default 0.0
        Fixed labor requirement per firm.
    fk : float, default 0.0
        Fixed capital requirement per firm.
    market_cfg : FlexibleMarketStructureConfig, optional
        Market structure configuration. If provided, fl and fk are extracted from it.

    Returns
    -------
    N : np.ndarray or float
        Endogenous firm variety count, non-negative everywhere.
    """
    if market_cfg is not None:
        fl = market_cfg.fl
        fk = market_cfg.fk

    # Zero or negative fixed cost fallback: return 1.0
    if fl <= 0.0 and fk <= 0.0:
        if isinstance(pi_op, (int, float, np.floating, np.integer)):
            return 1.0
        return np.ones_like(pi_op, dtype=float)

    fixed_cost = w * fl + r * fk
    if isinstance(pi_op, (int, float, np.floating, np.integer)) and isinstance(fixed_cost, (int, float, np.floating, np.integer)):
        if fixed_cost <= 0.0:
            return 1.0
        return max(float(pi_op) / float(fixed_cost), 0.0)

    fixed_cost_arr = np.asarray(fixed_cost, dtype=float)
    safe_fixed = np.where(fixed_cost_arr > 1e-12, fixed_cost_arr, 1e-12)
    pi_arr = np.asarray(pi_op, dtype=float)
    N_arr = np.maximum(pi_arr / safe_fixed, 0.0)
    return np.where(fixed_cost_arr > 1e-12, N_arr, 1.0)


# Backward-compatible alias
compute_condensed_varieties = compute_dixit_stiglitz_varieties


def compute_variety_price_scaling(
    p: np.ndarray | float | None = None,
    N: np.ndarray | float | None = None,
    N0: np.ndarray | float = 1.0,
    sigma_j: float = 6.0,
    market_cfg: FlexibleMarketStructureConfig | None = None,
) -> np.ndarray | float:
    """Compute effective price index accounting for Dixit-Stiglitz variety expansion.

    Formula:
        p_eff = p * (N / N0) ** (1.0 / (1.0 - sigma_j))

    At N == N0, scaling is exactly 1.0 (p_eff == p).

    Parameters
    ----------
    p : np.ndarray or float, optional, default 1.0
        Delivered price or base price index. If omitted, returns pure variety scaling factor.
    N : np.ndarray or float, optional
        Current firm variety count. If only one positional argument is given,
        it is treated as N with p=1.0.
    N0 : np.ndarray or float, default 1.0
        Benchmark firm variety count.
    sigma_j : float, default 6.0
        Within-industry variety substitution elasticity (sigma_j > 1.0).
    market_cfg : FlexibleMarketStructureConfig, optional
        Market structure configuration container.

    Returns
    -------
    p_eff : np.ndarray or float
        Effective price accounting for variety expansion.
    """
    if market_cfg is not None:
        sigma_j = market_cfg.sigma_j
        if not (market_cfg.condense_varieties or market_cfg.variety_condensation or market_cfg.variety_expansion):
            if p is None:
                return 1.0
            return p

    if N is None:
        if p is None:
            return 1.0
        # Single positional argument passed -> interpreted as N
        N = p
        p = 1.0

    if p is None:
        p = 1.0

    if sigma_j <= 1.0:
        raise ValueError(f"sigma_j must be strictly greater than 1.0, got {sigma_j}.")

    exponent = 1.0 / (1.0 - sigma_j)

    if isinstance(N, (int, float, np.floating, np.integer)) and isinstance(N0, (int, float, np.floating, np.integer)):
        if float(N) == float(N0):
            scale = 1.0
        else:
            ratio = max(float(N) / max(float(N0), 1e-12), 1e-12)
            scale = ratio ** exponent
        if isinstance(p, (int, float, np.floating, np.integer)):
            if scale == 1.0:
                return float(p)
            return float(p) * scale
        return np.asarray(p, dtype=float) * scale

    N_arr = np.asarray(N, dtype=float)
    N0_arr = np.asarray(N0, dtype=float)
    safe_N0 = np.maximum(N0_arr, 1e-12)
    ratio_arr = np.maximum(N_arr / safe_N0, 1e-12)
    scale_arr = ratio_arr ** exponent
    scale_arr = np.where(N_arr == N0_arr, 1.0, scale_arr)

    if isinstance(p, (int, float, np.floating, np.integer)) and float(p) == 1.0:
        return scale_arr
    return np.asarray(p, dtype=float) * scale_arr


def compute_convergence_diagnostics(
    residual: np.ndarray | Sequence[float] | None,
    calib: Any = None,
    layout: str = "codebase",
    n_top: int = 3,
) -> dict[str, Any]:
    """Compute market-specific convergence failure diagnostics for CGE residual equations.

    Extracts the top worst-offending equations by absolute residual magnitude,
    maps equation indices using codebase block indexing and Fortran order
    (k % ns = sector, k // ns = country), maps them to economic identities, and
    provides targeted remediation suggestions.

    Parameters
    ----------
    residual : array-like or None
        Residual vector of length 2*M + 4*nc - 1 (e.g. 2,001 for 77c x 11s).
    calib : Any, optional
        Calibration object containing country_codes, sector_codes, n_countries, n_sectors.
    layout : {"codebase", "spec"}, default "codebase"
        Equation block ordering:
        - 'codebase': Matches puremacro.trade runtime residual vector:
            [Goods Market Clearing, Zero-Profit Condition, Labor Market,
             Capital Market, Trade Balance, Fiscal Budget].
        - 'spec': Matches theoretical state-dual ordering:
            [Zero-profit gross output pricing, Market clearing for goods,
             Capital market clearing, Labor market clearing,
             Trade balance / transfers, Numeraire / price normalization].
    n_top : int, default 3
        Number of worst-offending equations to extract.

    Returns
    -------
    dict[str, Any]
        Diagnostic dictionary containing:
        - "top_equations": list of dicts (keys: rank, index, block, equation,
          country, sector, residual, abs_residual) with native Python types.
        - "max_residual": maximum finite absolute residual (float).
        - "l1_norm": sum of finite absolute residuals (float).
        - "remediation": actionable remediation advice string.
    """
    if residual is None:
        return {
            "top_equations": [],
            "max_residual": 0.0,
            "l1_norm": 0.0,
            "remediation": "No residual vector provided.",
        }

    res_arr = np.asarray(residual, dtype=float).ravel()
    if len(res_arr) == 0:
        return {
            "top_equations": [],
            "max_residual": 0.0,
            "l1_norm": 0.0,
            "remediation": "Empty residual vector provided.",
        }

    # Resolve dimensions
    if calib is not None and hasattr(calib, "n_countries") and hasattr(calib, "n_sectors"):
        nc = int(calib.n_countries)
        ns = int(calib.n_sectors)
    else:
        n_tot = len(res_arr)
        if n_tot == 2001:
            nc, ns = 77, 11
        elif n_tot == 15:
            nc, ns = 2, 2
        else:
            nc, ns = 2, 2
    M = ns * nc

    raw_c_codes = getattr(calib, "country_codes", None)
    if raw_c_codes and len(raw_c_codes) == nc:
        c_codes = [str(x) for x in raw_c_codes]
    else:
        c_codes = [f"C{i:02d}" for i in range(nc)]

    raw_s_codes = getattr(calib, "sector_codes", None)
    if raw_s_codes and len(raw_s_codes) == ns:
        s_codes = [str(x) for x in raw_s_codes]
    else:
        s_codes = [f"S{j:02d}" for j in range(ns)]

    abs_res = np.abs(res_arr)
    rank_keys = np.where(np.isnan(abs_res), np.inf, abs_res)
    sorted_indices = np.argsort(-rank_keys, kind="stable")

    k_top = min(max(int(n_top), 0), len(res_arr))
    top_list: list[dict[str, Any]] = []

    for rank, idx in enumerate(sorted_indices[:k_top], start=1):
        idx = int(idx)
        raw_val = float(res_arr[idx])
        abs_val = float(abs_res[idx]) if not np.isnan(raw_val) else float("nan")

        if layout == "spec":
            if idx < M:
                block = "Zero-Profit Condition"
                s = idx % ns
                c = idx // ns
                country, sector = c_codes[c], s_codes[s]
                eq_str = f"Zero-profit gross output pricing: Sector {sector}, Country {country}"
            elif idx < 2 * M:
                block = "Goods Market Clearing"
                rem = idx - M
                s = rem % ns
                c = rem // ns
                country, sector = c_codes[c], s_codes[s]
                eq_str = f"Market clearing for gross output / goods: Sector {sector}, Country {country}"
            elif idx < 2 * M + nc:
                block = "Capital Market"
                c = idx - 2 * M
                country, sector = c_codes[c], None
                eq_str = f"Capital market clearing: Country {country}"
            elif idx < 2 * M + 2 * nc:
                block = "Labor Market"
                c = idx - (2 * M + nc)
                country, sector = c_codes[c], None
                eq_str = f"Labor market clearing: Country {country}"
            elif idx < 2 * M + 3 * nc - 1:
                block = "Trade Balance"
                c = idx - (2 * M + 2 * nc)
                country, sector = c_codes[c], None
                eq_str = f"Trade balance / transfers: Country {country}"
            else:
                block = "Fiscal Budget"
                c = idx - (2 * M + 3 * nc - 1)
                country, sector = c_codes[c], None
                eq_str = f"Numeraire / price normalization: Country {country}"
        else:  # "codebase" layout (default)
            if idx < M:
                block = "Goods Market Clearing"
                s = idx % ns
                c = idx // ns
                country, sector = c_codes[c], s_codes[s]
                eq_str = f"Goods Market Clearing: Sector {sector}, Country {country}"
            elif idx < 2 * M:
                block = "Zero-Profit Condition"
                rem = idx - M
                s = rem % ns
                c = rem // ns
                country, sector = c_codes[c], s_codes[s]
                eq_str = f"Zero Profit Condition: Sector {sector}, Country {country}"
            elif idx < 2 * M + nc:
                block = "Labor Market Clearing"
                c = idx - 2 * M
                country, sector = c_codes[c], None
                eq_str = f"Labor Market Clearing: Country {country}"
            elif idx < 2 * M + 2 * nc:
                block = "Capital Market Clearing"
                c = idx - (2 * M + nc)
                country, sector = c_codes[c], None
                eq_str = f"Capital Market Clearing: Country {country}"
            elif idx < 2 * M + 3 * nc - 1:
                block = "Trade Balance"
                c = idx - (2 * M + 2 * nc)
                country, sector = c_codes[c], None
                eq_str = f"Trade Balance / Current Account: Country {country}"
            else:
                block = "Fiscal Budget Consistency"
                c = idx - (2 * M + 3 * nc - 1)
                country, sector = c_codes[c], None
                eq_str = f"Fiscal Budget Consistency: Country {country}"

        top_list.append({
            "rank": int(rank),
            "index": int(idx),
            "block": str(block),
            "equation": str(eq_str),
            "country": str(country),
            "sector": str(sector) if sector is not None else None,
            "residual": float(raw_val),
            "abs_residual": float(abs_val),
        })

    top_eq = top_list[0] if top_list else None
    if top_eq is None:
        remediation = "No residual equations evaluated."
    elif np.isnan(top_eq["abs_residual"]) or np.isinf(top_eq["abs_residual"]):
        remediation = (
            f"Severe numerical divergence detected in {top_eq['block']}. "
            "Check for division by zero, non-positive factor prices, or degenerate substitution bounds."
        )
    elif top_eq["block"] == "Goods Market Clearing":
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Review intermediate sourcing tariffs (tau), final demand expenditure shares, "
            "or increase solver damping (damping <= 0.5)."
        )
    elif top_eq["block"] == "Zero-Profit Condition":
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Review value-added CES substitution elasticity (rho_va), factor prices, "
            "or intermediate cost shares."
        )
    elif top_eq["block"] in ("Labor Market", "Labor Market Clearing"):
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Adjust wage adjustment damping, examine labor endowment calibration (l_endow), "
            "or check labor share (1 - alpha)."
        )
    elif top_eq["block"] in ("Capital Market", "Capital Market Clearing"):
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Adjust rental rate damping, examine capital endowment calibration (k_endow), "
            "or check capital share (alpha)."
        )
    elif top_eq["block"] == "Trade Balance":
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Verify Armington trade elasticity (sigma_trade) and import price indices, "
            "or check foreign trade transfer closure."
        )
    else:
        remediation = (
            f"Largest residual is in {top_eq['equation']}. "
            "Suggestion: Verify tariff revenue recycling parameters and government budget closure."
        )

    finite_vals = abs_res[np.isfinite(abs_res)]
    max_res = float(np.max(finite_vals)) if len(finite_vals) > 0 else 0.0
    l1_norm = float(np.sum(finite_vals)) if len(finite_vals) > 0 else 0.0

    return {
        "top_equations": top_list,
        "max_residual": max_res,
        "worst_residual": max_res,
        "l1_norm": l1_norm,
        "remediation": remediation,
    }


@dataclass
class FlexibleTradeEquilibriumResult:
    """Result container for flexible trade general equilibrium solutions.

    Encapsulates solved equilibrium state vectors, market prices, factor returns,
    endogenous markups, flow accounting, and welfare metrics. Adheres to
    progressive disclosure principles with high-level inspection methods.

    Parameters / Attributes
    ----------
    x_sol : np.ndarray, shape (2001,)
        Canonical general equilibrium state vector: [log(p); log(y); log(r); log(w); T; XN].
    converged : bool, default True
        Whether the nonlinear root solver converged within tolerance.
    iterations : int, default 0
        Number of solver iterations executed.
    residual_norm : float, default 0.0
        Final maximum absolute equation residual: max(|ff|).
    residuals : np.ndarray | None, default None
        Full residual vector ff at solution.
    metadata : dict[str, Any], default empty dict
        Convergence diagnostics, solver configurations, and provenance data.
    markups : pd.DataFrame, default empty DataFrame
        DataFrame of post-shock counterfactual markups by origin, destination, and sector.
    calib : TradeCalibrationResult | None, default None
        Calibrated structural baseline dataset.
    _factors_frame : pd.DataFrame | None, default None
        Cached factor allocation DataFrame.
    config : FlexibleTradeModelConfig | None, default None
        Resolved flexible model configuration used to generate this equilibrium.
    """

    x_sol: np.ndarray
    converged: bool = True
    iterations: int = 0
    residual_norm: float = 0.0
    residuals: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    markups: pd.DataFrame = field(default_factory=pd.DataFrame)
    calib: Any = None
    _factors_frame: pd.DataFrame | None = None
    config: FlexibleTradeModelConfig | None = None
    _postproc_cache: dict[str, Any] | None = field(default=None, repr=False)

    def _get_unpacked(self) -> Any:
        from puremacro.trade.equilibrium import unpack_equilibrium_vector
        if self.calib is not None:
            ns, nc, nfd = self.calib.n_sectors, self.calib.n_countries, self.calib.n_final_demand
        else:
            ns, nc, nfd = 11, 77, 3
        return unpack_equilibrium_vector(self.x_sol, ns=ns, nc=nc, nfd=nfd)

    def _get_postproc(self) -> dict[str, Any]:
        if self._postproc_cache is not None:
            return self._postproc_cache
        if self.calib is not None:
            from puremacro.trade.postprocessing import compute_postprocessing_flows
            flows = compute_postprocessing_flows(x_sol=self.x_sol, calib=self.calib)
            self._postproc_cache = flows
            return flows
        return {}

    @property
    def p_sol(self) -> np.ndarray:
        """Equilibrium gross output prices."""
        return self._get_unpacked().p

    @property
    def y_sol(self) -> np.ndarray:
        """Equilibrium gross output quantities."""
        return self._get_unpacked().y

    @property
    def r_sol(self) -> np.ndarray:
        """Equilibrium capital rental rates."""
        return self._get_unpacked().r

    @property
    def w_sol(self) -> np.ndarray:
        """Equilibrium wage rates."""
        return self._get_unpacked().w

    @property
    def T_sol(self) -> np.ndarray:
        """Equilibrium government tax revenues."""
        return self._get_unpacked().T

    @property
    def XN_sol(self) -> np.ndarray:
        """Equilibrium net foreign transfers."""
        return self._get_unpacked().XN

    @property
    def cpi(self) -> np.ndarray | None:
        """Domestic consumer price index relative to baseline."""
        return self._get_postproc().get("cpi")

    @property
    def terms_of_trade(self) -> np.ndarray | None:
        """National terms of trade index (export price index / import price index)."""
        return self._get_postproc().get("terms_of_trade")

    @property
    def exports(self) -> np.ndarray | None:
        """National total gross exports across intermediate and final goods."""
        return self._get_postproc().get("exports")

    @property
    def imports(self) -> np.ndarray | None:
        """National total gross imports across intermediate and final goods."""
        return self._get_postproc().get("imports")

    @property
    def gdp(self) -> np.ndarray | None:
        """National GDP at market prices."""
        return self._get_postproc().get("gdp")

    @property
    def gdp_fc(self) -> np.ndarray | None:
        """National GDP at factor cost (labor income plus capital income)."""
        return self._get_postproc().get("gdp_fc")

    def summary_markups(self, by_sector: bool = True) -> pd.DataFrame:
        """Summary statistics of calibrated and counterfactual markups.

        Parameters
        ----------
        by_sector : bool, default True
            If True, returns summary statistics (mean, min, max, std) grouped by
            sector across destination markets. If False, returns aggregate statistics.

        Returns
        -------
        pd.DataFrame
            Summary statistics table.
        """
        if isinstance(self.markups, pd.DataFrame) and not self.markups.empty and "markup" in self.markups.columns:
            if by_sector and "sector" in self.markups.columns:
                grouped = (
                    self.markups.groupby("sector", as_index=False, sort=False)["markup"]
                    .agg(mean="mean", min="min", max="max", std="std")
                )
                grouped["std"] = grouped["std"].fillna(0.0)
                return grouped
            return pd.DataFrame({
                "metric": ["mean", "min", "max", "std"],
                "value": [
                    float(self.markups["markup"].mean()),
                    float(self.markups["markup"].min()),
                    float(self.markups["markup"].max()),
                    float(self.markups["markup"].std()) if len(self.markups) > 1 else 0.0,
                ],
            })
        if by_sector and self.calib is not None:
            s_codes = list(self.calib.sector_codes) if self.calib.sector_codes else [f"S{i:02d}" for i in range(self.calib.n_sectors)]
            return pd.DataFrame({
                "sector": s_codes,
                "mean": [1.0] * len(s_codes),
                "min": [1.0] * len(s_codes),
                "max": [1.0] * len(s_codes),
                "std": [0.0] * len(s_codes),
            })
        return pd.DataFrame({"metric": ["mean", "min", "max", "std"], "value": [1.0, 1.0, 1.0, 0.0]})

    def factor_allocation_frame(self) -> pd.DataFrame:
        """Sectoral and national factor allocation DataFrame.

        Computes solved general equilibrium labor and capital factor demands (xl, xk)
        across all countries and sectors from factor returns, gross output, and
        CES technology parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ["country", "sector", "labor", "capital", "xl", "xk"]
            satisfying national endowment balance.
        """
        if self._factors_frame is not None:
            return self._factors_frame
        if self.calib is not None:
            calib = self.calib
            ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
            uv = self._get_unpacked()
            p, y, r, w = uv.p, uv.y, uv.r, uv.w

            tech_cfg = (
                self.config.technology
                if (self.config is not None and hasattr(self.config, "technology"))
                else FlexibleTechnologyConfig()
            )

            tau_a = getattr(calib, "tau_a", None)
            if tau_a is None:
                tau_a = np.ones((ns * nc, ns, nc), dtype=float)

            P_M, _, _ = compute_intermediate_composite_price(
                p=p, tau_a=tau_a, calib=calib, sigma=float(tech_cfg.sigma_inter)
            )
            c_va, c_y = compute_nested_ces_costs(r=r, w=w, P_M=P_M, calib=calib, tech_cfg=tech_cfg)
            xl, xk, _ = compute_nested_factor_demands(
                ytot=y, r=r, w=w, P_M=P_M, c_va=c_va, c_y=c_y, p=p, tau=tau_a,
                calib=calib, tech_cfg=tech_cfg, normalized=True,
            )

            c_codes = list(calib.country_codes) if calib.country_codes else [f"C{i:02d}" for i in range(nc)]
            s_codes = list(calib.sector_codes) if calib.sector_codes else [f"S{i:02d}" for i in range(ns)]

            records = []
            for c_idx, c_code in enumerate(c_codes):
                for s_idx, s_code in enumerate(s_codes):
                    xl_val = float(xl[0, s_idx, c_idx])
                    xk_val = float(xk[0, s_idx, c_idx])
                    records.append({
                        "country": c_code,
                        "sector": s_code,
                        "labor": xl_val,
                        "capital": xk_val,
                        "xl": xl_val,
                        "xk": xk_val,
                    })
            df = pd.DataFrame(records)
            self._factors_frame = df
            return df
        return pd.DataFrame(columns=["country", "sector", "labor", "capital", "xl", "xk"])

    def welfare_decomposition(self, base_result: Any = None) -> pd.DataFrame:
        """Hicksian Equivalent Variation, terms-of-trade, and allocative efficiency welfare decomposition.

        Parameters
        ----------
        base_result : FlexibleTradeEquilibriumResult | TradeEquilibriumResult | None, default None
            Benchmark equilibrium result. If None or self, self-identity returns zero welfare changes.

        Returns
        -------
        pd.DataFrame
            Welfare decomposition table with columns ["country", "EV", "terms_of_trade", "efficiency"].
        """
        if base_result is not None:
            x_base = getattr(base_result, "x_sol", None)
            if x_base is not None and len(x_base) != len(self.x_sol):
                raise ValueError(
                    f"Dimension mismatch in welfare decomposition: {len(x_base)} vs {len(self.x_sol)}."
                )

        calib = self.calib if self.calib is not None else getattr(base_result, "calib", None)
        if calib is None:
            countries = ["C01"]
            return pd.DataFrame({
                "country": countries,
                "EV": [0.0],
                "terms_of_trade": [0.0],
                "efficiency": [0.0],
            })

        nc = calib.n_countries
        ns = calib.n_sectors
        nfd = calib.n_final_demand
        c_codes = list(calib.country_codes) if calib.country_codes else [f"C{i:02d}" for i in range(nc)]

        x_base = getattr(base_result, "x_sol", None) if base_result is not None else None
        if base_result is None or base_result is self or (x_base is not None and np.allclose(self.x_sol, x_base, atol=1e-12)):
            return pd.DataFrame({
                "country": c_codes,
                "EV": [0.0] * nc,
                "terms_of_trade": [0.0] * nc,
                "efficiency": [0.0] * nc,
            })

        from puremacro.trade.equilibrium import unpack_equilibrium_vector

        uv1 = unpack_equilibrium_vector(self.x_sol, ns=ns, nc=nc, nfd=nfd)
        uv0 = unpack_equilibrium_vector(x_base, ns=ns, nc=nc, nfd=nfd)

        l_endow = np.asarray(calib.l_endow, dtype=float).ravel()
        k_endow = np.asarray(calib.k_endow, dtype=float).ravel()
        Y_con_0 = uv0.w.ravel() * l_endow + uv0.r.ravel() * k_endow + uv0.T.ravel()
        Y_con_1 = uv1.w.ravel() * l_endow + uv1.r.ravel() * k_endow + uv1.T.ravel()

        theta_hh = (
            calib.theta[:, 0:1, :].ravel()
            if calib.theta is not None
            else np.full(nc, 1.0 / nfd)
        )
        Y_C_0 = theta_hh * Y_con_0
        Y_C_1 = theta_hh * Y_con_1

        pref_cfg = (
            self.config.preference
            if (self.config is not None and hasattr(self.config, "preference"))
            else FlexiblePreferenceConfig()
        )
        sigma_trade = (
            float(pref_cfg.sigma_trade)
            if isinstance(pref_cfg.sigma_trade, (int, float))
            else 5.0
        )

        tau_fd = getattr(calib, "taufd_a", None)
        if tau_fd is None:
            tau_fd = np.ones((ns * nc, nfd, nc), dtype=float)

        P_C_3d_0 = compute_armington_purchaser_prices(uv0.p, tau_fd, calib, sigma_trade=sigma_trade)
        P_C_3d_1 = compute_armington_purchaser_prices(uv1.p, tau_fd, calib, sigma_trade=sigma_trade)
        P_C_0 = P_C_3d_0[:, 0, :]  # (ns, nc)
        P_C_1 = P_C_3d_1[:, 0, :]  # (ns, nc)

        mu_arr = _resolve_subsistence_shares(calib, pref_cfg)
        mu_2d = mu_arr[0]  # (ns, nc)

        if np.all(mu_2d == 0.0):
            _, _, theta_sec_0, _, _ = _extract_benchmark_household_data(calib)
            theta_2d = theta_sec_0[0]
            log_P0 = np.sum(theta_2d * np.log(np.maximum(P_C_0, 1e-12)), axis=0)
            log_P1 = np.sum(theta_2d * np.log(np.maximum(P_C_1, 1e-12)), axis=0)
            P_ratio = np.exp(log_P0 - log_P1)
            EV = Y_C_1 * P_ratio - Y_C_0
        else:
            Ycon_0_bm, E_C_0_bm, theta_sec_0, E_Cs0, c_s0 = _extract_benchmark_household_data(calib)
            c_s0_2d = c_s0[0]
            E_Cs0_2d = E_Cs0[0]
            E_C_0_2d = E_C_0_bm[0, 0, :]
            Ycon_0_2d = Ycon_0_bm[0, 0, :]

            u0 = np.maximum(Y_con_0 / np.maximum(Ycon_0_2d, 1e-12), 0.0)
            u1 = np.maximum(Y_con_1 / np.maximum(Ycon_0_2d, 1e-12), 0.0)
            g_u0 = smooth_subsistence_scaling(u0)
            g_u1 = smooth_subsistence_scaling(u1)
            c_bar_0 = mu_2d * c_s0_2d * g_u0[None, :]
            c_bar_1 = mu_2d * c_s0_2d * g_u1[None, :]

            num = (1.0 - mu_2d) * E_Cs0_2d
            sub_E0 = np.sum(mu_2d * E_Cs0_2d, axis=0, keepdims=True)
            denom = E_C_0_2d[None, :] - sub_E0
            denom_safe = np.where(denom > 1e-12, denom, 1.0)
            theta_LES = np.divide(num, denom_safe, out=theta_sec_0[0].copy(), where=(denom > 1e-12))
            sum_les = np.sum(theta_LES, axis=0, keepdims=True)
            theta_LES = np.where(sum_les > 0, theta_LES / sum_les, theta_sec_0[0])

            log_P0_les = np.sum(theta_LES * np.log(np.maximum(P_C_0, 1e-12)), axis=0)
            log_P1_les = np.sum(theta_LES * np.log(np.maximum(P_C_1, 1e-12)), axis=0)
            P_LES_ratio = np.exp(log_P0_les - log_P1_les)

            sub_exp_0 = np.sum(P_C_0 * c_bar_0, axis=0)
            sub_exp_1 = np.sum(P_C_1 * c_bar_1, axis=0)
            super_inc_1 = np.maximum(Y_C_1 - sub_exp_1, 0.0)

            e_P0_u1 = sub_exp_0 + super_inc_1 * P_LES_ratio
            EV = e_P0_u1 - Y_C_0

        tot_1 = np.asarray(self.terms_of_trade, dtype=float).ravel() if self.terms_of_trade is not None else np.ones(nc)
        tot_0 = (
            np.asarray(base_result.terms_of_trade, dtype=float).ravel()
            if getattr(base_result, "terms_of_trade", None) is not None
            else np.ones(nc)
        )
        exp_1 = np.asarray(self.exports, dtype=float).ravel() if self.exports is not None else np.zeros(nc)
        imp_1 = np.asarray(self.imports, dtype=float).ravel() if self.imports is not None else np.zeros(nc)
        trade_vol = 0.5 * (exp_1 + imp_1)
        tot_effect = (tot_1 - tot_0) * trade_vol

        efficiency = EV - tot_effect

        return pd.DataFrame({
            "country": c_codes,
            "EV": [float(v) for v in EV],
            "terms_of_trade": [float(v) for v in tot_effect],
            "efficiency": [float(v) for v in efficiency],
        })

    def welfare_summary(self, base_result: Any = None) -> pd.DataFrame:
        """Alias for welfare_decomposition."""
        return self.welfare_decomposition(base_result)

    @property
    def convergence_diagnostic(self) -> dict[str, Any]:
        """Convergence failure diagnostic dictionary from metadata."""
        diag = self.metadata.get("convergence_diagnostic", {})
        return diag if isinstance(diag, dict) else {}

    @property
    def top_offending_equations(self) -> list[dict[str, Any]]:
        """List of top worst-offending residual equations from convergence diagnostics."""
        diag = self.convergence_diagnostic
        return diag.get("top_equations", []) if isinstance(diag, dict) else []


FlexibleEquilibriumResult = FlexibleTradeEquilibriumResult


def _solve_inner_prices(
    xm_curr: np.ndarray,
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
    tau_a: np.ndarray,
    lu_P: Any,
    p_init: np.ndarray | None = None,
    replicate_matlab_precedence: bool = True,
) -> tuple[np.ndarray, int]:
    """Solve inner price fixed point p = T_price(p; xm) with latency cap <= 10.

    Under default settings (sigma_y < 1e-6, variable_markups=False, sigma_inter < 1e-6),
    short-circuits directly to LAPACK solve (0 inner iterations).
    Under flexible settings, enforces an inner fixed-point iteration cap <= 10.
    """
    nc = calib.n_countries
    ns = calib.n_sectors
    M = ns * nc
    tax_flat = calib.tax.flatten(order="F")

    r = np.exp(xm_curr[:nc]).reshape((1, 1, nc))
    w = np.exp(xm_curr[nc : 2 * nc]).reshape((1, 1, nc))

    # Evaluate Value-Added Unit Cost c_va(r, w)
    c_va, _ = compute_nested_ces_costs(
        r=r, w=w, P_M=np.ones((1, ns, nc)), calib=calib, tech_cfg=config.technology
    )
    v_P = c_va.flatten(order="F") / np.maximum(1.0 - tax_flat, 1e-12)

    # 1. Short-circuit directly to LAPACK solve under default linear settings
    is_linear = (
        float(getattr(config.technology, "sigma_y", 0.0)) < 1e-6
        and float(getattr(config.technology, "sigma_inter", 0.0)) < 1e-6
        and not bool(getattr(config.market_structure, "variable_markups", False))
    )
    if is_linear:
        p_vec = lu_P.solve(v_P)
        return p_vec, 0

    # 2. Flexible setting: Contractive Fixed-Point Iteration (capped <= 10)
    K_max = min(max(1, int(getattr(config, "max_inner_iter", 10))), 10)
    omega = 0.8
    tol_inner = 1e-8

    p_curr = p_init.copy() if p_init is not None else lu_P.solve(v_P)

    inner_iters = 0
    for _ in range(K_max):
        inner_iters += 1
        P_M, _, _ = compute_intermediate_composite_price(
            p=p_curr, tau_a=tau_a, calib=calib, sigma=float(config.technology.sigma_inter)
        )
        c_y = compute_outer_ces_cost(
            c_va=c_va, P_M=P_M, calib=calib, tech_cfg=config.technology, normalized=False
        )
        c_y_flat = c_y.flatten(order="F")

        if config.market_structure.variable_markups:
            s_0 = compute_benchmark_market_shares(calib)
            p_2d = p_curr.reshape((ns, nc), order="F")
            sigma_trade = getattr(calib, "sigma", 5.0)
            sig_val = float(sigma_trade) if isinstance(sigma_trade, (int, float)) else float(np.mean(sigma_trade))
            exponent = 1.0 - sig_val
            p_term = (p_2d[:, :, None]) ** exponent
            shares_unnorm = s_0 * p_term
            sum_shares = np.sum(shares_unnorm, axis=1, keepdims=True)
            s_ni = np.divide(shares_unnorm, sum_shares, out=s_0.copy(), where=(sum_shares > 1e-12))

            mu_0 = compute_benchmark_markups(calib, config.market_structure)
            mu_ni, _ = compute_atkeson_burstein_markups(
                s_ni=s_ni, c_i=None, market_cfg=config.market_structure, mu_0=mu_0
            )
            mu_mean = np.mean(mu_ni / np.maximum(mu_0, 1e-12), axis=2)
            mu_rel = mu_mean.ravel(order="F")
            p_target = (mu_rel * c_y_flat) / np.maximum(1.0 - tax_flat, 1e-12)
        else:
            p_target = c_y_flat / np.maximum(1.0 - tax_flat, 1e-12)

        diff = float(np.max(np.abs(p_target - p_curr)))
        p_curr = (1.0 - omega) * p_curr + omega * p_target

        if diff <= tol_inner:
            break

    return p_curr, inner_iters


def _quasi_condensed_solve(
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
    tau_a: np.ndarray | None = None,
    taufd_a: np.ndarray | None = None,
    tauf_vec: np.ndarray | None = None,
    tauf_fd_vec: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    replicate_matlab_precedence: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray, dict[str, Any]]:
    """Quasi-condensed Newton solver for general equilibrium with flexible trade extensions."""
    from puremacro.trade.solver import _resolve_tariffs, build_initial_guess, unpack_equilibrium_vector

    ns, nc = calib.n_sectors, calib.n_countries
    nfd = calib.n_final_demand
    M = ns * nc
    n_m = 4 * nc - 1

    if tau_a is None or taufd_a is None or tauf_vec is None or tauf_fd_vec is None:
        tau_a, taufd_a, tauf_vec, tauf_fd_vec = _resolve_tariffs(
            calib, tau=kwargs.get("tau"), tau_fd=kwargs.get("tau_fd")
        )

    # Pre-factorize Leontief operators for fast linear solves
    tax_flat = calib.tax.flatten(order="F")
    a_eff_2d = (calib.a * tau_a).reshape((M, M), order="F")
    denom = 1.0 - tax_flat[:, np.newaxis]
    B_T_mat = a_eff_2d.T / np.maximum(denom, 1e-12)
    a_2d = calib.a.reshape((M, M), order="F")

    M_P_dense = np.eye(M, dtype=float) - B_T_mat
    M_Y_dense = np.eye(M, dtype=float) - a_2d
    lu_P_piv = la.lu_factor(M_P_dense)
    lu_Y_piv = la.lu_factor(M_Y_dense)

    class _LUSolver:
        def __init__(self, piv: Any) -> None:
            self._piv = piv

        def solve(self, b: np.ndarray) -> np.ndarray:
            return la.lu_solve(self._piv, b)

    lu_P = _LUSolver(lu_P_piv)
    lu_Y = _LUSolver(lu_Y_piv)

    if x0 is None:
        x_init_full = build_initial_guess(calib)
    else:
        x_init_full = np.asarray(x0, dtype=float).ravel()

    vars0 = unpack_equilibrium_vector(x_init_full, ns=ns, nc=nc, nfd=nfd)
    xm = np.concatenate([
        np.log(np.maximum(vars0.r.ravel(), 1e-12)),
        np.log(np.maximum(vars0.w.ravel(), 1e-12)),
        vars0.T.flatten(order="F"),
        vars0.XN.ravel(),
    ])

    def eval_macro(xm_curr: np.ndarray, p_warm: np.ndarray | None = None, compute_full: bool = True):
        r = np.exp(xm_curr[:nc]).reshape((1, 1, nc))
        w = np.exp(xm_curr[nc : 2 * nc]).reshape((1, 1, nc))
        T = xm_curr[2 * nc : 3 * nc].reshape((1, 1, nc))
        XN = xm_curr[3 * nc :]
        invforT = np.append(XN, -np.sum(XN))

        p_vec, inner_iters = _solve_inner_prices(
            xm_curr=xm_curr,
            calib=calib,
            config=config,
            tau_a=tau_a,
            lu_P=lu_P,
            p_init=p_warm,
            replicate_matlab_precedence=replicate_matlab_precedence,
        )
        p_3d = p_vec.reshape((1, ns, nc), order="F")

        # Final demand
        ppfd = np.tensordot(p_vec, calib.afd * taufd_a, axes=(0, 0))[np.newaxis, :, :]
        Ycon = w * calib.l_endow + r * calib.k_endow + T
        cd = calib.theta * Ycon / ppfd
        tax_fd_arr = calib.tax_fd if calib.tax_fd is not None else np.zeros((1, nfd, nc))
        Tax_c = tax_fd_arr * ppfd * cd
        c = cd.copy()
        c[:, 1:2, :] -= invforT.reshape((1, 1, nc)) / ppfd[:, 1:2, :]
        xc = calib.afd * (c - tax_fd_arr * cd)
        xc_2d = xc.reshape((M, nfd * nc), order="F")
        d = np.sum(xc_2d, axis=1)

        y_vec = lu_Y.solve(d)
        ytot = y_vec.reshape((1, ns, nc), order="F")

        # Factor demands
        c_va, c_y = compute_nested_ces_costs(
            r=r, w=w, P_M=np.ones((1, ns, nc)), calib=calib, tech_cfg=config.technology
        )
        xl, xk, _ = compute_nested_factor_demands(
            ytot=ytot,
            r=r,
            w=w,
            P_M=np.ones((1, ns, nc)),
            c_va=c_va,
            c_y=c_y,
            p=p_3d,
            tau=tau_a,
            calib=calib,
            tech_cfg=config.technology,
        )

        # Bilateral flows
        pp_col = p_vec[:, np.newaxis]
        ppfd_row = ppfd.reshape((1, nfd * nc), order="F")
        y_blocks = y_vec.reshape(nc, ns)
        a_blocks = a_2d.reshape(M, nc, ns)
        x_sum_c2 = np.sum(a_blocks * y_blocks[None, :, :], axis=2)
        val_sum = x_sum_c2 * pp_col
        T_inter = np.sum(val_sum.reshape(nc, ns, nc), axis=1)

        xc_blocks = xc_2d.reshape(M, nc, nfd)
        ppfd_blocks = ppfd_row.reshape(nc, nfd)
        fd_sum_c2 = np.sum(xc_blocks * ppfd_blocks[None, :, :], axis=2)
        T_fd = np.sum(fd_sum_c2.reshape(nc, ns, nc), axis=1)
        np.fill_diagonal(T_inter, 0.0)
        np.fill_diagonal(T_fd, 0.0)

        X0 = np.sum(T_inter, axis=1)
        M0 = np.sum(T_inter, axis=0)
        XFD = np.sum(T_fd, axis=1)
        MFD = np.sum(T_fd, axis=0)
        invforT_realized = X0 + XFD - M0 - MFD

        Tax_Total = np.sum(calib.tax * ytot, axis=1).ravel() + np.sum(Tax_c, axis=1).ravel()
        Tarifs_Totals = M0 * tauf_vec + MFD * tauf_fd_vec

        ff2 = calib.l_endow.ravel() - np.sum(xl, axis=1).ravel()
        ff3 = calib.k_endow.ravel() - np.sum(xk, axis=1).ravel()
        ff4 = XN - invforT_realized[:nc - 1]
        ff5 = T.ravel() - (Tax_Total + Tarifs_Totals)
        f_m = np.concatenate([ff2, ff3, ff4, ff5])

        if compute_full:
            ff0 = y_vec - (a_2d @ y_vec + d)
            v_P_val = c_va.flatten(order="F") / np.maximum(1.0 - tax_flat, 1e-12)
            ff1 = p_vec - (v_P_val + B_T_mat @ p_vec)
            f_full = np.concatenate([ff0, ff1, ff2, ff3, ff4, ff5])
        else:
            f_full = None

        return f_m, f_full, p_vec, y_vec, inner_iters

    f_m, f_full, p_sol, y_sol, inner_iters = eval_macro(xm, compute_full=True)
    max_res = float(np.max(np.abs(f_full)))
    diff = float(np.sum(np.abs(f_full)))

    if max_iter == 0:
        x_full = np.concatenate([np.log(np.maximum(p_sol, 1e-12)), np.log(np.maximum(y_sol, 1e-12)), xm])
        diag = compute_convergence_diagnostics(f_full, calib)
        diag["remediation"] = "Increase solver max_iter or reduce shock magnitude."
        return x_full, False, 0, max_res, diff, f_full, {"convergence_diagnostic": diag}

    if max_res <= tol:
        x_full = np.concatenate([np.log(np.maximum(p_sol, 1e-12)), np.log(np.maximum(y_sol, 1e-12)), xm])
        return x_full, True, 0, max_res, diff, f_full, {}

    # Damped Newton loop on macro state
    h = np.array([eps_fd if j < 2 * nc else eps_fd * max(abs(xm[j]), 1.0) for j in range(n_m)])

    for it in range(max_iter):
        J = np.empty((n_m, n_m), dtype=float)
        for j in range(n_m):
            xm_pert = xm.copy()
            xm_pert[j] += h[j]
            f_p, _, _, _, _ = eval_macro(xm_pert, p_warm=p_sol, compute_full=False)
            J[:, j] = (f_p - f_m) / h[j]

        c_norm = np.linalg.norm(J, axis=0)
        c_norm[c_norm == 0] = 1.0
        D_R = 1.0 / c_norm
        J_scaled = J * D_R[np.newaxis, :]
        r_norm = np.linalg.norm(J_scaled, axis=1)
        r_norm[r_norm == 0] = 1.0
        D_L = 1.0 / r_norm
        J_equil = D_L[:, np.newaxis] * J_scaled
        rhs_equil = -D_L * f_m

        try:
            sol_u = la.solve(J_equil, rhs_equil)
        except la.LinAlgError:
            sol_u = la.lstsq(J_equil, rhs_equil)[0]
        delta_m = D_R * sol_u

        max_factor_step = float(np.max(np.abs(delta_m[:2 * nc])))
        if max_factor_step > 0.3:
            delta_m *= (0.3 / max_factor_step)

        alpha_step = 1.0
        norm_0 = float(np.linalg.norm(D_L * f_m))
        accepted = False

        for _ in range(15):
            xm_trial = xm + alpha_step * delta_m
            xm_trial[:2 * nc] = np.clip(xm_trial[:2 * nc], -5.0, 5.0)
            f_t_m, f_t_full, p_t, y_t, _ = eval_macro(xm_trial, p_warm=p_sol, compute_full=True)
            norm_t = float(np.linalg.norm(D_L * f_t_m))
            res_t = float(np.max(np.abs(f_t_full)))

            if res_t <= tol or norm_t < norm_0:
                accepted = True
                xm, f_m, f_full, p_sol, y_sol = xm_trial, f_t_m, f_t_full, p_t, y_t
                break
            alpha_step *= 0.5

        if not accepted:
            xm = xm_trial
            f_m, f_full, p_sol, y_sol = f_t_m, f_t_full, p_t, y_t

        max_res = float(np.max(np.abs(f_full)))
        diff = float(np.sum(np.abs(f_full)))

        if max_res <= tol or diff <= tol:
            x_full = np.concatenate([np.log(np.maximum(p_sol, 1e-12)), np.log(np.maximum(y_sol, 1e-12)), xm])
            return x_full, True, it + 1, max_res, diff, f_full, {}

    x_full = np.concatenate([np.log(np.maximum(p_sol, 1e-12)), np.log(np.maximum(y_sol, 1e-12)), xm])
    diag = compute_convergence_diagnostics(f_full, calib)
    return x_full, False, max_iter, max_res, diff, f_full, {"convergence_diagnostic": diag}


TECH_FIELDS = {
    "rho_va",
    "sigma_va",
    "sigma_y",
    "sigma_inter",
    "capacity_margins",
    "penalty_scale",
    "penalty_exponent",
    "replicate_matlab_precedence",
}
PREF_FIELDS = {"subsistence_shares", "mu_s", "subsistence_ratio", "sigma_trade"}
MARKET_FIELDS = {
    "variable_markups",
    "sigma_j",
    "theta_j",
    "variety_condensation",
    "condense_varieties",
    "variety_expansion",
    "markup_min",
    "markup_max",
    "markup_bounds",
    "cournot_weights",
    "clamping_threshold",
    "fl",
    "fk",
}
MODEL_FIELDS = {"max_inner_iter", "technology", "preference", "market_structure"}
SOLVER_KWARGS = {
    "tau",
    "tau_fd",
    "tauf",
    "tauf_fd",
    "tau_a",
    "taufd_a",
    "tol",
    "max_iter",
    "verbose",
    "return_history",
    "damping",
    "method",
    "backend",
    "fiscal_closure",
    "recycling_params",
    "x0",
    "replicate_matlab_precedence",
    "base_result",
    "full_block",
    "eps_fd",
}


def _resolve_flexible_model_config(
    config: FlexibleTradeModelConfig | None = None,
    rho_va: float | None = None,
    sigma_y: float | None = None,
    **kwargs: Any,
) -> tuple[FlexibleTradeModelConfig, dict[str, Any]]:
    """Resolve and validate model configuration and solver keyword arguments.

    Partitions caller arguments across structural configuration layers
    (technology, preference, market structure, solver) and synchronizes
    bidirectional aliases before instantiation or dataclass replacement to
    prevent desynchronization.

    Parameters
    ----------
    config : FlexibleTradeModelConfig | None, default None
        Base configuration to update, or None to construct from defaults and kwargs.
    rho_va : float | None, optional
        Value-added substitution elasticity override.
    sigma_y : float | None, optional
        Gross output substitution elasticity override.
    **kwargs : Any
        Keyword parameters for model configurations or numerical solver settings.

    Returns
    -------
    resolved_config : FlexibleTradeModelConfig
        Unified declarative model configuration.
    solver_kwargs : dict[str, Any]
        Filtered keyword arguments destined for the numerical solver.

    Raises
    ------
    TypeError
        If unknown keyword arguments outside allowed parameter sets are encountered.
    ValueError
        If parameter values or conflicting alias pairs violate economic constraints.
    """
    all_allowed = TECH_FIELDS | PREF_FIELDS | MARKET_FIELDS | MODEL_FIELDS | SOLVER_KWARGS
    spurious = set(kwargs.keys()) - all_allowed
    if spurious:
        raise TypeError(f"Unknown keyword argument(s): {spurious}")

    tech_kwargs: dict[str, Any] = {}
    pref_kwargs: dict[str, Any] = {}
    mkt_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    solver_kwargs: dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in TECH_FIELDS:
            tech_kwargs[k] = v
        elif k in PREF_FIELDS:
            pref_kwargs[k] = v
        elif k in MARKET_FIELDS:
            mkt_kwargs[k] = v
        elif k in MODEL_FIELDS:
            model_kwargs[k] = v
        elif k in SOLVER_KWARGS:
            solver_kwargs[k] = v

    if rho_va is not None:
        tech_kwargs["rho_va"] = rho_va
    if sigma_y is not None:
        tech_kwargs["sigma_y"] = sigma_y

    # Synchronize technology aliases (rho_va <-> sigma_va)
    if "rho_va" in tech_kwargs and "sigma_va" in tech_kwargs:
        if tech_kwargs["rho_va"] != tech_kwargs["sigma_va"]:
            raise ValueError(
                f"Conflicting values provided for rho_va ({tech_kwargs['rho_va']}) "
                f"and sigma_va ({tech_kwargs['sigma_va']})."
            )
    elif "rho_va" in tech_kwargs:
        tech_kwargs["sigma_va"] = tech_kwargs["rho_va"]
    elif "sigma_va" in tech_kwargs:
        tech_kwargs["rho_va"] = tech_kwargs["sigma_va"]

    # Synchronize preference aliases (subsistence_shares <-> mu_s <-> subsistence_ratio)
    sub_keys = [k for k in ("subsistence_shares", "mu_s", "subsistence_ratio") if k in pref_kwargs]
    if sub_keys:
        first_val = pref_kwargs[sub_keys[0]]
        for k in sub_keys[1:]:
            other_val = pref_kwargs[k]
            if isinstance(first_val, np.ndarray) or isinstance(other_val, np.ndarray):
                if not np.array_equal(first_val, other_val):
                    raise ValueError("Conflicting values provided for subsistence parameters.")
            elif first_val != other_val:
                raise ValueError("Conflicting values provided for subsistence parameters.")
        pref_kwargs["subsistence_shares"] = first_val
        pref_kwargs["mu_s"] = first_val
        pref_kwargs["subsistence_ratio"] = first_val

    # Synchronize market structure aliases ((markup_min, markup_max) <-> markup_bounds)
    if "markup_bounds" in mkt_kwargs:
        mb = mkt_kwargs["markup_bounds"]
        if isinstance(mb, (tuple, list)) and len(mb) == 2:
            mkt_kwargs["markup_min"] = float(mb[0])
            mkt_kwargs["markup_max"] = float(mb[1])
            mkt_kwargs["markup_bounds"] = (float(mb[0]), float(mb[1]))
    elif "markup_min" in mkt_kwargs or "markup_max" in mkt_kwargs:
        cur_min = config.market_structure.markup_min if config is not None else 1.0
        cur_max = config.market_structure.markup_max if config is not None else 5.0
        b_min = float(mkt_kwargs.get("markup_min", cur_min))
        b_max = float(mkt_kwargs.get("markup_max", cur_max))
        mkt_kwargs["markup_min"] = b_min
        mkt_kwargs["markup_max"] = b_max
        mkt_kwargs["markup_bounds"] = (b_min, b_max)

    # Synchronize variety condensation aliases
    if any(k in mkt_kwargs for k in ("condense_varieties", "variety_condensation", "variety_expansion")):
        c_flag = bool(
            mkt_kwargs.get("condense_varieties")
            or mkt_kwargs.get("variety_condensation")
            or mkt_kwargs.get("variety_expansion")
        )
        mkt_kwargs["condense_varieties"] = c_flag
        mkt_kwargs["variety_condensation"] = c_flag
        mkt_kwargs["variety_expansion"] = c_flag

    if config is None:
        tech = model_kwargs.get("technology")
        if tech is None:
            tech = FlexibleTechnologyConfig(**tech_kwargs)
        elif tech_kwargs:
            tech = replace(tech, **tech_kwargs)

        pref = model_kwargs.get("preference")
        if pref is None:
            pref = FlexiblePreferenceConfig(**pref_kwargs)
        elif pref_kwargs:
            pref = replace(pref, **pref_kwargs)

        mkt = model_kwargs.get("market_structure")
        if mkt is None:
            mkt = FlexibleMarketStructureConfig(**mkt_kwargs)
        elif mkt_kwargs:
            mkt = replace(mkt, **mkt_kwargs)

        m_args: dict[str, Any] = {"technology": tech, "preference": pref, "market_structure": mkt}
        if "max_inner_iter" in model_kwargs:
            m_args["max_inner_iter"] = model_kwargs["max_inner_iter"]
        resolved_config = FlexibleTradeModelConfig(**m_args)
    else:
        new_tech = model_kwargs.get("technology", config.technology)
        if tech_kwargs:
            new_tech = replace(new_tech, **tech_kwargs)

        new_pref = model_kwargs.get("preference", config.preference)
        if pref_kwargs:
            new_pref = replace(new_pref, **pref_kwargs)

        new_mkt = model_kwargs.get("market_structure", config.market_structure)
        if mkt_kwargs:
            new_mkt = replace(new_mkt, **mkt_kwargs)

        r_args: dict[str, Any] = {
            "technology": new_tech,
            "preference": new_pref,
            "market_structure": new_mkt,
        }
        if "max_inner_iter" in model_kwargs:
            r_args["max_inner_iter"] = model_kwargs["max_inner_iter"]
        resolved_config = replace(config, **r_args)

    return resolved_config, solver_kwargs


def _build_markups_frame(
    x_sol: np.ndarray,
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig,
) -> pd.DataFrame:
    """Compute post-solve counterfactual markups DataFrame."""
    from puremacro.trade.equilibrium import unpack_equilibrium_vector

    nc = calib.n_countries
    ns = calib.n_sectors
    c_codes = list(calib.country_codes) if calib.country_codes else [f"C{i:02d}" for i in range(nc)]
    s_codes = list(calib.sector_codes) if calib.sector_codes else [f"S{i:02d}" for i in range(ns)]
    c_arr = np.array(c_codes)
    s_arr = np.array(s_codes)

    origins = np.repeat(c_arr, nc * ns)
    dests = np.tile(np.repeat(c_arr, ns), nc)
    sectors = np.tile(s_arr, nc * nc)

    if not config.market_structure.variable_markups:
        return pd.DataFrame({
            "origin": origins,
            "destination": dests,
            "sector": sectors,
            "markup": np.ones(nc * nc * ns, dtype=float),
        })

    uv = unpack_equilibrium_vector(x_sol, ns=ns, nc=nc, nfd=calib.n_final_demand)
    p_2d = uv.p[0]

    s_0 = compute_benchmark_market_shares(calib)
    sigma_trade = getattr(calib, "sigma", 5.0)
    sig_val = float(sigma_trade) if isinstance(sigma_trade, (int, float)) else float(np.mean(sigma_trade))
    exponent = 1.0 - sig_val
    p_term = (p_2d[:, :, None]) ** exponent
    shares_unnorm = s_0 * p_term
    sum_shares = np.sum(shares_unnorm, axis=1, keepdims=True)
    s_ni = np.divide(shares_unnorm, sum_shares, out=s_0.copy(), where=(sum_shares > 1e-12))

    mu_0 = compute_benchmark_markups(calib, config.market_structure)
    mu_ni, _ = compute_atkeson_burstein_markups(
        s_ni=s_ni, c_i=None, market_cfg=config.market_structure, mu_0=mu_0
    )
    mu_ods = np.transpose(mu_ni, (1, 2, 0))

    return pd.DataFrame({
        "origin": origins,
        "destination": dests,
        "sector": sectors,
        "markup": mu_ods.ravel(),
    })


def solve_flexible_trade_equilibrium(
    calib: TradeCalibrationResult,
    config: FlexibleTradeModelConfig | None = None,
    rho_va: float | None = None,
    sigma_y: float | None = None,
    **kwargs: Any,
) -> FlexibleTradeEquilibriumResult:
    """Solve multi-country multi-sector general equilibrium with flexible trade model extensions.

    High-level declarative solver supporting progressive disclosure: allows solving
    scenarios directly via keyword arguments (e.g. `rho_va=0.7`, `variable_markups=True`,
    `subsistence_ratio=0.2`) or via pre-configured `FlexibleTradeModelConfig` dataclasses.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters from :func:`calibrate_trade_model`.
    config : FlexibleTradeModelConfig | None, default None
        Declarative model configuration specification. If None, constructed from defaults
        and passed keyword overrides.
    rho_va : float | None, optional
        Value-added substitution elasticity between labor and capital.
    sigma_y : float | None, optional
        Gross output substitution elasticity between value-added and intermediate composite.
    **kwargs : Any
        Additional configuration overrides or numerical solver settings, including:
        - Technology: `sigma_va`, `sigma_inter`, `capacity_margins`
        - Preferences: `subsistence_shares`, `mu_s`, `subsistence_ratio`, `sigma_trade`
        - Market structure: `variable_markups`, `sigma_j`, `theta_j`, `markup_bounds`
        - Solver controls: `method`, `tol`, `max_iter`, `tau`, `tau_fd`, `x0`

    Returns
    -------
    FlexibleTradeEquilibriumResult
        Inspectable general equilibrium solution container.

    Examples
    --------
    >>> from puremacro.trade import calibrate_trade_model, solve_flexible_trade_equilibrium
    >>> calib = calibrate_trade_model(...)
    >>> res = solve_flexible_trade_equilibrium(calib, rho_va=0.7, variable_markups=True)
    >>> df_factors = res.factor_allocation_frame()
    >>> df_welfare = res.welfare_decomposition()
    """
    config, solver_kwargs = _resolve_flexible_model_config(
        config=config, rho_va=rho_va, sigma_y=sigma_y, **kwargs
    )

    max_iter = solver_kwargs.get("max_iter", 100)
    tol = solver_kwargs.get("tol", 2.5e-3)
    method = solver_kwargs.get("method")

    if max_iter == 0:
        x_init = solver_kwargs.get("x0")
        if x_init is None:
            from puremacro.trade.solver import build_initial_guess
            x_init = build_initial_guess(calib)
        else:
            x_init = np.asarray(x_init, dtype=float).ravel()
        from puremacro.trade.equilibrium import compute_equilibrium_residuals
        raw_res = compute_equilibrium_residuals(x_init, calib)
        diag = compute_convergence_diagnostics(raw_res, calib)
        diag["remediation"] = "Increase solver max_iter or reduce shock magnitude."
        df_markups = _build_markups_frame(x_init, calib, config)
        return FlexibleTradeEquilibriumResult(
            x_sol=x_init,
            converged=False,
            iterations=0,
            residual_norm=float(np.max(np.abs(raw_res))),
            residuals=raw_res,
            metadata={"config": config, "convergence_diagnostic": diag},
            markups=df_markups,
            calib=calib,
            config=config,
        )

    if method == "quasi_condensed":
        x_sol, conv, iters, max_res, diff, res_vec, meta = _quasi_condensed_solve(
            calib=calib,
            config=config,
            **solver_kwargs,
        )
        diag = meta.get("convergence_diagnostic", {})
        if not conv and not diag:
            diag = compute_convergence_diagnostics(res_vec, calib)
        meta["config"] = config
        df_markups = _build_markups_frame(x_sol, calib, config)
        return FlexibleTradeEquilibriumResult(
            x_sol=x_sol,
            converged=conv,
            iterations=iters,
            residual_norm=max_res,
            residuals=res_vec,
            metadata=meta,
            markups=df_markups,
            calib=calib,
            config=config,
        )

    # Dispatch to solve_trade_equilibrium with dynamic sparsity and flexible configs
    solve_kwargs = dict(solver_kwargs)
    solve_kwargs["sigma_y"] = float(config.technology.sigma_y)
    solve_kwargs["variable_markups"] = bool(config.market_structure.variable_markups)
    solve_kwargs["config"] = config

    res_base = solve_trade_equilibrium(calib, **solve_kwargs)

    diag: dict[str, Any] = {}
    if not res_base.converged:
        from puremacro.trade.equilibrium import compute_equilibrium_residuals
        raw_res = (
            res_base.residuals
            if res_base.residuals is not None
            else compute_equilibrium_residuals(res_base.x_sol, calib)
        )
        diag = compute_convergence_diagnostics(raw_res, calib)

    df_markups = _build_markups_frame(res_base.x_sol, calib, config)
    meta = dict(getattr(res_base, "metadata", {}))
    meta["config"] = config
    if not res_base.converged:
        meta["convergence_diagnostic"] = diag

    return FlexibleTradeEquilibriumResult(
        x_sol=res_base.x_sol,
        converged=res_base.converged,
        iterations=res_base.iterations,
        residual_norm=res_base.residual_norm,
        residuals=res_base.residuals,
        metadata=meta,
        markups=df_markups,
        calib=calib,
        config=config,
    )


__all__ = [
    # Declarative configurations
    "FlexibleTechnologyConfig",
    "FlexiblePreferenceConfig",
    "FlexibleMarketStructureConfig",
    "FlexibleTradeModelConfig",
    # Result containers
    "FlexibleTradeEquilibriumResult",
    "FlexibleEquilibriumResult",
    # Technology functions
    "compute_nested_ces_costs",
    "compute_nested_factor_demands",
    # Preference functions
    "compute_stone_geary_final_demand",
    "smooth_subsistence_scaling",
    # Market structure functions
    "compute_atkeson_burstein_markups",
    "compute_benchmark_market_shares",
    "compute_benchmark_markups",
    "compute_dixit_stiglitz_varieties",
    "compute_condensed_varieties",
    "compute_variety_price_scaling",
    # Diagnostics & Solvers
    "compute_convergence_diagnostics",
    "solve_flexible_trade_equilibrium",
]

