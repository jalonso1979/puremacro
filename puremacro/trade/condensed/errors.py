"""Exception taxonomy of the condensed one-factor Leontief tariff model.

Every failure of the condensed path raises one of these classes. Drivers never
catch and continue, and nothing is reported as ``passed`` without a passing
certificate (a result solved with ``certify=False`` is ``converged`` but
neither ``passed`` nor ``certified``). Bad input raises a ``ValueError``
subclass; numerical failure raises a :class:`CondensedSolveError` (a
``RuntimeError`` subclass) that carries the last state the solver reached.
"""
from __future__ import annotations

from typing import Any


class CondensedModelError(Exception):
    """Base class for every error raised by :mod:`puremacro.trade.condensed`."""


class DataIntegrityError(CondensedModelError, ValueError):
    """The input table has inconsistent shapes, codes or accounting identities."""


class CalibrationError(CondensedModelError, ValueError):
    """A calibration gate failed (non-finite values, broken accounts, invalid shares)."""


class UnsupportedExtension(CondensedModelError, ValueError):
    """An option is outside the condensed model (for example tariffs on purchases abroad)."""


class CondensedSolveError(CondensedModelError, RuntimeError):
    """Base class of the numerical failures of the condensed solvers.

    Parameters
    ----------
    message : str
        Description of the failure.
    z_last : array-like, optional
        The last unknown vector ``(log w, Y/Y0)`` the solver evaluated.
    residual_last : float, optional
        The maximum scaled residual at ``z_last``.
    """

    def __init__(self, message: str, *, z_last: Any = None, residual_last: float | None = None) -> None:
        super().__init__(message)
        self.z_last = None if z_last is None else z_last
        self.residual_last = None if residual_last is None else float(residual_last)


class ExistenceViolation(CondensedSolveError):
    """A nonnegative matrix has a certified spectral lower bound of at least one."""


class ProductivityUncertified(CondensedSolveError):
    """The available numerical bounds do not certify ``rho < 1 - margin`` and do not reject.

    Two situations, told apart by the message: the bounds certify ``rho < 1``
    but not the safety margin ("productivity margin not met"), or they certify
    nothing ("productivity is inconclusive"). Neither is a claim of
    nonexistence. The gate fails closed in both cases.

    Parameters
    ----------
    message : str
        Description of the outcome.
    lower, upper : float
        Best certified bounds on the spectral radius (``upper`` may be ``inf``).
    margin : float
        The required gap below one.
    """

    def __init__(self, message: str, *, lower: float = float("nan"), upper: float = float("inf"),
                 margin: float = float("nan")) -> None:
        super().__init__(message)
        self.lower = float(lower)
        self.upper = float(upper)
        self.margin = float(margin)


class EquilibriumNotFound(CondensedSolveError):
    """Newton did not reach the stopping rule, or the line search found no descent."""


class ContinuationFailure(CondensedSolveError):
    """Continuation in the tariff scale could not reach ``lambda = 1``.

    Parameters
    ----------
    message : str
        Description of the failure.
    lam_reached : float
        Largest tariff scale at which an equilibrium was accepted.
    z_last : array-like, optional
        The unknown vector accepted at ``lam_reached``.
    """

    def __init__(self, message: str, lam_reached: float, *, z_last: Any = None) -> None:
        super().__init__(f"{message} (lambda reached: {lam_reached:.6g})", z_last=z_last)
        self.lam_reached = float(lam_reached)


class FoldDetected(CondensedSolveError):
    """Pseudo-arclength continuation could not continue regularly before ``lambda = 1``.

    Raised when ``dlambda/ds`` turns negative or ``det J_z`` changes sign: a
    turning point, or an existence boundary where the Jacobian becomes
    numerically singular (the only economic case in the tests is such a
    boundary, not a turning point).

    Parameters
    ----------
    lam : float
        Tariff scale at which the sign change was detected.
    sigma_min : float
        Smallest singular value of the Jacobian there, relative to the largest.
    """

    def __init__(self, lam: float, sigma_min: float, *, z_last: Any = None) -> None:
        super().__init__(
            f"fold detected at lambda={lam:.6g} (sigma_min/sigma_max={sigma_min:.3e})", z_last=z_last
        )
        self.lam = float(lam)
        self.sigma_min = float(sigma_min)


class MultipleEquilibria(CondensedSolveError):
    """A second admissible root was found by the near-start multistart.

    Parameters
    ----------
    message : str
        Description of the finding.
    roots : list
        Every distinct admissible root found, as arrays of unknowns.
    """

    def __init__(self, message: str, roots: list[Any]) -> None:
        super().__init__(message)
        self.roots = list(roots)


class CertificationFailure(CondensedSolveError):
    """The independent raw-flow certificate exceeded its tolerance in at least one block.

    Parameters
    ----------
    message : str
        Description of the failure.
    blocks : dict
        Every certificate block with its maximum scaled violation.
    """

    def __init__(self, message: str, blocks: dict[str, float]) -> None:
        super().__init__(message)
        self.blocks = dict(blocks)


__all__ = [
    "CalibrationError",
    "CertificationFailure",
    "CondensedModelError",
    "CondensedSolveError",
    "ContinuationFailure",
    "DataIntegrityError",
    "EquilibriumNotFound",
    "ExistenceViolation",
    "FoldDetected",
    "MultipleEquilibria",
    "ProductivityUncertified",
    "UnsupportedExtension",
]
