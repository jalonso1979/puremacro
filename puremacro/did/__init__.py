"""Modern staggered Difference-in-Differences estimators.

The headline estimators in the post-2020 literature, all in one place:

  ``callaway_santanna``        — Callaway-Sant'Anna 2021 ATT(g, t)
                                  with not-yet-treated / never-treated controls.
  ``sun_abraham``              — Sun-Abraham 2021 cohort-share-weighted
                                  event-study (a CS aggregation).
  ``borusyak_jaravel_spiess``  — BJS 2022 imputation estimator.
  ``synthetic_did``            — Arkhangelsky-Athey-Hirshberg-Imbens-
                                  Wager 2021 SDID (single-cohort).
  ``cdh_did``                  — de Chaisemartin-D'Haultfoeuille 2020
                                  DID_M / DID_M^l (switchers estimator).
  ``sdid_multi_cohort``        — Multi-cohort SDID aggregation
                                  (wraps ``synthetic_did``).

The CS / SA / BJS / SDID estimators take a long-format DataFrame with
columns ``(unit, time, outcome, treat_time)`` where ``treat_time`` is
the first-treatment period for a unit (NaN for never-treated controls).
CS and SA select the comparison group with ``control=`` (``"never_treated"``
or ``"not_yet_treated"``; ``control_group=`` is accepted as an alias).

The CdH and multi-cohort SDID estimators take **four 1-D arrays**
``(y, treatment, panel_id, time_id)`` instead — see their docstrings.

Every result dataclass exposes the same presentation contract:
``summary()``, ``to_frame()``, ``to_markdown()``, ``to_latex()``,
``to_typst()`` and ``plot()``.
"""

from .types import PanelDiD
from .callaway_santanna import callaway_santanna
from .sun_abraham import sun_abraham
from .borusyak_jaravel_spiess import borusyak_jaravel_spiess
from .synthetic_did import synthetic_did
from .cdh import cdh_did
from .sdid_multi import sdid_multi_cohort
from .sensitivity import honest_did, honest_did_sensitivity, HonestDiDResult
from ._results import (
    CallawaySantannaResult,
    SunAbrahamResult,
    BorusyakJaravelSpiessResult,
    SyntheticDiDResult,
    CdHResult,
    SDIDMultiResult,
)

__all__ = [
    "PanelDiD",
    "callaway_santanna",
    "sun_abraham",
    "borusyak_jaravel_spiess",
    "synthetic_did",
    "cdh_did",
    "sdid_multi_cohort",
    "honest_did",
    "honest_did_sensitivity",
    # Result dataclasses
    "CallawaySantannaResult",
    "SunAbrahamResult",
    "BorusyakJaravelSpiessResult",
    "SyntheticDiDResult",
    "CdHResult",
    "SDIDMultiResult",
    "HonestDiDResult",
]
