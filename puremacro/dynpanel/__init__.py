"""Dynamic panel GMM: Arellano-Bond and Blundell-Bond.

Provides:

- :func:`ab_gmm` — Arellano-Bond (1991) difference GMM.
- :func:`bb_gmm` — Blundell-Bond (1998) system GMM (extends ab_gmm with
  level-equation moment conditions).

Modern best-practice defaults are ON: two-step optimal weighting,
Windmeijer (2005) finite-sample SE correction, Roodman (2009) instrument
collapse, Hansen J overidentification test, Arellano-Bond AR(1)/AR(2)
serial-correlation tests, lag-window control.

References
----------
Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data: Monte Carlo evidence and an application to employment equations.
    Review of Economic Studies 58(2), 277-297.
Blundell, R. and Bond, S. (1998). Initial conditions and moment
    restrictions in dynamic panel data models. Journal of Econometrics
    87(1), 115-143.
Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators. Journal of Econometrics
    126, 25-51.
Roodman, D. (2009). How to do xtabond2: an introduction to difference and
    system GMM in Stata. Stata Journal 9(1), 86-136.
"""
from ._results import GMMResult
from .ab_gmm import ab_gmm
from .bb_gmm import bb_gmm

__all__ = ["GMMResult", "ab_gmm", "bb_gmm"]
