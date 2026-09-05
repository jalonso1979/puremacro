"""SVAR identification methods."""
from .cholesky import cholesky_svar as cholesky, compute_chol_shocks
from .bq import bq_svar as bq
from .sign import sign_restriction_svar as sign_restrictions
from .narrative_sign import identify_narrative_sign, narrative_sign_svar, NarrativeRestriction
from .proxy import proxy_svar as proxy, proxy_svar
from .hetero import rigobon_svar as hetero, HeteroResult
from .maxshare import maxshare, news_maxshare, identify_maxshare
from .sign_zero import sign_zero
from .sign_robust import gk_robust_bands, gk_robust_bands_from_gibbs
from .non_gaussian import non_gaussian_svar
from .magmav import magmav_svar
from .panel import mean_group_svar
from ._results import (
    ProxySVARResult,
    CholeskySVARResult,
    BQSVARResult,
    SignRestrictionResult,
    NarrativeSignResult,
    NarrativeSignSVARResult,
    GKRobustBandsResult,
    NonGaussianSVARResult,
    SignZeroResult,
    PanelSVARResult,
    MaxShareResult,
    MagMavSVARResult,
)

__all__ = [
    "cholesky", "compute_chol_shocks", "bq", "sign_restrictions", "proxy", "hetero",
    "identify_narrative_sign", "narrative_sign_svar", "NarrativeRestriction",
    "NarrativeSignResult", "NarrativeSignSVARResult",
    "maxshare", "news_maxshare", "identify_maxshare", "sign_zero", "gk_robust_bands",
    "gk_robust_bands_from_gibbs", "non_gaussian_svar", "magmav_svar", "mean_group_svar",
    "ProxySVARResult", "CholeskySVARResult", "BQSVARResult",
    "SignRestrictionResult", "GKRobustBandsResult",
    "NonGaussianSVARResult", "SignZeroResult", "HeteroResult", "PanelSVARResult",
    "MaxShareResult", "MagMavSVARResult",
]
