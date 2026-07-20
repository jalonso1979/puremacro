"""Regenerate puremacro/validation/goldens/state_space.json (dev-only; uses statsmodels).

The PACKAGE references are statsmodels' linear-Gaussian Kalman filter / RTS
smoother (``tsa.statespace.MLEModel``), an independent implementation of the
same recursions. We build the statsmodels models on the SAME system matrices
and the SAME seeded data as puremacro (imported from
``puremacro.validation.cases_state_space``) and pin a finite *known*
initialisation so the two priors are identical (no diffuse approximation).

Run from the package dir:  python tools/gen_validation_goldens_state_space.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels
from statsmodels.tsa.statespace.mlemodel import MLEModel

from puremacro.validation.cases_state_space import ar1_noise_demo, local_level_demo


class _UnivariateUC(MLEModel):
    """Univariate-observation unobserved-component model with FIXED matrices.

        alpha_{t+1} = phi * alpha_t + eta_t,   eta_t ~ N(0, q)
        y_t         = alpha_t + eps_t,         eps_t ~ N(0, h)

    All parameters are baked in (no estimation); ``smooth([])`` just runs the
    filter+smoother given a known finite initialisation.
    """

    def __init__(self, endog, phi: float, q: float, h: float, a0, P0):
        super().__init__(
            endog,
            k_states=1,
            k_posdef=1,
            initialization="known",
            constant=np.asarray(a0, dtype=float),
            stationary_cov=np.asarray(P0, dtype=float),
        )
        self["transition", 0, 0] = float(phi)
        self["design", 0, 0] = 1.0
        self["selection", 0, 0] = 1.0
        self["state_cov", 0, 0] = float(q)
        self["obs_cov", 0, 0] = float(h)

    @property
    def param_names(self):  # no free parameters
        return []

    @property
    def start_params(self):
        return np.array([])


def _fit(spec: dict):
    mod = _UnivariateUC(
        spec["y"], phi=spec["phi"], q=spec["q"], h=spec["h"],
        a0=spec["a0"], P0=spec["P0"],
    )
    return mod.smooth([])


def main() -> None:
    # --- Case 1: local-level filtered states + log-likelihood ---
    ll_spec = local_level_demo()
    res_ll = _fit(ll_spec)
    a_filt = np.asarray(res_ll.filtered_state, dtype=float).T.ravel()  # (T,)
    loglik = float(res_ll.llf)

    # --- Case 2: AR(1)+noise smoothed states + smoothed covariances ---
    ar_spec = ar1_noise_demo()
    res_ar = _fit(ar_spec)
    a_smooth = np.asarray(res_ar.smoothed_state, dtype=float).T.ravel()       # (T,)
    P_smooth = np.asarray(res_ar.smoothed_state_cov, dtype=float).T.ravel()   # (T,)

    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_state_space.py",
            "reference": (
                f"statsmodels {statsmodels.__version__} tsa.statespace.MLEModel.smooth() "
                "with initialization='known' on identical system matrices"
            ),
            "dgp": (
                "puremacro.validation.cases_state_space.local_level_demo "
                "(seed=20260531) and .ar1_noise_demo (seed=424242)"
            ),
            "regenerate": "python tools/gen_validation_goldens_state_space.py",
        },
        "filter_local_level_vs_statsmodels": {
            "a_filt": a_filt.tolist(),
            "loglik": loglik,
        },
        "smoother_ar1_vs_statsmodels": {
            "a_smooth": a_smooth.tolist(),
            "P_smooth": P_smooth.tolist(),
        },
    }

    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro"
        / "validation"
        / "goldens"
        / "state_space.json"
    )
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {dest}\n"
        f"  local_level: a_filt len {a_filt.shape[0]}, loglik {loglik:.6f}\n"
        f"  ar1_noise: a_smooth len {a_smooth.shape[0]}, P_smooth len {P_smooth.shape[0]}"
    )


if __name__ == "__main__":
    main()
