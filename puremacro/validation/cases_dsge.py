"""Validation cases for the ``dsge`` subsystem (linear RE solvers + state space).

Scope: independent, package-free checks of the two QZ rational-expectations
solvers (Klein 2000, Sims 2002 gensys) and the state-space construction that
feeds the Kalman likelihood. References are CLOSED-FORM (hand-solvable RE
models) or INTERNAL identities / cross-method agreement — never another
puremacro function, so there is no circularity, and no reference package is
imported (these are pure numpy/scipy linear algebra).

There are NO PACKAGE cases here, hence no ``goldens/dsge.json`` and no
reference-drift guard: every reference is recomputed live from a closed form
or an algebraic identity inside this module.

Mechanisms used
---------------
ANALYTICAL — a model with a known closed-form solution:
    * Klein:  x_t = a E_t x_{t+1} + b z_t with z_t AR(1, rho) has the unique
      stable policy  x_t = [b / (1 - a*rho)] z_t  and state law  z_{t+1}=rho z_t.
    * gensys: the same forward-looking block has stable generalised eigenvalue
      rho and unstable generalised eigenvalue 1/beta (Sims-shifted control row).
    * State space: an AR(1) observed without noise has an exact Gaussian
      log-likelihood; the Kalman recursion on ``make_state_space`` must match it.
INTERNAL — identities every valid solution must satisfy / cross-method:
    * Klein and gensys return the SAME generalised eigenvalues and the same
      dominant stable state-transition eigenvalue for one model.
    * gensys shock-impact formula  (Gamma0 - G Gamma1) Impact = Psi.
    * Blanchard-Kahn: under existence+uniqueness the number of stable
      generalised eigenvalues equals the number of predetermined variables.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# ---------------------------------------------------------------------------
# Seeded demo data (kept LOCAL to this module — the shared _fixtures.py is not
# touched). These build the puremacro input and its closed-form reference from
# byte-identical primitives.
# ---------------------------------------------------------------------------
def _fwd_params() -> dict:
    """Primitives of the hand-solvable forward-looking model.

    Structural equation (single forward-looking control x, single AR(1) state z):
        x_t = a * E_t[x_{t+1}] + b * z_t,     z_{t+1} = rho * z_t + eps_t.
    Closed-form stable solution:  x_t = b / (1 - a*rho) * z_t.
    Requires |a*rho| < 1 (here 0.5*0.6 = 0.3) for the geometric sum to converge.
    """
    return {"a": 0.5, "b": 0.8, "rho": 0.6, "beta": 0.8}


def _klein_forward_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Klein form A E_t[z_{t+1}] = B z_t + C eps for the forward model.

    z = [z (predetermined); x (forward control)].
      row 0 (state law):     z_{t+1} = rho z_t + eps
      row 1 (Euler, t-form): a x_{t+1} = x_t - b z_t   (no z_{t+1} term)
    """
    p = _fwd_params()
    a, b, rho = p["a"], p["b"], p["rho"]
    A = np.array([[1.0, 0.0], [0.0, a]])
    B = np.array([[rho, 0.0], [-b, 1.0]])
    C = np.array([[1.0], [0.0]])
    return A, B, C


def _gensys_forward_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """gensys form Gamma0 z_t = Gamma1 z_{t-1} + Psi eps + Pi eta.

    Same model, Sims convention (eta_t = z_t - E_{t-1} z_t). The control's
    Euler equation is shifted back one period so the lead becomes an
    expectation error on the forward variable (cf. Sims 2002):
        a x_t = x_{t-1} - b z_{t-1} + a eta_t.
    Stable gen-eigenvalue = rho ; unstable gen-eigenvalue = 1/a.
    """
    p = _fwd_params()
    a, b, rho = p["a"], p["b"], p["rho"]
    Gamma0 = np.array([[1.0, 0.0], [0.0, a]])
    Gamma1 = np.array([[rho, 0.0], [-b, 1.0]])
    Psi = np.array([[1.0], [0.0]])
    Pi = np.array([[0.0], [a]])
    return Gamma0, Gamma1, Psi, Pi


def _ar1_demo_data() -> dict:
    """Stationary AR(1) sample observed without measurement noise.

    x_t = rho x_{t-1} + sigma eps_t ; y_t = x_t. Drawn from the stationary
    distribution (x_0 ~ N(0, sigma^2/(1-rho^2))) so the exact Gaussian
    likelihood below has no initial-condition slack. Seeded → identical
    every call.
    """
    rho, sigma, T = 0.7, 0.5, 200
    rng = np.random.default_rng(20260531)
    x = np.zeros(T)
    x[0] = rng.standard_normal() * sigma / np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + sigma * rng.standard_normal()
    return {"y": x.reshape(-1, 1), "rho": rho, "sigma": sigma, "T": T}


# ---------------------------------------------------------------------------
# Compute callables (puremacro). Imports are INSIDE so module import stays
# pyodide-light and never pulls scipy/statsmodels at top level.
# ---------------------------------------------------------------------------
def _klein_forward_policy() -> dict:
    from puremacro.dsge.klein import klein_solve

    A, B, C = _klein_forward_matrices()
    sol = klein_solve(A, B, n_pre=1, C=C)
    # F is the policy loading x = F z; G is the state law z_{t+1} = G z_t.
    return {
        "eu_sum": float(sum(sol.eu)),       # 2.0 ⇒ unique stable solution
        "policy_loading": float(sol.F[0, 0]),
        "state_persistence": float(sol.G[0, 0]),
    }


def _gensys_forward_eigs() -> dict:
    from puremacro.dsge.gensys import gensys

    G0, G1, Psi, Pi = _gensys_forward_matrices()
    sol = gensys(G0, G1, Psi, Pi)
    eigs = np.sort(np.asarray(sol.eigenvalues, dtype=float))
    return {
        "eu_sum": float(sum(sol.eu)),
        "stable_eig": float(eigs[0]),
        "unstable_eig": float(eigs[1]),
    }


def _klein_gensys_cross() -> dict:
    """Solve ONE model with both solvers; compare invariant spectra."""
    from puremacro.dsge.gensys import gensys
    from puremacro.dsge.klein import klein_solve

    Ak, Bk, Ck = _klein_forward_matrices()
    sk = klein_solve(Ak, Bk, n_pre=1, C=Ck)
    G0, G1, Psi, Pi = _gensys_forward_matrices()
    sg = gensys(G0, G1, Psi, Pi)

    klein_geneigs = np.sort(np.abs(np.asarray(sk.eigenvalues, dtype=float)))
    gensys_geneigs = np.sort(np.abs(np.asarray(sg.eigenvalues, dtype=float)))
    klein_state_eig = float(np.max(np.abs(np.linalg.eigvals(sk.G))))
    gensys_state_eig = float(np.max(np.abs(np.linalg.eigvals(sg.G))))
    return {
        # gensys gen-eigs reported relative to klein's (cross-method residual = 0).
        "geneig_diff": (gensys_geneigs - klein_geneigs),
        "state_eig_diff": gensys_state_eig - klein_state_eig,
    }


def _gensys_impact_identity() -> dict:
    """Gamma0 Impact - Psi must lie in the column space of Pi.

    This case used to assert `(Gamma0 - G Gamma1) Impact = Psi`, and its own
    citation showed why that was wrong: it wrote the model as
    `Gamma0 z_t = Gamma1 z_{t-1} + Psi eps_t`, dropping the `Pi eta_t` term.
    The identity follows correctly from that truncated system and not from the
    one gensys actually solves, so the case validated the shipped `Impact`
    formula against a restatement of itself.

    With the expectation errors kept, substituting z_t = G z_{t-1} + Impact eps
    and matching the eps terms gives `Gamma0 Impact = Psi + Pi N`, where N is
    the loading of eta on eps. N is not known here, so the testable content is
    that `Gamma0 Impact - Psi` lies in the column space of Pi -- which the
    corrected solver satisfies and the old one does not.
    """
    from puremacro.dsge.gensys import gensys

    # A 2-shock forward model (Sims-shifted control), distinct from the
    # 1-shock analytical model above, to exercise the multi-shock path.
    G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
    G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
    Psi = np.array([[1.0, 0.5], [0.0, 0.3]])
    Pi = np.array([[0.0], [0.8]])
    sol = gensys(G0, G1, Psi, Pi)
    target = G0 @ sol.Impact - Psi
    N, *_ = np.linalg.lstsq(Pi, target, rcond=None)
    resid = target - Pi @ N
    return {"eu_sum": float(sum(sol.eu)), "impact_residual": resid.ravel()}


def _statespace_kalman_loglik() -> dict:
    """make_state_space(AR1) → Kalman log-likelihood at the stationary prior."""
    from puremacro.examples.dsge_ar1_demo import make_state_space
    from puremacro.state_space import kalman_filter

    d = _ar1_demo_data()
    rho, sigma = d["rho"], d["sigma"]
    ssm = make_state_space({"rho": rho, "sigma": sigma})
    P0 = np.array([[sigma ** 2 / (1.0 - rho ** 2)]])  # stationary state variance
    res = kalman_filter(d["y"], ssm, a0=np.zeros(1), P0=P0)
    return {"loglik": float(res["loglik"])}


def _blanchard_kahn_count() -> dict:
    """Under eu=(1,1), #stable generalised eigenvalues == #predetermined vars.

    Checked on TWO independent models: the 2-var analytical model (n_pre=1)
    and the bundled linearised RBC (n_pre=2).
    """
    from puremacro.dsge.klein import klein_solve
    from puremacro.examples.dsge_rbc_klein import _rbc_matrices

    Ak, Bk, Ck = _klein_forward_matrices()
    s1 = klein_solve(Ak, Bk, n_pre=1, C=Ck)
    n1 = int(np.sum(np.abs(np.asarray(s1.eigenvalues, dtype=float)) < 1.0))

    A, B, C, _ = _rbc_matrices()
    s2 = klein_solve(A, B, n_pre=2, C=C)
    n2 = int(np.sum(np.abs(np.asarray(s2.eigenvalues, dtype=float)) < 1.0))

    return {
        "eu_sums": np.array([float(sum(s1.eu)), float(sum(s2.eu))]),
        "n_stable": np.array([float(n1), float(n2)]),
    }


# ---------------------------------------------------------------------------
# Closed-form references (independent of puremacro — no circularity).
# ---------------------------------------------------------------------------
def _ref_klein_forward_policy() -> dict:
    p = _fwd_params()
    a, b, rho = p["a"], p["b"], p["rho"]
    return {
        "eu_sum": 2.0,
        "policy_loading": b / (1.0 - a * rho),  # geometric-sum solution
        "state_persistence": rho,
    }


def _ref_gensys_forward_eigs() -> dict:
    p = _fwd_params()
    a, rho = p["a"], p["rho"]
    return {"eu_sum": 2.0, "stable_eig": rho, "unstable_eig": 1.0 / a}


def _ref_ar1_loglik() -> dict:
    """Exact Gaussian log-likelihood of a stationary AR(1) observed w/o noise.

    y_1 ~ N(0, s2/(1-rho^2)); y_t | y_{t-1} ~ N(rho y_{t-1}, s2). This is a
    standard textbook result (Hamilton 1994, eq. 5.2.9) and does not call any
    puremacro code.
    """
    d = _ar1_demo_data()
    y = d["y"][:, 0]
    rho, sigma, T = d["rho"], d["sigma"], d["T"]
    s2 = sigma ** 2
    v0 = s2 / (1.0 - rho ** 2)
    ll = -0.5 * (np.log(2.0 * np.pi * v0) + y[0] ** 2 / v0)
    for t in range(1, T):
        ll += -0.5 * (np.log(2.0 * np.pi * s2) + (y[t] - rho * y[t - 1]) ** 2 / s2)
    return {"loglik": float(ll)}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="dsge.klein_forward_policy_loading",
        subsystem="dsge",
        title="Klein QZ policy loading matches the closed-form RE solution",
        title_es="La carga de política QZ de Klein coincide con la solución de ER en forma cerrada",
        mechanism=Mechanism.ANALYTICAL,
        compute=_klein_forward_policy,
        reference=_ref_klein_forward_policy,
        tol=Tol.TIGHT,
        citation=(
            "Closed-form rational-expectations solution of x_t = a E_t x_{t+1} + b z_t "
            "with AR(1) z (rho): x_t = b/(1-a*rho) z_t. Solver: Klein, P. (2000), "
            "'Using the generalised Schur form to solve a multivariate linear rational "
            "expectations model', JEDC 24, 1405-1423."
        ),
        notes=(
            "F and the state law G are basis-invariant; the QZ shock-loading N is "
            "left unchecked (its iid-impact formula is approximate for this layout)."
        ),
    ),
    ValidationCase(
        id="dsge.gensys_forward_eigenvalues",
        subsystem="dsge",
        title="gensys generalised eigenvalues equal rho and 1/beta",
        title_es="Los autovalores generalizados de gensys son iguales a rho y 1/beta",
        mechanism=Mechanism.ANALYTICAL,
        compute=_gensys_forward_eigs,
        reference=_ref_gensys_forward_eigs,
        tol=Tol.TIGHT,
        citation=(
            "Forward-looking RE block has stable generalised eigenvalue rho and "
            "unstable 1/a (Blanchard-Kahn classification). Solver: Sims, C. (2002), "
            "'Solving linear rational expectations models', Comp. Econ. 20, 1-20."
        ),
    ),
    ValidationCase(
        id="dsge.klein_gensys_cross_method",
        subsystem="dsge",
        title="Klein and gensys agree on the spectrum of one RE model",
        title_es="Klein y gensys coinciden en el espectro de un mismo modelo de ER",
        mechanism=Mechanism.INTERNAL,
        compute=_klein_gensys_cross,
        reference=lambda: {
            "geneig_diff": np.zeros(2),
            "state_eig_diff": 0.0,
        },
        tol=Tol.NUMERIC,
        citation=(
            "Cross-method identity: two distinct QZ solvers (Klein 2000; Sims 2002 "
            "gensys) of the same linear RE model must return identical generalised "
            "eigenvalues and the same dominant stable state-transition eigenvalue."
        ),
    ),
    ValidationCase(
        id="dsge.gensys_shock_impact_identity",
        subsystem="dsge",
        title="gensys impact obeys Gamma0 Impact - Psi in col(Pi)",
        title_es="El impacto de gensys cumple Gamma0 Impact - Psi en col(Pi)",
        mechanism=Mechanism.INTERNAL,
        compute=_gensys_impact_identity,
        reference=lambda: {"eu_sum": 2.0, "impact_residual": np.zeros(2 * 2)},
        tol=Tol.EXACT,
        citation=(
            "Shock-impact defining identity of the Sims (2002) solution form "
            "z_t = G z_{t-1} + Impact eps_t: substituting into the FULL system "
            "Gamma0 z_t = Gamma1 z_{t-1} + Psi eps_t + Pi eta_t and matching the "
            "eps terms gives Gamma0 Impact = Psi + Pi N, so Gamma0 Impact - Psi "
            "lies in the column space of Pi. The earlier form of this case "
            "asserted (Gamma0 - G Gamma1) Impact = Psi, which follows only if "
            "the Pi eta_t term is dropped from the model."
        ),
    ),
    ValidationCase(
        id="dsge.statespace_kalman_loglik_ar1",
        subsystem="dsge",
        title="Kalman log-likelihood of make_state_space(AR1) matches the exact AR(1) likelihood",
        title_es="La verosimilitud de Kalman de make_state_space(AR1) coincide con la verosimilitud exacta del AR(1)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_statespace_kalman_loglik,
        reference=_ref_ar1_loglik,
        tol=Tol.NUMERIC,
        citation=(
            "Exact Gaussian likelihood of a stationary AR(1) observed without noise "
            "(Hamilton, J. (1994), Time Series Analysis, eq. 5.2.9). Validates the "
            "state-space construction puremacro.examples.dsge_ar1_demo.make_state_space "
            "fed through puremacro.state_space.kalman_filter."
        ),
        notes=(
            "Residual ~7e-7 from the demo model's H=1e-8 measurement-noise ridge; "
            "well inside NUMERIC tolerance."
        ),
    ),
    ValidationCase(
        id="dsge.blanchard_kahn_stable_count",
        subsystem="dsge",
        title="Stable generalised-eigenvalue count equals the number of predetermined states",
        title_es="El número de autovalores generalizados estables es igual al número de estados predeterminados",
        mechanism=Mechanism.INTERNAL,
        compute=_blanchard_kahn_count,
        reference=lambda: {
            "eu_sums": np.array([2.0, 2.0]),         # both models: unique stable solution
            "n_stable": np.array([1.0, 2.0]),        # n_pre = 1 (analytical), 2 (RBC)
        },
        tol=Tol.EXACT,
        citation=(
            "Blanchard, O. and Kahn, C. (1980), 'The solution of linear difference "
            "models under rational expectations', Econometrica 48, 1305-1311: a unique "
            "stable solution requires exactly as many stable (inside-unit-circle) "
            "generalised eigenvalues as predetermined variables."
        ),
    ),
]
