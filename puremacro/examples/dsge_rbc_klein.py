"""Linearised RBC model solved with the Klein (2000) QZ method.

A bare-bones one-shock RBC: log-linearised consumption Euler equation
plus capital accumulation. The state is detrended capital
``k_hat_t``; the control is consumption ``c_hat_t``. With log
preferences and full depreciation (delta=1) the policy function is
known in closed form (c_t = (1 - alpha * beta) * y_t, k_{t+1} = alpha
* beta * y_t), which gives us a check on Klein's output.

We use a richer specification (delta < 1) so the policy function is
non-trivial, and demonstrate that Klein's QZ solver returns a stable
state-space form ready to feed into ``puremacro.state_space``.

Run:
    python -m puremacro.examples.dsge_rbc_klein

Español
-------
Modelo RBC linealizado resuelto con el método QZ de Klein (2000).

Un modelo RBC de un solo choque en su versión más básica: ecuación de
Euler del consumo log-linealizada más acumulación de capital. El estado es el
capital en desviación de la tendencia ``k_hat_t``; el control es el consumo
``c_hat_t``. Con preferencias logarítmicas y depreciación total (delta=1), la
función de política se conoce en forma cerrada (c_t = (1 - alpha * beta) * y_t,
k_{t+1} = alpha * beta * y_t), lo que sirve de verificación de los resultados
de Klein.

Se utiliza una especificación más rica (delta < 1) para que la función de
política sea no trivial, y se demuestra que el solver QZ de Klein devuelve una
forma espacio-estado estable lista para alimentar ``puremacro.state_space``.

Ejecución:
    python -m puremacro.examples.dsge_rbc_klein
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dsge import klein_solve


def _rbc_matrices(beta=0.99, alpha=0.36, delta=0.025, sigma=1.0,
                   rho_z=0.95):
    """Linearised one-sector RBC around steady state.

    Variables (in deviations from log steady state):
        x = [k_hat, z_hat]  (predetermined: k inherited; z exogenous AR(1))
        y = [c_hat]         (forward-looking control)

    The log-linearised system in Klein form is
        A * E_t [x_{t+1}; y_{t+1}] = B * [x_t; y_t] + C * eps_t.
    """
    # Steady-state ratios (well-known closed forms for RBC)
    r_ss = 1.0 / beta - 1.0  # net rate
    # marginal product of capital condition: alpha * (k/y)^(-1) = r_ss + delta
    ky = alpha / (r_ss + delta)
    # consumption-output ratio: c/y = 1 - delta * k/y
    cy = 1.0 - delta * ky
    # Investment-output: i/y = delta * k/y

    # Log-linear equations:
    # 1) Capital accumulation:
    #    k_hat_{t+1} = (1 - delta) * k_hat_t + delta * i_hat_t
    #    with i_hat_t = (1/(i/y)) (y_hat_t - (c/y) c_hat_t)
    #    and y_hat_t = alpha * k_hat_t + z_hat_t.
    # 2) Productivity AR(1): z_hat_{t+1} = rho_z * z_hat_t + eps_t.
    # 3) Consumption Euler:
    #    sigma * (c_hat_{t+1} - c_hat_t) = (1 - beta*(1-delta)) *
    #      (alpha-1) * k_hat_{t+1} + (1 - beta*(1-delta)) * z_hat_{t+1}
    #    [Hansen RBC log-linearisation]
    iy = delta * ky
    # equation 1: k_hat_{t+1} - (1-delta) k_hat_t - (delta/iy) [alpha k_hat_t + z_hat_t - cy c_hat_t] = 0
    a_kk = (delta / iy) * alpha + (1 - delta)
    a_kz = delta / iy
    a_kc = -(delta / iy) * cy
    # In Klein form A z_{t+1} = B z_t.
    # Row 1: k_{t+1} = a_kk k_t + a_kz z_t + a_kc c_t
    # Row 2: z_{t+1} = rho_z z_t
    # Row 3: c_{t+1} - c_t = (1/sigma) * (1 - beta*(1-delta)) *
    #     ((alpha-1) k_{t+1} + z_{t+1})
    #     ⇒  c_{t+1} - (kappa*(alpha-1)) k_{t+1} - kappa z_{t+1} = c_t
    kappa = (1.0 - beta * (1.0 - delta)) / sigma
    A = np.array([
        [1.0, 0.0, 0.0],                      # k_{t+1}
        [0.0, 1.0, 0.0],                      # z_{t+1}
        [-kappa * (alpha - 1.0), -kappa, 1.0],  # c_{t+1}
    ])
    B = np.array([
        [a_kk, a_kz, a_kc],
        [0.0, rho_z, 0.0],
        [0.0, 0.0, 1.0],
    ])
    C = np.array([[0.0], [1.0], [0.0]])  # productivity shock loads on z eq.
    return A, B, C, dict(beta=beta, alpha=alpha, delta=delta, rho_z=rho_z,
                         ky=ky, cy=cy)


def run_demo() -> dict:
    A, B, C, params = _rbc_matrices()
    sol = klein_solve(A, B, n_pre=2, C=C)
    # Simulate
    T = 200
    rng = np.random.default_rng(0)
    x = np.zeros((T, 2))
    y = np.zeros((T, 1))
    eps = rng.standard_normal(T) * 0.01
    for t in range(1, T):
        x[t] = sol.G @ x[t - 1] + sol.N @ np.array([eps[t]])
        y[t] = sol.F @ x[t]
    return {"sol": sol, "x_path": x, "y_path": y, "params": params}


def main() -> None:
    out = run_demo()
    sol = out["sol"]
    print("Linearised RBC model solved by Klein (2000) QZ method")
    print(f"  Existence / uniqueness flags: {sol.eu}")
    print(f"  Generalised eigenvalues |.|:  "
          f"{np.array2string(np.abs(sol.eigenvalues), precision=3)}")
    print(f"  → {sum(np.abs(sol.eigenvalues) < 1)} stable, "
          f"{sum(np.abs(sol.eigenvalues) >= 1)} unstable "
          f"(needed exactly 1 unstable for unique sol.)")
    print()
    print(f"  G (state transition):\n{np.round(sol.G, 4)}")
    print(f"  F (policy y = F x):\n{np.round(sol.F, 4)}")
    print(f"  N (state shock loading):\n{np.round(sol.N, 4)}")
    print(f"  L (control shock loading):\n{np.round(sol.L, 4)}")
    print()
    print(f"  Simulated path: var(k_hat) = {np.var(out['x_path'][:, 0]):.4f}, "
          f"var(z_hat) = {np.var(out['x_path'][:, 1]):.4f}, "
          f"var(c_hat) = {np.var(out['y_path']):.4f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        x = out["x_path"]; y = out["y_path"]
        fig, ax = plt.subplots(figsize=(7, 3.0))
        ax.plot(x[:, 0], color="0.0", lw=0.9, label="$\\hat k_t$ (capital)")
        ax.plot(x[:, 1], color="0.4", lw=0.7, ls="--",
                label="$\\hat z_t$ (productivity)")
        ax.plot(y[:, 0], color="0.6", lw=0.7, ls=":",
                label="$\\hat c_t$ (consumption)")
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.set_xlabel("t")
        ax.set_title("RBC simulation from Klein QZ policy function")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "dsge_rbc_klein.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'dsge_rbc_klein.png'}")


if __name__ == "__main__":
    main()
