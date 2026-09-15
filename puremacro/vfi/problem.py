"""Public API for the general discrete VFI engine: VFIProblem / VFISolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import pandas as pd

from puremacro import _backend as _bk
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.vfi.returnfn import build_return_tensor
from puremacro.vfi.solve import solve_vfi


@dataclass(frozen=True)
class VFISolution:
    """Solved value function and greedy policy (always returned as numpy)."""

    V: np.ndarray                 # (n_a, n_z)
    policy_aprime: np.ndarray     # (n_a, n_z) int indices into a_grid
    policy_d: np.ndarray | None   # (n_a, n_z) int indices into d_grid, or None
    n_iter: int
    sup_norm: float
    backend: str
    endo_shape: tuple = ()        # component sizes of the endogenous grid; (n_a,) if 1-D
    a_grid: np.ndarray | None = None
    z_grid: np.ndarray | None = None

    def policy_components(self) -> tuple[np.ndarray, ...]:
        """Per-asset next-state index arrays, each (n_a, n_z), unravelled from the
        flat ``policy_aprime`` over ``endo_shape``. A 1-tuple for a single asset."""
        shape = self.endo_shape if self.endo_shape else (self.policy_aprime.shape[0],)
        return tuple(np.unravel_index(np.asarray(self.policy_aprime), shape))

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of value function convergence and dimensions."""
        n_a, n_z = self.V.shape
        records = [
            {"Metric": "Endogenous Grid Dimension (n_a)", "Value": str(n_a)},
            {"Metric": "Exogenous Shock Dimension (n_z)", "Value": str(n_z)},
            {"Metric": "Total State Space", "Value": str(n_a * n_z)},
            {"Metric": "Iterations", "Value": str(self.n_iter)},
            {"Metric": "Supremum Norm", "Value": f"{self.sup_norm:.6e}"},
            {"Metric": "Backend", "Value": self.backend},
            {"Metric": "Discrete Choice Included", "Value": str(self.policy_d is not None)},
            {"Metric": "Mean Value (V)", "Value": f"{float(np.mean(self.V)):.6f}"},
            {"Metric": "Min Value (V)", "Value": f"{float(np.min(self.V)):.6f}"},
            {"Metric": "Max Value (V)", "Value": f"{float(np.max(self.V)):.6f}"},
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_frame(self) -> pd.DataFrame:
        """Tabulate state indices/values, value function, and policy choices as a DataFrame."""
        n_a, n_z = self.V.shape
        a_idx, z_idx = np.meshgrid(np.arange(n_a), np.arange(n_z), indexing="ij")
        data: dict[str, Any] = {
            "a_idx": a_idx.ravel(),
            "z_idx": z_idx.ravel(),
        }
        if self.a_grid is not None:
            a_arr = np.asarray(self.a_grid)
            if a_arr.ndim == 1 and len(a_arr) == n_a:
                data["a"] = a_arr[a_idx.ravel()]
        if self.z_grid is not None:
            z_arr = np.asarray(self.z_grid)
            if z_arr.ndim == 1 and len(z_arr) == n_z:
                data["z"] = z_arr[z_idx.ravel()]
        data["V"] = self.V.ravel()
        data["policy_aprime"] = self.policy_aprime.ravel()
        if self.a_grid is not None:
            a_arr = np.asarray(self.a_grid)
            if a_arr.ndim == 1 and len(a_arr) == n_a:
                data["aprime_val"] = a_arr[self.policy_aprime.ravel()]
        if self.policy_d is not None:
            data["policy_d"] = self.policy_d.ravel()
        return pd.DataFrame(data)

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        show: bool = False,
    ) -> Any:
        """Headless and WASM-safe plot of value functions and policy functions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array-like of Axes, optional
            Pre-existing axes for plotting. If None, a new 1x2 subplot figure is created.
        figsize : tuple[float, float], optional
            Figure dimension (width, height) in inches.
        show : bool, default False
            If True, calls ``plt.show()``. Always False in headless/test environments.

        Returns
        -------
        matplotlib.figure.Figure or axes
            The figure if created anew, or the passed axis/axes.
        """
        import matplotlib.pyplot as plt

        n_a, n_z = self.V.shape
        created_fig = False
        if ax is None:
            fig, (ax_v, ax_p) = plt.subplots(1, 2, figsize=figsize or (10, 4.5))
            created_fig = True
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
            ax_v, ax_p = ax[0], ax[1]
            fig = ax_v.get_figure()
        else:
            ax_v = ax
            ax_p = None
            fig = ax_v.get_figure()

        if self.a_grid is not None and np.asarray(self.a_grid).ndim == 1 and len(np.asarray(self.a_grid)) == n_a:
            a_x = np.asarray(self.a_grid)
            x_label = "Asset (a)"
        else:
            a_x = np.arange(n_a)
            x_label = "Asset Index (a_idx)"

        # Value Function Panel
        for iz in range(n_z):
            if self.z_grid is not None and np.asarray(self.z_grid).ndim == 1 and len(np.asarray(self.z_grid)) == n_z:
                label = f"z = {self.z_grid[iz]:.2f}"
            else:
                label = f"z_idx = {iz}"
            ax_v.plot(a_x, self.V[:, iz], label=label)
        ax_v.set_title("Value Function V(a, z)")
        ax_v.set_xlabel(x_label)
        ax_v.set_ylabel("V")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(frameon=False)

        # Policy Function Panel
        if ax_p is not None:
            for iz in range(n_z):
                if self.z_grid is not None and np.asarray(self.z_grid).ndim == 1 and len(np.asarray(self.z_grid)) == n_z:
                    label = f"z = {self.z_grid[iz]:.2f}"
                    pol_y = a_x[self.policy_aprime[:, iz]]
                else:
                    label = f"z_idx = {iz}"
                    pol_y = self.policy_aprime[:, iz]
                ax_p.plot(a_x, pol_y, label=label)
            ax_p.plot(a_x, a_x, "k--", alpha=0.5, label="45° line")
            ax_p.set_title("Policy Function a'(a, z)")
            ax_p.set_xlabel(x_label)
            ax_p.set_ylabel("a'")
            ax_p.grid(True, alpha=0.3)
            ax_p.legend(frameon=False)

        if show:
            plt.show()

        return fig if created_fig else ax


@dataclass(frozen=True)
class VFIProblem:
    """Infinite-horizon discrete VFI problem (VFIToolkit "Case 1").

    State: endogenous ``a`` (a_grid) + exogenous Markov ``z`` (z_grid, P_z).
    Choice: next-period ``a'`` on a_grid, plus optional contemporaneous ``d``
    (d_grid). ``return_fn`` is xp-generic -- see vfi.returnfn for the signature.
    ``options`` keys: howard (bool), n_howard (int), tol (float), max_iter (int),
    divide_and_conquer (bool, numba only): exploit a monotone a'-policy for an
    O(n_a' log n_a) greedy step -- exact for supermodular/concave problems.
    """
    a_grid: np.ndarray
    z_grid: np.ndarray
    P_z: np.ndarray
    return_fn: Callable
    beta: float
    params: dict = field(default_factory=dict)
    d_grid: np.ndarray | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0,1); got {self.beta}")
        P = np.asarray(self.P_z)
        if isinstance(self.a_grid, (list, tuple)):
            grids = [np.asarray(g) for g in self.a_grid]
            if len(grids) < 1:
                raise ValueError("a_grid list must contain >= 1 endogenous grid")
            for g in grids:
                if g.ndim != 1 or g.size < 1:
                    raise ValueError("each endogenous grid must be 1-D with >= 1 point")
                if not np.all(np.isfinite(g)):
                    raise ValueError("endogenous grids must be finite")
            if int(np.prod([g.size for g in grids])) < 2:
                raise ValueError("the product endogenous grid must have >= 2 points")
        else:
            a = np.asarray(self.a_grid)
            if a.ndim != 1 or a.size < 2:
                raise ValueError("a_grid must be 1-D with >= 2 points")
            if not np.all(np.isfinite(a)):
                raise ValueError("a_grid must be finite")
        if isinstance(self.z_grid, (list, tuple)):
            zgrids = [np.asarray(g) for g in self.z_grid]
            if len(zgrids) < 1:
                raise ValueError("z_grid list must contain >= 1 shock grid")
            for g in zgrids:
                if g.ndim != 1 or g.size < 1:
                    raise ValueError("each exogenous grid must be 1-D with >= 1 point")
                if not np.all(np.isfinite(g)):
                    raise ValueError("z_grid must be finite")
            n_z_total = int(np.prod([g.size for g in zgrids]))
        else:
            zarr = np.asarray(self.z_grid)
            if zarr.ndim != 1 or zarr.size < 1:
                raise ValueError("z_grid must be 1-D with >= 1 points")
            if not np.all(np.isfinite(zarr)):
                raise ValueError("z_grid must be finite")
            n_z_total = int(zarr.size)
        if P.shape != (n_z_total, n_z_total):
            raise ValueError(f"P_z must be ({n_z_total},{n_z_total}); got {P.shape}")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("P_z rows must sum to 1")
        if self.d_grid is not None:
            d = np.asarray(self.d_grid)
            if d.ndim != 1 or d.size < 1:
                raise ValueError("d_grid must be 1-D with >= 1 points or None")

    def solve(self, backend: str = "numpy") -> VFISolution:
        if not _bk.backend_available(backend):
            raise ValueError(
                f"backend {backend!r} not available; installed: "
                f"{_bk.available_backends()}"
            )
        opt = self.options
        howard = bool(opt.get("howard", True))
        n_howard = int(opt.get("n_howard", 20))
        tol = float(opt.get("tol", 1e-8))
        max_iter = int(opt.get("max_iter", 10_000))
        divide_and_conquer = bool(opt.get("divide_and_conquer", False))

        z_arg = (self.z_grid if isinstance(self.z_grid, (list, tuple))
                 else np.asarray(self.z_grid, dtype=float))
        P = np.asarray(self.P_z, dtype=float)
        d = None if self.d_grid is None else np.asarray(self.d_grid, dtype=float)
        if isinstance(self.a_grid, (list, tuple)):
            grids = [np.asarray(g, dtype=float) for g in self.a_grid]
            endo_shape = tuple(int(g.size) for g in grids)
            a_arg = grids
        else:
            a_arr = np.asarray(self.a_grid, dtype=float)
            endo_shape = (int(a_arr.size),)
            a_arg = a_arr
        n_a = int(np.prod(endo_shape))

        if backend == "numba":
            from puremacro.vfi import kernels_numba as KN

            R = build_return_tensor(self.return_fn, a_arg, z_arg, self.params,
                                    d_grid=d, xp=np)
            R = np.ascontiguousarray(R)
            if divide_and_conquer:
                V, flat, n_it, sup = KN.solve_vfi_dc_numba(
                    R, P, float(self.beta), howard, n_howard, tol, max_iter
                )
            else:
                V, flat, n_it, sup = KN.solve_vfi_numba(
                    R, P, float(self.beta), howard, n_howard, tol, max_iter
                )
            n_ap = n_a
        else:
            if divide_and_conquer:
                raise ValueError(
                    "divide_and_conquer=True requires backend='numba' "
                    "(it is an inner-loop algorithm with no vectorised form)"
                )
            xp = _bk.get_array_namespace(backend)
            R = build_return_tensor(self.return_fn, a_arg, z_arg, self.params,
                                    d_grid=d, xp=xp)
            V, flat, n_ap, n_it, sup = solve_vfi(
                R, xp.array(P), float(self.beta), howard=howard,
                n_howard=n_howard, tol=tol, max_iter=max_iter, xp=xp
            )
            V = _bk.to_numpy(V)
            flat = _bk.to_numpy(flat)

        flat = np.asarray(flat).astype(np.int64)
        policy_aprime = flat % n_ap
        policy_d = (flat // n_ap) if d is not None else None
        return VFISolution(
            V=np.asarray(V, dtype=float),
            policy_aprime=policy_aprime,
            policy_d=policy_d,
            n_iter=int(n_it),
            sup_norm=float(sup),
            backend=backend,
            endo_shape=endo_shape,
            a_grid=self.a_grid if isinstance(self.a_grid, (list, tuple)) else np.asarray(self.a_grid, dtype=float),
            z_grid=self.z_grid if isinstance(self.z_grid, (list, tuple)) else np.asarray(self.z_grid, dtype=float),
        )

    def stationary_distribution(self, solution, *, tol: float = 1e-12,
                                max_iter: int = 100_000):
        """Stationary agent distribution mu(a,z) for a solved policy.

        Uses this problem's ``P_z`` and ``solution.policy_aprime`` (the
        VFISolution from ``.solve()``). Returns an (n_a, n_z) numpy array
        summing to 1. See ``puremacro.vfi.distribution.stationary_distribution``.
        """
        from puremacro.vfi.distribution import stationary_distribution

        return stationary_distribution(
            solution.policy_aprime, np.asarray(self.P_z, dtype=float),
            tol=tol, max_iter=max_iter,
        )

    def aggregate(self, solution, mu, fn, *, params=None):
        """Integrate ``fn`` over the agent distribution ``mu`` at the solved policy.

        ``fn`` follows the eval-fn convention (see puremacro.vfi.aggregate); it
        receives this problem's ``params`` by default. Returns a float.
        """
        from puremacro.vfi.aggregate import aggregate

        return aggregate(
            fn, mu, solution.policy_aprime,
            (self.a_grid if isinstance(self.a_grid, (list, tuple))
             else np.asarray(self.a_grid, dtype=float)),
            (self.z_grid if isinstance(self.z_grid, (list, tuple))
             else np.asarray(self.z_grid, dtype=float)),
            params=self.params if params is None else params,
            policy_d=solution.policy_d,
            d_grid=None if self.d_grid is None else np.asarray(self.d_grid, dtype=float),
        )


__all__ = ["VFIProblem", "VFISolution"]
