"""Batched Parallel Jacobian Assembly & Automatic Differentiation for CGE Trade Models.

This module replaces serial column perturbation loops with high-performance batched
tensor operations across batch dimension B = 230 (using factor return equivalence
r_c == w_c) or B = 307.

Key Capabilities:
1. One-time pre-inversion of Leontief operators (I - B^T) and (I - A) per scenario/step.
2. Vectorized multi-RHS linear solves / GEMM ((3465, 3465) @ (3465, B)) on GPU.
3. Batched tensor contractions (einsum) for bilateral trade flow accounting.
4. Forward-mode automatic differentiation / JVP (via torch.func.jvp / mx.jvp) eliminating
   step-size sensitivity on small economies (e.g. Cyprus).
"""
from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import scipy.linalg as la

from puremacro.trade._results import TradeCalibrationResult
from puremacro.trade.gpu.backend import (
    detect_device,
    has_mlx,
    has_torch,
    select_compute_device,
    to_numpy,
    to_tensor,
)

if has_torch():
    import torch

if has_mlx():
    import mlx.core as mx


class BatchedJacobianEvaluator:
    """High-performance GPU-accelerated batched Jacobian evaluator for 45-sector CGE trade models.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated trade model containing technology matrices, endowments, and shares.
    tau_a : np.ndarray
        Effective intermediate tariff matrix (ns * nc, ns, nc) or 4D tensor.
    taufd_a : np.ndarray
        Effective final demand tariff matrix (ns * nc, nfd, nc) or 4D tensor.
    tauf_vec : np.ndarray
        National intermediate tariff vectors (nc,).
    tauf_fd_vec : np.ndarray
        National final demand tariff vectors (nc,).
    device : str | None
        Target device ('auto', 'cuda', 'mps', 'mlx', 'cpu').
    backend : str | None
        Target backend ('torch' or 'mlx').
    batch_size : int, default 230
        Perturbation dimension B: 230 (factor equivalence r_c == w_c) or 307.
    replicate_matlab_precedence : bool, default True
        Whether to replicate MATLAB operator precedence at ff_equi.m:31.
    """

    def __init__(
        self,
        calib: TradeCalibrationResult,
        tau_a: np.ndarray,
        taufd_a: np.ndarray,
        tauf_vec: np.ndarray,
        tauf_fd_vec: np.ndarray,
        device: str | None = None,
        backend: str | None = None,
        batch_size: int = 230,
        replicate_matlab_precedence: bool = True,
    ) -> None:
        self.calib = calib
        self.ns = calib.n_sectors
        self.nc = calib.n_countries
        self.nfd = calib.n_final_demand
        self.M = self.ns * self.nc
        self.batch_size = batch_size
        self.replicate_matlab_precedence = replicate_matlab_precedence

        pref = device if device not in (None, "auto") else backend
        sel_backend, sel_device = select_compute_device(pref)
        self.backend = sel_backend if backend in (None, "auto") else backend
        self.device = sel_device if device in (None, "auto") else device

        # Check precision capabilities
        dev_info = detect_device(self.device)
        self.supports_float64 = dev_info.supports_float64
        # On MPS, default tensor dtype is float32; on CUDA/CPU/MLX, float64
        if self.backend == "torch":
            self.dtype = torch.float32 if self.device == "mps" else torch.float64
        else:
            self.dtype = None

        # Pre-invert constant output operator (I - A) once
        self.a_2d = self.calib.a.reshape((self.M, self.M), order="F")
        self.inv_Y_np = la.inv(np.eye(self.M, dtype=float) - self.a_2d)

        # Pre-cache calibration constants
        self.tax_flat = self.calib.tax.flatten(order="F")
        self.denom_tax = np.maximum(1.0 - self.tax_flat, 1e-12)
        self.tax_fd_arr = (
            self.calib.tax_fd
            if self.calib.tax_fd is not None
            else np.zeros((1, self.nfd, self.nc), dtype=float)
        )

        # Invert pricing operator (I - B^T) for given initial tariffs
        self.update_tariffs(tau_a, taufd_a, tauf_vec, tauf_fd_vec)

    def update_tariffs(
        self,
        tau_a: np.ndarray,
        taufd_a: np.ndarray,
        tauf_vec: np.ndarray,
        tauf_fd_vec: np.ndarray,
    ) -> None:
        """Update tariff schedules and pre-invert the Leontief pricing operator (I - B^T)."""
        self.tau_a = np.asarray(tau_a, dtype=float)
        self.taufd_a = np.asarray(taufd_a, dtype=float)
        self.tauf_vec = np.asarray(tauf_vec, dtype=float).ravel()
        self.tauf_fd_vec = np.asarray(tauf_fd_vec, dtype=float).ravel()

        # Reshape tariffs if 4D
        if self.tau_a.ndim == 4:
            self.tau_a = self.tau_a.transpose(1, 0, 2, 3).reshape(self.M, self.ns, self.nc)
        if self.taufd_a.ndim == 4:
            self.taufd_a = self.taufd_a.transpose(1, 0, 2, 3).reshape(self.M, self.nfd, self.nc)

        a_eff_2d = (self.calib.a * self.tau_a).reshape((self.M, self.M), order="F")
        B_T_mat = a_eff_2d.T / self.denom_tax[:, np.newaxis]
        M_P_dense = np.eye(self.M, dtype=float) - B_T_mat
        self.inv_P_np = la.inv(M_P_dense)

        # afd * taufd product
        self.afd_taufd_np = (self.calib.afd * self.taufd_a).reshape((self.M, self.nfd * self.nc), order="F")

        # Transfer matrices to PyTorch
        if has_torch():
            torch_dev = self.device if self.backend == "torch" else "cpu"
            torch_dtype = self.dtype if self.backend == "torch" and self.dtype is not None else torch.float64
            self.inv_P_t = torch.as_tensor(self.inv_P_np, device=torch_dev, dtype=torch_dtype)
            self.inv_Y_t = torch.as_tensor(self.inv_Y_np, device=torch_dev, dtype=torch_dtype)
            self.a_blocks_t = torch.as_tensor(self.a_2d.reshape(self.M, self.nc, self.ns), device=torch_dev, dtype=torch_dtype)
            self.afd_taufd_t = torch.as_tensor(self.afd_taufd_np, device=torch_dev, dtype=torch_dtype)
            self.tax_flat_t = torch.as_tensor(self.tax_flat, device=torch_dev, dtype=torch_dtype)
            self.denom_t = torch.as_tensor(self.denom_tax, device=torch_dev, dtype=torch_dtype)
            self.tax_fd_t = torch.as_tensor(self.tax_fd_arr, device=torch_dev, dtype=torch_dtype)
            self.tax_t = torch.as_tensor(self.calib.tax, device=torch_dev, dtype=torch_dtype).reshape(1, 1, self.ns, self.nc)
            self.l_endow_t = torch.as_tensor(self.calib.l_endow, device=torch_dev, dtype=torch_dtype).reshape(1, 1, 1, self.nc)
            self.k_endow_t = torch.as_tensor(self.calib.k_endow, device=torch_dev, dtype=torch_dtype).reshape(1, 1, 1, self.nc)
            self.alpha_t = torch.as_tensor(self.calib.alpha, device=torch_dev, dtype=torch_dtype).reshape(1, 1, self.ns, self.nc)
            self.beta_t = torch.as_tensor(self.calib.beta, device=torch_dev, dtype=torch_dtype).reshape(1, 1, self.ns, self.nc)
            self.theta_t = torch.as_tensor(self.calib.theta, device=torch_dev, dtype=torch_dtype).reshape(1, 1, self.nfd, self.nc)
            self.afd_t = torch.as_tensor(self.calib.afd, device=torch_dev, dtype=torch_dtype).reshape(1, self.M, self.nfd, self.nc)
            self.tauf_vec_t = torch.as_tensor(self.tauf_vec, device=torch_dev, dtype=torch_dtype)
            self.tauf_fd_vec_t = torch.as_tensor(self.tauf_fd_vec, device=torch_dev, dtype=torch_dtype)

        # Transfer matrices to Apple MLX
        if has_mlx():
            self.inv_P_mx = mx.array(self.inv_P_np)
            self.inv_Y_mx = mx.array(self.inv_Y_np)
            self.a_blocks_mx = mx.array(self.a_2d.reshape(self.M, self.nc, self.ns))
            self.afd_taufd_mx = mx.array(self.afd_taufd_np)
            self.tax_flat_mx = mx.array(self.tax_flat)
            self.denom_mx = mx.array(self.denom_tax)
            self.tax_fd_mx = mx.reshape(mx.array(self.tax_fd_arr), (1, 1, self.nfd, self.nc))
            self.tax_mx = mx.reshape(mx.array(self.calib.tax), (1, 1, self.ns, self.nc))
            self.l_endow_mx = mx.reshape(mx.array(self.calib.l_endow), (1, 1, 1, self.nc))
            self.k_endow_mx = mx.reshape(mx.array(self.calib.k_endow), (1, 1, 1, self.nc))
            self.alpha_mx = mx.reshape(mx.array(self.calib.alpha), (1, 1, self.ns, self.nc))
            self.beta_mx = mx.reshape(mx.array(self.calib.beta), (1, 1, self.ns, self.nc))
            self.theta_mx = mx.reshape(mx.array(self.calib.theta), (1, 1, self.nfd, self.nc))
            self.afd_mx = mx.reshape(mx.array(self.calib.afd), (1, self.M, self.nfd, self.nc))
            self.tauf_vec_mx = mx.array(self.tauf_vec)
            self.tauf_fd_vec_mx = mx.array(self.tauf_fd_vec)

            # Precision-preserving float64 arrays on Apple Silicon UMA CPU stream
            with mx.stream(mx.cpu):
                self.inv_P_mx_64 = mx.array(self.inv_P_np, dtype=mx.float64)
                self.inv_Y_mx_64 = mx.array(self.inv_Y_np, dtype=mx.float64)
                self.a_blocks_mx_64 = mx.array(self.a_2d.reshape(self.M, self.nc, self.ns), dtype=mx.float64)
                self.afd_taufd_mx_64 = mx.array(self.afd_taufd_np, dtype=mx.float64)
                self.tax_flat_mx_64 = mx.array(self.tax_flat, dtype=mx.float64)
                self.denom_mx_64 = mx.array(self.denom_tax, dtype=mx.float64)
                self.tax_fd_mx_64 = mx.reshape(mx.array(self.tax_fd_arr, dtype=mx.float64), (1, 1, self.nfd, self.nc))
                self.tax_mx_64 = mx.reshape(mx.array(self.calib.tax, dtype=mx.float64), (1, 1, self.ns, self.nc))
                self.l_endow_mx_64 = mx.reshape(mx.array(self.calib.l_endow, dtype=mx.float64), (1, 1, 1, self.nc))
                self.k_endow_mx_64 = mx.reshape(mx.array(self.calib.k_endow, dtype=mx.float64), (1, 1, 1, self.nc))
                self.alpha_mx_64 = mx.reshape(mx.array(self.calib.alpha, dtype=mx.float64), (1, 1, self.ns, self.nc))
                self.beta_mx_64 = mx.reshape(mx.array(self.calib.beta, dtype=mx.float64), (1, 1, self.ns, self.nc))
                self.theta_mx_64 = mx.reshape(mx.array(self.calib.theta, dtype=mx.float64), (1, 1, self.nfd, self.nc))
                self.afd_mx_64 = mx.reshape(mx.array(self.calib.afd, dtype=mx.float64), (1, self.M, self.nfd, self.nc))
                self.tauf_vec_mx_64 = mx.array(self.tauf_vec, dtype=mx.float64)
                self.tauf_fd_vec_mx_64 = mx.array(self.tauf_fd_vec, dtype=mx.float64)

    def eval_macro_single(
        self,
        xm: np.ndarray,
        p_cached: tuple[np.ndarray, np.ndarray] | None = None,
        compute_full: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate macro residual equations at a single point in float64 on host CPU."""
        nc, ns, nfd, M = self.nc, self.ns, self.nfd, self.M

        if self.batch_size == 230:
            w = np.exp(xm[:nc]).reshape((1, 1, nc))
            r = w
            T = xm[nc : 2 * nc].reshape((1, 1, nc))
            XN = xm[2 * nc :]
        else:
            r = np.exp(xm[:nc]).reshape((1, 1, nc))
            w = np.exp(xm[nc : 2 * nc]).reshape((1, 1, nc))
            T = xm[2 * nc : 3 * nc].reshape((1, 1, nc))
            XN = xm[3 * nc :]

        invforT = np.append(XN, -np.sum(XN))

        term_r = (r / self.calib.alpha) ** self.calib.alpha
        if self.replicate_matlab_precedence:
            term_w = w / ((1.0 - self.calib.alpha) ** (1.0 - self.calib.alpha))
        else:
            term_w = (w / (1.0 - self.calib.alpha)) ** (1.0 - self.calib.alpha)
        val_va = (1.0 / self.calib.beta) * (term_r * term_w)

        if p_cached is None:
            v_P = val_va.flatten(order="F") / self.denom_tax
            p_vec = self.inv_P_np @ v_P
            p = p_vec.reshape((1, ns, nc), order="F")
        else:
            p_vec, p = p_cached

        ppfd = (p_vec @ self.afd_taufd_np).reshape(nc, nfd).T.reshape((1, nfd, nc))
        Ycon = w * self.calib.l_endow + r * self.calib.k_endow + T
        cd = self.calib.theta * Ycon / ppfd
        Tax_c = self.tax_fd_arr * ppfd * cd
        c = cd.copy()
        c[:, 1:2, :] -= invforT.reshape((1, 1, nc)) / ppfd[:, 1:2, :]
        xc = self.calib.afd * (c - self.tax_fd_arr * cd)
        d = np.sum(xc, axis=(1, 2))
        y_vec = self.inv_Y_np @ d
        ytot = y_vec.reshape((1, ns, nc), order="F")

        mask_active = (ytot > 0)
        ratio_rw = ((1.0 - self.calib.alpha) * r) / (self.calib.alpha * w)
        xl = np.where(mask_active, (ytot / self.calib.beta) * (ratio_rw ** self.calib.alpha), 0.0)

        a_blocks = self.a_2d.reshape(M, nc, ns)
        y_blocks = y_vec.reshape(nc, ns)
        x_sum_c2 = np.einsum("min,in->mi", a_blocks, y_blocks)
        val_sum = x_sum_c2 * p_vec[:, np.newaxis]
        T_inter = np.sum(val_sum.reshape(nc, ns, nc), axis=1)

        xc_2d = xc.reshape((M, nfd * nc), order="F")
        ppfd_row = ppfd.reshape((1, nfd * nc), order="F")
        xc_blocks = xc_2d.reshape(M, nc, nfd)
        ppfd_blocks = ppfd_row.reshape(nc, nfd)
        fd_sum_c2 = np.einsum("min,in->mi", xc_blocks, ppfd_blocks)
        T_fd = np.sum(fd_sum_c2.reshape(nc, ns, nc), axis=1)

        np.fill_diagonal(T_inter, 0.0)
        np.fill_diagonal(T_fd, 0.0)

        X0 = np.sum(T_inter, axis=1)
        M0 = np.sum(T_inter, axis=0)
        XFD = np.sum(T_fd, axis=1)
        MFD = np.sum(T_fd, axis=0)
        invforT_realized = X0 + XFD - M0 - MFD

        Tax_Total = np.sum(self.calib.tax * ytot, axis=1).ravel() + np.sum(Tax_c, axis=1).ravel()
        Tarifs_Totals = M0 * self.tauf_vec + MFD * self.tauf_fd_vec

        ff2 = self.calib.l_endow.ravel() - np.sum(xl, axis=1).ravel()
        ff4 = XN - invforT_realized[: nc - 1]
        ff5 = T.ravel() - (Tax_Total + Tarifs_Totals)

        if self.batch_size == 230:
            f_macro = np.concatenate([ff2, ff4, ff5])
        else:
            ratio_wr = (self.calib.alpha * w) / ((1.0 - self.calib.alpha) * r)
            xk = np.where(mask_active, (ytot / self.calib.beta) * (ratio_wr ** (1.0 - self.calib.alpha)), 0.0)
            ff3 = self.calib.k_endow.ravel() - np.sum(xk, axis=1).ravel()
            f_macro = np.concatenate([ff2, ff3, ff4, ff5])

        f_full = None
        if compute_full:
            ff0 = y_vec - (self.a_2d @ y_vec + d)
            v_P = val_va.flatten(order="F") / self.denom_tax
            ff1 = p_vec - self.inv_P_np @ v_P
            if self.batch_size == 230:
                ratio_wr = (self.calib.alpha * w) / ((1.0 - self.calib.alpha) * r)
                xk = np.where(mask_active, (ytot / self.calib.beta) * (ratio_wr ** (1.0 - self.calib.alpha)), 0.0)
                ff3 = self.calib.k_endow.ravel() - np.sum(xk, axis=1).ravel()
            f_full = np.concatenate([ff0, ff1, ff2, ff3, ff4, ff5])

        return f_macro, f_full, p_vec, p, y_vec

    def eval_macro_batched_torch(self, X_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate B column perturbations in parallel on PyTorch GPU (CUDA or MPS)."""
        B = X_batch.shape[0]
        nc, ns, nfd, M = self.nc, self.ns, self.nfd, self.M

        if self.batch_size == 230:
            w_b = torch.exp(X_batch[:, :nc])
            r_b = w_b
            T_b = X_batch[:, nc : 2 * nc]
            XN_b = X_batch[:, 2 * nc :]
        else:
            r_b = torch.exp(X_batch[:, :nc])
            w_b = torch.exp(X_batch[:, nc : 2 * nc])
            T_b = X_batch[:, 2 * nc : 3 * nc]
            XN_b = X_batch[:, 3 * nc :]

        invforT_b = torch.cat([XN_b, -torch.sum(XN_b, dim=1, keepdim=True)], dim=1)

        # 4D factor rates: (B, 1, 1, nc)
        w_b_4d = w_b.unsqueeze(1).unsqueeze(1)
        r_b_4d = r_b.unsqueeze(1).unsqueeze(1)
        T_b_4d = T_b.unsqueeze(1).unsqueeze(1)

        term_r = (r_b_4d / self.alpha_t) ** self.alpha_t
        if self.replicate_matlab_precedence:
            term_w = w_b_4d / ((1.0 - self.alpha_t) ** (1.0 - self.alpha_t))
        else:
            term_w = (w_b_4d / (1.0 - self.alpha_t)) ** (1.0 - self.alpha_t)
        val_va = (1.0 / self.beta_t) * (term_r * term_w)

        # Flatten in Fortran order
        val_va_flat = val_va.squeeze(1).permute(0, 2, 1).reshape(B, M)
        v_P_b = val_va_flat / self.denom_t.unsqueeze(0)

        # Solve prices via GEMM
        P_b = (self.inv_P_t @ v_P_b.T).T  # (B, M)

        # Final demand prices
        ppfd_flat = P_b @ self.afd_taufd_t  # (B, nfd * nc)
        ppfd_b = ppfd_flat.reshape(B, nc, nfd).permute(0, 2, 1).unsqueeze(1)

        # Consumer income & absorption
        Ycon_b = w_b_4d * self.l_endow_t + r_b_4d * self.k_endow_t + T_b_4d
        cd_b = self.theta_t * Ycon_b / ppfd_b
        Tax_c_b = self.tax_fd_t * ppfd_b * cd_b

        c_b = cd_b.clone()
        c_b[:, :, 1:2, :] -= invforT_b.unsqueeze(1).unsqueeze(1) / ppfd_b[:, :, 1:2, :]
        xc_b = self.afd_t * (c_b - self.tax_fd_t * cd_b)
        d_b = xc_b.sum(dim=(2, 3))  # (B, M)

        # Solve gross outputs via GEMM
        Y_b = (self.inv_Y_t @ d_b.T).T  # (B, M)
        Y_b_4d = Y_b.reshape(B, nc, ns).permute(0, 2, 1).unsqueeze(1)

        # Factor demand
        ratio_rw = ((1.0 - self.alpha_t) * r_b_4d) / (self.alpha_t * w_b_4d)
        xl_b = (Y_b_4d / self.beta_t) * (ratio_rw ** self.alpha_t)

        # Bilateral trade flows
        y_blocks_b = Y_b.reshape(B, nc, ns)
        x_sum_b = torch.einsum("min,bin->bmi", self.a_blocks_t, y_blocks_b)
        val_sum_b = x_sum_b * P_b.unsqueeze(-1)
        T_inter_b = val_sum_b.reshape(B, nc, ns, nc).sum(dim=2)

        # Exact Final demand bilateral flow contraction
        xc_blocks_b = xc_b.permute(0, 1, 3, 2)  # (B, M, nc, nfd)
        ppfd_blocks_b = ppfd_b.squeeze(1).permute(0, 2, 1)  # (B, nc, nfd)
        fd_sum_b = torch.einsum("bmin,bin->bmi", xc_blocks_b, ppfd_blocks_b)
        T_fd_b = fd_sum_b.reshape(B, nc, ns, nc).sum(dim=2)

        eye_mask = torch.eye(nc, device=self.device, dtype=torch.bool).unsqueeze(0)
        T_inter_b = T_inter_b.masked_fill(eye_mask, 0.0)
        T_fd_b = T_fd_b.masked_fill(eye_mask, 0.0)

        X0_b = T_inter_b.sum(dim=2)
        M0_b = T_inter_b.sum(dim=1)
        XFD_b = T_fd_b.sum(dim=2)
        MFD_b = T_fd_b.sum(dim=1)
        invforT_realized_b = X0_b + XFD_b - M0_b - MFD_b

        Tax_Total_b = (self.tax_t * Y_b_4d).sum(dim=2).squeeze(1) + Tax_c_b.sum(dim=2).squeeze(1)
        Tarifs_Totals_b = M0_b * self.tauf_vec_t + MFD_b * self.tauf_fd_vec_t

        ff2_b = self.l_endow_t.reshape(1, nc) - xl_b.sum(dim=2).squeeze(1)
        ff4_b = XN_b - invforT_realized_b[:, : nc - 1]
        ff5_b = T_b - (Tax_Total_b + Tarifs_Totals_b)

        if self.batch_size == 230:
            return torch.cat([ff2_b, ff4_b, ff5_b], dim=1)
        else:
            ratio_wr = (self.alpha_t * w_b_4d) / ((1.0 - self.alpha_t) * r_b_4d)
            xk_b = (Y_b_4d / self.beta_t) * (ratio_wr ** (1.0 - self.alpha_t))
            ff3_b = self.k_endow_t.reshape(1, nc) - xk_b.sum(dim=2).squeeze(1)
            return torch.cat([ff2_b, ff3_b, ff4_b, ff5_b], dim=1)

    def eval_macro_batched_mlx(self, X_batch: Any, stream: Any = None) -> Any:
        """Evaluate B column perturbations in parallel on Apple MLX Unified Memory."""
        B = X_batch.shape[0]
        nc, ns, nfd, M = self.nc, self.ns, self.nfd, self.M

        use_f64 = (X_batch.dtype == mx.float64) or (stream == mx.cpu)
        target_stream = stream if stream is not None else (mx.cpu if use_f64 else (mx.gpu if self.device != "cpu" else mx.cpu))
        with mx.stream(target_stream):
            if use_f64:
                inv_P = getattr(self, "inv_P_mx_64", self.inv_P_mx)
                inv_Y = getattr(self, "inv_Y_mx_64", self.inv_Y_mx)
                a_blocks = getattr(self, "a_blocks_mx_64", self.a_blocks_mx)
                afd_taufd = getattr(self, "afd_taufd_mx_64", self.afd_taufd_mx)
                denom = getattr(self, "denom_mx_64", self.denom_mx)
                tax_fd = getattr(self, "tax_fd_mx_64", self.tax_fd_mx)
                tax = getattr(self, "tax_mx_64", self.tax_mx)
                l_endow = getattr(self, "l_endow_mx_64", self.l_endow_mx)
                k_endow = getattr(self, "k_endow_mx_64", self.k_endow_mx)
                alpha = getattr(self, "alpha_mx_64", self.alpha_mx)
                beta = getattr(self, "beta_mx_64", self.beta_mx)
                theta = getattr(self, "theta_mx_64", self.theta_mx)
                afd = getattr(self, "afd_mx_64", self.afd_mx)
                tauf_vec = getattr(self, "tauf_vec_mx_64", self.tauf_vec_mx)
                tauf_fd_vec = getattr(self, "tauf_fd_vec_mx_64", self.tauf_fd_vec_mx)
                eye_diag = mx.eye(nc, dtype=mx.float64)
            else:
                inv_P = self.inv_P_mx
                inv_Y = self.inv_Y_mx
                a_blocks = self.a_blocks_mx
                afd_taufd = self.afd_taufd_mx
                denom = self.denom_mx
                tax_fd = self.tax_fd_mx
                tax = self.tax_mx
                l_endow = self.l_endow_mx
                k_endow = self.k_endow_mx
                alpha = self.alpha_mx
                beta = self.beta_mx
                theta = self.theta_mx
                afd = self.afd_mx
                tauf_vec = self.tauf_vec_mx
                tauf_fd_vec = self.tauf_fd_vec_mx
                eye_diag = mx.eye(nc)

            if self.batch_size == 230:
                w_b = mx.exp(X_batch[:, :nc])
                r_b = w_b
                T_b = X_batch[:, nc : 2 * nc]
                XN_b = X_batch[:, 2 * nc :]
            else:
                r_b = mx.exp(X_batch[:, :nc])
                w_b = mx.exp(X_batch[:, nc : 2 * nc])
                T_b = X_batch[:, 2 * nc : 3 * nc]
                XN_b = X_batch[:, 3 * nc :]

            invforT_b = mx.concatenate([XN_b, -mx.sum(XN_b, axis=1, keepdims=True)], axis=1)

            w_b_4d = mx.expand_dims(mx.expand_dims(w_b, 1), 1)
            r_b_4d = mx.expand_dims(mx.expand_dims(r_b, 1), 1)
            T_b_4d = mx.expand_dims(mx.expand_dims(T_b, 1), 1)

            term_r = (r_b_4d / alpha) ** alpha
            if self.replicate_matlab_precedence:
                term_w = w_b_4d / ((1.0 - alpha) ** (1.0 - alpha))
            else:
                term_w = (w_b_4d / (1.0 - alpha)) ** (1.0 - alpha)
            val_va = (1.0 / beta) * (term_r * term_w)

            val_va_flat = mx.reshape(mx.transpose(mx.squeeze(val_va, 1), (0, 2, 1)), (B, M))
            v_P_b = val_va_flat / mx.expand_dims(denom, 0)

            # Linear solve via GPU GEMM
            P_b = mx.transpose(inv_P @ mx.transpose(v_P_b))

            ppfd_flat = P_b @ afd_taufd
            ppfd_b = mx.expand_dims(mx.transpose(mx.reshape(ppfd_flat, (B, nc, nfd)), (0, 2, 1)), 1)

            Ycon_b = w_b_4d * l_endow + r_b_4d * k_endow + T_b_4d
            cd_b = theta * Ycon_b / ppfd_b
            Tax_c_b = tax_fd * ppfd_b * cd_b

            c_b = cd_b
            c_inv = mx.expand_dims(mx.expand_dims(invforT_b, 1), 1) / ppfd_b[:, :, 1:2, :]
            c_b = mx.concatenate([c_b[:, :, :1, :], c_b[:, :, 1:2, :] - c_inv, c_b[:, :, 2:, :]], axis=2)

            xc_b = afd * (c_b - tax_fd * cd_b)
            d_b = mx.sum(xc_b, axis=(2, 3))

            Y_b = mx.transpose(inv_Y @ mx.transpose(d_b))
            Y_b_4d = mx.expand_dims(mx.transpose(mx.reshape(Y_b, (B, nc, ns)), (0, 2, 1)), 1)

            ratio_rw = ((1.0 - alpha) * r_b_4d) / (alpha * w_b_4d)
            xl_b = (Y_b_4d / beta) * (ratio_rw ** alpha)

            y_blocks_b = mx.reshape(Y_b, (B, nc, ns))
            x_sum_b = mx.einsum("min,bin->bmi", a_blocks, y_blocks_b)
            val_sum_b = x_sum_b * mx.expand_dims(P_b, -1)
            T_inter_b = mx.sum(mx.reshape(val_sum_b, (B, nc, ns, nc)), axis=2)

            # Exact Final demand bilateral flow contraction
            xc_blocks_b = mx.transpose(xc_b, (0, 1, 3, 2))  # (B, M, nc, nfd)
            ppfd_blocks_b = mx.transpose(mx.squeeze(ppfd_b, 1), (0, 2, 1))  # (B, nc, nfd)
            fd_sum_b = mx.einsum("bmin,bin->bmi", xc_blocks_b, ppfd_blocks_b)
            T_fd_b = mx.sum(mx.reshape(fd_sum_b, (B, nc, ns, nc)), axis=2)

            # Zero out diagonal
            diag_mask = mx.expand_dims(1.0 - eye_diag, 0)
            T_inter_b = T_inter_b * diag_mask
            T_fd_b = T_fd_b * diag_mask

            X0_b = mx.sum(T_inter_b, axis=2)
            M0_b = mx.sum(T_inter_b, axis=1)
            XFD_b = mx.sum(T_fd_b, axis=2)
            MFD_b = mx.sum(T_fd_b, axis=1)
            invforT_realized_b = X0_b + XFD_b - M0_b - MFD_b

            Tax_Total_b = mx.squeeze(mx.sum(tax * Y_b_4d, axis=2), 1) + mx.squeeze(mx.sum(Tax_c_b, axis=2), 1)
            Tarifs_Totals_b = M0_b * tauf_vec + MFD_b * tauf_fd_vec

            ff2_b = mx.reshape(l_endow, (1, nc)) - mx.squeeze(mx.sum(xl_b, axis=2), 1)
            ff4_b = XN_b - invforT_realized_b[:, : nc - 1]
            ff5_b = T_b - (Tax_Total_b + Tarifs_Totals_b)

            if self.batch_size == 230:
                return mx.concatenate([ff2_b, ff4_b, ff5_b], axis=1)
            else:
                ratio_wr = (alpha * w_b_4d) / ((1.0 - alpha) * r_b_4d)
                xk_b = (Y_b_4d / beta) * (ratio_wr ** (1.0 - alpha))
                ff3_b = mx.reshape(k_endow, (1, nc)) - mx.squeeze(mx.sum(xk_b, axis=2), 1)
                return mx.concatenate([ff2_b, ff3_b, ff4_b, ff5_b], axis=1)

    def evaluate_batched_jacobian(
        self,
        xm: np.ndarray,
        f_base: np.ndarray | None = None,
        eps_fd: float = 1e-4,
        ad_mode: Literal["finite_diff", "forward", "vjp"] = "finite_diff",
        stream: Any = None,
    ) -> np.ndarray:
        """Evaluate the full (B, B) macro Jacobian using parallel batched GPU execution or forward-mode AD.

        Parameters
        ----------
        xm : np.ndarray
            Current macro point (length 230 or 307).
        f_base : np.ndarray | None
            Unperturbed macro residual vector. If None, evaluated automatically.
        eps_fd : float, default 1e-4
            Finite difference step size.
        ad_mode : str, default 'finite_diff'
            Differentiation mode: 'finite_diff' (GPU batched), 'forward' (AD/JVP), or 'vjp'.
        stream : Any, default None
            Target execution stream (e.g. mx.gpu or mx.cpu for Apple MLX).

        Returns
        -------
        np.ndarray
            Jacobian matrix J of shape (B, B).
        """
        B = len(xm)
        nc = self.nc

        if f_base is None:
            f_base, _, _, _, _ = self.eval_macro_single(xm)

        # Forward AD mode via torch.func
        if ad_mode == "forward" and self.backend == "torch" and has_torch():
            try:
                def f_call(x_tensor: torch.Tensor) -> torch.Tensor:
                    return self.eval_macro_batched_torch(x_tensor.unsqueeze(0)).squeeze(0)

                xm_t = torch.as_tensor(xm, device=self.device, dtype=self.dtype)
                J_t = torch.func.jacfwd(f_call)(xm_t)
                return to_numpy(J_t)
            except Exception:
                pass

        # Forward AD mode via Apple MLX JVP
        if ad_mode == "forward" and self.backend == "mlx" and has_mlx():
            try:
                def f_call_mlx(x_m: Any) -> Any:
                    return mx.squeeze(self.eval_macro_batched_mlx(mx.expand_dims(x_m, 0), stream=stream), 0)

                xm_mx = mx.array(xm)
                J_cols = []
                for b in range(B):
                    e_b = np.zeros(B, dtype=float)
                    e_b[b] = 1.0
                    _, col_b = mx.jvp(f_call_mlx, (xm_mx,), (mx.array(e_b),))
                    J_cols.append(to_numpy(col_b[0]))
                return np.column_stack(J_cols)
            except Exception:
                pass

        # High-performance Batched Finite Difference (default)
        n_factor_vars = nc if self.batch_size == 230 else 2 * nc
        h = np.empty(B, dtype=float)
        for j in range(B):
            if j < n_factor_vars:
                h[j] = eps_fd
            else:
                h[j] = eps_fd * max(abs(xm[j]), 1.0)

        # Construct perturbation matrix
        X_batch_np = np.tile(xm, (B, 1)) + np.diag(h)

        if self.backend == "torch" and has_torch():
            X_batch_t = torch.as_tensor(X_batch_np, device=self.device, dtype=self.dtype)
            F_batch_t = self.eval_macro_batched_torch(X_batch_t)
            F_batch = to_numpy(F_batch_t)
        elif self.backend == "mlx" and has_mlx():
            target_stream = stream if stream is not None else (mx.cpu if self.device == "cpu" else mx.gpu)
            if target_stream == mx.cpu:
                with mx.stream(mx.cpu):
                    X_batch_mx = mx.array(X_batch_np, dtype=mx.float64)
                    F_batch_mx = self.eval_macro_batched_mlx(X_batch_mx, stream=mx.cpu)
                    mx.eval(F_batch_mx)
                    F_batch = to_numpy(F_batch_mx)
            else:
                with mx.stream(mx.gpu):
                    X_batch_mx = mx.array(X_batch_np, dtype=mx.float32)
                    F_batch_mx = self.eval_macro_batched_mlx(X_batch_mx, stream=mx.gpu)
                    mx.eval(F_batch_mx)
                    F_batch = to_numpy(F_batch_mx)
        else:
            # CPU numpy fallback
            F_batch = np.empty((B, B), dtype=float)
            for b in range(B):
                F_batch[b], _, _, _, _ = self.eval_macro_single(X_batch_np[b])

        # Form Jacobian columns: J[:, b] = (F_batch[b] - f_base) / h[b]
        diff = F_batch - f_base[np.newaxis, :]
        J = diff.T / h[np.newaxis, :]
        return J

    def jvp(self, xm: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute exact directional derivative (Jacobian-Vector Product) J(xm) @ v."""
        if self.backend == "torch" and has_torch():
            try:
                def f_call(x_tensor: torch.Tensor) -> torch.Tensor:
                    return self.eval_macro_batched_torch(x_tensor.unsqueeze(0)).squeeze(0)

                xm_t = torch.as_tensor(xm, device=self.device, dtype=self.dtype)
                v_t = torch.as_tensor(v, device=self.device, dtype=self.dtype)
                _, tangent = torch.func.jvp(f_call, (xm_t,), (v_t,))
                return to_numpy(tangent)
            except Exception:
                pass

        if self.backend == "mlx" and has_mlx():
            try:
                def f_call_mlx(x_m: Any) -> Any:
                    return mx.squeeze(self.eval_macro_batched_mlx(mx.expand_dims(x_m, 0)), 0)

                xm_mx = mx.array(xm)
                v_mx = mx.array(v)
                _, tangent = mx.jvp(f_call_mlx, (xm_mx,), (v_mx,))
                return to_numpy(tangent[0])
            except Exception:
                pass

        eps = 1e-7
        f_plus, _, _, _, _ = self.eval_macro_single(xm + eps * v)
        f_minus, _, _, _, _ = self.eval_macro_single(xm - eps * v)
        return (f_plus - f_minus) / (2.0 * eps)
