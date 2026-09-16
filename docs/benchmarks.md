> 🇬🇧 English · 🇪🇸 [Español](es/benchmarks.md)

# Computational Performance Benchmarks

`puremacro` is engineered for ultra-fast, zero-overhead execution using vectorized NumPy and SciPy routines.

---

## Runtime Benchmarks (Apple Silicon M-Series / Linux x86_64)

The table below reports execution times across representative macroeconomic models and econometric estimators:

| Engine | Runtime (ms) | Target Reference | Implementation Details |
| :--- | :--- | :--- | :--- |
| **VAR(4) + Cholesky IRF** ($T=1000, N=5, H=40$) | **1.07 ms** | MATLAB / statsmodels | Pure NumPy vectorized MA recursion |
| **Sequence-Space HANK GE Solve** ($T=40, N_a=50$) | **111.8 ms** | Julia SSJ / Python SSJ | EGM + $O(T^3)$ direct matrix inversion |
| **DICE Climate Simulation** ($150\text{ yr}, 3\text{-box}$) | **0.89 ms** | GAMS / DICE2016 | Vectorized reservoir transition cycle |
| **Mixed-Frequency DFM Nowcast** ($T=120, N=15$) | **4.84 ms** | NY Fed / MATLAB | EM SVD missing-value convergence |
| **Elastic Net Coordinate Descent** ($T=200, P=40, 30\lambda$) | **39.08 ms** | glmnet / scikit-learn | Pure NumPy soft-thresholding loop |
| **Bandpass (CF) + Beveridge-Nelson** ($T=500$) | **3.17 ms** | MATLAB / Stata `tsfilter` | Linear-time state space projection |

The timings above are the reference CPU NumPy/SciPy code paths. The
hardware-accelerated alternatives are documented elsewhere: the `backend=`
keyword of the spatial and trade solvers in
[`docs/spatial_and_trade_ge.md`](spatial_and_trade_ge.md), the GPU tariff surface
— batched Jacobians, homotopy continuation, the optional `torch` / `mlx` solvers
— in [`docs/trade_gpu.md`](trade_gpu.md), and the continuous-time HJB solver in
[`docs/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## Reproducing Benchmarks

Run the automated benchmark suite locally:

```bash
python benchmarks/benchmark_suite.py
```
