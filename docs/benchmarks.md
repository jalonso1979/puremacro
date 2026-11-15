> 🇬🇧 English · 🇪🇸 [Español](es/benchmarks.md)

# Computational Performance Benchmarks

`puremacro` is engineered for ultra-fast, zero-overhead execution using vectorized NumPy and SciPy routines.

---

## Runtime Benchmarks (Apple Silicon M-Series, Python 3.11)

The table below reports mean execution times across `n_runs=5` iterations measured on 2026-11-15 (`puremacro` 6.6.0 tree, Python 3.11, Apple Silicon M-Series):

| Engine | Runtime (ms) | Target Reference | Implementation Details |
| :--- | :--- | :--- | :--- |
| **VAR(4) + Cholesky IRF** ($T=1000, N=5, H=40$) | **0.60 ms** | MATLAB / statsmodels | Pure NumPy vectorized MA recursion |
| **Sequence-Space HANK GE Solve** ($T=40, N_a=50$) | **36.06 ms** | Julia SSJ / Python SSJ | EGM + $O(T^3)$ direct matrix inversion |
| **DICE Climate Simulation** ($150\text{ yr}, 3\text{-box}$) | **8.77 ms** | GAMS / DICE2016 | Vectorized reservoir transition cycle |
| **Mixed-Frequency DFM Nowcast** ($T=120, N=15$) | **3.31 ms** | NY Fed / MATLAB | EM SVD missing-value convergence |
| **Elastic Net Coordinate Descent** ($T=200, P=40, 30\lambda$) | **31.43 ms** | glmnet / scikit-learn | Pure NumPy soft-thresholding loop |
| **Bandpass (CF) + Beveridge-Nelson** ($T=500$) | **2.45 ms** | MATLAB / Stata `tsfilter` | Linear-time state space projection |

The timings above are the reference CPU NumPy/SciPy code paths. The
hardware-accelerated alternatives are documented elsewhere: the `backend=`
keyword of the spatial solver in
[`docs/spatial_and_trade_ge.md`](spatial_and_trade_ge.md) and the continuous-time
HJB solver in [`docs/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## Reproducing Benchmarks

Run the automated benchmark suite locally:

```bash
python benchmarks/benchmark_suite.py
```
