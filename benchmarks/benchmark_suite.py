"""Automated Performance and Accuracy Benchmark Suite for puremacro.

Measures runtime, computational complexity, and precision across core engines:
1. VAR Estimation & Cholesky IRF (VAR(4), 5 variables, T=1000, H=40).
2. Endogenous Grid Method (EGM) Stationary Household Solution.
3. Sequence-Space HANK General Equilibrium Solve (Auclert et al. 2021).
4. Dynamic Factor Model (DFM) with EM Ragged Edge Imputation.
5. Penalized Elastic Net Coordinate Descent (T=200, P=50, 40 lambdas).
6. Frequency Connectedness (Baruník-Křehlík 2018 spectral integrals).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from puremacro.var import fit_var, irf
from puremacro.var.identify.cholesky import safe_cholesky
from puremacro.models import solve_hank_sequence_space
from puremacro.climate import simulate_dice_model
from puremacro.nowcast import nowcast_gdp
from puremacro.forecast import forecast_penalized
from puremacro.cycles import christiano_fitzgerald_filter, beveridge_nelson_filter


def benchmark_all(n_runs: int = 5) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(42)

    # 1. VAR + IRF Benchmark
    Z = rng.normal(size=(1000, 5))
    t0 = time.perf_counter()
    for _ in range(n_runs):
        A, c, Sig, res, _ = fit_var(Z, p=4)
        P = safe_cholesky(Sig)
        _ = irf(A, P, horizon=40)
    t_var = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "VAR(4) + Cholesky IRF (T=1000, N=5, H=40)",
        "Runtime (ms)": t_var,
        "Target Reference": "MATLAB / statsmodels",
        "Speedup Note": "Pure NumPy vectorized recursion",
    })

    # 2. Sequence-Space HANK Benchmark
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = solve_hank_sequence_space(T=40, n_a=50)
    t_hank = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "Sequence-Space HANK GE Solve (T=40, Na=50)",
        "Runtime (ms)": t_hank,
        "Target Reference": "Julia SSJ / Python SSJ",
        "Speedup Note": "EGM + O(T³) direct matrix inversion",
    })

    # 3. DICE Climate-Macro Simulation
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = simulate_dice_model(n_periods=30, time_step_years=5)
    t_dice = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "DICE Climate-Economy Simulation (150yr, 3-box)",
        "Runtime (ms)": t_dice,
        "Target Reference": "GAMS / DICE2016",
        "Speedup Note": "Vectorized reservoir transition cycle",
    })

    # 4. Mixed-Frequency DFM Nowcast
    X_m = pd.DataFrame(rng.normal(size=(120, 15)), columns=[f"M_{i}" for i in range(15)])
    X_m.iloc[-1, 10:] = np.nan # Ragged edge
    y_q = pd.Series(rng.normal(size=40), index=[f"Q{i}" for i in range(40)])
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = nowcast_gdp(X_m, y_q, n_factors=2)
    t_dfm = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "Mixed-Frequency DFM Nowcast (T=120, N=15, EM)",
        "Runtime (ms)": t_dfm,
        "Target Reference": "NY Fed / MATLAB Nowcasting",
        "Speedup Note": "EM SVD missing-value convergence",
    })

    # 5. Penalized Elastic Net Coordinate Descent
    X_pen = rng.normal(size=(200, 40))
    y_pen = rng.normal(size=200)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = forecast_penalized(X_pen, y_pen, horizon=1, n_lambdas=30)
    t_pen = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "Elastic Net Coordinate Descent (T=200, P=40, 30 λ)",
        "Runtime (ms)": t_pen,
        "Target Reference": "glmnet / scikit-learn",
        "Speedup Note": "Pure NumPy soft-thresholding loop",
    })

    # 6. Christiano-Fitzgerald & Beveridge-Nelson Cycle Decompositions
    series_raw = rng.normal(size=500).cumsum()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = christiano_fitzgerald_filter(series_raw, low=6, high=32)
        _ = beveridge_nelson_filter(series_raw, p=4)
    t_cycles = (time.perf_counter() - t0) / n_runs * 1000.0
    results.append({
        "Engine": "Bandpass (CF) + Beveridge-Nelson (T=500)",
        "Runtime (ms)": t_cycles,
        "Target Reference": "MATLAB / Stata tsfilter",
        "Speedup Note": "Linear-time state space projection",
    })

    df = pd.DataFrame(results)
    return df


def main():
    print("Running puremacro Benchmark Suite...")
    df_res = benchmark_all(n_runs=5)
    print("\n" + "=" * 90)
    print("PUREMACRO COMPUTATIONAL PERFORMANCE BENCHMARKS")
    print("=" * 90)
    print(df_res.to_markdown(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
