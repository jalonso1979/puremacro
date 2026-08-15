"""Cross-validation script: Python puremacro vs MATLAB puremacro."""

import subprocess
import numpy as np
from puremacro.cycles import hamilton_filter
from puremacro.var.estimate import estimate_var
from puremacro.var.identify.cholesky import cholesky_factor
from puremacro.var.irf import irf, fevd
from puremacro.dsge.gensys import gensys

def run_cross_validation():
    print("====================================================")
    print("Python vs MATLAB puremacro Benchmark Cross-Checks")
    print("====================================================\n")

    # 1. Hamilton filter benchmark
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100))
    cycle_py, trend_py = hamilton_filter(y, h=8, p=4)

    np.savetxt("matlab/tests/bench_y.csv", y, delimiter=",")
    
    cmd = """
    addpath('matlab/tests');
    addpath('matlab');
    y = csvread('matlab/tests/bench_y.csv');
    [c, t] = puremacro.cycles.hamilton(y, 8, 4);
    csvwrite('matlab/tests/bench_c_matlab.csv', c);
    """
    subprocess.run(["octave-cli", "--eval", cmd], check=True)

    c_mat = np.loadtxt("matlab/tests/bench_c_matlab.csv", delimiter=",")
    valid_mask = ~np.isnan(cycle_py)
    diff_hamilton = np.max(np.abs(cycle_py[valid_mask] - c_mat[valid_mask]))
    print(f"[1] Hamilton filter max diff: {diff_hamilton:.2e}")
    assert diff_hamilton < 1e-10, "Hamilton filter mismatch!"

    # 2. VAR(2) Companion & Impulse Response benchmark
    Y_var = np.random.randn(100, 2)
    A_list_py, c_py, Sigma_py, resid_py, _ = estimate_var(Y_var, p=2)
    B0_py = cholesky_factor(Sigma_py)
    irf_py = irf(A_list_py, B0_py, horizon=10)

    np.savetxt("matlab/tests/bench_Yvar.csv", Y_var, delimiter=",")
    cmd_var = """
    addpath('matlab/tests');
    addpath('matlab');
    Y = csvread('matlab/tests/bench_Yvar.csv');
    res = puremacro.var.estimate(Y, 2);
    B0 = puremacro.var.cholesky(res.Sigma);
    irfs = puremacro.var.irf(res.A_list, B0, 10);
    csvwrite('matlab/tests/bench_irf_matlab.csv', irfs(:,:));
    """
    subprocess.run(["octave-cli", "--eval", cmd_var], check=True)

    irf_mat = np.loadtxt("matlab/tests/bench_irf_matlab.csv", delimiter=",").reshape(11, 2, 2, order="F")
    diff_irf = np.max(np.abs(irf_py - irf_mat))
    print(f"[2] SVAR IRF max diff: {diff_irf:.2e}")
    assert diff_irf < 1e-10, "SVAR IRF mismatch!"

    # 3. Sims gensys solver benchmark (unstable root = 1.5, n_eta = 1)
    G0 = np.eye(2)
    G1 = np.array([[1.5, 0.0], [0.0, 0.5]])
    Psi = np.array([[1.0], [0.5]])
    Pi = np.array([[1.0], [0.0]])

    sol_py = gensys(G0, G1, Psi, Pi)
    cmd_dsge = """
    addpath('matlab/tests');
    addpath('matlab');
    G0 = eye(2); G1 = [1.5, 0; 0, 0.5]; Psi = [1; 0.5]; Pi = [1; 0];
    sol = puremacro.dsge.gensys(G0, G1, Psi, Pi);
    csvwrite('matlab/tests/bench_G_matlab.csv', sol.G);
    csvwrite('matlab/tests/bench_Impact_matlab.csv', sol.Impact);
    """
    subprocess.run(["octave-cli", "--eval", cmd_dsge], check=True)

    G_mat = np.loadtxt("matlab/tests/bench_G_matlab.csv", delimiter=",")
    Impact_mat = np.loadtxt("matlab/tests/bench_Impact_matlab.csv", delimiter=",").reshape(2, 1)

    diff_G = np.max(np.abs(sol_py.G - G_mat))
    diff_Impact = np.max(np.abs(sol_py.Impact - Impact_mat))
    print(f"[3] Sims gensys G transition max diff: {diff_G:.2e}")
    print(f"[3] Sims gensys Impact matrix max diff: {diff_Impact:.2e}")
    assert diff_G < 1e-10 and diff_Impact < 1e-10, "Gensys solver mismatch!"

    print("\n====================================================")
    print("ALL PYTHON VS MATLAB BENCHMARKS MATCHED PERFECTLY!")
    print("====================================================")

if __name__ == "__main__":
    run_cross_validation()
