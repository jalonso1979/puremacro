"""Smoke test for VAR Cholesky identification end-to-end."""
import numpy as np
import pytest

from puremacro.var import fit_var, irf, fevd
from puremacro.var.identify.cholesky import cholesky_factor


def test_cholesky_smoke(toy_var2):
    """Test Cholesky identification on toy VAR(1): fit → B0 → IRF → FEVD."""
    # Convert DataFrame to numpy array
    Y = toy_var2.values

    # Estimate VAR(1)
    result = fit_var(Y, p=1)
    assert len(result) == 5, f"Expected fit_var to return 5 elements, got {len(result)}"
    A_list, c, Sigma, residuals, X = result

    # Compute Cholesky factor (impact matrix B0)
    B0 = cholesky_factor(Sigma)

    # B0 should be lower triangular
    assert np.allclose(B0, np.tril(B0))

    # Cholesky factor must reconstruct Sigma
    assert np.allclose(B0 @ B0.T, Sigma, atol=1e-8), \
        f"B0 @ B0.T != Sigma:\nB0 @ B0.T =\n{B0 @ B0.T}\nSigma =\n{Sigma}"

    # Compute IRF with horizon 12
    irf_arr = irf(A_list, B0, horizon=12)

    # IRF should have shape (horizon+1, n_vars, n_vars)
    assert irf_arr.shape == (13, 2, 2), f"Expected shape (13, 2, 2), got {irf_arr.shape}"

    # IRF[0] should equal B0 (impact response)
    assert np.allclose(irf_arr[0], B0, atol=1e-8), \
        f"IRF[0] != B0:\nIRF[0] =\n{irf_arr[0]}\nB0 =\n{B0}"

    # Compute FEVD with horizon 12
    fev = fevd(A_list, B0, horizon=12)

    # FEVD should have same shape as IRF
    assert fev.shape == (13, 2, 2), f"Expected shape (13, 2, 2), got {fev.shape}"

    # FEVD shares should sum to 1 at each horizon (for h >= 1; h=0 is at impact)
    for h in range(1, 13):
        shares_sum = fev[h].sum(axis=1)  # sum across shocks for each variable
        assert np.allclose(shares_sum, 1.0, atol=1e-8), \
            f"FEVD shares don't sum to 1 at h={h}: {shares_sum}"

    # All FEVD entries should be between 0 and 1
    assert np.all(fev >= -1e-8) and np.all(fev <= 1.0 + 1e-8), \
        f"FEVD contains values outside [0, 1]"
