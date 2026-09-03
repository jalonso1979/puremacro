"""Pesaran (2021) CD test for cross-sectional dependence in panel residuals."""
import numpy as np
from scipy.stats import norm


def cd_test(residuals_panel):
    """
    Pesaran CD test for cross-sectional independence in a panel.

    CD = sqrt(2*T / (N(N-1))) * Σ_{i<j} ρ̂_ij

    Under H0 of cross-sectional independence: CD ~ N(0, 1) asymptotically.

    Parameters
    ----------
    residuals_panel : ndarray, shape (T, N)
        Panel residuals (T time periods, N cross-sectional units).

    Returns
    -------
    CD_stat : float
        Pesaran CD test statistic.
    p_value : float
        Two-tailed p-value from standard normal distribution.

    References
    ----------
    Pesaran, M. H. (2021). General diagnostic tests for cross-section
    dependence in panels. Empirical Economics, 60(1), 13-50.
    """
    residuals_panel = np.asarray(residuals_panel)
    assert residuals_panel.ndim == 2, "residuals_panel must be 2D: (T, N)"

    T, N = residuals_panel.shape

    # Compute pairwise correlations
    corr_matrix = np.corrcoef(residuals_panel.T)  # N x N

    # Sum of upper-triangle correlations (i < j)
    rho_sum = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rho_sum += corr_matrix[i, j]

    # CD test statistic
    cd_stat = np.sqrt(2 * T / (N * (N - 1))) * rho_sum

    # Two-tailed p-value from standard normal
    p_value = 2 * norm.sf(np.abs(cd_stat))

    return float(cd_stat), float(p_value)


# Public alias matching the test expectation
pesaran_cd = cd_test
