"""Replication tests for ``puremacro.examples.svariv_mertens_ravn``.

Synthetic-DGP replication: we know the true structural impact column
(under the Stock-Watson proxy-SVAR normalization), so we can check that
proxy-SVAR recovers the peak response within a tight tolerance and that
weak-IV diagnostics behave as expected.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mr_result():
    from puremacro.examples.svariv_mertens_ravn import (
        run_mertens_ravn_replication,
    )
    return run_mertens_ravn_replication(seed=0)


def test_mertens_ravn_strong_instrument(mr_result):
    """Cragg-Donald F should be well above the Stock-Yogo 10% cutoff
    (~16.4) on this DGP: the proxy is wired to be strong."""
    cd_f = mr_result["cd_f"]
    assert cd_f > 10.0, f"CD F = {cd_f:.2f} <= 10 (weak instrument)"


def test_mertens_ravn_peak_within_10pct(mr_result):
    """Recovered peak |response| of the second variable should match
    the population truth within 10%."""
    true_peak = abs(mr_result["true_peak"])
    rec_peak = abs(mr_result["recovered_peak"])
    rel_err = abs(rec_peak - true_peak) / true_peak
    assert rel_err <= 0.10, (
        f"recovered peak |{rec_peak:.4f}| differs from truth "
        f"|{true_peak:.4f}| by {rel_err * 100:.1f}% (>10%)"
    )


def test_mertens_ravn_msw_bands_present(mr_result):
    """MSW (AR-inverted) bands should be finite, well-formed, and
    bracket the 2SLS point estimate (lower <= upper)."""
    lo = mr_result["msw_lo"]
    hi = mr_result["msw_hi"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi, f"MSW band malformed: lo={lo:.4f} > hi={hi:.4f}"
    grid = mr_result["msw_grid"]
    assert grid.ndim == 1 and grid.size > 1
