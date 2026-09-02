"""Tests for VarEstimateResult dataclass and var.estimate_var return shape."""
import numpy as np
import pytest


def _toy_var2(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def test_var_estimate_result_is_frozen_and_has_expected_fields():
    from puremacro.var._results import VarEstimateResult

    r = VarEstimateResult(
        A_list=[np.eye(2)],
        c=np.zeros(2),
        Sigma=np.eye(2),
        resid=np.zeros((10, 2)),
        X=np.zeros((10, 3)),
    )
    assert r.A_list[0].shape == (2, 2)
    with pytest.raises(Exception):
        r.c = np.ones(2)  # frozen


def test_estimate_var_returns_var_estimate_result():
    from puremacro.var.estimate import estimate_var
    from puremacro.var._results import VarEstimateResult

    Y = _toy_var2()
    r = estimate_var(Y, p=2)
    assert isinstance(r, VarEstimateResult)
    assert len(r.A_list) == 2
    assert r.Sigma.shape == (2, 2)


def test_var_estimate_result_supports_legacy_tuple_unpack():
    """Backwards-compatibility tail: existing code that does `A, c, S, e, X = estimate_var(Y, p)` must still work."""
    from puremacro.var.estimate import estimate_var

    Y = _toy_var2()
    A_list, c, Sigma, resid, X = estimate_var(Y, p=2)
    assert len(A_list) == 2
    assert c.shape == (2,)


def test_estimate_var_rejects_non_finite_data_by_name():
    """`np.linalg.lstsq` does not raise on NaN — it returns NaN coefficients.

    LAPACK writes "On entry to DLASCL parameter number 4 had an illegal value"
    directly to the terminal, bypassing sys.stderr, so it survives a `2>`
    redirect and pytest never sees it. Everything downstream — Sigma, the
    residuals, every IRF and every bootstrap band — is then silently all-NaN
    but perfectly well-formed, which is exactly the failure mode this package
    promises not to have.
    """
    import numpy as np
    import pytest

    from puremacro.var.estimate import estimate_var

    rng = np.random.default_rng(0)
    Y = rng.standard_normal((60, 2))
    assert np.isfinite(estimate_var(Y, 2).Sigma).all()      # positive control

    for bad in (np.nan, np.inf, -np.inf):
        Yb = Y.copy()
        Yb[5, 0] = bad
        with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
            estimate_var(Yb, 2)


def test_estimate_var_still_accepts_a_dataframe():
    """The finiteness guard must not break the DataFrame callers."""
    import numpy as np
    import pandas as pd

    from puremacro.var.estimate import estimate_var

    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((60, 2)), columns=["a", "b"])
    # DataFrame in, DataFrame Sigma out — pre-existing behaviour; the point
    # here is only that the finiteness guard does not choke on it.
    assert np.isfinite(np.asarray(estimate_var(df, 2).Sigma)).all()
