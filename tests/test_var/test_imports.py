def test_var_public_imports():
    from puremacro.var import (
        fit_var, lag_select, companion, is_stable,
        irf, fevd, historical_decomp, bootstrap_bands,
    )
    from puremacro.var.identify import (
        cholesky, bq, sign_restrictions, proxy, hetero,
    )
    for fn in (fit_var, lag_select, companion, is_stable,
               irf, fevd, historical_decomp, bootstrap_bands,
               cholesky, bq, sign_restrictions, proxy, hetero):
        assert callable(fn)
