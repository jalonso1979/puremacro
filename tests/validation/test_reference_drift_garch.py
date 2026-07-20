"""CI-only drift-guard: recompute the LIVE arch reference and assert the frozen
goldens in goldens/garch.json are still faithful. This is the single place arch
is touched for GARCH validation. Run via: pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

arch = pytest.importorskip("arch")


def _fit_arch():
    from arch import arch_model

    from puremacro.validation.cases_garch import garch_demo_data

    d = garch_demo_data()
    res = arch_model(
        d["y"], mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    return res, d["burn"]


def test_garch_params_golden_matches_live_arch():
    from puremacro.validation._goldens import load_golden

    res, _ = _fit_arch()
    golden = load_golden("garch:params_vs_arch")
    live = {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
    }
    # If this fails, arch changed its estimate — regenerate the golden:
    #   python tools/gen_validation_goldens_garch.py
    for k in ("omega", "alpha", "beta"):
        np.testing.assert_allclose(live[k], golden[k], rtol=1e-6, atol=1e-9)


def test_garch_cond_vol_golden_matches_live_arch():
    from puremacro.validation._goldens import load_golden

    res, burn = _fit_arch()
    live = np.asarray(res.conditional_volatility, dtype=float)[burn:]
    golden = np.asarray(load_golden("garch:cond_vol_vs_arch")["sigma_tail"], dtype=float)
    np.testing.assert_allclose(live, golden, rtol=1e-6, atol=1e-9)
