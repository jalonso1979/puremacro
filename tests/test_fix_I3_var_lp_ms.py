"""Follow-up fixes of review keys VAR, MSAR and LALP in the files they did not own.

* BQ cumulation mask (``bq_svar(cumulate=...)``) now reaches the panel and
  teaching BQ paths: ``var.panel.mean_group_svar``,
  ``var.identify.panel.mean_group_svar``, ``teaching.svar_panel`` and
  ``teaching.bq_canonical.bq_gdp_urate`` (which must not cumulate the
  unemployment rate). ``BQSVARResult`` no longer claims every row is a level
  response.
* ``teaching.svar_panel.fit_per_country`` used ``scheme_kwargs.pop`` inside
  the country loop (so only the first country got the caller's ``n_boot`` /
  ``ci``) and seeded each country with ``hash()`` of a string (salted per
  process, so bands changed from run to run).
* ``var.regime.girf`` crashed on an MS-VAR fit with ``p = 0``; its docstring
  called the MSIH model a "Hamilton 1989 teaching spec".
* ``LPResult`` carries la_lp's lag specification through pandas operations;
  ``lp_hac`` no longer credits its HAC bandwidth to Plagborg-Moller & Wolf.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from puremacro.var.estimate import estimate_var
from puremacro.var.identify.bq import _bq_impact, bq_svar
from puremacro.var.irf import irf as compute_irf

# Canonical (dy, u) VAR(1) with a lower-triangular long-run matrix (same DGP
# as tests/test_fix_VAR_bq_cumulate.py): u is stationary in levels.
_A = np.array([[0.3, -0.2], [0.1, 0.8]])
_C1 = np.array([[1.0, 0.0], [0.5, 0.7]])
_B = np.linalg.solve(np.linalg.inv(np.eye(2) - _A), _C1)


def _simulate(T, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T + 200, 2)) @ _B.T
    X = np.zeros((T + 200, 2))
    for t in range(1, T + 200):
        X[t] = _A @ X[t - 1] + e[t]
    return X[200:]


def _panel(N=4, T=400):
    return {f"C{i}": _simulate(T, seed=10 + i) for i in range(N)}


def _country_raw_irf(Y, p, H):
    A_list, _, Sigma, _, _ = estimate_var(Y, p)
    return compute_irf(A_list, _bq_impact(A_list, Sigma, 0), H)


# --------------------------------------------------------------------------
# BQSVARResult docstring
# --------------------------------------------------------------------------

def test_bqsvar_result_docstring_describes_the_cumulate_mask():
    from puremacro.var.identify._results import BQSVARResult

    doc = BQSVARResult.__doc__
    assert "represents\n    the *level* response of each variable" not in doc
    assert "cumulate" in doc
    assert "must not be cumulated" in doc


# --------------------------------------------------------------------------
# Canonical mean-group panel SVAR (var.identify.panel)
# --------------------------------------------------------------------------

class TestIdentifyPanelBQ:
    def test_default_cumulates_every_row(self):
        from puremacro.var.identify.panel import mean_group_svar

        panel = _panel()
        res = mean_group_svar(panel, p=1, horizon=12, identification="bq")
        ref = np.mean([np.cumsum(_country_raw_irf(panel[c], 1, 12), axis=0)
                       for c in sorted(panel)], axis=0)
        np.testing.assert_allclose(res.irf_mean, ref, rtol=0, atol=1e-13)

    @pytest.mark.parametrize("spec", [[0], [True, False], np.array([0])])
    def test_cumulate_first_row_only(self, spec):
        from puremacro.var.identify.panel import mean_group_svar

        panel = _panel()
        H = 40
        res = mean_group_svar(panel, p=1, horizon=H, identification="bq",
                              cumulate=spec)
        for i, cid in enumerate(res.country_ids):
            ref = bq_svar(panel[cid], p=1, horizon=H, n_boot=0, cumulate=[0])
            np.testing.assert_allclose(res.country_irfs[i], ref.irf_point,
                                       rtol=0, atol=1e-13)
        # the unemployment row returns to zero; the cumulated one would not
        assert np.abs(res.irf_mean[H, 1, :]).max() < 0.05
        old = mean_group_svar(panel, p=1, horizon=H, identification="bq")
        assert np.abs(old.irf_mean[H, 1, :]).max() > 0.3

    def test_bad_cumulate_raises(self):
        from puremacro.var.identify.panel import mean_group_svar

        with pytest.raises(ValueError, match="out of range"):
            mean_group_svar(_panel(), p=1, horizon=4, identification="bq",
                            cumulate=[5])


# --------------------------------------------------------------------------
# Legacy mean-group panel SVAR (var.panel)
# --------------------------------------------------------------------------

class TestLegacyPanelBQ:
    def test_default_cumulates_every_row(self):
        from puremacro.var.panel import mean_group_svar

        panel = _panel()
        res = mean_group_svar(panel, p=1, horizon=12, identification="bq")
        ref = np.mean([np.cumsum(_country_raw_irf(panel[c], 1, 12), axis=0)
                       for c in panel], axis=0)
        np.testing.assert_allclose(res.irf_mean, ref, rtol=0, atol=1e-13)

    def test_cumulate_first_row_only(self):
        from puremacro.var.panel import mean_group_svar

        panel = _panel()
        res = mean_group_svar(panel, p=1, horizon=20, identification="bq",
                              cumulate=[0])
        raw = np.stack([_country_raw_irf(panel[c], 1, 20) for c in panel])
        np.testing.assert_allclose(res.country_irfs[:, :, 0, :],
                                   np.cumsum(raw[:, :, 0, :], axis=1),
                                   rtol=0, atol=1e-13)
        np.testing.assert_allclose(res.country_irfs[:, :, 1, :],
                                   raw[:, :, 1, :], rtol=0, atol=1e-13)

    def test_bad_cumulate_raises_instead_of_zero_substitution(self):
        """Per-country ValueErrors are replaced by zeros with a warning; a
        malformed mask must fail once, up front."""
        import warnings

        from puremacro.var.panel import mean_group_svar

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="out of range"):
                mean_group_svar(_panel(), p=1, horizon=4,
                                identification="bq", cumulate=[2])

    def test_cumulate_ignored_by_cholesky(self):
        from puremacro.var.panel import mean_group_svar

        panel = _panel()
        a = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        b = mean_group_svar(panel, p=1, horizon=4, identification="cholesky",
                            cumulate=[0])
        np.testing.assert_array_equal(a.irf_mean, b.irf_mean)


# --------------------------------------------------------------------------
# teaching.svar_panel
# --------------------------------------------------------------------------

def _wide_panel():
    """Long (code, date) panel with a levels variable u and an I(1) level y
    whose first difference is dy of the canonical DGP."""
    frames = []
    for i, code in enumerate(["AAA", "BBB", "CCC"]):
        X = _simulate(240, seed=40 + i)
        dates = pd.period_range("1960Q1", periods=241, freq="Q").to_timestamp()
        y = np.concatenate([[0.0], np.cumsum(X[:, 0])])
        u = np.concatenate([[np.nan], X[:, 1]])
        frames.append(pd.DataFrame({"code": code, "date": dates, "u": u, "y": y}))
    return pd.concat(frames).set_index(["code", "date"])


class TestTeachingSvarPanel:
    def test_fit_svar_country_forwards_cumulate(self):
        from puremacro.teaching.svar_panel import fit_svar_country

        Y = _simulate(300, seed=3)
        out = fit_svar_country(Y, p=1, horizon=12, scheme="bq", n_boot=20,
                               seed=1, cumulate=[0])
        ref = bq_svar(Y, p=1, horizon=12, n_boot=20, seed=1, cumulate=[0])
        np.testing.assert_array_equal(out["point"], ref.irf_point)
        np.testing.assert_array_equal(out["lo"], ref.irf_lower)
        np.testing.assert_array_equal(out["hi"], ref.irf_upper)

    def test_fit_svar_country_bad_cumulate_raises(self):
        from puremacro.teaching.svar_panel import fit_svar_country

        with pytest.raises(ValueError, match="out of range"):
            fit_svar_country(_simulate(300, seed=3), p=1, horizon=4,
                             scheme="bq", n_boot=5, cumulate=[3])

    def test_fit_per_country_cumulates_only_vars_diff(self):
        from puremacro.teaching.svar_panel import fit_per_country, prep_country_system

        wide = _wide_panel()
        res = fit_per_country(wide, ["AAA", "BBB"], vars_levels=["u"],
                              vars_diff=["y"], p=1, horizon=12, scheme="bq",
                              n_boot=10)
        assert set(res) == {"AAA", "BBB"}
        for code, r in res.items():
            _, Y, names = prep_country_system(wide, code, vars_levels=["u"],
                                              vars_diff=["y"])
            assert names == ["u", "y"]
            ref = bq_svar(Y, p=1, horizon=12, n_boot=10,
                          seed=zlib.crc32(f"{code}|bq".encode()),
                          cumulate=[False, True])
            np.testing.assert_array_equal(r["point"], ref.irf_point)
            np.testing.assert_array_equal(r["lo"], ref.irf_lower)

    def test_fit_per_country_explicit_cumulate_wins(self):
        from puremacro.teaching.svar_panel import fit_per_country

        wide = _wide_panel()
        a = fit_per_country(wide, ["AAA"], vars_levels=["u"], vars_diff=["y"],
                            p=1, horizon=6, scheme="bq", n_boot=5,
                            cumulate=True)["AAA"]
        b = fit_per_country(wide, ["AAA"], vars_levels=["u"], vars_diff=["y"],
                            p=1, horizon=6, scheme="bq", n_boot=5)["AAA"]
        np.testing.assert_array_equal(a["point"][:, 1], b["point"][:, 1])
        np.testing.assert_allclose(a["point"][:, 0],
                                   np.cumsum(b["point"][:, 0], axis=0))

    def test_every_country_gets_the_callers_n_boot_ci_and_a_stable_seed(self, monkeypatch):
        import importlib

        from puremacro.teaching.svar_panel import fit_per_country

        # the package re-exports a function named ``bq``; get the module
        bq_mod = importlib.import_module("puremacro.var.identify.bq")

        calls = []
        real = bq_mod.bq_svar

        def spy(Y, **kw):
            calls.append(kw)
            return real(Y, **kw)

        monkeypatch.setattr(bq_mod, "bq_svar", spy)
        codes = ["AAA", "BBB", "CCC"]
        fit_per_country(_wide_panel(), codes, vars_levels=["u"],
                        vars_diff=["y"], p=1, horizon=4, scheme="bq",
                        n_boot=7, ci=0.68)
        assert [c["n_boot"] for c in calls] == [7, 7, 7]
        assert [c["ci"] for c in calls] == [0.68, 0.68, 0.68]
        assert [c["seed"] for c in calls] == [
            zlib.crc32(f"{code}|bq".encode()) for code in codes]


# --------------------------------------------------------------------------
# teaching.bq_canonical
# --------------------------------------------------------------------------

def test_bq_gdp_urate_does_not_cumulate_the_unemployment_rate():
    from puremacro.teaching.bq_canonical import bq_gdp_urate

    X = _simulate(3000, seed=5)
    log_gdp = np.concatenate([[0.0], np.cumsum(X[:, 0])])
    urate = np.concatenate([[X[0, 1]], X[:, 1]])
    df = pd.DataFrame({"log_gdp_real": log_gdp, "urate": urate})
    H = 40
    point, lo, hi = bq_gdp_urate(df, p=1, horizon=H, n_boot=30, seed=0)

    Y = np.column_stack([np.diff(log_gdp), urate[1:]])
    raw = _country_raw_irf(Y, 1, H)
    signs = np.sign(raw[0, [0, 1], [0, 1]])          # impact_rule (0,0), (1,1)
    np.testing.assert_allclose(point[:, 0, :], np.cumsum(raw[:, 0, :], axis=0) * signs,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(point[:, 1, :], raw[:, 1, :] * signs,
                               rtol=0, atol=1e-12)
    assert np.all(np.abs(point[H, 1, :]) < 0.02)     # urate returns to zero
    assert np.all(lo <= hi + 1e-12)
    # the urate band is a band of the raw response: it shrinks towards zero
    assert np.all(np.abs(lo[H, 1, :]) < 0.05) and np.all(np.abs(hi[H, 1, :]) < 0.05)


def test_bq_canonical_docstring_says_only_gdp_is_cumulated():
    import puremacro.teaching.bq_canonical as m

    assert "Only the GDP responses" in m.__doc__
    assert "cumulate=[0]" in m.__doc__


# --------------------------------------------------------------------------
# var.regime: girf on p = 0, wording
# --------------------------------------------------------------------------

def _hmm_data(T=400, seed=2):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    mu = np.array([[0.5, 0.2], [-1.0, -0.4]])
    L = [np.linalg.cholesky(np.array([[0.3, 0.1], [0.1, 0.4]])),
         np.linalg.cholesky(np.array([[1.5, 0.5], [0.5, 1.0]]))]
    return np.stack([mu[s[t]] + L[s[t]] @ rng.standard_normal(2) for t in range(T)])


def test_girf_on_p0_ms_fit_is_impact_only():
    from puremacro.var.regime import girf, ms_var_fit

    Y = _hmm_data()
    fit = ms_var_fit(Y, K=2, p=0, n_iter=200)
    assert fit.A.shape == (2, 0)
    res = girf(fit, Y, shock=0, horizon=6, n_hist=10, n_sim=20,
               shock_size=[1.0, -2.0], n_boot=20, rng=1)
    assert res.girf_by_regime.shape == (2, 2, 7, 2)
    for k in range(2):
        chol = np.linalg.cholesky(fit.Sigma[k])
        for s, d in enumerate([1.0, -2.0]):
            np.testing.assert_allclose(res.girf_by_regime[k, s, 0],
                                       d * chol[:, 0], rtol=1e-10, atol=1e-12)
    assert np.all(res.girf_by_regime[:, :, 1:, :] == 0.0)


def test_girf_p1_ms_fit_unchanged_by_p0_guard():
    """The p>0 path still shifts the lag buffer (a response beyond h=0)."""
    from puremacro.var.regime import girf, ms_var_fit

    Y = _hmm_data()
    fit = ms_var_fit(Y, K=2, p=1, n_iter=100)
    res = girf(fit, Y, shock=0, horizon=4, n_hist=5, n_sim=10, n_boot=10, rng=0)
    np.testing.assert_allclose(
        res.girf_by_regime[0, 0, 1],
        fit.A @ res.girf_by_regime[0, 0, 0], rtol=1e-10, atol=1e-12)


def test_regime_docstrings_name_msih_not_hamilton():
    import importlib

    import puremacro.var.regime as reg

    # ``puremacro.var.regime.girf`` is shadowed by the re-exported function
    g = importlib.import_module("puremacro.var.regime.girf")

    assert "Hamilton 1989 teaching spec" not in g.__doc__
    assert "MSIH spec in Krolzig 1997" in " ".join(g.__doc__.split())
    assert "MSIH Markov-switching VAR with shared AR matrix" in " ".join(reg.__doc__.split())
    assert "ECM" in reg.__doc__


def test_ms_var_example_is_not_called_hamilton_style():
    from pathlib import Path

    import puremacro.examples.ms_var_business_cycle as ex

    assert "Hamilton (1989)-style" not in ex.__doc__
    assert "al estilo Hamilton" not in ex.__doc__
    src = Path(ex.__file__).read_text(encoding="utf-8")
    assert "Hamilton-style MS-VAR" not in src
    cat = (Path(ex.__file__).parent / "EXAMPLES_CATALOG.md").read_text(encoding="utf-8")
    assert "Hamilton (1989)-style" not in cat
    assert "Lag-augmented LP (Plagborg-Møller-Wolf 2021)" not in cat


# --------------------------------------------------------------------------
# LPResult metadata, lp_hac attribution, tool docstring
# --------------------------------------------------------------------------

def _lp_frame(T=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.5 * x[t] + rng.standard_normal()
    return pd.DataFrame({"y": y, "x": x})


def test_la_lp_lag_spec_survives_pandas_operations():
    from puremacro.lp import la_lp

    res = la_lp(_lp_frame(), y="y", x="x", horizons=range(0, 6), n_lags=3,
                extra_lags=2, control_lags=1)
    derived = {
        "set_index": res.set_index("h", drop=False),
        "copy": res.copy(),
        "iloc": res.iloc[:3],
        "mask": res[res["beta"] > -np.inf],
        "sort": res.sort_values("beta"),
    }
    for name, obj in derived.items():
        assert (obj.n_lags, obj.extra_lags, obj.control_lags) == (3, 2, 1), name


def test_lp_hac_docstring_does_not_credit_pmw_for_the_bandwidth():
    from puremacro.lp.jorda import lp_hac

    doc = " ".join(lp_hac.__doc__.split())
    assert "Plagborg-Møller-Wolf 2021 recommendation" not in doc
    assert "truncation lag ``h + 1``" in doc
    assert "No reference is claimed" in doc


def test_spec_curve_tool_credits_mopm():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "tools/run_uncertainty_ident_spec_curve.py").read_text(encoding="utf-8")
    assert "Plagborg-Moller-Wolf (2021) lag-augmented LP" not in src
    assert "Montiel Olea & Plagborg-Moller (2021) lag-augmented LP" in src
