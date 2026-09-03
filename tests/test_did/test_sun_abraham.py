

# ---------------------------------------------------------------------------
# Regression test for the event-study band aggregation (fixed after 1.9.0).
# ---------------------------------------------------------------------------
def test_event_study_band_agrees_with_its_own_standard_error():
    """`lo`/`hi` must be built from the aggregated `se`, not averaged.

    `se` is aggregated correctly as sqrt(sum_i w_i^2 se_i^2) — the standard
    error of a weighted sum. The band used to be `sum_i w_i lo_i` and
    `sum_i w_i hi_i`, a weighted mean of the per-cohort interval edges. A
    weighted mean of standard errors is not the standard error of a weighted
    mean: with K equally weighted cohorts of equal precision the half-width
    comes out sqrt(K) times too large.

    That is exactly what was measured on this design — the ratio of the
    reported half-width to `z * se` was 1.73 where three cohorts contributed
    (sqrt(3) = 1.732) and 1.37 where two did (sqrt(2) = 1.414). Every affected
    row contradicted the `se` printed beside it, so the defect is visible
    without any external reference: the row is inconsistent with itself.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    from puremacro.did.sun_abraham import sun_abraham

    rng = np.random.default_rng(0)
    cohorts = {1: 6, 2: 10, 3: 14}
    rows = []
    for u in range(120):
        g = cohorts.get(u % 4, np.nan)      # u % 4 == 0 -> never treated
        a_i = rng.normal()
        for t in range(1, 21):
            eff = 2.0 if (not np.isnan(g) and t >= g) else 0.0
            rows.append({"unit": u, "time": t, "treat_time": g,
                         "y": a_i + 0.02 * t + eff + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)

    alpha = 0.10
    res = sun_abraham(df, unit="unit", time="time", outcome="y",
                      treat_time="treat_time", n_boot=300, alpha=alpha, seed=1)
    es = res.att_event_study
    z = float(norm.ppf(1.0 - alpha / 2.0))

    multi = es[(es["n_cohorts"] > 1) & np.isfinite(es["se"]) & (es["se"] > 0)]
    assert len(multi) >= 4, "fixture must produce event times with >1 cohort"
    for _, r in multi.iterrows():
        np.testing.assert_allclose(
            r["hi"] - r["att"], z * r["se"], rtol=1e-9,
            err_msg=(
                f"event_time {r['event_time']}: half-width {r['hi'] - r['att']:.5f} "
                f"disagrees with z*se {z * r['se']:.5f} over {int(r['n_cohorts'])} cohorts"
            ),
        )
        np.testing.assert_allclose(r["att"] - r["lo"], z * r["se"], rtol=1e-9)
