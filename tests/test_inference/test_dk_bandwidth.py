"""The Driscoll-Kraay bandwidth must be a rule on periods, not on panel rows.

`driscoll_kraay` sums the scores across units first and then runs Newey-West
on the resulting length-T series, so the Bartlett bandwidth is a property of
the time dimension alone. `_focal_dk_se` derived it from `len(X)` — the panel
row count N*T — which inflates it by N**(2/9): at T = 100 the bandwidth went
7 -> 10 -> 13 as N went 10 -> 50 -> 200. Adding countries changed how many
quarters of autocorrelation the kernel believed the data had, and the
cross-section cannot carry that information: it has been summed away before
the kernel sees anything.

The bandwidth is captured rather than inferred from the standard error,
because the SE also falls with N for the ordinary reason that there is more
data — which would confound exactly the effect under test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.panel_dk import panel_lp_dk


def _panel(n_units: int, n_periods: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_periods).cumsum() * 0.2
    rows = []
    for i in range(n_units):
        a_i = rng.standard_normal()
        x = rng.standard_normal(n_periods)
        y = a_i + 0.5 * x + common + rng.standard_normal(n_periods) * 0.5
        for t in range(n_periods):
            rows.append({"code": i, "date": t, "y": y[t], "x": x[t]})
    return pd.DataFrame(rows).set_index(["code", "date"])


@pytest.mark.parametrize("n_units", [10, 50, 200])
def test_dk_bandwidth_depends_only_on_the_number_of_periods(monkeypatch, n_units):
    """The bandwidth must follow floor(4*(T/100)^(2/9)) on the T the kernel sees.

    T is taken from the time keys handed to `driscoll_kraay`, not from the raw
    panel: the horizon loop drops rows for lags and leads, so the estimation
    sample is shorter than the input. Deriving the expectation from what the
    kernel actually receives keeps the test honest about the rule while still
    pinning the thing that was wrong — the dependence on N.
    """
    import puremacro.inference.dk as dk_mod

    seen: list[tuple[int, int]] = []
    real = dk_mod.driscoll_kraay

    def spy(score, time_keys, lags):
        seen.append((int(lags), int(len(np.unique(time_keys)))))
        return real(score, time_keys, lags)

    monkeypatch.setattr(dk_mod, "driscoll_kraay", spy)

    n_periods = 100
    df = _panel(n_units, n_periods)
    panel_lp_dk(df, y="y", x="x", horizons=[1], n_lags=2)

    assert seen, "driscoll_kraay was never called — the spy did not attach"
    for lags, t_used in seen:
        expected = max(1, int(np.floor(4 * (t_used / 100) ** (2 / 9))))
        row_based = max(1, int(np.floor(4 * (len(df) / 100) ** (2 / 9))))
        assert lags == expected, (
            f"N={n_units}: bandwidth {lags} != {expected} for the T={t_used} "
            f"the kernel received. The panel-row rule would give {row_based}."
        )


def test_dk_bandwidth_is_identical_across_cross_section_sizes(monkeypatch):
    """Same T, different N: the kernel must be handed the same bandwidth.

    This is the property that failed. At T = 100 the bandwidth used to go
    7 -> 10 -> 13 as N went 10 -> 50 -> 200, because it was derived from the
    N*T row count. The cross-section has been summed away before the kernel
    sees anything, so it cannot carry information about time dependence.
    """
    import puremacro.inference.dk as dk_mod

    real = dk_mod.driscoll_kraay
    per_n = {}
    for n_units in (10, 50, 200):
        seen: list[int] = []

        def spy(score, time_keys, lags, _s=seen):
            _s.append(int(lags))
            return real(score, time_keys, lags)

        monkeypatch.setattr(dk_mod, "driscoll_kraay", spy)
        panel_lp_dk(_panel(n_units, 100), y="y", x="x", horizons=[1], n_lags=2)
        assert seen
        per_n[n_units] = sorted(set(seen))

    values = {tuple(v) for v in per_n.values()}
    assert len(values) == 1, (
        f"bandwidth varied with the cross-section size: {per_n}"
    )
