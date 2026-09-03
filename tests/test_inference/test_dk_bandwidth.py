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
    import puremacro.inference.dk as dk_mod

    seen: list[int] = []
    real = dk_mod.driscoll_kraay

    def spy(score, time_keys, lags):
        seen.append(int(lags))
        return real(score, time_keys, lags)

    monkeypatch.setattr(dk_mod, "driscoll_kraay", spy)

    n_periods = 100
    df = _panel(n_units, n_periods)
    panel_lp_dk(df, y="y", x="x", horizons=[1], n_lags=2)

    assert seen, "driscoll_kraay was never called — the spy did not attach"
    expected = max(1, int(round(4 * (n_periods / 100) ** (2 / 9))))
    row_based = max(1, int(round(4 * (len(df) / 100) ** (2 / 9))))
    assert set(seen) == {expected}, (
        f"N={n_units}: bandwidth {sorted(set(seen))} != {expected} derived from "
        f"T={n_periods}. The panel-row rule would give {row_based}."
    )
