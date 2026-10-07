# puremacro.hfi — High-frequency identification of monetary policy

Surprise construction and Jarociński-Karadi (2020) decomposition.

For external-IV SVAR, pipe a surprise series into
`puremacro.var.identify.proxy.proxy_svar`. HFI does not duplicate that
machinery; it provides only the surprise- and shock-construction layer.

## Quick start

```python
import numpy as np
import pandas as pd
from puremacro.hfi import gk2015_surprise, aggregate_to_period, jk_median_target
from puremacro.var.identify.proxy import proxy_svar

# 1) Surprises from FFR-futures around announcements. CME quotes the contracts
#    as prices (100 - rate), so pass quote="price": the result is the
#    implied-rate surprise, positive = tightening. With the default
#    quote="rate" the inputs are read as implied rates, and prices would give
#    the surprise with the wrong sign.
surprise = gk2015_surprise(pre_prices, post_prices, days_remaining_in_month,
                           days_in_month=30, quote="price")

# 2) Optionally combine multiple contracts via Nakamura-Steinsson 2018 first PC
# surprise, loadings = ns2018_first_pc(contract_changes, scale_to_idx=0)

# 3) Aggregate to monthly for VAR. The default method="sum" adds each surprise
#    to its own month; method="gk2015" is GK's footnote-11 monthly average, for
#    a VAR in monthly-average rates (its index runs one month further).
z = aggregate_to_period(surprise, announce_dates, freq="M")

# 4) Proxy-SVAR with Olea-Pflueger first-stage F and bootstrap bands.
#    policy_idx is the column of Y_macro holding the policy rate: the F is
#    computed on its residual, and sign=+1 makes the shock raise it.
#    Y_macro's rows must line up with z's months.
res = proxy_svar(Y_macro, p=2, horizon=24,
                 instrument_series=z.values, n_boot=500, seed=0,
                 shock_target_idx=policy_idx, sign_var=policy_idx, sign=+1.0)
print(res.summary())

# 5) Optionally split into MP vs information shocks (JK 2020). The rate and
#    asset surprises must come from the same announcement windows, so
#    decompose before aggregating, then aggregate each shock.
mp_info = jk_median_target(rate_surprise=surprise,
                           asset_surprise=spx_window_change,
                           n_rotations=10_000, seed=0)
z_mp = aggregate_to_period(mp_info.mp_shock, announce_dates, freq="M")
```

## Public API

- `gk2015_surprise(pre, post, days_remaining, days_in_month=30, *, quote="rate")` — Gertler-Karadi 2015 month-end-adjusted FFR-futures surprise (eq. 19, footnote 6 factor `T/(T - t)`). Returns the implied-rate surprise, positive = tightening. `quote="rate"` reads the inputs as implied rates; `quote="price"` reads CME prices (`100 - rate`) and negates the price change.
- `ns2018_first_pc(surprise_matrix, scale_to_idx=0)` — Nakamura-Steinsson 2018 first PC of K policy contracts.
- `aggregate_to_period(surprises, dates, freq="M", method="sum")` — announcement-day surprises into monthly/quarterly bins, zero-filling empty periods. `method="sum"` adds each surprise to its own period; `method="gk2015"` is the GK (2015) footnote-11 monthly average (a surprise on day `k` of a `D`-day month puts `(D - k + 1)/D` of itself in that month and `(k - 1)/D` in the next).
- `jk_poor_man(rate_surprise, asset_surprise)` — JK 2020 sign-of-comovement decomposition.
- `jk_median_target(rate_surprise, asset_surprise, n_rotations=10_000, seed=None)` — JK 2020 median admissible-rotation decomposition.

## What's NOT here

- The full Bayesian sign-restriction variant of JK 2020 — not implemented.
- Real surprise series — none are shipped; bring your own (Gertler-Karadi public dataset, Nakamura-Steinsson replication files, etc.).
- External-IV SVAR machinery — composes on top of `puremacro.var.identify.proxy.proxy_svar`.

## References

- Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. AEJ:Macro 7(1), 44-76.
- Nakamura, E. and Steinsson, J. (2018). High-frequency identification of monetary non-neutrality. QJE 133(3), 1283-1330.
- Jarociński, M. and Karadi, P. (2020). Deconstructing monetary policy surprises — the role of information shocks. AEJ:Macro 12(2), 1-43.
- Olea, J.L.M. and Pflueger, C. (2013). A robust test for weak instruments. JBES 31(3), 358-369.
