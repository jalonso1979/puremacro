# Observed-household policy and conditional trade-adjustment evidence

Generated on 2026-11-01 from the installed 6.1.0 development wheel with
`tools/reference_validation/check_household_trade.py`. Package files actually
imported, authenticated ENIGH inputs, environment and all output checksums are
recorded in `record.json`. The source payload is packaged once under
`puremacro/trade/_datafiles`; it is not duplicated here.

Reproduce from a source checkout into a new directory:

```sh
PYTHONPATH=. python tools/reference_validation/check_household_trade.py --output new-evidence
```

The static study includes 90,033 positive-spending observed households, four
policies and nine price/preference states. The dynamic study includes two
policies and eight tariff/adjustment/utility-curvature states. Both use explicit
balanced lump-sum compensation. Survey sampling uncertainty, endogenous labor
and saving, and national general-equilibrium responses are outside the study.
No probabilities are assigned to the assumption grid.

The recorded independent two-good CES welfare gap is below 1.1e-15 when scaled
by each household's baseline budget. All four 12/24-quarter sourcing cost
comparisons pass the declared 1e-4 tolerance. These checks establish numerical
agreement for the specified experiments, not empirical validity of their
preference, price-transmission or fiscal assumptions.

The bottom-two-quintile objective ranks targeted compensation first in every
enumerated static and dynamic state. The all-household mean money-metric EV
objective ranks the zero-compensation control first in the static experiment.
These are different explicit objectives; the record does not name a universal
best policy. INEGI data attribution and exclusions are retained in the manifests.
