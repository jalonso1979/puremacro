# Productive-capital block reference

The original `labor`, `investment`, `dividend` and `partial_ss` functions in
the Sequence-Jacobian authors' `examples/two_asset.py` generate this fixture.
Source commit: `eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9`.
The complete source file and its MIT license are retained verbatim here.

Regenerate independently of PureMacro:

```sh
python tools/reference_validation/hank_capital_reference.py \
  --source-folder tests/fixtures/hank_capital_reference \
  --output /tmp/new-capital-reference.json
```

The generator selects four original functions from the AST, removes their SSJ
decorators, translates `x.apply(f)` into `f(x)`, and supplies explicit adjacent
dates. It never imports PureMacro or transcribes the author equations. Scalar
root finding solves the original investment and valuation conditions on the
published aggregate calibration and 24 fixed-seed off-steady cases.

This is an independent capital-block reference, not a replication of the
published full model. PureMacro uses direct ownership of productive capital,
flexible union hours, a different household portfolio adjustment cost, and a
two-state income process. The original firm dividend includes investment;
PureMacro separately distributes production profits and capital returns.
The accounting bridge is `author dividend = production profit + rental*K -
investment goods`. This bridge is tested separately from the block outputs.
