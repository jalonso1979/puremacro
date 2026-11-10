# Bounded second-order HANK benchmark

This record belongs to the source commit in `provenance.json`. It covers small
one-asset calibrations with 12 or 20 asset points and two income states, retaining
all marginal controls and the full constrained distribution basis.

Reproduce with:

```sh
python tools/reference_validation/benchmark_hank_second_order.py --output /tmp/hank-second-order --references
```

`record.json` retains all 24 signed/zero amplitude cases, six quadrature checks,
four seeded 100-period held-out paths, and ten independent finite nonlinear
reference solves. Two 12-asset references fail the unchanged 1e-9 tolerance at
an approximately 1.47e-9 equilibrium floor. Eight 20-asset references converge.
A reference solver failure remains a failure even if its candidate is close.

`coefficients-and-paths.npz` preserves curvature tensors, variance corrections,
mean decompositions, and held-out state paths. CSV files expose the amplitude
and finite-reference comparisons; the PNG plots first/second-order equation
errors. Artifact and source hashes are retained. Wall times are descriptive,
not controlled performance claims.

The local signed-shock errors decrease approximately cubically above their
numerical floor. Euler/KKT errors at 20 asset points remain around 0.004;
small discrete-equation residuals do not certify household-grid convergence.
Finite stochastic references use a first-order terminal boundary and remain
horizon sensitive. This evidence does not establish infinite-tail, global,
reduced-basis, two-asset, capital, or welfare accuracy. See
`docs/hank_second_order.md` and `docs/es/hank_second_order.md` for the API and
interpretation.
