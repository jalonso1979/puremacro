# Empirical-to-structural parameter recovery

Simulated data generated from a closed-form New Keynesian equilibrium; LP moments are fitted by solving the model's equations with puremacro. This demonstrates the estimation workflow, not a historical empirical replication.

| parameter | estimate |       se | ci_lower | ci_upper | boundary | simulation_truth |
|-----------|----------|----------|----------|----------|----------|------------------|
|     sigma | 1.469875 | 0.059139 | 1.353964 | 1.585786 |    False |              1.5 |
|     kappa | 0.152528 | 0.004758 | 0.143203 | 0.161854 |    False |             0.15 |
|       rho | 0.594417 |  0.01293 | 0.569074 | 0.619759 |    False |              0.6 |

Joint moment covariance retained; local Jacobian rank 3/3. Horizons 0–4 fit the model; horizons 5–8 are descriptive checks. The manifest records the data hash, model, seed, shock normalization, and limitations.
