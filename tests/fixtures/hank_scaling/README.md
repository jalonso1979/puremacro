# HANK state/control scaling evidence

Run `tools/reference_validation/validate_hank_scaling.py --output NEW_DIRECTORY`
with all four BLAS thread environment variables set to 1. Each configuration
runs in a fresh process; macOS Accelerate threading is also verified. The
record binds numerical sources, matrices, held-out paths and nonlinear audits
by SHA-256. Failed configurations remain in the same record.

The default 500×7 one-asset case has a 229-equation projected system and retains
all 3500 household grid points during backward steps and transport. Its first 30
aggregate impulse responses differ by 7.33e-8 or less from an independent
200-period sequence-space solution. This comparison has a finite horizon.

The 40×40 two-asset case with distribution rank 24, marginal rank 48 and Krylov
training horizon 96 has 78 equations. Its finer marginal rank 128 comparison has
158 equations. Rank 80 at the same training horizon fails the real-policy
check, and the low-rank default one-asset case fails determinacy. These are
observed configurations, not universal safe rank thresholds.

All cases use the same independent 48-date innovation sequence (seed 9917,
standard deviation 1e-4), plus a quarter-amplitude nonlinear audit. Larger-shock
nonlinear audits fail even when linear projected equations and control-rank
comparisons are accurate. The default 500×7 case also misses the 1e-5 market
criterion at quarter amplitude; that failure is retained. Correcting these
paths requires a nonlinear solve, not relabeling projected convergence.

Peak RSS is measured at the end of the solve and over the full worker lifetime;
the latter includes audits and any sequence-space reference. It includes
imports/calibration, so it is not the memory of the Schur matrices alone.
Stored matrix bytes are reported separately. Times are observations from a
host also running regression checks, not exclusive-machine speed guarantees.
Two-asset grid changes recalibrate asset supplies; they are sensitivity
observations, not a fixed-supply household-grid convergence certificate.
