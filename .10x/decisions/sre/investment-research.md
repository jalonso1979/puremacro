# Operational boundary

No service, migration, broker action or scheduled task is introduced. Numerical
APIs operate on caller-supplied frames and arrays; callbacks are explicitly caller
code. Seeds, execution lag, costs, eligibility and assumptions remain explicit.
Missing held observations, unusable covariance, failed optimization and divergent
belief dynamics raise rather than produce plausible fallback outputs.

Replay preserves observed trading dates and timezones. Custom holiday frequency
rules are not reconstructed from generic aliases. Financial returns must already
use a common currency and the chosen corporate-action/distribution convention.
No claim is made about handling delisted assets without supplied data.

The deliverable is a local feature branch with numerical and documentation
evidence. No main merge, tag, deployment or publication is part of this task.
