# Input and replay review

No credentials, remote services, broker orders or live-source cache writes are
introduced by the numerical APIs. Replay uses JSON and reruns validated numeric
inputs; it does not use pickle, eval or executable callbacks. The SHA-256 digest
detects accidental changes, not authenticity. Public callbacks run caller code
and are explicitly not sandboxed. Filing provenance is preserved, not verified.
The developer-only Pyodide runner mounts the explicitly selected source tree.

Tests use the repository's isolated cache/credential fixtures. Notebook kernels,
IPython configuration and validation logs use /private/tmp. No original-worktree
user changes were edited. Broad tests regenerate unrelated course-site hashes;
those generated changes must be restored before delivery.
