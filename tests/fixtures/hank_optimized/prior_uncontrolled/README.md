# Initial timing observations

These are the original three-repetition measurements before Accelerate's
threading mode was explicitly pinned. Numerical solves and path comparisons
passed; the timings describe the then-current scheduling policy and are not
verified single-thread measurements. `runner-used.py.txt` is the exact original
runner, including its incomplete single-thread guidance. Its hash matches the
runner entry in this directory's `record.json`.

The parent directory contains the controlled replacement and explains its
verification. These original observations are retained, not used for the
current speed comparison. Source hashes here identify the historical run;
the current-source checks apply to the parent record.
