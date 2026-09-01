## 2024-05-18 - [Vectorized Matrix Solving with NumPy]
**Learning:** For batched log-likelihood or matrix solve operations in loops over a time dimension, NumPy provides fully vectorized primitives (e.g. `np.linalg.slogdet`, `np.linalg.solve`). `np.linalg.solve` handles stacks of matrices natively when broadcasting a matching right-hand side.
**Action:** Always prefer batched `np.linalg` functions over python loops in high-frequency mathematical evaluations.
