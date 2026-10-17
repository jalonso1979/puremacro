## 2026-08-31 - Memory reallocation in numpy simulation loops
**Learning:** Calling `np.concatenate` to manage rolling history buffers in tight simulation loops (like for Generalized IRF trajectories across `H` steps and `M` parallel histories) is a significant bottleneck due to constant reallocation and copying of the whole buffer, even though the arrays are relatively small per iteration. Pre-allocating the full future length works but increases peak memory. Simply slicing and re-assigning inplace (`buf[:, :-1] = buf[:, 1:]` and `buf[:, -1] = y`) gives ~15% speedup vs `np.concatenate` without the overhead and memory jump of full pre-allocation.
**Action:** When shifting time buffers inplace in high-throughput hot loops with Numpy, use slicing and inplace assignment instead of `np.concatenate` to minimize reallocation.

## 2024-05-19 - Vectorizing Monthly to Quarterly Transition Matrices
**Learning:** In `puremacro/labor_flows.py`, chaining monthly transition matrices using `iterrows()` inside a groupby is extremely slow due to object allocation overhead. We can get ~130x speedups by using `.values.reshape(N, 3, 3, 3)` and NumPy batch matrix multiplication (`@`).
**Action:** Always look for `groupby().head(N)` or strictly shaped continuous blocks of time series data when replacing `iterrows()` in mathematical chains; they easily map to `reshape` and batch multiplication.
