## 2026-08-31 - Memory reallocation in numpy simulation loops
**Learning:** Calling `np.concatenate` to manage rolling history buffers in tight simulation loops (like for Generalized IRF trajectories across `H` steps and `M` parallel histories) is a significant bottleneck due to constant reallocation and copying of the whole buffer, even though the arrays are relatively small per iteration. Pre-allocating the full future length works but increases peak memory. Simply slicing and re-assigning inplace (`buf[:, :-1] = buf[:, 1:]` and `buf[:, -1] = y`) gives ~15% speedup vs `np.concatenate` without the overhead and memory jump of full pre-allocation.
**Action:** When shifting time buffers inplace in high-throughput hot loops with Numpy, use slicing and inplace assignment instead of `np.concatenate` to minimize reallocation.

## 2024-10-04 - SQLite NumPy Types Incompatibility
**Learning:** `sqlite3.executemany` does not natively support NumPy numeric types like `np.float64` or `np.int64`, which are generated when vectorizing pandas operations. Attempting to insert them directly causes failures or silent truncation in the database driver.
**Action:** When vectorizing dataframes for SQLite insertion, ensure final column arrays are safely mapped back to native Python types (e.g., using `[float(x) if x is not None else None for x in arr]`) immediately before `zip` and insertion.
