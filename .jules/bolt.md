## 2024-05-19 - Pandas iterrows() bottleneck
**Learning:** The use of `iterrows()` in pandas is extremely slow for looping over rows and should be avoided for production operations. In `store_realtime_vintages` and `news.to_frame`, switching to vectorization and `pd.concat` yielded a ~20x performance improvement in benchmarks.
**Action:** Always prefer vectorization (`pd.to_numeric`, `pd.to_datetime`, column assignment) or list comprehension with `zip()` over `iterrows()` when processing DataFrames.
