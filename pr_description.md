🔒 Fix insecure deserialization vulnerability in disk_cache

🎯 **What:** The vulnerability fixed
Replaced `pickle` with `json` serialization in `puremacro/cache.py`'s `disk_cache` function. The cache files now use a `.json` extension instead of `.pkl`.

⚠️ **Risk:** The potential impact if left unfixed
Insecure Deserialization via `pickle.load`. An attacker who can write to the cache directory (`~/.cache/puremacro/` by default) could tamper with `.pkl` files to execute arbitrary code when a victim application retrieves the cached data. The blast radius would involve local code execution with the permissions of the user running the process.

🛡️ **Solution:** How the fix addresses the vulnerability
The `json` module is a safe serialization format that does not allow arbitrary code execution upon loading data, completely neutralizing the insecure deserialization risk. `json` also handles the types (dicts, lists) specified in the docstring correctly.
