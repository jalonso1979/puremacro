## 2024-05-18 - [Security] Validate SSL connections
**Learning:** Hardcoding unverified contexts (`ssl._create_unverified_context()`) disables verification, which leaves API calls vulnerable to MITM attacks. If the primary OS-bundled certificates are not enough, standard practice is to use `certifi.where()` as the default context CA file, rather than disabling verification completely.
**Action:** When making HTTP requests where you encounter SSL errors, provide a valid CA bundle path (`ssl.create_default_context(cafile=certifi.where())`) instead of skipping verification.
