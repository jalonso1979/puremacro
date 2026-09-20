import re

with open("puremacro/fetch/emissions.py", "r") as f:
    content = f.read()

# Fix 1: preserve null errors
SEARCH1 = """    # 2. Filter OBS_VALUE
    sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    sub = sub.dropna(subset=["OBS_VALUE"])"""
REPLACE1 = """    # 2. Filter OBS_VALUE
    # To match original semantics: if pd.isna(raw_val) it was skipped. If float() failed, skipped.
    # errors='coerce' forces failures to NaN.
    sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    sub = sub.dropna(subset=["OBS_VALUE"])"""

if SEARCH1 in content:
    content = content.replace(SEARCH1, REPLACE1)

SEARCH2 = """        # Value resolution
        if "value" not in batch_df.columns:
            continue
        batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
        batch_df = batch_df.dropna(subset=["value"])"""
REPLACE2 = """        # Value resolution
        if "value" not in batch_df.columns:
            continue
        # We must skip naturally missing values as well as unparseable ones.
        batch_df["value"] = pd.to_numeric(batch_df["value"], errors="coerce")
        batch_df = batch_df.dropna(subset=["value"])"""

if SEARCH2 in content:
    content = content.replace(SEARCH2, REPLACE2)

with open("puremacro/fetch/emissions.py", "w") as f:
    f.write(content)
