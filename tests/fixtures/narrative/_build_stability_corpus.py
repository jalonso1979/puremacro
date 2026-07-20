"""Deterministic 200-doc fixture corpus for the α.2 stability test.
Run once and commit the resulting .parquet."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "stability_corpus.parquet"
SEED = 0
N_DOCS = 200

BANKS = ["FED", "ECB", "BOJ", "PBOC", "RBA", "BCB", "BANXICO", "BIS"]
COUNTRIES = ["USA", "EA20", "JPN", "CHN", "AUS", "BRA", "MEX", "MULTI"]

UNCERTAINTY = ["uncertain", "uncertainty", "downside risk", "ambiguous",
               "unclear", "concern", "volatile", "headwind"]
LABOR = ["labor market", "employment", "workforce", "jobs", "unemployment",
         "hiring", "wage growth"]
TECH = ["automation", "artificial intelligence", "robotics", "digitisation",
        "machine learning", "AI adoption"]


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    quarters = pd.date_range("2018Q1", "2023Q4", freq="QS")
    for i in range(N_DOCS):
        bank_ix = i % len(BANKS)
        bank = BANKS[bank_ix]
        country = COUNTRIES[bank_ix]
        q = quarters[rng.randrange(len(quarters))]
        n_unc = rng.randint(0, 3)
        n_lab = rng.randint(0, 3)
        n_tech = rng.randint(0, 2)
        words = (rng.sample(UNCERTAINTY, n_unc) +
                 rng.sample(LABOR, n_lab) +
                 rng.sample(TECH, min(n_tech, len(TECH))) +
                 ["central bank", "policy", "outlook"] * 3)
        rng.shuffle(words)
        text = " ".join(words)
        rows.append({
            "date": q, "text": text,
            "url": f"https://example.org/{i}",
            "meta": {"bank_code": bank, "country": country,
                     "language": "en", "doctype": "speech"},
        })
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT} with {N_DOCS} docs")


if __name__ == "__main__":
    main()
