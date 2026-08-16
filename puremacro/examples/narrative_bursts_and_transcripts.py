"""Narrative Macroeconomics: Transcripts, Dynamic Topics, Burst Anomalies, and SVARs.

Demonstrates the puremacro v1.0.0 end-to-end narrative econometrics workflow:
1. Multi-Speaker Central Bank Press Conference Parsing (Chair Guidance vs. Press Questions).
2. Pure-Python Dynamic Topic Modeling (NMF time-varying topic shares).
3. Narrative Burst Anomaly Detection (Kleinberg z-score frequency surges).
4. Bayesian Narrative SVAR with Ludvigson-Ma-Ng magnitude shock bounds.

Outputs:
  puremacro/examples/output/narrative_bursts_and_transcripts.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.narrative import (
    DynamicTopicModel,
    detect_narrative_bursts,
    score_spanish_macro_sentiment,
)
from puremacro.narrative.sources import parse_transcript
from puremacro.var.identify import NarrativeRestriction, narrative_sign_svar


def main():
    print("--- 1. Parsing Central Bank Multi-Speaker Press Conference ---")
    raw_transcript = """
    FEDERAL RESERVE PRESS CONFERENCE
    WASHINGTON, D.C.
    
    CHAIR POWELL: Good afternoon. My colleagues and I remain firmly committed to bringing inflation back down to our 2 percent goal. We decided today to raise the target range for the federal funds rate by 25 basis points and continue reducing our securities holdings. The labor market remains extremely tight, with wage growth elevated.
    
    QUESTION AND ANSWER PERIOD
    
    REPORTER NICK: Nick Timiraos, Wall Street Journal. Chair Powell, how do recent banking sector stresses affect your trajectory for additional rate hikes?
    
    CHAIR POWELL: Thanks Nick. We will closely monitor incoming credit conditions. Inflation pressures remain persistent, and restoring price stability is essential.
    
    REPORTER STEVE: Steve Liesman, CNBC. Do you see a path to a soft landing without significant unemployment spikes?
    
    CHAIR POWELL: A soft landing remains plausible, but softening labor market conditions and anchored inflation expectations will be critical.
    """
    doc = parse_transcript(raw_transcript, title="FOMC Press Conference", date="2023-03-22", institution="FED")
    print(f"Document parsed: {doc.title} ({doc.institution}) with {len(doc.turns)} speaker turns.")
    print(f"Prepared statement length: {len(doc.opening_statement().split())} words.")
    print(f"Chair Q&A response length: {len(doc.chair_qa_text().split())} words.")

    print("\n--- 2. Fitting Pure-Python Dynamic Topic Model ---")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    
    corpus_templates = [
        "Inflation surge driven by energy prices and supply chain bottlenecks.",
        "Central bank hikes policy rate to anchor long term inflation expectations.",
        "Labor market conditions remain tight with low unemployment and wage pressure.",
        "Financial market stress and credit standards tightening across regional banks.",
        "Economic growth slowing down as restrictive monetary policy cools demand.",
        "Consumer spending resilient but household credit balances expanding rapidly.",
    ]
    
    dated_corpus = []
    for d in dates:
        # 4 documents per month
        for _ in range(4):
            idx = rng.choice(len(corpus_templates))
            text = corpus_templates[idx]
            # Add some burst in late 2023 for financial stress
            if d >= pd.Timestamp("2023-03-01") and d <= pd.Timestamp("2023-06-01"):
                text += " Bank run liquidity pressures and emergency discount window borrowing."
            dated_corpus.append((d, text))

    texts_only = [t[1] for t in dated_corpus]
    dates_only = [t[0] for t in dated_corpus]
    
    dtm = DynamicTopicModel(n_topics=3, random_state=42)
    dtm_res = dtm.fit_transform_corpus(texts_only, dates_only, freq="MS")
    print(f"Dynamic topics extracted across {len(dtm_res.topic_shares)} months.")
    for top_k, kw in dtm_res.topic_keywords.items():
        kw_str = ", ".join(w for w, _ in kw[:4])
        print(f"  {top_k}: {kw_str}")

    print("\n--- 3. Detecting Statistical Narrative Burst Anomalies ---")
    target_date = "2023-03-01"
    bursts = detect_narrative_bursts(
        dated_corpus,
        target_date=target_date,
        window_periods=12,
        freq="MS",
        min_count=2,
    )
    print(f"Detected {len(bursts)} narrative bursts on {target_date}:")
    for b in bursts[:5]:
        print(f"  Term: {b.term:15s} | Z-Score: {b.z_score:+5.2f} | Surge: {b.burst_magnitude:4.1f}x")

    print("\n--- 4. Estimating Bayesian Narrative SVAR with Shock Bounds ---")
    # Synthetic 2-variable macro system (Fed funds rate, Inflation)
    T = 120
    e = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.7 * Y[t-1, 0] + 0.1 * Y[t-1, 1] + e[t, 0]
        Y[t, 1] = 0.2 * Y[t-1, 0] + 0.6 * Y[t-1, 1] + 0.5 * e[t, 0] + e[t, 1]

    # Narrative restriction: shock 0 on date 21 had positive sign and magnitude >= 1.5 std
    restr = [
        NarrativeRestriction(
            kind="shock_bound",
            date=21,
            shock=0,
            min_magnitude=0.5,
            sign=+1,
        )
    ]
    sign_matrix = {0: np.array([[1, 0], [1, 1]])}
    svar_res = narrative_sign_svar(
        Y,
        p=1,
        horizon=10,
        sign_matrix=sign_matrix,
        restrictions=restr,
        bayes_draws=True,
        n_draws=500,
        seed=42,
    )
    print(f"Bayesian Narrative SVAR accepted {svar_res.n_narrative_accepted} posterior draws.")

    print("\n--- 5. Generating Publication Figure ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "narrative_bursts_and_transcripts.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Transcript Speaker Word Share
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    speaker_counts = {}
    for t in doc.turns:
        speaker_counts[t.speaker] = speaker_counts.get(t.speaker, 0) + len(t.text.split())
    speakers = list(speaker_counts.keys())
    words = list(speaker_counts.values())
    bars = ax_a.bar(speakers, words, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="#333", alpha=0.85)
    ax_a.set_title("(a) FOMC Transcript Speaker Breakdown", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Word Count")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{int(yval)} w", ha='center', va='bottom', fontsize=9)

    # Panel B: Dynamic Topic Proportions Over Time
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    dtm_res.topic_shares.plot(ax=ax_b, lw=2)
    ax_b.set_title("(b) Pure-Python NMF Dynamic Topic Trajectories", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Monthly Topic Share")
    ax_b.legend(title="Topics", framealpha=0.9, fontsize=8)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Top Narrative Bursts
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    if bursts:
        top_b = bursts[:5]
        terms = [b.term for b in top_b][::-1]
        z_scores = [b.z_score for b in top_b][::-1]
        colors = ["#d62728" if z > 2.0 else "#1f77b4" for z in z_scores]
        ax_c.barh(terms, z_scores, color=colors, edgecolor="#333", alpha=0.85)
        ax_c.axvline(2.0, color="gray", linestyle="--", alpha=0.7, label="Surge Threshold (z=2.0)")
    ax_c.set_title(f"(c) Narrative Anomaly Bursts ({target_date})", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Frequency Z-Score Spike")
    ax_c.legend(loc="lower right", fontsize=8)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Bayesian Narrative SVAR IRF with Bounds
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    h = np.arange(svar_res.irf_median.shape[0])
    irf_pt = svar_res.irf_median[:, 1, 0]
    irf_lo = svar_res.irf_lower[:, 1, 0]
    irf_hi = svar_res.irf_upper[:, 1, 0]
    ax_d.plot(h, irf_pt, color="#1f77b4", lw=2, label="Bayesian Median IRF")
    ax_d.fill_between(h, irf_lo, irf_hi, color="#1f77b4", alpha=0.25, label="90% Posterior Credible Band")
    ax_d.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_d.set_title("(d) Response to Monetary Shock (Ludvigson-Ma-Ng Bound)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Horizon (Months)")
    ax_d.set_ylabel("Response")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to: {out_file}")


if __name__ == "__main__":
    main()
