"""Publication + teaching report outputs for NarrativeSpecificationCurve."""
from __future__ import annotations

import pandas as pd


class NarrativeReport:
    """Wrap a spec-curve results DataFrame and produce tables + figures.

    Parameters
    ----------
    results : DataFrame returned by ``NarrativeSpecificationCurve.run()``.
    """

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results

    # ------------------------------------------------------------------ #
    # Publication outputs                                                  #
    # ------------------------------------------------------------------ #

    def to_latex_table_firststage(self) -> str:
        """First-stage F table: rows by target, columns by statistic."""
        df = self.results.copy()
        rows = []
        for target in sorted(df["target"].unique()):
            sub = df[df["target"] == target]
            rows.append({
                "Target":             target.capitalize(),
                "N specs":            len(sub),
                "Median CD-F":        round(float(sub["first_stage_f_cd"].median()), 2),
                "Median KP-F":        round(float(sub["first_stage_f_kp"].median()), 2),
                r"\% F > 10":         round(100 * float((sub["first_stage_f_cd"] > 10).mean()), 1),
                "Median $n_{obs}$":   int(sub["n_obs"].median()),
                "Median $n_{nz}$":    int(sub["n_nonzero_quarters"].median()),
            })
        tbl = pd.DataFrame(rows).set_index("Target")
        header = " & ".join(tbl.columns) + r" \\"
        lines = [
            r"\begin{tabular}{l" + "r" * len(tbl.columns) + "}",
            r"\toprule",
            "Target & " + header,
            r"\midrule",
        ]
        for idx, row in tbl.iterrows():
            lines.append(idx + " & " + " & ".join(str(v) for v in row.values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def to_latex_table_speccurve(self) -> str:
        """Spec-curve summary: rows by specification dimension."""
        df = self.results.copy()
        dim_cols = ["source_subset", "aggregation", "time_window",
                    "surprise_filter", "harmonized"]
        rows = []
        for dim in dim_cols:
            if dim not in df.columns:
                continue
            for val in sorted(df[dim].unique()):
                sub = df[df[dim] == val]
                rows.append({
                    "Dimension":   dim,
                    "Value":       str(val),
                    "Median F_{CD}": round(float(sub["first_stage_f_cd"].median()), 2),
                    r"\% F > 10":  round(100 * float((sub["first_stage_f_cd"] > 10).mean()), 1),
                    "IQR β_{h4}":  round(float(sub["beta_2sls_h4"].quantile(0.75)
                                              - sub["beta_2sls_h4"].quantile(0.25)), 3),
                })
        tbl = pd.DataFrame(rows)
        if tbl.empty:
            return r"\begin{tabular}{l}\toprule No data.\\\bottomrule\end{tabular}"
        lines = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            "Dimension & Value & Median $F_{CD}$ & \\% $F>10$ & IQR $\\hat{\\beta}_{h4}$ \\\\",
            r"\midrule",
        ]
        for _, row in tbl.iterrows():
            lines.append(
                " & ".join(str(v) for v in row.values) + r" \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Publication figures                                                  #
    # ------------------------------------------------------------------ #

    def plot_f_violin(self, *, figsize=(8, 4)):
        """Violin plot of F-statistic distribution by target."""
        import matplotlib.pyplot as plt

        df = self.results.dropna(subset=["first_stage_f_cd"])
        targets = sorted(df["target"].unique())
        fig, ax = plt.subplots(figsize=figsize)
        data = [df[df["target"] == t]["first_stage_f_cd"].values for t in targets]
        vp = ax.violinplot(data, positions=range(len(targets)), showmedians=True)
        ax.axhline(10, color="red", linestyle="--", linewidth=0.8, label="F = 10 (Stock-Yogo)")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([t.capitalize() for t in targets])
        ax.set_ylabel("Cragg-Donald F statistic")
        ax.set_title("First-stage F distribution across specifications")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def plot_beta_funnel(self, *, figsize=(8, 4)):
        """Doucouliagos-Stanley funnel: β_2SLS vs 1/SE."""
        import matplotlib.pyplot as plt

        df = self.results.dropna(subset=["beta_2sls_h4", "se_2sls"])
        df = df[df["se_2sls"] > 0].copy()
        df["precision"] = 1.0 / df["se_2sls"]

        targets = sorted(df["target"].unique())
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax = plt.subplots(figsize=figsize)
        for i, t in enumerate(targets):
            sub = df[df["target"] == t]
            ax.scatter(sub["beta_2sls_h4"], sub["precision"],
                       label=t.capitalize(), alpha=0.5, s=18,
                       color=colors[i % len(colors)])
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel(r"$\hat{\beta}_{2SLS}$ (h=4)")
        ax.set_ylabel(r"Precision $1/SE$")
        ax.set_title("Funnel plot of 2SLS estimates")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Teaching figures                                                     #
    # ------------------------------------------------------------------ #

    def plot_event_density(self, panel, *, figsize=(10, 4)):
        """Stacked bar: events per quarter, colored by target."""
        import matplotlib.pyplot as plt

        long = panel.to_long()
        if long.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("No events")
            return fig
        # String labels ("2010Q1") keep the bar chart categorical: pandas 3
        # routes a freq-inferable DatetimeIndex through its period converter,
        # which raises on bar plots ("Must supply freq for datetime value").
        long["quarter"] = pd.PeriodIndex(long["date"], freq="Q").astype(str)
        counts = (long.groupby(["quarter", "target"])
                      .size()
                      .unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=figsize)
        counts.plot(kind="bar", stacked=True, ax=ax, width=0.9)
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Event count")
        ax.set_title("Narrative event density by target")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        return fig

    def plot_spec_overlay(self, *, n: int = 50, horizon_col: str = "beta_2sls_h4",
                          figsize=(8, 4)):
        """Overlay n randomly sampled spec β estimates."""
        import matplotlib.pyplot as plt

        df = self.results.dropna(subset=[horizon_col])
        sample = df.sample(min(n, len(df)), random_state=0) if len(df) > n else df
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(range(len(sample)), sorted(sample[horizon_col]),
                   alpha=0.4, s=12, color="steelblue")
        ax.axhline(float(df[horizon_col].median()), color="red",
                   linestyle="--", linewidth=1.0, label="Median β")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Specification (sorted by β)")
        ax.set_ylabel(r"$\hat{\beta}_{2SLS}$")
        ax.set_title(f"Robustness: {len(sample)} specifications")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


__all__ = ["NarrativeReport"]
