"""puremacro-dynare — Drop-in command-line runner for Dynare .mod models.

Executes Dynare .mod macroeconomic models in pure Python with zero C++ compilation.
Computes steady states, 1st and 2nd order decision rules, IRFs, FEVD, historical
shock decompositions, and exports publication-ready LaTeX, Typst, and Markdown tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class DynareParser(argparse.ArgumentParser):
    """Dual-mode parser supporting both legacy model solving and parity verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parity_parser = argparse.ArgumentParser(
            prog="puremacro-dynare parity",
            description="Run automated parity verification against Dynare golden outputs.",
        )
        self._parity_parser.add_argument(
            "path",
            nargs="?",
            default=".",
            help="Path to .mod file, *_results.mat file, or directory of models. Default: .",
        )
        self._parity_parser.add_argument(
            "--order",
            type=int,
            choices=[1, 2],
            default=1,
            help="Perturbation order to verify (1 or 2). Default: 1.",
        )
        self._parity_parser.add_argument(
            "--tol",
            type=float,
            default=None,
            help="Maximum absolute deviation tolerance override.",
        )
        self._parity_parser.add_argument(
            "--format",
            choices=["markdown", "latex", "typst", "all", "text"],
            default="text",
            help="Output scorecard format. Default: text.",
        )
        self._parity_parser.add_argument(
            "--outdir",
            type=str,
            default=None,
            help="Directory to export scorecard artifacts.",
        )
        self._parity_parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress terminal summary output.",
        )
        self._parity_parser.set_defaults(command="parity")

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        if args and args[0] == "parity":
            return self._parity_parser.parse_args(args[1:], namespace=namespace)
        ns = super().parse_args(args, namespace=namespace)
        if not hasattr(ns, "command"):
            ns.command = "solve"
        return ns


def create_parser() -> argparse.ArgumentParser:
    parser = DynareParser(
        prog="puremacro-dynare",
        description="Run Dynare .mod macroeconomic models in pure Python without MATLAB/C++.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the Dynare .mod file (e.g. model.mod or sw07.mod).",
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=[1, 2],
        default=1,
        help="Perturbation order: 1 (linear Klein QZ) or 2 (pruned 2nd order). Default: 1.",
    )
    parser.add_argument(
        "--irf",
        type=int,
        default=40,
        help="Impulse response horizon in periods. Default: 40.",
    )
    parser.add_argument(
        "--fevd",
        action="store_true",
        help="Compute and export Forecast Error Variance Decomposition (FEVD).",
    )
    parser.add_argument(
        "--shock-decomp",
        type=str,
        default=None,
        metavar="DATA_CSV",
        help="Compute historical shock decomposition using observed series in DATA_CSV.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=0,
        help="Stochastic simulation length in periods (0 to skip stochastic simulation). Default: 0.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./dynare_results",
        help="Directory to save generated tables and figures. Default: ./dynare_results.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "typst", "all"],
        default="all",
        help="Publication export format for tables. Default: all.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save IRF and FEVD figures (PNG) in outdir.",
    )
    parser.add_argument(
        "--steady-only",
        action="store_true",
        help="Compute and display steady state without solving dynamic perturbation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress interactive terminal output.",
    )
    return parser


def run_parity_cli(args: argparse.Namespace) -> int:
    """Execute parity verification subcommand."""
    target_path = Path(args.path)
    if not target_path.exists():
        sys.stderr.write(f"Error: Path '{target_path}' not found.\n")
        return 1

    from puremacro.dsge.parity import compare_model_to_dynare, run_parity_suite

    if target_path.is_file():
        if target_path.suffix == ".mod":
            mat_candidates = [
                target_path.with_name(f"{target_path.stem}_results.mat"),
                target_path.with_suffix(".mat"),
                target_path.parent / "results" / f"{target_path.stem}_results.mat",
            ]
            mat_path = next((m for m in mat_candidates if m.exists()), None)
            if mat_path is None:
                sys.stderr.write(f"Error: No matching results .mat file found for {target_path}.\n")
                return 1
            res = compare_model_to_dynare(target_path, mat_path, order=args.order, tol=args.tol)
        elif target_path.suffix == ".mat":
            mod_candidates = [
                target_path.with_suffix(".mod"),
                target_path.with_name(f"{target_path.stem.replace('_results', '')}.mod"),
            ]
            mod_path = next((m for m in mod_candidates if m.exists()), None)
            if mod_path is None:
                sys.stderr.write(f"Error: No matching .mod file found for {target_path}.\n")
                return 1
            res = compare_model_to_dynare(mod_path, target_path, order=args.order, tol=args.tol)
        else:
            sys.stderr.write(f"Error: Unrecognized file type '{target_path.suffix}'; expected .mod or .mat\n")
            return 1
    else:
        res = run_parity_suite(target_path, order=args.order, tol=args.tol)

    if not args.quiet:
        sys.stdout.write(res.summary() + "\n")

    if args.outdir or (args.format and args.format != "text"):
        out_dir = Path(args.outdir) if args.outdir else Path("./dynare_parity_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.format in ("markdown", "all"):
            (out_dir / "parity_scorecard.md").write_text(res.to_markdown(), encoding="utf-8")
        if args.format in ("latex", "all"):
            (out_dir / "parity_scorecard.tex").write_text(res.to_latex(), encoding="utf-8")
        if args.format in ("typst", "all"):
            (out_dir / "parity_scorecard.typ").write_text(res.to_typst(), encoding="utf-8")

    return 0 if res.passed else 1


def _bk_status_line(model) -> str:
    """Blanchard-Kahn verdict read off the solved model, never assumed."""
    first = getattr(model, "first_order", None)
    src = first if first is not None else model
    solution = getattr(src, "solution", None)
    eu = tuple(getattr(solution, "eu", ())) if solution is not None else ()
    if eu == (1, 1):
        return "Blanchard-Kahn condition verified: unique stable rational expectations path."
    if eu == (1, 0):
        return "Blanchard-Kahn condition FAILED: indeterminacy (multiple stable solutions)."
    if eu == (0, 0):
        return "Blanchard-Kahn condition FAILED: no stable solution."
    return "Blanchard-Kahn status: not reported by this solution object."


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) == "parity":
        return run_parity_cli(args)

    mod_path = Path(args.model)
    if not mod_path.exists():
        sys.stderr.write(f"Error: Model file '{mod_path}' not found.\n")
        return 1

    from puremacro.dsge.dynare import load_mod
    from puremacro.dsge.klein import BlanchardKahnError

    if not args.quiet:
        sys.stdout.write("=" * 72 + "\n")
        sys.stdout.write(f"  puremacro-dynare: Solving {mod_path.name}\n")
        sys.stdout.write("=" * 72 + "\n")

    try:
        model = load_mod(mod_path, order=args.order)
    except BlanchardKahnError as exc:
        sys.stderr.write(f"Error solving model: {exc}\n")
        sys.stderr.write(
            "The Blanchard-Kahn condition fails, so there is no unique stable "
            "rational-expectations solution to report.\n"
        )
        return 3
    except Exception as exc:
        sys.stderr.write(f"Error parsing model: {exc}\n")
        return 2

    # Display steady state
    if not args.quiet:
        vars_list = list(getattr(model, "variables", []))
        shocks_list = list(getattr(model, "shocks", []))
        params_dict = (
            getattr(model, "parameters", None)
            or getattr(model, "params", None)
            or getattr(model, "_params", None)
            or {}
        )
        n_eqs = len(getattr(model, "equations", vars_list))
        sys.stdout.write("Model Summary:\n")
        sys.stdout.write(f"  Equations   : {n_eqs}\n")
        sys.stdout.write(f"  Endogenous  : {len(vars_list)}\n")
        sys.stdout.write(f"  Exogenous   : {len(shocks_list)}\n")
        sys.stdout.write(f"  Parameters  : {len(params_dict)}\n\n")

        sys.stdout.write("Steady State:\n")
        ss = getattr(model, "steady_state", {})
        if hasattr(ss, "items"):
            for var, val in sorted(ss.items(), key=lambda x: str(x[0])):
                sys.stdout.write(f"  {str(var):<20s} = {float(val):14.6f}\n")
        elif isinstance(ss, (list, tuple, np.ndarray)):
            for var, val in zip(vars_list, ss):
                sys.stdout.write(f"  {str(var):<20s} = {float(val):14.6f}\n")
        sys.stdout.write("\n")

    if args.steady_only:
        return 0

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Solve dynamic model
    if not args.quiet:
        sys.stdout.write(f"Solving dynamic perturbation at order {args.order}...\n")

    try:
        sim_res = model.stoch_simul(
            order=args.order,
            periods=args.periods,
            irf=args.irf,
        )
    except Exception as exc:
        sys.stderr.write(f"Error solving model: {exc}\n")
        return 3

    if not args.quiet:
        sys.stdout.write(f"{_bk_status_line(model)}\n")
        sys.stdout.write("Decision rules computed successfully.\n\n")

    # Export formats helper
    def write_exports(base_name: str, md_text: str, tex_text: str, typ_text: str):
        if args.format in ("markdown", "all"):
            (out_dir / f"{base_name}.md").write_text(md_text, encoding="utf-8")
        if args.format in ("latex", "all"):
            (out_dir / f"{base_name}.tex").write_text(tex_text, encoding="utf-8")
        if args.format in ("typst", "all"):
            (out_dir / f"{base_name}.typ").write_text(typ_text, encoding="utf-8")

    # 1. Summary tables
    write_exports(
        "model_moments",
        sim_res.to_markdown(),
        sim_res.to_latex(),
        sim_res.to_typst(),
    )

    # 2. IRF plots & data
    if sim_res.irfs:
        irf_df = pd.DataFrame(sim_res.irfs)
        irf_df.to_csv(out_dir / "irfs.csv", index=True)
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                fig = sim_res.plot(style="publication")
                if fig is not None:
                    fig.savefig(out_dir / "irfs.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
            except Exception as e:
                if not args.quiet:
                    sys.stdout.write(f"Warning saving plot: {e}\n")

    # 3. FEVD if requested
    if args.fevd:
        if not args.quiet:
            sys.stdout.write("Computing Forecast Error Variance Decomposition (FEVD)...\n")
        try:
            from puremacro.dsge.decomposition import compute_fevd
            fevd_res = compute_fevd(model, horizons=[1, 4, 8, 16, 32, None])
            write_exports(
                "fevd",
                fevd_res.to_markdown(),
                fevd_res.to_latex(),
                fevd_res.to_typst(),
            )
            fevd_res.table.to_csv(out_dir / "fevd.csv")
            if args.plot:
                import matplotlib.pyplot as plt
                fig_fevd = fevd_res.plot(style="publication")
                if fig_fevd is not None:
                    fig_fevd.savefig(out_dir / "fevd.png", dpi=300, bbox_inches="tight")
                    plt.close(fig_fevd)
            if not args.quiet:
                sys.stdout.write("FEVD exported successfully.\n")
        except Exception as exc:
            sys.stderr.write(f"Warning: Failed to compute FEVD: {exc}\n")

    # 4. Historical shock decomposition if requested
    if args.shock_decomp:
        decomp_path = Path(args.shock_decomp)
        if not decomp_path.exists():
            sys.stderr.write(f"Warning: Data file for shock decomposition '{decomp_path}' not found.\n")
        else:
            if not args.quiet:
                sys.stdout.write(f"Computing historical shock decomposition from {decomp_path.name}...\n")
            try:
                from puremacro.dsge.decomposition import compute_shock_decomposition
                data_df = pd.read_csv(decomp_path, index_col=0)
                decomp_res = compute_shock_decomposition(model, data_df)
                write_exports(
                    "shock_decomposition",
                    decomp_res.to_markdown(),
                    decomp_res.to_latex(),
                    decomp_res.to_typst(),
                )
                if not args.quiet:
                    sys.stdout.write("Historical shock decomposition exported successfully.\n")
            except Exception as exc:
                sys.stderr.write(f"Warning: Failed shock decomposition: {exc}\n")

    if not args.quiet:
        sys.stdout.write(f"\nAll results saved to: {out_dir.resolve()}\n")
        sys.stdout.write("=" * 72 + "\n")

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run_cli(argv))


if __name__ == "__main__":
    main()
