"""Tests for puremacro-dynare CLI runner."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.cli import create_parser, run_cli


def test_cli_parser_creation():
    parser = create_parser()
    assert parser.prog == "puremacro-dynare"
    args = parser.parse_args(["mymodel.mod", "--order", "2", "--irf", "20", "--fevd", "--plot"])
    assert args.model == "mymodel.mod"
    assert args.order == 2
    assert args.irf == 20
    assert args.fevd is True
    assert args.plot is True


def test_cli_file_not_found():
    code = run_cli(["non_existent_model_12345.mod", "--quiet"])
    assert code == 1


def test_cli_steady_only():
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    assert ref_mod.is_file()
    code = run_cli([str(ref_mod), "--steady-only", "--quiet"])
    assert code == 0


def test_cli_full_solve_and_export(tmp_path: Path):
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    out_dir = tmp_path / "results"
    
    code = run_cli([
        str(ref_mod),
        "--order", "1",
        "--irf", "10",
        "--fevd",
        "--outdir", str(out_dir),
        "--format", "all",
        "--quiet",
    ])
    assert code == 0
    
    # Check exported artifacts
    assert (out_dir / "model_moments.md").is_file()
    assert (out_dir / "model_moments.tex").is_file()
    assert (out_dir / "model_moments.typ").is_file()
    assert (out_dir / "irfs.csv").is_file()
    assert (out_dir / "fevd.md").is_file()
    assert (out_dir / "fevd.tex").is_file()
    assert (out_dir / "fevd.typ").is_file()
    assert (out_dir / "fevd.csv").is_file()
    
    # Check irfs.csv content
    df_irfs = pd.read_csv(out_dir / "irfs.csv", index_col=0)
    assert df_irfs.shape[0] == 11  # horizon 0 to 10


def test_cli_with_plots(tmp_path: Path):
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    out_dir = tmp_path / "plots_out"
    
    code = run_cli([
        str(ref_mod),
        "--order", "1",
        "--irf", "5",
        "--fevd",
        "--plot",
        "--outdir", str(out_dir),
        "--quiet",
    ])
    assert code == 0
    assert (out_dir / "irfs.png").is_file()
    assert (out_dir / "fevd.png").is_file()


def test_cli_with_shock_decomposition(tmp_path: Path):
    from puremacro.dsge import load_mod
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    m = load_mod(ref_mod)
    
    # Generate a small sample dataset matching observables
    np.random.seed(42)
    T_sim = 15
    data_dict = {v: np.random.randn(T_sim) * 0.01 for v in m.variables[:5]}
    data_df = pd.DataFrame(data_dict)
    csv_path = tmp_path / "data.csv"
    data_df.to_csv(csv_path)
    
    out_dir = tmp_path / "decomp_out"
    code = run_cli([
        str(ref_mod),
        "--shock-decomp", str(csv_path),
        "--outdir", str(out_dir),
        "--quiet",
    ])
    assert code == 0
    assert (out_dir / "shock_decomposition.md").is_file()
    assert (out_dir / "shock_decomposition.tex").is_file()
    assert (out_dir / "shock_decomposition.typ").is_file()
