"""tools/build_sw07_data.py — build the bundled Smets-Wouters (2007) dataset
and the SW07 replication fixture (dev-only; not shipped in the wheel).

Two sub-commands::

    python tools/build_sw07_data.py data     [--cache DIR] [--out CSV]
    python tools/build_sw07_data.py fixture  [--out NPZ ...]

``data`` downloads the FRED series below as public ``fredgraph.csv`` files
with ``urllib`` (no API key, nothing written under ``data/raw``) and writes
``puremacro/dsge/_sw07_data.csv``. ``fixture`` optimises the SW07 posterior on
that CSV and writes the replication fixture (see :func:`build_fixture`).

Data definitions — Smets & Wouters (2007), ECB Working Paper 722, data
appendix (printed p. 47, PDF p. 48; https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf):

* Real GDP (chained dollars); nominal personal consumption expenditures and
  nominal fixed private investment deflated with the GDP deflator.
* Inflation: first difference of the log GDP implicit price deflator.
* Hours and wages: BLS nonfarm-business (NFB) sector, all persons. The real
  wage is NFB hourly compensation divided by the GDP deflator. Hours are
  "the index of average hours for the NFB sector multiplied with the
  Civilian Employment (16 years and over)".
* Aggregate real variables are per capita: divided by the civilian
  population 16+.
* "Consumption, investment, GDP, wages and hours are expressed in 100 times
  log"; interest rate (federal funds rate) and inflation are quarterly rates.

FRED identifiers used (quarterly averages of monthly series):

    gdp       GDPC1        Real GDP, chained dollars (SW07: GDPC96)
    cons_nom  PCEC         Personal consumption expenditures, nominal
    inv_nom   FPI          Fixed private investment, nominal
    deflator  GDPDEF       GDP implicit price deflator
    hours_avg PRS85006023  NFB sector: average weekly hours, all workers (index)
    emp       CE16OV       Civilian employment level, 16+ (monthly)
    wage_nom  COMPNFB      NFB sector: hourly compensation, all workers (index;
                           BLS PRS85006103)
    ffr       FEDFUNDS     Effective federal funds rate (monthly, % p.a.)
    pop       CNP16OV      Civilian noninstitutional population, 16+ (monthly)

Observables written (1966Q1-2004Q4, 156 quarters; the column names are the
ones ``puremacro.dsge.sw07_observation.OBSERVED_VARS`` expects):

    gdp_growth   100 * dlog(GDPC1 / CNP16OV)
    cons_growth  100 * dlog(PCEC / GDPDEF / CNP16OV)
    inv_growth   100 * dlog(FPI / GDPDEF / CNP16OV)
    wage_growth  100 * dlog(COMPNFB / GDPDEF)
    log_hours    100 * log(PRS85006023 * CE16OV / CNP16OV), minus its
                 1966Q1-2004Q4 mean (the level of an index is arbitrary; the
                 model intercept ``constelab`` absorbs any mean)
    infl         100 * dlog(GDPDEF)
    ffr          FEDFUNDS / 4

Through puremacro 4.3.0 the bundled file used real chained PCE (PCECC96),
real gross private domestic investment including inventories (GPDIC1), and
hours = log(HOANBS / CNP16OV) WITHOUT the factor 100, i.e. hours entered the
likelihood at 1/100 of the model's percent units.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "puremacro" / "dsge" / "_sw07_data.csv"
FIXTURE_NAME = "sw07_parity_seed0_200draws.npz"
# One copy only: the package-data file that the replication cases and
# tests/test_dsge/test_sw07_wrapper.py read through importlib.resources. (An
# identical second copy under tests/fixtures/ had no reader and was removed.)
DEFAULT_FIXTURES = (
    ROOT / "puremacro" / "replication" / "data" / FIXTURE_NAME,
)

SERIES = {
    "gdp":       "GDPC1",
    "cons_nom":  "PCEC",
    "inv_nom":   "FPI",
    "deflator":  "GDPDEF",
    "hours_avg": "PRS85006023",
    "emp":       "CE16OV",
    "wage_nom":  "COMPNFB",
    "ffr":       "FEDFUNDS",
    "pop":       "CNP16OV",
}

FIRST, LAST = pd.Period("1966Q1", "Q"), pd.Period("2004Q4", "Q")
_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


# --------------------------------------------------------------------- data --

def download_fred(fred_id: str, cache_dir: Path | None = None) -> pd.Series:
    """One FRED series as a float Series on a DatetimeIndex.

    Reads ``<cache_dir>/<fred_id>.csv`` when present; otherwise downloads the
    public fredgraph CSV (and stores it in ``cache_dir`` if given).
    """
    text = None
    cached = None if cache_dir is None else Path(cache_dir) / f"{fred_id}.csv"
    if cached is not None and cached.is_file():
        text = cached.read_text(encoding="utf-8")
    if text is None:
        req = urllib.request.Request(_FREDGRAPH.format(fred_id),
                                     headers={"User-Agent": "puremacro-dev-tools"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
        time.sleep(0.3)
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    s = pd.Series(df.iloc[:, 1].to_numpy(dtype=float),
                  index=pd.to_datetime(df.iloc[:, 0]), name=fred_id)
    return s.dropna()


def _quarterly(s: pd.Series) -> pd.Series:
    """Quarterly average on a quarterly PeriodIndex (identity for quarterly data)."""
    q = s.groupby(s.index.to_period("Q")).mean()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q


def transform(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """SW07 observables from the raw FRED levels (keys as in ``SERIES``)."""
    q = pd.concat({k: _quarterly(v) for k, v in raw.items()}, axis=1)
    q = q[(q.index >= FIRST - 1) & (q.index <= LAST)]
    if q.isna().any().any():
        raise ValueError(f"missing raw observations:\n{q[q.isna().any(axis=1)]}")

    def dlog100(x: pd.Series) -> pd.Series:
        return 100.0 * np.log(x / x.shift(1))

    out = pd.DataFrame(index=q.index)
    out["gdp_growth"] = dlog100(q["gdp"] / q["pop"])
    out["cons_growth"] = dlog100(q["cons_nom"] / q["deflator"] / q["pop"])
    out["inv_growth"] = dlog100(q["inv_nom"] / q["deflator"] / q["pop"])
    out["wage_growth"] = dlog100(q["wage_nom"] / q["deflator"])
    hours = 100.0 * np.log(q["hours_avg"] * q["emp"] / q["pop"])
    out["log_hours"] = hours
    out["infl"] = dlog100(q["deflator"])
    out["ffr"] = q["ffr"] / 4.0
    out = out[(out.index >= FIRST) & (out.index <= LAST)]
    out["log_hours"] = out["log_hours"] - out["log_hours"].mean()
    if len(out) != 156 or out.isna().any().any():
        raise ValueError(f"expected 156 complete quarters, got {len(out)}")
    return out


def write_csv(out: pd.DataFrame, path: Path) -> None:
    header = [
        "# puremacro SW07 dataset — 1966Q1 to 2004Q4 (156 quarterly obs)",
        "# Definitions: Smets & Wouters (2007), ECB WP 722 data appendix (printed p.47).",
        "# Source FRED series (fredgraph.csv; quarterly averages of monthly series):",
    ]
    for nm, fred_id in SERIES.items():
        header.append(f"#   {nm}: {fred_id}")
    header += [
        "# gdp_growth  = 100*dlog(GDPC1/CNP16OV)",
        "# cons_growth = 100*dlog(PCEC/GDPDEF/CNP16OV)",
        "# inv_growth  = 100*dlog(FPI/GDPDEF/CNP16OV)",
        "# wage_growth = 100*dlog(COMPNFB/GDPDEF)",
        "# log_hours   = 100*log(PRS85006023*CE16OV/CNP16OV), demeaned over the sample",
        "# infl        = 100*dlog(GDPDEF);  ffr = FEDFUNDS/4 (quarterly percent)",
        f"# Built: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')} by tools/build_sw07_data.py",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in header:
            fh.write(line + "\n")
        out.to_csv(fh, index_label="date", lineterminator="\n")


def build_data(out_path: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    raw = {name: download_fred(fid, cache_dir) for name, fid in SERIES.items()}
    out = transform(raw)
    write_csv(out, out_path)
    return out


# ------------------------------------------------------------------ fixture --

def _objective(csv_path: Path):
    sys.path.insert(0, str(ROOT))
    from puremacro.dsge.estimate import _OPT_PENALTY, _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    df = pd.read_csv(csv_path, comment="#", index_col="date")
    y = df[list(OBSERVED_VARS)].to_numpy()
    names = param_names()
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS)
    nlp_opt = _make_neg_log_posterior(y, make_state_space, PRIORS, names,
                                      _FIXED_PARAMS, penalty=_OPT_PENALTY)
    lb = np.array([PRIORS[n]["lb"] for n in names])
    ub = np.array([PRIORS[n]["ub"] for n in names])
    return names, nlp, nlp_opt, lb, ub


def _starts(names) -> list[np.ndarray]:
    """Starting points: the .mod's estimated_params initial values, and the
    SW07 Table 1a/1b posterior mode (ECB WP 722, PDF pp. 35-36)."""
    mod_init = {
        "ea": 0.4618, "eb": 0.1818513, "eg": 0.6090, "eqs": 0.46017, "em": 0.2397,
        "epinf": 0.1455, "ew": 0.2089, "crhoa": 0.9676, "crhob": 0.2703,
        "crhog": 0.9930, "crhoqs": 0.5724, "crhoms": 0.3, "crhopinf": 0.8692,
        "crhow": 0.9546, "cmap": 0.7652, "cmaw": 0.8936, "csadjcost": 6.3325,
        "csigma": 1.2312, "chabb": 0.7205, "cprobw": 0.7937, "csigl": 2.8401,
        "cprobp": 0.7813, "cindw": 0.4425, "cindp": 0.3291, "czcap": 0.2648,
        "cfc": 1.4672, "crpi": 1.7985, "crr": 0.8258, "cry": 0.0893,
        "crdy": 0.2239, "constepinf": 0.7, "constebeta": 0.7420,
        "constelab": 1.2918, "ctrend": 0.3982, "cgy": 0.05, "calfa": 0.24,
    }
    table1_mode = {
        "ea": 0.45, "eb": 0.24, "eg": 0.52, "eqs": 0.45, "em": 0.24,
        "epinf": 0.14, "ew": 0.24, "crhoa": 0.95, "crhob": 0.18, "crhog": 0.97,
        "crhoqs": 0.71, "crhoms": 0.12, "crhopinf": 0.90, "crhow": 0.97,
        "cmap": 0.74, "cmaw": 0.88, "csadjcost": 5.48, "csigma": 1.39,
        "chabb": 0.71, "cprobw": 0.73, "csigl": 1.92, "cprobp": 0.65,
        "cindw": 0.59, "cindp": 0.22, "czcap": 0.54, "cfc": 1.61, "crpi": 2.03,
        "crr": 0.81, "cry": 0.08, "crdy": 0.22, "constepinf": 0.81,
        "constebeta": 0.16, "constelab": 0.0, "ctrend": 0.43, "cgy": 0.52,
        "calfa": 0.19,
    }
    return [np.array([d[n] for n in names], dtype=float) for d in (table1_mode, mod_init)]


def _grad(f, x, lb, ub, h=1e-5):
    """Central-difference gradient (one-sided at a bound)."""
    g = np.empty_like(x)
    for i in range(len(x)):
        hi = h * max(1.0, abs(x[i]))
        up, dn = x.copy(), x.copy()
        up[i] = min(x[i] + hi, ub[i])
        dn[i] = max(x[i] - hi, lb[i])
        g[i] = (f(up) - f(dn)) / (up[i] - dn[i])
    return g


def find_mode(csv_path: Path, *, verbose: bool = True) -> tuple[np.ndarray, float]:
    """Posterior mode of the SW07 model on ``csv_path``.

    L-BFGS-B (finite-difference gradients) from each start in
    :func:`_starts`; the best end point is then polished by damped Newton
    steps with a central-difference gradient and Hessian until the
    log posterior improves by less than 1e-9. Returns the mode and its log
    posterior (log likelihood + log prior).
    """
    from scipy.optimize import minimize

    from puremacro.numerics import numerical_hessian

    names, nlp, nlp_opt, lb, ub = _objective(csv_path)
    bounds = list(zip(lb, ub))
    best_x, best_f = None, np.inf
    for k, x0 in enumerate(_starts(names)):
        x = np.clip(x0, lb + 1e-6, ub - 1e-6)
        r = minimize(nlp_opt, x, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 2000, "maxfun": 200_000,
                              "ftol": 1e-12, "gtol": 1e-6})
        if verbose:
            print(f"start {k}: L-BFGS-B {r.nit} iterations, log posterior "
                  f"{-r.fun:.6f} ({r.message})", flush=True)
        if r.fun < best_f:
            best_x, best_f = np.asarray(r.x, dtype=float), float(r.fun)
    x, f = best_x, best_f
    for it in range(8):
        g = _grad(nlp_opt, x, lb, ub)
        H = numerical_hessian(nlp_opt, x, h=1e-4)
        H = 0.5 * (H + H.T)
        w, V = np.linalg.eigh(H)
        step = -V @ ((V.T @ g) / np.maximum(w, 1e-6 * max(w.max(), 1.0)))
        t, improved = 1.0, False
        while t > 1e-4:
            x_new = np.clip(x + t * step, lb + 1e-8, ub - 1e-8)
            f_new = nlp_opt(x_new)
            if f_new < f:
                improved = True
                break
            t *= 0.5
        if verbose:
            print(f"Newton {it}: |g|max {np.abs(g).max():.2e}, min eig H {w.min():.3e}, "
                  f"log posterior {-(f_new if improved else f):.9f}", flush=True)
        if not improved:
            break
        gain = f - f_new
        x, f = x_new, f_new
        if gain < 1e-9:
            break
    return x, -float(nlp(x))


def build_fixture(csv_path: Path, out_paths, *, n_keep: int = 200,
                  thin: int = 50, burn_in: int = 2000, seed: int = 0) -> dict:
    """Replication fixture: posterior mode, inverse Hessian, and 200 MH draws.

    * ``mode_values`` / ``log_post_mode``: :func:`find_mode` on ``csv_path``.
    * ``mode_hessian_inv``: inverse of the central-difference Hessian of the
      negative log posterior at the mode (must be positive definite).
    * ``draws`` (1, 200, 36) / ``log_posterior_trace`` (1, 200): a random-walk
      Metropolis chain started at the mode, proposal ``(2.38^2/36) * inv H``,
      scalar scale adapted during ``burn_in`` extra draws, then every
      ``thin``-th of ``n_keep * thin`` draws kept. ``accept_rates`` is the
      acceptance rate over those retained iterations.
    * ``data_sha256``: SHA-256 of the CSV the fixture was built on (with
      CRLF normalised to LF).
    """
    from puremacro.mcmc import random_walk_metropolis
    from puremacro.numerics import numerical_hessian

    names, nlp, _, lb, ub = _objective(csv_path)
    mode, lp_mode = find_mode(csv_path)
    h = np.minimum(1e-4, 0.5 * np.minimum(mode - lb, ub - mode))
    if np.any(h < 1e-4):
        raise RuntimeError("mode within 1e-4 of a prior bound; Hessian stencil leaves the support: "
                           f"{[n for n, hh in zip(names, h) if hh < 1e-4]}")
    H = numerical_hessian(nlp, mode, h=1e-4)
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    if eig.min() <= 0:
        raise RuntimeError(f"Hessian at the mode is not positive definite (min eig {eig.min():.3e})")
    inv_H = np.linalg.inv(H)
    inv_H = 0.5 * (inv_H + inv_H.T)
    n = len(names)
    out = random_walk_metropolis(lambda v: -nlp(v), mode, (2.38 ** 2 / n) * inv_H,
                                 n_draws=n_keep * thin, seed=seed,
                                 accept_target=0.25, adapt_burnin=burn_in)
    chain = np.asarray(out["chain"])[thin - 1::thin]
    trace = np.asarray(out["log_post"])[thin - 1::thin]
    # Line endings normalised (the replication cases hash the same way).
    sha = hashlib.sha256(Path(csv_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    payload = dict(
        draws=chain[None, :, :],
        log_posterior_trace=trace[None, :],
        accept_rates=np.array([float(out["accept_rate"])]),
        mode_values=mode,
        log_post_mode=np.array(lp_mode),
        mode_hessian_inv=inv_H,
        param_names=np.array(names),
        data_sha256=np.array(sha),
    )
    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **payload)
        print(f"wrote {p}")
    return payload


AUTHORS_CSV = ROOT / "puremacro" / "dsge" / "_sw07_usmodel_data.csv"
_AUTHORS_MAP = {"dy": "gdp_growth", "dc": "cons_growth", "dinve": "inv_growth", "dw": "wage_growth",
                "labobs": "log_hours", "pinfobs": "infl", "robs": "ffr"}


def write_authors_csv(mat_path: Path, out_path: Path = AUTHORS_CSV) -> pd.DataFrame:
    """Convert the authors' ``usmodel_data.mat`` (AER replication files, 1947Q3-2004Q4)
    verbatim into the bundled ``_sw07_usmodel_data.csv``. Development tool only: the
    library never reads MATLAB files."""
    import scipy.io as sio

    m = sio.loadmat(str(mat_path))
    n = int(np.asarray(m["dy"]).size)
    idx = [str(q) for q in pd.period_range("1947Q3", periods=n, freq="Q")]
    df = pd.DataFrame({v: np.asarray(m[k], dtype=float).ravel() for k, v in _AUTHORS_MAP.items()}, index=idx)
    df.index.name = "date"
    header = [
        f"# puremacro SW07 dataset, the authors' series: {idx[0]} to {idx[-1]} ({n} quarterly obs)",
        "# Source: usmodel_data.mat of the Smets and Wouters (2007) AER replication files",
        "# (https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586), as redistributed in Johannes Pfeifer's",
        "# DSGE_mod repository (Smets_Wouters_2007/usmodel_data.mat). The data are the authors' replication",
        "# data; the GPL-3 licence of DSGE_mod covers its Dynare code, not these numbers.",
        "# Converted verbatim by tools/build_sw07_data.py authors; column mapping dy->gdp_growth, dc->cons_growth,",
        "# dinve->inv_growth, dw->wage_growth, labobs->log_hours, pinfobs->infl, robs->ffr. Units as in the paper's",
        "# data appendix: growth rates and inflation in 100 x log differences, hours 100 x log (the authors' level,",
        "# mean -0.81 over 1966Q1-2004Q4; _sw07_data.csv demeans to 0), federal funds rate in quarterly percent.",
        "# The 1966Q1-2004Q4 rows are the paper's estimation sample; 1956Q1-1965Q4 is the Table 2 training sample.",
    ]
    out_path.write_text("\n".join(header) + "\n" + df.to_csv(float_format="%.12g", lineterminator="\n"),
                        encoding="utf-8")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("authors", help="convert the authors' usmodel_data.mat into the bundled CSV")
    a.add_argument("--mat", type=Path, required=True, help="path to usmodel_data.mat (AER replication files)")
    a.add_argument("--out", type=Path, default=AUTHORS_CSV)
    d = sub.add_parser("data", help="download FRED series and write the CSV")
    d.add_argument("--cache", type=Path, default=None, help="directory of cached fredgraph CSVs")
    d.add_argument("--out", type=Path, default=DEFAULT_CSV)
    f = sub.add_parser("fixture", help="optimise the posterior and write the replication fixture")
    f.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    f.add_argument("--out", type=Path, nargs="+", default=list(DEFAULT_FIXTURES))
    args = ap.parse_args(argv)
    if args.cmd == "authors":
        out = write_authors_csv(args.mat, args.out)
        print(f"wrote {args.out} ({len(out)} quarters)")
    elif args.cmd == "data":
        out = build_data(args.out, args.cache)
        print(f"wrote {args.out} ({len(out)} quarters)")
        print(out.describe().T[["mean", "std", "min", "max"]])
    else:
        build_fixture(args.csv, args.out)


if __name__ == "__main__":
    main()
