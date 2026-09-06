"""Nordhaus DICE-2016R forward simulator with a marginal-damage social cost of carbon.

The module prices a *scenario*: you supply a carbon-tax path, a climate
sensitivity and a damage coefficient, and it returns the joint trajectory of
output, emissions, carbon reservoirs, warming and damages, plus a social cost
of carbon (SCC) evaluated on that path. Nothing is optimised - there is no
welfare maximisation over the abatement or savings rate.

What is implemented (Nordhaus 2017, *PNAS* 114(7), DICE-2016R calibration):

- Cobb-Douglas production ``Y = A K^0.3 L^0.7`` with exogenous TFP, population
  and carbon-intensity paths; capital ``K' = (1-0.10)^dt K + dt * s * Y_net``
  with a fixed savings rate.
- Abatement rate ``mu`` read off DICE's first-order condition against the
  supplied tax, and DICE's abatement cost ``theta1 * mu^2.6``.
- The three-reservoir carbon cycle (atmosphere, upper ocean, deep ocean) with
  DICE-2016R's transfer coefficients, applied as a mass-conserving operator.
- CO2 forcing plus DICE's exogenous non-CO2 forcing path, and the two-layer
  (atmosphere / deep ocean) temperature recursion with DICE's per-step
  coefficients ``c1 = 0.1005, c3 = 0.088, c4 = 0.025``.
- A **quadratic** damage function ``D(T) = a2 * T^2`` (no other form is
  provided).
- An SCC computed the way DICE defines it: the present value of the
  consumption lost to one extra ton of CO2 emitted in period ``t``, obtained by
  perturbing the emissions path and discounting the consumption losses with
  the Ramsey factor ``(1 + rho)^-(s-t) * (c_t / c_s)^eta``.

Time-step rule: the DICE constants are per 5-year step. For
``time_step_years != 5`` the 5-year carbon-cycle and temperature transition
operators are raised to the power ``dt / 5`` (matrix fractional power), the
emission injection is scaled by ``dt``, and capital depreciation is geometric
in ``dt``, so a 10-year run samples the 5-year dynamics rather than running
them at a different speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# DICE-2016R calibration (Nordhaus 2017, PNAS). Rates are per year unless the
# comment says "per 5-year step".
# ---------------------------------------------------------------------------
_DICE_STEP_YEARS = 5.0

# Carbon cycle (GtC). b12 / b23 are per 5-year step; the reverse flows are
# pinned to the equilibrium reservoir sizes exactly as in DICE-2016R.
_M_AT_EQ, _M_UP_EQ, _M_LO_EQ = 588.0, 360.0, 1720.0
_B12, _B23 = 0.12, 0.007
_B21 = _B12 * _M_AT_EQ / _M_UP_EQ  # 0.196
_B32 = _B23 * _M_UP_EQ / _M_LO_EQ  # 0.001465
# Row i = "where reservoir i's carbon goes": row-stochastic by construction.
_B_ROW = np.array(
    [
        [1.0 - _B12, _B12, 0.0],
        [_B21, 1.0 - _B21 - _B23, _B23],
        [0.0, _B32, 1.0 - _B32],
    ]
)
_PHI_5 = _B_ROW.T  # column-stochastic operator: M' = _PHI_5 @ M conserves mass
_GTCO2_PER_GTC = 3.666

# Climate (per 5-year step)
_FCO22X = 3.6813  # W/m2 per CO2 doubling
_C1, _C3, _C4 = 0.1005, 0.088, 0.025
_FEX0, _FEX1, _FEX_YEARS = 0.5, 1.0, 85.0  # exogenous non-CO2 forcing ramp

# Economy
_ALPHA = 0.30
_DELTA_K = 0.10  # annual depreciation
_A0, _G_A0, _DELTA_A = 5.115, 0.0152, 0.005
_L0, _L_ASYM, _POP_ADJ = 7.79, 11.5, 0.134  # billions; popadj per 5-year step
_SIGMA0 = 35.85 / 105.5  # GtCO2 per $T of gross output (e0 / q0)
_G_SIGMA0, _DELTA_SIGMA = -0.0152, -math.log(1.0 - 0.001)
_E_LAND0, _DELTA_LAND = 2.6, 0.115  # GtCO2/yr; deland per 5-year step
_P_BACK0, _G_BACK = 550.0, 0.025  # $/tCO2; gback per 5-year step
_THETA2 = 2.6
_DAMAGE_CAP = 0.90

# Initial state, dated to the run's start_year (DICE-2016R's 2015 stocks with
# 2020 population and 2020 observed warming).
_K0 = 223.0
_M0 = np.array([851.0, 460.0, 1740.0])
_T_ATM0, _T_OCEAN0 = 1.15, 0.0068


def _matrix_fractional_power(mat: np.ndarray, power: float) -> np.ndarray:
    """``mat ** power`` for a diagonalisable matrix with real positive eigenvalues."""
    if power == 1.0:
        return mat.copy()
    w, v = np.linalg.eig(mat)
    if np.any(np.abs(w.imag) > 1e-12) or np.any(w.real <= 0.0):
        raise ValueError("transition operator has non-positive eigenvalues; cannot rescale")
    out = v @ np.diag(w.real ** power) @ np.linalg.inv(v)
    return np.asarray(out.real, dtype=float)


def min_climate_sensitivity() -> float:
    """Smallest ``climate_sensitivity`` for which DICE's per-step temperature
    recursion is stable and monotone (both eigenvalues of the two-layer
    transition matrix in ``(0, 1)``). About 0.37 degC per doubling."""
    a_max = 1.0 - _C1 * _C3 * _C4 / (1.0 - _C4)
    return _FCO22X / (a_max / _C1 - _C3)


def _temperature_operator(climate_sensitivity: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A_dt, b_dt)`` with ``x' = A_dt x + b_dt * F`` for x = (T_atm, T_ocean)."""
    lam = _FCO22X / climate_sensitivity
    a5 = np.array([[1.0 - _C1 * (lam + _C3), _C1 * _C3], [_C4, 1.0 - _C4]])
    b5 = np.array([_C1, 0.0])
    s = dt / _DICE_STEP_YEARS
    a_dt = _matrix_fractional_power(a5, s)
    # Sum of the geometric series I + A + ... over a fractional number of steps.
    b_dt = np.linalg.solve(np.eye(2) - a5, (np.eye(2) - a_dt) @ b5)
    return a_dt, b_dt


def _exogenous(t: float) -> Dict[str, float]:
    """Exogenous paths at elapsed calendar time ``t`` (years since start)."""
    steps = t / _DICE_STEP_YEARS
    pop = _L_ASYM * (_L0 / _L_ASYM) ** ((1.0 - _POP_ADJ) ** steps)
    tfp = _A0 * math.exp(_G_A0 / _DELTA_A * (1.0 - math.exp(-_DELTA_A * t)))
    sigma = _SIGMA0 * math.exp(_G_SIGMA0 * (1.0 - math.exp(-_DELTA_SIGMA * t)) / _DELTA_SIGMA)
    e_land = _E_LAND0 * (1.0 - _DELTA_LAND) ** steps
    p_back = _P_BACK0 * (1.0 - _G_BACK) ** steps
    f_exo = _FEX0 + (_FEX1 - _FEX0) * min(t, _FEX_YEARS) / _FEX_YEARS
    return {"pop": pop, "tfp": tfp, "sigma": sigma, "e_land": e_land, "p_back": p_back, "f_exo": f_exo}


def _forcing(m_at: float, f_exo: float) -> float:
    return _FCO22X * math.log(max(m_at, 1.0) / _M_AT_EQ) / math.log(2.0) + f_exo


@dataclass
class _State:
    capital: float
    carbon: np.ndarray
    t_atm: float
    t_ocean: float


def _simulate_path(
    *,
    state: _State,
    first_index: int,
    n_steps: int,
    dt: float,
    carbon_tax_initial: float,
    carbon_tax_growth: float,
    damage_coef: float,
    savings_rate: float,
    phi_dt: np.ndarray,
    a_dt: np.ndarray,
    b_dt: np.ndarray,
    pulse_index: Optional[int] = None,
    pulse_gtco2_per_year: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Run the model for ``n_steps`` starting at period ``first_index``.

    ``pulse_index`` adds ``pulse_gtco2_per_year`` GtCO2/yr of extra emissions
    during that one period (the marginal experiment behind the SCC).
    """
    cols = (
        "output_gross", "output_net", "capital", "consumption", "population",
        "emissions", "atmospheric_carbon", "upper_ocean_carbon", "deep_ocean_carbon",
        "radiative_forcing", "temperature_anomaly", "ocean_temperature",
        "climate_damages", "damage_fraction", "abatement_rate", "abatement_fraction",
        "carbon_tax",
    )
    out = {c: np.empty(n_steps) for c in cols}
    k = state.capital
    m = state.carbon.copy()
    t_atm, t_ocean = state.t_atm, state.t_ocean
    x = np.array([t_atm, t_ocean])

    for j in range(n_steps):
        i = first_index + j
        t = i * dt
        ex = _exogenous(t)
        pop, tfp, sigma = ex["pop"], ex["tfp"], ex["sigma"]

        y_gross = tfp * (k ** _ALPHA) * (pop ** (1.0 - _ALPHA))
        damage_frac = min(damage_coef * t_atm * t_atm, _DAMAGE_CAP)

        tax = carbon_tax_initial * (1.0 + carbon_tax_growth) ** t
        mu = min(1.0, (tax / ex["p_back"]) ** (1.0 / (_THETA2 - 1.0)))
        abatement_frac = (ex["p_back"] * sigma / 1000.0) * (mu ** _THETA2) / _THETA2

        y_net = y_gross * (1.0 - damage_frac - abatement_frac)
        consumption = (1.0 - savings_rate) * y_net
        investment = savings_rate * y_net

        e_ind = sigma * (1.0 - mu) * y_gross
        e_total = e_ind + ex["e_land"]
        if pulse_index is not None and i == pulse_index:
            e_total += pulse_gtco2_per_year

        forcing_now = _forcing(m[0], ex["f_exo"])

        out["output_gross"][j] = y_gross
        out["output_net"][j] = y_net
        out["capital"][j] = k
        out["consumption"][j] = consumption
        out["population"][j] = pop
        out["emissions"][j] = e_total
        out["atmospheric_carbon"][j] = m[0]
        out["upper_ocean_carbon"][j] = m[1]
        out["deep_ocean_carbon"][j] = m[2]
        out["radiative_forcing"][j] = forcing_now
        out["temperature_anomaly"][j] = t_atm
        out["ocean_temperature"][j] = t_ocean
        out["climate_damages"][j] = y_gross * damage_frac
        out["damage_fraction"][j] = damage_frac
        out["abatement_rate"][j] = mu
        out["abatement_fraction"][j] = abatement_frac
        out["carbon_tax"][j] = tax

        # --- state update (DICE ordering: carbon, then forcing at the new stock, then temperature)
        m = phi_dt @ m
        m[0] += e_total * dt / _GTCO2_PER_GTC
        forcing_next = _forcing(m[0], _exogenous(t + dt)["f_exo"])
        x = a_dt @ x + b_dt * forcing_next
        t_atm, t_ocean = float(x[0]), float(x[1])
        k = (1.0 - _DELTA_K) ** dt * k + dt * investment

    return out


@dataclass(frozen=True)
class DICEResult:
    """Simulation results from the DICE forward simulator.

    Attributes
    ----------
    trajectories : pd.DataFrame
        Indexed by calendar ``year``. Stocks (``capital``, the three carbon
        reservoirs, the two temperatures) are **start**-of-period values, so
        the first row is the initial condition; flows are per year.

        - ``output_gross``: gross world output before damages, $T/yr
        - ``output_net``: output after damages and abatement spending, $T/yr
        - ``capital``: capital stock, $T
        - ``consumption``: ``(1 - savings_rate) * output_net``, $T/yr
        - ``population``: billions
        - ``emissions``: industrial plus land-use CO2, GtCO2/yr
        - ``atmospheric_carbon`` / ``upper_ocean_carbon`` / ``deep_ocean_carbon``: GtC
        - ``radiative_forcing``: W/m2 (CO2 plus exogenous non-CO2)
        - ``temperature_anomaly``: surface warming, degC above pre-industrial
        - ``ocean_temperature``: deep-ocean layer, degC
        - ``climate_damages``: ``output_gross * damage_fraction``, $T/yr
        - ``damage_fraction``: share of gross output lost, ``a2 * T^2``
        - ``abatement_rate``: emissions-control rate ``mu`` in [0, 1]
        - ``abatement_fraction``: share of gross output spent on abatement
        - ``carbon_tax``: the tax path you supplied, $/tCO2
        - ``social_cost_of_carbon``: present value of the consumption lost to
          one extra ton of CO2 emitted in that year, $/tCO2
    scc_initial : float
        ``social_cost_of_carbon`` at ``start_year``.
    peak_temperature : float
        Maximum of ``temperature_anomaly`` over the reported horizon.
    end_century_damages : float
        ``damage_fraction`` at the index year nearest 2100.
    parameters : dict
        The keyword arguments the run was made with.
    """
    trajectories: pd.DataFrame
    scc_initial: float
    peak_temperature: float
    end_century_damages: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    # -- presentation --------------------------------------------------------
    def summary(self) -> str:
        df = self.trajectories
        dt = self.parameters.get("time_step_years", "?")
        rho = self.parameters.get("discount_rate")
        eta = self.parameters.get("elasticity_marginal_utility")
        lines = [
            "DICE-2016R forward simulation (Nordhaus 2017)",
            "=" * 72,
            f"Horizon                         : {len(df)} periods of {dt} years, "
            f"{int(df.index[0])}-{int(df.index[-1])}",
            f"Initial Social Cost of Carbon   : ${self.scc_initial:.2f} / tCO2"
            + (f"  (rho = {rho}, eta = {eta})" if rho is not None else ""),
            f"Peak Global Surface Warming     : {self.peak_temperature:.2f} degC above pre-industrial",
            f"End-Century (2100) Damage Share : {self.end_century_damages * 100:.2f}% of gross output",
            "-" * 72,
            "Key Benchmark Years (nearest available row):",
        ]
        shown: set[int] = set()
        for target in (2025, 2050, 2075, 2100):
            yr = int(min(df.index, key=lambda y: abs(y - target)))
            if yr in shown:
                continue
            shown.add(yr)
            row = df.loc[yr]
            lines.append(
                f"  Year {yr}: Temp +{row['temperature_anomaly']:.2f}degC | "
                f"Emissions {row['emissions']:.1f} GtCO2/yr | "
                f"Damage {row['damage_fraction'] * 100:.2f}% | "
                f"Tax ${row['carbon_tax']:.1f}/t | SCC ${row['social_cost_of_carbon']:.1f}/t"
            )
        return "\n".join(lines)

    _TABLE_COLUMNS = (
        "temperature_anomaly", "emissions", "atmospheric_carbon",
        "damage_fraction", "abatement_rate", "carbon_tax", "social_cost_of_carbon",
    )

    def to_frame(self, columns: Optional[Sequence[str]] = None, digits: Optional[int] = None) -> pd.DataFrame:
        """Trajectories (all columns by default), optionally rounded."""
        df = self.trajectories if columns is None else self.trajectories[list(columns)]
        df = df.copy()
        return df.round(digits) if digits is not None else df

    def _table(self, columns: Optional[Sequence[str]], digits: int) -> pd.DataFrame:
        return self.to_frame(columns=columns or self._TABLE_COLUMNS, digits=digits)

    def to_markdown(self, *, columns: Optional[Sequence[str]] = None, index: bool = True, digits: int = 3) -> str:
        """Markdown table of the main columns (year as the first column)."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self._table(columns, digits), index=index)

    def to_latex(self, *, columns: Optional[Sequence[str]] = None, index: bool = True, digits: int = 3) -> str:
        """LaTeX ``tabular`` of the main columns."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self._table(columns, digits), index=index)

    def to_typst(self, *, columns: Optional[Sequence[str]] = None, index: bool = True, digits: int = 3) -> str:
        """Typst ``#table`` of the main columns."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self._table(columns, digits), index=index)

    def plot(self, *, axes: Any = None, label: Optional[str] = None, **line_kwargs: Any):
        """Four panels: warming, emissions, damage share, carbon tax vs SCC.

        Pass ``axes=fig.axes`` from a previous call to overlay a second scenario.
        Returns the matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        df = self.trajectories
        if axes is None:
            fig, ax_arr = plt.subplots(2, 2, figsize=(11, 7.5))
            axes = list(np.ravel(ax_arr))
        else:
            axes = list(axes)
            fig = axes[0].figure
        if len(axes) < 4:
            raise ValueError("DICEResult.plot needs four axes")
        lab = label or f"tax ${self.parameters.get('carbon_tax_initial', '?')}/t"
        axes[0].plot(df.index, df["temperature_anomaly"], label=lab, **line_kwargs)
        axes[0].set_title("Surface warming (degC above pre-industrial)")
        axes[1].plot(df.index, df["emissions"], label=lab, **line_kwargs)
        axes[1].set_title("CO2 emissions (GtCO2/yr)")
        axes[2].plot(df.index, 100.0 * df["damage_fraction"], label=lab, **line_kwargs)
        axes[2].set_title("Climate damages (% of gross output)")
        axes[3].plot(df.index, df["social_cost_of_carbon"], label=f"SCC, {lab}", **line_kwargs)
        axes[3].plot(df.index, df["carbon_tax"], linestyle="--", label=f"tax, {lab}", **line_kwargs)
        axes[3].set_title("Social cost of carbon vs carbon tax ($/tCO2)")
        for ax in axes[:4]:
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_xlabel("Year")
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


def _validate_inputs(
    n_periods: int, time_step_years: float, start_year: int, carbon_tax_initial: float,
    carbon_tax_growth: float, climate_sensitivity: float, damage_coef: float,
    discount_rate: float, elasticity_marginal_utility: float, savings_rate: float,
    scc_horizon_years: float,
) -> None:
    if not isinstance(n_periods, (int, np.integer)) or isinstance(n_periods, bool) or n_periods < 1:
        raise ValueError(f"n_periods must be a positive integer, got {n_periods!r}")
    if not np.isfinite(time_step_years) or time_step_years <= 0:
        raise ValueError(f"time_step_years must be > 0, got {time_step_years!r}")
    if not isinstance(start_year, (int, np.integer)) or isinstance(start_year, bool):
        raise ValueError(f"start_year must be an integer, got {start_year!r}")
    if not np.isfinite(carbon_tax_initial) or carbon_tax_initial < 0:
        raise ValueError(f"carbon_tax_initial must be >= 0 $/tCO2, got {carbon_tax_initial!r}")
    if not np.isfinite(carbon_tax_growth) or carbon_tax_growth <= -1:
        raise ValueError(f"carbon_tax_growth must be > -1, got {carbon_tax_growth!r}")
    ecs_min = min_climate_sensitivity()
    if not np.isfinite(climate_sensitivity) or climate_sensitivity <= ecs_min:
        raise ValueError(
            f"climate_sensitivity must exceed {ecs_min:.3f} degC per doubling for DICE's "
            f"per-step temperature recursion to be stable and monotone, got {climate_sensitivity!r}"
        )
    if not np.isfinite(damage_coef) or damage_coef < 0:
        raise ValueError(f"damage_coef must be >= 0, got {damage_coef!r}")
    if not np.isfinite(discount_rate) or discount_rate <= -1:
        raise ValueError(f"discount_rate must be > -1, got {discount_rate!r}")
    if not np.isfinite(elasticity_marginal_utility) or elasticity_marginal_utility < 0:
        raise ValueError(f"elasticity_marginal_utility must be >= 0, got {elasticity_marginal_utility!r}")
    if not np.isfinite(savings_rate) or not 0.0 <= savings_rate < 1.0:
        raise ValueError(f"savings_rate must lie in [0, 1), got {savings_rate!r}")
    if not np.isfinite(scc_horizon_years) or scc_horizon_years < 0:
        raise ValueError(f"scc_horizon_years must be >= 0, got {scc_horizon_years!r}")


def simulate_dice_model(
    *,
    n_periods: int = 30,
    time_step_years: int = 5,
    start_year: int = 2020,
    carbon_tax_initial: float = 40.0,
    carbon_tax_growth: float = 0.02,
    climate_sensitivity: float = 3.1,
    damage_coef: float = 0.00236,
    discount_rate: float = 0.015,
    elasticity_marginal_utility: float = 1.45,
    savings_rate: float = 0.22,
    scc_horizon_years: int = 300,
) -> DICEResult:
    """Simulate the DICE-2016R climate-economy model forward under a given carbon tax.

    Parameters
    ----------
    n_periods : int, default 30
        Number of reported steps (30 steps of 5 years span 2020-2165).
    time_step_years : int, default 5
        Years per step. DICE's constants are per 5-year step; other values
        rescale the carbon-cycle and temperature operators by the matrix
        power ``dt/5`` (see the module docstring).
    start_year : int, default 2020
        Calendar year of the first row; the initial state is dated here.
    carbon_tax_initial : float, default 40.0
        Carbon price in ``start_year``, $/tCO2. ``0.0`` is a no-policy run.
    carbon_tax_growth : float, default 0.02
        Annual growth rate of the carbon tax (compounded over calendar years).
    climate_sensitivity : float, default 3.1
        Equilibrium warming per CO2 doubling, degC (DICE-2016R ``t2xco2``).
        Must exceed :func:`min_climate_sensitivity` (about 0.37).
    damage_coef : float, default 0.00236
        ``a2`` in ``D(T) = a2 * T^2`` (share of gross output).
    discount_rate : float, default 0.015
        Pure rate of social time preference ``rho`` (annual), used in the
        Ramsey discount factor of the SCC.
    elasticity_marginal_utility : float, default 1.45
        ``eta`` in the Ramsey factor ``(1+rho)^-(s-t) * (c_t/c_s)^eta``
        (DICE-2016R ``elasmu``). ``0`` discounts at the pure rate only.
    savings_rate : float, default 0.22
        Fixed gross saving rate out of net output.
    scc_horizon_years : int, default 300
        Years of consumption losses integrated into each period's SCC. The
        model is run internally past the reported horizon so late periods
        are not truncated.

    Returns
    -------
    DICEResult
    """
    _validate_inputs(
        n_periods, time_step_years, start_year, carbon_tax_initial, carbon_tax_growth,
        climate_sensitivity, damage_coef, discount_rate, elasticity_marginal_utility,
        savings_rate, scc_horizon_years,
    )
    dt = float(time_step_years)
    phi_dt = _matrix_fractional_power(_PHI_5, dt / _DICE_STEP_YEARS)
    a_dt, b_dt = _temperature_operator(climate_sensitivity, dt)

    n_extra = int(math.ceil(scc_horizon_years / dt))
    n_total = n_periods + n_extra
    common: Dict[str, Any] = dict(
        dt=dt, carbon_tax_initial=carbon_tax_initial, carbon_tax_growth=carbon_tax_growth,
        damage_coef=damage_coef, savings_rate=savings_rate, phi_dt=phi_dt, a_dt=a_dt, b_dt=b_dt,
    )
    state0 = _State(capital=_K0, carbon=_M0.copy(), t_atm=_T_ATM0, t_ocean=_T_OCEAN0)
    base = _simulate_path(state=state0, first_index=0, n_steps=n_total, **common)

    # --- marginal-damage SCC: perturb emissions in period i, discount the consumption losses
    pulse = 1e-2  # GtCO2/yr during one period (small enough to be a derivative)
    c_pc = base["consumption"] / base["population"]
    scc = np.empty(n_periods)
    for i in range(n_periods):
        state_i = _State(
            capital=float(base["capital"][i]),
            carbon=np.array([base["atmospheric_carbon"][i], base["upper_ocean_carbon"][i], base["deep_ocean_carbon"][i]]),
            t_atm=float(base["temperature_anomaly"][i]),
            t_ocean=float(base["ocean_temperature"][i]),
        )
        n_steps = min(n_total - i, n_extra + 1)
        pert = _simulate_path(
            state=state_i, first_index=i, n_steps=n_steps, pulse_index=i, pulse_gtco2_per_year=pulse, **common
        )
        d_cons = pert["consumption"] - base["consumption"][i:i + n_steps]  # $T/yr, <= 0
        years_ahead = dt * np.arange(n_steps)
        ramsey = (1.0 + discount_rate) ** (-years_ahead) * (c_pc[i] / c_pc[i:i + n_steps]) ** elasticity_marginal_utility
        # $T per (GtCO2/yr) -> $/tCO2 is a factor 1000 (1e12 / 1e9)
        scc[i] = -1000.0 * float(np.sum(ramsey * d_cons)) / pulse

    years = [start_year + int(round(i * dt)) for i in range(n_periods)]
    data = {k: v[:n_periods] for k, v in base.items()}
    data["social_cost_of_carbon"] = scc
    df = pd.DataFrame(data, index=pd.Index(years, name="year"))

    yr_2100 = min(df.index, key=lambda y: abs(y - 2100))
    params = dict(
        n_periods=n_periods, time_step_years=time_step_years, start_year=start_year,
        carbon_tax_initial=carbon_tax_initial, carbon_tax_growth=carbon_tax_growth,
        climate_sensitivity=climate_sensitivity, damage_coef=damage_coef,
        discount_rate=discount_rate, elasticity_marginal_utility=elasticity_marginal_utility,
        savings_rate=savings_rate, scc_horizon_years=scc_horizon_years,
    )
    return DICEResult(
        trajectories=df,
        scc_initial=float(scc[0]),
        peak_temperature=float(df["temperature_anomaly"].max()),
        end_century_damages=float(df.loc[yr_2100, "damage_fraction"]),
        parameters=params,
    )


__all__ = [
    "DICEResult",
    "min_climate_sensitivity",
    "simulate_dice_model",
]
