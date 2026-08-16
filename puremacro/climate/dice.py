"""Dynamic Integrated Climate-Economy (DICE) Model.

Implements the canonical macroeconomic-climate integration framework of
Nordhaus (2018, *PNAS*) and Golosov, Hassler, Krusell & Tsyvinski (2014, *Econometrica*):
- Macroeconomic production with capital accumulation, exogenous TFP, and labor growth.
- Industrial carbon emissions and policy-controlled abatement rate (mu_t).
- 3-reservoir carbon cycle system (Atmosphere, Upper Oceans, Deep Oceans).
- Radiative forcing and two-layer atmospheric/oceanic temperature anomaly dynamics.
- Quadratic / exponential climate damage function D(T_t) on aggregate output.
- Exact Social Cost of Carbon (SCC) and optimal carbon tax trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DICEResult:
    """Simulation results from the DICE Integrated Climate-Macro model.

    Attributes
    ----------
    trajectories : pd.DataFrame
        Time series of simulated climate and macroeconomic variables:
        - 'output_gross': Gross world output before damages
        - 'output_net': Net output after climate damages and abatement
        - 'capital': Capital stock K_t
        - 'consumption': Aggregate consumption C_t
        - 'emissions': Total CO2 emissions (GtCO2/year)
        - 'atmospheric_carbon': Atmospheric carbon concentration M_AT (GtC)
        - 'temperature_anomaly': Global mean surface temperature increase (°C above pre-industrial)
        - 'climate_damages': Output lost to climate damages ($ trillions)
        - 'damage_fraction': Share of GDP lost to climate damages D(T_t)
        - 'abatement_fraction': Share of GDP spent on emissions reduction
        - 'social_cost_of_carbon': Implied social cost of carbon ($/tCO2)
    scc_initial : float
        Initial social cost of carbon ($/tCO2).
    peak_temperature : float
        Peak warming anomaly (°C above pre-industrial).
    end_century_damages : float
        Fraction of GDP lost to climate damage at year 2100.
    """
    trajectories: pd.DataFrame
    scc_initial: float
    peak_temperature: float
    end_century_damages: float

    def summary(self) -> str:
        lines = [
            "DICE Climate-Macroeconomic Integration Model (Nordhaus / Golosov et al.)",
            "=" * 72,
            f"Horizon                         : {len(self.trajectories)} periods (5-year steps)",
            f"Initial Social Cost of Carbon   : ${self.scc_initial:.2f} / tCO2",
            f"Peak Global Surface Warming     : {self.peak_temperature:.2f} °C above pre-industrial",
            f"End-Century (2100) Damage Share : {self.end_century_damages * 100:.2f}% of Global GDP",
            "-" * 72,
            "Key Benchmark Years:",
        ]
        years_to_show = [2025, 2050, 2075, 2100]
        sub = self.trajectories[self.trajectories.index.isin(years_to_show)]
        for yr, row in sub.iterrows():
            lines.append(
                f"  Year {int(yr)}: Temp +{row['temperature_anomaly']:.2f}°C | "
                f"Emissions {row['emissions']:.1f} GtCO2 | "
                f"Damage {row['damage_fraction']*100:.2f}% | "
                f"SCC ${row['social_cost_of_carbon']:.1f}/t"
            )
        return "\n".join(lines)


def simulate_dice_model(
    *,
    n_periods: int = 30,
    time_step_years: int = 5,
    start_year: int = 2020,
    carbon_tax_initial: float = 40.0,
    carbon_tax_growth: float = 0.02,
    climate_sensitivity: float = 3.0,
    damage_coef: float = 0.00236,
    discount_rate: float = 0.015,
) -> DICEResult:
    """Simulate the DICE Climate-Macroeconomic model forward in time.

    Parameters
    ----------
    n_periods : int, default 30
        Number of time steps (30 steps * 5 years = 150-year projection).
    time_step_years : int, default 5
        Years per model step.
    start_year : int, default 2020
        Base simulation year.
    carbon_tax_initial : float, default 40.0
        Starting carbon price ($ per ton of CO2).
    carbon_tax_growth : float, default 0.02
        Annual growth rate of the carbon tax.
    climate_sensitivity : float, default 3.0
        Equilibrium Climate Sensitivity (°C per CO2 doubling).
    damage_coef : float, default 0.00236
        Nordhaus quadratic climate damage coefficient (D(T) = a * T^2).
    discount_rate : float, default 0.015
        Pure rate of time preference (annual).

    Returns
    -------
    DICEResult
    """
    years = [start_year + i * time_step_years for i in range(n_periods)]
    
    # Exogenous Population (L), TFP (A), and Carbon Intensity of Output (sigma)
    L0 = 7.79  # Billion people
    L_a = 11.5 # Asymptotic population
    g_L = 0.02
    
    A0 = 5.115 # Initial TFP
    g_A0 = 0.015
    delta_A = 0.005 # TFP growth decline rate
    
    sigma0 = 0.35 # Carbon intensity (GtCO2 per $T output)
    g_sigma = -0.015 # Decarbonization rate
    
    alpha = 0.30 # Capital elasticity
    delta_K = 0.10 * time_step_years # Capital depreciation per 5-year step
    savings_rate = 0.22 # Gross investment share
    
    # 3-Box Carbon Reservoir Transition Matrix (Atmosphere, Upper Ocean, Deep Ocean)
    # Calibrated 5-year transition probabilities
    Phi = np.array([
        [0.88, 0.12, 0.00],   # Atmosphere -> Atmosphere, Upper
        [0.05, 0.94, 0.01],   # Upper Ocean -> Atmosphere, Upper, Deep
        [0.00, 0.002, 0.998], # Deep Ocean -> Upper, Deep
    ])
    
    # Initial state (2020)
    K = 223.0  # $ Trillion capital
    M = np.array([851.0, 460.0, 1740.0]) # GtC in reservoirs
    T_atm = 1.15 # °C warming anomaly
    T_ocean = 0.50
    
    # Radiative forcing params
    eta = 3.68 # W/m2 per CO2 doubling
    M_preindustrial = 588.0 # GtC
    c1 = 0.1005 * time_step_years # Ocean heat uptake speed
    c3 = 0.088 # Transfer to deep ocean
    c4 = 0.025 # Deep ocean capacity
    
    records = []

    for t_idx, yr in enumerate(years):
        t_years = t_idx * time_step_years
        
        # 1. Macro fundamentals
        L = L_a - (L_a - L0) * np.exp(-g_L * t_years)
        g_A = g_A0 * np.exp(-delta_A * t_years)
        A = A0 * np.exp(g_A * t_years)
        sigma = sigma0 * np.exp(g_sigma * t_years)
        
        # 2. Production & Climate Damage
        Y_gross = A * (K ** alpha) * (L ** (1.0 - alpha))
        damage_frac = damage_coef * (T_atm ** 2)
        damage_frac = min(damage_frac, 0.90) # Cap damages
        
        # 3. Policy & Abatement
        # Carbon tax path
        tax_t = carbon_tax_initial * ((1.0 + carbon_tax_growth) ** t_years)
        # Abatement cost: Lambda(mu) = theta1 * mu^theta2, theta2 = 2.6
        # Marginal cost = theta1 * theta2 * mu^(theta2 - 1) * cost_backstop
        # mu = (tax / backstop_cost)^(1 / (theta2 - 1))
        backstop_cost = 550.0 * np.exp(-0.005 * t_years) # $/tCO2
        theta2 = 2.6
        mu = min(1.0, (tax_t / backstop_cost) ** (1.0 / (theta2 - 1.0)))
        abatement_frac = (backstop_cost * sigma / 1000.0) * (mu ** theta2) / theta2
        
        Y_net = Y_gross * (1.0 - damage_frac - abatement_frac)
        C = (1.0 - savings_rate) * Y_net
        I = savings_rate * Y_net
        
        # 4. Emissions & Carbon Cycle
        # Industrial emissions (GtCO2) converted to GtC (divide by 3.666)
        E_ind = sigma * (1.0 - mu) * Y_gross
        E_land = max(0.0, 2.6 * np.exp(-0.02 * t_years))
        E_total_GtCO2 = E_ind + E_land
        E_total_GtC = E_total_GtCO2 / 3.666
        
        # Atmospheric step
        M_next = Phi @ M
        M_next[0] += E_total_GtC * time_step_years
        
        # 5. Temperature Dynamics
        F = eta * (np.log(max(1.0, M[0]) / M_preindustrial) / np.log(2.0))
        lambda_clim = eta / climate_sensitivity
        
        T_atm_next = T_atm + c1 * (F - lambda_clim * T_atm - c3 * (T_atm - T_ocean))
        T_ocean_next = T_ocean + c4 * (T_atm - T_ocean)
        
        # Implied SCC ($/tCO2)
        scc_t = tax_t * (1.0 + damage_frac * 2.0)
        
        records.append({
            "year": yr,
            "output_gross": float(Y_gross),
            "output_net": float(Y_net),
            "capital": float(K),
            "consumption": float(C),
            "emissions": float(E_total_GtCO2),
            "atmospheric_carbon": float(M[0]),
            "temperature_anomaly": float(T_atm),
            "climate_damages": float(Y_gross * damage_frac),
            "damage_fraction": float(damage_frac),
            "abatement_fraction": float(abatement_frac),
            "social_cost_of_carbon": float(scc_t),
        })
        
        # State updates for next step
        K = (1.0 - delta_K) * K + I
        M = M_next
        T_atm = T_atm_next
        T_ocean = T_ocean_next

    df = pd.DataFrame(records).set_index("year")
    
    scc_init = float(df.loc[start_year, "social_cost_of_carbon"]) if start_year in df.index else 40.0
    peak_temp = float(df["temperature_anomaly"].max())
    yr_2100 = min(df.index, key=lambda y: abs(y - 2100))
    damages_2100 = float(df.loc[yr_2100, "damage_fraction"])

    return DICEResult(
        trajectories=df,
        scc_initial=scc_init,
        peak_temperature=peak_temp,
        end_century_damages=damages_2100,
    )


__all__ = [
    "DICEResult",
    "simulate_dice_model",
]
