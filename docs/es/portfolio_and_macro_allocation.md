> 🇬🇧 [English](../portfolio_and_macro_allocation.md) · 🇪🇸 Español

# Análisis de Portafolios y Asignación Macro de Activos (`puremacro.portfolio`)

`puremacro.portfolio` proporciona un motor cuantitativo en NumPy/SciPy puro para
optimización de portafolios, asignación táctica por cuadrantes macroeconómicos,
atribución multifactorial, backtesting con costos de transacción y simulación
Monte Carlo de patrimonio y retiro:

1. **Filtrado de Covarianza y Optimización de Portafolios (`puremacro.portfolio.allocation`)**:
   - `shrink_covariance`: Contracción analítica de Ledoit-Wolf y filtrado espectral por Teoría de Matrices Aleatorias (Marčenko-Pastur).
   - `optimize_portfolio`: Punto de entrada unificado para `"max_sharpe"`, `"min_variance"`, `"risk_parity"` (Contribución Igualitaria al Riesgo), `"hrp"` (Paridad de Riesgo Jerárquica de López de Prado 2016), `"min_cvar"` (Mínimo CVaR de Rockafellar-Uryasev) y `"kelly"` (criterio de Kelly fraccional).
   - `markowitz_frontier`: Traza la frontera eficiente e identifica los portafolios de Mínima Varianza, Tangencia (Máximo Sharpe) y Paridad de Riesgo.
   - `black_litterman`: Equilibrio de mercado inverso combinado con vistas absolutas o relativas del inversionista (incluyendo niveles de confianza de Idzorek de 0 a 100%).
2. **Asignación por Regímenes Macroeconómicos y Atribución Factorial (`puremacro.portfolio.macro_regime`)**:
   - `classify_macro_quadrants`: Clasifica el crecimiento e inflación en los 4 cuadrantes macro clásicos (`Goldilocks`, `Reflation`, `Stagflation`, `Deflationary Bust`).
   - `all_weather_portfolio`: Construcción de portafolios All-Weather balanceados por riesgo entre regímenes macro o inclinados según probabilidades de régimen.
   - `factor_attribution`: Descomposición multifactorial OLS + Newey-West HAC de alfa y betas con exposiciones móviles y atribución de varianza.
3. **Backtesting con Rebalanceo Realista y Simulación Patrimonial (`puremacro.portfolio.backtest`)**:
   - `backtest_portfolio`: Simula pesos flotantes, rebalanceo periódico o por bandas de deriva (`drift_band`), costos de transacción en puntos base, caídas máximas (drawdowns) y hoja de diagnóstico visual de 4 paneles (`.plot()`).
   - `simulate_wealth_glidepath`: Simulador Monte Carlo / bootstrap por bloques para acumulación (DCA) y desacumulación en el retiro (regla del 4%, probabilidad de ruina y gráficos de abanico).

---

## 1. Contracción de Covarianza, HRP, Paridad de Riesgo y Black-Litterman

```python
import numpy as np
import pandas as pd
from puremacro.portfolio import (
    shrink_covariance,
    optimize_portfolio,
    markowitz_frontier,
    hierarchical_risk_parity,
    black_litterman,
)

rng = np.random.default_rng(42)
dates = pd.date_range("2015-01-31", periods=120, freq="ME")
returns = pd.DataFrame(
    rng.normal([0.008, 0.004, 0.005, 0.003], [0.045, 0.020, 0.035, 0.015], size=(120, 4)),
    index=dates,
    columns=["Equities", "Bonds", "Gold", "TIPS"],
)

cov_lw = shrink_covariance(returns, method="ledoit_wolf")
hrp_res = hierarchical_risk_parity(returns)
rp_res = optimize_portfolio(returns, method="risk_parity")
print(hrp_res.summary())
print(rp_res.summary())

P = np.array([[1.0, -1.0, 0.0, 0.0]])
Q = np.array([0.04 / 12.0])
bl_res = black_litterman(
    cov_lw,
    market_weights=[0.50, 0.30, 0.10, 0.10],
    P=P,
    Q=Q,
    confidences=[0.70],
)
print(bl_res.summary())
```

---

## 2. Clasificación de Cuadrantes Macro y Portafolio All-Weather

```python
from puremacro.portfolio import (
    classify_macro_quadrants,
    all_weather_portfolio,
    factor_attribution,
)

growth = pd.Series(rng.normal(2.2, 1.0, size=120), index=dates, name="gdp_growth")
inflation = pd.Series(rng.normal(2.5, 0.8, size=120), index=dates, name="cpi_inflation")

quadrants = classify_macro_quadrants(growth, inflation, method="change")
aw_res = all_weather_portfolio(returns, growth, inflation, method="risk_parity")
print(aw_res.summary())
```

---

## 3. Backtesting con Rebalanceo y Simulación de Retiro

```python
from puremacro.portfolio import backtest_portfolio, simulate_wealth_glidepath

bt = backtest_portfolio(
    returns,
    weights=rp_res.weights,
    rebalance="quarterly",
    tc_bps=10.0,
)
print(bt.summary())

glide = simulate_wealth_glidepath(
    returns,
    weights=rp_res.weights,
    initial_wealth=500_000.0,
    horizon_years=30,
    annual_withdrawal=20_000.0,
    n_sims=500,
    seed=42,
)
print(glide.summary())
```
