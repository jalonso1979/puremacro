# Reasignación Comercial Dinámica e Incidencia en los Hogares

> 🇬🇧 [English](../trade_reallocation_incidence.md) · 🇪🇸 Español

El módulo `puremacro.trade.reallocation_incidence` enlaza las transiciones comerciales de previsión perfecta con dos países y dos sectores ([trade_reallocation.md](trade_reallocation.md)) con el bienestar de grupos de hogares normalizado al punto de referencia ([trade_household.md](trade_household.md)) y las cuentas ICIO de la OCDE incluidas. En lugar de tratar cada región como un único agente representativo, mide cómo los salarios sectoriales, las rentas del capital netas de inversión, las devoluciones arancelarias y los precios al consumidor distribuyen ganancias y pérdidas entre grupos de hogares en cada fecha $t = 0, \dots, T-1$.

## Estructura Matemática e Identidades Contables

### Agregación ICIO de la OCDE (`calibrate_reallocation_from_icio`)

Dada una tabla multirregional y multisectorial `(Z, F, VA_L, VA_K, TLS, TFD, output)` y dos particiones exhaustivas —`country_groups` (por defecto `USA` frente a `ROW` en la ICIO OCDE) y `sector_groups` (por defecto `Tradable` CIIU `A–C` frente a `Services`)—, el puente conserva el valor agregado total al mapear las cuotas empíricas de gasto hacia `ReallocationEconomy`:

$$
\theta_{c,s} = \frac{\sum_{i} E_{i,s,c}}{\sum_{s'} \sum_{i} E_{i,s',c}}, \qquad \text{home\_share}_{c,s} = \frac{E_{c,s,c}}{\sum_{i} E_{i,s,c}}, \qquad \alpha^{\text{pool}} = \frac{\sum_{c,s} VA^K_{c,s}}{\sum_{c,s} (VA^L_{c,s} + VA^K_{c,s})}.
$$

La oferta laboral de cada país se normaliza proporcionalmente al valor agregado laboral regional ($\sum_c L_c = 2$), mientras la base de flujos (`"final_demand"`, `"consumption"` o `"total_absorption"`) selecciona si $E_{i,s,c}$ mide la demanda final total `C + I + Cx`, el consumo `C` o la absorción bruta `Z + F`.

### Presupuestos de Grupos y Bienestar (`evaluate_transition_household_incidence`)

En cada fecha $t \in \{0, \dots, T-1\}$ de una transición `ReallocationTransitionResult` convergida y auditada, el gasto nominal agregado en consumo del país $c$ cumple:

$$
P_{c,t} C_{c,t} = \sum_{s=0}^{1} w_{t,c,s} L_{t,c,s} + \sum_{s=0}^{1} r_{t,c,s} K_{t,c,s} + T^{\text{reb}}_{t,c} - P_{c,t}\bigl(A_{c,t} - C_{c,t}\bigr).
$$

Cada grupo de hogares $g \in \mathcal{G}_c$ recibe pesos salariales sectoriales $(\lambda^L_{g,0}, \lambda^L_{g,1})$, peso de capital $\lambda^K_g$, peso de devolución arancelaria $\lambda^R_g$ y cuota de desembolso en inversión/ajuste $\lambda^I_g$, cada uno sumando $1.0$ dentro del país $c$:

$$
m_{t,g} = \sum_{s=0}^{1} \lambda^L_{g,s} w_{t,c,s} L_{t,c,s} + \lambda^K_g \sum_{s=0}^{1} r_{t,c,s} K_{t,c,s} + \lambda^R_g T^{\text{reb}}_{t,c} - \lambda^I_g P_{c,t}\bigl(A_{c,t} - C_{c,t}\bigr).
$$

Dado que cada componente de ingreso y gasto particiona exactamente el total nacional, la suma del gasto nominal de los grupos iguala de forma idéntica al consumo nacional ($\sum_{g \in \mathcal{G}_c} m_{t,g} = P_{c,t} C_{c,t}$) y el ingreso bruto grupal iguala a la absorción bruta ($P_{c,t} A_{c,t}$) en cada fecha.

Las preferencias de demanda se calibran en los precios de referencia del estado estacionario inicial $p^{\text{rel}}_{0,c,s} = 1$ mediante `puremacro.trade.household.calibrate_household` bajo `"fixed_baskets"`, `"cobb_douglas"`, `"stone_geary"` o `"ces"`. A lo largo de la senda, los precios relativos sectoriales $p^{\text{rel}}_{t,c,s} = P_{t,c,s} / P_{0,c,s}$ y presupuestos $m_{t,g}$ producen:

- **Índice de Costo de Vida de Konüs e Ingreso Real**: $P^{\text{COL}}_{t,g} = e_g(p^{\text{rel}}_t, u_{0,g}) / m_{0,g}$ y $m^{\text{real}}_{t,g} = m_{t,g} / P^{\text{COL}}_{t,g}$.
- **Bienestar Monetario por Periodo**: Variación equivalente $EV_{t,g} = e_g(1, u_{t,g}) - m_{0,g}$ y variación compensatoria $CV_{t,g} = m_{t,g} - e_g(p^{\text{rel}}_t, u_{0,g})$.
- **Resúmenes en Valor Presente de la Senda**: Valor presente $PV\_EV_g = \sum_{t=0}^{T-1} \beta^t EV_{t,g} + \frac{\beta^T}{1-\beta} EV^{\text{term}}_g$ y porcentaje equivalente en consumo intertemporal CRRA $\text{CEV}_g$ evaluado sobre $c^{\text{eq}}_{t,g} = m_{0,g} + EV_{t,g}$.
- **Brecha de Desigualdad entre Grupos**: Rango máximo-menos-mínimo y desviación estándar ponderada por presupuesto de $EV\%_{t,g}$ por país a lo largo de $t = 0, \dots, T-1$.

## Flujo de Calibración e Incidencia

El ejemplo siguiente calibra `(USA, ROW) x (Tradable, Services)` desde la tabla ICIO OCDE 2019 incluida, aplica un arancel estadounidense del 20% a las importaciones transables, resuelve la transición dinámica y evalúa la incidencia por grupos:

```python
from puremacro.trade.reallocation import solve_reallocation_transition
from puremacro.trade.reallocation_incidence import (
    calibrate_reallocation_from_icio,
    evaluate_transition_household_incidence,
    make_sector_household_groups,
    reallocation_tariffs_from_scenarios,
)
from puremacro.trade.scenarios import TariffScenario

cal = calibrate_reallocation_from_icio()
scenario = TariffScenario(
    importer="USA",
    tariffs={("ROW", "Tradable"): 0.20},
    name="us_tradable_20pct",
)
tariffs = reallocation_tariffs_from_scenarios(scenario, cal.economy, horizon=25)
transition = solve_reallocation_transition(cal.economy, tariffs)

groups = make_sector_household_groups(
    cal.economy,
    include_capital_owners=True,
    basket_tilt=0.05,
)
incidence = evaluate_transition_household_incidence(
    transition,
    groups,
    rule="cobb_douglas",
)
print(incidence.summary(lang="en"))
```

## Regiones Personalizadas y Preferencias Heterogéneas

Quien llama puede definir particiones ICIO 2x2 arbitrarias (como `MEX+USA` frente a `ROW`), inspeccionar la exposición inflacionaria desagregada por canasta de gasto y combinar distintas reglas de demanda (`fixed_baskets`, `stone_geary`, `ces`) en una misma evaluación:

```python
from puremacro.trade.data import load_icio_data
from puremacro.trade.reallocation import solve_reallocation_transition
from puremacro.trade.reallocation_incidence import (
    HouseholdGroupSpec,
    calibrate_reallocation_from_icio,
    evaluate_transition_household_incidence,
    reallocation_tariffs_from_scenarios,
)
from puremacro.trade.scenarios import TariffScenario

icio = load_icio_data(return_structured=True)
mex_usa = ("MEX", "USA")
row = tuple(c for c in icio.country_codes if c not in mex_usa)

cal_na = calibrate_reallocation_from_icio(
    icio,
    country_groups={"MEX+USA": mex_usa, "ROW": row},
    sector_groups={"Goods": ("A", "B", "C"), "Services": tuple(s for s in icio.sector_codes if s not in ("A", "B", "C"))},
    expenditure_basis="consumption",
)
tariffs = reallocation_tariffs_from_scenarios(
    TariffScenario(importer="MEX+USA", tariffs={("ROW", "Goods"): 0.15}),
    cal_na.economy,
    horizon=20,
)
tr = solve_reallocation_transition(cal_na.economy, tariffs)

custom_groups = (
    HouseholdGroupSpec(
        name="NA_Goods_Workers_LowElasticity",
        country="MEX+USA",
        labor_weights=(1.0, 0.0),
        capital_weight=0.0,
        rebate_weight=0.5,
        investment_weight=0.0,
        expenditure_shares=(0.65, 0.35),
        rule="stone_geary",
        supernumerary_share=0.5,
    ),
    HouseholdGroupSpec(
        name="NA_Services_and_Capital",
        country="MEX+USA",
        labor_weights=(0.0, 1.0),
        capital_weight=1.0,
        rebate_weight=0.5,
        investment_weight=1.0,
        expenditure_shares=(0.20, 0.80),
        rule="ces",
        ces_elasticity=2.0,
    ),
    HouseholdGroupSpec(
        name="ROW_Representative",
        country="ROW",
        labor_weights=(1.0, 1.0),
        non_labor_weight=1.0,
        rule="cobb_douglas",
    ),
)
res = evaluate_transition_household_incidence(tr, custom_groups)
print(res.inequality_dataframe().head())
```

## Límites Transparentes de Calibración y Divulgación

Las tablas insumo-producto de un solo año no identifican parámetros intertemporales, de sustitución ni de ajuste. Cada llamada a `calibrate_reallocation_from_icio` registra explícitamente las siguientes fronteras de modelización dentro de `result.audit.stylized_assumptions` y `result.diagnostic`:

| Parámetro o Supuesto | Origen en ICIO | Estado de Identificación y Divulgación |
|---|---|---|
| `sector_weights` | Demanda agregada observada | Identificado por cuotas de `final_demand`, `consumption` o `total_absorption`. |
| `home_shares` | Flujos domésticos e importados | Identificado bajo la base elegida; el alto sesgo local de `ROW` (`~0.98`) exige un `sourcing_cost` bajo. |
| `labor_supply` | `labor_va` agregado | Escala relativa regional de ingreso laboral normalizada para sumar `2.0`. |
| `capital_share` | `capital_va / va` incluido | El archivo OCDE 2023 usa la convención `2/3` trabajo y `1/3` capital; promediado a un escalar global. |
| `productivity` | No identificado por separado | Los flujos nominales en USD de un año confunden precios y PTF física; se fija en `ones((2, 2))`. |
| `beta`, `delta`, `risk_aversion` | Ausentes en IO estática | Supuestos estilizados anuales (`0.96`, `0.08`, `2.0`) en lugar de momentos estimados. |
| `trade_elasticity`, costos de ajuste | Ausentes en IO estática | Supuestos estilizados (`trade_elasticity=4.0`, `capital_cost=4.0`, `labor_cost=5.0`, `sourcing_cost=0.01`). |
| Insumos intermedios (`Z`) y aranceles | Omitidos o agregados en `TLS`/`TFD` | `ReallocationEconomy` modela valor agregado bajo autarquía financiera (balanza comercial nula) y aranceles base cero. |

## Auditorías de Conservación, Diagnósticos y Referencia de API

Cada evaluación aplica verificaciones estrictas de rechazo antes de devolver resultados:

- **Solo Transiciones Convergidas y Auditadas**: Pasar una instancia `ReallocationTransitionResult` no convergida o alterada lanza `TransitionError` (subclase de `ValueError`).
- **Partición Exacta de Factores y Presupuestos**: Sumas nacionales inválidas de pesos laborales, de capital, de devoluciones o de inversión (distintas de `1.0`), cuotas de gasto negativas o que no suman `1.0`, o presupuestos grupales no positivos lanzan `ValueError` / `TransitionError`.
- **Verificación de Identidad con Choque Cero**: Cuando `tariffs == 0` en toda la senda, `evaluate_transition_household_incidence` verifica $|EV_{t,g}| < 10^{-12}$, $|CV_{t,g}| < 10^{-12}$ y $|P^{\text{COL}}_{t,g} - 1| < 10^{-12}$ para todo grupo y fecha.
- **Diagnósticos Estandarizados y Exportación Bilingüe**: Tanto `ReallocationCalibrationResult` como `TransitionHouseholdIncidenceResult` adjuntan un `SolverDiagnosticRecord` (`puremacro.solvers.diagnostics`) y exponen `.summary(lang="en"|"es")`, `.to_dataframe()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` y `.to_dict()`.

Guías relacionadas: [trade_reallocation.md](trade_reallocation.md) · [trade_household.md](trade_household.md) · [trade_distributional.md](trade_distributional.md) · [trade_stability.md](trade_stability.md).
