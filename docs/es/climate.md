> 🇬🇧 [English](../climate.md) · 🇪🇸 Español

# Macroeconomía del Clima (DICE)

`puremacro.climate` incorpora un simulador compacto del modelo DICE de William Nordhaus cuyo propósito es cuantificar el impacto de un *escenario* de política ambiental predefinido. Dado un sendero de impuesto al carbono, una sensibilidad climática y un coeficiente de daños económicos, calcula la trayectoria conjunta 2020–2165 de producción, emisiones de gases de efecto invernadero, concentración de carbono atmosférico, aumento de temperatura y daños sobre el PIB mundial, permitiendo comparar alternativas de política dentro de un marco contable internamente coherente sin licencias de GAMS ni compiladores de C.

```python
from puremacro.climate import simulate_dice_model, DICEResult

res = simulate_dice_model()  # Valores por defecto: 2020-2165, impuesto $40/tCO2, +2%/año
print(res.summary())
```

Dos precisiones metodológicas cruciales sobre el alcance del módulo:
- **Es una simulación determinista hacia adelante, no un problema de optimización.** No maximiza bienestar intertemporal ni resuelve ecuaciones de Euler. La tasa de ahorro mundial se mantiene fija en 0.22 y la tasa de mitigación de emisiones se determina a partir de la condición de primer orden estática frente al sendero impositivo suministrado.
- **La columna `social_cost_of_carbon` refleja el sendero impositivo ajustado.** Representa el precio del carbono suministrado corregido por dos veces la proporción contemporánea de daños económicos.

---

## Parámetros de `simulate_dice_model()`

Todos los argumentos se pasan exclusivamente por clave (*keyword-only*):

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `n_periods` | `30` | Número de etapas. Con el paso temporal predeterminado, 30 etapas cubren 2020–2165 |
| `time_step_years` | `5` | Años transcurridos en cada etapa del modelo |
| `start_year` | `2020` | Año inicial de calibración del estado de la economía y la atmósfera |
| `carbon_tax_initial` | `40.0` | Precio inicial del carbono en el año base ($/tCO2). `0.0` modela un escenario sin política |
| `carbon_tax_growth` | `0.02` | Tasa de crecimiento **anual** del impuesto al carbono |
| `climate_sensitivity` | `3.0` | Sensibilidad climática al equilibrio ante una duplicación de CO2 (°C) |
| `damage_coef` | `0.00236` | Coeficiente $a$ en la función cuadrática de daño $D(T) = a T^2$ |
| `discount_rate` | `0.015` | Tasa de descuento pura (sin efecto en la trayectoria hacia adelante) |

---

## Estructura del resultado `DICEResult`

El objeto devuelto es una dataclass congelada cuyo campo principal `trajectories` es un DataFrame indexado por año:

| Columna | Unidad | Significado económico |
|---|---|---|
| `output_gross` | Billones de USD/año | Producto mundial bruto antes de daños climáticos |
| `output_net` | Billones de USD/año | Producto neto disponible tras descontar los daños |
| `emissions` | GtCO2/año | Emisiones globales de CO2 derivadas de la producción |
| `carbon_atmosphere` | GtC | Masa de carbono acumulada en la atmósfera |
| `temperature_surface` | °C sobre nivel preindustrial | Calentamiento medio global de la superficie |
| `damages_fraction` | Proporción [0, 1] | Fracción del PIB perdida debido al aumento térmico |
| `abatement_fraction` | Proporción [0, 1] | Fracción de emisiones abatidas por la política de carbono |

```python
# Gráficos de escenarios listos para publicación
res.plot()
print(res.to_latex())
```
