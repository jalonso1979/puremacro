> 🇪🇸 Español · 🇬🇧 [English](../data_ecosystem.md)

# Ecosistema de Datos Macroeconómicos Globales

`puremacro` proporciona un ecosistema de datos unificado, sin necesidad de claves API y con almacenamiento en caché para la investigación macroeconómica internacional, el análisis de políticas y el pronóstico. Todos los recolectores operan a través de la capa de caché unificada (`puremacro.fetch._http.cached_get`), cumplen estrictamente con la seguridad de Pyodide (cero importaciones de `requests` a nivel de módulo) y devuelven marcos de datos en formato largo estandarizados (`code, date, variable, value, source`).

Además de los recolectores de series individuales, los **constructores modulares de paneles** de alto nivel (`puremacro.build_climate_panel` y `puremacro.build_financial_panel`) ensamblan conjuntos de datos multipaís y multifrecuencia con agregación temporal automatizada y seguimiento de datos faltantes.

---

## 1. Emisiones y Gases de Efecto Invernadero (`puremacro.fetch.emissions`)

Recolecta datos internacionales de emisiones de los Indicadores del Desarrollo Mundial (WDI) del Banco Mundial y la base de datos de gases de efecto invernadero SDMX de la OCDE.

```python
from puremacro.fetch.emissions import (
    fetch_wdi_emissions,
    fetch_oecd_ghg_emissions,
    fetch_emissions_panel,
)

# Obtener emisiones de CO2 per cápita y GEI totales del Banco Mundial WDI
df_wdi = fetch_wdi_emissions(
    codes=["USA", "DEU", "JPN", "GBR", "MEX"],
    indicators=["EN.GHG.CO2.PC.CE.AR5", "EN.GHG.ALL.MT.CE.AR5"],
    start_year=2000,
)

# Obtener cuentas sectoriales de GEI de la OCDE (energía, transporte, manufactura)
df_oecd = fetch_oecd_ghg_emissions(
    codes=["USA", "DEU"],
    sectors=["1A1", "1A2", "1A3"],  # Energía, Industria, Transporte
    start_period="2010",
)

# Panel unificado de emisiones multifuente en una sola llamada
df_emissions = fetch_emissions_panel(
    codes=["USA", "DEU", "JPN"],
    start_year=2000,
)
```

### Indicadores Compatibles

| Módulo / Función | Código de Indicador / Flujo | Descripción |
|---|---|---|
| `fetch_wdi_emissions` | `EN.GHG.CO2.MT.CE.AR5` | Emisiones totales de CO2 (millones de toneladas métricas de CO2 equivalente) |
| `fetch_wdi_emissions` | `EN.GHG.CO2.PC.CE.AR5` | Emisiones de CO2 per cápita (toneladas métricas por habitante) |
| `fetch_wdi_emissions` | `EN.GHG.CH4.MT.CE.AR5` | Emisiones de metano (CH4) |
| `fetch_wdi_emissions` | `EN.GHG.N2O.MT.CE.AR5` | Emisiones de óxido nitroso (N2O) |
| `fetch_oecd_ghg_emissions` | `OECD.ENV.EPI,DSD_AIR_GHG@DF_AIR_GHG,` | Cuentas sectoriales de GEI (sectores IPCC: `1A1` Energía, `1A2` Industria, `1A3` Transporte, `1A4b` Residencial) |

---

## 2. Balances Energéticos y Transición (`puremacro.fetch.energy_transition`)

Recolecta generación de electricidad por fuente, cuotas de transición a energías limpias y balances de consumo de energía primaria de repositorios internacionales abiertos (OCDE, Ember, Energy Institute).

```python
from puremacro.fetch.energy_transition import (
    fetch_electricity_generation_mix,
    fetch_energy_transition_panel,
)

# Obtener matriz de generación eléctrica (renovables, fósiles, hidroeléctrica, nuclear)
df_mix = fetch_electricity_generation_mix(
    codes=["USA", "DEU", "FRA", "ESP"],
    start_year=2010,
)

# Panel completo de transición energética con cuota de energía limpia
df_energy = fetch_energy_transition_panel(
    codes=["USA", "DEU"],
    start_year=2000,
)
```

### Métricas Clave Suministradas
- `clean_elec_share`: Porcentaje de electricidad generada a partir de fuentes con cero emisiones de carbono (renovables, hidroeléctrica, nuclear).
- `fossil_elec_share`: Porcentaje de electricidad generada a partir de combustibles fósiles (carbón, gas, petróleo).
- `elec_gen_twh`: Generación eléctrica total anual en teravatios-hora (TWh).
- `energy_intensity`: Consumo energético por unidad de PIB real.

---

## 3. Suites de Precios de Referencia de Materias Primas (`puremacro.fetch.commodities`)

Índices estandarizados de precios de materias primas al contado y de futuros derivados del Pink Sheet del Banco Mundial y del Sistema de Precios Primarios de Materias Primas (PCPS) del FMI.

```python
from puremacro.fetch.commodities import (
    fetch_commodity_prices,
    fetch_commodity_indices,
    COMMODITY_CATEGORIES,
)

# Obtener niveles de precios brutos (en unidades nativas: USD/barril, USD/mmbtu, USD/tonelada)
df_prices = fetch_commodity_prices(
    commodities=["oil_brent", "oil_wti", "gas_eu", "copper", "gold"],
    frequency="M",
    start_date="2000-01-01",
)

# Obtener índices compuestos de grupos (2010=100)
df_indices = fetch_commodity_indices(
    categories=["energy", "metals", "agriculture", "fertilizers"],
    frequency="Q",
)
```

### Grupos de Materias Primas Disponibles

| Categoría | Índices y Puntos de Referencia | Unidades |
|---|---|---|
| **Energía** | Petróleo Brent (`oil_brent`), Petróleo WTI (`oil_wti`), Gas Natural Europeo (`gas_eu`), Henry Hub EE.UU. (`gas_us`), Carbón Australia (`coal_au`) | $/bbl, $/mmbtu, $/mt |
| **Metales Industriales** | Cobre (`copper`), Aluminio (`aluminum`), Mineral de Hierro (`iron_ore`), Níquel (`nickel`), Zinc (`zinc`) | $/mt, $/dmt |
| **Metales Preciosos** | Oro (`gold`), Plata (`silver`), Platino (`platinum`) | $/onza troy |
| **Agricultura y Alimentos** | Trigo (`wheat`), Maíz (`maize`), Soja (`soybeans`), Café (`coffee`), Cacao (`cocoa`) | $/mt, $/kg |
| **Fertilizantes** | Urea (`urea`), Roca Fosfórica (`phosphate_rock`), DAP (`dap`) | $/mt |

---

## 4. Estabilidad Financiera e Indicadores Macroprudenciales Internacionales (`puremacro.fetch.financial`)

Obtiene indicadores internacionales de estabilidad financiera, riesgo soberano y mercados crediticios del Banco de Pagos Internacionales (BPI / BIS), FRED y bancos centrales.

```python
from puremacro.fetch.financial import (
    fetch_sovereign_yields,
    fetch_policy_rates,
    fetch_credit_gap_panel,
    fetch_property_prices,
    fetch_financial_conditions,
)

# Rendimientos soberanos de referencia a 10 y 2 años + diferencial de plazos (spread 10Y-2Y)
df_yields = fetch_sovereign_yields(
    codes=["USA", "DEU", "GBR", "JPN"],
    maturities=["10Y", "2Y", "spread_10y2y"],
    frequency="M",
)

# Tasas de interés de política de bancos centrales (FED, BCE, BOE, BOJ)
df_cb = fetch_policy_rates(
    codes=["USA", "EA20", "GBR", "JPN"],
    frequency="M",
)

# Brecha crédito/PIB del BIS (indicador de colchón de capital anticíclico de Basilea III)
df_credit = fetch_credit_gap_panel(
    codes=["USA", "DEU", "GBR"],
    start_year=2000,
)

# Índice real de precios de la vivienda residencial del BIS
df_house = fetch_property_prices(
    codes=["USA", "DEU", "GBR"],
    start_year=2000,
)

# Indicadores de condiciones financieras y diferenciales de estrés
df_fci = fetch_financial_conditions(
    indicators=["NFCI", "TEDRATE", "BAMLH0A0HYM2"],
    start_date="2000-01-01",
)
```

---

## 5. Constructores Modulares de Paneles (`puremacro.build_climate_panel` y `build_financial_panel`)

Constructores de paneles de alto nivel que integran conjuntos balanceados o no balanceados listos para estimación econométrica en VAR, proyecciones locales o regresiones de panel.

### `build_climate_panel`

Combina emisiones, balances energéticos e indicadores macroeconómicos en un panel armonizado.

```python
from puremacro import build_climate_panel

panel = build_climate_panel(
    codes=["USA", "DEU", "JPN", "GBR", "MEX"],
    start_year=2000,
    frequency="A",  # "A" para Anual, "Q" para Trimestral (interpolado/agregado)
    include_gdp=True,
    include_energy=True,
)

print(panel.head())
```

**Columnas Clave**:
- `code`, `date`: Código ISO-3 del país e índice de fechas.
- `co2_total`, `co2_per_capita`, `ghg_total`: Métricas de emisiones.
- `clean_elec_share`, `fossil_elec_share`: Métricas de transición energética.
- `gdp`, `population`: Denominadores para normalización per cápita e intensidad.
- `imputed_*`: Banderas booleanas que indican series interpoladas.

### `build_financial_panel`

Combina curvas de rendimiento soberano, tasas de política de bancos centrales, brechas de crédito del BIS y precios de referencia de materias primas alineados a horizontes de pronóstico macroeconómico.

```python
from puremacro import build_financial_panel

panel = build_financial_panel(
    codes=["USA", "DEU", "GBR", "JPN"],
    start_date="2000-01-01",
    frequency="Q",  # "M" para Mensual, "Q" para Trimestral agregado
    include_commodities=True,
    include_credit=True,
)

print(panel.head())
```

**Columnas Clave**:
- `yield_10y`, `yield_2y`, `term_spread`: Curva de rendimientos de deuda soberana.
- `policy_rate`: Tasa de interés de política del banco central.
- `credit_gap`: Brecha crédito/PIB del BIS.
- `oil_brent`, `copper`, `gold`: Precios de referencia de materias primas.

---

## 6. Garantías Arquitectónicas y Pruebas Fuera de Línea

1. **Compatibilidad con Pyodide y Cero Dependencias de Red en Importación**:
   La importación de `puremacro`, `puremacro.fetch` o cualquiera de los constructores de paneles jamás realiza llamadas de red ni requiere `requests` a nivel de módulo.
2. **Artefactos Fijos para Pruebas Fuera de Línea (Fixtures)**:
   Cada recolector incluye respuestas pregrabadas en `tests/data/`, permitiendo que el conjunto completo de pruebas unitarias y CI se ejecute al 100 % sin conexión.
3. **Caché Unificada**:
   Las llamadas en vivo almacenan entradas con marca de tiempo en `data/raw/` con gestión automática de tiempo de vida (TTL) y alertas de datos desactualizados.
