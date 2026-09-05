> 🇬🇧 [English](../gertler_karadi.md) · 🇪🇸 Español

# Modelo DSGE de Gertler y Karadi (2011) con fricciones financieras

El módulo `puremacro.dsge.gertler_karadi` implementa el modelo canónico macrofinanciero de equilibrio general dinámico y estocástico (DSGE) con fricciones en la intermediación bancaria, amplificación de balance y política de crédito no convencional desarrollado por **Gertler y Karadi (2011, *Journal of Monetary Economics*)**, incorporando compatibilidad con regímenes que se activan ocasionalmente mediante **OccBin (Guerrieri e Iacoviello 2015)**.

A raíz de la crisis financiera global de 2007–2008, la modelización macroeconómica moderna reconoció que las perturbaciones en la intermediación financiera no son meros reflejos pasivos del ciclo económico, sino catalizadores primarios de severa amplificación macroeconómica. El marco de Gertler y Karadi (GK 2011) modeliza explícitamente a los intermediarios financieros (bancos) que financian las compras de capital productivo de las empresas emitiendo depósitos a los hogares, en presencia de un problema de agencia endógeno.

---

## 1. Estructura teórica del modelo

### 1.1 Balance bancario y fricción de agencia

Los bancos canalizan el ahorro desde los hogares hacia las empresas no financieras. El balance agregado del sector bancario iguala el valor total de los activos intermediados $S_t = Q_t K_t$ a la suma del patrimonio neto bancario $N_t$ y los depósitos captados de los hogares $B_t$:

$$Q_t K_t = N_t + B_t$$

Los banqueros maximizan el valor esperado del patrimonio acumulado que transferirán a sus respectivos hogares al momento de su retiro. Para introducir fricciones financieras, GK (2011) formulan un **problema de agencia por riesgo moral**:
- Al cierre de cada período, el banquero puede desviar una fracción $\lambda_b \in (0, 1)$ de los activos gestionados $Q_t K_t$ directamente hacia su hogar.
- Si el banquero desvía fondos, la entidad quiebra, los depositantes se incautan de la fracción restante $(1 - \lambda_b)$ y el banco es liquidado.

Para que los hogares depositen voluntariamente sus fondos, el valor de continuación del banco $V_t$ debe satisfacer la **restricción de compatibilidad de incentivos**:

$$V_t \ge \lambda_b Q_t K_t$$

El valor de continuación puede descomponerse linealmente como:

$$V_t = \nu_t Q_t K_t + \eta_t N_t$$

donde $\nu_t$ representa el valor marginal de expandir los activos del banco (manteniendo fijo el patrimonio neto) y $\eta_t$ mide el valor marginal de incrementar el patrimonio neto (manteniendo fijos los activos).

Cuando la restricción de incentivos resulta vinculante, determina de forma endógena el **ratio de apalancamiento bancario** $\phi_t$:

$$\phi_t \equiv \frac{Q_t K_t}{N_t} = \frac{\eta_t}{\lambda_b - \nu_t}$$

Esta restricción fija endógenamente el **diferencial de crédito (*credit spread*)** o prima de financiamiento externo:

$$\text{Spread}_t \equiv \mathbb{E}_t \left[ R_{k, t+1} - R_{t+1} \right]$$

### 1.2 Amplificación macroeconómica y perturbaciones a la calidad del capital

El experimento principal analizado en Gertler y Karadi (2011) es una **perturbación a la calidad del capital** $\xi_t$.

El stock efectivo de capital físico evoluciona según $K_{t+1} = \xi_{t+1} [(1 - \delta) K_t + I_t]$. Una innovación negativa $\varepsilon_\xi < 0$:
1. Reduce de forma directa la productividad técnica y física del capital instalado.
2. Deprime de inmediato el precio de mercado del activo de capital $Q_t$.
3. Induce cuantiosas pérdidas patrimoniales en los balances de los bancos. Dado que el sistema bancario opera con un apalancamiento basal $\phi \approx 4$, una depreciación del 1% en los activos contrae el patrimonio neto en aproximadamente **un 4%**:
   $$\frac{d N_t}{N_{ss}} \approx \phi \cdot \frac{d Q_t}{Q_{ss}}$$
4. Obliga a los bancos a restringir drásticamente la intermediación o elevar los diferenciales de crédito $\mathbb{E}_t[R_{k,t+1} - R_{t+1}]$ en cientos de puntos básicos para restablecer la confianza de los depositantes.
5. Desencadena una contracción profunda y duradera de la inversión agregada y del PIB.

---

## 2. Solucionadores duales

El paquete `puremacro.dsge.gertler_karadi` incorpora dos metodologías de resolución:

1. **Solucionador lineal QZ de Klein (1998) (`method='klein'`)**:  
   Resuelve perturbaciones racionales de primer orden en torno al estado estacionario determinista utilizando la descomposición de Schur generalizada de `puremacro.dsge.klein`. Es idóneo para simulaciones lineales veloces.
2. **Recursión lineal por tramos con OccBin (`method='occbin'`)**:  
   Resuelve modelos dinámicos con cambios de régimen que se activan ocasionalmente conforme a Guerrieri e Iacoviello (2015):
   - **Política de crédito no convencional (`constraint_type='credit_policy'`)**: El banco central intermedia crédito directamente adquiriendo activos privados cuando el diferencial de crédito supera un umbral crítico (por ejemplo, 100 puntos básicos):
     $$\psi_t = \nu_g \cdot \max \left( 0, \, \text{Spread}_t - \text{umbral} \right)$$
   - **Límites macroprudenciales de apalancamiento (`constraint_type='leverage_cap'`)**: Impone un techo regulatorio estricto sobre el apalancamiento bancario $\phi_t \le \phi_{max}$.

---

## 3. Calibración canónica (GK 2011, Tabla 1)

| Parámetro | Valor | Interpretación económica |
|---|---|---|
| `beta` | $0.99$ | Factor de descuento subjetivo trimestral (tasa natural $R_{ss} \approx 4.0\%$ anual) |
| `sigma` | $1.0$ | Elasticidad de sustitución intertemporal (utilidad logarítmica) |
| `h` | $0.815$ | Hábito en el consumo |
| `varphi` | $0.276$ | Inverso de la elasticidad de Frisch de oferta laboral |
| `alpha` | $0.33$ | Participación del capital en la función de producción |
| `delta` | $0.025$ | Tasa de depreciación física trimestral (10% anual) |
| `eta_i` | $1.728$ | Parámetro de costes de ajuste de la inversión |
| `theta_b` | $0.972$ | Probabilidad de supervivencia del banquero (antigüedad media $\approx 9$ años) |
| `lambda_b`| $0.381$ | Fracción de activos susceptible de desvío (intensidad del riesgo moral) |
| `omega_b` | $0.002$ | Dotación inicial transferida a los nuevos banqueros |
| `gamma` | $0.779$ | Parámetro de Calvo de rigidez de precios |
| `rho_xi` | $0.66$ | Persistencia autorregresiva del choque a la calidad del capital |

---

## 4. Ejemplos de uso y código ejecutable

### Simulación de un choque a la calidad del capital bajo Klein y OccBin

```python
import numpy as np
from puremacro.dsge.gertler_karadi import (
    solve_gertler_karadi,
    solve_steady_state,
)

# 1. Inspección del estado estacionario determinista
ee = solve_steady_state()
print(f"Apalancamiento bancario phi en EE : {ee['phi']:.2f}x")
print(f"Diferencial de crédito anualizado : {ee['spread_ann'] * 10000:.1f} pbs")

# 2. Simulación de perturbación a la calidad del capital (-5%) con el solucionador lineal de Klein
res_klein = solve_gertler_karadi(
    shock_type="capital_quality",
    shock_size=-0.05,
    horizon=40,
    method="klein",
)

# 3. Simulación con OccBin incorporando intervención del banco central en crédito
res_occbin = solve_gertler_karadi(
    shock_type="capital_quality",
    shock_size=-0.05,
    horizon=40,
    method="occbin",
    constraint_type="credit_policy",
    threshold=0.0025,  # Intervención si el diferencial excede 100 pbs anuales
)

# 4. Comparación de resultados entre regímenes
df_klein = res_klein.to_frame()
df_occbin = res_occbin.to_frame()

print("Caída máxima del patrimonio bancario (Klein) :", df_klein["n"].min())
print("Caída máxima del patrimonio bancario (OccBin):", df_occbin["n"].min())
print("Pico del diferencial de crédito (Klein, pbs) :", df_klein["prem"].max() * 40000)
print("Pico del diferencial de crédito (OccBin, pbs):", df_occbin["prem"].max() * 40000)

# 5. Informe estructurado y visualización gráfica
print(res_occbin.summary())
fig = res_occbin.plot()
```

---

## 5. Especificación completa de la API

### `solve_gertler_karadi`

```python
solve_gertler_karadi(
    params: Mapping[str, float] | None = None,
    shock_type: str = "capital_quality",
    shock_size: float = -0.05,
    horizon: int = 40,
    method: str = "occbin",
    constraint_type: str = "credit_policy",
    threshold: float | None = None,
    max_iter: int = 50,
) -> GertlerKaradiResult
```

#### Parámetros:
- `params`: Diccionario opcional para sobreescribir los parámetros canónicos de `GK2011_PARAMS`.
- `shock_type`: Identificador del choque: `'capital_quality'` ($\varepsilon_\xi$), `'tfp'` ($\varepsilon_a$) o `'monetary'` ($\varepsilon_r$).
- `shock_size`: Magnitud de la innovación en $t=0$ (por defecto `-0.05` para un choque contractivo del 5%).
- `horizon`: Longitud de la simulación en trimestres (por defecto `40`).
- `method`: Motor de resolución: `'occbin'` (lineal por tramos) o `'klein'` (perturbación lineal QZ).
- `constraint_type`: Restricción activa en OccBin: `'credit_policy'` o `'leverage_cap'`.
- `threshold`: Umbral numérico de activación del régimen (por defecto `0.0025` para 100 pbs de spread).
- `max_iter`: Número máximo de iteraciones retrógradas para el algoritmo de OccBin.

---

## 6. Interfaz de resultados

La clase `GertlerKaradiResult` almacena las trayectorias dinámicas y ofrece herramientas formales de reporte:

- **Atributos**:
  - `irf`: Diccionario con las trayectorias temporales $(T,)$ de todas las variables endógenas.
  - `variables`: Lista de identificadores de variables (`['y', 'c', 'i', 'q', 'k', 'n', 'phi', 'prem', ...]`).
  - `steady_state`: Diccionario con los valores del estado estacionario determinista.
  - `regime_history`: Vector booleano que registra los períodos en los cuales la restricción de OccBin estuvo activa.
  - `converged`: Estado de convergencia del solucionador numérico.
- **Métodos disponibles**:
  - `to_frame()`: Devuelve un `DataFrame` de pandas indexado por trimestres de simulación $t = 0, \dots, T-1$.
  - `.plot()`: Gráfico multipanel en Matplotlib que ilustra las trayectorias de Producto, Inversión, Patrimonio Bancario, Precio del Capital $Q$, Apalancamiento $\phi$ y Diferencial de Crédito.
  - `.summary()`: Informe técnico que desglosa los valores de estado estacionario, especificación del choque y picos de respuesta.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas estructuradas listas para su inclusión en publicaciones académicas.
