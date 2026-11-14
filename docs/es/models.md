> 🇬🇧 [English](../models.md) · 🇪🇸 Español

# HANK en el espacio de secuencias

Para el modelo publicado con horas endógenas y equilibrio de activos y trabajo,
consulte el [equilibrio HANK de un activo con referencia independiente](hank_equilibrium.md).
Las rutinas anteriores descritas abajo utilizan un cierre reducido y mantienen
sus propias APIs.

`puremacro.models` constituye la vertiente estructural del paquete: modelos que se resuelven para un estado estacionario y luego se proyectan a lo largo de una trayectoria de transición, sin compiladores, sin archivos MEX y sin Dynare. Todo el contenido de esta página se ejecuta de forma local y autónoma, sin acceder a la red.

```python
from puremacro.models import solve_hank_sequence_space

res = solve_hank_sequence_space(T=40, n_a=100, beta=0.985, r_ss=0.01,
                                phi_pi=1.5, kappa=0.1,
                                shock_magnitude=0.0025, shock_rho=0.7)
print(res.summary())
```

Esto simula un choque monetario contractivo de 25 puntos básicos en una economía nuevo keynesiana de un activo con agentes heterogéneos, y devuelve la respuesta en **0.21 segundos**.

## Contenido del submódulo

| Módulo | Ruta de importación | Qué resuelve |
|---|---|---|
| `hank_sequence_space` | `puremacro.models` | HANK de un activo: estado estacionario por EGM, algoritmo exacto Fake News en $\mathcal{O}(T^2)$, transferencias fiscales focalizadas y transición de equilibrio general mediante un sistema lineal $T \times T$ |
| `hank_fiscal` | `puremacro.models.hank_fiscal` | Cierres fiscales explícitos financiados para HANK (`balanced_budget` y `debt_rule` con impuestos proporcionales o de suma fija), auditando bienes, RGI y vaciado de activos (Walras) |
| `monetary_transmission` | `puremacro.models` | Transmisión de política monetaria y macroprudencial HANK vs. RANK con descomposición Kaplan-Moll-Violante |
| `dmp_regime_dependent` | `puremacro.models` | Modelo DMP con empresa representativa, salarios rígidos de Hall, coste de vacantes dependiente del régimen y choque de PTF |
| `nested_dmp` (paquete) | `puremacro.models.nested_dmp` | DMP con empresas heterogéneas y creencias bayesianas sobre la regla de política: estado estacionario, FRI con previsión perfecta, estimación por coincidencia de FRI, solución estocástica recursiva y análisis de bienestar |
| `smm` | `puremacro.models.smm` | Cargador de momentos y función objetivo para estimar `dmp_regime_dependent` frente a FRI de proyecciones locales |

Para el simulador de transmisión monetaria HANK frente a RANK y su laboratorio en el navegador (`curso/site/labs/`), consulte [Simulador de transmisión monetaria: HANK frente a RANK](monetary_transmission.md).

`puremacro.models.__all__` reexporta:
`DMPParameters`, `DMPState`, `dmp_steady_state`, `dmp_irf`,
`SequenceSpaceHANKResult`, `solve_hank_sequence_space`,
`FakeNewsResult`, `FiscalTransferResult`,
`fake_news_algorithm`, `simulate_targeted_transfer`,
`MonetaryTransmissionSimulator` y `MonetaryTransmissionResult`. `hank_fiscal`, `nested_dmp` y `smm` se importan mediante sus propias rutas de submódulo.

## Ventajas del método en el espacio de secuencias

Una solución global de un modelo de agentes heterogéneos trata la distribución como una variable de estado. Se aproxima (momentos de Krusell-Smith, bases de proyección, redes neuronales), se resuelve una ecuación de Bellman sobre el espacio conjunto (idiosincrásico y agregado) y el coste crece exponencialmente con la dimensión del estado agregado (*maldición de la dimensionalidad*).

Auclert, Bardóczy, Rognlie y Straub (2021) observan que si sólo se desea la respuesta **lineal** ante un choque agregado, jamás se necesita manipular ese objeto dimensional. Se requiere únicamente una matriz de tamaño $T \times T$ por cada variable de entrada — el jacobiano del consumo agregado respecto al precio en la fecha $s$ — y el equilibrio macroeconómico se convierte en un sistema lineal sobre la *secuencia* de agregados, no sobre un vector de estado.

Dos consecuencias computacionales directas:
- **El horizonte temporal es económico**: Con $n_a=20$ puntos de activos, la resolución completa toma 0.056 s para $T=40$, 0.182 s para $T=600$ y 0.415 s para $T=1000$. Multiplicar el horizonte por 25 sólo incrementa el tiempo por 7.4x, pues la resolución de $T \times T$ está limitada por BLAS y el llenado del jacobiano es de orden $\mathcal{O}(T^2)$.
- **El bloque de hogares se computa por separado**: Su coste escala con la malla de activos $n_a$ y no con $T$.

---

## 1. Algoritmo exacto Fake News (`fake_news_algorithm`)

Auclert et al. (2021) demostraron que calcular numéricamente el jacobiano de $T \times T$ por simulación directa hacia adelante requiere $T$ simulaciones independientes. El **Algoritmo Fake News** reduce esta complejidad a $\mathcal{O}(T^2)$ mediante la iteración hacia atrás de vectores de esperanza.

Define la *matriz de noticias falsas* $\mathcal{F}$, donde $\mathcal{F}_{t,s}$ mide la revisión en la expectativa de consumo en la fecha $t$ al recibir en $t=0$ la noticia de una perturbación que ocurrirá en la fecha $s$:
$$\mathcal{F}_{t,s} = (\mathbf{D}_{ss} \mathcal{E}_t)' \cdot d\mathbf{a}^*_s$$

Una vez construida $\mathcal{F}$, el jacobiano completo en el espacio de secuencias $\mathcal{J}$ se recupera mediante la identidad de acumulación fundamental:
$$\mathcal{J}_{t,s} = \mathcal{J}_{t-1,s-1} + \mathcal{F}_{t,s}$$

```python
from puremacro.models import fake_news_algorithm

# Cálculo exacto de jacobianos de consumo intertemporal
fn_res = fake_news_algorithm(T=40, n_a=100, beta=0.985, r_ss=0.01)
print(fn_res.summary())

# Matrices como DataFrames ordenados
df_jac = fn_res.to_frame(which="jacobian")
df_f   = fn_res.to_frame(which="fake_news")

# Mapa de calor doble (F y J) listo para publicación
fn_res.plot()

# Exportar tabla a LaTeX o Typst
print(fn_res.to_latex())
```

---

## 2. Transferencias fiscales focalizadas (`simulate_targeted_transfer`)

Los modelos de agentes heterogéneos permiten evaluar políticas distributivas que escapan a los modelos de agente representativo, como el efecto estímulo de cheques fiscales dirigidos a deciles específicos de riqueza e ingreso.

`puremacro.models.simulate_targeted_transfer` simula el impacto macroeconómico y distributivo de una transferencia fiscal:

```python
from puremacro.models import simulate_targeted_transfer

# Estímulo dirigido al 30% de hogares con menores tenencias de activos
transfer_res = simulate_targeted_transfer(
    ss_model=ss_res,          # solved steady state (distribution, policies, grid)
    target=[1, 2, 3],         # wealth deciles 1-3 (poorest 30%)
    amount=1.0,               # total outlay, in units of mean quarterly labour income
    T=30,
)

print(transfer_res.summary())
print("PMC agregada de impacto:", transfer_res.impact_mpc)
print("Multiplicador fiscal acumulado:", transfer_res.cumulative_multiplier)

# Gráfico de panel doble: respuesta dinámica del consumo e incidencia por deciles
transfer_res.plot()
```

---

## 3. Resolvedor paramétrico en el espacio de secuencias (`solve_hank_sequence_space`)

Junto con el motor Fake News, `solve_hank_sequence_space` proporciona un resolvedor analítico ultrarrápido que vincula el problema estático de los hogares con los jacobianos intertemporales:

```python
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,
    shock_rho=0.7,
)
```

Argumentos principales:
- `T`: Horizonte de truncamiento (longitud de la FRI y dimensión de los jacobianos).
- `beta`: Factor de descuento intertemporal del hogar y de la curva de Phillips nuevo keynesiana.
- `gamma`: Coeficiente de aversión relativa al riesgo (CRRA).
- `r_ss`: Tasa real trimestral en estado estacionario.
- `phi_pi`: Coeficiente de respuesta a la inflación en la regla de Taylor.
- `kappa`: Pendiente de la curva de Phillips nuevo keynesiana.
- `shock_magnitude`: Magnitud del choque monetario inicial en $t=0$.
- `shock_rho`: Persistencia autorregresiva AR(1) del choque.

> [!NOTE]
> **Alcance de `shock_var="G"` en `hank_sequence_space.py`:** Al llamar a `solve_nonlinear_transition(..., shock_var="G")`, el gasto público `dG_t` entra únicamente como una perturbación exógena en la ecuación de vaciado del mercado de bienes (`Y_t = C_t + G_t`), sin incrementar los impuestos pagados por los hogares (`T_t`) ni emitir deuda pública (`B_t`). Por ello representa un experimento no financiado de demanda agregada y arroja multiplicadores inflados frente a un cierre fiscal explícito. Para transiciones fiscales financiadas con auditoría presupuestaria y de activos, utilice `puremacro.models.hank_fiscal`.

## Cierres fiscales financiados (`puremacro.models.hank_fiscal`)

El módulo `puremacro.models.hank_fiscal` incorpora cierres fiscales explícitos y auditables sobre el bloque de hogares en espacio de secuencias sin alterar `hank_sequence_space.py`.

```python
from puremacro.models.hank_fiscal import compare_fiscal_closures, solve_hank_fiscal
import numpy as np

dG = 0.01 * (0.7 ** np.arange(120))
res = solve_hank_fiscal(
    shock_seq=dG,
    closure="debt_rule",      # "balanced_budget" o "debt_rule"
    tax_type="proportional",  # "proportional" o "lump_sum"
    horizon=120,
    phi_B=0.08,
    phi_G=0.0,
)
print(res.goods_market_residual, res.gov_budget_residual, res.asset_market_residual)
```

### Reglas de cierre e incidencia tributaria

- `closure="balanced_budget"`: Mantiene $B_t = B_{\text{ss}}$ con recaudación período a período $T_t = G_t + r_t B_{\text{ss}}$.
- `closure="debt_rule"`: Evoluciona la deuda mediante $B_t = (1 + r_t) B_{t-1} + G_t - T_t$ y la regla fiscal lineal $T_t = T_{\text{ss}} + \phi_B (B_{t-1} - B_{\text{ss}}) + \phi_G dG_t$. En HANK falla la equivalencia ricardiana: posponer impuestos mediante déficit eleva el consumo de hogares con restricción de liquidez frente al presupuesto equilibrado.
- `tax_type="proportional"` frente a `"lump_sum"`: El impuesto proporcional ajusta la tasa $\tau_t = T_t / Y_t$ sobre el ingreso laboral neto $(1 - \tau_t) w_t s$; el impuesto de suma fija distribuye uniformemente $\Delta T_t = T_t - T_{\text{ss}}$ entre todos los hogares.

### Estabilidad, ley de Walras y diagnósticos

- **Región estacionaria de deuda:** Exige $r_{\text{ss}} < \phi_B < 2 + r_{\text{ss}}$. Reglas con $\phi_B \le r_{\text{ss}}$ (explosivas / sin respaldo fiscal) lanzan `ValueError` salvo `allow_unstable_rule=True`, caso en el que se emite `RuntimeWarning` y el resultado queda marcado como no convergente (`converged=False`, `SupportStatus(is_supported=False, regime="non_stationary_explosive")`).
- **Discretización del límite superior de activos:** La identidad exacta de Walras $Y_t - C_t - G_t = (A_t - B_t) - (1+r_t)(A_{t-1} - B_{t-1})$ solo se cumple si las decisiones de ahorro no quedan censuradas en $a_{\max}$. Por defecto se usa `n_a=100, a_max=100.0`; mallas truncadas (como `a_max=30.0`) quedan señaladas como `regime="boundary_asset_truncation"`.
- **Objeto devuelto (`FiscalHANKResult`):** Dataclass inmutable con sendas (`Y`, `C`, `G`, `T`, `B`, `A`, `r`, `w`, `tau`, `pi`), solución linealizada (`linear_dY`, `linear_dC`), multiplicadores fiscales (`impact_multiplier`, `cumulative_multiplier`, `present_value_multiplier`), residuos (`goods_market_residual`, `gov_budget_residual`, `gov_intertemporal_residual`, `asset_market_residual`, `terminal_debt_gap`), `solver_diagnostics` y utilidades (`summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()`, `plot()`, `compare_fiscal_closures()`).
