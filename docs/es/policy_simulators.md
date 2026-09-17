> 🇬🇧 [English](../policy_simulators.md) · 🇪🇸 Español

# Simuladores de Política Macroeconómica y Laboratorios Interactivos para Navegador

`puremacro` ofrece una arquitectura dual e integrada para la simulación cuantitativa de contrafactuales de política macroeconómica:

1. **API de Producción en Python (`puremacro.models`)**:
   - **`TradePolicySimulator`**: Modelo de equilibrio general de comercio cuantitativo basado en **Caliendo y Parro (2015, *Review of Economic Studies*)**, que resuelve contrafactuales arancelarios bilaterales, términos de intercambio, encadenamientos insumo-producto y descomposición de bienestar con convergencia rigurosa en el vaciado de mercados ($\max_i |X_i - Y_i| < 10^{-6}$).
   - **`MonetaryTransmissionSimulator`**: Comparación lado a lado entre modelos **Neokeynesianos de Agentes Heterogéneos (HANK)** y de **Agente Representativo (RANK)**, implementando la descomposición de canales directo e indirecto de **Kaplan, Moll y Violante (2018, *American Economic Review*)** a lo largo de los deciles de propensión marginal a consumir (PMC).
2. **Laboratorios Interactivos en WebAssembly / Navegador (`curso/site/labs/`)**:
   - **`comercio-aranceles.html` / `.js`**: Simulador en tiempo real de guerra arancelaria (MEX-EE. UU.-CHN) con controles deslizantes bilaterales, solucionador de equilibrio general en JavaScript y visualizadores de desvío de comercio.
   - **`politica-monetaria-hank.html` / `.js`**: Laboratorio interactivo de transmisión monetaria HANK vs. RANK con distribución de PMC por deciles y descomposición de canales KMV.
   - Diseñados para funcionamiento fuera de línea, sin proceso de compilación (*zero-build*) y con compatibilidad total con Pyodide e iPad.

Ambas interfaces utilizan clases de resultados inmutables (*frozen dataclasses*) y ofrecen una suite integral de exportación (`.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`).

---

## 1. Simulador de Política Comercial Cuantitativa (`TradePolicySimulator`)

### 1.1 Marco Teórico: Modelo de Equilibrio General de Caliendo y Parro (2015)

Considere una economía global con $N$ países ($n, i \in \{1, \dots, N\}$) y $J$ sectores ($j, k \in \{1, \dots, J\}$). La producción combina mano de obra $L_{n, j}$ con insumos intermedios de todos los sectores bajo rendimientos constantes a escala.

Empleando el método de **Álgebra Exacta de Sombreros (*Exact Hat Algebra*)** (Dekle, Eaton y Kortum 2007; Caliendo y Parro 2015), los cambios contrafactuales en precios, salarios y gasto se expresan en variaciones proporcionales $\hat{x} \equiv x' / x$, prescindiendo de la necesidad de estimar niveles no observables de productividad o costos comerciales de transporte tipo *iceberg*.

Ante variaciones contrafactuales en los aranceles brutos bilaterales $\hat{\tau}_{ni}^j = \frac{1 + t_{ni}'^j}{1 + t_{ni}^j}$:

1. **Variación de Costos Unitarios**:
   $$\hat{c}_n^j = \hat{w}_n^{\gamma_n^j} \prod_{k=1}^J \left( \hat{P}_n^k \right)^{\gamma_n^{j, k}}$$
   donde $\gamma_n^j > 0$ es la participación del valor agregado y $\gamma_n^{j, k} \ge 0$ son las participaciones de insumos intermedios ($\gamma_n^j + \sum_k \gamma_n^{j, k} = 1$).
2. **Cuotas de Comercio Bilateral (Gravedad de Eaton-Kortum)**:
   $$\hat{\pi}_{ni}^j = \left( \frac{\hat{c}_i^j \hat{\tau}_{ni}^j}{\hat{P}_n^j} \right)^{-\theta_j}$$
   donde $\theta_j > 0$ es la elasticidad comercial sectorial.
3. **Índices de Precios Sectoriales**:
   $$\hat{P}_n^j = \left( \sum_{i=1}^N \pi_{ni}^j \left( \hat{c}_i^j \hat{\tau}_{ni}^j \right)^{-\theta_j} \right)^{-\frac{1}{\theta_j}}$$
4. **Vaciado de Mercados de Bienes y Factores**:
   $$\max_n \left| \sum_{j=1}^J \gamma_n^j Y_n'^j - w_n' L_n \right| < 10^{-6}$$
   Los salarios de equilibrio $\hat{w}_n$ se determinan mediante un algoritmo de tanteo (*tatonnement*) hasta alcanzar la tolerancia exigida.

### 1.2 Descomposición del Bienestar e Ingreso Real

La variación del bienestar e ingreso real $\widehat{\mathcal{W}}_n = \frac{\hat{I}_n}{\hat{P}_n}$ se descompone en tres mecanismos:
1. **Efecto Términos de Intercambio**: Variación en los precios de exportación frente a los de importación.
2. **Eficiencia Insumo-Producto**: Ganancias de eficiencia por reducción de costos intermedios en cadenas globales de valor.
3. **Efecto de Ingresos Arancelarios**: Recaudación arancelaria neta transferida a los hogares.

### 1.3 Ejemplo de Uso en Python

```python
import numpy as np
from puremacro.models import TradePolicySimulator

# 1. Instanciar el Simulador de Política Comercial usando el preajuste calibrado TLCAN-China (3 países)
simulador = TradePolicySimulator.from_preset("nafta_china")

# 2. Simular un aumento arancelario unilateral del 20% de EE. UU. sobre manufacturas mexicanas
contrafactual = simulador.simulate_tariff_counterfactual(
    tariff_shocks={("USA", "MEX", "Manufactures"): 0.20}
)

# Verificar convergencia de equilibrio general
print(f"¿Convergió? : {contrafactual.converged} (Iteraciones: {contrafactual.iterations})")
print(f"Error residual de vaciado : {contrafactual.market_clearing_residual:.2e} (< 1e-6)")

# Resumen de resultados por país
print(contrafactual.summary())

# Exportar tabla a LaTeX con formato de revista académica
print(contrafactual.to_latex())
```

---

## 2. Simulador de Transmisión Monetaria y Macroprudencial (`MonetaryTransmissionSimulator`)

### 2.1 Mecanismos de Transmisión: HANK frente a RANK

Los modelos Neokeynesianos de Agente Representativo (RANK) postulan que la política monetaria incide en el consumo agregado predominantemente a través del **canal directo de sustitución intertemporal**: tasas de interés más altas incentivan a los hogares a diferir consumo a lo largo de una única ecuación de Euler.

Por el contrario, los modelos de Agentes Heterogéneos (HANK) (Kaplan, Moll y Violante 2018; Auclert et al. 2021) incorporan riesgo idiosincrásico no asegurable y restricciones de endeudamiento. Una proporción sustancial de los hogares vive al día (*hand-to-mouth*), presentando elevadas Propensiones Marginales a Consumir (PMC). Como consecuencia, el **canal indirecto de equilibrio general**—que opera mediante la demanda de trabajo y los ingresos laborales—domina la respuesta agregada.

### 2.2 Descomposición de Kaplan-Moll-Violante (2018) en el Espacio de Secuencias

A través de la matriz jacobiana en el espacio de secuencias, la respuesta del consumo agregado $d\mathbf{C} \in \mathbb{R}^T$ en un horizonte $T$ se descompone en:

$$d\mathbf{C} = \underbrace{\mathbf{J}^{C, r} \, d\mathbf{r}}_{\text{Canal Directo (Sustitución)}} + \underbrace{\mathbf{J}^{C, Y} \, d\mathbf{Y}}_{\text{Canal Indirecto (Ingreso de Equilibrio General)}}$$

| Dimensión | Economía RANK | Economía HANK |
|---|---|---|
| **Estructura de Mercados** | Mercados completos de activos | Mercados incompletos, restricciones crediticias |
| **Distribución de la PMC** | Uniforme: $1 - \beta \approx 0.015$ | Heterogénea entre deciles: $0.02$ a $0.45+$ |
| **Participación Canal Directo** | $\approx 100\%$ | $20\% - 40\%$ |
| **Participación Canal Indirecto** | $\approx 0\%$ | $60\% - 80\%$ |
| **Motor de Transmisión** | Ecuación de Euler intertemporal | Retroalimentación de ingresos laborales |

### 2.3 Ejemplo de Uso en Python

```python
from puremacro.models import MonetaryTransmissionSimulator

# 1. Instanciar el Simulador de Transmisión Monetaria
sim_mon = MonetaryTransmissionSimulator(
    n_a=50,
    beta=0.985,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
)

# 2. Simular un alza contractiva de 25 pb en la tasa de interés con persistencia rho=0.7 y horizonte T=40
res = sim_mon.simulate_transmission(
    shock_type="rate",
    magnitude=0.0025,
    rho=0.7,
    T=40,
)

print(res.summary())
print(f"PMC agregada HANK : {res.aggregate_mpc_hank:.4f}")
print(f"PMC agregada RANK : {res.aggregate_mpc_rank:.4f}")
print(f"Participación canal indirecto HANK : {res.indirect_share_hank:.1f}%")
print(f"Participación canal indirecto RANK : {res.indirect_share_rank:.1f}%")

# Detalle de la PMC por deciles de riqueza
print(res.mpc_deciles_hank)
```

---

## 3. Laboratorios Interactivos para Navegador (`curso/site/labs/`)

Junto con la API en Python, `puremacro` distribuye dos laboratorios interactivos implementados en HTML5, CSS y Canvas nativos con JavaScript puro, sin dependencias de Node/npm y con funcionamiento fuera de línea.

### 3.1 `comercio-aranceles.html` y `.js` (Simulador de Guerra Arancelaria)

- **Ubicación**: `curso/site/labs/comercio-aranceles.html` y `curso/site/labs/comercio-aranceles.js`.
- **Prestaciones**:
  * Simula una guerra comercial trilateral entre México (MEX), Estados Unidos (USA) y China (CHN).
  * Deslizadores interactivos para aranceles bilaterales.
  * Solución instantánea de equilibrio general en el navegador.
  * Visualización en tiempo real de salarios reales, términos de intercambio y desvío de flujos comerciales.

### 3.2 `politica-monetaria-hank.html` y `.js` (Laboratorio HANK vs. RANK)

- **Ubicación**: `curso/site/labs/politica-monetaria-hank.html` y `curso/site/labs/politica-monetaria-hank.js`.
- **Prestaciones**:
  * Curvas de respuesta al impulso comparadas para producto, inflación y consumo.
  * Controles interactivos de magnitud del choque, persistencia y fracción de hogares de subsistencia.
  * Escalera dinámica de propensiones marginales a consumir por deciles de riqueza.
  * Gráfico dinámico de las contribuciones de los canales directo e indirecto de Kaplan-Moll-Violante.

Ambos laboratorios pueden visualizarse localmente con `python -m http.server` o integrarse en cuadernos pedagógicos interactivos con Pyodide.

---

## Referencias

1. Auclert, A., Bardóczy, B., Rognlie, M. y Straub, L. (2021). "Using the sequence-space Jacobian to solve and estimate heterogeneous-agent models." *Econometrica*, 89(6), 3115–3148.
2. Caliendo, L. y Parro, F. (2015). "Estimates of the trade and welfare effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
3. Dekle, R., Eaton, J. y Kortum, S. (2007). "Unbalanced trade." *American Economic Review*, 97(2), 351–355.
4. Kaplan, G., Moll, B. y Violante, G. L. (2018). "Monetary policy according to HANK." *American Economic Review*, 108(3), 697–743.
