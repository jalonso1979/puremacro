> 🇬🇧 [English](../narrative_sign_svar.md) · 🇪🇸 Español

# Restricciones narrativas de signo en modelos SVAR

El módulo `puremacro.var.identify.narrative_sign` implementa la metodología de identificación estructural para vectores autorregresivos (SVAR) desarrollada por **Antolín-Díaz y Rubio-Ramírez (2018, *American Economic Review*)**, junto con las extensiones de cotas de magnitud de **Ludvigson, Ma y Ng (2021, *Journal of Political Economy*)**.

En los modelos SVAR identificados mediante restricciones de signo convencionales (Faust 1998; Uhlig 2005; Rubio-Ramírez, Waggoner y Zha 2010), las regiones identificadas por conjuntos (*set identification*) suelen ser excesivamente amplias, arrojando intervalos de credibilidad que abarcan respuestas cualitativamente contradictorias. Las restricciones narrativas resuelven esta limitación al condicionar las matrices de rotación estructural sobre evidencia histórica concreta —tales como anuncios documentados de la banca central, perturbaciones geopolíticas en el mercado petrolero o episodios específicos de crisis financiera—, descartando aquellas extracciones ortogonales que contradicen el registro histórico establecido.

---

## 1. Fundamentos metodológicos

### 1.1 La forma reducida del VAR y las rotaciones estructurales

Considérese un modelo $\text{VAR}(p)$ en forma reducida con $n$ variables endógenas:

$$y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma)$$

donde $\Sigma$ es la matriz simétrica y definida positiva de covarianzas residuales. Sea $P = \text{chol}(\Sigma)$ el factor triangular inferior de Cholesky tal que $P P' = \Sigma$.

Las innovaciones de forma reducida $u_t$ se vinculan con las perturbaciones estructurales ortonormales $\varepsilon_t \sim \mathcal{N}(0, I_n)$ mediante:

$$u_t = B \varepsilon_t = P Q \varepsilon_t$$

donde $Q \in \mathcal{O}(n)$ es una matriz ortogonal de rotación que satisface $Q Q' = Q' Q = I_n$. Las respuestas al impulso estructurales en el horizonte $h$ se obtienen como:

$$\Psi_h = \Phi_h B = \Phi_h P Q$$

siendo $\Phi_h$ la matriz de coeficientes de media móvil de la forma reducida en el horizonte $h$ ($\Phi_0 = I_n$).

### 1.2 Tipología de restricciones narrativas

Antolín-Díaz y Rubio-Ramírez (2018) formalizan las restricciones narrativas sobre las realizaciones de las perturbaciones estructurales $\varepsilon_t = B^{-1} u_t = Q' P^{-1} u_t$ y su descomposición histórica en cuatro categorías fundamentales:

1. **Tipo I: Restricción sobre el signo de la perturbación (`'shock_sign'`)**  
   Fija el signo de la perturbación estructural $j$ en una fecha histórica precisa $t^*$:
   $$\text{sign}(\varepsilon_{j, t^*}) = s, \quad s \in \{-1, +1\}$$
   *Ejemplo*: «La perturbación de política monetaria en el cuarto trimestre de 1979 (endurecimiento de Paul Volcker) fue estrictamente positiva (contractiva)».

2. **Tipo II: Dominancia en la descomposición histórica (`'hd_dominance'`, `dominance='most'`)**  
   Exige que la perturbación estructural $j$ sea la principal contribuyente individual al cambio inesperado acumulado en la variable $i$ a lo largo de una ventana de $L$ períodos finalizada en $t_1^*$:
   $$|H_{i, j}(t_1^*, L)| \ge \max_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   donde la contribución histórica de la perturbación $k$ se define como:
   $$H_{i, k}(t_1^*, L) = \sum_{l=0}^L (\Phi_l B)_{i, k} \, \varepsilon_{k, t_1^* - l}$$

3. **Tipo III: Dominancia abrumadora en la descomposición histórica (`'hd_dominance'`, `dominance='overwhelming'`)**  
   Exige que la contribución absoluta de la perturbación $j$ supere la suma de las contribuciones de todas las demás perturbaciones estructurales combinadas:
   $$|H_{i, j}(t_1^*, L)| \ge \sum_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   *Ejemplo*: «La perturbación monetaria fue la causa abrumadora del incremento imprevisto de la tasa de fondos federales en 1979Q4».

4. **Tipo IV: Cotas de magnitud de la perturbación (`'shock_bound'`)**  
   Impone desigualdades sobre la magnitud absoluta de la perturbación estructural conforme a Ludvigson, Ma y Ng (2021):
   $$\underline{m} \le |\varepsilon_{j, t^*}| \le \bar{m}$$
   *Ejemplo*: «La perturbación monetaria durante el cambio de régimen de Volcker tuvo una magnitud de al menos 2 desviaciones estándar ($|\varepsilon_{j, t^*}| \ge 2.0$)».

### 1.3 Muestreo y ponderación por importancia (Algoritmo 1 de AD-RR)

1. **Generación de rotaciones de Haar**: Se muestrean matrices aleatorias gaussianas $Z \sim \mathcal{N}(0, I_n)$ y se calcula $Q$ mediante descomposición QR con normalización de signos positivos en la diagonal ($R_{ii} > 0$).
2. **Filtrado por restricciones de signo tradicionales**: Se verifica si las respuestas al impulso $\Psi_h = \Phi_h P Q$ satisfacen las restricciones de signo especificadas en `sign_matrix`.
3. **Validación narrativa**: Se calculan las perturbaciones estructurales históricas $\varepsilon = u (B^{-1})'$ y se comprueba el cumplimiento simultáneo de todas las restricciones narrativas en sus fechas correspondientes.
4. **Ponderación por importancia**: A fin de corregir el sesgo que favorecería a parámetros bajo los cuales los eventos narrativos fuesen fortuitamente probables, cada extracción superviviente se pondera por:
   $$w = \frac{1}{\omega}$$
   donde $\omega$ representa la probabilidad de que las restricciones narrativas se verifiquen cuando las perturbaciones en las fechas restringidas se extraen de forma independiente e idénticamente distribuida $\mathcal{N}(0, I_n)$:
   - Para restricciones puras de Tipo I (`'shock_sign'`) sobre $m$ pares distintos (fecha, perturbación), $\omega = 0.5^m$ se conoce de forma analítica cerrada ($w = 2^m$).
   - Para restricciones de Tipo II, III o IV, $\omega$ se estima mediante simulación de Monte Carlo con $S = \text{n\_weight\_sims}$ extracciones.
5. **Inferencia ponderada**: Las bandas de credibilidad y la mediana se obtienen mediante cuantiles ponderados puntuales sobre las extracciones aceptadas.

---

## 2. Diagnósticos de aceptación y métricas

El objeto `NarrativeSignResult` generado proporciona métricas diagnósticas exhaustivas:

| Atributo | Tipo | Significado analítico |
|---|---|---|
| `n_draws` | `int` | Total de rotaciones ortogonales de Haar evaluadas. |
| `n_traditional_accepted` | `int` | Número de extracciones que satisfacen las restricciones de signo tradicionales. |
| `n_narrative_accepted` | `int` | Extracciones que satisfacen tanto las restricciones tradicionales como las narrativas. |
| `acceptance_rate` | `float` | Tasa de aceptación global (`n_narrative_accepted / n_draws`). |
| `traditional_acceptance_rate`| `float` | Fracción de extracciones válidas tradicionales (`n_traditional_accepted / n_draws`). |
| `narrative_acceptance_rate`  | `float` | Tasa condicional de aceptación narrativa (`n_narrative_accepted / n_traditional_accepted`). |
| `weights` | `np.ndarray` | Ponderadores de importancia en bruto $1/\hat{\omega}$ para las extracciones supervivientes. |
| `ess` / `effective_draws` | `float` | Tamaño muestral efectivo de Kish: $(\sum w_i)^2 / \sum w_i^2$. Alerta sobre concentración excesiva en pocas extracciones. |
| `restriction_labels` | `tuple[str]` | Etiquetas legibles que identifican cada restricción impuesta. |
| `restriction_fail_counts` | `tuple[int]` | Diagnóstico de rigidez: extracciones tradicionales rechazadas por cada restricción individual. |

---

## 3. Replicación: Choque de política monetaria de Volcker (1979Q4)

En octubre de 1979, la Reserva Federal presidida por Paul Volcker adoptó un nuevo esquema operativo centrado en objetivos cuantitativos de reservas no prestadas, provocando un aumento histórico sin precedentes en la tasa de fondos federales con el propósito de erradicar la persistente inflación estadounidense.

En un modelo VAR de 3 variables compuesto por la Tasa de Fondos Federales (TFF), la Inflación y el Crecimiento del PIB, Antolín-Díaz y Rubio-Ramírez (2018) imponen:
- **Signos tradicionales**: +TFF, -Inflación, -Crecimiento del PIB en el horizonte contemporáneo $h=0$.
- **Restricción narrativa 1 (Tipo I)**: La perturbación monetaria en 1979Q4 fue contractiva ($\varepsilon_{MP, 1979Q4} > 0$).
- **Restricción narrativa 2 (Tipo III)**: La perturbación monetaria fue la contribuyente abrumadora al repunte inesperado de la tasa de fondos federales en 1979Q4.

El condicionamiento narrativo logra reducir la dispersión de las bandas de credibilidad al 68% en **más de un 20%** para todas las variables en comparación con las restricciones de signo tradicionales aisladas, descartando trayectorias expansivas anómalas del producto.

### Código de replicación ejecutable

```python
import numpy as np
import pandas as pd
from puremacro.var.identify import (
    NarrativeRestriction,
    identify_narrative_sign,
)

# 1. Preparación de datos macroeconómicos trimestrales (TFF, Inflación, PIB)
rng = np.random.default_rng(42)
T = 172
fechas = pd.date_range("1965-01-01", periods=T, freq="QE")
volcker_idx = fechas.get_loc("1979-10-01")

# Simulación de un VAR estacionario calibrado
Y = np.zeros((T, 3))
for t in range(1, T):
    Y[t] = 0.5 * Y[t-1] + rng.standard_normal(3)
# Inyección del episodio de ajuste de Volcker en 1979Q4
Y[volcker_idx, 0] += 3.5

# 2. Restricciones de signo tradicionales en h = 0
# Choque 0 = Política monetaria contractiva (+TFF, -Inflación, -PIB)
sign_matrix = {0: np.array([+1, -1, -1])}

# 3. Definición de restricciones narrativas
restricciones = [
    # Tipo I: Perturbación monetaria positiva en 1979Q4
    (volcker_idx, 0, +1),
    # Tipo III: Choque monetario como causa abrumadora del repunte de la TFF
    NarrativeRestriction(
        kind="hd_dominance",
        date=volcker_idx,
        shock=0,
        variable=0,  # TFF
        window=0,
        dominance="overwhelming",
    ),
]

# 4. Estimación del SVAR con restricciones narrativas
res_narr = identify_narrative_sign(
    Y=Y,
    p=2,
    horizon=16,
    sign_matrix=sign_matrix,
    restrictions=restricciones,
    n_draws=3000,
    n_weight_sims=300,
    ci=0.68,
    seed=123,
)

# 5. Diagnósticos de convergencia e informe resumen
print(res_narr.summary())
print(f"Aceptación tradicional: {res_narr.traditional_acceptance_rate:.2%}")
print(f"Aceptación narrativa  : {res_narr.narrative_acceptance_rate:.2%}")
print(f"Tamaño efectivo (ESS) : {res_narr.effective_draws:.1f}")

# 6. Visualización y exportación académica
fig = res_narr.plot(shock_idx=0)
tabla_md = res_narr.to_markdown()
tabla_tex = res_narr.to_latex()
```

---

## 4. Especificación completa de la API

### `identify_narrative_sign`

```python
identify_narrative_sign(
    Y: np.ndarray | pd.DataFrame | VarEstimateResult,
    restrictions: list | None = None,
    *,
    p: int | None = None,
    horizon: int | None = 20,
    sign_matrix: dict | np.ndarray | None = None,
    dates: Sequence[Any] | None = None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> NarrativeSignResult
```

#### Parámetros:
- `Y`: Matriz temporal $(T, n)$, `DataFrame` o resultado previo `VarEstimateResult`.
- `restrictions`: Lista de restricciones narrativas. Admite instancias de `NarrativeRestriction`, tuplas breves `(fecha, indice_choque, signo)` o eventos `puremacro.narrative.NarrativeEvent`.
- `p`: Orden autorregresivo del VAR (inferido automáticamente si `Y` es un `VarEstimateResult`).
- `horizon`: Horizonte de proyección $H$ para las respuestas al impulso (por defecto `20`).
- `sign_matrix`: Diccionario `{h: S}` con matrices de signo tradicionales $S \in \{-1, 0, 1\}$.
- `dates`: Índices o etiquetas temporales de longitud $T$ para vincular fechas con filas del panel.
- `bayes_draws`: Si es `True`, muestrea coeficientes $(A, \Sigma)$ de la distribución a posteriori Normal-Inversa-Wishart.
- `n_draws`: Cantidad total de rotaciones de Haar candidatas (por defecto `2000`).
- `n_weight_sims`: Muestras de Monte Carlo empleadas en el cómputo de $\omega$ para restricciones de Tipo II, III o IV (por defecto `500`).
- `ci`: Nivel de cobertura para las bandas de credibilidad (por defecto `0.90`).
- `seed`: Semilla generadora para reproducibilidad exacta.

---

## 5. Interfaz de resultados y capacidades analíticas

La clase `NarrativeSignResult` implementa las herramientas del protocolo de presentación de `puremacro`:

- `.plot(shock_idx=0, target_idx=None)`: Gráfico con la mediana de respuesta y las bandas de credibilidad sombreadas.
- `.summary()`: Informe estructurado que desglosa las tasas de aceptación, el tamaño muestral efectivo y las frecuencias de rechazo de cada restricción.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas tipográficas listas para su inserción en artículos de investigación.
- `.fevd(horizon=20)`: Descomposición de la varianza del error de pronóstico atribuible a cada perturbación estructural.
- `.historical_decomposition(variable=0, shock=0)`: Serie temporal de la descomposición histórica que ilustra el impacto de la perturbación sobre la variable seleccionada.
