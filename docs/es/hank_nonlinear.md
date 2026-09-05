> 🇬🇧 [English](../hank_nonlinear.md) · 🇪🇸 Español

# Transiciones no lineales en modelos HANK

El módulo `puremacro.models.hank_sequence_space.solve_nonlinear_transition` implementa la simulación de equilibrio general no lineal para modelos Nuevo Keynesianos con Agentes Heterogéneos (HANK) en el espacio de secuencias, fundamentado en la metodología desarrollada por **Auclert, Bardóczy, Rognlie y Straub (2021, *Econometrica*)**.

Si bien las técnicas de linealización basadas en jacobianos en el espacio de secuencias (el algoritmo *Fake News*) proporcionan soluciones instantáneas para perturbaciones infinitesimales, el análisis de política macroeconómica debe afrontar con frecuencia **perturbaciones de gran escala (*large MIT shocks*)** —tales como transferencias fiscales de emergencia, ajustes drásticos de la tasa de interés de política monetaria o ensanchamientos severos de las primas de riesgo—. En dichos escenarios, la presencia de restricciones de endeudamiento activas, motivos de ahorro precautorio asimétricos y la no linealidad de la curva de Phillips invalidan las aproximaciones lineales locales.

`solve_nonlinear_transition` calcula la **trayectoria de transición no lineal exacta** de todos los agregados macroeconómicos y distribuciones de agentes en equilibrio general empleando el algoritmo cuasi-Newton de Broyden con actualizaciones analíticas de rango 1 de Sherman-Morrison.

---

## 1. Equilibrio general en el espacio de secuencias

### 1.1 Representación matemática no lineal

Considérese un horizonte temporal discreto de simulación $t = 0, \dots, T-1$, donde $T$ es suficientemente extenso como para garantizar la convergencia de la economía a su estado estacionario determinista.

Sea $\mathbf{Z} = (Z_0, \dots, Z_{T-1})'$ la secuencia exógena de perturbaciones (*MIT shocks*, como la senda del tipo de interés o el gasto público) y $\mathbf{U} = (U_0, \dots, U_{T-1})'$ la secuencia de variables endógenas objetivo (como las desviaciones del producto agregado $dY_t$).

El equilibrio general macroeconómico se define mediante un sistema de $T$ condiciones no lineales de vaciado de mercado:

$$\mathbf{H}(\mathbf{U}, \mathbf{Z}) = \mathbf{0}$$

En un modelo canónico HANK de un activo, el residuo de vaciado en el mercado de bienes en cada período $t$ se formula como:

$$H_t(\mathbf{U}, \mathbf{Z}) = Y_t - C_t(\mathbf{Y}, \mathbf{r}(\mathbf{Y}, \mathbf{Z})) - G_t = 0$$

donde el consumo agregado $C_t$ resulta de integrar de manera no lineal sobre la distribución heterogénea de riqueza e ingresos de los hogares:
$$C_t = \int c_t(a, s) \, dD_t(a, s)$$

El bloque microeconómico de hogares se resuelve exactamente a lo largo de la transición:
1. **Iteración hacia atrás de las reglas de decisión**: Las funciones de consumo $c_t(a, s)$ y ahorro $a_t(a, s)$ se resuelven mediante el Método de la Malla Endógena (EGM) desde el estado estacionario final en $t = T$ hasta el período inicial $t = 0$.
2. **Propagación hacia adelante de la distribución**: La distribución de riqueza $D_t(a, s)$ evoluciona temporalmente a partir de la distribución de estado estacionario $D_0 = D_{ss}$ aplicando las matrices markovianas de transición inducidas por las políticas óptimas de ahorro.

### 1.2 El algoritmo cuasi-Newton de Broyden

Resolver $\mathbf{H}(\mathbf{U}, \mathbf{Z}) = \mathbf{0}$ mediante el algoritmo clásico de Newton-Raphson requeriría recalcular el jacobiano completo de dimensión $T \times T$ en cada iteración, lo que implicaría un coste computacional prohibitivo.

Auclert et al. (2021) implementan el **método cuasi-Newton de Broyden con la fórmula de Sherman-Morrison**:

1. **Inicialización**:  
   La aproximación inicial del jacobiano inverso $B_0 \approx J_{ss}^{-1}$ se establece a partir del inverso del jacobiano lineal de estado estacionario obtenido con el algoritmo *Fake News*:
   $$J_{ss} = \left. \frac{\partial \mathbf{H}}{\partial \mathbf{U}} \right|_{ss}$$
2. **Paso cuasi-Newton**:  
   En la iteración $k$, el vector de actualización propuesto para las variables endógenas es:
   $$\Delta \mathbf{U}_k = - B_k \mathbf{H}(\mathbf{U}_k, \mathbf{Z})$$
3. **Búsqueda lineal retrógrada (*Backtracking Line Search*)**:  
   Una búsqueda de tipo Armijo asegura la reducción monótona de la norma infinita residual:
   $$\|\mathbf{H}(\mathbf{U}_k + \alpha \Delta \mathbf{U}_k, \mathbf{Z})\|_\infty < \|\mathbf{H}(\mathbf{U}_k, \mathbf{Z})\|_\infty$$
4. **Actualización de rango 1 de Sherman-Morrison**:  
   Definiendo $\Delta \mathbf{H}_k = \mathbf{H}_{k+1} - \mathbf{H}_k$, el inverso del jacobiano se actualiza analíticamente sin inversión matricial:
   $$B_{k+1} = B_k + \frac{(\Delta \mathbf{U}_k - B_k \Delta \mathbf{H}_k) (\Delta \mathbf{U}_k^T B_k)}{\Delta \mathbf{U}_k^T B_k \Delta \mathbf{H}_k}$$
5. **Criterio de parada**:  
   El procedimiento concluye cuando el error máximo de vaciado satisface $\|\mathbf{H}\|_\infty < \text{tol}$ (típicamente $10^{-6}$).

---

## 2. Comparativa: Dinámica lineal frente a no lineal

El análisis no lineal en el espacio de secuencias pone de manifiesto fenómenos económicos inaccesibles a los modelos perturbativos de primer orden:
- **Asimetría de los multiplicadores**: Un incremento sustancial de la tasa de interés genera una recesión más profunda que la expansión inducida por un recorte de idéntica magnitud, debido al estrangulamiento de liquidez de los hogares endeudados.
- **Espirales de ahorro precautorio**: En contracciones económicas severas, la incertidumbre sobre el empleo intensifica los motivos de autoseguro de los agentes, deprimiendo el consumo agregado muy por encima de las proyecciones lineales.
- **Persistencia dependiente del estado**: La velocidad de recuperación macroeconómica varía en función de la distribución inicial de la riqueza líquida.

---

## 3. Ejemplo de aplicación y código ejecutable

### Simulación de una perturbación monetaria severa

```python
import numpy as np
from puremacro.models.hank_sequence_space import (
    solve_hank_sequence_space,
    solve_nonlinear_transition,
)

# 1. Solución del estado estacionario y jacobianos del modelo HANK
modelo_ee = solve_hank_sequence_space(
    T=300,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
)

# 2. Definición de un choque monetario MIT severo (+150 puntos básicos)
T_sim = 100
choque_r = 0.015 * (0.75 ** np.arange(T_sim))

# 3. Solución de la trayectoria de transición de equilibrio general no lineal
res_trans = solve_nonlinear_transition(
    ss_model=modelo_ee,
    shock_seq=choque_r,
    shock_var="r",
    horizon=T_sim,
    max_iter=50,
    tol=1e-6,
    backtracking=True,
)

# 4. Diagnósticos de convergencia numérica
print(res_trans.summary())
print(f"Convergencia lograda : {res_trans.converged}")
print(f"Iteraciones Broyden  : {res_trans.iterations}")
print(f"Norma residual final : {np.max(np.abs(res_trans.residuals)):.2e}")

# 5. Comparación de la contracción máxima del producto
print("Caída del PIB (Lineal)   :", np.min(res_trans.irf_output_linear))
print("Caída del PIB (No lineal):", np.min(res_trans.irf_output_nonlinear))

# 6. Gráfico comparativo de las trayectorias de transición
fig = res_trans.plot()
```

### Simulación de un choque fiscal de gasto público

```python
# Choque de gasto público: expansión inicial del 2% del PIB con persistencia 0.8
choque_gasto = 0.02 * (0.8 ** np.arange(T_sim))

res_fiscal = solve_nonlinear_transition(
    ss_model=modelo_ee,
    shock_seq=choque_gasto,
    shock_var="G",
    horizon=T_sim,
)

# Análisis de los multiplicadores fiscales de impacto
mult_lineal = res_fiscal.irf_output_linear[0] / choque_gasto[0]
mult_nolineal = res_fiscal.irf_output_nonlinear[0] / choque_gasto[0]
print(f"Multiplicador de impacto (Lineal)   : {mult_lineal:.3f}")
print(f"Multiplicador de impacto (No lineal): {mult_nolineal:.3f}")
```

---

## 4. Especificación completa de la API

### `solve_nonlinear_transition`

```python
solve_nonlinear_transition(
    ss_model: SequenceSpaceHANKResult | Mapping[str, Any] | None = None,
    shock_seq: Sequence[float] | np.ndarray | None = None,
    shock_var: str = "r",
    horizon: int = 300,
    max_iter: int = 100,
    tol: float = 1e-6,
    backtracking: bool = True,
    **kwargs: Any,
) -> NonlinearHANKResult
```

#### Parámetros:
- `ss_model`: Resultado previo del estado estacionario `SequenceSpaceHANKResult` o diccionario de parámetros. Si es `None`, se calcula automáticamente.
- `shock_seq`: Vector de la perturbación exógena de dimensión $T$.
- `shock_var`: Variable perturbada: `'r'` (tasa de interés real de política) o `'G'` (gasto del gobierno).
- `horizon`: Longitud del horizonte temporal de simulación $T$ en trimestres (por defecto `300`).
- `max_iter`: Número máximo de iteraciones cuasi-Newton de Broyden (por defecto `100`).
- `tol`: Tolerancia de convergencia para $\|\mathbf{H}\|_\infty$ (por defecto `1e-6`).
- `backtracking`: Activa la búsqueda lineal retrógrada para garantizar la contracción monótona del residuo.

---

## 5. Interfaz de resultados y presentación

La clase `NonlinearHANKResult` organiza las trayectorias de equilibrio y suministra métodos de presentación académica:

- **Atributos numéricos**:
  - `U` / `nonlinear_path`: Secuencia no lineal de las desviaciones del producto de equilibrio ($dY_t$).
  - `linear_path`: Trayectoria lineal de referencia proyectada por el modelo de espacio de secuencias.
  - `residuals`: Secuencia residual de vaciado de mercado $H_t(U, Z)$ a lo largo de los $T$ períodos.
  - `iterations`: Conteo de iteraciones requeridas para converger.
  - `converged`: Indicador booleano de convergencia exitosa ($\|\mathbf{H}\|_\infty < \text{tol}$).
  - `norm_history`: Evolución de la norma máxima del residuo entre iteraciones.
  - `irf_output_linear`, `irf_output_nonlinear`: Respuestas del producto (lineal y no lineal).
  - `irf_consumption_linear`, `irf_consumption_nonlinear`: Respuestas del consumo agregado.
  - `irf_rate_linear`, `irf_rate_nonlinear`: Dinámica de la tasa de interés real.
  - `irf_inflation_linear`, `irf_inflation_nonlinear`: Dinámica de la inflación agregada.
- **Métodos disponibles**:
  - `.plot()`: Gráfico comparativo en Matplotlib con subpaneles para producto, consumo, tasa de interés e inflación.
  - `.summary()`: Informe técnico con los parámetros del choque, métricas de convergencia y comparación de picos de respuesta.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas tipográficas comparativas listas para su publicación.
