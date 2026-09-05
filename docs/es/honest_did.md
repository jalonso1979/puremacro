> 🇬🇧 [English](../honest_did.md) · 🇪🇸 Español

# Análisis de sensibilidad en diferencias en diferencias honesto

El módulo `puremacro.did.honest_did` implementa el marco econométrico de análisis de sensibilidad para estimadores de diferencias en diferencias (DiD) escalonadas y estudios de eventos desarrollado por **Rambachan y Roth (2023, *Review of Economic Studies*)**.

En la evaluación empírica de políticas públicas, el supuesto causal indispensable es la existencia de **tendencias paralelas**: en ausencia del tratamiento, los resultados promedio de las cohortes tratadas y de control habrían evolucionado a lo largo de trayectorias paralelas. La práctica empírica habitual somete este supuesto a contraste verificando si los coeficientes previos al tratamiento en regresiones de estudio de eventos son estadísticamente nulos. No obstante, Rambachan y Roth demuestran que:
1. **Las pruebas previas carecen de potencia estadística**: Con frecuencia fracasan en rechazar violaciones reales y cuantitativamente severas de las tendencias paralelas.
2. **Sesgo de pre-prueba (*pre-test bias*)**: Condicionar la validez del análisis a la superación de una prueba previa induce distorsiones inferenciales y sesgos de selección sustanciales sobre los efectos estimados.

El enfoque del «DiD Honesto» sustituye los contrastes binarios de hipótesis por un **análisis de sensibilidad riguroso**, construyendo intervalos de confianza robustos y conjuntos identificados bajo violaciones potenciales acotadas del supuesto de tendencias paralelas en el período posterior al tratamiento.

---

## 1. Marco econométrico

Sea $\hat{\beta} = (\hat{\beta}_{pre}', \hat{\beta}_{post}')'$ el vector de coeficientes estimados en un estudio de eventos con matriz de covarianzas asintóticas $\hat{\Sigma}$. Sea $\delta = (\delta_{pre}', \delta_{post}')'$ la verdadera desviación subyacente de las tendencias paralelas, de tal suerte que:

$$\hat{\beta} \sim \mathcal{N}(\theta + \delta, \Sigma)$$

donde $\theta_{pre} = 0$ por construcción analítica y $\theta_{post}$ representa el vector de efectos causales de interés.

En presencia de discrepancias en las tendencias ($\delta_{post} \neq 0$), el parámetro causal queda únicamente **identificado por conjuntos (*partially identified*)**. Rambachan y Roth restringen las desviaciones contrafactuales admisibles $\delta \in \Delta$ a través de dos familias formales:

### 1.1 Cotas de suavidad / Segundas diferencias acotadas ($\Delta^{SD}(M)$)

Impone que la pendiente de la tendencia contrafactual no puede experimentar aceleraciones o deceleraciones bruscas entre períodos contiguos. Las segundas diferencias de $\delta$ quedan restringidas por la constante $M \ge 0$:

$$\Delta^{SD}(M) = \left\{ \delta \in \mathbb{R}^T : \left| (\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1}) \right| \le M, \quad \forall t \right\}$$

- $M = 0$: Exige una tendencia diferencial estrictamente lineal. Toda pendiente preexistente en el período previo se proyecta linealmente hacia el período post-tratamiento.
- $M > 0$: Concede holgura frente a tendencias puramente lineales, permitiendo cambios de curvatura tanto más pronunciados cuanto mayor sea el parámetro $M$.

### 1.2 Cotas de magnitud relativa ($\Delta^{RM}(\bar{M})$)

Acota las posibles desviaciones post-tratamiento en función de un múltiplo $\bar{M} \ge 0$ de la desviación máxima observada en el período previo al tratamiento:

$$\Delta^{RM}(\bar{M}) = \left\{ \delta \in \mathbb{R}^T : |\delta_l| \le \bar{M} \cdot \max_{s < 0} |\delta_s|, \quad \forall l \ge 0 \right\}$$

O bien, formulado en primeras diferencias:
$$|\delta_t - \delta_{t-1}| \le \bar{M} \cdot \max_{s \le 0} |\delta_s - \delta_{s-1}|$$

- $\bar{M} = 0$: Restaura el supuesto clásico de tendencias paralelas exactas tras el tratamiento ($\delta_{post} = 0$).
- $\bar{M} = 1$: Garantiza que la violación contrafactual post-tratamiento no sobrepasa la divergencia histórica más severa observada antes de la intervención.

---

## 2. Optimización e inferencia estadística

Para cualquier combinación lineal de interés $\theta = l' \tau_{post}$ (como el efecto promedio acumulado o el impacto en un horizonte específico $h$):

1. **Cálculo del conjunto identificado**:  
   El intervalo identificado $[\theta^{lo}(M), \theta^{hi}(M)]$ se calcula mediante programación lineal convexa exacta empleando el solucionador de punto interior y símplex HiGHS (`scipy.optimize.linprog(method='highs')`).
2. **Intervalos de confianza robustos**:  
   Se derivan conforme a los métodos de **Imbens y Manski (2004)** y **Stoye (2009)** para parámetros identificados por conjuntos:
   $$CI_{1-\alpha}(M) = \left[ \hat{\theta}^{lo} - c_\alpha \cdot \text{ee}(\hat{\theta}^{lo}), \; \hat{\theta}^{hi} + c_\alpha \cdot \text{ee}(\hat{\theta}^{hi}) \right]$$
   donde el valor crítico $c_\alpha$ satisface $\Phi(c_\alpha + \frac{\hat{\theta}^{hi} - \hat{\theta}^{lo}}{\max(\text{ee})}) - \Phi(-c_\alpha) = 1 - \alpha$.
3. **Valor de ruptura $M^*$ (*Breakdown Value*)**:  
   El umbral de ruptura $M^*$ representa la menor magnitud de violación para la cual el intervalo de confianza robusto al nivel $(1-\alpha)$ intersecta el valor nulo:
   $$M^* = \inf \{ M \ge 0 : 0 \in CI_{1-\alpha}(M) \}$$
   Se determina con precisión numérica mediante el algoritmo de Brent (`scipy.optimize.brentq`). Si el estimador convencional ya resulta no significativo en $M=0$, se reporta $M^* = 0.0$.

---

## 3. Replicación: Reducción del IVA en restaurantes franceses (Benzarti y Carloni 2019)

En julio de 2009, Francia aplicó una reducción tributaria del impuesto sobre el valor añadido (IVA) en el sector de la restauración del 19.6% al 5.5%. Benzarti y Carloni (2019) examinaron si la incidencia económica del recorte tributario benefició a las empresas mediante un ensanchamiento de sus márgenes de beneficio.

Rambachan y Roth (2023, Sección 6.1) analizan la robustez de la respuesta de los beneficios empresariales en 2009 tomando como base el período 2004–2007 (con año de referencia omitido en 2008):
- **Estimación basal por MCO (2009)**: Efecto estimado $\hat{\beta}_{2009} = 0.1960$ ($\text{ee} = 0.0190$, estadístico $t > 10$).
- **Desviación máxima previa al tratamiento**: $\max_{s \le 2007} |\hat{\beta}_s| = 0.0730$.
- **Sensibilidad bajo magnitud relativa ($\Delta^{RM}$)**: El valor de ruptura asciende a $M^* \approx 2.06$. El impacto positivo de la reforma sobre los beneficios empresariales preserva su significancia estadística incluso ante desviaciones contrafactuales que dupliquen (**más de 2.0 veces**) la peor divergencia histórica observada.

### Código de replicación ejecutable

```python
import numpy as np
from puremacro.did import honest_did

# 1. Coeficientes de estudio de eventos y matriz de covarianzas de Benzarti y Carloni (2019)
anios = [2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012]
anio_ref = 2008

beta_hat = np.array([
    0.006696, 0.029345, -0.006473, 0.073015,
    0.195961, 0.312064,  0.239542, 0.126043,
])

sigma_hat = np.array([
    [0.000843, 0.000477, 0.000262, 0.000235, 0.000168, 0.000113, 0.000020, -0.000137],
    [0.000477, 0.000643, 0.000399, 0.000244, 0.000220, 0.000180, 0.000038, -0.000030],
    [0.000262, 0.000399, 0.000523, 0.000212, 0.000184, 0.000146, 0.000070,  0.000060],
    [0.000235, 0.000244, 0.000212, 0.000309, 0.000120, 0.000133, 0.000102,  0.000108],
    [0.000168, 0.000220, 0.000184, 0.000120, 0.000361, 0.000295, 0.000163,  0.000085],
    [0.000113, 0.000180, 0.000146, 0.000133, 0.000295, 0.000472, 0.000248,  0.000142],
    [0.000020, 0.000038, 0.000070, 0.000102, 0.000163, 0.000248, 0.000412,  0.000221],
    [-0.000137,-0.000030, 0.000060, 0.000108, 0.000085, 0.000142, 0.000221,  0.000489],
])

# 2. Análisis de sensibilidad bajo magnitud relativa en el año inmediato (2009)
res_rm = honest_did(
    b_hat=beta_hat,
    sigma=sigma_hat,
    event_time=anios,
    base_period=anio_ref,
    target_horizon=2009,
    method="relative_magnitude",
    m_vec=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    alpha=0.05,
)

# 3. Presentación del informe y lectura del valor de ruptura M*
print(res_rm.summary())
print(f"Valor de ruptura M*: {res_rm.breakdown_value:.4f}")

# 4. Generación gráfica de la curva de sensibilidad
fig = res_rm.plot()

# 5. Exportación formal de resultados
tabla_df = res_rm.to_frame()
codigo_latex = res_rm.to_latex()
```

---

## 4. Especificación completa de la API

### `honest_did`

```python
honest_did(
    b_hat: Any = None,
    sigma: np.ndarray | Sequence[Sequence[float]] | None = None,
    se: Sequence[float] | None = None,
    method: str = "smoothness",
    m_vec: Sequence[float] | None = None,
    base_period: int = -1,
    alpha: float = 0.05,
    l_vec: Sequence[float] | np.ndarray | None = None,
    pre_periods: int | Sequence[int | float] | None = None,
    post_periods: int | Sequence[int | float] | None = None,
    *,
    result: Any = None,
    event_time: Sequence[int | float] | None = None,
    target_horizon: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> HonestDiDResult
```

#### Parámetros:
- `b_hat` / `result`: Vector de coeficientes del estudio de eventos, o un objeto de resultados procedente de `puremacro.did.callaway_santanna` o `puremacro.did.sun_abraham`.
- `sigma`: Matriz asintótica de varianzas y covarianzas $(T, T)$ de los coeficientes de eventos.
- `se`: Vector de errores estándar individuales (empleado si no se suministra la matriz completa).
- `method`: Familia de restricciones de sensibilidad:
  - `'smoothness'`: Segundas diferencias acotadas ($\Delta^{SD}(M)$).
  - `'relative_magnitude'`: Desviaciones proporcionales a las tendencias previas ($\Delta^{RM}(\bar{M})$).
- `m_vec`: Malla de valores de sensibilidad $M \ge 0$ evaluados.
- `base_period`: Período de referencia omitido normalizado a cero (por defecto `-1`).
- `alpha`: Nivel de significancia nominal (por defecto `0.05` para un intervalo del 95%).
- `target_horizon`: Horizonte temporal específico sobre el cual se evalúa la sensibilidad.
- `l_vec`: Vector de contrastes $l$ que define el parámetro lineal agregado $\theta = l' \tau_{post}$.

---

## 5. Interfaz de resultados y presentación

El contenedor `HonestDiDResult` dispone de métodos integrados para la comunicación académica de resultados:

- `.table` / `.to_frame()`: `pd.DataFrame` estructurado con la curva de sensibilidad:
  - `M`: Magnitud de la violación evaluada.
  - `id_lo`, `id_hi`: Límites inferior y superior del conjunto identificado.
  - `ci_lo`, `ci_hi`: Extremos del intervalo de confianza robusto de Imbens-Manski.
  - `significant`: Indicador booleano de significancia estadística ($0 \notin [ci_{lo}, ci_{hi}]$).
- `.breakdown_value`: Umbral de ruptura exacto $M^*$.
- `.plot()`: Gráfico en Matplotlib que traza el conjunto identificado y las bandas de confianza robustas en función de $M$, señalizando con una línea discontinua el punto de ruptura $M^*$.
- `.plot_ascii()`: Visualización esquemática en caracteres ASCII para terminales e interactividad rápida.
- `.summary()`: Informe técnico con métricas basales previas al tratamiento y valores de ruptura desglosados por horizonte.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas tipográficas listas para su inclusión en manuscritos.
