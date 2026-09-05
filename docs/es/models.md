> 🇬🇧 [English](../models.md) · 🇪🇸 Español

# HANK en el espacio de secuencias

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
| `dmp_regime_dependent` | `puremacro.models` | Modelo DMP con empresa representativa, salarios rígidos de Hall, coste de vacantes dependiente del régimen y choque de PTF |
| `nested_dmp` (paquete) | `puremacro.models.nested_dmp` | DMP con empresas heterogéneas y creencias bayesianas sobre la regla de política: estado estacionario, FRI con previsión perfecta, estimación por coincidencia de FRI, solución estocástica recursiva y análisis de bienestar |
| `smm` | `puremacro.models.smm` | Cargador de momentos y función objetivo para estimar `dmp_regime_dependent` frente a FRI de proyecciones locales |

`puremacro.models.__all__` reexporta:
`DMPParameters`, `DMPState`, `dmp_steady_state`, `dmp_irf`,
`SequenceSpaceHANKResult`, `solve_hank_sequence_space`,
`FakeNewsResult`, `FiscalTransferResult`,
`fake_news_algorithm` y `simulate_targeted_transfer`.

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
    transfer_amount=500.0,
    target_deciles=[1, 2, 3],
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
