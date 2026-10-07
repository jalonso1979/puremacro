> 🇬🇧 [English](../vfi_continuous_transition.md) · 🇪🇸 Español

# Dinámica de transición continua y choques MIT

`puremacro.vfi.continuous_transition` implementa la dinámica de transición no lineal en equilibrio general para modelos macroeconómicos continuos con agentes heterogéneos tras perturbaciones agregadas imprevistas (**choques MIT**). El motor acopla la recursión hacia atrás de las reglas de decisión mediante el Método de la Malla Endógena (EGM continuo) con la propagación hacia adelante de la distribución transversal mediante el método no estocástico de **Young (2010)**, vaciando simultáneamente los mercados en el espacio de secuencias mediante un algoritmo especializado de Broyden Cuasi-Newton (**Auclert et al. 2021**).

Si bien las perturbaciones linealizadas de primer orden proporcionan aproximaciones locales precisas cerca del estado estacionario determinista, resultan insuficientes para capturar las no linealidades inducidas por choques de política de gran magnitud (como subidas abruptas de tipos de interés soberanos, paquetes de estímulo fiscal extraordinarios o perturbaciones estructurales de productividad). Ante dichos choques, las restricciones de endeudamiento asimétricas se activan, los motivos precautorios se intensifican y la distribución transversal de la riqueza $\mu_t(k, z)$ experimenta reasignaciones cuantitativamente relevantes. `puremacro` calcula la **trayectoria exacta de transición de equilibrio no lineal** sobre horizontes arbitrarios $T$ garantizando la conservación estricta de la masa de probabilidad ($\sum \mu_t = 1.0 \pm 10^{-12}$) en cada fecha.

---

## 1. Marco teórico y computacional

### 1.1 El problema de transición en el espacio de secuencias

Considérese una economía inicializada en una distribución de estado estacionario $\mu_0(k, z)$ en la fecha $t = 0$. Una secuencia determinista e imprevista de choques agregados $\mathbf{Z} = (Z_0, Z_1, \dots, Z_{T-1})$ impacta la economía en $t = 0$, donde el horizonte $T$ se elige suficientemente amplio para que el sistema converja al estado estacionario final $\mu_T(k, z)$ en el período $T$.

El equilibrio general se define como las secuencias de precios factoriales $\{r_t, w_t\}_{t=0}^{T-1}$, reglas de decisión de los hogares $\{c_t(k, z), a'_t(k, z)\}_{t=0}^{T-1}$ y distribuciones continuas de riqueza $\{\mu_t(k, z)\}_{t=0}^{T}$ tales que:
1. Los hogares optimizan hacia atrás tomando la trayectoria de tipos de interés y salarios como dada.
2. La distribución transversal evoluciona hacia adelante bajo las reglas de decisión variantes en el tiempo.
3. Las empresas competitivas maximizan beneficios y el mercado de capitales se vacía en cada fecha $t \in [0, T-1]$:

$$H_t(\mathbf{r}) \equiv K_t^s(\mathbf{r}) - K_t^d(r_t; Z_t) = 0$$

donde la oferta agregada de capital viene dada por la integral continua transversal:

$$K_t^s = \sum_{m=1}^{n_z} \int_{\underline{k}}^{\bar{k}} k \, \mu_t(dk, z_m)$$

### 1.2 Recursión hacia atrás de políticas continuas vía EGM

Dada una trayectoria tentativa de tipos de interés $\mathbf{r} = (r_0, r_1, \dots, r_{T-1})$ y los correspondientes salarios competitivos $w_t = (1 - \alpha) Z_t (K_t^d / \bar{L})^\alpha$, el consumo y el ahorro óptimos se resuelven **hacia atrás en el tiempo** desde $t = T-1$ hasta $t = 0$ aplicando el Método de la Malla Endógena (EGM) sin sesgo de discretización.

En la fecha $t$, conocida la política de consumo de continuación $c_{t+1}(a', z')$, la ecuación de Euler define la utilidad marginal esperada:

$$\mathcal{EMU}_t(a', z_j) = \beta (1 + r_{t+1}) \sum_{m=1}^{n_z} P_z(j, m) \cdot u'\left( c_{t+1}(a', z_m) \right)$$

Invirtiendo la función de utilidad marginal se obtiene el consumo endógeno $c_t^{\text{endo}}(a', z_j) = (u')^{-1}(\mathcal{EMU}_t(a', z_j))$. El estado de riqueza previo a la decisión se deduce analíticamente de la restricción presupuestaria:

$$k_t^{\text{endo}}(a', z_j) = \frac{c_t^{\text{endo}}(a', z_j) + a' - w_t z_j}{1 + r_t}$$

Las funciones de política $a'_t(k, z)$ y $c_t(k, z)$ se interpolan sobre la rejilla densa y la de histograma, imponiendo la restricción de endeudamiento $a'_t \ge \underline{k}$ mediante truncamiento en la frontera inferior.

### 1.3 Propagación de densidad hacia adelante de Young (2010)

Partiendo de la distribución inicial predeterminada $\mu_0 = \mu_{\text{init}}^*$, la distribución transversal avanza período a período para $t = 0, \dots, T-1$ mediante los operadores lineales de lotería de Young (2010):

$$\mu_{t+1} = \mathcal{T}_t^* \mu_t = \text{continuous\_push\_distribution}(\mu_t, a'_t, \mathbf{k}, \mathbf{P}_z, \mathbf{z})$$

Para cada estado $(k_i, z_j)$, la elección continua $a'_t(k_i, z_j)$ se acota entre nodos adyacentes $k_l \le a'_t \le k_{l+1}$ asignando pesos de lotería lineal:

$$\omega_l = \frac{k_{l+1} - a'_t}{k_{l+1} - k_l}, \quad \omega_{l+1} = 1 - \omega_l$$

Esto garantiza:
- **Conservación exacta de la masa**: $\sum_{i, m} \mu_t(k_i, z_m) = 1.0 \pm 10^{-12}$ para todo $t \in [0, T]$.
- **Cero difusión numérica**: La oferta de capital $K_t^s = \sum_{i, m} k_i \mu_t(k_i, z_m)$ coincide con la agregación exacta de las decisiones individuales.

### 1.4 Vaciado de mercado en el espacio de secuencias vía Broyden

La trayectoria de tipos de interés de equilibrio $\mathbf{r}^* \in \mathbb{R}^T$ resuelve el sistema no lineal apilado:

$$\mathbf{H}(\mathbf{r}) = \begin{bmatrix} K_0^s(\mathbf{r}) - K_0^d(r_0; Z_0) \\ \vdots \\ K_{T-1}^s(\mathbf{r}) - K_{T-1}^d(r_{T-1}; Z_{T-1}) \end{bmatrix} = \mathbf{0}$$

Para resolver $\mathbf{H}(\mathbf{r}) = \mathbf{0}$ de forma eficiente sin reevaluar repetidamente matrices jacobianas numéricas, `puremacro` utiliza el **método Cuasi-Newton de Broyden con actualizaciones de Sherman-Morrison de rango 1**:

1. **Inicialización analítica**: La aproximación inicial del jacobiano inverso $\mathbf{B}_0 \approx \mathbf{J}^{-1}$ se inicializa mediante la derivada analítica de la demanda de capital:
   $$J_{t, t}^{\text{init}} \approx -\frac{\partial K_t^d}{\partial r_t} = \frac{1}{1 - \alpha} \frac{K_t^d}{r_t + \delta} > 0, \quad \mathbf{B}_0 = \text{diag}\left( \frac{1}{J_{t, t}^{\text{init}}} \right)$$
2. **Paso Cuasi-Newton**: En la iteración $k$, la actualización propuesta es:
   $$\Delta \mathbf{r}^{(k)} = - \mathbf{B}_k \mathbf{H}(\mathbf{r}^{(k)})$$
3. **Búsqueda lineal con retroceso monótono**: Partiendo de $\lambda = 1$, el paso se divide por dos (como mucho 12 veces) hasta que la norma euclídea del residuo baja lo suficiente:
   $$\|\mathbf{H}(\mathbf{r}^{(k)} + \lambda \Delta \mathbf{r}^{(k)})\|_2 \le (1 - 10^{-4} \lambda)\, \|\mathbf{H}(\mathbf{r}^{(k)})\|_2$$
   Si ningún paso lo cumple, $\mathbf{B}_k$ se reinicia a $\mathbf{B}_0$ y se da el paso amortiguado $\mathbf{r}^{(k)} - \omega\, \mathbf{B}_0 \mathbf{H}(\mathbf{r}^{(k)})$, con $\omega$ = `damping`. Ese paso de reserva es el único lugar en que Broyden usa `damping`. Con `backtracking=False` todos los pasos son de reserva y se descartan las actualizaciones de rango uno. La búsqueda lineal no falló en ninguna ejecución del cuaderno 52: con el valor por defecto `backtracking=True`, `damping` de 0.2, 0.4 y 0.8 dio trayectorias de Broyden idénticas bit a bit para los choques de $-8\%$, $-5\%$, $+0.1\%$, $+5\%$ y $+8\%$ con persistencia 0.5, 0.8 y 0.92 ($T = 40$). Con `backtracking=False` el mismo choque de $+5\%$ necesitó 31, 14 y 17 iteraciones.
4. **Actualización de Sherman-Morrison**: Definiendo $\Delta \mathbf{H} = \mathbf{H}^{(k+1)} - \mathbf{H}^{(k)}$ y $\Delta \mathbf{r} = \mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}$:
   $$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\Delta \mathbf{r} - \mathbf{B}_k \Delta \mathbf{H}) (\Delta \mathbf{r}^\top \mathbf{B}_k)}{\Delta \mathbf{r}^\top \mathbf{B}_k \Delta \mathbf{H}}$$
5. **Criterio de convergencia**: El proceso finaliza cuando $\|\mathbf{H}(\mathbf{r})\|_\infty < \text{tol}$ (típicamente $10^{-4}$).

---

## 2. Opciones metodológicas y del solucionador

| Característica | Broyden Cuasi-Newton (`solver="broyden"`) | Relajación de disparo (*shooting*) (`solver="shooting"`) |
|---|---|---|
| **Tasa de convergencia** | Superlineal (típicamente 4–8 iteraciones) | Lineal (20–60 iteraciones) |
| **Mecanismo de paso** | Actualización completa de Sherman-Morrison en rango 1 | Relajación amortiguada: $\mathbf{r}^{(n+1)} = (1 - \omega)\mathbf{r}^{(n)} + \omega \mathbf{r}^{\text{implied}}$ |
| **Búsqueda lineal** | Búsqueda retrógrada monótona de tipo Armijo sobre $\|\mathbf{H}\|_2$; `damping` solo escala el paso de reserva (sección 1.4) | Parámetro fijo de amortiguación $\omega$ = `damping` $\in (0, 1]$ en cada paso |
| **Robustez** | Excepcional ante choques persistentes y de gran tamaño | Contracción garantizada para innovaciones marginales |
| **Tiempo de cálculo** | $< 0.5$ segundos para $T = 150$ | $1.0$–$2.5$ segundos para $T = 150$ |

---

## 3. Calibración canónica y especificación de choques

La dinámica de transición admite tres clases de perturbaciones macroeconómicas imprevistas. En lo que sigue, $s_t$ es el elemento de `shock_path` en la fecha $t$; `continuous_mit_shock` construye $s_t = \Delta \cdot \rho^t$ a partir de `shock_size` $\Delta$ y `persistence` $\rho$ ($s_t = \Delta$ en todo $t$ cuando $\rho = 1$). En `continuous_mit_shock`, $\Delta$ es siempre una desviación, nunca un nivel: entrega a `solve_continuous_transition` los niveles $Z_t = 1 + s_t$ o $\beta_t = \beta(1 + s_t)$ siempre que la regla de abajo leería la trayectoria como niveles. Hasta la versión 4.3.0 pasaba siempre $s_t$, de modo que `shock_size=0.6` daba $Z_0 = 0.6$, una caída del 40%, en lugar de $1.6$. Un choque de PTF debe ser mayor que $-1$.

1. **Choque de Productividad Total de los Factores (`shock_var="z"`)**:
   $$Z_t = 1 + s_t$$
   cuando todo $s_t \le 0.5$; si algún elemento supera $0.5$, la trayectoria se lee en niveles, $Z_t = s_t$. Una trayectoria más corta que $T$ se mantiene en su último valor. Permite modelar perturbaciones transitorias ($\rho < 1$) o permanentes ($\rho = 1$); una permanente resuelve el estado estacionario final en $Z_{T-1}$.
2. **Choque monetario / cuña de tipo de interés (`shock_var="r"`)**:
   $$r_t^{\text{hh}} = r_t + s_t$$
   Los hogares obtienen $r_t + s_t$ en la ecuación de Euler y en la restricción presupuestaria, mientras que las empresas pagan $r_t$. Una trayectoria más corta que $T$ se completa con ceros. Simula endurecimiento de la política monetaria o ensanchamiento de diferenciales de crédito. Solo se admiten cuñas transitorias (véase la limitación de la condición terminal más abajo).
3. **Choque en el factor de descuento / paciencia (`shock_var="beta"`)**:
   $$\beta_t = \beta\,(1 + s_t)$$
   multiplicativo, con $\beta$ el factor de descuento del estado estacionario inicial, cuando todo $s_t \le 0.5$; si algún elemento supera $0.5$, la trayectoria se lee en niveles, $\beta_t = s_t$. $\beta_t$ descuenta la fecha $t+1$ en la ecuación de Euler de la fecha $t$. Una trayectoria más corta que $T$ se completa con $\beta$. Por ejemplo, `shock_size=0.01` con `persistence=0.8` da $\beta_0 = 1.01\,\beta$ y $\beta_1 = 1.008\,\beta$. Un choque permanente resuelve el estado estacionario final en $\beta_{T-1}$. Modela episodios de preferencia por la liquidez y aumentos repentinos del ahorro precautorio.

### 3.1 Condición terminal y su limitación

La trayectoria se resuelve hacia atrás desde un estado estacionario final en la fecha $T$: la política de consumo de continuación $c_T$ y el tipo $r_T$.

**`continuous_mit_shock`** lo elige solo a partir de `persistence`:

- `persistence < 1` (transitorio): siempre el estado estacionario inicial, y el choque se anula a partir de la fecha $T$. Cuando la fracción del impacto que queda en $T-1$, $\rho^{T-1}$, supera `truncation_tol` (por defecto $10^{-3}$), un `RuntimeWarning` indica el horizonte más corto que la lleva por debajo de la tolerancia. `metadata["mit_shock"]` informa del truncamiento en todos los casos (sección 6).
- `persistence == 1` (permanente, que hay que pedir explícitamente): para choques de PTF y de factor de descuento, un estado estacionario resuelto internamente en $Z = 1 + \Delta$ o $\beta(1 + \Delta)$ con los controles de la sección 5. Para una cuña de tipo de interés, véase más abajo.
- Un `terminal_steady_state` pasado como palabra clave se usa tal cual, con cualquier persistencia.

Hasta la versión 4.3.0 un choque transitorio seguía la regla de `solve_continuous_transition` de abajo. En cuanto $Z_{T-1}$ se alejaba de 1 más de unos $10^{-5}$, el choque se resolvía en silencio como permanente en $Z_{T-1}$. Con $T = 40$ y $\Delta = 0.05$ el modelo cambiaba en $\rho \approx 0.806$. Con $\rho = 0.92$ la economía de la sección 4 convergía a un estado estacionario con una PTF un 0.19% mayor y un 0.30% más de capital, y su trayectoria de capital se alejaba de la solución transitoria hasta en 0.026.

**`solve_continuous_transition`**, llamada directamente con una trayectoria, usa:

- `terminal_steady_state`, si se pasa uno;
- si no, cuando $Z_{T-1}$ difiere de 1 o $\beta_{T-1}$ de $\beta$ más de lo que permite `numpy.isclose(..., atol=1e-6)` (unos $10^{-5}$), un estado estacionario resuelto internamente en $Z_{T-1}$ y $\beta_{T-1}$ con los controles de la sección 5;
- en otro caso, el estado estacionario inicial.

Esa regla no distingue una trayectoria transitoria que decae despacio de una permanente. Para una trayectoria transitoria que no se ha extinguido en $T-1$, pase `terminal_steady_state=` el estado estacionario inicial (como hace `continuous_mit_shock`) o alargue el horizonte.

**Cómo elegir $T$.** El aviso de truncamiento solo mira el choque exógeno, y el capital vuelve más despacio que el choque. En el cuaderno 52 ($\rho = 0.8$, $T = 40$) el choque que queda en $T-1$ es $1.7 \times 10^{-4}$ del impacto, pero $K_{T-1} - K^*$ sigue siendo $0.040$. `metadata["mit_shock"]["capital_gap_last"]` informa de esa brecha; resolver de nuevo con un horizonte más largo mide el error de truncamiento.

**No se admite una cuña de tipo de interés permanente.** Con `shock_var="rate"` y una cuña que sigue siendo distinta de cero en $T-1$ (por ejemplo `persistence=1`), la condición terminal sigue siendo el estado estacionario inicial: $c_T$ es la política en $r^*$ y $r_T = r^*$ no lleva la cuña, aunque los hogares se enfrentan a $r + s$ para siempre. La trayectoria se resuelve entonces contra una condición terminal incoherente, y `converged=True` solo indica que el mercado de capitales se vacía en cada fecha de esa trayectoria. No indica que la trayectoria haya alcanzado el nuevo estado estacionario. `solve_continuous_transition` no emite ningún aviso; `continuous_mit_shock(..., shock_type="rate", persistence=1.0)` emite un `RuntimeWarning` y activa `metadata["mit_shock"]["truncated"]`. En la economía de la sección 4 ($N_k = 100$, $n_z = 3$), una cuña permanente de $+50$ pb devuelve `converged=True` con $r_{T-1} = 0.0332$ para $T = 40$ y $0.0325$ para $T = 80$, frente al tipo final $r_T = r^* = 0.0394$: la trayectoria sigue moviéndose en $T$. `solve_aiyagari_continuous` no tiene argumento de cuña, así que tampoco se puede pasar un estado estacionario final con ella. Use cuñas de tipo de interés que se hayan extinguido en $T-1$ y compruebe que `r_path[-1]` ha vuelto al tipo final.

---

## 4. Ejemplos prácticos ejecutables

El siguiente script calcula la trayectoria exacta de transición no lineal tras una innovación imprevista del $+5\%$ en la PTF en una economía de Aiyagari:

```python
import numpy as np
from puremacro.vfi import (
    solve_aiyagari_continuous,
    solve_continuous_transition,
    continuous_mit_shock,
    TransitionShock,
)
from puremacro.vfi.aggregate import lorenz_and_gini

# 1. Cálculo del equilibrio general estacionario inicial
init_ss = solve_aiyagari_continuous(
    beta=0.96,
    gamma=2.0,
    alpha=0.36,
    delta=0.08,
    N_k=100,
    n_z=3,
    max_evals=20,
)

# 2. Simulación del choque MIT no lineal (+5% PTF con persistencia 0.75)
T_sim = 40
res_trans = continuous_mit_shock(
    steady_state=init_ss,
    shock_type="tfp",
    shock_size=0.05,
    persistence=0.75,
    horizon=T_sim,
    solver="broyden",
    max_iter=30,
)

assert res_trans.converged
assert len(res_trans.r_path) == T_sim
assert res_trans.mass_conservation_error < 1e-12

# 3. Métricas dinámicas de desigualdad a lo largo de la transición
k_grid = init_ss.distribution.asset_grid
mu_0 = np.sum(res_trans.distributions[0], axis=1)
mu_T = np.sum(res_trans.distributions[-1], axis=1)

_, _, gini_0 = lorenz_and_gini(k_grid, mu_0)
_, _, gini_T = lorenz_and_gini(k_grid, mu_T)

summary_df = res_trans.summary()
print(f"Transición convergida en {res_trans.iterations} iters, residuo máx: {res_trans.max_residual:.2e}")
print(f"Gini de riqueza: t=0 -> {gini_0:.4f}, t={T_sim} -> {gini_T:.4f}")
```

---

## 5. Especificación completa de la API

```text
TransitionShock(
    path: np.ndarray,
    var: str = "z",
)

solve_continuous_transition(
    initial_steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    terminal_steady_state: AiyagariContinuousEquilibrium | dict[str, Any] | None = None,
    shock_path: np.ndarray | Sequence[float] | None = None,
    shock_var: str = "z",
    horizon: int = 150,
    solver: str = "broyden",
    damping: float = 0.3,
    tol: float = 1e-4,
    max_iter: int = 100,
    backtracking: bool = True,
    backend: str = "numpy",
    r_init_path: np.ndarray | Sequence[float] | None = None,
    **kwargs: Any,
) -> ContinuousTransitionResult

continuous_mit_shock(
    steady_state: AiyagariContinuousEquilibrium | dict[str, Any],
    shock_type: str = "tfp",
    shock_size: float = 0.05,
    persistence: float = 0.8,
    horizon: int = 150,
    solver: str = "broyden",
    *,
    truncation_tol: float = 1e-3,
    **kwargs: Any,
) -> ContinuousTransitionResult
```

#### Parámetros:
- `initial_steady_state`: Objeto de equilibrio general precalculado `AiyagariContinuousEquilibrium`, o un diccionario de configuración que se pasa a `solve_aiyagari_continuous` para resolver un nuevo estado estacionario. El diccionario puede contener **solo** palabras clave de `solve_aiyagari_continuous` (`beta`, `gamma`, `alpha`, `delta`, `rho_z`, `sigma_z`, `n_z`, `a_max`, `N_k`, `n_a`, `P_z`, `z_grid`, ...). Cualquier otra clave, como las mallas, la distribución o los precios de un estado estacionario resuelto en otro sitio, lanza `TypeError`. Para partir de un estado estacionario que ya tiene, pase el propio objeto `AiyagariContinuousEquilibrium`.
- `terminal_steady_state`: Equilibrio final de destino (objeto o diccionario, como arriba). Si se omite, `solve_continuous_transition` usa el estado estacionario inicial para trayectorias extinguidas en $T-1$ y, en otro caso, resuelve internamente un estado estacionario en $Z_{T-1}$ y $\beta_{T-1}$, lo que incluye una trayectoria transitoria que aún no se ha extinguido. `continuous_mit_shock` pasa el estado estacionario inicial para todo choque transitorio (sección 3.1). No se admite una cuña de tipo de interés permanente (sección 3.1).
- `shock_path`: Vector unidimensional con la trayectoria del choque de longitud $T$, interpretado según `shock_var` como en la sección 3.
- `shock_var` / `shock_type`: Variable objetivo: `'tfp'` / `'z'` / `'productivity'`, `'rate'` / `'r'` / `'monetary'`, o `'beta'` / `'discount'`.
- `horizon`: Longitud temporal del horizonte de simulación $T$ (por defecto $150$).
- `solver`: Solucionador de vaciado: `'broyden'` (Cuasi-Newton) o `'shooting'` (relajación con amortiguación).
- `damping`: El peso de relajación $\omega$ de `solver='shooting'`, usado en cada paso. Con `solver='broyden'` solo escala el paso de reserva que se da cuando falla la búsqueda lineal (y todos los pasos si `backtracking=False`); véase la sección 1.4. Con el valor por defecto `backtracking=True` no tuvo ningún efecto en las ejecuciones del cuaderno 52.
- `tol`: Tolerancia de vaciado en el mercado de capitales $\|K^s - K^d\|_\infty$ (por defecto $10^{-4}$).

#### Parámetros de `continuous_mit_shock`:
- `steady_state`: El estado estacionario inicial (objeto o diccionario de configuración, como `initial_steady_state` arriba).
- `shock_type`: `'tfp'`, `'rate'` o `'beta'`, o cualquier alias de `shock_var`.
- `shock_size`: El impacto $\Delta = s_0$, siempre una desviación: $+0.05$ es $+5\%$ de PTF, $+0.01$ es $+100$ pb en el tipo, y en `'beta'` da $\beta_0 = \beta(1 + \Delta)$. Un choque de PTF debe ser mayor que $-1$.
- `persistence`: $\rho \in [0, 1]$. $\rho = 1$ es un choque permanente; cualquier valor menor que 1, incluido $0.999$, es transitorio y mantiene el estado estacionario inicial como condición terminal (sección 3.1).
- `truncation_tol`: La mayor fracción $\rho^{T-1}$ del impacto que un choque transitorio puede conservar en $T-1$ sin un `RuntimeWarning` (por defecto $10^{-3}$).
- `**kwargs`: Se pasan a `solve_continuous_transition` (`damping`, `tol`, `max_iter`, `backtracking`, `backend`, `r_init_path`, `terminal_steady_state` y las palabras clave de abajo).

#### Parámetros estructurales y `**kwargs`:
- Los parámetros estructurales `beta`, `gamma`, `alpha` y `delta` se leen de `metadata["params"]` del estado estacionario inicial, que `solve_aiyagari_continuous` registra. Pasar uno de ellos como palabra clave con otro valor lanza `ValueError`. Las palabras clave solo se usan cuando el estado estacionario no registra ninguno (uno construido con `continuous_stationary_equilibrium` o a mano); si falta alguno, se usa el valor por defecto de `solve_aiyagari_continuous` con un `UserWarning`.
- `egm_tol`, `egm_max_iter`, `xtol`, `tol_ge`, `max_evals`, `dist_options`: controles del estado estacionario final resuelto internamente, con el mismo significado que en `solve_aiyagari_continuous`. Si no se dan, los cinco primeros se toman de los metadatos del estado estacionario inicial y, en su defecto, de los valores por defecto de esa función ($10^{-8}$, $10\,000$, $10^{-8}$, $10^{-4}$, $100$); `dist_options` no se lee de los metadatos y vale `None` por defecto.
- `r_min`, `r_max`: cotas de la trayectoria de prueba del tipo de interés (por defecto $10^{-4}$ y $\max(0.15,\ 1/\beta - 1 + 0.10)$).
- Cualquier otra palabra clave se ignora con un `FutureWarning`; lanzará `TypeError` en una versión futura.
- `continuous_mit_shock(..., **kwargs)` transmite estas palabras clave a `solve_continuous_transition`.

---

## 6. Interfaz de resultados y métricas dinámicas de desigualdad

`ContinuousTransitionResult` almacena la trayectoria dinámica completa y las métricas de desigualdad asociadas:

### Atributos del contenedor
- `r_path`: Trayectoria del tipo de interés real $r_0, \dots, r_{T-1}$.
- `w_path`: Trayectoria del salario real competitivo $w_0, \dots, w_{T-1}$.
- `K_s_path`: Secuencia de oferta agregada de capital $K_t^s = \int k \, d\mu_t$.
- `K_d_path`: Secuencia de demanda agregada de capital $K_t^d(r_t)$.
- `C_path`: Secuencia de consumo agregado $C_t = \int c_t \, d\mu_t$.
- `distributions`: Lista de $T+1$ distribuciones transversales conjuntas $\mu_0, \mu_1, \dots, \mu_T$.
- `residuals`: Trayectoria de excesos de demanda de capital $H_t = K_t^s - K_t^d$.
- `max_residual`: Residuo máximo de vaciado $\|H\|_\infty$.
- `converged`: Indicador booleano de convergencia exitosa.
- `mass_conservation_error`: Desviación absoluta máxima de masa a lo largo de los $T+1$ períodos.
- `metadata`: Diagnósticos del solucionador: `params`, `relaxation_converged`, `terminal_steady_state` (los diagnósticos de un estado estacionario final resuelto internamente, o `None`) y, salvo que un choque nulo devuelva el estado estacionario de inmediato, `Z_path` y `beta_path`. `continuous_mit_shock` añade `metadata["mit_shock"]`, con `shock_type`, `shock_size`, `persistence`, `permanent`, `terminal_condition` (`'initial_steady_state'`, `'solved_steady_state'` o `'user'`), `shock_at_last_date` ($s_{T-1}$), `remaining_share` ($\rho^{T-1}$), `truncation_tol`, `truncated`, `horizon_needed` (el menor $T$ con $\rho^{T-1} \le$ `truncation_tol`; `None` para un choque permanente) y `capital_gap_last` ($K_{T-1} - K^*$ para un choque transitorio).

### Métodos de serialización y gráficos
- `res.summary() -> pd.DataFrame`: Resumen de diagnósticos de convergencia, tiempo de cálculo y agregados iniciales/finales.
- `res.to_frame() -> pd.DataFrame`: DataFrame con las series temporales completas del espacio de secuencias ($r_t, w_t, K_t^s, K_t^d, C_t, H_t$).
- `res.to_markdown(digits=4) -> str`: Tabla en formato Markdown.
- `res.to_latex(digits=4) -> str`: Entorno tabular en LaTeX de calidad para publicaciones académicas.
- `res.to_typst(digits=4) -> str`: Código Typst formateado para informes científicos.
- `res.plot(figsize=(14, 8)) -> matplotlib.figure.Figure`: Gráfico diagnóstico multipanel que visualiza tipos de interés, salarios, vaciado de capital, consumo agregado y la evolución de la distribución de riqueza $\mu_t(k)$.

---

## Referencias

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *The Quarterly Journal of Economics*, 109(3), 659–684.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica*, 89(6), 3115–3148.
- Young, E. R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and a non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.
