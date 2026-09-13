> 🇬🇧 [English](../deep_macro.md) · 🇪🇸 Español

# Deep Macro y redes neuronales informadas por la física

`puremacro.vfi.deep_macro` implementa un solucionador para modelos macroeconómicos dinámicos en alta dimensión basado en **Redes Neuronales Informadas por la Física (PINN, *Physics-Informed Neural Networks*)** y muestreo a lo largo de trayectorias ergódicas, siguiendo la metodología pionera de **Maliar, Maliar y Winant (2021, *Journal of Monetary Economics*)**. Este motor resuelve modelos económicos dinámicos con $10+$ variables de estado continuas, donde los métodos tradicionales de colocación en producto tensorial o rejillas dispersas sucumben ante la maldición de la dimensionalidad.

De manera fundamental, toda la arquitectura de aprendizaje profundo —incluyendo la propagación hacia adelante del perceptrón multicapa, la retropropagación analítica exacta y el algoritmo de optimización Adam— está implementada en **NumPy puro vectorizado**. Este diseño garantiza un cumplimiento estricto del entorno de ejecución de Pyodide, WebAssembly y navegadores web (cero importaciones de PyTorch, TensorFlow o JAX), incorporando al mismo tiempo conectores opcionales de aceleración por hardware para Apple Silicon MLX y NVIDIA CuPy.

---

## 1. Marco teórico y computacional

### 1.1 La maldición de la dimensionalidad en macroeconomía dinámica

En modelos económicos dinámicos con producción multisectorial, encadenamientos comerciales internacionales o estructuras financieras complejas, el vector continuo de estados $s_t \in \mathbb{R}^D$ supera con frecuencia las $D \ge 10$ dimensiones (como 10 stocks de capital nacionales $k_t = (k_{1, t}, \dots, k_{10, t})'$).

Los métodos de proyección global convencionales discretizan cada dimensión en $N$ nodos:
- **Rejillas en producto tensorial completo**: Exigen $N^D$ puntos de evaluación. Para $N = 10$ y $D = 10$, esto implica $10^{10}$ nodos, una barrera computacional inabordable.
- **Mallas dispersas de Smolyak**: Reducen la complejidad a $O(D^\mu / \mu!)$, pero se encuentran restringidas en la práctica a dimensiones $D \le 6$ para órdenes polinómicos moderados.

Las Redes Neuronales Informadas por la Física eluden la discretización espacial en rejillas. Al parametrizar las reglas de decisión mediante redes neuronales profundas $c(s; \mathbf{w})$ y entrenar exclusivamente sobre el conjunto atractor ergódico económicamente relevante, el aprendizaje profundo escala eficazmente a espacios de estado de ultra-alta dimensión.

### 1.2 Formulación de las Redes Neuronales Informadas por la Física (PINN)

Considérese una economía regida por el sistema de residuos de Euler continuo:

$$\mathcal{R}(s, c(s); s', c(s')) \equiv u'(c(s)) - \beta \mathbb{E} \left[ u'(c(s')) \cdot \left( \mathbf{f}'(k', z') + 1 - \delta \right) \right] = \mathbf{0}$$

Un Perceptrón Multicapa (MLP) aproxima la función continua de política $c(s; \mathbf{w}): \mathbb{R}^D \to \mathbb{R}^P$, donde $\mathbf{w} = \{W^{(l)}, b^{(l)}\}_{l=0}^{L-1}$ representa las matrices de pesos y vectores de sesgo de la red.

El objetivo de la PINN consiste en minimizar el Error Cuadrático Medio de la ecuación de Euler sobre un lote de estados muestreados:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{B} \sum_{b=1}^B \left\| \mathcal{R}\left( s_b, c(s_b; \mathbf{w}); s'_b, c(s'_b; \mathbf{w}) \right) \right\|_2^2$$

A diferencia del aprendizaje supervisado estándar, que requiere etiquetas precalculadas o datos de entrenamiento externos, las PINN se entrenan directamente sobre las condiciones de equilibrio del modelo: la pérdida es cero si y solo si se satisface la ecuación intertemporal de Euler.

### 1.3 Arquitectura MLP en NumPy puro y seguridad para Pyodide

Para garantizar cero dependencias externas y ejecución inmediata en entornos de navegador (como Juno o Google Colab bajo Pyodide), `DeepMacroMLP` implementa las fases de propagación y retropropagación utilizando álgebra lineal vectorizada en NumPy:

#### Propagación hacia adelante
Para la capa $l = 0, \dots, L-1$:

$$z^{(l+1)} = a^{(l)} W^{(l)} + b^{(l)}, \quad a^{(l+1)} = \sigma\left( z^{(l+1)} \right)$$

donde $a^{(0)} = s \in \mathbb{R}^{B \times D}$, y $\sigma(\cdot)$ es una función de activación suave.

#### Retropropagación analítica
Mediante la regla de la cadena matricial exacta, los gradientes de salida $\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'(z^{(L)})$ se propagan hacia atrás:

$$\delta^{(l)} = \left( \delta^{(l+1)} (W^{(l)})^\top \right) \odot \sigma'\left( z^{(l)} \right)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{B} (a^{(l)})^\top \delta^{(l+1)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{B} \sum_{b=1}^B \delta_b^{(l+1)}$$

### 1.4 Muestreo en trayectorias ergódicas (Maliar, Maliar & Winant 2021)

El muestreo en rejillas uniformes desperdicia recursos evaluando regiones del espacio de estados cuya probabilidad de visita en equilibrio es nula. Maliar et al. (2021) introducen el **muestreo en trayectorias ergódicas**:

1. **Simulación**: Desde un estado inicial $s_0$, se simulan trayectorias temporales de equilibrio bajo la política neuronal corriente:
   $$s_{t+1} = \mathcal{T}\left( s_t, c(s_t; \mathbf{w}), \epsilon_{t+1} \right), \quad t = 0, \dots, T_{\text{sim}}$$
2. **Descarte de transición**: Se descarta el período inicial $T_{\text{burn}}$ para eliminar la dependencia respecto a las condiciones iniciales.
3. **Muestreo por minilotes**: Se extraen lotes de entrenamiento aleatorios $s_b \sim \{s_t\}_{t=T_{\text{burn}}}^{T_{\text{sim}}}$ exclusivamente del conjunto ergódico de alta densidad.
4. **Re-simulación periódica**: La trayectoria se vuelve a simular cada $K_{\text{resim}}$ épocas conforme la regla de política evoluciona.

### 1.5 Activación acotada y garantías de viabilidad física

Las decisiones de control económico están sujetas a restricciones de recursos fundamentales. Por ejemplo, el consumo debe ser estrictamente positivo ($c > 0$) y no puede superar la riqueza disponible o liquidez total ($c < W(s)$).

`DeepMacroModel` impone la viabilidad física aplicando una función de acotamiento sigmoide a las salidas brutas de la red $\tilde{c} = \text{MLP}(s)$:

$$c_i(s) = \epsilon_{\min} + (W_i(s) - 2\epsilon_{\min}) \cdot \sigma\left( \tilde{c}_i \right)$$

donde $W_i(s) = A_i k_i^\alpha + (1 - \delta)k_i$ representa la riqueza total disponible. Esta garantía estructural previene que el optimizador explore estados económicamente inviables (consumo o capital negativos) durante el entrenamiento.

---

## 2. Opciones metodológicas y del solucionador

### Funciones de activación suaves
`DeepMacroMLP` soporta funciones de activación suaves e infinitamente diferenciables con derivadas analíticas exactas:

- **SiLU / Swish** ($\sigma(x) = x \cdot \text{sigmoid}(x)$): Activación no lineal predeterminada que evita la anulación del gradiente.
- **GELU** ($\sigma(x) = 0.5 x (1 + \text{erf}(x / \sqrt{2}))$): Unidad lineal de error gaussiano estándar en modelos basados en transformers.
- **Tanh** ($\sigma(x) = \tanh(x)$): Activación suave simétrica centrada en cero.
- **Softplus** ($\sigma(x) = \ln(1 + e^x)$): Suavizado diferenciable de la función ReLU.

### Optimizador Adam en NumPy puro
La clase `AdamOptimizer` implementa la actualización de momentos con corrección de sesgo:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_{t-1}$$

incorporando truncamiento de la norma del gradiente ($\|g\|_2 \le \bar{G}$) para asegurar estabilidad numérica.

---

## 3. Calibración canónica y especificación en alta dimensión

El modelo canónico de acumulación de capital multipaís considera $N = 10$ economías interconectadas:

| Parámetro | Símbolo | Valor de referencia | Interpretación económica |
|---|---|---|---|
| Número de países (estados y controles) | $N$ | $10$ | 10 stocks de capital $k_i$, 10 decisiones de consumo $c_i$ |
| Participación del capital | $\alpha$ | $0.360$ | Coeficiente Cobb-Douglas en la función de producción |
| Factor de descuento subjetivo | $\beta$ | $0.960$ | Tasa anualizada de descuento temporal |
| Tasa de depreciación del capital | $\delta$ | $0.080$ | Depreciación física anual del capital |
| Topología de la red MLP | — | `(64, 64)` | 2 capas ocultas con 64 neuronas cada una (4,874 parámetros) |
| Activación oculta | — | `"silu"` | Función de activación SiLU |
| Épocas de entrenamiento | — | $50$–$400$ | Presupuesto de épocas (converge en $< 0.5$ s para 50 épocas) |
| Longitud de trayectoria ergódica | $T_{\text{sim}}$ | $1500$ | Horizonte temporal de la senda estocástica simulada |
| Tamaño de minilote | $B$ | $64$–$128$ | Número de vectores de estado por paso de gradiente |

---

## 4. Ejemplos prácticos ejecutables

El siguiente script define un modelo de crecimiento de 10 países ($D = 10$), entrena la red neuronal informada por la física sobre trayectorias ergódicas simuladas y valida el residuo de Euler fuera de muestra:

```python
import numpy as np
from puremacro.vfi.deep_macro import (
    DeepMacroModel,
    solve_deep_macro,
    DeepMacroSolution,
)

# 1. Definición del modelo de crecimiento de alta dimensión (10 estados, 10 controles)
model = DeepMacroModel.multi_country_growth(
    n_countries=10,
    alpha=0.36,
    beta=0.96,
    delta=0.08,
)
k_ss, c_ss = model.steady_state()
assert len(k_ss) == 10 and len(c_ss) == 10

# 2. Entrenamiento de la red neuronal informada por la física en NumPy puro
solution = solve_deep_macro(
    model,
    hidden_dims=(64, 64),
    activation="silu",
    n_epochs=50,
    batch_size=64,
    lr=2e-3,
    trajectory_length=1000,
    burn_in=100,
    seed=42,
    verbose=False,
)

assert isinstance(solution, DeepMacroSolution)
assert solution.converged
assert solution.test_euler_mse < 1e-3

# 3. Simulación de trayectoria y verificación de viabilidad física
sim_data = solution.simulate(periods=300, seed=123)
assert sim_data["physically_viable"]
summary_df = solution.summary()

print(f"Deep Macro resolvió el modelo de 10 países en {len(solution.loss_history)} épocas")
print(f"Pérdida final: {solution.loss_history[-1]:.2e}, ECM fuera de muestra: {solution.test_euler_mse:.2e}")
```

---

## 5. Especificación completa de la API

```text
DeepMacroModel(
    n_states: int,
    n_controls: int,
    beta: float = 0.96,
    params: dict[str, Any] = field(default_factory=dict),
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    transition_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray] | None = None,
    euler_residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    name: str = "Multi-Country Capital Accumulation",
)

DeepMacroModel.multi_country_growth(
    n_countries: int = 10,
    alpha: float = 0.36,
    beta: float = 0.96,
    delta: float = 0.08,
    A: float | np.ndarray = 1.0,
    gamma: float = 1.0,
    rho: float = 0.90,
    sigma_eps: float = 0.02,
) -> DeepMacroModel

DeepMacroMLP(
    input_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    output_dim: int = 1,
    activation: str = "silu",
    output_activation: str = "linear",
    seed: int = 42,
    backend: str = "numpy",
)

AdamOptimizer(
    params: list[np.ndarray],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
)

solve_deep_macro(
    model: DeepMacroModel,
    hidden_dims: Sequence[int] = (64, 64),
    activation: str = "silu",
    n_epochs: int = 400,
    batch_size: int = 128,
    lr: float = 2e-3,
    trajectory_length: int = 2000,
    burn_in: int = 200,
    resimulate_every: int = 50,
    weight_decay: float = 1e-6,
    grad_clip: float = 1.0,
    backend: str = "numpy",
    seed: int = 42,
    verbose: bool = False,
    **kwargs: Any,
) -> DeepMacroSolution
```

#### Parámetros:
- `model`: Objeto de problema que especifica las dimensiones de estado/control, el factor de descuento $\beta$ y los parámetros económicos.
- `hidden_dims`: Dimensiones de las capas ocultas de la red neuronal (p. ej. `(64, 64)`).
- `activation`: Función de activación suave (`"silu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"relu"`, `"softplus"`).
- `n_epochs`: Número total de épocas de entrenamiento.
- `batch_size`: Tamaño de minilote muestreado de la trayectoria ergódica.
- `lr`: Tasa de aprendizaje del optimizador Adam.
- `trajectory_length`: Longitud de la trayectoria ergódica simulada.
- `resimulate_every`: Frecuencia (en épocas) de re-simulación de la trayectoria de estados.
- `grad_clip`: Umbral de truncamiento de norma de gradiente.

---

## 6. Interfaz de resultados y validación diagnóstica

`DeepMacroSolution` reúne la red neuronal entrenada, el historial de convergencia y las métricas diagnósticas fuera de muestra:

### Atributos del contenedor
- `model`: Referencia al modelo `DeepMacroModel` original.
- `mlp`: Red neuronal `DeepMacroMLP` entrenada.
- `loss_history`: Historial del valor de la función de pérdida por época.
- `test_euler_mse`: Error Cuadrático Medio de Euler fuera de muestra ($\text{ECM} < 10^{-3}$).
- `test_euler_max`: Residuo absoluto máximo de la ecuación de Euler en la muestra de prueba.
- `test_trajectory`: Array de forma $(T_{\text{test}}, D)$ con la trayectoria simulada fuera de muestra.
- `elapsed_time`: Tiempo de ejecución en segundos.
- `converged`: Indicador booleano de convergencia exitosa ($\text{ECM} < 10^{-3}$).

### Métodos de evaluación y simulación
- `solution.policy(s) -> np.ndarray`: Evalúa la regla de decisión continua $c(s)$ en estados individuales o por lotes, garantizando viabilidad física ($0 < c_i < W_i$).
- `solution.simulate(s0=None, periods=300, seed=123) -> dict`: Simula trayectorias hacia adelante devolviendo estados, controles, residuos de Euler y viabilidad física.
- `solution.summary() -> pd.DataFrame`: Cuadro resumen con la arquitectura de la red, épocas, pérdida final, ECM de Euler y tiempo de cálculo.
- `solution.to_frame() -> pd.DataFrame`: Alias que devuelve el DataFrame del cuadro resumen.
- `solution.to_markdown(digits=4) -> str`: Tabla en formato Markdown.
- `solution.to_latex(digits=4) -> str`: Entorno tabular en LaTeX para publicaciones científicas.
- `solution.to_typst(digits=4) -> str`: Código Typst formateado.
- `solution.plot(figsize=(14, 4.5)) -> matplotlib.figure.Figure`: Gráfico diagnóstico multipanel que visualiza la convergencia de la pérdida, los residuos de Euler fuera de muestra y las sendas simuladas de acumulación de capital.

---

## Referencias

- Coleman, W. J. (1990). "Solving the stochastic growth model by policy-function iteration." *Journal of Business & Economic Statistics*, 8(1), 27–29.
- Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014). "Smolyak method for solving dynamic economic models: Lagrange interpolation, anisotropic grid and adaptive domain." *Journal of Economic Dynamics and Control*, 44, 92–123.
- Maliar, L., Maliar, S., & Winant, P. (2021). "Deep learning for solving dynamic economic models." *Journal of Monetary Economics*, 122, 76–101.
