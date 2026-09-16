> 🇬🇧 [English](../trade_gpu.md) · 🇪🇸 Español

# Backends de hardware: GPU, Apple MLX y jacobianos por lotes para el equilibrio general de comercio y espacial

`puremacro._backend` y `puremacro.trade.gpu` añaden aceleración por hardware opcional a los solucionadores de equilibrio general documentados en [Economía espacial cuantitativa y equilibrio general de comercio](spatial_and_trade_ge.md). Intervienen dos capas:

1. **La capa de espacio de nombres de arrays** (`puremacro._backend`): `numpy`, `mlx` (Apple Silicon) y `cupy` (NVIDIA) comparten una única API de arrays al estilo de NumPy. `solve_trade_equilibrium(..., backend=)` y `AllenArkolakisModel.solve_equilibrium(..., backend=)` / `solve_counterfactual(..., backend=)` encaminan a través de ella determinadas reducciones tensoriales, con un `RuntimeWarning` y un respaldo (*fallback*) a NumPy siempre que el backend solicitado falte o falle.
2. **La capa de dispositivo** (`puremacro.trade.gpu`): un motor PyTorch (CUDA / Apple MPS / CPU) y otro nativo de Apple MLX para el modelo de equilibrio general computable multipaís y multisectorial de Caliendo-Parro (2015), construido en torno a operadores de Leontief preinvertidos, un jacobiano por lotes evaluado como un único producto matricial (GEMM) con múltiples segundos miembros, un paso de Levenberg-Marquardt (1944, 1963) con búsqueda lineal con retroceso (*backtracking*) y continuación por homotopía adaptativa (Allgower y Georg, 1990) en el parámetro arancelario.

Ambas capas son opcionales. Nada de lo que figura en esta página es necesario para ejecutar los modelos: la ruta NumPy sigue siendo la implementación de referencia y el oráculo de corrección, y es la única ruta disponible en Pyodide (iPad, navegador). Todo resultado acelerado de esta página se compara contra ella.

---

## 1. Marco algorítmico

### 1.1 Qué backend ejecuta qué

`puremacro._backend.SUPPORTED` es `("numpy", "numba", "mlx", "cupy")`. `backend_available(name)` consulta `importlib.util.find_spec`, `available_backends()` enumera los instalados (NumPy en primer lugar) y `get_array_namespace(name)` devuelve `numpy`, `mlx.core` o `cupy` (`"numba"` es un backend de kernels compilados, no un espacio de nombres de arrays, y aquí lanza `ValueError`). `to_numpy(x)` trae cualquier array de vuelta al host, encaminando CuPy por `cp.asnumpy` porque CuPy bloquea la conversión implícita a memoria del host.

Los dos solucionadores que aceptan `backend=` utilizan el espacio de nombres de manera distinta:

- **`solve_trade_equilibrium(calib, ..., method="condensed", backend=)`**: el solucionador por eliminación en bloques / complemento de Schur mantiene las factorizaciones de Leontief, el paso de Newton sobre el bloque macro y todos los residuos en NumPy/SciPy float64, y delega al dispositivo dos reducciones por evaluación del residuo, la contracción de flujos bilaterales de intermedios $\sum_n a_{m i n}\, y_{i n}$ y la contracción de demanda final $\sum_n x^{c}_{m i n}\, \tilde p_{i n}$, copiando de vuelta los resultados $(M \times N)$. **Los arrays del dispositivo son siempre float64**: en Apple MLX se construyen bajo `mx.stream(mx.cpu)`, porque Metal no tiene float64 y unos residuos float32 hacen divergir el jacobiano por diferencias finitas en serie de este solucionador con los datos canónicos. El jacobiano macro se construye por diferencias finitas en serie, de modo que el residuo (y ambas reducciones) se ejecuta $B + 1$ veces por iteración de Newton más una vez por cada prueba de la búsqueda lineal. La palabra clave solo la consulta `method="condensed"`; los demás métodos (`"newton"`, `"sparse_lu"`, `"krylov"`, `"broyden"`, `"hybr"`, `"lm"`) la ignoran en silencio.

  El mecanismo de respaldo tiene tres capas y nunca es silencioso. Un backend que no está instalado, o cuyo espacio de nombres no se puede importar, produce un `RuntimeWarning` y la ruta NumPy. Una resolución acelerada que *lanza una excepción* produce `RuntimeWarning: Backend '<name>' failed during the condensed solve (...); falling back to 'numpy'` y se rehace en NumPy. Y una resolución acelerada que simplemente no converge produce `RuntimeWarning: Backend '<name>' did not converge (max residual ... > tol ...); falling back to 'numpy'` y *también* se rehace en NumPy, de modo que nunca se devuelve un resultado acelerado no convergido.
- **`AllenArkolakisModel.solve_equilibrium(backend=)`**: toda la aplicación contractiva amortiguada sobre $(w_i, L_i)$ (la matriz de kernel $\tau_{ji}^{-\theta}$, el acceso a mercados, el softmax de igualación de utilidades) se ejecuta en el dispositivo en `float32` para MLX y en `float64` para CuPy. Un dispositivo float32 solo puede certificar el punto fijo hasta su propio suelo, así que siempre que ese suelo sea más grueso que el `tol` solicitado —y en MLX siempre lo es, con suelos de $\max(\text{tol}, 10^{-6})$ en el residuo y $10^{-5}$ en la dispersión de utilidad— el bucle del dispositivo se trata como un arranque en caliente y la iteración **termina en el host en NumPy float64** con el `tol` solicitado. La tolerancia pedida se respeta, por tanto, y `converged` dice la verdad sea cual sea el backend; el Ejemplo 4.4 muestra a MLX y NumPy coincidiendo hasta $10^{-9}$ tras las mismas 27 iteraciones. Tras la convergencia, la población se renormaliza a $\bar L$, los salarios a media unitaria, y el índice de precios, los términos de acceso a mercados y los salarios reales se recalculan en NumPy float64, razón por la cual el residuo de conservación del trabajo es cero (salvo redondeo) en ambos backends. `solve_counterfactual` pasa `backend` tanto a la resolución base como a la contrafactual.

### 1.2 Selección de dispositivo y `DeviceInfo`

`puremacro.trade.gpu.select_compute_device(preferred)` asigna a un nombre un par `(backend, device)`, donde `backend` es `"torch"`, `"mlx"` o `"numpy"` y `device` es `"cuda"`, `"mps"`, `"gpu"` o `"cpu"`:

| `preferred` | Resultado | Si no está disponible |
|---|---|---|
| `None` / `"auto"` | el primero de: torch CUDA, torch MPS, MLX GPU, torch CPU, NumPy CPU | nunca falla |
| `"numpy"` | `("numpy", "cpu")`, el evaluador NumPy en serie | nunca falla |
| `"cpu"` | `("torch", "cpu")` cuando PyTorch está instalado, `("numpy", "cpu")` en caso contrario | nunca falla |
| `"cuda"`, `"cuda:N"` | ese dispositivo CUDA de PyTorch | `RuntimeError` (falta PyTorch, `torch.cuda.is_available()` es `False`, o el índice `N` está fuera de rango) |
| `"mps"` | `("torch", "mps")` | `RuntimeError` (falta PyTorch o `torch.backends.mps.is_available()` es `False`) |
| `"torch"` / `"pytorch"` | el mejor dispositivo de PyTorch: CUDA, luego MPS, luego CPU | `RuntimeError` si falta PyTorch |
| `"mlx"` / `"apple_mlx"` | `("mlx", "gpu")` | `RuntimeError` si falta `mlx` |
| `"gpu"` | la mejor GPU de cualquier backend: CUDA, luego MPS, luego MLX | `RuntimeError` si no hay ninguna de las tres |

Arriba están todas las cadenas aceptadas; cualquier otra lanza `ValueError: Unknown device/backend ...; expected one of (...)` en lugar de resolverse calladamente a `"auto"`. Dos consecuencias que conviene recordar: `"numpy"` es un nombre real, de modo que se puede pedir el evaluador en serie de forma explícita, y `"cpu"` ya no exige PyTorch —degrada a `("numpy", "cpu")`, que es lo que obtiene todo kernel Pyodide.

`detect_device(preferred)` envuelve la misma elección en una dataclass congelada:

| Campo de `DeviceInfo` | Significado |
|---|---|
| `backend` | `"torch"`, `"mlx"` o `"numpy"` |
| `device` | `"cuda"`, `"mps"`, `"gpu"` (MLX) o `"cpu"` |
| `device_name` | nombre del adaptador CUDA, `"Apple Metal (<cpu>)"`, `"Apple MLX (<cpu>)"` o el procesador del host |
| `total_memory_gb` | CUDA: `total_memory` de las propiedades del dispositivo; Apple Silicon: `sysctl hw.memsize` (la memoria unificada completa; `None` si `sysctl` no puede ejecutarse); `None` en CPU sin acelerador |
| `supports_float64` | `True` para CUDA y CPU, `False` para MPS, `False` para el stream de GPU de MLX |
| `is_uma` | `True` en Apple Silicon (memoria unificada: el host y la GPU comparten la misma memoria, por lo que no hay copia al dispositivo) |

El `repr` resume la situación de precisión, p. ej. `f32-only` para MPS y `f32-gpu/f64-cpu-stream` para MLX.

### 1.3 Precisión: float64 frente a float32

Una diferencia finita unilateral divide el error de redondeo del residuo por el paso $h = 10^{-4}$, de modo que un residuo float32 produce un error relativo $O(1)$ en el jacobiano y un paso de Newton divergente con datos reales. **La regla que sigue la capa de dispositivo es, por tanto: ensamblar el jacobiano en float64 allí donde el dispositivo pueda, y usar un dispositivo float32 solo cuando se pida explícitamente, y solo para la fase gruesa.**

- **CUDA y PyTorch en CPU** ejecutan en float64 de principio a fin; no ocurre nada especial.
- **MPS** (Apple Metal a través de PyTorch) carece de float64. Con `device="auto"`, el solucionador sustituye en silencio el evaluador float64 de **CPU** de PyTorch. Con `device="mps"` pedido explícitamente emite un `RuntimeWarning` («... only supports float32; finite-difference Jacobians are unusable in float32, so it is used for the coarse phase only and the Jacobian is polished in float64 on torch:cpu»), usa MPS mientras $\max_i |F_i| \ge 5 \times 10^6$ y solo en las primeras iteraciones, y después pule en float64.
- **MLX** conserva dos copias de cada constante: arrays float32 para el stream de GPU y arrays float64 creados bajo `mx.stream(mx.cpu)`, donde MLX ejecuta Accelerate/LAPACK (y donde `mx.exp`, que solo es exacto a nivel float32, se sustituye por un `e ** x` exacto). `BatchedJacobianEvaluator` **usa por defecto el stream de CPU**; al stream float32 de GPU solo se llega con `stream="gpu"` o por la fase gruesa de un dispositivo pedido explícitamente. `solve_trade_equilibrium_mlx` es una envoltura ligera de `solve_trade_equilibrium_gpu(device="mlx")` y sigue exactamente la misma política.
- **La capa de espacio de nombres** construye sus arrays de dispositivo en float64 en todos los backends —en MLX bajo `mx.stream(mx.cpu)`—, de modo que `backend="mlx"` en `solve_trade_equilibrium(method="condensed")` reproduce bit a bit la respuesta de NumPy en la calibración de juguete (Ejemplo 4.2). El solucionador espacial ejecuta su contracción en float32 en MLX pero termina en el host en float64 (sección 1.1), así que su respuesta también es float64.

Sea cual sea el dispositivo, `result.metadata["jacobian_evaluations"]` cuenta los jacobianos por `backend:device:dtype`, de modo que la ruta de precisión que realmente siguió una ejecución queda en el resultado y no en la memoria de quien lo ejecutó. En todos los solucionadores de comercio acelerados, el residuo de un solo punto `eval_macro_single`, los puntos de prueba de la búsqueda lineal y la comprobación final del residuo son NumPy float64 en el host, de modo que `max_residual` y `converged` son siempre magnitudes float64; solo las columnas del jacobiano de la fase gruesa pueden arrastrar redondeo float32, y la sección 4.3 lo mide: un jacobiano float32 de MPS difiere del float64 hasta en $10^{-3}$ en relación con su mayor entrada, mientras que el stream de CPU float64 de MLX y PyTorch en CPU reproducen el jacobiano NumPy en serie hasta $10^{-10}$.

### 1.4 El jacobiano por lotes

El sistema de Caliendo-Parro de `puremacro.trade` (véase la [sección 1 de la página de comercio](spatial_and_trade_ge.md)) se condensa sobre un vector macro. Con $N$ países, $S$ sectores y $M = SN$ pares país-sector, `BatchedJacobianEvaluator` trabaja sobre

$$x_m = \begin{bmatrix} \log w \\ T \\ X_N \end{bmatrix} \in \mathbb{R}^{3N-1} \quad\text{o}\quad x_m = \begin{bmatrix} \log r \\ \log w \\ T \\ X_N \end{bmatrix} \in \mathbb{R}^{4N-1},$$

donde $w$ son los salarios, $r$ las rentas del capital, $T$ la recaudación tributaria y $X_N$ las transferencias netas del exterior de los países $1,\dots,N-1$ (el último cierra la cuenta corriente mundial). La primera disposición impone la igualdad de las remuneraciones de los factores, $r_c = w_c$. Para la calibración ICIO de 77 países estas dimensiones son $3 \cdot 77 - 1 = 230$ y $4 \cdot 77 - 1 = 307$, de donde proceden los valores conocidos de `batch_size`: **el argumento es un interruptor de disposición (*layout*), no un tamaño de bloque.**

La disposición la elige ahora el argumento exclusivamente por nombre `factor_equivalence`, y `batch_size` se conserva por compatibilidad hacia atrás y se *valida* contra la calibración en lugar de depender de un número mágico. `factor_equivalence=None` (valor por defecto) selecciona la disposición reducida exactamente cuando es exacta para la calibración —es decir, cuando las participaciones del capital son uniformes entre los sectores de cada país— y la completa en caso contrario; `factor_equivalence=True` en una calibración donde no es exacta lanza `ValueError` en lugar de descartar calladamente la condición de vaciado del mercado de capital. Si se pasa `batch_size`, debe valer $3N-1$ o $4N-1$ para *su* $N$: en la calibración de juguete de 2 países que sigue, 5 o 7, y `batch_size=230` lanza `ValueError: batch_size=230 is inconsistent with a 2-country calibration: expected 5 (reduced layout, r == w) or 7 (full layout).` Pasar ambos argumentos con implicaciones contradictorias también lanza una excepción.

Dado $x_m$, los precios y los productos brutos se obtienen de dos sistemas de Leontief,

$$p = (I - B^\top)^{-1}\, v(x_m), \qquad y = (I - A)^{-1}\, d(x_m, p),$$

con $B^\top = (A \circ \tau)^\top / (1 - \text{tax})$ el operador de participaciones de costos con aranceles incluidos y $A$ los coeficientes insumo-producto. Ambas inversas se forman **una sola vez** (`(I - A)^{-1}` en la construcción, `(I - B^\top)^{-1}` en `update_tariffs`, es decir, una vez por esquema arancelario) y se copian al dispositivo. Un jacobiano por diferencias hacia adelante necesita $F$ en los $B$ puntos perturbados $x_m + h_j e_j$, con $h_j = \varepsilon$ para las entradas de precios de factores y $h_j = \varepsilon \max(|x_{m,j}|, 1)$ para las demás ($\varepsilon = 10^{-4}$). Apilando los puntos perturbados como filas de $X \in \mathbb{R}^{B \times B}$, las resoluciones de precios y productos para todo el lote se convierten en dos productos matriciales,

$$P = \left[(I - B^\top)^{-1} V^\top\right]^\top, \qquad Y = \left[(I - A)^{-1} D^\top\right]^\top, \qquad V, D \in \mathbb{R}^{B \times M},$$

y la contabilidad de flujos bilaterales se convierte en contracciones `einsum` por lotes (`"min,bin->bmi"`). Un único lanzamiento de kernel sustituye $B$ evaluaciones seriales del residuo, y

$$J_{:,j} = \frac{F(x_m + h_j e_j) - F(x_m)}{h_j}.$$

`ad_mode="forward"` sustituye las diferencias por diferenciación automática (DA) en modo directo (`torch.func.jacfwd` en PyTorch, `mx.jvp` columna a columna en MLX) y `ad_mode="vjp"` por modo inverso (`torch.func.jacrev`, `mx.vjp`); ambos eliminan la sensibilidad al tamaño de paso de las economías pequeñas. Ante cualquier fallo, el evaluador emite un `RuntimeWarning` y recurre a la ruta de diferencias finitas —el respaldo nunca es silencioso—, y en el backend `numpy` cualquiera de los dos modos de DA advierte y usa diferencias finitas, ya que no hay ninguna de las dos bibliotecas por la que diferenciar. Un vector macro cuya longitud no coincide con la disposición del evaluador lanza `ValueError`, igual que un `ad_mode` desconocido. El método `jvp(xm, v)` devuelve la derivada direccional $J v$ del mismo modo, con una diferencia central como último recurso.

### 1.5 Paso de Levenberg-Marquardt, equilibrado y búsqueda lineal

Cada iteración de `solve_trade_equilibrium_gpu` / `solve_trade_equilibrium_mlx` escala el jacobiano por ambos lados (equilibrado de Jacobi), $\tilde J = D_L J D_R$ con $D_R = \operatorname{diag}(1/\lVert J_{:,j}\rVert_2)$ y $D_L = \operatorname{diag}(1/\lVert (J D_R)_{i,:}\rVert_2)$, y resuelve las ecuaciones normales regularizadas

$$\left(\tilde J^\top \tilde J + \mu I\right) u = -\tilde J^\top D_L F(x_m), \qquad \Delta x_m = D_R\, u,$$

con `torch.linalg.solve` en float64 siempre que PyTorch sea importable (en el dispositivo CUDA cuando ese es el dispositivo de pulido, y en la CPU en caso contrario), luego `scipy.linalg.solve` y luego `scipy.linalg.lstsq`; se prueban en ese orden y gana la primera que tenga éxito. A continuación, el paso completo se reescala de modo que su bloque de precios de factores se mantenga dentro de $0.3$ en valor absoluto, y los logaritmos de los precios de factores se acotan al intervalo $[-5, 5]$. Una búsqueda lineal con retroceso sobre la función de mérito $\Phi(x) = \lVert D_L F(x) \rVert_2$ ensaya entonces $\alpha = 1, \tfrac12, \tfrac14, \dots$ (a lo sumo 12 puntos de prueba) y acepta el primer punto que cumpla $\max_i|F_i| \le \text{tol}$, $\Phi(x + \alpha \Delta) < \Phi(x)$ o $\max_i |F_i(x + \alpha\Delta)| < \max_i |F_i(x)|$: una regla de descenso simple en el patrón de retroceso de Armijo (1966), sin constante de descenso suficiente. La amortiguación sigue la actualización clásica de Marquardt: $\mu \leftarrow \max(0.3\mu, 10^{-8})$ tras un paso aceptado, $\mu \leftarrow \max(0.5\mu, 10^{-8})$ cuando solo la mejor prueba mejoró el mérito, y $\mu \leftarrow \min(5\mu, 10^{2})$ en caso contrario. La convergencia se declara cuando $\max_i |F_i| \le \text{tol}$ sobre el residuo **completo** (precios, productos, mercados de factores, transferencias, recaudación), que se reevalúa al final y se almacena en `residuals`.

### 1.6 Continuación por homotopía en el parámetro arancelario

Los choques arancelarios grandes y asimétricos pueden hacer fracasar un método de tipo Newton iniciado desde la línea base. `solve_homotopy_continuation` inserta el esquema objetivo en la trayectoria

$$\tau(\lambda) = (1-\lambda)\,\tau_0 + \lambda\,\tau_1, \qquad \tau^{fd}(\lambda) = (1-\lambda)\,\tau^{fd}_0 + \lambda\,\tau^{fd}_1, \qquad \lambda \in [0, 1],$$

(la misma interpolación se aplica a los vectores de aranceles nacionales) y la recorre con la continuación sin predictor y con arranque en caliente de Allgower y Georg (1990, cap. 1): se resuelve en $\lambda_0 = 0$ (o se acepta `x0`) y después se resuelve repetidamente en $\min(\lambda + \Delta\lambda, 1)$ partiendo del último estado convergido. El control del paso es adaptativo: un paso que convergió en a lo sumo 4 iteraciones multiplica el siguiente por $1.5$ (con tope en `max_step`), uno que necesitó al menos `max_iter_per_step - 5` lo multiplica por $0.75$ (con suelo en `min_step`), y un paso fallido se descarta y se reintenta con la mitad del tamaño. Si el paso cae por debajo de `min_step`, si la propia línea base no converge o si ningún paso llega a tener éxito, la función **lanza `RuntimeError`** (no devuelve un resultado parcial); el argumento `callback(lambda, result)`, invocado tras cada paso exitoso, es la vía para conservar los equilibrios intermedios.

---

## 2. Opciones metodológicas

| Opción | Capa de espacio de nombres (`backend=`) | `solve_trade_equilibrium_gpu` | `solve_trade_equilibrium_mlx` |
|---|---|---|---|
| **Nombres aceptados** | `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` advierte y recurre a NumPy —es un backend de kernels, no un espacio de nombres—; cualquier otra cadena lanza `ValueError`) | `device` en `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"`; `backend` en `"torch"`, `"mlx"`, `"numpy"` | ninguno: solo MLX, lanza `RuntimeError` sin él |
| **Precisión** | float64 en todos los backends (MLX en el stream de CPU) | jacobiano float64 allí donde el dispositivo lo permita; un dispositivo float32 solo si se pide explícitamente, y solo para la fase gruesa | la misma política, en el stream de CPU de MLX |
| **Jacobiano** | no aplica (el Newton sobre el bloque macro permanece en NumPy) | diferencias finitas por lotes, `ad_mode="forward"` (jacfwd) o `"vjp"` (jacrev) | diferencias finitas por lotes, `ad_mode="forward"` (jvp) o `"vjp"` (mx.vjp) |
| **Regla de parada** | comercio: `tol=2.5e-3` sobre $\max|F|$; espacial: el `tol` solicitado, terminado en el host cuando el suelo del dispositivo es más grueso | `tol=2.5e-3` sobre el residuo completo | `tol=2.5e-3` sobre el residuo completo |
| **Política ante fallos** | `RuntimeWarning`, luego NumPy — ante un backend ausente, ante una excepción *y* ante una resolución acelerada no convergida | las excepciones se propagan; los fallos de DA advierten y recurren a diferencias finitas | ídem |
| **Dónde se prueba** | calibración de juguete 2x2x3 y un modelo espacial de 5 regiones (`tests/test_trade_solver_accelerated.py`) | calibración ICIO de 45 sectores y 77 países (`tests/test_gpu_acceleration.py`), omitida cuando falta el acelerador o los datos privados | ídem |

La tolerancia de comercio por defecto es la misma $2.5 \times 10^{-3}$ para el solucionador NumPy y para los acelerados. Es laxa: en la calibración de juguete, dos métodos NumPy (`"newton"` y `"condensed"`) ya difieren en unos $10^{-5}$ en los salarios a esa tolerancia, lo que supera cualquier brecha entre backends medida en la sección 4.

---

## 3. Instalación de los aceleradores

Ninguno de los aceleradores forma parte de la instalación base, y la instalación base ejecuta todo lo de esta página por la ruta NumPy.

| Acelerador | Instalación | Plataformas | Proporciona |
|---|---|---|---|
| Apple MLX | `pip install "puremacro[backend]"` (el extra `[accel]` tiene el mismo contenido: `numba` más `mlx` en macOS) | solo macOS con Apple Silicon (marcador `sys_platform == 'darwin'`) | `backend="mlx"` en ambos solucionadores, `solve_trade_equilibrium_mlx`, `device="mlx"` en el motor GPU |
| NVIDIA CuPy | `pip install "puremacro[cuda]"` (`cupy-cuda12x`; elija el *wheel* que corresponda a su versión del CUDA Toolkit) | GPU NVIDIA con CUDA | `backend="cupy"` en ambos solucionadores |
| PyTorch | `pip install torch` (**no** existe un extra de puremacro para él; use el selector de pytorch.org para una compilación CUDA) | CUDA, Apple MPS o CPU | `solve_trade_equilibrium_gpu`, `solve_homotopy_continuation`, `device="cuda"/"mps"/"cpu"` |

`puremacro.trade.gpu` importa `torch` y `mlx` de forma **diferida**: `import puremacro.trade`, e incluso `from puremacro.trade.gpu import solve_trade_equilibrium_gpu`, dejan ambos fuera de `sys.modules` en una máquina donde están instalados, y tienen éxito en una máquina donde no lo están. Por tanto, todos los nombres de esta página existen en la instalación base; una llamada que realmente necesita un acelerador ausente lanza `RuntimeError` cuando lo pide por su nombre, mientras que `device=None` / `"auto"` se resuelve a `("numpy", "cpu")` y ejecuta el evaluador en serie. `puremacro.runtime.capabilities().backends` enumera los backends de *espacio de nombres* instalados (`numpy`, `numba`, `mlx`, `cupy`); PyTorch no forma parte de esa tupla ni siquiera cuando está instalado.

**Pyodide, iPad y el navegador.** Ni PyTorch, ni MLX ni CuPy pueden instalarse en un kernel Pyodide, de modo que allí `capabilities().backends == ("numpy",)` (véase [Ejecución en cualquier entorno](tablet.md)), `detect_device()` devuelve `numpy:cpu` por construcción, y `backend="mlx"` o `"cupy"` produce el `RuntimeWarning` y la respuesta NumPy. Un script escrito para la ruta NumPy se ejecuta sin cambios en una tableta.

---

## 4. Ejemplos prácticos ejecutables

Todos los ejemplos se ejecutan sin conexión. Fueron ejecutados en una estación de trabajo Apple Silicon con PyTorch 2.12.1 (MPS) y MLX 0.32.2 instalados y sin GPU NVIDIA; las líneas impresas reproducen esa ejecución. Cada ejemplo termina en bastante menos de dos segundos una vez importado `puremacro.trade` (véase la advertencia sobre el tiempo de importación en la sección 7).

### 4.1 ¿Qué hardware tengo?

```python
from puremacro._backend import available_backends
from puremacro.trade.gpu import detect_device, get_memory_usage, has_mlx, has_torch, select_compute_device

print("namespace backends:", available_backends())
print("torch installed:", has_torch(), "| mlx installed:", has_mlx())
print("auto selection   :", select_compute_device())
print(detect_device())
if has_mlx():
    print(detect_device("mlx"))
if has_torch():
    print(detect_device("cpu"))
try:
    select_compute_device("cuda")
except RuntimeError as exc:
    print("cuda:", exc)
print(get_memory_usage())
```

```text
namespace backends: ('numpy', 'numba', 'mlx')
torch installed: True | mlx installed: True
auto selection   : ('torch', 'mps')
<DeviceInfo: torch:mps (Apple Metal (arm), 36.0 GB, f32-only, UMA)>
<DeviceInfo: mlx:gpu (Apple MLX (arm), 36.0 GB, f32-gpu/f64-cpu-stream, UMA)>
<DeviceInfo: torch:cpu (arm, f64)>
cuda: CUDA requested but torch.cuda.is_available() is False.
{'backend': 'torch', 'device': 'mps', 'allocated_mb': 0.0, 'driver_mb': 1.453125, 'peak_mb': 0.0}
```

Con PyTorch instalado, `"auto"` prefiere MPS a MLX en un Mac, y el `DeviceInfo` de MPS informa `f32-only`. El diccionario de memoria siempre contiene `backend`, `device`, `allocated_mb` y `peak_mb`; MPS añade `driver_mb`, CUDA y MLX añaden `cache_mb`, y el respaldo NumPy informa el tamaño máximo del conjunto residente (RSS) del proceso obtenido de `resource.getrusage`.

### 4.2 Equilibrio de comercio en NumPy y MLX con la calibración de juguete

La calibración es la tabla de 2 países, 2 sectores y 3 componentes de demanda final de `tests/test_trade_solver_accelerated.py` (líneas 21-72). En la línea base, la tabla ya es un equilibrio (los solucionadores retornan tras cero iteraciones), por lo que el ejemplo impone un arancel del 25% sobre todo lo que el país 1 importa del país 0 para obligar al solucionador a iterar.

```python
import warnings

import numpy as np

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium


def toy_calibration():
    """Tabla de 2 países, 2 sectores y 3 componentes de demanda final (tests/test_trade_solver_accelerated.py)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((7, 10), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 15.0, 5.0, 2.0],
        [10.0, 25.0, 2.0, 8.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        data[i, 4:] = fd_row_sums[i] * np.array(shares)
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


calib = toy_calibration()
ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand

# Arancel del 25% sobre todo lo que el país 1 importa del país 0 (intermedios y demanda final)
tau = np.ones((ns, nc, ns, nc))
tau[:, 0, :, 1] = 1.25
tau_fd = np.ones((ns, nc, nfd, nc))
tau_fd[:, 0, :, 1] = 1.25

res_np = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="numpy")
res_mlx = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="mlx")
for name, res in [("numpy", res_np), ("mlx", res_mlx)]:
    print(f"{name:5s} converged={res.converged} iterations={res.iterations} "
          f"max_residual={res.max_residual:.3e} backend={res.metadata['backend']}")
print("wages (numpy):", res_np.w_sol.ravel())
print("max |dw| =", f"{np.max(np.abs(res_np.w_sol - res_mlx.w_sol)):.2e}",
      "| max |dp| =", f"{np.max(np.abs(res_np.p_sol - res_mlx.p_sol)):.2e}",
      "| max |dy| =", f"{np.max(np.abs(res_np.y_sol - res_mlx.y_sol)):.2e}")

# Un acelerador no instalado: RuntimeWarning y, después, la respuesta de NumPy
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res_cp = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="condensed", backend="cupy")
print(caught[0].category.__name__, "->", caught[0].message)
print("fallback identical to numpy:", np.array_equal(res_cp.w_sol, res_np.w_sol))
print(res_mlx.to_markdown())
```

```text
numpy converged=True iterations=2 max_residual=4.606e-04 backend=numpy
mlx   converged=True iterations=2 max_residual=4.606e-04 backend=mlx
wages (numpy): [0.86578283 0.96175774]
max |dw| = 0.00e+00 | max |dp| = 0.00e+00 | max |dy| = 0.00e+00
RuntimeWarning -> Backend 'cupy' is not available; falling back to 'numpy'.
fallback identical to numpy: True
|               Diagnostic |        Value |
|--------------------------|--------------|
|            Solver Status |    CONVERGED |
|               Iterations |            2 |
|         L1 Residual Norm | 9.512323e-04 |
|    Max Absolute Residual | 4.605624e-04 |
|         Mean Price Index |     0.904276 |
|       Total World Output |   5.5000e+02 |
|          Mean Wage Index |     0.913770 |
| Mean Capital Rental Rate |     0.913770 |
```

Ambos backends convergen en dos iteraciones a salarios, precios y productos *idénticos*: la capa de espacio de nombres ejecuta las dos contracciones delegadas en float64 en el stream de CPU de MLX, así que no queda redondeo float32 que perturbe la trayectoria de Newton. En un problema tan pequeño la ruta MLX es más lenta que NumPy, porque cada iteración envía dos tensores diminutos al dispositivo y de vuelta; la capa de espacio de nombres está pensada para los operadores de dimensión $M = 3465$ de la calibración ICIO.

### 4.3 El jacobiano por lotes sobre la misma calibración

Continuando desde el bloque anterior, el evaluador acepta directamente los tensores arancelarios 4-D y, con `batch_size` en `None`, elige la disposición reducida $[\log w; T; X_N]$ —exacta para esta calibración—, que tiene $B = 3 \cdot 2 - 1 = 5$ entradas. La entrada de MLX se pide como `device="mlx"`, de modo que el evaluador toma el stream de CPU float64; `stream="gpu"` sería la forma de pedir float32.

```python
from puremacro.trade.gpu import BatchedJacobianEvaluator

zeros = np.zeros(nc)                       # sin aranceles nacionales (uniformes)
x_m = np.concatenate([np.zeros(nc),        # log w  (r = w bajo la disposición B = 3*nc - 1)
                      calib.T.ravel(),     # T      (recaudación tributaria)
                      calib.invforT.ravel()[: nc - 1]])  # XN (transferencias netas de los países 1..nc-1)

ev_np = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device="cpu", backend="numpy")
J_np = ev_np.evaluate_batched_jacobian(x_m)
print("B =", len(x_m), "| J shape:", J_np.shape, "| max|J| =", f"{np.max(np.abs(J_np)):.2f}")

for backend, device in [("torch", "cpu"), ("torch", "mps"), ("mlx", "mlx")]:
    try:
        ev = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device=device, backend=backend)
    except RuntimeError as exc:            # acelerador ausente en esta máquina
        print(f"{backend}:{device} unavailable ({exc})")
        continue
    J = ev.evaluate_batched_jacobian(x_m)
    print(f"{backend}:{device:4s} dtype={ev.dtype}  max|J - J_numpy| = {np.max(np.abs(J - J_np)):.1e}")

ev_t = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device="cpu", backend="torch")
J_ad = ev_t.evaluate_batched_jacobian(x_m, ad_mode="forward")   # torch.func.jacfwd, sin tamaño de paso
print("forward-AD vs finite differences:", f"{np.max(np.abs(J_ad - J_np)):.1e}")
```

```text
B = 5 | J shape: (5, 5) | max|J| = 31.22
torch:cpu  dtype=torch.float64  max|J - J_numpy| = 1.4e-10
torch:mps  dtype=torch.float32  max|J - J_numpy| = 3.1e-02
mlx:mlx  dtype=None  max|J - J_numpy| = 2.8e-10
forward-AD vs finite differences: 1.7e-03
```

El jacobiano float64 de PyTorch en CPU reproduce el de NumPy en serie hasta $10^{-10}$, y MLX también, porque el evaluador usa por defecto el stream de CPU float64. El único dispositivo float32 que queda en la tabla, MPS, difiere en $3 \times 10^{-2}$ sobre entradas de tamaño 31 —$10^{-3}$ en términos relativos—, que es exactamente lo que hace una diferencia finita con el redondeo float32 dividido por $h = 10^{-4}$, y exactamente por lo que los solucionadores refinan en float64 (sección 1.3). La brecha de $1.7 \times 10^{-3}$ entre la DA en modo directo y las diferencias finitas es el error de truncamiento de las propias diferencias; el jacobiano por DA es el más preciso de los dos, y `ad_mode="vjp"` (modo inverso) lo reproduce. (`ev.dtype` solo se fija en la ruta PyTorch; MLX elige float32 o float64 según el stream.)

### 4.4 Equilibrio espacial en NumPy y MLX

```python
import numpy as np

from puremacro.spatial import AllenArkolakisModel

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
model = AllenArkolakisModel.from_coordinates(coords)

res_np = model.solve_equilibrium(backend="numpy")
res_mlx = model.solve_equilibrium(backend="mlx")
print("iterations: numpy", res_np.iterations, "| mlx", res_mlx.iterations)
print("population (numpy):", np.round(res_np.population, 6))
print("max |dw| =", f"{np.max(np.abs(res_np.wages - res_mlx.wages)):.1e}",
      "| max |dL| =", f"{np.max(np.abs(res_np.population - res_mlx.population)):.1e}",
      "| max |dP| =", f"{np.max(np.abs(res_np.price_index - res_mlx.price_index)):.1e}")
print("labor conservation:", res_np.labor_conservation_residual, res_mlx.labor_conservation_residual)

cf_np = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="numpy")
cf_mlx = model.solve_counterfactual(productivity_new=np.array([1.2, 1.0, 1.0, 1.0, 1.0]), backend="mlx")
print(f"welfare change: numpy {cf_np.welfare_pct:+.5f}% | mlx {cf_mlx.welfare_pct:+.5f}%")
```

```text
iterations: numpy 27 | mlx 27
population (numpy): [0.199763 0.199763 0.199763 0.199763 0.200947]
max |dw| = 5.4e-10 | max |dL| = 2.7e-10 | max |dP| = 2.7e-12
labor conservation: 0.0 0.0
welfare change: numpy +4.39312% | mlx +4.39312%
```

Ambos backends reportan las mismas 27 iteraciones y coinciden hasta $10^{-9}$, porque el bucle float32 de MLX cede el testigo al bucle float64 de NumPy en cuanto alcanza su propio suelo (sección 1.1): la ejecución se certifica contra el `tol=1e-8` solicitado, no contra el $10^{-6}$ del dispositivo, y `max_residual` vuelve en $7.3 \times 10^{-9}$ en MLX frente a $7.4 \times 10^{-9}$ en NumPy. El efecto sobre el bienestar de una ganancia de productividad del 20% en la primera región coincide hasta el quinto decimal. La población suma exactamente $\bar L$ en ambos backends porque la renormalización se realiza en NumPy.

### 4.5 Los solucionadores de dispositivo y la homotopía sobre la calibración de juguete

`solve_trade_equilibrium_gpu` y `solve_homotopy_continuation` están hechos para la calibración ICIO, pero nada impide ejecutarlos sobre la tabla 2x2x3 de 4.2 —y hacerlo es la forma más barata de ver los valores por defecto de los aranceles, la resolución de la disposición y la política de dispositivo en el resultado. Continuando desde el bloque de 4.2:

```python
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu

res_np = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="condensed")
res_gpu = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd)   # tauf/tauf_fd por defecto 0
for name, r in [("condensed", res_np), ("gpu", res_gpu)]:
    print(f"{name:10s} converged={r.converged} iterations={r.iterations} "
          f"max_residual={r.max_residual:.3e} w={np.round(r.w_sol.ravel(), 6)}")

meta = res_gpu.metadata
print("device used   :", f"{meta['backend']}:{meta['device']}",
      "| requested:", f"{meta['requested_backend']}:{meta['requested_device']}")
print("layout        :", "B =", meta["batch_size"], "factor_equivalence =", meta["factor_equivalence"])
print("jacobians     :", dict(meta["jacobian_evaluations"]))

steps = []
hom = solve_homotopy_continuation(calib, target_tau=tau, target_tau_fd=tau_fd,
                                  initial_step=0.25, max_step=0.5,
                                  callback=lambda lam, r: steps.append((round(lam, 3), r.iterations)))
print("homotopy steps:", steps)
print(f"homotopy       converged={hom.converged} max_residual={hom.max_residual:.3e} "
      f"w={np.round(hom.w_sol.ravel(), 6)}")
```

```text
condensed  converged=True iterations=2 max_residual=4.606e-04 w=[0.865783 0.961758]
gpu        converged=True iterations=3 max_residual=4.474e-04 w=[0.865783 0.961758]
device used   : torch:cpu | requested: torch:mps
layout        : B = 5 factor_equivalence = True
jacobians     : {'torch:cpu:float64': 2}
homotopy steps: [(0.25, 3), (0.625, 3), (1.0, 3)]
homotopy       converged=True max_residual=4.795e-06 w=[0.865775 0.961769]
```

El solucionador de dispositivo alcanza los mismos salarios que la referencia NumPy hasta el sexto decimal sin más argumentos arancelarios que `tau` y `tau_fd`, porque `tauf` y `tauf_fd` toman por defecto vectores de ceros exactamente igual que en `solve_trade_equilibrium`. Los metadatos dicen qué ocurrió en realidad: `"auto"` eligió MPS, la política float64 de la sección 1.3 trasladó en silencio el jacobiano a `torch:cpu`, se eligió la disposición reducida porque aquí es exacta ($B = 5$) y ambos jacobianos fueron float64. La homotopía dio tres pasos —$0.25$, luego $0.625$ tras el crecimiento de $1.5\times$ de un paso que convergió en a lo sumo 4 iteraciones, y luego $1.0$— y los arranques en caliente la dejan un orden de magnitud por debajo del residuo de la resolución directa.

### 4.6 Forma de la llamada de los solucionadores completos PyTorch / MLX (calibración ICIO de 45 sectores)

`solve_trade_equilibrium_gpu`, `solve_trade_equilibrium_mlx` y `solve_homotopy_continuation` están diseñados para la calibración ICIO completa de la OCDE, con 45 sectores y 77 países, y se prueban contra ella; sus operadores de Leontief de dimensión $M = 3465$ son los que el GEMM por lotes está pensado para amortizar. Ese conjunto de datos no se distribuye con el paquete, por lo que aquí solo se muestra la forma de las llamadas, no un ejemplo ejecutado:

```text
from puremacro.trade import build_tariff_matrices, calibrate_trade_model
from puremacro.trade.data import load_raw_45sector_icio
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu, solve_trade_equilibrium_mlx

raw   = load_raw_45sector_icio()                       # requiere data_2020_SML.csv (IO_RAW45_PATH)
calib = calibrate_trade_model(raw, ns=45, nc=77, nfd=3)
tau, tau_fd, tauf, tauf_fd = build_tariff_matrices("t10", calib)   # (ns,nc,ns,nc), (ns,nc,nfd,nc), (1,nc), (1,nc)

res = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd, tauf=tauf, tauf_fd=tauf_fd,
                                  device="auto", tol=2.5e-3, ad_mode="finite_diff")
res = solve_trade_equilibrium_mlx(calib, tau=tau, tau_fd=tau_fd, tauf=tauf, tauf_fd=tauf_fd)
res = solve_homotopy_continuation(calib, target_tau=tau, target_tau_fd=tau_fd,
                                  target_tauf=tauf, target_tauf_fd=tauf_fd,
                                  initial_step=0.1, max_step=0.5, min_step=1e-4,
                                  callback=lambda lam, r: print(lam, r.iterations, r.max_residual))
```

Los tres devuelven el mismo `TradeEquilibriumResult` que `solve_trade_equilibrium`, con `metadata["method"]` igual a `"gpu_accelerated"` o `"mlx_accelerated"`. En esta calibración $N = 77$, de modo que `batch_size=230` y `batch_size=307` son los valores que acepta el validador, y dejar `batch_size=None` elige entre ellos según si $r = w$ es exacto.

---

## 5. Especificación completa de la API

```text
# puremacro._backend
SUPPORTED = ("numpy", "numba", "mlx", "cupy")
backend_available(name: str) -> bool                     # ValueError para nombres desconocidos
available_backends() -> tuple[str, ...]                  # instalados, numpy en primer lugar
get_array_namespace(name: str)                           # numpy | mlx.core | cupy; ImportError con la sugerencia de pip
to_numpy(x) -> np.ndarray

# backend= en los solucionadores de los modelos
solve_trade_equilibrium(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                        method="newton", tol=2.5e-3, max_iter=50, replicate_matlab_precedence=True,
                        base_result=None, *, backend="numpy", ...) -> TradeEquilibriumResult
AllenArkolakisModel.solve_equilibrium(tol=1e-8, max_iter=2500, damping=0.35,
                                      backend="numpy") -> AllenArkolakisResult
AllenArkolakisModel.solve_counterfactual(trade_costs_new=None, productivity_new=None, amenity_new=None,
                                         tol=1e-8, max_iter=2500, damping=0.35,
                                         backend="numpy") -> AllenArkolakisResult

# puremacro.trade.gpu.backend
DeviceInfo(backend: Literal["torch", "mlx", "numpy"], device: str, device_name: str,
           total_memory_gb: float | None = None, supports_float64: bool = True, is_uma: bool = False)
has_torch() -> bool
has_mlx() -> bool
select_compute_device(preferred: str | None = None) -> tuple[str, str]
detect_device(preferred: str | None = None) -> DeviceInfo
device_context(device: str | None = None, backend: str | None = None)   # gestor de contexto -> (backend, device)
get_memory_usage(device: str | None = None, backend: str | None = None) -> dict[str, float | str]
reset_peak_memory(device: str | None = None, backend: str | None = None) -> None
to_tensor(array, device: str | None = None, dtype=None, backend: str = "torch")
to_numpy(tensor) -> np.ndarray

# puremacro.trade.gpu.batched_jacobian
BatchedJacobianEvaluator(calib: TradeCalibrationResult, tau_a, taufd_a, tauf_vec, tauf_fd_vec,
                         device: str | None = None, backend: str | None = None,
                         batch_size: int | None = None, factor_equivalence: bool | None = None,
                         replicate_matlab_precedence: bool = True)
    .update_tariffs(tau_a, taufd_a, tauf_vec, tauf_fd_vec) -> None
    .eval_macro_single(xm, p_cached=None, compute_full=False)
        -> (f_macro, f_full | None, p_vec, p, y_vec)
    .evaluate_batched_jacobian(xm, f_base=None, eps_fd=1e-4,
                               ad_mode="finite_diff" | "forward" | "vjp", stream=None) -> np.ndarray (B, B)
    .jvp(xm, v) -> np.ndarray
    .eval_macro_batched_torch(X_batch) / .eval_macro_batched_mlx(X_batch, stream=None)

# puremacro.trade.gpu.solver_gpu
solve_trade_equilibrium_gpu(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            device=None, backend=None, batch_size=None, max_iter=50, tol=2.5e-3,
                            damping=1e-4, ad_mode="finite_diff", replicate_matlab_precedence=True,
                            base_result=None, verbose=False, *,
                            factor_equivalence=None) -> TradeEquilibriumResult

# puremacro.trade.gpu.mlx_solver
solve_trade_equilibrium_mlx(calib, tau=None, tau_fd=None, tauf=None, tauf_fd=None, x0=None,
                            batch_size=None, max_iter=50, tol=2.5e-3, damping=1e-4,
                            ad_mode="finite_diff", replicate_matlab_precedence=True,
                            base_result=None, verbose=False, *,
                            factor_equivalence=None) -> TradeEquilibriumResult

# puremacro.trade.gpu.homotopy
solve_homotopy_continuation(calib, target_tau, target_tau_fd, target_tauf=None, target_tauf_fd=None,
                            base_tau=None, base_tau_fd=None, base_tauf=None, base_tauf_fd=None, x0=None,
                            device=None, backend=None, batch_size=None,
                            initial_step=0.1, min_step=1e-4, max_step=0.5, tol=2.5e-3, damping=1e-4,
                            max_iter_per_step=30, replicate_matlab_precedence=True, base_result=None,
                            verbose=False, callback=None, *,
                            factor_equivalence=None) -> TradeEquilibriumResult
```

`puremacro.trade` reexporta `DeviceInfo`, `detect_device`, `select_compute_device`, `device_context`, `get_memory_usage`, `reset_peak_memory`, `BatchedJacobianEvaluator`, `solve_trade_equilibrium_gpu`, `solve_trade_equilibrium_mlx` y `solve_homotopy_continuation`; `has_torch`, `has_mlx`, `to_tensor` y `to_numpy` se importan desde `puremacro.trade.gpu`.

### Parámetros comunes a los solucionadores acelerados

- `calib`: un `TradeCalibrationResult` de `calibrate_trade_model`.
- `tau`, `tau_fd`: multiplicadores arancelarios efectivos ($1 + $ tasa), bien 4-D `(ns, nc, ns, nc)` / `(ns, nc, nfd, nc)`, bien en la disposición interna `(M, ns, nc)` / `(M, nfd, nc)`; `None` significa todo unos.
- `tauf`, `tauf_fd`: *tasas* arancelarias nacionales uniformes de longitud `nc`, aplicadas a la factura de importaciones en la ecuación de recaudación. `None` significa todo ceros, exactamente igual que en `solve_trade_equilibrium`, de modo que la llamada por defecto resuelve la misma línea base que el solucionador NumPy.
- `x0`: `None` (partir de $\log w = 0$, la $T$ calibrada y las transferencias netas), el vector de estado completo de longitud $2M + 4N - 1$, o un vector macro de cualquiera de las dos disposiciones, $3N - 1$ o $4N - 1$, para *su* $N$; un vector reducido que siembra la disposición completa fija $r = w$. Cualquier otra longitud lanza `ValueError` nombrando las tres que espera.
- `device` / `backend`: los valores explícitos se conservan tal cual; `select_compute_device` completa el que falte de los dos, guiado por `device` si se dio y por `backend` en caso contrario. El par que acaba evaluando el jacobiano puede diferir del solicitado (la política float64 de la sección 1.3); `metadata["device"]` / `["backend"]` registran el par efectivo y `metadata["requested_device"]` / `["requested_backend"]` el solicitado. `solve_trade_equilibrium_mlx` no tiene ninguno de los dos argumentos.
- `batch_size` / `factor_equivalence`: el interruptor de disposición de la sección 1.4. Prefiera `factor_equivalence` (`None` = elegir según la exactitud, `True` = reducida $[\log w; T; X_N]$, `False` = completa $[\log r; \log w; T; X_N]$); `batch_size` es la grafía antigua y debe valer $3N-1$ o $4N-1$ para la calibración.
- `tol`: tolerancia en norma infinito sobre el residuo; `max_iter`: iteraciones de Levenberg-Marquardt (`max_iter_per_step` por paso de continuación).
- `damping`: $\mu$ inicial.
- `ad_mode`: `"finite_diff"` (diferencias por lotes, por defecto), `"forward"` (DA en modo directo: `torch.func.jacfwd` / `mx.jvp`) o `"vjp"` (modo inverso: `torch.func.jacrev` / `mx.vjp`). Un modo de DA que falle en el dispositivo advierte y recurre a diferencias finitas; un valor desconocido lanza `ValueError`.
- `replicate_matlab_precedence`: conservar la precedencia de operadores de la implementación MATLAB de referencia (`ff_equi.m`) en el término de costo unitario del valor agregado, $w / (1-\alpha)^{1-\alpha}$; `False` utiliza $(w / (1-\alpha))^{1-\alpha}$.
- `base_result`: equilibrio base que utiliza el posprocesamiento para los índices de IPC y de términos de intercambio.
- `verbose`: imprime por iteración `max_res`, el paso y $\mu$ (y, en la homotopía, el registro de pasos).
- `callback` (solo homotopía): `callback(lambda_value, result)` tras cada paso exitoso.

---

## 6. Interfaz de resultados

### `TradeEquilibriumResult` (solucionadores acelerados)

Los solucionadores acelerados devuelven la misma dataclass congelada que `solve_trade_equilibrium`, posprocesada por `postprocess_trade_equilibrium`, de modo que `x_sol`, `p_sol`, `y_sol`, `r_sol`, `w_sol`, `T_sol`, `XN_sol`, los tensores de flujos bilaterales, `gdp`, `imports`, `exports`, `tariffs`, `cpi` y `terms_of_trade` se documentan una sola vez en la [página de comercio](spatial_and_trade_ge.md). Los campos que los solucionadores de esta página fijan por sí mismos son:

- `converged`, `iterations`, `max_residual`, `diff` / `residual_norm` (norma L1) y `residuals`: evaluados sobre el residuo del sistema completo en float64.
- `metadata`: `method` (`"gpu_accelerated"`, `"mlx_accelerated"` o el nombre del método NumPy con el `backend` solicitado), el `device` y el `backend` efectivos junto con `requested_device` y `requested_backend`, `factor_equivalence`, `batch_size`, `jacobian_evaluations` (un recuento de jacobianos por `backend:device:dtype`), `ad_mode`, `tol`, `max_iter`, `solve_duration_seconds` y `memory_usage` (el diccionario de `get_memory_usage`). En el solucionador NumPy, `metadata["backend"]` refleja el backend *solicitado* incluso tras recurrir a NumPy.
- `summary(detailed=False) -> pd.DataFrame`: la tabla de diagnóstico del solucionador impresa en 4.2; `detailed=True` devuelve la tabla por país (PIB, comercio, aranceles, IPC, términos de intercambio).
- `to_frame(detailed=False)`, `to_markdown(detailed=False)`, `to_latex(detailed=False)`, `to_typst(detailed=False)`: la misma tabla en cada formato.
- `TradeEquilibriumResult` no tiene método `plot`; las figuras de escenarios provienen de `puremacro.trade.plot` (`plot_country_impacts`, `plot_tariff_escalation_curve`, ...) aplicadas a un `ScenarioBatchResult`.

### `AllenArkolakisResult` (solucionador espacial)

No cambia con el backend: `wages`, `population`, `price_index`, `real_wages`, `welfare`, `consumer_market_access`, `firm_market_access`, `trade_shares` y, para los contrafactuales, `w_hat`, `L_hat`, `welfare_hat`, `welfare_pct`; diagnósticos `converged`, `iterations`, `max_residual`, `labor_conservation_residual`, `spatial_utility_variance`; presentación `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` y `plot(kind="spatial")`. Nada en el resultado registra qué backend lo produjo.

### `DeviceInfo` y el diccionario de memoria

`DeviceInfo` es la dataclass congelada de la sección 1.2. `get_memory_usage()` devuelve `{"backend", "device", "allocated_mb", "peak_mb"}` más `cache_mb` (CUDA, MLX) o `driver_mb` (MPS); `reset_peak_memory()` reinicia los contadores de pico (CUDA, MLX) y vacía las cachés del dispositivo (CUDA, MPS, MLX), y `device_context(...)` lo invoca al salir y produce el par `(backend, device)` sin contrastar ambos, devolviendo las cadenas que se pasaron (`device_context("mlx")` produce `("mlx", "mlx")`; `device_context(backend="mlx")` por sí solo produce `("mlx", "mps")` en un Mac donde `"auto"` elige MPS).

---

## 7. Advertencias y limitaciones

- **`backend=` en `solve_trade_equilibrium` solo actúa con `method="condensed"`.** El `method="newton"` por defecto y los demás métodos lo ignoran sin advertencia, y `metadata["backend"]` registra el nombre solicitado en todos los casos, de modo que en esos métodos es una etiqueta y no un hecho. Los solucionadores de dispositivo acelerados, en cambio, registran el dispositivo y el backend efectivos junto al par solicitado.
- **Dos capas, dos vocabularios.** La capa de espacio de nombres acepta `"numpy"`, `"mlx"`, `"cupy"` —`"numba"` advierte y recurre a NumPy, y cualquier otra cosa, `"torch"` incluida, lanza `ValueError: Unknown backend`—. La capa de dispositivo acepta `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"` y lanza `ValueError` ante cualquier otra cosa. Los dos vocabularios no se solapan limpiamente: `"cupy"` no significa nada para la capa de dispositivo y `"mps"` no significa nada para la de espacio de nombres.
- **Precisión.** MPS carece de float64 y el stream Metal de MLX es float32, así que un dispositivo float32 pedido explícitamente se usa solo para la fase gruesa y lo advierte cuando ocurre. No espere ninguna diferencia entre backends en las rutas float64 (la sección 4 mide $0$ en la capa de espacio de nombres y $10^{-10}$ en el jacobiano) y en torno a $10^{-3}$ en términos relativos en un jacobiano float32 de MPS.
- **Tolerancias.** `tol=2.5e-3`, tanto en el solucionador de comercio NumPy como en los acelerados, es laxa; ajústela explícitamente para trabajos de paridad.
- **Costo de importar el módulo.** `import puremacro.trade` ya no arrastra PyTorch ni MLX —ambos se importan de forma perezosa, en la primera llamada que los necesita—, de modo que la importación cuesta alrededor de 1.3 s en la estación de trabajo utilizada aquí y tiene éxito estén o no instalados los aceleradores. La primera llamada acelerada paga la importación del framework.
- **Rutas no verificadas.** La rama CuPy de ambos solucionadores de la capa de espacio de nombres y la rama CUDA de la capa de dispositivo no se ejecutaron al redactar esta página (sin hardware NVIDIA); están cubiertas por la lógica de respaldo y por pruebas que se omiten cuando falta el acelerador o los datos ICIO privados.
- **Sin aceleración en problemas pequeños.** Todas las rutas de dispositivo añaden latencia de transferencia y de lanzamiento; la calibración de juguete y el modelo de 5 regiones se ejecutan más lentamente en MLX que en NumPy. [Pruebas comparativas de rendimiento](benchmarks.md) publica actualmente solo tiempos de CPU.
- **Sin conexión y determinista.** Nada de lo aquí descrito toca la red; las rutas aceleradas son tan deterministas como lo sean las reducciones del dispositivo, y no más, y la ruta NumPy es la referencia para las afirmaciones de reproducibilidad.

---

## Referencias

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Allgower, E. L., & Georg, K. (1990). *Numerical Continuation Methods: An Introduction*. Springer Series in Computational Mathematics 13. Berlin: Springer.
- Armijo, L. (1966). "Minimization of functions having Lipschitz continuous first partial derivatives." *Pacific Journal of Mathematics*, 16(1), 1–3.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares." *Quarterly of Applied Mathematics*, 2(2), 164–168.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441.
