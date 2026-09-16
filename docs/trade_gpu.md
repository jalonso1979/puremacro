> 🇬🇧 English · 🇪🇸 [Español](es/trade_gpu.md)

# Hardware Backends: GPU, Apple MLX & Batched Jacobians for Trade and Spatial GE

`puremacro._backend` and `puremacro.trade.gpu` add optional hardware acceleration to the general equilibrium solvers documented in [Quantitative Spatial Economics & Gravity Trade GE](spatial_and_trade_ge.md). Two layers are involved:

1. **The array-namespace layer** (`puremacro._backend`): `numpy`, `mlx` (Apple Silicon) and `cupy` (NVIDIA) share one NumPy-like array API. `solve_trade_equilibrium(..., backend=)` and `AllenArkolakisModel.solve_equilibrium(..., backend=)` / `solve_counterfactual(..., backend=)` route selected tensor reductions through it, with a `RuntimeWarning` and a NumPy fallback whenever the requested backend is missing or fails.
2. **The device layer** (`puremacro.trade.gpu`): a PyTorch (CUDA / Apple MPS / CPU) and native Apple MLX engine for the multi-country, multi-sector Caliendo-Parro (2015) computable general equilibrium model, built around pre-inverted Leontief operators, a batched Jacobian evaluated as one multi-right-hand-side GEMM, a Levenberg-Marquardt (1944, 1963) step with a backtracking line search, and adaptive homotopy continuation (Allgower & Georg, 1990) in the tariff parameter.

Both layers are optional. Nothing on this page is required to run the models: the NumPy path remains the reference implementation and the correctness oracle, and it is the only path available in Pyodide (iPad, browser). Every accelerated result on this page is compared against it.

---

## 1. Algorithmic Framework

### 1.1 Which backend runs what

`puremacro._backend.SUPPORTED` is `("numpy", "numba", "mlx", "cupy")`. `backend_available(name)` checks `importlib.util.find_spec`, `available_backends()` lists the installed ones (NumPy first), and `get_array_namespace(name)` returns `numpy`, `mlx.core` or `cupy` (`"numba"` is a compiled-kernel backend, not an array namespace, and raises `ValueError` here). `to_numpy(x)` brings any array home, routing CuPy through `cp.asnumpy` because CuPy blocks implicit host conversion.

The two solvers that accept `backend=` use the namespace differently:

- **`solve_trade_equilibrium(calib, ..., method="condensed", backend=)`**: the block-elimination / Schur-complement solver keeps the Leontief factorizations, the Newton step on the macro block and every residual in NumPy/SciPy float64, and offloads two reductions per residual evaluation, the bilateral intermediate-flow contraction $\sum_n a_{m i n}\, y_{i n}$ and the final-demand contraction $\sum_n x^{c}_{m i n}\, \tilde p_{i n}$, to the device and copies the $(M \times N)$ results back. **The device arrays are always float64**: on Apple MLX they are built under `mx.stream(mx.cpu)`, because Metal has no float64 and float32 residuals make the serial finite-difference Jacobian of this solver diverge on the canonical data. The macro Jacobian is built by serial finite differences, so the residual (and both reductions) runs $B + 1$ times per Newton iteration plus once per line-search trial. The keyword is only consulted by `method="condensed"`; the other methods (`"newton"`, `"sparse_lu"`, `"krylov"`, `"broyden"`, `"hybr"`, `"lm"`) ignore it silently.

  The fallback is three-layered and never silent. A backend that is not installed, or whose namespace fails to import, produces a `RuntimeWarning` and the NumPy path. An accelerated solve that *raises* produces `RuntimeWarning: Backend '<name>' failed during the condensed solve (...); falling back to 'numpy'` and is redone in NumPy. An accelerated solve that merely fails to converge produces `RuntimeWarning: Backend '<name>' did not converge (max residual ... > tol ...); falling back to 'numpy'` and is *also* redone in NumPy, so a non-converged accelerated result is never returned.
- **`AllenArkolakisModel.solve_equilibrium(backend=)`**: the whole damped contraction mapping on $(w_i, L_i)$ (kernel matrix $\tau_{ji}^{-\theta}$, market access, the utility-equalization softmax) runs on the device in `float32` for MLX and `float64` for CuPy. A float32 device can only certify the fixed point to its own floor, so whenever that floor is coarser than the `tol` you asked for — which it always is on MLX, where the device floors are $\max(\text{tol}, 10^{-6})$ on the residual and $10^{-5}$ on the utility dispersion — the device loop is treated as a warm start and the iteration is **finished on the host in NumPy float64** at the requested `tol`. The requested tolerance is therefore honored and `converged` is truthful whatever the backend; Example 4.4 shows MLX and NumPy agreeing to $10^{-9}$ after the same 27 iterations. After convergence the population is renormalized to $\bar L$, the wages to unit mean, and the price index, market-access terms and real wages are recomputed in NumPy float64, which is why the labor-conservation residual is zero (to rounding) on both backends. `solve_counterfactual` passes `backend` to both the baseline and the counterfactual solve.

### 1.2 Device selection and `DeviceInfo`

`puremacro.trade.gpu.select_compute_device(preferred)` maps a name to a `(backend, device)` pair, where `backend` is `"torch"`, `"mlx"` or `"numpy"` and `device` is `"cuda"`, `"mps"`, `"gpu"` or `"cpu"`:

| `preferred` | Result | If unavailable |
|---|---|---|
| `None` / `"auto"` | first of: torch CUDA, torch MPS, MLX GPU, torch CPU, NumPy CPU | never fails |
| `"numpy"` | `("numpy", "cpu")`, the serial NumPy evaluator | never fails |
| `"cpu"` | `("torch", "cpu")` when PyTorch is installed, `("numpy", "cpu")` otherwise | never fails |
| `"cuda"`, `"cuda:N"` | that PyTorch CUDA device | `RuntimeError` (PyTorch missing, `torch.cuda.is_available()` is `False`, or index `N` out of range) |
| `"mps"` | `("torch", "mps")` | `RuntimeError` (PyTorch missing or `torch.backends.mps.is_available()` is `False`) |
| `"torch"` / `"pytorch"` | best PyTorch device: CUDA, then MPS, then CPU | `RuntimeError` if PyTorch is missing |
| `"mlx"` / `"apple_mlx"` | `("mlx", "gpu")` | `RuntimeError` if `mlx` is missing |
| `"gpu"` | best GPU of any backend: CUDA, then MPS, then MLX | `RuntimeError` if none of the three is available |

Every accepted string is listed above; anything else raises `ValueError: Unknown device/backend ...; expected one of (...)` rather than quietly resolving to `"auto"`. Two consequences worth keeping in mind: `"numpy"` is a real name, so you can ask for the serial evaluator explicitly, and `"cpu"` no longer demands PyTorch — it degrades to `("numpy", "cpu")`, which is what every Pyodide kernel gets.

`detect_device(preferred)` wraps the same choice in a frozen dataclass:

| `DeviceInfo` field | Meaning |
|---|---|
| `backend` | `"torch"`, `"mlx"` or `"numpy"` |
| `device` | `"cuda"`, `"mps"`, `"gpu"` (MLX) or `"cpu"` |
| `device_name` | CUDA adapter name, `"Apple Metal (<cpu>)"`, `"Apple MLX (<cpu>)"` or the host processor |
| `total_memory_gb` | CUDA: `total_memory` of the device properties; Apple Silicon: `sysctl hw.memsize` (the unified pool; `None` if `sysctl` cannot be run); `None` on plain CPU |
| `supports_float64` | `True` for CUDA and CPU, `False` for MPS, `False` for the MLX GPU stream |
| `is_uma` | `True` on Apple Silicon (unified memory: host and GPU share one pool, so there is no device copy) |

The `repr` summarizes the precision situation, e.g. `f32-only` for MPS and `f32-gpu/f64-cpu-stream` for MLX.

### 1.3 Precision: float64 versus float32

A one-sided finite difference divides the rounding error of the residual by the step $h = 10^{-4}$, so a float32 residual produces an $O(1)$ relative Jacobian error and a diverging Newton step on real data. **The rule the device layer follows is therefore: assemble the Jacobian in float64 wherever the device can, and use a float32 device only when you ask for one explicitly, and only for the coarse phase.**

- **CUDA and PyTorch CPU** execute in float64 throughout; nothing special happens.
- **MPS** (Apple Metal through PyTorch) has no float64. With `device="auto"` the solver silently substitutes the float64 PyTorch **CPU** evaluator. With `device="mps"` requested explicitly it emits a `RuntimeWarning` ("... only supports float32; finite-difference Jacobians are unusable in float32, so it is used for the coarse phase only and the Jacobian is polished in float64 on torch:cpu"), uses MPS while $\max_i |F_i| \ge 5 \times 10^6$ and only in the first iterations, and polishes in float64 afterwards.
- **MLX** keeps two copies of every constant: float32 arrays for the GPU stream and float64 arrays created under `mx.stream(mx.cpu)`, where MLX runs Accelerate/LAPACK (and where `mx.exp`, which is only float32-accurate, is replaced by an exact `e ** x`). `BatchedJacobianEvaluator` **defaults to the CPU stream**; the float32 GPU stream is reached only through `stream="gpu"` or the explicit-device coarse phase. `solve_trade_equilibrium_mlx` is a thin wrapper over `solve_trade_equilibrium_gpu(device="mlx")` and follows exactly the same policy.
- **The namespace layer** builds its device arrays in float64 on every backend — on MLX under `mx.stream(mx.cpu)` — so `backend="mlx"` on `solve_trade_equilibrium(method="condensed")` reproduces the NumPy answer bit for bit on the toy calibration (Example 4.2). The spatial solver runs its contraction in float32 on MLX but finishes on the host in float64 (Section 1.1), so its answer is float64 too.

Whichever device is used, `result.metadata["jacobian_evaluations"]` counts the Jacobians by `backend:device:dtype`, so the precision path a run actually took is on the result rather than in your memory. In every accelerated trade solver the single-point residual `eval_macro_single`, the line-search trial points and the final residual check are NumPy float64 on the host, so `max_residual` and `converged` are always float64 quantities; only the coarse-phase Jacobian columns can carry float32 rounding, and Section 4.3 measures it: a float32 MPS Jacobian differs from the float64 one by up to $10^{-3}$ relative to its largest entry, while the float64 MLX CPU stream and PyTorch CPU reproduce the serial NumPy Jacobian to $10^{-10}$.

### 1.4 The batched Jacobian

The Caliendo-Parro system of `puremacro.trade` (see [section 1 of the trade page](spatial_and_trade_ge.md)) is condensed onto a macro vector. With $N$ countries, $S$ sectors and $M = SN$ country-sector pairs, `BatchedJacobianEvaluator` works on

$$x_m = \begin{bmatrix} \log w \\ T \\ X_N \end{bmatrix} \in \mathbb{R}^{3N-1} \quad\text{or}\quad x_m = \begin{bmatrix} \log r \\ \log w \\ T \\ X_N \end{bmatrix} \in \mathbb{R}^{4N-1},$$

where $w$ are wages, $r$ capital rentals, $T$ tax revenues and $X_N$ the net foreign transfers of countries $1,\dots,N-1$ (the last one closes the world current account). The first layout imposes the factor-return equivalence $r_c = w_c$. For the 77-country ICIO calibration these dimensions are $3 \cdot 77 - 1 = 230$ and $4 \cdot 77 - 1 = 307$, which is where the familiar `batch_size` values come from: **the argument is a layout switch, not a chunk size.**

The layout is now chosen by the keyword-only `factor_equivalence`, and `batch_size` is kept for backward compatibility and *validated* against the calibration rather than keyed off a magic number. `factor_equivalence=None` (the default) selects the reduced layout exactly when it is exact for the calibration — that is, when capital shares are uniform across each country's sectors — and the full layout otherwise; `factor_equivalence=True` on a calibration where it is not exact raises `ValueError` instead of quietly dropping the capital-market clearing condition. If you pass `batch_size`, it must equal $3N-1$ or $4N-1$ for *your* $N$: on the 2-country toy calibration below that is 5 or 7, and `batch_size=230` raises `ValueError: batch_size=230 is inconsistent with a 2-country calibration: expected 5 (reduced layout, r == w) or 7 (full layout).` Passing both keywords with conflicting implications also raises.

Given $x_m$, prices and gross outputs follow from two Leontief systems,

$$p = (I - B^\top)^{-1}\, v(x_m), \qquad y = (I - A)^{-1}\, d(x_m, p),$$

with $B^\top = (A \circ \tau)^\top / (1 - \text{tax})$ the tariff-inclusive cost-share operator and $A$ the input-output coefficients. Both inverses are formed **once** (`(I - A)^{-1}` at construction, `(I - B^\top)^{-1}` in `update_tariffs`, i.e. once per tariff schedule) and copied to the device. A forward-difference Jacobian needs $F$ at the $B$ perturbed points $x_m + h_j e_j$, with $h_j = \varepsilon$ for the factor-price entries and $h_j = \varepsilon \max(|x_{m,j}|, 1)$ otherwise ($\varepsilon = 10^{-4}$). Stacking the perturbed points as rows of $X \in \mathbb{R}^{B \times B}$, the price and output solves for the whole batch become two matrix products,

$$P = \left[(I - B^\top)^{-1} V^\top\right]^\top, \qquad Y = \left[(I - A)^{-1} D^\top\right]^\top, \qquad V, D \in \mathbb{R}^{B \times M},$$

and the bilateral flow accounting becomes batched `einsum` contractions (`"min,bin->bmi"`). One kernel launch replaces $B$ serial residual evaluations, and

$$J_{:,j} = \frac{F(x_m + h_j e_j) - F(x_m)}{h_j}.$$

`ad_mode="forward"` replaces the differences by forward-mode automatic differentiation (`torch.func.jacfwd` on PyTorch, `mx.jvp` column by column on MLX) and `ad_mode="vjp"` by reverse mode (`torch.func.jacrev`, `mx.vjp`); both remove the step-size sensitivity of small economies. On any failure the evaluator emits a `RuntimeWarning` and falls back to the finite-difference path — the fallback is never silent — and on the `numpy` backend either AD mode warns and uses finite differences, since neither library is there to differentiate through. A macro vector whose length does not match the evaluator's layout raises `ValueError`, and an unknown `ad_mode` raises `ValueError` as well. The `jvp(xm, v)` method returns the directional derivative $J v$ the same way, with a central difference as last resort.

### 1.5 Levenberg-Marquardt step, equilibration and line search

Each iteration of `solve_trade_equilibrium_gpu` / `solve_trade_equilibrium_mlx` scales the Jacobian on both sides (Jacobi equilibration), $\tilde J = D_L J D_R$ with $D_R = \operatorname{diag}(1/\lVert J_{:,j}\rVert_2)$ and $D_L = \operatorname{diag}(1/\lVert (J D_R)_{i,:}\rVert_2)$, and solves the regularized normal equations

$$\left(\tilde J^\top \tilde J + \mu I\right) u = -\tilde J^\top D_L F(x_m), \qquad \Delta x_m = D_R\, u,$$

with `torch.linalg.solve` in float64 whenever PyTorch is importable (on the CUDA device when that is the polish device, on the CPU otherwise), then `scipy.linalg.solve`, then `scipy.linalg.lstsq`; the three are tried in that order and the first that succeeds wins. The whole step is then rescaled so that its factor-price block stays within $0.3$ in absolute value, and the log factor prices are clipped to $[-5, 5]$. A backtracking line search on the merit function $\Phi(x) = \lVert D_L F(x) \rVert_2$ then tries $\alpha = 1, \tfrac12, \tfrac14, \dots$ (at most 12 trial points) and accepts the first trial with $\max_i|F_i| \le \text{tol}$, $\Phi(x + \alpha \Delta) < \Phi(x)$ or $\max_i |F_i(x + \alpha\Delta)| < \max_i |F_i(x)|$: a simple-decrease rule in the Armijo (1966) backtracking pattern, without a sufficient-decrease constant. The damping follows the classical Marquardt update: $\mu \leftarrow \max(0.3\mu, 10^{-8})$ after an accepted step, $\mu \leftarrow \max(0.5\mu, 10^{-8})$ when only the best trial improved the merit, and $\mu \leftarrow \min(5\mu, 10^{2})$ otherwise. Convergence is declared when $\max_i |F_i| \le \text{tol}$ on the **full** residual (prices, outputs, factor markets, transfers, revenues), which is re-evaluated at the end and stored in `residuals`.

### 1.6 Homotopy continuation in the tariff parameter

Large asymmetric tariff shocks can defeat a Newton-type method started from the baseline. `solve_homotopy_continuation` embeds the target schedule in the path

$$\tau(\lambda) = (1-\lambda)\,\tau_0 + \lambda\,\tau_1, \qquad \tau^{fd}(\lambda) = (1-\lambda)\,\tau^{fd}_0 + \lambda\,\tau^{fd}_1, \qquad \lambda \in [0, 1],$$

(the same interpolation applies to the national tariff vectors) and traces it with the predictor-free, warm-started continuation of Allgower & Georg (1990, ch. 1): solve at $\lambda_0 = 0$ (or accept `x0`), then repeatedly solve at $\min(\lambda + \Delta\lambda, 1)$ starting from the last converged state. Step control is adaptive: a step that converged in at most 4 iterations grows the next one by $1.5\times$ (capped at `max_step`), one that needed at least `max_iter_per_step - 5` shrinks it by $0.75$ (floored at `min_step`), and a failed step is discarded and retried with half the size. If the step falls below `min_step`, if the baseline itself does not converge, or if no step ever succeeds, the function **raises `RuntimeError`** (it does not return a partial result); the `callback(lambda, result)` hook, called after every successful step, is the way to keep intermediate equilibria.

---

## 2. Methodological Options

| Option | Namespace layer (`backend=`) | `solve_trade_equilibrium_gpu` | `solve_trade_equilibrium_mlx` |
|---|---|---|---|
| **Accepted names** | `"numpy"`, `"mlx"`, `"cupy"` (`"numba"` warns and falls back — it is a kernel backend, not a namespace; any other string raises `ValueError`) | `device` in `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"`; `backend` in `"torch"`, `"mlx"`, `"numpy"` | none: MLX only, raises `RuntimeError` without it |
| **Precision** | float64 on every backend (MLX on the CPU stream) | float64 Jacobian wherever the device allows; a float32 device only when requested explicitly, and only for the coarse phase | same policy, on the MLX CPU stream |
| **Jacobian** | not applicable (Newton on the macro block stays in NumPy) | batched finite differences, `ad_mode="forward"` (jacfwd) or `"vjp"` (jacrev) | batched finite differences, `ad_mode="forward"` (jvp) or `"vjp"` (mx.vjp) |
| **Stopping rule** | trade: `tol=2.5e-3` on $\max|F|$; spatial: the `tol` you asked for, finished on the host when the device floor is coarser | `tol=2.5e-3` on the full residual | `tol=2.5e-3` on the full residual |
| **Failure policy** | `RuntimeWarning`, then NumPy — for a missing backend, a raised exception *and* a non-converged accelerated solve | exceptions propagate; AD failures warn and fall back to finite differences | same |
| **Where tested** | 2x2x3 toy calibration and a 5-region spatial model (`tests/test_trade_solver_accelerated.py`) | 45-sector, 77-country ICIO calibration (`tests/test_gpu_acceleration.py`), skipped when the accelerator or the private data are absent | same |

The default trade tolerance is the same $2.5 \times 10^{-3}$ for the NumPy solver and the accelerated ones. It is loose: on the toy calibration two NumPy methods (`"newton"` and `"condensed"`) already differ by about $10^{-5}$ in wages at that tolerance, which is larger than any cross-backend gap measured in Section 4.

---

## 3. Installing the Accelerators

None of the accelerators is part of the base install, and the base install runs everything on this page through the NumPy path.

| Accelerator | Install | Platforms | Provides |
|---|---|---|---|
| Apple MLX | `pip install "puremacro[backend]"` (the `[accel]` extra has the same contents: `numba` plus `mlx` on macOS) | Apple Silicon macOS only (`sys_platform == 'darwin'` marker) | `backend="mlx"` on both solvers, `solve_trade_equilibrium_mlx`, `device="mlx"` in the GPU engine |
| NVIDIA CuPy | `pip install "puremacro[cuda]"` (`cupy-cuda12x`; choose the wheel matching your CUDA toolkit) | NVIDIA GPUs with CUDA | `backend="cupy"` on both solvers |
| PyTorch | `pip install torch` (there is **no** puremacro extra for it; use the selector at pytorch.org for a CUDA build) | CUDA, Apple MPS, or CPU | `solve_trade_equilibrium_gpu`, `solve_homotopy_continuation`, `device="cuda"/"mps"/"cpu"` |

`puremacro.trade.gpu` imports `torch` and `mlx` **lazily**: `import puremacro.trade`, and even `from puremacro.trade.gpu import solve_trade_equilibrium_gpu`, leave both out of `sys.modules` on a machine where they are installed, and succeed on a machine where they are not. Every name on this page therefore exists in the base install; a call that genuinely needs an absent accelerator raises `RuntimeError` when it asks for it by name, while `device=None` / `"auto"` resolves to `("numpy", "cpu")` and runs the serial evaluator. `puremacro.runtime.capabilities().backends` lists the installed *namespace* backends (`numpy`, `numba`, `mlx`, `cupy`); PyTorch is not part of that tuple even when installed.

**Pyodide, iPad and the browser.** Neither PyTorch, MLX nor CuPy can be installed in a Pyodide kernel, so `capabilities().backends == ("numpy",)` there (see [Running anywhere](tablet.md)), `detect_device()` returns `numpy:cpu` by construction, and `backend="mlx"` or `"cupy"` produces the `RuntimeWarning` and the NumPy answer. A script written against the NumPy path runs unchanged on a tablet.

---

## 4. Runnable Worked Examples

All examples run offline. They were executed on an Apple Silicon workstation with PyTorch 2.12.1 (MPS) and MLX 0.32.2 installed and no NVIDIA GPU; the printed lines reproduce that run. Every example finishes in well under two seconds once `puremacro.trade` is imported (see the import-time caveat in section 7).

### 4.1 What hardware do I have?

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

With PyTorch installed, `"auto"` prefers MPS over MLX on a Mac, and the MPS `DeviceInfo` reports `f32-only`. The memory dictionary always carries `backend`, `device`, `allocated_mb` and `peak_mb`; MPS adds `driver_mb`, CUDA and MLX add `cache_mb`, and the NumPy fallback reports the peak resident set size of the process from `resource.getrusage`.

### 4.2 Trade equilibrium on NumPy and MLX with the toy calibration

The calibration is the 2-country, 2-sector, 3-final-demand table of `tests/test_trade_solver_accelerated.py` (lines 21-72). At its baseline the table is already an equilibrium (the solvers return after zero iterations), so the example imposes a 25% tariff on everything country 1 imports from country 0 to make the solver work.

```python
import warnings

import numpy as np

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium


def toy_calibration():
    """2-country, 2-sector, 3-final-demand table (tests/test_trade_solver_accelerated.py)."""
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

# 25% tariff on everything country 1 imports from country 0 (intermediates and final demand)
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

# An accelerator that is not installed: RuntimeWarning, then the NumPy answer
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

Both backends converge in two iterations to *identical* wages, prices and outputs: the namespace layer runs the two offloaded contractions in float64 on the MLX CPU stream, so there is no float32 rounding left to perturb the Newton path. On a problem this small the MLX path is slower than NumPy, because each iteration ships two tiny tensors to the device and back; the namespace layer is meant for the $M = 3465$ operators of the ICIO calibration.

### 4.3 The batched Jacobian on the same calibration

Continuing from the previous block, the evaluator accepts the 4-D tariff tensors directly and, with `batch_size` left at `None`, picks the reduced layout $[\log w; T; X_N]$ — exact for this calibration — which has $B = 3 \cdot 2 - 1 = 5$ entries. The MLX entry is requested as `device="mlx"`, so the evaluator takes the float64 CPU stream; `stream="gpu"` would be the way to ask for float32.

```python
from puremacro.trade.gpu import BatchedJacobianEvaluator

zeros = np.zeros(nc)                       # no national (uniform) tariffs
x_m = np.concatenate([np.zeros(nc),        # log w  (r = w under the B = 3*nc - 1 layout)
                      calib.T.ravel(),     # T      (tax revenue)
                      calib.invforT.ravel()[: nc - 1]])  # XN (net transfers of countries 1..nc-1)

ev_np = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device="cpu", backend="numpy")
J_np = ev_np.evaluate_batched_jacobian(x_m)
print("B =", len(x_m), "| J shape:", J_np.shape, "| max|J| =", f"{np.max(np.abs(J_np)):.2f}")

for backend, device in [("torch", "cpu"), ("torch", "mps"), ("mlx", "mlx")]:
    try:
        ev = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device=device, backend=backend)
    except RuntimeError as exc:            # accelerator absent on this machine
        print(f"{backend}:{device} unavailable ({exc})")
        continue
    J = ev.evaluate_batched_jacobian(x_m)
    print(f"{backend}:{device:4s} dtype={ev.dtype}  max|J - J_numpy| = {np.max(np.abs(J - J_np)):.1e}")

ev_t = BatchedJacobianEvaluator(calib, tau, tau_fd, zeros, zeros, device="cpu", backend="torch")
J_ad = ev_t.evaluate_batched_jacobian(x_m, ad_mode="forward")   # torch.func.jacfwd, no step size
print("forward-AD vs finite differences:", f"{np.max(np.abs(J_ad - J_np)):.1e}")
```

```text
B = 5 | J shape: (5, 5) | max|J| = 31.22
torch:cpu  dtype=torch.float64  max|J - J_numpy| = 1.4e-10
torch:mps  dtype=torch.float32  max|J - J_numpy| = 3.1e-02
mlx:mlx  dtype=None  max|J - J_numpy| = 2.8e-10
forward-AD vs finite differences: 1.7e-03
```

The float64 PyTorch CPU Jacobian reproduces the serial NumPy one to $10^{-10}$, and so does MLX, because the evaluator defaults to the float64 CPU stream. The one float32 device left in the table, MPS, differs by $3 \times 10^{-2}$ on entries of size 31 — $10^{-3}$ relative — which is exactly what a finite difference does with float32 rounding divided by $h = 10^{-4}$, and exactly why the solvers polish in float64 (section 1.3). The $1.7 \times 10^{-3}$ gap between forward-mode AD and finite differences is the truncation error of the differences themselves; the AD Jacobian is the more accurate of the two, and `ad_mode="vjp"` (reverse mode) reproduces it. (`ev.dtype` is only set on the PyTorch path; MLX picks float32 or float64 per stream.)

### 4.4 Spatial equilibrium on NumPy and MLX

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

Both backends report the same 27 iterations and agree to $10^{-9}$, because the MLX float32 loop hands over to the NumPy float64 loop as soon as it reaches its own floor (section 1.1): the run is certified against the `tol=1e-8` you asked for, not against the device's $10^{-6}$, and `max_residual` comes back at $7.3 \times 10^{-9}$ on MLX against $7.4 \times 10^{-9}$ on NumPy. The welfare effect of a 20% productivity gain in the first region agrees to five decimals. The population sums to exactly $\bar L$ on both backends because the renormalization happens in NumPy.

### 4.5 The device solvers and the homotopy on the toy calibration

`solve_trade_equilibrium_gpu` and `solve_homotopy_continuation` are built for the ICIO calibration, but nothing stops them from running on the 2x2x3 table of 4.2 — and running them there is the cheapest way to see the tariff defaults, the layout resolution and the device policy on the result. Continuing from the block in 4.2:

```python
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu

res_np = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="condensed")
res_gpu = solve_trade_equilibrium_gpu(calib, tau=tau, tau_fd=tau_fd)   # tauf/tauf_fd default to 0
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

The device solver reaches the same wages as the NumPy reference to six decimals with no tariff arguments beyond `tau` and `tau_fd`, because `tauf` and `tauf_fd` default to zeros exactly as they do on `solve_trade_equilibrium`. The metadata says what actually happened: `"auto"` picked MPS, the float64 policy of section 1.3 silently moved the Jacobian to `torch:cpu`, the reduced layout was chosen because it is exact here ($B = 5$), and both Jacobians were float64. The homotopy took three steps — $0.25$, then $0.625$ after the $1.5\times$ growth of a step that converged in at most 4 iterations, then $1.0$ — and the warm starts leave it an order of magnitude below the direct solve's residual.

### 4.6 Call shape of the full PyTorch / MLX solvers (45-sector ICIO calibration)

`solve_trade_equilibrium_gpu`, `solve_trade_equilibrium_mlx` and `solve_homotopy_continuation` are designed for, and tested against, the full 45-sector, 77-country OECD ICIO calibration, whose $M = 3465$ Leontief operators are what the batched GEMM is meant to amortize. That data set is not distributed with the package, so the calls are shown as shapes only, not as an executed example:

```text
from puremacro.trade import build_tariff_matrices, calibrate_trade_model
from puremacro.trade.data import load_raw_45sector_icio
from puremacro.trade.gpu import solve_homotopy_continuation, solve_trade_equilibrium_gpu, solve_trade_equilibrium_mlx

raw   = load_raw_45sector_icio()                       # needs data_2020_SML.csv (IO_RAW45_PATH)
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

All three return the same `TradeEquilibriumResult` as `solve_trade_equilibrium`, with `metadata["method"]` equal to `"gpu_accelerated"` or `"mlx_accelerated"`. On this calibration $N = 77$, so `batch_size=230` and `batch_size=307` are the values the validator accepts, and leaving `batch_size=None` picks between them by whether $r = w$ is exact.

---

## 5. Full API Specification

```text
# puremacro._backend
SUPPORTED = ("numpy", "numba", "mlx", "cupy")
backend_available(name: str) -> bool                     # ValueError for unknown names
available_backends() -> tuple[str, ...]                  # installed, numpy first
get_array_namespace(name: str)                           # numpy | mlx.core | cupy; ImportError with the pip hint
to_numpy(x) -> np.ndarray

# backend= on the model solvers
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
device_context(device: str | None = None, backend: str | None = None)   # context manager -> (backend, device)
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

`puremacro.trade` re-exports `DeviceInfo`, `detect_device`, `select_compute_device`, `device_context`, `get_memory_usage`, `reset_peak_memory`, `BatchedJacobianEvaluator`, `solve_trade_equilibrium_gpu`, `solve_trade_equilibrium_mlx` and `solve_homotopy_continuation`; `has_torch`, `has_mlx`, `to_tensor` and `to_numpy` are imported from `puremacro.trade.gpu`.

### Parameters shared by the accelerated solvers

- `calib`: a `TradeCalibrationResult` from `calibrate_trade_model`.
- `tau`, `tau_fd`: effective tariff multipliers ($1 + $ rate), either 4-D `(ns, nc, ns, nc)` / `(ns, nc, nfd, nc)` or the internal `(M, ns, nc)` / `(M, nfd, nc)` layout; `None` means all ones.
- `tauf`, `tauf_fd`: national uniform tariff *rates* of length `nc`, applied to the import bill in the revenue equation. `None` means all zeros, exactly as on `solve_trade_equilibrium`, so the default call solves the same baseline as the NumPy solver.
- `x0`: `None` (start from $\log w = 0$, the calibrated $T$ and net transfers), the full state vector of length $2M + 4N - 1$, or a macro vector of either layout, $3N - 1$ or $4N - 1$, for *your* $N$; a reduced vector seeding the full layout sets $r = w$. Any other length raises `ValueError` naming the three it expects.
- `device` / `backend`: explicit values are kept as given; `select_compute_device` fills in whichever of the two is missing, driven by `device` if it was given and by `backend` otherwise. The pair that ends up evaluating the Jacobian may differ from the pair you asked for (the float64 policy of section 1.3); `metadata["device"]` / `["backend"]` record the effective pair and `metadata["requested_device"]` / `["requested_backend"]` the requested one. `solve_trade_equilibrium_mlx` has neither argument.
- `batch_size` / `factor_equivalence`: the layout switch of section 1.4. Prefer `factor_equivalence` (`None` = choose by exactness, `True` = reduced $[\log w; T; X_N]$, `False` = full $[\log r; \log w; T; X_N]$); `batch_size` is the older spelling and must equal $3N-1$ or $4N-1$ for the calibration.
- `tol`: infinity-norm tolerance on the residual; `max_iter`: Levenberg-Marquardt iterations (`max_iter_per_step` per continuation step).
- `damping`: initial $\mu$.
- `ad_mode`: `"finite_diff"` (batched differences, default), `"forward"` (forward-mode AD: `torch.func.jacfwd` / `mx.jvp`) or `"vjp"` (reverse mode: `torch.func.jacrev` / `mx.vjp`). An AD mode that fails on the device warns and falls back to finite differences; an unknown value raises `ValueError`.
- `replicate_matlab_precedence`: keep the operator precedence of the reference MATLAB implementation (`ff_equi.m`) in the value-added unit-cost term, $w / (1-\alpha)^{1-\alpha}$; `False` uses $(w / (1-\alpha))^{1-\alpha}$.
- `base_result`: baseline equilibrium used by the post-processing for CPI and terms-of-trade indices.
- `verbose`: print per-iteration `max_res`, step and $\mu$ (and, in the homotopy, the step log).
- `callback` (homotopy only): `callback(lambda_value, result)` after each successful step.

---

## 6. Result Interface

### `TradeEquilibriumResult` (accelerated solvers)

The accelerated solvers return the same frozen dataclass as `solve_trade_equilibrium`, post-processed by `postprocess_trade_equilibrium`, so `x_sol`, `p_sol`, `y_sol`, `r_sol`, `w_sol`, `T_sol`, `XN_sol`, the bilateral flow tensors, `gdp`, `imports`, `exports`, `tariffs`, `cpi` and `terms_of_trade` are documented once in the [trade page](spatial_and_trade_ge.md). The fields the solvers on this page set themselves are:

- `converged`, `iterations`, `max_residual`, `diff` / `residual_norm` (L1 norm) and `residuals`: evaluated on the full system residual in float64.
- `metadata`: `method` (`"gpu_accelerated"`, `"mlx_accelerated"`, or the NumPy method name with the requested `backend`), the effective `device` and `backend` together with `requested_device` and `requested_backend`, `factor_equivalence`, `batch_size`, `jacobian_evaluations` (a count of Jacobians by `backend:device:dtype`), `ad_mode`, `tol`, `max_iter`, `solve_duration_seconds` and `memory_usage` (the dictionary of `get_memory_usage`). On the NumPy solver, `metadata["backend"]` echoes the *requested* backend even after a fallback.
- `summary(detailed=False) -> pd.DataFrame`: the solver diagnostic table printed in 4.2; `detailed=True` gives the country table (GDP, trade, tariffs, CPI, terms of trade).
- `to_frame(detailed=False)`, `to_markdown(detailed=False)`, `to_latex(detailed=False)`, `to_typst(detailed=False)`: the same table in each format.
- There is no `plot` method on `TradeEquilibriumResult`; scenario figures come from `puremacro.trade.plot` (`plot_country_impacts`, `plot_tariff_escalation_curve`, ...) applied to a `ScenarioBatchResult`.

### `AllenArkolakisResult` (spatial solver)

Unchanged by the backend: `wages`, `population`, `price_index`, `real_wages`, `welfare`, `consumer_market_access`, `firm_market_access`, `trade_shares`, and for counterfactuals `w_hat`, `L_hat`, `welfare_hat`, `welfare_pct`; diagnostics `converged`, `iterations`, `max_residual`, `labor_conservation_residual`, `spatial_utility_variance`; presentation `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` and `plot(kind="spatial")`. Nothing in the result records which backend produced it.

### `DeviceInfo` and the memory dictionary

`DeviceInfo` is the frozen dataclass of section 1.2. `get_memory_usage()` returns `{"backend", "device", "allocated_mb", "peak_mb"}` plus `cache_mb` (CUDA, MLX) or `driver_mb` (MPS); `reset_peak_memory()` resets the peak counters (CUDA, MLX) and empties the device caches (CUDA, MPS, MLX), and `device_context(...)` calls it on exit and yields the `(backend, device)` pair without cross-checking the two, echoing the strings you passed (`device_context("mlx")` yields `("mlx", "mlx")`; `device_context(backend="mlx")` alone yields `("mlx", "mps")` on a Mac where `"auto"` picks MPS).

---

## 7. Caveats and Limitations

- **`backend=` on `solve_trade_equilibrium` only acts with `method="condensed"`.** The default `method="newton"` and the other methods ignore it without a warning, and `metadata["backend"]` records the requested name in every case, so on those methods it is a label rather than a fact. The accelerated device solvers, by contrast, record the effective device and backend alongside the requested pair.
- **Two layers, two vocabularies.** The namespace layer takes `"numpy"`, `"mlx"`, `"cupy"` — `"numba"` warns and falls back, and anything else, `"torch"` included, raises `ValueError: Unknown backend`. The device layer takes `"auto"`, `"numpy"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`, `"mlx"`, `"gpu"` and raises `ValueError` on anything else. The two vocabularies do not overlap cleanly: `"cupy"` means nothing to the device layer and `"mps"` means nothing to the namespace layer.
- **Precision.** MPS has no float64 and MLX's Metal stream is float32, so an explicitly requested float32 device is used for the coarse phase only and warns when it is. Expect no cross-backend difference at all on the float64 paths (section 4 measures $0$ on the namespace layer, $10^{-10}$ on the Jacobian) and about $10^{-3}$ relative on a float32 MPS Jacobian.
- **Tolerances.** `tol=2.5e-3` on both the NumPy and the accelerated trade solvers is loose; tighten it explicitly for parity work.
- **Import cost.** `import puremacro.trade` no longer pulls in PyTorch or MLX — both are imported lazily, at the first call that needs them — so the import costs about 1.3 s on the workstation used here and succeeds whether or not the accelerators are installed. The first accelerated call pays the framework import.
- **Unverified paths.** The CuPy branch of both namespace-layer solvers and the CUDA branch of the device layer were not executed while writing this page (no NVIDIA hardware); they are covered by the fallback logic and by tests that skip when the accelerator, or the private ICIO data, are absent.
- **No speedup on small problems.** All device paths add transfer and launch latency; the toy calibration and the 5-region model run slower on MLX than on NumPy. [Performance & Benchmarks](benchmarks.md) currently publishes CPU timings only.
- **Offline and deterministic.** Nothing here touches the network; the accelerated paths are only as deterministic as the device reductions, and the NumPy path is the reference for reproducibility claims.

---

## References

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Allgower, E. L., & Georg, K. (1990). *Numerical Continuation Methods: An Introduction*. Springer Series in Computational Mathematics 13. Berlin: Springer.
- Armijo, L. (1966). "Minimization of functions having Lipschitz continuous first partial derivatives." *Pacific Journal of Mathematics*, 16(1), 1–3.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares." *Quarterly of Applied Mathematics*, 2(2), 164–168.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of nonlinear parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441.
