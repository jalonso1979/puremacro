> 🇬🇧 [English](../benchmarks.md) · 🇪🇸 Español

# Pruebas comparativas de rendimiento computacional

`puremacro` está optimizado para una ejecución ultrarrápida y sin sobrecostes innecesarios, aprovechando al máximo las operaciones vectorizadas de NumPy y las rutinas BLAS/LAPACK de SciPy.

---

## Tiempos de ejecución de referencia (Apple Silicon M-Series, Python 3.11)

La siguiente tabla reporta los tiempos medios de ejecución en `n_runs=5` repeticiones medidos el 2026-11-15 (código `puremacro` 6.6.0, Python 3.11, Apple Silicon M-Series):

| Motor / Modelo | Tiempo de ejecución (ms) | Referencia metodológica | Detalles de implementación |
| :--- | :--- | :--- | :--- |
| **VAR(4) + FRI por Cholesky** ($T=1000, N=5, H=40$) | **0.60 ms** | MATLAB / statsmodels | Recursión de medias móviles vectorizada en NumPy puro |
| **Equilibrio general HANK en espacio de secuencias** ($T=40, N_a=50$) | **36.06 ms** | Julia SSJ / Python SSJ | EGM + inversión matricial directa en $\mathcal{O}(T^3)$ |
| **Simulación climática DICE** ($150\text{ años}, 3\text{ cajas}$) | **8.77 ms** | GAMS / DICE2016 | Ciclo de transición de reservorios vectorizado |
| **Nowcast por DFM con frecuencias mixtas** ($T=120, N=15$) | **3.31 ms** | Fed de Nueva York / MATLAB | Convergencia EM con SVD para valores faltantes |
| **Descenso por coordenadas Elastic Net** ($T=200, P=40, 30\lambda$) | **31.43 ms** | glmnet / scikit-learn | Bucle de umbralización suave en NumPy puro |
| **Filtro paso-banda (CF) + Beveridge-Nelson** ($T=500$) | **2.45 ms** | MATLAB / Stata `tsfilter` | Proyección de espacio de estados en tiempo lineal |

Los tiempos anteriores corresponden a las rutas de referencia en NumPy/SciPy
sobre CPU. Las alternativas aceleradas por hardware se documentan aparte: el
argumento `backend=` del solucionador espacial en
[`docs/es/spatial_and_trade_ge.md`](spatial_and_trade_ge.md) y el solucionador
HJB en tiempo continuo en [`docs/es/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## Reproducir las pruebas localmente

Ejecute la suite automatizada de pruebas de rendimiento en su equipo:

```bash
python benchmarks/benchmark_suite.py
```
