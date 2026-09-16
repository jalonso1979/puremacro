> 🇬🇧 [English](../benchmarks.md) · 🇪🇸 Español

# Pruebas comparativas de rendimiento computacional

`puremacro` está optimizado para una ejecución ultrarrápida y sin sobrecostes innecesarios, aprovechando al máximo las operaciones vectorizadas de NumPy y las rutinas BLAS/LAPACK de SciPy.

---

## Tiempos de ejecución de referencia (Apple Silicon M-Series / Linux x86_64)

La siguiente tabla reporta los tiempos de ejecución cronometrados en modelos y estimadores representativos:

| Motor / Modelo | Tiempo de ejecución (ms) | Referencia metodológica | Detalles de implementación |
| :--- | :--- | :--- | :--- |
| **VAR(4) + FRI por Cholesky** ($T=1000, N=5, H=40$) | **1.07 ms** | MATLAB / statsmodels | Recursión de medias móviles vectorizada en NumPy puro |
| **Equilibrio general HANK en espacio de secuencias** ($T=40, N_a=50$) | **111.8 ms** | Julia SSJ / Python SSJ | EGM + inversión matricial directa en $\mathcal{O}(T^3)$ |
| **Simulación climática DICE** ($150\text{ años}, 3\text{ cajas}$) | **0.89 ms** | GAMS / DICE2016 | Ciclo de transición de reservorios vectorizado |
| **Nowcast por DFM con frecuencias mixtas** ($T=120, N=15$) | **4.84 ms** | Fed de Nueva York / MATLAB | Convergencia EM con SVD para valores faltantes |
| **Descenso por coordenadas Elastic Net** ($T=200, P=40, 30\lambda$) | **39.08 ms** | glmnet / scikit-learn | Bucle de umbralización suave en NumPy puro |
| **Filtro paso-banda (CF) + Beveridge-Nelson** ($T=500$) | **3.17 ms** | MATLAB / Stata `tsfilter` | Proyección de espacio de estados en tiempo lineal |

Los tiempos anteriores corresponden a las rutas de referencia en NumPy/SciPy
sobre CPU. Las alternativas aceleradas por hardware se documentan aparte: el
argumento `backend=` de los solucionadores espaciales y de comercio en
[`docs/es/spatial_and_trade_ge.md`](spatial_and_trade_ge.md), la superficie
arancelaria en GPU — jacobianos por lotes, continuación por homotopía y los
solucionadores opcionales con `torch` / `mlx` — en
[`docs/es/trade_gpu.md`](trade_gpu.md), y el solucionador HJB en tiempo continuo
en [`docs/es/vfi_hjb_continuous.md`](vfi_hjb_continuous.md).

---

## Reproducir las pruebas localmente

Ejecute la suite automatizada de pruebas de rendimiento en su equipo:

```bash
python benchmarks/benchmark_suite.py
```
