> 🇬🇧 [English](../tablet.md) · 🇪🇸 Español

# Ejecución en cualquier entorno: iPad, Juno y el navegador

El núcleo numérico de `puremacro` se basa en NumPy, SciPy, Pandas y Matplotlib puros, lo que lo hace *importable* en un iPad o en un navegador web mediante Pyodide y WebAssembly. No obstante, ser importable no equivale a ser inmediatamente utilizable: cuatro servicios que un sistema de escritorio provee de forma transparente están ausentes en una tableta, y este módulo resuelve esa brecha operativa.

## Diagnóstico inicial: ¿qué puede hacer este dispositivo?

```python
from puremacro import runtime
print(runtime.report())
```

Salida típica en un entorno Pyodide:
```
puremacro runtime
  host       : pyodide 3.12.7 (wasm32)
  device     : tablet
  network    : js-fetch (call runtime.enable_browser_network())
  parquet    : unavailable -> use puremacro.runtime.store / pocket
  threads    : no (1 cpu, unknown)
  writable fs: yes
  backends   : numpy
```

La detección se realiza por heurística del sistema operativo, el entorno de aislamiento de iOS (`/var/mobile/`) y el agente de usuario del navegador. Cada campo puede fijarse manualmente mediante variables de entorno si la detección automática discrepa:
`PUREMACRO_HOST`, `PUREMACRO_DEVICE`, `PUREMACRO_SOCKETS`, `PUREMACRO_PARQUET`.

---

## 1. Sin sockets de red

Bajo WebAssembly / Pyodide no existe una pila de red TCP tradicional, por lo que las librerías `requests` y `urllib` fallan. El navegador, no obstante, sí puede realizar peticiones HTTP mediante la API nativa de JavaScript `fetch`. Una única llamada encamina toda la infraestructura de descarga a través del navegador:

```python
from puremacro import runtime
from puremacro.fetch import fetch_xrate_monthly

runtime.enable_browser_network()
fx = fetch_xrate_monthly(["MEX"])
```

Para revertir al modo tradicional, ejecute `runtime.disable_browser_network()`.

---

## 2. Sin motor Parquet (`pyarrow`)

Dado que `pyarrow` no dispone de binarios compilados para WebAssembly, `puremacro.pocket` empaqueta los datos en cartuchos portátiles `.pmz` autoverificables con firma SHA-256 que llevan consigo su propia procedencia histórica:

```python
from puremacro import pocket

# En la estación de trabajo:
pocket.pack(panel_macro, "panel_g7.pmz", source="OECD QNA", vintage="2026-09-01")

# En el iPad (incluso en modo avión):
cart = pocket.load("panel_g7.pmz")
panel = cart.frame()              # Comprobación de integridad por SHA-256 al leer
print(cart.provenance.vintage)    # '2026-09-01'
```

---

## 3. Suspensión de aplicaciones en iPadOS

Cuando el usuario conmuta entre aplicaciones en iPadOS, el sistema suspende los procesos en segundo plano, interrumpiendo cálculos largos. `puremacro.longrun` ejecuta el trabajo en bloques reanudables y almacena puntos de control (*checkpoints*):

```python
import numpy as np
from puremacro import longrun

job = longrun.bootstrap(iteracion_svar, n_draws=2000, checkpoint="svar_irf.ckpt")
job.run(seconds=30)  # Computa 30 segundos y guarda el avance en disco
# ... tras responder un mensaje o reabrir la aplicación:
job.run(seconds=30)
bandas = np.percentile(job.result(), [5, 95], axis=0)
```

Cada extracción $i$ utiliza deterministamente `default_rng([seed, i])`, de modo que un cálculo ejecutado a lo largo de varias sesiones fragmentadas genera resultados **bit a bit idénticos** a una ejecución continua, condición indispensable para publicaciones científicas.

---

## 4. Descarga de cómputo a Google Colab (`runtime.colab`)

Cuando una tarea excede la memoria o capacidad de cálculo del dispositivo móvil (como cadenas largas de MCMC o bootstraps de gran escala), `puremacro.runtime.colab` genera automáticamente un cuaderno de Google Colab conectado con Google Drive:

```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

nb = generate_colab_notebook(
    task_code="""
import puremacro as pm
res = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
pm.runtime.store.save_frame(res.summary(), "sw07_posterior.pmz")
""",
    mount_drive=True,
    export_result_file="sw07_posterior.pmz",
)

show_colab_offload_dialog(nb, filename="sw07_colab.ipynb")
posterior = load_colab_result("sw07_posterior.pmz")
```
