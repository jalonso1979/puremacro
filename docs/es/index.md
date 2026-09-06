> 🇬🇧 [English](../index.md) · 🇪🇸 Español

# puremacro

**Modelización macroeconométrica y estructural de agentes heterogéneos de nivel de producción, en Python puro y con cero extensiones en C.**

[![Versión en PyPI](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Playground JupyterLite](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ¿Qué es puremacro?

`puremacro` es una librería unificada de computación macroeconómica construida completamente en Python puro y NumPy. Elimina cadenas de compilación complejas en Fortran, C++ y archivos MEX, permitiendo que los modelos econométricos se ejecuten en cualquier entorno: portátiles locales, clústeres de alto rendimiento, Google Colab y **directamente en el navegador web mediante Pyodide / WebAssembly**.

### Subsistemas principales

1. **Modelos DSGE estructurales y paridad con Dynare**:
   - **Analizador de archivos `.mod` de Dynare y CLI `puremacro-dynare`**: Analiza y resuelve archivos `.mod` directamente desde terminal o scripts de Python.
   - **Perturbación de segundo orden con poda (*pruning*)**: Términos cruzados ($g_{xu}, g_{uu}$) y correcciones por riesgo ($g_{\sigma\sigma}$) de Schmitt-Grohé y Uribe (2004) con poda de Kim et al. (2008) y paridad completa con las reglas de decisión `oo_.dr` de Dynare.
   - **OccBin (restricciones ocasionalmente activas)**: Algoritmo lineal por tramos de Guerrieri e Iacoviello (2015) para la cota inferior de tasa cero (ZLB) y restricciones de endeudamiento.
   - **Previsión perfecta no lineal**: Solver de relajación de Newton-Raphson apilado de Boucekkine-Juillard para transiciones deterministas de gran escala.
   - **Estimación bayesiana de DSGE por MCMC**: Búsqueda de moda vía L-BFGS-B / Nelder-Mead, matriz de covarianza hessiana de Laplace y algoritmo adaptativo de Random-Walk Metropolis-Hastings.
   - **Momentos teóricos y descomposición de choques**: Momentos analíticos de Lyapunov, descomposición de varianza del error de pronóstico (FEVD) y descomposición histórica exacta de choques suavizada por Kalman.

2. **Modelos de agentes heterogéneos (HANK y VFI)**:
   - **HANK en el espacio de secuencias** (Auclert, Bardóczy, Rognlie y Straub 2021, *Econometrica*): Modelos de equilibrio general con mercados incompletos resueltos en tiempo $\mathcal{O}(T^3)$.
   - **Algoritmo Fake News**: Cálculo rápido en $\mathcal{O}(T^2)$ de los jacobianos de consumo intertemporal mediante vectores de esperanza e identidades de acumulación.
   - **Transferencias fiscales focalizadas**: Trayectorias dinámicas de consumo y multiplicadores fiscales acumulados entre deciles de riqueza.
   - **Búsqueda y emparejamiento DMP** (Mortensen-Pissarides): Transiciones del mercado laboral, salarios rígidos y dinámica de la curva de Beveridge.

3. **Motores econométricos y proyecciones locales**:
   - **`LPResult` unificado**: Proyecciones locales estandarizadas (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) con errores estándar HAC de Newey-West y Driscoll-Kraay para paneles.
   - **SVAR y FAVAR**: Cholesky, Blanchard-Quah, restricciones de signo, variables instrumentales externas/proxy, máxima participación de varianza (*news*) y VAR aumentado con factores.
   - **Diferencias en diferencias modernas**: Estimadores robustos a la heterogeneidad temporal y de cohortes (Callaway y Sant'Anna, Sun y Abraham, Borusyak-Jaravel-Spiess y DiD sintético).

4. **Nowcasting y aprendizaje automático**:
   - **Modelos de factores dinámicos de frecuencias mixtas (DFM)** (Giannone, Reichlin y Small 2008): Seguimiento del PIB en tiempo real con bordes irregulares y descomposición de noticias.
   - **Pronóstico macroeconómico penalizado**: Elastic Net y Lasso Adaptativo (Zou 2006) mediante descenso por coordenadas para selección de predictores de alta dimensión.

5. **Macroeconomía del clima**:
   - **DICE (Dynamic Integrated Climate-Economy)** (Nordhaus 2018): Modelo de 3 reservorios del ciclo de carbono, calentamiento global y contabilidad del coste social del carbono (SCC).

6. **Informes y exportación para publicaciones**:
   - Exportación directa de tablas listas para publicar en **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`) y **Markdown** (`.to_markdown()`), con estrellas de significancia y errores estándar.

7. **Ejecución en cualquier entorno**:
   - Compatibilidad completa con Pyodide/WebAssembly para tabletas e iPad, con descarga automática a Google Colab (`runtime.colab`), ejecución fragmentada resistente a suspensiones (`longrun`) y cartuchos portátiles `.pmz` (`pocket`).

8. **Métodos de frontera (2.3)**:
   - **Restricciones narrativas de signo** ([guía](narrative_sign_svar.md)), **DiD honesto** ([guía](honest_did.md)), **Proyecciones locales suavizadas** ([guía](smooth_lp.md)), **HANK no lineal en el espacio de secuencias** ([guía](hank_nonlinear.md)), **DSGE de Gertler-Karadi (2011)** ([guía](gertler_karadi.md)) y **BVAR con volatilidad estocástica** ([guía](bvar_sv.md)).
   - **Econometría espacial** ([guía](spatial.md)): matrices de pesos espaciales, I de Moran / C de Geary, HAC espacial de Conley en cortes transversales y proyecciones locales de panel, y VI shift-share con errores de Adão-Kolesár-Morales.

---

## Instalación

```bash
pip install puremacro
```

O instale con herramientas completas de cuadernos:

```bash
pip install "puremacro[notebooks]"
```
