> 🇪🇸 Español · 🇬🇧 [English](../index.md)

# puremacro

**Modelado macroeconométrico y estructural con agentes heterogéneos en Python puro, listo para producción y sin extensiones en C.**

[![Versión en PyPI](https://img.shields.io/pypi/v/puremacro.svg)](https://pypi.org/project/puremacro/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Entorno JupyterLite](https://img.shields.io/badge/JupyterLite-Live%20IDE-orange.svg)](https://jalonso1979.github.io/puremacro/)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ¿Qué es puremacro?

`puremacro` es una biblioteca unificada de computación macroeconómica desarrollada íntegramente en Python puro y NumPy. Elimina cadenas de compilación complejas en Fortran, C++ o MEX, permitiendo que los modelos econométricos se ejecuten en cualquier entorno: computadoras portátiles, clústeres de alto rendimiento, Google Colab y **directamente dentro del navegador web mediante Pyodide / WebAssembly**.

### Subsistemas Clave

1. **Modelos DSGE Estructurales y Paridad con Dynare (3.0 – 3.2)**:
   - **Analizador Nativo de Archivos `.mod` de Dynare y CLI `puremacro-dynare`**: Carga y resolución de archivos `.mod` estándar directamente en Python puro.
   - **Recursiones Analíticas Exactas del Gradiente (Score)**: Diferenciación simbólica sobre el AST y solucionador generalizado de Sylvester que evalúa $\nabla_\theta \ln L$ en una sola pasada hacia adelante sin diferencias finitas.
   - **HMC y NUTS en Python Puro**: Monte Carlo Hamiltoniano con adaptación dual averaging de tamaño de paso, adaptación de masa diagonal y diagnósticos de convergencia $\hat{R}$ normalizados por rangos.
   - **Política Discrecional Óptima vs Compromiso**: Iteraciones matriciales de Riccati de Dennis (2007) para política discrecional perfecta en el sentido de Markov frente a compromiso en perspectiva atemporal, descomponiendo los sesgos de inflación y estabilización.
   - **Modelado Híbrido DSGE-VAR**: Optimización de previas e identificación estructural DSGE-VAR($\lambda$) de Del Negro y Schorfheide (2004).
   - **Choques Anticipados de Noticias**: Aumento del espacio de estados con operadores nilpotentes que preservan la determinancia de Blanchard-Kahn.
   - **Filtrado No Lineal de Partículas y MS-DSGE**: Filtros de partículas bootstrap y auxiliares con volatilidad estocástica, OccBin diferenciable para NUTS y soluciones racionales con cambio de régimen de Markov de Foerster et al. (2016) con GIRFs analíticas cerradas.
   - **Perturbación Podada de Segundo Orden y OccBin**: Derivadas cruzadas de Schmitt-Grohé y Uribe (2004) con poda de Kim et al. (2008), y algoritmo lineal por partes de Guerrieri e Iacoviello (2015) para límites inferiores cero (ZLB).

2. **Modelos de Agentes Heterogéneos (HANK y VFI)**:
   - **HANK en el Espacio de Secuencias** (Auclert, Bardóczy, Rognlie y Straub 2021, *Econometrica*): Modelos de equilibrio general con mercados incompletos resueltos en $\mathcal{O}(T^3)$.
   - **Algoritmo de Noticias Falsas (Fake News)**: Cálculo acelerado $\mathcal{O}(T^2)$ de jacobianos de consumo en el espacio de secuencias mediante vectores de expectativa e identidades de acumulación.
   - **Puente HANK en Espacio de Secuencias**: Declaración directa del bloque `hetagent_block` en archivos `.mod` estándar de Dynare.
   - **Transferencias Fiscales Focalizadas**: Respuestas dinámicas de consumo y multiplicadores acumulados por deciles de riqueza.
   - **HJB en Tiempo Continuo**: Esquemas upwind en diferencias finitas de Achdou et al. (2022) para modelos de agentes heterogéneos en tiempo continuo.

3. **Motores Econométricos y Proyecciones Locales**:
   - **`LPResult` Unificado**: Proyecciones locales estandarizadas (`lp_hac`, `lp_iv`, `lp_state_dep`, `panel_lp`) con errores estándar Newey-West HAC, fixed-$b$ y Driscoll-Kraay.
   - **SVAR y FAVAR**: Cholesky, Blanchard-Quah, restricciones de signo, restricciones narrativas de signo, instrumentos externos/proxy, max-share/noticias y VAR aumentado por factores.
   - **Diferencias en Diferencias Modernas**: Estimadores con adopción escalonada robustos a heterogeneidad en el tratamiento (Callaway y Sant'Anna, Sun y Abraham, Borusyak-Jaravel-Spiess, SDID, Honest DiD).
   - **Econometría Espacial y VAR Global**: Errores estándar Conley HAC, modelos autorregresivos espaciales (`sar`, `sem`, `sdm`), proyecciones locales espaciales y VAR Global de Pesaran (`var.gvar`).

4. **Suites de Política Macroeconómica Aplicada (puremacro 3.2)**:
   - **Postura Monetaria y Gráficos de Abanico**: Evaluación de la postura monetaria en tiempo real frente a reglas de Taylor contrafactuales, proyecciones en abanico y sentimiento textual en comunicados de bancos centrales ([Guía](notebooks.md)).
   - **Nowcasting en Tiempo Real y Descomposición de Noticias**: Modelos de factores dinámicos (DFM) de frecuencia mixta (Giannone, Reichlin y Small), tratamiento de extremos irregulares (ragged edges) y atribución de sorpresas de datos ([Guía](nowcast.md)).
   - **Riesgo Macroprudencial y Estabilidad Financiera**: Densidades condicionales skew-$t$ de Growth-at-Risk (GaR) y redes de desbordamiento de riesgo sistémico de Diebold-Yilmaz ([Guía](notebooks.md)).
   - **Política Fiscal y DSA Soberano**: Multiplicadores fiscales trimétodo y análisis estocástico de sostenibilidad de la deuda pública ante escenarios de estrés macroeconómico y climático.

5. **Ecosistema de Datos Globales y Paneles Modulares (puremacro 3.2)**:
   - **Emisiones y Datos Climáticos**: Cuentas de emisiones y gases de efecto invernadero del Banco Mundial WDI y la OCDE SDMX a nivel internacional y sectorial ([Guía](data_ecosystem.md)).
   - **Balances Energéticos y Materias Primas**: Consumo de energía primaria, cuotas de generación limpia y suites ampliadas de precios de materias primas del Pink Sheet del Banco Mundial ([Guía](data_ecosystem.md)).
   - **Estabilidad Financiera Internacional**: Curvas de rendimiento soberano (10Y/2Y), tasas de política monetaria, brechas de crédito del BPI (BIS) y precios reales de vivienda ([Guía](data_ecosystem.md)).
   - **Constructores Modulares de Paneles**: Funciones de alto nivel `build_climate_panel` y `build_financial_panel` con armonización temporal automatizada (M$\to$Q, A$\to$Q).

6. **Generación de Informes y Publicación**:
   - Exportación de tablas con calidad de imprenta directamente a **LaTeX** (`.to_latex()`), **Typst** (`.to_typst()`) y **Markdown** (`.to_markdown()`), con errores estándar y estrellas de significancia ([Guía](reporting.md)).

7. **Ejecución en Cualquier Dispositivo**:
   - Compatibilidad completa con Pyodide/WebAssembly para tabletas y navegadores ([Guía](tablet.md)), descarga de cómputo a Google Colab (`runtime.colab`), ejecución segmentada (`longrun`) y cartuchos portátiles `.pmz` (`pocket`).

---

## Instalación

```bash
pip install puremacro
```

O con herramientas completas para cuadernos interactivos:

```bash
pip install "puremacro[notebooks]"
```
