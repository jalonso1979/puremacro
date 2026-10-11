> 🇬🇧 [English](../research_workflows.md) · 🇪🇸 Español

# Evidencia independiente, estimación estructural e incidencia comercial

El [flujo de pronósticos entre países](cross_country_forecasting.md) añade un
panel fijo de tres países, una evaluación con ventana creciente y procedencia
portátil. Usa una edición reciente de datos revisados; no es evidencia causal
ni de tiempo real. Está incluido en puremacro 4.8.0.

Estos flujos de investigación conectan validación independiente, estimación
estructural y análisis distributivo. Funcionan sin conexión desde el paquete
instalado con las dependencias numéricas estándar y conservan los supuestos,
las unidades y el alcance de la evidencia.

Los [estudios con datos reales](empirical_research.md) añaden estimación SW07
con momentos observados y diagnósticos condicionales a la calibración, y una
replicación base de Romer–Romer (2010) con datos originales, software independiente
y valores publicados redondeados. El ejemplo NK sintético sigue siendo una
prueba de recuperación; los nuevos estudios tienen datos, supuestos y
conclusiones propios.

| Tarea | API | Guía |
|---|---|---|
| Revisar evidencia numérica y de datos | `puremacro.validation.run_research_benchmarks` | [Benchmarks independientes](research_benchmarks.md) |
| Ajustar modelos a momentos empíricos correlacionados | `puremacro.structural.fit_structural` y `lp_moment_targets` | [Puente estructural](structural_bridge.md) |
| Medir incidencia por hogar o grupo | `puremacro.trade.compute_distributional_welfare` | [Incidencia distributiva](trade_distributional.md) |
| Comparar correcciones de momentos y ponderaciones SW07 | `puremacro.structural.run_sw07_estimator_experiment` | [Controles emparejados del estimador](sw07_estimator_experiment.md) |
| Conciliar precios arancelarios, ingresos y devoluciones a hogares | `python -m puremacro.examples.distributional_trade_ge` | [Aplicación GE sintética con canastas ENIGH](distributional_trade_ge.md) |

## Benchmarks

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)  # False: falla el redondeo publicado del estadístico t RR2010.
report.write("research_output/benchmarks")
```

Los ocho casos distinguen soluciones analíticas, optimización independiente del
gasto, resultados externos autenticados de Dynare, totales oficiales de INEGI
y la especificación original de Romer–Romer con software independiente y
valores publicados.
El expediente JSON/Markdown conserva valores observados y de referencia,
unidades, fuentes, tolerancias, controles negativos y limitaciones. No certifica
todos los modelos ni presenta ejercicios sintéticos como replicaciones empíricas.
Pasan siete de ocho casos: el estadístico t RR2010 queda fuera del intervalo
de redondeo publicado por aproximadamente 0.000003925. El fallo se conserva;
véase la [guía empírica](empirical_research.md).

## De proyecciones locales a un modelo nuevo keynesiano

```bash
python -m puremacro.examples.structural_irf_matching --output research_output/structural
```

La aplicación simula una economía de tres ecuaciones desde una solución en
forma cerrada independiente, con perturbación monetaria persistente y errores
de medición. Estima respuestas LP de producto e inflación con **covarianza HAC
conjunta**, y ajusta aversión al riesgo, pendiente de Phillips y persistencia
resolviendo el modelo mediante `load_mod`. La innovación es 0.01 en la tasa
trimestral: un punto porcentual.

Los horizontes 0–4 entran al ajuste; 5–8 son comprobaciones descriptivas con la
misma muestra. Se exportan observaciones, momentos, covarianza, parámetros,
comparaciones, figura y manifiesto con modelo, semilla y hash de los datos.
Es recuperación con datos simulados, no replicación histórica. Para un estudio
empírico sustituya el simulador por datos transformados y shocks identificados,
o construya `MomentTargets` con estimaciones y su covarianza conjunta. Frecuencia,
unidades, transformación y normalización deben coincidir con el modelo.

## Canastas observadas de México: ENIGH 2024

```python
import pandas as pd
from puremacro.datasets import load_enigh2024_deciles
from puremacro.trade import prepare_household_groups, compute_distributional_welfare

survey = load_enigh2024_deciles()
sectors = list(survey.attrs["categories"])
groups = prepare_household_groups(
    survey[sectors], survey["households"],
    income_exposure=survey[["cash_wages"]],
    monetary_unit="MXN", period="quarter", provenance=survey.attrs,
)
prices = pd.Series(1.0, index=sectors)
prices["food"] = 1.025  # Supuesto explícito de precios, no una estimación.
incidence = compute_distributional_welfare(
    groups, base_prices=pd.Series(1.0, index=sectors), prices=prices,
    rule="fixed_baskets",
)
print(incidence.groups[["ev_pct", "cv_pct"]])
```

Los datos proceden de los [cuadros oficiales 3.2 y 4.2 de ENIGH 2024](https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx).
Son pesos por trimestre **por todos los hogares del decil**, no únicamente por
los compradores de cada rubro. Los factores de expansión suman 38,830,230 hogares.
Ocho categorías de consumo excluyen transferencias de gasto y consumo no monetario.
La exposición salarial excluye remuneraciones en especie. `survey.attrs` conserva
hashes de origen y CSV, renglones, unidades, transformaciones y limitaciones.

Para reconstruir el CSV use `python tools/build_enigh_distributional_data.py
--workbook PATH.xlsx`; solo esta conversión necesita `openpyxl`, disponible en
`puremacro[io]`. Se rechazan archivos con un hash no revisado.

```bash
python -m puremacro.examples.distributional_trade_enigh --output research_output/distributional
```

La aplicación completa evalúa seis escenarios. Por defecto **supone** un arancel
de 10%, exposición importada del rubro alimentario de 25% y traslado completo,
que producen un aumento de precio de 2.5%. ENIGH no identifica estos parámetros.
Los salarios permanecen constantes. Una bolsa externa de 0.5% del consumo
agregado inicial se omite, se reparte por igual entre hogares o se dirige a los
cuatro deciles inferiores. **No se infiere que sea recaudación arancelaria.** Se
comparan canastas fijas y Cobb–Douglas. Los argumentos de línea de comandos
permiten modificar todos los supuestos (`--help`).

Se exportan EV/CV por grupo, agregados ponderados, atribución de precios/ingresos/
transferencias, figura y manifiesto. No se afirman intervalos de confianza de
encuesta, heterogeneidad intradecil, concordancia industrial ni shocks de
equilibrio general estimados. El puente con resultados comerciales auditados
acepta precios compuestos con soporte, y requiere un mapeo explícito de ingresos.

La diferencia entre patrones de consumo y exposición importada es sustantiva:
véanse [Fajgelbaum y Khandelwal](https://www.nber.org/papers/w20331) y
[Borusyak y Jaravel](https://www.nber.org/papers/w28957). No se pretende replicar
ninguno de esos trabajos.
