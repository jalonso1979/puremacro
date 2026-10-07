"""Translate the existing corrected EN science into its stale Spanish edition.

Existing Spanish text/code is retained where the scientific content agrees.
Changed numerical code cells mirror the English edition, with presentation
strings localized. Cell-end display calls are removed under the notebook style
contract; Jupyter's inline backend displays figures when the cell completes.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[3] / "notebooks"
stem = "45_dsge_discretion_dsge_var_and_news_shocks"
english = (root / f"{stem}.py").read_text()
spanish = (root / f"{stem}_es.py").read_text()
en_parts = re.split(r"(?m)^# %%", english)
es_parts = re.split(r"(?m)^# %%", spanish)
en = en_parts[1:]
es = es_parts[1:11] + ["", ""] + es_parts[11:]
assert len(en) == len(es) == 36

translations = {
0: r"""# Frontera DSGE: política discrecional, DSGE-VAR y noticias anticipadas

**¿Cómo equilibran los bancos centrales estabilización y credibilidad cuando reoptimizan en cada período, cómo disciplinar los vectores autorregresivos con distribuciones a priori de equilibrio general microfundamentadas y cómo reaccionan los agentes a las noticias antes de que cambien los fundamentos?**

Los modelos DSGE linealizados suelen suponer una regla fija para el instrumento (una regla de Taylor) y choques que llegan por sorpresa. Este cuaderno desarrolla tres extensiones:

1. **Discreción frente a compromiso** (Oudiz & Sachs 1985; Clarida, Galí & Gertler 1999; Dennis 2007):
   Una autoridad que no puede comprometer a sus sucesores reoptimiza cada período, tomando como dadas las expectativas privadas. Con una meta de producto superior al potencial ($y^* > 0$), esto genera el **sesgo inflacionario** de Kydland-Prescott / Barro-Gordon y, ante choques de costos, un **sesgo de estabilización**: la discreción no permite prometer la respuesta persistente que el compromiso utiliza para orientar las expectativas.

2. **DSGE-VAR** (Del Negro & Schorfheide 2004):
   Las autocovarianzas teóricas $\Gamma_k(\theta)$ del DSGE centran una distribución a priori conjugada normal-Wishart invertida para un VAR. Un hiperparámetro $\lambda$ determina cuántas observaciones artificiales del DSGE representa la distribución a priori; la log-densidad marginal $\ln p(Y \mid \lambda, \theta)$ a lo largo de $\lambda$ muestra cuánto peso asignan los datos a las restricciones del DSGE.

3. **Choques de noticias (anticipados)** (Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012):
   Las reformas fiscales, la tecnología y la orientación futura suelen anunciarse antes de entrar en vigor. Los choques anticipados se incorporan mediante una matriz de desplazamiento nilpotente $K_H$: el estado exógeno afectado no cambia antes de la fecha programada, mientras las variables prospectivas reaccionan al anuncio.

También utilizamos el **preprocesador macro de Dynare**, los **diagnósticos de identificación por rango** de Iskrev (2010) y Komunjer & Ng (2011), y el **widget de deslizadores de IRF**, todo en Python puro bajo el contrato de cuatro paquetes de Pyodide. Todos los modelos son calibraciones construidas para el ejemplo y todos los datos son simulados.""",
2: r"""## Intuición

**Intuición.** Tres problemas prospectivos recorren este cuaderno: credibilidad, especificación incorrecta e información que llega antes de que cambien los fundamentos.

Primero, un banco central sin un mecanismo de compromiso no puede prometer de forma creíble que mantendrá una política restrictiva cuando el choque se haya disipado: el público sabe que reoptimizará cada período. Si la meta de producto supera el potencial, el público espera mayor inflación promedio (*sesgo inflacionario*), y el banco pierde la capacidad de usar promesas de política futura ante un choque de costos (*sesgo de estabilización*).

Segundo, un DSGE con parametrización rígida es, como máximo, una aproximación al proceso generador de datos. El DSGE-VAR utiliza las covarianzas del modelo como distribución a priori para un VAR no restringido; la verosimilitud marginal a lo largo de $\lambda$ indica cuántas «observaciones del modelo» están dispuestos a aceptar los datos.

Tercero, los hogares y las empresas prospectivos reaccionan a anuncios creíbles. Una mejora tecnológica anunciada para dentro de cuatro trimestres modifica hoy las tasas reales esperadas, el producto y la inflación, aunque la productividad todavía no haya cambiado.""",
6: r"""### Deslizadores de parámetros y resoluciones QZ rápidas

`interactive_irf` crea un widget con deslizadores usando únicamente Matplotlib. Cada movimiento resuelve de nuevo el modelo mediante la descomposición de Schur generalizada (QZ) y actualiza las respuestas impulso. Para un modelo pequeño la resolución requiere unos milisegundos; la latencia exacta depende del equipo.""",
8: r"""### Identificación formal de parámetros por rango (Iskrev 2010; Komunjer & Ng 2011)

Antes de optimizar la política o estimar bayesianamente, es necesario verificar la identificación local de los parámetros estructurales. Evaluamos los jacobianos de solución $J_1$ y momentos $J_2$ de Iskrev (2010), junto con los rangos de la función de transferencia $J_H$ y de la densidad espectral $J_S$ de Komunjer & Ng (2011). Las cuatro variables ($y, \pi, r, u$) se consideran observadas y los jacobianos se evalúan en la calibración.""",
10: r"""El modelo **no** está identificado en esta calibración: todos los criterios tienen rango 5 de 7. La primera dirección nula aumenta conjuntamente $\phi_\pi$ y $\phi_y$. La siguiente celda muestra por qué: con $\sigma = 1$ y $\phi_\pi - \phi_y = 1$, la trayectoria $y_t = -\pi_t$ satisface exactamente la curva IS, de modo que la regla de Taylor solo distingue $(\phi_\pi - \phi_y)\,\pi_t$. Verificamos que $y_t + \pi_t$ sea cero en la respuesta al choque de costos y repetimos el análisis con $\phi_y = 0.25$, que rompe esta coincidencia particular.""",
12: r"""## 2. Política monetaria óptima: discreción frente a compromiso

Comparamos la **política discrecional Markov-perfecta**, consistente en el tiempo (Dennis 2007), con el referente de **compromiso**: el plan de Ramsey iniciado en el estado estacionario, cuya ley de movimiento corresponde a la perspectiva atemporal.

El banco central minimiza la función de pérdida cuadrática:
$$ \mathcal{L}_t = \mathbb{E}_t \sum_{\tau=0}^\infty \beta^\tau \left[ \pi_{t+\tau}^2 + 0.25 (y_{t+\tau} - y^*)^2 \right] $$
donde $y^* = 0.05$ representa una meta positiva de brecha del producto de 5% (por ejemplo, para contrarrestar distorsiones de competencia monopolística).""",
14: r"""### Cuantificación del bienestar: sesgos de inflación y estabilización

1. **Sesgo inflacionario**: como $y^* > 0$, la autoridad discrecional tiene incentivos para generar inflación inesperada y elevar el producto. Quienes fijan precios lo anticipan, por lo que aumenta la inflación promedio sin una ganancia sistemática de producto ($E[\pi^{\text{disc}}] > 0$). Con compromiso, la inflación promedio es cero.
2. **Sesgo de estabilización**: después de un choque de costos, un banco central con compromiso promete mantener el producto por debajo del potencial incluso cuando el choque se disipe. La promesa reduce la inflación esperada y suaviza la disyuntiva actual. Bajo discreción esa promesa no es creíble. Las pérdidas siguientes son esperanzas incondicionales (`loss_criterion="unconditional"`); `stabilization_bias` es su diferencia y toma NaN si no se calculó una solución con compromiso.""",
16: r"""Graficamos las respuestas impulso a un choque de costos de una desviación estándar ($\sigma_u = 1\%$) bajo discreción y compromiso, calculadas a partir de los dos modelos resueltos con sus reglas de política.""",
18: r"""## 3. Modelos híbridos DSGE-VAR (Del Negro & Schorfheide 2004)

Conectamos ahora el modelo DSGE teórico con un VAR(1) no restringido para las series macroeconómicas $Y_t = [y_t, \pi_t, r_t]'$.

Especificamos un modelo neokeynesiano con tres choques: tecnología ($a_t$), costos ($u_t$) y política monetaria ($r_t$). Simulamos $T=250$ trimestres de datos sintéticos **a partir de este mismo modelo** y evaluamos la log-densidad marginal a lo largo de la rejilla de peso a priori $\lambda \in [0.25, 5.0]$:""",
20: r"""Estimamos el DSGE-VAR(1) para valores candidatos del peso a priori $\lambda$. Cuando $\lambda \to \lambda_{\min}$, el estimador se aproxima al VAR no restringido por MCO; cuando $\lambda \to \infty$, impone estrictamente las restricciones teóricas del DSGE:""",
22: r"""### Perfil de optimización de la densidad marginal

Graficamos la log-densidad marginal $\ln p(Y \mid \lambda, \theta)$ como función del peso a priori $\lambda$. Un máximo interior indicaría que algunas, pero no todas, las restricciones del DSGE ayudan; un máximo en el mayor $\lambda$ indica que los datos favorecen tanto peso del modelo como permite la rejilla.""",
24: r"""## 4. Motor de choques anticipados y de noticias (Beaudry & Portier 2006)

¿Cómo reaccionan las economías ante anuncios creíbles de innovaciones futuras?

Examinamos un choque tecnológico anticipado anunciado en $t=0$ con $k=4$ trimestres de adelanto: hoy se conoce que el proceso tecnológico $a_t$ recibirá una innovación unitaria en $t=4$.

La construcción garantiza tres propiedades que verificamos numéricamente:
1. **Estado sin revisión anticipada**: el estado tecnológico exógeno no puede moverse antes de la fecha prevista ($a_t = 0$ para $t < 4$).
2. **Salto de las variables prospectivas**: al formar expectativas racionales, los hogares y las empresas modifican producto, inflación y tasa de política en $t=0$, cuando reciben el anuncio.
3. **Materialización exacta**: en $t=4$ se realiza exactamente la innovación del choque ($a_4 = 1.0$).""",
28: r"""### Descomposición de varianza del error de pronóstico: sorpresa y anticipaciones

¿Qué proporción de la varianza del error de pronóstico de cada variable proviene de la sorpresa y de cada anticipación $k \in \{1, 2, 3, 4\}$? `decompose_news` asigna **varianza unitaria de innovación** tanto a la sorpresa como a cada anticipación. Las participaciones describen la propagación de innovaciones del mismo tamaño, no una estimación de la importancia de las noticias en los datos.""",
32: r"""## Lectura de los resultados

**Lectura de los resultados.**
1. **Macroprocesamiento e identificación.** El preprocesador conserva la rama de indexación, por lo que `pi` pasa a ser un estado predeterminado; hay dos raíces estables para dos estados y el modelo es determinado. **No** está identificado en esta calibración: $J_1$, $J_2$, $J_H$ y $J_S$ tienen rango 5 de 7. Una dirección nula aumenta conjuntamente $\phi_\pi$ y $\phi_y$: con $\sigma = 1$ y $\phi_\pi - \phi_y = 1$, $y_t = -\pi_t$ es una relación exacta de equilibrio (el máximo $|y_t + \pi_t|$ en la respuesta al choque de costos es inferior a 1e-12), y la regla solo revela $\phi_\pi - \phi_y$. Con $\phi_y = 0.25$, el rango de $J_2$ sube a 6 de 7; la dirección nula restante combina $\beta$, $\kappa$ y $\gamma_p$, parámetros de la curva de Phillips que un único choque no permite separar. Superar esta verificación es necesario antes de estimar; no superarla, como aquí, implica que la verosimilitud es plana en esas direcciones.
2. **Discreción frente a compromiso.** Con $y^* = 0.05$, la discreción genera un sesgo inflacionario de +0.024752, mientras el compromiso mantiene la inflación promedio en cero. La pérdida incondicional es 1.5475e-04 bajo discreción y 1.1763e-04 bajo compromiso: el compromiso ahorra 24.0% de la pérdida discrecional (sesgo de estabilización 3.7124e-05). Las respuestas impulso muestran el mecanismo: ante un choque de costos de una desviación estándar, el producto cae 1.495 puntos porcentuales al impacto bajo discreción y 1.188 bajo compromiso. El compromiso mantiene el producto por debajo del potencial durante más tiempo y permite que la inflación caiga bajo cero (mínimo de -0.100 puntos porcentuales), en vez de converger desde valores positivos.
3. **DSGE-VAR.** La log-densidad marginal aumenta en todos los pasos de la rejilla, de 2820.55 con $\lambda = 0.25$ a 2829.49 con $\lambda = 5$; por tanto, $\hat{\lambda} = 5.0$ está en el extremo superior. Es una solución de esquina, no un máximo interior: los datos se simularon con el mismo modelo que centra la distribución a priori, de modo que más peso del modelo resulta favorable. Con datos reales o un modelo a priori mal especificado (véase el ejercicio intermedio), el perfil puede alcanzar su máximo con un $\lambda$ pequeño.
4. **Choques de noticias.** Ante noticias tecnológicas con cuatro trimestres de adelanto, el estado exógeno permanece en cero antes de $t = 4$ (desviación máxima inferior a 1e-12) y luego se materializa exactamente ($a_4 = 1.0000$). Producto (+0.0983), inflación (+0.1271) y tasa de política (+0.2153) reaccionan al anuncio; el producto sigue aumentando hasta un máximo de +0.5756 en el trimestre 3, justo antes de la innovación. Con varianzas de innovación iguales, las noticias explican 96.9% de la varianza del error de pronóstico del producto a 16 trimestres y 81.9% de la inflación; estas participaciones reflejan el supuesto de varianza unitaria, no datos observados.""",
33: r"""## Tu turno

La siguiente celda resuelve de nuevo la discreción para otra meta de producto y las noticias para otro adelanto. Verifica dos propiedades que deben cumplirse para cualquier elección admisible: el sesgo inflacionario es lineal en $y^*$ (el problema es lineal-cuadrático), y la respuesta del producto al anuncio satisface la curva IS iterada hacia adelante, $y_0 = -\sum_{t\ge 0}(r_t - \pi_{t+1}) - a_0$. La regla de Taylor descompone la tasa real acumulada en $(\phi_\pi - 1)\sum_t \pi_t + \phi_y \sum_t y_t + \pi_0$.

**Consignas.**
1. *Básica — origen del sesgo inflacionario.* Reconstruya el modelo de la sección 1 sin indexación (cambie `@#define USE_INDEXATION = 1` por `= 0`). A partir de la condición de primer orden discrecional $\kappa \pi + \lambda_y (y - y^*) = 0$ y de la curva de Phillips estacionaria $(1 - \beta)\pi = \kappa y$, derive la inflación promedio $\bar{\pi}(\kappa)$ en forma cerrada y compárela con `optimal_policy(..., rule="discretion", y_star=target_output).inflation_bias` (deben coincidir a 1e-9). Encuentre analíticamente el $\kappa^*$ que maximiza $\bar{\pi}$ y compruébelo: reconstruya el modelo para $\kappa = 0.01, 0.015, \dots, 0.20$ y verifique que el máximo de la biblioteca esté a no más de un paso de la rejilla de su $\kappa^*$. ¿Por qué desaparece el sesgo cuando $\kappa \to 0$ si $\beta < 1$, por qué disminuye con $\kappa$ grande y qué ocurriría con $\beta = 1$?
2. *Intermedia — ¿detecta $\hat{\lambda}$ la especificación incorrecta?* Simule $T = 250$ trimestres (semilla 123) con una copia de `NK_DSGE_VAR_MOD` que suavice la tasa, `r = 0.8*r(-1) + 0.2*(phi_pi*pi + phi_y*y) + eps_r;`, y reestime el DSGE-VAR manteniendo la distribución a priori centrada en el modelo con regla estática. Verifique que $\hat{\lambda}$ caiga al extremo inferior de la búsqueda mientras el caso correctamente especificado permanezca en el superior, y que el perfil log-MDD mal especificado disminuya en cada paso. ¿Por qué pueden compararse las formas de los perfiles, pero no sus niveles? Repita con las semillas 1 y 2.
3. *Avanzada — cambio de signo del efecto del anuncio.* Ejecute `lead_yt = 2, 4, 8` en la celda siguiente y registre $y_0$. Use la descomposición impresa de la tasa real acumulada para explicar por qué el producto aumenta ante anuncios cercanos, pero cae con $L = 8$. ¿Qué partes de la trayectoria representan relajación prometida y cuáles endurecimiento anticipado? ¿Entre qué dos adelantos cambia el signo?""",
35: r"""## ¿Qué tan exhaustivo es esto?

`puremacro.dsge` integra modelos estructurales con expectativas racionales en Python puro:
- `optimal_policy`: Resuelve la política discrecional Markov-perfecta (Dennis 2007; Oudiz & Sachs 1985) mediante iteración matricial de Riccati y la compara con compromiso desde el estado estacionario (Clarida, Galí & Gertler 1999), cuantificando sesgos de inflación y estabilización. El cuaderno 66 contrasta estos solvers con las soluciones cerradas de Clarida, Galí & Gertler (1999).
- `estimate_dsge_var`: Implementa DSGE-VAR($\lambda$) de Del Negro & Schorfheide (2004), vinculando momentos analíticos entre ecuaciones con distribuciones a priori Wishart invertidas para pruebas de especificación del modelo.
- `news_irf` y `decompose_news`: Incorporan estados mediante matrices compañeras nilpotentes para noticias anticipadas (Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012) y descomponen varianza entre noticias y sorpresas con varianzas de innovación iguales.
- `preprocess_macro`: Preprocesador macro de Dynare en Python puro que resuelve `@#define`, `@#for`, `@#if` e interpolaciones antes de compilar el AST.
- `identification`: Calcula los criterios de identificación por rango de Iskrev (2010) y Komunjer & Ng (2011) sobre jacobianos dinámicos, momentos de autocovarianza y funciones de transferencia espectral.""",
}

for index, text in translations.items():
    es[index] = " [markdown]\n" + "\n".join("# " + line if line else "#" for line in text.splitlines()) + "\n\n"

# Preserve established Spanish code for unchanged cells. Changed numerical
# cells are copied from the corrected English edition before localizing labels.
for index in (5, 7, 11, 15, 17, 23, 25, 29, 34):
    es[index] = en[index]

display_translations = {
    'label="Discretion"': 'label="Discreción"',
    'label="Commitment"': 'label="Compromiso"',
    'r"Inflation $\\pi_t$ (pp)"': 'r"Inflación $\\pi_t$ (pp)"',
    'r"Output Gap $y_t$ (pp)"': 'r"Brecha del producto $y_t$ (pp)"',
    'r"Nominal Rate $r_t$ (pp)"': 'r"Tasa nominal $r_t$ (pp)"',
    'set_xlabel("Quarters")': 'set_xlabel("Trimestres")',
    '"--- Welfare Bias Breakdown ---"': '"--- Desglose de sesgos de bienestar ---"',
    '"--- Interactive Slider Performance ---"': '"--- Desempeño del deslizador interactivo ---"',
    '(machine-dependent)': '(depende del equipo)',
    'Loss saved by commitment (% of disc.)': 'Pérdida ahorrada con compromiso (% de discreción)',
    'Log MDD rises at every grid step': 'Log-MDD aumenta en cada paso de la rejilla',
    '(largest grid value:': '(mayor valor de la rejilla:',
    '"Log Marginal Data Density"': '"Log-densidad marginal de los datos"',
    '(unit innovation)': '(innovación unitaria)',
    '(0 up to rounding)': '(0 salvo redondeo)',
}
for index in range(len(es)):
    if not es[index].startswith(" [markdown]"):
        for old, new in display_translations.items():
            es[index] = es[index].replace(old, new)

result = es_parts[0] + "".join("# %%" + cell for cell in es)
result = re.sub(r"(?m)^plt\.show\(\)\n", "", result)
(root / f"{stem}_es.py").write_text(result)

# Owned EN/ES NB10 and EN NB45: presentation-only style repair. Calculations,
# assertions, cell boundaries and scientific prose remain otherwise unchanged.
for filename in (f"{stem}.py", "10_staggered_did.py", "10_staggered_did_es.py"):
    path = root / filename
    path.write_text(re.sub(r"(?m)^plt\.show\(\)\n", "", path.read_text()))
