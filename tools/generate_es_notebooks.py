"""Generate idiomatic, rigorous Spanish editions for Notebooks 63, 64, and 65."""
from __future__ import annotations

import re
from pathlib import Path

def parse_jupytext(content: str) -> tuple[str, list[tuple[str, str]]]:
    cell_pattern = re.compile(r"^# %%(\s+\[markdown\])?", re.MULTILINE)
    splits = cell_pattern.split(content)
    header = splits[0]
    cells: list[tuple[str, str]] = []
    i = 1
    while i < len(splits):
        tag = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        cell_type = "markdown" if tag and "[markdown]" in tag else "code"
        cells.append((cell_type, body.strip()))
        i += 2
    return header, cells


def assemble_jupytext(header: str, cells: list[tuple[str, str]]) -> str:
    parts = [header.strip() + "\n\n"]
    for c_type, body in cells:
        if c_type == "markdown":
            parts.append(f"# %% [markdown]\n{body}\n\n")
        else:
            parts.append(f"# %%\n{body}\n\n")
    return "".join(parts).rstrip() + "\n"


# =============================================================================
# NOTEBOOK 63
# =============================================================================

NB63_MD = {0: '# # Juegos arancelarios: bienestar hicksiano, represalias y comprobaciones numéricas\n'
    '#\n'
    '# **¿Cómo distinguimos un candidato de juego arancelario de una mejor respuesta verificada '
    'numéricamente?**\n'
    '#\n'
    '# Esta economía de dos países, balanceada a mano, es sintética. Usamos la misma función de '
    'gasto de consumo y la misma contabilidad auditada en cada comparación. Los países A y B son '
    'ilustrativos; no se predice ninguna política nacional real.',
 1: '# ## El método en matemáticas\n'
    '#\n'
    '# El objetivo es la variación equivalente de consumo $EV_i=e_i(P_0,U_i)-e_i(P_0,U_{i0})$ '
    'respecto a una única referencia fija sin aranceles. EV es cero en esa referencia, por lo que '
    'las ganancias porcentuales se dividen entre el gasto inicial en el consumo seleccionado '
    '$m_{i0}$. La ganancia por desviarse unilateralmente es $r_i=\\max_t '
    'EV_i(t,\\tau_{-i})-EV_i(\\tau)$. Un candidato numérico de Nash debe satisfacer tanto '
    '$\\max_i|BR_i(\\tau_{-i})-\\tau_i|\\leq\\epsilon_\\tau$ como $\\max_i '
    'r_i/m_{i0}\\leq\\epsilon_r$ en el intervalo declarado.',
 2: '# ## Intuición\n'
    '#\n'
    '# **Intuición.** Los pasos pequeños por amortiguación pueden ocultar grandes incentivos para '
    'desviarse. Un arancel óptimo en la frontera depende del límite impuesto. El dilema del '
    'prisionero debe comprobarse en la matriz de pagos con acciones fijas; las represalias no lo '
    'implican. Un equilibrio general fallido no proporciona un pago.',
 3: '# ## Código resuelto\n'
    '#\n'
    '# Construimos una tabla balanceada con un sector productor por país, dos canastas de consumo '
    '(categorías 0 y 2), inversión separada (1), impuestos domésticos y saldos externos no nulos. '
    'Las remuneraciones fijas del trabajo y el capital cierran las cuentas de producción.',
 5: '# ### Una búsqueda unilateral acotada\n'
    '#\n'
    '# Seleccionamos explícitamente `metric="hicksian_ev"`. El modelo usa abastecimiento '
    'intermedio CES con sigma 2, canastas finales Leontief y devolución fiscal de suma fija. El '
    'objetivo es bienestar de consumo condicional; se excluye la inversión. Todas las búsquedas '
    'comparten una referencia auditada. Los porcentajes dividen EV entre el gasto inicial en '
    'consumo.',
 7: '# ### Diagnósticos numéricos de mejores respuestas\n'
    '#\n'
    '# Cada intento de equilibrio se audita. La recuperación automática prueba Newton, el método '
    'híbrido y luego continuación de Keller, sin relajar la tolerancia; si todos fallan, se '
    'produce un error. La convergencia exige mejores respuestas simultáneas en el vector final y '
    'ganancias por desviación normalizadas por consumo. El resultado puede ser inconcluso; debe '
    'comunicarse sin afirmar un teorema de Nash.',
 10: '# ## Lectura de los resultados\n'
     '#\n'
     '# La curva monetaria de bienestar tiene una referencia igual a cero. Su representación '
     'porcentual usa consumo inicial fijo; no divide entre EV inicial. Cada celda de pagos usa las '
     'mismas dos acciones por jugador. La convergencia corresponde a la búsqueda numérica '
     'declarada: revisa el límite arancelario, las fronteras, la resolución y la ganancia final '
     'por desviarse. El ejercicio no demuestra aranceles óptimos únicos ni interiores.',
 11: '# ## Tu turno\n'
     '#\n'
     '# Cambia el límite y compara el candidato y sus diagnósticos. Una respuesta en el límite '
     'superior puede cambiar con la restricción institucional.',
 13: '# ## ¿Qué tan exhaustivo es esto?\n'
     '#\n'
     '# Es un ejercicio docente reproducible, no un pronóstico empírico de una guerra comercial. '
     'Un equilibrio CES escalar derivado por separado y una minimización primal de gasto validan '
     'una cuadrícula completa de 41 por 41 perfiles; consulta '
     '`reviews/2026-09-20-hicksian-policy/REPORT.md`. Las búsquedas locales pueden omitir máximos '
     'estrechos. EV hicksiana está disponible para las preferencias y la contabilidad declaradas; '
     'la descomposición causal TOT/Alloc/TariffRec y la certificación de teoremas siguen sin estar '
     'disponibles. Consulta `docs/trade_policy.md`.'}


# =============================================================================
# NOTEBOOK 64
# =============================================================================

NB64_MD = {
    0: r"""# # Singularidades, Bifurcaciones de Pliegue y Continuación por Longitud de Pseudosegmento de Keller en Equilibrio General Multirregional
#
# **Pregunta motivadora / Motivating question:**
# **¿Cómo resuelven los economistas cuantitativos del comercio internacional las singularidades numéricas, las bifurcaciones de pliegue y el estancamiento numérico en microeconomías mal condicionadas cuando las rigideces de producción a corto plazo empujan los modelos de equilibrio general hacia regímenes profundamente inelásticos?**
#
# En la modelización del comercio en equilibrio general computable (CGE), los sistemas multipaís y multisectoriales dependen habitualmente de solucionadores estándar de Newton-Raphson para determinar los salarios de equilibrio de los factores, los precios de los bienes y los saldos comerciales multilaterales. Aunque el método de Newton exhibe convergencia cuadrática en la vecindad de equilibrios regulares, los choques económicos del mundo real con frecuencia empujan a las economías hacia dominios paramétricos extremos y mal condicionados. Dos modos profundos de falla numérica emergen en el análisis de frontera del comercio internacional:
#
# 1. **Rigidez a Corto Plazo y Bifurcaciones de Pliegue Nodo-Silla:** Cuando las elasticidades de sustitución de insumos intermedios colapsan hacia una tecnología de Leontief ($\sigma \to 0$), la sustitución en las cadenas de suministro se apaga. En torno al umbral crítico $\sigma_{\text{fold}} \approx 0.1238$ bajo choques arancelarios crecientes ($\tau \in [0.20, 0.25]$), la variedad de equilibrio general se pliega sobre sí misma. En este punto singular de retorno, el jacobiano estándar del estado degenera ($\det J_x \to 0$, $\kappa_2(J_x) > 10^5$). El método estándar de Newton-Raphson intenta dar un paso a través del pliegue con un valor fijo del parámetro, lo que causa divisiones por singularidad y divergencia explosiva.
# 2. **Topología de Red Mal Condicionada y Estancamiento en Microeconomías:** En las matrices globales de insumo-producto (como la OECD ICIO de 77 países), las microeconomías sumamente abiertas (como Chipre, `CYP`, que representa $< 0.03\%$ del producto mundial con una apertura comercial $> 70\%$) introducen una rigidez numérica extrema. En el sistema global de salarios de equilibrio, Chipre corresponde a una dirección espectral casi nula. Las iteraciones de Newton no regularizadas proyectan los errores residuales globales sobre este modo frágil, produciendo fluctuaciones salariales desmedidas ($\Delta \ln w_{\text{CYP}} > 10$) que desestabilizan la totalidad del sistema global.
#
# Para superar estas patologías estructurales, la teoría del comercio moderno incorpora algoritmos avanzados de continuación y estabilización numérica: el filtro de viabilidad espectral de Hawkins-Simon para descartar programas arancelarios inviables, la Continuación por Longitud de Pseudosegmento de Keller (PAC) sobre un jacobiano aumentado de dimensión $(K+1) \times (K+1)$ para sortear suavemente las bifurcaciones de pliegue, y el truncamiento modal por proyección SVD con aceleración de Anderson para doblegar la rigidez de las microeconomías. Asimismo, los teoremas de acotamiento analítico (Teoremas 1–4) delimitan la envolvente matemática exacta que acota las rigideces de Leontief y la sustitución flexible CES.""",

    1: r"""# ## El método en matemáticas — Continuación Numérica, Sistemas Aumentados y Singularidades
#
# **1. Bifurcaciones de Pliegue Nodo-Silla y Puntos Singulares de Retorno.**
# Sea $F(x, \lambda) = 0$ el sistema de equilibrio general de dimensión $K$, donde $x = (\ln w, \ln r, \ln p, \ln y, T, XN) \in \mathbb{R}^K$ es el vector de estado endógeno y $\lambda \in [0, 1]$ parametriza el choque de política comercial ($\tau(\lambda) = (1-\lambda)\tau_0 + \lambda \tau_{\text{target}}$).
#
# Una bifurcación de pliegue nodo-silla ocurre en un punto $(x^*, \lambda^*)$ donde:
# $$ F(x^*, \lambda^*) = 0, \quad \det\left(J_x(x^*, \lambda^*)\right) = 0, \quad \frac{\partial F}{\partial \lambda}(x^*, \lambda^*) \notin \text{rango}\left(J_x(x^*, \lambda^*)\right) $$
# En el punto de retorno, la tangente a lo largo del eje paramétrico se anula ($\tau_\lambda = \frac{d\lambda}{ds} \to 0$ e invierte su signo $\tau_\lambda < 0$). El número de condición del jacobiano explota:
# $$ \kappa_2(J_x) = \frac{\sigma_{\max}(J_x)}{\sigma_{\min}(J_x)} > 10^5 $$
# El paso clásico de Newton-Raphson $\Delta x = -J_x^{-1} F(x; \lambda)$ falla catastróficamente debido a la no invertibilidad del operador.
#
# **2. Continuación por Longitud de Pseudosegmento de Keller (PAC).**
# Keller (1977) reformula el problema de continuación parametrizando tanto el estado $x(s)$ como el parámetro $\lambda(s)$ en función de la longitud de arco $s$. La variedad solución se recorre resolviendo el sistema aumentado $G(x, \lambda; s) = 0$ de dimensión $(K+1) \times (K+1)$:
# $$ G(x, \lambda; s) = \begin{bmatrix} F(x, \lambda) \\ \tau_x^\top W^2 (x - x_k) + \tau_\lambda (\lambda - \lambda_k) - \Delta s \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ 0 \end{bmatrix} $$
# donde $(\tau_x, \tau_\lambda)$ es el vector tangente unitario que satisface el sistema tangente aumentado:
# $$ \begin{bmatrix} J_x & \frac{\partial F}{\partial \lambda} \\ \tau_x^\top W^2 & \tau_\lambda \end{bmatrix} \begin{bmatrix} v_x \\ v_\lambda \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ 1 \end{bmatrix}, \qquad (\tau_x, \tau_\lambda) = \frac{(v_x, v_\lambda)}{\|(v_x, v_\lambda)\|_2} $$
# Por el Lema de Aumento de Frontera (Bordering Lemma), el jacobiano aumentado orlado $J_{\text{aug}} \in \mathbb{R}^{(K+1) \times (K+1)}$ permanece no singular ($\kappa_2(J_{\text{aug}}) < 10^4$) en el punto de retorno, permitiendo que los pasos correctores de Newton converjan con precisión de máquina ($\|F(x)\|_\infty < 10^{-10}$).
#
# **3. Filtro de Viabilidad Espectral de Hawkins-Simon.**
# Antes de ejecutar los solucionadores no lineales, el sistema evalúa el radio espectral $\rho(\mathbf{B}_\tau)$ de la matriz insumo-producto aumentada por aranceles:
# $$ B_{\tau, ij} = \frac{a_{ij}(1 + \tau_{ij})}{1 - t_j} $$
# Empleando la iteración de potencias desplazada de Collatz-Wielandt en tiempo $\mathcal{O}(M^2)$ ($< 0.15$s), el filtro acota $\rho(\mathbf{B}_\tau)$:
# $$ \min_i \frac{[\mathbf{B}_\tau v]_i}{v_i} \le \rho(\mathbf{B}_\tau) \le \max_i \frac{[\mathbf{B}_\tau v]_i}{v_i} $$
# Si $\rho(\mathbf{B}_\tau) \ge 1.0$, se viola la condición de Hawkins-Simon (1949), lo que indica que ningún vector de precios no negativos puede sostener la producción. Los esquemas arancelarios prohibitivos ($\tau \ge 8.0$) se rechazan de inmediato antes de iniciar la resolución numérica.
#
# **4. Estabilización de Redes Mal Condicionadas (Chipre CYP).**
# En una microeconomía abierta con participación del producto $\omega_c \ll 1$ y elevada penetración de importaciones, los gradientes salariales de factores se tornan mal condicionados. Tres mecanismos de estabilización aseguran la convergencia:
# - **Truncamiento Modal de Componentes SVD:** Se descompone el jacobiano equilibrado $J = U \Sigma V^\top$. Se acotan los coeficientes de proyección modal $c_k = (U^\top f)_k / \sigma_k$ a $|\tilde{c}_k| \le 20.0$, y se limita el desplazamiento salarial físico a $\|\Delta \ln w\|_\infty \le 0.30$.
# - **Aceleración de Anderson:** Ponderaciones de mezcla de mínimos cuadrados con restricciones KKT de profundidad $m=4$, $\gamma \in \mathbb{R}^m$, resolviendo $\min_\gamma \|\sum_{i=1}^m \gamma_i f_i\|_2^2$ s.a. $\sum \gamma_i = 1.0$, amortiguando las oscilaciones dinámicas en los salarios entre países.
# - **Solucionador de Variedad Condicional 1D Desacoplado:** Particiona el sistema en el resto del mundo (76 países) y la ecuación salarial unidimensional de Chipre, resolviendo la variedad macroeconómica condicionalmente y evaluando a Chipre mediante un buscador de raíces secante acotado en 1D.
#
# **5. Envolvente de Acotamiento Analítico de los Teoremas 1–4.**
# - **Teorema 1 (Inversión de Cotas en Equilibrio General):** Bajo una Pequeña Economía Abierta (SOE), el PIB real de Leontief a doble deflación es invariante ($\Delta \text{RGDP}_{Leo} = 0$), mientras que el modelo CES flexible se contrae debido a la pérdida de peso muerto de Harberger:
#   $$ \text{DWL} = \frac{1}{2} \sigma \frac{s_M}{1 - s_M} Y_0 \tau^2 \implies \Delta \text{RGDP}_{Leo} \ge \Delta \text{RGDP}_{CES} $$
#   En una Gran Economía Abierta (LOE), las ganancias de términos de intercambio superan la pérdida de eficiencia asignativa para aranceles moderados, invirtiendo el ordenamiento ($\Delta W_{CES} > 0 = \Delta W_{Leo}$).
# - **Teorema 2 (Límite Superior de Precios en Fábrica y Amortiguamiento Salarial):** Condicionado a los costos de los factores, los costos de Leontief acotan superiormente a los de CES ($c_{Leo} \ge c_{CES}$). En equilibrio general, el amortiguamiento salarial ($w_{CES} > w_{Leo}$) invierte los precios por encima del umbral de intensidad laboral $\bar{s}_{L, cj} = \frac{\Delta \ln q}{\Delta \ln w + \Delta \ln q}$.
# - **Teorema 3 (Cota Inferior de Destrucción de Exportaciones Extranjeras):** Los volúmenes físicos de importación bajo Leontief dominan a los de CES ($M_{Leo} \ge M_{CES}$), estableciendo una cota inferior sobre la destrucción de exportaciones extranjeras:
#   $$ |\Delta X_{Leo}| \le |\Delta X_{CES}| $$
# - **Teorema 4 (Dominancia de Ingresos Arancelarios y Restitución del 99.3%):** La nula erosión de la base impositiva garantiza que $\text{TR}_{Leo} > \text{TR}_{CES}$ para todo $\tau > 0$. El modelo CES exhibe un pico único de Laffer en $\tau^* = \frac{1}{\sigma - 1}$, y la restitución en suma fija provee una compensación del 99.3% ($0.990 \le \text{offset} \le 0.996$).""",

    2: r"""# ## Intuición
#
# **Intuición.** ¿Por qué fallan los modelos de equilibrio general del comercio con elasticidades bajas, y por qué la continuación por longitud de pseudosegmento tiene éxito donde el método estándar de Newton diverge?
#
# 1. **La física de un pliegue nodo-silla:**
#    Cuando una economía opera con una sustitución estándar ($\sigma \in [2, 5]$), imponer un arancel del 25% induce a los productores domésticos a sustituir insumos intermedios foráneos por proveedores locales. Conforme $\sigma \to 0.1238$ (aproximándose a la rigidez de Leontief), la sustitución de insumos es prácticamente imposible. Cada unidad monetaria de costo arancelario debe ser absorbida por los costos unitarios de las empresas. En equilibrio general, al subir los costos, el poder adquisitivo de los consumidores cae, contrayendo la demanda y encogiendo las ventas. Cuando el arancel alcanza un umbral crítico, la espiral de costos supera la generación de ingresos: no existe ningún precio a esa tasa arancelaria fija que vacíe los mercados de factores. La variedad de equilibrio "se pliega hacia atrás" —existen dos equilibrios por debajo del arancel crítico y ninguno por encima de él—. El método estándar de Newton avanza horizontalmente a lo largo del eje del arancel, cayendo al abismo de la no existencia de soluciones.
#
# 2. **La geometría de longitud de arco de Keller:**
#    El método de pseudosegmento de Keller reconoce que el parámetro $\lambda$ (el multiplicador arancelario) no es un piso horizontal fijo, sino simplemente otra coordenada en la variedad solución. Al introducir una coordenada de longitud de arco $s$, el solucionador se desplaza tangencialmente sobre la superficie curva de soluciones. Cuando la variedad gira en sentido inverso ($\tau_\lambda < 0$), el jacobiano aumentado orlado permanece completamente suave y bien condicionado. El algoritmo bordea con naturalidad la "nariz" del pliegue, siguiendo la trayectoria a través de la singularidad sin incurrir en divisiones por cero.
#
# 3. **El dilema de rigidez en microeconomías (Chipre):**
#    En una red de comercio internacional como la OECD ICIO, Estados Unidos, China y Alemania cuentan con mercados internos inmensos que anclan los niveles salariales globales. Por el contrario, una microeconomía abierta como Chipre (`CYP`) produce una fracción ínfima del producto mundial pero importa más del 70% de su consumo. En términos numéricos, la ecuación salarial de Chipre actúa como un resorte microscópico y ultrasensible adherido a un cuerpo rígido masivo. Los pasos de Newton no condicionados intentan suprimir los pequeños desequilibrios monetarios de Chipre imponiendo ajustes salariales porcentuales gigantescos ($\Delta \ln w > 10$), lo cual vuelve negativos los precios domésticos de inmediato y colapsa el cálculo. El truncamiento modal SVD actúa como un amortiguador numérico: detecta el autovector de alta frecuencia de Chipre y restringe su desplazamiento a $|\Delta \ln w| \le 0.30$, mientras que la aceleración de Anderson suaviza las oscilaciones salariales entre países.
#
# 4. **La envolvente de acotamiento de la teoría económica:**
#    La tecnología de Leontief representa el límite rígido absoluto a corto plazo (las recetas de producción no se pueden alterar), mientras que CES representa la adaptación flexible a largo plazo. Los Teoremas 1–4 proveen un emparedado analítico que acota la realidad económica: Leontief preserva el producto bruto y los volúmenes de exportación pero maximiza la inflación de costos y los ingresos arancelarios, mientras que CES introduce triángulos de pérdida de peso muerto, erosión de la base gravable y curvas de Laffer con máximos definidos.""",

    12: r"""# ## Lectura de los resultados
#
# **Lectura de los resultados.** Los resultados computacionales ilustran los mecanismos matemáticos que gobiernan las singularidades numéricas, la continuación y el acotamiento teórico en sistemas de comercio multirregional:
#
# 1. **Viabilidad Previa de Hawkins-Simon (Experimento 1):** La iteración de potencias desplazada de Collatz-Wielandt evalúa la productividad de la matriz insumo-producto en $0.003$ segundos, holgadamente por debajo del límite de $0.15$ segundos. Para un arancel del 25%, el radio espectral $\rho(\mathbf{B}_\tau) = 0.6935 < 1.0$, acotado estrechamente por $[0.6894, 0.6958]$, certificando la existencia de precios de equilibrio. Cuando se introduce un esquema inviable ($\tau = 8.0$), $\rho(\mathbf{B}_\tau) = 1.5647 \ge 1.0$, y el filtro emite inmediatamente un `ValueError`, protegiendo a los solucionadores no lineales de una divergencia irrecuperable.
# 2. **Cruce del Pliegue Nodo-Silla mediante Keller PAC (Experimento 2):** En la variedad canónica con pliegue, el método estándar de Newton experimenta $\kappa_2(J_x) > 10^5$ y $\det(J_x) \to 0$, haciendo que la longitud del paso explote por encima de $10^3$. La Continuación por Longitud de Pseudosegmento de Keller aumenta el sistema a dimensión $(K+1) \times (K+1)$, manteniendo el número de condición aumentado $\kappa_2(J_{\text{aug}}) < 10^4$. Keller PAC atraviesa el punto de retorno hacia la rama invertida ($\tau_\lambda \le 0$) con un residuo máximo a lo largo de la trayectoria de $< 10^{-10}$.
# 3. **Continuación PAC en CGE (Experimento 3):** Con una elasticidad a corto plazo $\sigma_{\text{fold}} = 0.1238$ y un choque arancelario del 25%, `solve_keller_pac` sortea el punto de retorno del modelo CGE en menos de 15 pasos. El equilibrio refinado satisface $\|F(x)\|_\infty = 2.84 \times 10^{-14} \ll 10^{-10}$, verificando el vaciado de mercados con precisión de máquina float64.
# 4. **Estabilización de la Microeconomía de Chipre (Experimento 4):** En el sistema salarial empírico de 77 países, los pasos de Newton sin regularizar en el modo rígido de Chipre explotan hasta $2.8 \times 10^6$. El truncamiento modal SVD acota estrictamente los coeficientes modales a $|c_k| \le 20$ y limita los desplazamientos factoriales a $\|\Delta \ln w\|_\infty \le 0.30$. El solucionador condicional desacoplado en 1D converge con un residuo de $4.44 \times 10^{-16}$, mientras que la aceleración de Anderson logra la contracción del residuo de punto fijo en 8 iteraciones.
# 5. **Motor de Acotamiento Teórico (Experimento 5):** `verify_theorems_1_to_4` valida rigurosamente los cuatro teoremas de acotamiento: la invariancia del PIB real de Leontief acota superiormente a la contracción CES (Teorema 1); los precios en fábrica se amortiguan mediante la deflación salarial por encima del umbral de intensidad laboral $\bar{s}_L = 42.8\%$ (Teorema 2); la destrucción de exportaciones extranjeras bajo Leontief acota inferiormente a la de CES (Teorema 3); y los ingresos arancelarios de Leontief dominan a los de CES con una compensación exacta por restitución en suma fija del 99.3% y un pico analítico de Laffer en $\tau^* = 1.0$ para $\sigma = 2.0$ (Teorema 4).""",

    14: r"""# **Ejercicios y reflexiones.**
# 1. *Básico:* Aumente `sigma_custom` de $0.1238$ a $0.50$. Observe cómo el grado de curvatura del pliegue disminuye, permitiendo que Keller PAC recorra la trayectoria de continuación en menos pasos de longitud de arco.
# 2. *Intermedio:* Reduzca `max_disp_custom` de $0.30$ a $0.15$. Verifique que el solucionador desacoplado de la variedad 1D de Chipre respete el límite de desplazamiento más estricto sin generar divergencia numérica.
# 3. *Avanzado:* Disminuya el tamaño de paso inicial `ds_custom` a $0.02$ y compruebe que el número de condición aumentado permanezca estrictamente acotado bajo $\kappa_2(J_{\text{aug}}) < 10^4$ a lo largo de todo el punto de retorno.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro.trade` proporciona una suite completa de continuación numérica, estabilización y acotamiento teórico:
# - `puremacro.trade.solver`: Solucionadores CGE multialgorítmicos robustos que incluyen la Continuación por Longitud de Pseudosegmento de Keller (`solve_keller_pac`), el filtro de viabilidad espectral de Hawkins-Simon (`check_hawkins_simon_viability`), truncamiento modal por SVD (`svd_clamped_newton_step`), aceleración de Anderson (`anderson_accelerate`) y estabilización desacoplada 1D para microeconomías (`solve_cyprus_manifold_step`).
# - `puremacro.trade.policy_analytics`: Motor de verificación de teoremas analíticos de acotamiento (`verify_theorems_1_to_4`), descomposición aditiva exacta en 3 vías de la Variación Equivalente hicksiana (`decompose_hicksian_ev_3way`) y seguimiento de vulnerabilidad en cadenas de suministro mediante Ghosh (`compute_supply_chain_vulnerability`).
# - `puremacro.trade.optimal_tariffs`: Evaluación del bienestar soberano nacional (`evaluate_national_welfare`), aranceles óptimos unilaterales de términos de intercambio (`compute_unilateral_optimal_tariff`), iteración arancelaria de Nash multilateral no cooperativa (`solve_multilateral_nash_tariffs`) y matrices de pagos en forma normal del Dilema del Prisionero $2 \times 2$ (`compute_welfare_payoff_matrix`).
# - `puremacro.trade.data`: Adaptadores de ingesta MRIO para bases de datos internacionales OECD ICIO, Eurostat FIGARO, EXIOBASE 3, WIOD y Eora26 (`load_icio_data`, `load_figaro`, `load_exiobase`, `load_wiod`, `load_eora`).
# - `puremacro.trade.regularize`: Regularización contable walrasiana, piso no negativo de valor agregado, reconciliación TLS de residuos y balanceo matricial mediante RAS, GRAS y optimización cuadrática restringida (`regularize_mrio_table`, `balance_ras`, `balance_gras`, `balance_quadratic`).""",
}


# =============================================================================
# NOTEBOOK 65
# =============================================================================

NB65_MD = {0: '# # Propagación de costos en CGV, procedencia y límites de validación del bienestar\n'
    '#\n'
    '# **¿Cómo distinguir un contrafactual comercial resuelto de una afirmación de bienestar sin '
    'sustento?**\n'
    '#\n'
    '# Este tutorial utiliza tablas pequeñas generadas con estructuras de OECD ICIO, FIGARO y EXIOBASE. Son '
    'datos sintéticos para enseñanza, no observaciones de esas bases.',
 1: '# ## El método en matemáticas\n'
    '#\n'
    '# Con coeficientes de insumos fijos $A$, la propagación de costos satisface $dp=(I-A^T)^{-1}dv$, '
    'manteniendo fijos los precios de factores. El equilibrio general también modifica precios, cantidades, '
    'transferencias e impuestos. Un residuo pequeño solo verifica las ecuaciones especificadas.',
 2: '# ## Intuición\n'
    '#\n'
    '# **Intuición.** Un aumento de costos llega a los clientes mediante sus compras de insumos. Tablas '
    'sintéticas similares pueden generar cascadas similares por construcción; esto no es evidencia empírica '
    'entre bases. La variación equivalente (equivalent variation) requiere una función de gasto y utilidad '
    'inicial independientes.',
 3: '# ## Código resuelto\n'
    '#\n'
    '# Generamos los datos de forma explícita y mostramos su procedencia antes de reportar resultados.',
 5: '# ### Procedencia\n'
    '#\n'
    '# Los lectores de datos reales requieren un archivo por defecto. `fallback_to_synthetic=True` permite '
    'datos generados solo con autorización explícita. Las calibraciones conservan procedencia, conversiones '
    'y ajustes. La división entre trabajo y capital es un supuesto del modelo.',
 7: '# ### Propagación de costos con precios fijos\n'
    '#\n'
    '# La cota espectral superior verifica productividad de estas matrices no negativas. El residuo del '
    'sistema lineal verifica este cálculo, no un teorema de bienestar.',
 9: '# ### Un experimento arancelario resuelto\n'
    '#\n'
    '# Usamos una tabla didáctica separada, balanceada a mano, con dos países y categorías explícitas de '
    'consumo e inversión. Las tablas sintéticas anteriores ilustran redes y no son la fuente de este ejemplo '
    'de bienestar. Primero reportamos PIB nominal e ingresos arancelarios; después evaluamos el consumo.',
 11: '# ### Bienestar hicksiano del consumo\n'
     '#\n'
     '# La categoría 0 representa consumo; se excluye inversión. Con una canasta Leontief, la utilidad es su '
     'cantidad y la función de gasto es $e(P,U)=PU$. Por tanto, $EV=P_0(C_1-C_0)$, positiva para ganancias. '
     'La interfaz general también admite preferencias Cobb–Douglas entre categorías de consumo seleccionadas '
     'explícitamente. Las transferencias incluyen los aranceles una sola vez. La atribución a precios, '
     'ingreso factorial y transferencias compara estados finales; no es un teorema causal de términos de '
     'intercambio y eficiencia.',
 13: '# ### Presentación de resultados numéricos',
 15: '# ## Lectura de los resultados\n'
     '#\n'
     '# Las tablas identifican los datos como sintéticos. El residuo y las cotas espectrales son '
     'diagnósticos diferentes. EV y CV del consumo se obtienen de una función de gasto explícita y se '
     'contrastan con la valoración directa de cantidades. La nueva atribución utiliza precios de comprador, '
     'ingreso factorial y transferencias. La descomposición histórica TOT/Alloc y la certificación de '
     'teoremas siguen sin estar disponibles.',
 16: '# ## Tu turno\n'
     '#\n'
     '# Repite el experimento con otra semilla u otro proveedor. Compara CES y Leontief usando los mismos '
     'aranceles, cierre fiscal y tolerancia. Verifica los flujos reportados contra cada tecnología antes de '
     'interpretar diferencias.',
 17: '# ## ¿Qué tan exhaustivo es esto?\n'
     '#\n'
     '# Este ejemplo enseña procedencia, propagación de coeficientes fijos y una solución GE pequeña. No '
     'valida archivos nativos MRIO, comparaciones empíricas de bienestar, equilibrio de Nash o una '
     'descomposición causal de términos de intercambio y eficiencia. EV depende de las preferencias y del '
     'cierre fiscal declarados. Consulta `docs/STRUCTURAL_VALIDATION_STATUS.md` para los límites actuales.'}


def generate_es_notebook(stem: str, md_dict: dict[int, str], knob_cell_idx: int) -> None:
    en_path = Path(f"notebooks/{stem}.py")
    es_path = Path(f"notebooks/{stem}_es.py")

    header, cells = parse_jupytext(en_path.read_text(encoding="utf-8"))
    es_cells = []

    for idx, (c_type, body) in enumerate(cells):
        if c_type == "markdown":
            if idx in md_dict:
                es_cells.append(("markdown", md_dict[idx]))
            else:
                es_cells.append(("markdown", body))
        else:
            if idx == knob_cell_idx:
                # Prepend Tu turno comment
                knob_body = "# Tu turno: personaliza parámetros e inspecciona el comportamiento del modelo\n" + body
                es_cells.append(("code", knob_body))
            else:
                es_cells.append(("code", body))

    output = assemble_jupytext(header, es_cells)
    es_path.write_text(output, encoding="utf-8")
    print(f"Generated {es_path.name} with {len(es_cells)} cells ({len([c for c in es_cells if c[0] == 'code'])} code cells)")


def main() -> None:
    print("Generating Spanish editions for Notebooks 63, 64, and 65...")
    generate_es_notebook("63_trade_wars_and_nash_tariffs", NB63_MD, knob_cell_idx=12)
    generate_es_notebook("64_singularities_and_keller_pac", NB64_MD, knob_cell_idx=13)
    generate_es_notebook("65_gvc_cascades_and_welfare_decomposition", NB65_MD, knob_cell_idx=13)
    print("All 3 Spanish editions generated successfully!")


if __name__ == "__main__":
    main()
