**Español** · [English](../dsge_phase_c.md)

# Frontera DSGE: Política Óptima (Discreción vs Compromiso), Modelos Híbridos DSGE-VAR y Perturbaciones de Noticias (News Shocks)

`puremacro` incorpora tres fronteras computacionales de vanguardia para el análisis macroeconómico estructural:

1. **Regímenes de Política Óptima (Discreción vs. Compromiso)**: Resuelve la política discrecional temporalmente consistente (Markov-perfecta) mediante iteración de matrices de Riccati según Dennis (2007), en paralelo con la política de compromiso lineal-cuadrático (LQ) mediante aumentación de multiplicadores de Lagrange y descomposición QZ de Klein (2000) (el plan de Ramsey desde el estado estacionario, cuya ley de movimiento es la regla de perspectiva atemporal; sección 1.3). Formaliza la cuantificación rigurosa del **sesgo de inflación** de Kydland-Prescott / Barro-Gordon y del **sesgo de estabilización**.
2. **Modelos Híbridos DSGE-VAR (Del Negro & Schorfheide 2004)**: Conecta los microfundamentos de equilibrio general del DSGE con la flexibilidad econométrica de los vectores autorregresivos mediante una distribución a priori conjugada Normal-Wishart Invertida centrada en los momentos teóricos de ecuaciones cruzadas $\Gamma_k(\theta)$. Ofrece evaluación analítica exacta en forma cerrada de la log-densidad marginal de los datos $\ln p(Y \mid \lambda, \theta)$, optimización acotada del hiperparámetro $\hat{\lambda} \in [\lambda_{\min}, \infty)$ e identificación estructural mediante la rotación ortonormal del DSGE $Q^*$.
3. **Motor de Perturbaciones Anticipadas y de Noticias (Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012)**: Implementa la aumentación del espacio de estados complementario para anuncios prospectivos $\epsilon_t = \eta_t^0 + \sum_{k=1}^H \eta_{t-k}^k$. Preserva de forma exacta la determinabilidad de punto de silla de Blanchard-Kahn a través de operadores de transición nilpotentes con autovalores idénticamente nulos, calcula trayectorias de impulso-respuesta multilead y efectúa descomposiciones automáticas de varianza.

Todos los algoritmos están implementados en **Python puro** bajo el estricto contrato Pyodide de cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`), con cero dependencias compiladas en C, cero solvers externos y cero licencias comerciales.

---

## Comparativa Metodológica

| Dimensión | Discreción Óptima | Compromiso LQ | DSGE-VAR($\lambda$) | Motor de Noticias (News) |
|:---|:---|:---|:---|:---|
| **Referencia Fundamental** | Dennis (2007); Oudiz & Sachs (1985) | Currie & Levine (1993); Woodford (2003) | Del Negro & Schorfheide (2004) | Beaudry & Portier (2006); Schmitt-Grohé & Uribe (2012) |
| **Concepto de Equilibrio** | Nash Markov-perfecto temporalmente consistente | Compromiso: plan de Ramsey desde el estado estacionario, ley de movimiento de la perspectiva atemporal | VAR bayesiano con a priori conjugada | Expectativas racionales con anuncios futuros |
| **Espacio de Estados** | Estados físicos predeterminados $x_{t-1}$ | Aumentado con precios sombra pasados $\lambda_{t-1}$ | Compañero de rezagos observables $X_t$ | Aumentado con la cola de noticias $V_t$ |
| **Núcleo Computacional** | Iteración matricial de Riccati $\|F_{k+1}-F_k\|_\infty < 10^{-9}$ | Schur generalizado aumentado (Klein QZ) | Lyapunov analítico + Wishart Invertida | Operador de desplazamiento nilpotente $K_H$ ($\sigma=\{0\}$) |
| **Aporte Económico Principal** | Cuantifica sesgos de inflación y estabilización | Mínima pérdida condicional desde el estado estacionario (Ramsey); la pérdida incondicional puede superar a la de discreción (sección 1.4) | Cuantifica desalineación estructural $\hat{\lambda}$ | Distingue anticipación de realización |
| **Compatible con Pyodide** | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) | Sí (`numpy`, `scipy`) |

---

## 1. Regímenes de Política Óptima: Discreción vs Compromiso

### 1.1 Formulación Matemática

Considérese un modelo DSGE lineal en espacio de estados de primer orden con expectativas racionales:
$$A_0 y_t = A_1 y_{t-1} + A_2 \mathbb{E}_t y_{t+1} + B u_t + C \epsilon_t$$
donde el vector de variables endógenas $y_t$ se particiona en $n_x$ variables de estado físicas predeterminadas $x_t$ y $n_z$ variables de control o salto prospectivas $z_t$, $u_t$ es un vector de $m$ instrumentos de política (por ejemplo, la tasa de interés nominal de corto plazo $r_t$), y $\epsilon_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma_\epsilon)$.

El banco central minimiza la función intertemporal de pérdida cuadrática esperada:
$$\mathcal{L}_t = \mathbb{E}_t \sum_{s=0}^\infty \beta^s \left[ \frac{1}{2} y_{t+s}^\top W y_{t+s} + (y_{t+s} - y^*)^\top \Omega (y_{t+s} - y^*) \right]$$
donde $W$ y $\Omega$ son matrices simétricas semidefinidas positivas de ponderación de objetivos, $\beta \in (0, 1)$ es el factor de descuento del hacedor de política, y $y^*$ representa la distorsión del objetivo (como una meta de producto superior al producto natural de precios flexibles).

### 1.2 Política Discrecional (Dennis 2007)

Bajo discreción, la autoridad monetaria carece de tecnología de compromiso para ligar sus decisiones futuras. En cada periodo $t$, el banco central reoptimiza el instrumento $u_t$ tomando como dadas las reglas de formación de expectativas del sector privado. El equilibrio de Nash Markov-perfecto resultante es temporalmente consistente.

El sector privado forma expectativas prospectivas como función lineal de los estados predeterminados:
$$\mathbb{E}_t z_{t+1} = H x_t$$

La función de valor de continuación del banco central satisface la ecuación de Bellman:
$$\mathcal{V}(x_{t-1}) = \frac{1}{2} x_{t-1}^\top V x_{t-1} + d$$
donde la matriz simétrica semidefinida positiva de Riccati $V$ cumple:
$$V = G^\top W_{\text{full}} G + \beta G^\top V G$$
siendo $G$ la matriz de transición de estados en bucle cerrado $y_t = G x_{t-1} + N \epsilon_t$.

La regla de retroalimentación de política converge a:
$$u_t = F x_{t-1}$$
mediante el algoritmo de iteración de funciones de política de Dennis (2007):
$$\|F_{k+1} - F_k\|_\infty < 10^{-9}$$

### 1.3 Compromiso: Plan de Ramsey y Perspectiva Atemporal

Bajo compromiso, el banco central elige un plan contingente de estados y lo cumple. Incorporando multiplicadores de Lagrange $\lambda_t$ para las ecuaciones del sector privado que quedan al eliminar la regla del instrumento, las condiciones necesarias de primer orden producen un sistema aumentado:
$$\begin{bmatrix} y_t \\ \lambda_t \end{bmatrix} = G_{\text{comm}} \begin{bmatrix} y_{t-1} \\ \lambda_{t-1} \end{bmatrix} + N_{\text{comm}} \epsilon_t$$
resuelto directamente mediante el solver QZ de Klein (2000).

Esta ley de movimiento es la misma para el plan de Ramsey elegido en $t_0$ y para la regla de perspectiva atemporal; ambos difieren solo en el multiplicador inicial. El plan de Ramsey fija $\lambda_{-1} = 0$ sea cual sea la historia. La perspectiva atemporal aplica también en $t_0$ la condición de $t \ge 1$, de modo que usa el multiplicador implícito en la política pasada (Jensen y McCallum 2002, ecs. 4a–4c y 5). `lq_commitment` evalúa sus resultados así:

- Las respuestas al impulso y `conditional_loss` parten del estado estacionario, con $y_{-1} = 0$ y $\lambda_{-1} = 0$. Desde el estado estacionario el multiplicador implícito en la política pasada también es cero, así que ahí el plan de Ramsey y la regla atemporal coinciden.
- `loss` promedia sobre la distribución estacionaria de $(y_t, \lambda_t)$, es decir, evalúa la regla atemporal en promedio. Vale NaN, con un `RuntimeWarning`, cuando $G_{\text{comm}}$ tiene una raíz sobre o fuera del círculo unitario, porque no existe distribución estacionaria. La comprobación abarca toda la transición, así que una raíz unitaria confinada a una variable con peso nulo (un nivel de precios, por ejemplo) también da NaN.

El argumento `timeless` de `lq_commitment` y `ramsey_model` nunca cambió el resultado. Está obsoleto y emite un `FutureWarning`.

`ramsey_model(model, objective, ...)` llega a la misma solución a partir de una cadena objetivo como `"pi^2 + 0.25*x^2"`. Deriva simbólicamente las condiciones de primer orden del planificador, pero solo para mostrarlas: `focs` las guarda como ecuaciones legibles en sintaxis de Dynare, y `foc_nodes` como árboles de expresiones. El sistema que resuelve se arma numéricamente, a partir de las matrices lineales del modelo (o de las ecuaciones del `.mod` linealizadas en el estado estacionario) y del hessiano del objetivo en el estado estacionario, con los multiplicadores de estado estacionario iguales a cero. Sobre un modelo lineal comparte, por tanto, las matrices y el solver QZ con `lq_commitment`. Ambos difieren solo en cómo leen el objetivo: `ramsey_model` usa el hessiano de la cadena objetivo, y `lq_commitment` los pesos con un factor 1/2 en la pérdida. Eso solo reescala los multiplicadores, así que compararlos verifica el intérprete del objetivo y la contabilidad de los multiplicadores, no la solución. Para un objetivo de bienestar no lineal cuyos multiplicadores de estado estacionario no son cero, `ramsey_model` es solo una aproximación lineal-cuadrática.

### 1.4 Descomposición de los Sesgos de Bienestar

`discretionary_policy` reporta dos brechas entre discreción y compromiso. Se definen de manera distinta, y ninguna vuelve a resolver el modelo con otro objetivo.

1. **Sesgo de Inflación**: Originado en Kydland & Prescott (1977) y Barro & Gordon (1983). Cuando la pérdida persigue una brecha de producto $y^* > 0$ superior al nivel natural, la inflación de estado estacionario bajo discreción supera la inflación nula de estado estacionario del compromiso atemporal en
   $$\text{Sesgo}_{\pi} = \mathbb{E}[\pi^{\text{disc}}] - \mathbb{E}[\pi^{\text{comm}}] = \frac{\kappa \lambda_y}{\lambda_y (1 - \beta) + \kappa^2} y^*,$$
   donde $\lambda_y$ es el peso de la brecha dividido entre el peso de la inflación, leídos de las entradas de `weights` llamadas `y`, `y_gap`, `output`, `x` o `gap` y `pi`, `inflation`, `infl` o `pfe` (0.25 y 1 si faltan). `inflation_bias` evalúa esta forma cerrada; `y_star` no entra en la dinámica resuelta. La pendiente $\kappa$ se toma de `kappa=` (o `slope_pc=`), luego de un parámetro del modelo llamado `kappa`, `slope_pc`, `kap` o `pc_slope`, luego de la fila de la curva de Phillips en el jacobiano del modelo, y si no, vale 0.5 por omisión; pase `kappa=` cuando la pendiente tenga otro nombre.
2. **Sesgo de Estabilización**: Originado en Clarida, Galí & Gertler (1999) y Woodford (2003). Sin compromiso, el banco central no puede prometer la política dependiente de la historia que estabiliza la inflación presente con un costo menor en producto. `stabilization_bias` es una diferencia de pérdidas, y `loss_criterion` elige cuáles. La opción por defecto, `loss_criterion="unconditional"`, compara los campos `loss`, la pérdida promediada sobre la distribución estacionaria de cada régimen (el criterio que también usa `osr`):
   $$\text{Sesgo}_{\text{estab}} = \mathcal{L}^{\text{disc}} - \mathcal{L}^{\text{comm}}, \qquad \mathcal{L} = \sum_i w_i \operatorname{Var}(y_i)$$
   Su signo no está garantizado. La regla atemporal sigue honrando promesas pasadas y puede perder en promedio: con un factor de descuento lo bastante pequeño, la discreción supera a la regla atemporal bajo la pérdida incondicional siempre que la brecha de producto tenga algún peso y los precios no sean flexibles (Sauer 2010, Proposición 2). En el modelo de referencia de Sauer, que fija el parámetro de Calvo en 0.8722 (una pendiente de 0.02 con $\beta = 0.99$), con peso del producto 0.0625 y choques sin correlación serial, esto ocurre con $\beta < 0.839$. Jensen y McCallum (2002) señalan algo relacionado: bajo este criterio la regla atemporal ni siquiera es la mejor regla de su propia forma. En el modelo de Clarida, Galí y Gertler con pendiente 0.1, peso del producto 0.25 y persistencia del choque de costos 0.5, el sesgo incondicional es positivo con $\beta \ge 0.85$ y negativo con $\beta \le 0.8$ en la malla del [cuaderno 66](https://github.com/jalonso1979/puremacro/blob/main/notebooks/66_optimal_policy_cgg1999_replication_es.py). Con choques de costos sin correlación serial y $\beta = 0.5$ vale $-0.19$.
   `loss_criterion="conditional"` compara en cambio los campos `conditional_loss`, el criterio del propio planificador evaluado desde el estado estacionario:
   $$\mathcal{L}_0 = (1-\beta)\,E_0 \sum_{t \ge 0} \beta^t \sum_i w_i y_{i,t}^2, \qquad y_{-1} = 0,\ \lambda_{-1} = 0.$$
   Desde el estado estacionario, el compromiso es la política plenamente óptima (de Ramsey), así que este sesgo nunca es negativo: en el ejemplo con $\beta = 0.5$ vale $+0.031$. Todo resultado incluye tanto `loss` como `conditional_loss`, sea cual sea el criterio, y el cuaderno 66 contrasta ambos con fórmulas cerradas.

   `stabilization_bias` vale NaN cuando no se calcula (`compare_commitment=False`) o no puede calcularse: falla la resolución bajo compromiso, en cuyo caso `commitment_result` es None, o una pérdida bajo el criterio elegido es NaN. Los dos últimos casos emiten un `RuntimeWarning`, y `summary()` informa el sesgo como no disponible. Un sesgo calculado nunca se sustituye por 0.0.

### 1.5 Ejemplo Ejecutable en Python

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.policy import optimal_policy

# 1. Definir modelo neokeynesiano de 3 ecuaciones
mod = """
var y pi r u;
varexo eps_u;
parameters beta sigma kappa phi_pi phi_y rho_u;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; phi_y = 0.5; rho_u = 0.5;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1));
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y;
  u = rho_u*u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1.0; end;
"""
modelo = build_dynare(mod)

# 2. Resolver Discreción vs Compromiso
res = optimal_policy(
    modelo,
    loss={"pi": 1.0, "y": 0.25},
    rule="discretion",
    instruments="r",
    y_star=0.05,
    compare_commitment=True,
)

print(res.summary())
print(f"Sesgo de Inflación      : {res.inflation_bias:.6f}")
print(f"Sesgo de Estabilización : {res.stabilization_bias:.6f}")

# 3. Graficar respuestas de impulso comparativas
ax = res.plot(compare_commitment=True, periods=16)
ax.figure.savefig("output/dsge_optimal_discretion.png", bbox_inches="tight")
```

### 1.6 Reglas Simples Óptimas: Precisión Alcanzable

`osr(model, rule_params, weights, ...)` ajusta los coeficientes de una regla de instrumento para minimizar la pérdida incondicional $\sum_i w_i \operatorname{Var}(y_i)$, por defecto con Nelder–Mead. La pérdida es cuadrática en su mínimo, $\mathcal{L}(\gamma) \approx \mathcal{L}^*\,[1 + \tfrac{c}{2}((\gamma - \gamma^*)/\gamma^*)^2]$, con curvatura normalizada $c = \gamma^{*2} \mathcal{L}''(\gamma^*)/\mathcal{L}^*$. Por eso una búsqueda que compara valores de la pérdida localiza los coeficientes minimizadores, y la asignación que implican, solo hasta un error relativo de unos $\sqrt{2\varepsilon/c}$, donde $\varepsilon$ es la precisión relativa de la pérdida (en el mejor caso el épsilon de máquina, $2.2\times10^{-16}$). En un problema bien escalado ($c$ del orden de uno) son unos $10^{-8}$, sean cuales sean las tolerancias. Una pérdida plana o mal escalada se localiza con menos precisión. Por ejemplo, con la regla de metas $x_t = -\phi\,\pi_t$ bajo la curva de Phillips $\pi_t = \beta E_t\pi_{t+1} + \kappa x_t + u_t$ con choque de costos AR(1), donde $\phi^* = \kappa/(\alpha(1-\beta\rho)) = 500$ y $c = 0.02$, `osr` se detiene con un error relativo de $10^{-7}$, tanto con las tolerancias nuevas como con las antiguas. La pérdida misma es exacta a unos $10^{-15}$ relativos.

Los valores por defecto `xatol=1e-8` (coeficientes) y `fatol` igual a $10^{-12}$ veces la pérdida inicial (al menos $10^{-12}$) alcanzan ese piso con coeficientes del orden de uno. `xatol` es absoluto, en las unidades de los coeficientes, así que conviene escalarlo con ellos, a unas $10^{-8}$ veces su magnitud: con coeficientes mucho menores que uno, `1e-8` es una tolerancia relativa holgada. Tómese la economía de Clarida, Galí y Gertler con la regla $i_t = g_t/\varphi + \phi_\pi \pi_t + \phi_x x_t$, cotas $\phi_\pi \in [1, 10]$ y $\phi_x \in [0, 10]$, y 25 calibraciones aleatorias. Ahí la respuesta de la brecha de producto a un choque de costos coincide con la regla simple óptima en forma cerrada con un error relativo de $9\times10^{-9}$ (mediana) y $7\times10^{-8}$ (peor caso), y la pérdida con un error menor que $4\times10^{-15}$. Los valores por defecto del propio Nelder–Mead de SciPy ($10^{-4}$ para ambos), que `osr` usaba antes, se detenían con errores de hasta $2.7\times10^{-5}$ cuando el óptimo cae en una cota, con unas 1.8 veces menos evaluaciones de la pérdida. Pase `xatol=1e-4, fatol=1e-4` para reproducirlos, u `options={...}` para fijar las tolerancias de otro optimizador de SciPy. Cuando el óptimo es una cresta, muchas reglas implementan la misma asignación: la asignación queda determinada con esta precisión, pero los coeficientes son solo un punto de la cresta.

`loss_opt` se recalcula resolviendo de nuevo el modelo en los coeficientes devueltos. Si esa resolución falla, o la regla es indeterminada, `loss_opt` vale NaN con un `RuntimeWarning` (nunca el valor de penalización del optimizador), y lo mismo `loss_initial` cuando no pueden evaluarse los momentos de partida. En `variance_table`, `variance_reduction_pct` vale NaN donde no está definido.

---

## 2. Modelos Híbridos DSGE-VAR (Del Negro & Schorfheide 2004)

### 2.1 Motivación y Estructura Conceptual

Mientras que los modelos DSGE estructurales aportan capacidad para análisis contrafactuales microfundamentados, suelen presentar desalineaciones econométricas frente a vectores autorregresivos (VAR) no restringidos. Del Negro & Schorfheide (2004) desarrollan una arquitectura bayesiana en la que las restricciones de ecuaciones cruzadas del modelo DSGE operan como una distribución a priori conjugada sobre un $\text{VAR}(p)$ empírico:
$$Y_t = \Phi_0 + \sum_{l=1}^p \Phi_l Y_{t-l} + u_t, \quad u_t \sim \text{i.i.d.} \mathcal{N}(0, \Sigma)$$

El hiperparámetro de peso $\lambda \in [\lambda_{\min}, \infty)$ modula la influencia de la teoría económica:
- $\lambda \to \lambda_{\min}$: El híbrido coincide asintóticamente con el VAR por MCO no restringido.
- $\lambda \to \infty$: El híbrido converge de forma continua a la representación teórica del modelo DSGE.
- $\hat{\lambda} = \arg\max_\lambda \ln p(Y \mid \lambda, \theta)$: El peso óptimo constituye un estadístico formal de especificación del DSGE.

### 2.2 Autocovarianzas Teóricas y Momentos a Priori

Sea $\Sigma_s$ la matriz de covarianza estacionaria de estados del DSGE, solución de la ecuación discreta de Lyapunov:
$$\Sigma_s = G \Sigma_s G^\top + N \Sigma_\epsilon N^\top$$

Las autocovarianzas poblacionales teóricas para los observables $Y_t$ a rezagos $l = 0, \dots, p$ son:
$$\Gamma_{YY}^*(\theta) = \mathbb{E}_\theta [Y_t Y_t^\top] = Z \Sigma_s Z^\top$$
$$\Gamma_{YX}^*(\theta) = \mathbb{E}_\theta [Y_t X_t^\top] = \begin{bmatrix} Z G \Sigma_s Z^\top & \cdots & Z G^p \Sigma_s Z^\top \end{bmatrix}$$
$$\Gamma_{XX}^*(\theta) = \mathbb{E}_\theta [X_t X_t^\top]$$

Los coeficientes teóricos a priori del DSGE $\Phi^*(\theta)$ y la covarianza de innovaciones $\Sigma^*(\theta)$ satisfacen:
$$\Phi^*(\theta) = \Gamma_{XX}^{*-1}(\theta) \Gamma_{XY}^*(\theta), \quad \Sigma^*(\theta) = \Gamma_{YY}^*(\theta) - \Gamma_{YX}^*(\theta) \Gamma_{XX}^{*-1}(\theta) \Gamma_{XY}^*(\theta)$$

### 2.3 Distribución a Priori Conjugada Normal-Wishart Invertida

La función de densidad a priori adopta la forma analítica conjugada:
$$p(\Sigma \mid \lambda, \theta) \sim \mathcal{IW}\left(\lambda T \Sigma^*(\theta), \; \lambda T - k\right)$$
$$p(\Phi \mid \Sigma, \lambda, \theta) \sim \mathcal{N}\left(\Phi^*(\theta), \; \Sigma \otimes (\lambda T \Gamma_{XX}^*(\theta))^{-1}\right)$$
donde $k = n \cdot p + 1$ (con constante) y $T$ es el tamaño muestral efectivo.

**Condición de Admisibilidad**: Para garantizar la integrabilidad de la distribución Normal-Wishart Invertida propia:
$$\lambda \ge \lambda_{\min} \equiv \frac{k + n}{T}$$

### 2.4 Log-Densidad Marginal de los Datos en Forma Cerrada

La densidad marginal integrada $\ln p(Y \mid \lambda, \theta) = \int p(Y \mid \Phi, \Sigma) p(\Phi, \Sigma \mid \lambda, \theta) d\Phi d\Sigma$ posee la solución cerrada:
$$\begin{aligned}
\ln p(Y \mid \lambda, \theta) = &-\frac{n T}{2} \ln(2\pi) + \ln \left( \frac{\Gamma_n((1 + \lambda)T - k)}{\Gamma_n(\lambda T - k)} \right) \\
&-\frac{n}{2} \ln \left| \frac{\lambda T \Gamma_{XX}^*(\theta) + X^\top X}{\lambda T \Gamma_{XX}^*(\theta)} \right| \\
&-\frac{(1 + \lambda)T - k}{2} \ln \left| (1 + \lambda)T \tilde{\Sigma}(\lambda) \right| \\
&+\frac{\lambda T - k}{2} \ln \left| \lambda T \Sigma^*(\theta) \right|
\end{aligned}$$
donde $\ln \Gamma_n(a) = \frac{n(n-1)}{4} \ln \pi + \sum_{j=1}^n \ln \Gamma(a + \frac{1-j}{2})$ es la función log-gamma multivariada.

### 2.5 Identificación Estructural con la Rotación $Q^*$

Para calcular impulsos-respuesta estructurales, la covarianza reducida $\tilde{\Sigma}$ se vincula con los choques estructurales mediante:
$$\tilde{A}_0 = \operatorname{chol}(\tilde{\Sigma}) \cdot Q^*$$
donde $Q^*$ es una matriz ortonormal ($Q^* Q^{*\top} = I_n$) calculada mediante descomposición QR alineada con la matriz de impacto teórico del DSGE $A_0^{\text{dsge}}$:
$$A_0^{\text{dsge}} = \operatorname{chol}(\Sigma^*) \cdot Q_{\text{dsge}} \implies Q^* = Q_{\text{dsge}}$$

### 2.6 Ejemplo Ejecutable en Python

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.dsge_var import estimate_dsge_var

# 1. Compilar modelo DSGE y simular datos
mod = """
var y pi r a u;
varexo eps_a eps_u eps_r;
parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; phi_y = 0.5;
rho_a = 0.70; rho_u = 0.70;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y + eps_r;
  a = rho_a*a(-1) + eps_a;
  u = rho_u*u(-1) + eps_u;
end;
shocks;
  var eps_a; stderr 0.01;
  var eps_u; stderr 0.01;
  var eps_r; stderr 0.005;
end;
"""
modelo = build_dynare(mod)
datos = modelo.simulate(300, seed=42)[["y", "pi", "r"]]

# 2. Estimar DSGE-VAR(lambda) optimizando sobre la rejilla
res = estimate_dsge_var(
    modelo,
    datos,
    p=1,
    lamb="optimal",
    lambda_grid=[0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
    identification="dsge",
)

print(res.summary())
print(f"Hiperparámetro óptimo hat_lambda : {res.hat_lambda:.4f}")

# 3. Pronósticos y respuestas de impulso estructurales
fc = res.forecast(horizon=8, ci=0.90)
fig, ax = res.plot(kind="irf", shock="eps_a", target="y", horizon=16)
fig.savefig("output/dsge_var_irf.png", bbox_inches="tight")
```

---

## 3. Motor de Perturbaciones Anticipadas y de Noticias (News Shocks)

### 3.1 Fundamentos Macroeconómicos

Las perturbaciones anticipadas o de "noticias" (news shocks) representan información que los agentes económicos reciben en la fecha $t$ sobre una innovación exógena que se materializará físicamente en un horizonte futuro $t+k$ ($k \ge 1$). Aplicaciones canónicas incluyen:
- Rezagos legislativos en reformas tributarias y fiscales (Mertens & Ravn 2011).
- Anuncios de descubrimientos tecnológicos o patentes (Beaudry & Portier 2006).
- Comunicados de política monetaria con orientación prospectiva (Campbell et al. 2012).

El proceso de perturbación exógena $\epsilon_t$ se descompone en choques sorpresa contemporáneos y anuncios anticipados:
$$\epsilon_t = \eta_t^0 + \sum_{k=1}^H \eta_{t-k}^k$$
donde $\eta_t^0$ es la innovación sorpresa inesperada de la fecha $t$, y $\eta_t^k$ es el anuncio efectuado en $t$ respecto a la realización en $t+k$.

### 3.2 Aumentación del Espacio de Estados Complementario

`puremacro` introduce un vector de estados auxiliar para la cola de noticias $V_t = [\nu_{1, t}, \nu_{2, t}, \dots, \nu_{H, t}]^\top \in \mathbb{R}^H$ que evoluciona mediante la recursión lineal complementaria:
$$V_t = K_H V_{t-1} + \eta_t^{\text{noticias}}$$
donde la matriz compañera $K_H$ de orden $H \times H$ es estrictamente triangular superior con unos en la superdiagonal:
$$K_H = \begin{bmatrix}
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & 1 \\
0 & 0 & \cdots & 0 & 0
\end{bmatrix}$$

**Invarianza de Autovalores Nulos**: Al ser $K_H$ estrictamente triangular superior, su polinomio característico es:
$$\det(\lambda I_H - K_H) = \lambda^H = 0 \implies \sigma(K_H) = \{0, 0, \dots, 0\}$$
Los $H$ autovalores aumentados son idénticamente cero, situándose de forma trivial en el interior del círculo unitario ($|\lambda_j| = 0 < 1$). Por consiguiente, la aumentación de noticias **preserva de forma estricta la determinabilidad de Blanchard-Kahn** sin perturbar los autovalores económicos finitos del modelo base.

### 3.3 Las Tres Propiedades Cardinales de los Choques de Noticias

1. **Cero Revisión en Estados Físicos Predeterminados ($t < k$)**:
   Las variables de estado físicas (stock de capital $k_t$ o productividad $a_t$) satisfacen:
   $$\Delta x_t = 0 \quad \forall \; 0 \le t < k$$
2. **Salto Inmediato en Variables de Control Prospectivas ($t = 0$)**:
   Las variables de salto (brecha de producto $y_0$, inflación $\pi_0$, consumo $c_0$) responden en el acto en el momento del anuncio:
   $$z_0 - z_{ss} = L_{\text{aug}} \cdot \eta^k \neq 0$$
3. **Realización Física Exacta en la Fecha $t = k$**:
   En el horizonte $t = k$, la innovación alcanza la cabecera de la cola de noticias ($\nu_{1, k} = \text{magnitud}$), desencadenando la expansión física del estado.

### 3.4 Descomposición Automática de Varianza

La función `decompose_news` calcula la participación en la varianza del error de pronóstico atribuible a sorpresas frente a noticias:
$$\text{FEVD}_v(h) = \frac{\sum_{j=0}^h \left( \text{IRF}_{v, \text{sorpresa}}(j) \right)^2}{\text{Var Total}_v(h)} + \sum_{k=1}^H \frac{\sum_{j=0}^h \left( \text{IRF}_{v, \text{noticia-}k}(j) \right)^2}{\text{Var Total}_v(h)} = 1.0000$$

### 3.5 Sintaxis Dynare `.mod` para Choques Anticipados

`puremacro` analiza de forma nativa la sintaxis estándar de Dynare:
```dynare
shocks;
  var eps_a;
  periods 1:4;
  values 0.01;
end;
```

### 3.6 Ejemplo Ejecutable en Python

```python
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.news import news_irf, decompose_news, plot_news_vs_surprise

mod = """
var y pi r a;
varexo eps_a;
parameters beta sigma kappa phi_pi rho_a;
beta = 0.99; sigma = 1.0; kappa = 0.5; phi_pi = 1.5; rho_a = 0.85;
model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y;
  r = phi_pi*pi;
  a = rho_a*a(-1) + eps_a;
end;
shocks; var eps_a; stderr 0.01; end;
"""
modelo = build_dynare(mod)

# 1. Calcular respuesta a noticia con 4 periodos de anticipación
res_news = news_irf(modelo, shock="eps_a", lead=4, horizon=20)
print(res_news.summary())

# 2. Descomponer varianza entre sorpresas y noticias 1..8
decomp = decompose_news(modelo, shock="eps_a", horizon=20, max_lead=8)
print("\nParticipación en la Varianza (suman 1.0):")
print(decomp.variance_shares.round(4))

# 3. Comparación gráfica multilead
fig, axes = plot_news_vs_surprise(modelo, shock="eps_a", leads=[0, 2, 4, 8])
fig.savefig("output/dsge_news_shocks.png", bbox_inches="tight")
```

---

## 4. Tabla de Referencia de la API

### 4.1 Funciones y Constructores

| Función / Constructor | Módulo | Parámetros Principales | Tipo de Retorno | Descripción |
|:---|:---|:---|:---|:---|
| `optimal_policy(model, loss, rule, ...)` | `puremacro.dsge.policy` | `model`, `loss`, `rule="discretion"\|"commitment"`, `instruments`, `y_star`, `tol=1e-9` | `DiscretionaryPolicyResult` o `PolicyResult` | Despachador principal que resuelve discreción óptima (Dennis 2007) o compromiso LQ (sección 1.3). |
| `discretionary_policy(model, ...)` | `puremacro.dsge.policy` | `model`, `target_vars`, `weights`, `instruments`, `beta=0.99`, `y_star`, `tol=1e-9`, `loss_criterion="unconditional"` | `DiscretionaryPolicyResult` | Algoritmo de iteración de funciones de política de Dennis (2007). |
| `lq_commitment(model, ...)` | `puremacro.dsge.policy` | `model`, `target_vars`, `weights`, `instruments`, `beta=0.99` (`timeless` está obsoleto) | `PolicyResult` | Compromiso óptimo lineal-cuadrático mediante QZ de Klein: la ley de movimiento de la perspectiva atemporal, con respuestas y `conditional_loss` desde el estado estacionario (sección 1.3). |
| `ramsey_model(model, objective, ...)` | `puremacro.dsge.ramsey` | `model_or_dag`, `objective`, `planner_discount=0.99`, `instruments` (`timeless` está obsoleto) | `RamseyResult` | La misma solución bajo compromiso a partir de una cadena objetivo; condiciones de primer orden legibles en `focs` (sección 1.3). |
| `osr(model, rule_params, weights, ...)` | `puremacro.dsge.policy` | `rule_params`, `weights`, `bounds`, `optimizer="Nelder-Mead"`, `xatol=1e-8`, `fatol=None`, `options=None` | `OSRResult` | Regla simple óptima por minimización sin derivadas de la pérdida incondicional (sección 1.6). |
| `estimate_dsge_var(model, data, ...)` | `puremacro.dsge.dsge_var` | `model`, `data`, `p=4`, `lamb=None`, `identification="dsge"`, `lambda_grid=None` | `DSGEVARResult` | Estimador y optimizador de hiperparámetros DSGE-VAR (Del Negro & Schorfheide 2004). |
| `news_irf(model, shock, lead, ...)` | `puremacro.dsge.news` | `model`, `shock`, `lead=0`, `horizon=40`, `size=1.0` | `NewsIRFResult` | Funciones de impulso-respuesta en espacio de estados aumentado para noticias. |
| `decompose_news(model, shock, ...)` | `puremacro.dsge.news` | `model`, `shock=None`, `horizon=40`, `max_lead=8` | `NewsDecompositionResult` | Descomposición automática de varianza entre sorpresas y noticias. |
| `plot_news_vs_surprise(model, shock, ...)` | `puremacro.dsge.news` | `model`, `shock`, `leads=(0, 2, 4, 8)`, `variables=None`, `horizon=40` | `tuple[Figure, Any]` | Asistente de visualización comparativa multilead. |

### 4.2 Estructuras de Datos de Resultados

| Clase de Resultado | Módulo | Atributos Clave | Métodos de Presentación |
|:---|:---|:---|:---|
| `DiscretionaryPolicyResult` | `puremacro.dsge._results` | `F`, `V`, `loss`, `conditional_loss`, `inflation_bias`, `stabilization_bias`, `loss_criterion`, `converged`, `iterations`, `diff`, `commitment_result` | `.summary()`, `.plot(compare_commitment=True)`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `DSGEVARResult` | `puremacro.dsge.dsge_var` | `lamb`, `hat_lambda`, `lambda_min`, `log_mdd`, `log_mdd_grid`, `Phi_star`, `Sigma_star`, `Phi_ols`, `Sigma_ols`, `B0` | `.summary()`, `.plot(kind="irf"\|"mdd"\|"forecast")`, `.irf()`, `.forecast()`, `.fevd()`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `NewsIRFResult` | `puremacro.dsge.news` | `irf`, `surprise_irf`, `shock`, `lead`, `horizon`, `size`, `model` | `.summary()`, `.plot(compare_surprise=True)`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `NewsDecompositionResult` | `puremacro.dsge.news` | `variance_shares`, `dynamic_shares`, `shock`, `horizon`, `max_lead`, `model` | `.summary()`, `.plot()`, `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |

---

## Referencias

- **Barro, R. J., & Gordon, D. B. (1983).** "Rules, discretion and reputation in a model of monetary policy." *Journal of Monetary Economics*, 12(1), 101-121.
- **Beaudry, P., & Portier, F. (2006).** "Stock Prices, News, and Economic Fluctuations." *American Economic Review*, 96(4), 1293-1307.
- **Clarida, R., Galí, J., & Gertler, M. (1999).** "The science of monetary policy: A New Keynesian perspective." *Journal of Economic Literature*, 37(4), 1661-1707.
- **Currie, D., & Levine, P. (1993).** *Rules, Reputation and Macroeconomic Policy Coordination*. Cambridge University Press.
- **Del Negro, M., & Schorfheide, F. (2004).** "Priors from General Equilibrium Models for VARs." *International Economic Review*, 45(2), 643-673.
- **Dennis, R. (2007).** "Optimal Policy in Rational Expectations Models: New Solution Algorithms." *Macroeconomic Dynamics*, 11(1), 31-55.
- **Jensen, C., & McCallum, B. T. (2002).** "The Non-optimality of Proposed Monetary Policy Rules under Timeless-Perspective Commitment." *Economics Letters*, 77(2), 163-168. NBER Working Paper 8882.
- **Klein, P. (2000).** "Using the generalized Schur form to solve a multivariate linear rational expectations model." *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- **Kydland, F. E., & Prescott, E. C. (1977).** "Rules rather than discretion: The inconsistency of optimal plans." *Journal of Political Economy*, 85(3), 473-491.
- **Oudiz, G., & Sachs, J. (1985).** "International Policy Coordination in Dynamic Macroeconomic Models." In *International Economic Policy Coordination*, Cambridge University Press.
- **Sauer, S. (2010).** "Discretion Rather Than Rules? When Is Discretionary Policymaking Better Than the Timeless Perspective?" *International Journal of Central Banking*, 6(2), 1-29.
- **Schmitt-Grohé, S., & Uribe, M. (2012).** "What's News in Business Cycles." *Econometrica*, 80(6), 2733-2764.
- **Woodford, M. (2003).** *Interest and Prices: Foundations of a Theory of Monetary Policy*. Princeton University Press.
