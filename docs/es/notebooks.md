> 🇬🇧 [English](../notebooks.md) · 🇪🇸 Español

# Cuadernos de Demostración

`puremacro` incluye cuadernos de demostración interactivos con calidad de publicación que cubren los principales paradigmas de macroeconomía con agentes heterogéneos, econometría empírica, estimación bayesiana, aplicaciones especializadas de clima e historia macroeconómica y suites de frontera para política macroeconómica aplicada.

Cada cuaderno está escrito en formato Jupytext percent (`.py`), se ejecuta de forma determinista con semillas fijas, cumple con el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`) y cuenta con una edición gemela en español (`_es.py`). Los artefactos `.ipynb` precompilados con salidas y figuras prerrenderizadas se generan mediante `tools/build_notebooks.py`.

---

## La Arquitectura Pedagógica de 7 Secciones

Siguiendo `notebooks/_TEMPLATE.md`, los cuadernos profundizados y de frontera siguen un flujo estructural uniforme de 7 celdas:

1. **Pregunta Motivadora**: 1–2 oraciones que definen el problema económico y el dilema de política.
2. **El Método en Matemáticas**: Ecuaciones estructurales y econométricas en LaTeX riguroso y compacto ($...$ / $$...$$).
3. **Intuición**: Sección explícita `**Intuición.**` que traduce las ecuaciones algebraicas a mecanismos económicos intuitivos y lógica de identificación.
4. **Código Desarrollado**: Bloques ejecutables en NumPy puro, autocontenidos y comentados explicando el *porqué* de cada decisión.
5. **Lectura de Resultados**: Análisis narrativo en markdown que interpreta directamente los valores numéricos principales, estimaciones y figuras.
6. **Tu Turno**: Ejercicio exploratorio interactivo con controles `# ← modifica esto`, valores por defecto funcionales con aserciones y retos graduados.
7. **¿Qué tan Exhaustivo es Esto?**: Referencias contextuales que vinculan la demostración con otros módulos de `puremacro` y la literatura.

---

## Demostraciones de Programación Dinámica Continua, Deep Macro y GE Espacial (puremacro 3.3)

Las demostraciones `51` a `55` presentan los motores de vanguardia de `puremacro` en programación dinámica sobre espacios de estados continuos, transiciones no lineales de distribuciones de riqueza, gradientes analíticos de sensibilidad adjunta de precisión de máquina, redes neuronales para aproximación de políticas en alta dimensión y equilibrio general espacial cuantitativo de comercio internacional.

### `51_continuous_projection_collocation_and_fem_es`
- **Fuente**: `notebooks/51_continuous_projection_collocation_and_fem_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo resuelven los macroeconomistas cuantitativos modelos de equilibrio general dinámico sin sufrir la maldición de la dimensionalidad ni las distorsiones de discretización de la iteración sobre funciones de valor en grillas, y cuándo conviene implementar polinomios ortogonales globales de Chebyshev frente a proyecciones locales de Galerkin con elementos finitos?
- **Matemáticas y Algoritmos Rectores**:
  - **Colocación Ortogonal de Chebyshev**: Mapeo de $k \in [a, b]$ al dominio canónico $x \in [-1, 1]$ mediante $x = \frac{2k - (a+b)}{b - a}$. Los polinomios base satisfacen la recurrencia de 3 términos $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$ con nodos de extremos de Chebyshev-Gauss-Lobatto $x_i = -\cos(i\pi/N)$. Se anula el operador de residuo continuo de Euler:
    $$\mathcal{R}_{\text{Euler}}(\theta; k_i) \equiv 1 - \beta \frac{u'(c(g(k_i), g(g(k_i))))}{u'(c(k_i, g(k_i)))} f'(g(k_i)) = 0$$
  - **Proyección de Galerkin con Elementos Finitos (FEM)**: Partición de $[a, b]$ en $E$ elementos con funciones base lineales a tramos en sombrero $\phi_j(k)$ con soporte compacto $[k_{j-1}, k_{j+1}]$. Los residuos se proyectan contra funciones de prueba usando cuadratura de Gauss-Legendre de $Q$ puntos:
    $$\int_a^b \mathcal{R}(k) \phi_i(k) \, dk = 0, \quad \forall i = 1, \dots, N_{\text{nodos}}$$
  - **Operador NCP de Fischer-Burmeister para Restricciones de Endeudamiento**: Complementariedad suavizada y perturbada ($\epsilon = 10^{-12}$) para resolver $k' \ge \bar{k}$ y $\mathcal{R}(k) \ge 0$:
    $$\Psi_{\text{FB}}^\epsilon(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon} = 0, \quad a = k' - \bar{k}, \quad b = \mathcal{R}(k)$$
- **Intuición Económica**: Los polinomios globales de Chebyshev logran convergencia espectral exponencial ($\mathcal{O}(c^{-N})$) para reglas de decisión analíticas reales (crecimiento de Brock-Mirman), pero sufren oscilaciones catastróficas de Gibbs y violaciones del límite inferior ante esquinas angulosas. FEM renuncia al soporte global a favor de funciones locales en sombrero (convergencia algebraica $\mathcal{O}(h^2)$), suprimiendo por completo el fenómeno de Gibbs cuando se fija un nodo del mallado en el umbral de endeudamiento $k^*$.
- **Código Desarrollado**:
  ```text
  import numpy as np
  from puremacro.vfi import CollocationProblem, solve_collocation, FEMProblem, solve_fem

  # Modelo analítico de referencia de Brock-Mirman (1972): u(c)=ln(c), f(k)=k^alpha, delta=1.0
  alpha, beta = 0.36, 0.96
  k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
  domain = (0.5 * k_ss, 1.5 * k_ss)

  def euler_res(k, kp, kpp, s, sp):
      c = k**alpha - kp
      c_next = kp**alpha - kpp
      return 1.0 / c - beta * (1.0 / c_next) * (alpha * kp**(alpha - 1.0))

  prob_col = CollocationProblem(domain=domain, deg=7, beta=beta, euler_residual_fn=euler_res)
  sol_col = solve_collocation(prob_col)

  prob_fem = FEMProblem(
      domain=domain, n_elements=25, deg=1, beta=beta,
      return_fn=lambda k, kp, s: np.log(np.maximum(k**alpha - kp, 1e-8)),
      transition_fn=lambda k, kp, s: kp,
      euler_residual_fn=euler_res,
  )
  sol_fem = solve_fem(prob_fem)
  ```
- **Lectura de Resultados y Visualizaciones**: Cuatro figuras principales: (1) Convergencia semilogarítmica del error de política demostrando decaimiento espectral exponencial ($L^\infty < 10^{-6}$ con $N=8$) frente a decaimiento algebraico de FEM; (2) Detalle en la vecindad del quiebre que evidencia el timbre de Chebyshev frente a la perfecta conformidad de frontera de FEM; (3) Curvas estocásticas de política $g(k, z)$ preservando monotonicidad; (4) Comparativa de tiempo de cómputo en NumPy, Numba y MLX.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `order_custom = 8`, `elements_custom = 50`, `borrow_ratio_custom = 0.50` con aserciones automáticas `sol_custom_c.converged`, `sol_custom_f.converged`, `err_custom_c < 1e-3`, `viol_custom_f == 0.0`.
- **Literatura y Referencias Cruzadas**: Brock y Mirman (1972), Judd (1992, 1998), McGrattan (1996), Fischer (1992). Guía de usuario: [`docs/es/vfi_continuous_projection.md`](vfi_continuous_projection.md).

### `52_continuous_transition_mit_shocks_es`
- **Fuente**: `notebooks/52_continuous_transition_mit_shocks_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo evolucionan de forma no lineal las distribuciones continuas de riqueza de los hogares $\mu_t(k, z)$ y los precios factoriales de equilibrio $\{r_t, w_t\}$ tras un choque macroeconómico agregado imprevisto en una economía de mercados incompletos, y cómo influye el ahorro precautorio endógeno en la persistencia macroeconómica y la desigualdad de riqueza?
- **Matemáticas y Algoritmos Rectores**:
  - **EGM Continuo hacia Atrás**: Inversión de la utilidad marginal a lo largo de activos continuos $a' \ge 0$ dada la trayectoria de precios $\{r_t, w_t\}$:
    $$c_t^{\text{endo}}(a', z_i) = \left( \beta (1 + r_{t+1}) \sum_j P_z(z_i, z_j) [c_{t+1}(a', z_j)]^{-\gamma} \right)^{-1/\gamma}$$
    $$a_t^{\text{endo}}(a', z_i) = \frac{c_t^{\text{endo}}(a', z_i) + a' - w_t z_i}{1 + r_t}$$
  - **Operador de Loterías de Young (2010) Temporal hacia Adelante**: Avance de densidad con conservación exacta de masa a nivel de máquina ($\sum \mu_t = 1.0 \pm 10^{-15}$):
    $$\mu_{t+1}(k_m, z_l) = \sum_j P_z(z_j, z_l) \sum_i \mu_t(k_i, z_j) [w_{\text{lo}} \mathbf{1}_{\{m=j_{\text{lo}}\}} + w_{\text{hi}} \mathbf{1}_{\{m=j_{\text{hi}}\}}]$$
  - **Vaciado de Mercado en el Espacio de Secuencias**:
    $$H_t(\mathbf{r}) \equiv \sum_{i, j} k_i \mu_t(k_i, z_j) - L \left( \frac{r_t + \delta}{\alpha Z_t} \right)^{\frac{1}{\alpha - 1}} = 0, \quad \forall t \in [0, T-1]$$
  - **Solucionador Cuasi-Newton de Broyden**: Trata la trayectoria completa de tasas $\mathbf{r} \in \mathbb{R}^T$ como un sistema unificado con actualizaciones de rango 1 de Sherman-Morrison para la matriz jacobiana inversa aproximada.
- **Intuición Económica**: La oferta agregada de capital no puede ajustarse de manera instantánea en $t=0$ debido a que la distribución de riqueza $\mu_0(k, z)$ es físicamente predeterminada. Para vaciar los mercados factoriales tras un choque positivo imprevisto del $+5\%$ en la PTF, la tasa de interés real salta de inmediato ($+49.9$ pb). El incremento salarial beneficia desproporcionadamente a los trabajadores con restricciones de liquidez y baja riqueza, contrayendo transitoriamente el índice de Gini ($0.5269 \to 0.5212$). A medida que los hogares acumulan ahorro precautorio, el stock agregado de capital alcanza su cúspide en $t=7$ ($K^* = 8.792 \to K_{\text{pico}} = 9.039$), generando una persistencia endógena que supera con creces la autocorrelación exógena del choque ($0.80^7 \approx 0.21$).
- **Código Desarrollado**:
  ```text
  import numpy as np
  from puremacro.vfi import solve_continuous_transition, TransitionShock

  # Choque imprevisto de PTF del +5% con rho=0.80 en un horizonte T=40 trimestres
  shock_path = 0.05 * (0.80 ** np.arange(40))
  shock = TransitionShock(variable="TFP", path=shock_path)
  res = solve_continuous_transition(model, shock=shock, T=40, damping=0.40)
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero central de 4 paneles (Senda de la Tasa Real, Oferta vs Demanda de Capital, Evolución del Gini de Riqueza, Desplazamiento de Curvas de Lorenz) y superficie en perspectiva 3D $\mu_t(k)$ que ilustra la dispersión dinámica de los activos precautorios.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `user_shock_size = 0.05`, `user_persistence = 0.80`, `user_damping = 0.40` con aserciones `user_res.converged`, `user_res.max_residual < 1e-4`, `user_res.mass_conservation_error < 1e-12`.
- **Literatura y Referencias Cruzadas**: Aiyagari (1994), Young (2010), Boppart, Krusell y Mitman (2018), Auclert et al. (2021). Guías de usuario: [`docs/es/vfi_continuous_transition.md`](vfi_continuous_transition.md), [`docs/es/vfi_continuous_equilibrium.md`](vfi_continuous_equilibrium.md).

### `53_exact_analytic_ift_gradients_es`
- **Fuente**: `notebooks/53_exact_analytic_ift_gradients_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo pueden los macroeconomistas cuantitativos efectuar estimaciones estructurales basadas en gradientes (SMM/GMM) de modelos dinámicos continuos sin incurrir en el ruido numérico y el costo combinatorio de las aproximaciones por diferencias finitas, y cómo entrega el Teorema de la Función Implícita sensibilidades exactas de política en una sola resolución lineal?
- **Matemáticas y Algoritmos Rectores**:
  - **Teorema de la Función Implícita en Sistemas de Proyección Continua**: Diferenciando la identidad de residuo nulo $\mathcal{R}(c^*(\theta); \theta) \equiv \mathbf{0} \in \mathbb{R}^N$ con respecto a los parámetros estructurales $\theta \in \mathbb{R}^p$:
    $$\nabla_\theta c^*(\theta) = - \left[ \nabla_c \mathcal{R}(c^*; \theta) \right]^{-1} \nabla_\theta \mathcal{R}(c^*; \theta)$$
  - **Sensibilidades de Políticas Continuas y Agregados Macroeconómicos**:
    $$\nabla_\theta g(k; \theta) = \Phi(k) \, \nabla_\theta c^*(\theta), \qquad \frac{d K^*}{d \theta} = \frac{\nabla_\theta g(k^*; \theta)}{1 - g'(k^*; \theta)}$$
  - **Función Objetivo y Gradiente Exacto de GMM Estructural**:
    $$Q(\theta) = (m(\theta) - \hat{m})^\top W (m(\theta) - \hat{m}), \qquad \nabla_\theta Q(\theta) = 2 \, G(\theta)^\top W (m(\theta) - \hat{m})$$
    donde $G(\theta) \equiv \nabla_\theta m(\theta) = \nabla_c m \, \nabla_\theta c^*(\theta)$ se calcula analíticamente sin diferencias finitas.
- **Intuición Económica**: Las diferencias finitas enfrentan un dilema de paso numérico: pasos grandes generan sesgos de truncamiento ($\mathcal{O}(h^2)$), mientras que pasos pequeños desatan errores de cancelación por redondeo y oscilaciones de parada que deforman la superficie de optimización. El TFI reutiliza la factorización LU del jacobiano residual convergente $J_c = \nabla_c \mathcal{R}$, evaluando las sensibilidades exactas por sustitución hacia atrás con una **aceleración empírica de 68.8x** y cero ruido numérico, logrando superficies parabólicas suaves que permiten al optimizador BFGS converger cuadráticamente en 17 iteraciones.
- **Código Desarrollado**:
  ```text
  from puremacro.vfi import (
      CollocationProblem, solve_collocation,
      compute_ift_gradients, policy_parameter_jacobian,
  )

  sol = solve_collocation(problem)
  ift_res = compute_ift_gradients(sol, problem, params=["beta", "sigma", "alpha", "delta"])
  d_policy_fn = policy_parameter_jacobian(sol, problem, params=["beta"])
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero de 4 paneles (Sensibilidades de la Política a lo largo de $k$, Gráfico de Barras de Rendimiento demostrando 68.8x de aceleración, Superficie de Contorno de Pérdida 2D $\log_{10} Q(\beta, \sigma)$ con la trayectoria BFGS, y Gráfico de Discrepancia Modal de la Base).
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `user_k1 = 0.10`, `user_k2 = 0.28`, `user_init_beta = 0.93`, `user_init_sigma = 1.30` con puertas de validación `user_opt.success`, `np.isclose(user_beta_hat, 0.96, atol=1e-4)`, `np.isclose(user_sigma_hat, 1.50, atol=1e-4)`.
- **Literatura y Referencias Cruzadas**: Judd (1998), Su y Judd (2012), Rust (1994). Guía de usuario: [`docs/es/vfi_analytic_gradients.md`](vfi_analytic_gradients.md).

### `54_deep_macro_pinns_high_dim_es`
- **Fuente**: `notebooks/54_deep_macro_pinns_high_dim_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo pueden los macroeconomistas cuantitativos quebrar la maldición exponencial de la dimensionalidad para resolver modelos de equilibrio general dinámico con más de 10 variables de estado continuo en Python puro sin librerías externas de deep learning, y cómo imponen las redes neuronales informadas por la física la viabilidad física exacta a lo largo de trayectorias ergódicas?
- **Matemáticas y Algoritmos Rectores**:
  - **Acumulación de Capital Multipaís en Alta Dimensión**: Vector de estado $\mathbf{s}_t = (k_{1, t}, \dots, k_{N, t}) \in \mathbb{R}_{++}^N$; recursos disponibles (*cash-on-hand*) $W_i(\mathbf{s}_t) = A_i k_{i, t}^\alpha + (1 - \delta) k_{i, t}$; ley del capital $k_{i, t+1} = W_i(\mathbf{s}_t) - c_{i, t}$.
  - **Operador de Residuo Continuo Adimensional de Euler**:
    $$\mathcal{R}_i(\mathbf{s}_t, \mathbf{c}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}) \equiv 1 - \beta \left(\frac{c_{i, t+1}}{c_{i, t}}\right)^{-\gamma} (\alpha A_i k_{i, t+1}^{\alpha-1} + 1 - \delta) = 0, \quad \forall i = 1, \dots, N$$
  - **Parametrización Neuronal de Políticas y Viabilidad Física**: Perceptrón multicapa entrenable $\mathbf{z}(\mathbf{s}; \Theta) \in \mathbb{R}^N$ con activación sigmoide acotada:
    $$\phi_i(\mathbf{s}; \Theta) = \epsilon_{\text{bound}} + (1 - 2\epsilon_{\text{bound}}) \sigma(z_i(\ln(\mathbf{s}/\mathbf{s}_{\text{ss}}))), \qquad c_i(\mathbf{s}; \Theta) = \phi_i(\mathbf{s}; \Theta) W_i(\mathbf{s})$$
    donde $\epsilon_{\text{bound}} = 10^{-4}$ garantiza estrictamente $c_i > 0, k'_i > 0, c_i < W_i$ en todo el dominio.
  - **Muestreo por Trayectorias Ergódicas (Maliar et al. 2021)**: La simulación de trayectorias de equilibrio de longitud $T = 1200$ concentra el cómputo sobre la variedad compacta del atractor estocástico, reduciendo $10^{10}$ puntos de grilla a $\mathcal{O}(T \cdot D)$. La función de pérdida $\mathcal{L}(\Theta) = \frac{1}{B \cdot N} \sum_{b, i} (\phi_{i, b} - \phi_{i, b}^{\text{target}})^2$ se minimiza mediante Adam con gradientes analíticos vectorizados en NumPy puro.
- **Intuición Económica**: Discretizar un vector continuo de 10 dimensiones con apenas 10 puntos por dimensión requeriría $10^{10}$ nodos. Los sistemas económicos, no obstante, jamás exploran hipercubos de manera uniforme; los rendimientos marginales decrecientes y la depreciación del capital ejercen fuerzas gravitacionales de reversión a la media que restringen la dinámica a un atractor estocástico de dimensión baja. Al entrenar perceptrones con activaciones suaves (SiLU) exclusivamente sobre las trayectorias ergódicas visitadas, las PINNs resuelven modelos de 10 países en $<0.5$ segundos con cero dependencias externas de deep learning (PyTorch/TensorFlow).
- **Código Desarrollado**:
  ```text
  from puremacro.vfi import DeepMacroModel, solve_deep_macro

  model = DeepMacroModel(d_in=10, d_out=10, hidden_layers=[64, 64], activation="silu", seed=42)
  res = solve_deep_macro(
      model, euler_loss_fn=euler_loss, state_sampler=sampler,
      n_epochs=120, lr=1e-3, optimizer="adam", batch_size=256,
  )
  ```
- **Lectura de Resultados y Visualizaciones**: Tablero de 4 paneles: (1) Convergencia de la pérdida de entrenamiento ($9.58 \times 10^{-10}$); (2) Gráficos de caja de residuos de Euler fuera de muestra para los 10 países ($\text{MSE} = 3.35 \times 10^{-6} < 10^{-3}$); (3) Senda de profundización del capital en los 10 países partiendo de una severa depresión; (4) Sección unidimensional de la función de política de consumo continuo frente al capital disponible.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `hidden_dims_custom = (32, 32)`, `lr_custom = 2e-3`, `n_epochs_custom = 80`, `perturb_custom = 0.60` con aserciones `sol_custom.converged`, `sol_custom.test_euler_mse < 1e-3`, `sim_custom["physically_viable"]`.
- **Literatura y Referencias Cruzadas**: Maliar, Maliar y Winant (2021), Raissi, Perdikaris y Karniadakis (2019), Judd (1998). Guía de usuario: [`docs/es/deep_macro.md`](deep_macro.md).

### `55_quantitative_spatial_and_trade_ge_es`
- **Fuente**: `notebooks/55_quantitative_spatial_and_trade_ge_es.py` (Inglés: `.py`, compilado: `.ipynb`)
- **Pregunta Económica Motivadora**: ¿Cómo se propagan los choques arancelarios internacionales, las disrupciones de cadenas de suministro y las inversiones en infraestructura regional de transporte a través de los encadenamientos insumo-producto y la movilidad espacial laboral para transformar los flujos de comercio, las poblaciones regionales y el bienestar agregado?
- **Matemáticas y Algoritmos Rectores**:
  - **Álgebra Exacta de Sombreros de Caliendo-Parro (2015)**: $N$ países, $J$ sectores con participación de valor agregado $\gamma_n^j$ y participaciones de costo de insumos intermedios $\gamma_n^{j, k}$. Variaciones relativas $\hat{x} \equiv x'/x$:
    $$\ln \hat{c}_i^j = \gamma_i^j \ln \hat{w}_i + \sum_k \gamma_i^{j, k} \ln \hat{P}_i^k$$
    $$\hat{P}_n^j = \left[ \sum_i \pi_{ni}^j (\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j)^{-\theta_j} \right]^{-1/\theta_j}, \qquad \pi_{ni}'^j = \pi_{ni}^j \left(\frac{\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j}{\hat{P}_n^j}\right)^{-\theta_j}$$
    Vaciado de mercado para el ingreso factorial $w_n' L_n$ y variación de bienestar nacional $\hat{W}_n = \hat{I}_n / \prod_j (\hat{P}_n^j)^{\alpha_n^j}$.
  - **Equilibrio General Espacial Cuantitativo de Allen-Arkolakis (2014)**: Geografía continua con $N$ localidades, costos bilaterales tipo iceberg $\tau_{ij} \ge 1$, aglomeración marshalliana $\alpha > 0$ ($A_i = \bar{A}_i L_i^\alpha$) y congestión residencial $\beta < 0$ ($a_i = \bar{a}_i L_i^\beta$). La igualación de utilidad real $u_i = a_i \frac{w_i}{P_i} = \bar{u}$ determina:
    $$L_i = \bar{L} \frac{(\bar{a}_i w_i / P_i)^{-1/\beta}}{\sum_k (\bar{a}_k w_k / P_k)^{-1/\beta}}, \qquad P_i^{-\theta} = \sum_j \tau_{ji}^{-\theta} \left(\frac{w_j}{\bar{A}_j L_j^\alpha}\right)^{-\theta}$$
    Equilibrio espacial único garantizado cuando $\alpha + \beta \le \frac{\theta}{1+\theta}$ y $\alpha \le 1/\theta$.
- **Intuición Económica**: En el comercio internacional, los aranceles unilaterales generan mejoras en los términos de intercambio (+0.33% bienestar en EE.UU.) pero desvían comercio hacia terceros países. Las guerras comerciales bilaterales destruyen estas ventajas e imponen pérdidas netas de bienestar (-1.11% EE.UU., -0.69% China). Los insumos intermedios actúan como multiplicadores de la cadena de suministro: los aranceles se transmiten en cascada a través de las matrices insumo-producto, amplificando las mermas de bienestar en relación con los modelos ricardianos tradicionales. En geografía espacial, los corredores de transporte (-20% de fricción) amplían el acceso a mercados, atrayendo mano de obra móvil (+0.35% población regional) hasta ser contrarrestados por la congestión de amenidades, elevando el bienestar agregado (+0.0627%) bajo estricta conservación poblacional.
- **Código Desarrollado**:
  ```text
  from puremacro.trade import CaliendoParroModel
  from puremacro.spatial import AllenArkolakisModel

  # 1. Simulación de choque arancelario de Caliendo-Parro
  cp_res = cp_model.solve_counterfactual(tariff_hat=tariff_hat)

  # 2. Simulación de corredor de infraestructura de transporte de Allen-Arkolakis
  aa_res = aa_model.simulate_transport_shock(origin=0, destination=1, cost_reduction=0.20)
  ```
- **Lectura de Resultados y Visualizaciones**: Figuras de calidad editorial: (1) Gráfico de 3 paneles de comercio (Matriz inicial de participaciones, Matriz post-guerra arancelaria, Impacto en bienestar por país); (2) Red geográfica de 2 paneles con corredor modernizado y variaciones de salarios/población; (3) Comparativa del impacto en bienestar con y sin encadenamientos insumo-producto.
- **Tu Turno y Aserciones Interactivas**: Parámetros interactivos `tariff_rate_custom = 0.15`, `cost_reduction_custom = 0.20` con aserciones `res_cp_custom.converged`, `res_aa_custom.converged`, `res_cp_custom.market_clearing_residual < 1e-6`, `res_aa_custom.labor_conservation_residual < 1e-12`, `res_aa_custom.welfare_pct > 0.0`.
- **Literatura y Referencias Cruzadas**: Eaton y Kortum (2002), Caliendo y Parro (2015), Allen y Arkolakis (2014), Costinot y Rodríguez-Clare (2014). Guía de usuario: [`docs/es/spatial_and_trade_ge.md`](spatial_and_trade_ge.md).

---

## Cuadernos de Frontera en Política Macroeconómica Aplicada (puremacro 3.2)

Las demostraciones `47` a `50` conectan la macroeconometría teórica con la práctica de política en bancos centrales, ministerios de hacienda y mercados financieros.

### `47_applied_central_bank_policy_suite_es`
- **Fuente**: `notebooks/47_applied_central_bank_policy_suite_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Evaluación en tiempo real de la postura monetaria frente a reglas de Taylor (1993, 1999) y especificaciones con inercia.
  - Simulación de políticas contrafactuales: trayectorias macroeconómicas bajo sendas alternativas de tasas de interés.
  - Bandas de proyección en gráficos de abanico (fan charts) incorporando incertidumbre de choques y dispersión posterior de parámetros.
  - Extracción de tono y sentimiento en comunicados de bancos centrales (FOMC, BCE) mediante calificación léxica.

### `48_applied_realtime_nowcasting_and_news_es`
- **Fuente**: `notebooks/48_applied_realtime_nowcasting_and_news_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Nowcasting del PIB trimestral con modelos de factores dinámicos (DFM) siguiendo a Giannone, Reichlin y Small (2008).
  - Ingesta de series en tiempo real con extremos irregulares (ragged edges) combinando frecuencias mensuales y trimestrales.
  - **Descomposición de noticias**: atribución de las revisiones del pronóstico a sorpresas en publicaciones de datos (empleo, producción, encuestas).
  - Pruebas de ortogonalidad de Mankiw-Shapiro (1986) para evaluar si las revisiones representan noticias o ruido.

### `49_applied_macroprudential_gar_and_stress_es`
- **Fuente**: `notebooks/49_applied_macroprudential_gar_and_stress_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Regresión cuantil condicional para Growth-at-Risk (GaR) siguiendo a Adrian, Boyarchenko y Giannone (2019).
  - Ajuste de densidades paramétricas skew-$t$ de Azzalini para capturar el riesgo asimétrico a la baja en horizontes predictivos.
  - Redes de desbordamiento de riesgo sistémico mediante descomposición generalizada de varianza del error de pronóstico (GFEVD) (Diebold y Yílmaz 2014).
  - Escenarios de estrés para colchones de capital macroprudencial y probabilidades de recesión en gráficos de abanico.

### `50_applied_fiscal_multipliers_and_debt_sustainability_es`
- **Fuente**: `notebooks/50_applied_fiscal_multipliers_and_debt_sustainability_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Estimación trimétodo de multiplicadores fiscales: SVAR de Blanchard-Perotti, proyecciones locales narrativas de Romer-Romer y LP-IV de Mertens-Ravn con estadístico $F$ efectivo de primera etapa.
  - Análisis estocástico de sostenibilidad de la deuda soberana (DSA) con gráficos de abanico ante choques conjuntos de crecimiento, inflación y diferencial de tasas $(r - g)$.
  - Pruebas de estrés de la trayectoria de deuda ante escenarios de riesgo climático físico y de transición.

---

## Demostraciones de Frontera en DSGE Bayesiano y HANK (puremacro 3.0 y 3.1)

### `42_dsge_bayesian_estimation_and_diagnostics_es` (Insignia)
- **Fuente**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics_es.py` (Inglés: `.py`)
- **Modelo**: Economía estadounidense de Smets & Wouters (2007, AER) con 7 variables y 7 choques.
- **Capacidades Clave**: Búsqueda de moda con múltiples algoritmos (`lbfgs`, `csminwel`, `cmaes`), diagnósticos de curvatura `mode_check`, muestreo MCMC con diagnósticos $\hat{R}$ de Gelman-Rubin, suavizador de Kalman para descomposición de choques, pronósticos dinámicos y comparación de modelos por densidad marginal de Laplace y Geweke.

### `43_dsge_nuts_and_analytic_gradients_es`
- **Fuente**: `notebooks/43_dsge_nuts_and_analytic_gradients_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Recursión analítica exacta del gradiente de verosimilitud de Kalman ($\nabla_\theta \ln L$) con solucionadores de Sylvester generalizados, muestreador No-U-Turn (NUTS) con adaptación dual averaging y diagnósticos energéticos BFMI.

### `44_hank_sequence_space_bridge_es`
- **Fuente**: `notebooks/44_hank_sequence_space_bridge_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Puente de agentes heterogéneos en el espacio de secuencias desde archivos `.mod` de Dynare (`hetagent_block`), distribución estacionaria de riqueza $\mathcal{D}^*(a)$, jacobianos Fake-News ($J_{C,r}, J_{C,Y}$) y transiciones no lineales de Broyden.

### `45_dsge_discretion_dsge_var_and_news_shocks_es`
- **Fuente**: `notebooks/45_dsge_discretion_dsge_var_and_news_shocks_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Política óptima discrecional frente a compromiso (descomposición del sesgo de inflación y estabilización), optimización de previas DSGE-VAR($\lambda$) de Del Negro y Schorfheide (2004) y estado-espacio aumentado para choques anticipados de noticias.

### `46_dsge_particle_filtering_and_markov_switching_es`
- **Fuente**: `notebooks/46_dsge_particle_filtering_and_markov_switching_es.py` (Inglés: `.py`)
- **Capacidades Clave**: OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov de Foerster et al. (2016) con GIRFs analíticas cerradas y filtrado de partículas secuencial Monte Carlo con volatilidad estocástica.

---

## Catálogo Completo de Demostraciones

| Cuaderno | Tema y Metodología | Gemelo en Inglés |
|---|---|---|
| `00_whats_new_in_puremacro_3_0_es` | Hito 3.0: Gradientes analíticos de Kalman, NUTS HMC y puente HANK en espacio de secuencias | `00_whats_new_in_puremacro_3_0` |
| `00_whats_new_in_puremacro_2_0_es` | Hito 2.0: API unificada (`lags`, `horizon`, `ci`), objetos de resultado y exportadores | `00_whats_new_in_puremacro_2_0` |
| `01_wealth_inequality_es` | Mercados incompletos de Aiyagari y Huggett, heterogeneidad permanente en $\beta$, Lorenz y Gini | `01_wealth_inequality` |
| `02_aggregate_shocks_es` | Agregación aproximada de Krusell–Smith, dinámica de transición, agente representativo | `02_aggregate_shocks` |
| `03_life_cycle_and_demographics_es` | Consumo y ahorro en ciclo de vida finito, riqueza de cohortes por edad, mortalidad | `03_life_cycle_and_demographics` |
| `04_firm_dynamics_es` | Equilibrio de industria de Hopenhayn con entrada/salida endógena y selección | `04_firm_dynamics` |
| `05_portfolios_and_preferences_es` | Elección de cartera con dos activos, utilidad recursiva de Epstein–Zin, EGM frente a VFI | `05_portfolios_and_preferences` |
| `06_svar_identification_es` | Identificación en VAR estructural: ordenamiento de Cholesky frente a restricciones de signo | `06_svar_identification` |
| `07_local_projections_es` | Proyecciones locales de Jordà con inferencia HAC Newey-West y respuestas estado-dependientes | `07_local_projections` |
| `08_garch_volatility_es` | GARCH(1,1) por máxima verosimilitud y correlación condicional dinámica (DCC) de Engle | `08_garch_volatility` |
| `09_growth_at_risk_es` | Growth-at-Risk de Adrian-Boyarchenko-Giannone mediante regresión cuantil y densidad skew-$t$ | `09_growth_at_risk` |
| `10_staggered_did_es` | Diferencias en diferencias modernas: estimadores de Callaway-Sant'Anna y Sun-Abraham | `10_staggered_did` |
| `11_narrative_uncertainty_es` | Construcción de índices textuales de Incertidumbre de Política Económica (EPU) | `11_narrative_uncertainty` |
| `12_validation_gallery_es` | Cuadro de mando de validación comparando puremacro contra referencias independientes | `12_validation_gallery` |
| `13_build_your_own_index_es` | Construcción multirreceta de índices: Texto $\to$ EPU, Panel $\to$ JLN, Financiero $\to$ FCI | `13_build_your_own_index` |
| `14_tax_multiplier_three_ways_es` | Multiplicador fiscal: SVAR Blanchard-Perotti, LP narrativa Romer-Romer, LP-IV Mertens-Ravn | `14_tax_multiplier_three_ways` |
| `15_lp_did_es` | Proyecciones locales con diferencias en diferencias (LP-DiD) bajo adopción escalonada | `15_lp_did` |
| `16_regime_girf_es` | Respuestas a impulsos generalizadas (GIRF) para VAR de umbral y cambio de régimen de Markov | `16_regime_girf` |
| `17_identification_spec_curve_es` | Curvas de especificación de identificación para análisis de sensibilidad | `17_identification_spec_curve` |
| `18_beveridge_curve_es` | Curva de Beveridge, eficiencia del emparejamiento y desplazamientos vacantes-desempleo | `18_beveridge_curve` |
| `19_model_confidence_set_es` | Conjunto de confianza de modelos (MCS) de Hansen-Lunde-Nason para pronósticos | `19_model_confidence_set` |
| `20_unit_roots_with_power_es` | Pruebas de raíz unitaria de alta potencia DF-GLS de Elliott-Rothenberg-Stock y Ng-Perron | `20_unit_roots_with_power` |
| `21_dynare_vfi_dsl_es` | Especificación declarativa en DSL para VFI con aceleración de Howard | `21_dynare_vfi_dsl` |
| `22_continuous_time_hjb_es` | Esquema en diferencias finitas para Hamilton-Jacobi-Bellman en tiempo continuo | `22_continuous_time_hjb` |
| `23_aiyagari_endogenous_labor_es` | Equilibrio general de mercados incompletos con oferta laboral endógena | `23_aiyagari_endogenous_labor` |
| `24_synthetic_control_es` | Control sintético de Abadie et al. con pruebas de placebo espaciales y temporales | `24_synthetic_control` |
| `25_frequency_connectedness_es` | Redes de desbordamiento en el dominio de la frecuencia de Baruník y Křehlík | `25_frequency_connectedness` |
| `26_cycles_and_bandpass_es` | Filtros de ciclo de Baxter-King, Christiano-Fitzgerald y Beveridge-Nelson | `26_cycles_and_bandpass` |
| `27_garch_midas_macro_risk_es` | Modelado de volatilidad de frecuencias mixtas GARCH-MIDAS de Engle-Ghysels-Sohn | `27_garch_midas_macro_risk` |
| `28_weak_iv_anderson_rubin_es` | Conjuntos de confianza Anderson-Rubin robustos a instrumentos débiles para LP-IV | `28_weak_iv_anderson_rubin` |
| `29_synthetic_did_es` | Diferencias en diferencias sintéticas (SDID) de Arkhangelsky et al. | `29_synthetic_did` |
| `30_narrative_bursts_and_transcripts_es` | Detección de ráfagas de Kleinberg en transcripciones de discursos de bancos centrales | `30_narrative_bursts_and_transcripts` |
| `31_sequence_space_hank_es` | Método jacobiano en el espacio de secuencias de Auclert et al. (2021) para HANK | `31_sequence_space_hank` |
| `32_climate_macro_dice_es` | Modelo DICE de Nordhaus e impuestos óptimos al carbono | `32_climate_macro_dice` |
| `33_gdp_nowcasting_news_es` | Nowcasting del PIB con modelo de factores dinámicos y descomposición de noticias | `33_gdp_nowcasting_news` |
| `34_penalized_macro_forecasting_es` | Elastic Net y Lasso adaptativo para pronósticos macroeconómicos en alta dimensión | `34_penalized_macro_forecasting` |
| `35_empirical_benchmark_replications_es` | Cuadro de mando de replicación en literaturas de referencia de SVAR, LP y DiD | `35_empirical_benchmark_replications` |
| `36_climate_sovereign_debt_risk_es` | Transmisión del riesgo climático físico y de transición a la sostenibilidad de deuda | `36_climate_sovereign_debt_risk` |
| `37_central_bank_narrative_sentiment_es` | Sentimiento en comunicación de bancos centrales y proyecciones locales de alta frecuencia | `37_central_bank_narrative_sentiment` |
| `38_real_time_vintages_and_revisions_es` | Cuentas nacionales en tiempo real en más de 45 países y pruebas noticia vs. ruido | `38_real_time_vintages_and_revisions` |
| `39_multilingual_narrative_harvesting_es` | Recolección narrativa multifuente (50+ conectores) y puntuación macro en 8 idiomas | `39_multilingual_narrative_harvesting` |
| `40_quarterly_national_accounts_es` | Tres enfoques para cuentas nacionales: cambio de año base, identidades y contribuciones | `40_quarterly_national_accounts` |
| `41_dynare_frontier_showcase_es` | Smets-Wouters (2007) con funciones nativas de suavizado y estimación, ZLB y Ramsey | `41_dynare_frontier_showcase` |
| `42_dsge_bayesian_estimation_and_diagnostics_es` | Estimación bayesiana insignia de Smets-Wouters, mode_check, MCMC, fan charts y MDD | `42_dsge_bayesian_estimation_and_diagnostics` |
| `43_dsge_nuts_and_analytic_gradients_es` | Estimación DSGE bayesiana con NUTS y recursión exacta del score de Kalman | `43_dsge_nuts_and_analytic_gradients` |
| `44_hank_sequence_space_bridge_es` | Puente HANK en espacio de secuencias desde archivos `.mod`, jacobianos Fake-News | `44_hank_sequence_space_bridge` |
| `45_dsge_discretion_dsge_var_and_news_shocks_es` | Política discrecional vs compromiso, optimización de previas DSGE-VAR, choques de noticias | `45_dsge_discretion_dsge_var_and_news_shocks` |
| `46_dsge_particle_filtering_and_markov_switching_es` | OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov y partículas | `46_dsge_particle_filtering_and_markov_switching` |
| `47_applied_central_bank_policy_suite_es` | Postura monetaria, reglas de Taylor contrafactuales, abanicos de proyección, sentimiento | `47_applied_central_bank_policy_suite` |
| `48_applied_realtime_nowcasting_and_news_es` | Nowcasting DFM con series irregulares en tiempo real, descomposición de noticias | `48_applied_realtime_nowcasting_and_news` |
| `49_applied_macroprudential_gar_and_stress_es` | Densidades de Growth-at-Risk (skew-$t$), redes de interconexión financiera y estrés | `49_applied_macroprudential_gar_and_stress` |
| `50_applied_fiscal_multipliers_and_debt_sustainability_es` | Multiplicadores fiscales trimétodo, DSA estocástico soberano bajo estrés climático | `50_applied_fiscal_multipliers_and_debt_sustainability` |
| `51_continuous_projection_collocation_and_fem_es` | PD de estados continuos: colocación ortogonal de Chebyshev vs FEM Galerkin con restricciones de endeudamiento | `51_continuous_projection_collocation_and_fem` |
| `52_continuous_transition_mit_shocks_es` | Dinámica de transición continua no lineal y choques MIT acoplando EGM y operador de Young | `52_continuous_transition_mit_shocks` |
| `53_exact_analytic_ift_gradients_es` | Jacobianos analíticos exactos por el TFI, sensibilidades adjuntas y estimación estructural GMM | `53_exact_analytic_ift_gradients` |
| `54_deep_macro_pinns_high_dim_es` | Modelos dinámicos en alta dimensión (10+ estados) mediante Redes Neuronales Informadas por la Física | `54_deep_macro_pinns_high_dim` |
| `55_quantitative_spatial_and_trade_ge_es` | Equilibrio general espacial y comercial cuantitativo: aranceles Caliendo-Parro y geografía Allen-Arkolakis | `55_quantitative_spatial_and_trade_ge` |

---

## Ejecución y Construcción de Artefactos

Los cuadernos de demostración se pueden ejecutar y compilar en artefactos `.ipynb` prerrenderizados mediante la herramienta de línea de comandos:

```bash
# Ejecutar y compilar todos los cuadernos en .ipynb
python tools/build_notebooks.py

# Compilar un solo cuaderno
python tools/build_notebooks.py 51_continuous_projection_collocation_and_fem_es

# Ejecutar sin modificar archivos (falla si algún cuaderno genera excepción)
python tools/build_notebooks.py --check
```
