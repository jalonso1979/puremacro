> 🇬🇧 [English](../hank_capital.md) · 🇪🇸 Español

# Capital productivo en HANK de dos activos

`calibrate_capital_hank` construye una economía donde los hogares poseen bonos
nominales del gobierno y capital productivo instalado. La inversión modifica
la oferta de capital, su precio equilibra el mercado ilíquido y el retorno
incluye la renta operativa y el valor del capital restante.
`solve_hank_reduced_policy` resuelve su equilibrio estocástico local con
coordenadas reducidas de la distribución y de los valores marginales. Los pasos
hacia atrás del hogar y el transporte de probabilidades siguen usando la malla
completa para auditorías no lineales y correcciones de la continuación
estocástica condicionadas al estado.

```python
import numpy as np
from puremacro.models import calibrate_capital_hank, solve_hank_reduced_policy

# Malla pequeña de demostración, sin certificar convergencia de la malla.
ss = calibrate_capital_hank(n_a=10, n_b=10, a_max=100., b_max=15.)
policy = solve_hank_reduced_policy(
    ss, distribution_rank=12, marginal_rank=48, rho=.7, shock_std=1e-4,
)
audit = policy.audit(np.r_[-1e-5, np.zeros(7)], tolerance=1e-5)
print(ss.diagnostics, policy.diagnostics, audit.diagnostics)
print(audit.paths[["Y", "K", "I", "q", "r_a", "goods_mkt"]])
```

La simulación lineal devuelve desviaciones logarítmicas de los controles. Las
auditorías no lineales devuelven desviaciones en niveles y residuos. `K` es
capital físico heredado; `Knext` es el stock que se heredará en el período
siguiente. `I` es inversión física e `investment_goods` añade sus costos de
instalación. Las auditorías conservan la tasa nominal y el precio de capital
efectivamente cotizados en cada fecha.

## Producción e inversión

Sea \(K_t\) el capital heredado, \(q_{t-1}\) su precio efectivo de compra y
\(q_t\) el precio de capital instalado para \(t+1\). La producción, el costo
marginal y la renta de capital satisfacen

\[
Y_t=Z K_t^\alpha N_t^{1-\alpha},\qquad
mc_t=\frac{w_tN_t}{(1-\alpha)Y_t},\qquad
R^K_t=\frac{\alpha mc_tY_t}{K_t}.
\]

La inversión física y su costo de recursos son

\[
\Phi(I_t,K_t)=\frac{\psi}{2}\left(\frac{I_t}{K_t}-\delta\right)^2K_t,
\quad \psi=\frac1{\delta\epsilon_I},\quad
q_t=1+\psi\left(\frac{I_t}{K_t}-\delta\right),
\]
\[
I_t=K_t[\delta+(q_t-1)/\psi],\qquad
K_{t+1}=(1-\delta)K_t+I_t.
\]

La tecnología es reversible: se permite inversión negativa siempre que
\(K_{t+1}>0\). La inversión irreversible requeriría una condición adicional
de complementariedad que aquí no se impone.

El retorno incluye la renta de instalación
\(\Pi^I_t=(q_t-1)I_t-\Phi_t=-K_t\Phi_K(I_t,K_t)\):

\[
1+r^a_t=\frac{R^K_t+(1-\delta)q_t+\Pi^I_t/K_t}{q_{t-1}}
=\frac{R^K_t-(I_t+\Phi_t)/K_t+q_tK_{t+1}/K_t}{q_{t-1}}.
\]

Omitir esta renta cambia la valoración marginal cuando varía la inversión.
Las pruebas diferencian independientemente la tecnología respecto a inversión
y capital heredado para verificar tanto la condición de primer orden como
el retorno realizado.

## Hogares y cierre agregado

El solver existente de dos activos y costos convexos elige consumo, bonos y
riqueza ilíquida en la malla completa. Se vacían ambos mercados:

\[
\int a'_t\,dD_t=q_t K_{t+1},\qquad \int b'_t\,dD_t=\bar B.
\]

El gasto público permanece fijo en niveles después de calibrar. Los impuestos
financian ese gasto y los intereses líquidos, sin financiar retornos de capital:

\[
1+r^b_t=\frac{1+i_{t-1}}{1+\pi_t},\quad
T_t=r^b_t\bar B+G,\quad
i_t=\bar r^b+\phi\pi_t+\varepsilon_t.
\]

Los beneficios operativos son
\(\Pi^P_t=Y_t-w_tN_t-R^K_tK_t-\mathrm{pricecost}_t\).
El ingreso no financiero \(w_tN_t+\Pi^P_t-T_t\) se distribuye proporcionalmente
a la productividad, cuya media estacionaria es uno. La renta de instalación
ya entra en los retornos de capital y no se distribuye una segunda vez.
Las horas comunes y flexibles cumplen

\[
w_t\int e c_t^{-\gamma}\,dD_t=v_\phi N_t^{1/\mathrm{Frisch}}.
\]

El costo de ajustar precios es
\(\mathrm{pricecost}_t=\mu[\log(1+\pi_t)]^2Y_t/[2(\mu-1)\kappa]\).
La curva de Phillips de precios satisface

\[
\log(1+\pi_t)=\kappa(mc_t-1/\mu)
+E_t\left[\frac{Y_{t+1}}{Y_t}
\frac{\log(1+\pi_{t+1})}{1+r^b_{t+1}}\right].
\]

Los presupuestos de los hogares y ambos mercados implican la restricción de
recursos, auditada por separado:

\[
Y_t=C_t+\mathrm{CHI}_t+I_t+\Phi_t+\mathrm{pricecost}_t+G.
\]

`resource_accounting_res` comprueba la identidad también fuera del equilibrio:
relaciona el residuo de bienes con los residuos de los mercados y las brechas
de propiedad heredadas. Si la distribución transportada cumple
\(\int a_t\,dD_t\ne q_{t-1}K_t\), se conserva el precio anterior efectivo.
No se redefine el retorno para absorber un error del mercado ilíquido.

El estado recursivo contiene las coordenadas de la distribución más
`lag_nominal`, `monetary_shock` y `log_capital`.
Esta última es \(\log(K_t/\bar K)\). En equilibrio lineal se identifica el precio
heredado mediante \(q_{t-1}=\int a_t\,dD_t/K_t\); las trayectorias no lineales
fuera del equilibrio conservan explícitamente la cotización efectiva.

## Calibración y alcance de la referencia

Los valores físicos predeterminados usan las cifras redondeadas de ABRS:
\(\alpha=.33,\delta=.02,\mu=1.015,\epsilon_I=4\). La IES inversa es 2 y la
elasticidad de Frisch es 1. Dados los retornos estacionarios, se resuelve
**beta**, manteniendo fijos esos parámetros productivos:

\[
(\bar r^a+\delta)\bar K=\alpha\bar Y/\mu,\quad
\bar K=\bar A,\quad
\bar Y=\frac{\bar C+\overline{\mathrm{CHI}}+\delta\bar K}{1-g}.
\]

La deuda es la demanda líquida estacionaria. La productividad total y la
desutilidad laboral normalizan las horas estacionarias a uno. Al refinar la
malla pueden cambiar beta y estas normalizaciones; no cambian silenciosamente
la participación de capital, la depreciación o los costos de instalación.
Se rechazan los fallos del hogar y los intervalos de beta que no encierran una
raíz. El historial conserva advertencias intermedias; los diagnósticos finales
conservan masa en los límites, error de políticas y residuos de recursos.

Producción, inversión, ajuste de precios e identidad de valoración se comparan
con el [ejemplo de dos activos de los autores de SSJ, commit
eb4ac4c](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/src/sequence_jacobian/examples/two_asset.py).
Véanse su [tutorial](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/notebooks/two_asset.ipynb)
y el [artículo](https://doi.org/10.3982/ECTA17434).
La prueba de referencia independiente evalúa trayectorias fuera del estado
estacionario y no únicamente valores de los parámetros.

**No se replica toda la economía HANK publicada.** La referencia contiene
salarios rígidos, intermediación entre acciones y bonos, una prima de liquidez,
un proceso de ingresos más amplio y otra incidencia de beneficios e impuestos.
Su alpha exacto se calibra a objetivos; aquí se fija el valor redondeado .33.
El denominador de su costo de cartera es \((1+r^a)a+\chi_0\); el solver
existente de PureMacro usa \(a+\bar a\). Se conserva ese solver y se declara
la diferencia. Una replicación completa exige esos bloques de hogares,
salarios, intermediarios, acciones y calibración.

La convergencia del sistema proyectado no certifica precisión de la malla
ni de las ecuaciones marginales omitidas. Deben compararse rangos, revisarse
residuos completos y errores Euler/KKT, y auditarse historias no usadas para
construir la base. La continuación estocástica local resuelta no establece
precisión global ni efectos de bienestar de segundo orden.
