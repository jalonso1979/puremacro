> 🇬🇧 [English](../trade_reallocation.md) · 🇪🇸 Español

# Transiciones conjuntas de abastecimiento, capital y trabajo

`puremacro.trade.reallocation` resuelve un equilibrio competitivo
**experimental de dos países y dos sectores** con abastecimiento por origen
endógeno, acumulación de capital y reasignación laboral sectorial costosa.
Los precios de bienes y factores, el consumo y los tres márgenes de asignación
se resuelven conjuntamente. Los valores predeterminados son supuestos
sintéticos y el hogar comparte el ingreso nacional de ambos sectores.

Este modelo pequeño tiene un cierre explícito propio. El
[modelo MRIO dinámico](trade_dynamic.md) mantiene el abastecimiento fijo y
permite movilidad laboral nacional; el [abastecimiento condicionado](trade_adjustment.md)
toma la producción y los precios como dados. Combinar sus trayectorias no
produce este equilibrio. Aquí no hay insumos intermedios, migración
internacional, bonos negociados, incertidumbre ni trabajadores heterogéneos.

## Ejecutar el arancel temporal

AAA anuncia en la fecha cero un arancel de importación del 10% en S01 para
las fechas 2–7. Las demás tasas, incluidas las terminales, siguen en cero.

```python
import numpy as np
from puremacro.trade import (
    ReallocationEconomy, audit_reallocation, reallocation_stability,
    run_reallocation_horizon_ladder, solve_reallocation_steady_state,
)

economy = ReallocationEconomy()
initial = solve_reallocation_steady_state(economy)
assert reallocation_stability(initial)["stable_graph"]

def policy(horizon):
    tariffs = np.zeros((horizon, 2, 2))
    tariffs[2:8, 0, 0] = 0.10
    return tariffs

ladder = run_reallocation_horizon_ladder(economy, policy, initial=initial)
assert ladder.converged, ladder.to_frame()
run = ladder.accepted
assert run is not None and audit_reallocation(run)["accepted"]
print(ladder.to_frame())
print(run.welfare)
print(run.sector_frame().head())
```

El ejemplo completo compara la misma política eliminando cada costo de ajuste:

```bash
python -m puremacro.examples.trade_reallocation --output trade-example --lang es
```

Exporta los arreglos completos (`path-*.npz`), cuentas nacionales y sectoriales,
errores de auditoría por fecha, bienestar, intentos por horizonte, figuras
PNG/PDF, parámetros, versiones y hashes del código. Los intentos rechazados se
conservan; un caso sin resolver hace que el comando termine con error.
El ejemplo prueba 40, 80, 160 y, si hace falta, 320 fechas. La API se detiene
en 160 de forma predeterminada.

## Cantidades económicas y secuencia temporal

El capital $K_{cs,t}$ es heredado. El hogar elige el trabajo sectorial actual
$L_{cs,t}$, una receta actual de proveedores $q_{cso,t}$ y el capital siguiente
$K_{cs,t+1}$. El empleo y las razones de abastecimiento iniciales provienen de
un estado estacionario auditado. La oferta nacional de trabajo $\bar L_c$ es
fija. Todas las fechas de política se conocen desde el inicio.

Las empresas competitivas producen $Y_{cs}=Z_{cs}K_{cs}^aL_{cs}^{1-a}$, con
$w_{cs}=(1-a)p_{cs}Y_{cs}/L_{cs}$ y $R_{cs}=ap_{cs}Y_{cs}/K_{cs}$.
El hogar nacional posee el capital y comparte los ingresos de ambos sectores.

El bien compuesto final es $A_c=\prod_s(X_{cs}/b_{cs})^{b_{cs}}$, con pesos
positivos que suman uno. Los bienes sectoriales combinan orígenes mediante

$$
\left[\sum_o\alpha_{cso}(q_{cso}/\alpha_{cso})^\rho\right]^{1/\rho}=1,
\qquad \rho=(\sigma-1)/\sigma.
$$

Los pesos nacionales y extranjeros son `home_shares` y `1-home_shares`.
Son pesos de preferencias/tecnología, no cuotas de compra fijas. El código
incluye el límite Cobb–Douglas $\sigma=1$. La coordenada de elección es
$r_{cs}=\log[(q_{cs,foreign}/\alpha_{cs,foreign})/(q_{cs,home}/\alpha_{cs,home})]$.
Los aranceles del importador se aplican únicamente a compras extranjeras:

$$
P_{cs}=\sum_o(1+\tau_{cso})p_{os}q_{cso},\quad
P_c=\prod_sP_{cs}^{b_{cs}},\quad
X_{cs}=b_{cs}P_cA_c/P_{cs},\quad Q_{cso}=X_{cs}q_{cso}.
$$

Los mercados de bienes se equilibran en cada origen: $Y_{os}=\sum_cQ_{cso}$.
El precio del productor de BBB, S02 es el numerario en cada fecha.

## Costos reales de ajuste y cuentas

Sean $g_{cs}=K'_{cs}/K_{cs}-1$ e $I_{cs}=K'_{cs}-(1-\delta)K_{cs}$.
Todos los costos de ajuste consumen **bienes finales nacionales**:

$$
\Phi^K_{cs}=\frac{\kappa_K}{2}K_{cs}g_{cs}^2,\quad
\Phi^L_c=\frac{\kappa_L}{2\bar L_c}\sum_s(L_{cs}-L_{cs,prev})^2,\quad
\Phi^Q_{cs}=\frac{\kappa_Q}{2}\bar L_cb_{cs}(r_{cs}-r_{cs,prev})^2.
$$

La absorción es $A_c=C_c+\sum_s(I_{cs}+\Phi^K_{cs}+\Phi^Q_{cs})+\Phi^L_c$.
La recaudación $TR_c=\sum_{so}\tau_{cso}p_{os}Q_{cso}$ se devuelve mediante
transferencias de suma fija. El presupuesto del hogar es
$P_cA_c=\sum_s(w_{cs}L_{cs}+R_{cs}K_{cs})+TR_c$.
El hogar toma los precios y las transferencias como dados. La autarquía
financiera implica comercio equilibrado a precios del productor en cada fecha.
La inversión es reversible y puede ser negativa; el capital, el consumo, las
cantidades, los precios y el valor marginal del capital deben ser positivos.

Con $m_t=\beta(C_{t+1}/C_t)^{-\gamma}$, $q^K=1+\kappa_Kg$ y
$l_t=L_{1,t}/\bar L$, las condiciones interiores son

$$
q^K_t=m_t[R_{t+1}/P_{t+1}+1-\delta+\kappa_K(g_{t+1}+g_{t+1}^2/2)],
$$

$$
(w_{1,t}-w_{2,t})/P_t-2\kappa_L(l_t-l_{t-1})
+2m_t\kappa_L(l_{t+1}-l_t)=0,
$$

$$
(A_t/\bar L)\partial_{r_{s,t}}\log P_{s,t}
+\kappa_Q(r_{s,t}-r_{s,t-1})-m_t\kappa_Q(r_{s,t+1}-r_{s,t})=0.
$$

La derivada de abastecimiento multiplica la absorción bruta, incluida la
inversión y todos los costos. Eliminar cada costo produce, respectivamente,
la ecuación de Euler de capital sin fricción, igualdad salarial sectorial o
abastecimiento CES estático. El bienestar de equilibrio general no tiene por
qué mejorar de forma monótona al eliminar un costo.

## Aproximación terminal y aceptación

Se resuelve un estado estacionario auditado bajo el arancel final. El valor de
continuación es lineal en el capital terminal, con derivada $u'(C^*)/\beta$;
las derivadas respecto al trabajo y abastecimiento heredados son cero en el
estado estacionario. La última condición de capital es
$q^K_{T-1}=(C^*/C_{T-1})^{-\gamma}$ y desaparecen los términos futuros de
ajuste laboral y abastecimiento. La cola usa una aproximación de primer orden.

`run.converged` indica que pasaron las ecuaciones **finitas** y la auditoría
independiente. No certifica la precisión de la cola. El refinamiento exige:

| Cantidad | Límite |
| --- | --- |
| Error del solver y de contabilidad independiente | $10^{-8}$ |
| Cambio máximo de las primeras 20 coordenadas log/logit/abastecimiento | $10^{-6}$ |
| Cambio del consumo equivalente | $10^{-4}$ puntos porcentuales |
| Distancia máxima de las coordenadas terminales al estado estacionario | $10^{-5}$ |

Las últimas tres comparan soluciones finitas sucesivas aceptadas; todos los
prefijos de política deben coincidir y las extensiones deben conservar la
política terminal. Un fallo rompe la cadena de comparación. Son pruebas
observadas de refinamiento, no demostraciones de unicidad global ni cotas
rigurosas para una cola infinita.

El solver usa 16 incógnitas por fecha y un jacobiano disperso con tres
diagonales de bloques y derivadas por paso complejo. Sustituye un presupuesto
redundante por el numerario. `audit_reallocation` reconstruye ambos
presupuestos, bienes bilaterales, producción, pagos factoriales, balances
fiscales y externos, recursos de ajuste y todas las condiciones de optimalidad
a partir de los arreglos económicos. No usa el residuo guardado.
Los límites numéricos log/logit son ±30; se rechazan soluciones cercanas a
ellos. El presupuesto de evaluaciones cuenta `nfev` de SciPy y excluye el
trabajo del jacobiano. Véase
[mínimos cuadrados de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

`reallocation_stability` evalúa el sistema matricial compañero estacionario
local usando la [descomposición de Schur generalizada ordenada](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ordqz.html).
La representación estable exige 16 raíces estables, incluidos modos cero
de variables sin rezagos, una proyección bien condicionada y un residuo
matricial pequeño. La calibración predeterminada tiene 10 direcciones heredadas
activas. El fallo terminal rechaza una transición de forma predeterminada.
`require_stable_terminal=False` permite un experimento explícitamente finito;
la escalera de horizontes siempre conserva la comprobación. Este diagnóstico
es local y no demuestra optimalidad global de los hogares.

## Bienestar y calibración ilustrativa

`run.welfare` informa consumos equivalentes nacionales respecto al consumo
estacionario inicial. Incluye el consumo descontado observado, consumo
terminal constante y la corrección lineal por capital terminal. En utilidad
normalizada, la corrección es
$\beta^{T-1}(C^*/C_0)^{-\gamma}\sum_s(K_{s,T}-K_s^*)/C_0$.
Admite utilidad logarítmica y sus límites cercanos. El resultado mantiene
`tail_is_approximation=True`, incluso después de aprobar el refinamiento.

El costo de abastecimiento inicial **1.0** falló la comprobación estable
terminal (15 raíces estables en lugar de 16) y el piloto de horizontes.
Se conservó como rechazado y el valor ilustrativo predeterminado se cambió
explícitamente a **0.1**. No se relajó ninguna tolerancia ni se infiere un
límite superior universal para los costos de abastecimiento. Los demás
valores son $\beta=.96$, $\delta=.08$, $a=.3$, $\gamma=2$, $\sigma=4$,
$\kappa_K=4$, $\kappa_L=5$, peso nacional .75, pesos sectoriales iguales
y productividad y oferta laboral unitarias.

En el ejemplo, los cuatro casos pasan con 160 fechas. La comparación 80–160
del modelo completo tiene distancia de coordenadas $3.78\times10^{-8}$,
distancia de bienestar $6.23\times10^{-7}$ puntos porcentuales y distancia
terminal $1.45\times10^{-7}$. Los consumos equivalentes nacionales ilustrativos
son aproximadamente **+0.1233% para AAA** y **−0.1469% para BBB**. Estos valores
sintéticos no son recomendaciones arancelarias empíricas. Las tablas
sectoriales informan salarios e ingresos, pero el ingreso nacional compartido
no identifica el bienestar individual de los trabajadores.

Estudios de ajuste laboral como el de
[Artuc, Chaudhuri y McLaren](https://www.nber.org/system/files/working_papers/w13465/w13465.pdf)
motivan su tratamiento explícito. Este modelo continuo de hogar con ingreso
compartido no replica su modelo de elección de los trabajadores. Dimensiones
mayores, insumos MRIO, endeudamiento, heterogeneidad laboral, estimación
empírica y optimalidad global quedan fuera de esta primera implementación.
