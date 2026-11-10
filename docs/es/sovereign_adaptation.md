# Adaptación y riesgo soberano

[English](../sovereign_adaptation.md)

`puremacro.sovereign.adaptation` resuelve conjuntamente inversión pública en
adaptación, deuda externa de un período y default, sobre mallas finitas de
deuda/capital y un horizonte finito. Los acreedores valoran el default futuro
condicional en la nueva deuda y el capital de adaptación elegidos hoy. La
propagación exacta de probabilidades distingue nuevos defaults de economías
que ya estaban excluidas del crédito.

Este modelo experimental tiene desastres de Markov exógenos, utilidad
logarítmica, impuestos de suma fija y exclusión permanente hasta la última
fecha. Los hogares no se endeudan directamente. No incorpora bancos domésticos,
temperatura, emisiones, reingreso ni aproximación terminal de horizonte infinito.
Su contrato de desastre condona parte de un pago de un período; no lo difiere
ni reproduce una cláusula de desastre en deuda de largo plazo.

## Utilizar el modelo

```python
import numpy as np
from puremacro.sovereign import (
    SovereignAdaptationParameters, solve_sovereign_adaptation,
    audit_sovereign_adaptation, audit_adaptation_distribution,
)

result = solve_sovereign_adaptation(
    income=[1.0, 1.0], damage=[0.0, 0.4], disaster=[False, True],
    transition=[[0.9, 0.1], [0.75, 0.25]],
    debt_grid=np.linspace(0.0, 0.4, 21),
    adaptation_grid=np.linspace(0.0, 0.3, 11), horizon=6,
    parameters=SovereignAdaptationParameters(grant_rate=0.5),
)
path = result.distribution([1.0, 0.0])
assert audit_sovereign_adaptation(result)["accepted"]
assert audit_adaptation_distribution(path)["accepted"]
print(path.to_frame()[["investment", "disaster_damage", "new_default_probability"]])
print(path.value)
```

`result.to_frame(date=t)` muestra las cuentas óptimas realizadas en cada estado
con acceso al mercado; `excluded=True` muestra los estados excluidos. Incluye
impuestos, consumo, daño, pérdidas por exclusión, subsidios, servicio de deuda,
condonación contractual, impagos, inversión y retiro de capital. La utilidad de
pago forzado inviable queda ausente con `repayment_feasible=False`; el consumo
seleccionado es positivo. `pricing_frame(t)` presenta precio y probabilidad de
default para cada elección de deuda/capital siguientes. Las tablas son copias
nuevas y los arreglos almacenados tienen memoria propia de solo lectura.
Hay informes Markdown, LaTeX y Typst.

`distribution(initial_probabilities, debt_index=0, adaptation_index=0,
market_access=True)` comienza en nodos exactos y propaga probabilidades sin
Monte Carlo. Un estado inicialmente excluido debe tener deuda cero.
`new_default_probability` cuenta entradas, mientras que `excluded_probability`
incluye los defaults actuales y la exclusión previa.

## Fechas, financiamiento y daño

Las fechas son (t=0,\ldots,T-1). El estado corriente (z) contiene ingreso
sin daño (y_z>0), indicador de desastre (h_z), fracción de daño
\(\theta_z\in[0,1)\) y transición (P_{z,:}). El daño positivo requiere un
indicador de desastre. El riesgo es exógeno: la adaptación reduce la severidad,
no la probabilidad del desastre.

El capital instalado (a) protege el ingreso antes de depreciarse:

\[
\ell(a,z)=y_z\theta_z e^{-\eta a},\qquad Y(a,z)=y_z-\ell(a,z).
\]

El gobierno observa la deuda facial heredada (b\geq0), el capital instalado
y el choque. Decide pago o default. Si paga, elige la deuda siguiente (b')
y el capital instalado siguiente (a') sobre las mallas suministradas.
La inversión y el retiro de capital son

\[
i=\max\{a'-(1-\delta)a,0\},\quad
s=\max\{(1-\delta)a-a',0\},\quad a'=(1-\delta)a+i-s.
\]

La inversión nueva protege desde la fecha siguiente. El retiro libre **no
genera ingresos de reventa**. Permite transiciones exactas sin interpolación
ni redondeo obligatorio hacia arriba tras depreciar. Una malla gruesa puede
inducir retiro adicional; por eso se refinan conjuntamente deuda y capital.

Un donante reembolsa la fracción (g\in[0,1]) de la inversión. El contrato
promete (m_zb), donde (m_z=1-\rho h_z) y \(\rho\in[0,1]\) es la condonación
en desastre. Bajo pago, presupuesto público y consumo del hogar son

\[
\tau=m_zb+i-gi-q_t(b',a',z)b',\qquad c=Y(a,z)-\tau.
\]

El default cancela el servicio contractual restante, impide emitir inmediatamente
y causa exclusión durante todas las fechas restantes. Cada período excluido
destruye la fracción \(\phi\in[0,1)\) del producto después del daño. Sigue siendo
posible adaptar usando ingreso y el mismo reembolso del donante:

\[
c^E=(1-\phi)Y(a,z)-(1-g)i.
\]

Daño, pérdida por exclusión y condonación contractual se registran por separado.
La deuda facial es servicio pagado más condonación contractual más servicio
restante impagado en default. En cada estado, los recursos satisfacen

\[
c+i+\text{servicio}=y_z-\ell-\text{pérdida por exclusión}
 +\text{recursos de emisión}+\text{subsidio}.
\]

## Solución hacia atrás y precios

El gobierno maximiza \(\mathbb E\sum_t\beta^t\log c_t\), con
(0<\beta\leq1). Para cada capital/choque, se agotan todas las elecciones de
capital en exclusión y todos los pares deuda/capital bajo pago. Los acreedores
externos neutrales al riesgo observan ambas elecciones. Para (t<T-1),

\[
q_t(b',a',z)=\frac{1}{R}\sum_{z'}P_{z,z'}m_{z'}
 [1-d_{t+1}(b',a',z')].
\]

El default futuro se conoce por inducción hacia atrás; no se itera sobre precios
ni se impone monotonicidad. El precio de referencia sin default es
\(\sum Pm/R\), posiblemente menor que (1/R) por la condonación contractual.
No toda diferencia respecto de (1/R) es una prima de default.

En (T-1), emisión e inversión nueva son cero, y también el valor de continuación
y rescate. El capital existente sobrevive físicamente la depreciación, pero
no tiene valor de continuación modelado. Los índices terminales de capital son
`-1` y los arreglos terminales de precios tienen ceros de relleno.
`pricing_frame()` rechaza fechas terminales: esos ceros no son cotizaciones.
El default terminal aún compara consumo bajo pago y exclusión.

Gana el pago si la ventaja de default no supera `decision_tolerance` (por
defecto `1e-12`, permitido `[0,1e-8]` en utilidad logarítmica). Empates exactos
entre acciones de pago prefieren menor nueva deuda y luego menor capital
siguiente; en exclusión, menor capital. El pago inviable mantiene valor `-inf`
internamente, sin piso de consumo. El desbordamiento produce un error. Las
mallas comienzan en cero exacto y crecen estrictamente; `[0]` impone ausencia
de deuda nueva o adaptación. Las filas de transición suman uno con tolerancia
`1e-12` y se normalizan dentro de ese margen de redondeo. Los límites son 30
fechas, 500,000 celdas de estados con acceso y una cota conservadora de 250
millones de evaluaciones estado/acción.

## Validación e interpretación del bienestar

`audit_sovereign_adaptation()` recalcula independientemente todos los máximos
de Bellman, valores/factibilidad de las acciones, precios, default y cuentas.
`audit_adaptation_distribution()` comprueba flujos de masa, fechas y la igualdad
entre valor inicial de Bellman y **logaritmo** esperado descontado del consumo.
No usa el logaritmo del consumo medio. La tolerancia predeterminada es `1e-9`,
normalizando errores por `1 + abs(reference)`. Comprueba el modelo discreto
especificado, no precisión en estados continuos u horizonte infinito.

Las pruebas incluyen un programa dinámico escalar separado, enumeración de
todas las historias de choques en modelos pequeños, una condición analítica
de inversión de dos fechas, cambios de unidades/orden y límites sin desastres,
adaptación o nueva deuda. El umbral terminal es (m_zb>\phi Y(a,z)), sujeto a
la tolerancia de empate. Con pérdida de exclusión cero, la deuda positiva sin
garantías tiene precio cero en este modelo finito. La condonación total elimina
el pago contractual en ese estado y se refleja explícitamente en el precio.

Para dos distribuciones con el mismo beta y horizonte,
`alternative.consumption_equivalent(reference)` devuelve
\(\exp[(V_{alt}-V_{ref})/\sum_t\beta^t]-1\). Escala uniformemente el consumo
de referencia. Mide bienestar del hogar doméstico en este modelo; el costo
del donante se registra como subsidio, pero su bienestar no se incluye.

## Reproducir el experimento

```sh
python -m puremacro.examples.sovereign_adaptation --output adaptation-en --lang en
python -m puremacro.examples.sovereign_adaptation --output adaptation-es --lang es
```

El ejemplo sintético de seis fechas compara base, subsidio de 50%, condonación
total en desastre, ausencia de adaptación y ausencia de nueva deuda, con mallas
conjuntas de 21/11, 41/21 y 81/41 puntos. Conserva mallas y criterios de
refinamiento. El piloto de condonación tocó el límite de deuda, por lo que se
añadieron dos comprobaciones con dominio más amplio y espaciamiento fijo.
La comparación original de inversión sin deuda nueva incumplió su límite;
se reportan por separado comprobaciones de capital con 81/161 nodos, conservando
ese fallo. La comparación gruesa de subsidio conserva también una bandera
de límite debida al redondeo numérico.
Se conservan masa en límites, retiro de capital y comparaciones fallidas. Se
guardan arreglos completos de políticas/precios/valores y masas de probabilidad,
junto con cuentas, bienestar, huellas y auditorías. No es una calibración empírica.

[Mallucci (2020)](https://www.federalreserve.gov/econres/ifdp/files/ifdp1291r1.pdf)
motiva el mecanismo de desastre/deuda. La tecnología de adaptación, horizonte
finito, exclusión permanente y condonación de un período son supuestos distintos;
no se replica el modelo de deuda de largo plazo de ese artículo.
