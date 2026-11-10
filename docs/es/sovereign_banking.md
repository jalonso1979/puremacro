# Riesgo soberano y bancos domésticos

[English](../sovereign_banking.md)

`puremacro.sovereign.banking` resuelve una economía contractual de tres fechas.
El default del gobierno reduce el patrimonio bancario antes de la oferta de
crédito privado. Inversionistas externos valoran las decisiones de pago, y el
gobierno elige deuda sobre una malla finita declarada. Se conservan todos los
equilibrios **puros** de precio/default; el usuario selecciona explícitamente
el precio mayor o menor en cada nivel de deuda.

Es un modelo experimental separado. No incorpora bancos al solver recursivo
de Arellano. La exposición doméstica es obligatoria, el capital bancario del
hogar está inmovilizado y el hogar no ahorra ni se endeuda directamente. Estas
restricciones son supuestos del modelo, no comportamientos estimados. Los
bancos obtienen financiamiento externo seguro después de la decisión soberana,
sujeto a un límite de apalancamiento. No hay recapitalización, default selectivo,
corridas bancarias, emisiones recurrentes ni solución de equilibrios mixtos.

## Resolver e inspeccionar

```python
import numpy as np
from puremacro.sovereign import (
    SovereignBankingParameters, solve_sovereign_banking,
    audit_sovereign_banking,
)

result = solve_sovereign_banking(
    income=[0.7, 1.0, 1.3],
    productivity=[0.6, 0.9, 1.2],
    probabilities=[0.25, 0.50, 0.25],
    debt_grid=np.linspace(0.0, 0.6, 51),
    parameters=SovereignBankingParameters(domestic_share=0.4),
    price_selection="highest",
)
assert audit_sovereign_banking(result)["accepted"]
print(result.equilibria_frame().query("selected"))
print(result.to_frame()[["bank_equity", "credit", "consumption1", "consumption2"]])
print(result.debt_frame().query("multiple"))
assert len(result.candidates_frame()) == 51 * 2**3
```

`equilibria_frame()` conserva cada precio consistente, probabilidad de default,
utilidad del hogar y crédito esperado. `debt_frame()` registra multiplicidad,
ausencia de equilibrios puros y el índice seleccionado (`-1` si no existe).
`candidates_frame()` conserva todas las máscaras rechazadas y sus motivos.
`to_frame()` presenta las cuentas realizadas con la deuda/precio elegidos, una
fila por estado conjunto de ingreso/productividad. Las tablas son copias nuevas;
los arreglos almacenados tienen memoria propia y son de solo lectura. Se admiten
informes Markdown, LaTeX y Typst.

`counterfactual_frame()` fuerza pago y default **con la misma deuda y precio
contratados**. Son comparaciones contables condicionales, no nuevos equilibrios
con precios recalculados. El pago inviable conserva su consumo no positivo,
`feasible=False` y utilidad logarítmica ausente. Las asignaciones seleccionadas
tienen consumo positivo y utilidad finita. `sovereign_bank_accounts()` ofrece
el mismo cálculo condicional para un contrato especificado por el usuario.

## Fechas, propiedad y financiamiento

Todos los stocks y flujos usan un bien. (R>0) es el retorno seguro bruto por
fecha; (0<\beta\leq1) descuenta la utilidad. Los hogares son propietarios de
bancos y empresas. Los estados conjuntos ((Y,a)) tienen probabilidades positivas
y se observan en la fecha 1. La productividad (a) determina el producto de los
proyectos en la fecha 2 y se conoce antes de prestar.

En la fecha 0, el ingreso es (Y_0) y el capital bancario fijo es (E_0<Y_0).
El gobierno emite valor facial (B\geq0) a precio (q), transfiere los recursos
al hogar y exige que los bancos domésticos mantengan la fracción (0\leq s<1)
de la deuda. Los bancos pagan (qsB) y ahorran el resto en un activo externo seguro:

\[
A_0=E_0-qsB\geq0,\qquad C_0=Y_0-E_0+qB.
\]

Los acreedores soberanos externos compran el resto. La restricción (s<1)
mantiene un acreedor externo marginal que determina el precio común. Los bancos
no pueden endeudarse en la fecha 0 ni recibir capital adicional del hogar en la 1.

En la fecha 1, el gobierno elige default completo (d=1) o pago (d=0).
El pago impone al hogar un impuesto ((1-d)B). Una penalización exógena separada
destruye la fracción \(\phi\) del ingreso corriente en default:

\[
C_1=(1-\phi d)Y-(1-d)B,\qquad
E_1=RA_0+(1-d)sB.
\]

El patrimonio bancario se mide **después** de liquidar la deuda. Dado el mismo
contrato inicial, el default lo reduce exactamente en (sB). Esta pérdida no se
resta otra vez del ingreso: sus consecuencias reales surgen por el crédito y
los rendimientos futuros. La penalización exógena \(\phi dY\) se registra por
separado y puede fijarse en cero.

Los bancos financian proyectos competitivos con producción (F(L)=a\log(1+L/k)),
donde (k>0) es una escala de crédito. Obtienen financiamiento externo (D) a
retorno (R), pero los préstamos no pueden exceder \(\lambda E_1\), con
\(\lambda\geq1\). La asignación del mercado de crédito es

\[
L=\min\{\max(a/R-k,0),\lambda E_1\},\quad
D=\max(L-E_1,0),\quad S=\max(E_1-L,0).
\]

La tasa bruta del préstamo es \(r_L=\max\{R,a/(k+L)\}\). Sin demanda de crédito,
la tasa cotizada es (R). Si el patrimonio es cero y hay proyectos rentables,
el crédito es cero y se cotiza (a/k). El financiamiento (D) es un pasivo
bancario externo, **no un depósito del hogar ni una garantía gubernamental**.
Los rendimientos de los proyectos se conocen al prestar y cubren ese pasivo.
(S) es el activo seguro del banco. Su balance es (L+S=E_1+D).

En la fecha 2 se liquidan los bancos. Las utilidades de la empresa son
(F-r_LL), los dividendos bancarios (r_LL+RS-RD), y el ingreso terminal del
hogar es (Y_2>0):

\[
C_2=Y_2+F(L)+RS-RD.
\]

El soberano elige en la fecha 1 para maximizar
\(\log C_1+\beta\log C_2\). En la fecha 0 maximiza
\(\log C_0+\beta\mathbb E[\log C_1+\beta\log C_2]\).
Estos valores de utilidad comparan las asignaciones de consumo especificadas
en este modelo; no son estimaciones empíricas de bienestar.

## Precios, multiplicidad y alcance numérico

Para cada punto de deuda y cada una de las (2^n) máscaras de default por
estado, se fija \(q=\sum_j p_j(1-d_j)/R\). Se recalculan ambas acciones del
gobierno a ese precio y se conserva la máscara si las decisiones coinciden y
(A_0\geq0). El precio afecta las reservas bancarias; puede haber más de un
precio consistente. Se conservan los equilibrios de default total con (q=0):
esa deuda no obtiene recursos. La ausencia de equilibrios puros no se reemplaza
por interpolación ni por una solución mixta.

`price_selection="highest"` o `"lowest"` selecciona un precio por nivel de
deuda disponible. Después el gobierno elige el mejor punto de la malla. La
selección es un supuesto explícito, no un resultado de unicidad económica.
Se elige pago cuando la ganancia de default no supera `decision_tolerance`
(por defecto `1e-12`, intervalo permitido `[0,1e-8]` en utilidad logarítmica).
Los empates exactos de utilidad en fecha 0 eligen la deuda menor. La auditoría
registra la mayor ganancia por desviarse de la decisión y los casi empates.
Se informa contacto con el límite superior de deuda. Una malla finita no
establece el óptimo continuo ni un equilibrio de horizonte infinito.

Los insumos son vectores reales finitos. La deuda comienza en cero exacto y
crece estrictamente. Las probabilidades son positivas y suman uno con tolerancia
`1e-12`; se normalizan las sumas dentro de ese margen de redondeo. Se admiten
hasta 12 estados y 200,000 pares deuda/máscara. Se rechazan carteras inviables;
el desbordamiento numérico genera un error, sin truncar resultados.

## Cuentas y límites de referencia

La auditoría escalar independiente reconstruye cada máscara de precio y decisión
por estado, asignaciones, selección declarada y optimalidad discreta de deuda.
Comprueba presupuestos de bancos, gobierno y hogares, restricciones y
complementariedad del crédito y las tres identidades de recursos:

\[
\begin{aligned}
C_0+A_0 &= Y_0+(1-s)qB,\\
C_1+L+S &= Y-\phi dY+RA_0-(1-s)(1-d)B+D,\\
C_2+RD &= Y_2+F(L)+RS.
\end{aligned}
\]

La tolerancia de auditoría predeterminada es `1e-9`; las comparaciones numéricas
dividen el error entre `1 + abs(reference)`. Comprueba ecuaciones y malla finita,
sin acotar el error de aproximación económica. Pruebas independientes optimizan
numéricamente el excedente de crédito y recuperan todos los equilibrios de
ejemplos pequeños.

Con **exposición doméstica cero**, el patrimonio bancario es (RE_0) bajo ambas
acciones y hay default si y solo si (B>\phi Y), salvo la tolerancia de empate
declarada. Se recupera la base correspondiente de tres fechas, no el modelo
recursivo de Arellano. Si ambas restricciones de crédito son vinculantes, el
default condicional reduce el crédito en \(\lambda sB\). Si ambas asignaciones
son irrestrictas, el crédito puede no cambiar. Estas referencias distinguen
pérdidas soberanas de pérdidas de crédito.

## Reproducir la ilustración

```sh
python -m puremacro.examples.sovereign_banking --output banking-en --lang en
python -m puremacro.examples.sovereign_banking --output banking-es --lang es
```

El ejemplo sintético fijo de tres estados compara exposiciones (s=0,.2,.4,.6),
ambas selecciones de precio y mallas de 51/101/201 nodos de deuda en `[0,.6]`.
Guarda candidatos, equilibrios, cuentas, arreglos exactos, huellas, auditorías y
comparaciones entre mallas. Un panel de estrés contractual separado fija
`B=.25, q=.8`; explícitamente no recalcula precios de equilibrio. Se conservan
los incumplimientos de los criterios de refinamiento. La distribución de estados
es fija: las comprobaciones refinan solamente la malla de deuda.

El mecanismo de balances está motivado por [Gennaioli, Martin y Rossi,
Sovereign Default, Domestic Banks, and Financial Institutions](https://crei.cat/wp-content/uploads/users/working-papers/sovereign_default.pdf).
Las fechas, tecnología, carteras obligatorias y reglas de selección anteriores
definen esta implementación; no replica las carteras óptimas de ese artículo.
