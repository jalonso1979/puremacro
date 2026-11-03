> 🇬🇧 [English](../sovereign_extensions_validation.md) · 🇪🇸 Español

# Extensiones soberanas: ecuaciones contrastadas independientemente

PureMacro 6.1.0 incorpora referencias independientes en mallas
pequeñas para deuda larga, reestructuración y el planificador con colateral.
Complementan el [estudio Arellano de corto plazo](arellano_replication.md);
no replican momentos empíricos publicados de deuda larga o Mendoza.

## Deuda larga y reestructuración

`tools/reference_validation/sovereign_extensions.py` implementa por separado
el punto fijo de valores y precios con NumPy/SciPy. Con 17 estados de deuda y
5 de ingreso, las tasas de vencimiento 0,2, 0,5 y 1 cumplen: error máximo de
valor menor que `1.8e-9`, de precio menor que `1.9e-14`, y políticas modales de
deuda idénticas. La quita de 20% y el aumento de duración de 50% coinciden con
probabilidades de impago calculadas independientemente dentro de `7e-15`.
Los contrastes desactivan proyección monótona y aceleración Howard; una derivada
proyectada de precios no es por sí sola una prueba de dilución estructural.

Para `delta < 0.999`, el solucionador existente suaviza decisiones de deuda
con ponderaciones Boltzmann y `tau = min(0.005, max(0.0001, sigma_eps/2))`.
El valor de pago es el objetivo ponderado **sin bonificación de entropía o
preferencias**. Es una aproximación de política probabilística, no el máximo
determinista ni el valor inclusivo de un modelo de utilidad aleatoria para la
deuda. Es distinta de la perturbación de preferencias entre impagar y pagar.
La diferencia respecto del máximo determinista es aproximadamente 0,00528 y
0,00425 unidades de valor en los dos casos de deuda larga, y queda registrada.

`policy_b` sigue siendo la decisión modal. Simulación, diagnósticos ergódicos,
recuperación y valoración de derechos de supervivencia ahora usan toda la
distribución `metadata['debt_choice_probabilities']`. La deuda corta conserva
exactamente soluciones y simulaciones del algoritmo 6.0.1 archivado; cambian
los momentos simulados de deuda larga.

Una quita sola reutiliza el equilibrio y cambia la deuda heredada. Cambiar el
vencimiento resuelve otra vez la misma calibración con la nueva tasa. Se rechazan
entradas y escenarios sin convergencia, el período cero usa el ingreso elegido
y se informa el redondeo a la malla de deuda. Son escenarios impuestos, sin
negociación ni recuperación determinada endógenamente.

Para derechos de supervivencia con prestamistas neutrales al riesgo y recuperación
cero, $q_h=S_h/(1+r^*)^h$. La columna heredada `risk_premium_component` resta la
pérdida de supervivencia continuamente compuesta del diferencial anual discreto.
Es un ajuste de capitalización, no una prima identificada de riesgo o dilución.
Se contrasta con riesgo constante y recursiones de la distribución completa.

## Planificador con colateral y diagnósticos medidos

Es un planificador ilustrativo con costos de ajuste y $b'\leq\kappa q k'$.
No implementa el modelo competitivo completo de Mendoza con financiamiento de
capital de trabajo e insumos importados. Una referencia exhaustiva sin
aceleración Howard coincide exactamente en deuda, capital y consumo sobre una
malla 7 × 6 × 3 para proporciones de colateral 0,3 y 0,9. El primer caso tiene
estados vinculantes. Los errores de valor son menores que `2.2e-9`, y los residuos
finales de Bellman menores que `1e-10`. Se rechazan estados inviables en vez de
asignarles consumo positivo artificial.

- `collateral_binding_fraction` mide estados de malla donde el colateral excluye
  el nodo siguiente de mayor deuda al capital elegido. No es una tasa de crisis.
- `crisis_probability` se mide tras el calentamiento de una simulación de 2.500
  períodos: colateral vinculante, consumo que cae y cuenta corriente que aumenta.
  `simulate()` devuelve estas observaciones, capacidad y holgura.
- La capacidad promedia `kappa*q*k_next` sobre la trayectoria. No multiplica
  promedios separados de precio y capital. El colchón compara esa capacidad
  con la deuda media del período siguiente.
- La amplificación divide caídas medias de consumo entre productividad media
  y baja, con y sin restricción, evaluadas en los mismos estados simulados de
  deuda y capital y ponderadas por frecuencia de visita. Si la respuesta base
  es cero o creciente, o no hay convergencia, se devuelve `NaN` con un estado.
  Se eliminan la selección del máximo, el sustituto 1,35 y los límites inferiores
  impuestos. El cociente puede ser menor que uno; no es una elasticidad causal.

Son diagnósticos reproducibles. La convergencia económica de malla, los momentos
publicados y un modelo competitivo completo siguen siendo trabajo separado.

## Reproducir

Parámetros, hashes, umbrales y errores están en
`tests/fixtures/sovereign_extensions/independent_validation.json`.

```bash
python -m tools.reference_validation.check_sovereign_extensions --output /tmp/sovereign-extensions.json
python -m pytest tests/test_sovereign/test_extension_validation.py -q
```

El cuaderno 73 explora la interfaz con las mediciones corregidas. Su malla gruesa
no constituye la referencia independiente ni una calibración de política.
