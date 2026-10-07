> 🇬🇧 [English](../distributional_trade_ge.md) · 🇪🇸 Español

# Aplicación de comercio con conciliación entre EG y hogares

`puremacro.examples.distributional_trade_ge` conecta dos equilibrios comerciales
auditados con las canastas observadas por decil de la ENIGH 2024, exposiciones al
ingreso nominal y una bolsa de recaudación arancelaria conciliada explícitamente.
Compara devoluciones iguales y devoluciones a los cuatro deciles inferiores,
con una referencia sin devolución a los hogares, bajo canastas fijas y
preferencias Cobb–Douglas.

**La economía comercial es sintética.** El gasto, el número de hogares y los
salarios monetarios son observados; la tecnología, las participaciones por origen
y el cambio de política son supuestos. Es una aplicación pedagógica reproducible,
no una estimación del efecto de un arancel mexicano real. Incorporar ENIGH no
convierte un equilibrio sintético en evidencia empírica sobre comercio.

## Ejecutar la aplicación completa

```bash
python -m puremacro.examples.distributional_trade_ge \
  --output research_output/distributional_trade_ge --tariff-rate 0.10
```

El comando escribe seis escenarios de incidencia por decil, agregados, precios
al comprador, el mapeo de ingresos, todas las transacciones arancelarias, la
conciliación fiscal, los vectores de equilibrio, una figura, un informe y un
manifiesto con sumas de verificación. No requiere red; el [cargador ENIGH](trade_distributional.md)
verifica las observaciones incluidas. El manifiesto conserva la URL y la huella
de la fuente oficial, los supuestos, las huellas del código numérico, las
versiones y los residuos de equilibrio.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from puremacro.examples.distributional_trade_ge import run_application

with TemporaryDirectory() as directory:
    study = run_application(Path(directory), tariff_rate=0.10)
    print(study["results"]["fixed_baskets_equal_per_household"].summary())
    print(study["fiscal_allocation"][[
        "preferences", "rule", "fiscal_residual_mxn", "income_residual_mxn",
    ]])
```

## Qué se resuelve y qué se supone

El modelo tiene dos regiones hipotéticas simétricas, `HOME` y `REST`, y ocho
compuestos de consumo con los nombres de las categorías de la encuesta. No son
una concordancia estimada con industrias. El consumo inicial es de 100 unidades
del modelo por región; sus participaciones sectoriales provienen del gasto
ENIGH ponderado por hogares. Los costos intermedios representan 35% de la
producción y su participación importada es 20%. Las participaciones importadas
finales se especifican en el manifiesto: alimentos tiene 25%. La producción
combina coeficientes intermedios fijos con valor agregado Cobb–Douglas de trabajo
y capital. Los impuestos, aranceles y ahorro externo iniciales son cero.

El arancel grava alimentos de `REST` entregados a `HOME`, tanto para insumos
como para consumo final. Ambos estados usan `accounting="consistent"` y el
mismo numerario: el precio productor de alimentos de `HOME`. El puente auditado
recupera precios al comprador y verifica convergencia, etiquetas, aranceles y
ecuaciones. Todos los grupos comparten la misma canasta por origen dentro de
cada sector. En esta calibración simétrica particular, el arancel de 10% eleva
el precio de alimentos 2.5%; no es una estimación del traslado a precios.

## Ingresos y unidades monetarias

La conversión de una unidad del modelo a MXN trimestrales divide el consumo
observado agregado inicial por el consumo inicial de `HOME`. Es una
normalización explícita, no un tipo de cambio ni una calibración de cuentas
nacionales.

La participación laboral sintética es el cociente entre salarios monetarios
observados agregados y consumo observado, aproximadamente 0.9119. El resto del
ingreso factorial se representa mediante una **aproximación no observada al
otro factor**, asignada entre deciles proporcionalmente al consumo. No es
ingreso de capital observado en la encuesta.

| Exposición del hogar | Cambio nominal contrafactual |
|---|---|
| Salarios monetarios observados | Cambio salarial del EG de `HOME` |
| Aproximación supuesta al otro factor | Cambio de la renta del capital del EG de `HOME` |
| Exposición inicial menos consumo | Se mantiene fija dentro de cada decil |

Todo ingreso factorial incremental se gasta. Las exposiciones pueden superar
o quedar por debajo del consumo inicial de un decil; estos residuos se exportan
y suman cero a nivel nacional. Los ingresos expuestos ponderados por hogares
coinciden con las cuentas factoriales del EG convertidas a la misma escala en
ambos estados. Ninguna exposición incluye recaudación arancelaria.

## Distribuir una sola vez la recaudación existente

Para cada transacción intermedia y final se reconstruye
`recaudación = tasa_arancelaria × precio_productor × cantidad`. La suma debe
coincidir con los aranceles reportados y, al no existir otros impuestos, con el
presupuesto gubernamental. La bolsa es el cambio de recaudación de `HOME`,
convertido con la misma escala monetaria.

La devolución igual paga lo mismo a cada hogar. La focalización en los cuatro
deciles inferiores paga solo a esos grupos, por igual a cada hogar elegible.
Ambas agotan la misma bolsa. La conciliación verifica:

```text
devoluciones a hogares + recaudación retenida fuera de hogares = ingreso incremental
cambio presupuestario de hogares + recaudación retenida = cambio presupuestario del EG convertido
```

El cierre de EG ya incluye la recaudación fiscal en el ingreso nacional. La
aplicación mapea las remuneraciones factoriales por separado y después
distribuye esa recaudación existente una sola vez mediante la transferencia a
los hogares. Nunca vuelve a añadir aranceles al ingreso total del EG.

La referencia sin devolución retiene la bolsa fuera del gasto de los hogares,
manteniendo los precios del EG. Es una comparación condicional, no un cierre
alternativo de ahorro gubernamental resuelto. La focalización y las preferencias
alternativas también se evalúan después de resolver: la demanda de los hogares
no retroalimenta producción, comercio, salarios ni rentas del capital.

## Interpretar los resultados predeterminados

Con el arancel sintético de 10% y devoluciones iguales, la VE con canastas fijas
es +2.380% para el decil I y −0.677% para el X. La focalización en los cuatro
deciles inferiores cambia esas cifras a +8.084% y −1.238%. Los porcentajes usan
el consumo monetario inicial de cada grupo y son condicionales a los supuestos.

La VE monetaria agregada es cercana a cero y sensible a las preferencias:
−0.001055% con canastas fijas y devoluciones iguales, frente a +0.006066% con
Cobb–Douglas. La media de porcentajes ponderada por hogares responde otra
pregunta. Ninguna medida define una ordenación social del bienestar. Las medias
por grupo omiten heterogeneidad dentro del decil y no permiten intervalos de
confianza muestrales.

Véase [incidencia distributiva](trade_distributional.md) para las definiciones de
VE/VC, la atribución ordenada precios → ingresos → transferencias y los contratos
de entrada de los hogares.
