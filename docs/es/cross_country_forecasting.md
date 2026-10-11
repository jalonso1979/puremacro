> [English](../cross_country_forecasting.md) · Español

# De un panel congelado a una comparación de pronósticos

**Disponible en puremacro 4.8.0.** Instale esta versión para ejecutar el flujo.
Usa una copia fija de WDI del Banco
Mundial para Estados Unidos, México y Brasil. La aplicación funciona sin red
y exporta datos, pronósticos, tablas de precisión, figura, metadatos y un
cartucho portátil `.pmz`.

```bash
python -m pip install "puremacro==4.8.0"
python -m puremacro.examples.cross_country_forecasting --output research_output/cross_country_forecasting
```

Consulte `report.md` y `accuracy.csv`. El archivo `manifest.json` registra el
diseño, las sumas de verificación de datos y código, las versiones utilizadas
y la suma de verificación de cada resultado. La ejecución desde una rueda
instalada utiliza únicamente las dependencias base y no consulta la API.

## Pregunta y diseño fijo

Se comparan tres métodos sencillos para pronosticar el crecimiento anual del
PIB real a un año. Hay 65 niveles anuales por país, de 1960 a 2024, del indicador
`NY.GDP.MKTP.KN`. Las unidades son millones de moneda local constante, con el
año base o de referencia de cada país. Se compara el crecimiento dentro de
cada país: `100 * (log(PIB_t) - log(PIB_(t-1)))`, no sus niveles monetarios.

Los años objetivo son 2000–2024. La muestra de estimación comienza en 1961 y se
amplía hasta el año anterior al objetivo. Los métodos son:

- **Crecimiento cero:** el nivel del PIB permanece igual.
- **Media histórica:** promedio del crecimiento en la muestra de estimación.
- **AR(1):** regresión del crecimiento sobre una constante y su primer rezago,
  mediante el estimador público `fit_var`.

Países, transformación, muestra, rezago y métodos se fijaron antes de comparar
los errores. No hay selección posterior, imputación ni estandarización con la
muestra completa. Se conservan 2020 y la recuperación. Los años ausentes,
niveles no positivos, muestras insuficientes y regresiones AR no identificadas
generan errores en vez de cambiar silenciosamente la muestra.

`forecasts.csv` contiene 225 combinaciones país/método/año, con origen, objetivo,
rango y tamaño de entrenamiento, dato observado, pronóstico y error.
`accuracy.csv` presenta RMSE, MAE y error medio. El error es observado menos
pronosticado, en puntos porcentuales de crecimiento logarítmico anual. La
tabla conjunta pondera por igual a los países; su RMSE es la raíz de la media
del error cuadrático medio de cada país. Las tablas de cobertura y exclusiones
identifican 1960, sin crecimiento previo, y 1961–1999, usados para entrenamiento
inicial en lugar de evaluación.

## Interpretación

Es una **evaluación histórica con datos revisados de una edición reciente**.
La descarga se capturó el 11 de octubre de 2026 UTC (10 de octubre en Ciudad
de México), y WDI declara actualización del 8 de octubre de 2026. Estas fechas
no reconstruyen la información disponible en cada origen. No se modelan los
rezagos de publicación. Por tanto, no demuestra desempeño en tiempo real.

Las pérdidas describen tres países y 25 años objetivo. No demuestran
significancia estadística, superioridad general de un método, efectos causales
ni intervalos predictivos calibrados. Los métodos simples son referencias
sustantivas: no se presupone que el AR(1) los supere.

## Evidencia portátil

```python
from puremacro import pocket

cartridge = pocket.load("research_output/cross_country_forecasting/cross_country_forecasting.pmz")
cartridge.verify()
print(cartridge["levels"].attrs["source"])
print(cartridge["accuracy"])
```

El nuevo esquema v2 conserva los atributos admitidos de los DataFrames:
diccionarios, claves de tupla, listas, tuplas, escalares numéricos y fechas.
No usa pickle; objetos no admitidos y ciclos generan `StoreError`. Los archivos
v1 siguen siendo legibles y sus cartuchos verificables, aunque ese formato no
conservaba atributos. Los lectores necesitan puremacro 4.8.0 o posterior para
leer v2; los archivos v1 no requieren conversión. Modificar los metadatos de un
cartucho nuevo invalida su verificación.
Las sumas detectan cambios; no son firmas de autenticidad.

Las tablas CSV son deterministas. La fecha de creación del cartucho cambia al
repetirlo, por lo que no se exige igualdad binaria de todo el archivo `.pmz`.

## Reconstrucción de los datos

El directorio `reviews/2026-10-10-cross-country-forecasting/source/` contiene las
respuestas originales del catálogo de países y del indicador. Desde la raíz:

```bash
PYTHONPATH=. python tools/build_cross_country_forecasting_data.py --output-dir /tmp/puremacro-gdp-rebuild
```

El conversor verifica sus hashes, reproduce esas respuestas en el transporte
HTTP de `wdi_panel` y contrasta cada observación con una extracción directa del
JSON. No accede a la red ni acepta revisiones nuevas implícitamente. El diseño
fijo y la validación se conservan en el directorio de revisión.
La reproducción por usuarios externos sigue siendo un objetivo de adopción.

Fuente: Banco Mundial, *World Development Indicators*, PIB (moneda local
constante), [NY.GDP.MKTP.KN](https://data.worldbank.org/indicator/NY.GDP.MKTP.KN),
[CC BY 4.0](https://datacatalog.worldbank.org/public-licenses#cc-by).
