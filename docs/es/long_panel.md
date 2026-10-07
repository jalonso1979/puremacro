> 🇬🇧 [English](../long_panel.md) · 🇪🇸 Español

# El panel largo de cuentas nacionales

La función estándar `qna_panel` proporciona las cuentas nacionales trimestrales de la OCDE, las cuales comienzan en 1995 para la mayoría de los países europeos. `qna_long_panel` extiende retrospectivamente esa serie principal hacia el pasado para países seleccionados mediante el empalme por ratios de añadas nacionales archivadas.

```python
from puremacro.fetch import qna_long_panel

long, seams = qna_long_panel(["ESP", "JPN"], return_seams=True)

print(long.loc["ESP"].index.min())   # 1970-01-01   (OCDE estándar: 1995-01-01)
print(long.loc["JPN"].index.min())   # 1955-04-01   (OCDE estándar: 1994-01-01)
```

El esquema de columnas coincide con el de `qna_panel`, por lo que las funciones `qna_identity`, `qna_rebase` y `qna_contributions` operan de forma transparente. Junto a cada columna de valor se incluye una columna `src_<columna>` que identifica la añada estadística que produjo cada observación trimestral.

| País | Cobertura alcanzada | Ganancia histórica | Fuentes archivadas utilizadas |
|---|---|---|---|
| **España** | **1970T1** | **+100 trimestres** | Tablas del INE base 1995 (API JSON) + libro base 1986 |
| **Japón** | **1955T2** | **+155 trimestres** | Publicaciones históricas 93SNA y 68SNA de la Oficina del Gabinete |

---

## 1. Qué preserva el empalme por ratios

El único elemento que debe conservarse rigurosamente de una añada estadística antigua son sus **tasas de crecimiento**. Sus niveles en unidades monetarias responden a metodologías y años base que fueron posteriormente sustituidos, por lo que copiarlos directamente introduciría un escalón artificial en el trimestre de empalme.

El segmento histórico se re-escala multiplicándolo por el ratio promedio entre ambas series durante el intervalo de solapamiento muestral, empalmándose con la serie moderna. Factores de escala constantes (por ejemplo, series japonesas expresadas en miles de millones de yenes anualizados frente a millones trimestrales en la OCDE) quedan absorbidos automáticamente sin intervención manual.

---

## 2. La estabilidad del ratio como prueba de validez

Lo que un re-escalado por ratio no puede corregir es una discrepancia sistemática en la evolución temporal durante el período de solapamiento (*deriva del ratio*). Si el ratio muestra una tendencia marcada, ambas metodologías discrepan sobre la propia tasa de crecimiento económico, y el nivel empalmado dependería arbitrariamente del trimestre tomado como ancla.

Por ello, `qna_long_panel` calcula e informa la deriva del ratio en la tabla de costuras (`seams`):

```python
# Inspeccionar variables con derivas superiores a la tolerancia
seams[~seams.stable][["code", "column", "older", "overlap_n", "ratio_drift"]]
```

## 3. Empalmar muchas fuentes a la vez

`splice_sources` aplica las mismas reglas a un marco largo con columnas `code, date, variable, value, source`, con una fila por observación y tantas fuentes como haya. `to_long` construye ese marco a partir de cualquier panel ancho `(code, date)`. Se añade una columna `source` y se apilan los paneles:

```python
from puremacro.fetch.longpanel import splice_sources, to_long

long = pd.concat([to_long(qna).assign(source="oecd"),
                  to_long(old).assign(source="archive")])
panel, provenance, seams = splice_sources(long, {"oecd": 0, "archive": 1})
```

En cada par (código, variable), la fuente primaria es la mejor clasificada que tenga al menos `min_obs` observaciones. Sus valores nunca se tocan y sus huecos nunca se rellenan. Las fuentes más antiguas solo la prolongan hacia atrás. Cada una se reescala con el ratio medio de las primeras `min_overlap` fechas comunes, y un cambio de unidades, por ejemplo millones frente a miles de millones, aparece en `pow10`. Las variables declaradas `rate` o `ratio` en `kinds` se añaden sin reescalar, pero solo si la brecha en la costura no supera `rate_tol`. `seams` guarda una fila por cada intento, también los rechazados. `provenance` dice qué fuente es la primaria de cada serie y cuáles la prolongaron.
