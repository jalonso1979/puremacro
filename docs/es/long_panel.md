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
