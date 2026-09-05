> 🇬🇧 [English](../national_accounts.md) · 🇪🇸 Español

# Cuentas Nacionales Trimestrales

`puremacro.fetch` construye paneles de cuentas nacionales trimestrales a partir del servicio SDMX de la OCDE y se encarga del trabajo analítico posterior a la descarga: armonizar países bajo un año base de precios común, verificar las identidades contables y descomponer el crecimiento económico en contribuciones sectoriales y de demanda.

Con excepción de `qna_panel`, `qna_labor` y `qna_countries`, ninguna función de este módulo accede a la red. Un panel descargado en vivo y un archivo CSV archivado operan con idéntica interfaz.

```python
from puremacro.fetch import (
    qna_panel, qna_countries, qna_rebase, qna_identity, qna_contributions,
)

panel = qna_panel(["USA", "JPN", "DEU"], start="1995",
                  output=True, income=True, real=True)
```

---

## 1. Los tres componentes de la construcción

`qna_panel` devuelve simultáneamente: niveles a precios corrientes (nominales), deflactores implícitos de precios y — con `real=True` — medidas de volumen en términos reales, vinculadas exactamente por la identidad:

$$\text{Nominal} = \frac{\text{Real} \times \text{Deflactor}}{100}$$

para cada componente de la contabilidad nacional.

---

## 2. Los tres enfoques de medición del PIB

La OCDE publica tres estimaciones independientes del PIB procedentes de distintos sistemas estadísticos. `qna_panel` proporciona acceso unificado a las tres:

| Opción | Enfoque de medición | Identidad contable | Registro en puremacro |
|---|---|---|---|
| *(siempre)* | Gasto | $Y = C + G + I + X - M$ | `QNA_COMPONENTS` |
| `output=True` | Producción / Oferta | $Y = \sum VA_j + (D21 - D31) + YA1$ | `QNA_ACTIVITIES` |
| `income=True` | Renta / Ingreso | $Y = D1 + B2A3G + (D2 - D3)$ | `QNA_INCOME` |

Dos observaciones metodológicas codificadas en los registros del módulo:
- **Partidas informativas frente a sumandos**: `va_mfg` está contenida dentro de `va_ind`, y `va_services` agrega siete subsectores ya detallados. `QNA_VA_ADDITIVE` lista las diez ramas aditivas que suman exactamente el valor añadido total; `QNA_VA_MEMO` lista las partidas informativas. Sumar todas las columnas `va_*` duplicaría indebidamente un tercio de la economía.
- **Las rentas solo se expresan a precios corrientes**: No existen medidas de volumen para la remuneración de asalariados, por lo que carecen de deflactor y de columna `_real`.

---

## 3. Descomposición de contribuciones al crecimiento

`qna_contributions` calcula la aportación de cada componente de la demanda agregada a la tasa de variación trimestral del PIB, ponderando cada componente por su peso relativo en el período precedente:

```python
contributions = qna_contributions(panel)
print(contributions.head())
```
