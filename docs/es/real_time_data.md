> 🇬🇧 [English](../real_time_data.md) · 🇪🇸 Español

# Datos en tiempo real (añadas / vintages)

Una **añada** (*vintage*) es una edición histórica específica publicada de una serie temporal. Los institutos oficiales de estadística revisan continuamente sus estimaciones a medida que incorporan información más completa, por lo que cada trimestre de referencia acumula una secuencia de ediciones publicadas. La diferencia entre la primera estimación (avanzada o preliminar) y la última disponible define la revisión $r_t = y_{t}^{\text{final}} - y_{t}^{\text{preliminar}}$ analizada en la literatura de datos en tiempo real.

```python
from puremacro.fetch import vintage_panel

rev = vintage_panel(["USA", "DEU", "ESP", "MEX"], series="B1GQ", freq="Q")

rev.coverage()            # Cobertura temporal recuperada por país
rev.revisions("DEU")      # Datos preliminares, finales y revisión r_t por trimestre
rev.news_or_noise("DEU")  # Contraste econométrico de Mankiw-Shapiro
rev.news_or_noise_panel() # Tabla ordenada para todo el corte transversal
```

`news_or_noise_panel()` evalúa la célebre proposición empírica de si las revisiones macroeconómicas constituyen «noticias» (incorporación de nueva información no anticipable, errores no correlacionados con la estimación preliminar) o «ruido» (errores de medición independientes del dato final):
$$y_t^{\text{final}} - y_t^{\text{preliminar}} = \alpha + \beta y_t^{\text{preliminar}} + u_t$$

---

## Proveedores disponibles

| Proveedor | Países cubiertos | Periodicidad de ediciones | Significado de la fecha de la añada |
|---|---|---|---|
| `oecd_stes` *(por defecto)* | 42 | Mensual, desde 1999-02 | Mes de instantánea del archivo de la OCDE |
| `alfred` | 35 | Por publicación individual | Fecha de publicación de la fuente oficial |
| `bundesbank` | DE | 111 (+42 históricas a 1995) | **Fecha exacta de publicación en Alemania** |
| `ons` | UK | 746, retrocediendo a 1961 | Mes de publicación y fase del comunicado |
| `statcan` | CA | 55, desde 2012-11 | **Fecha real de publicación** (The Daily) |
| `ecb_rtd` | EA, JP, US | Historial desde 2001 | Marca temporal de difusión del BCE |

```python
from puremacro.fetch import vintage_catalog, available_providers
print(available_providers())
vintage_catalog("oecd_stes").head()
```
