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
| `banxico` | MX | Una por día de captura | Fecha de la instantánea local |
| `inegi` | MX | Una por día de captura | Fecha de la instantánea local |
| `bcb` | BR | Una por día de captura | Fecha de la instantánea local |
| `bcch` | CL | Una por día de captura | Fecha de la instantánea local |

```python
from puremacro.fetch import vintage_catalog, available_providers
print(available_providers())
vintage_catalog("oecd_stes").head()
```

Además de los agregados de cuentas nacionales, el catálogo incorpora dos variables de frecuencia mensual o diaria sobre las que se construyen los conectores latinoamericanos: `policy_rate` (alias `tpm`, `selic`, `tasa_objetivo`, `interest_rate`, `target_rate`, ...) y `activity`, el índice mensual de actividad económica (alias `igae`, `imacec`, `ibc_br`). La función `canonical_variable` traduce cualquier grafía admitida al nombre canónico.

### Los cuatro conectores latinoamericanos

`banxico`, `inegi`, `bcb` y `bcch` son conectores de **instantánea** (*snapshot*), no archivos históricos. Cada servicio publica únicamente la edición vigente en el momento de la consulta, de modo que aquí la fecha de la añada es el día en que *esta máquina* capturó la serie y la escribió en la tabla `realtime_vintages` de la base de datos de caché. El historial de revisiones se acumula, por tanto, solo a lo largo de capturas repetidas en días distintos: una descarga produce una única añada, y un panel ensamblado en una sola tarde no contiene revisión alguna que contrastar.

| Proveedor | País | Series | Credencial |
|---|---|---|---|
| `banxico` | MX | `policy_rate` SF61745, `cpi` SP1, `activity` SR17631 | token, enviado en la cabecera `Bmx-Token` |
| `inegi` | MX | `gdp_real` 735848, `cpi` 628197, `activity` 736184 | token, incrustado en la ruta de la URL |
| `bcb` | BR | `gdp_real` 22099, `cpi` 433, `policy_rate` 432, `activity` 24363 | ninguna (endpoint abierto) |
| `bcch` | CL | `gdp_real` F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T, `cpi` F074.IPC.IND.Z.Z.C.M, `policy_rate` F022.TPM.TPO.D001.NO.Z.D, `activity` F032.IMC.IND.Z.Z.EP18.Z.Z.0.M | usuario **y** contraseña, enviados en la cadena de consulta |

Las entradas del catálogo cuyo identificador no pudo verificarse contra el servicio en vivo llevan la marca `VERIFY ONLINE` en su columna `note`, de manera que un identificador conjeturado resulte visible en lugar de pasar por definitivo:

```python
from puremacro.fetch import vintage_catalog
vintage_catalog("inegi")[["variable", "series_id", "freq", "note"]]
```

### Las fechas de añada no son intercambiables

Los proveedores `banxico`, `inegi`, `bcb` y `bcch` registran una **fecha de instantánea**, no una fecha de publicación: el día en que esta máquina descargó la serie. Ordenar ediciones sigue siendo válido, pero la primera edición de un período de referencia es la primera que *usted capturó*, no la primera estimación del instituto, y ninguna edición es anterior a su primera descarga. Para las ediciones *históricas* de México, Brasil y Chile, el archivo de la OCDE (`oecd_stes`) continúa siendo la fuente; así, `providers_for("MEX", "gdp_real")` devuelve `['alfred', 'inegi', 'oecd_stes']`.

Las credenciales se documentan en [`docs/es/CREDENTIALS.md`](CREDENTIALS.md), la tabla de instantáneas en [`docs/es/CACHE_DB.md`](CACHE_DB.md), y el flujo completo — captura, cartucho y contraste de revisiones — en [`docs/es/real_time_latam.md`](real_time_latam.md).
