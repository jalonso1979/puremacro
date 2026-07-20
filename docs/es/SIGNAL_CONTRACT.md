> 🇬🇧 [English](../SIGNAL_CONTRACT.md) · 🇪🇸 Español

# Contrato de señal

> Estado: **Slice 1 incluido en 0.65.0** (calidad basada únicamente en escasez).
> El Slice 2 (draws + propagación) y el Slice 3 (calibración) están planificados.

El contrato de señal es la estructura de datos por `RiskIndex` que permite
a un estimador LP / SVAR aguas abajo conocer la fiabilidad de una lectura
del índice y, en versiones posteriores, propagar esa fiabilidad a las bandas
de confianza de las FIR.

## Esquema (Slice 1)

`puremacro.narrative.types.RiskIndex` incorpora dos campos opcionales:

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict
    quality: SignalQualityReport | None = None   # 0.65.0+
    draws:   pd.DataFrame | None          = None  # 0.66.0+
```

Ambos campos toman el valor `None` por defecto — todo código anterior a 0.65.0 funciona de manera idéntica.

## Activación (Slice 1)

```python
from puremacro.narrative.indices import lui

ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()      # one-row DataFrame: mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```

El mismo argumento de palabra clave `with_quality=False` está disponible en todos los índices
canónicos de `puremacro.narrative.indices.__all__`: `epu`, `mpu`, `gpr`, `tone`,
`wui`, `lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`, `lwui_wage`,
`bbui`, `cboui`, `ep_ui`, `erpui`, `eurlex_ui`, `sotuui`, `bluesky_ui`.

## `SignalQualityReport` (campos del Slice 1 ya poblados)

| Campo                 | Incorporado en | Descripción                                                        |
|-----------------------|----------------|--------------------------------------------------------------------|
| `n_docs_per_period`   | 0.65.0         | documentos por trimestre que alimentaron el kernel                 |
| `avg_doc_length`      | 0.65.0         | promedio de tokens por documento por trimestre                     |
| `coverage_gaps`       | 0.65.0         | trimestres sin documentos dentro del rango de fechas               |
| `kernel_agreement`    | 0.66.0         | correlación media por pares entre draws del kernel (Slice 2)       |
| `multilingual_parity` | 0.66.0         | correlación entre subconjuntos por idioma (Slice 2)                |
| `doc_bootstrap_sd`    | 0.66.0         | desviación estándar por período entre draws de bootstrap (Slice 2) |
| `corpus_loo_max_swing`| 0.66.0         | máximo \|Δ\| entre draws de validación cruzada por corpus (Slice 2)|
| `benchmark_scores`    | 0.67.0         | Pearson/Spearman/RMSE por clave frente al índice canónico (S3)     |
| `event_panel`         | 0.67.0         | correlación de rangos + aciertos en el decil superior + AUC frente al panel de eventos |
| `survey_scores`       | 0.67.0         | Pearson/RMSE por clave frente a series de encuestas                |

## Validación

`RiskIndex.__post_init__` valida `draws` (cuando se especifica):
- `draws.index` debe coincidir con `series.index`.
- `draws.columns` debe ser un `pd.MultiIndex` de dos niveles denominado `['source', 'draw_id']`.
- Cada valor del nivel `source` debe pertenecer al conjunto `{'kernel', 'lexicon', 'doc', 'corpus'}`.
