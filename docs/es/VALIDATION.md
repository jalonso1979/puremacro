> 🇬🇧 [English](../VALIDATION.md) · 🇪🇸 Español

# Validación

> Disponible a partir de puremacro **0.92.0**.

`puremacro` reimplementa muchos estimadores en numpy/scipy puro para que corran
en el navegador. Para demostrar que esas reimplementaciones son correctas, el
paquete incluye una **galería de validación**: un registro declarativo de casos,
cada uno comparando un estimador de puremacro con una referencia *independiente*.
Ejecútala tú mismo:

```python
from puremacro.validation import run_all, scorecard

scorecard()                 # una fila por caso: puremacro vs referencia, pasa/falla, margen
assert all(r.passed for r in run_all())
```

`scorecard()` y `run_all()` son compatibles con pyodide — solo necesitan las
cuatro dependencias centrales — así que la galería corre sin cambios en el
playground del navegador (cuaderno `12_validation_gallery`, también en español
como `12_validation_gallery_es`).

## Cómo se valida un caso

Cada caso declara un **mecanismo** (cómo se obtiene su referencia) y un **nivel de
tolerancia**. El mecanismo es el núcleo del argumento de confianza: la referencia
debe ser genuinamente independiente del código de puremacro bajo prueba.

| Mecanismo | Referencia | ¿Corre en vivo en el navegador? |
|---|---|---|
| `package` | statsmodels / linearmodels / arch, capturado una vez como valor *golden* congelado | sí — compara con el golden (sin dependencia pesada en tiempo de ejecución) |
| `scipy` | `scipy` / `numpy` (dependencia de ejecución), recalculado en vivo | sí |
| `analytical` | una solución en forma cerrada | sí |
| `published` | un número de un artículo, con cita | sí (constante congelada) |
| `internal` | una identidad entre métodos o una prueba de simular-y-recuperar | sí |

El mecanismo `package` mantiene puro el código distribuido: statsmodels/
linearmodels/arch **nunca se importan en puremacro**. Sus salidas se congelan como
JSON golden, y un guardián de deriva en integración continua (`pytest -m reference`)
recalcula el paquete en vivo y verifica que el golden siga siendo fiel. Así,
`run_all()` lee los goldens para los casos `package` y recalcula todo lo demás en
vivo, sin necesitar nunca los paquetes pesados.

Niveles de tolerancia: `EXACT` (rtol 1e-10) · `TIGHT` (1e-6) · `NUMERIC` (1e-2) ·
`COARSE` (10%) · `QUALITATIVE` (signo / ordenamiento / un umbral de cota inferior).

## Cobertura

**73 casos en 13 subsistemas — todos pasan.** Por mecanismo: internal 34,
analytical 21, package 11, scipy 6, published 1.

| Subsistema | Casos | Referencia(s) |
|---|---|---|
| `var` | 3 | FIR de Cholesky vs `orth_irfs` de statsmodels; FEVD-suma-1 y estabilidad ⇔ radio espectral del companion < 1 (identidades) |
| `lp` | 5 | Coeficientes/EE-HAC de la PL de Jordà vs OLS-HAC de statsmodels; PL-IV vs `IV2SLS` de linearmodels; EF a dos vías vs `PanelOLS`; identidad IV-se-reduce-a-OLS |
| `garch` | 7 | Parámetros/volatilidades GARCH(1,1) vs `arch`; simular-y-recuperar; descomposición de varianza de GARCH-MIDAS |
| `inference` | 8 | EE de Newey–West / OLS-HAC vs HAC de statsmodels; tabla de valores críticos de Stock–Yogo (publicada); valor crítico plug-in sup-t vs su forma cerrada i.i.d.; monotonía de sup-t sobre valores críticos puntuales |
| `state_space` | 6 | Estados del filtro/suavizador de Kalman + log-verosimilitud vs el espacio de estados de statsmodels; identidades de varianza del suavizador |
| `dynpanel` | 7 | El GMM de Arellano–Bond / Blundell–Bond recupera un ρ conocido; identificación exacta J = 0; inflación de varianza de muestras finitas de Windmeijer |
| `did` | 4 | Callaway–Sant'Anna recupera DiD 2x2 analíticamente; Sun–Abraham iguala Callaway–Sant'Anna en 2x2; imputación de Borusyak–Jaravel–Spiess recupera 2x2; Diferencia en Diferencias Sintética recupera efecto de tratamiento |
| `unit_root` | 3 | Desentendencia GLS de ERS recupera tendencia y constante analíticamente; identidad de prueba M de Ng–Perron MZt = MZa · MSB |
| `spectral` | 6 | DEP de Welch / espectro cruzado / coherencia vs `scipy.signal`; partición de la unidad de la potencia por banda; coherencia ∈ [0,1] |
| `forecast` | 6 | Forma cerrada del CRPS gaussiano (Gneiting–Raftery); convergencia del estimador insesgado de conjunto; calibración PIT; signo/empate de Diebold–Mariano; retención MCS |
| `vfi` | 5 | Tauchen/Rouwenhorst reproducen los momentos del AR(1); política en forma cerrada de Brock–Mirman; estacionaria de Markov vs autovector izquierdo de scipy; MGE = VFI |
| `dsge` | 6 | Klein = gensys en un modelo conocido; solución en forma cerrada con expectativas a futuro; log-verosimilitud de Kalman vs la verosimilitud analítica del AR(1) |
| `narrative` | 7 | Puntuación léxica de valor conocido sobre texto construido; identidades de monotonía / estandarización del índice |

Cada caso lleva su cita completa en el código (`ValidationCase.citation`),
mostrada en la columna `citation` de `scorecard()`. Las referencias clave incluyen
Lütkepohl (2005), Newey & West (1987), Stock & Yogo (2005), Gneiting & Raftery
(2007), Diebold & Mariano (1995), Brock & Mirman (1972), Rouwenhorst (1995),
Tauchen (1986), Engle (2002), Arellano & Bond (1991) y Blundell & Bond (1998).

## X-11/ARIMA nativo frente al binario real de X-13ARIMA-SEATS

`puremacro.sa.x11` (nativo, Python puro, seguro para Pyodide) se valida
contra el binario genuino X-13ARIMA-SEATS del Censo (v1.1.57) mediante
goldens congelados: `tools/gen_validation_goldens_sa.py` ejecuta el
binario con la especificación fijada a la configuración nativa v1
(log/nivel por AICC, modelo airline, sin regresores de outliers,
extensión ARIMA de 60 periodos, X-11 con filtro estacional elegido por
MSR) sobre nueve series reales — seis niveles de desempleo LAUS de
condados desde grandes hasta diminutos (Keweenaw MI, ~2k habitantes),
producción industrial sin desestacionalizar y dos agregados
trimestrales — y `tests/test_sa_x11_native.py` exige acuerdo dentro de
tolerancias **medidas y luego congeladas con margen** (re-medidas el
2026-07-22 tras aterrizar la cascada genuina de pesos B17/B20 — pesos
de extremos del irregular de cada etapa formando originales
modificados que alimentan las tendencias de la etapa siguiente, la
estructura que la cadena v1 había condensado): medianas interiores de
0.08–0.83%, máximos interiores por debajo de 0.7% en series macro
suaves y hasta ~7% en los meses de reemplazo de extremos de series de
condados pequeños y salvajes (aproximadamente la MITAD de las brechas
v1), con la selección del filtro estacional coincidiendo con el
binario en todas las series mensuales. Las fuentes de divergencia
restantes quedan documentadas en lugar de ocultas: el binario aplica
filtros asimétricos al inicio de la serie (no usa backcasts dentro de
X-11) donde el motor nativo usa filtros simétricos sobre backcasts, y
detalles residuales del valor de reemplazo en los puntos marcados. El CI nunca ejecuta
el binario — solo los goldens congelados. Eso es lo que promete el
nombre "estilo X-11/ARIMA, validado contra X-13": exactamente lo que
midieron las tolerancias congeladas, ni más ni menos.

## Alcance honesto

Donde no existe una referencia *independiente* sólida, el caso se **omite con una
razón explícita** en lugar de avalarse con una comprobación circular. Ejemplos
documentados: `inference.kleibergen_paap_f` (la implementación devuelve un
estadístico no estándar sin forma cerrada equivalente), las rutinas pesadas de
estimación de `dsge`, un contraste externo de coeficientes de Arellano–Bond (sin
conjunto de datos sin conexión / paquete de GMM en Python) y las rutas de LLM y
descarga en vivo de narrative. La galería valida lo que se puede validar de forma
independiente, y lo dice cuando no puede.

## Reverificar

```bash
# rápido y puro: puremacro vs las referencias contratadas
python -m pytest tests/validation/ -q

# guardián de deriva en CI: recalcula las referencias EN VIVO de
# statsmodels/linearmodels/arch y verifica que los goldens sigan siendo fieles
# (requiere el extra dev)
python -m pytest -m reference -q
```

Para ampliar la galería, deja un módulo `cases_<subsistema>.py` que exponga
`CASES: list[ValidationCase]` en `puremacro/validation/`; `run_all()` lo descubre
automáticamente.
