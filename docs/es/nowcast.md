> 🇬🇧 [English](../nowcast.md) · 🇪🇸 Español

# Nowcasting del PIB

El PIB es una magnitud trimestral que se publica con retraso considerable — entre cuatro y ocho semanas tras el cierre del trimestre de referencia, según el instituto de estadística. Hasta que dicha cifra se difunde, toda la información disponible sobre el trimestre en curso es de periodicidad mensual: producción industrial, empleo, ventas minoristas y encuestas de opinión empresarial, cada una con su propio calendario de publicación. El *nowcasting* responde a la pregunta: «¿cuál es la tasa de crecimiento del trimestre actual *a fecha de hoy*?». Los dos obstáculos principales para lograrlo son estructurales y no meramente estadísticos.

**Frecuencias mixtas.** La variable objetivo se observa cuatro veces al año; los predictores, doce. Cualquier método debe definir cómo se agrega la serie mensual en términos trimestrales — mediante promedios, sumas o una trayectoria mensual latente restringida a coincidir con el total trimestral.

**Bordes irregulares (*ragged edge*).** En cualquier día hábil, el panel de datos es un rectángulo con el extremo inferior deshilachado. Las encuestas de confianza del mes $m$ se publican a los pocos días de su finalización; los índices de producción industrial tardan semanas; otras series arrastran dos meses de desfase. Eliminar filas incompletas (*listwise deletion*) descartaría precisamente los períodos más recientes objeto del pronóstico.

`puremacro.nowcast` proporciona tres estimadores que resuelven estos desafíos:

| Función | Modelo | Entradas requeridas | Tratamiento del borde irregular |
|---|---|---|---|
| `nowcast_gdp` | Factores EM-PCA + regresión puente trimestral | Panel mensual **y** serie trimestral de PIB | Imputación iterativa por PCA |
| `kalman_dfm` | DFM en dos etapas (Doz-Giannone-Reichlin 2011) | Un panel en una sola frecuencia | Suavizador exacto de Kalman |
| `mf_var` | VAR de frecuencias mixtas (Mariano-Murasawa 2003) | Panel mensual con el dato trimestral al final de cada trimestre | Suavizador exacto de Kalman |

Todo el código se ejecuta sin dependencias de red, utilizando directamente los datos provistos en memoria.

---

## 1. `nowcast_gdp`: Factores dinámicos y ecuación puente

Es el método estándar de trabajo y el que genera cifras publicables directamente. Opera en cuatro fases integradas:
1. Estandariza el panel mensual, interpola inicialmente los datos faltantes, extrae factores mediante componentes principales (PCA) y re-imputa los valores omitidos hasta la convergencia del algoritmo EM (`max_em_iter=50`, `em_tol=1e-4`).
2. Promedia los factores mensuales dentro de cada trimestre natural.
3. Estima una regresión puente por MCO del PIB trimestral histórico sobre los factores trimestralizados.
4. Aplica los coeficientes estimados al promedio de factores del trimestre en curso.

```python
from puremacro.nowcast import nowcast_gdp

res = nowcast_gdp(monthly_df, gdp_series, n_factors=2)
print("Nowcast puntual:", res.nowcast)
print("Trimestre objetivo:", res.target_quarter)
print("R² de la ecuación puente:", res.model_r2)
print(res.summary())
```

---

## 2. Modelo de factores dinámicos con filtro de Kalman (`kalman_dfm`)

Implementa la metodología en dos etapas de Doz, Giannone y Reichlin (2011). Permite extraer el factor común subyacente y manejar formalmente la estructura de estado-espacio con datos faltantes mediante el suavizador de Kalman:

```python
from puremacro.nowcast import kalman_dfm

dfm_res = kalman_dfm(panel_mensual, n_factors=2, p=1)
print(dfm_res.summary())
```

---

## 3. Descomposición de noticias (*News Decomposition*)

Permite descomponer cuantitativamente la revisión del *nowcast* entre dos publicaciones sucesivas, asignando a cada nuevo dato publicado su contribución específica a la variación del pronóstico del PIB:

```python
print(res.news_decomposition)
```
