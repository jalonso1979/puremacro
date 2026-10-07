"""Historical generator of the Spanish editions of Notebooks 63 and 65.

The ``*_es.py`` twins of notebooks 63, 64 and 65 are now maintained by hand,
and the markdown dictionaries below are historical snapshots that may be
stale against the current English notebooks (their cell indices can drift).
``generate_es_notebook`` therefore refuses to overwrite an existing
``*_es.py`` unless ``force=True`` (``--force`` on the command line).

Notebook 64 is no longer generated. Its dictionary restated retracted claims
(a fold at ``sigma_fold = 0.1238`` traversed by Keller PAC where Newton
diverges, and a converged Cyprus step with residual 4.4e-16) and was removed
on 2026-09-30; see ``reviews/2026-09-30-notebook-review/REVIEW.md``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

def parse_jupytext(content: str) -> tuple[str, list[tuple[str, str]]]:
    cell_pattern = re.compile(r"^# %%(\s+\[markdown\])?", re.MULTILINE)
    splits = cell_pattern.split(content)
    header = splits[0]
    cells: list[tuple[str, str]] = []
    i = 1
    while i < len(splits):
        tag = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        cell_type = "markdown" if tag and "[markdown]" in tag else "code"
        cells.append((cell_type, body.strip()))
        i += 2
    return header, cells


def assemble_jupytext(header: str, cells: list[tuple[str, str]]) -> str:
    parts = [header.strip() + "\n\n"]
    for c_type, body in cells:
        if c_type == "markdown":
            parts.append(f"# %% [markdown]\n{body}\n\n")
        else:
            parts.append(f"# %%\n{body}\n\n")
    return "".join(parts).rstrip() + "\n"


# =============================================================================
# NOTEBOOK 63
# =============================================================================

NB63_MD = {0: '# # Juegos arancelarios: bienestar hicksiano, represalias y comprobaciones numéricas\n'
    '#\n'
    '# **¿Cómo distinguimos un candidato de juego arancelario de una mejor respuesta verificada '
    'numéricamente?**\n'
    '#\n'
    '# Esta economía de dos países, balanceada a mano, es sintética. Usamos la misma función de '
    'gasto de consumo y la misma contabilidad auditada en cada comparación. Los países A y B son '
    'ilustrativos; no se predice ninguna política nacional real.',
 1: '# ## El método en matemáticas\n'
    '#\n'
    '# El objetivo es la variación equivalente de consumo $EV_i=e_i(P_0,U_i)-e_i(P_0,U_{i0})$ '
    'respecto a una única referencia fija sin aranceles. EV es cero en esa referencia, por lo que '
    'las ganancias porcentuales se dividen entre el gasto inicial en el consumo seleccionado '
    '$m_{i0}$. La ganancia por desviarse unilateralmente es $r_i=\\max_t '
    'EV_i(t,\\tau_{-i})-EV_i(\\tau)$. Un candidato numérico de Nash debe satisfacer tanto '
    '$\\max_i|BR_i(\\tau_{-i})-\\tau_i|\\leq\\epsilon_\\tau$ como $\\max_i '
    'r_i/m_{i0}\\leq\\epsilon_r$ en el intervalo declarado.',
 2: '# ## Intuición\n'
    '#\n'
    '# **Intuición.** Los pasos pequeños por amortiguación pueden ocultar grandes incentivos para '
    'desviarse. Un arancel óptimo en la frontera depende del límite impuesto. El dilema del '
    'prisionero debe comprobarse en la matriz de pagos con acciones fijas; las represalias no lo '
    'implican. Un equilibrio general fallido no proporciona un pago.',
 3: '# ## Código resuelto\n'
    '#\n'
    '# Construimos una tabla balanceada con un sector productor por país, dos canastas de consumo '
    '(categorías 0 y 2), inversión separada (1), impuestos domésticos y saldos externos no nulos. '
    'Las remuneraciones fijas del trabajo y el capital cierran las cuentas de producción.',
 5: '# ### Una búsqueda unilateral acotada\n'
    '#\n'
    '# Seleccionamos explícitamente `metric="hicksian_ev"`. El modelo usa abastecimiento '
    'intermedio CES con sigma 2, canastas finales Leontief y devolución fiscal de suma fija. El '
    'objetivo es bienestar de consumo condicional; se excluye la inversión. Todas las búsquedas '
    'comparten una referencia auditada. Los porcentajes dividen EV entre el gasto inicial en '
    'consumo.',
 7: '# ### Diagnósticos numéricos de mejores respuestas\n'
    '#\n'
    '# Cada intento de equilibrio se audita. La recuperación automática prueba Newton, el método '
    'híbrido y luego continuación de Keller, sin relajar la tolerancia; si todos fallan, se '
    'produce un error. La convergencia exige mejores respuestas simultáneas en el vector final y '
    'ganancias por desviación normalizadas por consumo. El resultado puede ser inconcluso; debe '
    'comunicarse sin afirmar un teorema de Nash.',
 10: '# ## Lectura de los resultados\n'
     '#\n'
     '# La curva monetaria de bienestar tiene una referencia igual a cero. Su representación '
     'porcentual usa consumo inicial fijo; no divide entre EV inicial. Cada celda de pagos usa las '
     'mismas dos acciones por jugador. La convergencia corresponde a la búsqueda numérica '
     'declarada: revisa el límite arancelario, las fronteras, la resolución y la ganancia final '
     'por desviarse. El ejercicio no demuestra aranceles óptimos únicos ni interiores.',
 11: '# ## Tu turno\n'
     '#\n'
     '# Cambia el límite y compara el candidato y sus diagnósticos. Una respuesta en el límite '
     'superior puede cambiar con la restricción institucional.',
 13: '# ## ¿Qué tan exhaustivo es esto?\n'
     '#\n'
     '# Es un ejercicio docente reproducible, no un pronóstico empírico de una guerra comercial. '
     'Un equilibrio CES escalar derivado por separado y una minimización primal de gasto validan '
     'una cuadrícula completa de 41 por 41 perfiles; consulta '
     '`reviews/2026-09-20-hicksian-policy/REPORT.md`. Las búsquedas locales pueden omitir máximos '
     'estrechos. EV hicksiana está disponible para las preferencias y la contabilidad declaradas; '
     'la descomposición causal TOT/Alloc/TariffRec y la certificación de teoremas siguen sin estar '
     'disponibles. Consulta `docs/trade_policy.md`.'}


# =============================================================================
# NOTEBOOK 65
# =============================================================================

NB65_MD = {0: '# # Propagación de costos en CGV, procedencia y límites de validación del bienestar\n'
    '#\n'
    '# **¿Cómo distinguir un contrafactual comercial resuelto de una afirmación de bienestar sin '
    'sustento?**\n'
    '#\n'
    '# Este tutorial utiliza tablas pequeñas generadas con estructuras de OECD ICIO, FIGARO y EXIOBASE. Son '
    'datos sintéticos para enseñanza, no observaciones de esas bases.',
 1: '# ## El método en matemáticas\n'
    '#\n'
    '# Con coeficientes de insumos fijos $A$, la propagación de costos satisface $dp=(I-A^T)^{-1}dv$, '
    'manteniendo fijos los precios de factores. El equilibrio general también modifica precios, cantidades, '
    'transferencias e impuestos. Un residuo pequeño solo verifica las ecuaciones especificadas.',
 2: '# ## Intuición\n'
    '#\n'
    '# **Intuición.** Un aumento de costos llega a los clientes mediante sus compras de insumos. Tablas '
    'sintéticas similares pueden generar cascadas similares por construcción; esto no es evidencia empírica '
    'entre bases. La variación equivalente (equivalent variation) requiere una función de gasto y utilidad '
    'inicial independientes.',
 3: '# ## Código resuelto\n'
    '#\n'
    '# Generamos los datos de forma explícita y mostramos su procedencia antes de reportar resultados.',
 5: '# ### Procedencia\n'
    '#\n'
    '# Los lectores de datos reales requieren un archivo por defecto. `fallback_to_synthetic=True` permite '
    'datos generados solo con autorización explícita. Las calibraciones conservan procedencia, conversiones '
    'y ajustes. La división entre trabajo y capital es un supuesto del modelo.',
 7: '# ### Propagación de costos con precios fijos\n'
    '#\n'
    '# La cota espectral superior verifica productividad de estas matrices no negativas. El residuo del '
    'sistema lineal verifica este cálculo, no un teorema de bienestar.',
 9: '# ### Un experimento arancelario resuelto\n'
    '#\n'
    '# Usamos una tabla didáctica separada, balanceada a mano, con dos países y categorías explícitas de '
    'consumo e inversión. Las tablas sintéticas anteriores ilustran redes y no son la fuente de este ejemplo '
    'de bienestar. Primero reportamos PIB nominal e ingresos arancelarios; después evaluamos el consumo.',
 11: '# ### Bienestar hicksiano del consumo\n'
     '#\n'
     '# La categoría 0 representa consumo; se excluye inversión. Con una canasta Leontief, la utilidad es su '
     'cantidad y la función de gasto es $e(P,U)=PU$. Por tanto, $EV=P_0(C_1-C_0)$, positiva para ganancias. '
     'La interfaz general también admite preferencias Cobb–Douglas entre categorías de consumo seleccionadas '
     'explícitamente. Las transferencias incluyen los aranceles una sola vez. La atribución a precios, '
     'ingreso factorial y transferencias compara estados finales; no es un teorema causal de términos de '
     'intercambio y eficiencia.',
 13: '# ### Presentación de resultados numéricos',
 15: '# ## Lectura de los resultados\n'
     '#\n'
     '# Las tablas identifican los datos como sintéticos. El residuo y las cotas espectrales son '
     'diagnósticos diferentes. EV y CV del consumo se obtienen de una función de gasto explícita y se '
     'contrastan con la valoración directa de cantidades. La nueva atribución utiliza precios de comprador, '
     'ingreso factorial y transferencias. La descomposición histórica TOT/Alloc y la certificación de '
     'teoremas siguen sin estar disponibles.',
 16: '# ## Tu turno\n'
     '#\n'
     '# Repite el experimento con otra semilla u otro proveedor. Compara CES y Leontief usando los mismos '
     'aranceles, cierre fiscal y tolerancia. Verifica los flujos reportados contra cada tecnología antes de '
     'interpretar diferencias.',
 17: '# ## ¿Qué tan exhaustivo es esto?\n'
     '#\n'
     '# Este ejemplo enseña procedencia, propagación de coeficientes fijos y una solución GE pequeña. No '
     'valida archivos nativos MRIO, comparaciones empíricas de bienestar, equilibrio de Nash o una '
     'descomposición causal de términos de intercambio y eficiencia. EV depende de las preferencias y del '
     'cierre fiscal declarados. Consulta `docs/STRUCTURAL_VALIDATION_STATUS.md` para los límites actuales.'}


def generate_es_notebook(stem: str, md_dict: dict[int, str], knob_cell_idx: int,
                         *, force: bool = False) -> None:
    en_path = Path(f"notebooks/{stem}.py")
    es_path = Path(f"notebooks/{stem}_es.py")
    if es_path.exists() and not force:
        raise FileExistsError(
            f"{es_path} exists and is maintained by hand; this generator's text may be "
            "stale. Pass force=True (--force) to overwrite it.")

    header, cells = parse_jupytext(en_path.read_text(encoding="utf-8"))
    es_cells = []

    for idx, (c_type, body) in enumerate(cells):
        if c_type == "markdown":
            if idx in md_dict:
                es_cells.append(("markdown", md_dict[idx]))
            else:
                es_cells.append(("markdown", body))
        else:
            if idx == knob_cell_idx:
                # Prepend Tu turno comment
                knob_body = "# Tu turno: personaliza parámetros e inspecciona el comportamiento del modelo\n" + body
                es_cells.append(("code", knob_body))
            else:
                es_cells.append(("code", body))

    output = assemble_jupytext(header, es_cells)
    es_path.write_text(output, encoding="utf-8")
    print(f"Generated {es_path.name} with {len(es_cells)} cells ({len([c for c in es_cells if c[0] == 'code'])} code cells)")


def main(argv: list[str] | None = None) -> None:
    force = "--force" in (sys.argv[1:] if argv is None else argv)
    print("Generating Spanish editions for Notebooks 63 and 65...")
    generate_es_notebook("63_trade_wars_and_nash_tariffs", NB63_MD, knob_cell_idx=12, force=force)
    generate_es_notebook("65_gvc_cascades_and_welfare_decomposition", NB65_MD, knob_cell_idx=13, force=force)
    print("Both Spanish editions generated.")


if __name__ == "__main__":
    main()
