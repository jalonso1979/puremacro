"""Los cuatro bloques de contabilidad nacional trimestral que congela T01_B §6.

Un solo import para todos los cuadernos posteriores a T01_B: los datos ya están
en disco —49 países, trimestrales, desestacionalizados, con su ficha de
procedencia al lado—, así que **ningún cuaderno de aquí en adelante toca la
red**::

    from _datos import gasto, pais, crecimiento, DATOS, ruta_dato, ruta_modelo

    g   = gasto()                    # (code, date) x 30 series + deflactores
    esp = pais("ESP")                # los cuatro bloques de un país, un marco
    dy  = crecimiento("USA")         # crecimiento trimestral del PIB, %

Los cuatro bloques, tal como los escribe T01_B §6:

===================  ==========================================================
``gasto()``          C, G, I, X, M, PIB, FBCF por activo, consumo por
                     durabilidad — **más un deflactor por cada uno**. 49 países.
``renta()``          remuneración de asalariados, excedente bruto y renta
                     mixta, impuestos netos. Sólo a precios corrientes: no
                     existen "salarios a precios constantes". 40 países.
``produccion()``     valor añadido por rama CIIU Rev.4, impuestos netos sobre
                     productos, discrepancia de encadenamiento, con sus
                     deflactores. 42 países.
``mercado_laboral()`` ocupados y horas (asalariados y cuenta propia), horas por
                     semana, proporción de cuenta propia, las tasas de paro y
                     de vacantes, y —desde T01_B §5.3— las mismas cabezas y
                     horas otra vez para agricultura (``_agri``) y para
                     administración pública, educación y sanidad (``_public``).
                     38 países; u y v para 30.
===================  ==========================================================

Y un quinto, **anual**, que existe por una sola razón: Estados Unidos y Japón no
publican nada en el flujo trimestral de trabajo, y sí en el anual::

    ``actividad_anual()``  valor añadido y trabajo por rama, anual, a precios
                           corrientes y del año anterior. 46 países.
    ``sector_mercado()``   lo anterior menos agricultura y menos O–Q, ya
                           encadenado: el sector de mercado de T01_A §1.4.

**No se guardan volúmenes**, porque el volumen es ``100 × nominal / deflactor``
exacto hasta el último bit — que es toda la tesis de T01_B §1. :func:`volumen`
lo rehace, y :func:`crecimiento` lo usa.

**Cada país en su propio año de referencia de precios.** Los niveles de los
deflactores no son comparables entre países hasta que se rebasan;
``meta()["price_ref_year"]`` dice cuál es el de cada uno y
``puremacro.fetch.qna_rebase`` los unifica en una línea. Las *participaciones*
nominales (``cons_hh / gdp``) y las *tasas de crecimiento* no necesitan nada de
eso: son inmunes al año de referencia y al tipo de cambio.

Si los archivos no están, cada función lanza un ``FileNotFoundError`` que dice
qué hacer. No hay descarga de respaldo aquí a propósito: quien descarga es
T01_B, y hacerlo dos veces es cómo se acaba con dos versiones de los mismos
números.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATOS",
    "_safe_probe",
    "ruta_dato",
    "ruta_modelo",
    "gasto",
    "renta",
    "produccion",
    "mercado_laboral",
    "meta",
    "actividad_anual",
    "sector_mercado",
    "RAMAS_FUERA",
    "bloques",
    "paises",
    "pais",
    "volumen",
    "crecimiento",
    "ciclo",
]

#: Los cuatro archivos, y el mínimo de columnas que cada uno tiene que traer
#: para que valga la pena haberlo leído.
_BLOQUES: dict[str, tuple[str, ...]] = {
    "gasto":           ("gdp", "cons_hh", "inv", "exports", "imports", "gdp_defl"),
    "renta":           ("gdp_income", "comp_emp"),
    "produccion":      ("gdp_output", "va_total"),
    "mercado_laboral": ("emp", "hours", "hours_week", "self_emp_share",
                        "hours_factor"),
}


def _safe_probe(p: Path | str, is_dir: bool | str | None = None) -> bool:
    """Safe path existence check catching OSError/PermissionError in sandboxes.

    On iOS sandboxed environments (Juno), probing filesystem paths outside the sandbox
    container raises PermissionError (EACCES) instead of returning False. Standard
    pathlib methods (exists, is_dir, is_file) do not suppress PermissionError in Python <= 3.12.
    This function guarantees a safe boolean return under all OS conditions.

    Parameters
    ----------
    p : Path | str
        Filesystem path to probe.
    is_dir : bool | str | None, default=None
        If True or 'dir', checks if p is an existing directory.
        If False or 'file', checks if p is an existing regular file.
        If None (default), checks if p exists as either a file or directory.

    Returns
    -------
    bool
        True if target exists and matches kind; False otherwise or if permission denied.
    """
    try:
        path = p if hasattr(p, "is_dir") else Path(p)
        if is_dir is True or is_dir == "dir":
            return bool(path.is_dir())
        if is_dir is False or is_dir == "file":
            return bool(path.is_file())
        return bool(path.is_file() or path.is_dir())
    except (OSError, TypeError, ValueError):
        return False


def _legible(p: Path | str, kind: str = "dir") -> bool:
    """Legacy alias for _safe_probe(p, is_dir=kind)."""
    return _safe_probe(p, is_dir=kind)


def _resolver() -> Path:
    """Resolve the course dataset directory, prioritizing in-sandbox directories and bundle_2026A.

    On iOS sandboxes (Juno), upward traversal (e.g. `Path.cwd().parent` or `Path("..")`)
    raises `PermissionError` when crossing the sandbox container boundary. This resolver
    inspects local directory candidates first, checking both `data_curso/bundle_2026A`
    and `data_curso`, before safely attempting upward exploration within guarded try-except
    blocks.
    """
    try:
        aqui = Path(__file__).resolve().parent
    except OSError:
        aqui = Path(__file__).parent

    try:
        cwd = Path.cwd()
    except OSError:
        cwd = aqui

    candidates: list[Path] = [
        # 1. Look within module directory (notebooks/ or colab_es/)
        aqui / "data_curso" / "bundle_2026A",
        aqui / "data_curso",
        aqui / "bundle_2026A",
        aqui / "data",
        # 2. Look within current working directory (e.g. repo root or custom runner)
        cwd / "notebooks" / "data_curso" / "bundle_2026A",
        cwd / "data_curso" / "bundle_2026A",
        cwd / "colab_es" / "data_curso" / "bundle_2026A",
        cwd / "notebooks" / "data_curso",
        cwd / "data_curso",
        cwd / "colab_es" / "data_curso",
        cwd / "bundle_2026A",
        cwd / "data",
    ]

    # 3. Guarded upward traversal from aqui (only if OS sandbox permits)
    try:
        if aqui.parent != aqui:
            candidates.extend([
                aqui.parent / "notebooks" / "data_curso" / "bundle_2026A",
                aqui.parent / "data_curso" / "bundle_2026A",
                aqui.parent / "colab_es" / "data_curso" / "bundle_2026A",
                aqui.parent / "notebooks" / "data_curso",
                aqui.parent / "data_curso",
                aqui.parent / "colab_es" / "data_curso",
                aqui.parent / "bundle_2026A",
            ])
    except OSError:
        pass

    # 4. Guarded upward traversal from cwd (only if OS sandbox permits)
    try:
        if cwd.parent != cwd:
            candidates.extend([
                cwd.parent / "notebooks" / "data_curso" / "bundle_2026A",
                cwd.parent / "data_curso" / "bundle_2026A",
                cwd.parent / "colab_es" / "data_curso" / "bundle_2026A",
                cwd.parent / "notebooks" / "data_curso",
                cwd.parent / "data_curso",
                cwd.parent / "colab_es" / "data_curso",
                cwd.parent / "bundle_2026A",
            ])
    except OSError:
        pass

    for cand in candidates:
        if _safe_probe(cand, is_dir=True):
            try:
                return cand.resolve()
            except OSError:
                return cand

    # Ultimate fallback: return local bundle candidate
    fallback = aqui / "data_curso" / "bundle_2026A"
    try:
        return fallback.resolve()
    except OSError:
        return fallback


#: Carpeta de datos del curso, resuelta una vez al importar.
DATOS: Path = _resolver()


def ruta_dato(filename: str | Path) -> Path:
    """Locate a dataset file within DATOS or fallback bundle directories.

    Guarantees portable resolution across desktop, cloud (Google Colab), and
    sandboxed mobile environments (Juno iOS). Never raises uncaught OS or
    permission exceptions; returns the resolved Path if found, or canonical
    fallback `DATOS / filename` for downstream handling.

    Parameters
    ----------
    filename : str | Path
        Filename or relative path of the dataset (e.g. "GDPC1.csv",
        "data_curso/gali1999.csv", "enoe_transitions_quarterly_observed.parquet").

    Returns
    -------
    Path
        Resolved path to the dataset file.
    """
    p_in = Path(filename)
    nombre = p_in.name

    candidates: list[Path] = [
        DATOS / p_in,
        DATOS / nombre,
    ]

    # If DATOS is a parent directory (e.g. data_curso/), check bundle_2026A inside it
    if DATOS.name != "bundle_2026A":
        candidates.append(DATOS / "bundle_2026A" / nombre)

    # If DATOS is bundle_2026A, check parent directory (e.g. data_curso/)
    try:
        if DATOS.parent != DATOS:
            candidates.append(DATOS.parent / nombre)
            candidates.append(DATOS.parent / "bundle_2026A" / nombre)
    except OSError:
        pass

    # Safe probe relative to _datos.py location
    try:
        aqui = Path(__file__).resolve().parent
    except OSError:
        aqui = Path(__file__).parent

    candidates.extend([
        aqui / p_in,
        aqui / "data_curso" / "bundle_2026A" / nombre,
        aqui / "data_curso" / nombre,
        aqui / "bundle_2026A" / nombre,
        aqui / "data" / nombre,
    ])

    # Safe probe relative to current working directory
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = aqui

    candidates.extend([
        cwd / p_in,
        cwd / "notebooks" / "data_curso" / "bundle_2026A" / nombre,
        cwd / "data_curso" / "bundle_2026A" / nombre,
        cwd / "colab_es" / "data_curso" / "bundle_2026A" / nombre,
        cwd / "notebooks" / "data_curso" / nombre,
        cwd / "data_curso" / nombre,
        cwd / "colab_es" / "data_curso" / nombre,
        cwd / "bundle_2026A" / nombre,
        cwd / nombre,
    ])

    for cand in candidates:
        if _safe_probe(cand, is_dir=False):
            try:
                return cand.resolve()
            except OSError:
                return cand

    # Default fallback: return primary target under DATOS preserving subdirectories
    fallback = DATOS / p_in
    try:
        return fallback.resolve()
    except OSError:
        return fallback


def ruta_modelo(filename: str | Path) -> Path:
    """Locate a Dynare `.mod` model file in course model directories.

    Searches `modelos/` folders within the local sandbox boundary before
    safely attempting upward traversal, preventing iOS PermissionError.

    Parameters
    ----------
    filename : str | Path
        Name or relative path of the model file (e.g. "rbc_canonico.mod",
        "modelos/nk_zlb_ref.mod", "merz_andolfatto").

    Returns
    -------
    Path
        Resolved path to the model `.mod` file.
    """
    p_in = Path(filename)
    nombre = p_in.name
    nombre_mod = nombre if nombre.endswith(".mod") else f"{nombre}.mod"

    try:
        aqui = Path(__file__).resolve().parent
    except OSError:
        aqui = Path(__file__).parent

    try:
        cwd = Path.cwd()
    except OSError:
        cwd = aqui

    candidates: list[Path] = [
        # 1. In-sandbox modelos/ alongside notebooks
        aqui / "modelos" / nombre_mod,
        aqui / "modelos" / nombre,
        # 2. Current working directory modelos/
        cwd / "notebooks" / "modelos" / nombre_mod,
        cwd / "modelos" / nombre_mod,
        cwd / "notebooks" / "modelos" / nombre,
        cwd / "modelos" / nombre,
        # 3. Directly alongside notebooks or cwd
        aqui / nombre_mod,
        cwd / nombre_mod,
    ]

    # 4. Guarded upward traversal (only if OS sandbox permits)
    try:
        if aqui.parent != aqui:
            candidates.extend([
                aqui.parent / "modelos" / nombre_mod,
                aqui.parent / "notebooks" / "modelos" / nombre_mod,
                aqui.parent / "modelos" / nombre,
            ])
    except OSError:
        pass

    try:
        if cwd.parent != cwd:
            candidates.extend([
                cwd.parent / "modelos" / nombre_mod,
                cwd.parent / "notebooks" / "modelos" / nombre_mod,
                cwd.parent / "modelos" / nombre,
            ])
    except OSError:
        pass

    for cand in candidates:
        if _safe_probe(cand, is_dir=False):
            try:
                return cand.resolve()
            except OSError:
                return cand

    # Default fallback: return primary path under notebooks/modelos/
    fallback = aqui / "modelos" / nombre_mod
    try:
        return fallback.resolve()
    except OSError:
        return fallback


def _leer(nombre: str) -> pd.DataFrame:
    ruta = ruta_dato(f"qna_{nombre}.csv")
    if not _safe_probe(ruta, is_dir=False):
        raise FileNotFoundError(
            f"No encuentro {ruta}.\n"
            f"Los cuatro bloques los escribe la sección 6 de T01_B. O bien\n"
            f"  (a) corre T01_B una vez —tarda unos minutos y descarga—, o\n"
            f"  (b) copia la carpeta data_curso/ o bundle_2026A/ del curso junto a este cuaderno\n"
            f"      (en un iPad tiene que estar dentro de la carpeta del propio\n"
            f"      cuaderno: todo lo que esté por encima queda fuera del sandbox).\n"
            f"Busqué en {DATOS}.")
    df = (pd.read_csv(ruta, parse_dates=["date"])
            .set_index(["code", "date"]).sort_index())
    faltan = [c for c in _BLOQUES[nombre] if c not in df.columns]
    if faltan:
        raise ValueError(
            f"{ruta.name} existe pero le faltan columnas ({', '.join(faltan)}). "
            f"Probablemente lo escribió una versión anterior de T01_B; vuelve a "
            f"correr su sección 6.")
    return df


@lru_cache(maxsize=None)
def gasto() -> pd.DataFrame:
    """Enfoque del gasto: niveles a precios corrientes y deflactores implícitos."""
    return _leer("gasto")


@lru_cache(maxsize=None)
def renta() -> pd.DataFrame:
    """Enfoque de la renta. Sin deflactores: estas series sólo existen nominales."""
    return _leer("renta")


@lru_cache(maxsize=None)
def produccion() -> pd.DataFrame:
    """Enfoque de la producción: valor añadido por rama, con sus deflactores.

    Ojo con las dos columnas informativas: ``va_mfg`` está *dentro* de ``va_ind``
    y ``va_services`` es la suma de las siete de servicios que ya figuran
    aparte. Sumar todas las ``va_*`` cuenta un tercio de la economía dos veces.
    """
    return _leer("produccion")


#: Horas semanales por ocupado que se consideran posibles. Las horas del
#: archivo ya vienen reparadas; esto es la red de seguridad, no el arreglo.
_BANDA_HORAS = (25.0, 50.0)


@lru_cache(maxsize=None)
def mercado_laboral() -> pd.DataFrame:
    """Insumo de trabajo y mercado de trabajo: cabezas, horas, paro y vacantes.

    Columnas derivadas: ``hours_week`` (horas por ocupado y semana),
    ``self_emp_share`` (% del empleo por cuenta propia) y ``theta`` (v/u).

    **Las horas vienen ya sobre la base temporal correcta.** La OCDE declara
    ``hours`` como millones de horas trabajadas *en el trimestre*, y Chile y
    Costa Rica no lo cumplen: presentan una cifra semanal y una anual
    respectivamente, lo que las deja en 3,2 y 166 horas por semana. T01_B §5.1
    lo arregla con ``puremacro.fetch.qna_repair_hours`` antes de congelar el
    bloque —el factor lo identifica exigir una jornada plausible, no lo elige
    nadie— y ``hours_factor`` viaja en el archivo diciendo qué se aplicó a cada
    país (1,0 para los 31 que ya estaban bien).

    Esto importa aguas abajo: un Chile sin reparar mete un producto por hora
    trece veces demasiado alto en cualquier comparación de *niveles*. Las tasas
    de crecimiento son inmunes —un factor constante se cancela en
    :math:`\\Delta\\log h`—, así que el fallo es silencioso, y por eso hay una
    guardia aquí y no sólo una nota.
    """
    df = _leer("mercado_laboral")
    d = df[["hours", "emp"]].dropna()
    if len(d):
        semana = ((d["hours"] * 1e6) / (d["emp"] * 1e3) / 13.0
                  ).groupby(level="code").median()
        malos = semana[(semana < _BANDA_HORAS[0]) | (semana > _BANDA_HORAS[1])]
        if len(malos):
            raise ValueError(
                f"qna_mercado_laboral.csv trae horas sobre una base temporal "
                f"imposible en {', '.join(f'{c} ({v:.1f} h/semana)' for c, v in malos.items())}. "
                f"Lo espera dentro de {_BANDA_HORAS[0]:.0f}-{_BANDA_HORAS[1]:.0f} h. "
                f"El archivo lo escribió una versión de T01_B anterior al arreglo "
                f"de la base temporal (§5.1, qna_repair_hours): vuelve a correr "
                f"su sección 6, o copia un data_curso/ actualizado.")
    return df


@lru_cache(maxsize=None)
def meta() -> pd.DataFrame:
    """Ficha por país: moneda, base de precios, año de referencia, ajuste estacional."""
    ruta = ruta_dato("qna_meta.csv")
    if not _safe_probe(ruta, is_dir=False):
        raise FileNotFoundError(f"No encuentro {ruta}. Corre la sección 6 de T01_B.")
    return pd.read_csv(ruta)


#: Las dos ramas que se le quitan a la economía entera para dejar el **sector de
#: mercado**. Agricultura, porque casi todo su trabajo es por cuenta propia y
#: buena parte de su producto anual es el clima; y administración pública,
#: educación y sanidad, porque el SCN *define* su valor añadido como
#: remuneración más consumo de capital fijo, de modo que su crecimiento medido
#: de la productividad es cercano a cero por construcción y no por hallazgo.
RAMAS_FUERA: tuple[str, ...] = ("agri", "public")


@lru_cache(maxsize=None)
def actividad_anual() -> pd.DataFrame:
    """Valor añadido y trabajo por rama, **anual**, indexado por ``(code, year)``.

    El quinto bloque, y el único anual. Existe porque el flujo trimestral de
    trabajo devuelve cero filas para Estados Unidos y para Japón —no una rama
    ausente ni una muestra corta: nada— mientras que las tablas anuales de la
    OCDE sí los traen, con las mismas doce agrupaciones CIIU. Es la única
    frecuencia en la que las dos economías sobre las que está escrita la
    literatura del ciclo se pueden medir con el mismo concepto que las demás.

    Dos columnas por rama y precio: ``va_X`` a precios corrientes y ``va_X_py``
    a los del año anterior. **No hay volúmenes**, por la misma razón que en los
    bloques trimestrales y una más: los volúmenes encadenados no son aditivos,
    así que un agregado construido sumándolos está mal sin avisar. Los precios
    del año anterior sí son aditivos dentro de un año, y :func:`sector_mercado`
    los encadena.
    """
    ruta = ruta_dato("oecd_ana_actividad.csv")
    if not _safe_probe(ruta, is_dir=False):
        raise FileNotFoundError(
            f"No encuentro {ruta}.\n"
            f"Lo congela la sección 5.3 de T01_B. O bien corre T01_B una vez, o\n"
            f"copia la carpeta data_curso/ o bundle_2026A/ del curso junto a este cuaderno.")
    df = (pd.read_csv(ruta).set_index(["code", "year"]).sort_index())
    faltan = [c for c in ("va_total", "va_total_py", "va_agri", "va_public",
                          "hours_employees") if c not in df.columns]
    if faltan:
        raise ValueError(
            f"{ruta.name} existe pero le faltan columnas ({', '.join(faltan)}). "
            f"Probablemente lo escribió una versión anterior de T01_B; vuelve a "
            f"correr su sección 5.3.")
    return df


def sector_mercado(horas: str = "hours_employees",
                   ramas: tuple[str, ...] = RAMAS_FUERA) -> pd.DataFrame:
    """El sector de mercado anual: la economía menos ``ramas``, ya encadenada.

    Devuelve, por ``(code, year)``: ``q`` (índice de volumen del valor añadido
    retenido), ``h`` (horas) y ``e`` (ocupados), listos para
    :math:`Y = (Y/H)(H/E)E`.

    **Todo sale por resta desde el total, nunca por suma.** Japón no publica las
    secciones E, N, S, T ni U, así que sumar lo que se conserva lo deja fuera y
    restar lo que se quita lo mantiene dentro — misma aritmética donde las dos
    funcionan, y sólo una contesta para todos los países.

    ``horas`` por omisión son las de los **asalariados**, que es lo único que
    Estados Unidos y Japón publican por rama. Con ``horas="hours"`` se usan las
    de todos los ocupados, disponibles para la mayoría del panel y para ninguno
    de esos dos: la diferencia entre las dos llamadas es la cuña que T01_A §1.4
    mide en vez de suponer.
    """
    from puremacro.fetch import chain_volume

    an = actividad_anual()
    va = an["va_total"].copy()
    py = an["va_total_py"].copy()
    for r in ramas:
        va = va - an[f"va_{r}"]
        py = py - an[f"va_{r}_py"]
    out = {"q": chain_volume(va, py)}
    for etiqueta, stem in (("h", horas), ("e", "emp_employees"
                                          if horas.endswith("employees")
                                          else "emp")):
        if stem not in an.columns:
            continue
        s = an[stem].copy()
        for r in ramas:
            col = f"{stem}_{r}"
            if col not in an.columns:
                s = None
                break
            s = s - an[col]
        if s is not None:
            out[etiqueta] = s
    return pd.DataFrame(out).sort_index()


def bloques() -> dict[str, pd.DataFrame]:
    """Los cuatro, en un diccionario, por si quieres iterar sobre ellos."""
    return {"gasto": gasto(), "renta": renta(),
            "produccion": produccion(), "mercado_laboral": mercado_laboral()}


def paises(bloque: str = "gasto") -> list[str]:
    """Códigos ISO-3 con datos en ese bloque."""
    return sorted(bloques()[bloque].index.get_level_values("code").unique())


def pais(code: str, *, bloques_: tuple[str, ...] = ("gasto", "renta", "produccion",
                                                    "mercado_laboral")) -> pd.DataFrame:
    """Todo lo que hay de un país, en un marco con índice de fechas.

    Une los bloques por fecha. Un país que no publique alguno —Estados Unidos no
    publica ni producción ni bloque laboral— simplemente no trae esas columnas,
    en vez de traerlas llenas de NaN.
    """
    code = code.upper()
    partes = []
    for nombre in bloques_:
        df = bloques()[nombre]
        if code not in df.index.get_level_values("code"):
            continue
        g = df.loc[code].dropna(axis=1, how="all")
        if not g.empty:
            partes.append(g)
    if not partes:
        raise KeyError(
            f"{code} no aparece en ninguno de los cuatro bloques. "
            f"Disponibles en gasto: {', '.join(paises())}")
    out = pd.concat(partes, axis=1).sort_index()
    return out.loc[:, ~out.columns.duplicated()]


def volumen(nombre: str = "gdp", code: str | None = None,
            bloque: str = "gasto") -> pd.Series:
    """La medida de volumen, rehecha como ``100 × nominal / deflactor``.

    Exacta hasta el último bit — el deflactor *está definido* como ese cociente,
    no estimado, así que invertirlo no aproxima nada. Por eso el archivo no
    guarda volúmenes. Sólo está definida donde ambas patas son estrictamente
    positivas: los agregados que incluyen variación de existencias cambian de
    signo y ahí el cociente no significa nada.
    """
    df = bloques()[bloque]
    defl = f"{nombre}_defl"
    if defl not in df.columns:
        raise KeyError(
            f"{nombre} no lleva deflactor en el bloque '{bloque}', así que no "
            f"tiene medida de volumen. Las series de renta sólo existen a "
            f"precios corrientes.")
    num, den = df[nombre], df[defl]
    v = pd.Series(np.where((num > 0) & (den > 0), 100.0 * num / den, np.nan),
                  index=df.index, name=f"{nombre}_real")
    return v if code is None else v.loc[code.upper()]


def crecimiento(code: str, nombre: str = "gdp", *, anual: bool = False,
                bloque: str = "gasto") -> pd.Series:
    """Crecimiento porcentual del volumen: trimestral, o interanual con ``anual=True``.

    Las tasas de crecimiento son inmunes al año de referencia de precios y al
    tipo de cambio, así que son comparables entre países sin más trámite.
    """
    v = volumen(nombre, code, bloque=bloque).dropna()
    return 100.0 * (v.pct_change(4) if anual else v.pct_change())


def ciclo(code: str, nombre: str = "gdp", *, bloque: str = "gasto",
          hasta: str | None = None) -> pd.Series:
    """Componente cíclico HP del log del volumen, en puntos logarítmicos x100.

    ``hasta="2019-12-31"`` recorta antes de 2020, que es lo que hace T01_B §4.3
    y por la razón que explica ahí: el cierre administrativo de los servicios de
    contacto no es un ciclo económico.
    """
    from puremacro.data import hp_filter

    v = volumen(nombre, code, bloque=bloque).dropna()
    if hasta is not None:
        v = v.loc[:hasta]
    c, _ = hp_filter(100.0 * np.log(v))
    return pd.Series(np.asarray(c), index=v.index, name=f"{nombre}_ciclo")
