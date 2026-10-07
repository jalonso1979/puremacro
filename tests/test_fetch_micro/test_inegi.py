"""INEGI ENIGH provider against synthetic archives (offline)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import inegi


def _zip(csv_text: str, table="concentradohogar", year=2022, encoding="utf-8",
         extras=True) -> bytes:
    stem = f"conjunto_de_datos_{table}_enigh{year}_ns"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(f"{stem}/conjunto_de_datos/{stem}.csv", csv_text.encode(encoding))
        if extras:
            zf.writestr(f"{stem}/diccionario_de_datos/diccionario_datos_{table}"
                        f"_enigh{year}_ns.csv", "nemonico,nombre\ning_cor,Ingreso\n")
            zf.writestr(f"{stem}/catalogos/tam_loc.csv", "clave,descripcion\n1,100000+\n")
    return out.getvalue()


def _households(n=60, seed=0):
    rng = np.random.default_rng(seed)
    est = rng.integers(1, 6, n)
    return pd.DataFrame({
        "FOLIOVIV": [f"{(i % 3) + 1:02d}{i:08d}" for i in range(n)],
        "FOLIOHOG": "1",
        "UBICA_GEO": [f"{(i % 3) + 1:02d}001" for i in range(n)],
        "TAM_LOC": rng.integers(1, 5, n),
        "EST_DIS": [f"{e:03d}" for e in est],
        "UPM": [f"{e:03d}{rng.integers(0, 3):04d}" for e in est],
        "FACTOR": rng.integers(200, 900, n),
        "ING_COR": rng.lognormal(10.5, 0.8, n).round(2),
        "TOT_INTEG": rng.integers(1, 7, n),
        "SEXO_JEFE": rng.integers(1, 3, n),
    })


@pytest.fixture
def served(monkeypatch):
    df = _households()
    payload = _zip(df.to_csv(index=False))
    calls = []

    def fake(url, **k):
        calls.append(url)
        return payload

    monkeypatch.setattr(inegi._http, "cached_get", fake)
    return df, calls


def test_url_and_wave_checks():
    assert inegi.enigh_url(2022) == (
        "https://www.inegi.org.mx/contenidos/programas/enigh/nc/2022/microdatos/"
        "enigh2022_ns_concentradohogar_csv.zip")
    assert inegi.enigh_url(2018, "poblacion").endswith("enigh2018_ns_poblacion_csv.zip")
    assert inegi.available_years(2022) == [2016, 2018, 2020, 2022]
    with pytest.raises(ValueError, match="no ENIGH"):
        inegi.enigh_url(2021)
    with pytest.raises(ValueError, match="unknown ENIGH table"):
        inegi.enigh_url(2022, "hogar")


def test_fetch_builds_a_taylor_frame(served):
    df, calls = served
    mf = inegi.fetch_enigh(2022)
    assert calls == [inegi.enigh_url(2022)]
    assert mf.design.method == "taylor" and mf.unit == "household"
    assert mf.source == "inegi.enigh" and mf.vintage == "2022 concentradohogar"
    # identifiers keep their leading zeros; measures become numeric
    assert mf.data["folioviv"].iloc[0] == df["FOLIOVIV"].iloc[0]
    assert mf.data["est_dis"].iloc[0].startswith("00")
    assert not pd.api.types.is_numeric_dtype(mf.data["upm"])
    assert pd.api.types.is_numeric_dtype(mf.data["ing_cor"])
    res = mf.mean("ing_cor").iloc[0]
    w, y = df["FACTOR"].astype(float), df["ING_COR"]
    assert res["estimate"] == pytest.approx((w * y).sum() / w.sum())
    assert res["se"] > 0 and res["n"] == len(df)


def test_variable_selection_keeps_design_and_ids(served):
    mf = inegi.fetch_enigh(2022, ["ING_COR"])
    assert list(mf.data.columns) == ["folioviv", "foliohog", "ubica_geo",
                                     "est_dis", "upm", "factor", "ing_cor"]
    with pytest.raises(KeyError, match="not in ENIGH"):
        inegi.fetch_enigh(2022, ["ingreso"])


def test_latin1_archive_and_person_table(monkeypatch):
    csv = ("folioviv,foliohog,numren,est_dis,upm,factor,parentesco,nombre_loc\n"
           "0100000001,1,01,001,0010001,250,101,Jesús María\n"
           "0100000001,1,02,001,0010001,250,201,Jesús María\n"
           "0100000002,1,01,002,0020001,300,101,León\n")
    payload = _zip(csv, table="poblacion", encoding="latin-1")
    monkeypatch.setattr(inegi._http, "cached_get", lambda url, **k: payload)
    mf = inegi.fetch_enigh(2022, table="poblacion")
    assert mf.unit == "person" and len(mf) == 3
    assert mf.data["numren"].tolist() == ["01", "02", "01"]
    assert mf.data["nombre_loc"].iloc[2] == "León"
    assert mf.total("factor").iloc[0]["estimate"] == pytest.approx(2 * 250**2 + 300**2)


def test_flat_archive_as_inegi_publishes_it(monkeypatch):
    """Layout seen live in 2026: one root-level CSV, UTF-8 with BOM, CRLF."""
    csv = ("folioviv,foliohog,ubica_geo,est_dis,upm,factor,ing_cor,tot_integ\r\n"
           "0100005002,1,01001,003,0000001,206,56123.75,3\r\n"
           "0100005003,1,01001,003,0000002,206,108048.87,2\r\n")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("concentradohogar.csv", csv.encode("utf-8-sig"))
        zf.writestr("nota_bases_datos_enigh2018_ns.txt", "nota")
    monkeypatch.setattr(inegi._http, "cached_get", lambda url, **k: out.getvalue())
    mf = inegi.fetch_enigh(2018)
    assert mf.data.columns[0] == "folioviv"            # BOM stripped
    assert mf.data.loc[0, "upm"] == "0000001" and mf.data.loc[0, "est_dis"] == "003"
    assert mf.data["ing_cor"].tolist() == [56123.75, 108048.87]


def test_csv_picker():
    names = ["x/diccionario_de_datos/d_concentradohogar.csv",
             "x/conjunto_de_datos/conjunto_de_datos_concentradohogar_enigh2022_ns.csv",
             "x/catalogos/entidad.csv"]
    assert inegi._pick_csv(names, "concentradohogar") == names[1]
    assert inegi._pick_csv(["concentradohogar.csv"], "concentradohogar") == "concentradohogar.csv"
    with pytest.raises(ValueError, match="cannot tell"):
        inegi._pick_csv(["a.csv", "b.csv"], "concentradohogar")


def test_registered():
    from puremacro.fetch import registry
    info = registry.info("inegi.enigh")
    assert info.kind == "micro" and info.auth == "none"
    assert registry.load("inegi.enigh") is inegi.fetch_enigh
