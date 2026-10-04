"""Tests for ``puremacro.trade.mrio``: provenance, native readers, regularization, aggregation, coarse tariffs.

Analytic fixtures are hand-designed (``tests/fixtures/trade/mrio``), every
expected array below is derived by hand from the fixture text, and the parity
tests compare against the IO research implementations in a subprocess (the
vendored ``puremacro.trade.corrected`` package must never share a process with
the live package). Tests that need the IO volume skip when it is absent; the
real-file tests are marked ``slow``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import puremacro.trade.data as trade_data
from puremacro.trade import mrio as m
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    CANONICAL_SECTOR_CODES,
    RAW_45_SECTOR_CODES,
    RawIOData,
    load_icio_data,
    package_mrio_to_calibration_result,
    verify_accounting_invariants,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "trade" / "mrio"
FIGARO_FIXTURE = FIXTURES / "figaro_2c2s_26ed.csv"
EXIO_FIXTURE = FIXTURES / "exiobase_2r3s_ixi"
# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = pathlib.Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))
VENDOR = IO_ROOT / "headlinePaper/rebuild/vendor"
IO_SOURCES = IO_ROOT / "headlinePaper/rebuild/empirical_2019/sources"
ICIO_2019 = IO_ROOT / "ICIOextended/2019_SML.csv"
ICIO_2020 = IO_ROOT / "ICIOextended/2020_SML.csv"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def toy_table(**overrides):
    """Port of the IO ``toy_native_data``: two countries, two industries, a signed inventory drawdown."""
    z = sparse.csr_matrix([[1, .1, .2, 0], [.3, 1, 0, .1], [.1, 0, 2, .2], [0, .2, .1, 1]])
    C = np.array([[4., .5], [1., 1.], [.4, 5.], [1., 3.]])
    G = np.array([[1., .1], [2., .2], [.1, 1.5], [.2, 2.]])
    X = np.array([[.1, .2], [0, 0], [.3, .2], [0, 0]])
    V = np.array([[-.4, 0], [.2, .1], [.1, -.2], [0, 0]])
    VAL = np.array([[.02, 0], [0, .01], [0, 0], [0, 0]])
    F = np.stack([C, G, X, V, VAL], axis=2)
    output = np.asarray(z.sum(1)).ravel() + F.sum(axis=(1, 2))
    taxes = np.array([.1, -.1, .2, .1])
    va = output - np.asarray(z.sum(0)).ravel() - taxes
    prodtax = np.array([.02, .04, .01, .03])
    kw = dict(
        Z=z, F=F, VA=va, TLS=taxes, TFD=np.array([[.2, .1, 0, -.1, 0], [.4, .3, 0, .1, 0]]),
        country_codes=("A", "B"), sector_codes=("goods", "services"), fd_codes=m.FINAL_USE_CATEGORIES,
        output=output, production_taxes=prodtax, labor_compensation=.6 * (va - prodtax),
        operating_surplus=.4 * (va - prodtax), merchandise_mask=np.array([True, False]),
        metadata={"dataset": "synthetic_unit_fixture", "units": "synthetic monetary units"},
    )
    kw.update(overrides)
    return m.MRIOTable.from_arrays(**kw)


def random_table(seed=0, n=3, s=4, *, sectors=None, countries=None, fd_codes=("C", "G", "X", "V"),
                 z_scale=0.05, negative_inventories=True):
    """Balanced random table with strictly positive value added (no floors, no phantoms)."""
    rng = np.random.default_rng(seed)
    M = n * s
    k = len(fd_codes)
    Z = rng.uniform(0.0, z_scale, (M, M))
    F = rng.uniform(0.2, 1.0, (M, n, k))
    if "X" in fd_codes:
        F[:, :, fd_codes.index("X")] *= 0.1
    if "V" in fd_codes and negative_inventories:
        F[:, :, fd_codes.index("V")] = rng.uniform(-0.05, 0.05, (M, n))
    y = Z.sum(1) + F.sum((1, 2))
    TLS = 0.03 * y
    VA = y - Z.sum(0) - TLS
    assert np.all(VA > 0)
    TFD = 0.04 * F.sum(0)
    ccodes = countries or tuple(["USA", "CHN", "ROW", "DEU", "JPN"][:n])
    scodes = sectors or tuple(f"S{i}" for i in range(s))
    return m.MRIOTable.from_arrays(Z, F, VA, TLS, TFD, country_codes=ccodes, sector_codes=scodes, fd_codes=fd_codes,
                                   reference_country=ccodes[-1], metadata={"dataset": f"random{seed}", "year": 2019})


def random_45(seed=7, n=3):
    """A balanced 45-industry table with the OECD registry and goods mask."""
    t = random_table(seed, n, 45, sectors=RAW_45_SECTOR_CODES, z_scale=0.02)
    return replace(t, merchandise_mask=m.goods_mask(RAW_45_SECTOR_CODES))


def _subprocess(code, cwd=None):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return r.stdout


def _require_io(path=None):
    if not (path or IO_ROOT).exists():
        pytest.skip("IO research volume not mounted")


def _maxdiff(a, b):
    if sparse.issparse(a) or sparse.issparse(b):
        d = abs(sparse.csr_matrix(a) - sparse.csr_matrix(b))
        return float(d.max()) if d.nnz else 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _read_exio_dir(core=EXIO_FIXTURE, **kw):
    """Read an extracted EXIOBASE directory without its archive (warned as unauthenticated)."""
    with pytest.warns(RuntimeWarning, match="without its archive"):
        return m.read_exiobase_native(core, **kw)


def _write_exio_zip(target, core=EXIO_FIXTURE, prefix="IOT_2019_ixi/"):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(core.rglob("*")):
            if p.is_file():
                z.write(p, prefix + p.relative_to(core).as_posix())
    return target


def _tiny_oecd_csv(directory, name="2019_SML.csv"):
    """A 2 x 2 table in the OECD SML layout; returns (path, Z, F6, VA, TLS, TFD6, Y)."""
    countries, sectors = ("AAA", "BBB"), ("A01_02", "D")
    nodes = [f"{c}_{s}" for c in countries for s in sectors]
    fd = [f"{c}_{f}" for c in countries for f in m.OECD_FD_CODES]
    Z = np.array([[1, 2, 0.5, 0], [0.5, 3, 0, 1], [0.2, 0, 4, 1], [0, 0.3, 0.6, 2.]])
    F6 = np.zeros((4, 2, 6))
    F6[0] = [[5, 0.2, 1, 0.5, -0.1, 0.3], [0.4, 0, 0, 0.1, 0, 0]]
    F6[1] = [[2, 0.1, 0.5, 3, 0.2, 0], [0.1, 0, 0.2, 0.3, 0, 0]]
    F6[2] = [[0.3, 0, 0, 0.2, 0, 0], [6, 0.1, 1.5, 1, 0.1, 0.4]]
    F6[3] = [[0.1, 0, 0, 0.1, 0, 0], [3, 0.2, 0.9, 2, -0.2, 0]]
    Y = Z.sum(1) + F6.sum((1, 2))
    TLS = 0.05 * Y
    VA = Y - Z.sum(0) - TLS
    TFD6 = 0.02 * F6.sum(0)
    top = pd.DataFrame(np.hstack([Z, F6.reshape(4, 12), Y[:, None]]), index=nodes, columns=nodes + fd + ["OUT"])
    bottom = pd.DataFrame([np.r_[TLS, TFD6.ravel(), 0.0], np.r_[VA, np.zeros(12), 0.0], np.r_[Y, np.zeros(12), 0.0]],
                          index=["TLS", "VA", "OUT"], columns=top.columns)
    frame = pd.concat([top, bottom])
    frame.index.name = "V1"
    path = pathlib.Path(directory) / name
    frame.to_csv(path)
    return path, countries, sectors, Z, F6, VA, TLS, TFD6, Y


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_identify_source_matches_hashlib_and_rejects_changes(self, tmp_path):
        src = tmp_path / "source.csv"
        src.write_bytes(b"not an empirical fixture\n")
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        md5 = hashlib.md5(src.read_bytes()).hexdigest()
        rec = m.identify_source(src, expected_sha256=sha, expected_md5=md5)
        assert (rec.sha256, rec.md5, rec.bytes, rec.status) == (sha, md5, 25, "verified")
        assert rec.to_dict()["path"] == str(src.resolve())
        assert m.identify_source(src).status == "hashed"
        src.write_bytes(b"changed contents\n")
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.identify_source(src, expected_sha256=sha)
        with pytest.raises(m.MRIOIntegrityError, match="MD5 mismatch"):
            m.identify_source(src, expected_md5=md5)
        with pytest.raises(FileNotFoundError, match="Authenticated native source"):
            m.identify_source(tmp_path / "absent.csv")

    def test_source_record_renderers(self, tmp_path):
        src = tmp_path / "s.csv"
        src.write_bytes(b"abc\n")
        rec = m.identify_source(src)
        df = rec.to_dataframe()
        assert df.loc["status", "Value"] == "hashed" and df.loc["bytes", "Value"] == 4
        assert rec.summary().startswith("s.csv: hashed, 4 bytes")
        assert rec.to_markdown() and "tabular" in rec.to_latex() and "#table" in rec.to_typst()

    def test_verify_zip_member_detects_size_and_crc_mismatch(self, tmp_path):
        member = tmp_path / "x.txt"
        member.write_text("region\tsector\tindout\nAT\ta\t10.15\n", encoding="utf-8")
        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(member, "IOT/x.txt")
        rec = m.verify_zip_member(archive, "IOT/x.txt")
        assert rec.status == "verified" and rec.archive_member == "IOT/x.txt" and rec.crc32 is not None
        assert rec.sha256 == hashlib.sha256(member.read_bytes()).hexdigest()
        assert m.verify_zip_member(archive, "IOT/x.txt", member).sha256 == rec.sha256
        member.write_text("region\tsector\tindout\nAT\ta\t10.16\n", encoding="utf-8")
        with pytest.raises(m.MRIOIntegrityError, match="CRC-32"):
            m.verify_zip_member(archive, "IOT/x.txt", member)
        member.write_text("short\n", encoding="utf-8")
        with pytest.raises(m.MRIOIntegrityError, match="bytes"):
            m.verify_zip_member(archive, "IOT/x.txt", member)
        with pytest.raises(m.MRIOIntegrityError, match="no member"):
            m.verify_zip_member(archive, "IOT/missing.txt")

    def test_check_oecd_source_registries(self, tmp_path):
        path = tmp_path / "2019_SML.csv"
        path.write_text("V1,AAA_A\nAAA_A,1\n", encoding="utf-8")
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        # The real registries are facts about other files: this digest is unknown to them.
        assert md5 not in m.OECD_KNOWN_CORRUPTED_MD5 and md5 not in m.OECD_ICIO_MD5.values()
        with pytest.warns(RuntimeWarning, match="not a whitelisted"):
            rec = m.check_oecd_source(path)
        assert rec.status == "unknown_edition"
        # Test-only registries exercise the refusal and whitelist paths.
        with pytest.raises(m.MRIOIntegrityError, match="lost decimal points"):
            m.check_oecd_source(path, corrupted={md5: m.OECD_KNOWN_CORRUPTED_MD5["d1b887aaafa54ab3f28fde78fcd21cdf"]})
        rec = m.check_oecd_source(path, year=2019, whitelist={2019: md5})
        assert rec.status == "whitelisted" and "2019_SML.csv" in rec.note
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year"):
            m.check_oecd_source(path, year=2020, whitelist={2019: md5})
        archive = tmp_path / "2016-2020_SML.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(path, "2019_SML.csv")
        rec = m.check_oecd_source(archive, year=2019, whitelist={2019: md5})
        assert rec.status == "whitelisted" and rec.archive_member == "2019_SML.csv"
        with pytest.raises(ValueError, match="year is required"):
            m.check_oecd_source(archive)
        assert m.OECD_ICIO_MD5 == {2019: "28cba31491177955445051d459053744", 2020: "d3e0f4979d85d6c0bb7cf4c43e324287"}
        assert list(m.OECD_KNOWN_CORRUPTED_MD5) == ["d1b887aaafa54ab3f28fde78fcd21cdf"]


# ---------------------------------------------------------------------------
# Concordances and goods masks
# ---------------------------------------------------------------------------


class TestConcordance:
    def test_build_validation(self):
        with pytest.raises(ValueError, match="1-based"):
            m.Concordance.build("x", ("a", "b"), (0, 1))
        with pytest.raises(ValueError, match="entries"):
            m.Concordance.build("x", ("a", "b"), (1,))
        with pytest.raises(ValueError, match="empty"):
            m.Concordance.build("x", ("a", "b"), (1, 3))
        with pytest.raises(ValueError, match="coarse codes"):
            m.Concordance.build("x", ("a", "b"), (1, 2), ("g1",))
        with pytest.raises(ValueError, match="unique"):
            m.Concordance.build("x", ("a", "a"), (1, 1))
        c = m.Concordance.build("x", ("a", "b", "c"), (1, 2, 1))
        assert c.coarse_codes == ("G01", "G02") and c.members("G01") == ("a", "c") and c.group_of("b") == "G02"

    def test_from_mapping_identity_and_compose(self):
        c = m.Concordance.from_mapping("map", {"a": "g", "b": "h", "c": "g"})
        assert c.coarse_codes == ("g", "h") and c.groups == (1, 2, 1)
        with pytest.raises(ValueError, match="lacks"):
            m.Concordance.from_mapping("map", {"a": "g"}, fine_codes=("a", "b"))
        ident = m.Concordance.identity(("a", "b"))
        assert ident.groups == (1, 2) and ident.coarse_codes == ("a", "b")
        outer = m.Concordance.from_mapping("outer", {"g": "all", "h": "all"})
        composed = c.compose(outer)
        assert composed.groups == (1, 1, 1) and composed.coarse_codes == ("all",)
        with pytest.raises(ValueError, match="fine_codes"):
            c.compose(c)

    def test_projection_matches_kron_and_coarse_mask_rules(self):
        c = m.Concordance.build("x", ("a", "b", "c"), (1, 2, 1))
        C = c.matrix
        assert C.shape == (3, 2) and np.array_equal(C, [[1, 0], [0, 1], [1, 0]])
        P = c.projection(2)
        assert np.array_equal(P, np.kron(np.eye(2), C.T))
        assert np.array_equal(c.projection(2, sparse_matrix=True).toarray(), P)
        mask = np.array([True, False, False])
        assert c.coarse_mask(mask, "all").tolist() == [False, False]
        assert c.coarse_mask(mask, "any").tolist() == [True, False]
        assert c.coarse_mask(np.array([True, False, True]), "majority").tolist() == [True, False]
        with pytest.raises(ValueError):
            c.coarse_mask(mask, "none")
        df = c.to_dataframe()
        assert list(df.index) == ["a", "b", "c"] and df.loc["c", "coarse"] == "G01"
        assert "| fine" in c.to_markdown() and "tabular" in c.to_latex() and "#table" in c.to_typst()
        assert c.summary() == "concordance 'x': 3 fine codes -> 2 groups (sizes 2, 1)"

    def test_registered_concordances_and_truthful_names(self):
        ag, isic = m.CONCORDANCES["agregar"], m.CONCORDANCES["isic_section"]
        assert ag.fine_codes == RAW_45_SECTOR_CODES == isic.fine_codes
        assert (ag.n_coarse, isic.n_coarse) == (11, 11)
        assert ag.groups == m.AGREGAR_11_CONCORDANCE and isic.groups == m.ISIC_SECTION_11
        assert ag.members("AGRI_MIN_FOOD") == RAW_45_SECTOR_CODES[:6]
        assert ag.members("UTILITIES") == ("D", "E")
        assert isic.members("A") == ("A01_02", "A03") and isic.members("C") == RAW_45_SECTOR_CODES[5:22]
        assert isic.members("REST") == ("I", "M", "N", "R", "S", "T")
        # The truthful labels of the bundled positions: group 1 is agriculture, mining AND food, group 3 utilities.
        assert "mining and food" in m.AGG11_SECTOR_NAMES["AGRI_MIN_FOOD"]
        assert m.AGG11_SECTOR_NAMES["UTILITIES"].startswith("Utilities")
        assert m.AGG11_SECTOR_CODES[0] == "AGRI_MIN_FOOD" and m.AGG11_SECTOR_CODES[2] == "UTILITIES"
        assert m.ISIC_SECTION_11_CODES == ("A", "B", "C", "DE", "F", "G", "H", "J", "KL", "OPQ", "REST")

    def test_goods_mask_classifications(self):
        oecd = m.goods_mask(RAW_45_SECTOR_CODES)
        assert oecd.sum() == 22 and oecd[:22].all() and not oecd[22:].any()
        assert m.goods_mask(m.AGG11_SECTOR_CODES).tolist() == [True, True] + [False] * 9
        assert m.goods_mask(m.ISIC_SECTION_11_CODES).tolist() == [True, True, True] + [False] * 8
        fig = m.goods_mask(m.FIGARO_NATIVE_64_SECTORS)
        assert fig.sum() == 22 and not fig[m.FIGARO_NATIVE_64_SECTORS.index("C33")]
        assert m.goods_mask(("i01.a", "i01.w.1", "i37", "i38", "i60.2")).tolist() == [True, False, True, False, False]
        with pytest.raises(ValueError, match="by position"):
            m.goods_mask(CANONICAL_SECTOR_CODES)
        assert m.goods_mask(CANONICAL_SECTOR_CODES, "agregar11").tolist() == [True, True] + [False] * 9
        assert m.goods_mask(CANONICAL_SECTOR_CODES, "canonical11").tolist() == [True, True, True] + [False] * 8
        with pytest.raises(ValueError, match="unknown"):
            m.goods_mask(("foo", "bar"))
        # The Agregar group travels with each canonical label, whatever the order or subset.
        assert m.goods_mask(tuple(reversed(CANONICAL_SECTOR_CODES)), "agregar11").tolist() == [False] * 9 + [True, True]
        assert m.goods_mask(("AGRI", "MINQ", "MANU", "ENEG", "CONS"), "agregar11").tolist() == [True, True, False, False, False]
        assert m.goods_mask(("MANUF_EXFOOD", "MINQ", "TRADE"), "agregar11").tolist() == [True, True, False]
        # A code outside the named registry is refused instead of silently counted as a service.
        for codes, kind in ((("AGRI", "foo"), "agregar11"), (("A01_02", "AGRI"), "oecd45"), (("A01", "C99"), "figaro64"),
                            (("A", "ZZ"), "isic_section11"), (("AGRI", "A01"), "canonical11")):
            with pytest.raises(ValueError, match=f"outside the {kind}"):
                m.goods_mask(codes, kind)


# ---------------------------------------------------------------------------
# The table contract
# ---------------------------------------------------------------------------


class TestTableContract:
    def test_from_arrays_validation(self):
        t = toy_table()
        with pytest.raises(ValueError, match="Z shape"):
            toy_table(Z=np.eye(3))
        with pytest.raises(ValueError, match="F shape"):
            toy_table(F=np.zeros((4, 3, 5)))
        with pytest.raises(ValueError, match="canonical order"):
            toy_table(fd_codes=("G", "C", "X", "V", "VAL"))
        with pytest.raises(ValueError, match="subset"):
            toy_table(fd_codes=("C", "G", "X", "V", "OTHER"))
        with pytest.raises(m.MRIOIntegrityError, match="nonnegative"):
            toy_table(Z=-np.eye(4))
        with pytest.raises(m.MRIOIntegrityError, match="NaN"):
            toy_table(output=np.array([1., np.nan, 2., 3.]))
        F = np.array(t.F)
        F[0, 0, 1] = -1.0
        with pytest.raises(m.MRIOIntegrityError, match="negative fixed investment"):
            toy_table(F=F)
        assert toy_table(F=F, strict=False).accounting_report().negative_investment_entries == 1
        with pytest.raises(ValueError, match="together"):
            toy_table(operating_surplus=None)
        with pytest.raises(ValueError, match="reference_country"):
            toy_table(reference_country="ZZZ")
        with pytest.raises(ValueError, match="unique"):
            toy_table(country_codes=("A", "A"))
        with pytest.raises(ValueError, match="merchandise_mask"):
            toy_table(merchandise_mask=np.array([1, 2]))
        with pytest.raises(ValueError, match="TFD shape"):
            toy_table(TFD=np.zeros((2, 4)))

    def test_arrays_are_read_only_copies(self):
        F = np.ones((4, 2, 5))
        t = toy_table(F=F, strict=False)
        F[0, 0, 0] = 100.0
        assert t.F[0, 0, 0] == 1.0
        with pytest.raises(ValueError):
            t.F[0, 0, 0] = 5.0
        with pytest.raises(ValueError):
            t.VA[0] = 5.0
        assert t.is_sparse and not t.Z.data.flags.writeable
        dense = toy_table(Z=np.eye(4))
        assert not dense.is_sparse and not dense.Z.flags.writeable and np.array_equal(dense.Z_dense, np.eye(4))

    def test_toy_accounts(self):
        d = toy_table()
        assert d.fd_codes == m.FINAL_USE_CATEGORIES == ("C", "G", "X", "V", "VAL")
        assert d.final_use("G")[1, 0] == 2 and d.final_use("X")[1, 0] == 0
        assert d.final_use("V")[0, 0] == -.4 and d.final_use("VAL")[0, 0] == .02
        np.testing.assert_array_equal(d.country_of_cell, [0, 0, 1, 1])
        assert d.cell_labels == ("A_goods", "A_services", "B_goods", "B_services")
        rep = d.accounting_report()
        assert rep.row_output_max_absolute_gap < 1e-13 and rep.column_output_max_absolute_gap < 1e-13
        assert rep.factor_decomposition_max_absolute_gap < 1e-13
        assert rep.negative_inventory_entries == 2 and rep.zero_output_cells == 0
        assert d.transformations == () and d.is_raw
        np.testing.assert_allclose(d.factor_VA, d.VA - d.production_taxes)
        np.testing.assert_allclose(d.capital_share, .4)
        assert d.capital_share_available.all()
        assert not np.allclose(d.operating_surplus / d.VA, d.capital_share)
        np.testing.assert_allclose(d.final_demand, d.F.sum(axis=2))
        assert rep.final_category_totals["V"] == pytest.approx(-.2)

    def test_missing_and_negative_factors_remain_unavailable_without_imputation(self):
        d = toy_table()
        missing = toy_table(labor_compensation=None, operating_surplus=None, production_taxes=None)
        assert np.isnan(missing.capital_share).all() and not missing.capital_share_available.any()
        np.testing.assert_allclose(missing.factor_VA, missing.VA)
        cap = np.array(d.operating_surplus)
        cap[0] = -1
        negative = toy_table(operating_surplus=cap)
        assert negative.operating_surplus[0] == -1 and np.isnan(negative.capital_share[0])
        assert not negative.capital_share_available[0]
        assert negative.accounting_report().factor_decomposition_max_absolute_gap > 0

    def test_zero_output_cell_is_preserved_without_phantom(self):
        d = toy_table()
        z = d.Z.toarray()
        z[0] = 0
        z[:, 0] = 0
        changes = {"Z": z}
        for key in ("F", "VA", "TLS", "production_taxes", "labor_compensation", "operating_surplus", "output"):
            arr = np.array(getattr(d, key))
            arr[0] = 0
            changes[key] = arr
        empty = toy_table(**changes)
        assert not empty.active_mask[0] and empty.output[0] == empty.VA[0] == 0
        assert empty.accounting_report().zero_output_cells == 1

    def test_source_rounding_and_signed_inventory_are_reported_without_repair(self):
        d = toy_table()
        output = np.array(d.output)
        output[0] = -.001
        F = np.array(d.F)
        F[0, 0, 0] = -.001
        raw = toy_table(output=output, F=F)
        assert raw.output[0] == -.001 and raw.final_use("C")[0, 0] == -.001
        rep = raw.accounting_report()
        assert rep.negative_output_cells == rep.negative_consumption_entries == 1
        assert rep.row_output_max_absolute_gap > 0

    def test_net_exports_sum_to_zero_and_reference_absorbs_roundoff(self):
        t = random_table(3, 4, 3)
        xn = t.net_exports()
        assert abs(xn.sum()) < 1e-12
        B = t.bilateral_flows()
        np.testing.assert_allclose(xn[:-1], (B.sum(1) - B.sum(0))[:-1], atol=1e-12)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        np.testing.assert_allclose(sp.net_exports(), xn, atol=1e-12)
        np.testing.assert_allclose(sp.bilateral_flows(), B, atol=1e-12)

    def test_from_raw_layouts(self):
        Z = np.array([[1, 2, 0.5, 0], [0.5, 3, 0, 1], [0.2, 0, 4, 1], [0, 0.3, 0.6, 2.]])
        F6 = np.abs(np.random.default_rng(1).normal(size=(4, 2, 6)))
        F6[:, :, 4] -= 0.3
        Y = Z.sum(1) + F6.sum((1, 2))
        raw = RawIOData(countries=["AAA", "BBB"], sectors=["A01_02", "D"], intermediate_matrix=Z,
                        final_demand_matrix=F6.reshape(4, 12), value_added=Y - Z.sum(0), taxes_less_subsidies=np.zeros(4),
                        gross_output=Y, fd_categories=list(m.OECD_FD_CODES), year=2019, source="test",
                        taxes_less_subsidies_fd=np.zeros(12))
        t = m.MRIOTable.from_raw(raw)
        assert t.fd_codes == ("C", "G", "X", "V")
        np.testing.assert_allclose(t.final_use("C"), F6[:, :, :3].sum(2))
        np.testing.assert_allclose(t.final_use("V"), F6[:, :, 4])
        np.testing.assert_allclose(t.final_use("X"), F6[:, :, 5])
        assert set(t.metadata["final_use_components"]) == set(m.OECD_FD_CODES)
        # Legacy C, I, Cx with a negative investment cell.
        F3 = np.stack([F6[:, :, :3].sum(2), F6[:, :, 3] + F6[:, :, 4], F6[:, :, 5]], axis=2)
        assert (F3[:, :, 1] < 0).any()
        legacy = replace(raw, final_demand_matrix=F3.reshape(4, 6), fd_categories=["C", "I", "Cx"],
                         taxes_less_subsidies_fd=np.zeros(6))
        with pytest.raises(m.MRIOIntegrityError, match="to_inventory"):
            m.MRIOTable.from_raw(legacy)
        moved = m.MRIOTable.from_raw(legacy, negative_investment="to_inventory")
        assert moved.fd_codes == ("C", "G", "X", "V") and moved.final_use("G").min() == 0
        np.testing.assert_allclose(moved.final_use("G") + moved.final_use("V"), F3[:, :, 1])
        assert any("negative investment" in x for x in moved.transformations)
        with pytest.raises(ValueError, match="negative_investment"):
            m.MRIOTable.from_raw(legacy, negative_investment="clip")
        native = replace(raw, final_demand_matrix=F6[:, :, :2].reshape(4, 4), fd_categories=["C", "V"],
                         taxes_less_subsidies_fd=None)
        assert m.MRIOTable.from_raw(native).fd_codes == ("C", "V")
        with pytest.raises(ValueError, match="fd_mapping"):
            m.MRIOTable.from_raw(replace(raw, fd_categories=["h", "n", "g", "i", "s", "d"]))
        custom = m.MRIOTable.from_raw(replace(raw, fd_categories=["h", "n", "g", "i", "s", "d"]),
                                      fd_mapping={"h": "C", "n": "C", "g": "C", "i": "G", "s": "V", "d": "X"})
        np.testing.assert_allclose(custom.F, t.F)

    def test_renderers_and_summary(self):
        t = toy_table()
        for which in ("summary", "countries", "categories", "sources"):
            df = t.to_dataframe(which)
            assert isinstance(df, pd.DataFrame)
            assert t.to_markdown(which) and "tabular" in t.to_latex(which) and "#table" in t.to_typst(which)
        with pytest.raises(ValueError):
            t.to_dataframe("other")
        assert "synthetic_unit_fixture" in t.summary()
        rep = t.accounting_report()
        assert rep.to_markdown() and rep.to_latex() and rep.to_typst() and "active cells" in rep.summary()
        reg, build = m.regularize_table(t)
        for which in ("summary", "gates", "floored"):
            assert build.to_markdown(which) and build.to_latex(which) and build.to_typst(which)
        assert "phantoms=0" in build.summary()
        assert json.dumps(build.to_dict()) and json.dumps(rep.to_dict())
        countries = t.to_dataframe("countries")
        assert list(countries.columns[:3]) == ["output", "value_added", "TLS"] and "V_expenditure" in countries.columns
        assert t.with_metadata(dataset="renamed").dataset == "renamed" and t.dataset == "synthetic_unit_fixture"


# ---------------------------------------------------------------------------
# Readers on the hand fixtures
# ---------------------------------------------------------------------------

FIGARO_F = np.array([
    [[43., 3., 0., -0.3], [6.5, 1., 0., 0.2]],
    [[66., 8., 0., 1.], [0., 2., 0., 0.4995]],
    [[7.2, 0.5, 0., 0.], [54.5, 6., 0., -1.2]],
    [[0., 0., 0., 0.], [0., 0., 0., 0.]],
])
FIGARO_Z = np.array([[10, 30, 2, 0], [5, 20, 3, 0], [4, 6, 15, 0], [0, 0, 0, 0]], dtype=float)


class TestFigaroReader:
    def _read(self, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return m.read_figaro_native(FIGARO_FIXTURE, **kw)

    def test_fixture_arrays_with_rounding_repair(self):
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            t = m.read_figaro_native(FIGARO_FIXTURE)
        assert t.country_codes == ("AUT", "ROW") and t.sector_codes == ("A01", "C10T12")
        assert t.fd_codes == ("C", "G", "X", "V") and t.reference_country == "ROW"
        np.testing.assert_allclose(t.Z, FIGARO_Z)
        np.testing.assert_allclose(t.F, FIGARO_F, atol=1e-15)
        np.testing.assert_allclose(t.VA, [73.4, 43.9995, 70.0, 0.0])
        np.testing.assert_allclose(t.labor_compensation, [40, 30, 45, 0])
        np.testing.assert_allclose(t.production_taxes, [-1.4, 1, 0.5, 0])
        np.testing.assert_allclose(t.operating_surplus, [34.8, 12.9995, 24.5, 0])
        np.testing.assert_allclose(t.TLS, [3, 5.5, 2, 0])
        np.testing.assert_allclose(t.TFD, [[4.3, 0.5, 0, 0.05], [5.4, 0.6, 0, -0.02]])
        np.testing.assert_allclose(t.output, [95.4, 105.4995, 92.0, 0.0])
        assert t.merchandise_mask.tolist() == [True, True]
        assert t.metadata["purchases_abroad_residents"] == pytest.approx(2.3)
        assert t.metadata["purchases_by_nonresidents"] == pytest.approx(2.3)
        adj = t.metadata["rounding_adjustments"]
        assert [a["reason"] for a in adj] == ["negative empty-cell rounding", "rounding reclassified from C to V"]
        assert adj[0]["cell"] == "FIGW1_C10T12" and adj[0]["output"] == -0.001 and adj[0]["absolute_input_mass"] == pytest.approx(0.002)
        assert adj[1]["cell"] == "AT_C10T12" and adj[1]["destination"] == "FIGW1" and adj[1]["amount"] == -0.0005
        assert len(t.transformations) == 1 and t.build_report.rounding_adjustments == adj
        rep = t.accounting_report()
        assert rep.row_column_max_absolute_gap < 1e-12 and rep.zero_output_cells == 1 and rep.negative_inventory_entries == 2
        assert rep.world_production_taxes == pytest.approx(0.1)
        assert t.sources[0].status == "unknown_edition" and t.sources[0].sha256 == hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        # Repairs preserve every row total and each country's total expenditure.
        raw = self._read(rounding_repair=False)
        np.testing.assert_allclose(t.row_sales()[:3], raw.row_sales()[:3])
        np.testing.assert_allclose(t.F.sum(axis=(0, 2))[:2], raw.F.sum(axis=(0, 2))[:2] + np.array([0, 0.001]))

    def test_fixture_without_repair_keeps_source_rounding(self):
        raw = self._read(rounding_repair=False)
        assert raw.is_raw and raw.output[3] == -0.001 and raw.TLS[3] == 0.001
        assert raw.final_use("C")[1, 1] == -0.0005 and raw.final_use("V")[1, 1] == 0.5
        assert raw.final_use("C")[3, 1] == -0.001
        rep = raw.accounting_report()
        assert rep.negative_output_cells == 1 and rep.negative_consumption_entries == 2

    def test_layout_errors_and_digest_checks(self, tmp_path):
        text = FIGARO_FIXTURE.read_text(encoding="utf-8")
        sha = hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok = m.read_figaro_native(FIGARO_FIXTURE, expected_sha256=sha)
        assert ok.sources[0].status == "verified"
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_figaro_native(FIGARO_FIXTURE, expected_sha256="0" * 64)
        bad = tmp_path / "bad.csv"
        lines = text.splitlines()
        bad.write_text("\n".join(lines[:5] + lines[6:7] + lines[5:6] + lines[7:]) + "\n", encoding="utf-8")  # swap two trailing rows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(m.MRIOIntegrityError, match="must end with the rows"):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("AT_P3_S13,AT_P3_S14", "AT_P3_S14,AT_P3_S13"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError, match="final-use columns"):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("AT_A01,AT_C10T12,FIGW1_A01,FIGW1_C10T12,AT_P3", "AT_A01,FIGW1_A01,AT_C10T12,FIGW1_C10T12,AT_P3"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError):
                m.read_figaro_native(bad)
            bad.write_text(text.replace("FIGW1_C10T12,0,0,0,0,0,0,0,0,0,0,-0.001", "FIGW1_C10T12,0,0,0,0,0,0,0,0,0,0,-0.5"), encoding="utf-8")
            with pytest.raises(m.MRIOIntegrityError, match="Non-rounding negative output"):
                m.read_figaro_native(bad)
            sp = m.read_figaro_native(FIGARO_FIXTURE, sparse_matrix=True)
        assert sp.is_sparse and _maxdiff(sp.Z, FIGARO_Z) == 0


class TestFigaroRegisteredDigest:
    def test_registered_digest_is_bound_to_its_year(self, monkeypatch):
        sha = hashlib.sha256(FIGARO_FIXTURE.read_bytes()).hexdigest()
        monkeypatch.setattr(m, "FIGARO_2026ED_2019_SHA256", sha)  # test-only: treat the fixture as the registered file
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = m.read_figaro_native(FIGARO_FIXTURE)
        assert t.sources[0].status == "verified" and t.year == 2019
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year 2020"):
            m.read_figaro_native(FIGARO_FIXTURE, year=2020)


EXIO_F = np.array([
    [[6.0, 0.4, 0, -0.3, 0.05], [0.7, 0.2, 0, 0.1, 0]],
    [[0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[4.3, 0.5, 0, 0, 0], [0.3, 0.1, 0, 0.02, 0]],
    [[0.8, 0.3, 0, 0.05, 0], [9.8, 1.0, 0, -0.4, 0.1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0.3, 0.4, 0, 0, 0], [5.1, 1.2, 0, 0.1, 0.02]],
])
EXIO_Z = np.array([
    [2.0, 0.1, 0.5, 0.3, 0, 0.1],
    [0.2, 0.4, 0.0, 0.0, 0, 0.0],
    [0.6, 0.1, 1.5, 0.2, 0, 0.4],
    [0.4, 0.0, 0.1, 3.0, 0, 0.7],
    [0, 0, 0, 0, 0, 0],
    [0.1, 0.0, 0.3, 0.9, 0, 2.5],
])


class TestExiobaseReader:
    def test_fixture_arrays(self):
        t = _read_exio_dir()
        assert t.country_codes == ("AUT", "ROW_ASIA_PACIFIC") and t.sector_codes == ("i01.a", "i01.w.1", "i60.2")
        assert t.fd_codes == ("C", "G", "X", "V", "VAL") and t.is_sparse and t.reference_country == "ROW_ASIA_PACIFIC"
        assert _maxdiff(t.Z, EXIO_Z) == 0 and t.Z.nnz == 20
        np.testing.assert_allclose(t.F, EXIO_F, atol=1e-15)
        np.testing.assert_allclose(t.TFD, [[0.62, 0.05, 0, 0.01, 0], [1.08, 0.1, 0, -0.01, 0.005]], atol=1e-15)
        np.testing.assert_allclose(t.TLS, [0.35, 0.01, 0.2, 0.45, 0, 0.3])
        np.testing.assert_allclose(t.production_taxes, [-0.1, 0.02, 0.12, 0.2, 0, -0.05])
        np.testing.assert_allclose(t.labor_compensation, [3.5, 0.05, 3.0, 6.3, 0, 4.0])
        np.testing.assert_allclose(t.operating_surplus, [3.1, 0.02, 2.3, 4.5, 0, 2.97])
        np.testing.assert_allclose(t.VA, [6.5, 0.09, 5.42, 11.0, 0, 6.92])
        np.testing.assert_allclose(t.output, [10.15, 0.7, 8.02, 15.85, 0, 10.92])
        assert t.merchandise_mask.tolist() == [True, False, False]
        assert t.metadata["nowcast"] is True and t.metadata["sector_names"][2] == "Other land transport"
        assert [s.archive_member for s in t.sources] == ["Z.txt", "Y.txt", "x.txt", "industries.txt", "satellite/F.txt",
                                                          "satellite/F_Y.txt", "finaldemands.txt", "metadata.json"]
        # Without its archive a directory is hashed, not authenticated.
        assert {s.status for s in t.sources} == {"unknown_edition"} and t.metadata["source_authenticated"] is False
        assert set(t.metadata["factor_components"]) == set(m.EXIOBASE_FACTOR_NAMES)
        rep = t.accounting_report()
        assert rep.row_column_max_absolute_gap < 1e-12 and rep.zero_output_cells == 1 and rep.negative_inventory_entries == 2
        assert rep.final_category_totals["VAL"] == pytest.approx(0.17)
        assert t.is_raw

    def test_zip_equals_directory_and_chunking_and_dense(self, tmp_path):
        archive = _write_exio_zip(tmp_path / "IOT_2019_ixi.zip")
        base = _read_exio_dir()
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            z = m.read_exiobase_native(archive)
        assert _maxdiff(z.Z, base.Z) == 0 and np.array_equal(z.F, base.F) and np.array_equal(z.VA, base.VA)
        assert z.sources[0].status == "unknown_edition" and z.sources[1].archive_member == "IOT_2019_ixi/Z.txt"
        assert all(s.status == "verified" and s.crc32 is not None for s in z.sources[1:])
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = m.read_exiobase_native(archive, expected_sha256=sha)
        assert v.sources[0].status == "verified"
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_exiobase_native(archive, expected_sha256="0" * 64)
        for chunk in (1, 2, 5, 100):
            c = _read_exio_dir(chunksize=chunk)
            assert _maxdiff(c.Z, base.Z) == 0
        dense = _read_exio_dir(sparse_matrix=False, chunksize=2)
        assert not dense.is_sparse and np.array_equal(dense.Z, EXIO_Z)
        # A ZIP without the year prefix is accepted when the members sit at the root.
        flat = _write_exio_zip(tmp_path / "flat.zip", prefix="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = m.read_exiobase_native(flat)
        assert _maxdiff(f.Z, base.Z) == 0
        with pytest.raises(FileNotFoundError):
            m.read_exiobase_native(tmp_path / "absent")

    def test_directory_authenticated_against_its_archive(self, tmp_path, monkeypatch):
        archive = _write_exio_zip(tmp_path / "IOT_2019_ixi.zip")
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = m.read_exiobase_native(EXIO_FIXTURE, archive=archive, expected_sha256=sha)
        assert t.sources[0].status == "verified" and t.sources[0].sha256 == sha and t.build_report.sha256 == sha
        members = t.sources[1:]
        assert [s.archive_member for s in members] == ["IOT_2019_ixi/" + x for x in (
            "Z.txt", "Y.txt", "x.txt", "industries.txt", "satellite/F.txt", "satellite/F_Y.txt", "finaldemands.txt",
            "metadata.json")]
        assert all(s.status == "verified" and s.crc32 is not None for s in members)
        assert members[0].path == str((EXIO_FIXTURE / "Z.txt").resolve()) and t.metadata["source_authenticated"] is True
        base = _read_exio_dir()
        assert _maxdiff(t.Z, base.Z) == 0 and np.array_equal(t.F, base.F) and np.array_equal(t.VA, base.VA)
        # An unregistered archive is read with a warning; its extracted members are still size- and CRC-checked.
        with pytest.warns(RuntimeWarning, match="unauthenticated"):
            u = m.read_exiobase_native(EXIO_FIXTURE, archive=archive)
        assert u.sources[0].status == "unknown_edition" and all(s.status == "verified" for s in u.sources[1:])
        assert u.metadata["source_authenticated"] is False
        with pytest.raises(m.MRIOIntegrityError, match="SHA-256 mismatch"):
            m.read_exiobase_native(EXIO_FIXTURE, archive=archive, expected_sha256="0" * 64)
        # One changed byte, or one added byte, in an extracted member is refused.
        core = tmp_path / "core"
        shutil.copytree(EXIO_FIXTURE, core)
        # Bytes, not text: text mode on Windows writes CRLF, which changes the
        # size and trips the size check before the CRC check under test.
        text = (core / "x.txt").read_bytes()
        (core / "x.txt").write_bytes(text.replace(b"10.15", b"10.16"))
        with pytest.raises(m.MRIOIntegrityError, match="CRC-32"):
            m.read_exiobase_native(core, archive=archive, expected_sha256=sha)
        (core / "x.txt").write_bytes(text.replace(b"10.15", b"10.150"))
        with pytest.raises(m.MRIOIntegrityError, match="bytes"):
            m.read_exiobase_native(core, archive=archive, expected_sha256=sha)
        # The archive digest cannot authenticate a directory on its own; archive= needs a directory.
        with pytest.raises(ValueError, match="pass archive"):
            m.read_exiobase_native(EXIO_FIXTURE, expected_sha256=sha)
        with pytest.raises(ValueError, match="extracted directory"):
            m.read_exiobase_native(archive, archive=archive)
        with pytest.raises(FileNotFoundError, match="archive not found"):
            m.read_exiobase_native(EXIO_FIXTURE, archive=tmp_path / "absent.zip")
        # The registered digest is bound to 2019.
        monkeypatch.setattr(m, "EXIOBASE_382_2019_IXI_SHA256", sha)  # test-only: treat this archive as the registered one
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = m.read_exiobase_native(EXIO_FIXTURE, archive=archive)
        assert r.sources[0].status == "verified" and r.metadata["source_authenticated"] is True
        with pytest.raises(m.MRIOIntegrityError, match="not the requested year 2020"):
            m.read_exiobase_native(archive, year=2020)

    @pytest.mark.filterwarnings("ignore:.*without its archive:RuntimeWarning")
    def test_rejections(self, tmp_path):
        def variant(edit):
            core = tmp_path / "core"
            if core.exists():
                shutil.rmtree(core)
            shutil.copytree(EXIO_FIXTURE, core)
            edit(core)
            return core

        def set_exports(core):
            p = core / "Y.txt"
            lines = p.read_text(encoding="utf-8").splitlines()
            parts = lines[3].split("\t")
            parts[2 + 6] = "0.5"  # AT exports column of the first cell
            lines[3] = "\t".join(parts)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="unallocated exports"):
            m.read_exiobase_native(variant(set_exports))

        def set_fd_factor(core):
            p = core / "satellite" / "F_Y.txt"
            lines = p.read_text(encoding="utf-8").splitlines()
            parts = lines[4].split("\t")  # "Other net taxes on production" row
            parts[1] = "0.1"
            lines[4] = "\t".join(parts)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="primary-factor payments"):
            m.read_exiobase_native(variant(set_fd_factor))

        def rename_industry(core):
            p = core / "industries.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Other land transport", "Other transport"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="ordering"):
            m.read_exiobase_native(variant(rename_industry))

        def rename_factor(core):
            p = core / "satellite" / "F.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Other net taxes on production", "Other taxes"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="factor labels"):
            m.read_exiobase_native(variant(rename_factor))

        def drop_member(core):
            (core / "x.txt").unlink()

        with pytest.raises(FileNotFoundError, match="x.txt"):
            m.read_exiobase_native(variant(drop_member))

        def rename_category(core):
            p = core / "finaldemands.txt"
            p.write_text(p.read_text(encoding="utf-8").replace("Changes in valuables", "Valuables"), encoding="utf-8")

        with pytest.raises(m.MRIOIntegrityError, match="finaldemands"):
            m.read_exiobase_native(variant(rename_category))


class TestOecdReader:
    def test_tiny_csv_with_monkeypatched_registries(self, tmp_path, monkeypatch):
        path, countries, sectors, Z, F6, VA, TLS, TFD6, Y = _tiny_oecd_csv(tmp_path)
        monkeypatch.setattr(trade_data, "OECD_77_COUNTRIES", countries)
        monkeypatch.setattr(trade_data, "OECD_45_SECTORS", sectors)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        with pytest.warns(RuntimeWarning, match="not a whitelisted"):
            t = m.read_oecd_native(path, 2019)
        assert t.fd_codes == ("C", "G", "X", "V") and t.country_codes == countries and t.sector_codes == sectors
        np.testing.assert_allclose(t.Z, Z)
        np.testing.assert_allclose(t.final_use("C"), F6[:, :, :3].sum(2))
        np.testing.assert_allclose(t.final_use("G"), F6[:, :, 3])
        np.testing.assert_allclose(t.final_use("V"), F6[:, :, 4])
        np.testing.assert_allclose(t.final_use("X"), F6[:, :, 5])
        np.testing.assert_allclose(t.VA, VA)
        np.testing.assert_allclose(t.TLS, TLS)
        np.testing.assert_allclose(t.output, Y)
        np.testing.assert_allclose(t.TFD, np.column_stack([TFD6[:, :3].sum(1), TFD6[:, 3], TFD6[:, 5], TFD6[:, 4]]))
        assert t.merchandise_mask.tolist() == [True, False] and t.labor_compensation is None
        assert t.sources[0].status == "unknown_edition" and t.metadata["whitelist_status"] == "unknown_edition"
        assert t.metadata["production_taxes_observed"] is False and t.reference_country == "BBB"
        assert t.metadata["final_use_components"]["HFCE"].shape == (4, 2)
        assert not t.is_sparse and t.build_report.md5 == md5
        with pytest.raises(m.MRIOIntegrityError, match="refused"):
            m.read_oecd_native(path, 2019, corrupted={md5: "a test-only corrupted registry entry"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w = m.read_oecd_native(path, 2019, whitelist={2019: md5})
            u = m.read_oecd_native(path, 2019, check=False)
        assert w.sources[0].status == "whitelisted" and u.sources[0].status == "hashed"
        archive = tmp_path / "2016-2020_SML.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(path, "2019_SML.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            zt = m.read_oecd_native(archive, 2019, whitelist={2019: md5})
        assert np.array_equal(zt.F, t.F) and zt.sources[0].archive_member == "2019_SML.csv"
        raw = trade_data.load_oecd_icio_granular(2019, file_path=path)
        assert raw.n_sectors == 2
        # to_raw("cix") reproduces the C, I, Cx condensation used by load_oecd_icio_granular.
        from puremacro.trade._oecd_icio import condense_final_demand, read_native

        cond = condense_final_demand(read_native(path, 2019))
        np.testing.assert_allclose(t.to_raw("cix").final_demand_matrix, cond.final_demand_matrix)
        np.testing.assert_allclose(t.to_raw("cix").taxes_less_subsidies_fd, cond.taxes_less_subsidies_fd)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class TestRegularize:
    def test_figaro_fixture_phantom_and_residual(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, rep = m.regularize_table(t)
        assert rep.passed and rep.phantoms == ("ROW_C10T12",) and rep.floored == () and rep.n_zero_output == 1
        assert rep.skipped_phantoms == () and rep.failed_gates == ()
        assert reg.F[3, 1, 0] == 1e-6 and reg.VA[3] == 1e-6 and reg.output[3] == pytest.approx(1e-6)
        np.testing.assert_allclose(reg.TLS, [3, 5.5, 2, 0], atol=1e-15)
        np.testing.assert_allclose(reg.output[:3], [95.4, 105.4995, 92.0])
        assert rep.dtls_accounting_abs < 1e-13 and rep.dtls_floor_abs == 0 and rep.world_abs_tls == pytest.approx(10.5)
        assert rep.world_gdp == pytest.approx(73.4 + 43.9995 + 70 + 1e-6 + 10.5 + (4.3 + 0.5 + 0.05 + 5.4 + 0.6 - 0.02))
        assert rep.negative_fd_cells == {"C": 0, "G": 0, "X": 0, "V": 2}
        assert rep.rounding_adjustments == t.metadata["rounding_adjustments"]
        assert len(reg.transformations) == 2 and reg.metadata["regularized"] is True and reg.build_report is rep
        assert all(g["passed"] for g in rep.gates.values()) and set(rep.gates) == {
            "nonnegative_output", "row_identity_median_rel", "zero_output_cells_empty", "dtls_world_share",
            "dtls_cell_share_of_output", "trade_balance_world_sum"}
        # The regularized accounts are exactly balanced.
        assert reg.accounting_report().row_column_max_absolute_gap < 1e-13
        # Regularizing twice is idempotent.
        again, rep2 = m.regularize_table(reg)
        np.testing.assert_allclose(again.TLS, reg.TLS, atol=1e-15)
        assert rep2.phantoms == ()

    def test_gates_strict_and_report_flags(self):
        t = toy_table()
        Z = t.Z.toarray()
        Z[0] = 0
        Z[:, 0] = 0
        out = np.array(t.output)
        out[0] = 0
        F = np.array(t.F)
        F[0] = 0
        junk = toy_table(Z=Z, output=out, F=F, strict=False)  # VA and TLS of the dead cell still non-zero
        with pytest.raises(m.MRIOIntegrityError, match="zero-output cell has non-zero"):
            m.regularize_table(junk)
        _, rep = m.regularize_table(junk, strict=False)
        assert not rep.passed and not rep.gates["zero_output_cells_empty"]["passed"]
        neg = toy_table(output=np.array([-1., 1, 1, 1]))
        with pytest.raises(m.MRIOIntegrityError, match="negative gross output"):
            m.regularize_table(neg)
        skewed = toy_table(output=np.array(t.output) * 1.5)
        with pytest.raises(m.MRIOIntegrityError, match="median relative row identity"):
            m.regularize_table(skewed)
        _, rep = m.regularize_table(skewed, strict=False)
        assert rep.gates["row_identity_median_rel"]["value"] == pytest.approx(1 / 3)
        # A large residual-tax change trips the per-cell gate.
        big = toy_table(TLS=np.array(t.TLS) + np.array([2., 0, 0, 0]), strict=False)
        with pytest.raises(m.MRIOIntegrityError, match="moves cell A_goods"):
            m.regularize_table(big)
        _, rep = m.regularize_table(big, strict=False)
        assert not rep.gates["dtls_cell_share_of_output"]["passed"]
        assert rep.gates["dtls_cell_share_of_output"]["value"] == pytest.approx(2 / max(t.output[0], 1))
        with pytest.raises(ValueError, match="inactive"):
            m.regularize_table(t, inactive="drop")
        with pytest.raises(TypeError):
            m.regularize_table(np.eye(2))

    def test_failed_gates_are_never_marked_regularized(self):
        t = toy_table()
        big = toy_table(TLS=np.array(t.TLS) + np.array([2., 0, 0, 0]), strict=False)
        reg, rep = m.regularize_table(big, strict=False)
        assert not rep.passed and "dtls_cell_share_of_output" in rep.failed_gates
        g = rep.gates["dtls_cell_share_of_output"]
        assert g["cells"] == ("A_goods",) and g["n_cells"] == 1 and g["worst_cell"] == "A_goods"
        assert reg.metadata["regularized"] is False and reg.metadata["regularization_passed"] is False
        assert reg.metadata["failed_gates"] == rep.failed_gates
        assert "FAILED gates" in reg.transformations[-1] and "failed:" in rep.summary()
        assert rep.to_dataframe().loc["failed_gates", "Value"] == ", ".join(rep.failed_gates)
        assert json.loads(json.dumps(rep.to_dict()))["gates"]["dtls_cell_share_of_output"]["cells"] == ["A_goods"]
        # The failure travels through aggregation and into the calibration bridge, loudly.
        with pytest.warns(RuntimeWarning, match="gates failed"):
            agg = m.aggregate_mrio(reg, m.Concordance.identity(reg.sector_codes))
        assert agg.metadata["regularized"] is False and agg.build_report.passed is False
        assert agg.metadata["aggregation"]["input_gates_passed"] is False
        with pytest.warns(RuntimeWarning, match="failed regularize_table gates"):
            m.to_calibration_matrix(agg)
        # A table that passes is marked and moves on silently.
        ok, ok_rep = m.regularize_table(t)
        assert ok.metadata["regularized"] is True and ok.metadata["failed_gates"] == () and ok_rep.failed_gates == ()
        assert ok_rep.gates["dtls_cell_share_of_output"]["n_cells"] == 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agg_ok = m.aggregate_mrio(ok, m.Concordance.identity(ok.sector_codes))
            m.to_calibration_matrix(agg_ok)
        assert agg_ok.metadata["aggregation"]["input_gates_passed"] is True

    def test_phantoms_skip_dirty_and_negative_cells_under_strict_false(self):
        t = toy_table()
        Z = t.Z.toarray()
        Z[0] = 0
        Z[:, 0] = 0
        out = np.array(t.output)
        out[0] = 0
        F = np.array(t.F)
        F[0] = 0
        junk = toy_table(Z=Z, output=out, F=F, strict=False)  # VA and TLS of the dead cell still non-zero
        with pytest.raises(m.MRIOIntegrityError, match="non-zero entries: A_goods"):
            m.regularize_table(junk)
        reg, rep = m.regularize_table(junk, strict=False)
        assert rep.phantoms == () and rep.skipped_phantoms == ("A_goods",)
        assert reg.VA[0] == junk.VA[0] and reg.F[0].sum() == 0  # the stray value added is kept, not overwritten
        assert rep.gates["zero_output_cells_empty"]["cells"] == ("A_goods",)
        assert "left unchanged" in reg.transformations[-1] and reg.metadata["regularized"] is False
        neg = toy_table(output=np.array([-1., 1, 1, 1]))
        with pytest.raises(m.MRIOIntegrityError, match="negative gross output at A_goods"):
            m.regularize_table(neg)
        reg_n, rep_n = m.regularize_table(neg, strict=False)
        assert rep_n.skipped_phantoms == ("A_goods",) and rep_n.phantoms == () and reg_n.VA[0] == neg.VA[0]
        assert rep_n.gates["nonnegative_output"]["cells"] == ("A_goods",)

    def test_mask_mode_keeps_empty_cells(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, rep = m.regularize_table(t, inactive="mask")
        assert rep.inactive == "mask" and rep.phantoms == ("ROW_C10T12",)
        assert reg.output[3] == 0 and reg.VA[3] == 0 and reg.F[3].sum() == 0 and reg.TLS[3] == 0

    def test_floor_only_where_va_nonpositive_and_dtls_bookkeeping(self):
        t = random_table(11, 2, 3)
        VA = np.array(t.VA)
        VA[2] = -0.5
        VA[4] = 0.0
        reg, rep = m.regularize_table(replace(t, VA=VA))
        labels = t.cell_labels
        assert [f[0] for f in rep.floored] == [labels[2], labels[4]]
        assert rep.floored[0][1] == -0.5 and rep.floored[0][2] == pytest.approx(1e-3 * t.output[2])
        assert reg.VA[2] == pytest.approx(1e-3 * t.output[2]) and reg.VA[4] == pytest.approx(1e-3 * t.output[4])
        untouched = [i for i in range(6) if i not in (2, 4)]
        np.testing.assert_allclose(reg.VA[untouched], t.VA[untouched])
        # Residual TLS absorbs the floor at the floored cells only; the rest is unchanged.
        expected = t.output - t.Z.sum(0) - reg.VA
        np.testing.assert_allclose(reg.TLS, expected, atol=1e-13)
        assert rep.dtls_floor_abs == pytest.approx(abs(expected[2] - t.TLS[2]) + abs(expected[4] - t.TLS[4]))
        assert rep.dtls_accounting_abs < 1e-12
        assert reg.accounting_report().row_column_max_absolute_gap < 1e-12

    def test_custom_gate_values_and_spectral_option(self):
        t = random_table(5, 3, 3)
        reg, rep = m.regularize_table(t, spectral=True, phantom=1e-4, va_floor_rel=0.01)
        assert rep.spectral_radius is not None and 0 < rep.spectral_radius[0] < 1
        lo, hi = rep.spectral_radius[1], rep.spectral_radius[2]
        assert lo <= rep.spectral_radius[0] + 1e-9 and rep.spectral_radius[0] <= hi + 1e-9
        assert "spectral_radius" in rep.to_dataframe().index
        with pytest.raises(ValueError, match="nonnegative"):
            m.regularize_table(t, phantom=-1)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        reg_sp, rep_sp = m.regularize_table(sp, spectral=True)
        assert rep_sp.spectral_radius is None and reg_sp.is_sparse
        np.testing.assert_allclose(reg_sp.TLS, reg.TLS, atol=1e-14)

    def test_commutes_with_aggregation_when_no_floors_apply(self):
        t = random_45(3)
        reg, _ = m.regularize_table(t)
        ag_then_reg, _ = m.regularize_table(m.aggregate_mrio(t, "agregar"))
        reg_then_ag = m.aggregate_mrio(reg, "agregar")
        assert _maxdiff(ag_then_reg.TLS, reg_then_ag.TLS) < 1e-12
        assert _maxdiff(ag_then_reg.output, reg_then_ag.output) < 1e-12
        assert _maxdiff(ag_then_reg.Z, reg_then_ag.Z) < 1e-12


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _check_conservation(fine, coarse, conc):
    n, k = fine.n_countries, fine.n_categories
    assert np.isclose(np.asarray(coarse.Z.sum()), np.asarray(fine.Z.sum()), rtol=1e-12)
    np.testing.assert_allclose(coarse.F.sum(axis=0), fine.F.sum(axis=0), rtol=1e-12, atol=1e-13)
    assert np.isclose(coarse.VA.sum(), fine.VA.sum(), rtol=1e-12) and np.isclose(coarse.TLS.sum(), fine.TLS.sum(), rtol=1e-12)
    np.testing.assert_array_equal(coarse.TFD, fine.TFD)
    g = conc.n_coarse
    np.testing.assert_allclose(coarse.output.reshape(n, g).sum(1), fine.output.reshape(n, -1).sum(1), rtol=1e-12)
    np.testing.assert_allclose(coarse.VA.reshape(n, g).sum(1), fine.VA.reshape(n, -1).sum(1), rtol=1e-12)
    np.testing.assert_allclose(coarse.bilateral_flows(), fine.bilateral_flows(), rtol=1e-12)


class TestAggregate:
    def test_conservation_on_random_fixture_and_oecd_fixture(self):
        t = random_table(1, 3, 4)
        conc = m.Concordance.build("pairs", t.sector_codes, (1, 1, 2, 2), ("P", "Q"))
        agg = m.aggregate_mrio(t, conc)
        _check_conservation(t, agg, conc)
        assert agg.metadata["aggregation"]["aggregated_identity_rel"] <= 1e-8
        assert agg.sector_codes == ("P", "Q") and agg.transformations[-1].startswith("aggregate_mrio")
        assert agg.metadata["aggregation"]["fine_balanced"] is True
        from tools.reference_validation.validate_oecd import load_fixture

        oecd = m.MRIOTable.from_raw(load_fixture(), metadata={"dataset": "oecd_icio_3x3"})
        reg, _ = m.regularize_table(oecd)
        conc2 = m.Concordance.from_mapping("two", {"RESOURCES": "GOODS", "MANUFACTURING": "GOODS", "SERVICES": "SERVICES"})
        agg2 = m.aggregate_mrio(reg, conc2)
        _check_conservation(reg, agg2, conc2)
        assert agg2.metadata["aggregation"]["aggregated_identity_rel"] <= 1e-8
        # Grouping ids or a mapping give the same table as the Concordance object.
        by_ids = m.aggregate_mrio(reg, (1, 1, 2), coarse_codes=("GOODS", "SERVICES"))
        by_map = m.aggregate_mrio(reg, {"RESOURCES": "GOODS", "MANUFACTURING": "GOODS", "SERVICES": "SERVICES"})
        for other in (by_ids, by_map):
            assert _maxdiff(other.Z, agg2.Z) == 0 and np.array_equal(other.F, agg2.F)

    def test_identity_concordance_is_a_noop(self):
        t = toy_table()
        ident = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes))
        assert _maxdiff(ident.Z, t.Z) == 0 and np.array_equal(ident.F, t.F) and np.array_equal(ident.VA, t.VA)
        assert ident.is_sparse and ident.merchandise_mask.tolist() == [True, False]
        np.testing.assert_array_equal(ident.labor_compensation, t.labor_compensation)

    def test_composition_and_explicit_kron(self):
        t = random_45(2)
        agregar = m.CONCORDANCES["agregar"]
        first = m.aggregate_mrio(t, agregar)
        outer = m.Concordance.from_mapping("two", {c: ("GOODS" if c in ("AGRI_MIN_FOOD", "MANUF_EXFOOD") else "SERV")
                                                  for c in m.AGG11_SECTOR_CODES})
        two_step = m.aggregate_mrio(first, outer)
        one_step = m.aggregate_mrio(t, agregar.compose(outer))
        assert _maxdiff(two_step.Z, one_step.Z) < 1e-12 and _maxdiff(two_step.F, one_step.F) < 1e-12
        assert _maxdiff(two_step.VA, one_step.VA) < 1e-12 and two_step.sector_codes == one_step.sector_codes
        P = np.kron(np.eye(3), agregar.matrix.T)
        assert _maxdiff(P @ t.Z @ P.T, first.Z) < 1e-12
        np.testing.assert_allclose(P @ t.VA, first.VA, rtol=1e-13)
        sp = replace(t, Z=sparse.csr_matrix(t.Z))
        agg_sp = m.aggregate_mrio(sp, agregar)
        assert agg_sp.is_sparse and _maxdiff(agg_sp.Z, first.Z) < 1e-12

    def test_region_concordance(self):
        t = random_table(4, 4, 3, countries=("USA", "CHN", "DEU", "ROW"))
        regions = m.Concordance.from_mapping("blocs", {"USA": "USA", "CHN": "REST", "DEU": "REST", "ROW": "REST"})
        agg = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes), region_concordance=regions)
        assert agg.country_codes == ("USA", "REST") and agg.n_cells == 6 and agg.reference_country == "REST"
        assert np.isclose(np.asarray(agg.Z.sum()), np.asarray(t.Z.sum()), rtol=1e-12)
        np.testing.assert_allclose(agg.TFD, np.vstack([t.TFD[0], t.TFD[1:].sum(0)]), rtol=1e-12)
        np.testing.assert_allclose(agg.F.sum(axis=(0, 2)), np.r_[t.F[:, 0].sum(), t.F[:, 1:].sum()], rtol=1e-12)
        B = t.bilateral_flows()
        np.testing.assert_allclose(agg.bilateral_flows()[0, 1], B[0, 1:].sum(), rtol=1e-12)
        by_map = m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes),
                                  region_concordance={"USA": "USA", "CHN": "REST", "DEU": "REST", "ROW": "REST"})
        assert _maxdiff(by_map.Z, agg.Z) == 0
        with pytest.raises(ValueError, match="other country codes"):
            m.aggregate_mrio(t, m.Concordance.identity(t.sector_codes),
                             region_concordance=m.Concordance.identity(("X", "Y", "Z", "W")))

    def test_goods_consistency_45_to_11(self):
        t = random_45(5)
        ag = m.aggregate_mrio(t, "agregar")
        isic = m.aggregate_mrio(t, "isic_section")
        assert ag.merchandise_mask.tolist() == [True, True] + [False] * 9
        assert isic.merchandise_mask.tolist() == [True, True, True] + [False] * 8
        fine_goods = (t.output.reshape(3, 45) * t.merchandise_mask).sum(1)
        np.testing.assert_allclose((ag.output.reshape(3, 11) * ag.merchandise_mask).sum(1), fine_goods, rtol=1e-12)
        np.testing.assert_allclose((isic.output.reshape(3, 11) * isic.merchandise_mask).sum(1), fine_goods, rtol=1e-12)
        assert ag.metadata["aggregation"]["mixed_goods_groups"] == () and isic.metadata["aggregation"]["mixed_goods_groups"] == ()
        with pytest.raises(ValueError, match="other sector codes"):
            m.aggregate_mrio(random_table(1, 3, 4), "agregar")
        with pytest.raises(ValueError, match="unknown concordance"):
            m.aggregate_mrio(t, "nace")

    def test_mixed_goods_groups_recorded_and_raw_tables_allowed(self):
        t = random_45(6)
        mixed = m.Concordance.build("mixed", RAW_45_SECTOR_CODES, [1] * 23 + [2] * 22, ("GOODS_AND_D", "REST"))
        agg = m.aggregate_mrio(t, mixed)
        assert agg.merchandise_mask.tolist() == [False, False]
        assert agg.metadata["aggregation"]["mixed_goods_groups"] == ("GOODS_AND_D",)
        out = np.array(t.output) * 1.01  # an unbalanced (raw) table is aggregated and recorded, not refused
        raw = replace(t, output=out)
        agg_raw = m.aggregate_mrio(raw, "agregar")
        assert agg_raw.metadata["aggregation"]["fine_balanced"] is True  # row = column identity does not involve output
        skew = replace(t, TLS=np.array(t.TLS) + 1.0)
        agg_skew = m.aggregate_mrio(skew, "agregar")
        assert agg_skew.metadata["aggregation"]["fine_balanced"] is False
        assert agg_skew.metadata["aggregation"]["aggregated_identity_rel"] > 1e-8


# ---------------------------------------------------------------------------
# Coarse tariff rules
# ---------------------------------------------------------------------------


class TestCoarseTariffs:
    def test_uniform_rates_within_group_reproduce_delta(self):
        t = random_45(8)
        conc = m.CONCORDANCES["agregar"]
        rng = np.random.default_rng(0)
        group_rates = rng.uniform(0, 0.5, 11)
        delta = np.tile(group_rates[np.asarray(conc.groups) - 1], (3, 1))
        delta[0] = 0  # USA row
        for rule in ("output", "bilateral"):
            res = m.coarse_tariff_rates(delta, t, conc, rule)
            assert res.importer == "USA" and res.rule == rule and res.concordance == "agregar"
            np.testing.assert_allclose(res.rates[1:], np.tile(group_rates, (2, 1)), atol=1e-14)
            assert not res.rates[0].any()
            assert res.to_dataframe().shape == (3, 11) and res.to_markdown() and res.to_latex() and res.to_typst()
        res = m.coarse_tariff_rates(delta, t, conc, "bilateral")
        own = slice(0, 11)
        for o in (1, 2):
            block = res.tau[o * 11:(o + 1) * 11, own]
            for g in range(11):
                positive = block[g][block[g] != 1.0]
                np.testing.assert_allclose(positive, 1 + group_rates[g], atol=1e-14)
        assert "benchmark revenue" in res.summary()

    def test_zero_weights_importer_row_and_explicit_weights(self):
        t = random_45(9)
        delta = np.full((3, 45), 0.2)
        weights = np.ones((3, 45))
        weights[1, :6] = 0.0  # CHN has no imports in the first group
        res = m.coarse_tariff_rates(delta, weights, "agregar")
        assert res.importer is None and res.rates[1, 0] == 0.0 and res.rates[1, 1] == pytest.approx(0.2)
        assert res.tau is None and res.benchmark_revenue_fine is None and res.coarse_codes == m.AGG11_SECTOR_CODES
        by_ids = m.coarse_tariff_rates(delta, weights, m.AGREGAR_11_CONCORDANCE, coarse_codes=m.AGG11_SECTOR_CODES)
        np.testing.assert_array_equal(by_ids.rates, res.rates)
        with pytest.raises(ValueError, match="defined on"):
            m.coarse_tariff_rates(delta[:, :4], weights[:, :4], "agregar")
        with pytest.raises(ValueError, match="needs the fine MRIOTable"):
            m.coarse_tariff_rates(delta, weights, "agregar", "bilateral")
        with pytest.raises(ValueError, match="importer needs"):
            m.coarse_tariff_rates(delta, weights, "agregar", importer="USA")
        res_t = m.coarse_tariff_rates(delta, t, "agregar")
        assert not res_t.rates[0].any() and not res_t.weights[0].any()
        np.testing.assert_allclose(res_t.rates[1:], 0.2)
        with pytest.raises(ValueError, match="never tariffed"):
            m.coarse_tariff_rates(delta, t, "agregar", fd_tariffed=("C", "X"))
        with pytest.raises(ValueError, match="rule"):
            m.coarse_tariff_rates(delta, t, "agregar", "mean")
        for bad in (-1.5, -0.5, -1e-12):
            with pytest.raises(ValueError, match="nonnegative"):
                m.coarse_tariff_rates(np.full((3, 45), bad), t, "agregar")
        with pytest.raises(ValueError, match="shape"):
            m.coarse_tariff_rates(np.zeros((2, 45)), t, "agregar")

    def test_convexity_with_nonnegative_weights(self):
        t = random_45(10)
        conc = m.CONCORDANCES["agregar"]
        rng = np.random.default_rng(3)
        delta = rng.uniform(0, 1, (3, 45))
        delta[0] = 0
        # With clipping every weight is nonnegative and each coarse rate is a convex combination.
        res = m.coarse_tariff_rates(delta, t, conc, negative_weights="clip")
        assert res.weights.min() >= 0
        C = conc.matrix
        for o in (1, 2):
            for g in range(11):
                members = np.flatnonzero(C[:, g])
                assert delta[o, members].min() - 1e-12 <= res.rates[o, g] <= delta[o, members].max() + 1e-12
        # An independent weighted average.
        W = res.weights
        expected = np.divide((W * delta) @ C, W @ C, out=np.zeros((3, 11)), where=(W @ C) > 0)
        np.testing.assert_allclose(res.rates, expected, atol=1e-15)
        keep = m.coarse_tariff_rates(delta, t, conc, negative_weights="keep")
        assert keep.metadata["negative_weights"] == "keep"
        # Keeping the signed inventories lowers the weights exactly by the drawdowns.
        drawdown = np.minimum(t.F[:, 0, t.fd_codes.index("V")], 0.0).reshape(3, 45)
        drawdown[0] = 0.0
        assert drawdown.min() < 0
        np.testing.assert_allclose(res.weights - keep.weights, -drawdown, atol=1e-15)

    def test_bilateral_preserves_benchmark_revenue_group_by_group(self):
        t = random_45(12)
        conc = m.CONCORDANCES["agregar"]
        rng = np.random.default_rng(5)
        delta = rng.uniform(0, 0.8, (3, 45))
        delta[0] = 0
        # With clipped inventories every block has nonnegative purchases and revenue is preserved exactly.
        res = m.coarse_tariff_rates(delta, t, conc, "bilateral", negative_weights="clip")
        assert res.benchmark_revenue_coarse == pytest.approx(res.benchmark_revenue_fine, rel=1e-12)
        usa = 0
        cols = slice(usa * 45, (usa + 1) * 45)
        Zi = t.Z[:, cols]
        C = conc.matrix
        R = np.kron(np.eye(3), C.T)
        ZG = R @ Zi @ C
        fine_rev = R @ (Zi * delta.reshape(-1)[:, None]) @ C
        coarse_rev = ZG * (res.tau[:, 0:11] - 1.0)
        np.testing.assert_allclose(coarse_rev, fine_rev, atol=1e-12)
        for code in ("C", "G", "V"):
            k = t.fd_codes.index(code)
            f = np.maximum(t.F[:, usa, k], 0.0)
            fine_f = R @ (f * delta.reshape(-1))
            coarse_f = (R @ f) * (res.tau_fd[:, usa, k] - 1.0)
            np.testing.assert_allclose(coarse_f, fine_f, atol=1e-12)
        assert np.all(res.tau_fd[:, :, t.fd_codes.index("X")] == 1.0)
        # With signed inventories kept, a block whose purchases are not positive keeps a unit
        # multiplier and drops its fine duty (the IO tot > 0 guard); the gap is exactly that duty.
        keep = m.coarse_tariff_rates(delta, t, conc, "bilateral", negative_weights="keep")
        kV = t.fd_codes.index("V")
        v = t.F[:, usa, kV]
        VG, VG_r = R @ v, R @ (v * delta.reshape(-1))
        assert (VG <= 0).any()
        dropped = VG_r[VG <= 0].sum()
        assert keep.benchmark_revenue_fine - keep.benchmark_revenue_coarse == pytest.approx(dropped, abs=1e-12)
        assert np.all(keep.tau_fd[VG <= 0, usa, kV] == 1.0)
        np.testing.assert_allclose(keep.tau[:, 0:11], res.tau[:, 0:11], atol=1e-12)  # Z blocks are unaffected
        assert np.all(res.tau[:, 11:] == 1.0) and np.all(res.tau[0:11, :] == 1.0)
        assert np.all(res.tau_fd[:, 1:, :] == 1.0) and np.all(res.tau_fd[0:11] == 1.0)
        # The output rule preserves each origin-group total (its weights are exactly the tariffed
        # purchases) but not the allocation of that revenue across buyer groups.
        out = m.coarse_tariff_rates(delta, t, conc, "output")
        assert out.benchmark_revenue_fine == pytest.approx(keep.benchmark_revenue_fine)
        assert out.benchmark_revenue_coarse == pytest.approx(out.benchmark_revenue_fine, rel=1e-12)
        by_buyer_output = ZG * out.rates.reshape(-1)[:, None]
        assert not np.allclose(by_buyer_output, fine_rev, rtol=1e-6, atol=1e-9)
        untariffed_v = m.coarse_tariff_rates(delta, t, conc, "bilateral", fd_tariffed=("C", "G"))
        assert np.all(untariffed_v.tau_fd[:, :, t.fd_codes.index("V")] == 1.0)

    def test_valuables_are_tariffed_with_inventories_by_default(self):
        t = toy_table()
        conc = m.Concordance.identity(t.sector_codes)
        rates = np.array([[0.1, 0.2], [0.0, 0.0]])
        kVAL = t.fd_codes.index("VAL")
        res = m.coarse_tariff_rates(rates, t, conc, "bilateral", importer="B")
        assert res.fd_tariffed == ("C", "G", "V", "VAL") and res.metadata["fd_tariffed_present"] == ("C", "G", "V", "VAL")
        assert res.tau_fd[1, 1, kVAL] == pytest.approx(1.2)  # A_services valuables bought by B
        old = m.coarse_tariff_rates(rates, t, conc, "bilateral", importer="B", fd_tariffed=("C", "G", "V"))
        assert np.all(old.tau_fd[:, :, kVAL] == 1.0)
        assert res.benchmark_revenue_fine - old.benchmark_revenue_fine == pytest.approx(0.2 * 0.01)
        out = m.coarse_tariff_rates(rates, t, conc, "output", importer="B")
        out_old = m.coarse_tariff_rates(rates, t, conc, "output", importer="B", fd_tariffed=("C", "G", "V"))
        np.testing.assert_allclose(out.weights[0] - out_old.weights[0], t.final_use("VAL")[:2, 1])
        # A table without VAL (the OECD layout) is unaffected by the default.
        assert m.coarse_tariff_rates(rates, random_table(2, 2, 2, countries=("A", "B")), (1, 2), importer="B").metadata[
            "fd_tariffed_present"] == ("C", "G", "V")

    def test_importer_other_than_usa(self):
        t = random_table(13, 3, 45, sectors=RAW_45_SECTOR_CODES, z_scale=0.02, negative_inventories=False)
        assert t.final_use("V").min() >= 0
        delta = np.random.default_rng(2).uniform(0, 0.3, (3, 45))
        res = m.coarse_tariff_rates(delta, t, "agregar", "bilateral", importer="CHN")
        assert res.importer == "CHN" and not res.rates[1].any() and res.rates[[0, 2]].any()
        assert np.all(res.tau[:, :11] == 1.0) and np.all(res.tau[:, 22:] == 1.0) and np.all(res.tau[11:22, :] == 1.0)
        assert res.benchmark_revenue_coarse == pytest.approx(res.benchmark_revenue_fine, rel=1e-12)
        no_usa = random_table(1, 2, 4, countries=("DEU", "FRA"))
        with pytest.raises(ValueError, match="importer is required"):
            m.coarse_tariff_rates(np.zeros((2, 4)), no_usa, (1, 1, 2, 2))
        ok = m.coarse_tariff_rates(np.full((2, 4), 0.1), no_usa, (1, 1, 2, 2), importer="FRA")
        np.testing.assert_allclose(ok.rates[0], 0.1)


# ---------------------------------------------------------------------------
# Bridges to the legacy calibration
# ---------------------------------------------------------------------------


class TestCalibrationBridge:
    def test_oecd_fixture_calibrates_and_solves_at_benchmark(self):
        from puremacro.trade import solve_trade_equilibrium
        from tools.reference_validation.validate_oecd import load_fixture

        oecd = m.MRIOTable.from_raw(load_fixture(), metadata={"dataset": "oecd_icio_3x3"})
        with pytest.warns(RuntimeWarning, match="regularize_table"):
            m.to_calibration_matrix(oecd)
        reg, rep = m.regularize_table(oecd)
        assert rep.passed and rep.phantoms == () and rep.floored == ()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            D = m.to_calibration_matrix(reg)
        assert D.shape == (12, 18)
        np.testing.assert_allclose(D[:9].sum(1), D[:, :9].sum(0), rtol=1e-13)
        calib = calibrate_trade_model(D, ns=3, nc=3, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"]
        np.testing.assert_allclose(D[10, :9] / (D[10, :9] + D[11, :9]), 2 / 3)
        base = solve_trade_equilibrium(calib, tariff_revenue_mode="schedule", accounting="consistent", tol=1e-5, max_iter=100)
        assert base.converged and np.max(np.abs(base.p_sol - 1)) < 1e-8
        np.testing.assert_allclose(base.y_sol, calib.ytot, rtol=1e-8)
        packaged = package_mrio_to_calibration_result(reg.to_raw(), regularize=False)
        assert verify_accounting_invariants(packaged)["valid"]
        np.testing.assert_allclose(packaged.ytot, calib.ytot, rtol=1e-12)
        raw = reg.to_raw("native")
        assert raw.fd_categories == list(reg.fd_codes) and raw.metadata["transformations"] == list(reg.transformations)
        assert raw.metadata["final_demand_mapping"]["C"] == ["C"]

    def test_figaro_factor_detail_and_production_taxes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = m.read_figaro_native(FIGARO_FIXTURE)
        reg, _ = m.regularize_table(t)
        D = m.to_calibration_matrix(reg)
        M = 4
        np.testing.assert_allclose(D[:M].sum(1), D[:, :M].sum(0), atol=1e-12)
        # Production taxes go to the TLS row; labour and capital split the factor value added by observed shares.
        np.testing.assert_allclose(D[M, :M], reg.TLS + reg.production_taxes)
        factor = reg.VA - reg.production_taxes
        lab, cap = reg.labor_compensation, reg.operating_surplus
        both = (lab > 0) & (cap > 0)
        share = np.where(both, lab / np.where(both, lab + cap, 1), 2 / 3)
        np.testing.assert_allclose(D[M + 1, :M], share * factor)
        np.testing.assert_allclose(D[M + 2, :M], (1 - share) * factor)
        assert D[M + 1, 3] == pytest.approx(2 / 3 * 1e-6)  # the phantom cell has no observed split
        # I merges G and V; Cx is the zero X column; final taxes condense the same way.
        F3 = D[:M, M:].reshape(M, 2, 3)
        np.testing.assert_allclose(F3[:, :, 0], reg.final_use("C"))
        np.testing.assert_allclose(F3[:, :, 1], reg.final_use("G") + reg.final_use("V"))
        assert not F3[:, :, 2].any()
        np.testing.assert_allclose(D[M, M:].reshape(2, 3), np.column_stack([reg.TFD[:, 0], reg.TFD[:, 1] + reg.TFD[:, 3], reg.TFD[:, 2]]))
        D2 = m.to_calibration_matrix(reg, production_taxes="to_factors")
        np.testing.assert_allclose(D2[M, :M], reg.TLS)
        np.testing.assert_allclose(D2[M + 1, :M] + D2[M + 2, :M], reg.VA)
        with pytest.raises(ValueError):
            m.to_calibration_matrix(reg, production_taxes="drop")
        with pytest.raises(ValueError):
            m.to_calibration_matrix(reg, labor_share=1.5)
        calib = calibrate_trade_model(D, ns=2, nc=2, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"]

    def test_zero_factor_and_large_production_tax_cells_still_calibrate(self):
        t = toy_table()
        lab, cap = np.array(t.labor_compensation), np.array(t.operating_surplus)
        prod = np.array(t.production_taxes)
        fva = t.VA - prod
        lab[1], cap[1] = fva[1], 0.0              # zero operating surplus
        prod[2] = t.VA[2] + 0.5                   # production taxes above value added
        lab[2], cap[2] = 0.3, -0.8
        lab[3], cap[3] = 0.0, fva[3]              # zero labour
        odd = toy_table(labor_compensation=lab, operating_surplus=cap, production_taxes=prod)
        assert odd.accounting_report().factor_decomposition_max_absolute_gap < 1e-12
        reg, rep = m.regularize_table(odd)
        assert rep.passed and rep.floored == ()
        with pytest.warns(RuntimeWarning, match=r"production taxes of 1 cell are .*\(B_goods\)"):
            D, det = m.to_calibration_matrix(reg, return_details=True)
        assert det["production_taxes_retained_cells"] == ("B_goods",)
        assert set(det["share_fallback_cells"]) == {"A_services", "B_goods", "B_services"}
        assert det["observed_share_cells"] == 1 and det["zero_factor_cells"] == ()
        M = 4
        assert (D[M + 1, :M] > 0).all() and (D[M + 2, :M] > 0).all()
        np.testing.assert_allclose(D[M, :M], reg.TLS + np.where(np.arange(M) == 2, 0.0, reg.production_taxes))
        np.testing.assert_allclose(D[M + 1, 2] + D[M + 2, 2], reg.VA[2])  # production taxes kept inside value added
        np.testing.assert_allclose(D[M + 1, 1] / (D[M + 1, 1] + D[M + 2, 1]), 2 / 3)
        np.testing.assert_allclose(D[M + 1, 0] / (D[M + 1, 0] + D[M + 2, 0]), 0.6)
        np.testing.assert_allclose(D[:M].sum(1), D[:, :M].sum(0), atol=1e-12)
        calib = calibrate_trade_model(D, ns=2, nc=2, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert verify_accounting_invariants(calib)["valid"] and np.all((calib.alpha > 0) & (calib.alpha < 1))
        D2, det2 = m.to_calibration_matrix(reg, production_taxes="to_factors", return_details=True)
        assert det2["production_taxes_retained_cells"] == ()
        np.testing.assert_allclose(D2[M + 1, :M] + D2[M + 2, :M], reg.VA)
        calibrate_trade_model(D2, ns=2, nc=2, nfd=3)

    def test_negative_factor_value_added_is_refused(self):
        t = toy_table()
        VA = np.array(t.VA)
        VA[1] = -0.5
        raw = toy_table(VA=VA, labor_compensation=None, operating_surplus=None, production_taxes=None)
        with pytest.raises(m.MRIOIntegrityError, match="negative factor value added.*A_services"):
            m.to_calibration_matrix(raw, check_balance=False)
        reg, rep = m.regularize_table(raw)
        assert rep.passed and [f[0] for f in rep.floored] == ["A_services"]
        D = m.to_calibration_matrix(reg)  # the value-added floor makes the cell calibratable
        calibrate_trade_model(D, ns=2, nc=2, nfd=3)

    def test_from_trade_calibration_bundled_negative_cells_and_round_trip(self):
        calib = calibrate_trade_model(load_icio_data())
        with pytest.raises(m.MRIOIntegrityError, match="to_inventory"):
            m.MRIOTable.from_trade_calibration(calib)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert t.fd_codes == ("C", "G", "X", "V") and t.n_countries == 77 and t.n_sectors == 11
        assert (t.final_use("V") < 0).sum() == 3 and t.final_use("G").min() == 0
        assert "3 negative investment cells moved into V" in t.transformations
        assert t.transformations[0] == "bridged from TradeCalibrationResult.data_calibra"
        assert t.merchandise_mask is None and "positional" in t.metadata["sector_label_note"]
        assert t.production_taxes is None and t.labor_compensation is not None
        np.testing.assert_allclose(t.labor_compensation / (t.labor_compensation + t.operating_surplus), 2 / 3)
        D = m.to_calibration_matrix(t)
        assert D.shape == calib.data_calibra.shape
        np.testing.assert_allclose(D, calib.data_calibra, rtol=1e-12, atol=1e-6)
        again = calibrate_trade_model(D)
        np.testing.assert_allclose(again.ytot, calib.ytot, rtol=1e-12)
        with pytest.raises(TypeError):
            m.MRIOTable.from_trade_calibration(D)
        with pytest.raises(ValueError, match="by position"):
            m.goods_mask(t.sector_codes)
        assert m.goods_mask(t.sector_codes, "agregar11").sum() == 2
        assert t.accounting_report().row_column_max_absolute_gap < 1e-4 * t.output.max()


class TestCalibrationBridgeFinalUseSemantics:
    """The Cx slot of a loader calibration is residents' purchases abroad only for OECD (DPABR)."""

    LOADERS = ("load_figaro", "load_wiod", "load_eora")

    @staticmethod
    def _calib(loader):
        return getattr(trade_data, loader)(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)

    @pytest.mark.parametrize("loader", LOADERS)
    def test_provider_inventories_are_never_bridged_as_purchases_abroad(self, loader):
        calib = self._calib(loader)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert "X" not in t.fd_codes and not t.final_use("X").any()
        M, n = calib.n_countries * calib.n_sectors, calib.n_countries
        F3 = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)
        np.testing.assert_array_equal(t.final_use("V"), F3[:, :, 2])
        np.testing.assert_array_equal(t.final_use("G"), F3[:, :, 1])
        assert t.metadata["final_use_bridge"]["cx_category"] == "V"
        assert t.metadata["final_use_bridge"]["final_use_mapping"] == calib.metadata["final_use_mapping"]

    def test_exiobase_export_column_is_refused_unless_named(self):
        calib = self._calib("load_exiobase")
        with pytest.raises(ValueError, match=r"INVNT', 'VALUABLES', 'EXPORT'.*cx_category"):
            m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory", cx_category="V")
        assert "X" not in t.fd_codes and t.metadata["final_use_bridge"]["rule"] == "explicit cx_category"
        with pytest.raises(ValueError, match="cx_category must be one of"):
            m.MRIOTable.from_trade_calibration(calib, cx_category="G")

    def test_oecd_mapping_and_the_bundled_layout_bridge_cx_as_purchases_abroad(self):
        calib = self._calib("load_oecd_icio_granular")
        assert calib.metadata["final_use_mapping"]["Cx"] == ["DPABR"]
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        unmapped = replace(calib, metadata={k: v for k, v in calib.metadata.items() if k != "final_use_mapping"})
        legacy = m.MRIOTable.from_trade_calibration(unmapped, negative_investment="to_inventory")
        assert t.fd_codes == legacy.fd_codes and "X" in t.fd_codes
        for name in ("F", "VA", "TLS", "TFD"):
            np.testing.assert_array_equal(getattr(t, name), getattr(legacy, name))
        np.testing.assert_array_equal(t.Z_dense, legacy.Z_dense)
        assert t.final_use("X").sum() > 0

    def test_unrecognised_mapping_is_refused_with_the_mapping_named(self):
        calib = self._calib("load_figaro")
        odd = replace(calib, metadata={**calib.metadata, "final_use_mapping": {"C": ["C"], "I": ["I"], "Cx": ["FOO"]}})
        with pytest.raises(ValueError, match=r"\['FOO'\].*fd_mapping"):
            m.MRIOTable.from_trade_calibration(odd, negative_investment="to_inventory")

    @staticmethod
    def _with_cell(calib, column, value, *, row=0, dest=1):
        """Set one final-use cell of ``data_calibra`` (column 0=C, 1=I, 2=Cx), keeping the row total."""
        n, s = calib.n_countries, calib.n_sectors
        M = n * s
        D = np.array(calib.data_calibra, dtype=float, copy=True)
        col, c_col = M + 3 * dest + column, M + 3 * dest
        D[row, c_col] += D[row, col] - value
        D[row, col] = value
        return replace(calib, data_calibra=D)

    def test_negative_fixed_investment_moves_into_v_when_cx_is_v(self):
        # Provider layouts bridge Cx to V, so the negative-investment policy must
        # add to that V column; the three bridges must agree cell by cell.
        from puremacro.trade.condensed.table import BalancedIOTable
        from puremacro.trade.dynamic.accounts import FINAL_CATEGORIES, DynamicAccounts

        calib = self._with_cell(self._calib("load_figaro"), 1, -1.0)
        with pytest.raises(m.MRIOIntegrityError, match="negative_investment='to_inventory'"):
            m.MRIOTable.from_trade_calibration(calib)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert t.fd_codes == ("C", "G", "V") and t.final_use("G").min() == 0.0
        assert "1 negative investment cells moved into V" in t.transformations
        M, n = calib.n_countries * calib.n_sectors, calib.n_countries
        F3 = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)
        np.testing.assert_array_equal(t.final_use("G") + t.final_use("V"), F3[:, :, 1] + F3[:, :, 2])
        dyn = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
        cond = BalancedIOTable.from_trade_calibration(calib, negative_investment="to_inventory",
                                                      merchandise_mask=np.array([True, False]))
        for j, cat in enumerate(("C", "G", "X", "V")):
            mrio_cat = t.final_use(cat) if cat in t.fd_codes else np.zeros((M, n))
            np.testing.assert_array_equal(mrio_cat, dyn.F[:, :, FINAL_CATEGORIES.index(cat)])
            np.testing.assert_array_equal(mrio_cat, cond.F[:, :, j])

    def test_signed_inventory_cells_stay_in_v(self):
        calib = self._with_cell(self._calib("load_figaro"), 2, -1.0)
        t = m.MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
        assert "X" not in t.fd_codes and t.final_use("V")[0, 1] == -1.0
        assert "negative investment" not in " ".join(t.transformations)


# ---------------------------------------------------------------------------
# Parity with the IO implementation (vendored package in a subprocess)
# ---------------------------------------------------------------------------

_RANDOM45 = """
import numpy as np
rng = np.random.default_rng(7)
N, S = 3, 45; M = N * S
Z = rng.uniform(0.0, 0.02, (M, M))
F = rng.uniform(0.2, 1.0, (M, N, 4)); F[:, :, 2] *= 0.1; F[:, :, 3] = rng.uniform(-0.05, 0.05, (M, N))
y = Z.sum(1) + F.sum((1, 2)); TLS = 0.03 * y; VA = y - Z.sum(0) - TLS; TFD = 0.04 * F.sum(0)
"""


class TestParityWithIO:
    def test_small_native_regularize_parity(self, tmp_path):
        _require_io(VENDOR)
        out = tmp_path / "io.npz"
        _subprocess(f"""
import sys; sys.path.insert(0, {str(VENDOR)!r})
import numpy as np
from puremacro.trade.corrected import data as cd
rng = np.random.default_rng(283)
Z = rng.uniform(.02, .1, (6, 6)); F = rng.uniform(.1, .3, (6, 3, 4)); F[:, :, 2] = 0; F[:, :, 3] *= .03
y = Z.sum(1) + F.sum((1, 2)); TLS = .04 * y; VA = y - Z.sum(0) - TLS; TFD = .05 * F.sum(0)
table, rep = cd.table_from_arrays(Z, F, VA, TLS, TFD, country_codes=('USA', 'CHN', 'RESIDUAL'),
    sector_codes=('native_goods', 'native_services'), fd_layout='cgxv', year=2019, reference_country='RESIDUAL')
d = rep.to_dict()
np.savez({str(out)!r}, Z=table.Z, F=table.F, VA=table.VA, TLS=table.TLS, TFD=table.TFD, xn=table.net_exports(),
         med=d['row_identity_median_rel'], dtls=d['dtls_accounting_abs'], share=d['dtls_accounting_max_share_of_output'],
         world_abs_tls=d['world_abs_tls'], xn0=d['xn0_raw_sum'], gdp=d['world_gdp'])
""")
        io = np.load(out)
        rng = np.random.default_rng(283)
        Z = rng.uniform(.02, .1, (6, 6))
        F = rng.uniform(.1, .3, (6, 3, 4))
        F[:, :, 2] = 0
        F[:, :, 3] *= .03
        y = Z.sum(1) + F.sum((1, 2))
        TLS = .04 * y
        VA = y - Z.sum(0) - TLS
        TFD = .05 * F.sum(0)
        t = m.MRIOTable.from_arrays(Z, F, VA, TLS, TFD, country_codes=("USA", "CHN", "RESIDUAL"),
                                    sector_codes=("native_goods", "native_services"), reference_country="RESIDUAL")
        reg, rep = m.regularize_table(t)
        for key, mine in (("Z", reg.Z), ("F", reg.F), ("VA", reg.VA), ("TLS", reg.TLS), ("TFD", reg.TFD), ("xn", reg.net_exports())):
            assert _maxdiff(io[key], mine) < 1e-13, key
        for key, mine in (("med", rep.row_identity_median_rel), ("dtls", rep.dtls_accounting_abs),
                          ("share", rep.dtls_accounting_max_share_of_output), ("world_abs_tls", rep.world_abs_tls),
                          ("xn0", rep.xn0_raw_sum), ("gdp", rep.world_gdp)):
            assert abs(float(io[key]) - mine) < 1e-13, key

    def test_random45_aggregate_and_coarse_rule_parity(self, tmp_path):
        _require_io(VENDOR)
        out = tmp_path / "io.npz"
        _subprocess(f"""
import sys; sys.path.insert(0, {str(VENDOR)!r})
{_RANDOM45}
from puremacro.trade.corrected import data as cd, tariffs as ct
tab, rep = cd.table_from_arrays(Z, F, VA, TLS, TFD, country_codes=("USA", "CHN", "ROW"), sector_codes=cd.RAW_45_SECTOR_CODES, fd_layout="cgxv")
ag = cd.aggregate_icio(tab, "agregar"); ai = cd.aggregate_icio(tab, "isic_section")
delta = rng.uniform(0, 0.8, (N, S)); delta[0] = 0
r11o = ct.coarse_11o(delta, tab); tau11, taufd11, tauV11 = ct.coarse_11b(delta, tab)
np.savez({str(out)!r}, Z=tab.Z, F=tab.F, VA=tab.VA, TLS=tab.TLS, agZ=ag.Z, agF=ag.F, agVA=ag.VA, agTLS=ag.TLS,
         aiZ=ai.Z, aiF=ai.F, aiVA=ai.VA, aiTLS=ai.TLS, delta=delta, r11o=r11o, tau11=tau11, taufd11=taufd11, tauV11=tauV11,
         aggrel=ag.report.aggregated_identity_rel)
""")
        io = np.load(out)
        ns = {}
        exec(_RANDOM45, ns)
        t = m.MRIOTable.from_arrays(ns["Z"], ns["F"], ns["VA"], ns["TLS"], ns["TFD"], country_codes=("USA", "CHN", "ROW"),
                                    sector_codes=RAW_45_SECTOR_CODES, merchandise_mask=m.goods_mask(RAW_45_SECTOR_CODES))
        reg, _ = m.regularize_table(t)
        assert _maxdiff(io["Z"], reg.Z) < 1e-13 and _maxdiff(io["TLS"], reg.TLS) < 1e-13
        ag, ai = m.aggregate_mrio(reg, "agregar"), m.aggregate_mrio(reg, "isic_section")
        for key, mine in (("agZ", ag.Z), ("agF", ag.F), ("agVA", ag.VA), ("agTLS", ag.TLS),
                          ("aiZ", ai.Z), ("aiF", ai.F), ("aiVA", ai.VA), ("aiTLS", ai.TLS)):
            assert _maxdiff(io[key], mine) < 1e-12, key
        assert ag.metadata["aggregation"]["aggregated_identity_rel"] <= max(float(io["aggrel"]), 1e-14) * 10 + 1e-15
        delta = io["delta"]
        ro = m.coarse_tariff_rates(delta, reg, "agregar", "output")
        rb = m.coarse_tariff_rates(delta, reg, "agregar", "bilateral")
        assert _maxdiff(io["r11o"], ro.rates) < 1e-12 and _maxdiff(io["r11o"], rb.rates) < 1e-12
        assert _maxdiff(io["tau11"], rb.tau) < 1e-12
        kC, kG, kX, kV = (reg.fd_codes.index(c) for c in ("C", "G", "X", "V"))
        assert _maxdiff(io["taufd11"][:, :, 0], rb.tau_fd[:, :, kC]) < 1e-12
        assert _maxdiff(io["taufd11"][:, :, 1], rb.tau_fd[:, :, kG]) < 1e-12
        assert _maxdiff(io["taufd11"][:, :, 2], rb.tau_fd[:, :, kX]) < 1e-12
        assert _maxdiff(io["tauV11"], rb.tau_fd[:, :, kV]) < 1e-12


# ---------------------------------------------------------------------------
# Real sources on the IO volume (slow; recorded totals from the IO audits)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealSources:
    def test_icio_2019_native_regularize_aggregate_and_coarse(self, tmp_path):
        _require_io(ICIO_2019)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            icio = m.read_oecd_native(ICIO_2019, 2019)
        assert icio.sources[0].status == "whitelisted" and icio.sources[0].md5 == m.OECD_ICIO_MD5[2019]
        assert icio.sources[0].sha256 == m.OECD_ICIO_SHA256[2019]
        assert (icio.n_countries, icio.n_sectors) == (77, 45) and icio.is_raw
        acc = icio.accounting_report()
        assert acc.active_cells == 3440 and acc.zero_output_cells == 25 and acc.negative_inventory_entries == 453
        assert acc.negative_consumption_entries == 0 and acc.world_reported_value_added == pytest.approx(81589615.1063)
        assert acc.world_product_taxes == pytest.approx(6264465.9931, abs=1e-3)
        assert acc.final_category_totals["X"] == pytest.approx(1149949.5916, abs=1e-3)
        assert acc.final_category_totals["V"] == pytest.approx(1319079.2177, abs=1e-3)
        reg, rep = m.regularize_table(icio)
        assert len(rep.phantoms) == 25 and rep.floored == () and rep.passed
        assert rep.row_identity_median_rel == pytest.approx(3.99e-7, rel=0.01)
        assert rep.world_gdp == pytest.approx(87854087.49032502, rel=1e-9)
        assert rep.dtls_accounting_max_share_of_output == pytest.approx(0.0100, abs=1e-4)
        assert abs(rep.xn0_raw_sum) < 1e-7
        ag = m.aggregate_mrio(reg, "agregar")
        assert ag.metadata["aggregation"]["aggregated_identity_rel"] < 1e-13
        delta = np.zeros((77, 45))
        delta[:, RAW_45_SECTOR_CODES.index("C24")] = 0.25
        delta[reg.country_index("USA")] = 0
        ro = m.coarse_tariff_rates(delta, reg, "agregar", "output")
        assert ro.rates[reg.country_index("CHN"), 1] == pytest.approx(0.00202, abs=5e-6)
        assert ro.rates[reg.country_index("DEU"), 1] == pytest.approx(0.01130, abs=5e-6)
        assert not ro.rates[:, [0] + list(range(2, 11))].any()
        out = tmp_path / "io.npz"
        _subprocess(f"""
import sys; sys.path.insert(0, {str(VENDOR)!r})
import numpy as np
from puremacro.trade.corrected import data as cd, tariffs as ct
tab, rep = cd.build_icio(2019, 45, path={str(ICIO_2019)!r})
ag = cd.aggregate_icio(tab, "agregar")
delta = np.zeros((77, 45)); delta[:, cd.RAW_45_SECTOR_CODES.index("C24")] = 0.25; delta[tab.country_index("USA")] = 0
np.savez({str(out)!r}, Z=tab.Z, F=tab.F, VA=tab.VA, TLS=tab.TLS, TFD=tab.TFD, agZ=ag.Z, agF=ag.F, r11o=ct.coarse_11o(delta, tab),
         xn=tab.net_exports(), gdp=rep.world_gdp, med=rep.row_identity_median_rel, share=rep.dtls_accounting_max_share_of_output)
""")
        io = np.load(out)
        for key, mine in (("Z", reg.Z), ("F", reg.F), ("VA", reg.VA), ("TLS", reg.TLS), ("TFD", reg.TFD),
                          ("agZ", ag.Z), ("agF", ag.F), ("r11o", ro.rates), ("xn", reg.net_exports())):
            assert _maxdiff(io[key], mine) < 1e-9, key
        assert abs(float(io["gdp"]) - rep.world_gdp) < 1e-6 and abs(float(io["med"]) - rep.row_identity_median_rel) < 1e-15
        assert abs(float(io["share"]) - rep.dtls_accounting_max_share_of_output) < 1e-15

    def test_figaro_2019_native_matches_io_readers(self, tmp_path):
        path = IO_SOURCES / "matrix_eu-ic-io_ind-by-ind_26ed_2019.csv"
        _require_io(path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            raw = m.read_figaro_native(path, rounding_repair=False)
            fig = m.read_figaro_native(path)
        assert fig.sources[0].status == "verified" and fig.sources[0].sha256 == m.FIGARO_2026ED_2019_SHA256
        assert (fig.n_countries, fig.n_sectors) == (50, 64) and fig.country_codes[-1] == "ZAF" and "ROW" in fig.country_codes
        acc = raw.accounting_report()
        assert acc.active_cells == 3133 and acc.negative_output_cells == 1 and acc.negative_inventory_entries == 20342
        assert acc.column_output_max_scaled_gap < 1e-10 and acc.world_reported_value_added == pytest.approx(73775596.651)
        assert acc.world_production_taxes == pytest.approx(1502531.645)
        adj = fig.metadata["rounding_adjustments"]
        assert len(adj) == 1 and adj[0]["cell"] == "AL_T" and adj[0]["output"] == pytest.approx(-0.001)
        assert fig.merchandise_mask.sum() == 22
        reg, rep = m.regularize_table(fig)
        assert len(rep.phantoms) == 67 and len(rep.floored) == 9 and rep.passed
        assert rep.world_gdp == pytest.approx(79054913.75506693, rel=1e-9)
        assert rep.negative_fd_cells["V"] == 20342
        # The regularized native table calibrates the legacy model: 3 cells keep production taxes inside value
        # added (they exceed it), and 104 active non-phantom cells lack two strictly positive observed factors.
        with pytest.warns(RuntimeWarning, match=r"\(ARG_A01, CYP_H51, JPN_M73\)"):
            D, det = m.to_calibration_matrix(reg, return_details=True)
        assert det["production_taxes_retained_cells"] == ("ARG_A01", "CYP_H51", "JPN_M73")
        assert len(det["share_fallback_cells"]) == 171 and set(rep.phantoms) <= set(det["share_fallback_cells"])
        M = reg.n_cells
        assert (D[M + 1, :M] > 0).all() and (D[M + 2, :M] > 0).all()
        calib = calibrate_trade_model(D, ns=64, nc=50, nfd=3, country_codes=reg.country_codes, sector_codes=reg.sector_codes)
        assert np.all((calib.alpha > 0) & (calib.alpha < 1))
        del D, calib
        out = tmp_path / "io.npz"
        _subprocess(f"""
import sys, os
os.chdir({str(IO_ROOT / 'headlinePaper/rebuild')!r}); sys.path.insert(0, {str(IO_ROOT / 'headlinePaper/rebuild')!r})
import numpy as np
import calibrate_databases_2019 as cdb
kwargs, prov, registry = cdb.figaro()
np.savez({str(out)!r}, Z=kwargs["Z"], F=kwargs["F"], VA=kwargs["VA"], TLS=kwargs["TLS"], TFD=kwargs["TFD"],
         goods=registry.merchandise_candidate.to_numpy(), n_adj=len(prov["rounding_adjustments"]))
""", cwd=str(IO_ROOT / "headlinePaper/rebuild"))
        io = np.load(out)
        for key, mine in (("Z", fig.Z), ("F", fig.F), ("VA", fig.VA), ("TLS", fig.TLS), ("TFD", fig.TFD)):
            assert _maxdiff(io[key], mine) < 1e-9, key
        assert np.array_equal(io["goods"], fig.merchandise_mask) and int(io["n_adj"]) == 1
        out2 = tmp_path / "native.npz"
        _subprocess(f"""
import sys; sys.path.insert(0, {str(IO_ROOT)!r})
import numpy as np
from dynamic_model.native_data import load_figaro
d = load_figaro({str(IO_ROOT)!r}, year=2019)
np.savez({str(out2)!r}, C=d.final_consumption, G=d.final_investment, V=d.inventory_changes, VA=d.VA, TLS=d.TLS, out=d.output,
         TFD=d.TFD, lab=d.labor_compensation, cap=d.operating_surplus, prod=d.production_taxes, countries=np.array(d.countries))
""", cwd=str(IO_ROOT))
        nat = np.load(out2)
        for key, mine in (("C", raw.final_use("C")), ("G", raw.final_use("G")), ("V", raw.final_use("V")), ("VA", raw.VA),
                          ("TLS", raw.TLS), ("out", raw.output), ("lab", raw.labor_compensation),
                          ("cap", raw.operating_surplus), ("prod", raw.production_taxes)):
            assert _maxdiff(nat[key], mine) < 1e-9, key
        assert _maxdiff(nat["TFD"][:, :4], raw.TFD) < 1e-9 and tuple(nat["countries"]) == raw.country_codes

    def test_exiobase_2019_native_matches_io_loader(self, tmp_path):
        core = IO_SOURCES / "exiobase_core"
        archive = IO_SOURCES / "IOT_2019_ixi.zip"
        _require_io(core / "Z.txt")
        _require_io(archive)
        # The extracted core is authenticated against the registered archive, member by member (size and CRC-32).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exio = m.read_exiobase_native(core, archive=archive)
        assert exio.sources[0].sha256 == m.EXIOBASE_382_2019_IXI_SHA256 and exio.sources[0].status == "verified"
        assert len(exio.sources) == 9 and exio.sources[0].archive_member is None
        assert all(s.status == "verified" and s.crc32 is not None for s in exio.sources[1:])
        assert exio.sources[1].archive_member == "IOT_2019_ixi/Z.txt" and exio.metadata["source_authenticated"] is True
        assert (exio.n_countries, exio.n_sectors) == (49, 163) and exio.is_sparse and exio.Z.nnz == 38380615
        assert exio.country_codes[-5:] == ("ROW_ASIA_PACIFIC", "ROW_AMERICA", "ROW_EUROPE", "ROW_AFRICA", "ROW_MIDDLE_EAST")
        assert exio.reference_country == "ROW_ASIA_PACIFIC"
        acc = exio.accounting_report()
        assert acc.zero_output_cells == 1084 and acc.nonpositive_active_factor_VA == 73 and acc.active_cells == 6903
        assert acc.final_category_totals["V"] == pytest.approx(1100405.960067401)
        assert acc.final_category_totals["VAL"] == pytest.approx(62203.98461110878)
        assert acc.world_reported_value_added == pytest.approx(73017735.1533808)
        assert acc.column_output_max_scaled_gap < 2e-6
        reg, rep = m.regularize_table(exio)
        assert len(rep.phantoms) == 1084 and len(rep.floored) == 89 and rep.passed
        assert rep.world_gdp == pytest.approx(78110996.72078681, rel=1e-9)
        out = tmp_path / "native.npz"
        _subprocess(f"""
import sys; sys.path.insert(0, {str(IO_ROOT)!r})
import numpy as np
from scipy import sparse
from dynamic_model.native_data import load_exiobase
d = load_exiobase({str(IO_ROOT)!r}, year=2019)
sparse.save_npz({str(tmp_path / 'Z.npz')!r}, d.Z)
np.savez({str(out)!r}, C=d.final_consumption, G=d.final_investment, V=d.inventory_changes, VAL=d.valuables, VA=d.VA, TLS=d.TLS,
         out=d.output, TFD=d.TFD, lab=d.labor_compensation, cap=d.operating_surplus, prod=d.production_taxes,
         countries=np.array(d.countries), sectors=np.array(d.sectors))
""", cwd=str(IO_ROOT))
        nat = np.load(out)
        assert _maxdiff(sparse.load_npz(tmp_path / "Z.npz"), exio.Z) == 0
        for key, mine in (("C", exio.final_use("C")), ("G", exio.final_use("G")), ("V", exio.final_use("V")),
                          ("VAL", exio.final_use("VAL")), ("VA", exio.VA), ("TLS", exio.TLS), ("out", exio.output),
                          ("lab", exio.labor_compensation), ("cap", exio.operating_surplus), ("prod", exio.production_taxes)):
            assert _maxdiff(nat[key], mine) < 1e-9, key
        assert _maxdiff(nat["TFD"], exio.TFD) < 1e-9
        assert tuple(nat["countries"]) == exio.country_codes and tuple(nat["sectors"]) == exio.sector_codes

    def test_bundled_table_is_not_an_aggregate_of_the_clean_2020_release(self):
        _require_io(ICIO_2020)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c20 = m.read_oecd_native(ICIO_2020, 2020)
        assert c20.sources[0].status == "whitelisted"
        reg, rep = m.regularize_table(c20, strict=False)  # the 2020 file sits just above the IO per-cell gate
        assert rep.failed_gates == ("dtls_cell_share_of_output",) and reg.metadata["regularized"] is False
        assert rep.gates["dtls_cell_share_of_output"]["value"] == pytest.approx(0.0106, abs=1e-4)
        with pytest.warns(RuntimeWarning, match="gates failed"):
            agg = m.aggregate_mrio(reg, "agregar")
        with np.load(trade_data.bundled_icio_path()) as bundle:
            D = np.asarray(bundle["data"], dtype=float)
        M = 847
        bundled_va = D[M + 1, :M].sum() + D[M + 2, :M].sum()
        assert bundled_va / agg.VA.sum() > 1000  # 7.05e11 versus about 7.97e7 USD million
        Zb, Za = D[:M, :M], np.asarray(agg.Z)
        nonzero = (Zb != 0) | (Za != 0)
        agree = np.isclose(Zb, Za, rtol=1e-6, atol=1e-6)[nonzero].mean()
        # Lost decimal points leave low-precision tokens intact: about a quarter of the nonzero
        # intermediate cells agree and three quarters differ (measured 0.2549 on 2026-09-22).
        assert 0.15 < agree < 0.35
        assert (Zb > 1e8).sum() > 500 and (Za > 1e8).sum() == 0
        # The value-added row agrees in 1 of 847 cells (0.12% on 2026-09-22): the tokens there carry
        # three or more decimals, so the lost decimal points touch almost every cell.
        va_agree = np.isclose(D[M + 1, :M] + D[M + 2, :M], agg.VA, rtol=1e-6).mean()
        assert va_agree < 0.01
