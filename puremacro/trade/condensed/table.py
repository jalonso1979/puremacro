"""Balanced inter-country input-output tables for the condensed tariff model.

The condensed model reads a balanced table with four final-use categories:
C (household, non-profit and government consumption), G (gross fixed capital
formation), X (residents' purchases abroad) and V (signed inventory changes).
:func:`table_from_arrays` accepts the shared wave-1 layouts described in the
implementation brief and applies a fixed build contract ported from the IO
research engine (``corrected/data.py::table_from_arrays``):

1. Shape and finiteness checks; the country and sector registries define the
   country-major cell order ``i = country * S + sector``.
2. Identity guard: with a provider output column, the median relative row error
   must be at most ``1e-5``; zero-output cells must be completely empty.
3. Final demand is condensed to (C, G, X, V) by summing every received code
   into its category. The six OECD categories map as C = HFCE + NPISH + GGFC,
   G = GFCF, X = DPABR, V = INVNT; VAL (valuables, also EXIOBASE ``VALUABLES``
   and Eora ``ACQ_VAL``) is merged into V and recorded; the FIGARO
   (``P3_S14``, ``P3_S15``, ``P3_S13`` -> C, ``P51G`` -> G, ``P5M`` -> V) and
   WIOD (``CONS_h``, ``CONS_np``, ``CONS_g`` -> C, ``GFCF`` -> G, ``INVT`` -> V)
   vocabularies of ``puremacro.trade.data`` are accepted, and the EXIOBASE
   ``EXPORT`` column must be identically zero (a closed table has no sales
   outside the modelled world). C and G are required; X and V are zero when
   absent, and a table without purchases abroad calibrates only with
   ``allow_empty_purchases_abroad=True``. The three-category layout (C, G, X)
   of puremacro's bundled 77x11 table needs an explicit ``negative_investment``
   policy because it merges inventories into G.
4. Every empty cell gets a phantom: own-country C purchases and value added of
   ``phantom`` (default 1e-6 in the table's units), so its row and column totals
   agree exactly.
5. Value added is floored at ``va_floor_rel`` of output where it is not positive.
6. Net product taxes on production (TLS) are recomputed as the residual that
   closes every column; the change is gated and reported.
7. Trade balances are measured fob over all uses; round-off in their world sum
   is assigned to the reference country.

Nothing here reads provider files: use ``puremacro.trade._oecd_icio.read_native``
or the loaders in ``puremacro.trade.data`` and pass their arrays or
``RawIOData`` through :meth:`BalancedIOTable.from_raw`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .errors import DataIntegrityError, UnsupportedExtension

FD_CODES: tuple[str, ...] = ("C", "G", "X", "V")
"""Final-demand categories in storage order: consumption, investment, purchases abroad, inventories."""

ENDOGENOUS_FD: tuple[str, ...] = ("C", "G", "X")
"""Categories with Cobb-Douglas spending shares; V is an exogenous quantity vector."""

OECD_SIX_FD_CODES: tuple[str, ...] = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")
"""Final-demand columns of the OECD ICIO SML layout, in file order."""

PHANTOM_DEFAULT = 1e-6
VA_FLOOR_REL_DEFAULT = 1e-3
ROW_IDENTITY_TOL = 1e-5
DTLS_WORLD_TOL = 1e-4
DTLS_CELL_TOL = 1e-2
XN_SUM_TOL = 1e-9

# Sector registries whose merchandise (ISIC A-C) membership is known without a
# concordance module: the raw OECD 45 industries, puremacro's bundled 11
# composites, the Agregar and ISIC-section 11-sector maps and the frozen
# 3-region x 3-sector fixture. Anything else needs an explicit mask.
_GOODS_CODES: frozenset[str] = frozenset({
    "A01_02", "A03", "B05_06", "B07_08", "B09", "C10T12", "C13T15", "C16", "C17_18", "C19",
    "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31T33",
    "AGRI", "MINQ", "MANU", "AGRI_MIN_FOOD", "MANUF_EXFOOD", "A", "B", "C",
    "RESOURCES", "MANUFACTURING",
})
_SERVICE_CODES: frozenset[str] = frozenset({
    "D", "E", "F", "G", "H49", "H50", "H51", "H52", "H53", "I", "J58T60", "J61", "J62_63",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "ENEG", "CONS", "TRAD", "TRAN", "INFO", "FIN", "GOV", "OTHS",
    "UTILITIES", "CONSTRUCTION", "TRADE", "TRANSPORT", "ACCOM_ICT", "FIN_REALESTATE",
    "PROF_ADMIN_PUBADM", "EDU_HEALTH", "ARTS_OTHER", "DE", "H", "J", "KL", "OPQ", "REST",
    "SERVICES",
})


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


# Final-demand vocabularies accepted by table_from_arrays, mapped onto the
# stored categories; several codes may map to one category and are summed.
# EXPORT (EXIOBASE "Exports: Total (fob)") is accepted only when identically
# zero: a closed multi-region table has no purchases outside the modelled world.
_FD_VOCABULARY: dict[str, str] = {
    # generic (C, G, X, V[, VAL]) and the legacy (C, I, Cx) of the bundled 77x11 layout
    "C": "C", "G": "G", "X": "X", "V": "V", "VAL": "VAL", "I": "G", "CX": "X",
    # OECD ICIO (six codes), EXIOBASE 3 and Eora 26
    "HFCE": "C", "NPISH": "C", "GGFC": "C", "GFCF": "G", "INVNT": "V", "DPABR": "X",
    "VALUABLES": "VAL", "ACQ_VAL": "VAL", "EXPORT": "EXPORT",
    # FIGARO (Eurostat): households, NPISH, government, GFCF, inventories and valuables
    "P3_S14": "C", "P3_S15": "C", "P3_S13": "C", "P51G": "G", "P5M": "V",
    # WIOD 2016
    "CONS_H": "C", "CONS_NP": "C", "CONS_G": "C", "INVT": "V",
}


def _kv_frame(items: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(items), columns=["Item", "Value"]).set_index("Item")


def infer_merchandise_mask(sector_codes: Sequence[str]) -> np.ndarray:
    """Boolean mask of merchandise (ISIC A-C) sectors for a known sector registry.

    Parameters
    ----------
    sector_codes : sequence of str
        Codes from the OECD 45-industry registry, puremacro's 11 composites
        (``AGRI``, ``MINQ``, ``MANU`` are goods), the Agregar or ISIC-section
        11-sector maps, or the frozen 3-sector fixture (``RESOURCES``,
        ``MANUFACTURING``, ``SERVICES``).

    Returns
    -------
    np.ndarray of bool, shape (S,)

    Raises
    ------
    UnsupportedExtension
        A code outside the known registries; pass ``merchandise_mask`` explicitly.
    """
    codes = tuple(str(c) for c in sector_codes)
    unknown = [c for c in codes if c not in _GOODS_CODES and c not in _SERVICE_CODES]
    if unknown:
        raise UnsupportedExtension(
            f"unknown sector codes {unknown}; pass an explicit merchandise_mask (True for ISIC A-C)"
        )
    return np.array([c in _GOODS_CODES for c in codes], dtype=bool)


@dataclass(frozen=True)
class TableBuildReport:
    """Provenance and gate values of one table build.

    Attributes
    ----------
    source, year, unit : str, int or None, str
        Provenance written by the caller.
    n_countries, n_sectors : int
    fd_codes_in : tuple of str
        Final-demand layout received; the table always stores (C, G, X, V).
    row_identity_median_rel : float
        Median relative gap between row totals and the provider output column
        (0 when no output column was given).
    n_zero_output : int
        Empty cells that received phantoms.
    phantoms : tuple of str
        Labels ``COUNTRY_SECTOR`` of the phantom cells.
    floored : tuple of (label, old, new)
        Value-added floors applied.
    dtls_accounting_abs, dtls_floor_abs : float
        Absolute change in TLS from the residual closure, split into the part
        that changes the accounts and the part that offsets the value-added floor.
    dtls_accounting_max_share_of_output : float
        Largest cell change relative to ``max(output, 1)``.
    world_abs_tls, xn0_raw_sum, world_gdp : float
        World absolute TLS, raw world sum of trade balances, and world GDP.
    negative_fd_cells : dict
        Count of negative cells per stored category.
    negative_investment : str
        Policy applied to negative G cells (``"raise"`` or ``"to_inventory"``).
    negative_investment_cells : tuple of str
        Labels ``COUNTRY_SECTOR->DEST`` of G cells moved into V.
    valuables_merged : bool
        True when a VAL category was merged into V.
    metadata : dict
        Free-form provenance.
    """

    source: str
    year: int | None
    unit: str
    n_countries: int
    n_sectors: int
    fd_codes_in: tuple[str, ...]
    row_identity_median_rel: float
    n_zero_output: int
    phantoms: tuple[str, ...]
    floored: tuple[tuple[str, float, float], ...]
    dtls_accounting_abs: float
    dtls_floor_abs: float
    dtls_accounting_max_share_of_output: float
    world_abs_tls: float
    xn0_raw_sum: float
    world_gdp: float
    negative_fd_cells: dict[str, int]
    negative_investment: str
    negative_investment_cells: tuple[str, ...]
    valuables_merged: bool
    phantom: float = PHANTOM_DEFAULT
    va_floor_rel: float = VA_FLOOR_REL_DEFAULT
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation."""
        return {
            "source": self.source, "year": self.year, "unit": self.unit,
            "n_countries": self.n_countries, "n_sectors": self.n_sectors,
            "fd_codes_in": list(self.fd_codes_in),
            "row_identity_median_rel": self.row_identity_median_rel,
            "n_zero_output": self.n_zero_output, "phantoms": list(self.phantoms),
            "floored": [list(x) for x in self.floored],
            "dtls_accounting_abs": self.dtls_accounting_abs, "dtls_floor_abs": self.dtls_floor_abs,
            "dtls_accounting_max_share_of_output": self.dtls_accounting_max_share_of_output,
            "world_abs_tls": self.world_abs_tls, "xn0_raw_sum": self.xn0_raw_sum,
            "world_gdp": self.world_gdp, "negative_fd_cells": dict(self.negative_fd_cells),
            "negative_investment": self.negative_investment,
            "negative_investment_cells": list(self.negative_investment_cells),
            "valuables_merged": self.valuables_merged, "phantom": self.phantom,
            "va_floor_rel": self.va_floor_rel, "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column table of the scalar gate values."""
        return _kv_frame([
            ("source", self.source), ("year", self.year), ("unit", self.unit),
            ("countries", self.n_countries), ("sectors", self.n_sectors),
            ("final-demand layout received", " ".join(self.fd_codes_in)),
            ("median relative row identity", self.row_identity_median_rel),
            ("phantom cells", self.n_zero_output), ("value-added floors", len(self.floored)),
            ("|dTLS| changing the accounts", self.dtls_accounting_abs),
            ("|dTLS| offsetting floors", self.dtls_floor_abs),
            ("max |dTLS| / max(output, 1)", self.dtls_accounting_max_share_of_output),
            ("world |TLS|", self.world_abs_tls), ("raw world sum of trade balances", self.xn0_raw_sum),
            ("world GDP", self.world_gdp),
            ("negative G cells moved to V", len(self.negative_investment_cells)),
            ("valuables merged into V", self.valuables_merged),
        ])

    def summary(self) -> str:
        """One line with the table size, source and the number of repairs applied."""
        return (
            f"TableBuildReport: {self.n_countries} economies x {self.n_sectors} industries from "
            f"{self.source}; phantoms={self.n_zero_output}; VA floors={len(self.floored)}; "
            f"|dTLS|={self.dtls_accounting_abs:.3e}; negative G cells moved={len(self.negative_investment_cells)}"
        )

    def to_markdown(self, **kwargs: Any) -> str:
        """The gate table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The gate table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The gate table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)


@dataclass(frozen=True, eq=False)
class BalancedIOTable:
    """A balanced inter-country input-output table in the condensed model's layout.

    Cells are country-major: cell ``i = country * S + sector``. All flows are at
    basic prices in the table's own units (the OECD tables use USD million).

    Attributes
    ----------
    Z : np.ndarray, shape (M, M)
        Intermediate flows; row = selling cell, column = buying cell.
    F : np.ndarray, shape (M, N, 4)
        Final-demand flows by destination country and category (C, G, X, V).
    VA : np.ndarray, shape (M,)
        Value added (one composite factor).
    TLS : np.ndarray, shape (M,)
        Net product taxes on each cell's purchases (the residual that balances columns).
    TFD : np.ndarray, shape (N, 4)
        Net product taxes on final demand by destination and category.
    country_codes, sector_codes : tuple of str
        Registries in storage order.
    fd_codes : tuple of str
        Always ``("C", "G", "X", "V")``.
    active : np.ndarray of bool, shape (M,)
        Cells with genuine output (False for phantoms).
    merchandise_mask : np.ndarray of bool, shape (S,)
        Goods sectors (ISIC A-C), inferred from the registry or given explicitly.
    report : TableBuildReport
        Provenance and gate values of the build.
    reference_country : str
        Country whose income equation is replaced by the numeraire and which
        absorbs round-off in world trade balances (default: the last code).
    metadata : dict
        Free-form provenance.
    """

    Z: np.ndarray
    F: np.ndarray
    VA: np.ndarray
    TLS: np.ndarray
    TFD: np.ndarray
    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    fd_codes: tuple[str, ...]
    active: np.ndarray
    merchandise_mask: np.ndarray
    report: TableBuildReport
    reference_country: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_countries(self) -> int:
        """Number of economies."""
        return len(self.country_codes)

    @property
    def n_sectors(self) -> int:
        """Number of industries per economy."""
        return len(self.sector_codes)

    @property
    def n_cells(self) -> int:
        """Number of country-industry cells, ``N * S``."""
        return self.n_countries * self.n_sectors

    @property
    def output(self) -> np.ndarray:
        """Gross output by cell from the cost side, ``sum_i Z_ij + TLS_j + VA_j``."""
        return self.Z.sum(axis=0) + self.TLS + self.VA

    @property
    def country_of_cell(self) -> np.ndarray:
        """Country index of every cell (country-major layout)."""
        return np.repeat(np.arange(self.n_countries), self.n_sectors)

    @property
    def labels(self) -> tuple[str, ...]:
        """``COUNTRY_SECTOR`` label of every cell."""
        return tuple(f"{c}_{s}" for c in self.country_codes for s in self.sector_codes)

    def country_index(self, code: str) -> int:
        """Storage position of a country code."""
        try:
            return self.country_codes.index(code)
        except ValueError as exc:
            raise DataIntegrityError(f"country {code!r} is not in the table") from exc

    @property
    def reference(self) -> int:
        """Storage position of the reference country."""
        return self.country_index(self.reference_country)

    def net_exports(self) -> np.ndarray:
        """Fob trade balances over all uses, with the world sum's round-off assigned to the reference country."""
        xn, _ = _net_exports(self.Z, self.F, self.n_countries, self.n_sectors)
        xn[self.reference] -= xn.sum()
        return xn

    def summary(self) -> str:
        """One-paragraph description of the table and its build."""
        r = self.report
        return (
            f"BalancedIOTable: {self.n_countries} economies x {self.n_sectors} industries "
            f"({r.source}, {r.year}, {r.unit}); phantoms={len(r.phantoms)}; VA floors={len(r.floored)}; "
            f"median row identity={r.row_identity_median_rel:.2e}; reference={self.reference_country}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Country totals: value added, production taxes, final-demand taxes, GDP and net exports."""
        n, s = self.n_countries, self.n_sectors
        va = self.VA.reshape(n, s).sum(axis=1)
        tls = self.TLS.reshape(n, s).sum(axis=1)
        tfd = self.TFD.sum(axis=1)
        frame = pd.DataFrame({
            "value_added": va, "production_taxes": tls, "final_demand_taxes": tfd,
            "gdp": va + tls + tfd, "net_exports": self.net_exports(),
        }, index=pd.Index(self.country_codes, name="country"))
        return frame

    def to_markdown(self, **kwargs: Any) -> str:
        """The country totals of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The country totals of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The country totals of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, Z, F, VA, TLS, TFD, **kwargs: Any) -> "BalancedIOTable":
        """Build a table from arrays; see :func:`table_from_arrays` (the report is ``table.report``)."""
        table, _ = table_from_arrays(Z, F, VA, TLS, TFD, **kwargs)
        return table

    @classmethod
    def from_io_table(cls, table: Any, **kwargs: Any) -> "BalancedIOTable":
        """Build from any object following the shared wave-1 IO-table contract.

        The object must expose ``Z`` (dense or ``scipy.sparse``), ``F`` (M, N, K),
        ``fd_codes``, ``VA``, ``TLS``, ``TFD``, ``country_codes`` and
        ``sector_codes``; ``output``, ``merchandise_mask`` and ``metadata`` are
        used when present. Keyword arguments go to :func:`table_from_arrays`.
        """
        Z = table.Z
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        kwargs.setdefault("country_codes", tuple(table.country_codes))
        kwargs.setdefault("sector_codes", tuple(table.sector_codes))
        kwargs.setdefault("fd_codes", tuple(table.fd_codes))
        output = getattr(table, "output", None)
        if output is not None and "output" not in kwargs:
            kwargs["output"] = np.asarray(output, dtype=float)
        mask = getattr(table, "merchandise_mask", None)
        if mask is not None and "merchandise_mask" not in kwargs:
            kwargs["merchandise_mask"] = np.asarray(mask, dtype=bool)
        meta = getattr(table, "metadata", None)
        if isinstance(meta, Mapping) and "metadata" not in kwargs:
            kwargs["metadata"] = dict(meta)
        kwargs.setdefault("source", str(kwargs.get("metadata", {}).get("source", type(table).__name__)))
        table_out, _ = table_from_arrays(Z, table.F, table.VA, table.TLS, table.TFD, **kwargs)
        return table_out

    @classmethod
    def from_raw(cls, raw: Any, **kwargs: Any) -> "BalancedIOTable":
        """Build from a ``puremacro.trade.data.RawIOData`` container.

        ``F`` may be ``(M, C * K_F)`` or ``(M, C, K_F)``; a two-dimensional
        value-added block ``(K_VA, M)`` is summed over factors; ``fd_categories``
        may be the six OECD codes, (C, G, X, V), (C, G, X, V, VAL), (C, G, X),
        the legacy (C, I, Cx), or the FIGARO, EXIOBASE, WIOD and Eora
        vocabularies of the ``puremacro.trade.data`` loaders (see
        :func:`table_from_arrays`). Those four have no purchases-abroad column,
        so the resulting table calibrates only with
        ``allow_empty_purchases_abroad=True``; their sector registries need an
        explicit ``merchandise_mask``. A missing ``taxes_less_subsidies_fd`` is
        treated as zero final-demand taxes.
        """
        C, S, K = raw.C, raw.S, raw.K_F
        M = C * S
        F = np.asarray(raw.final_demand_matrix, dtype=float).reshape(M, C, K)
        VA = np.asarray(raw.value_added, dtype=float)
        if VA.ndim == 2:
            VA = VA.sum(axis=0)
        tfd = getattr(raw, "taxes_less_subsidies_fd", None)
        TFD = np.zeros((C, K)) if tfd is None else np.asarray(tfd, dtype=float).reshape(C, K)
        kwargs.setdefault("country_codes", tuple(raw.countries))
        kwargs.setdefault("sector_codes", tuple(raw.sectors))
        kwargs.setdefault("fd_codes", tuple(raw.fd_categories))
        kwargs.setdefault("output", np.asarray(raw.gross_output, dtype=float))
        kwargs.setdefault("source", str(getattr(raw, "source", "RawIOData")))
        kwargs.setdefault("year", getattr(raw, "year", None) or None)
        kwargs.setdefault("unit", str(getattr(raw, "unit", "")))
        meta = getattr(raw, "metadata", None)
        if isinstance(meta, Mapping):
            kwargs.setdefault("metadata", dict(meta))
        table, _ = table_from_arrays(raw.intermediate_matrix, F, VA, raw.taxes_less_subsidies, TFD, **kwargs)
        return table

    @classmethod
    def from_trade_calibration(cls, calib: Any, *, negative_investment: str = "raise",
                               reference_country: str | None = None, cx_category: str | None = None,
                               **kwargs: Any) -> "BalancedIOTable":
        """Bridge from puremacro's bundled 77x11 layout (``TradeCalibrationResult``).

        ``calib.data_calibra`` rows ``M .. M+2`` are production taxes, labour and
        capital; its final-demand columns are C, I and Cx (country-major). Labour
        and capital are summed into the single factor; I becomes G and Cx
        becomes X. The bundled table merges inventories into I, so three of its
        cells are negative: ``negative_investment="raise"`` (default) refuses the
        table, ``"to_inventory"`` moves the negative cells into the exogenous V
        category and records them in the report. G therefore still contains
        the positive part of inventory changes.

        Cx becomes X only for the bundled layout (no
        ``calib.metadata["final_use_mapping"]``) or the OECD ``DPABR`` mapping.
        The FIGARO, WIOD and Eora loaders' Cx (signed changes in inventories and
        valuables) becomes the exogenous V category, so such a table has no
        purchases abroad and calibrates only with
        ``allow_empty_purchases_abroad=True``; EXIOBASE's Cx (which includes its
        export column) or an unrecognised mapping raises ``ValueError`` unless
        ``cx_category`` (``"X"``, ``"V"`` or ``"VAL"``, merged into V) is given.
        The decision is recorded in ``metadata["final_use_bridge"]``.
        """
        from puremacro.trade.dynamic.accounts import _calibration_cx_category

        data = getattr(calib, "data_calibra", None)
        if data is None:
            raise DataIntegrityError("the calibration has no data_calibra table; calibrate with validate=True")
        nc, ns, nfd = int(calib.n_countries), int(calib.n_sectors), int(calib.n_final_demand)
        M = nc * ns
        data = np.asarray(data, dtype=float)
        if data.shape != (M + 3, M + nc * nfd):
            raise DataIntegrityError(f"data_calibra has shape {data.shape}, expected {(M + 3, M + nc * nfd)}")
        Z = data[:M, :M]
        F = data[:M, M:].reshape(M, nc, nfd)
        TLS = data[M, :M]
        VA = data[M + 1, :M] + data[M + 2, :M]
        TFD = data[M, M:].reshape(nc, nfd)
        codes = tuple(calib.country_codes) if calib.country_codes else tuple(f"C{i:02d}" for i in range(nc))
        sectors = tuple(calib.sector_codes) if calib.sector_codes else tuple(f"S{i:02d}" for i in range(ns))
        if nfd != 3:
            raise UnsupportedExtension(
                f"a calibration with {nfd} final-demand categories cannot be bridged; the bridge expects C, I, Cx"
            )
        cx_target, bridge_record = _calibration_cx_category(
            calib, cx_category, caller="BalancedIOTable.from_trade_calibration")
        fd_in = ("C", "G", cx_target)
        kwargs.setdefault("source", "puremacro TradeCalibrationResult.data_calibra")
        kwargs.setdefault("unit", str(calib.metadata.get("unit", "calibration value units")))
        kwargs.setdefault("year", calib.metadata.get("year", None) or None)
        source_mapping = bridge_record["final_use_mapping"]
        investment_codes = None if source_mapping is None else list(source_mapping.get("I", ()))
        kwargs.setdefault("metadata", {"bridge": "from_trade_calibration",
                                       "labour_capital_summed": True,
                                       "investment_includes_inventories": (
                                           investment_codes is None
                                           or any(c in ("INVNT", "INVT", "P5M") for c in investment_codes))})
        kwargs["metadata"] = {**dict(kwargs["metadata"] or {}), "final_use_bridge": bridge_record}
        table, _ = table_from_arrays(
            Z, F, VA, TLS, TFD, country_codes=codes, sector_codes=sectors, fd_codes=fd_in,
            negative_investment=negative_investment, reference_country=reference_country, **kwargs,
        )
        return table


def _net_exports(Z: np.ndarray, F: np.ndarray, n: int, s: int) -> tuple[np.ndarray, float]:
    """Fob exports minus imports by country over all uses, and their raw world sum."""
    bilateral = Z.reshape(n, s, n, s).sum(axis=(1, 3)) + F.sum(axis=2).reshape(n, s, n).sum(axis=1)
    xn = bilateral.sum(axis=1) - bilateral.sum(axis=0)
    return xn, float(xn.sum())


def _condense_final_demand(F: np.ndarray, TFD: np.ndarray, fd_codes: tuple[str, ...]
                           ) -> tuple[np.ndarray, np.ndarray, bool]:
    """Map a received final-demand layout onto (C, G, X, V) by summing the codes of each category."""
    codes = tuple(str(c).strip() for c in fd_codes)
    upper = tuple(c.upper() for c in codes)
    m, n = F.shape[0], F.shape[1]
    if len(set(upper)) != len(upper):
        raise DataIntegrityError(f"duplicate final-demand codes in {codes}")
    unknown = [c for c, u in zip(codes, upper) if u not in _FD_VOCABULARY]
    if unknown:
        raise DataIntegrityError(
            f"unknown final-demand code(s) {unknown}; accepted layouts are the six OECD codes, "
            "(C, G, X, V), (C, G, X, V, VAL), (C, G, X), the legacy (C, I, Cx) and the FIGARO, "
            "EXIOBASE, WIOD and Eora vocabularies of puremacro.trade.data"
        )
    mapped = [_FD_VOCABULARY[u] for u in upper]
    if "C" not in mapped or "G" not in mapped:
        raise DataIntegrityError(f"final-demand layout {codes} must contain consumption (C) and investment (G)")
    F4 = np.zeros((m, n, 4))
    T4 = np.zeros((n, 4))
    valuables = False
    for j, cat in enumerate(mapped):
        if cat == "EXPORT":
            if np.any(F[:, :, j] != 0.0) or np.any(TFD[:, j] != 0.0):
                raise DataIntegrityError(
                    f"final-demand column {codes[j]!r} (sales outside the modelled world) must be identically "
                    "zero in a closed multi-region table"
                )
            continue
        if cat == "VAL":
            valuables = True
            k = FD_CODES.index("V")
        else:
            k = FD_CODES.index(cat)
        F4[:, :, k] += F[:, :, j]
        T4[:, k] += TFD[:, j]
    return F4, T4, valuables


def table_from_arrays(
    Z: np.ndarray,
    F: np.ndarray,
    VA: np.ndarray,
    TLS: np.ndarray,
    TFD: np.ndarray,
    *,
    country_codes: Sequence[str],
    sector_codes: Sequence[str],
    fd_codes: Sequence[str] = FD_CODES,
    output: np.ndarray | None = None,
    negative_investment: str = "raise",
    reference_country: str | None = None,
    merchandise_mask: np.ndarray | None = None,
    phantom: float = PHANTOM_DEFAULT,
    va_floor_rel: float = VA_FLOOR_REL_DEFAULT,
    source: str = "arrays",
    year: int | None = None,
    unit: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BalancedIOTable, TableBuildReport]:
    """Apply the build contract to raw arrays and return the table with its report.

    Parameters
    ----------
    Z : np.ndarray, shape (M, M)
        Intermediate flows, rows = seller cell, columns = buyer cell.
    F : np.ndarray, shape (M, N, K)
        Final-demand flows by destination country and category, in the layout
        named by ``fd_codes``.
    VA, TLS : np.ndarray, shape (M,)
        Value added and the table's own net product taxes on production.
    TFD : np.ndarray, shape (N, K)
        Net product taxes on final demand, in the layout of ``F``.
    country_codes, sector_codes : sequence of str
        Registries; the cell order is country-major.
    fd_codes : sequence of str
        The six OECD codes, ``("C","G","X","V")``, ``("C","G","X","V","VAL")``
        (VAL merged into V), ``("C","G","X")``, the legacy ``("C","I","Cx")``,
        or the loader vocabularies of ``puremacro.trade.data``: FIGARO
        (``P3_S14``, ``P3_S15``, ``P3_S13``, ``P51G``, ``P5M``), EXIOBASE
        (``HFCE``, ``NPISH``, ``GGFC``, ``GFCF``, ``INVNT``, ``VALUABLES``,
        ``EXPORT`` which must be identically zero), WIOD (``CONS_h``,
        ``CONS_np``, ``CONS_g``, ``GFCF``, ``INVT``) and Eora (``HFCE``,
        ``NPISH``, ``GGFC``, ``GFCF``, ``INVNT``, ``ACQ_VAL``). Codes mapping to
        one category are summed; C and G are required, X and V are zero when
        absent (a table without X needs ``allow_empty_purchases_abroad=True``
        at calibration).
    output : np.ndarray, shape (M,), optional
        Provider output column; row totals are used when absent.
    negative_investment : {"raise", "to_inventory"}
        Policy for negative G cells. ``"to_inventory"`` moves them into V.
    reference_country : str, optional
        Country that absorbs world round-off and whose income equation the
        numeraire replaces. Default: the last country code.
    merchandise_mask : array of bool, shape (S,), optional
        Goods sectors. Inferred from known registries when absent.
    phantom, va_floor_rel : float
        Phantom flow for empty cells and the value-added floor share.
    source, year, unit, metadata
        Provenance written into the report.

    Returns
    -------
    (BalancedIOTable, TableBuildReport)

    Raises
    ------
    DataIntegrityError
        Wrong shapes, non-finite data, negative output, a zero-output cell with
        entries, an unknown or duplicated final-demand code, a nonzero EXPORT
        column, a residual TLS change above ``1e-4`` of world |TLS| or ``1e-2``
        of a cell's output, or a world trade-balance sum above ``1e-9`` of GDP.
    UnsupportedExtension
        Unknown policy names or unknown sector codes without a mask.
    """
    ccodes = tuple(str(c) for c in country_codes)
    scodes = tuple(str(s) for s in sector_codes)
    n, s = len(ccodes), len(scodes)
    m = n * s
    if n == 0 or s == 0 or len(set(ccodes)) != n or len(set(scodes)) != s:
        raise DataIntegrityError("country and sector codes must be non-empty and unique")
    if negative_investment not in ("raise", "to_inventory"):
        raise UnsupportedExtension(f"negative_investment={negative_investment!r}; use 'raise' or 'to_inventory'")
    if reference_country is None:
        reference_country = ccodes[-1]
    if reference_country not in ccodes:
        raise DataIntegrityError(f"reference country {reference_country!r} is not in the registry")
    if not (np.isfinite(phantom) and phantom > 0 and np.isfinite(va_floor_rel) and 0 < va_floor_rel < 1):
        raise DataIntegrityError("phantom must be positive and va_floor_rel in (0, 1)")
    fd_in = tuple(str(c) for c in fd_codes)
    k = len(fd_in)
    Z = np.array(Z, dtype=np.float64, copy=True)
    F = np.array(F, dtype=np.float64, copy=True)
    VA = np.array(VA, dtype=np.float64, copy=True).reshape(-1)
    TLS_raw = np.array(TLS, dtype=np.float64, copy=True).reshape(-1)
    TFD = np.array(TFD, dtype=np.float64, copy=True)
    if F.ndim == 2 and F.shape == (m, n * k):
        F = F.reshape(m, n, k)
    if TFD.ndim == 1 and TFD.shape == (n * k,):
        TFD = TFD.reshape(n, k)
    if Z.shape != (m, m) or F.shape != (m, n, k) or VA.shape != (m,) or TLS_raw.shape != (m,) or TFD.shape != (n, k):
        raise DataIntegrityError(
            f"array shapes do not match {n} economies x {s} industries with {k} final-demand columns"
        )
    for name, arr in (("Z", Z), ("F", F), ("VA", VA), ("TLS", TLS_raw), ("TFD", TFD)):
        if not np.all(np.isfinite(arr)):
            raise DataIntegrityError(f"{name} contains NaN or Inf")
    if np.any(Z < 0):
        raise DataIntegrityError("negative intermediate flows are not admissible")
    if output is None:
        out = Z.sum(axis=1) + F.sum(axis=(1, 2))
    else:
        out = np.array(output, dtype=np.float64, copy=True).reshape(-1)
    if out.shape != (m,):
        raise DataIntegrityError("output column has the wrong length")
    if np.any(out < 0):
        raise DataIntegrityError("negative gross output")
    active = out > 0
    rel = np.abs(Z.sum(axis=1) + F.sum(axis=(1, 2)) - out)[active] / out[active]
    med = float(np.median(rel)) if rel.size else 0.0
    if med > ROW_IDENTITY_TOL:
        raise DataIntegrityError(
            f"median relative row identity error {med:.2e} exceeds {ROW_IDENTITY_TOL:.0e}; the table is not balanced"
        )
    zero = ~active
    junk = (np.abs(Z[zero]).sum() + np.abs(Z[:, zero]).sum() + np.abs(F[zero]).sum()
            + np.abs(VA[zero]).sum() + np.abs(TLS_raw[zero]).sum())
    if junk > 0:
        raise DataIntegrityError("a zero-output cell has non-zero entries")
    # The final-demand layout is validated before the sector registry, so a bad layout raises
    # DataIntegrityError whatever the sector codes are.
    F4, TFD4, valuables = _condense_final_demand(F, TFD, fd_in)
    if merchandise_mask is None:
        goods = infer_merchandise_mask(scodes)
    else:
        goods = np.asarray(merchandise_mask, dtype=bool)
        if goods.shape != (s,):
            raise DataIntegrityError(f"merchandise_mask must have shape {(s,)}")

    labels = [f"{c}_{x}" for c in ccodes for x in scodes]
    moved: list[str] = []
    neg_g = F4[:, :, 1] < 0
    if np.any(neg_g):
        cells = [f"{labels[i]}->{ccodes[d]}" for i, d in np.argwhere(neg_g)]
        if negative_investment == "raise":
            raise DataIntegrityError(
                f"{len(cells)} negative investment (G) cells, for example {cells[:3]}; pass "
                "negative_investment='to_inventory' to move them into the exogenous inventory category"
            )
        F4[:, :, 3] += np.minimum(F4[:, :, 1], 0.0)
        F4[:, :, 1] = np.maximum(F4[:, :, 1], 0.0)
        moved = cells
    F4 = np.ascontiguousarray(F4)

    country = np.repeat(np.arange(n), s)
    zero_idx = np.flatnonzero(zero)
    for i in zero_idx:
        F4[i, country[i], 0] = phantom
        VA[i] = phantom
    hit = active & (VA <= 0)
    floored = tuple((labels[i], float(VA[i]), float(va_floor_rel * out[i])) for i in np.flatnonzero(hit))
    VA[hit] = va_floor_rel * out[hit]

    tls = Z.sum(axis=1) + F4.sum(axis=(1, 2)) - Z.sum(axis=0) - VA
    dtls = tls - TLS_raw
    acc = np.where(hit, 0.0, dtls)
    world_abs_tls = float(np.abs(TLS_raw).sum())
    cell_scale = np.maximum(out, 1.0)
    acc_share = float(np.max(np.abs(acc[active]) / cell_scale[active])) if active.any() else 0.0
    if world_abs_tls > 0 and float(np.abs(acc).sum()) > DTLS_WORLD_TOL * world_abs_tls:
        raise DataIntegrityError(
            f"residual TLS changes the accounts by {np.abs(acc).sum():.3e}, more than "
            f"{DTLS_WORLD_TOL:.0e} of world |TLS| ({world_abs_tls:.3e})"
        )
    if acc_share > DTLS_CELL_TOL * (1.0 + 1e-9):
        raise DataIntegrityError(f"residual TLS moves a cell by {acc_share:.2%} of max(output, 1)")
    xn0, raw_sum = _net_exports(Z, F4, n, s)
    world_gdp = float(VA.sum() + tls.sum() + TFD4.sum())
    if abs(raw_sum) > XN_SUM_TOL * abs(world_gdp):
        raise DataIntegrityError(f"world sum of trade balances {raw_sum:.3e} exceeds {XN_SUM_TOL:.0e} of world GDP")

    report = TableBuildReport(
        source=str(source), year=None if year is None else int(year), unit=str(unit),
        n_countries=n, n_sectors=s, fd_codes_in=fd_in, row_identity_median_rel=med,
        n_zero_output=int(zero.sum()), phantoms=tuple(labels[i] for i in zero_idx), floored=floored,
        dtls_accounting_abs=float(np.abs(acc).sum()),
        dtls_floor_abs=float(np.abs(np.where(hit, dtls, 0.0)).sum()),
        dtls_accounting_max_share_of_output=acc_share, world_abs_tls=world_abs_tls,
        xn0_raw_sum=raw_sum, world_gdp=world_gdp,
        negative_fd_cells={c: int((F4[:, :, j] < 0).sum()) for j, c in enumerate(FD_CODES)},
        negative_investment=negative_investment, negative_investment_cells=tuple(moved),
        valuables_merged=valuables, phantom=float(phantom), va_floor_rel=float(va_floor_rel),
        metadata=dict(metadata or {}),
    )
    table = BalancedIOTable(
        Z=_readonly(Z), F=_readonly(F4), VA=_readonly(VA), TLS=_readonly(tls),
        TFD=_readonly(np.ascontiguousarray(TFD4)), country_codes=ccodes, sector_codes=scodes,
        fd_codes=FD_CODES, active=_readonly(active.copy()), merchandise_mask=_readonly(goods.copy()),
        report=report, reference_country=reference_country, metadata=dict(metadata or {}),
    )
    return table, report


__all__ = [
    "BalancedIOTable",
    "TableBuildReport",
    "infer_merchandise_mask",
    "table_from_arrays",
]
