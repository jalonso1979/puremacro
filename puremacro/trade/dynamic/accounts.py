"""Unaggregated source accounts for the dynamic sector-capital model.

:class:`DynamicAccounts` is a record of a published (or bundled) table, not a
stationary calibration. No empty production cell is replaced by a phantom, no
negative factor payment is floored, and no inventory or purchases-abroad flow
is treated as fixed investment. The stationary calibration
(:func:`puremacro.trade.dynamic.calibrate_dynamic`) reports every
transformation of these data explicitly.

Rows are seller country-industry cells and columns buyer cells, both in
country-major order (``cell = country_index * S + sector_index``). All money
values keep the source's units. Final uses are stored by seller cell,
destination country and the five contract categories
``("C", "G", "X", "V", "VAL")``: C = household + NPISH + government
consumption, G = gross fixed capital formation, X = residents' purchases
abroad, V = signed inventory changes, VAL = valuables. ``TFD`` carries the net
taxes on final purchases by destination and category in the same order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._results import _ReportMixin

__all__ = ["DynamicAccounts", "analytic_two_country_accounts"]

FINAL_CATEGORIES: tuple[str, ...] = ("C", "G", "X", "V", "VAL")
_CATEGORY_LONG_NAMES = {"C": "consumption", "G": "fixed_investment", "X": "purchases_abroad",
                        "V": "inventory_changes", "VAL": "valuables"}
# Source final-use codes that puremacro's loaders and the contract use, mapped
# to the five contract categories. Unknown codes must be mapped explicitly.
# The EXIOBASE loader's "EXPORT" column (exports to regions outside the table)
# is deliberately absent: it is not residents' purchases abroad, so a table
# carrying it needs an explicit ``fd_map`` (for example ``{"EXPORT": "X"}``).
KNOWN_FINAL_USE_CODES: dict[str, str] = {
    "C": "C", "HFCE": "C", "NPISH": "C", "GGFC": "C", "P3_S14": "C", "P3_S15": "C", "P3_S13": "C",
    "CONS_h": "C", "CONS_np": "C", "CONS_g": "C",
    "G": "G", "I": "G", "GFCF": "G", "P51G": "G",
    "X": "X", "Cx": "X", "DPABR": "X",
    "V": "V", "INVNT": "V", "INVT": "V", "P5M": "V",
    "VAL": "VAL", "VALUABLES": "VAL", "ACQ_VAL": "VAL",
}
_NEGATIVE_INVESTMENT_POLICIES = ("raise", "to_inventory")

# The OECD condensation (``_oecd_icio.condense_final_demand``) records
# ``Cx = DPABR``: only that mapping (or no mapping, the bundled 77x11 layout)
# makes the calibration's third final-use slot residents' purchases abroad.
_OECD_CX_CODES: tuple[str, ...] = ("DPABR",)
_CX_BRIDGE_CATEGORIES: tuple[str, ...] = ("X", "V", "VAL")


def _calibration_cx_category(calib: Any, cx_category: str | None = None, *,
                             caller: str = "from_trade_calibration") -> tuple[str, dict[str, Any]]:
    """Contract category of the ``Cx`` column of a ``TradeCalibrationResult``.

    Shared by the three ``from_trade_calibration`` bridges
    (:class:`DynamicAccounts`, ``puremacro.trade.mrio.MRIOTable`` and
    ``puremacro.trade.condensed.BalancedIOTable``). The calibration's third
    final-use slot means residents' direct purchases abroad (contract X) only
    when ``calib.metadata["final_use_mapping"]`` is absent (the bundled 77x11
    layout) or maps Cx to the OECD ``DPABR`` code. The FIGARO, EXIOBASE, WIOD
    and Eora loaders put signed changes in inventories and valuables there
    (``P5M``, ``INVNT``, ``INVT`` -> V; ``VALUABLES``, ``ACQ_VAL`` -> VAL; a
    column mixing both is stored as V, which then also holds valuables).
    EXIOBASE's ``EXPORT`` column, and any code outside
    :data:`KNOWN_FINAL_USE_CODES`, is refused unless ``cx_category`` names
    the category explicitly (``"X"``, ``"V"`` or ``"VAL"``).

    Returns the category and a record for the bridge's metadata.
    """
    meta = getattr(calib, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    mapping = meta.get("final_use_mapping")
    semantics = meta.get("final_use_semantics")
    cx_meaning = semantics.get("Cx") if isinstance(semantics, Mapping) else None
    record: dict[str, Any] = {"final_use_mapping": None if mapping is None else dict(mapping),
                              "cx_semantics": cx_meaning}
    if cx_category is not None:
        category = str(cx_category)
        if category not in _CX_BRIDGE_CATEGORIES:
            raise ValueError(f"{caller}: cx_category must be one of {_CX_BRIDGE_CATEGORIES} or None, "
                             f"got {cx_category!r}")
        record.update(cx_category=category, rule="explicit cx_category")
        return category, record
    if mapping is None:
        record.update(cx_category="X", rule="no final_use_mapping recorded (bundled 77x11 layout): "
                                            "Cx read as residents' purchases abroad")
        return "X", record
    remedy = ("pass cx_category='V' (signed inventories and valuables), 'VAL' or 'X' (tariff-exempt "
              "residents' purchases abroad) after checking calib.metadata['final_use_semantics'], or "
              "keep the native final-use categories: read the provider file with "
              "puremacro.trade.mrio.read_figaro_native, read_exiobase_native or read_oecd_native (they "
              "return an MRIOTable, accepted by DynamicAccounts.from_icio and BalancedIOTable.from_io_table), "
              "or, for a synthetic table, map generate_synthetic_mrio(...) explicitly with "
              "MRIOTable.from_raw(fd_mapping=...), DynamicAccounts.from_icio(fd_map=...) or "
              "BalancedIOTable.from_raw")
    codes_raw = mapping.get("Cx") if isinstance(mapping, Mapping) else None
    if codes_raw is None:
        raise ValueError(f"{caller}: calib.metadata['final_use_mapping'] = {mapping!r} does not describe the "
                         f"Cx column; {remedy}")
    codes = [str(codes_raw)] if isinstance(codes_raw, str) else [str(c) for c in codes_raw]
    if tuple(codes) == _OECD_CX_CODES:
        record.update(cx_category="X", rule="OECD mapping: Cx = DPABR (residents' purchases abroad)")
        return "X", record
    if "EXPORT" in codes:
        raise ValueError(f"{caller}: calib.metadata['final_use_mapping'] maps Cx to {codes}, which includes "
                         "EXIOBASE's 'EXPORT' column (exports to regions outside the table), so Cx is not "
                         f"residents' purchases abroad and cannot be split after condensation; {remedy}")
    targets = {KNOWN_FINAL_USE_CODES.get(code) for code in codes}
    if not codes or not targets <= {"V", "VAL"}:
        raise ValueError(f"{caller}: calib.metadata['final_use_mapping'] maps Cx to {codes}, which is neither "
                         f"the OECD DPABR column nor inventories and valuables; {remedy}")
    category = "V" if "V" in targets else "VAL"
    meaning = {frozenset({"V"}): "signed changes in inventories", frozenset({"VAL"}): "valuables",
               frozenset({"V", "VAL"}): "signed changes in inventories and valuables"}[frozenset(targets)]
    record.update(cx_category=category, rule=f"Cx = {'+'.join(codes)} ({meaning}) mapped to {category}")
    return category, record


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.array(value, dtype=float, copy=True)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def _map_final_uses(F: np.ndarray, TFD: np.ndarray, codes: Sequence[str],
                    fd_map: Mapping[str, str] | None) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Sum source final-use categories into the five contract categories."""
    codes = tuple(str(c) for c in codes)
    if len(set(codes)) != len(codes):
        raise ValueError("fd_codes must be unique")
    mapping = dict(KNOWN_FINAL_USE_CODES)
    if fd_map:
        mapping.update({str(k): str(v) for k, v in fd_map.items()})
    resolved: dict[str, str] = {}
    for code in codes:
        target = mapping.get(code)
        if target not in FINAL_CATEGORIES:
            raise ValueError(f"Final-use code {code!r} has no mapping onto {FINAL_CATEGORIES}; pass fd_map")
        resolved[code] = target
    M, N, K = F.shape
    if K != len(codes) or TFD.shape != (N, K):
        raise ValueError("F, TFD and fd_codes disagree on the number of final-use categories")
    out_F = np.zeros((M, N, len(FINAL_CATEGORIES)))
    out_T = np.zeros((N, len(FINAL_CATEGORIES)))
    for k, code in enumerate(codes):
        j = FINAL_CATEGORIES.index(resolved[code])
        out_F[:, :, j] += F[:, :, k]
        out_T[:, j] += TFD[:, k]
    return out_F, out_T, resolved


@dataclass(frozen=True)
class DynamicAccounts(_ReportMixin):
    """Source accounts at native resolution (section-2 IO table contract).

    ``accounting_report()`` returns the raw-account diagnostics as a dict and
    ``to_dataframe()`` (with ``to_markdown``/``to_latex``/``to_typst``) as a
    one-column table; nothing is rebalanced here.

    Attributes
    ----------
    country_codes, sector_codes : tuple of str
        Registries; ``M = len(country_codes) * len(sector_codes)``.
    Z : np.ndarray, shape (M, M)
        Basic-price intermediate deliveries, rows sellers, columns buyers.
    F : np.ndarray, shape (M, N, 5)
        Final deliveries by seller cell, destination country and category in
        ``fd_codes`` order ``("C", "G", "X", "V", "VAL")``.
    VA : np.ndarray, shape (M,)
        Value added as recorded (factor income plus production taxes).
    TLS : np.ndarray, shape (M,)
        Net taxes less subsidies on the buyer cell's intermediate purchases.
    production_taxes : np.ndarray, shape (M,)
        Other net taxes on production (zeros when the source has no detail).
    output : np.ndarray, shape (M,)
        Gross output.
    TFD : np.ndarray, shape (N, 5)
        Net taxes on final purchases by destination and category.
    labor_compensation, operating_surplus : np.ndarray or None, shape (M,)
        Factor detail when the source or bridge supplies it (both or neither).
    merchandise_mask : np.ndarray or None, shape (S,)
        Goods sectors (ISIC A-C) when known.
    dataset : str
        Source label.
    metadata : dict
        Provenance and every transformation applied by a bridge.
    """

    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    Z: np.ndarray
    F: np.ndarray
    VA: np.ndarray
    TLS: np.ndarray
    production_taxes: np.ndarray
    output: np.ndarray
    TFD: np.ndarray
    labor_compensation: np.ndarray | None = None
    operating_surplus: np.ndarray | None = None
    merchandise_mask: np.ndarray | None = None
    fd_codes: tuple[str, ...] = FINAL_CATEGORIES
    dataset: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- registries ---------------------------------------------------------
    @property
    def countries(self) -> tuple[str, ...]:
        """Alias of ``country_codes``."""
        return self.country_codes

    @property
    def sectors(self) -> tuple[str, ...]:
        """Alias of ``sector_codes``."""
        return self.sector_codes

    @property
    def n_countries(self) -> int:
        """Number of countries N."""
        return len(self.country_codes)

    @property
    def n_sectors(self) -> int:
        """Number of sectors S."""
        return len(self.sector_codes)

    @property
    def n_cells(self) -> int:
        """Number of source cells M = N * S (active or not)."""
        return self.n_countries * self.n_sectors

    @property
    def country_index(self) -> np.ndarray:
        """Country index of every cell (country-major order)."""
        return np.repeat(np.arange(self.n_countries), self.n_sectors)

    @property
    def cell_labels(self) -> tuple[str, ...]:
        """``country:sector`` labels of every cell in country-major order."""
        return tuple(f"{c}:{s}" for c in self.country_codes for s in self.sector_codes)

    # -- final uses ---------------------------------------------------------
    @property
    def final_consumption(self) -> np.ndarray:
        """(M, N) household, NPISH and government consumption."""
        return self.F[:, :, 0]

    @property
    def final_investment(self) -> np.ndarray:
        """(M, N) gross fixed capital formation."""
        return self.F[:, :, 1]

    @property
    def purchases_abroad(self) -> np.ndarray:
        """(M, N) residents' direct purchases abroad."""
        return self.F[:, :, 2]

    @property
    def inventory_changes(self) -> np.ndarray:
        """(M, N) signed changes in inventories."""
        return self.F[:, :, 3]

    @property
    def valuables(self) -> np.ndarray:
        """(M, N) acquisitions less disposals of valuables."""
        return self.F[:, :, 4]

    @property
    def final_demand(self) -> np.ndarray:
        """(M, N) all final deliveries."""
        return self.F.sum(axis=2)

    # -- factors ------------------------------------------------------------
    @property
    def active_mask(self) -> np.ndarray:
        """Positive reported output; this is not a factor-feasibility certificate."""
        return self.output > 0

    @property
    def factor_VA(self) -> np.ndarray:
        """Value added excluding production taxes (observed split when available)."""
        if self.labor_compensation is None:
            return self.VA - self.production_taxes
        return self.labor_compensation + self.operating_surplus

    @property
    def capital_share_available(self) -> np.ndarray:
        """Cells whose observed factor detail yields a capital share (positive factor VA, nonnegative parts)."""
        if self.operating_surplus is None:
            return np.zeros(self.n_cells, dtype=bool)
        return (self.factor_VA > 0) & (self.labor_compensation >= 0) & (self.operating_surplus >= 0)

    @property
    def capital_share(self) -> np.ndarray:
        """Observed operating surplus over factor value added (NaN when unavailable)."""
        out = np.full(self.n_cells, np.nan)
        if self.operating_surplus is not None:
            np.divide(self.operating_surplus, self.factor_VA, out=out, where=self.capital_share_available)
        return out

    def accounting_report(self) -> dict[str, Any]:
        """Independent raw-account diagnostics; nothing is rebalanced here."""
        row = self.Z.sum(axis=1) + self.final_demand.sum(axis=1)
        col = self.Z.sum(axis=0) + self.VA + self.TLS
        scale = np.maximum(np.abs(self.output), 1.0)
        out: dict[str, Any] = {
            "dataset": self.dataset, "n_countries": self.n_countries, "n_sectors": self.n_sectors,
            "n_cells": self.n_cells, "Z_nnz": int(np.count_nonzero(self.Z)),
            "active_cells": int(self.active_mask.sum()),
            "zero_output_cells": int((self.output == 0).sum()),
            "negative_output_cells": int((self.output < 0).sum()),
            "negative_consumption_entries": int((self.final_consumption < 0).sum()),
            "negative_investment_entries": int((self.final_investment < 0).sum()),
            "negative_inventory_entries": int((self.inventory_changes < 0).sum()),
            "nonpositive_active_factor_VA": int(((self.factor_VA <= 0) & self.active_mask).sum()),
            "unavailable_active_capital_shares": int((~self.capital_share_available & self.active_mask).sum()),
            "row_output_max_absolute_gap": float(np.max(np.abs(row - self.output), initial=0.0)),
            "row_output_max_scaled_gap": float(np.max(np.abs(row - self.output) / scale, initial=0.0)),
            "column_output_max_absolute_gap": float(np.max(np.abs(col - self.output), initial=0.0)),
            "column_output_max_scaled_gap": float(np.max(np.abs(col - self.output) / scale, initial=0.0)),
            "world_reported_value_added": float(self.VA.sum()),
            "world_product_taxes": float(self.TLS.sum() + self.TFD.sum()),
            "world_production_taxes": float(self.production_taxes.sum()),
            "final_category_totals": {code: float(self.F[:, :, k].sum())
                                      for k, code in enumerate(FINAL_CATEGORIES)},
        }
        if self.labor_compensation is not None:
            out["factor_decomposition_max_absolute_gap"] = float(np.max(np.abs(
                self.VA - self.factor_VA - self.production_taxes), initial=0.0))
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """The accounting report as a one-column table indexed by check name.

        Nested final-category totals appear as ``final_total_<code>`` rows.
        """
        report = self.accounting_report()
        totals = report.pop("final_category_totals")
        rows = list(report.items()) + [(f"final_total_{code}", value) for code, value in totals.items()]
        return pd.DataFrame({"value": [value for _, value in rows]},
                            index=pd.Index([key for key, _ in rows], name="check"))

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_arrays(cls, *, country_codes: Sequence[str], sector_codes: Sequence[str],
                    Z: Any, F: Any, VA: Any, TLS: Any, output: Any, TFD: Any,
                    fd_codes: Sequence[str] = FINAL_CATEGORIES,
                    production_taxes: Any | None = None,
                    labor_compensation: Any | None = None,
                    operating_surplus: Any | None = None,
                    merchandise_mask: Any | None = None,
                    fd_map: Mapping[str, str] | None = None,
                    negative_investment: str = "raise",
                    dataset: str = "arrays",
                    metadata: Mapping[str, Any] | None = None) -> "DynamicAccounts":
        """Build validated accounts from raw arrays (the section-2 contract).

        ``F`` may be ``(M, N, K)`` or destination-major ``(M, N*K)`` and
        ``TFD`` ``(N, K)`` or flat ``(N*K,)``, both in the order of
        ``fd_codes``. A two-dimensional ``TFD`` must be exactly ``(N, K)``
        (destination rows, category columns). A transposed ``(K, N)`` table
        is rejected when ``N != K``; when the number of countries equals the
        number of final-use codes the shapes coincide and the orientation
        cannot be detected, so pass destination rows and category columns.
        Codes are mapped onto the
        five contract categories through ``KNOWN_FINAL_USE_CODES`` (extended
        or overridden by ``fd_map``); categories that are absent become zeros.
        The EXIOBASE ``EXPORT`` column has no default mapping (it records
        exports to regions outside the table, not residents' purchases
        abroad) and must be mapped explicitly through ``fd_map``. ``Z`` may
        be dense or sparse.

        ``negative_investment="raise"`` rejects negative fixed-investment
        cells; ``"to_inventory"`` moves them into the signed inventory category
        (absorption by seller and destination is unchanged; the investment
        basket then excludes them) and records the cells and mass in metadata.
        Purchases abroad must be nonnegative. Tiny negative consumption and
        inventory entries are source data and are preserved.
        """
        if negative_investment not in _NEGATIVE_INVESTMENT_POLICIES:
            raise ValueError(f"negative_investment must be one of {_NEGATIVE_INVESTMENT_POLICIES}")
        countries = tuple(str(c) for c in country_codes)
        sectors = tuple(str(s) for s in sector_codes)
        if not countries or not sectors:
            raise ValueError("country and sector registries cannot be empty")
        if len(set(countries)) != len(countries) or len(set(sectors)) != len(sectors):
            raise ValueError("country and sector registries must have unique labels")
        N, S = len(countries), len(sectors)
        M = N * S
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        Z = _array(Z, (M, M), "Z")
        if (Z < 0).any():
            raise ValueError("Intermediate transactions must be nonnegative")
        codes = tuple(str(c) for c in fd_codes)
        K = len(codes)
        F = np.array(F, dtype=float, copy=True)
        if F.ndim == 2 and F.shape == (M, N * K):
            F = F.reshape(M, N, K)
        F = _array(F, (M, N, K), "F")
        TFD = np.array(TFD, dtype=float, copy=True)
        if TFD.ndim == 1 and TFD.size == N * K:
            TFD = TFD.reshape(N, K)
        TFD = _array(TFD, (N, K), "TFD")
        F5, T5, resolved = _map_final_uses(F, TFD, codes, fd_map)
        VA = _array(VA, (M,), "VA")
        TLS = _array(TLS, (M,), "TLS")
        output = _array(output, (M,), "output")
        production_taxes = (np.zeros(M) if production_taxes is None
                            else _array(production_taxes, (M,), "production_taxes"))
        if (labor_compensation is None) != (operating_surplus is None):
            raise ValueError("labor_compensation and operating_surplus must be supplied together")
        labor = None if labor_compensation is None else _array(labor_compensation, (M,), "labor_compensation")
        surplus = None if operating_surplus is None else _array(operating_surplus, (M,), "operating_surplus")
        if merchandise_mask is not None:
            mask = np.asarray(merchandise_mask)
            if mask.shape != (S,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("merchandise_mask must be a boolean sector vector")
            mask = mask.astype(bool)
        else:
            mask = None
        meta: dict[str, Any] = dict(metadata or {})
        meta["final_use_mapping"] = resolved
        negative = F5[:, :, 1] < 0
        if negative.any():
            count, mass = int(negative.sum()), float(F5[:, :, 1][negative].sum())
            if negative_investment == "raise":
                raise ValueError(f"{count} negative fixed-investment cells (mass {mass:.6g}); pass "
                                 "negative_investment='to_inventory' to move them into signed inventories")
            moved = np.where(negative, F5[:, :, 1], 0.0)
            F5[:, :, 3] += moved
            F5[:, :, 1] -= moved
            meta["negative_investment"] = {"policy": "to_inventory", "cells": count, "mass": mass,
                                           "note": "negative GFCF entries moved into signed inventory changes"}
        else:
            meta.setdefault("negative_investment", {"policy": negative_investment, "cells": 0, "mass": 0.0})
        if (F5[:, :, 2] < 0).any():
            raise ValueError("Negative residents' purchases abroad are not supported")
        return cls(country_codes=countries, sector_codes=sectors, Z=_readonly(Z), F=_readonly(F5),
                   VA=_readonly(VA), TLS=_readonly(TLS), production_taxes=_readonly(production_taxes),
                   output=_readonly(output), TFD=_readonly(T5),
                   labor_compensation=None if labor is None else _readonly(labor),
                   operating_surplus=None if surplus is None else _readonly(surplus),
                   merchandise_mask=None if mask is None else _readonly(mask),
                   fd_codes=FINAL_CATEGORIES, dataset=str(dataset), metadata=meta)

    @classmethod
    def from_icio(cls, icio: Any, *, negative_investment: str = "raise",
                  merchandise_sectors: Sequence[str] | str | None = "isic_a_c",
                  factor_split: tuple[float, float] | None = None,
                  labor_compensation: Any | None = None,
                  operating_surplus: Any | None = None,
                  fd_map: Mapping[str, str] | None = None) -> "DynamicAccounts":
        """Bridge from :class:`puremacro.trade.ICIOData`, :class:`RawIOData` or a section-2 table.

        * ``ICIOData`` (bundled 77x11 layout, final uses C, I, Cx): C maps to
          consumption, I to fixed investment and Cx to purchases abroad. The
          bundled I column already contains inventory changes and three cells
          are negative, so ``negative_investment`` must be chosen explicitly
          (``"to_inventory"`` keeps the flows in the consumption pool). The
          labour and capital rows of the bundled table are a mechanical
          2/3-1/3 split of value added, so capital shares are uniform 1/3.
          There is no production-tax detail: the TLS row is the whole tax.
        * ``RawIOData`` (loaders and ``generate_synthetic_mrio``): final-use
          codes are mapped through ``KNOWN_FINAL_USE_CODES``/``fd_map`` (the
          EXIOBASE ``EXPORT`` column needs an explicit ``fd_map`` entry).
          Without observed factor detail the split defaults to labour 2/3 and
          capital 1/3 of value added (``factor_split``), recorded in metadata,
          unless ``labor_compensation`` and ``operating_surplus`` are given.
        * Any object with ``Z, F, fd_codes, VA, TLS, TFD, output, country_codes,
          sector_codes`` (section-2 contract) is accepted as is.

        ``merchandise_sectors="isic_a_c"`` marks a sector as goods (ISIC
        sections A-C) when its code is one of the bundled labels ``AGRI``,
        ``MINQ``, ``MANU``, one of the 22 OECD ICIO 45-sector goods codes
        (``A01_02`` to ``C31T33``), or a code whose first letter is ``A``,
        ``B`` or ``C`` and that is either that single letter (the ISIC-11
        sections ``A``/``B``/``C``, FIGARO mining ``B``) or continues with a
        digit (``A01``, ``C10T12``), except FIGARO's ``C33`` (repair and
        installation of machinery, which is not a merchandise shipment).
        A section-2 table object that carries its own ``merchandise_mask``
        keeps that mask under the default. Other registries (for
        example EXIOBASE ``i01...`` codes) get no goods under this rule and
        need an explicit sequence. The rule and the resulting goods list are
        recorded in ``metadata["merchandise_sectors"]``. A sequence of sector
        codes marks exactly those; ``None`` leaves the mask undefined.
        """
        from puremacro.trade.data import ICIOData, RawIOData

        meta: dict[str, Any] = {}
        mask_rule = _MERCHANDISE_RULE if isinstance(merchandise_sectors, str) else "explicit sector list"
        if isinstance(icio, ICIOData):
            countries, sectors = tuple(icio.country_codes), tuple(icio.sector_codes)
            M, N = len(countries) * len(sectors), len(countries)
            codes = tuple(icio.fd_codes)
            Z = icio.intermediate_matrix
            F = np.asarray(icio.final_demand_matrix, dtype=float).reshape(M, N, len(codes))
            TFD = np.asarray(icio.net_taxes_final_demand, dtype=float).reshape(N, len(codes))
            TLS = np.asarray(icio.net_taxes_intermediate, dtype=float)
            labor = np.asarray(icio.labor_va, dtype=float)
            capital = np.asarray(icio.capital_va, dtype=float)
            VA = labor + capital
            output = np.asarray(icio.gross_output, dtype=float)
            production = np.zeros(M)
            meta.update(source="ICIOData", factor_detail="table rows n+1 (labour) and n+2 (capital); "
                        "the bundled 77x11 table splits value added 2/3-1/3 mechanically",
                        production_taxes="not separated in this layout; TLS row carries all taxes",
                        investment_note="the bundled I column merges gross fixed capital formation and inventories")
            dataset = "OECD_ICIO_bundled" if (M, N) == (847, 77) else "ICIOData"
        elif isinstance(icio, RawIOData):
            countries, sectors = tuple(str(c) for c in icio.countries), tuple(str(s) for s in icio.sectors)
            M, N = len(countries) * len(sectors), len(countries)
            codes = tuple(str(c) for c in icio.fd_categories)
            Z = np.asarray(icio.intermediate_matrix, dtype=float)
            F = np.asarray(icio.final_demand_matrix, dtype=float).reshape(M, N, len(codes))
            if icio.taxes_less_subsidies_fd is None:
                TFD = np.zeros((N, len(codes)))
                meta["final_taxes"] = "source has no final-use tax detail; TFD set to zero"
            else:
                TFD = np.asarray(icio.taxes_less_subsidies_fd, dtype=float).reshape(N, len(codes))
            VA_raw = np.asarray(icio.value_added, dtype=float)
            VA = VA_raw.sum(axis=0) if VA_raw.ndim == 2 else VA_raw
            TLS = np.asarray(icio.taxes_less_subsidies, dtype=float)
            output = np.asarray(icio.gross_output, dtype=float)
            production = np.zeros(M)
            if labor_compensation is None and operating_surplus is None:
                share = (2.0 / 3.0, 1.0 / 3.0) if factor_split is None else tuple(float(x) for x in factor_split)
                if len(share) != 2 or not all(np.isfinite(share)) or any(x <= 0 for x in share) or abs(sum(share) - 1) > 1e-12:
                    raise ValueError("factor_split must be two positive shares summing to one")
                labor, capital = share[0] * VA, share[1] * VA
                meta["factor_detail"] = f"assumed labour/capital split {share[0]:.6g}/{share[1]:.6g} of value added"
            else:
                labor, capital = labor_compensation, operating_surplus
                meta["factor_detail"] = "user-supplied labour and capital rows"
            meta.update(source=f"RawIOData:{icio.source}", year=int(icio.year), unit=icio.unit,
                        source_fd_categories=list(codes))
            dataset = str(icio.source or "RawIOData")
        elif all(hasattr(icio, name) for name in ("Z", "F", "fd_codes", "VA", "TLS", "TFD", "output",
                                                    "country_codes", "sector_codes")):
            countries, sectors = tuple(icio.country_codes), tuple(icio.sector_codes)
            codes = tuple(icio.fd_codes)
            Z, F, TFD = icio.Z, icio.F, icio.TFD
            VA, TLS, output = icio.VA, icio.TLS, icio.output
            production = getattr(icio, "production_taxes", None)
            labor = labor_compensation if labor_compensation is not None else getattr(icio, "labor_compensation", None)
            capital = operating_surplus if operating_surplus is not None else getattr(icio, "operating_surplus", None)
            meta.update(source=type(icio).__name__, **dict(getattr(icio, "metadata", {}) or {}))
            dataset = str(getattr(icio, "dataset", type(icio).__name__))
            if merchandise_sectors == "isic_a_c" and getattr(icio, "merchandise_mask", None) is not None:
                merchandise_sectors = tuple(s for s, g in zip(sectors, icio.merchandise_mask) if g)
                mask_rule = "the source table's merchandise_mask"
        else:
            raise TypeError("from_icio expects ICIOData, RawIOData or a section-2 IO table object")
        mask = _merchandise(sectors, merchandise_sectors)
        if mask is not None:
            meta["merchandise_sectors"] = {"rule": mask_rule, "goods": [s for s, g in zip(sectors, mask) if g]}
        return cls.from_arrays(country_codes=countries, sector_codes=sectors, Z=Z, F=F, VA=VA, TLS=TLS,
                               output=output, TFD=TFD, fd_codes=codes, production_taxes=production,
                               labor_compensation=labor, operating_surplus=capital, merchandise_mask=mask,
                               fd_map=fd_map, negative_investment=negative_investment,
                               dataset=dataset, metadata=meta)

    @classmethod
    def from_trade_calibration(cls, calib: Any, *, negative_investment: str = "raise",
                               merchandise_sectors: Sequence[str] | str | None = "isic_a_c",
                               cx_category: str | None = None) -> "DynamicAccounts":
        """Bridge from :class:`puremacro.trade.TradeCalibrationResult`.

        Uses ``calib.data_calibra`` (the reconstructed benchmark table, rows
        ``n..n+2`` = TLS, labour, capital; final-use columns ``n:`` = C, I, Cx
        per destination). G therefore contains inventory changes, three cells
        of the bundled table are negative, and ``negative_investment`` must be
        chosen explicitly (``"raise"`` or ``"to_inventory"``). Capital shares
        come from the labour/capital rows (uniform 1/3 on the bundled table).

        Cx becomes X (residents' purchases abroad) only for the bundled layout
        (no ``calib.metadata["final_use_mapping"]``) or the OECD ``DPABR``
        mapping; the FIGARO, WIOD and Eora loaders' Cx (signed inventories and
        valuables) becomes V, and EXIOBASE's Cx (which includes its export
        column) or an unrecognised mapping raises ``ValueError`` unless
        ``cx_category`` (``"X"``, ``"V"`` or ``"VAL"``) is given. The decision
        is recorded in ``metadata["final_use_bridge"]``.
        """
        from puremacro.trade._results import TradeCalibrationResult
        from puremacro.trade.data import ICIOData

        if not isinstance(calib, TradeCalibrationResult):
            raise TypeError("calib must be a TradeCalibrationResult")
        if calib.data_calibra is None:
            raise ValueError("calib.data_calibra is missing; calibrate with the reconstructed table available")
        nc, ns, nfd = int(calib.n_countries), int(calib.n_sectors), int(calib.n_final_demand)
        countries = tuple(calib.country_codes) if calib.country_codes else tuple(f"C{i:02d}" for i in range(nc))
        sectors = tuple(calib.sector_codes) if calib.sector_codes else tuple(f"S{i:02d}" for i in range(ns))
        if nfd == 3:
            codes: tuple[str, ...] = ("C", "I", "Cx")
        else:
            raise ValueError("from_trade_calibration expects the three-category C/I/Cx layout")
        table = np.asarray(calib.data_calibra, dtype=float)
        if table.shape != (ns * nc + 3, ns * nc + nfd * nc):
            raise ValueError("data_calibra has an unexpected shape for this calibration")
        cx_target, bridge_record = _calibration_cx_category(
            calib, cx_category, caller="DynamicAccounts.from_trade_calibration")
        icio = ICIOData(matrix=table, country_codes=countries, sector_codes=sectors, fd_codes=codes)
        accounts = cls.from_icio(icio, negative_investment=negative_investment,
                                 merchandise_sectors=merchandise_sectors,
                                 fd_map=None if cx_target == "X" else {"Cx": cx_target})
        meta = dict(accounts.metadata)
        mapping = bridge_record["final_use_mapping"]
        if mapping is None:
            investment_note = "G is the bundled I column: gross fixed capital formation plus inventories"
        else:
            investment_note = f"G is the calibration's I column (source codes {mapping.get('I')})"
        meta.update(source="TradeCalibrationResult.data_calibra",
                    unit=calib.metadata.get("unit", "calibration value units"),
                    investment_note=investment_note, final_use_bridge=bridge_record)
        return cls(**{**accounts.__dict__, "metadata": meta, "dataset": "TradeCalibrationResult"})


_MERCHANDISE_RULE = ("isic_a_c: bundled AGRI/MINQ/MANU, OECD 45-sector goods codes, or a code starting "
                     "with A, B or C that is that single letter or continues with a digit, except C33")
_BUNDLED_GOODS = frozenset({"AGRI", "MINQ", "MANU"})
_OECD45_GOODS = frozenset({"A01_02", "A03", "B05_06", "B07_08", "B09", "C10T12", "C13T15", "C16", "C17_18",
                           "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30",
                           "C31T33"})
_NOT_MERCHANDISE = frozenset({"C33"})  # FIGARO repair and installation of machinery


def _is_isic_a_c(code: str) -> bool:
    if code in _BUNDLED_GOODS or code in _OECD45_GOODS:
        return True
    if code in _NOT_MERCHANDISE or code[:1] not in ("A", "B", "C"):
        return False
    return len(code) == 1 or code[1].isdigit()


def _merchandise(sectors: Sequence[str], spec: Sequence[str] | str | None) -> np.ndarray | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec != "isic_a_c":
            raise ValueError("merchandise_sectors must be 'isic_a_c', a sequence of sector codes or None")
        return np.array([_is_isic_a_c(str(s)) for s in sectors], dtype=bool)
    chosen = set(str(s) for s in spec)
    unknown = chosen - set(sectors)
    if unknown:
        raise ValueError(f"Unknown merchandise sectors {sorted(unknown)}")
    return np.array([s in chosen for s in sectors], dtype=bool)


def analytic_two_country_accounts() -> DynamicAccounts:
    """Two-country, two-sector analytic accounts with separately identifiable flows.

    A software fixture, never empirical evidence: it carries distinct
    consumption, investment, purchases-abroad, inventory and valuables flows,
    distinct product and production taxes, and cell-specific capital shares
    ``(.35, .48, .30, .52)``. Copied from the IO engine's test fixture
    (``dynamic_model/tests/test_native_dynamics.py::analytic_native_data``).
    """
    z = np.array([[7, 3, 1, 2], [2, 9, 2, 1], [1, 2, 8, 3], [2, 1, 3, 7]], dtype=float)
    consumption = np.array([[22, 3], [20, 4], [3, 22], [4, 20]], dtype=float)
    investment = np.array([[12, 2], [10, 3], [3, 11], [2, 10]], dtype=float)
    tourism = np.array([[0, .5], [0, .3], [.4, 0], [.2, 0]])
    inventory = np.array([[.4, 0], [.2, 0], [0, .3], [0, .2]])
    valuables = np.array([[.1, 0], [0, .1], [0, .1], [.1, 0]])
    output = z.sum(axis=1) + (consumption + investment + tourism + inventory + valuables).sum(axis=1)
    product_tax = np.array([.3, .5, .4, .2])
    production_tax = np.array([.4, .3, .2, .5])
    va = output - z.sum(axis=0) - product_tax
    factor_va = va - production_tax
    alpha = np.array([.35, .48, .30, .52])
    F = np.stack([consumption, investment, tourism, inventory, valuables], axis=2)
    return DynamicAccounts.from_arrays(
        country_codes=("AAA", "BBB"), sector_codes=("S1", "S2"), Z=z, F=F, fd_codes=FINAL_CATEGORIES,
        VA=va, TLS=product_tax, production_taxes=production_tax, output=output,
        TFD=np.array([[1.3, .8, .1, .02, .03], [1.1, .7, .08, .01, .02]]),
        labor_compensation=(1 - alpha) * factor_va, operating_surplus=alpha * factor_va,
        merchandise_mask=np.array([True, False]), dataset="analytic_test_fixture",
        metadata={"source_kind": "analytic software fixture; not an empirical database"})
