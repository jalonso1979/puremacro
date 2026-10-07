"""Tariff wedges of the condensed model.

Ad-valorem import duties ``r >= 0`` become gross multipliers ``tau = 1 + r``
on every purchase of a foreign cell by the importing country: intermediate
purchases (``tau``, shape (M, M), column = buying cell), endogenous final
demand (``tau_fd``, shape (M, N, 3) for C, G, X) and inventories (``tau_V``,
shape (M, N)). Domestic transactions always carry ``tau = 1`` and residents'
purchases abroad (X) are never tariffed (``tau_fd[:, :, 2] == 1``), as in the
IO engine. Duties are collected by the importing country and
rebated lump-sum through its income equation.

This module replaces the IO engine's scenario registry, policy JSON loaders
and 45-to-11 coarse rules with explicit rate tables: pass a scalar, an
``(N, S)`` origin-by-sector table for one importer, an ``(N, N, S)`` array
``rates[origin, importer, sector]``, a mapping ``{importer: ...}`` or a
callable ``rate(origin_code, importer_code, sector_code)`` such as
``puremacro.trade.scenarios.TariffScenario.get_rate``. In the mapping form
``{importer: {origin | "*": rate}}`` a named origin always overrides ``"*"``,
whatever the order of the keys.

Multipliers below one (import subsidies) are rejected: with every
``tau >= 1`` the cost matrix ``B_tau(lam) = a (1 + lam (tau - 1)) / (1 - t)`` is
entrywise nondecreasing in the tariff scale, which is what lets one existence
certificate at ``lam = 1`` cover every continuation stage.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

from .errors import DataIntegrityError, UnsupportedExtension
from .table import ENDOGENOUS_FD

FD_TARIFFABLE: tuple[str, ...] = ("C", "G", "V")
"""Final-demand categories that may carry import duties; X (purchases abroad) never does."""


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=float)
    a.flags.writeable = False
    return a


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class TariffWedges:
    """Gross tariff multipliers of one experiment.

    Attributes
    ----------
    tau : np.ndarray, shape (M, M)
        Intermediate multipliers ``tau_ij = 1 + r`` (seller cell i, buyer cell j).
    tau_fd : np.ndarray, shape (M, N, 3)
        Final-demand multipliers for C, G, X; the X slice is identically one.
    tau_V : np.ndarray, shape (M, N)
        Inventory multipliers.
    rates : np.ndarray, shape (N, N, S)
        Ad-valorem rates ``rates[origin, importer, sector]`` behind the
        multipliers (zeros when the wedges were given directly).
    sha256 : str
        Digest of the four arrays, for provenance.
    metadata : dict
        Coverage, tariffed categories and inventory treatment.
    """

    tau: np.ndarray
    tau_fd: np.ndarray
    tau_V: np.ndarray
    rates: np.ndarray
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        """Number of country-industry cells ``M``."""
        return int(self.tau.shape[0])

    @property
    def n_countries(self) -> int:
        """Number of economies ``N``."""
        return int(self.tau_fd.shape[1])

    @property
    def n_sectors(self) -> int:
        """Number of industries per economy ``S = M / N``."""
        return self.n_cells // self.n_countries

    @property
    def is_free_trade(self) -> bool:
        """True when every multiplier equals one."""
        return bool(np.all(self.tau == 1.0) and np.all(self.tau_fd == 1.0) and np.all(self.tau_V == 1.0))

    @property
    def delta(self) -> np.ndarray:
        """Intermediate additional duties ``tau - 1``."""
        return self.tau - 1.0

    @property
    def delta_fd(self) -> np.ndarray:
        """Final-demand additional duties ``tau_fd - 1``."""
        return self.tau_fd - 1.0

    def rate_table(self, importer: int) -> np.ndarray:
        """Ad-valorem rates ``(N_origin, S)`` levied by one importer (index)."""
        return np.array(self.rates[:, int(importer), :], copy=True)

    @classmethod
    def free_trade(cls, calib: Any) -> "TariffWedges":
        """All multipliers equal to one for a calibration (or any object with ``n_cells``, ``n_countries``, ``n_sectors``)."""
        m, n, s = int(calib.n_cells), int(calib.n_countries), int(calib.n_sectors)
        return cls.from_arrays(np.ones((m, m)), np.ones((m, n, 3)), np.ones((m, n)),
                               rates=np.zeros((n, n, s)), metadata={"coverage": "none"})

    @classmethod
    def from_arrays(cls, tau: np.ndarray, tau_fd: np.ndarray, tau_V: np.ndarray, *,
                    rates: np.ndarray | None = None, metadata: Mapping[str, Any] | None = None) -> "TariffWedges":
        """Wrap explicit multiplier arrays after validating shapes, finiteness and positivity.

        Every multiplier must be at least one (ad-valorem rates ``r >= 0``;
        import subsidies are outside the model), domestic entries must equal
        one and the X slice of ``tau_fd`` must be identically one. ``rates``
        defaults to zeros of shape (N, N, S).

        Raises
        ------
        DataIntegrityError
            Wrong shapes, non-finite or non-positive entries, a multiplier
            below one (the message names the first offending entry) or a duty
            on a domestic transaction.
        UnsupportedExtension
            A duty on residents' purchases abroad (X).
        """
        tau = np.array(tau, dtype=float, copy=True)
        tau_fd = np.array(tau_fd, dtype=float, copy=True)
        tau_V = np.array(tau_V, dtype=float, copy=True)
        if tau.ndim != 2 or tau.shape[0] != tau.shape[1]:
            raise DataIntegrityError("tau must be square (M, M)")
        m = tau.shape[0]
        if tau_fd.ndim != 3 or tau_fd.shape[0] != m or tau_fd.shape[2] != 3:
            raise DataIntegrityError("tau_fd must have shape (M, N, 3)")
        n = tau_fd.shape[1]
        if n == 0 or m % n != 0:
            raise DataIntegrityError("the number of cells must be a multiple of the number of countries")
        s = m // n
        if tau_V.shape != (m, n):
            raise DataIntegrityError("tau_V must have shape (M, N)")
        for name, arr in (("tau", tau), ("tau_fd", tau_fd), ("tau_V", tau_V)):
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise DataIntegrityError(f"{name} must be finite and strictly positive")
        for name, arr in (("tau", tau), ("tau_fd", tau_fd), ("tau_V", tau_V)):
            if np.any(arr < 1.0):
                idx = tuple(int(i) for i in np.argwhere(arr < 1.0)[0])
                raise DataIntegrityError(
                    f"{name}{list(idx)} = {arr[idx]:.6g} is below one: import subsidies (negative ad-valorem "
                    "rates) are not part of the condensed model; multipliers must be 1 + r with r >= 0"
                )
        if np.any(tau_fd[:, :, ENDOGENOUS_FD.index("X")] != 1.0):
            raise UnsupportedExtension("tariffs on residents' purchases abroad (X) are not part of the model")
        country = np.repeat(np.arange(n), s)
        for k in range(n):
            own = country == k
            if (np.any(tau[np.ix_(own, own)] != 1.0) or np.any(tau_fd[own, k, :] != 1.0)
                    or np.any(tau_V[own, k] != 1.0)):
                raise DataIntegrityError("domestic transactions cannot bear import duties (tau must be one)")
        if rates is None:
            rates = np.zeros((n, n, s))
        rates = np.array(rates, dtype=float, copy=True)
        if rates.shape != (n, n, s) or not np.all(np.isfinite(rates)):
            raise DataIntegrityError("rates must be a finite array of shape (N, N, S)")
        return cls(tau=_readonly(tau), tau_fd=_readonly(tau_fd), tau_V=_readonly(tau_V),
                   rates=_readonly(rates), sha256=_digest(tau, tau_fd, tau_V, rates),
                   metadata=dict(metadata or {}))

    def to_dataframe(self, country_codes: Sequence[str] | None = None,
                     sector_codes: Sequence[str] | None = None) -> pd.DataFrame:
        """Nonzero ad-valorem rates, one row per (origin, importer, sector).

        Labels default to ``metadata["country_codes"]`` and
        ``metadata["sector_codes"]`` (recorded by :func:`build_tariff_wedges`),
        else to integer indices. Wedges built directly with
        :meth:`from_arrays` and no ``rates`` give an empty table.
        """
        n, s = self.n_countries, self.n_sectors
        cc = list(country_codes if country_codes is not None else self.metadata.get("country_codes", range(n)))
        sc = list(sector_codes if sector_codes is not None else self.metadata.get("sector_codes", range(s)))
        if len(cc) != n or len(sc) != s:
            raise ValueError(f"expected {n} country codes and {s} sector codes")
        rows = [(cc[o], cc[d], sc[j], float(self.rates[o, d, j])) for o, d, j in np.argwhere(self.rates != 0.0)]
        return pd.DataFrame(rows, columns=["origin", "importer", "sector", "rate"])

    def summary(self) -> str:
        """One line: the number of nonzero rates, their range and the non-unit multipliers."""
        nz = self.rates[self.rates != 0.0]
        span = f"rates in [{nz.min():.4g}, {nz.max():.4g}]" if nz.size else "no nonzero rates recorded"
        dutiable = int((self.tau != 1.0).sum() + (self.tau_fd != 1.0).sum() + (self.tau_V != 1.0).sum())
        return (f"TariffWedges: {nz.size} nonzero (origin, importer, sector) rates, {span}; {dutiable} "
                f"multipliers above one; coverage {self.metadata.get('coverage', 'explicit')}")

    def to_markdown(self, **kwargs: Any) -> str:
        """The rate table of :meth:`to_dataframe` as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """The rate table of :meth:`to_dataframe` as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """The rate table of :meth:`to_dataframe` as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)

    def scaled(self, lam: float) -> "TariffWedges":
        """Wedges ``1 + lam * (tau - 1)`` for the continuation parameter ``lam``."""
        lam = float(lam)
        if not np.isfinite(lam):
            raise ValueError("lam must be finite")
        return TariffWedges(
            tau=_readonly(1.0 + lam * (self.tau - 1.0)), tau_fd=_readonly(1.0 + lam * (self.tau_fd - 1.0)),
            tau_V=_readonly(1.0 + lam * (self.tau_V - 1.0)), rates=_readonly(lam * self.rates),
            sha256="" if lam != 1.0 else self.sha256, metadata={**self.metadata, "lambda": lam},
        )


def scale_wedges(wedges: TariffWedges, lam: float) -> TariffWedges:
    """Scale every additional duty by ``lam``: ``tau(lam) = 1 + lam * (tau - 1)``."""
    return wedges.scaled(lam)


def _resolve_rates(calib: Any, rates: Any, importer: str | None) -> tuple[np.ndarray, bool]:
    """Turn any accepted specification into a finite (N, N, S) rate array.

    Returns the array and whether the specification named sectors explicitly
    (an array with a sector axis, an ``(S,)`` array inside a mapping, or a
    callable); scalar specifications mean "every covered sector".
    """
    codes = tuple(calib.country_codes)
    sectors = tuple(calib.sector_codes)
    n, s = len(codes), len(sectors)
    full = np.zeros((n, n, s))
    explicit = {"sectors": False}

    def fill_importer(d: int, spec: Any) -> None:
        if isinstance(spec, Mapping):
            # "*" is applied first so a named origin always overrides it, whatever the key order.
            ordered = sorted(spec.items(), key=lambda item: item[0] != "*")
            for origin, value in ordered:
                value = np.asarray(value, dtype=float)
                if value.ndim == 0:
                    value = np.full(s, float(value))
                else:
                    explicit["sectors"] = True
                if value.shape != (s,):
                    raise DataIntegrityError(f"a per-origin rate must be a scalar or an array of shape {(s,)}")
                targets = [o for o in range(n) if o != d] if origin == "*" else [calib.index(str(origin))]
                for o in targets:
                    full[o, d, :] = value
            return
        arr = np.asarray(spec, dtype=float)
        if arr.ndim == 0:
            full[:, d, :] = float(arr)
        elif arr.shape == (n, s):
            full[:, d, :] = arr
            explicit["sectors"] = True
        elif arr.shape == (n,):
            full[:, d, :] = arr[:, None]
        else:
            raise DataIntegrityError(f"rates for one importer must be a scalar, (N,), (N, S) or a mapping; got {arr.shape}")

    if callable(rates):
        explicit["sectors"] = True
        for o, oc in enumerate(codes):
            for d, dc in enumerate(codes):
                if o == d:
                    continue
                for j, sc in enumerate(sectors):
                    full[o, d, j] = float(rates(oc, dc, sc))
    elif isinstance(rates, Mapping):
        if importer is not None:
            raise ValueError("pass importer only with scalar or array rates; a mapping names its importers")
        for dest, spec in rates.items():
            fill_importer(calib.index(str(dest)), spec)
    else:
        arr = np.asarray(rates, dtype=float)
        if arr.ndim == 3:
            if arr.shape != (n, n, s):
                raise DataIntegrityError(f"a full rate array must have shape {(n, n, s)}")
            full = np.array(arr, copy=True)
            explicit["sectors"] = True
        else:
            if importer is None:
                raise ValueError("importer is required with scalar or (N, S) rates")
            fill_importer(calib.index(str(importer)), arr)
    for k in range(n):
        full[k, k, :] = 0.0
    if not np.all(np.isfinite(full)):
        raise DataIntegrityError("tariff rates must be finite")
    if np.any(full < 0):
        raise ValueError("ad-valorem tariff rates must be nonnegative (0 = free trade)")
    return full, explicit["sectors"]


def build_tariff_wedges(
    calib: Any,
    rates: Any,
    *,
    importer: str | None = None,
    coverage: str = "goods",
    fd_tariffed: Sequence[str] = FD_TARIFFABLE,
    inventories: str = "tariffed",
    metadata: Mapping[str, Any] | None = None,
) -> TariffWedges:
    """Build the wedge arrays of an experiment from explicit ad-valorem rates.

    Parameters
    ----------
    calib : CondensedCalibration
        Supplies the registries and the goods mask ``calib.goods``.
    rates : float, array, mapping or callable
        ``float``: one rate on every foreign origin and covered sector into
        ``importer``. ``(N, S)`` array: rates by origin and sector into
        ``importer``. ``(N, N, S)`` array: ``rates[origin, importer, sector]``.
        Mapping ``{importer_code: spec}`` where ``spec`` is a float, an ``(N,)``
        or ``(N, S)`` array, or ``{origin_code | "*": float | (S,) array}``;
        ``"*"`` means every foreign origin and a named origin overrides it
        whatever the key order.
        Callable ``rate(origin_code, importer_code, sector_code) -> float``
        (for example ``TariffScenario.get_rate``).
    importer : str, optional
        Importing country for scalar or ``(N, S)`` rates.
    coverage : {"goods", "all"}
        ``"goods"`` keeps duties on merchandise sectors only (``calib.goods``).
        A sector-explicit specification (an array with a sector axis, an
        ``(S,)`` array inside a mapping, or a callable) that places duties on
        other sectors is trimmed with a ``RuntimeWarning`` and the trimmed
        sectors are recorded in ``metadata["dropped_service_rates"]``; scalar
        rates mean "every covered sector" and are restricted silently.
    fd_tariffed : sequence of str
        Subset of ``("C", "G", "V")``; ``"X"`` raises :class:`UnsupportedExtension`.
    inventories : {"tariffed", "untariffed"}
        Whether inventory purchases (V) carry the duty when ``"V"`` is tariffed.

    Returns
    -------
    TariffWedges

    Raises
    ------
    ValueError
        Negative rates or a missing importer.
    UnsupportedExtension
        Tariffs on X, unknown coverage or inventory treatment.
    """
    if coverage not in ("goods", "all"):
        raise UnsupportedExtension(f"coverage must be 'goods' or 'all', got {coverage!r}")
    if inventories not in ("tariffed", "untariffed"):
        raise UnsupportedExtension(f"inventories must be 'tariffed' or 'untariffed', got {inventories!r}")
    cats = tuple(str(c) for c in fd_tariffed)
    if "X" in cats:
        raise UnsupportedExtension("tariffs on residents' purchases abroad (X) are not part of the model")
    for c in cats:
        if c not in FD_TARIFFABLE:
            raise UnsupportedExtension(f"unknown final-demand category {c!r} in fd_tariffed")
    n, s, m = int(calib.n_countries), int(calib.n_sectors), int(calib.n_cells)
    full, sector_explicit = _resolve_rates(calib, rates, importer)
    dropped: tuple[str, ...] = ()
    if coverage == "goods":
        goods = np.asarray(calib.goods, dtype=bool)
        if sector_explicit:
            hit = np.any(full[:, :, ~goods] != 0.0, axis=(0, 1))
            if np.any(hit):
                dropped = tuple(str(c) for c, h in zip(np.asarray(calib.sector_codes)[~goods], hit) if h)
                warnings.warn(
                    f"coverage='goods' dropped explicit duties on non-merchandise sectors {list(dropped)}; "
                    "pass coverage='all' to keep them",
                    RuntimeWarning, stacklevel=2,
                )
        full[:, :, ~goods] = 0.0
    tau = np.ones((m, m))
    tau_fd = np.ones((m, n, 3))
    tau_V = np.ones((m, n))
    for d in range(n):
        ri = full[:, d, :].reshape(m)
        cols = slice(d * s, (d + 1) * s)
        tau[:, cols] = 1.0 + ri[:, None]
        for cat_idx, cat in enumerate(ENDOGENOUS_FD):
            if cat in cats:
                tau_fd[:, d, cat_idx] = 1.0 + ri
        if "V" in cats and inventories == "tariffed":
            tau_V[:, d] = 1.0 + ri
    tau_fd[:, :, ENDOGENOUS_FD.index("X")] = 1.0
    meta = {"coverage": coverage, "fd_tariffed": cats, "inventories": inventories,
            "country_codes": tuple(str(c) for c in calib.country_codes),
            "sector_codes": tuple(str(c) for c in calib.sector_codes),
            "importers": tuple(calib.country_codes[d] for d in range(n) if np.any(full[:, d, :] != 0)),
            "dropped_service_rates": dropped}
    meta.update(dict(metadata or {}))
    return TariffWedges.from_arrays(tau, tau_fd, tau_V, rates=full, metadata=meta)


__all__ = ["TariffWedges", "build_tariff_wedges", "scale_wedges"]
