"""Provider-agnostic scaffolding for real-time (vintage) data.

Every real-time repository in :mod:`puremacro.fetch.realtime` — ALFRED,
the EABCN/ECB RTD, Bundesbank, ONS, Statistics Canada, Banxico/INEGI —
lands on the same tidy shape::

    country  variable  date        vintage     value   provider  series_id

``date`` is the reference period the number describes; ``vintage`` is
the edition the number was published in. One (country, variable, date)
therefore has as many rows as there are editions of it.

WHAT ``vintage`` MEANS IS PROVIDER-SPECIFIC
-------------------------------------------
This is the subtlety that silently corrupts cross-country work, so it
is carried in the data rather than left to the reader's memory:
:data:`VINTAGE_SEMANTICS` records, per provider, what its vintage date
actually is, and :meth:`VintagePanel.vintage_semantics` reports it for
whatever is in a given panel.

For ordering editions — first release, second release, latest — every
provider's dates are equally usable. For dating a *release event*
against, say, a monetary-policy announcement, only providers whose
vintage is the national statistical office's own publication date will
do. Mixing the two without noticing is how a study ends up claiming a
central bank reacted to a number that had not been published yet.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...vintages import (
    MankiwShapiroResult,
    mankiw_shapiro,
    revision_frame,
    revision_triangle,
)


VINTAGE_COLUMNS = [
    "country", "variable", "date", "vintage", "value", "provider",
    "series_id", "units",
]


#: What each provider's ``vintage`` date actually denotes. Read this
#: before using a vintage date as an event date.
VINTAGE_SEMANTICS: dict[str, str] = {
    "alfred": (
        "ALFRED's release date, set by a documented three-step fallback: "
        "the source's own release date when the source provides one; "
        "otherwise the date given by FRED's data providers; otherwise "
        "the date the series first appeared in FRED. Usually the real "
        "release date for well-documented foreign series, but the "
        "payload does not say which branch produced any given date. "
        "Reliable for ordering editions; check the release's "
        "download-dates page before using it as an event date."
    ),
    "ecb_rtd": (
        "VALID_FROM of the observation version — the timestamp at which "
        "the ECB Data Portal disseminated that value. Tracks the ECB "
        "Economic Bulletin cycle, not a national release date."
    ),
    "bundesbank": (
        "The genuine publication date, carried as three dimensions of "
        "the series key (release year / month / day). Day codes 40 and "
        "41 mean the exact day is unknown and are mapped to the first "
        "of the named month."
    ),
    "ons": (
        "Publication MONTH plus release stage (1st / M1 / M2 / QNA) of "
        "the workbook column — not an exact release day. The day "
        "component is assigned by this library to keep same-month "
        "editions distinct and correctly ordered; it is an ordering "
        "device, not a release date."
    ),
    "statcan": (
        "Statistics Canada Daily release date of the table version — a "
        "genuine national publication date. (Note: the WDS JSON's "
        "`releaseTime` field is NOT this; it is the cube's last-refresh "
        "timestamp and is identical across vintages.)"
    ),
    "oecd_stes": (
        "Month of the OECD's snapshot of the national series (EDITION = "
        "YYYYMM), not the national statistical office's release date. "
        "The OECD ingests with a lag, so an edition can still carry the "
        "previous month's published figure. Fine for ordering editions "
        "and measuring revisions; not a release date."
    ),
}


#: Statuses worth retrying: the server is asking us to slow down or is
#: briefly unavailable, not telling us the request was wrong.
RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


def fetch_with_backoff(
    url: str,
    *,
    timeout: float,
    use_cache: bool = True,
    user_agent: str = "puremacro (real-time vintage reader)",
    attempts: int = 5,
    base_delay: float = 2.0,
    rate_limit_seconds: float = 1.0,
    sleep=None,
) -> bytes:
    """GET ``url``, backing off on rate-limit and transient-server codes.

    Bulk vintage archives are fetched one country at a time, and a
    cross-country panel therefore issues dozens of large requests in a
    row. The OECD SDMX endpoint answers that with HTTP 429, and a
    connector without backoff turns a rate limit into "this country
    publishes no vintages" — which is precisely the misreading this
    package exists to prevent.

    Honours ``Retry-After`` when the server sends one, and otherwise
    doubles ``base_delay`` per attempt. Non-retryable errors propagate
    immediately; the caller records them.

    This is deliberately narrower than a general retry loop: it retries
    only :data:`RETRYABLE_STATUS`, a bounded number of times. See
    ``puremacro/narrative/sources/RETRY_POLICY.md`` for why the shared
    HTTP helpers do not loop.
    """
    import time
    import urllib.error

    from ..._http import safe_get_bytes, safe_get_bytes_cached

    _sleep = sleep or time.sleep
    last: Exception | None = None
    for k in range(max(1, attempts)):
        try:
            if use_cache:
                return safe_get_bytes_cached(
                    url, timeout, user_agent=user_agent,
                    rate_limit_seconds=rate_limit_seconds,
                )
            return safe_get_bytes(url, timeout, user_agent=user_agent)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or k == attempts - 1:
                raise
            last = exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else base_delay * (2 ** k)
            except (TypeError, ValueError):
                delay = base_delay * (2 ** k)
            _sleep(min(delay, 60.0))
    raise last if last is not None else RuntimeError(f"could not fetch {url}")


def normalize_vintage_frame(
    df: pd.DataFrame,
    *,
    country: str,
    variable: str,
    provider: str,
    series_id: str,
    units: str = "level",
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> pd.DataFrame:
    """Coerce a provider's parsed output into the canonical tidy shape.

    Every connector routes its result through here, so a schema slip in
    one provider cannot propagate into the panel.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=VINTAGE_COLUMNS)
    missing = {date_col, vintage_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"{provider}:{series_id} parser returned columns "
            f"{sorted(df.columns)}; missing {sorted(missing)}"
        )
    out = pd.DataFrame({
        "country": str(country).upper(),
        "variable": str(variable),
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "vintage": pd.to_datetime(df[vintage_col], errors="coerce"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
        "provider": provider,
        "series_id": series_id,
        "units": units,
    })
    out = out.dropna(subset=["date", "vintage", "value"])
    return out[VINTAGE_COLUMNS].reset_index(drop=True)


@dataclass
class VintagePanel:
    """A multi-country, multi-provider real-time panel.

    ``df`` carries the tidy columns listed in :data:`VINTAGE_COLUMNS`.
    The methods are the operations everyone otherwise rebuilds by hand:
    the revision triangle, the first and latest estimate of each
    period, ``r_t = y_f - y_p``, and the Mankiw-Shapiro news-vs-noise
    test — the last one both per country and across the whole panel.
    """
    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.df is None or len(self.df) == 0:
            self.df = pd.DataFrame(columns=VINTAGE_COLUMNS)
            return
        if "units" not in self.df.columns:
            self.df = self.df.assign(units="level")
        missing = set(VINTAGE_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(
                f"VintagePanel missing required columns: {sorted(missing)}"
            )
        d = self.df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["vintage"] = pd.to_datetime(d["vintage"])
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["date", "vintage", "value"])
        key = ["country", "variable", "date", "vintage"]
        d = d.sort_values(key + ["provider"])
        clash = d.duplicated(subset=key, keep=False)
        if clash.any():
            offenders = (d.loc[clash]
                         .groupby(["country", "variable"])["provider"]
                         .apply(lambda x: sorted(set(x))))
            multi = {k: v for k, v in offenders.items() if len(v) > 1}
            if multi:
                # Two providers for one series is never a merge that can be
                # done correctly: they differ in units and in what a
                # vintage date means, so keeping one row at random splices
                # two incompatible series into a fabricated revision.
                warnings.warn(
                    "VintagePanel received the same (country, variable, "
                    f"date, vintage) from several providers: {multi}. Only "
                    "one row per key is kept, which mixes series that are "
                    "not comparable. Filter to one provider before "
                    "combining.",
                    UserWarning, stacklevel=3,
                )
        self.df = (
            d[VINTAGE_COLUMNS]
            .drop_duplicates(subset=key, keep="last")
            .sort_values(key)
            .reset_index(drop=True)
        )

    # -- inventory ---------------------------------------------------
    @property
    def countries(self) -> list[str]:
        return sorted(self.df["country"].unique().tolist()) if len(self.df) else []

    @property
    def variables(self) -> list[str]:
        return sorted(self.df["variable"].unique().tolist()) if len(self.df) else []

    @property
    def providers(self) -> list[str]:
        return sorted(self.df["provider"].unique().tolist()) if len(self.df) else []

    def __len__(self) -> int:
        return int(len(self.df))

    def is_empty(self) -> bool:
        return len(self.df) == 0

    def vintage_semantics(self) -> dict[str, str]:
        """What the ``vintage`` column means, for each provider present."""
        return {p: VINTAGE_SEMANTICS.get(p, "undocumented") for p in self.providers}

    # -- slicing -----------------------------------------------------
    def long(
        self,
        country: str | None = None,
        variable: str | None = None,
    ) -> pd.DataFrame:
        """Long ``[date, vintage, value]`` slice for one country/variable.

        With no arguments returns the whole tidy frame. This is the
        shape :mod:`puremacro.vintages` consumes.

        Naming a country without a variable is only allowed when that
        country carries exactly one variable; otherwise it raises. A
        long frame that silently interleaved GDP and consumption rows
        would look perfectly well-formed and produce nonsense
        revisions.
        """
        if country is None and variable is None:
            return self.df.copy()
        if country is None:
            var = self._canon(variable)
            sub = self.df[self.df["variable"] == var]
            if sub.empty:
                raise KeyError(
                    f"no rows for variable {variable!r}; panel has "
                    f"{self.variables}"
                )
            present = sorted(sub["country"].unique())
            if len(present) > 1:
                raise ValueError(
                    f"variable {variable!r} spans several countries "
                    f"{present}; pass country= to disambiguate. Returning "
                    "them concatenated would look like one country's "
                    "history and silently produce a nonsense triangle."
                )
            return sub[["date", "vintage", "value"]].reset_index(drop=True)
        c, v = self._resolve(country, variable)
        sub = self.df[(self.df["country"] == c) & (self.df["variable"] == v)]
        return sub[["date", "vintage", "value"]].reset_index(drop=True)

    def _canon(self, variable: str) -> str:
        """Match the caller's spelling against the panel's own.

        The panel stores canonical names (``vintage_panel`` canonicalises
        at the entry point), so ``long("DEU", "B1GQ")`` would otherwise
        match nothing and report an empty slice as if the country had no
        data. A spelling the panel already uses verbatim wins, so a
        hand-built panel carrying an uncatalogued name still works.
        """
        if variable is None:
            return variable
        if len(self.df) and variable in set(self.df["variable"]):
            return variable
        try:
            from .catalog import canonical_variable
            return canonical_variable(variable)
        except Exception:
            return variable

    def _resolve(self, country: str, variable: str | None) -> tuple[str, str]:
        c = str(country).upper()
        variable = self._canon(variable)
        if variable is None:
            avail = sorted(self.df[self.df["country"] == c]["variable"].unique())
            if not avail:
                raise KeyError(
                    f"no rows for country {c!r}; panel has {self.countries}"
                )
            if len(avail) > 1:
                raise ValueError(
                    f"country {c!r} carries several variables {avail}; "
                    "pass variable= to disambiguate"
                )
            variable = avail[0]
        return c, variable

    def units_for(self, country: str, variable: str | None = None) -> str:
        """Units the catalogue declared for one series."""
        c, v = self._resolve(country, variable)
        sub = self.df[(self.df["country"] == c) & (self.df["variable"] == v)]
        vals = sub["units"].dropna().unique()
        return str(vals[0]) if len(vals) else "level"

    def _transform_for(self, country: str, variable: str | None,
                       transform: str | None) -> str:
        """Resolve ``transform=None`` from the series' declared units.

        A level must be differenced before a news/noise test; a series
        already published as a growth rate must not be. Getting this
        wrong differences an already-differenced series and yields a
        well-formed, meaningless answer.
        """
        if transform is not None:
            return transform
        from .catalog import UNITS_TRANSFORM
        return UNITS_TRANSFORM.get(self.units_for(country, variable),
                                   "log_diff_pct")

    def coverage(self) -> pd.DataFrame:
        """Per (country, variable) diagnostic: how much real-time data is here.

        The first thing to look at when a cross-country revision study
        comes back thinner than expected — it shows immediately which
        countries carry only one edition and therefore cannot support a
        revision test at all.
        """
        if self.is_empty():
            return pd.DataFrame(columns=[
                "country", "variable", "provider", "series_id", "n_obs",
                "n_vintages", "n_periods", "first_date", "last_date",
                "first_vintage", "last_vintage",
            ])
        g = self.df.groupby(["country", "variable"], as_index=False)
        rows = []
        for (c, v), sub in g:
            rows.append({
                "country": c,
                "variable": v,
                "provider": ", ".join(sorted(sub["provider"].unique())),
                "series_id": ", ".join(sorted(sub["series_id"].unique())),
                "n_obs": int(len(sub)),
                "n_vintages": int(sub["vintage"].nunique()),
                "n_periods": int(sub["date"].nunique()),
                "first_date": sub["date"].min(),
                "last_date": sub["date"].max(),
                "first_vintage": sub["vintage"].min(),
                "last_vintage": sub["vintage"].max(),
            })
        return pd.DataFrame(rows).sort_values(
            ["country", "variable"]).reset_index(drop=True)

    def as_of(self, vintage_date) -> pd.DataFrame:
        """What the whole panel looked like on ``vintage_date``.

        Returns a ``(country, date)``-indexed wide frame — the dataset a
        researcher would have had in hand that day.
        """
        v = pd.Timestamp(vintage_date)
        sub = self.df[self.df["vintage"] <= v]
        if sub.empty:
            return pd.DataFrame()
        latest = (
            sub.sort_values(["country", "variable", "date", "vintage"])
            .groupby(["country", "variable", "date"], as_index=False)
            .last()
        )
        piv = latest.pivot_table(
            index=["country", "date"], columns="variable", values="value"
        ).sort_index()
        piv.columns.name = None
        return piv

    # -- revision analysis -------------------------------------------
    def triangle(
        self,
        country: str,
        variable: str | None = None,
        *,
        transform: str = "level",
        periods_per_year: int = 4,
    ) -> pd.DataFrame:
        """The (reference date x vintage) revision triangle for one series."""
        c, v = self._resolve(country, variable)
        return revision_triangle(
            self.long(c, v), transform=transform,
            periods_per_year=periods_per_year,
        )

    def revisions(
        self,
        country: str,
        variable: str | None = None,
        *,
        transform: str | None = None,
        periods_per_year: int = 4,
        release: int = 0,
    ) -> pd.DataFrame:
        """``[preliminary, final, revision]`` per reference period.

        ``revision`` is ``r_t = y_f - y_p``. ``transform=None`` picks
        the transform this series' declared units imply.
        """
        c, v = self._resolve(country, variable)
        return revision_frame(
            self.long(c, v),
            transform=self._transform_for(c, v, transform),
            periods_per_year=periods_per_year, release=release,
        )

    def news_or_noise(
        self,
        country: str,
        variable: str | None = None,
        *,
        transform: str | None = None,
        periods_per_year: int = 4,
        release: int = 0,
        hac_lags: int | str = 0,
        significance: float = 0.05,
    ) -> MankiwShapiroResult:
        """Run the Mankiw-Shapiro test pair on one country's revisions."""
        c, v = self._resolve(country, variable)
        tf = self._transform_for(c, v, transform)
        frame = self.revisions(
            c, v, transform=tf,
            periods_per_year=periods_per_year, release=release,
        )
        return mankiw_shapiro(
            frame["preliminary"], frame["final"],
            hac_lags=hac_lags, significance=significance, transform=tf,
        )

    def news_or_noise_panel(
        self,
        variable: str | None = None,
        *,
        transform: str | None = None,
        periods_per_year: int = 4,
        release: int = 0,
        hac_lags: int | str = 0,
        significance: float = 0.05,
        min_obs: int = 12,
    ) -> pd.DataFrame:
        """Run the test for every country and return one tidy table.

        This is the cross-section the teaching notebook plots: one row
        per country with ``beta_on_preliminary`` and its standard
        error, so "GDP revisions are noise, not news" stops being a
        claim about the United States and becomes a comparable fact.

        Countries with fewer than ``min_obs`` usable revision pairs are
        reported with ``ok=False`` and a populated ``note`` rather than
        dropped, because a silently short panel is exactly the failure
        this is meant to make visible.
        """
        rows = []
        pairs = (
            self.df[["country", "variable"]].drop_duplicates()
            .sort_values(["country", "variable"]).itertuples(index=False)
        )
        for c, v in pairs:
            if variable is not None and v != variable:
                continue
            tf = self._transform_for(c, v, transform)
            base = {"country": c, "variable": v,
                    "units": self.units_for(c, v), "transform": tf}
            try:
                frame = self.revisions(
                    c, v, transform=tf,
                    periods_per_year=periods_per_year, release=release,
                )
            except Exception as exc:                    # pragma: no cover
                rows.append({**base, "n_obs": 0, "ok": False,
                             "note": f"could not build revisions: {exc}"})
                continue
            n = int(len(frame))
            if n < max(3, min_obs):
                rows.append({**base, "n_obs": n, "ok": False,
                             "note": f"only {n} revision pairs "
                                     f"(min_obs={min_obs})"})
                continue
            res = mankiw_shapiro(
                frame["preliminary"], frame["final"],
                hac_lags=hac_lags, significance=significance, transform=tf,
            )
            rows.append({
                **base,
                "n_obs": res.n_obs,
                "mean_revision": res.mean_revision,
                "std_revision": res.std_revision,
                "beta_on_preliminary": res.beta_on_preliminary,
                "se_beta_on_preliminary": res.se_beta_on_preliminary,
                "p_beta_on_preliminary": res.p_beta_on_preliminary,
                "beta_on_final": res.beta_on_final,
                "se_beta_on_final": res.se_beta_on_final,
                "p_beta_on_final": res.p_beta_on_final,
                "noise_share": res.noise_share,
                "verdict": res.verdict,
                "ok": True,
                "note": "",
            })
        cols = [
            "country", "variable", "units", "transform",
            "n_obs", "mean_revision", "std_revision",
            "beta_on_preliminary", "se_beta_on_preliminary",
            "p_beta_on_preliminary", "beta_on_final", "se_beta_on_final",
            "p_beta_on_final", "noise_share", "verdict", "ok", "note",
        ]
        out = pd.DataFrame(rows)
        for col in cols:
            if col not in out.columns:
                out[col] = np.nan
        # Failure rows carry only a few keys, so the string columns come
        # back as NaN for them. Fill per CELL: a column-level fill leaves
        # `verdict` as NaN in any table that mixes ok and failed rows,
        # and `NaN` reads as a verdict rather than as "not estimated".
        for col in ("note", "verdict", "units", "transform"):
            out[col] = out[col].fillna("")
        out["ok"] = out["ok"].fillna(False).astype(bool)
        return out[cols].sort_values(["variable", "country"]).reset_index(drop=True)

    # -- combination -------------------------------------------------
    def filter(
        self,
        *,
        countries: Iterable[str] | None = None,
        variables: Iterable[str] | None = None,
        providers: Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        start_vintage: str | None = None,
        end_vintage: str | None = None,
    ) -> "VintagePanel":
        sub = self.df
        if countries is not None:
            sub = sub[sub["country"].isin({str(c).upper() for c in countries})]
        if variables is not None:
            sub = sub[sub["variable"].isin(set(variables))]
        if providers is not None:
            sub = sub[sub["provider"].isin(set(providers))]
        if start_date is not None:
            sub = sub[sub["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            sub = sub[sub["date"] <= pd.Timestamp(end_date)]
        if start_vintage is not None:
            sub = sub[sub["vintage"] >= pd.Timestamp(start_vintage)]
        if end_vintage is not None:
            sub = sub[sub["vintage"] <= pd.Timestamp(end_vintage)]
        return VintagePanel(df=sub.copy(),
                            metadata={**self.metadata, "filtered": True})

    @classmethod
    def concat(cls, panels: Iterable["VintagePanel"]) -> "VintagePanel":
        frames, meta = [], {}
        for p in panels:
            if p is None:
                continue
            # Merge dict-valued metadata rather than replacing it, and do
            # it for EMPTY panels too: a provider that returned nothing
            # is exactly the one whose `failed` reasons matter, and
            # dict.update would have dropped them.
            for k, v in p.metadata.items():
                if isinstance(v, dict) and isinstance(meta.get(k), dict):
                    meta[k] = {**meta[k], **v}
                elif isinstance(v, list) and isinstance(meta.get(k), list):
                    meta[k] = meta[k] + [x for x in v if x not in meta[k]]
                else:
                    meta[k] = v
            if p.is_empty():
                continue
            frames.append(p.df)
        if not frames:
            return cls(df=pd.DataFrame(columns=VINTAGE_COLUMNS), metadata=meta)
        return cls(df=pd.concat(frames, ignore_index=True), metadata=meta)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
#: name -> callable(countries, variables, **kwargs) -> VintagePanel
_PROVIDER_REGISTRY: dict[str, Callable[..., VintagePanel]] = {}
#: name -> set of country codes the provider can serve
_PROVIDER_COVERAGE: dict[str, frozenset[str]] = {}


def register_provider(
    name: str,
    fetch_fn: Callable[..., VintagePanel],
    countries: Iterable[str],
) -> None:
    """Register a real-time provider under ``name``."""
    _PROVIDER_REGISTRY[name] = fetch_fn
    _PROVIDER_COVERAGE[name] = frozenset(str(c).upper() for c in countries)


def available_providers() -> list[str]:
    """Names of every registered real-time provider."""
    return sorted(_PROVIDER_REGISTRY)


def provider_countries(name: str) -> frozenset[str]:
    """Country codes ``name`` can serve."""
    if name not in _PROVIDER_COVERAGE:
        raise KeyError(
            f"unknown provider {name!r}; registered: {available_providers()}"
        )
    return _PROVIDER_COVERAGE[name]


def get_provider(name: str) -> Callable[..., VintagePanel]:
    if name not in _PROVIDER_REGISTRY:
        raise KeyError(
            f"unknown provider {name!r}; registered: {available_providers()}"
        )
    return _PROVIDER_REGISTRY[name]


__all__ = [
    "VINTAGE_COLUMNS",
    "RETRYABLE_STATUS",
    "fetch_with_backoff",
    "VINTAGE_SEMANTICS",
    "VintagePanel",
    "normalize_vintage_frame",
    "register_provider",
    "available_providers",
    "provider_countries",
    "get_provider",
]
