"""``vintage_panel`` — one call for a cross-country real-time panel.

The revision-oriented counterpart to
:func:`puremacro.fetch.qna_panel`::

    rev = vintage_panel(["USA", "DEU", "ESP"], series="B1GQ", freq="Q")
    rev.coverage()             # what actually came back
    rev.revisions("DEU")       # preliminary / final / r_t
    rev.news_or_noise("DEU")   # the Mankiw-Shapiro pair
    rev.news_or_noise_panel()  # every country, one table
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

from ._base import VintagePanel, available_providers, get_provider
from .catalog import canonical_variable, provider_country_codes, resolve_series


#: Frequencies the real-time catalogues cover. Kept explicit so an
#: unsupported request fails loudly instead of returning an empty panel
#: that reads as "no vintages exist for these countries".
SUPPORTED_FREQUENCIES = {"Q"}

#: Order tried when ``providers="auto"``. The OECD STES revisions
#: archive leads because it covers 42 economies through *one* pipeline
#: with editions back to 1999, so a panel built from it is internally
#: comparable. The national repositories that follow carry better
#: vintage-date semantics (genuine release dates) but each in its own
#: format and each for one country.
DEFAULT_PROVIDER_ORDER = (
    "oecd_stes", "alfred", "bundesbank", "ons", "statcan", "ecb_rtd",
)


def vintage_panel(
    countries: Iterable[str] | str | None = None,
    *,
    series: str = "gdp_real",
    variables: Sequence[str] | None = None,
    freq: str = "Q",
    providers: str | Sequence[str] = "oecd_stes",
    catalog: dict | None = None,
    store=None,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float | None = None,
    warn_on_mixed_providers: bool = True,
) -> VintagePanel:
    """Assemble a cross-country panel of published *editions*.

    Parameters
    ----------
    countries : ISO3 codes, e.g. ``["USA", "DEU", "ESP"]``. ``None``
        takes every country the chosen provider(s) cover.
    series : the variable to fetch, in any spelling
        :func:`~puremacro.fetch.realtime.catalog.canonical_variable`
        accepts — ``"gdp_real"``, ``"B1GQ"``, ``"gdp"``.
    variables : fetch several variables at once; overrides ``series``.
    freq : only ``"Q"`` today; anything else raises.
    providers : ``"oecd_stes"`` (default) for one uniform pipeline
        across 42 economies, ``"auto"`` to fall back through
        :data:`DEFAULT_PROVIDER_ORDER` per country, or an explicit
        sequence tried in order. Use ``["bundesbank"]`` / ``["ons"]`` /
        ``["statcan"]`` when you need genuine national release dates.
    catalog : override the built-in series table; see
        :func:`~puremacro.fetch.realtime.catalog.resolve_series`.
    timeout : per-request timeout in seconds. ``None`` (default) lets
        each provider keep its own — the bulk providers need far longer
        than the per-series ones, and a single global default low
        enough for one is short enough to silently truncate the other.
    warn_on_mixed_providers : warn when the assembled panel draws on
        more than one provider. Mixing matters: providers disagree on
        what a vintage *date* means (ingestion date vs. national
        release date), so a cross-country comparison built from
        several of them is comparing slightly different objects. See
        ``VintagePanel.vintage_semantics()``.

    Returns
    -------
    VintagePanel

    Notes
    -----
    Countries that yield nothing are not an error — they are reported
    by ``VintagePanel.coverage()`` and by ``metadata["missing"]``.
    A country with only one archived edition cannot support a revision
    test at all, and ``news_or_noise_panel`` flags it rather than
    quietly dropping it.
    """
    if freq not in SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"freq={freq!r} is not supported; the real-time catalogues "
            f"cover {sorted(SUPPORTED_FREQUENCIES)}. Monthly and annual "
            "vintages exist at several providers but are not mapped yet."
        )

    if variables is not None:
        var_list = [canonical_variable(v) for v in variables]
    else:
        var_list = [canonical_variable(series)]

    if providers == "auto":
        provider_list = [p for p in DEFAULT_PROVIDER_ORDER
                         if p in available_providers()]
    elif isinstance(providers, str):
        provider_list = [providers]
    else:
        provider_list = list(providers)
    unknown = [p for p in provider_list if p not in available_providers()]
    if unknown:
        raise ValueError(
            f"unknown provider(s) {unknown}; registered: "
            f"{available_providers()}"
        )

    if countries is None:
        country_list = sorted({
            c for p in provider_list for c in provider_country_codes(p)
        })
    elif isinstance(countries, str):
        country_list = [countries.upper()]
    else:
        country_list = [str(c).upper() for c in countries]

    panels: list[VintagePanel] = []
    used: dict[str, str] = {}
    failed: dict[str, str] = {}

    # One call per provider, not one per country: the bulk archives
    # accept multi-country keys and rate-limit aggressively, so 42
    # single-country requests get throttled where 7 batched ones do not.
    #
    # The work queue is keyed on (country, VARIABLE) pairs, not on
    # countries. Keying on country alone would retire a country from the
    # queue as soon as ANY of its variables came back, so a request for
    # GDP + deflator against a provider carrying only GDP would silently
    # never look for the deflator anywhere else.
    remaining = {(c, v) for c in country_list for v in var_list}
    for prov in provider_list:
        if not remaining:
            break
        servable = sorted(
            (c, v) for (c, v) in remaining
            if resolve_series(prov, c, v, catalog=catalog) is not None
        )
        if not servable:
            continue
        countries_here = sorted({c for c, _ in servable})
        vars_here = sorted({v for _, v in servable})
        fetch_fn = get_provider(prov)
        kwargs = dict(catalog=catalog, store=store, refresh=refresh,
                      use_cache=use_cache)
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            part = fetch_fn(countries_here, vars_here, **kwargs)
        except Exception as exc:
            for c, v in servable:
                failed[f"{c}:{v}:{prov}"] = f"{type(exc).__name__}: {exc}"
            continue
        if part is None:
            continue
        # Providers record per-series reasons in their own metadata;
        # surface them rather than reporting a bare "missing", which
        # reads as "this country publishes no vintages" when it usually
        # means a fetch failed.
        for k, v in (part.metadata.get("failed") or {}).items():
            failed[f"{k}:{prov}"] = v
        if part.is_empty():
            continue
        panels.append(part)
        got = set(zip(part.df["country"], part.df["variable"]))
        for c, _v in got:
            used.setdefault(c, prov)
        remaining -= got

    missing_pairs = sorted(remaining)
    missing = sorted({c for c, _ in missing_pairs})

    panel = VintagePanel.concat(panels)
    panel.metadata.update({
        "requested_countries": country_list,
        "variables": var_list,
        "providers_tried": provider_list,
        "provider_used": used,
        "missing": missing,
        "missing_pairs": missing_pairs,
        "failed": failed,
        "freq": freq,
    })

    if missing:
        detail = "; ".join(f"{k} -> {v}" for k, v in list(failed.items())[:5])
        warnings.warn(
            f"vintage_panel returned nothing for {len(missing)} of "
            f"{len(country_list)} requested countries: {missing}. "
            + (f"Reasons include: {detail}. " if detail else
               "No provider in the list carries these (country, variable) "
               "pairs — check vintage_catalog() and known_gaps(). ")
            + "This is reported rather than raised because a partial panel "
              "is usually still usable, but a country absent here is NOT "
              "evidence that it publishes no vintages.",
            UserWarning, stacklevel=2,
        )

    if warn_on_mixed_providers and len(panel.providers) > 1:
        warnings.warn(
            "vintage_panel assembled data from several providers "
            f"({panel.providers}), which do not share a definition of the "
            "vintage date — some are archive-ingestion dates, others are "
            "national release dates. Editions still order correctly within "
            "each country, but do not treat vintage dates as comparable "
            "release timestamps across countries. Call "
            "VintagePanel.vintage_semantics() for the details.",
            UserWarning, stacklevel=2,
        )

    return panel


__all__ = ["vintage_panel", "SUPPORTED_FREQUENCIES", "DEFAULT_PROVIDER_ORDER"]
