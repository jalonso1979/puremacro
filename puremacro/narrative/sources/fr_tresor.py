"""French Direction générale du Trésor (DG Trésor) press feed.

DG Trésor publishes news at https://www.tresor.economie.gouv.fr/. The
RSS feed is in French; LLM-based scoring is recommended since the
default keyword lexicon is English-only.
"""
from __future__ import annotations

from typing import Iterator

from ._rss import iter_rss


_FEED = "https://www.tresor.economie.gouv.fr/Flux/Atom/Articles/Home"


def iter_tresor_press(*, feed_url: str | None = None) -> Iterator[tuple]:
    """Yield DG Trésor press items.

    The Trésor publishes an Atom feed at
    ``/Flux/Atom/Articles/Home`` (verified live, May 2026). Items are
    in French — pair with LLM scoring or a French keyword lexicon.
    """
    from ._rss import iter_atom
    yield from iter_atom(feed_url or _FEED)


__all__ = ["iter_tresor_press"]
