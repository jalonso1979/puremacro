"""Parser schema versioning + landmark-assertion framework (0.66.0+).

Each narrative connector parser declares a module-level
``PARSER_SCHEMA_VERSION`` and calls :func:`assert_landmarks` near the
top of its body parser. When the upstream source's HTML/JSON layout
drifts, the missing landmark triggers a loud ``ParserSchemaMismatchError``
naming the connector + the missing landmark. The ``iter_<source>``
wrapper catches the error and emits a ``UserWarning`` while yielding
empty (per RETRY_POLICY.md §4.1: "yield, don't raise").

The framework is intentionally minimal — landmarks are substring checks,
not CSS selectors or XPath expressions. Substring matching covers 95%
of real-world layout drift (renamed sections, removed headings,
restructured pages) at zero parser dependency cost.
"""
from __future__ import annotations


class ParserSchemaMismatchError(RuntimeError):
    """Raised by :func:`assert_landmarks` when an expected landmark is
    missing from the upstream body — i.e., the source layout has
    drifted away from what the parser was written against.

    Caught by ``iter_<source>`` generators, which emit a
    ``UserWarning`` naming the connector + missing landmark and then
    yield empty.
    """


def assert_landmarks(
    text: str,
    *,
    source: str,
    expected_version: int,
    landmarks: list,
) -> None:
    """Raise :class:`ParserSchemaMismatchError` if any landmark is missing.

    Parameters
    ----------
    text : the raw body about to be parsed (HTML, JSON, XML — anything
        string-shaped).
    source : the connector's canonical name (e.g. ``"beige_book"``,
        ``"eu_eurlex"``). Appears in the error message and in the
        ``warnings.warn`` emitted by the wrapper.
    expected_version : the parser's currently-locked schema version
        (matches the module's ``PARSER_SCHEMA_VERSION`` constant).
        Appears in the error message so a researcher knows which
        version the parser expected.
    landmarks : list of items. Each item is either:
        - ``str``: a substring that must appear in ``text``.
        - ``(selector, expected)`` tuple: the selector is informational
          (helps the error message); the check is
          ``expected in text``.
    """
    for landmark in landmarks:
        if isinstance(landmark, tuple):
            selector, expected = landmark
            if expected in text:
                continue
            raise ParserSchemaMismatchError(
                f"{source!r}: missing landmark ({selector}, {expected!r}) "
                f"(version={expected_version}). Upstream layout has "
                f"likely drifted; inspect the source HTML and bump "
                f"PARSER_SCHEMA_VERSION."
            )
        else:
            if landmark in text:
                continue
            raise ParserSchemaMismatchError(
                f"{source!r}: missing landmark {landmark!r} "
                f"(version={expected_version}). Upstream layout has "
                f"likely drifted; inspect the source HTML and bump "
                f"PARSER_SCHEMA_VERSION."
            )


__all__ = ["ParserSchemaMismatchError", "assert_landmarks"]
