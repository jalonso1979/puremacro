"""Tests for the Fed minutes announcement-page link extractor."""
from __future__ import annotations


def test_extract_minutes_body_link_modern_pattern():
    """Post-2014 announcement pages link to /monetarypolicy/fomcminutes…"""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<p>The minutes were released today.</p>'
        '<a href="/monetarypolicy/fomcminutes20240501.htm">Minutes (HTML)</a>'
        '</body></html>'
    )
    href = _extract_minutes_body_link(html)
    assert href == "/monetarypolicy/fomcminutes20240501.htm"


def test_extract_minutes_body_link_pre_2014_pattern():
    """Pre-2014 announcement pages link to /fomc/minutes/{meeting-date}.htm"""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<p>Released today: minutes of the December 13, 2005 meeting.</p>'
        '<a href="/fomc/minutes/20051213.htm">View the minutes</a>'
        '</body></html>'
    )
    href = _extract_minutes_body_link(html)
    assert href == "/fomc/minutes/20051213.htm"


def test_extract_minutes_body_link_returns_none_when_no_link():
    """If the announcement page has no minutes-body link, return None."""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = '<html><body><p>Just an announcement.</p></body></html>'
    assert _extract_minutes_body_link(html) is None


def test_extract_minutes_body_link_picks_first_match():
    """If multiple matching links exist, return the first one."""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<a href="/monetarypolicy/fomcminutes20240501.htm">First</a>'
        '<a href="/monetarypolicy/fomcminutes20240601.htm">Second</a>'
        '</body></html>'
    )
    assert _extract_minutes_body_link(html) == "/monetarypolicy/fomcminutes20240501.htm"
