"""Tests for narrative.sources._extractors body-extraction module."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Fed: <div id="article">…</div>
# ---------------------------------------------------------------------------
def test_extract_fed_finds_article_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><head><title>x</title></head><body>'
        '<nav>Top menu Economic Research Data Bank Assets</nav>'
        '<div id="article">'
        '<p>Recent indicators suggest that economic activity continued.</p>'
        '<p>The Committee judges that risks to its outlook remain.</p>'
        '</div>'
        '<footer>Footer chrome about uncertainty in policy</footer>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="FED")
    assert "Recent indicators" in body
    assert "Committee judges" in body
    # Navigation chrome should NOT leak in
    assert "Top menu" not in body
    assert "Footer chrome" not in body


def test_extract_fed_falls_back_to_generic_on_no_article_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<main><p>The body is in main, not in an article div.</p></main>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="FED")
    assert "body is in main" in body


# ---------------------------------------------------------------------------
# ECB: <main id="main-wrapper"> or <div class="section">
# ---------------------------------------------------------------------------
def test_extract_ecb_finds_main_wrapper():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<nav>menu</nav>'
        '<main id="main-wrapper">'
        '<p>The Governing Council decided today to raise rates.</p>'
        '</main>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="ECB")
    assert "Governing Council" in body
    assert "menu" not in body


# ---------------------------------------------------------------------------
# Generic fallback: largest text-dense container
# ---------------------------------------------------------------------------
def test_extract_default_picks_largest_div():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<div class="menu">x</div>'
        '<div class="content">'
        '<p>This is the actual statement body and it is the longest div on the page.</p>'
        '</div>'
        '<div class="footer">y</div>'
        '</body></html>'
    )
    body = extract_body(html)
    assert "actual statement body" in body


def test_extract_default_strips_script_and_style():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<script>var x = "should not appear"</script>'
        '<style>.foo { color: red; }</style>'
        '<div><p>Real body text.</p></div>'
        '</body></html>'
    )
    body = extract_body(html)
    assert "Real body text" in body
    assert "should not appear" not in body
    assert "color: red" not in body


def test_extract_handles_empty_html():
    from puremacro.narrative.sources._extractors import extract_body
    assert extract_body("") == ""
    assert extract_body("<html></html>") == ""


def test_extract_handles_malformed_html():
    """Crude regex extraction should not crash on broken HTML."""
    from puremacro.narrative.sources._extractors import extract_body
    body = extract_body("<html><body><p>unclosed paragraph<div>nested",
                        bank_code="FED")
    # No assertion on content; just confirm it doesn't raise.
    assert isinstance(body, str)


def test_extract_unknown_bank_code_uses_generic():
    from puremacro.narrative.sources._extractors import extract_body
    html = '<html><body><div><p>Body text here.</p></div></body></html>'
    body = extract_body(html, bank_code="NOT_A_BANK")
    assert "Body text here" in body


def test_extract_fed_handles_nested_divs():
    """The article container has nested divs. Extractor must find the
    matching balanced close tag, not the first inner one."""
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<div id="article">'
        '  <div class="heading"><h1>Title</h1></div>'   # nested div
        '  <div class="body">'                          # another nested div
        '    <p>Recent indicators suggest economic activity expanded.</p>'
        '    <p>The Committee is firmly committed to its 2 percent inflation goal.</p>'
        '  </div>'
        '</div>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="FED")
    assert "Recent indicators" in body
    assert "Committee is firmly committed" in body


def test_extract_ecb_handles_nested_divs():
    from puremacro.narrative.sources._extractors import extract_body
    html = (
        '<html><body>'
        '<main id="main-wrapper">'
        '  <div class="header">x</div>'
        '  <div class="content">'
        '    <p>The Governing Council decided today to cut rates.</p>'
        '  </div>'
        '</main>'
        '</body></html>'
    )
    body = extract_body(html, bank_code="ECB")
    assert "Governing Council" in body
