"""Shared Playwright stealth-helper for bot-protected CB archive pages.

Three families of CB sites refuse plain ``requests``-based crawls (Akamai
on BoE, Incapsula on BCCh, Radware on BanRep, generic Cloudfront WAF on
BCB, etc.) but admit a headless-Chromium client that passes:
- A realistic ``User-Agent`` header
- ``navigator.webdriver`` patched to ``undefined``
- ``--disable-blink-features=AutomationControlled``

This helper centralises that setup. Connectors that need it call
``fetch_with_playwright(url)`` and get back the rendered HTML; they parse
the HTML with the same regex-based approach as the pure-``requests``
connectors.

Playwright is a soft / opt-in dependency. We import lazily so a
pyodide / no-extras install of puremacro still imports cleanly; the
import failure surfaces only when a Playwright-backed connector is
actually called.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
_STEALTH_INIT = (
    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
)
_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]


def _running_in_async_loop() -> bool:
    """True iff the current thread has a running asyncio event loop
    (jupyter notebooks, async test runners, etc.). Playwright's sync API
    refuses to start under those conditions; we route through a thread
    to escape."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _do_fetch(url, wait_for, timeout_ms, viewport, locale):
    """Run the actual sync_playwright call. Module-level so it can be
    pickled / scheduled into a worker thread when an event loop blocks
    direct sync use (e.g., inside jupyter)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "Playwright is not installed. Run `pip install playwright && "
            "playwright install chromium` to enable Playwright-backed connectors."
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        try:
            ctx = browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": viewport[0], "height": viewport[1]},
                locale=locale,
            )
            page = ctx.new_page()
            page.add_init_script(_STEALTH_INIT)
            page.goto(url, timeout=timeout_ms)
            try:
                page.wait_for_load_state(wait_for, timeout=timeout_ms)
            except Exception:
                pass  # be tolerant of long-lived connections
            html = page.content()
            ctx.close()
            return html
        finally:
            browser.close()


@functools.lru_cache(maxsize=128)
def fetch_with_playwright(
    url: str,
    *,
    wait_for: str = "networkidle",
    timeout_ms: int = 20000,
    viewport: tuple[int, int] = (1440, 900),
    locale: str = "en-US",
) -> str:
    """Fetch ``url`` via stealth-Chromium and return rendered HTML.

    Cached: repeat calls for the same URL return the cached HTML
    (avoids spinning up a new browser for every page in a multi-page
    scrape). The cache is process-local; tests / health-checks see
    fresh content each pytest run.

    If called from inside a running asyncio loop (jupyter notebooks
    being the main case), the underlying ``sync_playwright()`` would
    refuse to start. We detect that and route the call through a
    worker thread which gets its own fresh event loop.

    Raises ``ImportError`` if Playwright isn't installed.
    """
    if _running_in_async_loop():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(
                _do_fetch, url, wait_for, timeout_ms, viewport, locale,
            ).result()
    return _do_fetch(url, wait_for, timeout_ms, viewport, locale)


__all__ = ["fetch_with_playwright"]
