# Narrative connector HTTP fixture cache

Each `<sha256>.json` file in this directory is a cached HTTP response
recorded for one of the narrative connectors in
`puremacro/narrative/sources/`. The cache key is the SHA-256 of
`URL + sorted-headers-json` (see `tests/_http_fixtures.py`).

## Recording new fixtures

```
PUREMACRO_RECORD_HTTP=1 pytest tests/test_narrative_offline.py
```

This fires the real HTTP, writes the response body into the cache, and
reruns the parser logic against the live data.

## Replay (default)

In replay mode (the default), the offline tests read fixtures only;
any cache miss raises a clear `FileNotFoundError` pointing back at this
file.

## Why some connectors are intentionally skipped

Some live sources require auth, have aggressive WAFs, or change their
response contract often enough that recording a snapshot is not
useful for parser-regression detection. Those connectors keep their
existing live-network tests in `tests/test_narrative.py` and are
excluded from `tests/test_narrative_offline.py` via
`@pytest.mark.skip`.

## File format

```json
{
  "url": "https://example.org/some/feed.xml",
  "status": 200,
  "content_type": "application/rss+xml",
  "encoding": "text",
  "body": "<?xml version=\"1.0\"?>..."
}
```

Binary bodies use `"encoding": "base64"` and `"body_b64": "..."`.
