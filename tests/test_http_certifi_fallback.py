import pytest
import ssl
import urllib.request
import urllib.error
from unittest.mock import patch, MagicMock

import puremacro._http as http

def test_request_ssl_fallback_uses_certifi():
    """Ensure that the SSL fallback uses certifi, not unverified context."""
    # We want to mock urllib.request.urlopen to fail once with URLError(SSLError),
    # then succeed on the second call.
    # We also want to mock certifi.where to return a dummy path.
    # We also want to mock ssl.create_default_context to assert it's called with the dummy path.

    mock_resp = MagicMock()
    mock_resp.read.return_value = b"success"
    mock_resp.__enter__.return_value = mock_resp

    # Create an SSLError wrapped in URLError, matching how urlopen fails
    ssl_err = urllib.error.URLError(ssl.SSLError("Mock SSL Error"))

    responses = [ssl_err, mock_resp]

    def side_effect(*args, **kwargs):
        res = responses.pop(0)
        if isinstance(res, Exception):
            raise res
        return res

    with patch("urllib.request.urlopen", side_effect=side_effect) as mock_urlopen, \
         patch("certifi.where", return_value="/mock/path/to/cacert.pem") as mock_where, \
         patch("ssl.create_default_context") as mock_create_ctx:

        mock_ctx = MagicMock()
        mock_create_ctx.return_value = mock_ctx

        res = http._request("https://test.invalid", timeout=1.0)

        assert res == b"success"

        # Verify the fallback logic used certifi
        mock_where.assert_called_once()
        mock_create_ctx.assert_called_once_with(cafile="/mock/path/to/cacert.pem")

        # Verify urlopen was called with the context
        assert mock_urlopen.call_count == 2
        assert mock_urlopen.call_args_list[1].kwargs["context"] == mock_ctx

def test_post_json_ssl_fallback_uses_certifi():
    """Ensure that the SSL fallback in post_json uses certifi, not unverified context."""

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status": "success"}'
    mock_resp.__enter__.return_value = mock_resp

    ssl_err = urllib.error.URLError(ssl.SSLError("Mock SSL Error"))

    responses = [ssl_err, mock_resp]

    def side_effect(*args, **kwargs):
        res = responses.pop(0)
        if isinstance(res, Exception):
            raise res
        return res

    with patch("urllib.request.urlopen", side_effect=side_effect) as mock_urlopen, \
         patch("certifi.where", return_value="/mock/path/to/cacert.pem") as mock_where, \
         patch("ssl.create_default_context") as mock_create_ctx:

        mock_ctx = MagicMock()
        mock_create_ctx.return_value = mock_ctx

        res = http.post_json("https://test.invalid", {"a": 1}, timeout=1.0)

        assert res == {"status": "success"}

        mock_where.assert_called_once()
        mock_create_ctx.assert_called_once_with(cafile="/mock/path/to/cacert.pem")

        assert mock_urlopen.call_count == 2
        assert mock_urlopen.call_args_list[1].kwargs["context"] == mock_ctx
