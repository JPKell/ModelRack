"""Tests for :mod:`modelrack.providers._http` — the transport plumbing every real adapter shares.

Exercised directly here, not only indirectly through
:mod:`modelrack.providers.ollama`'s own tests: this module is written to be reused by Phase 4's
OpenAI-compatible adapter, and its own contract deserves tests that do not depend on any one
caller's shape.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from baseaicore import ValidationError

from modelrack.errors import (
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
)
from modelrack.providers._http import (
    is_loopback_host,
    iter_capped_lines,
    read_capped_json,
    translate_stream_interruption,
    translate_transport_error,
    truncated_text,
    validate_base_url,
)


class TestTruncatedText:
    def test_short_text_is_returned_unchanged(self) -> None:
        assert truncated_text("hello", limit=10) == "hello"

    def test_long_text_is_cut_with_an_ellipsis(self) -> None:
        result = truncated_text("x" * 20, limit=10)

        assert result == "x" * 10 + "…"
        assert len(result) == 11

    def test_text_exactly_at_the_limit_is_unchanged(self) -> None:
        assert truncated_text("x" * 10, limit=10) == "x" * 10


class TestIsLoopbackHost:
    @pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "127.0.0.1", "127.55.0.9", "::1"])
    def test_loopback_hosts(self, host: str) -> None:
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["gpu-box.lan", "10.0.0.5", "example.com", "192.168.1.1"])
    def test_non_loopback_hosts(self, host: str) -> None:
        assert is_loopback_host(host) is False

    def test_a_hostname_that_is_not_an_ip_literal_and_not_localhost_is_not_loopback(self) -> None:
        """The fallback path when `ip_address()` cannot parse the host at all."""
        assert is_loopback_host("not-an-ip-address") is False


class TestValidateBaseUrl:
    def test_a_loopback_url_is_accepted_and_not_remote(self) -> None:
        url, is_remote = validate_base_url("http://127.0.0.1:11434")

        assert url == "http://127.0.0.1:11434"
        assert is_remote is False

    def test_a_non_loopback_url_is_accepted_and_flagged_remote(self) -> None:
        _url, is_remote = validate_base_url("http://gpu-box.lan:11434")

        assert is_remote is True

    def test_https_is_accepted(self) -> None:
        _url, is_remote = validate_base_url("https://localhost:11434")

        assert is_remote is False

    @pytest.mark.parametrize("scheme_url", ["ftp://x", "ws://x", "file:///etc/passwd"])
    def test_a_disallowed_scheme_is_refused(self, scheme_url: str) -> None:
        with pytest.raises(ValidationError) as raised:
            validate_base_url(scheme_url)

        assert raised.value.details["field"] == "base_url"

    @pytest.mark.parametrize("hostless_url", ["not-a-url", "http://", "http:///path"])
    def test_a_url_with_no_host_is_refused(self, hostless_url: str) -> None:
        with pytest.raises(ValidationError) as raised:
            validate_base_url(hostless_url)

        assert raised.value.details["field"] == "base_url"


class TestTranslateTransportError:
    def test_a_timeout_becomes_provider_timeout(self) -> None:
        error = translate_transport_error(httpx.ConnectTimeout("x"), base_url="http://x")

        assert isinstance(error, ProviderTimeout)
        assert error.details["base_url"] == "http://x"

    def test_a_read_timeout_also_becomes_provider_timeout(self) -> None:
        assert isinstance(
            translate_transport_error(httpx.ReadTimeout("x"), base_url="http://x"), ProviderTimeout
        )

    def test_malformed_http_framing_becomes_a_protocol_error(self) -> None:
        error = translate_transport_error(
            httpx.RemoteProtocolError("bad framing"), base_url="http://x"
        )

        assert isinstance(error, ProviderProtocolError)
        assert "bad framing" in error.details["body"]

    def test_connection_refused_is_classified_from_the_message(self) -> None:
        """The exact string httpcore produces for ECONNREFUSED (verified against httpx 0.28);
        no real socket is opened — the socket guard in ``tests/conftest.py`` would refuse one.
        """
        error = translate_transport_error(
            httpx.ConnectError("[Errno 111] Connection refused"), base_url="http://x"
        )

        assert isinstance(error, ProviderUnavailable)
        assert error.details["reason"] == ProviderUnavailableReason.CONNECTION_REFUSED.value

    def test_dns_failure_is_classified_from_the_message(self) -> None:
        """The exact string httpcore produces for a getaddrinfo (EAI_NONAME) failure."""
        error = translate_transport_error(
            httpx.ConnectError("[Errno -2] Name or service not known"), base_url="http://x"
        )

        assert isinstance(error, ProviderUnavailable)
        assert error.details["reason"] == ProviderUnavailableReason.DNS_FAILURE.value

    def test_a_tls_message_is_classified_as_tls_failure(self) -> None:
        error = translate_transport_error(
            httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
            base_url="http://x",
        )

        assert isinstance(error, ProviderUnavailable)
        assert error.details["reason"] == ProviderUnavailableReason.TLS_FAILURE.value

    def test_an_unrecognized_message_falls_back_to_network_error(self) -> None:
        error = translate_transport_error(
            httpx.ConnectError("something went sideways"), base_url="http://x"
        )

        assert isinstance(error, ProviderUnavailable)
        assert error.details["reason"] == ProviderUnavailableReason.NETWORK_ERROR.value

    def test_a_positive_errno_other_than_connection_refused_is_network_error(self) -> None:
        """A matched `[Errno N]` that is neither negative (DNS) nor 111 (refused)."""
        error = translate_transport_error(
            httpx.ConnectError("[Errno 113] No route to host"), base_url="http://x"
        )

        assert isinstance(error, ProviderUnavailable)
        assert error.details["reason"] == ProviderUnavailableReason.NETWORK_ERROR.value

    def test_never_returns_a_raw_httpx_exception(self) -> None:
        for exc in (
            httpx.ConnectError("x"),
            httpx.ConnectTimeout("x"),
            httpx.RemoteProtocolError("x"),
            httpx.ReadError("x"),
        ):
            error = translate_transport_error(exc, base_url="http://x")
            assert not isinstance(error, httpx.HTTPError)


class TestTranslateStreamInterruption:
    """The mid-stream classifier, deliberately different from the pre-flight one."""

    def test_a_timeout_mid_stream_is_still_a_timeout(self) -> None:
        error = translate_stream_interruption(httpx.ReadTimeout("x"), base_url="http://x")

        assert isinstance(error, ProviderTimeout)

    def test_a_reset_mid_stream_is_a_protocol_error_not_unavailable(self) -> None:
        """A connection that already delivered content was not 'unreachable'."""
        error = translate_stream_interruption(
            httpx.ReadError("connection reset by peer"), base_url="http://x"
        )

        assert isinstance(error, ProviderProtocolError)

    def test_malformed_framing_mid_stream_is_a_protocol_error(self) -> None:
        error = translate_stream_interruption(
            httpx.RemoteProtocolError("bad chunk"), base_url="http://x"
        )

        assert isinstance(error, ProviderProtocolError)


class TestReadCappedJson:
    @respx.mock
    def test_reads_a_json_object_under_the_cap(self) -> None:
        respx.get("http://x/y").mock(return_value=httpx.Response(200, json={"a": 1}))

        with httpx.Client() as client, client.stream("GET", "http://x/y") as response:
            assert read_capped_json(response, max_bytes=1000, base_url="http://x") == {"a": 1}

    @respx.mock
    def test_reads_a_json_array_too(self) -> None:
        respx.get("http://x/y").mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        with httpx.Client() as client, client.stream("GET", "http://x/y") as response:
            assert read_capped_json(response, max_bytes=1000, base_url="http://x") == [1, 2, 3]

    @respx.mock
    def test_an_oversize_body_is_rejected_without_full_buffering(self) -> None:
        respx.get("http://x/y").mock(return_value=httpx.Response(200, content=b"x" * 1000))

        with httpx.Client() as client, client.stream("GET", "http://x/y") as response:
            with pytest.raises(ProviderProtocolError) as raised:
                read_capped_json(response, max_bytes=100, base_url="http://x")

        assert raised.value.details["limit_bytes"] == 100

    @respx.mock
    def test_a_non_json_body_raises_with_a_truncated_body_in_details(self) -> None:
        respx.get("http://x/y").mock(return_value=httpx.Response(200, content=b"not json"))

        with httpx.Client() as client, client.stream("GET", "http://x/y") as response:
            with pytest.raises(ProviderProtocolError) as raised:
                read_capped_json(response, max_bytes=1000, base_url="http://x")

        assert raised.value.details["body"] == "not json"

    @respx.mock
    def test_a_json_scalar_is_rejected_as_not_an_object_or_array(self) -> None:
        respx.get("http://x/y").mock(return_value=httpx.Response(200, content=b'"just a string"'))

        with httpx.Client() as client, client.stream("GET", "http://x/y") as response:
            with pytest.raises(ProviderProtocolError):
                read_capped_json(response, max_bytes=1000, base_url="http://x")


class TestIterCappedLines:
    def test_lines_under_the_cap_pass_through_unchanged(self) -> None:
        result = list(iter_capped_lines(iter(["a", "bb", "ccc"]), max_chunk_bytes=10, base_url="x"))

        assert result == ["a", "bb", "ccc"]

    def test_a_line_over_the_cap_raises_a_protocol_error(self) -> None:
        with pytest.raises(ProviderProtocolError) as raised:
            list(
                iter_capped_lines(iter(["ok", "x" * 100]), max_chunk_bytes=10, base_url="http://x")
            )

        assert raised.value.details["limit_bytes"] == 10

    def test_lines_before_the_oversize_one_are_yielded_first(self) -> None:
        seen = []
        with pytest.raises(ProviderProtocolError):
            for line in iter_capped_lines(
                iter(["a", "b", "x" * 100]), max_chunk_bytes=10, base_url="http://x"
            ):
                seen.append(line)

        assert seen == ["a", "b"]

    def test_the_cap_is_measured_in_utf8_bytes_not_characters(self) -> None:
        """A multi-byte character can push a line over the byte cap while under the char count."""
        multibyte_line = "日" * 4  # 3 bytes each in UTF-8 = 12 bytes, 4 characters
        with pytest.raises(ProviderProtocolError):
            list(iter_capped_lines(iter([multibyte_line]), max_chunk_bytes=10, base_url="http://x"))
