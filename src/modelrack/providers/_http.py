"""Domain-adjacent module — the HTTP plumbing every real adapter shares.

Imports :mod:`baseaicore`, this package's own errors and ``httpx``; the one module in this package
allowed to import ``httpx`` at all. Nothing above this layer sees a transport exception or a raw
status code — that is [spec §11.7](../../../docs/packages/modelrack/spec.md): *no adapter raises a
raw* ``httpx`` *exception*. :mod:`modelrack.providers.ollama` is the first caller; Phase 4's
OpenAI-compatible adapter is the second, which is why this is its own module rather than folded
into either — the development plan names it as a Phase 3 file precisely because two adapters need
it.

**Why classification of a connect failure is best-effort.** ``httpx``'s transport layer
(``httpcore``) catches the raw ``OSError`` a socket call raised, formats it with ``str()``, and
re-raises a *new* exception carrying only that string — the original ``errno`` attribute does not
survive the crossing (verified against httpx 0.28: ``ConnectionRefusedError`` and
``socket.gaierror`` both collapse to a bare ``httpx.ConnectError`` whose only distinguishing
feature is its message text). :func:`_classify_unavailable` recovers the distinction the way the
message itself preserves it — a leading ``[Errno N]`` where a **negative** ``N`` is a
``getaddrinfo`` (``EAI_*``) code and a **positive** one is a POSIX ``errno`` — documented here as
what it is: message-sniffing, not a structured read. It degrades to
:attr:`~modelrack.errors.ProviderUnavailableReason.NETWORK_ERROR` rather than guessing when the
pattern does not match, which is the same discipline
ADR-0016 applies to every other measurement
this package cannot make honestly.
"""

from __future__ import annotations

import json
import re
import ssl
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import httpx
from baseaicore import ValidationError

from modelrack.errors import (
    ProviderError,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DEFAULT_MAX_CHUNK_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "build_client",
    "is_loopback_host",
    "iter_capped_lines",
    "read_capped_json",
    "translate_stream_interruption",
    "translate_transport_error",
    "truncated_text",
    "validate_base_url",
]

DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0

DEFAULT_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
"""Spec §14's default total cap on one non-streamed response body."""

DEFAULT_MAX_CHUNK_BYTES: Final[int] = 8 * 1024 * 1024
"""Spec §14's default cap on one streamed chunk — here, one NDJSON line."""

_MAXIMUM_BODY_CHARACTERS: Final[int] = 512
_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
_LOOPBACK_HOSTNAMES: Final[frozenset[str]] = frozenset({"localhost"})

# httpcore/httpx format a wrapped OSError as "[Errno N] message", discarding the original
# exception's structured `.errno`. A getaddrinfo (EAI_*) failure carries a *negative* code in that
# position; a POSIX errno is positive. This is the only distinction left to read by the time the
# exception reaches here (see the module docstring).
_ERRNO_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[Errno (-?\d+)\]")
_CONNECTION_REFUSED_ERRNO: Final[int] = 111


def truncated_text(text: str, *, limit: int = _MAXIMUM_BODY_CHARACTERS) -> str:
    """Return ``text`` cut to ``limit`` characters, spec §13's "truncated" body for ``details``.

    An error's ``details`` is not where an unbounded provider response belongs — the caller's own
    artifact storage is — so this is applied at every site that puts a response body into an
    exception.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def is_loopback_host(host: str) -> bool:
    """Report whether ``host`` names this machine, for :attr:`ProviderHealth.is_remote`.

    Spec §14: a non-loopback provider is permitted, never silently allowed — the health result
    must flag it so a caller can surface egress. ``localhost`` is accepted by name without a DNS
    lookup (resolving it here would be an outbound call this package does not otherwise make);
    every other hostname is treated as remote even if it happens to resolve locally today, because
    a resolution can change and a health flag that could too would defeat the purpose.

    Args:
        host: The hostname or literal IP from a base URL, already stripped of port and brackets.

    Returns:
        ``True`` for ``localhost`` and a loopback literal (``127.0.0.0/8`` or ``::1``).
    """
    if host.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def validate_base_url(base_url: str) -> tuple[str, bool]:
    """Validate a provider base URL and report whether it names this machine.

    Args:
        base_url: The URL an adapter was constructed with.

    Returns:
        ``(base_url, is_remote)``.

    Raises:
        ValidationError: If the URL has no host, or its scheme is not ``http``/``https``
            (spec §14: "only http/https schemes accepted").
    """
    parts = urlsplit(base_url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValidationError(
            f"base_url must use http or https; got {base_url!r} with scheme "
            f"{parts.scheme or '(none)'!r}.",
            details={"field": "base_url", "value": base_url, "scheme": parts.scheme},
        )
    if not parts.hostname:
        raise ValidationError(
            f"base_url must name a host; got {base_url!r}.",
            details={"field": "base_url", "value": base_url},
        )
    return base_url, not is_loopback_host(parts.hostname)


def build_client(
    *,
    base_url: str,
    timeout: float | httpx.Timeout,
    headers: dict[str, str] | None,
    verify: ssl.SSLContext | str | bool,
) -> httpx.Client:
    """Construct the one pooled client an adapter instance uses for every call.

    A single :class:`httpx.Client` per adapter instance is what spec §15 means by "connection
    pooling via a shared ``httpx.Client``" — a new client per request would open a new TCP
    connection (and, over HTTPS, a new TLS handshake) every time.

    Args:
        base_url: Already validated by :func:`validate_base_url`.
        timeout: A uniform timeout in seconds, or a fully configured :class:`httpx.Timeout` for
            distinct connect/read/write/pool limits (spec §12).
        headers: Sent on every request from this client — where a caller's API key lives, for the
            adapter that has one.
        verify: TLS verification, passed straight through to ``httpx``.

    Returns:
        A client with redirects disabled: a local inference runtime has no legitimate redirect to
        follow, and following one would be a silent change of the server actually being talked to.
    """
    return httpx.Client(
        base_url=base_url,
        timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
        headers=headers,
        verify=verify,
        follow_redirects=False,
    )


def _classify_unavailable(message: str) -> ProviderUnavailableReason:
    """Best-effort classification of a connect failure from its message text alone.

    See the module docstring: httpx discards the structured ``errno`` a raw ``OSError`` carried,
    so this reads the same ``[Errno N]`` text the platform's own ``strerror``/``gai_strerror``
    produced. Never guessed past what the text actually supports —
    :attr:`~ProviderUnavailableReason.NETWORK_ERROR` is the honest fallback.
    """
    lowered = message.lower()
    if "ssl" in lowered or "certificate" in lowered:
        return ProviderUnavailableReason.TLS_FAILURE
    match = _ERRNO_PATTERN.search(message)
    if match is not None:
        code = int(match.group(1))
        if code < 0:
            return ProviderUnavailableReason.DNS_FAILURE
        if code == _CONNECTION_REFUSED_ERRNO:
            return ProviderUnavailableReason.CONNECTION_REFUSED
    return ProviderUnavailableReason.NETWORK_ERROR


def translate_transport_error(exc: httpx.HTTPError, *, base_url: str) -> ProviderError:
    """Translate a raw ``httpx`` exception into the typed error spec §13 names for it.

    The one function every adapter method funnels a transport-layer failure through, so that
    "no adapter raises a raw ``httpx`` exception" (spec §11.7) is true by construction rather than
    by every call site remembering to wrap its own.

    Args:
        exc: Whatever ``httpx`` raised — during connection, during the request, or while reading
            the response.
        base_url: Included in a :class:`~modelrack.errors.ProviderUnavailable`'s ``details``
            (spec §13), because "unavailable" without an address does not tell a caller what to
            check.

    Returns:
        :class:`~modelrack.errors.ProviderTimeout` for any :class:`httpx.TimeoutException`;
        :class:`~modelrack.errors.ProviderProtocolError` for a malformed HTTP response
        (:class:`httpx.RemoteProtocolError` — the server sent something that was not HTTP);
        :class:`~modelrack.errors.ProviderUnavailable` for everything else ``httpx`` raises
        reaching the server at all, which covers connection refusal, DNS failure, TLS failure and
        every other transport error this adapter does not distinguish further.
    """
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeout(
            f"The provider at {base_url} did not answer in time.",
            details={"base_url": base_url},
        )
    if isinstance(exc, httpx.RemoteProtocolError):
        return ProviderProtocolError(
            f"The provider at {base_url} sent a response that was not valid HTTP.",
            details={"base_url": base_url, "body": truncated_text(str(exc))},
        )
    return ProviderUnavailable(
        f"Cannot reach the provider at {base_url}: {exc}",
        details={"base_url": base_url, "reason": _classify_unavailable(str(exc)).value},
    )


def translate_stream_interruption(exc: httpx.HTTPError, *, base_url: str) -> ProviderError:
    """Translate an ``httpx`` exception raised *after* a stream has already yielded content.

    Deliberately not the same mapping as :func:`translate_transport_error`. A connection that
    was already open and delivering deltas is not "unreachable" the way a failed connect
    attempt is — :class:`~modelrack.errors.ProviderUnavailable`'s own docstring is explicit that
    it means the provider "could not be reached **at all**", which a stream that had already
    produced output contradicts. A reset or malformed-framing failure here reads instead as spec
    §13's "stream truncated without a terminal chunk" row: the connection stopped, mid-story,
    for a reason the caller cannot act on differently than any other truncation. A timeout is the
    one exception kept distinct even here, because a caller's retry/backoff policy legitimately
    treats "it was too slow" differently from "it broke".

    Args:
        exc: What ``httpx`` raised while a stream's iterator was already in progress.
        base_url: For the resulting error's diagnostics.

    Returns:
        :class:`~modelrack.errors.ProviderTimeout` for a :class:`httpx.TimeoutException`;
        :class:`~modelrack.errors.ProviderProtocolError` for everything else.
    """
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeout(
            f"The provider at {base_url} stopped answering part-way through a stream.",
            details={"base_url": base_url},
        )
    return ProviderProtocolError(
        f"The stream from {base_url} ended abnormally: {exc}",
        details={"base_url": base_url, "body": truncated_text(str(exc))},
    )


def read_capped_json(
    response: httpx.Response, *, max_bytes: int, base_url: str
) -> dict[str, Any] | list[Any]:
    """Read a non-streamed response body under a size cap and parse it as JSON.

    Reads incrementally rather than via ``response.json()`` directly, because the whole point of a
    cap (spec §14: "prevent memory exhaustion") is never holding more than the limit in memory —
    checking the size only *after* an unbounded read would have already spent the memory it exists
    to save.

    Args:
        response: An already-received response, not yet read.
        max_bytes: The cap. A body at or under it is read in full; over it, reading stops and the
            response is closed without buffering the rest.
        base_url: For the oversize error's diagnostics.

    Returns:
        The parsed JSON body — an object or an array, whichever the provider sent.

    Raises:
        ProviderProtocolError: If the body exceeds ``max_bytes``, or is not parsable JSON.
    """
    chunks: list[bytes] = []
    total = 0
    oversize = False
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            oversize = True
            break
        chunks.append(chunk)
    response.close()
    if oversize:
        raise ProviderProtocolError(
            f"The response from {base_url} exceeded the {max_bytes}-byte size cap.",
            details={"base_url": base_url, "limit_bytes": max_bytes},
        )
    body = b"".join(chunks)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderProtocolError(
            f"The response from {base_url} was not valid JSON.",
            details={"base_url": base_url, "body": truncated_text(body.decode("utf-8", "replace"))},
        ) from exc
    if not isinstance(parsed, dict | list):
        raise ProviderProtocolError(
            f"The response from {base_url} was valid JSON but not an object or array.",
            details={"base_url": base_url, "body": truncated_text(body.decode("utf-8", "replace"))},
        )
    return parsed


def iter_capped_lines(
    lines: Iterable[str], *, max_chunk_bytes: int, base_url: str
) -> Iterable[str]:
    """Yield each line, raising if any single one exceeds the per-chunk size cap.

    Applied to the already-reassembled lines from :meth:`httpx.Response.iter_lines`, which is what
    keeps a NDJSON line intact even when the transport split it — or a multi-byte character inside
    it — across two TCP reads. The cap this function enforces is spec §14's *per-chunk* figure
    (default 8 MiB), distinct from :func:`read_capped_json`'s *total-response* cap.
    """
    for line in lines:
        if len(line.encode("utf-8")) > max_chunk_bytes:
            raise ProviderProtocolError(
                f"A streamed line from {base_url} exceeded the {max_chunk_bytes}-byte "
                "per-chunk size cap.",
                details={"base_url": base_url, "limit_bytes": max_chunk_bytes},
            )
        yield line
