"""Domain module — the OpenAI chat-completions wire shape, translated to this package's types.

Imports :mod:`baseaicore` and this package's own types; performs no I/O and reads no clock. Every
function here is pure, which is what lets both adapters that speak this shape assert against
recorded fixtures without a server.

Extracted from :mod:`modelrack.providers.openai_compatible` when a second adapter needed it: a
llama.cpp server answers chat requests in exactly this shape (with its own extensions layered on
top, which :mod:`modelrack.providers._llamacpp_wire` reads), and two copies of the tool-call
assembly logic would be two places for the "malformed arguments are preserved, not dropped" rule
to drift apart. Nothing here changed in the move; the OpenAI-compatible adapter's own tests are
what prove it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from modelrack.types import FinishReason, Message, ResponseFormat, ResponseFormatKind, ToolCall

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from modelrack.types import ToolDefinition

__all__ = [
    "extract_error",
    "finish_reason_for",
    "first_choice",
    "iter_sse_events",
    "message_payload",
    "parse_tool_calls",
    "request_tool_definitions",
    "response_format_payload",
    "tool_call_fragment",
    "tool_call_from_parts",
    "tool_call_index",
]

_FINISH_REASON_MAP: Final[dict[str, FinishReason]] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``payload["choices"][0]``, or ``{}`` when the array is missing or empty."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return {}


def message_payload(message: Message) -> dict[str, Any]:
    """Build one OpenAI-shaped chat message from a :class:`~modelrack.types.Message`."""
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, sort_keys=True),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload


def request_tool_definitions(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Build the OpenAI-shaped tool list from :class:`~modelrack.types.ToolDefinition` values."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in tools
    ]


def response_format_payload(response_format: ResponseFormat) -> dict[str, Any]:
    """Build the OpenAI-shaped ``response_format`` object for a text/JSON/schema request."""
    if response_format.kind is ResponseFormatKind.JSON:
        return {"type": "json_object"}
    if response_format.kind is ResponseFormatKind.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "modelrack_response",
                "schema": dict(response_format.schema or {}),
                "strict": True,
            },
        }
    return {"type": "text"}


def tool_call_from_parts(
    *, call_id: str | None, name: str | None, raw_arguments: str | None, fallback_id: str
) -> ToolCall:
    """Assemble one :class:`~modelrack.types.ToolCall` from its wire pieces.

    Shared between the non-streaming path (one complete entry) and the streaming path (fragments
    accumulated across many chunks) — both eventually have the same three pieces: an id the
    provider may or may not have sent, a name, and an ``arguments`` string that may or may not be
    valid JSON. A string that will not parse is kept as ``raw_arguments`` and yields empty
    ``arguments`` rather than raising: a malformed-arguments response is a real failure mode this
    package's :mod:`modelrack.testing` scripts on purpose, and dropping it here would make that
    scripted case untestable against a real adapter.
    """
    arguments: dict[str, Any] = {}
    if raw_arguments:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            arguments = parsed
    return ToolCall(
        id=call_id if call_id else fallback_id,
        name=name if name else "unknown",
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def parse_tool_calls(raw_calls: Any, *, call_prefix: str) -> tuple[ToolCall, ...]:  # noqa: ANN401
    """Parse a non-streaming response's ``message.tool_calls`` into the vocabulary's tuple."""
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for offset, entry in enumerate(raw_calls):
        if not isinstance(entry, Mapping):
            continue
        call_id = entry.get("id")
        function = entry.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        raw_arguments = function.get("arguments")
        calls.append(
            tool_call_from_parts(
                call_id=call_id if isinstance(call_id, str) and call_id else None,
                name=name if isinstance(name, str) and name else None,
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
                fallback_id=f"{call_prefix}-{offset}",
            )
        )
    return tuple(calls)


def tool_call_index(fragment: Mapping[str, Any], *, fallback: int) -> int:
    """Return a streamed tool-call fragment's ``index``, or ``fallback`` if it is absent/invalid."""
    index = fragment.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return fallback
    return index


def tool_call_fragment(
    fragment: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Return ``(id, name, arguments_fragment)`` from one streamed tool-call delta entry."""
    call_id = fragment.get("id")
    function = fragment.get("function")
    function = function if isinstance(function, Mapping) else {}
    name = function.get("name")
    arguments = function.get("arguments")
    return (
        call_id if isinstance(call_id, str) and call_id else None,
        name if isinstance(name, str) and name else None,
        arguments if isinstance(arguments, str) and arguments else None,
    )


def finish_reason_for(raw: Any, *, has_tool_calls: bool) -> FinishReason:  # noqa: ANN401
    """Map the wire ``finish_reason`` string to the vocabulary's finish reason.

    A message carrying tool calls wins regardless of what ``finish_reason`` said, the same
    defensive precedence :func:`modelrack.providers._ollama_wire.finish_reason_for` applies —
    :class:`~modelrack.types.GenerationResult` requires ``TOOL_CALLS`` whenever tool calls are
    present, and a server that populated ``tool_calls`` but reported a stale ``finish_reason``
    should not be able to violate that invariant.
    """
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    if isinstance(raw, str):
        return _FINISH_REASON_MAP.get(raw, FinishReason.UNKNOWN)
    return FinishReason.UNKNOWN


def extract_error(payload: Any) -> tuple[str | None, str | None]:  # noqa: ANN401 — provider JSON
    """Return ``(message, code)`` from a parsed error body, or ``(None, None)`` if there is none.

    Handles both documented error shapes: ``{"error": {"message": ..., "code": ...}}`` (the
    OpenAI-originated convention every server here follows) and the bare-string
    ``{"error": "..."}`` a few minimal servers still send.
    """
    if not isinstance(payload, Mapping):
        return None, None
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error, None
    if isinstance(error, Mapping):
        message = error.get("message")
        code = error.get("code")
        return (
            message if isinstance(message, str) and message else None,
            code if isinstance(code, str) else None,
        )
    return None, None


def iter_sse_events(lines: Iterable[str]) -> Iterator[str]:
    """Group raw SSE lines into event ``data`` payloads.

    Implements just the subset of the `Server-Sent Events grammar
    <https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation>`_
    this protocol uses: a ``data:`` field line (its value joined across consecutive ``data:``
    lines with ``\\n``, per the spec), a blank line dispatching whatever has been buffered, and a
    ``:``-prefixed line — a comment, sent by some servers as a keep-alive — ignored outright. Any
    other field name (``event:``, ``id:``, ``retry:``) is ignored: nothing this adapter reads uses
    them. A final event with no trailing blank line is still dispatched, defensively, since a
    stream ending exactly on ``data: [DONE]`` with no newline after it is a shape worth surviving
    rather than silently dropping the last event.
    """
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)
