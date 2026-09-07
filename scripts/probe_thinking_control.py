#!/usr/bin/env python3
"""Measure what an Ollama model does with the ``think`` control, and record the raw stream.

`ADR-0099 <../docs/adr/0099-a-task-profile-may-ask-for-reduced-thinking.md>`_ made the control
reachable from a task profile and routing enforces that a candidate's *provider* can carry it.
Nothing enforces that the *model* honours it, and the difference is not academic: a model that
accepts ``think: false`` and relocates its reasoning into ``content`` fails a caller's
``require_valid_json`` and, on at least one model, ends its stream with no terminal chunk. This
script is what produced the table in the suite's ``apps/loadcoach/routing.md`` §2 and the capture
behind ``packages/modelrack/spec.md``'s ``ProviderProtocolError`` row (I6, 2026-09-07).

It talks straight to Ollama over HTTP rather than through :class:`~modelrack.providers.ollama.\
OllamaProvider`, deliberately: a capture taken with no adapter in the path is what separates
"the parser mis-read the stream" from "the server ended it". It writes nothing outside its
output directory, changes no configuration, and is **not** part of any test run.

Re-measure after an Ollama upgrade or a new model pull::

    python scripts/probe_thinking_control.py probe --out /tmp/i6
    python scripts/probe_thinking_control.py capture gpt-oss:20b --runs 6 --out /tmp/i6

``probe`` streams two requests per model — ``think: true`` then ``think: false`` — and prints one
JSON summary per cell. ``capture`` streams one cell repeatedly and writes every raw read, with a
millisecond offset, to one file per run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# The prompt is one string for every model and every cell, so a column means the same thing
# everywhere. This is PromptCadence's `planner.draft` 1.1.0 record rendered in I3 gate D's shape:
# long enough to make a model reason, and the body under which the stream defect reproduces.
_SYSTEM = (
    "You draft plans for a governed agent harness. You propose; you do not execute, and you do "
    "not decide whether the plan runs — a deterministic validator checks the document and an "
    "approval policy decides each step. Return only the JSON document: no prose before or after "
    "it, no code fence."
)
_USER = """TASK
Read the repository at /home/jpk/ai/suite/PromptCadence and write a short note naming the three \
files a new contributor should read first and why.

DATA CLASSIFICATION
The task's data is classified "internal". A step may declare that level or a lower one, never a \
higher one. A step that reads the task text itself handles data at the task's level.

TOOLS AVAILABLE TO THIS TRAJECTORY
list_dir — List the entries of one directory inside the trajectory's read roots.
read_file — Read one file inside the trajectory's read roots, as text.
grep — Search the read roots for a regular expression and return matching lines.
write_file — Write one file inside the trajectory's write root.
run_command — Run one allowlisted command inside the sandbox.
Declare only tools from this list; [] when a step needs none.

TIERS
local_fast — local, ceiling confidential.
local_reasoning — local, ceiling confidential.
remote_frontier — remote, ceiling internal.
Declare exactly one tier name from this list per step.

PLAN
Produce between 1 and 20 steps, each one unit of work a single model session can finish with the \
tools it declares. Steps form a directed acyclic graph through depends_on. Keep the plan as \
short as the task allows: a one-step plan is the right plan for a one-step task.

FOR EACH STEP
- step_id: a short identifier unique within the plan, such as "s1".
- description: what the step must achieve, in one or two sentences.
- depends_on: the step_ids that must finish first; [] when none.
- tools: the tools the step will call, from the list above.
- tier: one tier name from the list above.
- data_classification: "public", "internal" or "confidential", at or below "internal".
- expected_turns: an integer from 1 to 100, your estimate of tool round trips. Advisory only; it \
sizes nothing.

RETURN
{"steps": [{"step_id": "...", "description": "...", "depends_on": [], "tools": [], "tier": \
"...", "data_classification": "...", "expected_turns": 1}]}
and nothing else."""

# `tools.plan`'s execution block, which is the profile the control was measured under.
_TEMPERATURE = 0.1
_MAX_OUTPUT_TOKENS = 4096
_TIMEOUT_SECONDS = 900.0


def build_body(
    model_name: str, *, think: bool | None, json_format: bool, unload: bool = False
) -> dict[str, Any]:
    """Build one streamed ``/api/chat`` body.

    Args:
        model_name: The Ollama tag to address.
        think: Ollama's top-level ``think`` key. ``None`` sends no key at all, which is
            byte-identical to a request built before the field existed.
        json_format: Whether to send ``format: "json"``. It is half the trigger — on gpt-oss:20b
            the stream dies only when this and ``think: false`` are sent together.
        unload: Send ``keep_alive: 0``, so the model is evicted when the request finishes. One
            load per model instead of one per request.

    Returns:
        The request body, ready for :meth:`httpx.Client.stream`.
    """
    body: dict[str, Any] = {
        "model": model_name,
        "stream": True,
        "options": {"temperature": _TEMPERATURE, "num_predict": _MAX_OUTPUT_TOKENS},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER},
        ],
    }
    if json_format:
        body["format"] = "json"
    if think is not None:
        body["think"] = think
    if unload:
        body["keep_alive"] = 0
    return body


def probe_one(base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    """Stream one request and report where the model put its reasoning.

    Judges nothing about answer quality — a model is never a test oracle. It records which
    channel carried what, whether the accumulated ``content`` parses as JSON, and how the stream
    ended, including the two ways it can end badly: Ollama's own mid-stream ``{"error": …}``
    object, and a well-formed stream that simply stops with no ``done: true`` line.

    Args:
        base_url: Where Ollama is listening.
        body: A body from :func:`build_body`.

    Returns:
        One row of the measurement, JSON-serialisable.
    """
    content: list[str] = []
    thinking: list[str] = []
    saw_thinking_key = False
    tool_calls = 0
    saw_done = False
    done_reason: str | None = None
    ended = "clean-eof"
    detail = ""
    status_code: int | None = None
    started_at = time.monotonic()
    try:
        with (
            httpx.Client(base_url=base_url, timeout=_TIMEOUT_SECONDS) as client,
            client.stream("POST", "/api/chat", json=body) as response,
        ):
            status_code = response.status_code
            for line in response.iter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("error"), str):
                    ended, detail = "error-object", payload["error"]
                    continue
                message = payload.get("message") or {}
                if "thinking" in message:
                    saw_thinking_key = True
                    thinking.append(message["thinking"] or "")
                content.append(message.get("content") or "")
                tool_calls += len(message.get("tool_calls") or ())
                if payload.get("done") is True:
                    saw_done, done_reason = True, payload.get("done_reason")
    except Exception as exc:  # noqa: BLE001 — recording whatever arrives is the whole point
        ended, detail = type(exc).__name__, str(exc)
    text = "".join(content)
    content_is_json: bool | None = None
    if text.strip():
        try:
            json.loads(text)
            content_is_json = True
        except json.JSONDecodeError:
            content_is_json = False
    return {
        "model": body["model"],
        "think": body.get("think"),
        "format": body.get("format"),
        "status_code": status_code,
        "saw_done": saw_done,
        "done_reason": done_reason,
        "ended": ended,
        "detail": detail[:300],
        "thinking_key_present": saw_thinking_key,
        "thinking_chars": len("".join(thinking)),
        "content_chars": len(text),
        "content_is_json": content_is_json,
        "tool_calls": tool_calls,
        "wall_ms": round((time.monotonic() - started_at) * 1000, 1),
        "content_head": text[:220],
        "thinking_head": "".join(thinking)[:220],
    }


def capture_one(base_url: str, body: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Stream one request and write every raw read, with its millisecond offset, to a file.

    This is the half that answers "did the server end the stream, or did the parser lose it?" —
    it reassembles nothing and interprets nothing, so a stream that ends with no ``done: true``
    line is visible as the absence of that line rather than as an exception type.

    Args:
        base_url: Where Ollama is listening.
        body: A body from :func:`build_body`.
        destination: The file to write. Overwritten if it exists.

    Returns:
        A summary row: the status, how it ended, and whether a terminal chunk ever arrived.
    """
    started_at = time.monotonic()
    saw_done = False
    ended = "clean-eof"
    detail = ""
    status_code: int | None = None
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(f"# body: {json.dumps(body)}\n")
        try:
            with (
                httpx.Client(base_url=base_url, timeout=_TIMEOUT_SECONDS) as client,
                client.stream("POST", "/api/chat", json=body) as response,
            ):
                status_code = response.status_code
                handle.write(f"# status: {status_code}\n")
                handle.write(f"# headers: {json.dumps(dict(response.headers))}\n")
                for chunk in response.iter_bytes():
                    offset_ms = (time.monotonic() - started_at) * 1000
                    handle.write(f"{offset_ms:9.1f}\t{chunk!r}\n")
                    if b'"done":true' in chunk.replace(b" ", b""):
                        saw_done = True
        except Exception as exc:  # noqa: BLE001 — recording whatever arrives is the whole point
            ended, detail = type(exc).__name__, str(exc)
            handle.write(f"# EXCEPTION {ended}: {detail}\n")
    return {
        "model": body["model"],
        "think": body.get("think"),
        "format": body.get("format"),
        "status_code": status_code,
        "saw_done": saw_done,
        "ended": ended,
        "detail": detail[:300],
        "raw_path": str(destination),
        "wall_ms": round((time.monotonic() - started_at) * 1000, 1),
    }


def installed_models(base_url: str) -> list[str]:
    """Return every model tag this Ollama has pulled, largest last is not guaranteed."""
    response = httpx.get(f"{base_url}/api/tags", timeout=30.0)
    response.raise_for_status()
    return [model["name"] for model in response.json()["models"]]


def _slug(model_name: str) -> str:
    """Return a filename-safe form of an Ollama tag."""
    return model_name.replace("/", "_").replace(":", "-")


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand and print a JSON line per request."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=Path(), help="Where results are written.")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Two cells per model: think true, then think false.")
    probe.add_argument("models", nargs="*", help="Tags; every installed model when omitted.")

    capture = sub.add_parser("capture", help="Record the raw bytes of one cell, repeatedly.")
    capture.add_argument("model")
    capture.add_argument("--runs", type=int, default=6)
    capture.add_argument("--think", choices=("true", "false", "unset"), default="false")
    capture.add_argument("--no-json-format", action="store_true")

    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if args.command == "probe":
        for model_name in args.models or installed_models(args.base_url):
            for think in (True, False):
                # `unload` rides on the second cell: one load per model, not one per request.
                body = build_body(model_name, think=think, json_format=True, unload=not think)
                row = probe_one(args.base_url, body)
                rows.append(row)
                print(json.dumps(row), flush=True)
    else:
        think_value: bool | None = {"true": True, "false": False, "unset": None}[args.think]
        for run_index in range(args.runs):
            body = build_body(args.model, think=think_value, json_format=not args.no_json_format)
            path = args.out / f"{_slug(args.model)}__think-{args.think}__{run_index}.raw"
            row = capture_one(args.base_url, body, path)
            rows.append(row)
            print(json.dumps(row), flush=True)

    (args.out / f"{args.command}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
