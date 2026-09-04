"""Provider adapter — llama.cpp's server, spawned and supervised by this package.

Imports :mod:`baseaicore`, this package's own types, ``httpx`` (through :mod:`_http`) and the
standard library; performs network I/O to a server it started itself, local file I/O under two
directories the constructing application names, and process I/O through an injected launcher.
Every test in ``tests/unit/test_llamacpp_adapter.py`` runs against a recorded transport and a fake
launcher, because the machine this was written on has no ``llama-server`` — and neither does CI.

The third real adapter, and the first that does not talk to a daemon somebody else manages
([ADR-0062](../../../docs/adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md)):
``load()`` *spawns* ``llama-server`` with the base GGUF and the profile's flags, ``unload()``
terminates it, ``list_resident()`` reads the process table. The ``Provider`` protocol did not
change to make that fit — decision 1 of that ADR calls that the evidence the seam was drawn in
the right place — and this module is the proof.

**Where things live.** This module owns the client, the HTTP calls, the streaming state machine
and the residency decisions. :mod:`modelrack.providers._llamacpp_process` owns everything about
a process: spawning, the health wait, kill-tree, pid files, orphan recovery, ports.
:mod:`modelrack.providers._llamacpp_wire` owns every translation of a payload, pure and
fixture-testable. :mod:`modelrack.providers._gguf` reads a model file's header and hashes its
content. :mod:`modelrack.providers._openai_wire` supplies the chat-completions shape the server's
chat endpoint answers in, shared with the OpenAI-compatible adapter.

**Identity is digest-bound here.** Ollama names a model by a tag it can repoint, so its
identities are ``name_only`` unless it reports a digest; this adapter hashes the file it serves,
so every identity carries the sha256 of the artifact — the identity-confidence gain ADR-0062
decision 6 names — and a request pinned to a digest that no longer matches the file at that path
is refused with :class:`~modelrack.errors.ModelNotFound` rather than served from different weights.

**What a digest costs, and the decision about it.** Hashing a 9 GB file takes seconds to tens of
seconds; hashing the reference machine's directory takes about 45 s. The header a descriptor is
built from is cached the way every adapter's metadata is — the in-memory
:class:`~modelrack.cache.MetadataCache`, its TTL, and ``refresh=True`` — and additionally
checked against the file's :class:`~modelrack.providers._gguf.ArtifactStamp` on every hit, so a
replaced file is never described from its predecessor's header. The digest is **not** on a TTL:
a content hash does not go stale with time, only with content, so it is keyed by path *and*
stamp in a :class:`DigestStore` and recomputed when the stamp changes or ``refresh=True`` says
so. The default store is :class:`JsonFileDigestStore`, one versioned JSON file inside the
application-named ``state_dir`` (ADR-0071): the first discovery against a fresh ``state_dir``
hashes every file once, and every process after that pays a ``stat`` per file. It is clearable
through :meth:`LlamaCppProvider.clear_digest_cache` or by deleting the file; nothing here
substitutes a cheaper digest for the content digest the identity contract names.

**Usage is read to ADR-0070 from the first commit** — see :mod:`_llamacpp_wire`'s docstring for
the three cases, and for the ``tokens_cached`` trap that module exists to avoid.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
from baseaicore import (
    UNSUPPORTED,
    AdapterIdentity,
    IdentityConfidence,
    ModelIdentity,
    ProviderKind,
    ValidationError,
    elapsed_ms,
    is_supported,
    monotonic_ns,
    utc_now,
    verify_adapter_base_compatibility,
)

from modelrack.adapters import AdapterRegistration, AdapterState, AdapterStatus
from modelrack.cache import DEFAULT_METADATA_TTL_SECONDS, CacheStats, MetadataCache
from modelrack.errors import (
    AdapterNotFound,
    CapabilityUnsupported,
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderUnavailable,
    ProviderUnavailableReason,
)
from modelrack.events import EventEmitter
from modelrack.provider import (
    LoadResult,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
)
from modelrack.providers._gguf import (
    ArtifactStamp,
    GgufFormatError,
    GgufHeader,
    read_gguf_header,
    sha256_of_file,
)
from modelrack.providers._http import (
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    build_client,
    iter_capped_lines,
    read_capped_json,
    translate_stream_interruption,
    translate_transport_error,
    truncated_text,
)
from modelrack.providers._llamacpp_process import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_PORT_RANGE,
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_STDERR_TAIL_BYTES,
    LaunchSpec,
    LlamaServerSupervisor,
    PosixProcessTable,
    ProcessLauncher,
    ProcessTable,
    ServerHandle,
    ServerProcess,
    SubprocessLauncher,
    loopback_port_is_free,
)
from modelrack.providers._llamacpp_wire import (
    DEFAULT_SERVER_EXECUTABLE,
    FORBIDDEN_LAUNCH_FLAGS,
    FORBIDDEN_REQUEST_KEYS,
    SLOT_PINNING_KEYS,
    LlamaCppError,
    ServerAdapter,
    build_chat_body,
    build_completion_body,
    build_descriptor,
    build_launch_argv,
    completion_finish_reason,
    header_kind,
    identity_for,
    is_shard,
    launch_flags,
    lora_field,
    model_name_for,
    read_backend_timing,
    read_build_info,
    read_chat_usage,
    read_completion_usage,
    read_error,
    read_lora_adapters,
    read_served_context,
)
from modelrack.providers._openai_wire import (
    finish_reason_for,
    first_choice,
    iter_sse_events,
    parse_tool_calls,
    tool_call_fragment,
    tool_call_from_parts,
    tool_call_index,
)
from modelrack.residency import require_force_unload, require_residency_query
from modelrack.streaming import (
    StreamCompleted,
    StreamFailed,
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.types import GenerationResult, Timing

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence
    from datetime import datetime

    from baseaicore import Measurement, ModelDescriptor, RuntimeProfile

    from modelrack.events import EventCallback
    from modelrack.streaming import StreamEvent
    from modelrack.types import GenerationRequest, ToolCall

__all__ = [
    "DEFAULT_PORT_RANGE",
    "DEFAULT_SERVER_EXECUTABLE",
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DigestStore",
    "InMemoryDigestStore",
    "JsonFileDigestStore",
    "LaunchSpec",
    "LlamaCppProvider",
    "PosixProcessTable",
    "ProcessLauncher",
    "ProcessTable",
    "ServerHandle",
    "ServerProcess",
    "SubprocessLauncher",
]

logger = logging.getLogger(__name__)

_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    streaming=True,
    tool_calling=True,
    structured_output=True,
    json_mode=True,
    token_counts=True,
    token_level_chunks=True,
    thinking_control=False,
    logprobs=False,
    force_unload=True,
    residency_query=True,
    kv_metrics=False,
    context_configurable=True,
    embedding=False,
    adapter_hot_swap=True,
)
"""What a llama-server this adapter spawns can do.

``token_level_chunks`` is ``True`` on the strength of the server's own loop: it emits one partial
result per decoded token and holds back only an incomplete UTF-8 sequence until the next token
completes it (``server-context.cpp``, ``send_partial_response``), so a streamed delta is one
token's text. ``thinking_control`` is ``False``: the server *reports* reasoning
(``reasoning_content``, read into ``thinking`` where present) but requesting or suppressing it is
a per-model template argument this phase does not expose. ``force_unload``, ``residency_query``
and ``context_configurable`` are ``True`` because they are literally what supervision is —
spawn, terminate, read the process table, ``--ctx-size``. ``logprobs``, ``kv_metrics`` and
``embedding`` stay ``False``: the server offers each, and nothing here reads them, which is what
"a capability nobody tested" means (ADR-0007 rule 2). ``adapter_hot_swap`` is the one flag ``True``
in this package: llama-server registers adapters at launch and selects among them per request with
no reload (ADR-0062), and this is the adapter that supervises that process.
"""

_REQUEST_HEADERS: Final[dict[str, str]] = {"Content-Type": "application/json"}
_HEALTH_PATH: Final[str] = "/health"
_PROPS_PATH: Final[str] = "/props"
_LORA_PATH: Final[str] = "/lora-adapters"
_COMPLETION_PATH: Final[str] = "/completion"
_CHAT_PATH: Final[str] = "/v1/chat/completions"
_DONE_SENTINEL: Final[str] = "[DONE]"
_GGUF_GLOB: Final[str] = "*.gguf"
_MODEL_KIND: Final[str] = "model"


class DigestStore(Protocol):
    """Where computed artifact digests are kept between calls, keyed by path and stamp.

    The seam that decides whether a 40 GB directory is hashed once per process or once per
    machine. The key already encodes the file's :class:`~modelrack.providers._gguf.ArtifactStamp`,
    so a store never has to decide whether an entry is stale: a changed file has a different key
    and simply misses. A store may forget anything at any time; the only cost is a re-hash.

    The default is :class:`JsonFileDigestStore` over ``<state_dir>/digests.json`` (ADR-0071);
    :class:`InMemoryDigestStore` is the no-persistence alternative, and an application may inject
    any other implementation backed by its own data root.
    """

    def get(self, key: str) -> str | None:
        """Return the digest stored under ``key``, or ``None``."""
        ...

    def put(self, key: str, digest: str, *, path: Path) -> None:
        """Store ``digest`` under ``key``, for the file at ``path``."""
        ...

    def clear(self) -> None:
        """Forget every digest, so the next discovery hashes again."""
        ...


class InMemoryDigestStore:
    """The default :class:`DigestStore`: a dict that lives and dies with the adapter.

    Thread-safe, inspectable through :func:`len`, and clearable for tests. Holds digests for the
    process's lifetime — there is no TTL, deliberately, because a content hash is invalidated by
    content changing (which the key captures), not by time passing.
    """

    __slots__ = ("_entries", "_lock")

    def __init__(self) -> None:
        """Create an empty store."""
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        """Return the digest stored under ``key``, or ``None``."""
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, digest: str, *, path: Path) -> None:
        """Store ``digest`` under ``key``, replacing any earlier value. ``path`` is not used."""
        with self._lock:
            self._entries[key] = digest

    def clear(self) -> None:
        """Forget every digest, so the next discovery hashes again."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return how many digests are held."""
        with self._lock:
            return len(self._entries)


_DIGEST_FILE_VERSION: Final[int] = 1
_DIGEST_FILE_NAME: Final[str] = "digests.json"


class JsonFileDigestStore:
    """The default :class:`DigestStore`: one versioned JSON file in the application's ``state_dir``.

    ADR-0071's decision, as code. A content digest is invalidated by the file's bytes changing —
    which the key's stamp captures — and by nothing else, so holding it only in memory would
    re-pay seconds to a minute of hashing on every process start for no information gained.

    Every write reads the current file, merges the new entry, drops entries whose ``path`` no
    longer exists, and replaces the file through a temporary sibling and :func:`os.replace`, so
    two processes writing the same file cannot corrupt it and a race costs at most one re-hash.
    An unreadable or differently versioned file is treated as empty, logged at DEBUG, and
    overwritten on the next write: it is a cache, and nothing may fail because of it.

    Args:
        path: The file. Its directory is created on the first write. ``<state_dir>/digests.json``
            when the adapter builds it.
        clock: Where each entry's ``computed_at`` comes from, for a human reading the file.
    """

    __slots__ = ("_clock", "_lock", "_path")

    def __init__(self, path: Path, *, clock: Callable[[], datetime] = utc_now) -> None:
        """Bind to a file; nothing is read or written until asked."""
        self._path = path
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Where the digests live."""
        return self._path

    def get(self, key: str) -> str | None:
        """Return the digest stored under ``key``, or ``None``."""
        with self._lock:
            entry = self._read().get(key)
        if not isinstance(entry, dict):
            return None
        digest = entry.get("digest")
        return digest if isinstance(digest, str) and digest else None

    def put(self, key: str, digest: str, *, path: Path) -> None:
        """Merge ``digest`` for ``path`` into the file, atomically, pruning entries for files
        that no longer exist.
        """
        with self._lock:
            entries = self._read()
            entries[key] = {
                "path": str(path),
                "digest": digest,
                "computed_at": self._clock().isoformat(),
            }
            kept = {
                stored_key: entry
                for stored_key, entry in entries.items()
                if isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and Path(entry["path"]).exists()
            }
            self._write(kept)

    def clear(self) -> None:
        """Remove the file. The next discovery hashes every file again."""
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return

    def __len__(self) -> int:
        """Return how many digests the file currently holds."""
        with self._lock:
            return len(self._read())

    def _read(self) -> dict[str, Any]:
        """Return the file's entries, or an empty mapping for an absent, unreadable or
        differently versioned file.
        """
        try:
            payload = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.debug(
                "llamacpp.digests.unreadable", extra={"path": str(self._path), "error": str(exc)}
            )
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _DIGEST_FILE_VERSION
            or not isinstance(payload.get("entries"), dict)
        ):
            logger.debug("llamacpp.digests.unrecognised", extra={"path": str(self._path)})
            return {}
        entries: dict[str, Any] = payload["entries"]
        return entries

    def _write(self, entries: dict[str, Any]) -> None:
        """Replace the file atomically with ``entries``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {"version": _DIGEST_FILE_VERSION, "entries": entries}, indent=2, sort_keys=True
            )
            + "\n"
        )
        temporary.replace(self._path)


@dataclass(frozen=True, slots=True)
class _Entry:
    """One served file as discovery found it: its name, path, header and when it was read."""

    name: str
    path: Path
    header: GgufHeader
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class _Selection:
    """The adapter axis one request resolved to, carried to the result that reports it."""

    name: str
    identity: AdapterIdentity
    confidence: IdentityConfidence


@dataclass(frozen=True, slots=True)
class _Verification:
    """What launch-time verification decided about one registration against one served base.

    ``confidence`` set means the adapter was registered; ``reason`` set means it was refused with
    that explanation. Exactly one is ever set — a candidate is either applied or refused, never
    dropped without a record (ADR-0058 rule 5).
    """

    confidence: IdentityConfidence | None = None
    reason: str | None = None


class _InFlightLease:
    """One request's claim on a server, released exactly once however the request ends.

    A restart that folds in a new adapter, and a restart forced by a profile change, both wait on
    the count these leases keep. Release is idempotent because a stream can end three ways —
    drained, abandoned, or collected — and two of them can happen to one iterator.
    """

    __slots__ = ("_counts", "_lock", "_model_name", "_released")

    def __init__(self, counts: dict[str, int], lock: threading.RLock, model_name: str) -> None:
        """Take the claim; the caller holds no lock."""
        self._counts = counts
        self._lock = lock
        self._model_name = model_name
        self._released = False
        with lock:
            counts[model_name] = counts.get(model_name, 0) + 1

    def release(self) -> None:
        """Give the claim back. Safe to call any number of times, from any thread."""
        with self._lock:
            if self._released:
                return
            self._released = True
            remaining = self._counts.get(self._model_name, 1) - 1
            if remaining > 0:
                self._counts[self._model_name] = remaining
            else:
                self._counts.pop(self._model_name, None)


@dataclass(frozen=True, slots=True)
class _HeaderSnapshot:
    """A parsed header with the instant it was read, for the metadata cache."""

    header: GgufHeader
    observed_at: datetime


class LlamaCppProvider:
    """A real :class:`~modelrack.provider.Provider` that runs ``llama-server`` itself.

    Discovers GGUF files under ``model_directory``, hashes them into digest-bound identities,
    and serves each on demand by spawning one ``llama-server`` per base on a loopback port from
    ``port_range``, supervised until ``unload()`` (or ``close()``, or this object's collection at
    the latest) terminates it. Chat-style requests reach the server's chat endpoint, where the
    model's template and tool-call parsing live; completion-style requests reach the native
    ``/completion`` endpoint with the prompt untouched.

    Every method raises the typed errors in :mod:`modelrack.errors`, never a raw ``httpx`` or
    ``subprocess`` exception (spec §11.7); every unavailable measurement is ``UNSUPPORTED``, and
    every token class the protocol cannot bill is ``0`` (ADR-0070).

    **Residency semantics.** A request whose runtime profile matches the running server's launch
    flags is served by it. A request whose profile differs — a different context size, offload,
    KV precision — **restarts** the server under the new flags, because every profile field is a
    launch-time property on llama-server and serving it from a server launched otherwise would
    record a runtime profile the run did not use (ADR-0023). Several different bases may be
    resident at once, each on its own port; whether they fit on the GPU is the caller's policy
    (ADR-0038), not this adapter's. ``keep_alive`` has no effect: a server stays until unloaded.

    Args:
        model_directory: Where the GGUF files are. Searched recursively; a file's served name is
            its path below this directory without ``.gguf``. Must exist at construction.
        state_dir: Where pid files and each server's captured stderr live. Created on the first
            spawn, not at construction. Named by the constructing application inside its own data
            root, because this package picks no directory of its own (spec §12) and pid files are
            what make an orphan from a crashed process recoverable by the next one (ADR-0062).
        adapters: LoRA adapters this provider may serve, supplied by the application that read
            the operator's directory — **this package never reads it** (ADR-0061 rule 3). Each is
            verified against the base actually launched, per base, at spawn; more may be handed
            over later through :meth:`register_adapters`, and
            :meth:`list_adapters` reports what became of each.
        server_path: The ``llama-server`` executable — a bare name resolved on ``PATH`` or a
            path. Not checked at construction: :meth:`health` reports a missing binary as
            ``UNAVAILABLE`` rather than the constructor refusing to build an adapter the health
            endpoint exists to ask about.
        port_range: Inclusive loopback ports servers may listen on. One port per resident base.
        startup_timeout_seconds: How long a spawned server may take to answer its first healthy
            probe before it is killed and :class:`~modelrack.errors.ProviderTimeout` raised.
        shutdown_timeout_seconds: The grace between ``SIGTERM`` and ``SIGKILL`` on unload.
        timeout: The default HTTP timeout for a call that names none (spec §12). Never ``None``.
        client: An already-constructed :class:`httpx.Client` to use instead of building one.
        max_response_bytes: The total-body cap for a non-streamed response (spec §14).
        max_chunk_bytes: The per-line cap for a streamed response (spec §14).
        metadata_ttl_seconds: How long a parsed header is reused before the file is re-read
            (spec §10, default 300 s). ``0`` disables the header cache. Every hit is also checked
            against the file's stamp, so the TTL bounds staleness of nothing but the header's
            contents for an unchanged file. Residency and health are never cached.
        digest_store: Where content digests are kept, keyed by path and stamp. Defaults to a
            :class:`JsonFileDigestStore` over ``<state_dir>/digests.json`` (ADR-0071), so a
            directory is hashed once per ``state_dir`` rather than once per process; pass an
            :class:`InMemoryDigestStore` for no persistence.
        on_event: An optional observer called as requests start, stream and finish, and as
            servers are loaded (spec §17). Receives no prompt, no generated text and no path.
        clock: Where a descriptor's ``observed_at`` and a handle's ``started_at`` come from.
        launcher: The spawn seam. Defaults to the real
            :class:`~modelrack.providers._llamacpp_process.SubprocessLauncher`; a test injects a
            fake so every supervision path runs without a binary.
        process_table: How processes without a handle are inspected and signalled, for orphan
            recovery. Defaults to the real POSIX table.
        port_is_free: Whether a port may be used. Defaults to a loopback bind probe.
        sleep: Called between health probes while a server starts. Injected so a test need not
            wait through a real startup.
        monotonic: The monotonic nanosecond clock timeouts and durations are measured with.

    Raises:
        ValidationError: If ``model_directory`` is not an existing directory, or the port range
            or a timeout is not valid (see
            :class:`~modelrack.providers._llamacpp_process.LlamaServerSupervisor`).
    """

    kind: ProviderKind = ProviderKind.LLAMACPP

    def __init__(
        self,
        model_directory: str | Path,
        *,
        state_dir: str | Path,
        adapters: Sequence[AdapterRegistration] = (),
        server_path: str | Path = DEFAULT_SERVER_EXECUTABLE,
        port_range: tuple[int, int] = DEFAULT_PORT_RANGE,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        metadata_ttl_seconds: float = DEFAULT_METADATA_TTL_SECONDS,
        digest_store: DigestStore | None = None,
        on_event: EventCallback | None = None,
        clock: Callable[[], datetime] = utc_now,
        launcher: ProcessLauncher | None = None,
        process_table: ProcessTable | None = None,
        port_is_free: Callable[[int], bool] = loopback_port_is_free,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], int] = monotonic_ns,
    ) -> None:
        """Validate the directory, build the supervisor and the pooled client."""
        directory = Path(model_directory)
        if not directory.is_dir():
            raise ValidationError(
                f"model_directory must be an existing directory; got {str(directory)!r}.",
                details={"field": "model_directory", "value": str(directory)},
            )
        self._model_directory = directory
        self._server_path = str(server_path)
        self._max_response_bytes = max_response_bytes
        self._max_chunk_bytes = max_chunk_bytes
        self._clock = clock
        self._monotonic = monotonic
        self._headers: MetadataCache[_HeaderSnapshot] = MetadataCache(
            ttl_seconds=metadata_ttl_seconds
        )
        self._digests: DigestStore = (
            digest_store
            if digest_store is not None
            else JsonFileDigestStore(Path(state_dir) / _DIGEST_FILE_NAME, clock=clock)
        )
        self._events = EventEmitter(on_event, provider_kind=self.kind)
        self._supervisor = LlamaServerSupervisor(
            state_dir=Path(state_dir),
            port_range=port_range,
            launcher=launcher,
            process_table=process_table,
            port_is_free=port_is_free,
            sleep=sleep,
            monotonic=monotonic,
            clock=clock,
            startup_timeout_seconds=startup_timeout_seconds,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS,
            stderr_tail_bytes=DEFAULT_STDERR_TAIL_BYTES,
        )
        self._identities: dict[str, ModelIdentity] = {}
        # Registration order is server-id order, so a dict (insertion-ordered) is the record.
        self._registrations: dict[str, AdapterRegistration] = {}
        # Per running server: what it registered, and what launch-time verification decided.
        self._server_adapters: dict[str, tuple[ServerAdapter, ...]] = {}
        self._verifications: dict[str, dict[str, _Verification]] = {}
        # Per running server: how many requests are using it right now. A restart waits on this.
        self._in_flight: dict[str, int] = {}
        self._lock = threading.RLock()
        for registration in adapters:
            self._registrations[registration.name] = registration
        self._idle_base_url = f"http://127.0.0.1:{port_range[0]}"
        self._client = client or build_client(
            base_url=self._idle_base_url,
            timeout=timeout,
            headers=dict(_REQUEST_HEADERS),
            verify=True,
        )
        # The safety net for a caller that never unloads: when this adapter is collected — or at
        # interpreter exit, whichever comes first — every server it still tracks is terminated.
        # Bound to the supervisor rather than to `self`, so the finalizer keeps nothing alive.
        self._finalizer = weakref.finalize(self, self._supervisor.terminate_all)

    # ------------------------------------------------------------------------------- protocol

    def health(self) -> ProviderHealth:
        """Report whether this adapter can serve, and what it is serving.

        Returns:
            ``UNAVAILABLE`` when the ``llama-server`` binary cannot be found or the model
            directory cannot be read — nothing could be spawned, and an operator should know
            which; ``DEGRADED`` when a supervised server is running but not answering its own
            health probe; ``OK`` otherwise, with the model count (headers only — no file is
            hashed by a health check), how many servers are resident, and the running build
            where a server is up. **Never a raised exception.** ``is_remote`` is always
            ``False``: a supervised server listens on loopback and nowhere else.

        Note:
            ``provider_version`` is ``None`` while no server is running. This adapter does not
            spawn a process to ask a binary its version; the build is read from ``/props`` once
            a server is up and carried on every result it produces.
        """
        start_ns = self._monotonic()
        resolved = self._resolved_server_path()
        if resolved is None:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                base_url=self._idle_base_url,
                detail=f"llama-server not found: {self._server_path!r}",
                latency_ms=elapsed_ms(start_ns, self._monotonic()),
            )
        try:
            entries = self._entries(refresh=False)
        except ProviderError as exc:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                base_url=self._idle_base_url,
                detail=f"model directory unreadable ({exc.code})",
                latency_ms=elapsed_ms(start_ns, self._monotonic()),
            )
        with self._lock:
            self._reap()
            handles = self._supervisor.handles()
        version = next((h.build_info for h in handles if h.build_info is not None), None)
        base_url = handles[0].base_url if handles else self._idle_base_url
        unanswered = [h for h in handles if not self._probe(h.port)]
        if unanswered:
            return ProviderHealth(
                status=ProviderStatus.DEGRADED,
                base_url=unanswered[0].base_url,
                detail=(
                    f"llama.cpp {version or '(unknown build)'}: {len(unanswered)} of "
                    f"{len(handles)} supervised servers not answering /health"
                ),
                provider_version=version,
                model_count=len(entries),
                latency_ms=elapsed_ms(start_ns, self._monotonic()),
            )
        return ProviderHealth(
            status=ProviderStatus.OK,
            base_url=base_url,
            detail=(
                f"llama.cpp {version or '(no server running)'}, {len(entries)} models, "
                f"{len(handles)} resident"
            ),
            provider_version=version,
            model_count=len(entries),
            latency_ms=elapsed_ms(start_ns, self._monotonic()),
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report what this adapter can do — the static declaration, no request made."""
        return _CAPABILITIES

    # ------------------------------------------------------------------------------- adapters

    def list_adapters(self) -> Sequence[AdapterState]:
        """Report every registration this provider holds, and whether each can be selected now.

        Returns:
            One state per registration, in the order the registrations arrived — which is also the
            order they are passed to ``--lora``, and so the order the server assigns ids in. A
            registration is :attr:`~modelrack.adapters.AdapterStatus.REGISTERED` when a running
            server has it, :attr:`~modelrack.adapters.AdapterStatus.PENDING_RESTART` when its
            base is running but was launched before it arrived,
            :attr:`~modelrack.adapters.AdapterStatus.INCOMPATIBLE` when launch-time verification
            refused it against the base actually served, and
            :attr:`~modelrack.adapters.AdapterStatus.AWAITING_BASE` when no server for its base is
            running at all.

            A snapshot, not a live view. Nothing here probes: the states are read from what the
            last spawn recorded, so calling this is free and never starts a server.
        """
        with self._lock:
            return tuple(
                self._state_for(registration) for registration in self._registrations.values()
            )

    def register_adapters(self, adapters: Sequence[AdapterRegistration]) -> None:
        """Add adapters this provider may serve, without disturbing anything already running.

        Registration is a **launch-time** property of a llama-server process, so nothing here
        touches a running server: an adapter whose base is already up becomes
        :attr:`~modelrack.adapters.AdapterStatus.PENDING_RESTART` and folds in at the next natural
        idle — the next request that arrives with nothing in flight against that server, or an
        :meth:`unload` (ADR-0062 decision 3). Never mid-work, and never by restarting a server
        under a stream.

        Args:
            adapters: The registrations to add. A name already held is **replaced**: a rescan that
                found new bytes under a familiar name is a new subject, and keeping the old one
                would leave the provider able to serve an identity the operator has retired.
                Replacing one whose base is running makes the replacement pending, so the running
                server keeps serving the identity it actually launched with until it restarts.
        """
        with self._lock:
            for registration in adapters:
                self._registrations[registration.name] = registration

    def _state_for(self, registration: AdapterRegistration) -> AdapterState:
        """Decide one registration's state from what the running servers recorded. Lock held."""
        for model_name, verifications in self._verifications.items():
            verification = verifications.get(registration.name)
            if verification is None:
                continue
            if verification.reason is not None:
                return AdapterState(
                    adapter=registration,
                    status=AdapterStatus.INCOMPATIBLE,
                    base_model_name=model_name,
                    reason=verification.reason,
                )
            server_id = next(
                (
                    adapter.server_id
                    for adapter in self._server_adapters.get(model_name, ())
                    if adapter.name == registration.name
                ),
                None,
            )
            if server_id is None:
                # Verified compatible at launch, but the server did not report it back: it was
                # launched with an argv this registration was not on, or the server rejected the
                # file. Either way it is not selectable, and saying so beats implying it is.
                return AdapterState(
                    adapter=registration,
                    status=AdapterStatus.PENDING_RESTART,
                    base_model_name=model_name,
                    base_confidence=verification.confidence,
                    reason=(
                        f"The server for {model_name!r} did not report this adapter; it folds in "
                        "at the next restart."
                    ),
                )
            return AdapterState(
                adapter=registration,
                status=AdapterStatus.REGISTERED,
                base_model_name=model_name,
                base_confidence=verification.confidence,
                server_id=server_id,
            )
        # No running server has verified it. Is one running that it *would* apply to?
        for model_name, base in self._identities.items():
            verification = self._verify(registration, base)
            if verification is None:
                continue
            if verification.reason is not None:
                return AdapterState(
                    adapter=registration,
                    status=AdapterStatus.INCOMPATIBLE,
                    base_model_name=model_name,
                    reason=verification.reason,
                )
            return AdapterState(
                adapter=registration,
                status=AdapterStatus.PENDING_RESTART,
                base_model_name=model_name,
                base_confidence=verification.confidence,
                reason=(
                    f"The server for {model_name!r} was launched before this adapter was "
                    "registered; it folds in at the next idle."
                ),
            )
        return AdapterState(
            adapter=registration,
            status=AdapterStatus.AWAITING_BASE,
            base_model_name=registration.base_model_name,
        )

    @staticmethod
    def _verify(registration: AdapterRegistration, base: ModelIdentity) -> _Verification | None:
        """Decide whether ``registration`` applies to ``base``, and how well it was proved.

        Candidacy comes first and is deliberately **either** handle: a registration that declares a
        base digest is a candidate for any base serving those bytes, whatever the file is called —
        which is what makes a rename safe (ADR-0061 rule 5) — and one that declares only a name is
        a candidate for the base of that name. Everything else is another base's adapter, and
        silence about it is correct.

        Returns:
            ``None`` when this is not this base's adapter at all; a :class:`_Verification` with a
            confidence when it applies; one with a reason when it is **refused**, which is the
            dangerous case — the manifest names this base and its digest says otherwise, meaning
            either the manifest is stale or the file changed under it.
        """
        declared_digest = registration.base_artifact_digest
        names_match = registration.base_model_name == base.provider_model_name
        if declared_digest is not None:
            if declared_digest != base.artifact_digest and not names_match:
                return None
        elif not names_match:
            return None
        try:
            confidence = verify_adapter_base_compatibility(
                base,
                declared_base_name=registration.base_model_name,
                declared_base_digest=declared_digest,
            )
        except ValidationError as exc:
            return _Verification(reason=str(exc))
        return _Verification(confidence=confidence)

    def _verify_all(self, base: ModelIdentity) -> dict[str, _Verification]:
        """Verify every held registration against one served base. Lock held."""
        verified: dict[str, _Verification] = {}
        for name, registration in self._registrations.items():
            verification = self._verify(registration, base)
            if verification is not None:
                verified[name] = verification
        return verified

    # ------------------------------------------------------------------------- metadata cache

    @property
    def metadata_cache_ttl_seconds(self) -> float:
        """How long a parsed GGUF header is reused, in seconds; ``0.0`` when disabled."""
        return self._headers.ttl_seconds

    def metadata_cache_stats(self) -> CacheStats:
        """Report what the header cache has done since construction or the last clear."""
        return self._headers.stats()

    def clear_metadata_cache(self) -> None:
        """Drop every cached header. Digests are keyed by content stamp and are not affected;
        see :meth:`clear_digest_cache`, or pass ``refresh=True`` to any discovery method to
        re-hash a file whose stamp is unchanged.
        """
        self._headers.clear()

    def clear_digest_cache(self) -> None:
        """Forget every computed digest, so the next discovery hashes every file again.

        For the default store this removes ``<state_dir>/digests.json`` (ADR-0071); deleting
        that file by hand is equally safe.
        """
        self._digests.clear()

    @property
    def model_directory(self) -> Path:
        """Where this adapter looks for GGUF files."""
        return self._model_directory

    @property
    def supervisor(self) -> LlamaServerSupervisor:
        """The process supervisor, for a test that asserts on handles and pid files."""
        return self._supervisor

    # -------------------------------------------------------------------------- discovery

    def list_models(self, *, refresh: bool = False) -> Sequence[ModelDescriptor]:
        """List the base models under the directory, each with a digest-bound identity.

        Reads every file's header (cached, stamp-checked) and **hashes every file whose digest is
        not already in the store** — the first call in a process against an unhashed directory
        costs minutes on a 40 GB directory and milliseconds thereafter. Shards of a split GGUF,
        adapters, projectors and files that are not GGUF at all are skipped with a DEBUG log,
        never listed as models.

        Args:
            refresh: Re-read every header and re-hash every file, ignoring both caches.

        Returns:
            One descriptor per base model, sorted by name. Empty when the directory holds none.

        Raises:
            ProviderUnavailable: If the directory cannot be read.
        """
        return tuple(
            self._describe(entry, refresh=refresh)
            for entry in self._entries(refresh=refresh).values()
        )

    def inspect_model(self, identity: ModelIdentity, *, refresh: bool = False) -> ModelDescriptor:
        """Fetch the descriptor for one model, matched on ``provider_model_name``.

        Args:
            identity: The model to inspect.
            refresh: Re-read the header and re-hash the file.

        Returns:
            The descriptor, with the whole GGUF metadata block in ``raw``.

        Raises:
            ModelNotFound: If no base model has that name.
            ProviderUnavailable: If the directory cannot be read.
        """
        return self._describe(
            self._entry_named(identity.provider_model_name, refresh=refresh), refresh=refresh
        )

    def resolve(self, reference: str, *, refresh: bool = False) -> ModelIdentity:
        """Resolve a user-supplied reference to a digest-bound identity.

        Tries, in order: an exact served name; the same with a trailing ``.gguf`` removed (a user
        who typed the filename); a unique prefix over every served name. An ambiguous prefix is
        refused (spec §11.8). Only the resolved file is hashed.

        Args:
            reference: What the user typed.
            refresh: Re-read headers and re-hash the resolved file.

        Returns:
            The identity, carrying the file's content digest.

        Raises:
            ModelNotFound: If nothing matches, or a prefix matches more than one model.
            ProviderUnavailable: If the directory cannot be read.
        """
        entries = self._entries(refresh=refresh)
        name = self._resolve_name(reference, entries)
        entry = entries[name]
        if reference != name:
            logger.debug(
                "llamacpp.model.resolved", extra={"reference": reference, "resolved_to": name}
            )
        return identity_for(name, self._digest_for(entry, refresh=refresh))

    # -------------------------------------------------------------------------- residency

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        """Spawn a server for ``identity`` under ``profile``, or report it already resident.

        Args:
            identity: The model to serve. A pinned digest must match the file's.
            profile: How to launch it. Every field becomes a launch flag
                (:func:`~modelrack.providers._llamacpp_wire.launch_flags`); a server already
                running under different flags is restarted under these.

        Returns:
            ``already_resident=True`` with ``load_ms`` ``UNSUPPORTED`` when a server under these
            flags was already up; otherwise ``load_ms`` is this process's own measurement of the
            time from spawn to the first healthy probe — llama-server reports no load time, so
            the observed figure is the honest one and it is never ``0``.

        Raises:
            ModelNotFound: If no base model has that name, or its digest does not match a
                pinned one.
            ProviderUnavailable: If the binary cannot be launched, no port is free, or the
                server exited before it was healthy — ``details`` carries the reason, the argv,
                and for an exit the code and the captured stderr tail.
            ProviderTimeout: If the server did not become healthy in time; it has been killed.
        """
        require_force_unload(_CAPABILITIES, action="load a model on demand")
        entry = self._locate(identity)
        with self._lock:
            self._reap()
            handle = self._supervisor.handle_for(entry.name)
            key = self._launch_key(entry, profile)
            if handle is not None and handle.launch_key == key:
                return LoadResult(
                    identity=identity,
                    already_resident=True,
                    load_ms=UNSUPPORTED,
                    profile_hash=profile.profile_hash,
                )
            if handle is not None:
                self._terminate(handle)
            handle = self._spawn(entry, identity, profile)
            return LoadResult(
                identity=identity,
                already_resident=False,
                load_ms=handle.startup_ms,
                profile_hash=profile.profile_hash,
            )

    def unload(self, identity: ModelIdentity) -> bool:
        """Terminate the server for ``identity``: ``SIGTERM`` its group, grace, ``SIGKILL``.

        Matched on ``provider_model_name`` alone, so a model whose file has since been removed
        can still be unloaded — the process is what is being freed, not the file.

        Args:
            identity: The model to evict.

        Returns:
            ``True`` if a server was running and has been stopped, ``False`` if none was — the
            state the caller wanted, not a failure.
        """
        require_force_unload(_CAPABILITIES, action="evict a model on demand")
        with self._lock:
            self._reap()
            handle = self._supervisor.handle_for(identity.provider_model_name)
            if handle is None:
                return False
            start_ns = self._monotonic()
            self._events.started(operation="unload", model_name=handle.model_name, metadata={})
            self._terminate(handle)
            self._events.completed(
                operation="unload",
                model_name=handle.model_name,
                metadata={},
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
            return True

    def list_resident(self) -> Sequence[ResidentModel]:
        """List the bases with a live supervised server, from the process table.

        Sweeps orphaned pid files and drops servers that exited on their own first, so the
        answer is the live set. ``context_length`` is what each server reported serving via
        ``/props`` — a *reported* served context (ADR-0023 §4) — and ``vram_bytes`` and
        ``total_bytes`` are ``UNSUPPORTED``: llama-server exposes no memory figure this adapter
        reads, and a number from elsewhere would not be the provider's.

        Returns:
            One entry per resident base, sorted by name.
        """
        require_residency_query(_CAPABILITIES)
        with self._lock:
            self._supervisor.sweep_orphans()
            self._reap()
            handles = self._supervisor.handles()
            return tuple(
                ResidentModel(
                    identity=self._identities.get(handle.model_name)
                    or identity_for(handle.model_name, None),
                    context_length=handle.served_context,
                )
                for handle in handles
            )

    def close(self) -> None:
        """Terminate every supervised server and close the HTTP client.

        Idempotent. An application calls this at shutdown; a test calls it in teardown. A
        provider that is simply dropped is cleaned up by its finalizer, but that runs at
        collection or interpreter exit, which is later than an application wants a GPU freed.
        """
        self._supervisor.terminate_all()
        self._identities.clear()
        self._client.close()

    # ------------------------------------------------------------------------ generation

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one non-streamed generation, spawning the model's server first if needed.

        A cancellation token on the request has no effect: a blocking round trip offers no
        boundary at which it could take effect (spec §13).

        Args:
            request: What to generate, and how. ``messages`` reach the chat endpoint;
                ``prompt`` reaches ``/completion`` raw.

        Returns:
            The complete outcome. ``client_ttft_ms`` is ``UNSUPPORTED``; ``provider_version`` is
            the running build.

        Raises:
            ModelNotFound: If the model is not under the directory, or a pinned digest differs.
            CapabilityUnsupported: If ``prompt`` is combined with ``tools`` — the native
                endpoint has no concept of tools, and dropping them silently is the dishonesty
                ADR-0007 rule 2 forbids.
            ContextLimitExceeded: If the server reports the request exceeds its context, with
                the requested and served sizes where the server stated them.
            ProviderRejected: If the server understood the request and refused it.
            ProviderUnavailable: If the server could not be spawned, exited, or is still loading.
            ProviderProtocolError: If the response cannot be parsed.
            ProviderTimeout: If the server does not answer in time.
        """
        is_chat, path = self._route(request)
        handle = self._ensure_server(request)
        selection = self._select_adapter(request, handle)
        body = self._build_body(request, handle, is_chat=is_chat, stream=False, selection=selection)
        lease = self._lease(handle)
        start_ns = self._monotonic()
        self._events.started(
            operation="generate",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        try:
            result = self._generate_once(
                request, handle, path, body, start_ns, is_chat=is_chat, selection=selection
            )
        except ProviderError as exc:
            self._events.failed(
                operation="generate",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=exc.code,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
            raise
        finally:
            lease.release()
        self._events.completed(
            operation="generate",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
            elapsed_ms=result.timing.client_wall_ms,
            output_tokens=result.usage.tokens.output_tokens,
            finish_reason=result.finish_reason.value,
        )
        return result

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Run one generation, yielding events as the server's SSE stream delivers them.

        Everything that can fail before a byte arrives — the request's shape, the server's
        spawn, the connection, a rejected request — raises from this call. Every failure after
        that is delivered as :class:`~modelrack.streaming.StreamFailed`, the terminal event,
        with the connection closed either way. A token already cancelled yields one terminal
        event and spawns nothing and opens nothing.

        Args:
            request: What to generate, and how. Its ``cancel`` token is honoured within one
                streamed event.

        Yields:
            Deltas, then one terminal event.

        Raises:
            CapabilityUnsupported: If ``prompt`` is combined with ``tools``.
            ModelNotFound: If the model is not under the directory, or a pinned digest differs.
            ContextLimitExceeded: If the server refuses the request as too large, before any
                content arrives.
            ProviderRejected: If the server refuses the request before streaming.
            ProviderUnavailable: If the server could not be spawned or is unreachable.
            ProviderTimeout: If it does not answer in time.
        """
        is_chat, path = self._route(request)
        if request.cancel is not None and request.cancel.is_cancelled:
            return iter((self._already_cancelled(request),))
        handle = self._ensure_server(request)
        selection = self._select_adapter(request, handle)
        body = self._build_body(request, handle, is_chat=is_chat, stream=True, selection=selection)
        lease = self._lease(handle)
        start_ns = self._monotonic()
        self._events.started(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        prepared = self._client.build_request(
            "POST", handle.base_url + path, json=body, timeout=self._timeout_for(request)
        )
        try:
            response = self._client.send(prepared, stream=True)
        except httpx.HTTPError as exc:
            lease.release()
            error = translate_transport_error(exc, base_url=handle.base_url)
            self._events.failed(
                operation="stream",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=error.code,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
            raise error from exc
        try:
            if response.status_code >= 400:
                self._raise_for_status(
                    response, handle, context_size=request.runtime_profile.context_size
                )
        except ProviderError as exc:
            response.close()
            lease.release()
            self._events.failed(
                operation="stream",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=exc.code,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
            raise
        except BaseException:
            response.close()
            lease.release()
            raise
        events = self._walk(
            request, response, start_ns, handle, is_chat=is_chat, selection=selection, lease=lease
        )
        # A caller may take the iterator and never start it — the response is already open, so the
        # claim is already taken and `_walk`'s `finally` will never run to give it back. Releasing
        # on collection is what keeps an abandoned stream from making a server un-restartable for
        # the life of the process.
        weakref.finalize(events, lease.release)
        return events

    def _lease(self, handle: ServerHandle) -> _InFlightLease:
        """Claim the server for one request, so a restart waits rather than cuts in."""
        return _InFlightLease(self._in_flight, self._lock, handle.model_name)

    # ----------------------------------------------------------------------- server control

    def _ensure_server(self, request: GenerationRequest) -> ServerHandle:
        """Return a live server for the request's model and profile, spawning or restarting one.

        A server that exited on its own since the last call is reported **once**, as a typed
        :class:`~modelrack.errors.ProviderUnavailable` carrying its exit code and stderr tail;
        the call after that spawns afresh. Respawning silently would hide a crash-looping
        server behind ordinary latency, which is this row's named failure mode.
        """
        entry = self._locate(request.identity)
        with self._lock:
            for exited, code in self._reap():
                if exited.model_name == entry.name:
                    raise ProviderUnavailable(
                        f"llama-server for {entry.name!r} exited with code {code} since the last "
                        "call; the next call will start a new one. Its captured output is "
                        "attached.",
                        details={
                            "reason": ProviderUnavailableReason.PROCESS_EXITED.value,
                            "exit_code": code,
                            "port": exited.port,
                            "stderr_path": str(exited.stderr_path),
                            "stderr_tail": self._supervisor.stderr_tail(exited),
                        },
                    )
            handle = self._supervisor.handle_for(entry.name)
            key = self._launch_key(entry, request.runtime_profile)
            if handle is not None and handle.launch_key != key:
                # A **required** restart: this server was launched under different flags and
                # cannot serve this profile at all (ADR-0023). It still waits for idle — killing a
                # server another thread is streaming from would fail that request with a dropped
                # connection, which looks like a server fault and is not one.
                self._require_idle(entry.name, restart_reason="profile_change")
                logger.debug(
                    "llamacpp.server.restart_for_profile",
                    extra={"model_name": entry.name, "port": handle.port},
                )
                self._terminate(handle)
                handle = None
            if handle is not None and self._folds_in_at_idle(entry.name):
                # A **deferred** restart: adapters arrived after this server started. This is the
                # next natural idle, so take it now — one restart per newly registered adapter is
                # the honest floor (ADR-0062 decision 3), and it is paid at a boundary between
                # requests rather than inside one.
                logger.debug(
                    "llamacpp.server.restart_for_adapters",
                    extra={"model_name": entry.name, "port": handle.port},
                )
                self._terminate(handle)
                handle = None
            if handle is None:
                handle = self._spawn(entry, request.identity, request.runtime_profile)
            return handle

    def _require_idle(self, model_name: str, *, restart_reason: str) -> None:
        """Refuse a restart that would cut into work in flight. Lock held.

        Raises:
            ProviderUnavailable: With reason ``restart_pending`` when another request is using
                this server. Availability, not reliability (ADR-0067 rule 2): the subject is
                unusable for a moment, and a router should stand it down rather than count a
                failure against it.
        """
        in_flight = self._in_flight.get(model_name, 0)
        if in_flight == 0:
            return
        raise ProviderUnavailable(
            f"The server for {model_name!r} must restart before it can serve this request, and "
            f"{in_flight} request(s) are still in flight against it. A restart never interrupts "
            "work; retry when they finish.",
            details={
                "reason": ProviderUnavailableReason.RESTART_PENDING.value,
                "restart_reason": restart_reason,
                "model_name": model_name,
                "in_flight": in_flight,
            },
        )

    def _folds_in_at_idle(self, model_name: str) -> bool:
        """Answer whether a running server should restart now to pick up new adapters. Lock held.

        ``True`` only when the set this provider would launch with differs from what the server
        actually reported registering **and** nothing is in flight. A busy server is left alone —
        the pending adapters stay pending, which is what the state is for.
        """
        base = self._identities.get(model_name)
        if base is None or self._in_flight.get(model_name, 0) > 0:
            return False
        wanted = tuple(
            name
            for name, verification in self._verify_all(base).items()
            if verification.reason is None
        )
        registered = tuple(adapter.name for adapter in self._server_adapters.get(model_name, ()))
        return wanted != registered

    def _spawn(
        self, entry: _Entry, identity: ModelIdentity, profile: RuntimeProfile
    ) -> ServerHandle:
        """Spawn a server for ``entry`` under ``profile``, reporting it as a ``load`` operation."""
        start_ns = self._monotonic()
        served = identity_for(entry.name, self._digest_for(entry, refresh=False))
        # Verified against the base actually being launched, before its argv exists — a manifest
        # that names this base and disagrees about its digest never reaches the command line.
        verifications = self._verify_all(served)
        registered = tuple(
            self._registrations[name]
            for name, verification in verifications.items()
            if verification.reason is None
        )
        for name, verification in verifications.items():
            if verification.reason is not None:
                logger.warning(
                    "llamacpp.adapter.refused",
                    extra={
                        "model_name": entry.name,
                        "adapter": name,
                        "reason": verification.reason,
                    },
                )
        self._events.started(operation="load", model_name=entry.name, metadata={})
        try:
            handle = self._supervisor.spawn(
                model_name=entry.name,
                build_argv=lambda port: build_launch_argv(
                    server_path=self._server_path,
                    model_path=entry.path,
                    alias=entry.name,
                    port=port,
                    profile=profile,
                    adapter_paths=[item.artifact_path for item in registered],
                ),
                probe=self._probe,
                launch_key=self._launch_key(entry, profile),
            )
        except ProviderError as exc:
            self._events.failed(
                operation="load",
                model_name=entry.name,
                metadata={},
                error_code=exc.code,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
            raise
        self._identities[entry.name] = identity if identity.artifact_digest is not None else served
        self._verifications[entry.name] = verifications
        self._read_props(handle)
        self._server_adapters[entry.name] = self._read_registered_adapters(handle, registered)
        self._events.completed(
            operation="load",
            model_name=entry.name,
            metadata={},
            elapsed_ms=handle.startup_ms,
        )
        return handle

    def _terminate(self, handle: ServerHandle) -> None:
        """Stop a server and forget everything that was true only while it ran."""
        self._supervisor.terminate(handle)
        self._forget(handle.model_name)

    def _reap(self) -> tuple[tuple[ServerHandle, int], ...]:
        """Drop servers that exited on their own, forgetting their identities."""
        exited = self._supervisor.reap_exited()
        for handle, _code in exited:
            self._forget(handle.model_name)
        return exited

    def _forget(self, model_name: str) -> None:
        """Drop the per-server state a stopped server's answers were derived from.

        The adapter states go with it: a registration that was ``registered`` on a server that is
        no longer running is ``awaiting_base``, and reporting it as registered would name a
        server id nothing will honour.
        """
        self._identities.pop(model_name, None)
        self._server_adapters.pop(model_name, None)
        self._verifications.pop(model_name, None)

    def _launch_key(self, entry: _Entry, profile: RuntimeProfile) -> str:
        """What decides whether a running server can serve a profile: the file and the flags."""
        return json.dumps([str(entry.path), list(launch_flags(profile))])

    def _probe(self, port: int) -> bool:
        """Answer whether the server on ``port`` says it is healthy. Never raises."""
        try:
            response = self._client.get(f"http://127.0.0.1:{port}{_HEALTH_PATH}")
        except httpx.HTTPError:
            return False
        response.close()
        return response.status_code == 200

    def _read_props(self, handle: ServerHandle) -> None:
        """Record the build and the served context from ``/props``; a failure is only logged."""
        try:
            props = self._get_json(handle, _PROPS_PATH)
        except ProviderError as exc:
            logger.debug(
                "llamacpp.props.unreadable",
                extra={"model_name": handle.model_name, "port": handle.port, "code": exc.code},
            )
            return
        if isinstance(props, dict):
            handle.build_info = read_build_info(props)
            handle.served_context = read_served_context(props)

    def _read_registered_adapters(
        self, handle: ServerHandle, launched: Sequence[AdapterRegistration]
    ) -> tuple[ServerAdapter, ...]:
        """Ask the server which adapters it registered, and under which ids.

        Read back rather than inferred from argv order: the id in a request's ``lora`` field is
        the server's own index, and assuming it would put a wrong-adapter bug — the kind that
        produces plausible, confident, wrong output — one refactor away. An adapter the server
        does not report is left out, so it shows as pending rather than as selectable.

        Returns:
            The registered adapters in server-id order, or ``()`` when none were launched or the
            endpoint could not be read (logged, never raised: a server that answers ``/health``
            and ``/props`` but not this is still usable for the bare base).
        """
        if not launched:
            return ()
        try:
            payload = self._get_json(handle, _LORA_PATH)
        except ProviderError as exc:
            logger.warning(
                "llamacpp.lora_adapters.unreadable",
                extra={"model_name": handle.model_name, "port": handle.port, "code": exc.code},
            )
            return ()
        return read_lora_adapters(
            payload, by_path={str(item.artifact_path): item.name for item in launched}
        )

    def _resolved_server_path(self) -> str | None:
        """Return the executable this adapter would launch, or ``None`` if it cannot be found."""
        candidate = Path(self._server_path)
        if candidate.parent != Path():
            return str(candidate) if candidate.is_file() else None
        return shutil.which(self._server_path)

    # -------------------------------------------------------------------------- discovery

    def _entries(self, *, refresh: bool) -> dict[str, _Entry]:
        """Return every base model under the directory, by served name, headers parsed.

        Never hashes. Files that are not base models — shards, adapters, projectors, files that
        do not parse as GGUF — are skipped with a DEBUG log naming the reason.
        """
        try:
            paths = sorted(p for p in self._model_directory.rglob(_GGUF_GLOB) if p.is_file())
        except OSError as exc:
            raise ProviderUnavailable(
                f"Cannot read the model directory {str(self._model_directory)!r}: {exc}",
                details={
                    "reason": ProviderUnavailableReason.LAUNCH_FAILED.value,
                    "model_directory": str(self._model_directory),
                    "error": str(exc),
                },
            ) from exc
        entries: dict[str, _Entry] = {}
        for path in paths:
            if is_shard(path):
                logger.debug(
                    "llamacpp.discovery.skipped", extra={"path": str(path), "reason": "shard"}
                )
                continue
            try:
                snapshot = self._header_snapshot(path, refresh=refresh)
            except (GgufFormatError, OSError) as exc:
                logger.debug(
                    "llamacpp.discovery.skipped",
                    extra={"path": str(path), "reason": str(exc)},
                )
                continue
            kind = header_kind(snapshot.header)
            if kind != _MODEL_KIND:
                logger.debug(
                    "llamacpp.discovery.skipped", extra={"path": str(path), "reason": kind}
                )
                continue
            name = model_name_for(path, root=self._model_directory)
            entries[name] = _Entry(
                name=name, path=path, header=snapshot.header, observed_at=snapshot.observed_at
            )
        return dict(sorted(entries.items()))

    def _header_snapshot(self, path: Path, *, refresh: bool) -> _HeaderSnapshot:
        """Return a file's parsed header, from the cache when its stamp still matches."""
        key = f"gguf:{path}"
        if not refresh:
            cached = self._headers.get(key)
            if cached is not None and cached.header.stamp == ArtifactStamp.of(path):
                return cached
        snapshot = _HeaderSnapshot(header=read_gguf_header(path), observed_at=self._clock())
        self._headers.put(key, snapshot)
        return snapshot

    def _digest_for(self, entry: _Entry, *, refresh: bool) -> str:
        """Return the file's content digest, hashing it unless the store already has it."""
        key = f"{entry.path}|{entry.header.stamp.key}"
        if not refresh:
            cached = self._digests.get(key)
            if cached is not None:
                return cached
        logger.debug(
            "llamacpp.artifact.hashing",
            extra={"path": str(entry.path), "size_bytes": entry.header.stamp.size_bytes},
        )
        digest = sha256_of_file(entry.path)
        self._digests.put(key, digest, path=entry.path)
        return digest

    def _describe(self, entry: _Entry, *, refresh: bool) -> ModelDescriptor:
        """Build the descriptor for one entry, hashing its file if needed."""
        return build_descriptor(
            entry.header,
            name=entry.name,
            digest=self._digest_for(entry, refresh=refresh),
            observed_at=entry.observed_at,
        )

    def _entry_named(self, name: str, *, refresh: bool) -> _Entry:
        """Return the entry served under exactly ``name``, or raise ``ModelNotFound``."""
        entries = self._entries(refresh=refresh)
        entry = entries.get(name)
        if entry is None:
            raise ModelNotFound(
                f"No model named {name!r} is served from {str(self._model_directory)!r}.",
                details={"reference": name, "known_model_count": len(entries)},
            )
        return entry

    def _locate(self, identity: ModelIdentity) -> _Entry:
        """Return the entry an identity names, refusing a pinned digest the file does not match."""
        entry = self._entry_named(identity.provider_model_name, refresh=False)
        if identity.artifact_digest is not None:
            actual = self._digest_for(entry, refresh=False)
            if actual != identity.artifact_digest:
                raise ModelNotFound(
                    f"The file served as {entry.name!r} no longer has the digest the identity "
                    "pins; refusing to run different weights than were asked for.",
                    details={
                        "reference": identity.provider_model_name,
                        "known_model_count": len(self._entries(refresh=False)),
                        "reason": "digest_mismatch",
                        "expected_digest": identity.artifact_digest,
                        "actual_digest": actual,
                    },
                )
        return entry

    @staticmethod
    def _resolve_name(reference: str, entries: Mapping[str, _Entry]) -> str:
        """Resolve a reference to one served name: exact, filename, then a unique prefix."""
        if reference in entries:
            return reference
        stripped = reference.removesuffix(".gguf")
        if stripped in entries:
            return stripped
        prefixed = [name for name in entries if name.startswith(reference)]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ModelNotFound(
                f"{reference!r} is a prefix of {len(prefixed)} models; it names none of them. "
                "Give enough of the name to pick one — resolving an ambiguous reference by "
                "choosing would run weights you did not ask for.",
                details={
                    "reference": reference,
                    "known_model_count": len(entries),
                    "matched_model_count": len(prefixed),
                },
            )
        raise ModelNotFound(
            f"No model matching {reference!r} is served from the model directory.",
            details={"reference": reference, "known_model_count": len(entries)},
        )

    # ------------------------------------------------------------------------- generation

    def _route(self, request: GenerationRequest) -> tuple[bool, str]:
        """Validate the request's shape and choose its endpoint: ``(is_chat, path)``.

        Everything here is decidable from the request alone, so it runs **before** a server is
        spawned and before a cancellation token is consulted: a request naming something this
        adapter refuses is malformed whichever way the token points, and reporting it as a
        cancellation would hide the caller's own bug.

        Raises:
            CapabilityUnsupported: If ``prompt`` is combined with ``tools`` — the native endpoint
                has no concept of tools, and dropping them silently is what ADR-0007 rule 2
                forbids.
            ProviderRejected: If ``provider_options`` carries the adapter selection itself
                (``lora``), a slot pin, or a ``--lora`` launch flag. Each would change what the
                weights do, or which cache answers, without changing the subject this adapter
                records — the fabricated comparability ADR-0058 exists to prevent. The supported
                channel is :attr:`~modelrack.types.GenerationRequest.adapter`.
        """
        self._refuse_smuggled_options(request)
        if request.prompt is not None:
            if request.tools:
                raise CapabilityUnsupported(
                    "llama-server's completion endpoint (/completion) has no concept of tools; "
                    "use messages instead of prompt to call one.",
                    details={"capability": "tool_calling"},
                )
            return False, _COMPLETION_PATH
        return True, _CHAT_PATH

    @staticmethod
    def _refuse_smuggled_options(request: GenerationRequest) -> None:
        """Refuse a ``provider_options`` entry that would move the subject behind this adapter."""
        options = request.runtime_profile.provider_options
        for key in sorted(options):
            if key in FORBIDDEN_REQUEST_KEYS:
                raise ProviderRejected(
                    f"provider_options[{key!r}] selects a LoRA adapter directly. That would run "
                    "different weights than the recorded subject names, so the result could not "
                    "be attributed. Use GenerationRequest.adapter, which this adapter resolves, "
                    "verifies against the served base and reports back on the result.",
                    details={"status_code": 0, "provider_message": f"forbidden option {key!r}"},
                )
            if key in SLOT_PINNING_KEYS:
                raise ProviderRejected(
                    f"provider_options[{key!r}] pins this request to one of the server's slots, "
                    "and so to that slot's prompt cache. Slot choice is the server's, because "
                    "that is what lets it clear a cache built under a different adapter; pinning "
                    "reaches past that rule (ADR-0062 decision 4).",
                    details={"status_code": 0, "provider_message": f"forbidden option {key!r}"},
                )
            if key in FORBIDDEN_LAUNCH_FLAGS:
                raise ProviderRejected(
                    f"provider_options[{key!r}] would register a LoRA adapter behind this "
                    "adapter's back: it would have no digest, no verified base and no name a "
                    "result could report. Register adapters through register_adapters(), which "
                    "verifies the base first.",
                    details={"status_code": 0, "provider_message": f"forbidden option {key!r}"},
                )

    def _select_adapter(
        self, request: GenerationRequest, handle: ServerHandle
    ) -> _Selection | None:
        """Resolve ``request.adapter`` against what the serving process actually registered.

        Returns:
            ``None`` when the request names no adapter — the bare-base subject, unchanged from
            before the adapter axis existed — else the selection the result will report.

        Raises:
            AdapterNotFound: If the name was never registered with this provider (``reason``
                ``unknown``), or was **refused** for this base at launch (``reason``
                ``incompatible_base``, with both digests). Never a bare-base generation: that
                would answer with a different subject than the caller asked for and say nothing
                (ADR-0062 decision 4).
            ProviderUnavailable: With reason ``restart_pending`` if the adapter is compatible but
                its base's server was launched before it was registered and work is in flight, so
                the restart that would fold it in cannot be taken yet.
        """
        name = request.adapter
        if name is None:
            return None
        with self._lock:
            known = list(self._registrations)
            registration = self._registrations.get(name)
            if registration is None:
                raise AdapterNotFound(
                    f"No adapter named {name!r} is registered with this provider.",
                    details={"adapter": name, "registered": known, "reason": "unknown"},
                )
            base = self._identities[handle.model_name]
            registered = self._server_adapters.get(handle.model_name, ())
            # What the launch decided, where this adapter existed then; otherwise what it would
            # decide now — an adapter registered after the server started has no launch verdict,
            # and refusing it as "another base's" would be wrong as well as unhelpful.
            verification = self._verifications.get(handle.model_name, {}).get(name)
            if verification is None:
                verification = self._verify(registration, base)
            if verification is None or verification.reason is not None:
                raise AdapterNotFound(
                    f"Adapter {name!r} cannot be applied to the base this server is running "
                    f"({handle.model_name!r}): "
                    + (
                        verification.reason
                        if verification is not None and verification.reason is not None
                        else (
                            f"it declares base {registration.base_model_name!r}. An adapter "
                            "applies to the base it was trained on and to nothing else."
                        )
                    ),
                    details={
                        "adapter": name,
                        "registered": known,
                        "reason": "incompatible_base",
                        "declared_base_digest": registration.base_artifact_digest,
                        "served_base_digest": base.artifact_digest,
                    },
                )
            if not any(adapter.name == name for adapter in registered):
                # Compatible, but this server was launched without it. It folds in at an idle,
                # and until then the honest answer is "not yet", never the bare base.
                self._require_idle(handle.model_name, restart_reason="adapter_registration")
                raise AdapterNotFound(  # pragma: no cover — an idle server has already folded in
                    f"Adapter {name!r} is registered but the running server did not load it.",
                    details={"adapter": name, "registered": known, "reason": "unknown"},
                )
            return _Selection(
                name=name,
                identity=registration.identity,
                confidence=verification.confidence or IdentityConfidence.NAME_ONLY,
            )

    def _build_body(
        self,
        request: GenerationRequest,
        handle: ServerHandle,
        *,
        is_chat: bool,
        stream: bool,
        selection: _Selection | None,
    ) -> dict[str, Any]:
        """Build the request body, stating the whole adapter configuration where there is one."""
        with self._lock:
            registered = self._server_adapters.get(handle.model_name, ())
        lora = lora_field(registered, selected=selection.name if selection else None)
        if is_chat:
            return build_chat_body(
                request,
                alias=request.identity.provider_model_name,
                stream=stream,
                lora=lora,
            )
        return build_completion_body(request, stream=stream, lora=lora)

    def _generate_once(
        self,
        request: GenerationRequest,
        handle: ServerHandle,
        path: str,
        body: Mapping[str, Any],
        start_ns: int,
        *,
        is_chat: bool,
        selection: _Selection | None = None,
    ) -> GenerationResult:
        """Run the round trip :meth:`generate` wraps in its event pair."""
        payload = self._post_json(
            handle,
            path,
            body,
            timeout=request.timeout_seconds,
            context_size=request.runtime_profile.context_size,
        )
        if not isinstance(payload, dict):
            raise ProviderProtocolError(
                f"The server at {handle.base_url} returned something other than a JSON object.",
                details={"base_url": handle.base_url, "body": truncated_text(json.dumps(payload))},
            )
        error = read_error(payload)
        if error is not None:
            raise self._build_message_error(
                error,
                handle,
                status_code=200,
                context_size=request.runtime_profile.context_size,
            )
        wall_ms = elapsed_ms(start_ns, self._monotonic())
        if is_chat:
            choice = first_choice(payload)
            message = choice.get("message")
            message = message if isinstance(message, Mapping) else {}
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            reasoning = message.get("reasoning_content")
            tool_calls = parse_tool_calls(
                message.get("tool_calls"), call_prefix=f"llamacpp-{start_ns}"
            )
            fingerprint = payload.get("system_fingerprint")
            return GenerationResult(
                text=text,
                identity=request.identity,
                finish_reason=finish_reason_for(
                    choice.get("finish_reason"), has_tool_calls=bool(tool_calls)
                ),
                usage=read_chat_usage(payload, text=text),
                timing=self._timing(payload, client_wall_ms=wall_ms, client_ttft_ms=UNSUPPORTED),
                tool_calls=tool_calls,
                thinking=reasoning if isinstance(reasoning, str) else UNSUPPORTED,
                provider_version=(
                    fingerprint
                    if isinstance(fingerprint, str) and fingerprint
                    else handle.build_info
                ),
                adapter=selection.identity if selection is not None else None,
                adapter_base_confidence=selection.confidence if selection is not None else None,
                raw=dict(payload),
            )
        content = payload.get("content")
        text = content if isinstance(content, str) else ""
        return GenerationResult(
            text=text,
            identity=request.identity,
            finish_reason=completion_finish_reason(payload),
            usage=read_completion_usage(payload, text=text),
            timing=self._timing(payload, client_wall_ms=wall_ms, client_ttft_ms=UNSUPPORTED),
            thinking=UNSUPPORTED,
            provider_version=handle.build_info,
            adapter=selection.identity if selection is not None else None,
            adapter_base_confidence=selection.confidence if selection is not None else None,
            raw=dict(payload),
        )

    @staticmethod
    def _timing(
        payload: Mapping[str, Any], *, client_wall_ms: Measurement, client_ttft_ms: Measurement
    ) -> Timing:
        """Join the server's ``timings`` with this process's own two readings (spec §11.3)."""
        backend = read_backend_timing(payload)
        return Timing(
            client_wall_ms=client_wall_ms,
            client_ttft_ms=client_ttft_ms,
            backend_prompt_eval_ms=backend.backend_prompt_eval_ms,
            backend_decode_ms=backend.backend_decode_ms,
        )

    def _already_cancelled(self, request: GenerationRequest) -> StreamFailed:
        """The one terminal event a stream cancelled before it began is entitled to.

        Delivered rather than raised, so a caller has one cancellation path; no server is
        spawned and no connection opened for a request nobody wants any more.
        """
        self._events.started(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        event = self._cancelled("")
        self._events.failed(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
            error_code=event.error.code,
        )
        return event

    def _walk(
        self,
        request: GenerationRequest,
        response: httpx.Response,
        start_ns: int,
        handle: ServerHandle,
        *,
        is_chat: bool,
        selection: _Selection | None = None,
        lease: _InFlightLease | None = None,
    ) -> Iterator[StreamEvent]:
        """Observe every event :meth:`_drain` produces, then hand it on unchanged.

        See :meth:`modelrack.providers.ollama.OllamaProvider._walk` for why observation lives in
        one wrapper and why the inner generator is closed explicitly in ``finally``.
        """
        model_name = request.identity.provider_model_name
        events = self._drain(
            request, response, start_ns, handle, is_chat=is_chat, selection=selection
        )
        try:
            for event in events:
                self._observe(event, request, start_ns, model_name)
                yield event
        finally:
            events.close()
            if lease is not None:
                lease.release()

    def _observe(
        self, event: StreamEvent, request: GenerationRequest, start_ns: int, model_name: str
    ) -> None:
        """Emit the observability event matching one stream event, if anyone is listening."""
        if not self._events.is_observed:
            return
        if isinstance(event, StreamCompleted):
            self._events.completed(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                elapsed_ms=event.result.timing.client_wall_ms,
                output_tokens=event.result.usage.tokens.output_tokens,
                finish_reason=event.result.finish_reason.value,
            )
        elif isinstance(event, StreamFailed):
            self._events.failed(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                error_code=event.error.code,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )
        else:
            self._events.chunk(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                chunk_index=event.index,
                elapsed_ms=elapsed_ms(start_ns, self._monotonic()),
            )

    def _drain(  # noqa: C901 — one state machine, two wire shapes; splitting it would hide the terminal rule
        self,
        request: GenerationRequest,
        response: httpx.Response,
        start_ns: int,
        handle: ServerHandle,
        *,
        is_chat: bool,
        selection: _Selection | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Drain one SSE stream, owning ``response`` for its entire remaining lifetime.

        Both shapes arrive as ``data:`` events (``server-common.cpp``, ``format_oai_sse``) and
        both report a mid-stream error as ``data: {"error": …}``. They end differently: the chat
        stream with ``data: [DONE]`` after a usage chunk, the native stream with a chunk whose
        ``stop`` is ``true`` — and **no** ``[DONE]``. A stream that ends any other way is
        truncated, and truncation is a :class:`~modelrack.streaming.StreamFailed`, never a short
        answer.
        """
        cancel = request.cancel
        answer = StringIO()
        thinking = StringIO()
        saw_thinking = False
        tool_fragments: dict[int, dict[str, Any]] = {}
        tool_order: list[int] = []
        first_delta_ns: int | None = None
        delta_index = 0
        finish_raw: Any = None
        terminal: dict[str, Any] = {}
        fingerprint: str | None = None
        completed = False
        try:
            lines = iter_capped_lines(
                response.iter_lines(),
                max_chunk_bytes=self._max_chunk_bytes,
                base_url=handle.base_url,
            )
            for data in iter_sse_events(lines):
                if cancel is not None and cancel.is_cancelled:
                    yield self._cancelled(answer.getvalue())
                    return
                if is_chat and data == _DONE_SENTINEL:
                    completed = True
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed event from {handle.base_url} was not valid JSON.",
                            details={"base_url": handle.base_url, "body": truncated_text(data)},
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return
                if not isinstance(payload, dict):
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed event from {handle.base_url} was not a JSON object.",
                            details={"base_url": handle.base_url, "body": truncated_text(data)},
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return
                error = read_error(payload)
                if error is not None:
                    yield StreamFailed(
                        error=self._build_message_error(
                            error,
                            handle,
                            status_code=response.status_code,
                            context_size=request.runtime_profile.context_size,
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return
                if is_chat:
                    usage = payload.get("usage")
                    if isinstance(usage, Mapping):
                        terminal["usage"] = usage
                    timings = payload.get("timings")
                    if isinstance(timings, Mapping):
                        terminal["timings"] = timings
                    marker = payload.get("system_fingerprint")
                    if isinstance(marker, str) and marker:
                        fingerprint = marker
                    choice = first_choice(payload)
                    delta = choice.get("delta")
                    delta = delta if isinstance(delta, Mapping) else {}
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        first_delta_ns = first_delta_ns or self._monotonic()
                        saw_thinking = True
                        thinking.write(reasoning)
                        yield ThinkingDelta(text=reasoning, index=delta_index)
                        delta_index += 1
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        first_delta_ns = first_delta_ns or self._monotonic()
                        answer.write(content)
                        yield TokenDelta(text=content, index=delta_index)
                        delta_index += 1
                    raw_tool_calls = delta.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        for fragment in raw_tool_calls:
                            if not isinstance(fragment, Mapping):
                                continue
                            first_delta_ns = first_delta_ns or self._monotonic()
                            call_index = tool_call_index(fragment, fallback=len(tool_order))
                            if call_index not in tool_fragments:
                                tool_fragments[call_index] = {
                                    "id": None,
                                    "name": None,
                                    "arguments": [],
                                }
                                tool_order.append(call_index)
                            entry = tool_fragments[call_index]
                            fragment_id, fragment_name, fragment_arguments = tool_call_fragment(
                                fragment
                            )
                            if fragment_id is not None:
                                entry["id"] = fragment_id
                            if fragment_name is not None:
                                entry["name"] = fragment_name
                            if fragment_arguments is not None:
                                entry["arguments"].append(fragment_arguments)
                            yield ToolCallDelta(
                                call_index=call_index,
                                id=fragment_id,
                                name=fragment_name,
                                arguments_fragment=fragment_arguments,
                                index=delta_index,
                            )
                            delta_index += 1
                    choice_finish = choice.get("finish_reason")
                    if choice_finish is not None:
                        finish_raw = choice_finish
                    continue
                content = payload.get("content")
                if isinstance(content, str) and content:
                    first_delta_ns = first_delta_ns or self._monotonic()
                    answer.write(content)
                    yield TokenDelta(text=content, index=delta_index)
                    delta_index += 1
                if payload.get("stop") is True:
                    terminal = payload
                    completed = True
                    break
            else:
                yield StreamFailed(
                    error=ProviderProtocolError(
                        f"The stream from {handle.base_url} ended without its terminal "
                        f"{'[DONE] event' if is_chat else 'stop chunk'}.",
                        details={"base_url": handle.base_url},
                    ),
                    partial_text=answer.getvalue(),
                )
                return

            if not completed:  # pragma: no cover — the `else` above always returns first
                raise AssertionError("stream loop exited without completing or returning")
            if cancel is not None and cancel.is_cancelled:  # pragma: no cover — see the sibling
                # Reachable only if cancel() fires from another thread between the top-of-loop
                # check on the terminal event and this one; the terminal event is its own SSE
                # event on both shapes, so a single-threaded test cannot land here.
                yield self._cancelled(answer.getvalue())
                return
            wall_ms = elapsed_ms(start_ns, self._monotonic())
            ttft_ms = (
                elapsed_ms(start_ns, first_delta_ns) if first_delta_ns is not None else UNSUPPORTED
            )
            text = answer.getvalue()
            if is_chat:
                tool_calls: tuple[ToolCall, ...] = tuple(
                    tool_call_from_parts(
                        call_id=tool_fragments[index]["id"],
                        name=tool_fragments[index]["name"],
                        raw_arguments="".join(tool_fragments[index]["arguments"]) or None,
                        fallback_id=f"llamacpp-{start_ns}-{position}",
                    )
                    for position, index in enumerate(tool_order)
                )
                result = GenerationResult(
                    text=text,
                    identity=request.identity,
                    finish_reason=finish_reason_for(finish_raw, has_tool_calls=bool(tool_calls)),
                    usage=read_chat_usage(terminal, text=text),
                    timing=self._timing(terminal, client_wall_ms=wall_ms, client_ttft_ms=ttft_ms),
                    tool_calls=tool_calls,
                    thinking=thinking.getvalue() if saw_thinking else UNSUPPORTED,
                    provider_version=fingerprint or handle.build_info,
                    adapter=selection.identity if selection is not None else None,
                    adapter_base_confidence=(
                        selection.confidence if selection is not None else None
                    ),
                    raw={key: dict(value) for key, value in terminal.items()},
                )
            else:
                result = GenerationResult(
                    text=text,
                    identity=request.identity,
                    finish_reason=completion_finish_reason(terminal),
                    usage=read_completion_usage(terminal, text=text),
                    timing=self._timing(terminal, client_wall_ms=wall_ms, client_ttft_ms=ttft_ms),
                    thinking=UNSUPPORTED,
                    provider_version=handle.build_info,
                    adapter=selection.identity if selection is not None else None,
                    adapter_base_confidence=(
                        selection.confidence if selection is not None else None
                    ),
                    raw=dict(terminal),
                )
            yield StreamCompleted(result=result)
        except httpx.HTTPError as exc:
            yield StreamFailed(
                error=translate_stream_interruption(exc, base_url=handle.base_url),
                partial_text=answer.getvalue(),
            )
        except ProviderProtocolError as exc:
            # Raised by `iter_capped_lines` for an oversize line — already typed, and the stream
            # has begun, so it is delivered rather than raised.
            yield StreamFailed(error=exc, partial_text=answer.getvalue())
        finally:
            response.close()

    def _cancelled(self, partial_text: str) -> StreamFailed:
        """The terminal event for a stream the caller stopped, its output attached."""
        return StreamFailed(
            error=GenerationCancelled(
                "Generation was cancelled by the caller's token.",
                details={"partial_text": partial_text},
            ),
            partial_text=partial_text,
        )

    # ------------------------------------------------------------------------- transport

    def _build_message_error(
        self,
        error: LlamaCppError,
        handle: ServerHandle,
        *,
        status_code: int,
        context_size: int | None,
    ) -> ProviderError:
        """Classify an error the server sent as content, by its type and then its message.

        Shared between a real 4xx, a 200 whose body is an error object, and an in-band streamed
        error. ``exceed_context_size_error`` — or the prose a build without the type uses — is
        :class:`~modelrack.errors.ContextLimitExceeded`, with the requested and served sizes the
        server stated (real numbers here, where Ollama's adapter can only say ``UNSUPPORTED``).
        ``unavailable_error`` is the server still loading:
        :class:`~modelrack.errors.ProviderUnavailable` with reason ``not_ready``. Anything else
        with a message is
        :class:`~modelrack.errors.ProviderRejected`, the message preserved verbatim.
        """
        if error.is_context_overflow:
            return ContextLimitExceeded(
                error.message,
                details={
                    "requested_tokens": (
                        error.n_prompt_tokens if error.n_prompt_tokens is not None else UNSUPPORTED
                    ),
                    "maximum_tokens": (
                        error.n_ctx
                        if error.n_ctx is not None
                        else (
                            handle.served_context
                            if is_supported(handle.served_context)
                            else (context_size if context_size is not None else UNSUPPORTED)
                        )
                    ),
                },
            )
        if error.is_not_ready:
            return ProviderUnavailable(
                error.message,
                details={
                    "base_url": handle.base_url,
                    "reason": ProviderUnavailableReason.NOT_READY.value,
                },
            )
        return ProviderRejected(
            error.message,
            details={
                "status_code": error.status_code if error.status_code is not None else status_code,
                "provider_message": error.message,
                "error_type": error.error_type,
            },
        )

    def _timeout_for(self, request: GenerationRequest) -> float | httpx._client.UseClientDefault:
        """Return the per-request timeout override, or the client's own default."""
        if request.timeout_seconds is not None:
            return request.timeout_seconds
        return httpx.USE_CLIENT_DEFAULT

    def _raise_for_status(
        self, response: httpx.Response, handle: ServerHandle, *, context_size: int | None
    ) -> None:
        """Translate a non-2xx response into the typed error spec §13 names, and raise it."""
        try:
            payload = read_capped_json(
                response, max_bytes=self._max_response_bytes, base_url=handle.base_url
            )
            raw_text = json.dumps(payload)
        except ProviderProtocolError as exc:
            if "limit_bytes" in exc.details:
                raise
            payload, raw_text = None, str(exc.details.get("body", ""))
        error = read_error(payload)
        if error is not None:
            raise self._build_message_error(
                error, handle, status_code=response.status_code, context_size=context_size
            )
        raise ProviderProtocolError(
            f"The server at {handle.base_url} returned status {response.status_code} with an "
            "unexpected body.",
            details={
                "base_url": handle.base_url,
                "status_code": response.status_code,
                "body": truncated_text(raw_text),
            },
        )

    def _post_json(
        self,
        handle: ServerHandle,
        path: str,
        body: Mapping[str, Any],
        *,
        timeout: float | None,
        context_size: int | None,
    ) -> Any:  # noqa: ANN401 — the server's own JSON shape
        """POST a JSON body to a supervised server and return the parsed JSON response."""
        try:
            with self._client.stream(
                "POST",
                handle.base_url + path,
                json=body,
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_status(response, handle, context_size=context_size)
                return read_capped_json(
                    response, max_bytes=self._max_response_bytes, base_url=handle.base_url
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=handle.base_url) from exc

    def _get_json(self, handle: ServerHandle, path: str) -> Any:  # noqa: ANN401 — server JSON
        """GET a path on a supervised server and return the parsed JSON response."""
        try:
            with self._client.stream("GET", handle.base_url + path) as response:
                if response.status_code >= 400:
                    self._raise_for_status(response, handle, context_size=None)
                return read_capped_json(
                    response, max_bytes=self._max_response_bytes, base_url=handle.base_url
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=handle.base_url) from exc

    def __repr__(self) -> str:
        """Name the directory and the port range, for a debugger session."""
        low, high = self._supervisor.port_range
        return (
            f"LlamaCppProvider(model_directory={str(self._model_directory)!r}, ports={low}-{high})"
        )
