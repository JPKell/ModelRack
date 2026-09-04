"""Domain-adjacent module — supervising one ``llama-server`` process per served base.

Imports :mod:`baseaicore`, this package's errors and the standard library; performs **process and
local-file I/O** — spawning, signalling, pid files, a captured-stderr log — and **no network I/O**.
The health probe it polls while a server starts is injected by the adapter, which owns the HTTP
client; this module never learns what a probe does, only whether it answered yes.

This is the "process lifecycle" half of what
[ADR-0062](../../../docs/adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md)
decision 6 says the suite takes over when it serves GGUF files itself, and the three risks the
[adapter roadmap §4.1](../../../docs/roadmap/adapter-roadmap.md) names for it are each answered
here, by name:

* **Orphaned processes.** Every server is spawned into its own session, so a signal to its
  process *group* reaches anything it forked; :meth:`LlamaServerSupervisor.terminate` sends
  ``SIGTERM``, waits a bounded grace, then ``SIGKILL``. A pid file is written *before* the health
  wait begins, recording the server's pid, the supervising process's pid, the port and the exact
  argv, so that if the supervising process itself dies the next supervisor to open the same
  ``state_dir`` finds the record, sees its owner is gone, verifies the pid still runs the same
  command line, and kills it (:meth:`LlamaServerSupervisor.sweep_orphans`). A record whose owner
  is still alive is left alone — another application's live server is not an orphan.
* **Port management.** Ports come from a configured range; one is chosen by skipping every port
  this supervisor holds or a live pid file claims, then asking the injected ``port_is_free``.
* **Startup-failure diagnosis.** The child's stderr goes to a file in ``state_dir`` (never to a
  pipe nobody drains — a full pipe buffer would stall the server itself). A server that exits
  before it is healthy raises :class:`~modelrack.errors.ProviderUnavailable` carrying its exit
  code and the tail of that file; one that never becomes healthy is killed and raises
  :class:`~modelrack.errors.ProviderTimeout` carrying the same tail. Neither is ever "it did not
  start" with no reason attached.

**Everything that touches the operating system is injected**, because on the machine this was
written on — and in CI — there is no ``llama-server`` to launch: :class:`ProcessLauncher` (spawn),
:class:`ProcessTable` (liveness and command lines of processes this supervisor did not spawn),
``port_is_free``, ``sleep`` and ``monotonic``. The defaults, :class:`SubprocessLauncher` and
:class:`PosixProcessTable`, are the real thing and are exercised in the default suite against an
ordinary shell command rather than against llama.cpp. Phase 8's leak tests (twenty load/unload
cycles, no orphan, flat memory) are written against the same seam, which is why it exists from
the first commit rather than being retrofitted.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from baseaicore import UNSUPPORTED, Measurement, ValidationError, elapsed_ms, monotonic_ns, utc_now

from modelrack.errors import ProviderTimeout, ProviderUnavailable, ProviderUnavailableReason

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_PORT_RANGE",
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_STDERR_TAIL_BYTES",
    "LaunchSpec",
    "LlamaServerSupervisor",
    "OrphanAction",
    "OrphanReport",
    "PidRecord",
    "PosixProcessTable",
    "ProcessLauncher",
    "ProcessTable",
    "ServerHandle",
    "ServerProcess",
    "SubprocessLauncher",
    "loopback_port_is_free",
]

logger = logging.getLogger(__name__)

DEFAULT_PORT_RANGE: Final[tuple[int, int]] = (8180, 8189)
"""Ten ports, chosen to sit clear of llama-server's own default (8080) and the suite's application
ports (8765–8768), so a server this package spawns never collides with one an operator started by
hand. Configurable; a range is what makes several bases resident at once possible at all."""

DEFAULT_STARTUP_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long a server may take to answer its first healthy probe. A 9 GB model loads in seconds
from the page cache and in a minute or so cold from disk; five minutes leaves room for a larger
model on a slower disk without letting a wedged server hold a port and a GPU forever."""

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 20.0
"""The grace between ``SIGTERM`` and ``SIGKILL``. llama-server exits promptly on ``SIGTERM``; the
grace exists for a server mid-generation to release its memory cleanly rather than be torn down."""

DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 0.25
"""How often the health probe is retried while a server starts."""

DEFAULT_STDERR_TAIL_BYTES: Final[int] = 16 * 1024
"""How much of the end of the stderr log a typed startup error carries. Enough for llama.cpp's
model-load banner and the error that followed it; bounded so an error object never becomes a log
file (the same discipline :func:`modelrack.providers._http.truncated_text` applies to bodies)."""

_KILL_WAIT_SECONDS: Final[float] = 5.0
_PID_FILE_SUFFIX: Final[str] = ".pid.json"
_STDERR_FILE_SUFFIX: Final[str] = ".stderr.log"
_LOOPBACK: Final[str] = "127.0.0.1"
_MAX_PORT: Final[int] = 65535


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Everything a :class:`ProcessLauncher` needs to start one server.

    Attributes:
        argv: The complete command line, the executable first. Built by the adapter's wire module
            from the model path, the port and the runtime profile; the launcher runs it verbatim
            and adds nothing.
        port: The loopback port the server was told to listen on, for diagnostics.
        stderr_path: Where the launcher must send the child's stderr (and stdout). A file, not a
            pipe — see the module docstring.
        model_name: The model being served, for diagnostics and pid records.
    """

    argv: tuple[str, ...]
    port: int
    stderr_path: Path
    model_name: str


class ServerProcess(Protocol):
    """A handle on one spawned server, as much of :class:`subprocess.Popen` as supervision needs.

    Implemented by the real launcher's return value and by a test's fake. The two signal methods
    address the whole process *group*, never the single pid: the server is spawned as a session
    leader precisely so that anything it forks dies with it.
    """

    @property
    def pid(self) -> int:
        """The server's pid, which is also its process-group id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code if the server has exited, else ``None``."""
        ...

    def wait(self, timeout_seconds: float) -> int | None:
        """Wait up to ``timeout_seconds`` for exit; the code, or ``None`` if still running."""
        ...

    def terminate(self) -> None:
        """Send ``SIGTERM`` to the process group. A no-op if it has already gone."""
        ...

    def kill(self) -> None:
        """Send ``SIGKILL`` to the process group. A no-op if it has already gone."""
        ...


type ProcessLauncher = Callable[[LaunchSpec], ServerProcess]
"""The spawn seam: given a launch spec, start the process and return a handle on it.

Raises :class:`OSError` — ``FileNotFoundError`` for a missing binary, ``PermissionError`` for one
that cannot be executed — and nothing else; the supervisor translates. Injected into
:class:`~modelrack.providers.llamacpp.LlamaCppProvider` the way ``httpx.Client`` and clocks are
injected elsewhere in this package, so that every supervision path is testable without a
``llama-server`` on the machine.
"""


class ProcessTable(Protocol):
    """How the supervisor reads and signals processes it holds no handle on.

    Needed only for orphan recovery: a server left behind by a supervisor that died is known
    through its pid file, not through a :class:`ServerProcess`.
    """

    def is_alive(self, pid: int) -> bool:
        """Report whether a process with this pid exists."""
        ...

    def command_line(self, pid: int) -> tuple[str, ...] | None:
        """Return the process's argv, or ``None`` where the platform cannot say."""
        ...

    def signal_group(self, pid: int, signum: int) -> None:
        """Send ``signum`` to the process group led by ``pid``; a no-op if it has gone."""
        ...


class _PopenProcess:
    """The real :class:`ServerProcess`, over a :class:`subprocess.Popen` in its own session."""

    __slots__ = ("_popen",)

    def __init__(self, popen: subprocess.Popen[bytes]) -> None:
        self._popen = popen

    @property
    def pid(self) -> int:
        """The child's pid; equal to its process-group id because it was started as a leader."""
        return self._popen.pid

    def poll(self) -> int | None:
        """Return the exit code if the child has exited, else ``None``."""
        return self._popen.poll()

    def wait(self, timeout_seconds: float) -> int | None:
        """Wait up to ``timeout_seconds``; return the exit code or ``None`` if still running."""
        try:
            return self._popen.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self) -> None:
        """``SIGTERM`` the whole group; tolerate a group that has already gone."""
        _signal_group(self._popen.pid, signal.SIGTERM)

    def kill(self) -> None:
        """``SIGKILL`` the whole group; tolerate a group that has already gone."""
        _signal_group(self._popen.pid, signal.SIGKILL)


def _signal_group(pid: int, signum: int) -> None:
    """Signal the process group led by ``pid``, ignoring a group that no longer exists."""
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        return


class SubprocessLauncher:
    """The real :class:`ProcessLauncher`: :class:`subprocess.Popen` in a new session.

    ``start_new_session=True`` makes the child a session and process-group leader, which is what
    turns "terminate the server" into "terminate the server and everything it forked". stdin is
    ``/dev/null`` — a server must never block on a terminal it does not have — and stdout and
    stderr both go to the spec's ``stderr_path``, opened for writing and closed in the parent
    once the child holds it.
    """

    def __call__(self, spec: LaunchSpec) -> ServerProcess:
        """Start the server described by ``spec``.

        Args:
            spec: What to run and where its output goes.

        Returns:
            A handle on the running child.

        Raises:
            OSError: If the executable cannot be found or run, or the stderr file cannot be
                opened. Not translated here — the supervisor turns it into a typed error with
                the attempted argv attached.
        """
        with spec.stderr_path.open("wb") as stderr:
            popen = subprocess.Popen(  # noqa: S603 — argv is built by this package, not from input
                spec.argv,
                stdin=subprocess.DEVNULL,
                stdout=stderr,
                stderr=stderr,
                start_new_session=True,
            )
        return _PopenProcess(popen)


class PosixProcessTable:
    """The real :class:`ProcessTable`, over ``kill(pid, 0)``, ``/proc`` and ``killpg``.

    ``command_line`` reads ``/proc/<pid>/cmdline`` where that filesystem exists and answers
    ``None`` elsewhere — an honest "cannot say", which the orphan sweep treats as *not
    verified* and therefore as no reason to spare a pid whose owner is gone.
    """

    def is_alive(self, pid: int) -> bool:
        """Report whether ``pid`` exists. A process we may not signal still exists."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def command_line(self, pid: int) -> tuple[str, ...] | None:
        """Return ``/proc/<pid>/cmdline`` split on NUL, or ``None`` if it cannot be read."""
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        if not raw:
            return None
        return tuple(
            part.decode("utf-8", errors="replace") for part in raw.rstrip(b"\0").split(b"\0")
        )

    def signal_group(self, pid: int, signum: int) -> None:
        """Signal the group led by ``pid``; a group that has gone is not an error."""
        _signal_group(pid, signum)


def loopback_port_is_free(port: int) -> bool:
    """Report whether nothing is currently bound to ``127.0.0.1:port``.

    The default ``port_is_free`` for :class:`LlamaServerSupervisor`. A bind-and-release probe:
    it opens no connection, so it does not trip the test suite's socket guard, and it is the same
    check the server itself will make a moment later. The window between this answer and the
    server's own bind is real and unclosable from here; a server that loses that race exits with
    a bind error, which the startup diagnosis reports with its stderr rather than hiding.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((_LOOPBACK, port))
        except OSError:
            return False
    return True


@dataclass(frozen=True, slots=True)
class PidRecord:
    """What a pid file says, so a later supervisor can recognise and recover a server.

    Attributes:
        pid: The server's pid (and process-group id).
        owner_pid: The pid of the Python process that spawned it. A record whose owner is still
            alive belongs to a live supervisor and is never touched by another.
        port: The loopback port the server was told to use.
        model_name: Which model it serves.
        argv: The exact command line, so a reused pid running something else is not killed by
            mistake.
        started_at: When it was spawned, RFC 3339.
        stderr_path: Where its output went.
    """

    pid: int
    owner_pid: int
    port: int
    model_name: str
    argv: tuple[str, ...]
    started_at: str
    stderr_path: str

    def to_json(self) -> str:
        """Serialize for the pid file."""
        return json.dumps(
            {
                "pid": self.pid,
                "owner_pid": self.owner_pid,
                "port": self.port,
                "model_name": self.model_name,
                "argv": list(self.argv),
                "started_at": self.started_at,
                "stderr_path": self.stderr_path,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> PidRecord:
        """Parse a pid file.

        Raises:
            ValueError: If the text is not the JSON object :meth:`to_json` writes. A sweep
                treats that as an unreadable record — removed, never acted on.
        """
        payload: Any = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("pid record is not a JSON object")
        argv = payload["argv"]
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("pid record argv is not a list of strings")
        return cls(
            pid=int(payload["pid"]),
            owner_pid=int(payload["owner_pid"]),
            port=int(payload["port"]),
            model_name=str(payload["model_name"]),
            argv=tuple(argv),
            started_at=str(payload["started_at"]),
            stderr_path=str(payload["stderr_path"]),
        )


class OrphanAction(StrEnum):
    """What :meth:`LlamaServerSupervisor.sweep_orphans` did about one pid file."""

    KILLED = "killed"
    """The owner was gone and the pid still ran the recorded command line: killed, file removed."""

    STALE_FILE_REMOVED = "stale_file_removed"
    """The owner was gone and so was the server: only the file was left, and it was removed."""

    PID_REUSED = "pid_reused"
    """The owner was gone but the pid now runs something else: file removed, process untouched."""

    FOREIGN_OWNER = "foreign_owner"
    """The owner is still alive: another supervisor's server, left exactly as it was."""

    UNREADABLE_FILE_REMOVED = "unreadable_file_removed"
    """The file was not a pid record this package wrote: removed, nothing signalled."""


@dataclass(frozen=True, slots=True)
class OrphanReport:
    """One pid file's fate during a sweep, for logging and for the tests that prove the sweep.

    Attributes:
        path: The pid file.
        action: What was done.
        pid: The recorded server pid, or ``None`` when the file was unreadable.
        port: The recorded port, or ``None`` when the file was unreadable.
    """

    path: Path
    action: OrphanAction
    pid: int | None = None
    port: int | None = None


@dataclass(slots=True)
class ServerHandle:
    """One supervised server this supervisor spawned and still tracks.

    Mutable — ``build_info`` and ``served_context`` are learned from the server after it is
    healthy — and owned by the supervisor; callers read it and never construct one.

    Attributes:
        model_name: The model it serves, and the key it is tracked under.
        port: Its loopback port.
        argv: The exact command line it was started with. Two requests whose launch flags agree
            share a server; one whose flags differ gets a restart — so this, not a profile hash,
            is what "already resident under this profile" compares.
        process: The handle from the launcher.
        pid_path: Its pid file.
        stderr_path: Its captured output.
        started_at: When it was spawned.
        startup_ms: How long it took to answer its first healthy probe — this process's own
            observation, which is what :attr:`~modelrack.provider.LoadResult.load_ms` reports
            for a supervised server, since llama-server reports no load time of its own.
        build_info: llama.cpp's ``build_info`` string from ``/props``, once read; ``None`` until
            then or if the read failed.
        served_context: The context size the server reports serving, from ``/props``;
            ``UNSUPPORTED`` until read or if it could not be read.
        launch_key: An opaque string the adapter derives from everything that decides the
            launch flags — the model file and the runtime profile — so "is this server already
            running the way this request wants?" is one comparison. Empty when the caller did
            not supply one.
    """

    model_name: str
    port: int
    argv: tuple[str, ...]
    process: ServerProcess
    pid_path: Path
    stderr_path: Path
    started_at: datetime
    startup_ms: Measurement = UNSUPPORTED
    build_info: str | None = None
    served_context: Measurement = UNSUPPORTED
    launch_key: str = ""

    @property
    def base_url(self) -> str:
        """Where this server listens."""
        return f"http://{_LOOPBACK}:{self.port}"


class LlamaServerSupervisor:
    """Spawns, health-waits, tracks and terminates ``llama-server`` processes for one adapter.

    One instance per :class:`~modelrack.providers.llamacpp.LlamaCppProvider`. Tracks at most one
    server per model name — one base per process, per ADR-0062 decision 1 — and any number of
    models at once, each on its own port from the range; whether several fit on the GPU is the
    caller's residency policy (ADR-0038), not this class's.

    Thread-safe: every method that reads or changes the set of handles holds one re-entrant
    lock. The lock is held across a spawn's health wait, so two threads asking for the same
    model get one server rather than two.

    Args:
        state_dir: Where pid files and stderr logs live. Created on the first spawn, not at
            construction, so an adapter built only to answer a health check touches nothing.
            Named by the constructing application inside its own data root — this package reads
            no configuration and picks no directory of its own (spec §12).
        port_range: Inclusive ``(low, high)`` of loopback ports servers may listen on.
        launcher: The spawn seam. Defaults to :class:`SubprocessLauncher`.
        process_table: How processes without a handle are read and signalled, for orphan
            recovery. Defaults to :class:`PosixProcessTable`.
        port_is_free: Whether a port may be used. Defaults to :func:`loopback_port_is_free`.
        sleep: Called between health probes and between orphan-kill checks. Injected so a test
            need not wait through a real startup.
        monotonic: The monotonic nanosecond clock the startup timeout and ``startup_ms`` are
            measured with.
        clock: The wall clock a handle's ``started_at`` and a pid record's timestamp come from.
        startup_timeout_seconds: How long a server may take to become healthy before it is
            killed and a :class:`~modelrack.errors.ProviderTimeout` raised.
        shutdown_timeout_seconds: The grace between ``SIGTERM`` and ``SIGKILL``.
        poll_interval_seconds: How often the health probe is retried during startup.
        stderr_tail_bytes: How much captured output a startup error carries.

    Raises:
        ValidationError: If the port range is empty, reversed or outside 1–65535, or any
            timeout or interval is not positive.
    """

    def __init__(
        self,
        *,
        state_dir: Path,
        port_range: tuple[int, int] = DEFAULT_PORT_RANGE,
        launcher: ProcessLauncher | None = None,
        process_table: ProcessTable | None = None,
        port_is_free: Callable[[int], bool] = loopback_port_is_free,
        sleep: Callable[[float], None],
        monotonic: Callable[[], int] = monotonic_ns,
        clock: Callable[[], datetime] = utc_now,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
    ) -> None:
        """Validate the configuration; the state directory is created on the first spawn."""
        low, high = port_range
        if not (1 <= low <= high <= _MAX_PORT):
            raise ValidationError(
                f"port_range must be an inclusive (low, high) within 1..{_MAX_PORT} with low <= "
                f"high; got {port_range!r}.",
                details={"field": "port_range", "value": list(port_range)},
            )
        for name, value in (
            ("startup_timeout_seconds", startup_timeout_seconds),
            ("shutdown_timeout_seconds", shutdown_timeout_seconds),
            ("poll_interval_seconds", poll_interval_seconds),
        ):
            if value <= 0:
                raise ValidationError(
                    f"{name} must be positive; got {value!r}. There is no 'no timeout' for a "
                    "process this package supervises — an unbounded wait is a wedged server "
                    "holding a GPU.",
                    details={"field": name, "value": value},
                )
        if stderr_tail_bytes < 1:
            raise ValidationError(
                f"stderr_tail_bytes must be at least 1; got {stderr_tail_bytes}.",
                details={"field": "stderr_tail_bytes", "value": stderr_tail_bytes},
            )
        self._state_dir = state_dir
        self._port_range = (low, high)
        self._launcher: ProcessLauncher = launcher or SubprocessLauncher()
        self._process_table: ProcessTable = process_table or PosixProcessTable()
        self._port_is_free = port_is_free
        self._sleep = sleep
        self._monotonic = monotonic
        self._clock = clock
        self._startup_timeout_seconds = startup_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stderr_tail_bytes = stderr_tail_bytes
        self._lock = threading.RLock()
        self._handles: dict[str, ServerHandle] = {}

    # -------------------------------------------------------------------------------- reading

    @property
    def state_dir(self) -> Path:
        """Where this supervisor keeps pid files and stderr logs."""
        return self._state_dir

    @property
    def port_range(self) -> tuple[int, int]:
        """The inclusive port range servers are spawned on."""
        return self._port_range

    def handles(self) -> tuple[ServerHandle, ...]:
        """Return every server currently tracked, sorted by model name.

        Returns:
            A snapshot. A server that has exited on its own is still listed until
            :meth:`reap_exited` runs — callers that want the live set call that first.
        """
        with self._lock:
            return tuple(self._handles[name] for name in sorted(self._handles))

    def handle_for(self, model_name: str) -> ServerHandle | None:
        """Return the tracked server for ``model_name``, or ``None``."""
        with self._lock:
            return self._handles.get(model_name)

    def is_running(self, handle: ServerHandle) -> bool:
        """Report whether the handle's process is still alive."""
        return handle.process.poll() is None

    def stderr_tail(self, handle: ServerHandle) -> str:
        """Return the end of a server's captured output, for diagnostics.

        Args:
            handle: The server.

        Returns:
            At most ``stderr_tail_bytes`` from the end of its log, decoded leniently; ``""`` if
            the log cannot be read.
        """
        return self._read_tail(handle.stderr_path)

    # ------------------------------------------------------------------------------ lifecycle

    def spawn(
        self,
        *,
        model_name: str,
        build_argv: Callable[[int], tuple[str, ...]],
        probe: Callable[[int], bool],
        launch_key: str = "",
    ) -> ServerHandle:
        """Start a server for ``model_name`` and wait until it answers a healthy probe.

        Sweeps orphans first, allocates a port, writes the pid file, launches, then polls
        ``probe`` until it answers ``True``, the process exits, or the startup timeout passes.

        Args:
            model_name: The model to serve. If a server for it is already tracked, that is a
                caller error — the adapter checks residency first — and it is refused rather
                than silently doubled.
            build_argv: Builds the command line for the port this method chooses.
            probe: Answers whether the server on a port is healthy. Provided by the adapter,
                which owns the HTTP client; must not raise.
            launch_key: Recorded on the handle unchanged; see :attr:`ServerHandle.launch_key`.

        Returns:
            The handle, with ``startup_ms`` set from this process's own clock.

        Raises:
            ProviderUnavailable: With ``reason`` ``launch_failed`` if no port is free or the
                binary cannot be run (``details`` carries ``argv`` where one was built); with
                ``reason`` ``process_exited`` if the server exited before it was healthy
                (``details`` carries ``exit_code``, ``stderr_tail``, ``stderr_path``,
                ``argv``). The pid file is removed in every failure case.
            ProviderTimeout: If the server was still running but not healthy when the startup
                timeout passed. It is killed — group signal, grace, ``SIGKILL`` — before this is
                raised, so a timeout never leaves a process behind. ``details`` carries
                ``elapsed_seconds``, ``limit_seconds``, ``stderr_tail`` and ``argv``.
            ValidationError: If a server for ``model_name`` is already tracked.
        """
        with self._lock:
            if model_name in self._handles:
                raise ValidationError(
                    f"A server for {model_name!r} is already tracked; unload it before spawning "
                    "another. Two servers for one base is never what a caller meant.",
                    details={"model_name": model_name},
                )
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self.sweep_orphans()
            port = self._allocate_port()
            argv = build_argv(port)
            stderr_path = self._state_dir / f"llama-server-{port}{_STDERR_FILE_SUFFIX}"
            pid_path = self._state_dir / f"llama-server-{port}{_PID_FILE_SUFFIX}"
            spec = LaunchSpec(argv=argv, port=port, stderr_path=stderr_path, model_name=model_name)
            start_ns = self._monotonic()
            started_at = self._clock()
            try:
                process = self._launcher(spec)
            except OSError as exc:
                raise ProviderUnavailable(
                    f"Could not launch llama-server for {model_name!r}: {exc}",
                    details={
                        "reason": ProviderUnavailableReason.LAUNCH_FAILED.value,
                        "argv": list(argv),
                        "port": port,
                        "error": str(exc),
                    },
                ) from exc
            handle = ServerHandle(
                model_name=model_name,
                port=port,
                argv=argv,
                process=process,
                pid_path=pid_path,
                stderr_path=stderr_path,
                started_at=started_at,
                launch_key=launch_key,
            )
            self._write_pid_file(handle)
            self._wait_until_healthy(handle, probe, start_ns)
            handle.startup_ms = elapsed_ms(start_ns, self._monotonic())
            self._handles[model_name] = handle
            logger.debug(
                "llamacpp.server.started",
                extra={
                    "model_name": model_name,
                    "port": port,
                    "pid": process.pid,
                    "startup_ms": handle.startup_ms,
                },
            )
            return handle

    def terminate(self, handle: ServerHandle) -> int | None:
        """Stop a server: ``SIGTERM`` its group, wait the grace, ``SIGKILL`` if needed.

        Args:
            handle: The server. Removed from the tracked set and its pid file deleted whether or
                not it was still running — a handle whose process already exited is terminated
                trivially.

        Returns:
            The exit code, or ``None`` if the process did not report one within the kill wait —
            which, after ``SIGKILL``, means the kernel has not reaped it yet rather than that it
            survived.
        """
        with self._lock:
            self._handles.pop(handle.model_name, None)
            code = self._stop(handle.process)
            self._remove_pid_file(handle.pid_path)
            logger.debug(
                "llamacpp.server.terminated",
                extra={"model_name": handle.model_name, "port": handle.port, "exit_code": code},
            )
            return code

    def terminate_all(self) -> None:
        """Stop every tracked server. Safe to call more than once and at interpreter exit."""
        with self._lock:
            for handle in list(self._handles.values()):
                try:
                    self.terminate(handle)
                except OSError:  # pragma: no cover — defensive: exit-time cleanup must finish
                    logger.debug(
                        "llamacpp.server.terminate_failed",
                        extra={"model_name": handle.model_name, "port": handle.port},
                    )

    def reap_exited(self) -> tuple[tuple[ServerHandle, int], ...]:
        """Drop every tracked server whose process has exited on its own.

        Returns:
            The handles dropped, each with its exit code, so the caller can report a crash
            rather than let the next request meet a refused connection with no explanation.
            Their pid files are removed.
        """
        with self._lock:
            exited: list[tuple[ServerHandle, int]] = []
            for name, handle in list(self._handles.items()):
                code = handle.process.poll()
                if code is None:
                    continue
                del self._handles[name]
                self._remove_pid_file(handle.pid_path)
                exited.append((handle, code))
                logger.debug(
                    "llamacpp.server.exited",
                    extra={"model_name": name, "port": handle.port, "exit_code": code},
                )
            return tuple(exited)

    def sweep_orphans(self) -> tuple[OrphanReport, ...]:
        """Recover servers left behind by supervisors that died, from their pid files.

        For each pid file in ``state_dir`` that no current handle owns: if its recorded owner
        process is still alive, it is another supervisor's and is left alone; if the server pid
        is gone, the stale file is removed; if the pid is alive but runs a different command
        line, the pid was reused and only the file is removed; otherwise the server is an orphan
        and its group is terminated, then killed after the grace.

        Returns:
            One report per pid file examined. Empty when the directory holds none.
        """
        with self._lock:
            held_ports = {handle.port for handle in self._handles.values()}
            reports: list[OrphanReport] = []
            for path in sorted(self._state_dir.glob(f"*{_PID_FILE_SUFFIX}")):
                try:
                    record = PidRecord.from_json(path.read_text())
                except (OSError, ValueError, KeyError, TypeError):
                    self._remove_pid_file(path)
                    reports.append(OrphanReport(path, OrphanAction.UNREADABLE_FILE_REMOVED))
                    continue
                if record.port in held_ports:
                    continue
                reports.append(self._recover(path, record))
            for report in reports:
                logger.debug(
                    "llamacpp.server.sweep",
                    extra={
                        "path": str(report.path),
                        "action": report.action.value,
                        "pid": report.pid,
                        "port": report.port,
                    },
                )
            return tuple(reports)

    # -------------------------------------------------------------------------------- internals

    def _recover(self, path: Path, record: PidRecord) -> OrphanReport:
        """Decide and apply the fate of one pid file this supervisor does not hold a handle on."""
        table = self._process_table
        owner_alive = record.owner_pid != os.getpid() and table.is_alive(record.owner_pid)
        if owner_alive:
            return OrphanReport(path, OrphanAction.FOREIGN_OWNER, record.pid, record.port)
        if not table.is_alive(record.pid):
            self._remove_pid_file(path)
            return OrphanReport(path, OrphanAction.STALE_FILE_REMOVED, record.pid, record.port)
        command_line = table.command_line(record.pid)
        if command_line is not None and command_line != record.argv:
            self._remove_pid_file(path)
            return OrphanReport(path, OrphanAction.PID_REUSED, record.pid, record.port)
        self._kill_group(record.pid)
        self._remove_pid_file(path)
        return OrphanReport(path, OrphanAction.KILLED, record.pid, record.port)

    def _kill_group(self, pid: int) -> None:
        """Terminate, then kill, a process group known only by pid."""
        table = self._process_table
        table.signal_group(pid, signal.SIGTERM)
        if self._wait_for_death(pid, self._shutdown_timeout_seconds):
            return
        table.signal_group(pid, signal.SIGKILL)
        self._wait_for_death(pid, _KILL_WAIT_SECONDS)

    def _wait_for_death(self, pid: int, limit_seconds: float) -> bool:
        """Poll the process table until ``pid`` is gone or ``limit_seconds`` pass."""
        start_ns = self._monotonic()
        limit_ns = int(limit_seconds * 1_000_000_000)
        while self._process_table.is_alive(pid):
            if self._monotonic() - start_ns >= limit_ns:
                return False
            self._sleep(self._poll_interval_seconds)
        return True

    def _stop(self, process: ServerProcess) -> int | None:
        """Kill-tree with grace: ``SIGTERM``, wait, ``SIGKILL``, wait."""
        code = process.poll()
        if code is not None:
            return code
        process.terminate()
        code = process.wait(self._shutdown_timeout_seconds)
        if code is not None:
            return code
        process.kill()
        return process.wait(_KILL_WAIT_SECONDS)

    def _wait_until_healthy(
        self, handle: ServerHandle, probe: Callable[[int], bool], start_ns: int
    ) -> None:
        """Poll until the server answers, exits, or runs out of time — never leaving it running
        on the failure paths.
        """
        limit_ns = int(self._startup_timeout_seconds * 1_000_000_000)
        while True:
            exit_code = handle.process.poll()
            if exit_code is not None:
                tail = self._read_tail(handle.stderr_path)
                self._remove_pid_file(handle.pid_path)
                raise ProviderUnavailable(
                    f"llama-server for {handle.model_name!r} exited with code {exit_code} before "
                    f"it became healthy on port {handle.port}. Its captured output is attached.",
                    details={
                        "reason": ProviderUnavailableReason.PROCESS_EXITED.value,
                        "exit_code": exit_code,
                        "port": handle.port,
                        "argv": list(handle.argv),
                        "stderr_path": str(handle.stderr_path),
                        "stderr_tail": tail,
                    },
                )
            if probe(handle.port):
                return
            elapsed_ns = self._monotonic() - start_ns
            if elapsed_ns >= limit_ns:
                self._stop(handle.process)
                tail = self._read_tail(handle.stderr_path)
                self._remove_pid_file(handle.pid_path)
                raise ProviderTimeout(
                    f"llama-server for {handle.model_name!r} did not become healthy on port "
                    f"{handle.port} within {self._startup_timeout_seconds:g} s; it has been "
                    "killed. Its captured output is attached.",
                    details={
                        "elapsed_seconds": elapsed_ns / 1_000_000_000,
                        "limit_seconds": self._startup_timeout_seconds,
                        "port": handle.port,
                        "argv": list(handle.argv),
                        "stderr_path": str(handle.stderr_path),
                        "stderr_tail": tail,
                    },
                )
            self._sleep(self._poll_interval_seconds)

    def _allocate_port(self) -> int:
        """Pick the first port in the range that nothing this supervisor knows of is using."""
        held = {handle.port for handle in self._handles.values()}
        claimed = self._ports_claimed_by_pid_files()
        low, high = self._port_range
        for port in range(low, high + 1):
            if port in held or port in claimed:
                continue
            if self._port_is_free(port):
                return port
        raise ProviderUnavailable(
            f"No free port for llama-server in {low}-{high}: every port is held by a tracked "
            "server, claimed by a live pid file, or bound by something else. Unload a model or "
            "widen port_range.",
            details={
                "reason": ProviderUnavailableReason.LAUNCH_FAILED.value,
                "port_range": [low, high],
                "held_ports": sorted(held | claimed),
            },
        )

    def _ports_claimed_by_pid_files(self) -> set[int]:
        """Ports named by pid files still present after a sweep — live foreign servers."""
        ports: set[int] = set()
        for path in self._state_dir.glob(f"*{_PID_FILE_SUFFIX}"):
            try:
                ports.add(PidRecord.from_json(path.read_text()).port)
            except (OSError, ValueError, KeyError, TypeError):  # pragma: no cover — see below
                # The sweep that always precedes this removed every unreadable file; only a
                # writer racing between the two could put one here. Skipped, never trusted.
                continue
        return ports

    def _write_pid_file(self, handle: ServerHandle) -> None:
        """Record the spawn before the health wait, so a crash of this process leaves a trail."""
        record = PidRecord(
            pid=handle.process.pid,
            owner_pid=os.getpid(),
            port=handle.port,
            model_name=handle.model_name,
            argv=handle.argv,
            started_at=handle.started_at.isoformat(),
            stderr_path=str(handle.stderr_path),
        )
        handle.pid_path.write_text(record.to_json())

    @staticmethod
    def _remove_pid_file(path: Path) -> None:
        """Delete a pid file, tolerating one already gone."""
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _read_tail(self, path: Path) -> str:
        """Return the last ``stderr_tail_bytes`` of a log, or ``""`` if it cannot be read."""
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                stream.seek(max(0, size - self._stderr_tail_bytes))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def __repr__(self) -> str:
        """Name the state directory, port range and tracked count, for a debugger."""
        low, high = self._port_range
        return (
            f"LlamaServerSupervisor(state_dir={str(self._state_dir)!r}, ports={low}-{high}, "
            f"tracked={len(self._handles)})"
        )
