"""Tests for :mod:`modelrack.providers._llamacpp_process` — spawning, health-waiting,
terminating and recovering ``llama-server`` processes.

Two layers, deliberately. :class:`TestSupervisor` drives :class:`LlamaServerSupervisor` through
the injected seam — a fake launcher, a fake process table, a stepping clock — and proves every
supervision path without a binary: this is the suite that runs in CI, where there is no
``llama-server`` and never will be. :class:`TestRealLauncher` then proves the *defaults* against
ordinary shell processes: that ``start_new_session`` plus a group signal really does kill a
grandchild, that stderr really lands in the file, that the process table really reads ``/proc``.
Neither layer needs llama.cpp; together they are the claim that the seam is honest.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from baseaicore import ValidationError, is_supported

from conftest import (
    FakeLauncher,
    FakeMonotonic,
    FakeProcessTable,
    FakeServerProcess,
    FakeSleep,
)
from modelrack import ProviderTimeout, ProviderUnavailable, ProviderUnavailableReason
from modelrack.providers._llamacpp_process import (
    LaunchSpec,
    LlamaServerSupervisor,
    OrphanAction,
    PidRecord,
    PosixProcessTable,
    SubprocessLauncher,
    loopback_port_is_free,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_PORTS = (18080, 18083)


def _argv(port: int) -> tuple[str, ...]:
    return ("llama-server", "--model", "/models/m.gguf", "--port", str(port))


def _ready(_port: int) -> bool:
    return True


def _never(_port: int) -> bool:
    return False


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
def table() -> FakeProcessTable:
    return FakeProcessTable()


@pytest.fixture
def sleeper() -> FakeSleep:
    return FakeSleep()


@pytest.fixture
def supervisor(
    tmp_path: Path, launcher: FakeLauncher, table: FakeProcessTable, sleeper: FakeSleep
) -> LlamaServerSupervisor:
    (tmp_path / "state").mkdir()  # the application's data root exists before the adapter does
    return LlamaServerSupervisor(
        state_dir=tmp_path / "state",
        port_range=_PORTS,
        launcher=launcher,
        process_table=table,
        port_is_free=lambda _port: True,
        sleep=sleeper,
        monotonic=FakeMonotonic(step_seconds=0.1),
        startup_timeout_seconds=1.0,
        shutdown_timeout_seconds=0.5,
        poll_interval_seconds=0.05,
        stderr_tail_bytes=64,
    )


class TestConstruction:
    def test_the_state_directory_is_created_on_the_first_spawn_not_at_construction(
        self, tmp_path: Path, sleeper: FakeSleep, launcher: FakeLauncher
    ) -> None:
        state = tmp_path / "deep" / "state"

        supervisor = LlamaServerSupervisor(
            state_dir=state, sleep=sleeper, launcher=launcher, port_is_free=lambda _port: True
        )
        assert not state.exists(), "a supervisor built for a health check touches nothing"
        supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        assert state.is_dir()
        assert supervisor.state_dir == state
        assert supervisor.port_range == (8180, 8189)
        assert "8180-8189" in repr(supervisor)

    @pytest.mark.parametrize("port_range", [(0, 10), (9000, 8999), (1, 70000)])
    def test_an_invalid_port_range_is_refused(
        self, tmp_path: Path, sleeper: FakeSleep, port_range: tuple[int, int]
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            LlamaServerSupervisor(state_dir=tmp_path, port_range=port_range, sleep=sleeper)

        assert raised.value.details["field"] == "port_range"

    @pytest.mark.parametrize(
        "field", ["startup_timeout_seconds", "shutdown_timeout_seconds", "poll_interval_seconds"]
    )
    def test_a_non_positive_timeout_is_refused(
        self, tmp_path: Path, sleeper: FakeSleep, field: str
    ) -> None:
        overrides: dict[str, Any] = {field: 0.0}

        with pytest.raises(ValidationError) as raised:
            LlamaServerSupervisor(state_dir=tmp_path, sleep=sleeper, **overrides)

        assert raised.value.details["field"] == field

    def test_a_zero_stderr_tail_is_refused(self, tmp_path: Path, sleeper: FakeSleep) -> None:
        with pytest.raises(ValidationError) as raised:
            LlamaServerSupervisor(state_dir=tmp_path, sleep=sleeper, stderr_tail_bytes=0)

        assert raised.value.details["field"] == "stderr_tail_bytes"


class TestSpawn:
    def test_a_healthy_server_is_tracked_with_its_startup_time(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready, launch_key="k")

        assert handle.port == _PORTS[0]
        assert handle.argv == _argv(_PORTS[0])
        assert handle.launch_key == "k"
        assert handle.base_url == f"http://127.0.0.1:{_PORTS[0]}"
        assert is_supported(handle.startup_ms)
        assert handle.startup_ms > 0
        assert supervisor.handle_for("m") is handle
        assert supervisor.handles() == (handle,)
        assert supervisor.is_running(handle)
        assert launcher.specs[0] == LaunchSpec(
            argv=_argv(_PORTS[0]),
            port=_PORTS[0],
            stderr_path=handle.stderr_path,
            model_name="m",
        )

    def test_the_pid_file_is_written_before_the_health_wait_begins(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        """A crash of *this* process during the wait must leave a trail for the next one."""
        seen: list[bool] = []

        def probe(port: int) -> bool:
            path = supervisor.state_dir / f"llama-server-{port}.pid.json"
            seen.append(path.exists())
            return True

        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=probe)

        assert seen == [True]
        record = PidRecord.from_json(handle.pid_path.read_text())
        assert record.pid == launcher.processes[0].pid
        assert record.owner_pid == os.getpid()
        assert record.port == handle.port
        assert record.model_name == "m"
        assert record.argv == handle.argv
        assert record.stderr_path == str(handle.stderr_path)

    def test_the_health_wait_polls_until_the_probe_answers(
        self, supervisor: LlamaServerSupervisor, sleeper: FakeSleep
    ) -> None:
        answers = iter([False, False, True])

        handle = supervisor.spawn(
            model_name="m", build_argv=_argv, probe=lambda _port: next(answers)
        )

        assert supervisor.is_running(handle)
        assert sleeper.calls == [0.05, 0.05]

    def test_a_server_that_exits_during_startup_is_a_typed_error_with_its_stderr(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        launcher.stderr_text = "load_model: failed to load model\nerror: unable to load model\n"
        launcher.plan(lambda spec: FakeServerProcess(41, exit_after_polls=2, exit_code=3))

        with pytest.raises(ProviderUnavailable) as raised:
            supervisor.spawn(model_name="m", build_argv=_argv, probe=_never)

        details = raised.value.details
        assert details["reason"] == ProviderUnavailableReason.PROCESS_EXITED.value
        assert details["exit_code"] == 3
        assert details["port"] == _PORTS[0]
        assert details["argv"] == list(_argv(_PORTS[0]))
        assert "unable to load model" in details["stderr_tail"]
        assert supervisor.handles() == ()
        assert not list(supervisor.state_dir.glob("*.pid.json")), "the pid file must be gone"

    def test_the_stderr_tail_is_bounded(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        launcher.stderr_text = "x" * 500 + "THE END"
        launcher.plan(lambda spec: FakeServerProcess(41, exit_after_polls=1, exit_code=2))

        with pytest.raises(ProviderUnavailable) as raised:
            supervisor.spawn(model_name="m", build_argv=_argv, probe=_never)

        tail = raised.value.details["stderr_tail"]
        assert tail.endswith("THE END")
        assert len(tail) == 64

    def test_a_server_that_never_becomes_healthy_is_killed_and_a_timeout_raised(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        """The failure that would otherwise hold a port and a GPU forever."""
        launcher.plan(lambda spec: FakeServerProcess(41, survives_terminate=True))

        with pytest.raises(ProviderTimeout) as raised:
            supervisor.spawn(model_name="m", build_argv=_argv, probe=_never)

        process = launcher.processes[0]
        assert process.terminated and process.killed, "SIGTERM, then SIGKILL after the grace"
        assert process.poll() == -9
        details = raised.value.details
        assert details["limit_seconds"] == 1.0
        assert details["elapsed_seconds"] >= 1.0
        assert details["argv"] == list(_argv(_PORTS[0]))
        assert "listening" in details["stderr_tail"]
        assert supervisor.handles() == ()
        assert not list(supervisor.state_dir.glob("*.pid.json"))

    def test_a_missing_binary_is_launch_failed_with_the_argv(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        launcher.plan(FileNotFoundError(2, "No such file or directory", "llama-server"))

        with pytest.raises(ProviderUnavailable) as raised:
            supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        assert raised.value.details["reason"] == ProviderUnavailableReason.LAUNCH_FAILED.value
        assert raised.value.details["argv"] == list(_argv(_PORTS[0]))
        assert "llama-server" in raised.value.details["error"]
        assert not list(supervisor.state_dir.glob("*.pid.json"))

    def test_a_second_server_for_the_same_model_is_refused(
        self, supervisor: LlamaServerSupervisor
    ) -> None:
        supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        with pytest.raises(ValidationError):
            supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)


class TestPorts:
    def test_each_server_gets_the_next_free_port(self, supervisor: LlamaServerSupervisor) -> None:
        first = supervisor.spawn(model_name="a", build_argv=_argv, probe=_ready)
        second = supervisor.spawn(model_name="b", build_argv=_argv, probe=_ready)

        assert (first.port, second.port) == (_PORTS[0], _PORTS[0] + 1)

    def test_a_port_something_else_holds_is_skipped(
        self, tmp_path: Path, launcher: FakeLauncher, table: FakeProcessTable, sleeper: FakeSleep
    ) -> None:
        supervisor = LlamaServerSupervisor(
            state_dir=tmp_path,
            port_range=_PORTS,
            launcher=launcher,
            process_table=table,
            port_is_free=lambda port: port != _PORTS[0],
            sleep=sleeper,
        )

        handle = supervisor.spawn(model_name="a", build_argv=_argv, probe=_ready)

        assert handle.port == _PORTS[0] + 1

    def test_a_port_a_live_foreign_pid_file_claims_is_skipped(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        table.alive.add(777)
        record = PidRecord(
            pid=778,
            owner_pid=777,
            port=_PORTS[0],
            model_name="theirs",
            argv=("x",),
            started_at="2026-09-03T10:00:00+00:00",
            stderr_path="",
        )
        (supervisor.state_dir / f"llama-server-{_PORTS[0]}.pid.json").write_text(record.to_json())

        handle = supervisor.spawn(model_name="a", build_argv=_argv, probe=_ready)

        assert handle.port == _PORTS[0] + 1

    def test_an_exhausted_range_is_launch_failed_naming_the_range(
        self, supervisor: LlamaServerSupervisor
    ) -> None:
        for name in "abcd":
            supervisor.spawn(model_name=name, build_argv=_argv, probe=_ready)

        with pytest.raises(ProviderUnavailable) as raised:
            supervisor.spawn(model_name="e", build_argv=_argv, probe=_ready)

        assert raised.value.details["reason"] == ProviderUnavailableReason.LAUNCH_FAILED.value
        assert raised.value.details["port_range"] == list(_PORTS)
        assert raised.value.details["held_ports"] == list(range(_PORTS[0], _PORTS[1] + 1))


class TestTerminate:
    def test_terminate_signals_the_group_removes_the_pid_file_and_forgets_the_handle(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        code = supervisor.terminate(handle)

        process = launcher.processes[0]
        assert process.terminated and not process.killed
        assert code == -15
        assert not handle.pid_path.exists()
        assert supervisor.handle_for("m") is None

    def test_a_server_that_ignores_sigterm_is_killed_after_the_grace(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        launcher.plan(lambda spec: FakeServerProcess(41, survives_terminate=True))
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        code = supervisor.terminate(handle)

        assert launcher.processes[0].killed
        assert code == -9

    def test_terminating_an_already_exited_server_is_trivial(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)
        launcher.processes[0].crash(0)

        code = supervisor.terminate(handle)

        assert code == 0
        assert not launcher.processes[0].terminated

    def test_terminate_all_stops_every_tracked_server_and_is_idempotent(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        supervisor.spawn(model_name="a", build_argv=_argv, probe=_ready)
        supervisor.spawn(model_name="b", build_argv=_argv, probe=_ready)

        supervisor.terminate_all()
        supervisor.terminate_all()

        assert supervisor.handles() == ()
        assert all(process.terminated for process in launcher.processes)
        assert not list(supervisor.state_dir.glob("*.pid.json"))

    def test_reap_exited_drops_servers_that_died_on_their_own(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        crashed = supervisor.spawn(model_name="crashed", build_argv=_argv, probe=_ready)
        alive = supervisor.spawn(model_name="alive", build_argv=_argv, probe=_ready)
        launcher.processes[0].crash(137)

        reaped = supervisor.reap_exited()

        assert reaped == ((crashed, 137),)
        assert supervisor.handles() == (alive,)
        assert not crashed.pid_path.exists()
        assert alive.pid_path.exists()

    def test_the_stderr_tail_of_a_live_server_is_readable(
        self, supervisor: LlamaServerSupervisor, launcher: FakeLauncher
    ) -> None:
        launcher.stderr_text = "srv  main: listening on http://127.0.0.1:18080\n"
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        assert "listening" in supervisor.stderr_tail(handle)

    def test_terminate_tolerates_a_pid_file_already_removed(
        self, supervisor: LlamaServerSupervisor
    ) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)
        handle.pid_path.unlink()

        assert supervisor.terminate(handle) == -15

    def test_a_missing_stderr_log_reads_as_empty(self, supervisor: LlamaServerSupervisor) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)
        handle.stderr_path.unlink()

        assert supervisor.stderr_tail(handle) == ""


class TestOrphanSweep:
    """The recovery path: pid files left by a supervisor that died."""

    @staticmethod
    def _record(
        supervisor: LlamaServerSupervisor,
        *,
        pid: int,
        owner_pid: int,
        port: int,
        argv: tuple[str, ...] = ("llama-server", "--port", "x"),
    ) -> Path:
        record = PidRecord(
            pid=pid,
            owner_pid=owner_pid,
            port=port,
            model_name="left-behind",
            argv=argv,
            started_at="2026-09-03T10:00:00+00:00",
            stderr_path=str(supervisor.state_dir / "old.log"),
        )
        path = supervisor.state_dir / f"llama-server-{port}.pid.json"
        path.write_text(record.to_json())
        return path

    def test_an_orphan_running_the_recorded_command_line_is_killed(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        argv = ("llama-server", "--port", "18080")
        path = self._record(supervisor, pid=500, owner_pid=499, port=18080, argv=argv)
        table.alive.add(500)
        table.command_lines[500] = argv

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.KILLED
        assert (reports[0].pid, reports[0].port) == (500, 18080)
        assert table.signals == [(500, signal.SIGTERM)]
        assert 500 not in table.alive
        assert not path.exists()

    def test_an_orphan_that_ignores_sigterm_is_killed_after_the_grace(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable, sleeper: FakeSleep
    ) -> None:
        argv = ("llama-server",)
        self._record(supervisor, pid=500, owner_pid=499, port=18080, argv=argv)
        table.alive.add(500)
        table.command_lines[500] = argv
        table.ignores_term.add(500)

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.KILLED
        assert table.signals == [(500, signal.SIGTERM), (500, signal.SIGKILL)]
        assert sleeper.calls, "the grace was waited out before SIGKILL"
        assert 500 not in table.alive

    def test_an_orphan_whose_command_line_cannot_be_read_is_still_killed(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        """``None`` from the table is "cannot say", never a reason to spare the pid."""
        self._record(supervisor, pid=500, owner_pid=499, port=18080)
        table.alive.add(500)

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.KILLED

    def test_a_record_whose_owner_is_alive_is_left_alone(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        path = self._record(supervisor, pid=500, owner_pid=499, port=18080)
        table.alive.update({499, 500})

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.FOREIGN_OWNER
        assert table.signals == []
        assert path.exists()

    def test_a_stale_record_is_removed_without_signalling(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        path = self._record(supervisor, pid=500, owner_pid=499, port=18080)

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.STALE_FILE_REMOVED
        assert table.signals == []
        assert not path.exists()

    def test_a_reused_pid_running_something_else_is_not_killed(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        path = self._record(supervisor, pid=500, owner_pid=499, port=18080)
        table.alive.add(500)
        table.command_lines[500] = ("postgres", "-D", "/var/lib/postgresql")

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.PID_REUSED
        assert table.signals == []
        assert not path.exists()

    def test_a_record_owned_by_this_process_with_no_handle_is_an_orphan(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        """A spawn this very process lost track of — a crash mid-spawn — is recovered too."""
        self._record(supervisor, pid=500, owner_pid=os.getpid(), port=18080)
        table.alive.add(500)

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.KILLED

    def test_a_record_for_a_port_this_supervisor_holds_is_skipped(
        self, supervisor: LlamaServerSupervisor
    ) -> None:
        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        assert supervisor.sweep_orphans() == ()
        assert handle.pid_path.exists()

    @pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"pid": 1}', '{"pid": 1, "argv": 3}'])
    def test_an_unreadable_record_is_removed_and_nothing_is_signalled(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable, text: str
    ) -> None:
        path = supervisor.state_dir / "llama-server-18080.pid.json"
        path.write_text(text)

        reports = supervisor.sweep_orphans()

        assert reports[0].action is OrphanAction.UNREADABLE_FILE_REMOVED
        assert reports[0].pid is None
        assert table.signals == []
        assert not path.exists()

    def test_spawn_sweeps_before_allocating(
        self, supervisor: LlamaServerSupervisor, table: FakeProcessTable
    ) -> None:
        argv = ("llama-server",)
        self._record(supervisor, pid=500, owner_pid=499, port=_PORTS[0], argv=argv)
        table.alive.add(500)
        table.command_lines[500] = argv

        handle = supervisor.spawn(model_name="m", build_argv=_argv, probe=_ready)

        assert 500 not in table.alive
        assert handle.port == _PORTS[0], "the recovered port is free again"


class TestPidRecord:
    def test_round_trips_through_json(self) -> None:
        record = PidRecord(
            pid=1,
            owner_pid=2,
            port=3,
            model_name="m",
            argv=("a", "b"),
            started_at="2026-09-03T10:00:00+00:00",
            stderr_path="x.log",
        )

        assert PidRecord.from_json(record.to_json()) == record
        assert json.loads(record.to_json())["argv"] == ["a", "b"]


@pytest.fixture
def real_supervisor(tmp_path: Path) -> Iterator[LlamaServerSupervisor]:
    """A supervisor over the real launcher and process table, with real (short) timeouts."""
    supervisor = LlamaServerSupervisor(
        state_dir=tmp_path / "state",
        port_range=(18090, 18091),
        port_is_free=lambda _port: True,
        sleep=lambda seconds: None,
        startup_timeout_seconds=10.0,
        shutdown_timeout_seconds=2.0,
        poll_interval_seconds=0.01,
    )
    yield supervisor
    supervisor.terminate_all()


def _shell(script: str) -> Callable[[int], tuple[str, ...]]:
    return lambda _port: ("sh", "-c", script)


class TestRealLauncher:
    """The default seam implementations, against shell processes rather than llama.cpp.

    Real signals, real pids, a real ``/proc``. What these prove is the part a fake cannot: that
    a group signal reaches a grandchild the server forked, that stderr really lands in the file,
    and that the process table answers about pids this supervisor never spawned.
    """

    def test_terminate_kills_the_whole_process_group(
        self, real_supervisor: LlamaServerSupervisor
    ) -> None:
        handle = real_supervisor.spawn(
            model_name="m", build_argv=_shell("sleep 30 & exec sleep 30"), probe=_ready
        )
        pgid = handle.process.pid
        assert os.getpgid(pgid) == pgid, "spawned as its own group leader"

        code = real_supervisor.terminate(handle)

        assert code == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)  # nothing left in the group — the backgrounded sleep died too

    def test_a_child_that_traps_sigterm_is_killed_after_the_grace(
        self, real_supervisor: LlamaServerSupervisor, tmp_path: Path
    ) -> None:
        ready = tmp_path / "ready"
        ignore = (
            "import pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
        )
        handle = real_supervisor.spawn(
            model_name="m",
            build_argv=lambda _port: (sys.executable, "-c", ignore),
            probe=lambda _port: ready.exists(),  # healthy only once the handler is installed
        )

        code = real_supervisor.terminate(handle)

        assert code == -signal.SIGKILL

    def test_stderr_reaches_the_log_file_and_the_exit_code_is_reported(
        self, real_supervisor: LlamaServerSupervisor
    ) -> None:
        script = "echo 'failed to load model: boom' >&2; exit 3"

        with pytest.raises(ProviderUnavailable) as raised:
            real_supervisor.spawn(model_name="m", build_argv=_shell(script), probe=_never)

        assert raised.value.details["exit_code"] == 3
        assert "boom" in raised.value.details["stderr_tail"]

    def test_a_missing_executable_is_launch_failed(
        self, real_supervisor: LlamaServerSupervisor
    ) -> None:
        with pytest.raises(ProviderUnavailable) as raised:
            real_supervisor.spawn(
                model_name="m",
                build_argv=lambda _port: ("/nonexistent/llama-server",),
                probe=_ready,
            )

        assert raised.value.details["reason"] == ProviderUnavailableReason.LAUNCH_FAILED.value

    def test_the_pid_file_survives_this_process_and_a_new_supervisor_recovers_the_orphan(
        self, tmp_path: Path
    ) -> None:
        """The whole point of pid files, end to end: a child of a *different* Python process is
        left running, and a fresh supervisor over the same state directory kills it.
        """
        state = tmp_path / "state"
        state.mkdir()
        script = (
            "import subprocess, sys, json, os\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen(['sleep', '60'], start_new_session=True,"
            " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "record = {'pid': child.pid, 'owner_pid': os.getpid(), 'port': 18090,"
            " 'model_name': 'm', 'argv': ['sleep', '60'], 'started_at': 'x', 'stderr_path': ''}\n"
            f"Path({str(state)!r}, 'llama-server-18090.pid.json').write_text(json.dumps(record))\n"
            "print(child.pid)\n"
        )
        orphan_pid = int(
            subprocess.run(  # noqa: S603 — a fixed script, spawned by this test on purpose
                [sys.executable, "-c", script], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        assert PosixProcessTable().is_alive(orphan_pid)

        supervisor = LlamaServerSupervisor(
            state_dir=state, sleep=lambda _s: None, poll_interval_seconds=0.01
        )
        reports = supervisor.sweep_orphans()

        assert [report.action for report in reports] == [OrphanAction.KILLED]
        assert not PosixProcessTable().is_alive(orphan_pid)
        assert not list(state.glob("*.pid.json"))


class TestPosixProcessTable:
    def test_this_process_is_alive_and_names_its_interpreter(self) -> None:
        table = PosixProcessTable()

        assert table.is_alive(os.getpid())
        command_line = table.command_line(os.getpid())
        assert command_line is not None
        assert "python" in command_line[0]

    def test_a_pid_that_does_not_exist_is_not_alive_and_has_no_command_line(self) -> None:
        table = PosixProcessTable()
        pid = 2**22 - 1  # above Linux's default pid_max

        assert not table.is_alive(pid)
        assert table.command_line(pid) is None
        table.signal_group(pid, signal.SIGTERM)  # tolerated, never raises

    def test_a_zombie_has_no_command_line(self) -> None:
        """An exited, unreaped child keeps its pid and loses its argv: ``None``, not ``()``."""
        child = subprocess.Popen(["true"])  # noqa: S603, S607 — a fixed, harmless command
        # Wait for the exit without reaping it, so the child is a zombie for the assertions:
        # its pid still exists, its argv is gone. (Polling cmdline instead would race the
        # moment of exec, when it is momentarily empty for a process that is very much alive.)
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        try:
            assert PosixProcessTable().is_alive(child.pid)
            assert PosixProcessTable().command_line(child.pid) is None
        finally:
            child.wait()

    def test_init_is_alive_even_though_it_cannot_be_signalled(self) -> None:
        """``PermissionError`` from ``kill(pid, 0)`` means the process exists."""
        assert PosixProcessTable().is_alive(1)


class TestLoopbackPortIsFree:
    def test_a_bound_port_is_not_free_and_a_released_one_is(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            port = holder.getsockname()[1]

            assert not loopback_port_is_free(port)
        assert loopback_port_is_free(port)


class TestSubprocessLauncherDirectly:
    def test_the_handle_reports_the_pid_and_exit_code(self, tmp_path: Path) -> None:
        spec = LaunchSpec(
            argv=("sh", "-c", "exit 7"), port=1, stderr_path=tmp_path / "e.log", model_name="m"
        )

        process = SubprocessLauncher()(spec)

        assert process.pid > 0
        assert process.wait(5.0) == 7
        assert process.poll() == 7
        process.terminate()  # on an exited group: tolerated
        process.kill()

    def test_wait_returns_none_while_the_process_runs(self, tmp_path: Path) -> None:
        spec = LaunchSpec(
            argv=("sleep", "5"), port=1, stderr_path=tmp_path / "e.log", model_name="m"
        )
        process = SubprocessLauncher()(spec)
        try:
            assert process.wait(0.01) is None
            assert process.poll() is None
        finally:
            process.kill()
            process.wait(5.0)
