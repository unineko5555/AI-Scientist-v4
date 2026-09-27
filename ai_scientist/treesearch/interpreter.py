"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import json
import os
import select
import signal
import socket
import stat
import sys
import time
import traceback
import subprocess
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin
from ai_scientist.inference.process import _process_identity as _kernel_process_identity
from ai_scientist.inference.process import _read_ownership as _read_group_ownership
from ai_scientist.inference.environment import generated_code_environment
from ai_scientist.inference.errors import InferenceError

# Never fork the worker's in-memory SDK clients or broker credentials into
# model-generated code. Child processes and locks share the fresh spawn context.
_EXECUTION_CONTEXT = get_context("spawn")
Process = _EXECUTION_CONTEXT.Process

logger = logging.getLogger("ai-scientist")


def _process_identity(pid):
    """Share the CLI runner's precise kernel identity; never trust ps seconds."""
    return _kernel_process_identity(pid)


def _read_worker_ownership(path):
    try:
        if path.is_symlink() or path.parent.is_symlink():
            return None
        parent = path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or metadata.st_size > 16384:
                return None
            record = json.loads(handle.read(16385))
        identity = record.get("identity")
        pid = record.get("pid")
        if record.get("version") != 1 or type(pid) is not int or pid <= 1 or pid == os.getpid():
            return None
        if not isinstance(identity, dict) or identity.get("pid") != pid or identity.get("uid") != os.getuid():
            return None
        return record
    except (OSError, ValueError, TypeError):
        return None


def _group_has_live_processes(group_id):
    result = subprocess.run(
        # setsid makes the owned session/group IDs equal on Linux and macOS.
        ["ps", "-o", "stat=", "-g", str(group_id)],
        capture_output=True, text=True, timeout=3,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Could not inspect owned experiment process group")
    return any(line.strip() and not line.strip().startswith("Z") for line in result.stdout.splitlines())


def _pid_is_live(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, timeout=3,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Could not inspect owned process")
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def stop_owned_workers(ownership_dir):
    """Finite fallback for pool workers blocked inside legacy API SDK calls."""
    for record in Path(ownership_dir).glob("*.json"):
        metadata = _read_worker_ownership(record)
        if metadata is None:
            if not record.exists():
                continue
            raise RuntimeError("Cannot prove owned worker record; manual cleanup required")
        pid = metadata["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            identity = _process_identity(pid)
            if identity != metadata["identity"]:
                if identity is None and not _pid_is_live(pid):
                    break
                raise RuntimeError("Cannot prove owned worker identity; GPU remains reserved")
            try:
                os.kill(pid, sig)  # Pool workers are not group leaders.
            except ProcessLookupError:
                break
            deadline = time.monotonic() + 2
            while _pid_is_live(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not _pid_is_live(pid):
                break
        if _pid_is_live(pid):
            raise RuntimeError("Owned pool worker did not stop; GPU remains reserved")
        record.unlink(missing_ok=True)


def stop_owned_sessions(ownership_dir):
    """Stop only this run's recorded experiment groups after worker failure.

    A crashed pool worker can no longer execute its Interpreter finally block.
    Records are private and include birth identity; no executor internals or
    command-line keyword matching are used to select processes for termination.
    """
    if ownership_dir is None or os.name != "posix":
        return
    for record in Path(ownership_dir).glob("*.json"):
        metadata = _read_group_ownership(record)
        if metadata is None:
            if not record.exists():
                continue
            raise RuntimeError("Cannot prove owned Interpreter record; manual cleanup required")
        pid = metadata["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            identity = _process_identity(pid)
            if identity != metadata["identity"]:
                # Read-only grace lets an orphan guardian finish teardown. A
                # numeric PGID alone never grants authority to signal again.
                deadline = time.monotonic() + 2
                while identity is None and _group_has_live_processes(pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                    identity = _process_identity(pid)
                if identity is None and not _group_has_live_processes(pid):
                    break
                raise RuntimeError("Cannot prove owned Interpreter identity; GPU remains reserved, manual cleanup required")
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + 2
            while _group_has_live_processes(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not _group_has_live_processes(pid):
                break
        if _group_has_live_processes(pid):
            raise RuntimeError("Owned experiment process group did not stop; GPU remains reserved")
        record.unlink(missing_ok=True)


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


class InterpreterError(InferenceError, RuntimeError):
    """Execution infrastructure failure, never scientific model feedback."""


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "treesearch/" not in l and "importlib" not in l]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        # Bound an individual stdout frame without dropping any completed data.
        for offset in range(0, len(msg), 8192):
            self.queue.put(msg[offset:offset + 8192])
        return len(msg)

    def flush(self):
        pass


class _SocketSender:
    def __init__(self, channel):
        self.channel = channel
        # Generated experiments can fork DataLoader workers which inherit
        # stdout. The lock must be shared across those processes, not copied.
        self.lock = _EXECUTION_CONTEXT.RLock()

    def send(self, kind, value):
        payload = json.dumps([kind, value], ensure_ascii=False).encode("utf-8") + b"\n"
        with self.lock:
            self.channel.sendall(payload)


class _ChannelQueue:
    """Child-only put facade, keeping stdout/exception/ready contracts intact."""
    def __init__(self, sender, kind):
        self.sender, self.kind = sender, kind

    def put(self, value):
        self.sender.send(self.kind, value)


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] | None = None,
        cancel_event=None,
        interrupt_grace_seconds: float = 2.0,
        ownership_dir=None,
        subscription_only: bool = False,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars (dict[str, str], optional): Environment variables to set in the child process. Defaults to {}.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore
        self._owns_process_group = False
        self.subscription_only = subscription_only
        self.env_vars = generated_code_environment(env_vars or {}, subscription_only=subscription_only)
        self.cancel_event = cancel_event
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self.ownership_dir = Path(ownership_dir) if ownership_dir else None
        self._ownership_file = None
        self._owner_pid = os.getpid()
        self._socket = None
        self.last_result = None

    def child_proc_setup(self, result_outq) -> None:
        # Own a separate process group so generated-code children are cleaned up
        # without touching another experiment or the user's CLI sessions.
        if os.name == "posix":
            os.setsid()
            parent_pid = self._owner_pid

            def stop_if_orphaned():
                while os.getppid() == parent_pid:
                    time.sleep(0.2)
                # Covers the narrow startup interval before the parent records
                # this child's PID, including an abrupt pool-worker SIGKILL.
                os.killpg(os.getpid(), signal.SIGKILL)

            threading.Thread(target=stop_if_orphaned, daemon=True).start()
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()

        child_environment = dict(os.environ)
        child_environment.update({key: value for key, value in self.env_vars.items() if value is not None})
        child_environment = generated_code_environment(child_environment, subscription_only=self.subscription_only)
        os.environ.clear()
        os.environ.update(child_environment)

        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(self, channel) -> None:
        # multiprocessing spawn serializes this bound instance too; close the
        # parent's socket endpoint in the child so EOF has one real producer.
        if self._socket is not None:
            self._socket.close()
        channel.setblocking(True)
        sender = _SocketSender(channel)
        result_outq = _ChannelQueue(sender, "output")
        event_outq = _ChannelQueue(sender, "event")
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        code_stream = channel.makefile("rb")
        for frame in code_stream:
            code = json.loads(frame)["code"]
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                RedirectQueue(result_outq).write(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")
        code_stream.close()
        channel.close()

    def create_process(self) -> None:
        self._owns_process_group = False
        self._owner_pid = os.getpid()
        # Parent socket IO is nonblocking and incrementally framed. Unlike
        # multiprocessing.Queue.get(timeout), a producer killed mid-frame
        # cannot strand the parent inside a blocking recv_bytes call.
        self._socket, child_socket = socket.socketpair()
        self._socket.setblocking(False)
        self._stream_fragments = []
        self._stream_closed = False
        self._events = []
        self._outputs = []
        self._pending_input = b""
        self._input_offset = 0
        self._saw_eof = False
        self.process = Process(
            target=self._run_session,
            args=(child_socket,),
        )
        try:
            self.process.start()
        finally:
            child_socket.close()
        if self.ownership_dir is not None and os.name == "posix":
            self.ownership_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.ownership_dir.chmod(0o700)
            self._ownership_file = self.ownership_dir / f"{self.process.pid}_{uuid.uuid4().hex}.json"
            identity = _process_identity(self.process.pid)
            deadline = time.monotonic() + 2
            while identity is not None and identity.get("pgid") != self.process.pid and time.monotonic() < deadline:
                time.sleep(0.01)
                identity = _process_identity(self.process.pid)
            if identity is None or identity.get("pgid") != self.process.pid or identity.get("uid") != os.getuid():
                raise InterpreterError("Cannot establish precise owned Interpreter identity")
            self._owns_process_group = True
            temporary = self._ownership_file.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                temporary.chmod(0o600)
                json.dump({"version": 1, "pid": self.process.pid, "pgid": self.process.pid, "uid": os.getuid(), "owner_pid": os.getpid(), "identity": identity, "nonce": uuid.uuid4().hex}, handle)
            temporary.replace(self._ownership_file)

    def _accept_frame(self, frame):
        kind, value = json.loads(frame)
        if kind == "event":
            self._events.append(tuple(value))
        elif kind == "output":
            if value == "<|EOF|>":
                self._saw_eof = True
            else:
                self._outputs.append(value)
        else:
            raise InterpreterError("Unknown experiment IPC message")

    def _poll_stream(self, timeout=0.1):
        if self._socket is None or self._stream_closed:
            return
        writable = [self._socket] if self._input_offset < len(self._pending_input) else []
        readable, ready_write, _ = select.select([self._socket], writable, [], timeout)
        if ready_write:
            try:
                written = self._socket.send(self._pending_input[self._input_offset:self._input_offset + 65536])
                self._input_offset += written
            except BlockingIOError:
                pass
            except (BrokenPipeError, ConnectionResetError):
                self._stream_closed = True
        if not readable:
            return
        # Bound work per pass so unlimited stdout never starves deadlines.
        for _ in range(32):
            try:
                data = self._socket.recv(8192)
            except BlockingIOError:
                break
            except ConnectionResetError:
                data = b""
            if not data:
                self._stream_closed = True
                break
            fragments = data.split(b"\n")
            self._stream_fragments.append(fragments[0])
            if len(fragments) > 1:
                self._accept_frame(b"".join(self._stream_fragments))
                for frame in fragments[1:-1]:
                    self._accept_frame(frame)
                self._stream_fragments = [fragments[-1]] if fragments[-1] else []

    def _signal_owned_process(self, sig):
        if self.process is None:
            return
        try:
            if os.name == "posix" and os.getpgid(self.process.pid) == self.process.pid:
                os.killpg(self.process.pid, sig)
            elif self.process.is_alive():
                os.kill(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def _stop_process(self):
        if self.process is None:
            return
        self._signal_owned_process(signal.SIGTERM)
        self.process.join(timeout=2)
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self._signal_owned_process(signal.SIGKILL)
            self.process.join(timeout=2)
        if self.process.exitcode is None:
            raise InterpreterError("Owned experiment process could not be stopped")
        # A child can finish before its descendants. Its process group is still
        # ours; kill any remaining members even when the leader has exited.
        if os.name == "posix" and self._owns_process_group:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while _group_has_live_processes(self.process.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _group_has_live_processes(self.process.pid):
                raise InterpreterError("Owned experiment descendants did not stop")

    def cleanup_session(self):
        if self.process is None:
            return
        self._stop_process()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.process.close()
        self.process = None  # type: ignore
        if self._ownership_file is not None:
            self._ownership_file.unlink(missing_ok=True)
            self._ownership_file = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")
        self.last_result = None

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self._pending_input = json.dumps({"code": code}, ensure_ascii=False).encode("utf-8") + b"\n"
        self._input_offset = 0
        self._outputs = []
        self._events = []
        self._saw_eof = False
        started = False
        start_time = time.monotonic()
        startup_deadline = start_time + 10
        child_in_overtime = False
        state = None

        while state is None:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._stop_process()
                state = ("state:finished", "Cancelled", {}, [])
                break
            self._poll_stream(timeout=0.1)
            while self._events:
                event = self._events.pop(0)
                if event[0] == "state:ready":
                    started = True
                    self._owns_process_group = os.name == "posix"
                    start_time = time.monotonic()
                elif event[0] == "state:finished":
                    state = event
                    break
                else:
                    raise InterpreterError("Invalid experiment event")
            if state is not None:
                break
            now = time.monotonic()
            if not started and now > startup_deadline:
                self._stop_process()
                state = ("state:finished", "TimeoutError", {"phase": "startup"}, [])
                break
            if not self.process.is_alive():
                # Tail data may contain a complete finished event; read it
                # without ever waiting for an incomplete frame's missing body.
                self._poll_stream(timeout=0)
                state = next((event for event in self._events if event[0] == "state:finished"),
                             ("state:finished", "ProcessExited", {}, []))
                break
            if started and self.timeout is not None and now - start_time > self.timeout:
                if not child_in_overtime:
                    self._signal_owned_process(signal.SIGINT)
                    child_in_overtime = True
                if now - start_time > self.timeout + self.interrupt_grace_seconds:
                    self._stop_process()
                    state = ("state:finished", "TimeoutError", {}, [])
                    break

        exec_time = time.monotonic() - start_time
        self._pending_input = b""
        drain_deadline = time.monotonic() + 2.0
        while not self._saw_eof and not self._stream_closed and time.monotonic() < drain_deadline:
            self._poll_stream(timeout=0.05)
        output = self._outputs
        if self._stream_fragments:
            output.append("\n[Incomplete IPC output frame retained after experiment interruption]\n")
            output.append(b"".join(self._stream_fragments).decode("utf-8", errors="replace"))
            self._stream_fragments = []

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        self.last_result = ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
        if e_cls_name == "ProcessExited" or (exc_info or {}).get("phase") == "startup":
            # Infrastructure failures must not become scientific buggy nodes.
            # The worker's emergency snapshot can still retain last_result.
            raise InterpreterError(f"Experiment process failed: {e_cls_name} (startup={not started})")
        return self.last_result
