"""Bounded, cancellable subprocesses owned by one inference request.

There is deliberately no shell, global process search, shared daemon shutdown, or
unbounded ``communicate()`` here.  POSIX process groups are the unit of ownership.
"""

from __future__ import annotations

import math
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import (
    Cancelled,
    IncompleteOutput,
    InferenceTimeout,
    ProviderError,
    UnsupportedCapability,
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float


# An owned process waits for registration before executing the actual CLI. If
# its owner dies in the Popen->record window, the pipe closes and no CLI starts.
# -I -S avoids importing user Python startup customization in this tiny gate.
_EXEC_GATE = """import os, sys
gate_fd = int(sys.argv[1])
permit = os.read(gate_fd, 1)
os.close(gate_fd)
if permit != b'G':
    sys.exit(125)
try:
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
except OSError:
    sys.stderr.write('Unable to execute configured CLI\\n')
    sys.exit(127)
"""


def _process_identity(pid: int) -> dict | None:
    """Read only one exact PID's kernel start identity; never enumerate processes.

    A second-resolution ``ps lstart`` is intentionally not used: two different
    processes could have the same PID and apparent start second after reuse.
    Unknown platforms/ABIs fail closed rather than guessing ownership.
    """
    try:
        if sys.platform.startswith("linux"):
            location = Path("/proc") / str(pid)
            value = (location / "stat").read_text()
            # comm may itself contain spaces or parentheses. Everything after
            # the last ')' starts at field 3; starttime is field 22 (index 19).
            tail = value[value.rindex(")") + 2:].split()
            return {
                "kind": "linux-proc-start-v1", "pid": pid,
                "pgid": int(tail[2]), "uid": location.stat().st_uid,
                "start_ticks": int(tail[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            }
        if sys.platform == "darwin":
            # Public XNU bsd/sys/proc_info.h, struct proc_bsdinfo (136 bytes).
            # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h
            class ProcBsdInfo(ctypes.Structure):
                _fields_ = [
                    ("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
                    ("xstatus", ctypes.c_uint32), ("pid", ctypes.c_uint32),
                    ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                    ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32),
                    ("rgid", ctypes.c_uint32), ("svuid", ctypes.c_uint32),
                    ("svgid", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                    ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32),
                    ("nfiles", ctypes.c_uint32), ("pgid", ctypes.c_uint32),
                    ("jobc", ctypes.c_uint32), ("tdev", ctypes.c_uint32),
                    ("tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32),
                    ("start_seconds", ctypes.c_uint64), ("start_microseconds", ctypes.c_uint64),
                ]

            info = ProcBsdInfo()
            if ctypes.sizeof(info) != 136:
                return None
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                            ctypes.c_void_p, ctypes.c_int]
            library.proc_pidinfo.restype = ctypes.c_int
            count = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if count != ctypes.sizeof(info) or info.pid != pid:
                return None
            return {
                "kind": "darwin-proc-start-v1", "pid": int(info.pid),
                "pgid": int(info.pgid), "uid": int(info.uid),
                "start_seconds": int(info.start_seconds),
                "start_microseconds": int(info.start_microseconds),
            }
    except (OSError, ValueError, IndexError, AttributeError):
        return None
    return None


def _read_ownership(path: Path) -> dict | None:
    try:
        if path.is_symlink() or path.parent.is_symlink():
            return None
        parent = path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077 or metadata.st_size > 16384):
                return None
            record = json.loads(stream.read(16385))
        if not isinstance(record, dict) or record.get("version") != 1:
            return None
        pid, pgid = record.get("pid"), record.get("pgid")
        if type(pid) is not int or type(pgid) is not int or pid != pgid or pid <= 1:
            return None
        if record.get("uid") != os.getuid() or not isinstance(record.get("identity"), dict):
            return None
        identity = record["identity"]
        if identity.get("pid") != pid or identity.get("pgid") != pgid or identity.get("uid") != record["uid"]:
            return None
        if not isinstance(record.get("nonce"), str):
            return None
        return record
    except (OSError, ValueError, RecursionError):
        return None


def _write_ownership(path: Path, process: subprocess.Popen) -> dict:
    parent = path.parent.stat()
    if path.parent.is_symlink() or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ProviderError("Process ownership records require a private directory owned by this user")
    identity = _process_identity(process.pid)
    if identity is None or identity["pgid"] != process.pid or identity["uid"] != os.getuid():
        raise UnsupportedCapability("Cannot establish a reliable owned CLI process identity")
    record = {
        "version": 1, "pid": process.pid, "pgid": process.pid,
        "uid": os.getuid(), "owner_pid": os.getpid(),
        "identity": identity, "nonce": uuid.uuid4().hex,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _remove_ownership(path: Path, record: dict) -> None:
    # Do not remove a later process's replacement record, even if a caller errs
    # and reuses a request directory concurrently.
    current = _read_ownership(path)
    if current is not None and current.get("nonce") == record.get("nonce"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def cleanup_owned_process(record_path: Path, grace_seconds: float = 2.0) -> bool:
    """Clean a recorded orphan after its broker has stopped.

    Requires matching UID, private record, PGID, and precise kernel process-start
    identity before *each* signal. Returns False without signaling if any proof
    is unavailable. The caller must not race a live broker updating the record.

    If the group leader vanished but descendants remain, identity cannot be
    established and cleanup deliberately refuses a numeric-PGID-only kill.
    Simultaneous launcher+broker SIGKILL still requires external supervision.
    """
    if isinstance(grace_seconds, bool) or not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace_seconds must be finite and nonnegative")
    path = Path(record_path)
    record = _read_ownership(path)
    if record is None:
        return False
    pid = record["pid"]
    identity = record["identity"]
    if _process_identity(pid) != identity:
        return False
    try:
        os.killpg(record["pgid"], signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        current = _process_identity(pid)
        if current != identity:
            if current is None and not _group_exists(record["pgid"]):
                _remove_ownership(path, record)
                return True
            if current is not None:
                return False
            # A just-exited leader/group may remain visible briefly while the
            # kernel reaps it. Read-only polling is safe; never signal again
            # without the leader's exact identity.
        time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    # Repeat identity verification immediately before escalation. A stale record
    # can never authorize signaling a PID with a different kernel start time.
    current = _process_identity(pid)
    if current != identity:
        if current is not None:
            return False
        # Even grace=0 allows a bounded 50ms observation of kernel teardown;
        # this is not another termination grace and sends no signal.
        observed_until = time.monotonic() + 0.05
        while True:
            if not _group_exists(record["pgid"]):
                _remove_ownership(path, record)
                return True
            if time.monotonic() >= observed_until:
                return False
            if _process_identity(pid) is not None:
                return False
            time.sleep(0.005)
    try:
        os.killpg(record["pgid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        return False
    _remove_ownership(path, record)
    return True


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_owned_group(process: subprocess.Popen, grace: float) -> None:
    """Terminate descendants too, even if the group's original leader exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    until = time.monotonic() + grace
    while _group_exists(process.pid) and time.monotonic() < until:
        # Reap the group leader so a zombie alone doesn't consume the grace.
        process.poll()
        time.sleep(min(0.025, max(0.0, until - time.monotonic())))
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError("Owned CLI process could not be reaped after SIGKILL") from exc


def _private_log(path: Path | None):
    if path is None:
        return None
    # Refuse symlinks and existing logs: one attempt owns one set of artifacts.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb", buffering=0)


def run_process(
    argv: Sequence[str],
    *,
    input_bytes: bytes = b"",
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    cancel_event=None,
    max_output_bytes: int = 16 * 1024 * 1024,
    termination_grace_seconds: float = 2.0,
    drain_timeout_seconds: float = 1.0,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    ownership_path: Path | None = None,
) -> ProcessResult:
    """Run a single owned process, concurrently pumping all three pipes.

    The output limit covers stdout and stderr together, including bytes written
    to optional private logs. Cancellation/timeout kills only this process group.
    No raw output (which may contain private paper content) is put in exceptions.
    """
    if os.name != "posix":
        raise UnsupportedCapability("CLI process isolation currently requires POSIX")
    if not argv or not all(isinstance(arg, str) and "\x00" not in arg for arg in argv):
        raise ValueError("argv must be a nonempty sequence of NUL-free strings")
    _positive(timeout_seconds, "timeout_seconds")
    _positive(termination_grace_seconds, "termination_grace_seconds")
    _positive(drain_timeout_seconds, "drain_timeout_seconds")
    if isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("CLI request cancelled before process start")

    started = time.monotonic()
    deadline = started + timeout_seconds
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    logs = {}
    process = None
    ownership_record = None
    gate_read_fd = gate_write_fd = None
    selector = selectors.DefaultSelector()
    try:
        logs["stdout"] = _private_log(stdout_path)
        logs["stderr"] = _private_log(stderr_path)
        command = list(argv)
        pass_fds = ()
        if ownership_path is not None:
            gate_read_fd, gate_write_fd = os.pipe()
            command = [sys.executable, "-I", "-S", "-c", _EXEC_GATE,
                       str(gate_read_fd), *command]
            pass_fds = (gate_read_fd,)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=dict(env),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                close_fds=True,
                pass_fds=pass_fds,
                bufsize=0,
            )
        except OSError as exc:
            raise ProviderError("Unable to start the configured CLI executable") from exc
        if ownership_path is not None:
            os.close(gate_read_fd)
            gate_read_fd = None
            ownership_record = _write_ownership(Path(ownership_path), process)
            os.write(gate_write_fd, b"G")
            os.close(gate_write_fd)
            gate_write_fd = None
        for name in ("stdout", "stderr"):
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        if input_bytes:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
        position = 0
        output_count = 0
        exited_at = None

        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled("CLI request cancelled")
            if now >= deadline:
                raise InferenceTimeout("CLI request exceeded its execution deadline")
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = now
                if now - exited_at >= drain_timeout_seconds and selector.get_map():
                    raise IncompleteOutput("CLI exited but a descendant kept an output pipe open")
            events = selector.select(timeout=min(0.05, max(0, deadline - now)))
            for key, _ in events:
                pipe = key.fileobj
                if key.data == "stdin":
                    try:
                        sent = os.write(pipe.fileno(), input_bytes[position:position + 65536])
                    except BrokenPipeError:
                        # Preserve the CLI's useful terminal error; incomplete stdin
                        # is never enough by itself to declare inference successful.
                        selector.unregister(pipe)
                        pipe.close()
                        continue
                    except BlockingIOError:
                        continue
                    position += sent
                    if position == len(input_bytes):
                        selector.unregister(pipe)
                        pipe.close()
                    continue
                try:
                    chunk = os.read(pipe.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(pipe)
                    pipe.close()
                    continue
                output_count += len(chunk)
                if output_count > max_output_bytes:
                    raise IncompleteOutput("CLI output exceeded the configured byte limit")
                streams[key.data].extend(chunk)
                if logs[key.data] is not None:
                    logs[key.data].write(chunk)

        returncode = process.wait(timeout=1.0)
        if returncode == 0 and position != len(input_bytes):
            raise IncompleteOutput("CLI exited before consuming the complete request")
        return ProcessResult(
            returncode=returncode,
            stdout=bytes(streams["stdout"]),
            stderr=bytes(streams["stderr"]),
            duration_seconds=time.monotonic() - started,
        )
    finally:
        # Cleanup also happens on KeyboardInterrupt and parser/IO failures.
        try:
            if process is not None:
                _stop_owned_group(process, termination_grace_seconds)
                if ownership_record is not None:
                    _remove_ownership(Path(ownership_path), ownership_record)
        finally:
            for descriptor in (gate_read_fd, gate_write_fd):
                if descriptor is not None:
                    os.close(descriptor)
            selector.close()
            if process is not None:
                for name in ("stdin", "stdout", "stderr"):
                    pipe = getattr(process, name)
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            for log in logs.values():
                if log is not None:
                    log.close()
