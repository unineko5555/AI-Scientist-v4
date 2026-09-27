"""Persistent research workers with durable, finite result collection.

Only the small Process startup arguments use multiprocessing serialization.
Tasks/results use private atomic files, so killing a worker halfway through a
large Node result cannot strand a ProcessPoolExecutor result-reader thread.
The Future interface preserves the caller's existing batch and journal order.
"""

from __future__ import annotations

from concurrent.futures import Future
import importlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
import traceback
import uuid

from omegaconf import OmegaConf

from ai_scientist.inference.errors import Cancelled, ConfigurationError, InferenceError
from ai_scientist.inference.process import _process_identity


_IDEAS = {
    "ai_scientist.treesearch.parallel_agent.AblationIdea",
    "ai_scientist.treesearch.parallel_agent.HyperparamTuningIdea",
}


class CancellationFlag:
    """Process-shared cancellation without a kill-sensitive semaphore lock."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)

    def is_set(self):
        return self.path.exists()

    def set(self):
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(descriptor)


def encode_value(value):
    """Lossless allowlisted codec; never stringify or silently drop fields."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if OmegaConf.is_config(value):
        return {"type": "omegaconf", "value": encode_value(OmegaConf.to_container(value, resolve=True))}
    if type(value) in (tuple, list):
        return {"type": "tuple" if type(value) is tuple else "list", "value": [encode_value(item) for item in value]}
    if type(value) is dict:
        return {"type": "dict", "value": [[encode_value(key), encode_value(item)] for key, item in value.items()]}
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    if name in _IDEAS:
        if set(vars(value)) != {"name", "description"}:
            raise ConfigurationError(f"Unknown fields in worker idea {name}")
        return {"type": "idea", "class": name, "value": encode_value(vars(value))}
    raise ConfigurationError(f"Unsupported worker payload type: {name}")


def decode_value(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if not isinstance(value, dict) or set(value) not in ({"type", "value"}, {"type", "class", "value"}):
        raise ConfigurationError("Invalid worker payload envelope")
    kind, body = value["type"], value["value"]
    if kind != "idea" and "class" in value:
        raise ConfigurationError("Unexpected worker payload fields")
    try:
        if kind == "float" and body in ("inf", "-inf", "nan"):
            return float(body)
        if kind == "path" and isinstance(body, str):
            return Path(body)
        if kind in ("list", "tuple") and isinstance(body, list):
            items = [decode_value(item) for item in body]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict" and isinstance(body, list):
            pairs = [(decode_value(key), decode_value(item)) for key, item in body]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ConfigurationError("Duplicate keys in worker payload")
            return result
        if kind == "omegaconf":
            return OmegaConf.create(decode_value(body), flags={"allow_objects": True})
        if kind == "idea" and value.get("class") in _IDEAS:
            fields = decode_value(body)
            if not isinstance(fields, dict) or set(fields) != {"name", "description"}:
                raise ConfigurationError("Unknown fields in worker idea")
            module, name = value["class"].rsplit(".", 1)
            return getattr(importlib.import_module(module), name)(**fields)
    except InferenceError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigurationError("Malformed worker payload") from error
    raise ConfigurationError(f"Unsupported worker payload tag: {kind}")


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise ConfigurationError("Worker artifact is not an owned private regular file")
        return json.load(handle)


def _error_payload(error):
    if isinstance(error, InferenceError):
        return encode_value(error.to_dict())
    return encode_value(InferenceError(
        f"Research worker failed: {type(error).__name__}: {error}",
        metadata={"exception_type": type(error).__name__, "traceback": traceback.format_exc()},
    ).to_dict())


def _worker_main(directory, function, initializer, initargs, stop):
    directory = Path(directory)
    try:
        if initializer is not None:
            initializer(*initargs)
        while not stop.is_set():
            pending = sorted(directory.glob("*.pending.json"))
            if not pending:
                # multiprocessing.Event/Condition can itself deadlock when a
                # waiter is SIGKILLed while owning its semaphore. No shared
                # lock appears anywhere on this process-control path.
                time.sleep(0.05)
                continue
            for path in pending:
                if stop.is_set():
                    return
                job_id = path.name.removesuffix(".pending.json")
                running = directory / f"{job_id}.running.{os.getpid()}.json"
                try:
                    path.rename(running)
                except FileNotFoundError:
                    continue  # Another persistent worker claimed this job.
                try:
                    payload = _read_json(running)
                    if set(payload) != {"args", "kwargs"}:
                        raise ConfigurationError("Invalid worker task fields")
                    if stop.is_set():
                        raise Cancelled("Research worker stopped before task execution")
                    result = function(*decode_value(payload["args"]), **decode_value(payload["kwargs"]))
                    response = {"status": "completed", "value": encode_value(result)}
                except BaseException as error:
                    response = {"status": "failed", "error": _error_payload(error)}
                # A committed result always precedes Future completion. Keep
                # input/claim files as diagnostics after interrupted runs.
                _atomic_json(directory / f"{job_id}.result.json", response)
    except BaseException as error:
        _atomic_json(directory / f"worker_{os.getpid()}.error.json", {"error": _error_payload(error)})
        raise


class OwnedProcessPool:
    """Small fixed-function executor using only public Process lifecycle APIs."""

    def __init__(self, *, max_workers, mp_context, directory, function, initializer=None, initargs=()):
        if max_workers < 1:
            raise ConfigurationError("Research worker count must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.function = function
        self._lock = threading.Lock()
        self._jobs = {}
        self._shutdown = False
        self._pool_failure = None
        self._stop = CancellationFlag(self.directory / "stop.flag")
        self._monitor_stop = threading.Event()
        self._processes = []  # Our Process handles, not executor internals.
        self._identities = {}
        pool_id = uuid.uuid4().hex
        try:
            for index in range(max_workers):
                process = mp_context.Process(
                    target=_worker_main,
                    args=(str(self.directory), function, initializer, initargs, self._stop),
                    # A new stage/pool must not inherit another stage's working
                    # .npy files. Reuse is limited to this persistent pool.
                    name=f"research_worker_{pool_id}_{index}",
                )
                process.start()
                self._processes.append(process)
                identity = _process_identity(process.pid)
                if identity is None:
                    raise RuntimeError("Cannot establish owned research worker identity")
                self._identities[process.pid] = identity
        except BaseException:
            self._stop.set()
            self._stop_workers()
            raise
        self._monitor = threading.Thread(target=self._monitor_results, name="research-result-monitor", daemon=True)
        self._monitor.start()

    @property
    def worker_pids(self):
        return tuple(process.pid for process in self._processes)

    @property
    def failure(self):
        """Pool-wide failure, including an idle worker's unexpected exit."""
        return self._pool_failure

    def submit(self, function, /, *args, **kwargs):
        if function != self.function:
            raise ConfigurationError("This research pool only accepts its configured worker function")
        # Encode synchronously: invalid/unknown fields fail before submission
        # and before the caller relinquishes its GPU assignment.
        payload = {"args": encode_value(args), "kwargs": encode_value(kwargs)}
        with self._lock:
            if self._shutdown or self._pool_failure is not None:
                raise Cancelled("Research worker pool is stopped")
            job_id = f"{len(self._jobs):08d}_{uuid.uuid4().hex}"
            future = Future()
            future.set_running_or_notify_cancel()
            self._jobs[job_id] = future
            try:
                _atomic_json(self.directory / f"{job_id}.pending.json", payload)
            except BaseException:
                self._jobs.pop(job_id)
                raise
        return future

    def _drain_results(self):
        with self._lock:
            jobs = list(self._jobs.items())
        for job_id, future in jobs:
            path = self.directory / f"{job_id}.result.json"
            if future.done() or not path.exists():
                continue
            try:
                payload = _read_json(path)
                if payload.get("status") == "completed" and set(payload) == {"status", "value"}:
                    value, error = decode_value(payload["value"]), None
                elif payload.get("status") == "failed" and set(payload) == {"status", "error"}:
                    value, error = None, InferenceError.from_dict(decode_value(payload["error"]))
                else:
                    raise ConfigurationError("Invalid research result fields")
            except BaseException as error:
                future.set_exception(error)
            else:
                if error is None:
                    future.set_result(value)
                else:
                    future.set_exception(error)

    def _observe_exits(self):
        dead = {process.pid for process in self._processes if process.exitcode is not None}
        if not dead:
            return
        if self._pool_failure is None and not self._shutdown:
            self._pool_failure = InferenceError("Owned research worker exited unexpectedly", metadata={"worker_pids": sorted(dead)})
            self._stop.set()
        # Read committed results before declaring any assigned work lost.
        self._drain_results()
        with self._lock:
            jobs = list(self._jobs.items())
        for job_id, future in jobs:
            if future.done():
                continue
            claims = list(self.directory.glob(f"{job_id}.running.*.json"))
            if claims and int(claims[0].name.split(".")[-2]) not in dead:
                continue
            future.set_exception(self._pool_failure or Cancelled("Research worker stopped; partial artifacts retained"))

    def _monitor_results(self):
        try:
            while not self._monitor_stop.wait(0.05):
                self._drain_results()
                self._observe_exits()
            self._drain_results()
            self._observe_exits()
        except BaseException as error:
            self._pool_failure = error
            self._stop.set()
            with self._lock:
                futures = list(self._jobs.values())
            for future in futures:
                if not future.done():
                    future.set_exception(error)

    def _stop_workers(self):
        deadline = time.monotonic() + 2
        for process in self._processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for operation in ("terminate", "kill"):
            for process in self._processes:
                if process.is_alive():
                    if self._identities.get(process.pid) != _process_identity(process.pid):
                        raise RuntimeError("Cannot prove owned worker identity; GPU remains reserved")
                    getattr(process, operation)()
            deadline = time.monotonic() + 2
            for process in self._processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in self._processes):
            raise RuntimeError("Owned research worker did not stop; GPU remains reserved")

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Always finite and fully reaped. wait=False must not leave an atexit
        # join waiting forever on a ProcessPool result pipe or a live worker.
        with self._lock:
            self._shutdown = True
        self._stop.set()
        self._stop_workers()
        self._monitor_stop.set()
        self._monitor.join(timeout=3)
        if self._monitor.is_alive():
            raise RuntimeError("Research result monitor did not stop; artifacts retained")
        self._drain_results()
        self._observe_exits()
