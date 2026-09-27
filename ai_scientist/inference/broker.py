"""Run-local, process-shared scheduling for subscription CLI inference.

Only the broker process launches inference transports. Clients exchange bounded,
length-prefixed JSON over a private Unix socket, never pickle or executable RPC.
One submit connection owns one request: disconnecting cancels it rather than
silently replaying an inference whose execution status may be unknown.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import importlib
import json
import multiprocessing
import math
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

from .contracts import ROLES, InferenceRequest, InferenceResult
from .errors import (
    AuthenticationRequired,
    BrokerUnavailable,
    Cancelled,
    ContextLimitExceeded,
    InferenceError,
    QueueTimeout,
    QuotaExhausted,
)


_PROTOCOL_VERSION = 1
_MAX_FRAME_BYTES = 128 * 1024 * 1024
_POLL_SECONDS = 0.05
_PROVIDERS = frozenset(("codex_cli", "claude_cli"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_provenance() -> dict[str, Any]:
    """Read Git identity from this module's project, in the owning process only.

    No environment dump, worktree paths, diffs, credentials, or stderr is saved.
    Git overrides are not inherited, optional index writes and fsmonitor hooks
    are disabled, and each read-only command has an independent finite deadline.
    """
    provenance: dict[str, Any] = {
        "revision": None, "dirty": None, "status": "unknown", "reason": None,
        "captured_at": _now(), "scope": "module_project_root",
    }
    project_root = Path(__file__).resolve().parents[2]
    environment = {
        "PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "LANG": "C",
        "GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
    }
    prefix = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false"]

    def read(arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, *arguments], cwd=project_root, env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", timeout=2.0,
            shell=False, check=False,
        )

    try:
        revision = read(["rev-parse", "--verify", "HEAD"])
        candidate = revision.stdout.strip()
        if revision.returncode != 0:
            provenance["reason"] = "git_revision_unavailable"
            return provenance
        if len(candidate) not in (40, 64) or any(character not in "0123456789abcdef" for character in candidate):
            provenance["reason"] = "invalid_git_revision_output"
            return provenance
        provenance["revision"] = candidate
        worktree = read(["status", "--porcelain", "--untracked-files=normal"])
        if worktree.returncode != 0:
            provenance.update(status="partial", reason="git_dirty_status_unavailable")
            return provenance
        provenance.update(dirty=bool(worktree.stdout.strip()), status="known")
    except FileNotFoundError:
        provenance["reason"] = "git_unavailable"
    except subprocess.TimeoutExpired:
        provenance["reason"] = "git_timeout"
    except OSError:
        provenance["reason"] = "git_read_failed"
    if provenance["reason"] is not None and provenance["revision"] is not None:
        provenance["status"] = "partial"
    return provenance


def _atomic_json(path: Path, value: Any) -> None:
    """Replace an artifact atomically without temporarily publishing its data."""
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _send_frame(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_FRAME_BYTES:
        raise ValueError("Inference IPC frame exceeds the supported size")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_frame(connection: socket.socket, deadline: float) -> dict[str, Any]:
    def read_exactly(length: int) -> bytes:
        chunks: list[bytes] = []
        while length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Inference IPC response deadline exceeded")
            connection.settimeout(min(remaining, 0.5))
            try:
                chunk = connection.recv(min(length, 1024 * 1024))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("Inference broker connection closed")
            chunks.append(chunk)
            length -= len(chunk)
        return b"".join(chunks)

    size = struct.unpack("!I", read_exactly(4))[0]
    if size == 0 or size > _MAX_FRAME_BYTES:
        raise ValueError("Invalid inference IPC frame size")
    value = json.loads(read_exactly(size))
    if not isinstance(value, dict):
        raise ValueError("Inference IPC message must be a JSON object")
    return value


def _request_directory(root: Path, request_id: str) -> Path:
    # Neither callers nor model output can select filesystem paths.
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return root / "requests" / digest


def _error_response(error: InferenceError) -> dict[str, Any]:
    return {"ok": False, "error": error.to_dict()}


def _request_fingerprint(request: InferenceRequest) -> str:
    return _canonical_sha256(request.to_dict())


@dataclass
class _Entry:
    request: InferenceRequest
    fingerprint: str
    directory: Path
    accepted_at: str = field(default_factory=_now)
    queued_at: float = field(default_factory=time.monotonic)
    state: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    requires_validation: bool = False
    validation: dict[str, Any] | None = None


class _BrokerState:
    def __init__(
        self,
        config: dict[str, Any],
        run_id: str,
        artifact_dir: Path,
        executor: Callable[..., InferenceResult],
        source_provenance: dict[str, Any],
    ) -> None:
        self.config = config
        self.config_hash = _canonical_sha256(config)
        self.source_provenance = source_provenance
        default = config.get("default", {})
        roles = config.get("roles", {})
        self.resolved_roles = {}
        for role in ROLES:
            resolved = {**default, **roles.get(role, {})}
            self.resolved_roles[role] = {
                "backend": resolved.get("backend"), "model": resolved.get("model"),
            }
        self.run_id = run_id
        self.root = artifact_dir
        self.executor = executor
        self.condition = threading.Condition(threading.RLock())
        self.entries: dict[str, _Entry] = {}
        self.queues: dict[str, OrderedDict[str, deque[_Entry]]] = {}
        self.queued: dict[str, int] = {}
        self.threads: list[threading.Thread] = []
        self.closing = False
        self.stop_error: InferenceError | None = None
        self.run_status = "running"
        self.started_at = _now()
        self.queue_timeout = float(config.get("queue_timeout_seconds", 3600))
        self.queue_capacity = int(config.get("max_queue_size", 64))
        for backend, provider in config.get("providers", {}).items():
            if backend not in _PROVIDERS:
                continue
            self.queues[backend] = OrderedDict()
            self.queued[backend] = 0
            for index in range(int(provider.get("max_in_flight", 1))):
                self.threads.append(
                    threading.Thread(
                        target=self._worker,
                        args=(backend,),
                        name=f"inference-{backend}-{index}",
                        daemon=True,
                    )
                )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "requests").mkdir(exist_ok=True, mode=0o700)
        self._write_manifest()

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def _metadata(self, entry: _Entry) -> dict[str, Any]:
        request = entry.request
        value: dict[str, Any] = {
            "request_id": request.request_id,
            "input_sha256": entry.fingerprint,
            "backend": request.backend,
            "role": request.role,
            "state": entry.state,
            "transport_status": entry.state,
            "validation_status": (
                ("accepted" if entry.validation["accepted"] else "rejected")
                if entry.validation is not None
                else ("pending" if entry.requires_validation else "not_required")
            ),
            "accepted_at": entry.accepted_at,
            "started_at": entry.started_at,
            "finished_at": entry.finished_at,
            "artifact_dir": str(entry.directory),
        }
        value["requested_model"] = request.model
        for name in ("node_id", "stage_id", "ensemble_index", "logical_request_id", "attempt"):
            value[name] = request.metadata.get(name)
        if entry.validation is not None:
            value["validation"] = entry.validation
        if entry.response and not entry.response["ok"]:
            value["error"] = entry.response["error"]
        elif entry.response:
            result = entry.response["result"]
            value["usage"] = result.get("usage", {})
            result_metadata = result.get("metadata") or {}
            value["cli_version"] = result_metadata.get("cli_version")
            value["actual_model"] = result_metadata.get("actual_model")
            value["effective_parameters"] = result_metadata.get("effective_parameters", {})
        value["requested_parameters"] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        return value

    def _write_manifest(self) -> None:
        # Aggregate terminal results once, not repeated stream usage events.
        usage: dict[str, int | float | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
            "cache_creation_input_tokens": None,
            "reasoning_output_tokens": None,
            "subscription_billing_cost": None,
        }
        by_provider: dict[str, dict[str, Any]] = {}
        versions: dict[str, set[str]] = {backend: set() for backend in self.queues}
        unknown_versions = {backend: 0 for backend in self.queues}
        observed_successes = {backend: 0 for backend in self.queues}
        measured = 0
        for entry in self.entries.values():
            if not entry.response or not entry.response["ok"]:
                continue
            backend = entry.request.backend
            observed_successes[backend] += 1
            version = (entry.response["result"].get("metadata") or {}).get("cli_version")
            if isinstance(version, str) and version.strip() and len(version) <= 256:
                versions[backend].add(version)
            else:
                unknown_versions[backend] += 1
            item = entry.response["result"].get("usage") or {}
            if not isinstance(item, dict):
                continue
            if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in item.values()):
                measured += 1
            provider_usage = by_provider.setdefault(
                entry.request.backend, {"subscription_billing_cost": None}
            )
            if "input_includes_cached" in item:
                provider_usage["input_includes_cached"] = item["input_includes_cached"]
            for key, value in item.items():
                if key == "subscription_billing_cost" or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    usage[key] = (usage.get(key) or 0) + value
                    provider_usage[key] = (provider_usage.get(key) or 0) + value
                elif value is None and key not in provider_usage:
                    provider_usage[key] = None
        pending = [
            request_id
            for request_id, entry in self.entries.items()
            if entry.state in ("queued", "running")
        ]
        pending_validation = [
            request_id
            for request_id, entry in self.entries.items()
            if entry.requires_validation and entry.validation is None
            and entry.response and entry.response["ok"]
        ]
        _atomic_json(
            self.root / "manifest.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "resolved_roles": self.resolved_roles,
                "source_revision": self.source_provenance["revision"],
                "source_dirty": self.source_provenance["dirty"],
                "source_provenance": self.source_provenance,
                "cli_versions": {
                    backend: sorted(values) if values else None for backend, values in versions.items()
                },
                "cli_version_status": {
                    backend: "observed_in_results" if values else "unknown_not_observed"
                    for backend, values in versions.items()
                },
                "cli_version_unknown_results": unknown_versions,
                "verified_capabilities": {
                    "live_probe_status": "unknown_not_verified_by_broker",
                    "full_v2_parity": "unknown_not_verified",
                    "observed_transport_successes": observed_successes,
                    "evidence_scope": "Completed transports and host schema validation are request-level evidence only; they do not establish live image fidelity, tool isolation, or full V2 pipeline parity.",
                },
                "run_status": self.run_status,
                "run_status_scope": "inference_transport; scientific run completion is recorded separately",
                "started_at": self.started_at,
                "updated_at": _now(),
                "provider_limits": {
                    backend: int(provider.get("max_in_flight", 1))
                    for backend, provider in self.config.get("providers", {}).items()
                    if backend in _PROVIDERS
                },
                "requests": {
                    request_id: self._metadata(entry)
                    for request_id, entry in self.entries.items()
                },
                "pending_request_ids": pending,
                "pending_validation_request_ids": pending_validation,
                "validation_summary": {
                    "accepted": sum(entry.validation is not None and entry.validation["accepted"] for entry in self.entries.values()),
                    "rejected": sum(entry.validation is not None and not entry.validation["accepted"] for entry in self.entries.values()),
                    "pending": len(pending_validation),
                },
                "usage_summary": usage,
                "usage_by_provider": by_provider,
                "usage_summary_semantics": "Sum of provider-reported fields; cached tokens are not added to input_tokens. See usage_by_provider for inclusion conventions.",
                "requests_with_measured_usage": measured,
                "requests_without_measured_usage": sum(
                    entry.response is not None for entry in self.entries.values()
                ) - measured,
                "stop_reason": self.stop_error.to_dict() if self.stop_error else None,
            },
        )

    def _finish(
        self, entry: _Entry, *, result: InferenceResult | None = None,
        error: InferenceError | None = None,
    ) -> None:
        if entry.done.is_set():
            return
        if entry.state == "queued":
            self.queued[entry.request.backend] -= 1
            # Remove cancelled/expired entries rather than accumulating queue
            # tombstones behind a long-running request.
            roles = self.queues[entry.request.backend]
            queued_role = roles.get(entry.request.role)
            if queued_role is not None:
                remaining = deque(candidate for candidate in queued_role if candidate is not entry)
                if remaining:
                    roles[entry.request.role] = remaining
                else:
                    del roles[entry.request.role]
        if error is not None:
            entry.state = "cancelled" if isinstance(error, Cancelled) else "failed"
            entry.response = _error_response(error)
        else:
            assert result is not None
            entry.state = "completed"
            entry.response = {"ok": True, "result": result.to_dict()}
        entry.finished_at = _now()
        try:
            if error is not None:
                _atomic_json(entry.directory / "error.json", error.to_dict())
            else:
                _atomic_json(entry.directory / "result.json", result.to_dict())
            _atomic_json(entry.directory / "request_metadata.json", self._metadata(entry))
            self._write_manifest()
        except (OSError, ValueError, TypeError) as exception:
            # A full/broken artifact volume must fail explicitly, not leave a
            # client waiting forever or report an unrecorded success.
            storage_error = BrokerUnavailable(
                "Cannot persist inference result artifacts",
                request_id=entry.request.request_id,
                metadata={"exception_type": type(exception).__name__, "phase": "artifacts"},
            )
            entry.state = "failed"
            entry.response = _error_response(storage_error)
            self.stop_error = storage_error
            self.run_status = "failed_transport"
            for pending in self.entries.values():
                if pending is entry or pending.done.is_set():
                    continue
                pending.cancel.set()
                if pending.state == "queued":
                    self.queued[pending.request.backend] -= 1
                    pending.state = "failed"
                    pending.finished_at = _now()
                    pending_error = InferenceError.from_dict(storage_error.to_dict())
                    pending_error.request_id = pending.request.request_id
                    pending.response = _error_response(pending_error)
                    pending.done.set()
            for queue in self.queues.values():
                queue.clear()
            try:
                _atomic_json(entry.directory / "error.json", storage_error.to_dict())
                _atomic_json(entry.directory / "request_metadata.json", self._metadata(entry))
                self._write_manifest()
            except (OSError, ValueError, TypeError):
                pass
        # Completed history can be very large (paper text and base64 images).
        # Keep identity/usage for delivery deduplication, not all prompt copies.
        entry.request.messages = []
        entry.request.system = ""
        entry.request.schema = None
        entry.done.set()
        self.condition.notify_all()

    def submit(self, request: InferenceRequest) -> _Entry:
        with self.condition:
            if request.run_id != self.run_id:
                raise BrokerUnavailable("Request belongs to a different run")
            if (
                not isinstance(request.request_id, str)
                or not 1 <= len(request.request_id) <= 256
                or not isinstance(request.role, str)
                or not request.role
                or request.backend not in self.queues
            ):
                raise BrokerUnavailable("Invalid request ID or unconfigured CLI provider")
            fingerprint = _request_fingerprint(request)
            if len(json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")) > int(
                self.config.get("max_payload_bytes", 32 * 1024 * 1024)
            ):
                raise ContextLimitExceeded(
                    "Inference request exceeds configured IPC size limit; input was not truncated",
                    request_id=request.request_id,
                    metadata={"executed": False},
                )
            previous = self.entries.get(request.request_id)
            if previous:
                if previous.fingerprint != fingerprint:
                    raise BrokerUnavailable(
                        "Request ID was already used for different input",
                        request_id=request.request_id,
                    )
                if not previous.done.is_set():
                    raise BrokerUnavailable(
                        "Request ID is already in flight; it was not replayed",
                        request_id=request.request_id,
                    )
                return previous
            if self.stop_error is not None:
                stopped = InferenceError.from_dict(self.stop_error.to_dict())
                stopped.metadata["blocked_by_request_id"] = stopped.request_id
                stopped.request_id = request.request_id
                raise stopped
            if self.closing:
                raise Cancelled("Inference broker is closing", request_id=request.request_id)
            if self.queued[request.backend] >= self.queue_capacity:
                raise QueueTimeout(
                    "Inference queue is full; request was not executed",
                    request_id=request.request_id,
                    metadata={"phase": "queue", "executed": False},
                )
            directory = _request_directory(self.root, request.request_id)
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            entry = _Entry(request, fingerprint, directory, requires_validation=request.schema is not None)
            self.entries[request.request_id] = entry
            self.queued[request.backend] += 1
            self.queues[request.backend].setdefault(request.role, deque()).append(entry)
            try:
                _atomic_json(directory / "request_metadata.json", self._metadata(entry))
                self._write_manifest()
            except (OSError, ValueError, TypeError) as exception:
                error = BrokerUnavailable(
                    "Cannot persist queued inference request",
                    request_id=request.request_id,
                    metadata={"exception_type": type(exception).__name__, "executed": False},
                )
                self._finish(entry, error=error)
                raise error from exception
            self.condition.notify_all()
            return entry

    def cancel(self, request_id: str, reason: str = "Request cancelled") -> bool:
        with self.condition:
            entry = self.entries.get(request_id)
            if entry is None or entry.done.is_set():
                return False
            entry.cancel.set()
            if entry.state == "queued":
                self._finish(entry, error=Cancelled(reason, request_id=request_id))
            self.condition.notify_all()
            return True

    def cancel_all(self, reason: str = "Run cancelled") -> None:
        with self.condition:
            if self.stop_error is None:
                self.stop_error = Cancelled(reason)
                self.run_status = "cancelled"
            for entry in list(self.entries.values()):
                self.cancel(entry.request.request_id, reason)
            self._write_manifest()

    def record_validation(
        self, request_id: str, *, accepted: bool, error: dict[str, Any] | None = None,
    ) -> None:
        with self.condition:
            entry = self.entries.get(request_id)
            if entry is None or not entry.done.is_set() or not entry.response or not entry.response["ok"]:
                raise BrokerUnavailable("Validation requires a completed transport result", request_id=request_id)
            if type(accepted) is not bool or (error is not None and not isinstance(error, dict)):
                raise BrokerUnavailable("Invalid inference validation result", request_id=request_id)
            if entry.validation is not None:
                if entry.validation["accepted"] == accepted and entry.validation["error"] == error:
                    return
                raise BrokerUnavailable("Request validation was already finalized", request_id=request_id)
            entry.validation = {
                "request_id": request_id, "accepted": accepted, "error": error, "validated_at": _now(),
            }
            _atomic_json(entry.directory / "validation.json", entry.validation)
            _atomic_json(entry.directory / "request_metadata.json", self._metadata(entry))
            self._write_manifest()

    def expire(self, entry: _Entry) -> None:
        with self.condition:
            if entry.state == "queued" and time.monotonic() - entry.queued_at >= self.queue_timeout:
                self._finish(
                    entry,
                    error=QueueTimeout(
                        "Inference queue deadline exceeded; request was not executed",
                        request_id=entry.request.request_id,
                        metadata={"phase": "queue", "executed": False},
                    ),
                )

    def shutdown(self, cancel: bool) -> None:
        with self.condition:
            self.closing = True
            if cancel and any(not entry.done.is_set() for entry in self.entries.values()):
                self.cancel_all("Inference broker closed")
            self.condition.notify_all()

    def finish_shutdown(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            self.cancel_all("Inference broker shutdown deadline reached")
            for thread in self.threads:
                thread.join(max(0, deadline + 5 - time.monotonic()))
        with self.condition:
            if any(not entry.done.is_set() for entry in self.entries.values()):
                self.run_status = "failed_transport"
            elif self.run_status == "running":
                self.run_status = "completed"
            self._write_manifest()

    def _next(self, backend: str) -> _Entry | None:
        queue = self.queues[backend]
        while queue:
            role, requests = queue.popitem(last=False)
            entry = requests.popleft()
            if requests:
                queue[role] = requests
            if entry.done.is_set():
                continue
            self.expire(entry)
            if not entry.done.is_set():
                return entry
        return None

    def _worker(self, backend: str) -> None:
        while True:
            with self.condition:
                entry = self._next(backend)
                while entry is None:
                    if self.closing:
                        return
                    self.condition.wait(_POLL_SECONDS)
                    entry = self._next(backend)
                self.queued[backend] -= 1
                entry.state = "running"
                entry.started_at = _now()
            try:
                with self.condition:
                    _atomic_json(entry.directory / "request_metadata.json", self._metadata(entry))
                    self._write_manifest()
                provider = dict(self.config["providers"][backend])
                provider.setdefault(
                    "request_timeout_seconds", self.config.get("request_timeout_seconds", 600)
                )
                result = self.executor(entry.request, provider, entry.directory, entry.cancel)
                if not isinstance(result, InferenceResult) or result.request_id != entry.request.request_id:
                    raise BrokerUnavailable("Inference transport returned an invalid result")
                with self.condition:
                    if entry.cancel.is_set():
                        self._finish(
                            entry, error=Cancelled("Request cancelled", request_id=entry.request.request_id)
                        )
                    else:
                        self._finish(entry, result=result)
            except BaseException as exception:
                error = exception if isinstance(exception, InferenceError) else BrokerUnavailable(
                    "Inference transport failed unexpectedly",
                    request_id=entry.request.request_id,
                    metadata={"exception_type": type(exception).__name__},
                )
                with self.condition:
                    if isinstance(error, (AuthenticationRequired, QuotaExhausted)):
                        self.stop_error = error
                        self.run_status = (
                            "stopped_auth" if isinstance(error, AuthenticationRequired) else "stopped_quota"
                        )
                    self._finish(entry, error=error)
                    if isinstance(error, (AuthenticationRequired, QuotaExhausted)):
                        for pending in list(self.entries.values()):
                            if pending.done.is_set():
                                continue
                            pending.cancel.set()
                            if pending.state == "queued":
                                pending_error = InferenceError.from_dict(error.to_dict())
                                pending_error.request_id = pending.request.request_id
                                pending_error.metadata["blocked_by_request_id"] = entry.request.request_id
                                self._finish(pending, error=pending_error)
                        self._write_manifest()


def _resolve_executor(reference: str | None) -> Callable[..., InferenceResult]:
    if reference is None:
        from .cli import execute

        return execute
    module_name, separator, name = reference.partition(":")
    if not separator or "<locals>" in name:
        raise ValueError("The internal executor seam must identify an importable callable")
    value: Any = importlib.import_module(module_name)
    for component in name.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError("The internal executor seam is not callable")
    return value


def _serve(
    config: dict[str, Any], run_id: str, artifact_dir: str,
    endpoint: dict[str, Any], executor_reference: str | None, owner_pid: int,
    source_provenance: dict[str, Any],
) -> None:
    # All files produced by the broker and its CLI descendants are private.
    os.umask(0o077)
    state = _BrokerState(
        config, run_id, Path(artifact_dir), _resolve_executor(executor_reference), source_provenance,
    )
    stopping = threading.Event()
    handlers: set[threading.Thread] = set()
    handlers_lock = threading.Lock()
    capacity = max(8, len(state.queues) * state.queue_capacity + len(state.threads) + 16)
    slots = threading.BoundedSemaphore(capacity)

    def interrupted(_signum: int, _frame: Any) -> None:
        # Avoid locks inside signal handlers; the main loop performs cancellation.
        stopping.set()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def handle(connection: socket.socket) -> None:
        entry: _Entry | None = None
        try:
            packet = _receive_frame(connection, time.monotonic() + 10)
            if not isinstance(packet.get("token"), str) or not hmac.compare_digest(
                packet["token"], endpoint["token"]
            ):
                raise BrokerUnavailable("Invalid inference broker authentication")
            operation = packet.get("op")
            if operation == "submit":
                entry = state.submit(InferenceRequest.from_dict(packet["request"]))
                while not entry.done.wait(_POLL_SECONDS):
                    state.expire(entry)
                    readable, _, _ = select.select([connection], [], [], 0)
                    if readable:
                        # No second packet is permitted on a submit connection.
                        state.cancel(entry.request.request_id, "Inference client disconnected")
                        return
                assert entry.response is not None
                _send_frame(connection, entry.response)
            elif operation == "cancel":
                _send_frame(connection, {"ok": True, "cancelled": state.cancel(str(packet["request_id"]))})
            elif operation == "cancel_all":
                state.cancel_all(str(packet.get("reason", "Run cancelled"))[:1000])
                _send_frame(connection, {"ok": True})
            elif operation == "record_validation":
                state.record_validation(
                    str(packet["request_id"]), accepted=packet["accepted"], error=packet.get("error"),
                )
                _send_frame(connection, {"ok": True})
            elif operation == "shutdown":
                state.shutdown(bool(packet.get("cancel", True)))
                _send_frame(connection, {"ok": True})
                stopping.set()
            elif operation == "ping":
                _send_frame(connection, {"ok": True, "run_id": run_id})
            else:
                raise BrokerUnavailable("Unknown inference broker operation")
        except (OSError, ConnectionError, TimeoutError):
            if entry is not None:
                state.cancel(entry.request.request_id, "Inference client disconnected")
        except Exception as exception:
            error = exception if isinstance(exception, InferenceError) else BrokerUnavailable(
                "Invalid inference broker request", metadata={"exception_type": type(exception).__name__}
            )
            try:
                _send_frame(connection, _error_response(error))
            except OSError:
                pass
        finally:
            connection.close()
            with handlers_lock:
                handlers.discard(threading.current_thread())
            slots.release()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(endpoint["socket_path"])
        os.chmod(endpoint["socket_path"], 0o600)
        listener.listen(capacity)
        listener.settimeout(0.2)
        state.start()
        while not stopping.is_set():
            if os.getppid() != owner_pid:
                # A killed launcher cannot run its atexit hook. Do not leave a
                # broker or its inference descendants serving an orphaned run.
                stopping.set()
                break
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            if not slots.acquire(blocking=False):
                connection.close()
                continue
            thread = threading.Thread(target=handle, args=(connection,), daemon=True)
            with handlers_lock:
                handlers.add(thread)
            thread.start()
    finally:
        listener.close()
        if not state.closing:
            state.shutdown(cancel=True)
        state.finish_shutdown(float(config.get("shutdown_timeout_seconds", 10)))
        deadline = time.monotonic() + 1
        with handlers_lock:
            remaining = list(handlers)
        for thread in remaining:
            thread.join(max(0, deadline - time.monotonic()))
        Path(endpoint["socket_path"]).unlink(missing_ok=True)


class BrokerClient:
    """A serializable endpoint handle; connections are fresh per operation."""

    def __init__(self, endpoint: dict[str, Any]) -> None:
        if endpoint.get("protocol_version") != _PROTOCOL_VERSION:
            raise BrokerUnavailable("Unsupported inference broker protocol")
        self.endpoint = dict(endpoint)

    def _call(self, operation: str, *, timeout: float = 10, **payload: Any) -> dict[str, Any]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(min(timeout, 10))
            connection.connect(self.endpoint["socket_path"])
            _send_frame(connection, {"op": operation, "token": self.endpoint["token"], **payload})
            response = _receive_frame(connection, time.monotonic() + timeout)
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as exception:
            raise BrokerUnavailable(
                "Inference broker connection failed; execution state may be unknown and was not retried",
                metadata={"exception_type": type(exception).__name__, "replayed": False},
            ) from exception
        finally:
            connection.close()
        if not response.get("ok"):
            raise InferenceError.from_dict(response.get("error", {}))
        return response

    def submit(self, request: InferenceRequest) -> InferenceResult:
        response = self._call(
            "submit",
            timeout=max(
                float(self.endpoint.get("response_timeout_seconds", 4215)),
                float(self.endpoint.get("queue_timeout_seconds", 3600)) + request.timeout_seconds + 20,
            ),
            request=request.to_dict(),
        )
        return InferenceResult.from_dict(response["result"])

    def cancel(self, request_id: str) -> bool:
        return bool(self._call("cancel", request_id=request_id).get("cancelled"))

    def cancel_all(self, reason: str = "Run cancelled") -> None:
        self._call("cancel_all", reason=reason)

    def record_validation(
        self, request_id: str, *, accepted: bool, error: dict[str, Any] | None = None,
    ) -> None:
        self._call("record_validation", request_id=request_id, accepted=accepted, error=error)


class Broker:
    """Owner of one run's broker process; never pass this object to workers.

    ``_executor`` is an internal, importable-callable test seam. It is not read
    from configuration, RPC requests, model output, or environment variables.
    """

    def __init__(
        self, config: dict[str, Any], run_id: str, artifact_dir: str | Path,
        *, _executor: str | Callable[..., InferenceResult] | None = None,
    ) -> None:
        if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
            raise BrokerUnavailable("The CLI inference broker requires Linux or macOS")
        self.config = json.loads(json.dumps(config, allow_nan=False))
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if int(self.config.get("max_queue_size", 64)) < 1:
            raise ValueError("max_queue_size must be positive")
        for name, default in (
            ("queue_timeout_seconds", 3600), ("request_timeout_seconds", 600),
            ("shutdown_timeout_seconds", 10),
        ):
            value = float(self.config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        providers = self.config.get("providers", {})
        if not any(backend in _PROVIDERS for backend in providers):
            raise ValueError("At least one CLI provider is required")
        for backend, provider in providers.items():
            if backend in _PROVIDERS and int(provider.get("max_in_flight", 1)) < 1:
                raise ValueError("Provider max_in_flight must be positive")
        self.run_id = run_id
        self.artifact_dir = Path(artifact_dir).resolve()
        self.manifest_path = self.artifact_dir / "manifest.json"
        self._owner_pid = os.getpid()
        self._process: multiprocessing.Process | None = None
        self._endpoint: dict[str, Any] | None = None
        self._private_dir: Path | None = None
        self._artifact_identity: tuple[int, int] | None = None
        self._closed = False
        self._executor = (
            f"{_executor.__module__}:{_executor.__qualname__}"
            if callable(_executor) else _executor
        )

    @property
    def endpoint(self) -> dict[str, Any]:
        if self._endpoint is None:
            raise BrokerUnavailable("Inference broker has not been started")
        return dict(self._endpoint)

    def start(self) -> Broker:
        if os.getpid() != self._owner_pid:
            raise BrokerUnavailable("Only the owning process can start an inference broker")
        if self._closed:
            raise BrokerUnavailable("A closed inference broker cannot be restarted")
        if self._process is not None:
            if self._process.is_alive():
                return self
            raise BrokerUnavailable("Inference broker process exited")
        self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.manifest_path.exists():
            raise BrokerUnavailable(
                "Inference artifact directory already contains a run; use a fresh directory"
            )
        artifact_stat = self.artifact_dir.stat()
        self._artifact_identity = (artifact_stat.st_dev, artifact_stat.st_ino)
        source_provenance = _source_provenance()
        # macOS has a particularly short sockaddr_un path limit. Do not put the
        # socket under a potentially deep run directory or the default TMPDIR.
        self._private_dir = Path(tempfile.mkdtemp(prefix="ais-broker-", dir="/tmp"))
        os.chmod(self._private_dir, 0o700)
        self._endpoint = {
            "protocol_version": _PROTOCOL_VERSION,
            "socket_path": str(self._private_dir / "ipc.sock"),
            "token": secrets.token_urlsafe(32),
            "run_id": self.run_id,
            "queue_timeout_seconds": float(self.config.get("queue_timeout_seconds", 3600)),
            "response_timeout_seconds": float(self.config.get("queue_timeout_seconds", 3600))
            + float(self.config.get("request_timeout_seconds", 600)) + 20,
        }
        self._process = multiprocessing.get_context("spawn").Process(
            target=_serve,
            args=(self.config, self.run_id, str(self.artifact_dir), self.endpoint, self._executor, self._owner_pid, source_provenance),
            name=f"inference-broker-{self.run_id[:20]}",
        )
        atexit.register(self.close)
        deadline = time.monotonic() + 15
        try:
            self._process.start()
            while time.monotonic() < deadline:
                if not self._process.is_alive():
                    raise BrokerUnavailable("Inference broker failed to start")
                if Path(self.endpoint["socket_path"]).exists():
                    BrokerClient(self.endpoint)._call("ping", timeout=1)
                    return self
                time.sleep(_POLL_SECONDS)
            raise BrokerUnavailable("Inference broker startup deadline exceeded")
        except BaseException as exception:
            self.close()
            if isinstance(exception, Exception) and not isinstance(exception, InferenceError):
                raise BrokerUnavailable(
                    "Unable to start the inference broker",
                    metadata={"exception_type": type(exception).__name__},
                ) from exception
            raise

    def _record_abnormal_exit(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            pending = manifest.get("pending_request_ids", [])
            if manifest.get("run_status") != "running" and not pending:
                return
            error = BrokerUnavailable(
                "Inference broker exited before finalizing requests; execution status is unknown",
                metadata={"replayed": False},
            ).to_dict()
            manifest["run_status"] = "failed_transport"
            manifest["stop_reason"] = error
            manifest["updated_at"] = _now()
            for request_id in pending:
                item = manifest.get("requests", {}).get(request_id)
                if item:
                    item["state"] = "unknown"
                    item["transport_status"] = "unknown"
                    item["error"] = error
            _atomic_json(self.manifest_path, manifest)
        except (OSError, ValueError, TypeError):
            # Never conceal the original shutdown/startup failure with a second
            # error while attempting to preserve diagnostics.
            pass

    def cleanup_owned_processes(self) -> int:
        """Clean verified CLI groups after broker loss, while its owner survives.

        This is not a system process scan. The transport records exact process
        creation identity in each private request directory. Stale, symlinked,
        replaced, or unverifiable records are never authority to signal a PID.
        A simultaneous SIGKILL of both launcher and broker still requires an
        external supervisor; this method does not claim crash-proof recovery.
        """
        process = self._process
        if (
            os.getpid() != self._owner_pid or process is None or process.pid is None
            or process.is_alive() or self._artifact_identity is None
        ):
            return 0
        try:
            root_stat = self.artifact_dir.stat()
            if self.artifact_dir.is_symlink() or (root_stat.st_dev, root_stat.st_ino) != self._artifact_identity:
                return 0
            requests = self.artifact_dir / "requests"
            if requests.is_symlink() or not requests.is_dir():
                return 0
            if self.manifest_path.is_symlink():
                return 0
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            accepted = manifest.get("requests", {})
            if manifest.get("run_id") != self.run_id or not isinstance(accepted, dict):
                return 0
            owned_directories = {
                hashlib.sha256(request_id.encode("utf-8")).hexdigest()
                for request_id in accepted if isinstance(request_id, str)
            }
            # One shared grace budget prevents many groups from each consuming
            # the full shutdown delay. Identity checks are independently bounded
            # by the transport's helper.
            deadline = time.monotonic() + 10
            terminated = 0
            for directory in requests.iterdir():
                if (
                    len(directory.name) != 64
                    or any(character not in "0123456789abcdef" for character in directory.name)
                    or directory.name not in owned_directories
                    or directory.is_symlink() or not directory.is_dir()
                ):
                    continue
                record = directory / "process.json"
                if record.is_symlink() or not record.is_file():
                    continue
                from .process import cleanup_owned_process

                try:
                    terminated += bool(cleanup_owned_process(
                        record, grace_seconds=min(2.0, max(0.0, deadline - time.monotonic())),
                    ))
                except (OSError, ValueError, InferenceError):
                    # Exact ownership could not be established. Never guess at
                    # another process or expand the filesystem/PID search scope.
                    continue
            return terminated
        except (OSError, ValueError, TypeError):
            return 0

    def close(self, cancel: bool = True) -> None:
        if self._closed or os.getpid() != self._owner_pid:
            return
        self._closed = True
        atexit.unregister(self.close)
        process = self._process
        if process is not None and process.pid is not None:
            if process.is_alive() and self._endpoint is not None:
                try:
                    BrokerClient(self.endpoint)._call("shutdown", cancel=cancel, timeout=2)
                except InferenceError:
                    pass
            process.join(float(self.config.get("shutdown_timeout_seconds", 10)) + 7)
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(2)
            self.cleanup_owned_processes()
            self._record_abnormal_exit()
        if self._private_dir is not None:
            (self._private_dir / "ipc.sock").unlink(missing_ok=True)
            try:
                self._private_dir.rmdir()
            except OSError:
                pass

    def __enter__(self) -> Broker:
        return self.start()

    def __exit__(self, *_exception: Any) -> None:
        self.close()
