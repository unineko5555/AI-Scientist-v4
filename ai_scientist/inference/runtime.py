"""One run configuration and one CLI broker, shared with spawned V2 workers."""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from pathlib import Path

from .config import InferenceConfig, config_from_args
from .contracts import CLI_BACKENDS, ROLES, InferenceClient
from .errors import BrokerUnavailable, ConfigurationError, InferenceError

ENV_CONFIG = "AI_SCIENTIST_INFERENCE_CONFIG"
ENV_ENDPOINT = "AI_SCIENTIST_INFERENCE_ENDPOINT"
ENV_RUN_ID = "AI_SCIENTIST_INFERENCE_RUN_ID"
ENV_ARTIFACT_DIR = "AI_SCIENTIST_INFERENCE_ARTIFACT_DIR"
ENV_OWNER_PID = "AI_SCIENTIST_INFERENCE_OWNER_PID"
_ENV_KEYS = (ENV_CONFIG, ENV_ENDPOINT, ENV_RUN_ID, ENV_ARTIFACT_DIR, ENV_OWNER_PID)
_runtime = None
_lock = threading.RLock()


class InferenceRuntime:
    def __init__(self, config: InferenceConfig, *, artifact_dir=None, run_id=None,
                 endpoint=None, owner_pid=None, explicit=True, attached=False):
        self.config = config
        self.run_id = run_id or uuid.uuid4().hex
        self.owner_pid = owner_pid if owner_pid is not None else os.getpid()
        self.process_id = os.getpid()
        self.explicit = explicit
        base = Path(artifact_dir or (Path.cwd() / ".inference")).expanduser().resolve()
        self.artifact_dir = base if attached else base / self.run_id
        self._endpoint = endpoint
        self._broker = None
        self._client = None
        self._preflight = None
        self._lock = threading.RLock()
        self._warnings = set()
        self._closed = False
        self._previous_environment = {key: os.environ.get(key) for key in _ENV_KEYS}
        if endpoint is not None:
            if endpoint.get("run_id") != self.run_id:
                raise ConfigurationError("Broker endpoint belongs to a different inference run")
            from .broker import BrokerClient
            self._client = BrokerClient(endpoint)

    @property
    def is_cli(self) -> bool:
        return self.config.mode == "cli"

    def export_config(self) -> dict:
        return self.config.to_dict()

    def export_environment(self) -> dict[str, str]:
        values = {
            ENV_CONFIG: json.dumps(self.config.to_dict(), allow_nan=False),
            ENV_RUN_ID: self.run_id,
            ENV_ARTIFACT_DIR: str(self.artifact_dir),
            ENV_OWNER_PID: str(self.owner_pid),
        }
        if self._endpoint is not None:
            values[ENV_ENDPOINT] = json.dumps(self._endpoint, allow_nan=False)
        os.environ.update(values)
        if self._endpoint is None:
            os.environ.pop(ENV_ENDPOINT, None)
        return values

    def preflight(self, *, force=False) -> dict:
        if not self.is_cli:
            return {"mode": "api", "providers": {}, "live_inference_verified": False}
        with self._lock:
            if self._preflight is not None and not force:
                return self._preflight
            from .cli import inspect_provider
            backends = sorted({self.config.resolve(role).backend for role in ROLES})
            providers = {}
            for backend in backends:
                providers[backend] = inspect_provider(backend, self.config.providers[backend])
            self._preflight = {
                "mode": "cli", "subscription_only": True,
                "providers": providers,
                "roles": {role: {"backend": self.config.resolve(role).backend,
                                   "model": self.config.resolve(role).model} for role in ROLES},
                "live_inference_verified": False,
                "note": "Read-only checks are not a live image/schema/full-pipeline acceptance test.",
            }
            return self._preflight

    def ensure_started(self):
        if not self.is_cli:
            return
        with self._lock:
            if self._closed:
                raise BrokerUnavailable("Inference runtime is already closed")
            if self._client is not None:
                return
            if self.owner_pid != os.getpid():
                raise BrokerUnavailable("Parent must start the inference broker before spawning workers")
            self.preflight()
            from .broker import Broker, BrokerClient
            self._broker = Broker(self.config.to_dict(), self.run_id, self.artifact_dir).start()
            self._endpoint = self._broker.endpoint
            self._client = BrokerClient(self._endpoint)
            self.export_environment()

    def submit(self, request):
        if not self.is_cli:
            raise ConfigurationError("A CLI inference request cannot use the API runtime")
        self.ensure_started()
        try:
            return self._client.submit(request)
        except BrokerUnavailable:
            self._cleanup_after_broker_loss()
            raise

    def _cleanup_after_broker_loss(self):
        # A dead broker cannot run its own process-runner finally blocks. The
        # launcher may reclaim only the groups with verified ownership records.
        if self._broker is not None and self.owner_pid == os.getpid():
            self._broker.cleanup_owned_processes()

    def record_validation(self, request_id: str, *, accepted: bool, error: dict | None = None):
        if self._client is not None and hasattr(self._client, "record_validation"):
            try:
                self._client.record_validation(request_id, accepted=accepted, error=error)
            except BrokerUnavailable:
                self._cleanup_after_broker_loss()
                raise

    def cancel_all(self, reason="Run cancelled"):
        if self._client is not None:
            try:
                self._client.cancel_all(reason=reason)
            except BrokerUnavailable:
                self._cleanup_after_broker_loss()
                raise

    def generate(self, client, **kwargs):
        from .gateway import generate
        return generate(client, runtime=self, **kwargs)

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._broker is not None and self.owner_pid == os.getpid():
                    self._broker.close(cancel=True)
            finally:
                self._closed = True
                self._client = None
                if self.owner_pid == os.getpid():
                    for key, value in self._previous_environment.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value


def is_configured() -> bool:
    return ENV_CONFIG in os.environ or (_runtime is not None and _runtime.explicit)


def _from_environment() -> InferenceRuntime:
    try:
        config = InferenceConfig.from_dict(json.loads(os.environ[ENV_CONFIG]))
        endpoint = json.loads(os.environ[ENV_ENDPOINT]) if os.environ.get(ENV_ENDPOINT) else None
        owner = int(os.environ.get(ENV_OWNER_PID, os.getpid()))
        run_id = os.environ.get(ENV_RUN_ID)
        if not run_id:
            raise ConfigurationError("Inherited inference configuration lacks run identity")
        return InferenceRuntime(config, artifact_dir=os.environ.get(ENV_ARTIFACT_DIR), run_id=run_id,
                                endpoint=endpoint, owner_pid=owner, attached=True)
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigurationError("Invalid inherited inference runtime") from exc


def get_runtime() -> InferenceRuntime:
    global _runtime
    with _lock:
        if _runtime is None or _runtime.process_id != os.getpid():
            _runtime = _from_environment() if ENV_CONFIG in os.environ else InferenceRuntime(InferenceConfig(), explicit=False)
        return _runtime


def configure_inference(config=None, *, artifact_dir=None) -> InferenceRuntime:
    global _runtime
    resolved = config if isinstance(config, InferenceConfig) else InferenceConfig.from_dict(config)
    with _lock:
        if _runtime is not None and not _runtime._closed:
            if _runtime.config.to_dict() == resolved.to_dict() and artifact_dir is None and _runtime.explicit:
                return _runtime
            if _runtime.owner_pid != os.getpid() and _runtime.is_cli:
                raise ConfigurationError("A worker cannot replace its inherited inference runtime")
            _runtime.close()
        _runtime = InferenceRuntime(resolved, artifact_dir=artifact_dir)
        _runtime.export_environment()
        return _runtime


def configure_from_args(args, *, artifact_dir=None, argv=None, default_config=None) -> InferenceRuntime:
    config = config_from_args(args, list(sys.argv[1:] if argv is None else argv), default_config=default_config)
    return configure_inference(config, artifact_dir=getattr(args, "inference_log_dir", None) or artifact_dir)


def shutdown_inference():
    global _runtime
    with _lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None


def get_cli_client(model=None, role="default") -> InferenceClient | None:
    runtime = get_runtime()
    if not runtime.is_cli:
        return None
    # Legacy default model strings are deliberately not CLI model overrides.
    # Explicit flags are resolved into the role configuration before this point.
    client = runtime.config.resolve(role or "default")
    if client.backend not in CLI_BACKENDS:
        raise ConfigurationError("API inference is forbidden in a CLI run")
    return client


def resolve_api_model(legacy_model, role="default"):
    """Use explicit profile/role models, falling back to unchanged V2 defaults."""
    runtime = get_runtime()
    if runtime.is_cli:
        raise ConfigurationError("An API model cannot be resolved in a CLI-only run")
    configured = runtime.config.resolve(role or "default").model
    return configured if configured is not None else legacy_model


def is_cli_client(client) -> bool:
    return isinstance(client, InferenceClient) and client.backend in CLI_BACKENDS
