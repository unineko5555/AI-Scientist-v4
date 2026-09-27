"""Explicit transport configuration; API model names never select CLI providers."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .contracts import CLI_BACKENDS, ROLES, InferenceClient, json_copy
from .errors import ConfigurationError


def _providers() -> dict:
    return {"codex_cli": {"executable": "codex", "max_in_flight": 1}, "claude_cli": {"executable": "claude", "max_in_flight": 1}}


@dataclass
class InferenceConfig:
    mode: str = "api"
    subscription_only: bool = False
    default: dict = field(default_factory=lambda: {"backend": "api", "model": None})
    roles: dict = field(default_factory=dict)
    providers: dict = field(default_factory=_providers)
    history_mode: str = "explicit_stateless"
    unsupported_parameter_policy: str = "warn_and_record"
    overflow_policy: str = "error"
    queue_timeout_seconds: float = 3600.0
    request_timeout_seconds: float = 1800.0
    transient_retry_limit: int = 2
    schema_repair_limit: int = 1
    max_queue_size: int = 64
    max_payload_bytes: int = 32 * 1024 * 1024
    fallback: str = "none"

    def __post_init__(self):
        self.default = deepcopy(self.default)
        self.roles = deepcopy(self.roles)
        self.providers = deepcopy(self.providers)
        if self.mode not in ("api", "cli"):
            raise ConfigurationError("inference.mode must be api or cli")
        if type(self.subscription_only) is not bool:
            raise ConfigurationError("subscription_only must be boolean")
        if self.history_mode != "explicit_stateless" or self.overflow_policy != "error":
            raise ConfigurationError("Only explicit_stateless history and error-on-overflow are supported")
        if self.fallback not in ("none", None):
            raise ConfigurationError("Automatic inference fallback is not supported")
        if self.unsupported_parameter_policy not in ("warn_and_record", "error"):
            raise ConfigurationError("unsupported_parameter_policy must be warn_and_record or error")
        for name in ("queue_timeout_seconds", "request_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive")
        for name, minimum in (("transient_retry_limit", 0), ("schema_repair_limit", 0), ("max_queue_size", 1), ("max_payload_bytes", 1024)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.default, dict) or not isinstance(self.roles, dict) or not isinstance(self.providers, dict):
            raise ConfigurationError("default, roles, and providers must be objects")
        if set(self.default) - {"backend", "model"}:
            raise ConfigurationError("Unknown default inference fields")
        if not isinstance(self.default.get("backend"), str) or self.default["backend"] not in CLI_BACKENDS | {"api"}:
            raise ConfigurationError("Unknown default inference backend")
        for role, setting in self.roles.items():
            if role not in ROLES or not isinstance(setting, dict) or set(setting) - {"backend", "model"}:
                raise ConfigurationError(f"Invalid inference role: {role}")
        for role in ROLES:
            resolved = self.resolve(role)
            if not isinstance(resolved.backend, str) or resolved.backend not in CLI_BACKENDS | {"api"}:
                raise ConfigurationError(f"Unknown backend for role {role}")
            if self.mode == "cli" and resolved.backend not in CLI_BACKENDS:
                raise ConfigurationError(f"CLI mode requires a CLI backend for role {role}")
            if self.mode == "api" and resolved.backend != "api":
                raise ConfigurationError("Use mode=cli for CLI providers; mixed API/CLI execution is not supported")
            if self.subscription_only and resolved.backend == "api":
                raise ConfigurationError("subscription_only prohibits API backends")
            if resolved.model is not None and (not isinstance(resolved.model, str) or not resolved.model.strip() or resolved.model.startswith("-")):
                raise ConfigurationError(f"Invalid model for role {role}")
        if set(self.providers) - CLI_BACKENDS:
            raise ConfigurationError("Unknown CLI provider configuration")
        for backend, provider in self.providers.items():
            if not isinstance(provider, dict):
                raise ConfigurationError(f"Provider {backend} must be an object")
            executable = provider.get("executable", _providers()[backend]["executable"])
            if not isinstance(executable, str) or not executable or "\x00" in executable:
                raise ConfigurationError("CLI executable must be a nonempty path or command name")
            limit = provider.get("max_in_flight", 1)
            if type(limit) is not int or not 1 <= limit <= 32:
                raise ConfigurationError("Provider max_in_flight must be between 1 and 32")
            forbidden = {"env", "api_key", "auth_token", "skip_auth", "skip_preflight", "allow_api", "fallback"}
            if forbidden & set(provider):
                raise ConfigurationError("Provider settings cannot override authentication or enable fallback")
        json_copy(asdict(self))

    def resolve(self, role: str = "default") -> InferenceClient:
        if role not in ROLES:
            raise ConfigurationError(f"Unknown inference role: {role}")
        settings = {**self.default, **self.roles.get(role, {})}
        return InferenceClient(settings.get("backend", "api"), settings.get("model"), role)

    def to_dict(self) -> dict:
        return json_copy(asdict(self))

    @classmethod
    def from_dict(cls, data: dict | None) -> "InferenceConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("inference configuration must be an object")
        if "inference" in data:
            data = data["inference"]
        if not isinstance(data, dict):
            raise ConfigurationError("inference configuration must be an object")
        data = json_copy(data)
        allowed = {f.name for f in fields(cls)}
        if set(data) - allowed:
            raise ConfigurationError("Unknown inference settings: " + ", ".join(sorted(set(data) - allowed)))
        for key in ("default", "roles", "providers"):
            if key in data and not isinstance(data[key], dict):
                raise ConfigurationError(f"{key} must be an object")
        default_backend = data.get("default", {}).get("backend")
        if data.get("mode") == "cli" or (isinstance(default_backend, str) and default_backend in CLI_BACKENDS):
            data.setdefault("mode", "cli")
            data.setdefault("subscription_only", True)
            data.setdefault("default", {"backend": "codex_cli", "model": None})
        defaults = _providers()
        for backend, settings in data.get("providers", {}).items():
            if backend not in defaults or not isinstance(settings, dict):
                raise ConfigurationError("Invalid CLI provider settings")
            defaults[backend].update(settings)
        data["providers"] = defaults
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError("Invalid inference configuration") from exc


def load_config(path: str | Path) -> InferenceConfig:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() == ".json":
                data = json.load(stream)
            else:
                try:
                    import yaml
                except ImportError as exc:
                    raise ConfigurationError("Reading YAML inference configuration requires PyYAML") from exc
                try:
                    data = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise ConfigurationError("Cannot parse inference YAML configuration") from exc
    except (OSError, ValueError) as exc:
        raise ConfigurationError("Cannot read inference configuration file") from exc
    return InferenceConfig.from_dict(data)


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    # Model provenance is based on explicitly supplied full option names.
    # Reject abbreviations instead of accepting an override then ignoring it.
    parser.allow_abbrev = False
    group = parser.add_argument_group("Inference transport")
    group.add_argument("--inference-config", help="JSON/YAML inference settings")
    group.add_argument("--llm-backend", choices=["api", "codex_cli", "claude_cli"], default=None,
                       help="Explicit model transport; CLI backends use subscription authentication")
    group.add_argument("--llm-model", default=None, help="Default CLI model; omitted uses the CLI's default")
    group.add_argument("--inference-log-dir", default=None, help="Private inference artifacts directory")


def config_from_args(args, argv: list[str], *, default_config: dict | InferenceConfig | None = None) -> InferenceConfig:
    config_path = getattr(args, "inference_config", None)
    if config_path:
        config = load_config(config_path)
    elif isinstance(default_config, InferenceConfig):
        config = default_config
    else:
        config = InferenceConfig.from_dict(default_config)
    data = config.to_dict()
    backend = getattr(args, "llm_backend", None)
    if backend:
        data["mode"] = "api" if backend == "api" else "cli"
        data["subscription_only"] = backend != "api"
        data["default"]["backend"] = backend
    model = getattr(args, "llm_model", None)
    if model is not None:
        if data["mode"] != "cli":
            raise ConfigurationError("--llm-model requires a CLI backend")
        data["default"]["model"] = model
    explicit_flags = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for destination, roles in getattr(args, "_inference_model_roles", {}).items():
        flags = {"--" + destination, "--" + destination.replace("_", "-")}
        if explicit_flags & flags:
            value = getattr(args, destination, None)
            if value is not None:
                for role in ([roles] if isinstance(roles, str) else roles):
                    data["roles"].setdefault(role, {})["model"] = value
    return InferenceConfig.from_dict(data)
