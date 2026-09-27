"""Plain JSON contracts shared by callers, the broker, and CLI transports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .errors import ConfigurationError

CLI_BACKENDS = frozenset({"codex_cli", "claude_cli"})
ROLES = (
    "default", "ideation", "experiment_code", "execution_feedback",
    "experiment_vision", "stage_control", "node_selection", "summary", "report",
    "aggregation", "citation", "writing", "writing_vision", "review_text", "review_vision",
)


def json_copy(value: Any) -> Any:
    """Copy through the actual wire format; reject NaN and non-JSON values."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Inference data must contain only finite JSON values") from exc


@dataclass(frozen=True)
class InferenceClient:
    backend: str
    model: str | None = None
    role: str = "default"


@dataclass
class InferenceRequest:
    request_id: str
    run_id: str
    role: str
    backend: str
    model: str | None = None
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    schema: dict | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float = 1800.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id or not isinstance(self.run_id, str) or not self.run_id:
            raise ConfigurationError("Requests require nonempty request_id and run_id")
        if not isinstance(self.role, str) or not self.role:
            raise ConfigurationError("Requests require a nonempty role")
        if not isinstance(self.backend, str) or self.backend not in CLI_BACKENDS:
            raise ConfigurationError("A CLI request must select codex_cli or claude_cli")
        if not isinstance(self.messages, list) or not all(isinstance(m, dict) for m in self.messages):
            raise ConfigurationError("messages must be an ordered list of objects")
        if not isinstance(self.system, str):
            raise ConfigurationError("system instruction must be text")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ConfigurationError("model must be a nonempty string or null")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("request timeout must be finite and positive")
        if self.temperature is not None and (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature)):
            raise ConfigurationError("temperature must be a finite number")
        if self.max_output_tokens is not None and (type(self.max_output_tokens) is not int or self.max_output_tokens <= 0):
            raise ConfigurationError("max_output_tokens must be a positive integer")
        if self.schema is not None and not isinstance(self.schema, dict):
            raise ConfigurationError("schema must be an object")
        if not isinstance(self.metadata, dict):
            raise ConfigurationError("metadata must be an object")
        self.messages = json_copy(self.messages)
        self.schema = json_copy(self.schema)
        self.metadata = json_copy(self.metadata)

    def to_dict(self) -> dict:
        return json_copy(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceRequest":
        allowed = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ConfigurationError("Invalid inference request fields")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError("Missing or invalid inference request fields") from exc


@dataclass
class InferenceResult:
    request_id: str
    text: str = ""
    structured_data: Any = None
    backend: str = ""
    model: str | None = None
    usage: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict:
        return json_copy(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceResult":
        allowed = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ConfigurationError("Invalid inference result fields")
        try:
            result = cls(**json_copy(data))
        except TypeError as exc:
            raise ConfigurationError("Missing inference result fields") from exc
        if not isinstance(result.text, str) or not isinstance(result.usage, dict) or not isinstance(result.metadata, dict):
            raise ConfigurationError("Malformed inference result")
        return result
