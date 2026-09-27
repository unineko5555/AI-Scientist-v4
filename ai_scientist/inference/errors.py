"""Serializable infrastructure errors, distinct from scientific evaluation failures."""

from __future__ import annotations

from typing import Any


class InferenceError(Exception):
    code = "inference_error"
    retryable = False

    def __init__(self, message: str, *, request_id: str | None = None, metadata: dict | None = None):
        super().__init__(message)
        self.request_id = request_id
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "request_id": self.request_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceError":
        error_type = _ERROR_TYPES.get(data.get("code"), ProviderError)
        return error_type(str(data.get("message", "Inference failed")), request_id=data.get("request_id"), metadata=data.get("metadata") or {})

    def __reduce__(self):
        return (deserialize_error, (self.to_dict(),))


class ConfigurationError(InferenceError):
    code = "configuration_error"


class AuthenticationRequired(InferenceError):
    code = "authentication_required"


class QuotaExhausted(InferenceError):
    code = "quota_exhausted"


class UnsupportedCapability(InferenceError):
    code = "unsupported_capability"


class ContextLimitExceeded(InferenceError):
    code = "context_limit_exceeded"


class TransientProviderError(InferenceError):
    code = "transient_provider_error"
    retryable = True


class QueueTimeout(InferenceError):
    code = "queue_timeout"


class InferenceTimeout(InferenceError):
    code = "inference_timeout"


class InvalidStructuredOutput(InferenceError):
    code = "invalid_structured_output"


class IncompleteOutput(InferenceError):
    code = "incomplete_output"


class Cancelled(InferenceError):
    code = "cancelled"


class BrokerUnavailable(InferenceError):
    code = "broker_unavailable"


class ProviderError(InferenceError):
    code = "provider_error"


_ERROR_TYPES = {c.code: c for c in (
    InferenceError, ConfigurationError, AuthenticationRequired, QuotaExhausted,
    UnsupportedCapability, ContextLimitExceeded, TransientProviderError, QueueTimeout,
    InferenceTimeout, InvalidStructuredOutput, IncompleteOutput, Cancelled,
    BrokerUnavailable, ProviderError,
)}


def serialize_error(error: InferenceError) -> dict:
    return error.to_dict()


def deserialize_error(data: dict) -> InferenceError:
    return InferenceError.from_dict(data)
