"""Subscription CLI inference. Importing this package never starts a model call."""

from .config import InferenceConfig, add_inference_arguments, load_config
from .contracts import InferenceClient, InferenceRequest, InferenceResult
from .errors import InferenceError
from .gateway import generate
from .runtime import (
    InferenceRuntime, configure_from_args, configure_inference, get_cli_client,
    get_runtime, is_cli_client, is_configured, resolve_api_model, shutdown_inference,
)

__all__ = [
    "InferenceConfig", "InferenceClient", "InferenceRequest", "InferenceResult", "InferenceError",
    "InferenceRuntime", "add_inference_arguments", "load_config", "configure_from_args",
    "configure_inference", "get_cli_client", "get_runtime", "is_cli_client", "is_configured",
    "shutdown_inference", "generate", "resolve_api_model",
]
