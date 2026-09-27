"""Unified, stateless text/vision requests with finite retries and host validation."""

from __future__ import annotations

import json
import time
import uuid
import warnings

from .contracts import InferenceClient, InferenceRequest
from .errors import (ConfigurationError, ContextLimitExceeded, InferenceError,
                     InvalidStructuredOutput, TransientProviderError, UnsupportedCapability)
from .history import build_messages
from .images import freeze_messages
from .runtime import get_runtime
from .schema import provider_schema, unpack_structured


def _parameters(runtime, client, temperature, max_tokens):
    unsupported = []
    if temperature is not None:
        unsupported.append("temperature")
    if max_tokens is not None:
        unsupported.append("max_output_tokens")
    if unsupported and runtime.config.unsupported_parameter_policy == "error":
        raise UnsupportedCapability("CLI does not provide API-equivalent control for: " + ", ".join(unsupported))
    key = (client.backend, client.role, tuple(unsupported))
    if unsupported and key not in runtime._warnings:
        warnings.warn(f"{client.backend}/{client.role}: requested {', '.join(unsupported)} cannot be applied as API sampling controls; recorded in metadata", RuntimeWarning, stacklevel=3)
        runtime._warnings.add(key)
    return {"requested_parameters": {"temperature": temperature, "max_output_tokens": max_tokens},
            "unsupported_parameters": unsupported,
            "effective_parameters": {"temperature": "cli_default",
                                     "max_output_tokens": "cli_default"},
            "sampling_equivalent_to_api": not unsupported}


def _repair_messages(messages, error, previous=None):
    # Only an actual final answer may be inserted as assistant history. A native
    # schema failure without one adds a corrective user instruction, not a
    # fabricated assistant message or raw CLI protocol/debug output.
    repair = []
    if previous is not None:
        repair.append({"role": "assistant", "content": [{"type": "text", "text": previous}]})
    repair.append({"role": "user", "content": [{"type": "text", "text": (
        "The response failed the requested output contract: " + str(error) +
        ". Return a corrected response in the required schema. Do not change a "
        "legitimate negative scientific evaluation to a positive one."
    )}]})
    return messages + repair


def generate(client: InferenceClient, *, prompt, system_message="", msg_history=None,
             images=None, schema=None, temperature=None, max_tokens=None, runtime=None):
    runtime = runtime or get_runtime()
    if not isinstance(client, InferenceClient) or not runtime.is_cli:
        raise ConfigurationError("The CLI gateway requires an explicit CLI client and runtime")
    if runtime.config.resolve(client.role) != client:
        raise ConfigurationError("Inference client does not match this run's resolved role")
    if system_message is None:
        system_message = ""
    if not isinstance(system_message, str):
        raise ConfigurationError("System instructions must be compiled to text")
    parameters = _parameters(runtime, client, temperature, max_tokens)
    messages = freeze_messages(build_messages(prompt, msg_history, images), runtime.config.max_payload_bytes)
    native_schema, schema_instruction, wrapped = provider_schema(schema, client.backend)
    system = system_message + schema_instruction
    logical_id = uuid.uuid4().hex
    repairs = 0
    transient_retries = 0
    attempts = 0
    while True:
        attempts += 1
        request = InferenceRequest(
            request_id=uuid.uuid4().hex, run_id=runtime.run_id, role=client.role,
            backend=client.backend, model=client.model, system=system, messages=messages,
            schema=native_schema, temperature=temperature, max_output_tokens=max_tokens,
            timeout_seconds=runtime.config.request_timeout_seconds,
            metadata={**parameters, "logical_request_id": logical_id, "attempt": attempts,
                      "history_mode": "explicit_stateless", "schema_encoding": "json_string_envelope" if wrapped else "native",
                      "original_schema": schema, "max_payload_bytes": runtime.config.max_payload_bytes},
        )
        encoded_size = len(json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8"))
        if encoded_size > runtime.config.max_payload_bytes:
            raise ContextLimitExceeded("Encoded inference request exceeds max_payload_bytes; input was not truncated")
        try:
            result = runtime.submit(request)
        except TransientProviderError:
            if transient_retries >= runtime.config.transient_retry_limit:
                raise
            transient_retries += 1
            time.sleep(min(2 ** (transient_retries - 1), 8))
            continue
        except InvalidStructuredOutput as exc:
            # These are definite native output-contract failures, not timeout,
            # disconnect, cancellation, or another uncertain execution state.
            # The broker already persists the failed transport attempt.
            if schema is None or repairs >= runtime.config.schema_repair_limit:
                raise
            repairs += 1
            messages = _repair_messages(messages, exc)
            continue
        if result.request_id != request.request_id or result.status != "completed" or result.backend != request.backend:
            raise InvalidStructuredOutput("Transport did not return a matching completed request")
        # Count every returned attempt, including a schema rejection later repaired.
        # The broker remains authoritative across all worker processes.
        try:
            from ai_scientist.utils.token_tracker import token_tracker
            token_tracker.add_normalized_usage(result.backend, result.model, result.usage, request_id=result.request_id)
        except ImportError:
            pass
        try:
            if schema is not None:
                result.structured_data = unpack_structured(result, schema, wrapped)
                result.text = json.dumps(result.structured_data, ensure_ascii=False, allow_nan=False)
            elif not result.text.strip():
                raise InvalidStructuredOutput("CLI returned an empty final answer")
        except InvalidStructuredOutput as exc:
            runtime.record_validation(request.request_id, accepted=False, error=exc.to_dict())
            if schema is None or repairs >= runtime.config.schema_repair_limit:
                raise
            repairs += 1
            previous = result.text or json.dumps(result.structured_data, ensure_ascii=False)
            messages = _repair_messages(messages, exc, previous)
            continue
        runtime.record_validation(request.request_id, accepted=True)
        result.metadata.update(parameters)
        result.metadata.update({"logical_request_id": logical_id, "attempt_count": attempts,
                                "schema_repairs": repairs, "transient_retries": transient_retries,
                                "history_mode": "explicit_stateless", "schema_validated": schema is not None})
        return result
