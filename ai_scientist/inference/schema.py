"""Provider schema envelopes plus validation against the original V2 contract."""

from __future__ import annotations

import json
import math
from copy import deepcopy

from .errors import ConfigurationError, InvalidStructuredOutput, UnsupportedCapability

_ENVELOPE_KEY = "answer_json"


def parse_json(text: str):
    def invalid_constant(_):
        raise ValueError("Non-finite JSON number")
    try:
        return json.loads(text, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidStructuredOutput("The final response is not a complete finite JSON value") from exc


def validate_structured(value, schema: dict):
    try:
        import jsonschema
    except ImportError as exc:
        raise UnsupportedCapability("Structured inference requires jsonschema") from exc
    try:
        jsonschema.Draft7Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ConfigurationError("Invalid requested JSON schema") from exc
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidStructuredOutput("Structured output must contain only finite JSON values") from exc
    errors = list(jsonschema.Draft7Validator(schema).iter_errors(value))
    if errors:
        first = errors[0]
        location = ".".join(str(p) for p in first.absolute_path) or "<root>"
        raise InvalidStructuredOutput(f"Structured output failed {first.validator} validation at {location}")
    if isinstance(value, dict):
        for flag, collection in (("valid_metrics_received", "metric_names"), ("valid_plots_received", "plot_analyses")):
            if value.get(flag) is True and collection in value and not value[collection]:
                raise InvalidStructuredOutput(f"{flag}=true requires a nonempty {collection}")
    return value


def provider_schema(schema: dict | None, backend: str) -> tuple[dict | None, str, bool]:
    """Codex's strict subset cannot express every V2 schema losslessly.

    A JSON-string envelope admits all original JSON values without inventing
    nullable/default scientific fields. The host still validates the original
    schema; this is intentionally not described as native business-schema validation.
    """
    if schema is None:
        return None, "", False
    try:
        import jsonschema
        jsonschema.Draft7Validator.check_schema(schema)
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except ImportError as exc:
        raise UnsupportedCapability("Structured inference requires jsonschema") from exc
    except (ValueError, TypeError, jsonschema.SchemaError) as exc:
        raise ConfigurationError("Invalid requested JSON schema") from exc
    if backend == "codex_cli":
        wrapped = {"type": "object", "properties": {_ENVELOPE_KEY: {"type": "string"}},
                   "required": [_ENVELOPE_KEY], "additionalProperties": False}
        instructions = (
            "\nReturn a JSON object with exactly one field, answer_json. Its value must be a "
            "JSON-encoded string containing the actual answer that validates against the "
            "following schema. Preserve legitimate negative evaluation results; do not "
            "fabricate successful experiments. Original answer schema:\n" + encoded
        )
        return wrapped, instructions, True
    return deepcopy(schema), "\nThe response must validate against this JSON schema:\n" + encoded, False


def unpack_structured(result, original_schema: dict, wrapped: bool):
    value = result.structured_data if result.structured_data is not None else parse_json(result.text)
    if wrapped:
        if not isinstance(value, dict) or set(value) != {_ENVELOPE_KEY} or not isinstance(value[_ENVELOPE_KEY], str):
            raise InvalidStructuredOutput("Missing Codex answer_json schema envelope")
        value = parse_json(value[_ENVELOPE_KEY])
    return validate_structured(value, original_schema)
