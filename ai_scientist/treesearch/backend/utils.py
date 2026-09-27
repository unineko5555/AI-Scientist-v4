from copy import deepcopy
from dataclasses import asdict, dataclass
import json

import jsonschema
try:
    from dataclasses_json import DataClassJsonMixin
except ModuleNotFoundError as exc:
    if exc.name != "dataclasses_json":
        raise

    class DataClassJsonMixin:
        """Optional API/research-stack dependency; FunctionSpec is JSON-native."""

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


import logging
from typing import Callable

logger = logging.getLogger("ai-scientist")


def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    # This dependency is only needed by the legacy API backends. CLI errors
    # never enter this API retry loop.
    import backoff

    @backoff.on_exception(backoff.expo, retry_exceptions, max_value=60, factor=1.5)
    def call():
        return create_fn(*args, **kwargs)

    return call()


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """Convert a prompt into markdown format"""
    try:
        logger.debug(f"compile_prompt_to_md input: type={type(prompt)}")
        if isinstance(prompt, (list, dict)):
            logger.debug(f"prompt content: {prompt}")

        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt.strip() + "\n"

        if isinstance(prompt, list):
            # Handle empty list case
            if not prompt:
                return ""
            # Special handling for multi-modal messages
            if all(isinstance(item, dict) and "type" in item for item in prompt):
                # For multi-modal messages, just pass through without modification
                return prompt

            try:
                result = "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])
                return result
            except Exception as e:
                logger.error(f"Error processing list items: {e}")
                logger.error("List contents:")
                for i, item in enumerate(prompt):
                    logger.error(f"  Item {i}: type={type(item)}, value={item}")
                raise

        if isinstance(prompt, dict):
            # Check if this is a single multi-modal message
            if "type" in prompt:
                return prompt

            # Regular dict processing
            try:
                out = []
                header_prefix = "#" * _header_depth
                for k, v in prompt.items():
                    logger.debug(f"Processing dict key: {k}")
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:
                logger.error(f"Error processing dict: {e}")
                logger.error(f"Dict contents: {prompt}")
                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        logger.error("Error in compile_prompt_to_md:")
        logger.error(f"Input type: {type(prompt)}")
        logger.error(f"Input content: {prompt}")
        logger.error(f"Error: {str(e)}")
        raise


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    def validate(self, value):
        """Validate the returned value, not only the schema definition."""
        jsonschema.Draft7Validator(self.json_schema).validate(value)
        return value

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**deepcopy(value))

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, value, **kwargs):
        return cls.from_dict(json.loads(value, **kwargs))

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }


def coerce_function_spec(value):
    """Accept the existing node-summary ``parameters`` dictionary contract."""
    if value is None or isinstance(value, FunctionSpec):
        return value
    if not isinstance(value, dict):
        raise TypeError("func_spec must be a FunctionSpec or function dictionary")
    definition = value.get("function", value)
    schema = definition.get("json_schema", definition.get("parameters"))
    if schema is None or not definition.get("name"):
        raise ValueError("Function specification requires name and parameters/json_schema")
    return FunctionSpec(
        name=definition["name"],
        description=definition.get("description", ""),
        json_schema=deepcopy(schema),
    )
