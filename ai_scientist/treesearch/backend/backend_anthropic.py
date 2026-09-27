import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from ai_scientist.utils.token_tracker import record_token_usage


def get_ai_client(model : str, max_retries=2):
    from ai_scientist.llm import _require_api_client
    _require_api_client(None)
    import anthropic
    client = anthropic.AnthropicBedrock(max_retries=max_retries)
    return client

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    import anthropic

    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        raise NotImplementedError(
            "Anthropic does not support function calling for now."
        )

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError,
         anthropic.InternalServerError, anthropic.APIStatusError),
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    record_token_usage(message, prompt=user_message, system_message=system_message)
    print(filtered_kwargs)

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
