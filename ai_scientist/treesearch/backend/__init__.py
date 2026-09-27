from copy import deepcopy
from ai_scientist.inference import generate, get_cli_client, resolve_api_model
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md, coerce_function_spec

def get_ai_client(model=None, *, role="default", **model_kwargs):
    """
    Get the appropriate AI client based on the model string.

    Args:
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        An instance of the appropriate AI client.
    """
    cli_client = get_cli_client(model, role=role)
    if cli_client is not None:
        return cli_client
    model = resolve_api_model(model, role=role)
    if not isinstance(model, str):
        raise ValueError("An explicit model is required in API mode")
    if "claude-" in model:
        from . import backend_anthropic
        return backend_anthropic.get_ai_client(model=model, **model_kwargs)
    else:
        from . import backend_openai
        return backend_openai.get_ai_client(model=model, **model_kwargs)

def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    role: str = "default",
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    func_spec = coerce_function_spec(func_spec)
    cli_client = get_cli_client(model, role=role)
    if cli_client is not None:
        system = compile_prompt_to_md(system_message) if system_message is not None else ""
        user = compile_prompt_to_md(user_message) if user_message is not None else ""
        # V2 often puts the entire request in system_message. A text system
        # instruction remains separate; multimodal content must be an input.
        if not isinstance(system, str):
            if user:
                raise ValueError("Multimodal system and user prompts cannot be merged implicitly")
            system, user = "", system
        result = generate(
            cli_client,
            prompt=user,
            system_message=system,
            schema=deepcopy(func_spec.json_schema) if func_spec else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if func_spec is not None:
            return func_spec.validate(result.structured_data)
        return result.text

    model = resolve_api_model(model, role=role)
    if not isinstance(model, str):
        raise ValueError("An explicit model is required in API mode")
    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    if "claude-" in model:
        from .backend_anthropic import query as query_func
    else:
        from .backend_openai import query as query_func
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
