import base64
from copy import deepcopy
import io
from typing import Any
import os
from PIL import Image
from ai_scientist.utils.token_tracker import record_token_usage, track_token_usage
from ai_scientist.inference import generate, get_cli_client, is_cli_client, resolve_api_model
from ai_scientist.inference.errors import ConfigurationError, UnsupportedCapability
from ai_scientist.llm import _api_retry, _require_api_client, _require_sdk, openai, extract_json_between_markers

MAX_NUM_TOKENS = 4096

AVAILABLE_VLMS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "o3-mini",

    # Ollama models

    # llama4
    "ollama/llama4:16x17b",

    # mistral
    "ollama/mistral-small3.2:24b",

    # qwen
    "ollama/qwen2.5vl:32b",

    "ollama/z-uo/qwen2.5vl_tools:32b",
]


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Save to bytes
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    _require_api_client(client)
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    else:
        raise ValueError(f"Model {model} not supported.")

@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt, max_tokens=MAX_NUM_TOKENS):
    _require_api_client(client)
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def prepare_vlm_prompt(msg, image_paths, max_images):
    """Keep every image occurrence, in order, with its actual bytes and MIME."""
    if isinstance(image_paths, (str, os.PathLike, bytes)):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 0:
        raise ConfigurationError("max_images must be a non-negative integer")
    if len(image_paths) > max_images:
        raise UnsupportedCapability(
            f"Received {len(image_paths)} images; max_images={max_images}. "
            "No images were silently dropped. Increase the explicit limit if supported."
        )
    content = [{"type": "text", "text": msg}]
    for source in image_paths:
        try:
            if isinstance(source, bytes):
                image_bytes = source
            else:
                with open(source, "rb") as image_file:
                    image_bytes = image_file.read()
            with Image.open(io.BytesIO(image_bytes)) as image:
                mime = Image.MIME.get(image.format)
                image.verify()
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigurationError("Cannot read a valid image for the VLM request") from exc
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise UnsupportedCapability(f"Unsupported image format: {mime}")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "low"},
        })
    return content


@_api_retry
def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
    max_tokens: int = MAX_NUM_TOKENS,
    schema: dict | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from vision-language model."""
    if msg_history is None:
        msg_history = []

    if is_cli_client(client):
        content = prepare_vlm_prompt(msg, image_paths, max_images)
        history = deepcopy(msg_history)
        result = generate(
            client,
            prompt=content,
            system_message=system_message,
            msg_history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            schema=schema,
        )
        return result.text, history + [
            {"role": "user", "content": content},
            {"role": "assistant", "content": result.text},
        ]

    _require_api_client(client)
    if model in AVAILABLE_VLMS:
        content = prepare_vlm_prompt(msg, image_paths, max_images)
        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        response = make_vlm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if schema is not None:
        from ai_scientist.inference.schema import validate_structured
        validate_structured(extract_json_between_markers(content), schema)

    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return content, new_msg_history


def create_client(model=None, *, role="default") -> tuple[Any, str | None]:
    """Create client for vision-language model."""
    cli_client = get_cli_client(model, role=role)
    if cli_client is not None:
        return cli_client, cli_client.model
    model = resolve_api_model(model, role=role)
    if not isinstance(model, str):
        raise ValueError("An explicit model is required in API mode")
    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "o3-mini",
    ]:
        print(f"Using OpenAI API with model {model}.")
        return _require_sdk(openai, "openai").OpenAI(), model
    elif model.startswith("ollama/"):
        print(f"Using Ollama API with model {model}.")
        return _require_sdk(openai, "openai").OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url="http://localhost:11434/v1"
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")


@_api_retry
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
    schema: dict | None = None,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from vision-language model for the same input.

    Args:
        msg: Text message to send
        image_paths: Path(s) to image file(s)
        client: OpenAI client instance
        model: Name of model to use
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate

    Returns:
        Tuple of (list of response strings, list of message histories)
    """
    if msg_history is None:
        msg_history = []

    if not isinstance(n_responses, int) or isinstance(n_responses, bool) or n_responses < 1:
        raise ValueError("n_responses must be a positive integer")
    if is_cli_client(client):
        responses = [
            get_response_from_vlm(
                msg, image_paths, client, model, system_message,
                msg_history=deepcopy(msg_history), temperature=temperature,
                max_images=max_images, schema=schema,
            )
            for _ in range(n_responses)
        ]
        return [item[0] for item in responses], [item[1] for item in responses]

    _require_api_client(client)
    if model in AVAILABLE_VLMS:
        content = prepare_vlm_prompt(msg, image_paths, max_images)

        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        if model.startswith("ollama/"):
            response = client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                seed=0,
            )
        else:
            # Get multiple responses
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                seed=0,
            )

        # Extract content from all responses
        contents = [r.message.content for r in response.choices]
        new_msg_histories = [deepcopy(new_msg_history) + [{"role": "assistant", "content": c}] for c in contents]
        record_token_usage(response, prompt=msg, system_message=system_message)
        if schema is not None:
            from ai_scientist.inference.schema import validate_structured
            for answer in contents:
                validate_structured(extract_json_between_markers(answer), schema)
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        # Just print the first response
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, new_msg_histories
