"""Child-process environment boundaries for generated research code."""

from __future__ import annotations

import os
from collections.abc import Mapping


def generated_code_environment(
    source: Mapping[str, str] | None = None, *, subscription_only: bool = True,
) -> dict[str, str]:
    """Return a copy with inference authority removed in subscription-only mode.

    Keep legacy API experiments unchanged. CUDA settings, PATH, dataset cache
    locations and AI_SCIENTIST_ROOT remain available. This is not an OS sandbox:
    generated code retains its user's filesystem and network permissions.
    """
    environment = dict(os.environ if source is None else source)
    if not subscription_only:
        return environment
    prefixes = (
        "OPENAI_", "ANTHROPIC_", "AZURE_OPENAI_", "CODEX_", "CLAUDE_",
        "BEDROCK_", "VERTEX_", "AWS_", "GOOGLE_", "CLOUD_ML_",
        "DEEPSEEK_", "OPENROUTER_", "GEMINI_", "MISTRAL_", "COHERE_",
        "GROQ_", "TOGETHER_", "AI_SCIENTIST_INFERENCE_",
    )
    credentials = {
        "HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_API_KEY", "OLLAMA_API_KEY",
    }
    return {
        key: value for key, value in environment.items()
        if not key.startswith(prefixes) and key not in credentials
    }
