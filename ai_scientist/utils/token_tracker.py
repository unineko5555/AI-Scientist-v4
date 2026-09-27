from functools import wraps
from typing import Dict, Optional, List
from dataclasses import asdict, is_dataclass
from collections import defaultdict
import asyncio
from datetime import datetime
import inspect
import logging
from threading import RLock


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        self.normalized_usage = []
        self._seen_requests = set()
        self._lock = RLock()

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        with self._lock:
            self.token_counts[model]["prompt"] += prompt_tokens
            self.token_counts[model]["completion"] += completion_tokens
            self.token_counts[model]["reasoning"] += reasoning_tokens
            self.token_counts[model]["cached"] += cached_tokens

    def add_normalized_usage(self, backend, model, usage, *, request_id=None):
        """Record provider-neutral usage without treating missing fields as zero.

        This is a local view only. The run broker owns cross-process aggregation.
        Subscription billing is not inferable from CLI usage or cost estimates.
        """
        if is_dataclass(usage):
            usage = asdict(usage)
        elif usage is not None and not isinstance(usage, dict):
            usage = vars(usage)
        usage = dict(usage or {})
        record = {
            "backend": backend,
            "model": model,
            "request_id": request_id,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "input_includes_cached": usage.get("input_includes_cached"),
            "reasoning_tokens": usage.get("reasoning_tokens", usage.get("reasoning_output_tokens")),
            "api_equivalent_cost_usd": usage.get("api_equivalent_cost_usd"),
            "subscription_billing_cost_usd": None,
        }
        with self._lock:
            key = (backend, request_id)
            if request_id is not None and key in self._seen_requests:
                return
            if request_id is not None:
                self._seen_requests.add(key)
            self.normalized_usage.append(record)

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        with self._lock:
            self.interactions[model].append(
                {
                    "system_message": system_message,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": timestamp,
                }
            )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        self.normalized_usage = []
        self._seen_requests = set()

    def calculate_cost(self, model: str) -> float | None:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return None

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        with self._lock:
            api_counts = {model: tokens.copy() for model, tokens in self.token_counts.items()}
            normalized_usage = list(self.normalized_usage)
        for model, tokens in api_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        # CLI costs are deliberately not passed through the historical API
        # price table; an API-equivalent estimate is not a subscription bill.
        for record in normalized_usage:
            key = f"{record['backend']}:{record['model'] or 'default'}"
            entry = summary.setdefault(key, {
                "tokens": {"prompt": None, "completion": None, "reasoning": None, "cached": None},
                "cost (USD)": None,
                "subscription_billing_cost_usd": None,
                "input_includes_cached": record["input_includes_cached"],
                "cache_creation_input_tokens": None,
                "requests": 0,
                "usage_complete": True,
            })
            entry["requests"] += 1
            if record["cache_creation_input_tokens"] is not None:
                entry["cache_creation_input_tokens"] = (entry["cache_creation_input_tokens"] or 0) + record["cache_creation_input_tokens"]
            for target, source in (("prompt", "input_tokens"), ("completion", "output_tokens"),
                                   ("reasoning", "reasoning_tokens"), ("cached", "cached_input_tokens")):
                value = record[source]
                if value is not None:
                    entry["tokens"][target] = (entry["tokens"][target] or 0) + value
                elif target in ("prompt", "completion"):
                    entry["usage_complete"] = False
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def record_token_usage(result, *, prompt=None, system_message=None):
    """Record a raw API SDK response; facade tuples never masquerade as SDK data."""
    usage = _field(result, "usage")
    model = _field(result, "model")
    if usage is None or model is None:
        return
    prompt_count = _field(usage, "prompt_tokens", _field(usage, "input_tokens"))
    completion_count = _field(usage, "completion_tokens", _field(usage, "output_tokens"))
    if prompt_count is None or completion_count is None:
        return
    reasoning = _field(_field(usage, "completion_tokens_details"), "reasoning_tokens", 0) or 0
    cached = _field(_field(usage, "prompt_tokens_details"), "cached_tokens", 0) or 0
    token_tracker.add_tokens(model, prompt_count, completion_count, reasoning, cached)
    choices = _field(result, "choices")
    if choices:
        response = _field(_field(choices[0], "message"), "content")
    else:
        content = _field(result, "content", [])
        response = "\n".join(_field(part, "text", "") for part in content)
    token_tracker.add_interaction(
        model, system_message, prompt, response,
        _field(result, "created", datetime.now()),
    )


def track_token_usage(func):
    """Transparent for positional arguments, absent SDK detail fields and tuples."""
    signature = inspect.signature(func)

    def record(result, args, kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        record_token_usage(result, prompt=bound.get("prompt"), system_message=bound.get("system_message"))

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        record(result, args, kwargs)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        record(result, args, kwargs)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
