"""Subscription-authenticated, stateless official CLI inference transports.

The application never reads OAuth storage and never implements an LLM HTTP API
here.  CLI authentication is checked before inference, and uncertain or API
authentication is rejected.  Capability inspection is *not* a live model test.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Any

from .contracts import InferenceRequest, InferenceResult
from .errors import (
    AuthenticationRequired,
    Cancelled,
    ContextLimitExceeded,
    IncompleteOutput,
    InferenceError,
    InferenceTimeout,
    InvalidStructuredOutput,
    ProviderError,
    QuotaExhausted,
    TransientProviderError,
    UnsupportedCapability,
)
from .history import render_history
from .images import iter_images, materialize_messages
from .process import ProcessResult, run_process


_BACKENDS = {"codex_cli": "codex", "claude_cli": "claude"}
_MIN_VERSIONS = {"codex_cli": (0, 157, 1), "claude_cli": (2, 1, 259)}
_PROTOCOL_PROFILE = "subscription-stateless-v1"
_MAX_ARG_BYTES = 64 * 1024
_CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "hooks", "multi_agent", "multi_agent_v2",
    "memories", "apps", "plugins", "remote_plugin", "recommended_plugins",
    "browser_use", "browser_use_external", "computer_use", "image_generation",
    "code_mode", "code_mode_host", "workspace_dependencies", "skill_search",
    "skill_mcp_dependency_install", "unbounded_connection_retries",
)
_CODEX_REQUIRED_FLAGS = (
    "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
    "--sandbox", "--color", "--cd", "--output-last-message", "--output-schema",
    "--image", "--strict-config", "--disable",
)
_CLAUDE_REQUIRED_FLAGS = (
    "--input-format", "--output-format", "--verbose", "--no-session-persistence",
    "--system-prompt-file", "--tools", "--strict-mcp-config", "--mcp-config",
    "--settings", "--json-schema",
)
# Some documented flags are intentionally omitted by Claude --help. Their
# acceptance is checked on the isolated, non-inference auth-status invocation
# below, rather than treating help as a complete registry. Isolation flags are
# also version-gated: safe-mode (2.1.169), restricted (2.1.248), permission-prompts
# (2.1.259). No fallback removes a required flag if the installed CLI rejects it.
_CLAUDE_AUTH_PROBED_FLAGS = frozenset({"--system-prompt-file"})
_SUBSCRIPTION_TYPES = {"pro", "max", "team", "enterprise"}
_CODEX_DISABLED_CODE_MODE_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. "
    "Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."
)


def subscription_environment(backend: str, source: dict | None = None) -> dict:
    """Copy the environment, removing API/provider overrides only in the child."""
    if backend not in _BACKENDS:
        raise UnsupportedCapability(f"Unknown CLI backend: {backend}")
    env = dict(os.environ if source is None else source)
    protected = {"CODEX_HOME", "CLAUDE_CONFIG_DIR"}
    if backend == "claude_cli":
        protected.add("CLAUDE_CODE_OAUTH_TOKEN")
    prefixes = (
        "OPENAI_", "ANTHROPIC_", "AZURE_OPENAI_", "BEDROCK_", "VERTEX_",
        "AWS_", "GOOGLE_", "CLOUD_ML_", "CODEX_", "CLAUDE_",
        "AI_SCIENTIST_INFERENCE_",
    )
    for key in list(env):
        if key not in protected and key.startswith(prefixes):
            del env[key]
    for key in ("NODE_OPTIONS", "NODE_PATH", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        env.pop(key, None)
    env["NO_COLOR"] = "1"
    if backend == "claude_cli":
        env.update({
            "CLAUDE_CODE_SAFE_MODE": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTO_COMPACT": "1",
            "DISABLE_COMPACT": "1",
        })
    return env


def _private_file(path: Path, content: str | bytes) -> Path:
    data = content.encode("utf-8") if isinstance(content, str) else content
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


def _private_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ProviderError("The CLI request directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ProviderError("The CLI request directory must be owned by this user")
    path.chmod(0o700)
    return path.resolve()


def _number(provider: dict, key: str, default: float) -> float:
    value = provider.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedCapability(f"{key} must be a positive number")
    if value <= 0 or not math.isfinite(value):
        raise UnsupportedCapability(f"{key} must be finite and positive")
    return value


def _integer(provider: dict, key: str, default: int) -> int:
    value = provider.get(key, default)
    if type(value) is not int or value <= 0:
        raise UnsupportedCapability(f"{key} must be a positive integer")
    return value


def _executable(backend: str, provider: dict) -> str:
    if backend not in _BACKENDS:
        raise UnsupportedCapability(f"Unknown CLI backend: {backend}")
    value = provider.get("executable", _BACKENDS[backend])
    if not isinstance(value, str) or not value or "\x00" in value:
        raise UnsupportedCapability("CLI executable must be one path or executable name")
    located = shutil.which(value)
    if located is None:
        raise UnsupportedCapability(f"Install and authenticate the official {_BACKENDS[backend]} CLI first")
    # Resolve once per request so changing cwd cannot select another executable.
    return str(Path(located).absolute())


def _version(text: str, backend: str) -> str:
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", text)
    if match is None:
        raise UnsupportedCapability("The CLI did not report a recognizable version")
    parts = tuple(int(part) for part in match.groups())
    if parts < _MIN_VERSIONS[backend] or parts[0] != _MIN_VERSIONS[backend][0]:
        required = ".".join(map(str, _MIN_VERSIONS[backend]))
        raise UnsupportedCapability(f"{backend} requires the documented {required}+ protocol generation")
    return ".".join(match.groups())


def _claude_profile_files(directory: Path) -> tuple[Path, Path]:
    settings = _private_file(directory / "cli-settings.json", json.dumps({
        "disableAllHooks": True,
        "disableClaudeAiConnectors": True,
        "permissions": {"defaultMode": "dontAsk"},
    }))
    mcp = _private_file(directory / "empty-mcp.json", '{"mcpServers":{}}')
    return settings, mcp


def _claude_isolation(settings: Path, mcp: Path) -> list[str]:
    return [
        "--safe-mode", "--restricted", "--tools", "", "--disallowedTools", "mcp__*",
        "--strict-mcp-config", "--mcp-config", str(mcp), "--settings", str(settings),
        "--permission-prompts", "none",
    ]


def _preflight_command(argv: list[str], directory: Path, env: dict, provider: dict,
                       cancel_event=None, deadline: float | None = None) -> ProcessResult:
    timeout = _number(provider, "preflight_timeout_seconds", 15.0)
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise InferenceTimeout("CLI preflight exceeded the request deadline")
    return run_process(
        argv, cwd=directory, env=env, timeout_seconds=timeout,
        cancel_event=cancel_event, max_output_bytes=1024 * 1024,
        termination_grace_seconds=_number(provider, "termination_grace_seconds", 2.0),
        drain_timeout_seconds=_number(provider, "drain_timeout_seconds", 1.0),
        ownership_path=directory / "process.json",
    )


def _inspect(backend: str, provider: dict, directory: Path, env: dict,
             cancel_event=None, deadline=None, claude_files=None) -> dict:
    executable = _executable(backend, provider)

    def probe(args):
        return _preflight_command([executable, *args], directory, env, provider,
                                  cancel_event, deadline)

    version_result = probe(["--version"])
    if version_result.returncode:
        _raise_provider_failure(version_result, "CLI version inspection failed")
    version = _version(version_result.stdout.decode("utf-8", "replace"), backend)
    help_result = probe(["exec", "--help"] if backend == "codex_cli" else ["--help"])
    if help_result.returncode:
        _raise_provider_failure(help_result, "CLI capability inspection failed")
    help_text = help_result.stdout.decode("utf-8", "replace")
    flags = _CODEX_REQUIRED_FLAGS if backend == "codex_cli" else _CLAUDE_REQUIRED_FLAGS
    auth_probed_flags = _CLAUDE_AUTH_PROBED_FLAGS if backend == "claude_cli" else frozenset()
    missing = [flag for flag in flags if flag not in help_text and flag not in auth_probed_flags]
    if missing:
        raise UnsupportedCapability("Required CLI flags unavailable: " + ", ".join(missing))

    if backend == "codex_cli":
        root_help = probe(["--help"])
        if root_help.returncode or "--no-daemon" not in root_help.stdout.decode("utf-8", "replace"):
            raise UnsupportedCapability("This Codex version cannot isolate the owned CLI from its shared daemon")
        # Do not enforce a different login method while *inspecting* auth:
        # Codex documents that a forced-login mismatch can log the user out.
        # Inspect existing status, reject API auth in the host, and enforce the
        # subscription method only on the actual inference invocation below.
        auth = probe(["login", "status"])
        # Never return/log auth text: some versions include identifying details.
        status = (auth.stdout + b"\n" + auth.stderr).decode("utf-8", "replace").strip().lower()
        if auth.returncode or "logged in using chatgpt" not in status or "api key" in status:
            raise AuthenticationRequired("Codex must report ChatGPT subscription login; run codex login yourself")
        authentication = {"method": "chatgpt", "subscription": True}
    else:
        settings, mcp = claude_files or _claude_profile_files(directory)
        inspection_system = _private_file(
            directory / "preflight-system.txt", "Non-inference CLI capability inspection.\n",
        )
        # The same isolation flags and child environment apply before inference.
        # Documented system-prompt-file may be hidden from --help. Verify its
        # argument acceptance using only a private harmless file, not user input.
        # Global options MUST precede auth: Claude rejects them after a subcommand.
        # Unknown flags fail here; they are never removed to make auth succeed.
        auth = probe([
            *_claude_isolation(settings, mcp), "--system-prompt-file", str(inspection_system),
            "auth", "status",
        ])
        if auth.returncode:
            if _looks_like_unsupported((auth.stdout + b"\n" + auth.stderr).decode("utf-8", "replace")):
                raise UnsupportedCapability("Claude auth status cannot verify this isolated subscription profile")
            raise AuthenticationRequired("Claude subscription login could not be verified; run claude auth login yourself")
        try:
            status = _strict_json(auth.stdout)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise AuthenticationRequired("Claude auth status did not return verifiable JSON") from exc
        if not isinstance(status, dict):
            raise AuthenticationRequired("Claude auth status returned an unknown format")
        plan = str(status.get("subscriptionType", "")).lower()
        if (status.get("loggedIn") is not True
                or status.get("authMethod") != "claude.ai"
                or status.get("apiProvider") != "firstParty"
                or plan not in _SUBSCRIPTION_TYPES):
            raise AuthenticationRequired("Claude must report first-party claude.ai subscription authentication")
        authentication = {"method": "claude.ai", "subscription": True, "plan": plan}

    return {
        "backend": backend, "executable": executable, "version": version,
        "authentication": authentication, "protocol_profile": _PROTOCOL_PROFILE,
        "capabilities": {"text": "declared", "images": "declared", "json_schema": "declared"},
        "live_inference_verified": False,
        "warnings": [
            "Read-only CLI inspection does not prove live image/schema/model compatibility.",
            "Managed policy and tool isolation require the documented live deployment checks.",
            "Subscription extra usage/account spending policy is controlled by the account owner.",
        ],
    }


def inspect_provider(backend: str, provider: dict) -> dict:
    """Inspect binary/version/flags/auth without sending any inference request.

    CLI auth status is called, not token storage. Only sanitized status fields
    leave this function. A private temporary directory is removed on return.
    """
    with tempfile.TemporaryDirectory(prefix="ai-scientist-cli-check-") as location:
        directory = _private_directory(Path(location))
        return _inspect(backend, provider, directory, subscription_environment(backend))


def _looks_like_unsupported(message: str) -> bool:
    text = message.lower()
    return any(value in text for value in (
        "unknown option", "unexpected argument", "unrecognized option", "unknown field",
        "unsupported parameter", "unknown feature", "unrecognized subcommand",
    ))


def _classified_failure(message: str, default: str) -> InferenceError:
    text = message.lower()
    if _looks_like_unsupported(message):
        return UnsupportedCapability("The CLI rejected the required isolation or capability profile")
    if any(value in text for value in (
        "context_length_exceeded", "context window", "prompt is too long", "input is too long",
        "maximum context", "too many tokens", "image limit", "too many images",
    )):
        return ContextLimitExceeded("The CLI rejected the complete input because it exceeds a model limit")
    if any(value in text for value in (
        "usage_limit_reached", "insufficient_quota", "quota_exceeded", "usage limit",
        "quota exceeded", "limit reached", "out of extra usage", "credit balance",
        "hit your limit", "subscription limit",
    )):
        return QuotaExhausted("The CLI reported exhausted subscription usage")
    if any(value in text for value in (
        "authentication_error", "unauthorized", "not logged in", "login required",
        "invalid api key", "invalid bearer token", "token expired", "please run /login",
        "expired token", "authentication failed",
    )):
        return AuthenticationRequired("The CLI reported an authentication failure")
    if any(value in text for value in (
        "overloaded_error", "rate_limit_error", "rate limit", "temporarily unavailable",
        "connection reset", "connection refused", "service unavailable", "network error",
    )):
        return TransientProviderError("The CLI reported a temporary provider/connection failure")
    return ProviderError(default)


def _raise_provider_failure(result: ProcessResult, default: str):
    raise _classified_failure((result.stdout + b"\n" + result.stderr).decode("utf-8", "replace"), default)


def _jsonl(raw: bytes) -> list[dict]:
    try:
        lines = raw.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError as exc:
        raise IncompleteOutput("CLI stdout was not UTF-8 JSONL") from exc
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = _strict_json(line)
        except (ValueError, RecursionError) as exc:
            raise IncompleteOutput("CLI stdout contained non-JSON protocol output") from exc
        if not isinstance(event, dict):
            raise IncompleteOutput("CLI output event was not an object")
        if not isinstance(event.get("type"), str):
            raise IncompleteOutput("CLI output event had no valid type")
        if "subtype" in event and not isinstance(event["subtype"], str):
            raise IncompleteOutput("CLI output event had an invalid subtype")
        events.append(event)
    return events


def _strict_json(text: str | bytes):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON constant")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def _compaction_event(event: dict) -> bool:
    # Inspect protocol labels, never answer text: a paper may discuss compaction.
    item = event.get("item", {})
    labels = [event.get("type", ""), event.get("subtype", "")]
    if isinstance(item, dict):
        labels += [item.get("type", "")]
    return any("compact" in str(label).lower() for label in labels)


def _numeric(value: Any):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _usage(raw: Any, backend: str) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "input_tokens": _numeric(raw.get("input_tokens")),
        "output_tokens": _numeric(raw.get("output_tokens")),
        "cached_input_tokens": _numeric(raw.get("cached_input_tokens", raw.get("cache_read_input_tokens"))),
        "cache_creation_input_tokens": _numeric(raw.get("cache_creation_input_tokens")),
        "reasoning_output_tokens": _numeric(raw.get("reasoning_output_tokens")),
        "input_includes_cached": backend == "codex_cli",
        "cost_usd": None, "billing": "subscription",
    }


def _read_final(path: Path, limit: int) -> str:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise IncompleteOutput("Codex final output is not a bounded regular file")
            os.fchmod(handle.fileno(), 0o600)
            data = handle.read(limit + 1)
    except FileNotFoundError as exc:
        raise IncompleteOutput("Codex did not create its final-answer artifact") from exc
    except OSError as exc:
        raise IncompleteOutput("Codex final-answer artifact could not be safely opened") from exc
    if len(data) > limit:
        raise IncompleteOutput("Codex final output exceeded the byte limit")
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise IncompleteOutput("Codex final output was not UTF-8") from exc
    if not text.strip():
        raise IncompleteOutput("Codex returned an empty final answer")
    return text


def _parse_codex(result: ProcessResult, final_path: Path, request: InferenceRequest,
                 output_limit: int) -> tuple[str, Any, dict, dict]:
    if result.returncode:
        _raise_provider_failure(result, "Codex exited unsuccessfully")
    events = _jsonl(result.stdout)
    terminal = None
    final_messages = []
    details = {}
    for event in events:
        kind = event.get("type")
        if _compaction_event(event):
            raise ContextLimitExceeded("Codex compacted the supplied history; the response is not accepted")
        if kind in {"turn.failed", "error"}:
            raise _classified_failure(json.dumps(event), "Codex reported a failed turn")
        if terminal is not None:
            raise IncompleteOutput("Codex emitted protocol events after its terminal response")
        if kind == "turn.completed":
            terminal = event
        elif kind == "thread.started":
            details["thread_id"] = event.get("thread_id")
        elif kind == "turn.started":
            continue
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                raise IncompleteOutput("Codex returned a malformed item")
            item_type = item.get("type")
            if item_type == "error":
                message = item.get("message")
                if not isinstance(message, str):
                    raise IncompleteOutput("Codex returned a malformed error item")
                # Observed on official Codex 0.157.1 with both code_mode and
                # code_mode_host intentionally disabled. This exact diagnostic
                # confirms fail-closed isolation; it is not a failed model turn.
                # No other error item, prefix match, or top-level error is waived.
                if kind == "item.completed" and message == _CODEX_DISABLED_CODE_MODE_NOTICE:
                    details.setdefault("isolation_notices", []).append(message)
                    continue
                raise _classified_failure(message, "Codex reported an error item")
            if item_type not in {"agent_message", "reasoning"}:
                # Even a blocked tool call is evidence that this isn't the
                # required pure-inference profile, not material for the paper.
                raise UnsupportedCapability("Codex emitted a forbidden tool or unknown item type")
            if kind == "item.completed" and item_type == "agent_message":
                message = item.get("text")
                if not isinstance(message, str):
                    raise IncompleteOutput("Codex assistant message had no text")
                final_messages.append(message)
        else:
            raise UnsupportedCapability("Codex emitted an unknown protocol event")
    if terminal is None or not final_messages:
        raise IncompleteOutput("Codex ended without a completed turn and final assistant message")
    text = _read_final(final_path, output_limit)
    if text.rstrip("\n") != final_messages[-1].rstrip("\n"):
        raise IncompleteOutput("Codex final artifact and final assistant event disagree")
    structured = None
    if request.schema is not None:
        try:
            structured = _strict_json(text)
        except (ValueError, RecursionError) as exc:
            raise InvalidStructuredOutput("Codex did not return the requested JSON output") from exc
    details["terminal_event"] = "turn.completed"
    return text, structured, _usage(terminal.get("usage"), "codex_cli"), details


def _parse_claude(result: ProcessResult, request: InferenceRequest) -> tuple[str, Any, dict, dict]:
    if result.returncode:
        _raise_provider_failure(result, "Claude exited unsuccessfully")
    events = _jsonl(result.stdout)
    terminal = None
    details = {}
    internal_tool_ids = set()
    for event in events:
        kind = event.get("type")
        if _compaction_event(event):
            raise ContextLimitExceeded("Claude compacted the supplied history; the response is not accepted")
        if terminal is not None:
            raise IncompleteOutput("Claude emitted protocol events after its terminal response")
        if kind == "result":
            terminal = event
        elif kind == "system" and event.get("subtype") == "init":
            # Native structured output may advertise its internal schema tool.
            allowed = {"StructuredOutput"} if request.schema is not None else set()
            tools = event.get("tools", [])
            if not isinstance(tools, list) or any(not isinstance(tool, str) or tool not in allowed for tool in tools):
                raise UnsupportedCapability("Claude advertised forbidden operation tools")
            if event.get("mcp_servers"):
                raise UnsupportedCapability("Claude loaded MCP servers in the isolated profile")
            details.update({"session_id": event.get("session_id"), "actual_model": event.get("model")})
        elif kind == "assistant":
            message = event.get("message")
            if not isinstance(message, dict):
                raise IncompleteOutput("Claude returned a malformed assistant message")
            content = message.get("content", [])
            if not isinstance(content, list) or not all(isinstance(part, dict) for part in content):
                raise IncompleteOutput("Claude returned malformed assistant content")
            for part in content:
                part_type = part.get("type")
                if part_type == "tool_use" and request.schema is not None and part.get("name") == "StructuredOutput":
                    tool_id = part.get("id")
                    if not isinstance(tool_id, str) or not tool_id:
                        raise IncompleteOutput("Claude schema tool had no identity")
                    internal_tool_ids.add(tool_id)
                elif part_type not in {"text", "thinking", "redacted_thinking"}:
                    raise UnsupportedCapability("Claude attempted a forbidden operation tool")
            if event.get("error"):
                raise _classified_failure(json.dumps(event), "Claude reported an assistant error")
        elif kind == "user":
            # Internal structured-output tool acknowledgment is not a new user
            # request; all other tool results violate the profile.
            message = event.get("message")
            if not isinstance(message, dict):
                raise IncompleteOutput("Claude returned a malformed user message")
            content = message.get("content", [])
            if not isinstance(content, list) or not all(isinstance(part, dict) for part in content):
                raise IncompleteOutput("Claude returned malformed user content")
            for part in content:
                if part.get("type") != "tool_result" or part.get("tool_use_id") not in internal_tool_ids:
                    raise UnsupportedCapability("Claude returned a result for an unrecognized tool")
        elif kind == "rate_limit_event":
            info = event.get("rate_limit_info", {})
            if not isinstance(info, dict):
                raise IncompleteOutput("Claude returned malformed rate limit metadata")
            if info.get("status") in {"rejected", "rate_limited"}:
                raise QuotaExhausted("Claude reported exhausted subscription usage")
        else:
            raise UnsupportedCapability("Claude emitted an unknown or disallowed protocol event")
    if terminal is None:
        raise IncompleteOutput("Claude ended without a final result envelope")
    if terminal.get("is_error") is not False or terminal.get("subtype") != "success":
        message = json.dumps(terminal)
        if "error_max_structured_output_retries" in message:
            raise InvalidStructuredOutput("Claude exhausted native structured-output retries")
        if terminal.get("subtype") in {"error_max_turns", "error_max_budget_usd"}:
            raise IncompleteOutput("Claude stopped at a turn or budget limit")
        raise _classified_failure(message, "Claude reported an unsuccessful final result")
    terminal_reason = terminal.get("terminal_reason")
    if terminal_reason is not None and not isinstance(terminal_reason, str):
        raise IncompleteOutput("Claude reported a malformed terminal reason")
    if terminal_reason in {"aborted_streaming", "aborted_tools", "cancelled", "interrupted"}:
        raise Cancelled("Claude reported an interrupted turn")
    if terminal_reason not in {None, "completed"}:
        raise IncompleteOutput("Claude reported an unrecognized or incomplete terminal reason")
    if terminal.get("permission_denials"):
        raise UnsupportedCapability("Claude attempted an operation requiring permission")
    allowed_stops = {None, "end_turn", "stop_sequence"}
    # Claude's native schema result is an internal StructuredOutput tool call,
    # not permission to execute general tools. Success is still envelope-gated.
    if request.schema is not None and "structured_output" in terminal:
        allowed_stops.add("tool_use")
    stop_reason = terminal.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise IncompleteOutput("Claude reported a malformed stop reason")
    if stop_reason not in allowed_stops:
        raise IncompleteOutput("Claude output was truncated or stopped for an unsupported reason")
    structured = terminal.get("structured_output") if request.schema is not None else None
    if request.schema is not None:
        if "structured_output" not in terminal:
            raise InvalidStructuredOutput("Claude did not return native structured_output")
        text = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
    else:
        text = terminal.get("result")
        if not isinstance(text, str) or not text.strip():
            raise IncompleteOutput("Claude returned an empty or non-text final answer")
    details["terminal_event"] = "result.success"
    details["stop_reason"] = terminal.get("stop_reason")
    details["api_equivalent_cost_usd"] = _numeric(terminal.get("total_cost_usd"))
    if isinstance(terminal.get("modelUsage"), dict):
        details["reported_models"] = sorted(terminal["modelUsage"])
    return text, structured, _usage(terminal.get("usage"), "claude_cli"), details


def _codex_command(executable: str, request: InferenceRequest, directory: Path,
                   system_file: Path, images: list[dict]) -> list[str]:
    argv = [executable, "--no-daemon", "exec", "--json", "--ephemeral",
            "--ignore-user-config", "--strict-config", "--skip-git-repo-check",
            "--sandbox", "read-only", "--color", "never", "--cd", str(directory)]
    settings = {
        "forced_login_method": "chatgpt", "model_provider": "openai",
        "approval_policy": "never", "model_instructions_file": str(system_file),
        "project_doc_max_bytes": 0, "web_search": "disabled",
        "apps._default.enabled": False, "history.persistence": "none",
    }
    for key, value in settings.items():
        argv += ["-c", key + "=" + json.dumps(value, ensure_ascii=False)]
    for feature in _CODEX_DISABLED_FEATURES:
        argv += ["--disable", feature]
    argv += ["--output-last-message", str(directory / "final.txt")]
    if request.model is not None:
        argv += ["--model", request.model]
    for image in images:
        argv += ["--image", image["path"]]
    if request.schema is not None:
        schema_file = _private_file(directory / "output-schema.json", json.dumps(request.schema, ensure_ascii=False))
        argv += ["--output-schema", str(schema_file)]
    return [*argv, "-"]


def _claude_input(history: str, images: list[dict]) -> bytes:
    content = [{"type": "text", "text": history}]
    for image in images:
        # Keep occurrences separate even when the underlying bytes are equal.
        content.append({"type": "text", "text": "Attached image occurrence: " + image["occurrence_id"]})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": image["mime_type"],
            "data": base64.b64encode(Path(image["path"]).read_bytes()).decode("ascii"),
        }})
    envelope = {"type": "user", "message": {"role": "user", "content": content},
                "parent_tool_use_id": None}
    return (json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _execute(request: InferenceRequest, provider: dict, request_dir: Path,
             cancel_event=None) -> InferenceResult:
    """Perform exactly one CLI inference request; retry policy belongs to gateway."""
    started = time.monotonic()
    deadline = started + request.timeout_seconds
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("CLI request cancelled before preflight")
    if request.model is not None and (request.model.startswith("-") or "\x00" in request.model):
        raise UnsupportedCapability("Model must not be a CLI option or contain NUL")
    directory = _private_directory(Path(request_dir))
    env = subscription_environment(request.backend)
    claude_files = _claude_profile_files(directory) if request.backend == "claude_cli" else None
    inspection = _inspect(request.backend, provider, directory, env, cancel_event, deadline, claude_files)
    materialized_limit = _integer(provider, "max_materialized_bytes", 32 * 1024 * 1024)
    normalized = materialize_messages(request.messages, directory, max_bytes=materialized_limit)
    images = list(iter_images(normalized))
    max_images = _integer(provider, "max_images", 200)
    if len(images) > max_images:
        raise ContextLimitExceeded("Request exceeds the configured image count; no images were dropped")
    history = render_history(normalized)
    if images:
        correspondence = [{"attachment_index": i + 1, "occurrence_id": image["occurrence_id"]}
                          for i, image in enumerate(images)]
        history += "\nImage attachment correspondence (ordered occurrences):\n" + json.dumps(correspondence)
    text_limit = _integer(provider, "max_input_text_bytes", 512 * 1024)
    if len(history.encode("utf-8")) + len(request.system.encode("utf-8")) > text_limit:
        raise ContextLimitExceeded("Complete text input exceeds the configured byte limit; no history was truncated")
    system_file = _private_file(directory / "system.txt", request.system)
    executable = inspection["executable"]
    if request.backend == "codex_cli":
        argv = _codex_command(executable, request, directory, system_file, images)
        stdin = history.encode("utf-8")
    else:
        settings, mcp = claude_files
        argv = [executable, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--no-session-persistence", "--system-prompt-file", str(system_file),
                *_claude_isolation(settings, mcp)]
        if request.model is not None:
            argv += ["--model", request.model]
        if request.schema is not None:
            argv += ["--json-schema", json.dumps(request.schema, ensure_ascii=False, separators=(",", ":"))]
        stdin = _claude_input(history, images)
    if sum(len(arg.encode("utf-8")) + 1 for arg in argv) > _MAX_ARG_BYTES:
        raise ContextLimitExceeded("CLI command metadata/schema exceeds the safe argv limit")
    if len(stdin) > _integer(provider, "max_stdin_bytes", 64 * 1024 * 1024):
        raise ContextLimitExceeded("Complete CLI input exceeds the configured byte limit")
    output_limit = _integer(provider, "max_output_bytes", 16 * 1024 * 1024)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InferenceTimeout("CLI request deadline expired during preflight")
    result = run_process(
        argv, input_bytes=stdin, cwd=directory, env=env, timeout_seconds=remaining,
        cancel_event=cancel_event, max_output_bytes=output_limit,
        termination_grace_seconds=_number(provider, "termination_grace_seconds", 2.0),
        drain_timeout_seconds=_number(provider, "drain_timeout_seconds", 1.0),
        stdout_path=directory / "stdout.jsonl", stderr_path=directory / "stderr.log",
        ownership_path=directory / "process.json",
    )
    if request.backend == "codex_cli":
        text, structured, usage, details = _parse_codex(result, directory / "final.txt", request, output_limit)
    else:
        text, structured, usage, details = _parse_claude(result, request)
    unsupported = {}
    if request.temperature is not None:
        unsupported["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        unsupported["max_output_tokens"] = request.max_output_tokens
    return InferenceResult(
        request_id=request.request_id, text=text, structured_data=structured,
        backend=request.backend, model=request.model, usage=usage,
        duration_seconds=time.monotonic() - started,
        metadata={
            **details, "cli_version": inspection["version"],
            "protocol_profile": _PROTOCOL_PROFILE,
            "authentication": inspection["authentication"],
            "live_capability_preverified": False,
            "unsupported_parameters": unsupported,
            "effective_parameters": {}, "history_mode": "explicit_stateless",
            "compaction_policy": "reject_detected; no blanket no-compaction guarantee",
            "image_occurrences": [{"occurrence_id": i["occurrence_id"], "mime_type": i["mime_type"]} for i in images],
            "stdout_path": str(directory / "stdout.jsonl"), "stderr_path": str(directory / "stderr.log"),
        },
    )


def execute(request: InferenceRequest, provider: dict, request_dir: Path,
            cancel_event=None) -> InferenceResult:
    """Execute with typed, request-correlated errors and no credential output."""
    try:
        return _execute(request, provider, request_dir, cancel_event)
    except InferenceError as exc:
        if exc.request_id is None:
            exc.request_id = request.request_id
        exc.metadata.setdefault("backend", request.backend)
        raise
    except OSError as exc:
        raise ProviderError(
            "Cannot access private CLI request artifacts",
            request_id=request.request_id,
            metadata={"backend": request.backend, "phase": "local_io"},
        ) from exc
