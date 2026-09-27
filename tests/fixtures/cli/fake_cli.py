#!/usr/bin/env python3
"""Offline stand-in for executable-level transport contract tests only.

Production code has no knowledge of these fixture variables and no test switch.
The test creates a private executable with this source and its Python shebang.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


args = sys.argv[1:]
backend = "claude" if "claude" in Path(sys.argv[0]).name else "codex"
scenario = os.environ.get("FAKE_CLI_SCENARIO", "success")
log = os.environ.get("FAKE_CLI_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"args": args, "cwd": os.getcwd()}) + "\n")

if "--version" in args:
    default = "2.1.259 (Claude Code)" if backend == "claude" else "codex-cli 0.157.1"
    print(os.environ.get("FAKE_CLI_VERSION", default))
    sys.exit(0)

if "--help" in args:
    flags = """--no-daemon --json --ephemeral --ignore-user-config --skip-git-repo-check
--sandbox --color --cd --output-last-message --output-schema --image --strict-config
--disable --input-format --output-format --verbose --no-session-persistence
--system-prompt-file --tools --strict-mcp-config --mcp-config --settings --json-schema"""
    if scenario == "missing_flag":
        flags = flags.replace("--image", "").replace("--tools", "")
    if scenario in {"hidden_system_prompt_file", "rejected_system_prompt_file", "rejected_system_prompt_file_stdout"}:
        flags = flags.replace("--system-prompt-file", "")
    print(flags)
    sys.exit(0)

if "status" in args and ("login" in args or "auth" in args):
    if backend == "claude":
        # Official Claude treats these as root/global options, not auth options.
        trailing_options = [arg for arg in args[args.index("auth") + 2:] if arg.startswith("--")]
        if trailing_options:
            print(f"unknown option '{trailing_options[0]}'", file=sys.stderr)
            sys.exit(2)
        if scenario in {"rejected_system_prompt_file", "rejected_system_prompt_file_stdout"} and "--system-prompt-file" in args:
            stream = sys.stdout if scenario.endswith("_stdout") else sys.stderr
            print("unknown option '--system-prompt-file'", file=stream)
            sys.exit(2)
        if "--system-prompt-file" in args:
            # Actually open the supplied harmless file before reporting status.
            Path(args[args.index("--system-prompt-file") + 1]).read_text()
    if scenario == "auth_unsupported":
        print("unknown option '--restricted'", file=sys.stderr)
        sys.exit(2)
    if backend == "codex":
        if any("forced_login_method" in argument for argument in args):
            marker = os.environ.get("FAKE_CLI_LOGOUT_MARKER")
            if marker:
                Path(marker).write_text("login policy mismatch cleared credentials")
            sys.exit(1)
        print("Logged in using an API key. PRIVATE_SECRET" if scenario == "auth_api" else "Logged in using ChatGPT", file=sys.stderr)
    else:
        print(json.dumps({
            "loggedIn": scenario != "auth_logged_out",
            "authMethod": "api_key" if scenario == "auth_api" else "claude.ai",
            "apiProvider": "firstParty", "subscriptionType": None if scenario == "auth_unknown_plan" else "max",
            "email": "PRIVATE_IDENTITY@example.invalid",
        }))
    sys.exit(0)

data = sys.stdin.read()
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stdin": data, "environment_keys": sorted(os.environ),
                                 "home": os.environ.get("HOME")}) + "\n")

if scenario == "hang":
    time.sleep(60)
if scenario == "stdout_flood":
    sys.stdout.write("x" * 200000)
    sys.stdout.flush()
    time.sleep(60)
if scenario == "quota_exit":
    print("usage_limit_reached: hit your limit", file=sys.stderr)
    sys.exit(1)
if scenario == "auth_expired":
    print("authentication_error: token expired", file=sys.stderr)
    sys.exit(1)
if scenario == "unknown_flag_exit":
    print("unknown option --safe-mode", file=sys.stderr)
    sys.exit(2)
if scenario == "network_exit":
    print("connection reset", file=sys.stderr)
    sys.exit(1)
if scenario == "context_exit":
    print("context_length_exceeded", file=sys.stderr)
    sys.exit(1)
if scenario == "nonzero_with_final":
    if "--output-last-message" in args:
        Path(args[args.index("--output-last-message") + 1]).write_text("partial")
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "partial"}))
    sys.exit(1)

schema = "--json-schema" in args or "--output-schema" in args
answer = '{"valid":false,"analyses":[]}' if schema else "FINAL ANSWER — 日本語"
if scenario == "invalid_json":
    answer = "not JSON"

if backend == "codex":
    if scenario != "missing_final":
        Path(args[args.index("--output-last-message") + 1]).write_text("disagrees" if scenario == "mismatch" else answer)
    print(json.dumps({"type": "thread.started", "thread_id": "fixture-thread"}))
    print(json.dumps({"type": "turn.started"}))
    print(json.dumps({"type": "item.completed", "item": {"id": "reason", "type": "reasoning", "text": "NOT THE ANSWER"}}))
    if scenario == "tool":
        print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "ignored"}}))
    if scenario == "compaction":
        print(json.dumps({"type": "context.compacted"}))
    if scenario == "unknown_event":
        print(json.dumps({"type": "future.unknown"}))
    print(json.dumps({"type": "item.completed", "item": {"id": "final", "type": "agent_message", "text": answer}}))
    if scenario == "quota_event":
        print(json.dumps({"type": "turn.failed", "error": {"message": "usage_limit_reached"}}))
    elif scenario != "missing_terminal":
        print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 123, "cached_input_tokens": 23, "output_tokens": 17}}))
else:
    print(json.dumps({"type": "system", "subtype": "init", "tools": ["Bash"] if scenario == "tool" else [], "mcp_servers": [], "model": "fixture-model", "session_id": "fixture-session"}))
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "NOT THE ANSWER"}]}}))
    if scenario == "structured_tool":
        print(json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "schema-tool-1", "name": "StructuredOutput", "input": {"valid": False, "analyses": []}}]}}))
        print(json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "schema-tool-1", "content": "valid"}]}}))
    if scenario == "unrecognized_tool_result":
        print(json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "bash", "content": "forbidden"}]}}))
    if scenario == "malformed_assistant":
        print(json.dumps({"type": "assistant", "message": "not an object"}))
    if scenario == "compaction":
        print(json.dumps({"type": "system", "subtype": "compact_boundary"}))
    if scenario == "unknown_event":
        print(json.dumps({"type": "future.unknown"}))
    result = {"type": "result", "subtype": "success", "is_error": False,
              "result": answer, "usage": {"input_tokens": 100, "cache_read_input_tokens": 23, "output_tokens": 17},
              "total_cost_usd": 0.1, "stop_reason": "end_turn"}
    if schema and scenario != "missing_structured":
        result["structured_output"] = {"valid": False, "analyses": []}
    if scenario == "quota_event":
        result.update({"is_error": True, "subtype": "error_during_execution", "errors": ["usage limit reached"]})
    if scenario == "truncated":
        result["stop_reason"] = "max_tokens"
    if scenario == "structured_tool":
        result["stop_reason"] = "tool_use"
        result["terminal_reason"] = "completed"
    if scenario == "aborted_success":
        result["terminal_reason"] = "aborted_streaming"
    if scenario == "unknown_terminal_reason":
        result["terminal_reason"] = "future_unknown"
    if scenario == "permission_denial":
        result["permission_denials"] = [{"tool_name": "Bash"}]
    if scenario == "missing_stop_reason":
        result.pop("stop_reason")
    if scenario == "schema_retry_error":
        result.update({"is_error": True, "subtype": "error_max_structured_output_retries"})
    if scenario != "missing_terminal":
        print(json.dumps(result))

if scenario == "garbage_tail":
    print("not-json")
if scenario == "duplicate_keys":
    print('{"type":"unknown","type":"turn.completed"}')
