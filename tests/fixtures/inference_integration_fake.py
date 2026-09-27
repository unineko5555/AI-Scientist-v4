#!/usr/bin/env python3
"""Self-contained offline executable for the full inference integration tests.

This is a protocol fixture, not a replacement model. It echoes explicit inputs so
the tests can check that the production transport lost no history or image bytes.
No SDK, network client, account file, or real CLI is used.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


ARGS = sys.argv[1:]
BACKEND = "claude_cli" if "claude" in Path(sys.argv[0]).name else "codex_cli"


def record(event):
    event.update({"backend": BACKEND, "pid": os.getpid(), "parent_pid": os.getppid()})
    descriptor = os.open(os.environ["INTEGRATION_FAKE_CLI_LOG"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(event) + "\n").encode())
    finally:
        os.close(descriptor)


def option(name):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else None


def emit(value):
    print(json.dumps(value), flush=True)


if "--version" in ARGS:
    record({"event": "inspection", "kind": "version"})
    print("2.1.259 (Claude Code)" if BACKEND == "claude_cli" else "codex-cli 0.157.1")
    sys.exit(0)

if "--help" in ARGS:
    record({"event": "inspection", "kind": "help"})
    print("""--no-daemon --json --ephemeral --ignore-user-config --skip-git-repo-check
--sandbox --color --cd --output-last-message --output-schema --image --strict-config
--disable --input-format --output-format --verbose --no-session-persistence
--system-prompt-file --tools --strict-mcp-config --mcp-config --settings --json-schema""")
    sys.exit(0)

if "status" in ARGS and ("login" in ARGS or "auth" in ARGS):
    record({"event": "inspection", "kind": "auth"})
    if BACKEND == "codex_cli":
        print("Logged in using ChatGPT", file=sys.stderr)
    else:
        emit({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
              "subscriptionType": "max", "email": "FAKE_ACCOUNT_NEVER_PERSIST@example.invalid"})
    sys.exit(0)

stdin = sys.stdin.read()
native_images = []
if BACKEND == "codex_cli":
    history_text = stdin
    for index, value in enumerate(ARGS):
        if value == "--image":
            image = Path(ARGS[index + 1])
            native_images.append((image.read_bytes(), {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif",
            }[image.suffix]))
    settings = [ARGS[index + 1] for index, value in enumerate(ARGS) if value == "-c"]
    system_path = next(json.loads(value.split("=", 1)[1]) for value in settings
                       if value.startswith("model_instructions_file="))
else:
    envelope = json.loads(stdin)
    assert envelope["type"] == "user" and envelope["message"]["role"] == "user"
    parts = envelope["message"]["content"]
    history_text = parts[0]["text"]
    for part in parts:
        if part["type"] == "image":
            source = part["source"]
            assert source["type"] == "base64"
            native_images.append((base64.b64decode(source["data"], validate=True), source["media_type"]))
    system_path = option("--system-prompt-file")

# The production renderer uses a one-line JSON envelope after its instructions;
# image correspondence, when present, is deliberately outside that envelope.
history = json.loads(history_text.splitlines()[1])
assert history["format"] == "ai-scientist.history.v1"
directive = None
for message in history["messages"]:
    if message["role"] != "user":
        continue
    for part in message["content"]:
        if part["type"] == "text":
            try:
                candidate = json.loads(part["text"])
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict) and candidate.get("fixture_request") is True:
                directive = candidate
assert directive is not None, "Every fixture invocation requires an explicit test directive"

identifier = uuid.uuid4().hex
record({"event": "started", "call_id": identifier, "time": time.monotonic(),
        "model": option("--model"), "label": directive.get("label"),
        "environment_keys": sorted(os.environ), "args": ARGS})
time.sleep(directive.get("sleep", 0))
if directive.get("mode") == "quota":
    record({"event": "finished", "call_id": identifier, "time": time.monotonic(), "quota": True})
    print("usage_limit_reached: subscription fixture quota reached", file=sys.stderr)
    sys.exit(1)

occurrences = [part["occurrence_id"] for message in history["messages"]
               for part in message["content"] if part["type"] == "image"]
assert len(occurrences) == len(native_images), "Native attachments must match every history occurrence"
answer = {
    "call_id": identifier,
    "backend": BACKEND,
    "model": option("--model"),
    "history": history["messages"],
    "system": Path(system_path).read_text(),
    "image_sha256": [hashlib.sha256(raw).hexdigest() for raw, _ in native_images],
    "image_mime": [mime for _, mime in native_images],
    "image_occurrences": occurrences,
    "valid_plots_received": False,
    "plot_analyses": [],
}
if directive.get("mode") == "repair" and not any(
    part["type"] == "text" and "The response failed the requested output contract:" in part["text"]
    for message in history["messages"] for part in message["content"]
):
    answer["valid_plots_received"] = True  # Host semantic validation must reject true + no plots.

structured = option("--output-schema") is not None or option("--json-schema") is not None
if BACKEND == "codex_cli" and structured:
    native_schema = json.loads(Path(option("--output-schema")).read_text())
    assert set(native_schema["properties"]) == {"answer_json"}
    native_answer = {"answer_json": json.dumps(answer)}
else:
    native_answer = answer
text = json.dumps(native_answer)

if BACKEND == "codex_cli":
    Path(option("--output-last-message")).write_text(text)
    emit({"type": "thread.started", "thread_id": "offline-" + identifier})
    emit({"type": "turn.started"})
    emit({"type": "item.completed", "item": {"id": "final", "type": "agent_message", "text": text}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 31, "cached_input_tokens": 7, "output_tokens": 11}})
else:
    emit({"type": "system", "subtype": "init", "tools": [], "mcp_servers": [],
          "model": option("--model") or "fixture-default", "session_id": "offline-" + identifier})
    result = {"type": "result", "subtype": "success", "is_error": False, "result": text,
              "usage": {"input_tokens": 31, "cache_read_input_tokens": 7, "output_tokens": 11},
              "stop_reason": "end_turn"}
    if structured:
        result["structured_output"] = native_answer
    emit(result)
record({"event": "finished", "call_id": identifier, "time": time.monotonic()})
