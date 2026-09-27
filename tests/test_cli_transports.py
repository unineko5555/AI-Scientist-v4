"""Executable-level CLI tests. Never invoke a real CLI or read credentials."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from ai_scientist.inference.cli import _parse_codex, execute, inspect_provider, subscription_environment
from ai_scientist.inference.contracts import InferenceRequest
from ai_scientist.inference.errors import (
    AuthenticationRequired,
    Cancelled,
    ContextLimitExceeded,
    IncompleteOutput,
    InferenceTimeout,
    InvalidStructuredOutput,
    ProviderError,
    QuotaExhausted,
    TransientProviderError,
    UnsupportedCapability,
)
from ai_scientist.inference.process import ProcessResult, cleanup_owned_process, run_process


@pytest.fixture(params=["codex_cli", "claude_cli"])
def fake_provider(request, tmp_path, monkeypatch):
    backend = request.param
    executable = tmp_path / ("fake-claude" if backend == "claude_cli" else "fake-codex")
    fixture = Path(__file__).parent / "fixtures" / "cli" / "fake_cli.py"
    source = fixture.read_text().split("\n", 1)[1]
    executable.write_text(f"#!{sys.executable}\n" + source)
    executable.chmod(0o700)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    return backend, {"executable": str(executable), "termination_grace_seconds": 0.1}, log


def make_request(backend, **kwargs):
    return InferenceRequest(
        request_id="request-1", run_id="run-1", role="review_text", backend=backend,
        system="Be a careful reviewer.", messages=[{"role": "user", "content": "paper"}],
        timeout_seconds=5, **kwargs,
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_text_success_uses_only_final_and_subscription_usage(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    result = execute(make_request(backend), provider, tmp_path / "request")
    assert result.text == "FINAL ANSWER — 日本語"
    assert "NOT THE ANSWER" not in result.text
    assert result.usage["output_tokens"] == 17
    assert result.usage["cost_usd"] is None
    assert result.usage["billing"] == "subscription"
    assert result.metadata["authentication"]["subscription"] is True
    assert result.metadata["live_capability_preverified"] is False
    calls = records(log)
    args = next(call["args"] for call in calls if "-p" in call.get("args", []) or "exec" in call.get("args", []) and "--help" not in call["args"])
    assert "--model" not in args
    assert "--resume" not in args and "--continue" not in args
    assert "--dangerously-skip-permissions" not in args
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert "--bare" not in args
    assert "paper" not in " ".join(args)
    assert (tmp_path / "request").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "request" / "system.txt").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "request" / "stdout.jsonl").stat().st_mode & 0o777 == 0o600


def test_schema_false_evaluation_is_preserved(fake_provider, tmp_path):
    backend, provider, _ = fake_provider
    schema = {"type": "object", "properties": {"valid": {"type": "boolean"}, "analyses": {"type": "array"}}, "required": ["valid", "analyses"]}
    result = execute(make_request(backend, schema=schema), provider, tmp_path / "request")
    assert result.structured_data == {"valid": False, "analyses": []}
    assert json.loads(result.text) == result.structured_data


def test_default_model_omitted_and_explicit_model_is_one_argument(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    model = "custom-model; touch NOT_EXECUTED"
    execute(make_request(backend, model=model), provider, tmp_path / "request")
    args = next(r["args"] for r in records(log) if "--model" in r.get("args", []))
    assert args[args.index("--model") + 1] == model
    assert not (tmp_path / "request" / "NOT_EXECUTED").exists()


def test_inspection_does_not_infer_or_expose_auth_identity(fake_provider):
    backend, provider, log = fake_provider
    info = inspect_provider(backend, provider)
    assert info["live_inference_verified"] is False
    assert all("stdin" not in record for record in records(log))
    assert "PRIVATE_IDENTITY" not in json.dumps(info)


@pytest.mark.parametrize("operation", ["inspect", "execute"])
def test_claude_hidden_system_prompt_flag_is_actively_verified(fake_provider, tmp_path, monkeypatch, operation):
    backend, provider, log = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude hidden global flags")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", "hidden_system_prompt_file")
    if operation == "inspect":
        info = inspect_provider(backend, provider)
        assert info["live_inference_verified"] is False
    else:
        result = execute(make_request(backend), provider, tmp_path / "request")
        assert result.text == "FINAL ANSWER — 日本語"
    calls = records(log)
    auth_calls = [record for record in calls if "status" in record.get("args", [])]
    assert len(auth_calls) == 1
    auth_args = auth_calls[0]["args"]
    assert auth_args[-2:] == ["auth", "status"]
    for flag in ("--safe-mode", "--restricted", "--tools", "--disallowedTools", "--strict-mcp-config",
                 "--mcp-config", "--settings", "--permission-prompts", "--system-prompt-file"):
        assert flag in auth_args[:-2]
    inspection_system = Path(auth_args[auth_args.index("--system-prompt-file") + 1])
    assert inspection_system == Path(auth_calls[0]["cwd"]) / "preflight-system.txt"
    if operation == "inspect":
        assert all("stdin" not in record for record in calls)
        assert not inspection_system.exists()  # The private probe directory was removed.
    else:
        assert inspection_system.stat().st_mode & 0o777 == 0o600
        assert inspection_system.read_text() == "Non-inference CLI capability inspection.\n"
        inference_args = next(record["args"] for record in calls if "-p" in record.get("args", []))
        actual_system = Path(inference_args[inference_args.index("--system-prompt-file") + 1])
        assert actual_system != inspection_system
        assert actual_system.read_text() == "Be a careful reviewer."
        assert sum("stdin" in record for record in calls) == 1


@pytest.mark.parametrize("scenario", ["rejected_system_prompt_file", "rejected_system_prompt_file_stdout"])
@pytest.mark.parametrize("operation", ["inspect", "execute"])
def test_claude_rejected_hidden_flag_fails_closed_before_inference(fake_provider, tmp_path, monkeypatch, scenario, operation):
    backend, provider, log = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude hidden global flags")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises(UnsupportedCapability):
        if operation == "inspect":
            inspect_provider(backend, provider)
        else:
            execute(make_request(backend), provider, tmp_path / "request")
    calls = records(log)
    auth_args = [record["args"] for record in calls if "status" in record.get("args", [])]
    assert len(auth_args) == 1  # Never retry by dropping the unsupported flag.
    assert auth_args[0][-2:] == ["auth", "status"]
    assert "--system-prompt-file" in auth_args[0]
    assert all("stdin" not in record and "-p" not in record.get("args", []) for record in calls)


def test_codex_auth_inspection_cannot_force_logout_of_api_session(fake_provider, tmp_path, monkeypatch):
    backend, provider, log = fake_provider
    if backend != "codex_cli":
        pytest.skip("Codex forced login-method semantics")
    marker = tmp_path / "logged-out.txt"
    monkeypatch.setenv("FAKE_CLI_SCENARIO", "auth_api")
    monkeypatch.setenv("FAKE_CLI_LOGOUT_MARKER", str(marker))
    with pytest.raises(AuthenticationRequired):
        inspect_provider(backend, provider)
    auth_args = next(r["args"] for r in records(log) if "status" in r.get("args", []))
    assert auth_args == ["login", "status"]
    assert all("stdin" not in r for r in records(log))
    assert not marker.exists()


def test_codex_inference_still_enforces_subscription_auth(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    if backend != "codex_cli":
        pytest.skip("Codex forced login-method semantics")
    execute(make_request(backend), provider, tmp_path / "request")
    calls = records(log)
    auth_args = next(r["args"] for r in calls if "status" in r.get("args", []))
    assert auth_args == ["login", "status"]
    exec_args = next(r["args"] for r in calls if "--output-last-message" in r.get("args", []))
    assert 'forced_login_method="chatgpt"' in exec_args


def test_child_api_env_removed_without_parent_or_home_changes(fake_provider, tmp_path, monkeypatch):
    backend, provider, log = fake_provider
    variables = {
        "OPENAI_API_KEY": "DO_NOT_LEAK", "CODEX_API_KEY": "DO_NOT_LEAK",
        "OPENAI_BASE_URL": "https://invalid.example", "ANTHROPIC_API_KEY": "DO_NOT_LEAK",
        "ANTHROPIC_AUTH_TOKEN": "DO_NOT_LEAK", "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1", "AWS_ACCESS_KEY_ID": "DO_NOT_LEAK",
        "ANTHROPIC_BASE_URL": "https://invalid.example", "CODEX_HOME": "/auth-home-unchanged",
        "CLAUDE_CONFIG_DIR": "/claude-home-unchanged",
        "AI_SCIENTIST_INFERENCE_RUNTIME": "PRIVATE_LOCAL_BROKER_TOKEN",
    }
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    execute(make_request(backend), provider, tmp_path / "request")
    event = next(r for r in records(log) if "stdin" in r)
    assert event["home"] == os.environ["HOME"]
    for key in variables:
        assert os.environ[key] == variables[key]
        if key not in {"CODEX_HOME", "CLAUDE_CONFIG_DIR"}:
            assert key not in event["environment_keys"]


@pytest.mark.parametrize("scenario,error", [
    ("auth_api", AuthenticationRequired), ("missing_flag", UnsupportedCapability),
    ("quota_exit", QuotaExhausted), ("quota_event", QuotaExhausted),
    ("auth_expired", AuthenticationRequired), ("network_exit", TransientProviderError),
    ("context_exit", ContextLimitExceeded), ("unknown_flag_exit", UnsupportedCapability),
    ("missing_terminal", IncompleteOutput), ("garbage_tail", IncompleteOutput),
    ("unknown_event", UnsupportedCapability), ("tool", UnsupportedCapability),
    ("compaction", ContextLimitExceeded), ("nonzero_with_final", ProviderError),
])
def test_typed_failures_never_return_partial_success(fake_provider, tmp_path, monkeypatch, scenario, error):
    backend, provider, _ = fake_provider
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises(error) as raised:
        execute(make_request(backend), provider, tmp_path / "request")
    assert "PRIVATE_SECRET" not in str(raised.value)


@pytest.mark.parametrize("scenario", ["auth_logged_out", "auth_unknown_plan", "auth_unsupported"])
def test_claude_auth_ambiguity_stops_before_any_inference(fake_provider, tmp_path, monkeypatch, scenario):
    backend, provider, log = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude auth result format")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises((AuthenticationRequired, UnsupportedCapability)):
        execute(make_request(backend), provider, tmp_path / "request")
    assert all("stdin" not in r for r in records(log))


@pytest.mark.parametrize("scenario,error", [("mismatch", IncompleteOutput), ("missing_final", IncompleteOutput), ("invalid_json", InvalidStructuredOutput)])
def test_codex_requires_consistent_final_file(fake_provider, tmp_path, monkeypatch, scenario, error):
    backend, provider, _ = fake_provider
    if backend != "codex_cli":
        pytest.skip("Codex final artifact")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises(error):
        execute(make_request(backend, schema={"type": "object"}), provider, tmp_path / "request")


@pytest.mark.parametrize("scenario,error", [("missing_structured", InvalidStructuredOutput), ("truncated", IncompleteOutput), ("schema_retry_error", InvalidStructuredOutput)])
def test_claude_native_result_gates(fake_provider, tmp_path, monkeypatch, scenario, error):
    backend, provider, _ = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude result envelope")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises(error):
        execute(make_request(backend, schema={"type": "object"}), provider, tmp_path / "request")


def test_claude_native_schema_tool_is_not_a_general_operation_tool(fake_provider, tmp_path, monkeypatch):
    backend, provider, _ = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude schema tool result")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", "structured_tool")
    result = execute(make_request(backend, schema={"type": "object"}), provider, tmp_path / "request")
    assert result.structured_data == {"valid": False, "analyses": []}
    assert result.metadata["stop_reason"] == "tool_use"


@pytest.mark.parametrize("scenario,error", [
    ("aborted_success", Cancelled), ("unknown_terminal_reason", IncompleteOutput),
    ("permission_denial", UnsupportedCapability), ("malformed_assistant", IncompleteOutput),
    ("unrecognized_tool_result", UnsupportedCapability),
])
def test_claude_protocol_safety_gates(fake_provider, tmp_path, monkeypatch, scenario, error):
    backend, provider, _ = fake_provider
    if backend != "claude_cli":
        pytest.skip("Claude protocol")
    monkeypatch.setenv("FAKE_CLI_SCENARIO", scenario)
    with pytest.raises(error):
        execute(make_request(backend, schema={"type": "object"}), provider, tmp_path / "request")


def test_duplicate_protocol_keys_are_rejected(fake_provider, tmp_path, monkeypatch):
    backend, provider, _ = fake_provider
    monkeypatch.setenv("FAKE_CLI_SCENARIO", "duplicate_keys")
    with pytest.raises(IncompleteOutput):
        execute(make_request(backend), provider, tmp_path / "request")


def test_rejects_old_cli_before_auth_or_inference(fake_provider, monkeypatch):
    backend, provider, log = fake_provider
    monkeypatch.setenv("FAKE_CLI_VERSION", "0.1.0" if backend == "codex_cli" else "2.1.42")
    with pytest.raises(UnsupportedCapability):
        inspect_provider(backend, provider)
    assert len(records(log)) == 1


def test_long_history_uses_single_stdin_turn_without_truncation(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    request = make_request(backend)
    paper = "研究全文" * 9000
    request.messages = [{"role": "assistant", "content": "chosen sliced history"}, {"role": "user", "content": paper}]
    execute(request, provider, tmp_path / "request")
    stdin = next(r["stdin"] for r in records(log) if "stdin" in r)
    assert paper in stdin
    if backend == "claude_cli":
        envelope = json.loads(stdin)
        assert len(stdin.splitlines()) == 1
        assert envelope["message"]["role"] == "user"
    assert request.messages[0]["content"] == "chosen sliced history"


def test_native_images_preserve_occurrences_and_mime(fake_provider, tmp_path):
    from PIL import Image
    backend, provider, log = fake_provider
    image = tmp_path / "plot.png"
    Image.new("RGB", (4, 4), "red").save(image)
    raw = image.read_bytes()
    uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()  # Legacy wrong label.
    request = make_request(backend)
    request.messages = [{"role": "user", "content": [
        {"type": "text", "text": "Compare BOTH occurrences, even if identical."},
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "image_url", "image_url": {"url": uri}},
    ]}]
    result = execute(request, provider, tmp_path / "request")
    occurrences = result.metadata["image_occurrences"]
    assert len(occurrences) == 2
    assert len({i["occurrence_id"] for i in occurrences}) == 2
    assert {i["mime_type"] for i in occurrences} == {"image/png"}
    calls = records(log)
    if backend == "codex_cli":
        args = next(r["args"] for r in calls if "--output-last-message" in r.get("args", []))
        assert args.count("--image") == 2
        for index, value in enumerate(args):
            if value == "--image":
                assert Path(args[index + 1]).read_bytes() == raw
    else:
        stdin = next(r["stdin"] for r in calls if "stdin" in r)
        parts = json.loads(stdin)["message"]["content"]
        sources = [part["source"] for part in parts if part["type"] == "image"]
        assert len(sources) == 2
        assert all(source["media_type"] == "image/png" and base64.b64decode(source["data"]) == raw for source in sources)


def test_text_overflow_never_silently_drops_history(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    provider["max_input_text_bytes"] = 10
    with pytest.raises(ContextLimitExceeded):
        execute(make_request(backend), provider, tmp_path / "request")
    assert all("stdin" not in r for r in records(log))


def test_output_limit_is_finite(fake_provider, tmp_path, monkeypatch):
    backend, provider, _ = fake_provider
    monkeypatch.setenv("FAKE_CLI_SCENARIO", "stdout_flood")
    provider["max_output_bytes"] = 1024
    started = time.monotonic()
    with pytest.raises(IncompleteOutput):
        execute(make_request(backend), provider, tmp_path / "request")
    assert time.monotonic() - started < 4
    assert (tmp_path / "request" / "stdout.jsonl").stat().st_size <= 1024


def test_cancelled_before_start_does_not_launch(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        execute(make_request(backend), provider, tmp_path / "request", event)
    assert not log.exists()


def test_subprocess_handles_large_simultaneous_pipes(tmp_path):
    program = "import sys; sys.stderr.buffer.write(b'e' * 200000); sys.stderr.flush(); data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)"
    result = run_process([sys.executable, "-c", program], input_bytes=b"a" * 300000,
                         cwd=tmp_path, env=dict(os.environ), timeout_seconds=3)
    assert result.returncode == 0
    assert result.stdout == b"a" * 300000
    assert result.stderr == b"e" * 200000


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancel_kill_owned_child_but_not_unrelated_process(tmp_path, cancel):
    pidfile = tmp_path / "owned.pid"
    program = "import pathlib, subprocess, sys, time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    event = threading.Event()
    timer = threading.Timer(0.35, event.set) if cancel else None
    if timer:
        timer.start()
    try:
        with pytest.raises(Cancelled if cancel else InferenceTimeout):
            run_process([sys.executable, "-c", program, str(pidfile)], cwd=tmp_path,
                        env=dict(os.environ), timeout_seconds=5 if cancel else 0.35,
                        cancel_event=event, termination_grace_seconds=0.1)
        assert unrelated.poll() is None
        child_pid = int(pidfile.read_text())
        # POSIX may leave an adopted zombie briefly. ps is diagnostic only and
        # distinguishes a zombie from a still-running leaked child.
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True).stdout.strip()
        assert not status or status.startswith("Z")
    finally:
        if timer:
            timer.cancel()
        os.killpg(unrelated.pid, signal.SIGTERM)
        unrelated.wait(timeout=2)


def test_pipe_inherited_by_descendant_cannot_hang_after_parent_exit(tmp_path):
    program = "import subprocess, sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
    started = time.monotonic()
    with pytest.raises(IncompleteOutput):
        run_process([sys.executable, "-c", program], cwd=tmp_path, env=dict(os.environ),
                    timeout_seconds=5, drain_timeout_seconds=0.1, termination_grace_seconds=0.1)
    assert time.monotonic() - started < 2


def test_success_removes_active_process_ownership_record(tmp_path):
    path = tmp_path / "process.json"
    result = run_process([sys.executable, "-c", "print('finished')"], cwd=tmp_path,
                         env=dict(os.environ), timeout_seconds=3, ownership_path=path)
    assert result.stdout == b"finished\n"
    assert not path.exists()


def test_materialized_input_limit_is_configurable(fake_provider, tmp_path):
    backend, provider, log = fake_provider
    provider["max_materialized_bytes"] = 1
    with pytest.raises(ContextLimitExceeded):
        execute(make_request(backend), provider, tmp_path / "request")
    assert all("stdin" not in r for r in records(log))


@pytest.mark.parametrize("limit", [0, -1, True, 3.5, "1024"])
def test_materialized_limit_requires_positive_integer(fake_provider, tmp_path, limit):
    backend, provider, _ = fake_provider
    provider["max_materialized_bytes"] = limit
    with pytest.raises(UnsupportedCapability):
        execute(make_request(backend), provider, tmp_path / "request")


def test_owned_orphan_cleanup_rejects_stale_identity_and_preserves_other_process(tmp_path):
    # A separate owner stands in for a broker. Its process group does not include
    # the CLI group, so killing the owner genuinely exercises orphan cleanup.
    owner_program = """
import os, pathlib, sys
from ai_scientist.inference.process import run_process
root = pathlib.Path(sys.argv[1])
run_process([sys.executable, '-c', 'import time; time.sleep(60)'],
            cwd=root, env=dict(os.environ), timeout_seconds=60,
            ownership_path=root/'process.json', termination_grace_seconds=0.1)
"""
    root = Path(__file__).resolve().parents[1]
    owner = subprocess.Popen([sys.executable, "-c", owner_program, str(tmp_path)], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    registry = tmp_path / "process.json"
    record = None
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if registry.exists():
                try:
                    record = json.loads(registry.read_text())
                    break
                except ValueError:
                    pass
            time.sleep(0.01)
        assert record is not None, "owner did not register its process"
        assert record["pid"] == record["pgid"]
        assert registry.stat().st_mode & 0o777 == 0o600
        assert record["identity"]["kind"] in {"linux-proc-start-v1", "darwin-proc-start-v1"}
        # A recycled PID with a different kernel start identity is not owned.
        stale = {**record, "identity": {**record["identity"], "start_seconds": -1}}
        registry.write_text(json.dumps(stale))
        assert cleanup_owned_process(registry, grace_seconds=0) is False
        assert unrelated.poll() is None
        registry.write_text(json.dumps(record))
        owner.kill()
        owner.wait(timeout=2)
        assert cleanup_owned_process(registry, grace_seconds=0) is True
        assert unrelated.poll() is None
        assert not registry.exists()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(record["pid"])], capture_output=True, text=True).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.02)
        assert not status or status.startswith("Z")
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        if record is not None:
            # Exact same recorded identity must still match for fallback cleanup.
            registry.write_text(json.dumps(record))
            registry.chmod(0o600)
            cleanup_owned_process(registry, grace_seconds=0)
        os.killpg(unrelated.pid, signal.SIGTERM)
        unrelated.wait(timeout=2)


def test_unverifiable_ownership_never_signals(monkeypatch, tmp_path):
    import ai_scientist.inference.process as process_module
    record = {"version": 1, "pid": 123, "pgid": 123, "uid": os.getuid(),
              "nonce": "fixture", "identity": {"kind": "unknown", "pid": 123, "pgid": 123, "uid": os.getuid()}}
    registry = tmp_path / "process.json"
    registry.write_text(json.dumps(record))
    registry.chmod(0o600)
    monkeypatch.setattr(process_module, "_process_identity", lambda pid: None)
    def forbidden_signal(*args):
        pytest.fail("Unverifiable ownership must not signal any process")
    monkeypatch.setattr(os, "killpg", forbidden_signal)
    assert cleanup_owned_process(registry) is False


def test_symlink_ownership_registry_is_not_followed(monkeypatch, tmp_path):
    import ai_scientist.inference.process as process_module
    target = tmp_path / "untrusted.json"
    target.write_text('{}')
    registry = tmp_path / "process.json"
    registry.symlink_to(target)
    monkeypatch.setattr(process_module, "_process_identity", lambda pid: pytest.fail("Must not inspect a symlink record"))
    assert cleanup_owned_process(registry) is False


def test_owner_death_before_registration_never_starts_cli(tmp_path):
    owner_program = """
import os, pathlib, sys, time
import ai_scientist.inference.process as runner
directory = pathlib.Path(sys.argv[1])
def blocked_registration(path, process):
    (directory/'gate-pid.txt').write_text(str(process.pid))
    time.sleep(60)
runner._write_ownership = blocked_registration
payload = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('CLI STARTED')"
runner.run_process([sys.executable, '-c', payload, str(directory/'started.txt')],
                   cwd=directory, env=dict(os.environ), timeout_seconds=60,
                   ownership_path=directory/'process.json')
"""
    root = Path(__file__).resolve().parents[1]
    owner = subprocess.Popen([sys.executable, "-c", owner_program, str(tmp_path)], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        pidfile = tmp_path / "gate-pid.txt"
        deadline = time.monotonic() + 4
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        owner.kill()
        owner.wait(timeout=2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.02)
        assert not status or status.startswith("Z")
        assert not (tmp_path / "started.txt").exists()
        assert not (tmp_path / "process.json").exists()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)


def _recorded_codex_isolation_events():
    path = Path(__file__).parent / "fixtures" / "cli" / "codex_disabled_code_mode.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _parse_recorded_codex(events, tmp_path, *, returncode=0, answer="fixture-live-nonce"):
    final = tmp_path / "final.txt"
    final.write_text(answer)
    raw = ("\n".join(json.dumps(event) for event in events) + "\n").encode()
    result = ProcessResult(returncode=returncode, stdout=raw, stderr=b"", duration_seconds=0.1)
    return _parse_codex(result, final, make_request("codex_cli"), 1024 * 1024)


def test_recorded_codex_disabled_code_mode_diagnostic_is_metadata_only(tmp_path):
    events = _recorded_codex_isolation_events()
    text, structured, usage, details = _parse_recorded_codex(events, tmp_path)
    assert text == "fixture-live-nonce"
    assert structured is None
    assert usage["output_tokens"] == 16
    assert details["terminal_event"] == "turn.completed"
    assert details["isolation_notices"] == [events[1]["item"]["message"]]


@pytest.mark.parametrize("message,error", [
    ("Unknown provider error", ProviderError),
    ("usage_limit_reached", QuotaExhausted),
    ("authentication_error", AuthenticationRequired),
    ("context_length_exceeded", ContextLimitExceeded),
])
def test_other_codex_error_items_are_classified_not_ignored(tmp_path, message, error):
    events = _recorded_codex_isolation_events()
    events[1]["item"]["message"] = message
    with pytest.raises(error):
        _parse_recorded_codex(events, tmp_path)


@pytest.mark.parametrize("alteration", ["prefix", "suffix", "top_level", "not_completed"])
def test_codex_isolation_exception_requires_exact_completed_item(tmp_path, alteration):
    events = _recorded_codex_isolation_events()
    if alteration == "prefix":
        events[1]["item"]["message"] = "Unexpected: " + events[1]["item"]["message"]
    elif alteration == "suffix":
        events[1]["item"]["message"] += " Additional failure."
    elif alteration == "top_level":
        events[1] = {"type": "error", "message": events[1]["item"]["message"]}
    else:
        events[1]["type"] = "item.started"
    with pytest.raises(ProviderError):
        _parse_recorded_codex(events, tmp_path)


@pytest.mark.parametrize("failure,error", [
    ("exit_nonzero", ProviderError), ("missing_terminal", IncompleteOutput),
    ("missing_answer", IncompleteOutput), ("mismatch", IncompleteOutput),
])
def test_codex_isolation_notice_does_not_relax_success_gates(tmp_path, failure, error):
    events = _recorded_codex_isolation_events()
    if failure == "missing_terminal":
        events.pop()
    elif failure == "missing_answer":
        events.pop(3)
    with pytest.raises(error):
        _parse_recorded_codex(events, tmp_path,
                             returncode=1 if failure == "exit_nonzero" else 0,
                             answer="different answer" if failure == "mismatch" else "fixture-live-nonce")
