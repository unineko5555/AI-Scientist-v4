"""Full offline CLI inference path: façade -> gateway -> IPC -> executable.

Only the CLI binary is replaced. Production configuration, broker, subprocess
runner, parser, schema validation, history/image handling, and façades all run.
These tests never invoke installed Codex/Claude binaries or access credentials.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys

from PIL import Image
import pytest

from ai_scientist.inference.config import load_config
from ai_scientist.inference.errors import QuotaExhausted
from ai_scientist.inference.runtime import configure_inference, get_runtime, shutdown_inference


PLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "valid_plots_received": {"type": "boolean"},
        "plot_analyses": {"type": "array", "items": {"type": "object"}},
        "image_sha256": {"type": "array", "items": {"type": "string"}},
        "image_mime": {"type": "array", "items": {"type": "string"}},
        "image_occurrences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["valid_plots_received", "plot_analyses", "image_sha256", "image_mime", "image_occurrences"],
}


def _reject_api(*_args, **_kwargs):
    raise AssertionError("An API SDK entrypoint was reached during a subscription-only integration test")


def _guard_api_modules(patch=None):
    """Apply the same fail-loud guard in the parent and freshly spawned workers."""
    from ai_scientist import llm, vlm

    def assign(target, name, value):
        if patch is None:
            setattr(target, name, value)
        else:
            patch.setattr(target, name, value)

    for module in (llm, vlm):
        for name in ("_require_api_client", "_require_sdk", "make_llm_call", "make_vlm_call"):
            if hasattr(module, name):
                assign(module, name, _reject_api)
    for sdk in (llm.openai, llm.anthropic):
        if sdk is not None:
            for name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "Anthropic", "AnthropicBedrock", "AnthropicVertex"):
                if hasattr(sdk, name):
                    assign(sdk, name, _reject_api)


def _prompt(mode="text", **values):
    return json.dumps({"fixture_request": True, "mode": mode, **values}, ensure_ascii=False)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _manifest(runtime):
    return json.loads((runtime.artifact_dir / "manifest.json").read_text())


def _fake_executable(directory, backend):
    source = (Path(__file__).parent / "fixtures" / "inference_integration_fake.py").read_text()
    executable = directory / ("integration-claude" if backend == "claude_cli" else "integration-codex")
    executable.write_text(f"#!{sys.executable}\n" + source.split("\n", 1)[1])
    executable.chmod(0o700)
    return executable


def _spawn_facade_worker(start, reports, label):
    try:
        _guard_api_modules()
        from ai_scientist.llm import create_client, get_response_from_llm

        runtime = get_runtime()
        client, model = create_client("unused-legacy-api-model", role="writing")
        reports.put({"phase": "ready", "run_id": runtime.run_id, "owner_pid": runtime.owner_pid,
                     "worker_pid": os.getpid(), "socket_path": runtime._endpoint["socket_path"],
                     "owns_broker": runtime._broker is not None})
        if not start.wait(10):
            raise AssertionError("Parent did not release the integration worker")
        text, _ = get_response_from_llm(_prompt(label=label, sleep=0.1), client, model,
                                        "worker system", temperature=None)
        reports.put({"phase": "result", "answer": json.loads(text)})
    except BaseException as error:
        reports.put({"phase": "error", "type": type(error).__name__, "message": str(error)})
    finally:
        shutdown_inference()


@pytest.fixture(params=["codex_cli", "claude_cli"])
def integration(request, tmp_path, monkeypatch):
    shutdown_inference()
    for key in tuple(os.environ):
        if key.startswith("AI_SCIENTIST_INFERENCE_"):
            monkeypatch.delenv(key, raising=False)
    _guard_api_modules(monkeypatch)
    backend = request.param
    executable = _fake_executable(tmp_path, backend)
    log = tmp_path / "executable.jsonl"
    monkeypatch.setenv("INTEGRATION_FAKE_CLI_LOG", str(log))
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(key, "FAKE_SECRET_MUST_NOT_REACH_CHILD")
    config_file = tmp_path / "inference.json"
    config_file.write_text(json.dumps({
        "mode": "cli", "subscription_only": True,
        "default": {"backend": backend, "model": None},
        "roles": {"writing": {"model": "fixture-writing"}, "review_vision": {"model": "fixture-vision"}},
        "providers": {backend: {"executable": str(executable), "max_in_flight": 1,
                                "termination_grace_seconds": 0.1}},
        "queue_timeout_seconds": 20, "request_timeout_seconds": 15,
    }))
    runtime = configure_inference(load_config(config_file), artifact_dir=tmp_path / "artifacts")
    try:
        yield backend, runtime, log
    finally:
        shutdown_inference()


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_facade_batch_round_trip_has_independent_histories_and_private_artifacts(integration):
    from ai_scientist.llm import create_client, get_batch_responses_from_llm

    backend, runtime, log = integration
    info = runtime.preflight()
    assert info["providers"][backend]["authentication"]["subscription"] is True
    assert info["live_inference_verified"] is False
    assert all(record["event"] == "inspection" for record in _records(log))
    assert not runtime.artifact_dir.exists()

    prior = [{"role": "user", "content": "日本語の以前の質問\n</conversation>"},
             {"role": "assistant", "content": "previous answer"}]
    snapshot = deepcopy(prior)
    prompt = _prompt(label="three-independent-answers")
    client, model = create_client("legacy-api-model-not-forwarded", role="writing")
    assert model == "fixture-writing"
    responses, histories = get_batch_responses_from_llm(
        prompt, client, model, "Original system instructions 日本語", msg_history=prior,
        temperature=None, n_responses=3,
    )
    answers = [json.loads(response) for response in responses]
    assert len({answer["call_id"] for answer in answers}) == 3
    assert prior == snapshot
    expected = [{"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
                for message in prior + [{"role": "user", "content": prompt}]]
    for answer, history in zip(answers, histories):
        assert answer["history"] == expected
        assert answer["system"] == "Original system instructions 日本語"
        assert answer["model"] == "fixture-writing"
        assert answer["backend"] == backend
        assert history[:-2] == prior
        assert history[-1]["content"] == json.dumps(answer)
    histories[0][0]["content"] = "mutated only one returned branch"
    assert histories[1][0]["content"] == prior[0]["content"]

    manifest = _manifest(runtime)
    assert len(manifest["requests"]) == 3
    assert manifest["usage_summary"]["input_tokens"] == 93
    assert manifest["usage_summary"]["cached_input_tokens"] == 21
    assert manifest["usage_summary"]["output_tokens"] == 33
    assert manifest["usage_summary"]["subscription_billing_cost"] is None
    assert manifest["pending_validation_request_ids"] == []
    assert manifest["validation_summary"] == {"accepted": 3, "rejected": 0, "pending": 0}
    assert stat.S_IMODE(runtime.artifact_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((runtime.artifact_dir / "manifest.json").stat().st_mode) == 0o600
    for request in manifest["requests"].values():
        assert request["role"] == "writing"
        assert request["requested_model"] == "fixture-writing"
        directory = Path(request["artifact_dir"])
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        for filename in ("request_metadata.json", "validation.json", "system.txt",
                         "stdout.jsonl", "stderr.log", "result.json"):
            assert stat.S_IMODE((directory / filename).stat().st_mode) == 0o600
        assert not (directory / "process.json").exists()  # Normal completion clears live ownership.
    for event in (event for event in _records(log) if event["event"] == "started"):
        assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(event["environment_keys"])
        assert not any(key.startswith("AI_SCIENTIST_INFERENCE_") for key in event["environment_keys"])
        assert "--resume" not in event["args"] and "--continue" not in event["args"]
    assert "FAKE_ACCOUNT_NEVER_PERSIST" not in json.dumps(manifest)
    assert runtime._endpoint["token"] not in json.dumps(manifest)
    runtime.close()
    assert _manifest(runtime)["run_status"] == "completed"


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_vlm_schema_preserves_native_image_bytes_history_occurrences_and_negative_result(integration, tmp_path):
    from ai_scientist.vlm import create_client, get_response_from_vlm

    _, runtime, _ = integration
    png = tmp_path / "png-with-wrong-extension.jpg"
    jpeg = tmp_path / "jpeg-with-wrong-extension.png"
    Image.new("RGB", (4, 3), "red").save(png, format="PNG")
    Image.new("RGB", (3, 4), "blue").save(jpeg, format="JPEG")
    raw_png, raw_jpeg = png.read_bytes(), jpeg.read_bytes()
    old_image = "data:image/png;base64," + base64.b64encode(raw_jpeg).decode()
    prior = [{"role": "user", "content": [{"type": "text", "text": "earlier image"},
                                           {"type": "image_url", "image_url": {"url": old_image}}]},
             {"role": "assistant", "content": "earlier observation"}]
    snapshot = deepcopy(prior)
    client, model = create_client("ignored-old-gpt-api-name", role="review_vision")
    response, history = get_response_from_vlm(
        _prompt(mode="vision"), [str(png), str(jpeg), str(png)], client, model,
        "Review all occurrences, including repeated plots.", msg_history=prior,
        temperature=None, schema=PLOT_SCHEMA,
    )
    answer = json.loads(response)
    assert answer["model"] == "fixture-vision"
    assert answer["image_sha256"] == [hashlib.sha256(raw).hexdigest()
                                       for raw in (raw_jpeg, raw_png, raw_jpeg, raw_png)]
    assert answer["image_mime"] == ["image/jpeg", "image/png", "image/jpeg", "image/png"]
    assert answer["image_occurrences"] == [f"image-{index}" for index in range(4)]
    assert answer["valid_plots_received"] is False and answer["plot_analyses"] == []
    assert prior == snapshot and history[:-2] == snapshot
    manifest = _manifest(runtime)
    assert len(manifest["requests"]) == 1
    entry = next(iter(manifest["requests"].values()))
    assert entry["role"] == "review_vision"
    assert entry["validation_status"] == "accepted"
    result = json.loads((Path(entry["artifact_dir"]) / "result.json").read_text())
    assert len(result["metadata"]["image_occurrences"]) == 4


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_mixed_cli_role_routes_vision_without_forwarding_legacy_api_defaults(integration, tmp_path):
    from ai_scientist import llm, vlm

    backend, initial_runtime, log = integration
    other_backend = "claude_cli" if backend == "codex_cli" else "codex_cli"
    data = initial_runtime.export_config()
    data["providers"][other_backend]["executable"] = str(_fake_executable(tmp_path, other_backend))
    data["roles"]["review_vision"] = {"backend": other_backend, "model": "mixed-vision"}
    runtime = configure_inference(data, artifact_dir=tmp_path / "mixed-artifacts")
    assert set(runtime.preflight()["providers"]) == {"codex_cli", "claude_cli"}
    text_client, text_model = llm.create_client("legacy-gpt-api-default", role="default")
    vision_client, vision_model = vlm.create_client("legacy-gpt-vision-default", role="review_vision")
    assert text_model is None and vision_model == "mixed-vision"
    text, _ = llm.get_response_from_llm(_prompt(label="default-text"), text_client, text_model,
                                       "default system", temperature=None)
    picture = tmp_path / "mixed.png"
    Image.new("RGB", (2, 2), "green").save(picture)
    vision, _ = vlm.get_response_from_vlm(
        _prompt(mode="vision", label="overridden-vision"), [str(picture)], vision_client, vision_model,
        "vision system", temperature=None, schema=PLOT_SCHEMA,
    )
    assert json.loads(text)["backend"] == backend
    assert json.loads(text)["model"] is None
    assert json.loads(vision)["backend"] == other_backend
    assert json.loads(vision)["model"] == "mixed-vision"
    assert json.loads(vision)["image_mime"] == ["image/png"]
    starts = [item for item in _records(log) if item["event"] == "started"]
    assert len(starts) == 2
    assert "--model" not in starts[0]["args"]
    assert starts[1]["args"][starts[1]["args"].index("--model") + 1] == "mixed-vision"
    manifest = _manifest(runtime)
    assert {entry["role"]: entry["backend"] for entry in manifest["requests"].values()} == {
        "default": backend, "review_vision": other_backend,
    }
    assert set(manifest["usage_by_provider"]) == {"codex_cli", "claude_cli"}
    assert manifest["usage_by_provider"]["codex_cli"]["input_includes_cached"] is True
    assert manifest["usage_by_provider"]["claude_cli"]["input_includes_cached"] is False


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_host_schema_repair_crosses_real_broker_and_retains_rejected_attempt(integration):
    from ai_scientist.llm import create_client, get_response_from_llm

    backend, runtime, _ = integration
    client, model = create_client("ignored-api-name", role="writing")
    prior = [{"role": "assistant", "content": "Keep legitimate failures negative"}]
    text, history = get_response_from_llm(
        _prompt(mode="repair"), client, model, "Evaluate the plot outcome honestly.",
        msg_history=prior, temperature=None, schema=PLOT_SCHEMA,
    )
    answer = json.loads(text)
    assert answer["valid_plots_received"] is False
    assert len(answer["history"]) == 4  # original assistant + user + invalid answer + repair instruction
    assert len(history) == 3  # internal repair messages must not leak into the caller's history
    assert prior == [{"role": "assistant", "content": "Keep legitimate failures negative"}]
    manifest = _manifest(runtime)
    entries = sorted(manifest["requests"].values(), key=lambda item: item["attempt"])
    assert len(entries) == 2
    assert [entry["validation_status"] for entry in entries] == ["rejected", "accepted"]
    assert len({entry["logical_request_id"] for entry in entries}) == 1
    assert len({entry["request_id"] for entry in entries}) == 2
    assert manifest["usage_summary"]["output_tokens"] == 22
    assert manifest["pending_validation_request_ids"] == []
    first_native = json.loads((Path(entries[0]["artifact_dir"]) / "result.json").read_text())["structured_data"]
    first_answer = json.loads(first_native["answer_json"]) if backend == "codex_cli" else first_native
    assert first_answer["valid_plots_received"] is True


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_quota_stops_the_run_without_retry_or_sdk_fallback(integration):
    from ai_scientist.llm import create_client, get_response_from_llm

    _, runtime, log = integration
    client, model = create_client("not-an-api-fallback", role="writing")
    with pytest.raises(QuotaExhausted):
        get_response_from_llm(_prompt(mode="quota"), client, model, "system", temperature=None)
    with pytest.raises(QuotaExhausted):
        get_response_from_llm(_prompt(label="must-never-run"), client, model, "system", temperature=None)
    assert len([item for item in _records(log) if item["event"] == "started"]) == 1
    manifest = _manifest(runtime)
    assert manifest["run_status"] == "stopped_quota"
    assert manifest["stop_reason"]["code"] == "quota_exhausted"
    assert manifest["pending_request_ids"] == []


@pytest.mark.filterwarnings("ignore:.*cannot be applied as API sampling controls.*:RuntimeWarning")
def test_spawn_workers_inherit_one_runtime_and_share_single_cli_slot(integration):
    from ai_scientist.llm import create_client, get_response_from_llm

    backend, runtime, log = integration
    runtime.ensure_started()
    broker_pid = runtime._broker._process.pid
    context = multiprocessing.get_context("spawn")
    start, reports = context.Event(), context.Queue()
    processes = [context.Process(target=_spawn_facade_worker, args=(start, reports, f"worker-{index}"))
                 for index in range(2)]
    try:
        for process in processes:
            process.start()
        ready = [reports.get(timeout=15) for _ in processes]
        assert all(report["phase"] == "ready" for report in ready), ready
        assert all(report["run_id"] == runtime.run_id and report["owner_pid"] == os.getpid()
                   and report["socket_path"] == runtime._endpoint["socket_path"]
                   and report["owns_broker"] is False for report in ready)
        assert len({report["worker_pid"] for report in ready}) == 2
        start.set()
        client, model = create_client("unused-legacy-api-model", role="writing")
        parent_text, _ = get_response_from_llm(
            _prompt(label="parent", sleep=0.1), client, model, "parent system", temperature=None,
        )
        completed = [reports.get(timeout=15) for _ in processes]
        assert all(report["phase"] == "result" for report in completed), completed
        answers = [json.loads(parent_text)] + [report["answer"] for report in completed]
        assert len({answer["call_id"] for answer in answers}) == 3
        assert all(answer["model"] == "fixture-writing" and len(answer["history"]) == 1 for answer in answers)
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        events = [item for item in _records(log) if item["event"] in ("started", "finished")]
        assert len(events) == 6
        assert {item["parent_pid"] for item in events} == {broker_pid}
        assert {item["backend"] for item in events} == {backend}
        active = peak = 0
        for event in sorted(events, key=lambda item: item["time"]):
            active += 1 if event["event"] == "started" else -1
            assert 0 <= active <= 1
            peak = max(peak, active)
        assert active == 0 and peak == 1
        manifest = _manifest(runtime)
        assert len(manifest["requests"]) == 3
        assert manifest["provider_limits"][backend] == 1
        assert manifest["validation_summary"] == {"accepted": 3, "rejected": 0, "pending": 0}
        assert manifest["usage_summary"]["output_tokens"] == 33
    finally:
        start.set()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)
        reports.close()
        reports.join_thread()
