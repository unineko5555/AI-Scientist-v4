"""Doctor tests never launch a CLI or send a live inference request."""

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ai_scientist.inference import doctor
from ai_scientist.inference.contracts import InferenceResult, ROLES
from ai_scientist.inference.errors import (
    AuthenticationRequired, ConfigurationError, QuotaExhausted, UnsupportedCapability,
)


class FakeRuntime:
    def __init__(self, config, *, artifact_dir=None):
        self.config = config
        self.artifact_dir = Path(artifact_dir or "/nonexistent-doctor-test") / "run"
        self.closed = False
        self.preflight_calls = 0
        self.calls = []

    def preflight(self):
        self.preflight_calls += 1
        return {"providers": {
            backend: {"version": "0.157.1" if backend == "codex_cli" else "2.1.259",
                      "authentication": {"method": "chatgpt" if backend == "codex_cli" else "claude.ai",
                                         "subscription": True, "plan": "max", "token": "DO_NOT_PRINT"},
                      "protocol_profile": "subscription-stateless-v1", "stdout": "DO_NOT_PRINT"}
            for backend in {self.config.resolve(role).backend for role in ROLES}
        }}

    def generate(self, client, **kwargs):
        self.calls.append((client, kwargs))
        if "images" in kwargs:
            color_names = {rgb: name for name, rgb in doctor._COLOR_RGB.items()}
            colors = []
            for path in kwargs["images"]:
                with Image.open(path) as img:
                    colors.append(color_names[img.getpixel((0, 0))])
                assert Path(path).stat().st_mode & 0o777 == 0o600
            return SimpleNamespace(text="", structured_data={"colors": colors})
        if "schema" in kwargs:
            return SimpleNamespace(text="", structured_data=json.loads(kwargs["prompt"].split(": ", 1)[1]))
        return SimpleNamespace(text=kwargs["prompt"].split(": ", 1)[1], structured_data=None)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_runtime(monkeypatch):
    instances = []

    def factory(*args, **kwargs):
        instance = FakeRuntime(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(doctor, "InferenceRuntime", factory)
    return instances


def test_default_inspection_never_issues_inference_or_creates_artifacts(fake_runtime, tmp_path, capsys):
    path = tmp_path / "private"
    result = doctor.main(["--llm-backend", "codex_cli", "--inference-log-dir", str(path)])
    output = capsys.readouterr().out
    report = json.loads(output)
    assert result == 0
    assert report["status"] == "ready_read_only"
    assert report["live_requests_attempted"] == report["planned_live_requests"] == 0
    assert report["live_inference_verified"] is False
    assert report["all_three_checks_verified"] is False
    assert report["selected_probe_checks"] == report["passed_probe_checks"] == []
    assert report["full_v2_parity_verified"] is False
    assert report["api_fallback_enabled"] is False
    assert set(report["roles"]) == set(ROLES)
    assert report["artifacts"]["created"] is False
    assert not path.exists()
    assert "DO_NOT_PRINT" not in output
    assert fake_runtime[0].preflight_calls == 1
    assert fake_runtime[0].calls == []
    assert fake_runtime[0].closed


def test_real_runtime_read_only_path_never_starts_broker(monkeypatch, tmp_path, capsys):
    from ai_scientist.inference import cli

    inspections = []

    def inspect(backend, provider):
        inspections.append(backend)
        return {"version": "0.157.1", "authentication": {"method": "chatgpt", "subscription": True}}

    monkeypatch.setattr(cli, "inspect_provider", inspect)
    monkeypatch.setattr(doctor.InferenceRuntime, "ensure_started", lambda *a: pytest.fail("Read-only doctor started a broker"))
    assert doctor.main(["--llm-backend", "codex_cli", "--inference-log-dir", str(tmp_path / "private")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert inspections == ["codex_cli"]
    assert report["status"] == "ready_read_only"
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize("backend", ["codex_cli", "claude_cli"])
def test_probes_use_real_gateway_with_frozen_images_and_original_schema(monkeypatch, tmp_path, capsys, backend):
    from ai_scientist.inference import cli

    requests = []
    validated = []
    monkeypatch.setattr(cli, "inspect_provider", lambda *a: {
        "version": "0.157.1", "authentication": {"method": "chatgpt", "subscription": True},
    })

    def submit(self, request):
        requests.append(request)
        parts = request.messages[0]["content"]
        prompt = parts[0]["text"]
        if request.schema is None:
            return InferenceResult(request.request_id, text=prompt.split(": ", 1)[1], backend=backend)
        if any(part["type"] == "image" for part in parts):
            names = {rgb: name for name, rgb in doctor._COLOR_RGB.items()}
            colors = []
            for part in parts:
                if part["type"] == "image":
                    assert part["source"]["media_type"] == "image/png"
                    with Image.open(io.BytesIO(base64.b64decode(part["source"]["data"]))) as image:
                        colors.append(names[image.getpixel((0, 0))])
            data = {"colors": colors}
        else:
            data = json.loads(prompt.split(": ", 1)[1])
        if backend == "codex_cli":
            assert set(request.schema["properties"]) == {"answer_json"}
            data = {"answer_json": json.dumps(data)}
        return InferenceResult(request.request_id, text=json.dumps(data), structured_data=data, backend=backend)

    monkeypatch.setattr(doctor.InferenceRuntime, "submit", submit)
    monkeypatch.setattr(doctor.InferenceRuntime, "record_validation", lambda self, request_id, **kw: validated.append((request_id, kw["accepted"])))
    assert doctor.main(["--llm-backend", backend, "--probe", "--inference-log-dir", str(tmp_path / "private")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "probes_passed"
    assert len(requests) == len(validated) == 3
    assert all(accepted for _, accepted in validated)
    assert all(request.temperature is request.max_output_tokens is None for request in requests)
    assert requests[1].metadata["original_schema"]["properties"]["by_id"]["additionalProperties"] == {"type": "integer"}
    assert requests[1].metadata["original_schema"]["required"] == ["echo", "count", "by_id"]
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize("args", [[], ["--probe"], ["--llm-backend", "api"], ["--llm-backend", "api", "--probe"]])
def test_api_mode_is_rejected_before_runtime(fake_runtime, capsys, args):
    assert doctor.main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["error"]["code"] == "configuration_error"
    assert report["live_requests_attempted"] == 0
    assert fake_runtime == []


@pytest.mark.parametrize("error_type", [AuthenticationRequired, UnsupportedCapability, ConfigurationError])
def test_preflight_errors_are_typed_and_do_not_print_exception_secrets(fake_runtime, monkeypatch, capsys, error_type):
    def fail(self):
        raise error_type("sk-secret-provider-output", metadata={"token": "oauth-secret"})

    monkeypatch.setattr(FakeRuntime, "preflight", fail)
    result = doctor.main(["--llm-backend", "claude_cli", "--probe"])
    output = capsys.readouterr().out
    report = json.loads(output)
    assert result != 0
    assert report["error"]["code"] == error_type.code
    assert "sk-secret" not in output and "oauth-secret" not in output
    assert report["live_requests_attempted"] == 0
    assert not fake_runtime[0].calls
    assert fake_runtime[0].closed


def test_probe_covers_distinct_backend_model_pairs_once(fake_runtime, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "mode": "cli", "default": {"backend": "codex_cli", "model": None},
        "request_timeout_seconds": 1000, "transient_retry_limit": 9, "schema_repair_limit": 9,
        "roles": {"writing": {"backend": "claude_cli", "model": "writer"},
                  "review_text": {"backend": "claude_cli", "model": "writer"},
                  "review_vision": {"backend": "claude_cli", "model": "vision"}},
    }))
    assert doctor.main(["--inference-config", str(config), "--probe", "--probe-timeout", "35"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "probes_passed"
    assert report["live_inference_verified"] is True
    assert report["all_three_checks_verified"] is True
    assert report["passed_probe_checks"] == list(doctor._PROBE_CHECKS)
    assert report["full_v2_parity_verified"] is False
    assert report["planned_live_requests"] == report["live_requests_attempted"] == 9
    assert len(report["probes"]) == 3
    assert all([check["name"] for check in target["checks"]] == ["text", "structured_output", "ordered_images"]
               for target in report["probes"])
    assert all(check["status"] == "passed" for target in report["probes"] for check in target["checks"])
    assert report["probes"][1]["roles"] == ["writing", "review_text"]
    assert report["probes"][0]["checks"][1]["schema_transport"].startswith("json_string_envelope")
    runtime = fake_runtime[0]
    assert runtime.config.request_timeout_seconds == 35
    assert runtime.config.transient_retry_limit == 0
    assert runtime.config.schema_repair_limit == 0
    assert runtime.closed
    assert len(runtime.calls) == 9
    for _, kwargs in runtime.calls:
        if "images" in kwargs:
            assert len(kwargs["images"]) == 3
            assert all(not Path(path).exists() for path in kwargs["images"])
            assert "red" not in kwargs["prompt"]


@pytest.mark.parametrize("failed_check", ["text", "structured_output", "ordered_images"])
def test_incorrect_probe_answers_fail_closed(fake_runtime, monkeypatch, capsys, failed_check):
    original = FakeRuntime.generate

    def wrong(self, client, **kwargs):
        result = original(self, client, **kwargs)
        check = "ordered_images" if "images" in kwargs else "structured_output" if "schema" in kwargs else "text"
        if check == failed_check:
            if check == "text":
                result.text = "unverified"
            elif check == "structured_output":
                result.structured_data["optional_note"] = "invented default"
            else:
                result.structured_data["colors"].reverse()
        return result

    monkeypatch.setattr(FakeRuntime, "generate", wrong)
    assert doctor.main(["--llm-backend", "codex_cli", "--probe"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert report["live_inference_verified"] is False
    assert report["error"]["code"] == "unsupported_capability"
    failed = report["probes"][0]["checks"][-1]
    assert {key: value for key, value in failed.items() if key != "expected_output_fingerprint"} == {"name": failed_check, "status": "failed", **(
        {"schema_transport": "json_string_envelope_with_original_schema_host_validation"}
        if failed_check == "structured_output" else {})}
    assert report["live_requests_attempted"] == ["text", "structured_output", "ordered_images"].index(failed_check) + 1
    assert fake_runtime[0].closed


def test_quota_error_stops_remaining_probes_and_closes_runtime(fake_runtime, monkeypatch, capsys):
    def quota(self, client, **kwargs):
        self.calls.append((client, kwargs))
        raise QuotaExhausted("secret quota account details", metadata={"access_token": "secret-token"})

    monkeypatch.setattr(FakeRuntime, "generate", quota)
    assert doctor.main(["--llm-backend", "claude_cli", "--probe"]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert "secret" not in output
    assert report["error"]["code"] == "quota_exhausted"
    assert report["live_requests_attempted"] == len(fake_runtime[0].calls) == 1
    assert fake_runtime[0].closed


def test_interrupt_closes_runtime_and_emits_json(fake_runtime, monkeypatch, capsys):
    monkeypatch.setattr(FakeRuntime, "generate", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert doctor.main(["--llm-backend", "codex_cli", "--probe"]) == 130
    report = json.loads(capsys.readouterr().out)
    assert report["error"]["code"] == "cancelled"
    assert fake_runtime[0].closed


def test_cleanup_failure_returns_nonzero_without_leaking_details(fake_runtime, monkeypatch, capsys):
    def failure(self):
        raise OSError("secret-private-path")

    monkeypatch.setattr(FakeRuntime, "close", failure)
    assert doctor.main(["--llm-backend", "codex_cli"]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["status"] == "error"
    assert report["cleanup_error"]["code"] == "doctor_internal_error"
    assert "secret-private-path" not in output


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-time"])
def test_invalid_probe_deadline_never_starts_runtime(fake_runtime, capsys, value):
    with pytest.raises(SystemExit) as error:
        doctor.main(["--llm-backend", "codex_cli", "--probe-timeout", value])
    assert error.value.code == 2
    assert fake_runtime == []


def test_help_warns_of_usage_and_is_read_only(fake_runtime, capsys):
    with pytest.raises(SystemExit) as error:
        doctor.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "consumes subscription usage" in output
    assert "--probe" in output
    assert fake_runtime == []


@pytest.mark.parametrize("check", doctor._PROBE_CHECKS)
def test_check_selection_requires_explicit_probe_without_starting_runtime(fake_runtime, capsys, check):
    assert doctor.main(["--llm-backend", "codex_cli", "--probe-check", check]) == 2
    report = json.loads(capsys.readouterr().out)
    assert "requires --probe" in report["hint"]
    assert report["error"]["code"] == "configuration_error"
    assert report["planned_live_requests"] == report["live_requests_attempted"] == 0
    assert fake_runtime == []


@pytest.mark.parametrize("selected", [
    ["text"], ["structured_output"], ["ordered_images"],
    ["structured_output", "ordered_images"],
    ["ordered_images", "structured_output", "ordered_images"],
])
def test_subset_probes_only_make_selected_calls_and_never_claim_full_verification(fake_runtime, capsys, selected):
    argv = ["--llm-backend", "codex_cli", "--probe"]
    for check in selected:
        argv.extend(["--probe-check", check])
    assert doctor.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    expected = [check for check in doctor._PROBE_CHECKS if check in selected]
    assert report["status"] == "passed_selected_probes"
    assert report["selected_probe_checks"] == expected
    assert report["passed_probe_checks"] == expected
    assert report["live_inference_verified"] is False
    assert report["all_three_checks_verified"] is False
    assert report["full_v2_parity_verified"] is False
    assert report["planned_live_requests"] == report["live_requests_attempted"] == len(expected)
    assert len(fake_runtime[0].calls) == len(expected)
    assert [check["name"] for check in report["probes"][0]["checks"]] == expected
    assert report["probes"][0]["passed_checks"] == expected
    assert report["probes"][0]["all_three_checks_verified"] is False
    assert fake_runtime[0].closed


def test_explicit_all_checks_still_verify_full_smoke_suite(fake_runtime, capsys):
    argv = ["--llm-backend", "codex_cli", "--probe"]
    for check in reversed(doctor._PROBE_CHECKS):
        argv.extend(["--probe-check", check])
    assert doctor.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "probes_passed"
    assert report["live_inference_verified"] is report["all_three_checks_verified"] is True
    assert report["planned_live_requests"] == len(fake_runtime[0].calls) == 3


def test_subset_request_count_covers_every_backend_model_target(fake_runtime, tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mode": "cli", "roles": {"writing": {"backend": "claude_cli"}}}))
    assert doctor.main([
        "--inference-config", str(config), "--probe", "--probe-check", "structured_output",
        "--probe-check", "ordered_images",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["planned_live_requests"] == report["live_requests_attempted"] == 4
    assert report["probe_target_count"] == 2
    assert len(report["probes"]) == 2
    assert all(target["passed_checks"] == ["structured_output", "ordered_images"] for target in report["probes"])
    assert report["live_inference_verified"] is False


def test_partial_target_success_does_not_claim_full_or_universal_check_success(fake_runtime, tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mode": "cli", "roles": {"writing": {"backend": "claude_cli"}}}))
    original = FakeRuntime.generate

    def fail_second_target(self, client, **kwargs):
        if client.backend == "claude_cli":
            raise QuotaExhausted("quota")
        return original(self, client, **kwargs)

    monkeypatch.setattr(FakeRuntime, "generate", fail_second_target)
    assert doctor.main(["--inference-config", str(config), "--probe"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["probes"][0]["passed_checks"] == list(doctor._PROBE_CHECKS)
    assert report["probes"][0]["all_three_checks_verified"] is True
    assert report["probes"][1]["passed_checks"] == []
    assert report["passed_probe_checks"] == []
    assert report["live_inference_verified"] is report["all_three_checks_verified"] is False
    assert report["planned_live_requests"] == 6
    assert report["live_requests_attempted"] == 4


def test_images_only_probe_does_not_generate_text_nonce(fake_runtime, monkeypatch, capsys):
    monkeypatch.setattr(doctor.secrets, "token_hex", lambda *a: pytest.fail("Unselected text/schema nonce generated"))
    assert doctor.main(["--llm-backend", "codex_cli", "--probe", "--probe-check", "ordered_images"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["live_requests_attempted"] == 1


@pytest.mark.parametrize("value,text_output,encoded", [
    ("  exact-nonce\n", True, b'"exact-nonce"'),
    ({"b": 2, "a": 1}, False, b'{"a":1,"b":2}'),
    ({"colors": ["blue", "red", "green"]}, False, b'{"colors":["blue","red","green"]}'),
    ({"label": "日本語"}, False, '{"label":"日本語"}'.encode("utf-8")),
])
def test_output_fingerprint_encoding_is_reproducible_without_plaintext(value, text_output, encoded):
    fingerprint = doctor._output_fingerprint(value, text_output=text_output)
    assert fingerprint["algorithm"] == "sha256"
    assert fingerprint["encoding"] == "utf-8"
    assert fingerprint["digest"] == hashlib.sha256(encoded).hexdigest()
    assert fingerprint["normalization"] == ("text.strip()" if text_output else "structured_json_value")
    assert set(fingerprint) == {"algorithm", "encoding", "canonicalization", "normalization", "digest"}


def test_fingerprint_is_recorded_before_generate_and_survives_parser_failure():
    expected = "doctor-exact-original-nonce"
    report = {"stage": "starting", "live_requests_attempted": 0}
    target = {"checks": []}

    def failure(client, **kwargs):
        assert "expected_output" not in kwargs
        check = target["checks"][0]
        assert check["status"] == "running"
        assert check["expected_output_fingerprint"] == doctor._output_fingerprint(expected, text_output=True)
        assert report["live_requests_attempted"] == 1
        raise UnsupportedCapability("Parser rejected a completed response")

    with pytest.raises(UnsupportedCapability):
        doctor._call_probe(SimpleNamespace(generate=failure), object(), report, target, "text",
                           expected_output=expected, prompt="not printed")
    assert target["checks"][0]["status"] == "failed"
    assert target["checks"][0]["expected_output_fingerprint"] == doctor._output_fingerprint(expected, text_output=True)
    assert expected not in json.dumps(target)


def test_all_successful_checks_include_matching_expected_fingerprints(fake_runtime, monkeypatch, capsys):
    original = FakeRuntime.generate
    expected = []

    def capture(self, client, **kwargs):
        result = original(self, client, **kwargs)
        text_output = "schema" not in kwargs
        expected.append(doctor._output_fingerprint(result.text if text_output else result.structured_data,
                                                   text_output=text_output))
        return result

    monkeypatch.setattr(FakeRuntime, "generate", capture)
    assert doctor.main(["--llm-backend", "claude_cli", "--probe"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [check["expected_output_fingerprint"] for check in report["probes"][0]["checks"]] == expected
    assert report["status"] == "probes_passed"
    assert report["live_requests_attempted"] == report["planned_live_requests"] == 3


def test_failed_probe_json_preserves_original_nonce_fingerprint(fake_runtime, monkeypatch, capsys):
    nonce = "doctor-not-an-authentication-token"
    monkeypatch.setattr(doctor.secrets, "token_hex", lambda *a: nonce.removeprefix("doctor-"))
    monkeypatch.setattr(FakeRuntime, "generate", lambda *a, **kw: (_ for _ in ()).throw(UnsupportedCapability("parser failure")))
    assert doctor.main(["--llm-backend", "claude_cli", "--probe"]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert nonce not in output
    assert report["probes"][0]["checks"][0]["expected_output_fingerprint"] == doctor._output_fingerprint(nonce, text_output=True)
    assert report["live_requests_attempted"] == 1
    assert report["live_inference_verified"] is False
