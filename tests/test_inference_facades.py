"""Offline compatibility tests: no SDK constructors or CLI processes are used."""

from copy import deepcopy
import base64
import json
from types import SimpleNamespace

import jsonschema
import pytest
from PIL import Image

from ai_scientist import llm, vlm
from ai_scientist.inference.contracts import InferenceClient, InferenceResult
from ai_scientist.inference.errors import ConfigurationError, InvalidStructuredOutput, QuotaExhausted, UnsupportedCapability
from ai_scientist import perform_vlm_review as figures
from ai_scientist.treesearch import backend
from ai_scientist.treesearch.backend.utils import FunctionSpec, coerce_function_spec
from ai_scientist.utils.token_tracker import TokenTracker, token_tracker, track_token_usage


@pytest.fixture(autouse=True)
def isolated_legacy_mode(monkeypatch):
    # Isolate these facade unit tests from runtime state in other test modules.
    for module in (llm, vlm, backend):
        monkeypatch.setattr(module, "get_cli_client", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "resolve_api_model", lambda model, role="default": model)
    token_tracker.reset()


@pytest.fixture(params=["codex_cli", "claude_cli"])
def cli_client(request):
    return InferenceClient(request.param, None, "writing")


@pytest.fixture
def png_path(tmp_path):
    path = tmp_path / "figure.png"
    Image.new("RGBA", (3, 2), (80, 40, 10, 130)).save(path)
    return path


def result(text="answer", structured_data=None):
    return InferenceResult("request-test", text=text, structured_data=structured_data)


def test_cli_factories_precede_all_api_dispatch(monkeypatch, cli_client):
    calls = []

    def resolver(model=None, role="default"):
        calls.append((model, role))
        return cli_client

    def forbidden_sdk(*args, **kwargs):
        pytest.fail("CLI mode constructed an API SDK client")

    monkeypatch.setattr(llm, "openai", SimpleNamespace(OpenAI=forbidden_sdk))
    monkeypatch.setattr(llm, "anthropic", SimpleNamespace(Anthropic=forbidden_sdk, AnthropicBedrock=forbidden_sdk))
    monkeypatch.setattr(vlm, "openai", SimpleNamespace(OpenAI=forbidden_sdk))
    for module in (llm, vlm, backend):
        monkeypatch.setattr(module, "get_cli_client", resolver)
        monkeypatch.setattr(module, "resolve_api_model", forbidden_sdk)
    for model in (None, "gpt-4o", "claude-legacy", "bedrock/anthropic.claude-legacy", "unlisted-model"):
        assert llm.create_client(model, role="writing") == (cli_client, None)
        assert vlm.create_client(model, role="writing_vision") == (cli_client, None)
        assert backend.get_ai_client(model, role="summary") is cli_client
    assert {role for _, role in calls} == {"writing", "writing_vision", "summary"}


@pytest.mark.parametrize("legacy_model", [None, "claude-legacy", "deepcoder-14b", "gpt-4o", "unlisted"])
def test_cli_text_ignores_api_dispatch_and_keeps_selected_history(monkeypatch, cli_client, legacy_model):
    selected = [{"role": "assistant", "content": [{"type": "text", "text": "selected-last-only"}]}]
    before = deepcopy(selected)
    calls = []

    def fake_generate(client, **kwargs):
        calls.append((client, deepcopy(kwargs)))
        return result("new answer")

    monkeypatch.setattr(llm, "generate", fake_generate)
    text, history = llm.get_response_from_llm("next", cli_client, legacy_model, "system", msg_history=selected)
    assert text == "new answer"
    assert len(calls) == 1
    assert calls[0][1]["msg_history"] == before
    assert calls[0][1]["system_message"] == "system"
    assert history == before + [{"role": "user", "content": "next"}, {"role": "assistant", "content": "new answer"}]
    history[0]["content"][0]["text"] = "mutated"
    assert selected == before


def test_cli_ensemble_independent_histories(monkeypatch, cli_client):
    seed = [{"role": "assistant", "content": [{"type": "text", "text": "seed"}]}]
    seen = []

    def fake_generate(client, **kwargs):
        seen.append(deepcopy(kwargs["msg_history"]))
        return result(f"branch-{len(seen)}")

    monkeypatch.setattr(llm, "generate", fake_generate)
    answers, histories = llm.get_batch_responses_from_llm("prompt", cli_client, None, "system", msg_history=seed, n_responses=3)
    assert answers == ["branch-1", "branch-2", "branch-3"]
    assert seen == [seed, seed, seed]
    histories[0][0]["content"][0]["text"] = "branch mutation"
    assert histories[1][0] == seed[0]
    assert histories[2][0] == seed[0]


def test_cli_text_schema_reaches_each_independent_branch(monkeypatch, cli_client):
    schema = {"type": "object", "properties": {"valid": {"type": "boolean"}}, "required": ["valid"]}
    seen = []
    monkeypatch.setattr(llm, "generate", lambda *args, **kwargs: seen.append(deepcopy(kwargs)) or result('{"valid":false}', {"valid": False}))
    answers, histories = llm.get_batch_responses_from_llm("q", cli_client, None, "s", n_responses=2, schema=schema)
    assert answers == ['{"valid":false}', '{"valid":false}']
    assert [call["schema"] for call in seen] == [schema, schema]
    assert histories[0] == histories[1]
    assert histories[0] is not histories[1]


def test_api_fallback_batch_preserves_history(monkeypatch):
    seen = []

    def create(**kwargs):
        seen.append(deepcopy(kwargs["messages"]))
        return SimpleNamespace(model="claude-test", content=[SimpleNamespace(text="ok")])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    selected = [{"role": "assistant", "content": "past"}]
    answers, histories = llm.get_batch_responses_from_llm("new", client, "claude-test", "sys", msg_history=selected, n_responses=2)
    assert answers == ["ok", "ok"]
    assert seen[0][0] == selected[0] == seen[1][0]
    histories[0][0]["content"] = "changed"
    assert histories[1][0] == selected[0]


def test_api_native_batch_tracks_raw_response_not_tuple():
    response = SimpleNamespace(
        model="gpt-test", created=1,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=7),
        choices=[SimpleNamespace(message=SimpleNamespace(content="a")), SimpleNamespace(message=SimpleNamespace(content="b"))],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    answers, histories = llm.get_batch_responses_from_llm("q", client, "gpt-test", "s", n_responses=2)
    assert answers == ["a", "b"]
    assert len(histories) == 2
    histories[0][0]["content"] = "changed"
    assert histories[1][0]["content"] == "q"
    assert token_tracker.get_summary()["gpt-test"]["tokens"] == {"prompt": 10, "completion": 7, "reasoning": 0, "cached": 0}


def test_vlm_keeps_png_bytes_and_duplicate_occurrences(monkeypatch, cli_client, png_path):
    calls = []
    monkeypatch.setattr(vlm, "generate", lambda client, **kwargs: calls.append(deepcopy(kwargs)) or result("vision"))
    selected = [{"role": "assistant", "content": "old"}]
    text, history = vlm.get_response_from_vlm("compare", [png_path, png_path], cli_client, "unlisted", "system", msg_history=selected)
    assert text == "vision"
    parts = calls[0]["prompt"]
    assert len(parts) == 3
    assert parts[1] == parts[2]
    uri = parts[1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == png_path.read_bytes()
    assert history[-2]["content"] == parts
    history[0]["content"] = "changed"
    assert selected[0]["content"] == "old"


def test_vlm_image_limit_fails_without_call_or_slicing(monkeypatch, cli_client, png_path):
    monkeypatch.setattr(vlm, "generate", lambda *args, **kwargs: pytest.fail("over-limit request was sent"))
    with pytest.raises(UnsupportedCapability, match="silently dropped"):
        vlm.get_response_from_vlm("compare", [png_path, png_path], cli_client, None, "sys", max_images=1)


def test_cli_vision_ensemble_has_independent_image_histories(monkeypatch, cli_client, png_path):
    monkeypatch.setattr(vlm, "generate", lambda *args, **kwargs: result("a"))
    answers, histories = vlm.get_batch_responses_from_vlm("compare", [png_path], cli_client, None, "system", n_responses=2)
    assert answers == ["a", "a"]
    original_uri = histories[1][0]["content"][1]["image_url"]["url"]
    histories[0][0]["content"][1]["image_url"]["url"] = "changed"
    assert histories[1][0]["content"][1]["image_url"]["url"] == original_uri


@pytest.mark.parametrize("schema_key", ["parameters", "json_schema"])
def test_tree_schema_dictionary_and_multimodal_contract(monkeypatch, cli_client, png_path, schema_key):
    schema = {"type": "object", "properties": {"valid": {"type": "boolean"}}, "required": ["valid"]}
    spec = {"name": "scientific_check", "description": "Check", schema_key: schema}
    content = vlm.prepare_vlm_prompt("inspect", [png_path], 1)
    calls = []
    monkeypatch.setattr(backend, "get_cli_client", lambda *args, **kwargs: cli_client)
    monkeypatch.setattr(backend, "generate", lambda client, **kwargs: calls.append(deepcopy(kwargs)) or result('{"valid":false}', {"valid": False}))
    assert backend.query("system", content, "claude-old", func_spec=spec, role="experiment_vision") == {"valid": False}
    assert calls[0]["schema"] == schema
    assert calls[0]["prompt"] == content
    assert calls[0]["system_message"].strip() == "system"
    assert spec[schema_key] == schema


def test_tree_factory_handle_can_be_used_by_text_facade(monkeypatch, cli_client):
    monkeypatch.setattr(backend, "get_cli_client", lambda *args, **kwargs: cli_client)
    monkeypatch.setattr(llm, "generate", lambda *args, **kwargs: result("summary"))
    client = backend.get_ai_client("gpt-4o", role="summary")
    assert llm.get_response_from_llm("summarize", client, "gpt-4o", "system")[0] == "summary"


def test_function_spec_validates_actual_output_and_serialization():
    spec = coerce_function_spec({"name": "check", "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}})
    assert spec.validate({"ok": False}) == {"ok": False}
    with pytest.raises(jsonschema.ValidationError):
        spec.validate({"ok": "false"})
    assert FunctionSpec.from_json(spec.to_json()) == spec


def test_cli_mode_rejects_stale_raw_sdk_clients(monkeypatch, cli_client, png_path):
    monkeypatch.setattr(llm, "get_cli_client", lambda *args, **kwargs: cli_client)
    raw_client = SimpleNamespace()
    with pytest.raises(ConfigurationError, match="Raw SDK"):
        llm.get_response_from_llm("q", raw_client, "gpt-4o", "s")
    with pytest.raises(ConfigurationError, match="Raw SDK"):
        vlm.get_response_from_vlm("q", png_path, raw_client, "gpt-4o", "s")
    with pytest.raises(ConfigurationError, match="Raw SDK"):
        llm.make_llm_call(raw_client, "gpt-4o", 0.7, "s", [])


def test_duplicate_figures_uses_shared_facade_and_occurrence_labels(monkeypatch, cli_client, png_path, tmp_path):
    screenshots = [
        {"img_name": "Figure 1", "images": [str(png_path)]},
        {"img_name": "Figure 2", "images": [str(png_path)]},
    ]
    monkeypatch.setattr(figures, "extract_figure_screenshots", lambda *args: screenshots)
    seen = []

    def fake_vlm(*args, **kwargs):
        seen.append((args, kwargs))
        return "Duplicate figures: Figure 1, Figure 2", []

    monkeypatch.setattr(figures, "get_response_from_vlm", fake_vlm)
    assert "Duplicate figures" in figures.detect_duplicate_figures(cli_client, None, str(tmp_path / "paper.pdf"))
    args, kwargs = seen[0]
    assert args[1] == [str(png_path), str(png_path)]
    assert "1: Figure 1" in args[0] and "2: Figure 2" in args[0]
    assert kwargs["max_images"] == 2
    assert kwargs["max_tokens"] == 1000


def test_duplicate_figures_propagates_quota(monkeypatch, cli_client, png_path, tmp_path):
    monkeypatch.setattr(figures, "extract_figure_screenshots", lambda *args: [{"img_name": "Figure 1", "images": [str(png_path)]}])

    def exhausted(*args, **kwargs):
        raise QuotaExhausted("subscription limit")

    monkeypatch.setattr(figures, "get_response_from_vlm", exhausted)
    with pytest.raises(QuotaExhausted):
        figures.detect_duplicate_figures(cli_client, None, str(tmp_path / "paper.pdf"))


def test_duplicate_figures_empty_extraction_is_not_a_negative_judgment(monkeypatch, cli_client, tmp_path):
    monkeypatch.setattr(figures, "extract_figure_screenshots", lambda *args: [])
    monkeypatch.setattr(figures, "get_response_from_vlm", lambda *args, **kwargs: pytest.fail("No images available"))
    assessment = figures.detect_duplicate_figures(cli_client, None, str(tmp_path / "paper.pdf"))
    assert "not performed" in assessment
    assert "No duplicates found" not in assessment


def test_figure_review_preserves_existing_string_schema(monkeypatch, cli_client, png_path):
    fields = ("Img_description", "Img_review", "Caption_review", "Figrefs_review", "Overall_comments", "Containing_sub_figures", "Informative_review")
    expected = dict.fromkeys(fields, "review")
    seen = []

    def fake_vlm(*args, **kwargs):
        seen.append(kwargs["schema"])
        return json.dumps(expected), []

    monkeypatch.setattr(figures, "get_response_from_vlm", fake_vlm)
    image = {"images": [str(png_path)], "caption": "caption", "main_text_figrefs": []}
    assert figures.generate_vlm_img_selection_review(image, "abstract", None, cli_client, "four pages") == expected
    assert seen[0]["required"] == list(fields)
    assert all(value == {"type": "string"} for value in seen[0]["properties"].values())
    monkeypatch.setattr(figures, "get_response_from_vlm", lambda *args, **kwargs: ('{"Img_review": false}', []))
    with pytest.raises(InvalidStructuredOutput):
        figures.generate_vlm_img_review(image, None, cli_client)


@pytest.mark.parametrize("prefix,suffix", [("", ""), ("THOUGHT\n```json\n", "\n```"), ("Notes first\n", "\nNotes after")])
def test_json_extraction_preserves_nested_native_output(prefix, suffix):
    expected = {"a": {"b": [1, {"c": "brace } inside a string"}]}, "valid": False}
    text = prefix + json.dumps(expected) + suffix
    assert llm.extract_json_between_markers(text) == expected
    assert vlm.extract_json_between_markers(text) == expected


def test_token_decorator_is_transparent_for_positional_tuple_contract():
    @track_token_usage
    def returns_tuple(prompt, system_message):
        return ["answer"], [[{"role": "assistant", "content": "answer"}]]

    assert returns_tuple("prompt", "system")[0] == ["answer"]


def test_cli_token_tracker_unknown_cost_and_usage_are_not_zero():
    tracker = TokenTracker()
    tracker.add_normalized_usage("codex_cli", "gpt-4o-2024-08-06", {}, request_id="one")
    tracker.add_normalized_usage("codex_cli", "gpt-4o-2024-08-06", {}, request_id="one")
    summary = tracker.get_summary()["codex_cli:gpt-4o-2024-08-06"]
    assert summary["requests"] == 1
    assert summary["tokens"]["prompt"] is None
    assert summary["tokens"]["completion"] is None
    assert summary["cost (USD)"] is None
    assert summary["subscription_billing_cost_usd"] is None
    assert summary["usage_complete"] is False


def test_cli_token_tracker_does_not_add_cached_to_input():
    tracker = TokenTracker()
    tracker.add_normalized_usage("codex_cli", "model", {"input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 3}, request_id="one")
    summary = tracker.get_summary()["codex_cli:model"]
    assert summary["tokens"]["prompt"] == 10
    assert summary["tokens"]["cached"] == 3
    assert summary["tokens"]["completion"] == 4


def test_api_vlm_validates_opt_in_schema(png_path):
    response = SimpleNamespace(
        model="gpt-4o-2024-08-06", created=1,
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"valid": "false"}'))],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    schema = {"type": "object", "properties": {"valid": {"type": "boolean"}}, "required": ["valid"]}
    with pytest.raises(InvalidStructuredOutput):
        vlm.get_response_from_vlm("check", png_path, client, "gpt-4o-2024-08-06", "s", schema=schema)


def _use_api_profile(monkeypatch, *, default_model=None, roles=None):
    """Exercise actual profile resolution with no global runtime or SDK setup."""
    from ai_scientist.inference import InferenceConfig
    from ai_scientist.inference import runtime as inference_runtime
    config = InferenceConfig.from_dict({
        "mode": "api",
        "default": {"backend": "api", "model": default_model},
        "roles": roles or {},
    })
    runtime = SimpleNamespace(config=config, is_cli=False)
    monkeypatch.setattr(inference_runtime, "get_runtime", lambda: runtime)
    for module in (llm, vlm, backend):
        monkeypatch.setattr(module, "resolve_api_model", inference_runtime.resolve_api_model)


@pytest.mark.parametrize(
    "default_model,role_model,expected_model,expected_provider",
    [
        (None, None, "gpt-4o", "openai"),
        ("gpt-4o", "bedrock/anthropic.claude-profile", "anthropic.claude-profile", "bedrock"),
        ("gpt-4o", "claude-writing", "claude-writing", "anthropic"),
        ("claude-default", "gpt-4o-mini", "gpt-4o-mini", "openai"),
        ("claude-default", None, "claude-default", "anthropic"),
    ],
)
def test_api_text_factory_resolves_profile_before_provider_dispatch(monkeypatch, default_model, role_model, expected_model, expected_provider):
    roles = {} if role_model is None else {"writing": {"model": role_model}}
    _use_api_profile(monkeypatch, default_model=default_model, roles=roles)
    calls = []
    monkeypatch.setattr(llm, "openai", SimpleNamespace(OpenAI=lambda **kwargs: calls.append("openai") or "openai-client"))
    monkeypatch.setattr(llm, "anthropic", SimpleNamespace(
        Anthropic=lambda **kwargs: calls.append("anthropic") or "anthropic-client",
        AnthropicBedrock=lambda **kwargs: calls.append("bedrock") or "bedrock-client",
    ))
    client, model = llm.create_client("gpt-4o", role="writing")
    assert model == expected_model
    assert client == f"{expected_provider}-client"
    assert calls == [expected_provider]


def test_api_vision_factory_resolves_default_and_role_models(monkeypatch):
    _use_api_profile(monkeypatch, default_model="gpt-4o-2024-11-20", roles={
        "writing_vision": {"model": "ollama/qwen2.5vl:32b"},
    })
    calls = []
    monkeypatch.setattr(vlm, "openai", SimpleNamespace(OpenAI=lambda **kwargs: calls.append(kwargs) or "client"))
    assert vlm.create_client(None, role="review_vision") == ("client", "gpt-4o-2024-11-20")
    assert vlm.create_client("gpt-4o-2024-08-06", role="writing_vision") == ("client", "ollama/qwen2.5vl:32b")
    assert calls[0] == {}
    assert calls[1]["base_url"] == "http://localhost:11434/v1"


@pytest.mark.parametrize("profile_model,expected_provider", [("claude-feedback", "anthropic"), ("gpt-4o-mini", "openai")])
def test_api_tree_factory_and_query_resolve_profile_model(monkeypatch, profile_model, expected_provider):
    from ai_scientist.treesearch.backend import backend_anthropic, backend_openai
    _use_api_profile(monkeypatch, default_model="gpt-default", roles={"execution_feedback": {"model": profile_model}})
    factories = []
    queries = []
    for provider, module in (("anthropic", backend_anthropic), ("openai", backend_openai)):
        monkeypatch.setattr(module, "get_ai_client", lambda provider=provider, **kwargs: factories.append((provider, kwargs)) or provider)
        monkeypatch.setattr(module, "query", lambda provider=provider, **kwargs: queries.append((provider, kwargs)) or ("reply", 0, 0, 0, {}))
    assert backend.get_ai_client("gpt-legacy", role="execution_feedback", max_retries=0) == expected_provider
    assert backend.query("system", "user", "gpt-legacy", role="execution_feedback", temperature=0.4) == "reply"
    assert factories == [(expected_provider, {"model": profile_model, "max_retries": 0})]
    assert queries[0][0] == expected_provider
    assert queries[0][1]["model"] == profile_model
    assert queries[0][1]["temperature"] == 0.4


def test_api_profile_null_role_model_preserves_legacy_fallback(monkeypatch):
    _use_api_profile(monkeypatch, default_model="claude-default", roles={"writing": {"model": None}})
    monkeypatch.setattr(llm, "openai", SimpleNamespace(OpenAI=lambda **kwargs: "openai-client"))
    assert llm.create_client("gpt-4o", role="writing") == ("openai-client", "gpt-4o")


def test_api_raw_response_does_not_re_resolve_factory_model(monkeypatch):
    def must_not_resolve(*args, **kwargs):
        pytest.fail("Raw SDK response path re-resolved a factory-normalized model")

    monkeypatch.setattr(llm, "resolve_api_model", must_not_resolve)
    calls = []
    response = SimpleNamespace(model="claude-factory-model", content=[SimpleNamespace(text="answer")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response))
    assert llm.get_response_from_llm("q", client, "claude-factory-model", "s")[0] == "answer"
    assert calls[0]["model"] == "claude-factory-model"
