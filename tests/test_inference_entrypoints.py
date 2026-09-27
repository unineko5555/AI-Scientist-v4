"""Offline workflow tests, without importing GPU/PDF SDK dependencies.

The production function definitions are compiled unchanged from their ASTs.
Only external services (inference, experiments, TeX) are replaced by fakes; the
entrypoint control flow, disk artifacts and failure propagation are exercised.
"""

import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from ai_scientist import perform_llm_review as paper_review
from ai_scientist.inference import InferenceClient
from ai_scientist.inference.errors import (
    ConfigurationError,
    InferenceError as RuntimeInferenceError,
    IncompleteOutput,
    InvalidStructuredOutput,
    QuotaExhausted,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = "launch_scientist_bfts.py"
WRITERS = [
    "ai_scientist/perform_writeup.py",
    "ai_scientist/perform_icbinb_writeup.py",
]


class InferenceError(Exception):
    """Test transport failure with the production catch-contract name."""


def _add_inference_arguments(parser):
    parser.add_argument("--inference-config")
    parser.add_argument("--llm-backend", choices=["api", "codex_cli", "claude_cli"])
    parser.add_argument("--llm-model")
    parser.add_argument("--inference-log-dir")


def _load(relative_path, names, **overrides):
    """Load real functions/constants, excluding heavyweight module imports."""
    namespace = {
        "__file__": str(ROOT / relative_path),
        "argparse": argparse,
        "datetime": datetime,
        "json": json,
        "os": os,
        "osp": os.path,
        "re": re,
        "shutil": shutil,
        "sys": sys,
        "traceback": traceback,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "InferenceError": InferenceError,
        "ConfigurationError": ConfigurationError,
        "add_inference_arguments": _add_inference_arguments,
    }
    namespace.update(overrides)
    source = ast.parse((ROOT / relative_path).read_text())
    selected = []
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
    assert selected, f"No definitions selected from {relative_path}"
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(ROOT / relative_path), "exec"), namespace)
    return namespace


def _raise_inference(*args, **kwargs):
    raise InferenceError("subscription quota exhausted")


@pytest.mark.parametrize(
    "filenames,expected",
    [
        ([], None),
        (["paper.pdf"], "paper.pdf"),
        (["paper_9.pdf", "paper_10.pdf"], "paper_10.pdf"),
        (["paper_9.pdf", "paper_final.pdf"], "paper_final.pdf"),
        (["paper_reflection1.pdf", "paper_reflection10.pdf"], "paper_reflection10.pdf"),
        (
            ["paper_reflection10.pdf", "paper_reflection_final_page_limit.pdf"],
            "paper_reflection_final_page_limit.pdf",
        ),
    ],
)
def test_pdf_selection_supports_both_v2_formats(tmp_path, filenames, expected):
    for name in filenames:
        (tmp_path / name).touch()
    namespace = _load(LAUNCHER, {"find_pdf_path_for_review"})
    result = namespace["find_pdf_path_for_review"](str(tmp_path))
    assert result == (str(tmp_path / expected) if expected else None)


def test_launcher_accepts_cli_model_names_and_declares_role_aliases():
    namespace = _load(LAUNCHER, {"parse_arguments"})
    args = namespace["parse_arguments"](
        ["--llm-backend", "claude_cli", "--model_review", "sonnet"]
    )
    assert args.model_review == "sonnet"
    assert args._inference_model_roles["model_review"] == ["review_text", "review_vision"]
    assert args._inference_model_roles["model_writeup"] == ["writing"]


def test_launcher_help_does_not_require_pytorch():
    # Deny torch even on developer machines where it happens to be installed.
    # This executes only --help, never authentication, inference, or training.
    command = (
        "import runpy, sys; sys.modules['torch'] = None; "
        "sys.argv = ['launch_scientist_bfts.py', '--help']; "
        "runpy.run_path('launch_scientist_bfts.py', run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=ROOT, capture_output=True,
        text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--llm-backend" in result.stdout
    assert "--inference-config" in result.stdout


def test_explicit_gpu_list_can_be_parsed_without_pytorch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    namespace = _load(LAUNCHER, {"get_available_gpus"})
    assert namespace["get_available_gpus"]("0,2") == [0, 2]
    with pytest.raises(ModuleNotFoundError):
        namespace["get_available_gpus"]()


def test_gpu_discovery_still_uses_pytorch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 3)
    ))
    namespace = _load(LAUNCHER, {"get_available_gpus"})
    assert namespace["get_available_gpus"]() == [0, 1, 2]


@pytest.mark.parametrize(
    "relative,extra,expected_roles",
    [
        (WRITERS[0], ["--big-model", "opus"], ["citation", "writing_vision"]),
        (WRITERS[1], ["--big-model", "opus"], ["citation", "writing_vision"]),
        ("ai_scientist/perform_plotting.py", [], ["aggregation"]),
        ("ai_scientist/perform_ideation_temp_free.py", [], ["ideation"]),
    ],
)
def test_standalone_config_preflight_and_shutdown(tmp_path, relative, extra, expected_roles):
    calls = []
    captured = []

    def configure(args, **kwargs):
        captured.append(args)
        calls.append("configure")
        return SimpleNamespace(preflight=lambda: calls.append("preflight"))

    def run(**kwargs):
        calls.append("run")
        raise InferenceError("quota")

    namespace = _load(
        relative,
        {"main"},
        configure_from_args=configure,
        shutdown_inference=lambda: calls.append("shutdown"),
        perform_writeup=run,
        aggregate_plots=run,
        create_client=lambda model, **kw: (object(), model),
        generate_temp_free_idea=run,
    )
    argv = ["--llm-backend", "claude_cli", "--model", "sonnet"] + extra
    if "ideation" in relative:
        workshop = tmp_path / "workshop.md"
        workshop.write_text("A workshop.")
        argv += ["--workshop-file", str(workshop)]
    else:
        argv += ["--folder", str(tmp_path)]
    with pytest.raises(InferenceError):
        namespace["main"](argv)
    assert calls == ["configure", "preflight", "run", "shutdown"]
    assert captured[0]._inference_model_roles["model"] == expected_roles


def test_launcher_preflight_failure_stops_before_gpu_or_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ideas_file = tmp_path / "ideas.json"
    ideas_file.write_text(json.dumps([{"Name": "offline_test"}]))
    calls = []
    namespace = _load(
        LAUNCHER,
        {"main", "parse_arguments", "write_run_status"},
        configure_from_args=lambda *a, **kw: SimpleNamespace(preflight=_raise_inference),
        load_experiment_inference_defaults=lambda path: {},
        get_available_gpus=lambda: pytest.fail("GPU work before preflight"),
        run_pipeline=lambda *a: pytest.fail("Pipeline ran after failed preflight"),
        save_token_tracker=lambda path: calls.append("save"),
        shutdown_inference=lambda: calls.append("shutdown"),
        cleanup_owned_processes=lambda: calls.append("cleanup"),
    )
    assert namespace["main"](["--load_ideas", str(ideas_file)]) == 2
    status_files = list((tmp_path / "experiments").glob("*/run_status.json"))
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text())
    assert status["status"] == "inference_failed"
    assert status["stage"] == "preflight"
    assert status["error_type"] == "InferenceError"
    assert calls == ["save", "shutdown", "cleanup"]


@pytest.mark.parametrize(
    "content,expected",
    [
        ("agent: {}\n", {}),
        ("inference: {}\nagent: {}\n", {}),
        (
            "inference:\n  mode: cli\n  subscription_only: true\n"
            "  default:\n    backend: codex_cli\n    model: null\nagent: {}\n",
            {"mode": "cli", "subscription_only": True,
             "default": {"backend": "codex_cli", "model": None}},
        ),
    ],
)
def test_launcher_reads_only_experiment_inference_defaults(tmp_path, content, expected):
    path = tmp_path / "bfts_config.yaml"
    path.write_text(content)
    namespace = _load(LAUNCHER, {"load_experiment_inference_defaults"})
    assert namespace["load_experiment_inference_defaults"](path) == expected


@pytest.mark.parametrize(
    "content", ["", "[]", "inference: false\n", "inference: null\n", "inference: [unclosed"]
)
def test_invalid_experiment_yaml_never_defaults_to_api(tmp_path, content):
    path = tmp_path / "bfts_config.yaml"
    path.write_text(content)
    namespace = _load(LAUNCHER, {"load_experiment_inference_defaults"})
    with pytest.raises(ConfigurationError):
        namespace["load_experiment_inference_defaults"](path)


@pytest.mark.parametrize(
    "override,expected_backend",
    [("none", "codex_cli"), ("profile", "claude_cli"), ("backend", "api"),
     ("profile_and_backend", "api")],
)
def test_launcher_yaml_policy_and_explicit_override_precedence(
    tmp_path, monkeypatch, override, expected_backend
):
    from ai_scientist.inference.config import config_from_args

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bfts_config.yaml").write_text(
        "inference:\n  mode: cli\n  subscription_only: true\n"
        "  default:\n    backend: codex_cli\n    model: null\nagent: {}\n"
    )
    ideas_file = tmp_path / "ideas.json"
    ideas_file.write_text(json.dumps([{"Name": "configuration_test"}]))
    profile = tmp_path / "explicit.json"
    profile.write_text(json.dumps({
        "mode": "cli", "subscription_only": True,
        "default": {"backend": "claude_cli", "model": None},
    }))
    argv = ["--load_ideas", str(ideas_file)]
    if "profile" in override:
        argv += ["--inference-config", str(profile)]
    if "backend" in override:
        argv += ["--llm-backend", "api"]
    resolved = []
    calls = []

    def configure(args, *, default_config, argv, **kwargs):
        # Use the real core config resolver, not a test recreation of the
        # precedence rules. Only process startup/model execution are mocked.
        config = config_from_args(args, argv, default_config=default_config)
        resolved.append(config)
        return SimpleNamespace(
            preflight=lambda: calls.append("preflight"),
            ensure_started=lambda: calls.append("start"),
        )

    namespace = _load(
        LAUNCHER,
        {"main", "parse_arguments", "write_run_status", "load_experiment_inference_defaults"},
        InferenceError=RuntimeInferenceError,
        configure_from_args=configure,
        get_available_gpus=lambda: calls.append("gpu") or [],
        run_pipeline=lambda *a: calls.append("pipeline"),
        save_token_tracker=lambda path: None,
        shutdown_inference=lambda: calls.append("shutdown"),
        cleanup_owned_processes=lambda: None,
    )
    assert namespace["main"](argv) == 0
    assert len(resolved) == 1
    assert resolved[0].default["backend"] == expected_backend
    assert resolved[0].subscription_only is (expected_backend != "api")
    assert resolved[0].mode == ("api" if expected_backend == "api" else "cli")
    assert calls == ["preflight", "start", "gpu", "pipeline", "shutdown"]


def test_launcher_invalid_yaml_policy_stops_before_preflight(tmp_path, monkeypatch):
    from ai_scientist.inference.config import config_from_args

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bfts_config.yaml").write_text(
        "inference:\n  mode: api\n  subscription_only: true\n"
    )
    ideas_file = tmp_path / "ideas.json"
    ideas_file.write_text(json.dumps([{"Name": "invalid_policy"}]))

    def configure(args, *, default_config, argv, **kwargs):
        config_from_args(args, argv, default_config=default_config)
        pytest.fail("Invalid subscription-only policy was accepted")

    namespace = _load(
        LAUNCHER,
        {"main", "parse_arguments", "write_run_status", "load_experiment_inference_defaults"},
        InferenceError=RuntimeInferenceError,
        configure_from_args=configure,
        get_available_gpus=lambda: pytest.fail("GPU work after invalid configuration"),
        run_pipeline=lambda *a: pytest.fail("Pipeline after invalid configuration"),
        save_token_tracker=lambda path: None,
        shutdown_inference=lambda: None,
        cleanup_owned_processes=lambda: None,
    )
    assert namespace["main"](["--load_ideas", str(ideas_file)]) == 2
    status_file = next((tmp_path / "experiments").glob("*/run_status.json"))
    assert json.loads(status_file.read_text())["error_type"] == "ConfigurationError"


@pytest.mark.parametrize("writeup_type", ["normal", "icbinb"])
@pytest.mark.parametrize("vision_failure", [False, True])
def test_launcher_preserves_pipeline_reviews_and_partial_artifacts(
    tmp_path, writeup_type, vision_failure
):
    calls = []
    client_roles = []

    def experiment(_config):
        calls.append("experiments")
        data = tmp_path / "logs/0-run/experiment_results"
        data.mkdir(parents=True)
        (data / "metrics.json").write_text('{"accuracy": 0.75}')

    def writeup(kind, **kwargs):
        assert kwargs["citations_text"] == "existing references"
        calls.append(kind)
        (tmp_path / "paper_final.pdf").write_text("offline PDF placeholder")
        return True

    def client(model, *, role):
        client_roles.append(role)
        return role, model

    def vision(*args):
        calls.append("vision_review")
        if vision_failure:
            raise InferenceError("quota")
        return {"images": "reviewed"}

    namespace = _load(
        LAUNCHER,
        {"run_pipeline", "parse_arguments", "find_pdf_path_for_review", "write_run_status"},
        idea_to_markdown=lambda *a: None,
        edit_bfts_config_file=lambda *a: "config.yaml",
        perform_experiments_bfts=experiment,
        aggregate_plots=lambda **kw: calls.append("plots"),
        gather_citations=lambda *a, **kw: "existing references",
        perform_writeup=lambda **kw: writeup("normal", **kw),
        perform_icbinb_writeup=lambda **kw: writeup("icbinb", **kw),
        load_paper=lambda path: "paper content",
        create_client=client,
        create_vlm_client=client,
        perform_review=lambda *a: calls.append("text_review") or {"Decision": "Reject"},
        perform_imgs_cap_ref_review=vision,
        save_token_tracker=lambda path: calls.append("usage"),
    )
    args = namespace["parse_arguments"](["--writeup-type", writeup_type])
    if vision_failure:
        with pytest.raises(InferenceError):
            namespace["run_pipeline"](args, [{"Name": "test"}], str(tmp_path))
    else:
        namespace["run_pipeline"](args, [{"Name": "test"}], str(tmp_path))
        assert calls[-1] == "usage"
        assert json.loads((tmp_path / "review_img_cap_ref.json").read_text()) == {
            "images": "reviewed"
        }
    assert calls.index("experiments") < calls.index("plots") < calls.index(writeup_type)
    assert client_roles == ["review_text", "review_vision"]
    assert json.loads((tmp_path / "review_text.txt").read_text())["Decision"] == "Reject"
    assert (tmp_path / "logs/0-run/experiment_results/metrics.json").exists()


@pytest.mark.parametrize("relative", WRITERS)
def test_citation_helpers_propagate_inference_failure(relative):
    namespace = _load(
        relative,
        {"get_citation_addition"},
        get_response_from_llm=_raise_inference,
    )
    with pytest.raises(InferenceError):
        namespace["get_citation_addition"](None, "model", ("report", ""), 0, 20, "idea")


def test_citation_progress_preserved_on_subscription_limit(tmp_path):
    count = 0

    def citation(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise InferenceError("quota")
        return "@article{example,\n title = {A measured result}\n}", False

    namespace = _load(
        WRITERS[1],
        {"gather_citations"},
        load_idea_text=lambda path: "idea",
        load_exp_summaries=lambda path: {},
        filter_experiment_summaries=lambda data, **kw: data,
        create_client=lambda *a, **kw: (None, "model"),
        get_citation_addition=citation,
    )
    with pytest.raises(InferenceError):
        namespace["gather_citations"](str(tmp_path), num_cite_rounds=20)
    assert count == 2
    assert "A measured result" in (tmp_path / "cached_citations.bib").read_text()
    assert json.loads((tmp_path / "citations_progress.json").read_text()) == {
        "completed_rounds": 1,
        "status": "inference_failed",
    }


def test_ideation_keeps_completed_ideas_before_later_inference_failure(tmp_path):
    responses = iter([
        'ACTION: FinalizeIdea\nARGUMENTS: {"idea": {"Name": "first_idea"}}',
        InferenceError("quota"),
    ])

    def respond(**kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result, []

    namespace = _load(
        "ai_scientist/perform_ideation_temp_free.py",
        {"generate_temp_free_idea"},
        get_response_from_llm=respond,
        idea_generation_prompt="{workshop_description}\n{prev_ideas_string}",
        idea_reflection_prompt="{current_round}/{num_reflections}: {last_tool_results}",
        system_prompt="research",
        tools_dict={},
    )
    idea_file = tmp_path / "ideas.json"
    with pytest.raises(InferenceError):
        namespace["generate_temp_free_idea"](
            str(idea_file), None, "model", "workshop", max_num_generations=2
        )
    assert json.loads(idea_file.read_text()) == [{"Name": "first_idea"}]


@pytest.mark.parametrize("fail_during_reflection", [False, True])
def test_aggregation_does_not_swallow_inference_failure(tmp_path, fail_during_reflection):
    count = 0

    def respond(**kwargs):
        nonlocal count
        count += 1
        if fail_during_reflection and count == 1:
            return "```python\nprint('offline')\n```", []
        raise InferenceError("quota")

    namespace = _load(
        "ai_scientist/perform_plotting.py",
        {"aggregate_plots", "extract_code_snippet"},
        create_client=lambda *a, **kw: (None, "model"),
        load_idea_text=lambda path: "idea",
        load_exp_summaries=lambda path: {},
        filter_experiment_summaries=lambda data, **kw: data,
        build_aggregator_prompt=lambda *a: "plot",
        AGGREGATOR_SYSTEM_MSG="plotting",
        MAX_FIGURES=12,
        get_response_from_llm=respond,
        run_aggregator_script=lambda *a: "executed by fake",
    )
    with pytest.raises(InferenceError):
        namespace["aggregate_plots"](str(tmp_path))
    assert count == (2 if fail_during_reflection else 1)


@pytest.mark.parametrize("cli_mode", [False, True])
def test_aggregator_subprocess_uses_mode_appropriate_environment(tmp_path, monkeypatch, cli_mode):
    from ai_scientist.inference.environment import generated_code_environment

    protected_names = {
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY",
        "AI_SCIENTIST_INFERENCE_ENDPOINT", "AI_SCIENTIST_INFERENCE_CONFIG",
    }
    for name in protected_names:
        monkeypatch.setenv(name, "offline-test-placeholder")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("SCIENTIST_TEST_DATASET_CACHE", str(tmp_path / "datasets"))
    observed = {}

    def run(command, **kwargs):
        environment = kwargs["env"]
        # Record names and nonsecret test markers only. A failed assertion must
        # never dump the developer's complete inherited environment.
        observed["protected_names"] = set(environment).intersection(protected_names)
        observed["gpu"] = environment.get("CUDA_VISIBLE_DEVICES")
        observed["cache"] = environment.get("SCIENTIST_TEST_DATASET_CACHE")
        observed["has_path"] = "PATH" in environment
        observed["command"] = command
        return SimpleNamespace(stdout="offline aggregator", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    namespace = _load(
        "ai_scientist/perform_plotting.py", {"run_aggregator_script"},
        subprocess=subprocess,
        get_runtime=lambda: SimpleNamespace(is_cli=cli_mode),
        generated_code_environment=generated_code_environment,
    )
    result = namespace["run_aggregator_script"](
        "print('offline aggregator')", str(tmp_path / "aggregator.py"),
        str(tmp_path), "aggregator.py",
    )
    assert result == "offline aggregator\n"
    assert observed["protected_names"] == (set() if cli_mode else protected_names)
    assert observed["gpu"] == "2"
    assert observed["cache"] == str(tmp_path / "datasets")
    assert observed["has_path"]
    assert observed["command"] == [sys.executable, "aggregator.py"]


def test_aggregator_environment_configuration_failure_is_not_a_scientific_error(tmp_path):
    namespace = _load(
        "ai_scientist/perform_plotting.py", {"run_aggregator_script"},
        subprocess=subprocess,
        get_runtime=_raise_inference,
        generated_code_environment=lambda **kwargs: {},
    )
    with pytest.raises(InferenceError):
        namespace["run_aggregator_script"](
            "print('not executed')", str(tmp_path / "aggregator.py"),
            str(tmp_path), "aggregator.py",
        )


@pytest.mark.parametrize("relative", WRITERS)
@pytest.mark.parametrize("fail_inference", [False, True])
def test_writeups_keep_formats_and_zero_reflection_artifacts(
    tmp_path, monkeypatch, relative, fail_inference
):
    monkeypatch.chdir(ROOT)
    client_roles = []
    prompts = []
    latex = r"\documentclass{article}\begin{document}A result.\end{document}"

    def client(model, *, role):
        client_roles.append(role)
        return role, model

    def respond(*, prompt, **kwargs):
        # Requiring prompt= also catches the old normal-writeup msg= bug.
        prompts.append(prompt)
        if fail_inference:
            raise InferenceError("quota")
        return (f"```latex\n{latex}\n```" if len(prompts) == 1 else "I am done"), []

    def compile_pdf(folder, destination):
        Path(destination).write_text(Path(folder, "template.tex").read_text())

    namespace = _load(
        relative,
        {"perform_writeup", "writeup_prompt", "writeup_system_message_template"},
        create_client=client,
        create_vlm_client=client,
        get_response_from_llm=respond,
        compile_latex=compile_pdf,
        load_idea_text=lambda path: "idea",
        load_exp_summaries=lambda path: {},
        filter_experiment_summaries=lambda data, **kw: data,
        get_reflection_page_info=lambda *a: "Current main text uses 4 pages.",
    )
    kwargs = dict(
        base_folder=str(tmp_path),
        citations_text="@article{measured, title = {Measured result}}",
        n_writeup_reflections=0,
    )
    if fail_inference:
        with pytest.raises(InferenceError):
            namespace["perform_writeup"](**kwargs)
        assert len(prompts) == 1
        assert (tmp_path / "latex/template.tex").exists()
    else:
        assert namespace["perform_writeup"](**kwargs)
        final_pdfs = list(tmp_path.glob("*final*.pdf"))
        assert len(final_pdfs) == 1
        assert final_pdfs[0].read_text() == latex
        assert "Measured result" in prompts[0]
        if "icbinb" in relative:
            assert "Current main text uses 4 pages." in prompts[1]
            assert "{reflection_page_info}" not in prompts[1]
    assert client_roles == ["writing_vision", "writing"]


def _complete_review(**overrides):
    review = {
        "Summary": "A reproducible negative result.",
        "Strengths": [],
        "Weaknesses": ["The hypothesis was not supported."],
        "Originality": 1,
        "Quality": 1,
        "Clarity": 1,
        "Significance": 1,
        "Questions": [],
        "Limitations": ["Only the stated experimental setting was studied."],
        "Ethical Concerns": False,
        "Soundness": 1,
        "Presentation": 1,
        "Contribution": 1,
        "Overall": 1,
        "Confidence": 1,
        "Decision": "Reject",
    }
    review.update(overrides)
    return review


def test_review_reflection_rounds_are_formatted(monkeypatch):
    prompts = []

    def respond(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(_complete_review()), []

    monkeypatch.setattr(paper_review, "get_response_from_llm", respond)
    result = paper_review.perform_review(
        "paper", "model", None, num_fs_examples=0, num_reflections=3
    )
    assert result["Decision"] == "Reject"
    assert prompts[1].startswith("Round 2/3.")
    assert prompts[2].startswith("Round 3/3.")


def test_ensemble_review_preserves_averaging_and_meta_decision(monkeypatch):
    reviews = [
        _complete_review(Decision="Reject", Overall=2, Confidence=2),
        _complete_review(Decision="Reject", Overall=6, Confidence=4),
        _complete_review(Decision="Accept", Overall=7, Confidence=3),
    ]
    histories = [
        [{"role": "user", "content": "paper"}, {"role": "assistant", "content": str(i)}]
        for i in range(3)
    ]
    monkeypatch.setattr(
        paper_review, "get_batch_responses_from_llm",
        lambda *a, **kw: ([json.dumps(review) for review in reviews], histories),
    )
    monkeypatch.setattr(
        paper_review, "get_meta_review",
        lambda *a: _complete_review(Decision="Accept", Overall=9, Confidence=5),
    )
    review, history = paper_review.perform_review(
        "paper", "model", None, num_fs_examples=0, num_reviews_ensemble=3,
        return_msg_history=True,
    )
    assert review == _complete_review(Decision="Accept", Overall=5, Confidence=3)
    assert history[0] == {"role": "user", "content": "paper"}
    assert "3 reviewers" in history[-1]["content"]
    assert histories[0][-1]["content"] == "0"  # Independent branch history retained.


SCORE_LIMITS = {
    "Originality": 4, "Quality": 4, "Clarity": 4, "Significance": 4,
    "Soundness": 4, "Presentation": 4, "Contribution": 4,
    "Overall": 10, "Confidence": 5,
}


@pytest.mark.parametrize("field,maximum", SCORE_LIMITS.items())
def test_review_validator_accepts_every_integer_in_each_score_domain(field, maximum):
    for value in range(1, maximum + 1):
        review = _complete_review(**{field: value})
        original = json.dumps(review, sort_keys=True)
        assert paper_review.validate_review(review) is review
        assert json.dumps(review, sort_keys=True) == original


@pytest.mark.parametrize("field,maximum", SCORE_LIMITS.items())
def test_review_validator_rejects_wrong_types_and_both_score_boundaries(field, maximum):
    for value in (-1, 0, maximum + 1, 1.0, 1.5, "1", True, False, None, float("nan")):
        with pytest.raises(InvalidStructuredOutput):
            paper_review.validate_review(_complete_review(**{field: value}))


@pytest.mark.parametrize("missing", _complete_review().keys())
def test_review_validator_requires_all_sixteen_fields(missing):
    review = _complete_review()
    del review[missing]
    with pytest.raises(InvalidStructuredOutput):
        paper_review.validate_review(review)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("Summary", []),
        ("Strengths", "none"),
        ("Weaknesses", [False]),
        ("Questions", [1]),
        ("Limitations", [None]),
        ("Ethical Concerns", "false"),
        ("Ethical Concerns", 0),
        ("Decision", "Weak Accept"),
        ("Decision", "reject"),
        ("Decision", True),
        ("Invented field", "not part of V2"),
    ],
)
def test_review_validator_rejects_invalid_non_score_fields(field, invalid):
    with pytest.raises(InvalidStructuredOutput):
        paper_review.validate_review(_complete_review(**{field: invalid}))


@pytest.mark.parametrize("decision", ["Accept", "Reject"])
@pytest.mark.parametrize("ethical_concerns", [False, True])
def test_review_validator_preserves_negative_evaluations(decision, ethical_concerns):
    review = _complete_review(**{"Decision": decision, "Ethical Concerns": ethical_concerns})
    assert paper_review.validate_review(review) == review
    assert review["Strengths"] == []
    assert all(review[score] == 1 for score in SCORE_LIMITS)


def test_review_validation_generated_roundtrip_and_nonmutation():
    # The declared development dependency is optional for minimal installations;
    # the concrete boundary tests above run even when Hypothesis is unavailable.
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    properties = {
        "Summary": st.text(max_size=80),
        "Strengths": st.lists(st.text(max_size=40), max_size=3),
        "Weaknesses": st.lists(st.text(max_size=40), max_size=3),
        "Questions": st.lists(st.text(max_size=40), max_size=3),
        "Limitations": st.lists(st.text(max_size=40), max_size=3),
        "Ethical Concerns": st.booleans(),
        "Decision": st.sampled_from(["Accept", "Reject"]),
        **{field: st.integers(min_value=1, max_value=maximum)
           for field, maximum in SCORE_LIMITS.items()},
    }

    @hypothesis.settings(max_examples=100, deadline=None)
    @hypothesis.example(_complete_review())
    @hypothesis.given(st.fixed_dictionaries(properties))
    def valid_reviews_roundtrip_without_scientific_changes(review):
        before = json.loads(json.dumps(review))
        validated = paper_review.validate_review(review)
        assert validated is review
        assert validated == before
        # Object field order and JSON serialization must not alter a decision,
        # ethical flag, list contents, or a low/negative numerical evaluation.
        reordered = dict(reversed(list(json.loads(json.dumps(validated)).items())))
        assert paper_review.validate_review(reordered) == before

    valid_reviews_roundtrip_without_scientific_changes()


@pytest.mark.parametrize("backend", ["codex_cli", "claude_cli"])
@pytest.mark.parametrize("failure", ["missing", "invalid", "history_missing", "empty"])
def test_cli_ensemble_never_accepts_incomplete_or_invalid_members(monkeypatch, backend, failure):
    answers = [json.dumps(_complete_review()), json.dumps(_complete_review())]
    histories = [[{"role": "assistant", "content": answer}] for answer in answers]
    expected = IncompleteOutput
    if failure == "missing":
        answers, histories = answers[:1], histories[:1]
    elif failure == "invalid":
        answers[1] = '{"Decision":"Reject"}'
        expected = InvalidStructuredOutput
    elif failure == "history_missing":
        histories = histories[:1]
    else:
        answers, histories = [], []
    monkeypatch.setattr(
        paper_review, "get_batch_responses_from_llm", lambda *a, **kw: (answers, histories)
    )
    monkeypatch.setattr(
        paper_review, "get_meta_review", lambda *a: pytest.fail("Incomplete ensemble was used")
    )
    with pytest.raises(expected):
        paper_review.perform_review(
            "paper", None, InferenceClient(backend), num_fs_examples=0, num_reviews_ensemble=2
        )


def test_api_ensemble_subset_is_explicit_and_uses_first_valid_history(monkeypatch):
    answers = ["{}", json.dumps(_complete_review(Overall=3)), json.dumps(_complete_review(Overall=7))]
    histories = [
        [{"role": "user", "content": f"branch {i}"}, {"role": "assistant", "content": answer}]
        for i, answer in enumerate(answers)
    ]
    monkeypatch.setattr(
        paper_review, "get_batch_responses_from_llm", lambda *a, **kw: (answers, histories)
    )
    monkeypatch.setattr(paper_review, "get_meta_review", lambda *a: _complete_review(Decision="Accept"))
    with pytest.warns(RuntimeWarning) as reported:
        review, history = paper_review.perform_review(
            "paper", "api-model", object(), num_fs_examples=0, num_reviews_ensemble=3,
            return_msg_history=True,
        )
    assert any("2 valid reviews out of 3 requested" in str(item.message) for item in reported)
    assert review["Overall"] == 5
    assert review["Decision"] == "Accept"
    assert history[0]["content"] == "branch 1"
    assert "2 reviewers" in history[-1]["content"]


def test_api_ensemble_with_no_valid_member_fails_explicitly(monkeypatch):
    monkeypatch.setattr(
        paper_review, "get_batch_responses_from_llm", lambda *a, **kw: (["{}", "{}"], [[], []])
    )
    with pytest.warns(RuntimeWarning), pytest.raises(InvalidStructuredOutput):
        paper_review.perform_review(
            "paper", "api-model", object(), num_fs_examples=0, num_reviews_ensemble=2
        )


@pytest.mark.parametrize("cli_mode", [False, True])
def test_invalid_meta_review_legacy_fallback_is_explicit_and_cli_has_none(monkeypatch, cli_mode):
    answers = [json.dumps(_complete_review(Overall=2)), json.dumps(_complete_review(Overall=6))]
    monkeypatch.setattr(
        paper_review, "get_batch_responses_from_llm", lambda *a, **kw: (answers, [[], []])
    )
    monkeypatch.setattr(paper_review, "get_meta_review", lambda *a: None)
    client = InferenceClient("codex_cli") if cli_mode else object()
    if cli_mode:
        with pytest.raises(InvalidStructuredOutput):
            paper_review.perform_review("paper", None, client, num_fs_examples=0, num_reviews_ensemble=2)
    else:
        with pytest.warns(RuntimeWarning, match="first valid review"):
            review = paper_review.perform_review(
                "paper", "api-model", client, num_fs_examples=0, num_reviews_ensemble=2
            )
        assert review["Decision"] == "Reject"
        assert review["Overall"] == 4


@pytest.mark.parametrize("stage", ["single", "ensemble", "meta", "reflection"])
@pytest.mark.parametrize("cli_mode", [False, True])
def test_review_never_swallows_quota_errors(monkeypatch, stage, cli_mode):
    calls = []

    def respond(*a, **kwargs):
        calls.append("response")
        if stage == "single" or (stage == "reflection" and len(calls) == 2):
            raise QuotaExhausted("quota")
        return json.dumps(_complete_review()), []

    def batch(*a, **kwargs):
        calls.append("batch")
        if stage == "ensemble":
            raise QuotaExhausted("quota")
        return [json.dumps(_complete_review())] * 2, [[], []]

    def meta(*a, **kwargs):
        calls.append("meta")
        raise QuotaExhausted("quota")

    monkeypatch.setattr(paper_review, "get_response_from_llm", respond)
    monkeypatch.setattr(paper_review, "get_batch_responses_from_llm", batch)
    monkeypatch.setattr(paper_review, "get_meta_review", meta)
    client = InferenceClient("claude_cli") if cli_mode else object()
    with pytest.raises(QuotaExhausted):
        paper_review.perform_review(
            "paper", None, client, num_fs_examples=0,
            num_reviews_ensemble=2 if stage in ("ensemble", "meta") else 1,
            num_reflections=3 if stage == "reflection" else 1,
        )
    assert len(calls) == (2 if stage in ("meta", "reflection") else 1)


def test_cli_reflection_uses_host_validation_and_keeps_early_stop(monkeypatch):
    calls = []
    valid_review = _complete_review()

    def respond(prompt, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return json.dumps(valid_review), []
        return "THOUGHT: I am done\nREVIEW JSON:\n```json\n" + json.dumps(valid_review) + "\n```", []

    monkeypatch.setattr(paper_review, "get_response_from_llm", respond)
    result = paper_review.perform_review(
        "paper", None, InferenceClient("codex_cli"), num_fs_examples=0, num_reflections=5
    )
    assert result == valid_review
    assert len(calls) == 2
    assert calls[0]["schema"] == paper_review.REVIEW_SCHEMA
    assert calls[1].get("schema") is None


@pytest.mark.parametrize("stage", ["single", "reflection", "meta"])
def test_invalid_review_outputs_fail_without_invented_defaults_or_retry(monkeypatch, stage):
    calls = []

    def respond(*a, **kwargs):
        calls.append(kwargs)
        if stage == "reflection" and len(calls) == 1:
            return json.dumps(_complete_review()), []
        return '{"Decision":"Accept","Overall":10}', []

    monkeypatch.setattr(paper_review, "get_response_from_llm", respond)
    client = InferenceClient("claude_cli")
    with pytest.raises(InvalidStructuredOutput):
        if stage == "meta":
            paper_review.get_meta_review(None, client, 0.75, [_complete_review()])
        else:
            paper_review.perform_review(
                "paper", None, client, num_fs_examples=0,
                num_reflections=3 if stage == "reflection" else 1,
            )
    assert len(calls) == (2 if stage == "reflection" else 1)


@pytest.mark.parametrize(
    "relative",
    [LAUNCHER, *WRITERS, "ai_scientist/perform_ideation_temp_free.py",
     "ai_scientist/perform_plotting.py"],
)
def test_all_entrypoint_client_factories_have_explicit_roles(relative):
    tree = ast.parse((ROOT / relative).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"create_client", "create_vlm_client"}
    ]
    assert calls
    assert all(any(keyword.arg == "role" for keyword in call.keywords) for call in calls)


def test_cleanup_targets_only_owned_descendants(monkeypatch):
    events = []
    child = SimpleNamespace(
        send_signal=lambda signum: events.append("terminate child"),
        kill=lambda: events.append("kill child"),
    )
    current = SimpleNamespace(children=lambda recursive: [child])
    waits = iter([([], [child]), ([child], [])])
    fake_psutil = SimpleNamespace(
        Process=lambda: current,
        wait_procs=lambda children, timeout: next(waits),
        NoSuchProcess=ProcessLookupError,
        AccessDenied=PermissionError,
        process_iter=lambda *a: pytest.fail("Must never scan global processes"),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    namespace = _load(LAUNCHER, {"cleanup_owned_processes"})
    namespace["cleanup_owned_processes"]()
    assert events == ["terminate child", "kill child"]
