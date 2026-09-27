"""Offline operator tests: no real provider, model request or research experiment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from hypothesis import example, given, settings, strategies as st
import pytest

from ai_scientist.inference.config import config_from_args
from ai_scientist.inference.doctor import build_parser as doctor_parser


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/ai-scientist"
HELPER = SKILL / "scripts/operate.py"
spec = importlib.util.spec_from_file_location("ai_scientist_skill_operator", HELPER)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "ai_scientist").symlink_to(ROOT / "ai_scientist", target_is_directory=True)
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copyfile(ROOT / "bfts_config.yaml", root / "bfts_config.yaml")
    # Tests that --execute reaches this stub, never the actual experiment runner.
    (root / "launch_scientist_bfts.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'pid': os.getpid(), 'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'environment_keys': sorted(os.environ)}), flush=True)\n"
        "raise SystemExit(7)\n"
    )
    (root / "topic.md").write_text("# CPU research topic\n")
    (root / "ideas.json").write_text(json.dumps([{"Name": "cpu_trial", "Title": "CPU trial"}, {"Name": "second"}]))
    return root


def parse(*arguments):
    return operator.build_parser().parse_args(arguments)


def invoke(root, *arguments, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(HELPER), "--project-root", str(root), *arguments],
        cwd=cwd or root, env=env, text=True, capture_output=True, timeout=20,
    )


@pytest.fixture
def fake_profile(project, monkeypatch):
    executable = project / "fake-claude"
    source = (ROOT / "tests/fixtures/cli/fake_cli.py").read_text().split("\n", 1)[1]
    executable.write_text(f"#!{sys.executable}\n" + source)
    executable.chmod(0o700)
    profile = project / "fake-profile.json"
    profile.write_text(json.dumps({"mode": "cli", "subscription_only": True,
        "default": {"backend": "claude_cli", "model": None},
        "providers": {"claude_cli": {"executable": str(executable), "termination_grace_seconds": 0.1}}}))
    log = project / "fake-calls.jsonl"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    return profile, log


def test_both_hosts_resolve_to_one_skill_and_helper():
    claude_skill = ROOT / ".claude/skills/ai-scientist"
    assert claude_skill.is_symlink()
    assert claude_skill.resolve() == SKILL
    assert (claude_skill / "scripts/operate.py").resolve() == HELPER
    assert operator.project_root() == ROOT
    # All maintained references can be opened through either discovery path.
    for skill in (SKILL, claude_skill):
        reference = skill / "references/operations.md"
        assert reference.is_file()
        assert (reference.parent / "../../../../docs/cli-inference.md").resolve().is_file()


@pytest.mark.parametrize("action,extra", [
    ("doctor", []), ("ideate", ["--workshop-file", "topic.md"]), ("run", ["--ideas", "ideas.json"]),
])
def test_provider_is_required(action, extra):
    with pytest.raises(SystemExit) as error:
        parse(action, *extra)
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["doctor", "--provider", "api"],
    ["doctor", "--provider", "codex", "--inference-config", "profile.yaml"],
    ["doctor", "--provider", "codex", "--llm-backend", "api"],
    ["doctor", "--provider", "codex", "--exec"],
    ["doctor", "--provider", "codex", "--execute", "--dry-run"],
    ["doctor", "--provider", "codex", "--", "--llm-backend", "api"],
    ["run", "--provider", "codex", "--ideas", "ideas.json", "--idea-index", "-1"],
    ["ideate", "--provider", "claude", "--workshop-file", "topic.md", "--num-reflections", "0"],
])
def test_unknown_ambiguous_or_invalid_options_fail_before_any_execution(arguments):
    with pytest.raises(SystemExit) as error:
        parse(*arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("profile", [
    {}, {"mode": "api"},
    {"mode": "cli", "subscription_only": False},
    {"mode": "cli", "roles": {"review_text": {"backend": "api"}}},
    {"mode": "cli", "fallback": "api"},
    {"mode": "cli", "providers": {"claude_cli": {"skip_auth": True}}},
])
def test_bad_profiles_never_launch_a_child(project, profile):
    path = project / "bad-profile.json"
    path.write_text(json.dumps(profile))
    result = invoke(project, "run", "--inference-config", str(path), "--ideas", "ideas.json", "--execute")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert not (project / "experiments").exists()


def test_invalid_yaml_fails_cleanly_before_child(project):
    path = project / "bad-profile.yaml"
    path.write_text("inference: [broken")
    result = invoke(project, "doctor", "--inference-config", str(path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("provider", operator.PROVIDERS)
def test_provider_plan_resolves_all_roles_and_never_infers(project, provider, fake_profile):
    _, log = fake_profile
    result = invoke(project, "doctor", "--provider", provider, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["execution_requested"] is False
    assert plan["mode"] == "cli" and plan["subscription_only"] is True
    backends = {role["backend"] for role in plan["roles"].values()}
    expected = {"codex_cli", "claude_cli"} if provider == "mixed" else {provider + "_cli"}
    assert backends == expected
    assert len(plan["roles"]) == 15
    assert all(role["model"] is None for role in plan["roles"].values())
    assert not log.exists()
    assert not (project / ".inference").exists()


def test_diagnostic_only_checks_fake_cli_without_inference(project, fake_profile, monkeypatch):
    profile, log = fake_profile
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    result = invoke(project, "doctor", "--inference-config", str(profile))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ready_read_only"
    assert report["live_requests_attempted"] == 0
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("status" in record["args"] for record in records)
    assert all("stdin" not in record and "-p" not in record["args"] for record in records)


@pytest.mark.parametrize("provider,count", [("codex", 3), ("claude", 3), ("mixed", 6)])
def test_live_probes_are_plan_only_and_count_distinct_targets(project, provider, count):
    result = invoke(project, "doctor", "--provider", provider, "--probe")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["planned_live_requests"] == count
    assert plan["execution_requested"] is False
    assert plan["live_inference"] is True
    assert plan["inference_limits"]["request_timeout_seconds"] == 120
    assert plan["inference_limits"]["transient_retry_limit"] == 0
    assert plan["inference_limits"]["schema_repair_limit"] == 0
    assert not (project / ".inference").exists()


def test_selected_probes_deduplicate_and_need_probe_flag(project):
    args = parse("doctor", "--provider", "mixed", "--probe", "--probe-check", "text", "--probe-check", "text")
    plan = operator.build_plan(args, project)
    assert plan["planned_live_requests"] == 2
    assert plan["argv"].count("--probe-check") == 1
    args = parse("doctor", "--provider", "claude", "--probe-check", "text")
    with pytest.raises(operator.OperatorError, match="requires --probe"):
        operator.build_plan(args, project)


def test_live_probe_requires_execute_even_with_valid_fake_auth(project, fake_profile):
    profile, log = fake_profile
    result = invoke(project, "doctor", "--inference-config", str(profile), "--probe")
    assert result.returncode == 0
    assert not log.exists()
    result = invoke(project, "doctor", "--inference-config", str(profile), "--probe", "--execute")
    # The generic fake responds, but deliberately does not know the probe nonce.
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["live_requests_attempted"] == 1
    assert report["status"] == "error"
    assert report["live_inference_verified"] is False
    assert any("stdin" in json.loads(line) for line in log.read_text().splitlines())


def test_ideation_plan_does_not_write_or_execute(project):
    result = invoke(project, "ideate", "--provider", "claude", "--workshop-file", "topic.md")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["execution_requested"] is False
    assert plan["idea_output"] == str(project / "topic.json")
    assert plan["argv"][plan["argv"].index("--inference-log-dir") + 1] == str(project / ".inference")
    assert not (project / "topic.json").exists()
    assert not (project / ".inference").exists()


def test_existing_ideas_need_explicit_rewrite_and_cannot_alias_topic(project):
    archive = project / "topic.json"
    archive.write_text("[]")
    args = parse("ideate", "--provider", "codex", "--workshop-file", str(project / "topic.md"))
    with pytest.raises(operator.OperatorError, match="already exists"):
        operator.build_plan(args, project)
    args.allow_existing_ideas = True
    assert operator.build_plan(args, project)["rewrites_existing_ideas"] is True
    assert archive.read_text() == "[]"
    args.workshop_file = str(archive)
    with pytest.raises(operator.OperatorError, match="must not also"):
        operator.build_plan(args, project)


def test_symlink_idea_output_is_not_followed(project):
    (project / "topic.json").symlink_to(project / "missing.json")
    args = parse("ideate", "--provider", "codex", "--workshop-file", str(project / "topic.md"), "--allow-existing-ideas")
    with pytest.raises(operator.OperatorError, match="symlink"):
        operator.build_plan(args, project)


@pytest.mark.parametrize("ideas,index", [({}, 0), ([], 0), ([{}], 0), ([{"Name": "../escape"}], 0),
    ([{"Name": "x/y"}], 0), ([{"Name": "x\\y"}], 0), ([{"Name": "ok"}], 1)])
def test_invalid_idea_selection_never_launches(project, ideas, index):
    path = project / "invalid.json"
    path.write_text(json.dumps(ideas))
    result = invoke(project, "run", "--provider", "codex", "--ideas", str(path), "--idea-index", str(index), "--execute")
    assert result.returncode == 2
    assert result.stdout == ""


def test_run_preview_preserves_v2_stages_and_does_not_start(project):
    original = (project / "bfts_config.yaml").read_bytes()
    result = invoke(project, "run", "--provider", "mixed", "--ideas", "ideas.json", "--idea-index", "1", "--writeup-type", "normal")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["execution_requested"] is False
    assert plan["skipped_stages"] == []
    assert plan["selected_idea_name"] == "second"
    assert plan["writeup_type"] == "normal"
    assert plan["research_settings"]["num_workers"] == 4
    assert plan["research_settings"]["multi_seed_eval"]["num_seeds"] == 3
    assert plan["research_settings"]["execution_timeout_seconds"] == 3600
    assert plan["inference_limits"]["max_in_flight"] == {"codex_cli": 1, "claude_cli": 1}
    assert plan["inference_limits"]["request_timeout_seconds"] == 1800
    assert plan["inference_limits"]["queue_timeout_seconds"] == 3600
    assert len(plan["research_settings"]["stages"]) == 4
    assert (project / "bfts_config.yaml").read_bytes() == original
    assert not (project / "experiments").exists()


@pytest.mark.parametrize("flags,expected", [
    (["--skip-writeup"], ["citations", "writeup", "review"]),
    (["--skip-review"], ["review"]),
    (["--skip-writeup", "--skip-review"], ["citations", "writeup", "review"]),
])
def test_plan_reports_effective_skips(project, flags, expected):
    args = parse("run", "--provider", "claude", "--ideas", str(project / "ideas.json"), *flags)
    plan = operator.build_plan(args, project)
    assert plan["skipped_stages"] == expected


def test_relative_inputs_and_root_are_independent_of_callers_directory(project, tmp_path):
    caller = tmp_path / "other directory"
    caller.mkdir()
    (caller / "ideas.json").write_text('[{"Name": "outside"}]')
    result = invoke(project, "run", "--provider", "codex", "--ideas", "ideas.json", cwd=caller)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["cwd"] == str(project)
    assert plan["argv"][plan["argv"].index("--load_ideas") + 1] == str(caller / "ideas.json")


def test_saved_idea_object_can_prepare_a_new_array_without_changing_old_run(project):
    old_run = project / "old-run"
    old_run.mkdir()
    saved = old_run / "idea.json"
    saved.write_text('{"Name":"saved", "Description":"unchanged"}')
    old_bytes = saved.read_bytes()
    new_input = project / "restart-ideas.json"
    new_input.write_text(json.dumps([json.loads(saved.read_text())]))
    result = invoke(project, "run", "--provider", "claude", "--ideas", str(new_input), "--attempt-id", "1")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["selected_idea_name"] == "saved"
    assert not plan["execution_requested"]
    assert saved.read_bytes() == old_bytes
    assert not (project / "experiments").exists()


def test_exec_preserves_entrypoint_exit_code_cwd_arguments_and_clears_old_broker(project, monkeypatch):
    monkeypatch.setenv("AI_SCIENTIST_INFERENCE_ENDPOINT", "private-old-run")
    monkeypatch.setenv("AI_SCIENTIST_INFERENCE_CONFIG", "private-old-config")
    monkeypatch.setenv("SKILL_TEST_KEEP", "yes")
    result = invoke(project, "run", "--provider", "claude", "--ideas", "ideas.json", "--execute")
    assert result.returncode == 7
    record = json.loads(result.stdout)
    assert record["cwd"] == str(project)
    assert record["args"][record["args"].index("--load_ideas") + 1] == str(project / "ideas.json")
    assert not any(key.startswith("AI_SCIENTIST_INFERENCE_") for key in record["environment_keys"])
    assert "SKILL_TEST_KEEP" in record["environment_keys"]
    assert os.environ["AI_SCIENTIST_INFERENCE_ENDPOINT"] == "private-old-run"


def test_exec_has_no_extra_supervisor_and_interrupt_reaches_runner(project):
    marker = project / "interrupted"
    (project / "launch_scientist_bfts.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "try:\n    print(os.getpid(), flush=True)\n    time.sleep(60)\n"
        f"except KeyboardInterrupt:\n    Path({str(marker)!r}).write_text('cleaned')\n    raise SystemExit(130)\n"
    )
    process = subprocess.Popen([sys.executable, str(HELPER), "--project-root", str(project),
        "run", "--provider", "codex", "--ideas", str(project / "ideas.json"), "--execute"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert int(process.stdout.readline()) == process.pid
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        assert marker.read_text() == "cleaned"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_status_is_local_and_does_not_treat_inference_as_research_success(project):
    directory = project / "experiments" / "2026-01-01_trial"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text('{"status": "completed"}')
    result = invoke(project, "status", "--run-dir", str(directory))
    report = json.loads(result.stdout)
    assert report["runs"][0]["recorded_status"] == "unknown"
    assert report["liveness"] == "not_checked"
    assert report["automatic_resume_supported"] is False
    (directory / "run_status.json").write_text(json.dumps({"status": "running", "stage": "experiments", "secret": "omit"}))
    result = invoke(project, "status")
    report = json.loads(result.stdout)
    assert report["runs"][0]["recorded_status"] == "running"
    assert report["liveness"] == "not_checked"
    assert "omit" not in result.stdout
    assert not (project / ".inference").exists()


@pytest.mark.parametrize("data", [b"broken", b"[]", b'{"status": 3}', b" " * (64 * 1024 + 1)])
def test_malformed_status_is_unknown(project, data):
    directory = project / "saved"
    directory.mkdir()
    (directory / "run_status.json").write_bytes(data)
    report = operator.status_report(parse("status", "--run-dir", str(directory)), project)
    assert report["runs"][0]["recorded_status"] == "unknown"


def test_help_and_empty_status_work_without_site_packages(project):
    for arguments in (["--help"], ["--project-root", str(project), "status"]):
        result = subprocess.run([sys.executable, "-S", str(HELPER), *arguments], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


@given(provider=st.sampled_from(operator.PROVIDERS), suffix=st.text(alphabet="abc日本語123 ;'$()\"-_=", max_size=50))
@example(provider="codex", suffix="; --llm-backend api $(touch SHOULD_NOT_EXIST)")
@settings(max_examples=100, deadline=None)
def test_model_argument_roundtrips_through_real_doctor_parser_without_changing_transport(provider, suffix):
    model = "model-" + suffix
    plan = operator.build_plan(parse("doctor", "--provider", provider, "--model", model, "--dry-run"), ROOT)
    assert shlex.split(plan["command_preview"]) == plan["argv"]
    parsed = doctor_parser().parse_args(plan["argv"][3:])
    effective = config_from_args(parsed, plan["argv"][3:])
    assert parsed.llm_model == model
    assert effective.mode == "cli" and effective.subscription_only
    assert all(effective.resolve(role).backend != "api" for role in plan["roles"])
    assert not plan["execution_requested"]


@given(name=st.text(alphabet="ab日本語123 ;'$()\"-_=", min_size=1, max_size=40))
@example(name="topic $(touch SHOULD_NOT_EXIST);'研究'")
@settings(max_examples=60, deadline=None)
def test_arbitrary_topic_paths_remain_one_literal_argument(name):
    with tempfile.TemporaryDirectory(prefix="ai-scientist-skill-path-") as directory:
        path = Path(directory) / ("topic-" + name + ".md")
        path.write_text("# Topic")
        plan = operator.build_plan(parse("ideate", "--provider", "claude", "--workshop-file", str(path)), ROOT)
        recovered = shlex.split(plan["command_preview"])
        assert recovered == plan["argv"]
        assert recovered[recovered.index("--workshop-file") + 1] == str(path.resolve())
        assert not path.with_suffix(".json").exists()
        assert path.read_text() == "# Topic"
