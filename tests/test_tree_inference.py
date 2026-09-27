"""Deterministic tree/runtime tests without GPU/SDK installations or inference.

Production definitions are compiled unchanged from their source ASTs. Heavy
research imports are replaced at the boundary; tree control flow, filesystem
snapshots, image bytes, real futures and owned-process cleanup are exercised.
"""

import ast
import base64
from collections import Counter
from concurrent.futures import CancelledError, Future, wait
import copy
from dataclasses import dataclass, field
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import socket
import select
import stat
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import uuid
from urllib.parse import quote, unquote, urlparse

import pytest
from PIL import Image
from hypothesis import example, given, settings, strategies as st

from ai_scientist.inference.errors import (
    Cancelled, ConfigurationError, InferenceError, InvalidStructuredOutput, QuotaExhausted,
)
from ai_scientist.inference.process import _process_identity, _read_ownership
from ai_scientist.inference.environment import generated_code_environment


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "ai_scientist" / "treesearch"


def _definitions(filename, names, namespace):
    source = ast.parse((TREE / filename).read_text())
    body = [
        node for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    exec(compile(module, str(TREE / filename), "exec"), namespace)
    return namespace


@pytest.fixture
def harness(monkeypatch):
    module = ModuleType("_tree_inference_test_harness")
    module.__path__ = []
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    ns.update({
        "__package__": module.__name__,
        "Path": Path, "os": os, "json": json, "copy": copy,
        "dataclass": dataclass, "field": field, "uuid": uuid, "time": time,
        "DataClassJsonMixin": object,
        "Counter": Counter, "base64": base64, "threading": threading,
        "multiprocessing": multiprocessing, "signal": signal,
        "queue": queue, "sys": sys, "subprocess": subprocess,
        "socket": socket, "select": select, "stat": stat,
        "_kernel_process_identity": _process_identity,
        "_read_group_ownership": _read_ownership,
        "generated_code_environment": generated_code_environment,
        "_EXECUTION_CONTEXT": multiprocessing.get_context("spawn"),
        "get_runtime": lambda: SimpleNamespace(is_cli=False, cancel_all=lambda **_: None),
        "resolve_api_model": lambda model, **_: model,
        "logger": logging.getLogger("tree-inference-test"),
        "InferenceError": InferenceError, "Cancelled": Cancelled,
        "InvalidStructuredOutput": InvalidStructuredOutput,
        "CancelledError": CancelledError, "wait": wait,
        "cast": lambda _type, value: value,
        "_WORKER_CANCEL_EVENT": None,
        "_WORKER_OWNERSHIP_DIR": None,
        "vlm_feedback_spec": object(), "plot_selection_spec": object(),
        "wrap_code": lambda code, **_: code,
        "MetricValue": lambda **kwargs: SimpleNamespace(**kwargs),
        "WorstMetricValue": lambda: SimpleNamespace(value=None, maximize=None),
        "humanize": SimpleNamespace(naturaldelta=str),
    })
    _definitions("interpreter.py", {
        "ExecutionResult", "InterpreterError", "RedirectQueue", "Interpreter", "exception_summary",
        "_process_identity", "_group_has_live_processes", "_pid_is_live", "stop_owned_sessions", "stop_owned_workers",
        "_read_worker_ownership", "_SocketSender", "_ChannelQueue",
    }, ns)
    _definitions("journal.py", {"_artifact_path", "Node", "Journal"}, ns)
    _definitions("parallel_agent.py", {
        "MinimalAgent", "ParallelAgent", "GPUManager", "_initialize_tree_worker",
        "_check_worker_cancelled", "_save_partial_node", "_selected_plot_paths", "_plot_data_url",
        "_retain_interpreter_result",
    }, ns)
    return ns


def _plots(tmp_path, count):
    result = []
    for index in range(count):
        path = tmp_path / f"plot_{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(path)
        result.append(str(path))
    return result


def _minimal(harness):
    agent = object.__new__(harness["MinimalAgent"])
    agent.task_desc = "Fixture experiment"
    agent.cfg = SimpleNamespace(agent=SimpleNamespace(
        feedback=SimpleNamespace(model="legacy-api-id", temp=0.5),
        vlm_feedback=SimpleNamespace(model="legacy-api-id", temp=0.5),
    ))
    agent._determine_datasets_successfully_tested = lambda node: ["fixture"]
    return agent


def test_selected_plots_reject_outside_candidate_and_duplicate_overuse(harness, tmp_path):
    paths = _plots(tmp_path, 3)
    select = harness["_selected_plot_paths"]
    assert select(paths, [paths[2], paths[0]]) == [paths[2], paths[0]]
    with pytest.raises(InvalidStructuredOutput):
        select(paths[:2], [paths[2]])  # An existing but unoffered image is invalid.
    with pytest.raises(InvalidStructuredOutput):
        select(paths, [paths[0], paths[0]])
    assert select([paths[0], paths[0]], [paths[0], paths[0]]) == [paths[0], paths[0]]


def test_vlm_uses_selected_order_and_real_image_mime(harness, tmp_path):
    paths = _plots(tmp_path, 11)
    selected = [paths[10], paths[2]]
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        if kwargs["role"] == "execution_feedback":
            return {"selected_plots": selected}
        return {"valid_plots_received": True, "plot_analyses": [
            {"analysis": "ten"}, {"analysis": "two"},
        ], "vlm_feedback_summary": "valid"}

    harness["query"] = query
    node = harness["Node"](plot_paths=paths, is_buggy=False)
    _minimal(harness)._analyze_plots_with_vlm(node)
    assert [analysis["plot_path"] for analysis in node.plot_analyses] == selected
    images = calls[1]["user_message"][1:]
    assert all(image["image_url"]["url"].startswith("data:image/png;base64,") for image in images)
    assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == Path(selected[0]).read_bytes()
    assert node.is_buggy_plots is False


def test_negative_plot_evaluation_with_empty_analysis_is_valid(harness, tmp_path):
    harness["query"] = lambda **_: {
        "valid_plots_received": False, "plot_analyses": [], "vlm_feedback_summary": "blank image",
    }
    node = harness["Node"](plot_paths=_plots(tmp_path, 1))
    _minimal(harness)._analyze_plots_with_vlm(node)
    assert node.is_buggy_plots is True
    assert node.plot_analyses == []


def test_incomplete_positive_plot_analysis_does_not_set_success(harness, tmp_path):
    harness["query"] = lambda **_: {
        "valid_plots_received": True, "plot_analyses": [], "vlm_feedback_summary": "incorrect",
    }
    node = harness["Node"](plot_paths=_plots(tmp_path, 1))
    with pytest.raises(InvalidStructuredOutput):
        _minimal(harness)._analyze_plots_with_vlm(node)
    assert node.is_buggy_plots is None


def test_node_round_trip_preserves_external_paths_and_plot_execution(harness, tmp_path):
    Node = harness["Node"]
    node = Node(code="print('done')", plot_paths=_plots(tmp_path, 1), exp_results_dir=str(tmp_path))
    node.absorb_plot_exec_result(harness["ExecutionResult"](["plot output"], 1.2, "ValueError"))
    data = node.to_dict()
    before = copy.deepcopy(data)
    restored = Node.from_dict(data)
    assert data == before
    assert restored.plot_paths == node.plot_paths
    assert restored.plot_exc_type == "ValueError"
    assert restored.plot_term_out == ["plot output"]
    assert restored.metric.value is None


def _parallel(harness):
    agent = object.__new__(harness["ParallelAgent"])
    agent.journal = harness["Journal"]()
    agent.gpu_manager = harness["GPUManager"](2)
    agent._cancel_event = threading.Event()
    agent._cancel_lock = threading.Lock()
    agent._batch_failure = None
    agent._cancel_started = None
    agent._owned_process_dir = Path("unused-test-ownership-directory")
    agent.executor = SimpleNamespace(shutdown=lambda **_: None, failure=None)
    agent._update_hyperparam_tuning_state = lambda node: None
    agent._update_ablation_state = lambda node: None
    harness["get_runtime"] = lambda: SimpleNamespace(cancel_all=lambda **_: None)
    harness["stop_owned_sessions"] = lambda path: None
    harness["stop_owned_workers"] = lambda path: None
    return agent


def test_future_completion_order_does_not_change_journal_order(harness):
    agent = _parallel(harness)
    first, second = Future(), Future()
    second.set_result(harness["Node"](id="second").to_dict())
    timer = threading.Timer(0.03, lambda: first.set_result(harness["Node"](id="first").to_dict()))
    timer.start()
    try:
        agent._collect_results([first, second], ["worker_0", "worker_1"])
    finally:
        timer.join()
    assert [(node.id, node.step) for node in agent.journal.nodes] == [("first", 0), ("second", 1)]


def test_quota_keeps_later_completed_future_and_releases_gpu_only_after_done(harness):
    agent = _parallel(harness)
    first, second = Future(), Future()
    first.set_exception(QuotaExhausted("limit reached"))
    second.set_result(harness["Node"](id="completed-later").to_dict())
    releases = []
    agent.gpu_manager = SimpleNamespace(release_gpu=lambda process_id: releases.append((process_id, first.done(), second.done())))
    with pytest.raises(QuotaExhausted):
        agent._collect_results([first, second], ["worker_0", "worker_1"])
    assert [node.id for node in agent.journal.nodes] == ["completed-later"]
    assert agent._cancel_event.is_set()
    assert releases == [("worker_0", True, True), ("worker_1", True, True)]


def test_stuck_legacy_api_worker_is_stopped_after_cancellation_grace(harness):
    agent = _parallel(harness)
    pending = Future()
    stopped = []
    agent._request_batch_cancel(Cancelled("user interrupted"))
    agent._cancel_started = time.monotonic() - 10

    def stop_workers(path):
        stopped.append("worker")
        if not pending.done():
            pending.set_exception(Cancelled("owned worker stopped"))

    harness["stop_owned_workers"] = stop_workers
    agent.executor.shutdown = lambda **_: stop_workers(None)
    harness["stop_owned_sessions"] = lambda path: stopped.append("experiment")
    agent.gpu_manager = SimpleNamespace(release_gpu=lambda pid: stopped.append("gpu"))
    with pytest.raises(Cancelled):
        agent._collect_results([pending], ["worker_0"])
    assert stopped.index("worker") < stopped.index("gpu")
    assert stopped.index("experiment") < stopped.index("gpu")


@pytest.mark.parametrize("failure_phase", ["execution_feedback", "plot_execution", "plot_evaluation"])
def test_worker_saves_experiment_and_unfinished_evaluation(harness, monkeypatch, tmp_path, failure_phase):
    Node, Result = harness["Node"], harness["ExecutionResult"]
    cleaned = []

    class FakeInterpreter:
        def __init__(self, working_dir, **kwargs):
            self.working_dir = Path(working_dir)

        def run(self, code, reset_session):
            if code == "plot":
                Image.new("RGB", (2, 2)).save(self.working_dir / "working" / "result.png")
                if failure_phase == "plot_execution":
                    self.last_result = Result(["plot process stopped"], 1, "ProcessExited")
                    raise harness["InterpreterError"]("Plot process exited")
            return Result(["experiment completed"], 1.0, None)

        def cleanup_session(self):
            cleaned.append(True)

    class FakeAgent:
        stage_name = "1_initial"

        def __init__(self, **kwargs):
            pass

        def _draft(self):
            return Node(id="partial-node", code="experiment")

        def parse_exec_result(self, node, **kwargs):
            if failure_phase == "execution_feedback":
                raise QuotaExhausted("subscription limit")
            node.is_buggy = False

        def _generate_plotting_code(self, node, *args):
            node.plot_code = "plot"
            return "plot"

        def _analyze_plots_with_vlm(self, node):
            raise QuotaExhausted("subscription limit")

    for suffix, members in {
        "interpreter": {"Interpreter": FakeInterpreter},
        "journal": {"Node": Node, "Journal": harness["Journal"]},
    }.items():
        module = ModuleType(f"{harness['__package__']}.{suffix}")
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    harness["MinimalAgent"] = FakeAgent
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = SimpleNamespace(
        workspace_dir=tmp_path / "workspace", log_dir=tmp_path / "logs",
        exec=SimpleNamespace(timeout=1, format_tb_ipython=False, agent_file_name="run.py"),
    )
    with pytest.raises(InferenceError) as stopped:
        harness["ParallelAgent"]._process_node_wrapper(None, "idea", cfg)
    snapshot = json.loads(Path(stopped.value.metadata["partial_node_path"]).read_text())
    assert snapshot["phase"] == failure_phase
    assert snapshot["status"] == "incomplete"
    assert snapshot["node"]["code"] == "experiment"
    assert snapshot["node"]["_term_out"] == ["experiment completed"]
    assert snapshot["node"]["is_buggy_plots"] is None
    assert cleaned
    if failure_phase == "plot_evaluation":
        assert snapshot["node"]["is_buggy"] is False
        assert Path(snapshot["node"]["plot_paths"][0]).exists()
    if failure_phase == "plot_execution":
        assert snapshot["node"]["plot_term_out"] == ["plot process stopped"]
        assert snapshot["node"]["plot_exc_type"] == "ProcessExited"
        assert snapshot["node"]["is_buggy"] is False


def test_multi_seed_submission_uses_independent_node_dicts(harness):
    agent = _parallel(harness)
    agent.cfg = SimpleNamespace(agent=SimpleNamespace(multi_seed_eval=SimpleNamespace(num_seeds=3)))
    agent.task_desc, agent.evaluation_metrics, agent.stage_name = "idea", "metric", "1_initial"
    agent.gpu_manager = None
    submitted = []

    def submit(fn, *args):
        submitted.append(args)
        future = Future()
        future.set_result(harness["Node"](id=f"seed-{len(submitted)}").to_dict())
        return future

    agent.executor = SimpleNamespace(submit=submit)
    agent._run_multi_seed_evaluation(harness["Node"](code="original"))
    assert len({id(args[0]) for args in submitted}) == 3
    for index, args in enumerate(submitted):
        assert f"seed = {index}\n" in args[0]["code"]


def test_save_run_default_never_selects_best_node_or_exports_tree(harness, tmp_path):
    calls = []
    harness["serialize"] = SimpleNamespace(dump_json=lambda journal, path: path.write_text(json.dumps(journal.to_dict())))
    harness["OmegaConf"] = SimpleNamespace(save=lambda config, f: f.write_text("fixture: true\n"))
    harness["tree_export"] = SimpleNamespace(generate=lambda *args: calls.append("unexpected export"))
    _definitions("utils/config.py", {"save_run"}, harness)
    journal = harness["Journal"](nodes=[harness["Node"](id="retained")])
    journal.get_best_node = lambda **kwargs: (_ for _ in ()).throw(AssertionError("model called"))
    target = harness["save_run"](SimpleNamespace(log_dir=tmp_path), journal, stage_name="stage_1")
    assert json.loads((target / "journal.json").read_text())["nodes"][0]["id"] == "retained"
    assert (target / "config.yaml").exists()
    assert calls == []


def test_inference_quota_not_swallowed_by_best_node_fallback(harness):
    harness["node_selection_spec"] = object()
    harness["query"] = lambda **kwargs: (_ for _ in ()).throw(QuotaExhausted("quota"))
    journal = harness["Journal"](nodes=[harness["Node"](id="a"), harness["Node"](id="b")])
    with pytest.raises(QuotaExhausted):
        journal.get_best_node(only_good=False)


def test_best_node_selection_rejects_unknown_ids(harness):
    harness["node_selection_spec"] = object()
    harness["query"] = lambda **kwargs: {"selected_id": "not-a-candidate", "reasoning": "invalid"}
    journal = harness["Journal"](nodes=[harness["Node"](id="a"), harness["Node"](id="b")])
    with pytest.raises(InvalidStructuredOutput):
        journal.get_best_node(only_good=False)


@pytest.mark.skipif(os.name != "posix", reason="Initial supported runtime is POSIX")
def test_interpreter_timeout_is_finite_and_preserves_unrelated_process(harness, monkeypatch, tmp_path):
    # This executes only a tiny deterministic Python loop, never model output.
    # fork keeps lightweight test-only dependency replacements in the child.
    context = multiprocessing.get_context("fork")
    harness.update({"Process": context.Process, "Queue": context.Queue})
    shutup = ModuleType("shutup")
    shutup.mute_warnings = lambda: None
    monkeypatch.setitem(sys.modules, "shutup", shutup)
    ownership = tmp_path / "owned"
    interpreter = harness["Interpreter"](tmp_path, timeout=0.1, interrupt_grace_seconds=0.1, ownership_dir=ownership)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    started = time.monotonic()
    try:
        result = interpreter.run("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('started', flush=True)\nwhile True: time.sleep(0.01)")
        assert result.exc_type == "TimeoutError"
        assert "started" in "".join(result.term_out)
        assert time.monotonic() - started < 8
        assert unrelated.poll() is None
        harness["stop_owned_sessions"](ownership)
        assert list(ownership.glob("*.json")) == []
    finally:
        interpreter.cleanup_session()
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_continuous_output_cannot_starve_timeout(tmp_path):
    from ai_scientist.treesearch.interpreter import Interpreter

    interpreter = Interpreter(tmp_path, timeout=0.1, interrupt_grace_seconds=0.1)
    started = time.monotonic()
    try:
        result = interpreter.run("import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True: print('x'*8192)")
        assert time.monotonic() - started < 8
        assert result.exc_type == "TimeoutError"
        assert "x" in "".join(result.term_out)
    finally:
        interpreter.cleanup_session()


@pytest.mark.skipif(os.name != "posix", reason="Initial supported runtime is POSIX")
def test_orphan_registry_cleanup_targets_only_owned_group(harness, tmp_path):
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    record = tmp_path / "owned.json"
    tmp_path.chmod(0o700)
    record.write_text(json.dumps({"version": 1, "pid": owned.pid, "pgid": owned.pid, "uid": os.getuid(), "nonce": uuid.uuid4().hex, "identity": harness["_process_identity"](owned.pid)}))
    record.chmod(0o600)
    try:
        harness["stop_owned_sessions"](tmp_path)
        owned.wait(timeout=3)
        assert unrelated.poll() is None
        assert not record.exists()
    finally:
        if owned.poll() is None:
            owned.kill()
        owned.wait(timeout=3)
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_registry_does_not_signal_recycled_pid(harness, monkeypatch, tmp_path):
    record = tmp_path / "owned.json"
    tmp_path.chmod(0o700)
    record.write_text(json.dumps({"version": 1, "pid": 123456, "pgid": 123456, "uid": os.getuid(), "nonce": uuid.uuid4().hex, "identity": {"pid": 123456, "pgid": 123456, "uid": os.getuid(), "birth": "old"}}))
    record.chmod(0o600)
    harness["_process_identity"] = lambda pid: {"pid": pid, "birth": "new"}
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="identity"):
        harness["stop_owned_sessions"](tmp_path)
    assert signals == []
    assert record.exists()


def test_all_tree_query_calls_have_explicit_roles():
    for filename in ["parallel_agent.py", "journal.py", "agent_manager.py", "journal2report.py"]:
        tree = ast.parse((TREE / filename).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "query"]
        assert calls
        assert all(any(keyword.arg == "role" for keyword in node.keywords) for node in calls), filename


def test_summary_groups_substages_without_losing_nodes(harness, monkeypatch):
    _definitions("log_summarization.py", {"overall_summarize"}, harness)
    harness["annotate_history"] = lambda *args, **kwargs: None
    harness["get_ai_client"] = lambda *args, **kwargs: None
    harness["get_stage_summary"] = lambda journal, *args: [node.id for node in journal.nodes]
    harness["get_node_log"] = lambda node: {"id": node.id}
    journals = [
        ("1_initial_1_first", harness["Journal"](nodes=[harness["Node"](id="first")])),
        ("1_initial_2_second", harness["Journal"](nodes=[harness["Node"](id="second")])),
    ]
    results = harness["overall_summarize"](iter(journals))
    assert len(results) == 4
    assert results[0] == ["first", "second"]
    assert results[1]["status"] == "not_run"


def test_stage_and_unified_html_plot_urls_resolve_to_real_artifacts(harness, tmp_path):
    harness.update({"quote": quote, "unquote": unquote, "urlparse": urlparse})
    _definitions("utils/tree_export.py", {"_plot_urls", "_rebase_plot_urls"}, harness)
    log_dir = tmp_path / "custom logs" / "0-run"
    artifact = log_dir / "experiment_results" / "node1" / "result 1.png"
    artifact.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(artifact)
    html = log_dir / "stage_2_baseline" / "tree_plot.html"
    node = harness["Node"](plot_paths=[str(artifact)], plots=["../../logs/broken.png"])
    stage_urls = harness["_plot_urls"](node, html)
    assert (html.parent / unquote(stage_urls[0])).resolve() == artifact.resolve()
    assert "%20" in stage_urls[0]
    unified = harness["_rebase_plot_urls"]({"plots": [stage_urls]}, html.parent, log_dir)
    assert (log_dir / unquote(unified["plots"][0][0])).resolve() == artifact.resolve()
    assert node.plots == ["../../logs/broken.png"]


def test_generated_code_environment_preserves_api_mode_and_scrubs_cli_authority():
    source = {
        "OPENAI_API_KEY": "dummy", "ANTHROPIC_API_KEY": "dummy",
        "AI_SCIENTIST_INFERENCE_ENDPOINT": "private broker",
        "AI_SCIENTIST_INFERENCE_CONFIG": "private config",
        "AWS_SECRET_ACCESS_KEY": "dummy", "GOOGLE_APPLICATION_CREDENTIALS": "/private/file",
        "HUGGINGFACE_API_KEY": "dummy", "OLLAMA_API_KEY": "dummy", "HF_TOKEN": "dummy",
        "HF_HOME": "/dataset/cache", "CUDA_VISIBLE_DEVICES": "2", "PATH": "/bin",
        "AI_SCIENTIST_ROOT": "/project",
    }
    assert generated_code_environment(source, subscription_only=False) == source
    assert generated_code_environment(source) == {key: source[key] for key in (
        "HF_HOME", "CUDA_VISIBLE_DEVICES", "PATH", "AI_SCIENTIST_ROOT",
    )}
    assert source["OPENAI_API_KEY"] == "dummy"


def test_real_spawn_interpreter_scrubs_inherited_and_explicit_credentials(monkeypatch, tmp_path):
    from ai_scientist.treesearch.interpreter import Interpreter

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-parent")
    monkeypatch.setenv("AI_SCIENTIST_INFERENCE_ENDPOINT", "dummy-private-endpoint")
    interpreter = Interpreter(tmp_path, timeout=2, subscription_only=True,
                              env_vars={"ANTHROPIC_API_KEY": "dummy-explicit", "CUDA_VISIBLE_DEVICES": "3"})
    try:
        result = interpreter.run("import os\nassert 'OPENAI_API_KEY' not in os.environ\nassert 'ANTHROPIC_API_KEY' not in os.environ\nassert 'AI_SCIENTIST_INFERENCE_ENDPOINT' not in os.environ\nprint(os.environ['CUDA_VISIBLE_DEVICES'])")
        assert result.exc_type is None
        assert "3\n" in "".join(result.term_out)
        assert os.environ["OPENAI_API_KEY"] == "dummy-parent"
        # Persistent interpreter sessions retain execution variables as before.
        assert interpreter.run("x = 7", reset_session=False).exc_type is None
        assert "7\n" in "".join(interpreter.run("print(x)", reset_session=False).term_out)
    finally:
        interpreter.cleanup_session()


def test_unexpected_experiment_exit_is_infrastructure_and_keeps_stdout(tmp_path):
    from ai_scientist.treesearch.interpreter import Interpreter

    interpreter = Interpreter(tmp_path, timeout=2)
    try:
        with pytest.raises(RuntimeError, match="ProcessExited"):
            interpreter.run("import os\nprint('before-exit', flush=True)\nos._exit(3)")
        assert interpreter.last_result.exc_type == "ProcessExited"
        assert "before-exit" in "".join(interpreter.last_result.term_out)
    finally:
        interpreter.cleanup_session()


def _send_socket_frames(sender, character, count):
    for _ in range(count):
        sender.send("output", character * 8192)


@pytest.mark.skipif(os.name != "posix", reason="Initial supported runtime is POSIX")
def test_real_socket_sender_keeps_forked_experiment_stdout_frames_separate():
    from ai_scientist.treesearch.interpreter import _SocketSender

    receiver, writer = socket.socketpair()
    writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    sender = _SocketSender(writer)
    children = [multiprocessing.get_context("fork").Process(target=_send_socket_frames, args=(sender, char, 25)) for char in "AB"]
    try:
        for child in children:
            child.start()
        writer.close()
        receiver.settimeout(5)
        data = b""
        while True:
            chunk = receiver.recv(8192)
            if not chunk:
                break
            data += chunk
        frames = [json.loads(frame) for frame in data.splitlines()]
        assert len(frames) == 50
        assert Counter(value[0] for kind, value in frames) == {"A": 25, "B": 25}
        assert all(kind == "output" and len(set(value)) == 1 and len(value) == 8192 for kind, value in frames)
    finally:
        receiver.close()
        writer.close()
        for child in children:
            child.join(timeout=2)
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
        assert all(child.exitcode == 0 for child in children)


@pytest.mark.parametrize("kind", ["output", "event"])
def test_truncated_large_socket_frame_cannot_block_timeout(tmp_path, kind):
    from ai_scientist.treesearch.interpreter import Interpreter, _SocketSender

    receiver, writer = socket.socketpair()
    writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    sender = _SocketSender(writer)
    child = multiprocessing.get_context("fork").Process(target=sender.send, args=(kind, "x" * (8 * 1024 * 1024)))
    child.start()
    writer.close()
    interpreter = Interpreter(tmp_path, timeout=0.01)
    interpreter._socket = receiver
    receiver.setblocking(False)
    interpreter._stream_closed = False
    interpreter._stream_fragments = []
    interpreter._pending_input = b""
    interpreter._input_offset = 0
    interpreter._events, interpreter._outputs = [], []
    try:
        assert select.select([receiver], [], [], 5)[0]
        child.kill()
        child.join(timeout=2)
        started = time.monotonic()
        while not interpreter._stream_closed and time.monotonic() - started < 1:
            interpreter._poll_stream(timeout=0.01)
        assert interpreter._stream_closed
        assert interpreter._stream_fragments
        assert time.monotonic() - started < 1
    finally:
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(timeout=2)


def test_js_location_defaults_respect_exported_stage_metadata():
    import shutil
    if shutil.which("node") is None:
        pytest.skip("Node.js not installed")
    script = (TREE / "utils/viz_templates/template.js").read_text()
    function = script.split("function applyTreeLocationDefaults", 1)[1].split("// Initialize with", 1)[0]
    code = "function applyTreeLocationDefaults" + function + """
const href = 'https://example.test/logs/run/stage_2_baseline_1_first/tree_plot.html';
const configured = applyTreeLocationDefaults({log_dir_path:'..', current_stage:'Stage_2'}, href);
const fallback = applyTreeLocationDefaults({}, href);
console.log(JSON.stringify({configured, fallback}));
"""
    result = subprocess.run(["node", "-e", code], check=True, capture_output=True, text=True, timeout=5)
    data = json.loads(result.stdout)
    assert data["configured"] == {"log_dir_path": "..", "current_stage": "Stage_2"}
    assert data["fallback"] == {"log_dir_path": "https://example.test/logs/run/", "current_stage": "Stage_2"}


codec_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=100),
    lambda values: st.lists(values, max_size=5) | st.dictionaries(st.text(max_size=30), values, max_size=5) | st.tuples(values, values),
    max_leaves=30,
)


@settings(max_examples=150, deadline=None)
@example(value={"type": "dict", "value": [(1, None)], "unknown_future_field": {"nested": []}})
@given(value=codec_values)
def test_owned_pool_codec_round_trip_preserves_all_fields_and_types(value):
    from ai_scientist.treesearch.utils.owned_pool import decode_value, encode_value

    decoded = decode_value(json.loads(json.dumps(encode_value(value))))
    assert decoded == value
    assert type(decoded) is type(value)


def test_owned_pool_codec_preserves_all_real_wrapper_arguments_and_rejects_unknown_fields(tmp_path):
    from omegaconf import OmegaConf
    from ai_scientist.treesearch.parallel_agent import AblationIdea, HyperparamTuningIdea
    from ai_scientist.treesearch.journal import Node
    from ai_scientist.treesearch.utils.owned_pool import decode_value, encode_value

    cfg = OmegaConf.load(ROOT / "bfts_config.yaml")
    cfg.workspace_dir = tmp_path
    cfg.inference = {"mode": "cli", "future_option": {"all_fields": True}}
    node = Node(code="print('science')", plot_exc_info={"args": ["unfinished"]}, plot_exc_stack=[("plot.py", 5, "plot", "raise")])
    args = (node.to_dict(), {"task": "fixture"}, cfg, 2, "memory", "metric", "4_ablation", AblationIdea("a", "all ablation fields"), HyperparamTuningIdea("h", "all tuning fields"), "plot3", "plot2", "plot1", True)
    restored = decode_value(encode_value(args))
    assert restored[:2] == args[:2]
    assert OmegaConf.to_container(restored[2]) == OmegaConf.to_container(cfg)
    assert isinstance(restored[2].workspace_dir, Path)
    assert restored[3:7] == args[3:7]
    assert vars(restored[7]) == vars(args[7])
    assert vars(restored[8]) == vars(args[8])
    assert restored[9:] == args[9:]
    args[7].unrecognized = "must not disappear"
    with pytest.raises(ConfigurationError, match="Unknown fields"):
        encode_value(args)
    with pytest.raises(ConfigurationError, match="Unsupported worker payload"):
        encode_value(object())


def _pool_fixture(action, value=None, *, delay=0):
    time.sleep(delay)
    if action == "quota":
        raise QuotaExhausted("fixture quota", metadata={"phase": "vision", "retained": [1, 2]})
    if action == "exit":
        os._exit(17)
    if action == "hang":
        time.sleep(30)
    if action == "slow_result":
        # Pause the actual production atomic JSON writer after it has written
        # a large value chunk, before the result document/rename is complete.
        original = json.JSONEncoder.iterencode

        def interrupted_writer(self, obj, *args, **kwargs):
            for chunk in original(self, obj, *args, **kwargs):
                yield chunk
                if len(chunk) > 1024 * 1024:
                    time.sleep(30)

        json.JSONEncoder.iterencode = interrupted_writer
    return {"value": value, "pid": os.getpid(), "name": multiprocessing.current_process().name}


def _real_pool(tmp_path, count=2):
    from ai_scientist.treesearch.utils.owned_pool import OwnedProcessPool
    return OwnedProcessPool(max_workers=count, mp_context=multiprocessing.get_context("spawn"), directory=tmp_path / "jobs", function=_pool_fixture)


def test_owned_pool_real_workers_persist_and_preserve_large_results(tmp_path):
    pool = _real_pool(tmp_path)
    try:
        first = pool.submit(_pool_fixture, "echo", "first", delay=0.1)
        second = pool.submit(_pool_fixture, "echo", "second")
        assert [future.result(timeout=10)["value"] for future in (first, second)] == ["first", "second"]
        pid_set = set(pool.worker_pids)
        large = pool.submit(_pool_fixture, "echo", "x" * (8 * 1024 * 1024))
        result = large.result(timeout=10)
        assert result["value"] == "x" * (8 * 1024 * 1024)
        assert result["pid"] in pid_set
        assert len(list((tmp_path / "jobs").glob("*.result.json"))) == 3
        assert all(path.stat().st_mode & 0o077 == 0 for path in (tmp_path / "jobs").iterdir())
    finally:
        pool.shutdown()
    assert all(_process_identity(pid) is None for pid in pool.worker_pids)


def test_owned_pool_preserves_typed_inference_errors_and_completed_later_results(tmp_path):
    pool = _real_pool(tmp_path)
    try:
        error = pool.submit(_pool_fixture, "quota", delay=0.2)
        success = pool.submit(_pool_fixture, "echo", "completed-later")
        with pytest.raises(QuotaExhausted) as raised:
            error.result(timeout=10)
        assert raised.value.metadata == {"phase": "vision", "retained": [1, 2]}
        assert success.result(timeout=10)["value"] == "completed-later"
    finally:
        pool.shutdown()


def test_owned_pool_crash_does_not_lose_a_committed_result_or_hang(tmp_path):
    pool = _real_pool(tmp_path)
    started = time.monotonic()
    try:
        completed = pool.submit(_pool_fixture, "echo", "durable")
        assert completed.result(timeout=10)["value"] == "durable"
        crashed = pool.submit(_pool_fixture, "exit")
        with pytest.raises(InferenceError):
            crashed.result(timeout=10)
        assert completed.result()["value"] == "durable"
    finally:
        pool.shutdown()
    assert time.monotonic() - started < 10
    assert all(_process_identity(pid) is None for pid in pool.worker_pids)


def test_owned_pool_idle_worker_failure_is_visible_and_stuck_worker_shutdown_is_finite(tmp_path):
    pool = _real_pool(tmp_path)
    try:
        stuck = pool.submit(_pool_fixture, "hang")
        deadline = time.monotonic() + 10
        while not list(pool.directory.glob("*.running.*.json")) and time.monotonic() < deadline:
            time.sleep(0.01)
        claim = next(pool.directory.glob("*.running.*.json"))
        busy_pid = int(claim.name.split(".")[-2])
        idle_pid = next(pid for pid in pool.worker_pids if pid != busy_pid)
        os.kill(idle_pid, signal.SIGKILL)
        while pool.failure is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.failure is not None
        assert not stuck.done()  # No false claim that the live work has stopped.
        started = time.monotonic()
        pool.shutdown()
        assert time.monotonic() - started < 7
        with pytest.raises(InferenceError):
            stuck.result(timeout=0.1)
    finally:
        pool.shutdown()


def test_owned_pool_kill_mid_result_write_retains_completed_and_partial_files(tmp_path):
    pool = _real_pool(tmp_path)
    try:
        completed = pool.submit(_pool_fixture, "echo", "already-committed")
        assert completed.result(timeout=10)["value"] == "already-committed"
        interrupted = pool.submit(_pool_fixture, "slow_result", "x" * (8 * 1024 * 1024))
        deadline = time.monotonic() + 10
        partial = []
        while not partial and time.monotonic() < deadline:
            partial = [path for path in pool.directory.glob("*.result.json.*.tmp") if path.stat().st_size > 1024 * 1024]
            time.sleep(0.005)
        assert partial, "fixture never reached the production writer's incomplete result"
        job_id = partial[0].name.split(".result.json", 1)[0]
        claim = next(pool.directory.glob(f"{job_id}.running.*.json"))
        writer_pid = int(claim.name.split(".")[-2])
        os.kill(writer_pid, signal.SIGKILL)
        with pytest.raises(InferenceError):
            interrupted.result(timeout=5)
        assert completed.result()["value"] == "already-committed"
        assert partial[0].exists()
        assert not (pool.directory / f"{job_id}.result.json").exists()
        started = time.monotonic()
        pool.shutdown()
        assert time.monotonic() - started < 7
    finally:
        pool.shutdown()


def test_owned_pool_worker_names_do_not_reuse_another_stage_workspace(tmp_path):
    names = []
    for stage in ("stage_1", "stage_2"):
        pool = _real_pool(tmp_path / stage, count=1)
        try:
            names.append(pool.submit(_pool_fixture, "echo").result(timeout=10)["name"])
            assert pool.submit(_pool_fixture, "echo").result(timeout=10)["name"] == names[-1]
        finally:
            pool.shutdown()
    assert names[0] != names[1]
    assert len({tmp_path / f"process_{name}" for name in names}) == 2


def test_seed_aggregation_round_trip_retains_metric_sentinel_for_stage_control():
    from ai_scientist.treesearch.journal import Node
    from ai_scientist.treesearch.utils.metric import MetricValue

    node = Node(code="# aggregate", is_seed_agg_node=True, is_seed_node=True, is_buggy=False)
    restored = Node.from_dict(node.to_dict())
    assert isinstance(restored.metric, MetricValue)
    assert restored.metric.value is None
    # Stage-control success checks access .value on every non-buggy node.
    assert [n for n in [restored] if not n.is_buggy and n.metric.value is not None] == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_owned_pool_codec_preserves_nonfinite_scientific_metrics(value):
    from ai_scientist.treesearch.journal import Node
    from ai_scientist.treesearch.utils.metric import MetricValue
    from ai_scientist.treesearch.utils.owned_pool import decode_value, encode_value

    node = Node(metric=MetricValue(value=value), is_buggy=False)
    encoded = json.loads(json.dumps(encode_value(node.to_dict()), allow_nan=False))
    restored = Node.from_dict(decode_value(encoded))
    if math.isnan(value):
        assert math.isnan(restored.metric.value)
    else:
        assert restored.metric.value == value
