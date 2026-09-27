"""Offline broker integration tests, including real spawn-process IPC.

The importable executor is a constructor-only seam. These tests never start an
installed CLI, inspect account credentials, or make a network request.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ai_scientist.inference.broker import Broker, BrokerClient, _receive_frame, _send_frame, _source_provenance
from ai_scientist.inference.contracts import ROLES, InferenceRequest, InferenceResult
from ai_scientist.inference.errors import (
    AuthenticationRequired,
    BrokerUnavailable,
    Cancelled,
    ContextLimitExceeded,
    InferenceTimeout,
    QueueTimeout,
    QuotaExhausted,
)


def fake_executor(request, provider, request_dir, cancel_event):
    """Deterministic transport, executed inside the actual broker process."""
    started = time.monotonic()
    (request_dir / "started.marker").write_text(str(os.getpid()), encoding="utf-8")
    mode = request.metadata.get("mode", "complete")
    duration = request.metadata.get("sleep", 0.05)
    if mode == "subprocess":
        from ai_scientist.inference.process import run_process

        run_process(
            [sys.executable, "-c", "from pathlib import Path; import time; Path('executed.marker').write_text('yes'); time.sleep(30)"],
            cwd=request_dir, env={"PATH": os.defpath},
            timeout_seconds=request.timeout_seconds, cancel_event=cancel_event,
            ownership_path=request_dir / "process.json",
        )
    if mode == "quota":
        time.sleep(duration)
        raise QuotaExhausted("Subscription quota reached", request_id=request.request_id)
    if mode == "auth":
        raise AuthenticationRequired("Subscription login required", request_id=request.request_id)
    while mode == "wait" or time.monotonic() - started < duration:
        if cancel_event.wait(0.005):
            (request_dir / "cancelled.marker").write_text("cancelled", encoding="utf-8")
            raise Cancelled("Fake transport cancelled", request_id=request.request_id)
        if time.monotonic() - started > request.timeout_seconds:
            raise InferenceTimeout("Fake transport deadline", request_id=request.request_id)
    finished = time.monotonic()
    if mode == "artifact_error":
        (request_dir / "result.json").mkdir()
    return InferenceResult(
        request_id=request.request_id,
        backend=request.backend,
        text=json.dumps({
            "pid": os.getpid(), "started": started, "finished": finished,
            "backend": request.backend, "request_id": request.request_id,
        }),
        usage={"input_tokens": 3, "output_tokens": 2, "cached_input_tokens": 1},
        duration_seconds=finished - started,
        metadata={"cli_version": request.metadata.get("cli_version")},
    )


def _submit_in_process(endpoint, requests, results):
    try:
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            values = list(executor.map(
                lambda value: BrokerClient(endpoint).submit(InferenceRequest.from_dict(value)).to_dict(),
                requests,
            ))
        results.put({"ok": True, "results": values})
    except BaseException as exception:
        results.put({"ok": False, "exception": type(exception).__name__})


def _wait_submit(endpoint, request):
    BrokerClient(endpoint).submit(InferenceRequest.from_dict(request))


def _wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.015)
    raise AssertionError("Condition did not become true before the test deadline")


def _pid_running(pid):
    value = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True, text=True, timeout=2,
    )
    return bool(value.stdout.strip()) and not value.stdout.strip().startswith("Z")


@unittest.skipUnless(os.name == "posix", "The broker targets Linux/macOS")
class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inference-broker-test-")
        self.root = Path(self.temporary.name)
        self.brokers = []

    def tearDown(self):
        for broker in reversed(self.brokers):
            broker.close()
        self.temporary.cleanup()

    def broker(self, **overrides):
        config = {
            "providers": {
                "codex_cli": {"max_in_flight": 1},
                "claude_cli": {"max_in_flight": 1},
            },
            "queue_timeout_seconds": 5,
            "request_timeout_seconds": 5,
            "shutdown_timeout_seconds": 1,
            "max_queue_size": 64,
        }
        config.update(overrides)
        value = Broker(
            config, "test-run", self.root / f"run-{len(self.brokers)}",
            _executor=fake_executor,
        )
        self.brokers.append(value)
        return value.start()

    def request(self, identifier, **metadata):
        return InferenceRequest(
            request_id=identifier,
            run_id="test-run",
            role=metadata.pop("role", "writing"),
            backend=metadata.pop("backend", "codex_cli"),
            system="private input not retained in manifest",
            messages=[{"role": "user", "content": "PRIVATE_PROMPT_SENTINEL"}],
            timeout_seconds=metadata.pop("timeout_seconds", 5),
            metadata=metadata,
        )

    def manifest(self, broker):
        return json.loads(broker.manifest_path.read_text(encoding="utf-8"))

    def request_state(self, broker, identifier):
        return self.manifest(broker).get("requests", {}).get(identifier, {}).get("state")

    def test_result_manifest_usage_and_private_socket(self):
        broker = self.broker()
        endpoint = broker.endpoint
        self.assertEqual(stat.S_IMODE(Path(endpoint["socket_path"]).stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(Path(endpoint["socket_path"]).parent.stat().st_mode), 0o700)
        self.assertLess(len(endpoint["socket_path"]), 104)
        result = BrokerClient(endpoint).submit(self.request("normal", node_id="node-1"))
        self.assertNotEqual(json.loads(result.text)["pid"], os.getpid())
        self.assertEqual(result.backend, "codex_cli")
        manifest = self.manifest(broker)
        self.assertEqual(manifest["usage_summary"]["input_tokens"], 3)
        self.assertEqual(manifest["usage_summary"]["cached_input_tokens"], 1)
        self.assertIsNone(manifest["usage_summary"]["subscription_billing_cost"])
        self.assertEqual(manifest["requests"]["normal"]["node_id"], "node-1")
        self.assertEqual(manifest["pending_request_ids"], [])
        serialized = json.dumps(manifest)
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", serialized)
        self.assertNotIn(endpoint["token"], serialized)
        self.assertEqual(stat.S_IMODE(broker.manifest_path.stat().st_mode), 0o600)
        request_dir = Path(manifest["requests"]["normal"]["artifact_dir"])
        self.assertTrue((request_dir / "result.json").exists())
        broker.close()
        self.assertEqual(self.manifest(broker)["run_status"], "completed")
        self.assertFalse(Path(endpoint["socket_path"]).parent.exists())
        self.assertFalse(broker._process.is_alive())

    def test_run_provenance_and_request_digest_do_not_claim_full_parity(self):
        captured = {
            "revision": "a" * 40, "dirty": True, "status": "known", "reason": None,
            "captured_at": "2026-09-27T00:00:00+00:00", "scope": "module_project_root",
        }
        with patch("ai_scientist.inference.broker._source_provenance", return_value=captured) as collect:
            broker = self.broker(
                default={"backend": "codex_cli", "model": None},
                roles={"writing": {"model": "writer"}, "experiment_vision": {"backend": "claude_cli", "model": "vision"}},
            )
            collect.assert_called_once_with()
        manifest = self.manifest(broker)
        canonical = json.dumps(broker.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self.assertEqual(manifest["config_hash"], hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        self.assertEqual(set(manifest["resolved_roles"]), set(ROLES))
        self.assertEqual(len(manifest["resolved_roles"]), 15)
        self.assertEqual(manifest["resolved_roles"]["writing"], {"backend": "codex_cli", "model": "writer"})
        self.assertEqual(manifest["resolved_roles"]["experiment_vision"], {"backend": "claude_cli", "model": "vision"})
        self.assertEqual(manifest["source_revision"], "a" * 40)
        self.assertTrue(manifest["source_dirty"])
        self.assertEqual(manifest["source_provenance"], captured)
        self.assertIsNone(manifest["cli_versions"]["codex_cli"])
        self.assertEqual(manifest["cli_version_status"]["claude_cli"], "unknown_not_observed")
        client = BrokerClient(broker.endpoint)
        request = self.request("versioned", cli_version="fixture-cli 1.0")
        canonical_input = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        expected_digest = hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()
        client.submit(request)
        client.record_validation(request.request_id, accepted=True)
        client.submit(self.request("versioned-again", cli_version="fixture-cli 1.0"))
        client.submit(self.request("upgraded", cli_version="fixture-cli 1.1"))
        client.submit(self.request("unobserved"))
        manifest = self.manifest(broker)
        self.assertEqual(manifest["cli_versions"]["codex_cli"], ["fixture-cli 1.0", "fixture-cli 1.1"])
        self.assertEqual(manifest["cli_version_unknown_results"]["codex_cli"], 1)
        self.assertEqual(manifest["requests"][request.request_id]["input_sha256"], expected_digest)
        request_path = Path(manifest["requests"][request.request_id]["artifact_dir"]) / "request_metadata.json"
        self.assertEqual(json.loads(request_path.read_text())["input_sha256"], expected_digest)
        self.assertEqual(manifest["verified_capabilities"]["full_v2_parity"], "unknown_not_verified")
        self.assertEqual(manifest["verified_capabilities"]["live_probe_status"], "unknown_not_verified_by_broker")
        self.assertEqual(manifest["verified_capabilities"]["observed_transport_successes"]["codex_cli"], 4)
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", json.dumps(manifest))

    def test_source_provenance_is_read_only_bounded_and_environment_minimal(self):
        replies = [
            subprocess.CompletedProcess([], 0, stdout="b" * 40 + "\n"),
            subprocess.CompletedProcess([], 0, stdout=" M private-filename.py\n"),
        ]
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "never-read-or-persist-this-value",
            "GIT_DIR": "/unrelated/git/repository",
        }), patch("ai_scientist.inference.broker.subprocess.run", side_effect=replies) as run:
            value = _source_provenance()
        self.assertEqual(value["revision"], "b" * 40)
        self.assertTrue(value["dirty"])
        self.assertEqual(value["status"], "known")
        self.assertIsNone(value["reason"])
        self.assertNotIn("private-filename", json.dumps(value))
        self.assertNotIn("never-read-or-persist", json.dumps(value))
        self.assertEqual(run.call_count, 2)
        for invocation in run.call_args_list:
            self.assertEqual(invocation.args[0][0], "git")
            self.assertFalse(invocation.kwargs["shell"])
            self.assertLessEqual(invocation.kwargs["timeout"], 2)
            self.assertEqual(invocation.kwargs["cwd"], Path(__file__).resolve().parents[1])
            environment = invocation.kwargs["env"]
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("GIT_DIR", environment)
            self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
            self.assertIn("core.fsmonitor=false", invocation.args[0])

    def test_source_provenance_missing_git_and_partial_timeout_are_explicit(self):
        with patch("ai_scientist.inference.broker.subprocess.run", side_effect=FileNotFoundError):
            unavailable = _source_provenance()
        self.assertIsNone(unavailable["revision"])
        self.assertIsNone(unavailable["dirty"])
        self.assertEqual(unavailable["status"], "unknown")
        self.assertEqual(unavailable["reason"], "git_unavailable")
        with patch("ai_scientist.inference.broker.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout="c" * 40 + "\n"),
            subprocess.TimeoutExpired("git", 2),
        ]):
            partial = _source_provenance()
        self.assertEqual(partial["revision"], "c" * 40)
        self.assertIsNone(partial["dirty"])
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["reason"], "git_timeout")

    def test_shared_limits_across_processes_and_threads(self):
        broker = self.broker(providers={
            "codex_cli": {"max_in_flight": 2}, "claude_cli": {"max_in_flight": 1},
        })
        context = multiprocessing.get_context("spawn")
        reports = context.Queue()
        processes = []
        for index in range(2):
            requests = [
                self.request(f"child-{index}-codex", sleep=0.15).to_dict(),
                self.request(f"child-{index}-claude", backend="claude_cli", sleep=0.15).to_dict(),
            ]
            process = context.Process(target=_submit_in_process, args=(broker.endpoint, requests, reports))
            process.start()
            processes.append(process)
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    BrokerClient(broker.endpoint).submit,
                    [self.request(f"parent-{index}", sleep=0.15) for index in range(4)],
                ))
            for _ in processes:
                report = reports.get(timeout=10)
                self.assertTrue(report["ok"], report)
                results.extend(InferenceResult.from_dict(value) for value in report["results"])
            for process in processes:
                process.join(5)
                self.assertEqual(process.exitcode, 0)
            samples = [json.loads(value.text) for value in results]
            self.assertEqual({sample["pid"] for sample in samples}, {broker._process.pid})
            for backend, limit in (("codex_cli", 2), ("claude_cli", 1)):
                events = sorted(
                    event for sample in samples if sample["backend"] == backend
                    for event in ((sample["started"], 1), (sample["finished"], -1))
                )
                active = peak = 0
                for _, change in events:
                    active += change
                    peak = max(peak, active)
                self.assertLessEqual(peak, limit)
                self.assertEqual(peak, limit)
            self.assertEqual(len(self.manifest(broker)["requests"]), 8)
            self.assertEqual(self.manifest(broker)["usage_summary"]["input_tokens"], 24)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            reports.close()
            reports.join_thread()

    def test_queue_timeout_does_not_execute_or_cancel_running_request(self):
        broker = self.broker(queue_timeout_seconds=0.08)
        client = BrokerClient(broker.endpoint)
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(client.submit, self.request("running", sleep=0.3))
            _wait_until(lambda: self.request_state(broker, "running") == "running")
            with self.assertRaises(QueueTimeout) as caught:
                client.submit(self.request("queued"))
            self.assertFalse(caught.exception.metadata["executed"])
            self.assertEqual(running.result(timeout=3).request_id, "running")
        directory = Path(self.manifest(broker)["requests"]["queued"]["artifact_dir"])
        self.assertFalse((directory / "started.marker").exists())

    def test_bounded_queue_backpressure(self):
        broker = self.broker(max_queue_size=1)
        client = BrokerClient(broker.endpoint)
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(client.submit, self.request("running", mode="wait"))
            _wait_until(lambda: self.request_state(broker, "running") == "running")
            queued = executor.submit(client.submit, self.request("queued"))
            _wait_until(lambda: self.request_state(broker, "queued") == "queued")
            with self.assertRaises(QueueTimeout):
                client.submit(self.request("overflow"))
            client.cancel_all()
            for future in (running, queued):
                with self.assertRaises(Cancelled):
                    future.result(timeout=3)

    def test_cancellation_waits_for_transport_cleanup(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(client.submit, self.request("cancel-me", mode="wait"))
            _wait_until(lambda: self.request_state(broker, "cancel-me") == "running")
            self.assertTrue(client.cancel("cancel-me"))
            with self.assertRaises(Cancelled):
                future.result(timeout=3)
        manifest = self.manifest(broker)
        directory = Path(manifest["requests"]["cancel-me"]["artifact_dir"])
        self.assertTrue((directory / "cancelled.marker").exists())
        self.assertEqual(manifest["pending_request_ids"], [])
        self.assertFalse(client.cancel("cancel-me"))
        self.assertFalse(client.cancel("unknown"))

    def test_client_process_death_cancels_owned_request(self):
        broker = self.broker()
        process = multiprocessing.get_context("spawn").Process(
            target=_wait_submit,
            args=(broker.endpoint, self.request("abandoned", mode="wait").to_dict()),
        )
        process.start()
        try:
            _wait_until(lambda: self.request_state(broker, "abandoned") == "running")
            process.terminate()
            process.join(3)
            _wait_until(lambda: self.request_state(broker, "abandoned") == "cancelled")
            directory = Path(self.manifest(broker)["requests"]["abandoned"]["artifact_dir"])
            self.assertTrue((directory / "cancelled.marker").exists())
        finally:
            if process.is_alive():
                process.kill()
            process.join(3)

    def test_quota_stops_new_and_queued_requests_without_fallback(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        with ThreadPoolExecutor(max_workers=2) as executor:
            failing = executor.submit(client.submit, self.request("quota", mode="quota", sleep=0.2))
            _wait_until(lambda: self.request_state(broker, "quota") == "running")
            queued = executor.submit(client.submit, self.request("behind-quota"))
            _wait_until(lambda: self.request_state(broker, "behind-quota") == "queued")
            for future in (failing, queued):
                with self.assertRaises(QuotaExhausted):
                    future.result(timeout=3)
        with self.assertRaises(QuotaExhausted):
            client.submit(self.request("new-provider", backend="claude_cli"))
        manifest = self.manifest(broker)
        self.assertEqual(manifest["run_status"], "stopped_quota")
        self.assertEqual(manifest["pending_request_ids"], [])
        directory = Path(manifest["requests"]["behind-quota"]["artifact_dir"])
        self.assertFalse((directory / "started.marker").exists())
        self.assertEqual(manifest["requests"]["behind-quota"]["error"]["request_id"], "behind-quota")
        broker.close()
        self.assertEqual(self.manifest(broker)["run_status"], "stopped_quota")

    def test_authentication_error_remains_typed(self):
        broker = self.broker()
        with self.assertRaises(AuthenticationRequired):
            BrokerClient(broker.endpoint).submit(self.request("auth", mode="auth"))
        self.assertEqual(self.manifest(broker)["run_status"], "stopped_auth")

    def test_inference_timeout_is_not_a_queue_timeout(self):
        broker = self.broker()
        with self.assertRaises(InferenceTimeout):
            BrokerClient(broker.endpoint).submit(self.request("timeout", mode="wait", timeout_seconds=0.04))
        request = self.manifest(broker)["requests"]["timeout"]
        self.assertEqual(request["error"]["code"], "inference_timeout")
        self.assertIsNotNone(request["started_at"])

    def test_artifact_failure_is_reported_without_hanging_or_replay(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        with self.assertRaises(BrokerUnavailable) as caught:
            client.submit(self.request("cannot-save", mode="artifact_error"))
        self.assertEqual(caught.exception.metadata["phase"], "artifacts")
        self.assertEqual(self.manifest(broker)["run_status"], "failed_transport")
        with self.assertRaises(BrokerUnavailable):
            client.submit(self.request("not-executed"))
        self.assertNotIn("not-executed", self.manifest(broker)["requests"])

    def test_payload_limit_rejects_instead_of_truncating(self):
        broker = self.broker(max_payload_bytes=1024)
        request = self.request("too-large")
        request.system = "A" * 2048
        with self.assertRaises(ContextLimitExceeded) as caught:
            BrokerClient(broker.endpoint).submit(request)
        self.assertFalse(caught.exception.metadata["executed"])
        self.assertEqual(self.manifest(broker)["requests"], {})

    def test_request_delivery_deduplication_not_prompt_cache(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        request = self.request("same-id")
        first = client.submit(request)
        again = client.submit(request)
        independent = client.submit(self.request("new-id"))
        self.assertEqual(first.to_dict(), again.to_dict())
        self.assertNotEqual(json.loads(first.text)["started"], json.loads(independent.text)["started"])
        request.system = "different request under the same ID"
        with self.assertRaises(BrokerUnavailable):
            client.submit(request)
        self.assertEqual(self.manifest(broker)["usage_summary"]["input_tokens"], 6)

    def test_business_validation_is_separate_from_transport_success(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        request = self.request("structured", logical_request_id="logical", attempt=1)
        request.schema = {"type": "object", "required": ["metric"]}
        result = client.submit(request)
        metadata = self.manifest(broker)["requests"]["structured"]
        self.assertEqual(metadata["transport_status"], "completed")
        self.assertEqual(metadata["validation_status"], "pending")
        self.assertEqual(metadata["logical_request_id"], "logical")
        error = {"code": "invalid_structured_output", "message": "Missing metric"}
        client.record_validation("structured", accepted=False, error=error)
        metadata = self.manifest(broker)["requests"]["structured"]
        self.assertEqual(metadata["transport_status"], "completed")
        self.assertEqual(metadata["validation_status"], "rejected")
        directory = Path(metadata["artifact_dir"])
        self.assertTrue((directory / "validation.json").exists())
        self.assertEqual(json.loads((directory / "result.json").read_text())["text"], result.text)
        # Identical delivery is idempotent; repairs must get a new request ID.
        client.record_validation("structured", accepted=False, error=error)
        with self.assertRaises(BrokerUnavailable):
            client.record_validation("structured", accepted=True)

    def test_fair_scheduling_rotates_roles(self):
        broker = self.broker()
        client = BrokerClient(broker.endpoint)
        with ThreadPoolExecutor(max_workers=5) as executor:
            first = executor.submit(client.submit, self.request("block", mode="wait", role="writing"))
            _wait_until(lambda: self.request_state(broker, "block") == "running")
            bulk = [
                executor.submit(client.submit, self.request(f"bulk-{index}", role="writing"))
                for index in range(3)
            ]
            _wait_until(lambda: all(self.request_state(broker, f"bulk-{index}") == "queued" for index in range(3)))
            other = executor.submit(client.submit, self.request("other-role", role="summary"))
            _wait_until(lambda: self.request_state(broker, "other-role") == "queued")
            client.cancel("block")
            with self.assertRaises(Cancelled):
                first.result(timeout=3)
            results = [future.result(timeout=3) for future in bulk] + [other.result(timeout=3)]
        order = [value["request_id"] for value in sorted(
            (json.loads(result.text) for result in results), key=lambda value: value["started"]
        )]
        self.assertLessEqual(order.index("other-role"), 1)

    def test_close_cancels_and_reaps_running_broker(self):
        broker = self.broker()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(BrokerClient(broker.endpoint).submit, self.request("pending", mode="wait"))
            _wait_until(lambda: self.request_state(broker, "pending") == "running")
            started = time.monotonic()
            broker.close()
            self.assertLess(time.monotonic() - started, 5)
            with self.assertRaises(Cancelled):
                future.result(timeout=2)
        self.assertFalse(broker._process.is_alive())
        self.assertEqual(self.manifest(broker)["run_status"], "cancelled")
        self.assertEqual(self.manifest(broker)["pending_request_ids"], [])

    def test_dead_broker_is_not_automatically_replayed(self):
        broker = self.broker()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(BrokerClient(broker.endpoint).submit, self.request("interrupted", mode="wait"))
            _wait_until(lambda: self.request_state(broker, "interrupted") == "running")
            broker._process.kill()
            broker._process.join(3)
            with self.assertRaises(BrokerUnavailable) as caught:
                future.result(timeout=3)
            self.assertFalse(caught.exception.metadata["replayed"])
        broker.close()
        manifest = self.manifest(broker)
        self.assertEqual(manifest["run_status"], "failed_transport")
        self.assertEqual(manifest["requests"]["interrupted"]["state"], "unknown")
        self.assertEqual(len(manifest["requests"]), 1)

    def test_dead_broker_owned_cli_cleanup_leaves_unrelated_process_alive(self):
        broker = self.broker()
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        owned_pid = None
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    BrokerClient(broker.endpoint).submit,
                    self.request("orphan-cli", mode="subprocess", timeout_seconds=30),
                )
                _wait_until(lambda: self.request_state(broker, "orphan-cli") == "running")
                directory = Path(self.manifest(broker)["requests"]["orphan-cli"]["artifact_dir"])
                record = directory / "process.json"
                _wait_until(record.exists)
                _wait_until((directory / "executed.marker").exists)
                owned_pid = json.loads(record.read_text(encoding="utf-8"))["pid"]
                from ai_scientist.inference.process import _process_identity, _read_ownership
                ownership = _read_ownership(record)
                self.assertIsNotNone(ownership)
                self.assertEqual(_process_identity(owned_pid), ownership["identity"])
                self.assertTrue(_pid_running(owned_pid))
                # Never race the active transport's own process registry.
                self.assertEqual(broker.cleanup_owned_processes(), 0)
                broker._process.kill()
                broker._process.join(3)
                with self.assertRaises(BrokerUnavailable):
                    future.result(timeout=3)
                cleaned = broker.cleanup_owned_processes()
                _wait_until(lambda: not _pid_running(owned_pid))
                self.assertEqual(cleaned, 1)
                self.assertIsNone(unrelated.poll())
                broker.close()
                self.assertEqual(self.manifest(broker)["run_status"], "failed_transport")
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)
            if owned_pid is not None and _pid_running(owned_pid):
                # Test-owned sleeper only, never an account CLI or OS-wide scan.
                try:
                    os.killpg(owned_pid, 9)
                except ProcessLookupError:
                    pass

    def test_json_rpc_authentication_and_path_containment(self):
        broker = self.broker()
        endpoint = broker.endpoint
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(endpoint["socket_path"])
            _send_frame(connection, {"op": "ping", "token": "invalid"})
            reply = _receive_frame(connection, time.monotonic() + 2)
            self.assertFalse(reply["ok"])
        identifier = "../../outside-run"
        BrokerClient(endpoint).submit(self.request(identifier))
        directory = Path(self.manifest(broker)["requests"][identifier]["artifact_dir"])
        self.assertTrue(directory.is_relative_to(broker.artifact_dir / "requests"))
        self.assertNotIn("outside-run", directory.name)
        other_run = self.request("wrong-run")
        other_run.run_id = "another-run"
        with self.assertRaises(BrokerUnavailable):
            BrokerClient(endpoint).submit(other_run)

    def test_orphan_cleanup_only_considers_accepted_private_request_directories(self):
        broker = self.broker()
        BrokerClient(broker.endpoint).submit(self.request("registered"))
        accepted = Path(self.manifest(broker)["requests"]["registered"]["artifact_dir"])
        broker.close()
        (accepted / "process.json").write_text("{}", encoding="utf-8")
        forged = broker.artifact_dir / "requests" / ("a" * 64)
        forged.mkdir()
        (forged / "process.json").write_text("{}", encoding="utf-8")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "process.json").write_text("{}", encoding="utf-8")
        (broker.artifact_dir / "requests" / ("b" * 64)).symlink_to(outside, target_is_directory=True)
        with patch("ai_scientist.inference.process.cleanup_owned_process", return_value=True) as cleanup:
            self.assertEqual(broker.cleanup_owned_processes(), 1)
            self.assertEqual(cleanup.call_count, 1)
            self.assertEqual(cleanup.call_args.args[0], accepted / "process.json")

    def test_reusing_an_artifact_directory_is_rejected(self):
        broker = self.broker()
        BrokerClient(broker.endpoint).submit(self.request("saved"))
        broker.close()
        second = Broker(broker.config, "new-run", broker.artifact_dir, _executor=fake_executor)
        self.brokers.append(second)
        with self.assertRaises(BrokerUnavailable):
            second.start()
        self.assertEqual(self.manifest(broker)["run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
