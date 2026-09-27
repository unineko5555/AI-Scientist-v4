"""Read-only CLI diagnostics, with explicitly opt-in subscription probes.

Run ``python -m ai_scientist.inference.doctor --llm-backend codex_cli``
without ``--probe`` to inspect the installed CLI and its authentication only.
No credentials are read by this module and no model request is made by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile

from .config import InferenceConfig, add_inference_arguments, config_from_args
from .contracts import ROLES
from .errors import ConfigurationError, InferenceError, UnsupportedCapability
from .runtime import InferenceRuntime


_USAGE_NOTICE = (
    "--probe sends model requests and consumes subscription usage. Account extra-usage "
    "or spending settings remain controlled by the account owner."
)
_PARITY_NOTICE = (
    "These diagnostics do not establish full V2 pipeline parity, research quality, "
    "long-context behavior, or managed-policy/tool-isolation guarantees."
)
_PROBE_CHECKS = ("text", "structured_output", "ordered_images")
_COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}
_ERROR_MESSAGES = {
    "configuration_error": (
        "CLI configuration is required and must be valid. Select --llm-backend "
        "codex_cli or claude_cli, or supply a CLI --inference-config."
    ),
    "authentication_required": (
        "Subscription authentication could not be verified. Authenticate the selected "
        "CLI yourself with codex login or claude auth login, then rerun the doctor."
    ),
    "unsupported_capability": (
        "A required CLI binary, protocol version, capability, or probe dependency is "
        "unavailable. Check installation and the supported CLI profile."
    ),
    "quota_exhausted": "The CLI reported exhausted subscription usage; no API fallback was attempted.",
    "context_limit_exceeded": "The complete diagnostic input exceeded a configured or model limit.",
    "transient_provider_error": "The CLI reported a temporary provider or connection failure.",
    "queue_timeout": "The diagnostic request exceeded its queue deadline.",
    "inference_timeout": "The diagnostic request exceeded its execution deadline.",
    "invalid_structured_output": "The model response did not satisfy the requested output contract.",
    "incomplete_output": "The CLI did not return a complete verifiable final response.",
    "cancelled": "Diagnostics were cancelled; owned inference work is being closed.",
    "broker_unavailable": "The shared inference broker could not complete the diagnostic request.",
    "provider_error": "The CLI transport failed; no API fallback was attempted.",
    "inference_error": "Inference diagnostics failed; no API fallback was attempted.",
}


def _positive_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose subscription CLI inference without issuing model requests by default.",
        epilog=_USAGE_NOTICE + " " + _PARITY_NOTICE,
    )
    add_inference_arguments(parser)
    parser.add_argument(
        "--probe", action="store_true",
        help=("Explicitly consume subscription usage: by default send three small application requests per "
              "distinct backend/model combination (text, structured output, ordered images)."),
    )
    parser.add_argument(
        "--probe-check", action="append", choices=_PROBE_CHECKS,
        help=("Run only selected checks; requires --probe. Repeat to select multiple checks; "
              "each check runs once. Omitted means all three checks."),
    )
    parser.add_argument(
        "--probe-timeout", type=_positive_timeout, default=120.0, metavar="SECONDS",
        help="Per-request probe deadline, capped by the configured deadline (default: 120); probes do not retry.",
    )
    return parser


def _configuration_summary(config: InferenceConfig) -> dict:
    used = {config.resolve(role).backend for role in ROLES}
    return {
        "mode": config.mode,
        "subscription_only": config.subscription_only,
        "history_mode": config.history_mode,
        "fallback": config.fallback,
        "request_timeout_seconds": config.request_timeout_seconds,
        "transient_retry_limit": config.transient_retry_limit,
        "schema_repair_limit": config.schema_repair_limit,
        "providers": {
            backend: {"executable": config.providers[backend]["executable"],
                      "max_in_flight": config.providers[backend]["max_in_flight"]}
            for backend in sorted(used)
        },
    }


def _provider_summary(backend: str, value: dict) -> dict:
    """Keep a small allowlist; never forward CLI output or arbitrary metadata."""
    auth = value.get("authentication", {})
    if not isinstance(auth, dict):
        auth = {}
    method = auth.get("method")
    version = value.get("version")
    result = {
        "backend": backend,
        "version": version if isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version) else None,
        "authentication": {
            "method": method if method in ("chatgpt", "claude.ai") else "unknown",
            "subscription": auth.get("subscription") is True,
        },
        "protocol_profile": "subscription-stateless-v1" if value.get("protocol_profile") == "subscription-stateless-v1" else "unknown",
        "capabilities": {name: "declared_not_live_verified" for name in ("text", "images", "json_schema")},
    }
    if auth.get("plan") in ("pro", "max", "team", "enterprise"):
        result["authentication"]["plan"] = auth["plan"]
    return result


def _safe_error(exc: BaseException) -> dict:
    """Exception text/metadata can contain provider output; never print either."""
    code = exc.code if isinstance(exc, InferenceError) else "doctor_internal_error"
    if code not in _ERROR_MESSAGES:
        code = "doctor_internal_error"
    return {
        "code": code,
        "message": _ERROR_MESSAGES.get(code, "Diagnostics failed unexpectedly; no provider output or credentials were printed."),
    }


def _probe_targets(config: InferenceConfig) -> list[tuple]:
    grouped = {}
    for role in ROLES:
        client = config.resolve(role)
        key = (client.backend, client.model)
        if key not in grouped:
            grouped[key] = (client, [])
        grouped[key][1].append(role)
    return list(grouped.values())


def _output_fingerprint(value, *, text_output: bool) -> dict:
    """Replayable expected-answer identity without printing the answer itself.

    Text is stripped exactly as in the live comparison, then encoded as a JSON
    string. Structured data is encoded directly as JSON, with object keys sorted
    and array order preserved. This is also sufficient to check saved responses
    after a transport/parser failure, without issuing another model request.
    """
    normalized = value.strip() if text_output else value
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {
        "algorithm": "sha256",
        "encoding": "utf-8",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)",
        "normalization": "text.strip()" if text_output else "structured_json_value",
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def _call_probe(runtime, client, report, target, name, *, expected_output, **kwargs):
    check = {"name": name, "status": "running",
             "expected_output_fingerprint": _output_fingerprint(expected_output, text_output=name == "text")}
    target["checks"].append(check)
    report["stage"] = name
    report["live_requests_attempted"] += 1
    try:
        result = runtime.generate(client, **kwargs)
    except BaseException:
        check["status"] = "failed"
        raise
    return result, check


def _expect(check: dict, condition: bool) -> None:
    if not condition:
        check["status"] = "failed"
        raise UnsupportedCapability("A live capability probe returned an incorrect response")
    check["status"] = "passed"


def _summarize_probes(report: dict) -> None:
    """Do not turn a successful subset, or an incomplete target set, into full verification."""
    targets = report["probes"]
    for target in targets:
        target["passed_checks"] = [check["name"] for check in target["checks"] if check["status"] == "passed"]
        target["all_three_checks_verified"] = set(target["passed_checks"]) == set(_PROBE_CHECKS)
    all_targets_present = bool(report["probe_target_count"] and len(targets) == report["probe_target_count"])
    report["passed_probe_checks"] = [
        check for check in _PROBE_CHECKS
        if all_targets_present and all(check in target["passed_checks"] for target in targets)
    ]
    report["all_three_checks_verified"] = all_targets_present and all(
        target["all_three_checks_verified"] for target in targets
    )
    report["live_inference_verified"] = report["all_three_checks_verified"]


def _run_probes(runtime: InferenceRuntime, report: dict) -> None:
    selected = report["selected_probe_checks"]
    if "ordered_images" in selected:
        try:
            from PIL import Image
        except ImportError as exc:
            raise UnsupportedCapability("Live image diagnostics require Pillow") from exc
    for client, roles in _probe_targets(runtime.config):
        target = {"backend": client.backend, "model": client.model, "roles": roles, "checks": []}
        report["probes"].append(target)
        nonce = "doctor-" + secrets.token_hex(8) if any(name in selected for name in ("text", "structured_output")) else None
        if "text" in selected:
            result, check = _call_probe(
                runtime, client, report, target, "text",
                expected_output=nonce,
                prompt=f"Return exactly the following token with no explanation, quotes, or formatting: {nonce}",
                system_message="You are performing a minimal inference transport diagnostic.",
            )
            _expect(check, result.text.strip() == nonce)

        if "structured_output" in selected:
            schema = {
                "type": "object",
                "properties": {
                    "echo": {"type": "string", "enum": [nonce]},
                    "count": {"type": "integer", "enum": [3]},
                    "by_id": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "optional_note": {"type": "string"},
                },
                "required": ["echo", "count", "by_id"],
                "additionalProperties": False,
            }
            expected = {"echo": nonce, "count": 3, "by_id": {"alpha": 1, "beta": 2}}
            result, check = _call_probe(
                runtime, client, report, target, "structured_output", schema=schema, expected_output=expected,
                prompt=("Return this exact JSON value through the required output contract; omit the optional_note field: "
                        + json.dumps(expected)),
                system_message="You are verifying structured data preservation, including optional fields and dynamic object keys.",
            )
            check["schema_transport"] = (
                "json_string_envelope_with_original_schema_host_validation"
                if client.backend == "codex_cli" else "native_schema_with_host_validation"
            )
            _expect(check, result.structured_data == expected)

        if "ordered_images" in selected:
            # Random order, distinct colors, and neutral file names prevent a fixed
            # answer or filename-only model from passing this vision smoke test.
            colors = secrets.SystemRandom().sample(list(_COLOR_RGB), 3)
            with tempfile.TemporaryDirectory(prefix="ai-scientist-doctor-images-") as location:
                images = []
                for index, color in enumerate(colors):
                    path = Path(location) / f"image-{index + 1}.png"
                    Image.new("RGB", (96, 96), _COLOR_RGB[color]).save(path, format="PNG")
                    path.chmod(0o600)
                    images.append(str(path))
                image_schema = {
                    "type": "object",
                    "properties": {"colors": {"type": "array", "items": {"type": "string", "enum": list(_COLOR_RGB)},
                                               "minItems": 3, "maxItems": 3}},
                    "required": ["colors"], "additionalProperties": False,
                }
                result, check = _call_probe(
                    runtime, client, report, target, "ordered_images", images=images, schema=image_schema,
                    expected_output={"colors": colors},
                    prompt=("Look at all three attached images. Each is filled with one solid color. "
                            "Return its color name in attachment order as the colors array. "
                            "Read the actual pixels; the file names do not encode colors."),
                    system_message="You are verifying that multiple real image inputs arrive in order.",
                )
                _expect(check, result.structured_data == {"colors": colors})


def _artifact_summary(runtime: InferenceRuntime) -> dict:
    path = runtime.artifact_dir
    exists = path.exists()
    result = {"directory": str(path), "created": exists, "expected_directory_mode": "0700"}
    if exists:
        result["directory_mode"] = f"{stat.S_IMODE(path.stat().st_mode):04o}"
    return result


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    report = {
        "doctor_version": 1, "status": "error", "stage": "configuration",
        "live_probe_requested": args.probe,
        "live_requests_attempted": 0,
        "planned_live_requests": 0,
        "live_inference_verified": False,
        "all_three_checks_verified": False,
        "selected_probe_checks": [check for check in _PROBE_CHECKS if check in (args.probe_check or _PROBE_CHECKS)] if args.probe else [],
        "passed_probe_checks": [],
        "probe_target_count": 0,
        "full_v2_parity_verified": False,
        "api_fallback_enabled": False,
        "probes": [],
        "notes": [_USAGE_NOTICE, _PARITY_NOTICE],
    }
    runtime = None
    exit_code = 1
    try:
        if args.probe_check and not args.probe:
            report["hint"] = "--probe-check requires --probe; selecting checks alone never enables live inference."
            raise ConfigurationError("--probe-check requires --probe")
        config = config_from_args(args, argv)
        if config.mode != "cli":
            raise ConfigurationError("The CLI doctor cannot issue API requests")
        if args.probe:
            # One request per selected check/target, without hidden retries.
            values = config.to_dict()
            values.update({"transient_retry_limit": 0, "schema_repair_limit": 0,
                           "request_timeout_seconds": min(config.request_timeout_seconds, args.probe_timeout)})
            config = InferenceConfig.from_dict(values)
        report["configuration"] = _configuration_summary(config)
        report["roles"] = {
            role: {"backend": config.resolve(role).backend, "model": config.resolve(role).model}
            for role in ROLES
        }
        report["probe_target_count"] = len(_probe_targets(config)) if args.probe else 0
        report["planned_live_requests"] = len(report["selected_probe_checks"]) * report["probe_target_count"]
        runtime = InferenceRuntime(config, artifact_dir=args.inference_log_dir)
        report["stage"] = "read_only_preflight"
        preflight = runtime.preflight()
        report["providers"] = {
            backend: _provider_summary(backend, value)
            for backend, value in preflight["providers"].items()
        }
        report["read_only_preflight_passed"] = True
        if args.probe:
            _run_probes(runtime, report)
            _summarize_probes(report)
            report["status"] = "probes_passed" if report["all_three_checks_verified"] else "passed_selected_probes"
        else:
            report["status"] = "ready_read_only"
        report["stage"] = "completed"
        exit_code = 0
    except KeyboardInterrupt:
        report["error"] = {"code": "cancelled", "message": _ERROR_MESSAGES["cancelled"]}
        exit_code = 130
    except Exception as exc:
        report["error"] = _safe_error(exc)
        exit_code = 2 if isinstance(exc, ConfigurationError) else 1
    finally:
        _summarize_probes(report)
        if runtime is not None:
            try:
                runtime.close()
                report["artifacts"] = _artifact_summary(runtime)
            except Exception as exc:
                report["status"] = "error"
                report["cleanup_error"] = _safe_error(exc)
                exit_code = 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
