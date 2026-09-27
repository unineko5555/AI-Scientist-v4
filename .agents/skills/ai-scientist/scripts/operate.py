#!/usr/bin/env python3
"""Thin, subscription-only operator for the unchanged V2 entrypoints.

Ideation, research and live probes print a plan unless --execute is supplied.
Execution replaces this process: the existing runner retains signal handling,
process ownership and artifact management. There is no detached job or resume API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys


PROVIDERS = ("codex", "claude", "mixed")
PROBE_CHECKS = ("text", "structured_output", "ordered_images")
PROJECT_MARKERS = ("launch_scientist_bfts.py", "ai_scientist/inference/config.py", "bfts_config.yaml")


class OperatorError(ValueError):
    """Invalid operator input; never a reason to fall back to an API."""


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", help="Checkout location; normally found from this script, including symlinks")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("doctor", "ideate", "run"):
        sub = commands.add_parser(action, allow_abbrev=False)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--provider", choices=PROVIDERS)
        source.add_argument("--inference-config", help="Existing subscription-only CLI JSON/YAML profile")
        sub.add_argument("--model", help="Explicit default CLI model; otherwise use the selected profile")
        sub.add_argument("--inference-log-dir", help="Private inference artifacts; relative to the caller's working directory")
        execution = sub.add_mutually_exclusive_group()
        execution.add_argument("--execute", action="store_true", help="Execute the displayed operation; live operations consume usage")
        execution.add_argument("--dry-run", action="store_true", help="Print the plan only; do not invoke a CLI or write artifacts")
        if action == "doctor":
            sub.add_argument("--probe", action="store_true", help="Plan live checks; also requires --execute to send requests")
            sub.add_argument("--probe-check", action="append", choices=PROBE_CHECKS)
            sub.add_argument("--probe-timeout", type=positive, default=120)
        elif action == "ideate":
            sub.add_argument("--workshop-file", required=True)
            sub.add_argument("--max-num-generations", type=positive, default=1)
            sub.add_argument("--num-reflections", type=positive, default=5)
            sub.add_argument("--allow-existing-ideas", action="store_true", help="Allow V2 to reload and rewrite the adjacent existing idea archive")
        else:
            sub.add_argument("--ideas", required=True)
            sub.add_argument("--idea-index", type=nonnegative, default=0)
            sub.add_argument("--attempt-id", type=nonnegative, default=0)
            sub.add_argument("--writeup-type", choices=("normal", "icbinb"), default="icbinb")
            sub.add_argument("--writeup-retries", type=positive, default=3)
            sub.add_argument("--num-cite-rounds", type=nonnegative, default=20)
            sub.add_argument("--load-code", action="store_true")
            sub.add_argument("--add-dataset-ref", action="store_true")
            sub.add_argument("--skip-writeup", action="store_true")
            sub.add_argument("--skip-review", action="store_true")
    status = commands.add_parser("status", allow_abbrev=False)
    status.add_argument("--run-dir", help="Exact experiment directory; otherwise list recent experiments")
    status.add_argument("--limit", type=positive, default=10)
    return parser


def project_root(value=None):
    candidates = [Path(value).expanduser().resolve()] if value else Path(__file__).resolve().parents
    for candidate in candidates:
        if all((candidate / marker).is_file() for marker in PROJECT_MARKERS):
            return candidate
    raise OperatorError("AI-Scientist checkout not found. Supply --project-root before the subcommand.")


def input_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise OperatorError(f"Input file does not exist: {path}")
    return path


def read_json(path, limit=8 * 1024 * 1024):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise OperatorError(f"JSON file exceeds {limit} bytes: {path}")
    return json.loads(raw)


def cli_config(args, root):
    # Import only after --help/status; neither needs the research dependencies.
    if not sys.path or sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    from ai_scientist.inference.config import InferenceConfig, load_config
    from ai_scientist.inference.errors import ConfigurationError

    path = input_file(root / "configs" / f"inference.{args.provider}.yaml" if args.provider else args.inference_config)
    try:
        config = load_config(path)
        if config.mode != "cli" or not config.subscription_only:
            raise OperatorError("This skill requires mode: cli and subscription_only: true; API profiles are not allowed.")
        if args.model is not None:
            data = config.to_dict()
            data["default"]["model"] = args.model
            config = InferenceConfig.from_dict(data)
    except ConfigurationError as exc:
        raise OperatorError(str(exc)) from exc
    return path, config


def build_plan(args, root):
    profile, config = cli_config(args, root)
    from ai_scientist.inference.contracts import ROLES

    options = ["--inference-config", str(profile)]
    if args.model is not None:
        options += ["--llm-model", args.model]
    log_dir = args.inference_log_dir
    if log_dir is None and args.action in ("doctor", "ideate"):
        log_dir = str(root / ".inference")
    if log_dir is not None:
        options += ["--inference-log-dir", str(Path(log_dir).expanduser().resolve())]
    plan = {
        "action": args.action, "cwd": str(root), "inference_config": str(profile),
        "mode": config.mode, "subscription_only": config.subscription_only,
        "api_fallback_enabled": False,
        "roles": {role: {"backend": config.resolve(role).backend, "model": config.resolve(role).model} for role in ROLES},
        "live_inference": args.action != "doctor" or args.probe,
        "full_v2_parity_verified": False,
        "inference_limits": {
            "max_in_flight": {backend: config.providers[backend].get("max_in_flight", 1)
                              for backend in sorted({config.resolve(role).backend for role in ROLES})},
            "request_timeout_seconds": config.request_timeout_seconds,
            "queue_timeout_seconds": config.queue_timeout_seconds,
            "transient_retry_limit": config.transient_retry_limit,
            "schema_repair_limit": config.schema_repair_limit,
        },
    }
    if args.action == "doctor":
        if args.probe_check and not args.probe:
            raise OperatorError("--probe-check requires --probe; it never enables live inference by itself.")
        argv = [sys.executable, "-m", "ai_scientist.inference.doctor"]
        plan["planned_live_requests"] = 0
        if args.probe:
            plan["inference_limits"].update({
                "request_timeout_seconds": min(config.request_timeout_seconds, args.probe_timeout),
                "transient_retry_limit": 0, "schema_repair_limit": 0,
            })
            argv += ["--probe", "--probe-timeout", str(args.probe_timeout)]
            checks = list(dict.fromkeys(args.probe_check or PROBE_CHECKS))
            for check in checks:
                argv += ["--probe-check", check]
            targets = {(config.resolve(role).backend, config.resolve(role).model) for role in ROLES}
            plan["planned_live_requests"] = len(checks) * len(targets)
    elif args.action == "ideate":
        workshop = input_file(args.workshop_file)
        output = workshop.with_suffix(".json")
        if output.resolve() == workshop or (output.exists() and output.samefile(workshop)):
            raise OperatorError("The workshop must not also be the .json idea output.")
        if output.exists() and not args.allow_existing_ideas:
            raise OperatorError("The adjacent idea archive already exists. Use a new topic path or explicitly --allow-existing-ideas.")
        if output.is_symlink():
            raise OperatorError("Refusing to write ideas through a symlink.")
        argv = [sys.executable, "-m", "ai_scientist.perform_ideation_temp_free",
                "--workshop-file", str(workshop), "--max-num-generations", str(args.max_num_generations),
                "--num-reflections", str(args.num_reflections)]
        plan["idea_output"] = str(output)
        plan["rewrites_existing_ideas"] = output.exists()
    else:
        ideas_path = input_file(args.ideas)
        ideas = read_json(ideas_path)
        if not isinstance(ideas, list) or args.idea_index >= len(ideas):
            raise OperatorError("--ideas must contain an array with the selected --idea-index.")
        selected = ideas[args.idea_index]
        name = selected.get("Name") if isinstance(selected, dict) else None
        if not isinstance(name, str) or not name.strip() or any(char in name for char in ("/", "\\", "\x00")) or name in (".", ".."):
            raise OperatorError("The selected idea needs a nonempty Name without path separators.")
        if args.load_code:
            input_file(ideas_path.with_suffix(".py"))
        if args.add_dataset_ref:
            input_file(root / "hf_dataset_reference.py")
        argv = [sys.executable, str(root / "launch_scientist_bfts.py"),
                "--load_ideas", str(ideas_path), "--idea_idx", str(args.idea_index),
                "--attempt_id", str(args.attempt_id), "--writeup-type", args.writeup_type,
                "--writeup-retries", str(args.writeup_retries), "--num_cite_rounds", str(args.num_cite_rounds)]
        for field, flag in (("load_code", "--load_code"), ("add_dataset_ref", "--add_dataset_ref"),
                            ("skip_writeup", "--skip_writeup"), ("skip_review", "--skip_review")):
            if getattr(args, field):
                argv.append(flag)
        import yaml

        try:
            with (root / "bfts_config.yaml").open(encoding="utf-8") as stream:
                research = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise OperatorError("Cannot parse bfts_config.yaml.") from exc
        if not isinstance(research, dict) or not isinstance(research.get("agent"), dict):
            raise OperatorError("bfts_config.yaml must contain an agent mapping.")
        agent = research["agent"]
        plan["research_settings"] = {key: agent.get(key) for key in ("num_workers", "stages", "steps", "multi_seed_eval")}
        execution = research.get("exec", {})
        if not isinstance(execution, dict):
            raise OperatorError("bfts_config.yaml exec must be a mapping.")
        plan["research_settings"]["execution_timeout_seconds"] = execution.get("timeout")
        plan["experiment_parent"] = str(root / "experiments")
        plan["selected_idea_index"] = args.idea_index
        plan["selected_idea_name"] = name
        plan["writeup_type"] = args.writeup_type
        plan["skipped_stages"] = ["citations", "writeup", "review"] if args.skip_writeup else (["review"] if args.skip_review else [])
        plan["generated_code_execution"] = True
    plan["argv"] = argv + options
    plan["command_preview"] = shlex.join(plan["argv"])
    plan["execution_requested"] = args.execute or (args.action == "doctor" and not args.probe and not args.dry_run)
    plan["note"] = "Live operations consume subscription usage. This is not an OS sandbox or a total usage/time budget."
    return plan


def status_report(args, root):
    if args.run_dir:
        directory = Path(args.run_dir).expanduser().resolve()
        if not directory.is_dir():
            raise OperatorError(f"Experiment directory does not exist: {directory}")
        directories = [directory]
    else:
        parent = root / "experiments"
        directories = sorted((path for path in parent.iterdir() if path.is_dir()), reverse=True)[:args.limit] if parent.is_dir() else []
    runs = []
    for directory in directories:
        entry = {"run_dir": str(directory), "recorded_status": "unknown"}
        marker = directory / "run_status.json"
        try:
            data = read_json(marker, limit=64 * 1024)
            if not isinstance(data, dict) or not isinstance(data.get("status"), str):
                raise OperatorError("Invalid status record")
            entry["recorded_status"] = data["status"]
            for key in ("stage", "updated_at", "error_type"):
                if isinstance(data.get(key), str):
                    entry[key] = data[key]
        except (OSError, ValueError):
            entry["status_record"] = "missing_or_invalid"
        runs.append(entry)
    return {"action": "status", "runs": runs, "liveness": "not_checked", "automatic_resume_supported": False,
            "note": "Saved research status only; running may be stale. Inference completion alone is not research completion."}


def execute_plan(plan):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AI_SCIENTIST_INFERENCE_")}
    environment["PYTHONUNBUFFERED"] = "1"
    os.chdir(plan["cwd"])
    # No shell, backgrounding, extra supervisor or signal interception.
    os.execvpe(plan["argv"][0], plan["argv"], environment)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        root = project_root(args.project_root)
        if args.action == "status":
            print(json.dumps(status_report(args, root), ensure_ascii=False, indent=2))
            return 0
        plan = build_plan(args, root)
        print(json.dumps(plan, ensure_ascii=False, indent=2), file=sys.stderr if plan["execution_requested"] else sys.stdout, flush=True)
        if plan["execution_requested"]:
            execute_plan(plan)
        return 0
    except ImportError as exc:
        print(f"Skill dependency unavailable ({exc.name}). Use your research Python environment; CLI-only dependencies are in requirements-cli.txt.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"ai-scientist: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
