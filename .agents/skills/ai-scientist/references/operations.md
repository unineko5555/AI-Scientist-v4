# Operating the V2 runner

These examples assume the checkout root and an activated research Python
environment. Use an absolute helper path when outside the checkout. `--help`
and `status` require only Python's standard library. Planning needs the lightweight
inference dependencies; research still needs the original research dependencies.

```bash
# No inference: inspect the selected CLI's version/capabilities/login.
python .agents/skills/ai-scientist/scripts/operate.py doctor --provider codex
python .agents/skills/ai-scientist/scripts/operate.py doctor --provider claude

# No subprocess or artifact writes: inspect the command/configuration only.
python .agents/skills/ai-scientist/scripts/operate.py doctor --provider codex --dry-run

# Plan ideation; no model requests or writes to topic.json yet.
python .agents/skills/ai-scientist/scripts/operate.py ideate \
  --provider codex --workshop-file /path/to/topic.md

# Only after the user requests/authorizes this live ideation operation.
python .agents/skills/ai-scientist/scripts/operate.py ideate \
  --provider codex --workshop-file /path/to/topic.md --execute

# Plan all V2 research stages; default writeup is icbinb.
python .agents/skills/ai-scientist/scripts/operate.py run \
  --provider claude --ideas /path/to/topic.json --idea-index 0

# After authorization, add --execute to that same command.
python .agents/skills/ai-scientist/scripts/operate.py status
python .agents/skills/ai-scientist/scripts/operate.py status --run-dir /path/to/experiment
```

## Provider and resource choices

`--provider` selects the checked-in `configs/inference.<provider>.yaml` profile.
`mixed` explicitly selects the existing role-based Codex/Claude profile and needs
both subscriptions. `--inference-config /path/to/profile.yaml` is an alternative,
not an additional override; the helper rejects API mode, API/CLI mixed roles and
`subscription_only: false`. `--model` changes only the default CLI model, not
explicit role models. Prefer a custom profile for role-specific settings.

`run` supports `--writeup-type normal|icbinb`, `--attempt-id`,
`--writeup-retries`, `--num-cite-rounds`, `--load-code`, `--add-dataset-ref`,
`--skip-writeup`, and `--skip-review`. Skipping is never automatic. Tree workers,
stage limits, seeds and execution deadlines still come from `bfts_config.yaml`.
The helper previews them and does not edit that file. Inference concurrency and
timeouts are different from experimental concurrency and a total usage budget.
`--skip-writeup` also skips citations and review, matching the existing launcher;
the preview lists these effective skips. `--skip-review` alone keeps writing.

`ideate` supports `--max-num-generations` and `--num-reflections`, preserving the
entrypoint defaults (1 and 5). If an adjacent idea JSON already exists, stop and
ask whether to use another topic filename or explicitly extend/rewrite the
archive with `--allow-existing-ideas`. This is not checkpoint resume.

`doctor --probe` prints the planned application-request count, which is three per
distinct backend/model by default. `--probe-check` selects checks. Neither sends
requests without `--execute`. CLI-internal traffic and actual account charges are
not determined by that count. A diagnostic pass is not a V2 end-to-end pass.

## Dependencies and execution environment

Use [the research installation guide](../../../../README.md#installation) and
[CLI inference guide](../../../../docs/cli-inference.md) when setup is requested.
Installing packages, changing login or purchasing usage needs the user's relevant
authorization. Do not read token files or print environment secrets.

`requirements-cli.txt` suffices for the inference layer/doctor, not full research.
`requirements.txt` plus an appropriate PyTorch installation and PDF/LaTeX tools
remain necessary for research. Do not copy the CUDA installation command onto an
Apple Silicon Mac. Missing CUDA is not an error on the CPU path. No MPS or worker
count change is made by this skill.

The parent agent invokes the Python runner; the runner invokes isolated,
stateless, tool-free CLI inference children. They must not reload this skill and
recursively start a research run. The transport removes inherited parent-session
markers only in those children, and keeps its existing safe-mode/config/tool
restrictions. Do not weaken those restrictions to make a probe pass.
Codex's child command explicitly disables this checkout's operator `SKILL.md`
through a per-process `skills.config` override; Claude's safe mode disables skill
loading. The outer interactive session still discovers the skill normally.

The generated experimental code is executable code with filesystem/network
access, not an OS sandbox. Use a controlled research environment. The skill does
not create containers, remote machines or a new daemon.

## Artifacts and interrupted runs

- Ideation writes JSON next to the selected topic. CLI inference logs default to
  the checkout's ignored `.inference/`, not alongside a potentially tracked topic.
- The research launcher chooses `experiments/<timestamp>_<Name>_attempt_<id>/`.
  Observe the exact path in its output. The helper does not relocate or overwrite
  the run to implement custom output paths.
- `run_status.json` records the research status/stage. The inference
  `manifest.json` is not a substitute. `status` does not inspect authentication,
  invoke a CLI, infer process liveness, or claim automatic resume.
- Journals/workspaces/partial nodes remain the recovery evidence. Do not relaunch
  automatically after a lost session, quota stop or timeout: a new run consumes
  more usage and does not resume the old experiment tree.
- Logs can contain private prompts, images and papers. Do not publish or commit
  them. For an explicit `--inference-log-dir`, arrange appropriate Git exclusion
  with the user; only the default locations are already ignored.

## Preparing a fresh run after a stop

First preserve the previous run and check its stop reason. Use the original ideas
array and selected index if available. If only `<old-run>/idea.json` survives,
that file contains **one object**, not the array accepted by `--ideas`: read it and
create a new, separately named JSON file containing `[<the unchanged object>]`.
Use your file-editing tool; do not modify the old run or overwrite an existing
file. Preview `run --provider <chosen-provider> --ideas /path/restart-ideas.json
--idea-index 0` before any authorized execution.

Compare the old run's saved `bfts_config.yaml` and inference configuration with the
current choices. Resolve differences with the user instead of silently adopting
changed defaults. `--attempt-id` labels a new attempt; it does not load checkpoints.
A new run repeats the research and consumes usage; it does not continue the saved
tree or bypass an unresolved quota/authentication problem.
