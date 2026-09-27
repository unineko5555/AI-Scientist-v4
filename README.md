# AI-Scientist-v4

A fork that runs the AI-Scientist V2 research pipeline through subscription-authenticated Codex CLI or Claude Code CLI. It includes a shared `ai-scientist` skill for both CLIs.

## Features Inherited from V2

This repository is based on [Sakana AI's AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2). It preserves V2's Python research engine and research stages while adding CLI-based LLM/VLM inference.

- Research idea generation and novelty assessment from a topic description
- Agentic experiment tree search across four stages: baseline, hyperparameter tuning, research improvement, and ablation
- Multi-seed evaluation, result analysis, plot generation, visual assessment, and plot aggregation
- Literature search, citations, paper writing, PDF generation, and text and image review
- Both `normal` and `icbinb` writeup formats, with saved experiment logs and artifacts

The skill is an interface for operating this pipeline, not a replacement research engine. This fork retains the upstream [LICENSE](LICENSE). Review its AI-use disclosure requirements before publishing generated work.

## Subscription Codex/Claude CLI Instead of Direct API Calls

In CLI mode, all application-managed text, code, and image inference goes through the official CLIs using ChatGPT or Claude subscription authentication. No LLM API keys are required. Authentication failures or usage limits stop the run; they do not trigger an automatic switch to an API or another CLI.

| Inference provider | Skill/helper option | Configuration file |
| --- | --- | --- |
| Codex only | `--provider codex` | [inference.codex.yaml](configs/inference.codex.yaml) |
| Claude Code only | `--provider claude` | [inference.claude.yaml](configs/inference.claude.yaml) |
| Both, assigned by role | `--provider mixed` | [inference.mixed.yaml](configs/inference.mixed.yaml) |

Using a single provider does not require the other CLI or subscription. `mixed` requires both; the included profile assigns writing and review to Claude and all other roles to Codex. `model: null` uses the selected CLI's default model.

The CLI you use to operate the skill and the CLI used for research inference are independent choices. For example, a Codex conversation can launch a Claude-backed research run.

"No API required" means you do not need API keys or billing setup for the application to call LLM APIs directly. The official CLIs still contact their services, and literature searches through Semantic Scholar, dataset downloads, and other external network activity remain. Subscription limits apply; unlimited usage or the absence of additional usage charges is not guaranteed.

> For compatibility, the original Python entrypoints still default to API mode when CLI configuration is omitted. This README uses the shared helper, which requires an explicit CLI profile. When invoking an entrypoint directly, always specify a profile such as `--inference-config configs/inference.codex.yaml`.

## Skill Setup

<a id="installation"></a>

### 1. Prepare the Repository and Python Environment

The runner targets macOS and Linux. The following example uses Python 3.11. If you have already cloned the repository, work from that checkout's root directory.

```bash
git clone https://github.com/unineko5555/AI-Scientist-v4.git
cd AI-Scientist-v4

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cli.txt
```

These dependencies support inference-layer diagnostics and execution previews. To generate ideas or run research, install the research dependencies in the same Python environment:

```bash
python -m pip install -r requirements.txt
```

Experiments also require a [PyTorch](https://pytorch.org/get-started/locally/) installation suitable for your OS and compute environment. For a Mac, use the example below. For CUDA on Linux, select the appropriate installation command on the official PyTorch page.

```bash
# Example PyTorch installation on macOS
python -m pip install torch torchvision
```

Paper generation requires a LaTeX distribution such as TeX Live or MacTeX, with `pdflatex`, `bibtex`, `chktex`, and Poppler's `pdftotext` available. `requirements-cli.txt` alone is not enough for the full research pipeline. `requirements.txt` also includes SDKs for legacy API compatibility, but CLI mode does not use them for inference.

A MacBook Air M3 can use the existing CPU execution path; CUDA is not required. This fork does not add MPS support, so do not assume experiments will use the Apple GPU. Memory and runtime requirements depend on the research workload, and a complete research run on M3 has not been verified.

### 2. Install a CLI and Sign In with Your Subscription

Follow the official installation instructions for [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) or [Claude Code](https://code.claude.com/docs/en/quickstart). If already installed, check authentication for the CLI you intend to use.

For Codex, sign in with your ChatGPT account. See the [Codex authentication guide](https://learn.chatgpt.com/docs/auth) for details.

```bash
codex login
codex login status
```

For Claude Code, sign in with your Claude subscription account. Do not add `--console`, which selects API billing. See the [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).

```bash
claude auth login
claude auth status
```

### 3. Open the CLI in This Repository

The skill is included in the repository. No global copy, plugin, or additional MCP server is required.

- Codex loads `.agents/skills/ai-scientist/SKILL.md`.
- Claude Code loads `.claude/skills/ai-scientist/SKILL.md`, a relative symlink to the same skill.

Preserve symlinks when cloning the repository. This layout follows the [Codex skills](https://learn.chatgpt.com/docs/build-skills) and [Claude Code skills](https://code.claude.com/docs/en/skills) discovery conventions.

With your Python environment activated, start your chosen CLI from the repository root:

```bash
# For Codex
codex
```

```bash
# For Claude Code
claude
```

Enter the following `$ai-scientist` or `/ai-scientist` prompts in the CLI conversation, not in your shell.

Codex example:

```text
$ai-scientist Check the environment for Codex. Do not start research or live inference probes yet.
```

Claude Code example:

```text
/ai-scientist Check the environment for Claude. Do not start research or live inference probes yet.
```

If the skill does not appear, start a new CLI session in this repository.

## Using This Repository

The workflow is: diagnose the environment, prepare a topic, generate ideas, select an idea, run experiments and paper generation, then inspect the results. Research runs execute generated code, so use a dedicated, isolated environment. The skill itself does not provide OS-level isolation.

### Use the Skill Conversationally

Prefix your request with `$ai-scientist` in Codex or `/ai-scientist` in Claude Code. The examples below use Codex for both interaction and inference. To use Claude for both, change the prefix and explicitly request Claude inference.

```text
$ai-scientist Prepare a research topic in ai_scientist/ideas/my_topic.md for a Codex-backed run.
Define the hypothesis, data, and evaluation method for small experiments on a MacBook Air M3 CPU.
Do not generate ideas or run experiments yet.
```

After reviewing the topic, request idea generation:

```text
$ai-scientist Generate one research idea from ai_scientist/ideas/my_topic.md using Codex.
```

Review the generated JSON and select the idea you want to research:

```text
$ai-scientist I want to research idea index 0 from ai_scientist/ideas/my_topic.json using Codex.
Show the execution plan and compute settings, including all V2 stages. Do not execute it yet.
```

After confirming the selected idea, scope, and compute settings, ask: "Start the research using the plan we reviewed." Requests to diagnose or prepare a topic do not authorize a research run. Keep the session open while the run is executing.

### Run the Same Operations from a Terminal

Run these commands from the repository root with your Python environment activated. To use Claude, replace `--provider codex` with `--provider claude`.

First, run the non-inference CLI diagnostic. It checks the CLI version, capabilities, and authentication, but does not verify research dependencies or the complete pipeline.

```bash
python .agents/skills/ai-scientist/scripts/operate.py doctor --provider codex
```

Next, create `ai_scientist/ideas/my_topic.md` with `Title`, `Keywords`, `TL;DR`, and `Abstract` sections. Use the [existing topic example](ai_scientist/ideas/i_cant_believe_its_not_better.md) as a reference, and include your hypothesis, data, evaluation method, and compute constraints.

```bash
# Preview idea generation only
python .agents/skills/ai-scientist/scripts/operate.py ideate \
  --provider codex \
  --workshop-file ai_scientist/ideas/my_topic.md \
  --max-num-generations 1

# Execute after reviewing the plan; this consumes subscription usage
python .agents/skills/ai-scientist/scripts/operate.py ideate \
  --provider codex \
  --workshop-file ai_scientist/ideas/my_topic.md \
  --max-num-generations 1 \
  --execute
```

Ideas are saved as a JSON array in `my_topic.json` next to the topic file. If that JSON file already exists, the helper stops instead of proceeding. Use a new topic filename or follow the [instructions for existing idea archives](.agents/skills/ai-scientist/references/operations.md).

Review the JSON and specify the idea to run. `--idea-index` is zero-based; the example below selects the first idea.

```bash
# Preview a run with all V2 stages
python .agents/skills/ai-scientist/scripts/operate.py run \
  --provider codex \
  --ideas ai_scientist/ideas/my_topic.json \
  --idea-index 0

# Start experiments, paper generation, and review after checking the plan
python .agents/skills/ai-scientist/scripts/operate.py run \
  --provider codex \
  --ideas ai_scientist/ideas/my_topic.json \
  --idea-index 0 \
  --execute
```

`ideate` and `run` only display a plan unless `--execute` is supplied. The default writeup format is `icbinb`; use `--writeup-type normal` for the other format. Experiment workers, per-stage iteration limits, and seeds are configured in [bfts_config.yaml](bfts_config.yaml). The defaults are 4 workers and 3 seeds; they are not automatically reduced for M3. Preview the plan again after changing settings.

### Inspect Results

```bash
# Inspect saved status for recent research runs
python .agents/skills/ai-scientist/scripts/operate.py status

# Inspect one run; replace the path with the one printed during execution
python .agents/skills/ai-scientist/scripts/operate.py status \
  --run-dir "experiments/<timestamp>_<Name>_attempt_0"
```

Each research run is stored under `experiments/<timestamp>_<Name>_attempt_<id>/`.

- `run_status.json`: the saved research status and stage
- `logs/0-run/`: experiment logs and tree-search records, including `unified_tree_viz.html` when generated
- Paper PDFs, LaTeX, figures, and review results: saved when their respective stages produce them

`status` only reads saved records. It does not check whether a process is still running or resume an interrupted run. If authentication or usage limits stop a run, investigate the cause instead of automatically rerunning it. Logs can contain private research content and images; do not publish or commit them without review.

Ordinary `doctor` and `status` operations do not send model inference requests. The Codex or Claude conversation used to request those operations can still consume subscription usage. For a small live inference test, preview it with `doctor --provider codex --probe`, then add `--execute` only when you intend to run it.

Successful CLI diagnostics or small probes do not establish full-pipeline completion or paper quality. See the [CLI operations guide](docs/cli-inference.md) for detailed settings, limitations, and recovery procedures.
