import logging
import json
import pickle
from omegaconf import OmegaConf
from ai_scientist.inference import configure_inference, get_runtime, shutdown_inference
from ai_scientist.inference.errors import InferenceError
from . import backend
from .journal import Journal, Node
from .journal2report import journal2report
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize


logger = logging.getLogger("ai-scientist")


def journal_to_rich_tree(journal: Journal, cfg):
    best_node = journal.get_best_node(cfg=cfg)

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def _write_tree_state(cfg, manager, status, error=None):
    """Durable, inference-free manifest for completed and interrupted work."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "last_stage": manager.current_stage.name if manager and manager.current_stage else None,
        "completed_node_ids": {
            name: [node.id for node in journal.nodes]
            for name, journal in (manager.journals.items() if manager else [])
        },
        "pending_evaluations": sorted(str(path) for path in (log_dir / "partial_nodes").glob("*.json")),
        "worker_completed_artifacts": sorted(str(path) for path in (log_dir / "completed_nodes").glob("*.json")),
        "workspace": str(cfg.workspace_dir),
        "stop_reason": {"type": type(error).__name__, "message": str(error)} if error else None,
    }
    target = log_dir / "tree_run_state.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        temporary.chmod(0o600)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(target)


def perform_experiments_bfts(config_path: str):
    cfg = load_cfg(Path(config_path))
    # The launcher owns the shared runtime. Standalone tree runs may configure
    # one from YAML; workers only ever attach to its exported environment.
    from ai_scientist.inference import is_configured
    owns_runtime = not is_configured()
    if owns_runtime:
        configure_inference(OmegaConf.to_container(cfg.inference, resolve=True), artifact_dir=cfg.log_dir)
    runtime = get_runtime()
    cfg.inference = runtime.export_config()
    state = {"manager": None}
    try:
        runtime.ensure_started()
        _perform_experiments_bfts(cfg, state)
    except BaseException as error:
        manager = state["manager"]
        if manager:
            for name, journal in manager.journals.items():
                try:
                    save_run(cfg, journal, stage_name=f"stage_{name}")
                except Exception:
                    logger.exception("Emergency journal snapshot failed for %s", name)
        status = {
            "QuotaExhausted": "stopped_quota",
            "AuthenticationRequired": "stopped_auth",
            "Cancelled": "cancelled",
            "KeyboardInterrupt": "cancelled",
        }.get(type(error).__name__, "failed_transport" if isinstance(error, InferenceError) else "failed")
        _write_tree_state(cfg, manager, status, error)
        raise
    else:
        _write_tree_state(cfg, state["manager"], "completed")
    finally:
        if owns_runtime:
            shutdown_inference()


def _perform_experiments_bfts(cfg, state):
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    print(task_desc)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    global_step = 0

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    manager = AgentManager(
        task_desc=task_desc,
        cfg=cfg,
        workspace_dir=Path(cfg.workspace_dir),
    )
    state["manager"] = manager
    _write_tree_state(cfg, manager, "running")

    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Running experiments...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def create_exec_callback(status_obj):
        def exec_callback(*args, **kwargs):
            status_obj.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
            status_obj.update("[green]Generating code...")
            return res

        return exec_callback

    def step_callback(stage, journal):
        print("Step complete")
        # Saving must precede summarization/best-node/model-dependent HTML.
        save_run(cfg, journal, stage_name=f"stage_{stage.name}")
        _write_tree_state(cfg, manager, "running")
        try:
            # Generate and save notes for this step
            notes_dir = cfg.log_dir / f"stage_{stage.name}" / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            # Save latest node summary
            if journal.nodes:
                latest_node = journal.nodes[-1]
                if hasattr(latest_node, "_agent"):
                    summary = latest_node._agent._generate_node_summary(latest_node)
                    with open(
                        notes_dir / f"node_{latest_node.id}_summary.json", "w"
                    ) as f:
                        json.dump(summary, f, indent=2)


            if cfg.agent.get("summary", None) is not None:
                current_findings = journal.generate_summary(
                    include_code=False, 
                    **{
                        "model": cfg.agent.summary.model, 
                        "temp": cfg.agent.summary.temp
                    }
                )
            else:
                current_findings = journal.generate_summary(include_code=False)

            best_metric = journal.get_best_node(cfg=cfg)

            # Generate and save stage progress summary
            stage_summary = {
                "stage": stage.name,
                "total_nodes": len(journal.nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "good_nodes": len(journal.good_nodes),
                "best_metric": (
                    str(best_metric.metric)
                    if best_metric
                    else "None"
                ),
                "current_findings": current_findings,
            }

            with open(notes_dir / "stage_progress.json", "w") as f:
                json.dump(stage_summary, f, indent=2)

            # Save the run as before
            save_run(cfg, journal, stage_name=f"stage_{stage.name}", include_analysis=True)

        except InferenceError:
            raise
        except Exception as e:
            print(f"Error in step callback: {e}")

        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")
        print(f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name}")
        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")

    def generate_live(manager):
        current_stage = manager.current_stage
        current_journal = manager.journals.get(
            current_stage.name if current_stage else None, None
        )

        if current_journal:
            tree = journal_to_rich_tree(current_journal, cfg)
        else:
            tree = Tree("[bold blue]No results yet")

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]

        stage_info = [
            "[bold]Experiment Progress:",
            f"Current Stage: [cyan]{current_stage.name if current_stage else 'None'}[/cyan]",
            f"Completed Stages: [green]{', '.join(manager.completed_stages)}[/green]",
        ]

        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            Panel(Text("\n".join(stage_info)), title="Stage Progress"),
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    live = Live(
        generate_live(manager),
        refresh_per_second=16,
        screen=True,
    )

    manager.run(exec_callback=create_exec_callback(status), step_callback=step_callback)

    manager_pickle_path = cfg.log_dir / "manager.pkl"
    try:
        with open(manager_pickle_path, "wb") as f:
            pickle.dump(manager, f)
        logger.info(f"Saved manager state to: {manager_pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save full manager state: {e}")
        try:
            with open(manager_pickle_path, "wb") as f:
                pickle.dump(manager.journals.items(), f)
            logger.info(f"Saved manager journals to: {manager_pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save manager journals: {e}")

    if cfg.generate_report:
        print("Generating final report from all stages...")
        (
            draft_summary,
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items(), cfg)
        draft_summary_path = cfg.log_dir / "draft_summary.json"
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

        with open(draft_summary_path, "w") as draft_file:
            json.dump(draft_summary, draft_file, indent=2)

        with open(baseline_summary_path, "w") as baseline_file:
            json.dump(baseline_summary, baseline_file, indent=2)

        with open(research_summary_path, "w") as research_file:
            json.dump(research_summary, research_file, indent=2)

        with open(ablation_summary_path, "w") as ablation_file:
            json.dump(ablation_summary, ablation_file, indent=2)

        print(f"Summary reports written to files:")
        print(f"- Draft summary: {draft_summary_path}")
        print(f"- Baseline summary: {baseline_summary_path}")
        print(f"- Research summary: {research_summary_path}")
        print(f"- Ablation summary: {ablation_summary_path}")


if __name__ == "__main__":
    cfg_path = "treesearch/utils/config.yaml"
    perform_experiments_bfts(cfg_path)
