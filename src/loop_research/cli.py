from __future__ import annotations

import json
import subprocess
import sys
import time
from itertools import count
from pathlib import Path

import typer

from .engine import Research
from .models import MetricPolicy, Project
from .storage import Conflict

app = typer.Typer(no_args_is_help=True)


@app.command("onboard")
def onboard_project(project: Path, root: Path, objective: str = typer.Option(...),
                    execute: bool = False, interpreter: Path | None = None,
                    artifacts: Path | None = None, focus: str | None = None,
                    research: bool = typer.Option(False, "--research", help="基线通过后接入候选并运行研究。"),
                    structural_files: int = typer.Option(0, min=0, max=6, help="要求首次结构候选修改的最少源文件数。"),
                    metric_policy: Path | None = typer.Option(None, exists=True, dir_okay=False,
                                                              help="多目标优先级和硬门槛的 JSON 文件。"),
                    worker_timeout_seconds: int = typer.Option(240, min=10, max=1800, help="接入角色的单次调用时限。"),
                    retry_adaptation: bool = typer.Option(False, help="在保留原证据的前提下重开已耗尽的适配尝试。"),
                    rounds: int = typer.Option(2, min=1, max=1000),
                    candidates: int = typer.Option(6, min=1, max=1000)):
    """Run project discovery, generate a plan and independently review it."""
    from .onboarding import onboard
    if research and candidates < rounds:
        raise typer.BadParameter("候选预算应为每轮至少保留一项任务。", param_hint="--candidates")
    if structural_files and not research:
        raise typer.BadParameter("结构候选需要启用研究。", param_hint="--structural-files")
    if metric_policy and not research:
        raise typer.BadParameter("指标规则需要启用研究。", param_hint="--metric-policy")
    if retry_adaptation and not (execute or research):
        raise typer.BadParameter("重试适配需要启用执行。", param_hint="--retry-adaptation")
    policy = MetricPolicy.model_validate_json(metric_policy.read_bytes()) if metric_policy else None
    from .workers import CodexWorker
    worker = CodexWorker(root, timeout=worker_timeout_seconds) if worker_timeout_seconds != 240 else None
    if not execute and not research:
        show_json(onboard(project, root, objective, focus=focus, worker=worker))
        return
    if focus is None and (project / "experiment.json").is_file():
        if structural_files:
            raise typer.BadParameter("该选项用于原生项目接入。", param_hint="--structural-files")
        if policy:
            raise typer.BadParameter("使用 experiment.json 内的 metric_policy。", param_hint="--metric-policy")
        from .study_intake import preparation_summary, prepare
        if (root / "state.json").exists():
            engine = Research(root)
            receipt = root / "preparation.json"
            origin = (json.loads(receipt.read_bytes())["source_project"] if receipt.exists()
                      else engine.command.repository if engine.command else None)
            if origin != str(project.resolve()) or engine.project.objective != objective:
                raise typer.BadParameter("运行目录已绑定另一个项目或研究目标。")
            engine.references()
        else:
            result = prepare(project, root, objective=objective)
            if result["status"] != "ready":
                show_json(preparation_summary(root, result))
                raise typer.Exit(3)
            engine = initialize_study(root, read_project(root / "study-spec.json"))
        measured = engine.store.read().baseline_evidence
        show_json({"phase": "baseline_verified", "evidence": measured,
                   "metrics": engine.evidence.get(measured)["content"]["metrics"]})
        if research:
            run(root, interval=2, max_ticks=0)
        return
    from .onboard_native import qualify
    result = qualify(project, root, objective, interpreter=interpreter, artifacts=artifacts, focus=focus,
                     worker=worker, retry_adaptation=retry_adaptation)
    show_json(result)
    if result["phase"] != "baseline_verified":
        raise typer.Exit(3)
    if research:
        from .native_study import connect_native
        admission = connect_native(root.resolve(), rounds=rounds, candidates=candidates,
                                   structural_files=structural_files, metric_policy=policy, worker=worker)
        show_json(admission)
        if admission["phase"] != "study_ready":
            raise typer.Exit(3)
        study_root = Path(admission["study"])
        run(study_root, interval=2, max_ticks=0)
        terminal = Research(study_root).store.read()
        show_json({"phase": terminal.phase, "completed_batches": terminal.completed_batches,
                   "used": terminal.used, "stop_reason": terminal.stop_reason})
        if terminal.phase != "complete":
            raise typer.Exit(3)


@app.command("onboard-replay")
def onboard_historical_replay(root: Path, interpreter: Path = typer.Option(...), revise: bool = False):
    """Ask the workflow to stage and execute the existing historical baseline."""
    from .onboard_replay import propose_and_run
    result = propose_and_run(root, interpreter, revise=revise)
    typer.echo(json.dumps({key: value for key, value in result.items() if key not in {"copied", "sealed_inputs"}},
                          ensure_ascii=False, indent=2))
    if result["phase"] != "replay_executed":
        raise typer.Exit(2)


@app.command("onboard-admission")
def onboard_candidate_admission(root: Path):
    """Independently identify what must be wired before starting new candidates."""
    from .onboard_admission import assess
    typer.echo(json.dumps(assess(root), ensure_ascii=False, indent=2))


@app.command("onboard-connect")
def onboard_connect(root: Path, interpreter: Path = typer.Option(...)):
    """Generate a local bridge, qualify it and dispatch integration controls."""
    from .bridge_launch import connect
    result = connect(root, interpreter)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result["phase"] != "bridge_integration_passed":
        raise typer.Exit(3 if result["phase"] == "bridge_job_pending" else 2)


@app.command("run-stages")
def run_project_stages(plan: Path, receipts: Path):
    """Run a sealed local preparation plan and preserve stage receipts."""
    from .sealed_stage import run_plan
    result = run_plan(plan, receipts)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise typer.Exit(2)


@app.command()
def init(spec: Path, root: Path):
    """Create a research project and measure its baseline using a frozen evaluator."""
    engine = initialize_study(root, read_project(spec))
    typer.echo(engine.store.read().model_dump_json(indent=2))


@app.command("revise-metrics")
def revise_metrics(previous: Path, root: Path, policy: Path = typer.Option(..., exists=True, dir_okay=False),
                   objective: str = typer.Option(...), reason: str = typer.Option(...)):
    """Measure a new baseline under explicitly revised targets in a distinct run directory."""
    original, root = revision_origin(previous, root, reason)
    approved = MetricPolicy.model_validate_json(policy.read_bytes())
    data = {**original.project.model_dump(), "archive_uri": str(root), "objective": objective,
            "metrics": approved.measured_metrics(), "metric_policy": approved}
    if original.command:
        data["command_experiment"] = {**original.command.model_dump(), "revision": original.platform.source["revision"]}
    revised_baseline(original, root, Project.model_validate(data), reason)


@app.command("revise-study")
def revise_study(previous: Path, spec: Path, root: Path, reason: str = typer.Option(...)):
    """Measure a new baseline under a revised experiment specification and retain the old lineage."""
    original, root = revision_origin(previous, root, reason)
    revised_baseline(original, root, read_project(spec), reason)


def revision_origin(previous: Path, root: Path, reason: str) -> tuple[Research, Path]:
    previous, root = previous.resolve(), root.resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):
        raise typer.BadParameter("Use a distinct run directory outside the previous run.", param_hint="root")
    if not reason.strip():
        raise typer.BadParameter("Record the reason for the protocol revision.", param_hint="--reason")
    original = Research(previous)
    original.frozen_protocol()
    return original, root


def revised_baseline(original: Research, root: Path, project: Project, reason: str):
    engine = Research.initialize(root, project, revision_from=(original, reason))
    state = engine.store.read()
    show_json({"phase": "baseline_verified", "protocol": engine.references()["protocol"],
               "baseline": state.baseline_evidence, "metrics": state.best_metrics})


def read_project(spec: Path) -> Project:
    from .study_intake import normalize_project
    return normalize_project(json.loads(spec.read_bytes()), spec.resolve().parent)


def show_json(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command("prepare-study")
def prepare_study(source: Path, root: Path, objective: str | None = None, spec: Path | None = None,
                  answers: Path | None = None, non_interactive: bool = False, refresh: bool = False,
                  start: bool = typer.Option(False, "--start", help="自检和真实基线通过后启动优化。")):
    """发现实验接口、提交待补问题并执行启动前自检。"""
    from .study_intake import ANSWER_FIELDS, FIELD_TYPES, preparation_summary, prepare
    try:
        additions = json.loads(answers.read_bytes()) if answers else {}
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--answers") from exc
    if not isinstance(additions, dict):
        raise typer.BadParameter("请提供以字段名为键的 JSON 对象。", param_hint="--answers")
    try:
        result = prepare(source, root, objective=objective, spec=spec, answers=additions, refresh=refresh)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if result["issues"] and sys.stdin.isatty() and not non_interactive:
        typer.echo("补充已知项。回车可保留待补。命令、指标、路径清单和参数使用 JSON。")
        asked = set()
        while result["issues"]:
            replies = {}
            items = [item for item in result["issues"] if item["field"] not in asked and item["field"] in ANSWER_FIELDS]
            required = [item for item in items if item["field"] not in {"reversible", "affordable", "confirmed_knowledge"}]
            if not required and items:
                display = preparation_summary(root, result)["pending_configuration"]
                show_json(display)
            for item in required or items:
                field = item["field"]
                if field in asked:
                    continue
                asked.add(field)
                if item["detail"]:
                    typer.echo(item["detail"])
                answer = typer.prompt(item["question"], default="", show_default=False).strip()
                if not answer:
                    continue
                try:
                    replies[field] = json.loads(answer) if field in FIELD_TYPES else answer
                except ValueError:
                    typer.echo("此项需要有效的 JSON，已保留待补。")
            if not replies:
                break
            result = prepare(source, root, answers=replies)
    if result["status"] != "ready":
        show_json(preparation_summary(root, result))
        raise typer.Exit(3)
    if start:
        project = read_project(root / "study-spec.json")
        engine = initialize_study(root, project)
        show_json({"status": "baseline_passed", "metrics": engine.store.read().best_metrics})
        run(root, interval=3, max_ticks=100)
    else:
        show_json(preparation_summary(root, result))


@app.command()
def preflight(spec: Path):
    """检查完整配置所需的命令、数据划分及环境依赖。"""
    from .preflight import inspect
    result = inspect(read_project(spec))
    show_json(result)
    if result["issues"]:
        raise typer.Exit(3)


def initialize_study(root: Path, project: Project):
    from .preflight import inspect
    from .study_intake import preparation_summary
    checkpoint = root / "preparation.json"
    if not checkpoint.exists():
        result = inspect(project)
        if result["issues"]:
            show_json(result)
            raise typer.Exit(3)
    try:
        return Research.initialize(root, project)
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        if checkpoint.exists():
            record = json.loads(checkpoint.read_bytes())
            if record["status"] == "awaiting_input":
                show_json(preparation_summary(root, record))
                raise typer.Exit(3) from exc
        show_json({"status": "blocked", "stage": "initialization", "detail": str(exc),
                   "next": "核对接入预检及基线日志，修正配置或运行脚本后，使用同一运行目录重试。",
                   "baseline_logs": str(root / "baseline-logs")})
        raise typer.Exit(3) from exc


@app.command()
def tick(root: Path, event_id: str | None = None):
    """Perform one bounded state-machine step, then exit."""
    try:
        typer.echo(Research(root).tick(event_id).model_dump_json(indent=2))
    except Conflict as exc:
        typer.echo(str(exc))
        raise typer.Exit(2) from exc


@app.command()
def status(root: Path):
    """Read progress without invoking any model or training worker."""
    if not (root / "state.json").exists() and (root / "preparation.json").exists():
        from .study_intake import preparation_summary
        record = json.loads((root / "preparation.json").read_bytes())
        show_json(preparation_summary(root, record))
        return
    if not (root / "state.json").exists():
        for name in ("native-onboarding.json", "onboarding.json"):
            if (path := root / name).exists():
                show_json(json.loads(path.read_bytes()))
                return
    typer.echo(Research(root).store.read().model_dump_json(indent=2))


@app.command()
def budget(root: Path):
    """Count durable local job requests, replay audits and observed model usage."""
    engine = Research(root)
    state = engine.store.read()
    training = {"baseline": 1, "validation_requests": 0, "holdout_requests": 0,
                "review_replay_requests": 0, "retry_requests": 0}
    observed, reproductions, worker_issues = {}, 0, {}
    for path in engine.evidence.root.glob("*.json"):
        record = engine.evidence.get(path.stem)
        content = record["content"]
        if record["kind"] == "training_result" and content.get("request_sha256"):
            observed[content["request_sha256"]] = content["status"]
        elif record["kind"] == "reproduction":
            reproductions += 1
        elif record["kind"] == "worker_execution_issue":
            issue = content["type"]
            worker_issues[issue] = worker_issues.get(issue, 0) + 1
    results = {}
    for path in (engine.root / "jobs").glob("*/request.json"):
        request = json.loads(path.read_bytes())
        task = request["task_id"]
        if task.startswith("independent-review:"):
            training["review_replay_requests"] += 1
        else:
            split = "holdout_requests" if request.get("split") == "test" else "validation_requests"
            training[split] += 1
        if ":attempt:" in task and int(task.rsplit(":", 1)[-1]) > 0:
            training["retry_requests"] += 1
        result = path.with_name("result.json")
        status_name = (json.loads(result.read_bytes())["status"] if result.exists()
                       else observed.get(path.parent.name, "no_durable_result"))
        results[status_name] = results.get(status_name, 0) + 1
    calls, incomplete, usage = 0, 0, {}
    for path in (engine.root / "workers").glob("*/input.json"):
        calls += 1
        metadata = path.with_name("metadata.json")
        reported = json.loads(metadata.read_bytes()).get("usage") if metadata.exists() else None
        if reported is None:
            incomplete += 1
        for key, amount in (reported or {}).items():
            if type(amount) is int:
                usage[key] = usage.get(key, 0) + amount
    typer.echo(json.dumps({"candidate_budget": {"used": state.used, "limit": engine.project.max_experiments},
                           "training": training, "training_result_statuses": results, "replay_audits": reproductions,
                           "worker_requests": calls, "worker_issues": worker_issues, "requests_without_usage": incomplete,
                           "observed_usage": usage, "billing_total": None}, indent=2))


@app.command("resolve-review")
def resolve_review(root: Path, review_id: str, reviewer: str, rationale: str, accept: bool = False):
    """Resolve the current independent-review disagreement by exact evidence ID."""
    result = Research(root).resolve_review(review_id, reviewer, rationale, accept=accept)
    typer.echo(result.model_dump_json(indent=2))


@app.command("recheck-review")
def recheck_review(root: Path, review_id: str):
    """Reassess against newly available evidence without resolving a disagreement by fiat."""
    result = Research(root).recheck_review(review_id)
    typer.echo(result.model_dump_json(indent=2))


@app.command("record-refutation")
def record_refutation(root: Path, evidence_id: str, reviewer: str, rationale: str):
    """Block an exact candidate with reviewed negative evidence under its frozen protocol."""
    typer.echo(Research(root).record_refutation(evidence_id, reviewer, rationale))


@app.command("propose-direction-refutation")
def propose_direction_refutation(root: Path, measurements: list[str]):
    """Independently assess matched negative results and retain a proposal for human review."""
    from .refutations import propose
    show_json(propose(Research(root), measurements))


@app.command("decide-direction-refutation")
def decide_direction_refutation(root: Path, proposal: str, reviewer: str, rationale: str, decision: str):
    """Record an explicit approval, rejection or deferral by proposal ID."""
    from .refutations import decide
    if decision not in {"approve", "reject", "defer"}:
        raise typer.BadParameter("Choose approve, reject or defer.", param_hint="decision")
    show_json(decide(Research(root), proposal, reviewer, rationale, decision))


@app.command()
def directions(root: Path):
    """Read the screened direction pool and provenance without waking an Agent."""
    engine = Research(root)
    state = engine.store.read()
    if not state.direction_pool:
        typer.echo(json.dumps({"phase": state.phase, "research_stage": state.research_stage, "directions": []}))
        return
    engine.evidence.verify_tree(state.direction_pool)
    typer.echo(json.dumps({"evidence": state.direction_pool, **engine.evidence.get(state.direction_pool)["content"]},
                          ensure_ascii=False, indent=2))


@app.command()
def run(root: Path, interval: float = 3, max_ticks: int = 100):
    """Drive ticks. Set --max-ticks 0 to follow the study to a terminal phase."""
    if max_ticks < 0:
        raise typer.BadParameter("Use a nonnegative tick limit.", param_hint="--max-ticks")
    ticks = range(max_ticks) if max_ticks else count()
    for _ in ticks:
        engine = Research(root)
        state = engine.tick()
        typer.echo(f"revision={state.revision} phase={state.phase} batch={state.batch} used={state.used}")
        if state.phase in {"stopped", "complete", "review"}:
            engine.report()
            return
        time.sleep(interval)
    raise typer.Exit(3)


@app.command()
def resume(root: Path, interval: float = typer.Option(3, min=0), max_ticks: int = typer.Option(100, min=0),
           python: Path = typer.Option(Path(sys.executable))):
    """Continue an existing study with its checksum-verified archived controller."""
    from .reproducibility import resume_archived

    try:
        status = resume_archived(root, python=python, interval=interval, max_ticks=max_ticks)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise typer.BadParameter(str(exc), param_hint="root") from exc
    if status:
        raise typer.Exit(status)


@app.command()
def study(spec: Path, root: Path, interval: float = 3, max_ticks: int = 100):
    """Initialize and run a new local study, or resume the same frozen specification."""
    project = read_project(spec)
    if (root / "state.json").exists():
        engine = Research(root)
        engine.references()
        if engine.project != project:
            raise typer.BadParameter("This run root already belongs to another frozen specification")
    else:
        engine = initialize_study(root, project)
        typer.echo(json.dumps({"baseline": engine.store.read().best_metrics}, ensure_ascii=False))
    run(root, interval=interval, max_ticks=max_ticks)


@app.command()
def serve(root: Path, interval_seconds: int = typer.Option(60, min=1),
          max_ticks: int = typer.Option(0, min=0)):
    """Use one scheduler, with state leases guarding overlapping external triggers."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler(timezone="UTC")
    completed = 0

    def scheduled_tick():
        nonlocal completed
        try:
            state = Research(root).tick()
        except Conflict as exc:
            typer.echo(f"Skipped a concurrent tick: {exc}", err=True)
            return
        completed += 1
        typer.echo(f"revision={state.revision} phase={state.phase} batch={state.batch} used={state.used}")
        if state.phase in {"stopped", "complete", "review"} or completed == max_ticks:
            scheduler.shutdown(wait=False)

    scheduler.add_job(scheduled_tick, "interval", seconds=interval_seconds,
                      max_instances=1, coalesce=True, id="research-tick")
    scheduler.start()


@app.command()
def stop(root: Path, cancel_training: bool = False):
    typer.echo(Research(root).stop(cancel_training).model_dump_json(indent=2))


@app.command()
def reconcile(root: Path):
    """Reconcile receipts and completed training without invoking any model."""
    typer.echo(Research(root).reconcile().model_dump_json(indent=2))


@app.command()
def report(root: Path):
    typer.echo(str(Research(root).report()))


@app.command()
def reproduce(root: Path, evidence_id: str):
    from .replay import reproduce as replay
    result = replay(Research(root), evidence_id)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise typer.Exit(4)


@app.command("approve-memory")
def approve_memory(root: Path, proposal: str, reviewer: str, expected_hash: str, rationale: str):
    """Human action: approve a previously reviewed concrete candidate version."""
    engine = Research(root)
    engine.memory.approve(proposal, reviewer=reviewer, expected_hash=expected_hash, explanation=rationale)
    typer.echo(json.dumps({"approved": proposal}))


@app.command("memory-list")
def memory_list(root: Path):
    engine = Research(root)
    typer.echo(json.dumps(engine.memory.timeline(engine.store.read().project_id), ensure_ascii=False, indent=2))


@app.command("screen-memory")
def screen_memory(root: Path, proposal: str):
    typer.echo(json.dumps(Research(root).memory.screen(proposal), ensure_ascii=False, indent=2))


@app.command("review-memory")
def review_memory(root: Path, proposal: str):
    """Check the scope and logic of an evidence-backed draft with an independent worker."""
    typer.echo(json.dumps(Research(root).review_memory(proposal), ensure_ascii=False, indent=2))


@app.command("revise-memory")
def revise_memory(root: Path, previous: str, instruction: str,
                  evidence_id: list[str] = typer.Option(None, "--evidence-id")):
    """Draft and independently review a successor to an active knowledge record."""
    result = Research(root).revise_memory(previous, instruction, evidence_id or [])
    show_json(result)


@app.command("search-memory")
def search_memory(root: Path, query: str):
    engine = Research(root)
    typer.echo(json.dumps(engine.memory.search(query, engine.store.read().project_id), ensure_ascii=False, indent=2))


@app.command("revoke-memory")
def revoke_memory(root: Path, proposal: str, reason: str):
    Research(root).memory.revoke(proposal, reason)
    typer.echo(json.dumps({"revoked": proposal}))


@app.command("rollback-memory")
def rollback_memory(root: Path, proposal: str, reviewer: str, rationale: str):
    Research(root).memory.rollback(proposal, reviewer=reviewer, explanation=rationale)
    typer.echo(json.dumps({"restored": proposal}))


if __name__ == "__main__":
    app()
