from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Metric(StrictModel):
    name: str = Field(min_length=1)
    direction: Literal["max", "min"] = "max"
    threshold: float = Field(allow_inf_nan=False)

    def passes(self, value: float) -> bool:
        return math.isfinite(value) and (value >= self.threshold if self.direction == "max" else value <= self.threshold)


class MetricPolicy(StrictModel):
    targets: list[Metric] = Field(min_length=1, max_length=16)
    hard_limits: list[Metric] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_targets(self):
        if any(len({metric.name for metric in group}) != len(group) for group in (self.targets, self.hard_limits)):
            raise ValueError("Targets and hard limits must each name distinct metrics")
        limits = {metric.name: metric for metric in self.hard_limits}
        for target in self.targets:
            limit = limits.get(target.name)
            if limit and (limit.direction != target.direction or not limit.passes(target.threshold)):
                raise ValueError(f"The target for {target.name} must satisfy its hard limit")
        if len(self.measured_metrics()) > 16:
            raise ValueError("At most 16 frozen metrics can be used in a decision policy")
        return self

    def measured_metrics(self) -> list[Metric]:
        targets = {metric.name: metric for metric in self.targets}
        return [*targets.values(), *(metric for metric in self.hard_limits if metric.name not in targets)]

    def failed_limits(self, values) -> list[str]:
        return [metric.name for metric in self.hard_limits if not metric.passes(values[metric.name])]

    def failed_targets(self, values) -> list[str]:
        return [metric.name for metric in self.targets if not metric.passes(values[metric.name])]

    def rank(self, values) -> tuple[float, ...]:
        return tuple(values[metric.name] * (1 if metric.direction == "max" else -1) for metric in self.targets)


ParameterValue = bool | int | float | str


class EnvironmentRequirements(StrictModel):
    committed_files: list[str] = Field(default_factory=list)
    local_paths: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    programs: list[str] = Field(default_factory=list)
    gpu_count: int = Field(default=0, ge=0)


class CommandExperiment(StrictModel):
    repository: str
    revision: str = "HEAD"
    command: list[str] = Field(min_length=1)
    split_manifest: str
    baseline: dict[str, ParameterValue]
    choices: dict[str, list[ParameterValue]]
    timeout_seconds: int = Field(default=120, ge=1, le=86400)
    bridge_interpreter: str | None = None
    mutable_sources: list[str] = Field(default_factory=list, max_length=6)
    min_structural_files: int = Field(default=0, ge=0, le=6)

    @model_validator(mode="after")
    def validate_choices(self):
        if (not self.choices and not self.mutable_sources) or self.choices.keys() != self.baseline.keys():
            raise ValueError("Parameter names must match between baseline and choices")
        if self.min_structural_files > len(self.mutable_sources):
            raise ValueError("The required structural change exceeds the admitted mutable scope")
        if self.mutable_sources:
            paths = [PurePosixPath(name) for name in self.mutable_sources]
            if (self.command != ["isolated-native"] or len(set(paths)) != len(paths)
                    or any(path.is_absolute() or path.as_posix() != name or path.suffix != ".py"
                           or any(part.startswith(".") for part in path.parts)
                           for name, path in zip(self.mutable_sources, paths, strict=True))):
                raise ValueError("Only native studies may declare distinct relative Python source paths")
        for name, base in self.baseline.items():
            options = self.choices[name]
            if not name.isidentifier() or not options or not any(type(x) is type(base) and x == base for x in options):
                raise ValueError(f"Invalid baseline or choices for {name}")
            if any(type(x) is not type(base) or (type(x) is float and not math.isfinite(x)) for x in options):
                raise ValueError(f"Mixed types or nonfinite choices for {name}")
        if any(not x or "\0" in x for x in self.command):
            raise ValueError("Command arguments must be nonempty strings")
        if self.command[0] == "isolated-fold":
            if (not self.bridge_interpreter or len(self.command) != 3 or not self.command[1].startswith("/")
                    or len(self.command[2]) != 64 or any(c not in "0123456789abcdef" for c in self.command[2])):
                raise ValueError("An isolated fold experiment requires an interpreter, asset directory and projection SHA")
        elif self.command[0] in {"isolated-predict", "isolated-native"}:
            if len(self.command) != 1 or not (self.bridge_interpreter or "").startswith("/"):
                raise ValueError("Isolated execution requires an absolute Python interpreter and no extra arguments")
        elif self.bridge_interpreter and self.command != ["isolated-bridge"]:
            raise ValueError("An isolated bridge must use the controlled bridge entrypoint")
        return self


class Project(StrictModel):
    name: str = Field(min_length=1)
    objective: str = Field(min_length=10)
    knowledge: str = Field(min_length=20)
    archive_uri: str = Field(min_length=1)
    metrics: list[Metric] = Field(min_length=1)
    metric_policy: MetricPolicy | None = None
    max_experiments: int = Field(ge=1, le=1000)
    max_batches: int | None = Field(default=None, ge=1, le=1000)
    branches: int = Field(default=3, ge=1, le=4)
    concurrency: int = Field(default=3, ge=1, le=16)
    seed: int = 42
    stagnation_batches: int = Field(default=2, ge=1)
    infrastructure_retries: int = Field(default=2, ge=0, le=10)
    worker_timeout_seconds: int = Field(default=240, ge=10)
    task_lease_seconds: int = Field(default=1800, ge=60)
    reversible: bool
    affordable: bool
    confirmed_knowledge: bool
    dataset: Literal["breast_cancer"] = "breast_cancer"
    model: str | None = None
    command_experiment: CommandExperiment | None = None
    environment: EnvironmentRequirements | None = None
    independent_review: bool = False
    review_model: str | None = None
    followup_experiments: bool = False

    @model_serializer(mode="wrap")
    def serialize_project(self, handler):
        data = handler(self)
        if self.metric_policy is None:
            data.pop("metric_policy", None)
        return data

    @model_validator(mode="after")
    def validate_scope(self):
        if not (self.reversible and self.affordable and self.confirmed_knowledge):
            raise ValueError("Quantifiable, reversible, affordable experiments and confirmed knowledge required")
        if self.branches > self.max_experiments:
            raise ValueError("Budget cannot be smaller than initial batch")
        if self.command_experiment:
            config = self.command_experiment
            targets = sum(len(set(v)) > 1 for v in config.choices.values()) + bool(config.mutable_sources)
            if targets < self.branches:
                raise ValueError("Command experiment needs one distinct parameter or structural target per parallel branch")
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("Duplicate metric names")
        if self.metric_policy and self.metrics != self.metric_policy.measured_metrics():
            raise ValueError("Project metrics must match the ordered targets and hard limits in metric_policy")
        return self


class Hypothesis(StrictModel):
    title: str
    question: str
    axis: str
    change: str
    prediction: str
    falsification: str
    source_ids: list[str] = Field(min_length=1)
    parameter_bounds: str
    # Multiple axes require a separately accepted protocol, not an unannounced edit.
    single_variable: bool
    machine_target: str | None = None
    direction_id: str | None = None
    classifier_name: str | None = Field(default=None, description="Exact implementation copied from the selected direction for classifier.family; null for all other targets.")


class Plan(StrictModel):
    rationale: str
    hypotheses: list[Hypothesis] = Field(max_length=4)
    research_questions: list[str] = Field(default_factory=list, max_length=4)


class HypothesisOverlap(StrictModel):
    machine_targets: list[str] = Field(min_length=2, max_length=2)
    reason: str = Field(min_length=10, max_length=1500)


class OrthogonalityReview(StrictModel):
    overlaps: list[HypothesisOverlap] = Field(max_length=6)


class MetricReplan(StrictModel):
    evidence_ids: list[str] = Field(min_length=2, max_length=5)
    protected_metrics: list[str] = Field(min_length=1, max_length=16)
    relationship: str = Field(min_length=20, max_length=4000)
    next_check: str = Field(min_length=20, max_length=4000)


class TradeoffPlan(Plan):
    tradeoff: MetricReplan


class ResearchQuery(StrictModel):
    query: str = Field(min_length=4, max_length=180)
    gap: str = Field(min_length=10, max_length=1000)


class ResearchRequest(StrictModel):
    rationale: str
    criteria: list[str] = Field(min_length=2, max_length=4)
    queries: list[ResearchQuery] = Field(min_length=2, max_length=4)


class PaperQuote(StrictModel):
    source_id: str
    field: Literal["abstract", "body"]
    quote: str = Field(min_length=16, max_length=160)


class DirectionProposal(StrictModel):
    title: str = Field(min_length=5, max_length=250)
    machine_target: str
    hypothesis: str = Field(min_length=20, max_length=1500)
    rationale: str = Field(min_length=20, max_length=2000)
    parameter_bounds: str = Field(min_length=5, max_length=1000)
    quotes: list[PaperQuote] = Field(min_length=1, max_length=4)
    priority: int = Field(ge=0, le=100, description="Scheduling priority: 0-100, higher values rank earlier. This is not statistical confidence or predicted performance.")
    limitations: list[str] = Field(min_length=1, max_length=5)
    classifier_name: str | None = Field(default=None, description="Exact available classifier for classifier.family; null for all other targets.")


class SourceRejection(StrictModel):
    source_id: str
    reason: str


class LiteratureScreen(StrictModel):
    summary: str
    directions: list[DirectionProposal] = Field(max_length=12)
    excluded_sources: list[SourceRejection]
    unresolved: list[str]


class CodeResult(StrictModel):
    rationale: str
    code: str
    source_ids: list[str] = Field(min_length=1)
    changed_axis: str


class StructuralReview(StrictModel):
    accepted: bool
    rationale: str = Field(min_length=10)
    findings: list[str]


class Analysis(StrictModel):
    summary: str
    evidence_ids: list[str]
    explanation: str
    confidence: float = Field(ge=0, le=1)
    next_directions: list[str]
    limitations: list[str]
    # The controller, not this field, determines whether thresholds were reached.
    metric_tradeoffs: str
    research_questions: list[str] = Field(default_factory=list, max_length=4)


class BatchAccounting(StrictModel):
    candidate_iterations: int = Field(ge=0)
    successful_measurements: int = Field(ge=0)
    code_failures: int = Field(ge=0)


class FailureBatchAnalysis(Analysis):
    accounting: BatchAccounting


class IndependentReview(StrictModel):
    selected_evidence: str | None
    evidence_ids: list[str]
    rationale: str
    concerns: list[str]
    requires_human: bool = Field(default=False, description="True only if a human must resolve a material problem before the candidate can advance to independent replay. Ordinary statistical limitations belong in concerns.")


class MemoryDraft(StrictModel):
    title: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=20, max_length=16000)
    evidence_ids: list[str] = Field(min_length=1)
    applicability: str = Field(min_length=10)
    kind: Literal["experience", "skill", "refutation", "timeline"]


class MemoryLogicAssessment(StrictModel):
    verdict: Literal["grounded", "inconclusive", "unsupported"]
    observed_result: str = Field(min_length=10, max_length=4000)
    explanation: str = Field(min_length=10, max_length=4000)
    limitations: list[str] = Field(min_length=1, max_length=12)
    evidence_ids: list[str] = Field(min_length=1)


class Curation(StrictModel):
    summary: str
    drafts: list[MemoryDraft] = Field(max_length=8)
    unresolved: list[str]


class Task(StrictModel):
    id: str
    batch: int
    hypothesis: Hypothesis
    status: Literal["reserved", "coding", "submitted", "succeeded", "code_failed", "infra_failed",
                    "unclassified_failed", "contract_rejected", "cancelled"] = "reserved"
    code_artifact: str | None = None
    code_commit: str | None = None
    receipt: str | None = None
    evidence: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    attempts: int = 0
    worker_failures: int = Field(default=0, ge=0)
    counted: bool = False
    error: str | None = None
    mode: Literal["candidate", "probe", "combination"] = "candidate"


class State(StrictModel):
    project_id: str
    revision: int = 0
    phase: Literal["plan", "execute", "monitor", "converge", "research", "curate", "finalize", "stopped", "complete", "review"] = "plan"
    stopped: bool = False
    stop_reason: str | None = None
    batch: int = 0
    completed_batches: int = Field(default=0, ge=0)
    batch_started: float = 0
    active_plan: str | None = None
    used: int = 0
    active_tasks: list[Task] = Field(default_factory=list)
    best_evidence: str | None = None
    baseline_evidence: str | None = None
    best_metrics: dict[str, float] = Field(default_factory=dict)
    best_code: str | None = None
    last_analysis: str | None = None
    pending_review: str | None = None
    review_resolution: str | None = None
    review_analysis: str | None = None
    review_check: str | None = None
    review_receipt: str | None = None
    analysis_failures: int = 0
    last_analysis_rejection: str | None = None
    last_rejection: str | None = None
    rejected_batches: int = 0
    direction_pool: str | None = None
    attempted_directions: list[str] = Field(default_factory=list)
    research_mode: Literal["initial", "progressive", "escape"] = "initial"
    research_stage: Literal["query", "fetch", "screen"] = "query"
    research_request: str | None = None
    research_sources: str | None = None
    research_questions: list[str] = Field(default_factory=list)
    research_failures: int = 0
    research_rounds: int = 0
    research_feedback: str | None = None
    research_avoid_targets: list[str] = Field(default_factory=list)
    after_curation: Literal["plan", "research", "finalize"] = "plan"
    final_receipt: str | None = None
    final_evidence: str | None = None
    offline_accepted: bool = False
    stagnant_batches: int = 0
    refuted_keys: list[str] = Field(default_factory=list)
    # Lease + epoch prevent a late worker from committing after stop/recovery.
    lease: str | None = None
    lease_until: float = 0
    lease_pid: int | None = None
    lease_process_created: float | None = None
    event_ids: list[str] = Field(default_factory=list)
