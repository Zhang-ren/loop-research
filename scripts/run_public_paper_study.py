"""Run a bounded literature, source, reproduction, experiment and writing pilot."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import httpx
from bs4 import BeautifulSoup
from pydantic import Field

from loop_research.command_experiment import run_command
from loop_research.engine import Research
from loop_research.literature import LiteratureRadar, normalized_query
from loop_research.models import CommandExperiment, EnvironmentRequirements, Metric, Project, StrictModel
from loop_research.preflight import inspect
from loop_research.replay import reproduce
from loop_research.storage import Evidence, atomic, canonical, digest
from loop_research.workers import CodexWorker


class SourceChoice(StrictModel):
    repository: str
    tag: str
    example: str
    source_ids: list[str] = Field(min_length=1, max_length=6)
    rationale: str


class DiscoveryQueries(StrictModel):
    title: str
    paper_queries: list[str] = Field(min_length=2, max_length=4)
    github_queries: list[str] = Field(min_length=1, max_length=3)
    rationale: str
    primary_metric: str


class Adaptation(StrictModel):
    adapter: str = Field(min_length=200, max_length=14000)
    split_builder: str = Field(min_length=100, max_length=8000)
    baseline_json: str
    choices_json: str
    protocol: str = Field(min_length=60, max_length=6500)


class ProtocolReview(StrictModel):
    accepted: bool
    discrepancies: list[str]
    limitations: list[str]


class PaperDraft(StrictModel):
    title: str
    abstract: str
    introduction: str
    method: str
    results: str
    discussion: str
    conclusion: str
    source_ids: list[str] = Field(min_length=1)


class PaperReview(StrictModel):
    accurate: bool
    errors: list[str]
    limitations: list[str]


def github(path: str, params=None):
    with httpx.Client(timeout=25, headers={"Accept": "application/vnd.github+json", "User-Agent": "loop-research-local"}) as client:
        response = client.get("https://api.github.com" + path, params=params)
        response.raise_for_status()
        return response.json()


def raw_source(repository: str, commit: str, relative: str) -> bytes:
    parts = PurePosixPath(relative)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a GitHub repository and immutable commit")
    if parts.is_absolute() or ".." in parts.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", relative):
        raise ValueError("Invalid GitHub source path")
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    response = httpx.get(url, timeout=25)
    response.raise_for_status()
    if len(response.content) > 128_000:
        raise ValueError("Pilot source file exceeds size limit")
    return response.content


def read(path):
    return json.loads(path.read_bytes())


def submit_brief(root: Path, brief: str):
    store = Evidence(root)
    path = root / "queries.json"
    if path.exists():
        old = read(path)
        if store.get(old["brief"])["content"]["user_request"] != brief:
            raise ValueError("The existing run already has a different brief")
        return
    inputs = {"user_request": brief, "requirements": "For this CPU pilot, formulate 2–4 OpenAlex method-paper queries and "
              "1–3 licensed upstream GitHub repository queries for scikit-learn bundled 8×8 digits and the relevant classifier. "
              "Use repository qualifiers such as in:name and user: when known. Give a falsifiable question and metrics. Do not run tools."}
    parent = store.put("simulated_research_brief", inputs)
    result, metadata = CodexWorker(root).run("public-research-discovery", inputs, DiscoveryQueries)
    key = store.put("public_research_queries", {"query": result.model_dump(), "worker": metadata}, [parent])
    atomic(path, canonical({"evidence": key, "brief": parent}))


def follow_method_repository(root, search, candidates, store):
    """Resolve an author-published method paper via the discovered LIBSVM README."""
    pointer = root / "author-paper.json"
    if pointer.exists():
        key = read(pointer)["evidence"]
        store.verify_tree(key)
        return key
    name = "cjlin1/libsvm"
    if name not in candidates:
        return None
    upstream = github(f"/repos/{name}")
    commit = github(f"/repos/{name}/commits/{upstream['default_branch']}")["sha"]
    raw = raw_source(name, commit, "README")
    body = raw.decode()
    urls = re.findall(r"https?://www\.csie\.ntu\.edu\.tw/~cjlin/papers/libsvm\.pdf", body)
    if not urls:
        raise ValueError("The discovered method repository did not link the publication")
    readme_id = store.put("method_repository_readme", {"repo": name, "commit": commit,
        "readme_sha256": digest(raw), "citation_excerpt": body[body.index("Chih-Chung Chang and Chih-Jen Lin, LIBSVM"):].split(".pdf", 1)[0] + ".pdf"},
        [search["evidence"]])
    original = "https:" + urls[0].split(":", 1)[1]
    doi = "10.1145/1961189.1961199"
    location = "https://scholars.lib.ntu.edu.tw/entities/publication/3b78cddc-99c4-436e-ba75-bacc1c1bb2fd"
    response = httpx.get(location, timeout=30)
    response.raise_for_status()
    html = BeautifulSoup(response.content, "html.parser")
    identifier = html.find("meta", attrs={"name": "citation_doi"})
    description = html.find("meta", attrs={"name": "description"})
    if not identifier or identifier.get("content") != doi or not description or "implementation details of LIBSVM" not in description.get("content", ""):
        raise ValueError("Institutional abstract does not match the citation from the author repository")
    key = store.put("paper_source", {"title": "LIBSVM: A Library for Support Vector Machines",
        "url": location, "doi": "https://doi.org/" + doi, "year": 2011,
        "abstract": description["content"], "body_excerpt": "", "access": "abstract_page", "retracted": False,
        "landing_page_sha256": digest(response.content), "author_url": original,
        "provider": "official institutional DOI-matched abstract after author PDF failed"}, [readme_id])
    atomic(pointer, canonical({"evidence": key}))
    return key


def discover(root: Path, interpreter: Path) -> dict:
    store = Evidence(root)
    state = root / "discovery.json"
    if state.exists():
        record = read(state)
        store.verify_tree(record["evidence"])
        return record
    queries = read(root / "queries.json")
    plan = store.get(queries["evidence"])["content"]["query"]
    radar = LiteratureRadar(root, store)
    search_path = root / "public-searches.json"
    if search_path.exists():
        search = read(search_path)
        store.verify_tree(search["evidence"])
    else:
        def repository_search(query):
            try:
                body = github("/search/repositories", {"q": query, "per_page": 6, "sort": "stars"})
                repos = [{"name": row["full_name"], "url": row["html_url"], "description": row["description"],
                          "stars": row["stargazers_count"], "license": (row.get("license") or {}).get("spdx_id")}
                         for row in body["items"] if not row.get("archived")]
                return {"query": query, "repos": repos}
            except httpx.HTTPError as exc:
                return {"query": query, "repos": [], "error": str(exc)}

        with ThreadPoolExecutor(max_workers=3) as pool:
            papers_job = pool.submit(radar.search, plan["paper_queries"], max_papers=6)
            repos = list(pool.map(repository_search, plan["github_queries"]))
            papers = papers_job.result()
        search = {"papers": papers, "repositories": repos}
        key = store.put("public_research_search_results", search, [queries["evidence"], *papers["source_ids"],
                                                                    *(row["evidence"] for row in papers["searches"])])
        search = {**search, "evidence": key}
        atomic(search_path, canonical(search))
    candidates = {row["name"]: row for block in search["repositories"] for row in block["repos"] if row["license"]}
    if not candidates:
        raise ValueError("No open-source repository passed discovery")
    author_paper = follow_method_repository(root, search, candidates, store)
    papers = radar.cards(([author_paper] if author_paper else []) + search["papers"]["source_ids"])
    readable = [row for row in papers if row["access"] in {"abstract_page", "body_excerpt"} and not row["retracted"]]
    if not readable:
        raise ValueError("No readable primary paper; run targeted source recovery")
    version = subprocess.check_output([str(interpreter), "-I", "-c", "import sklearn; print(sklearn.__version__)"], text=True).strip()
    payload = {"brief": store.get(queries["brief"])["content"], "repos": list(candidates.values()),
               "papers": readable, "source_ids": [row["source_id"] for row in readable], "installed_sklearn_version": version,
               "requirements": "Choose an official, licensed upstream project from the supplied GitHub discovery results. "
               "Choose one existing CPU example of the exact 8x8 handwritten digits data set, including repository-relative source path. "
               "The repository field is only owner/name. For scikit-learn match the installed version tag. Include the directly recovered LIBSVM method "
               "paper when it applies. Exclude incidental papers from other domains such as text classification or quantum computing. "
               "Do not write code, claim results, or invoke tools. Only those exact fields."}
    chosen, worker = CodexWorker(root, timeout=240).run("public-research-source-review", payload, SourceChoice)
    chosen.repository = chosen.repository.removeprefix("https://github.com/").rstrip("/")
    if chosen.repository not in candidates or not set(chosen.source_ids) <= set(payload["source_ids"]):
        raise ValueError("Source selection left the discovered repositories or papers")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", chosen.tag):
        raise ValueError("Expected a single release tag")
    commit = github(f"/repos/{chosen.repository}/commits/{chosen.tag}")["sha"]
    upstream = raw_source(chosen.repository, commit, chosen.example)
    ast.parse(upstream)
    source_url = f"https://github.com/{chosen.repository}/blob/{commit}/{chosen.example}"
    record = {"selection": chosen.model_dump(), "worker": worker, "commit": commit, "source_url": source_url,
              "source_sha256": digest(upstream), "license": candidates[chosen.repository]["license"],
              "library_version": version}
    key = store.put("pinned_open_source_baseline", record, [search["evidence"], *chosen.source_ids])
    atomic(root / "upstream-example.py", upstream)
    record["evidence"] = key
    atomic(state, canonical(record))
    return record


def execute(interpreter: Path, script: Path, *args: str):
    env = {"PATH": os.defpath, "HOME": str(script.parent), "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.run([str(interpreter), str(script), *args], capture_output=True, env=env, cwd=script.parent,
                          stdin=subprocess.DEVNULL, timeout=90, check=True)


def copy_evidence(source: Evidence, target: Evidence, ids: list[str]):
    for artifact in set().union(*(source.verify_tree(key) for key in ids)):
        raw = (source.root / f"{artifact}.json").read_bytes()
        atomic(target.root / f"{artifact}.json", raw)


def prepare_study(root: Path, interpreter: Path) -> dict:
    discovery = discover(root, interpreter)
    store, worker = Evidence(root), CodexWorker(root, timeout=260)
    if (root / "prepared-study.json").exists():
        record = read(root / "prepared-study.json")
        store.verify_tree(record["evidence"])
        return record
    upstream = root / "upstream-example.py"
    if digest(upstream.read_bytes()) != discovery["source_sha256"]:
        raise ValueError("Pinned upstream source changed")
    if discovery["selection"]["repository"] != "scikit-learn/scikit-learn":
        raise ValueError("This pilot numerical auditor supports the scikit-learn digits benchmark")
    baseline_path = root / "upstream-reproduction.json"
    if baseline_path.exists():
        baseline_record = read(baseline_path)
        store.verify_tree(baseline_record["evidence"])
    else:
        # Execute the unmodified official example; read its exported arrays after execution.
        runner = root / "reproduce_upstream.py"
        atomic(runner, b'''import json, runpy, sys
from sklearn.metrics import accuracy_score, f1_score
result = runpy.run_path(sys.argv[1], run_name="__main__")
y, p = result["y_test"], result["predicted"]
print(json.dumps({"train_count": len(result["y_train"]), "test_count": len(y),
                  "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}))
''')
        run = execute(interpreter, runner, str(upstream))
        atomic(root / "upstream-stdout.log", run.stdout)
        atomic(root / "upstream-stderr.log", run.stderr)
        original = json.loads(run.stdout.splitlines()[-1])
        if original["train_count"] + original["test_count"] != 1797:
            raise ValueError("Unexpected upstream study scope")
        baseline_record = {"source_commit": discovery["commit"], "interpreter": str(interpreter),
                           "source_sha256": discovery["source_sha256"], "metrics": original,
                           "stdout_sha256": digest(run.stdout), "returncode": run.returncode}
        baseline_record["evidence"] = store.put("executed_upstream_example", baseline_record, [discovery["evidence"]])
        atomic(baseline_path, canonical(baseline_record))

    draft_ref = root / "adaptation.json"
    payload = {"upstream": upstream.read_text(), "source_url": discovery["source_url"], "bibliography": discovery["selection"]["source_ids"],
        "task": "Author a compact compatible experiment adapter. Do not invoke tools. Preserve the upstream unscaled "
                "load_digits images flattened into 64 features, RBF SVC(C=1.0,gamma=0.001). Baseline and search must share "
                "a NEW single stratified disjoint split: train is 60%, validation and test 20% each; seed 42 for both split stages. "
                "The official example uses a different split, do not align or compare cross-protocol performance. "
                "Split builder takes one output path argv[1] and writes exact train/validation/test keys with unique integer "
                "sample indices in order. LOOP_PARAMETERS points to an absolute JSON file, not the JSON itself. "
                "LOOP_SPLIT_MANIFEST is an absolute JSON file. LOOP_SPLIT is the string validation or test. "
                "Write to the absolute LOOP_METRICS path. Its JSON must be exactly {\"metrics\":{\"accuracy\":0.9,\"macro_f1\":0.89},"
                "\"predictions\":[...]}, using actual calculations, not the example numbers. "
                "LOOP_SPLIT_MANIFEST has exactly train and the chosen validation or test split. Fit only train IDs and score "
                "only selected IDs. Keep metric keys accuracy and macro_f1, and raw integer predictions in given evaluation ID order. "
                "Use exactly three single-variable axes: C, gamma, kernel. Candidate values should be modest, make baseline "
                "typed C=1.0, gamma=0.001, kernel=rbf; include the baseline value FIRST and exactly one alternative SECOND in each choices list. "
                "Choose interpretable alternatives. Keep each axis same type. "
                "Use metrics.accuracy_score and f1_score average macro. No normalization, model selection using test or timings. "
                "Generate baseline_json and choices_json as JSON objects encoded in those string fields. Document the protocol in Chinese. "
                "The three candidate budget is fixed; old upstream sample order and entire original half split are a separate replay. "
                "Do not report experimental values, write only the two short scripts and frozen design."}
    rejection = root / "adapter-contract-feedback.json"
    feedback_parents = []
    if rejection.exists():
        feedback_id = read(rejection)["evidence"]
        payload["previous_contract_feedback"] = store.get(feedback_id)["content"]
        feedback_parents.append(feedback_id)
    if not draft_ref.exists():
        draft, metadata = worker.run("public-study-adapter-coder", payload, Adaptation)
        ast.parse(draft.adapter, feature_version=(3, 9))
        ast.parse(draft.split_builder, feature_version=(3, 9))
        key = store.put("public_study_adapter", {"draft": draft.model_dump(), "worker": metadata},
                        [discovery["evidence"], baseline_record["evidence"], *feedback_parents])
        atomic(draft_ref, canonical({"evidence": key}))
    draft_id = read(draft_ref)["evidence"]
    draft = store.get(draft_id)["content"]["draft"]
    review_path = root / "adaptation-review.json"
    if not review_path.exists():
        assessment, metadata = worker.run("public-study-protocol-review", {
            "task": "Independently verify that the supplied code follows the requested original baseline and NEW frozen protocol. "
                    "Check separation, no heldout tuning, unchanged raw-feature RBF baseline, valid one-axis settings, "
                    "and that actual metrics and predictions are written as required. List only material errors as discrepancies. "
                    "Set accepted=true when material discrepancies are empty. No code changes or execution.",
            "requirements": payload["task"], "upstream": payload["upstream"], "draft": draft}, ProtocolReview)
        key = store.put("public_study_adapter_review", {"review": assessment.model_dump(), "worker": metadata}, [draft_id])
        atomic(review_path, canonical({"evidence": key}))
    review_id = read(review_path)["evidence"]
    report = store.get(review_id)["content"]["review"]
    if not report["accepted"] or report["discrepancies"]:
        raise ValueError(f"Adapter needs revision: {report['discrepancies']}")

    repository = root / "experiment"
    repository.mkdir(exist_ok=True)
    for name, value in {"evaluate.py": draft["adapter"], "make_split.py": draft["split_builder"], "PROTOCOL.md": draft["protocol"]}.items():
        atomic(repository / name, value.encode())
    with TemporaryDirectory(prefix="digits-split-") as directory:
        generated = Path(directory) / "split.json"
        execute(interpreter, repository / "make_split.py", str(generated))
        atomic(repository / "split.json", generated.read_bytes())
    splits = read(repository / "split.json")
    if sorted(splits.keys()) != ["test", "train", "validation"] or sorted(sum(splits.values(), [])) != list(range(1797)):
        raise ValueError("Split does not partition all 1797 samples exactly once")
    git = ["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "init", "-q", "--template="], check=True)
    subprocess.run([*git, "add", "--", "evaluate.py", "make_split.py", "PROTOCOL.md", "split.json"], check=True)
    if subprocess.check_output([*git, "diff", "--cached", "--name-only"]):
        subprocess.run([*git, "-c", "user.name=LOOP Research", "-c", "user.email=local@invalid", "commit", "-qm",
                        "Freeze worker-authored digits study and split"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    config = CommandExperiment(repository=str(repository), revision=revision, command=[str(interpreter), "evaluate.py"],
        split_manifest="split.json", baseline=json.loads(draft["baseline_json"]), choices=json.loads(draft["choices_json"]), timeout_seconds=45)
    if config.baseline != {"C": 1.0, "gamma": 0.001, "kernel": "rbf"}:
        raise ValueError("Adapter does not preserve the official default SVC control")
    project = Project(name="公开手写数字分类复现与限定改进", objective="Improve validation accuracy and macro F1 on the fixed CPU digits split with three conservative single-variable SVC experiments.",
        knowledge="这是公开 1797 个 sklearn 8×8 数字样本的新冻结划分。主指标 accuracy，次指标 macro_f1。"
                  "原示例后半段测试无法与新划分混比。不得用新测试集选择配置。\n" + draft["protocol"],
        metrics=[Metric(name="accuracy", threshold=1.0), Metric(name="macro_f1", threshold=1.0)],
        max_experiments=3, branches=3, concurrency=3, archive_uri="local", independent_review=True,
        reversible=True, affordable=True, confirmed_knowledge=True, command_experiment=config,
        environment=EnvironmentRequirements(python_modules=["sklearn", "numpy"], committed_files=["evaluate.py"]))
    checks = inspect(project)
    if checks["status"] != "ready":
        raise ValueError(f"Study preflight failed: {checks['issues']}")
    atomic(root / "experiment.json", canonical(project.model_dump()))
    record = {"status": "ready", "revision": revision, "preflight": checks,
              "split_counts": {name: len(indices) for name, indices in splits.items()}, "source": discovery["evidence"]}
    key = store.put("frozen_public_study_design", record, [review_id, baseline_record["evidence"]])
    atomic(root / "prepared-study.json", canonical({**record, "evidence": key}))
    return {**record, "evidence": key}


class StudyRadar(LiteratureRadar):
    """Use the method papers retrieved in the live discovery stage."""
    def search(self, queries, *, max_papers=8):
        initial = read(self.root / "admitted-sources.json")
        rows = []
        for query in queries:
            content = {"query": query, "provider": "screened_public_discovery_sources", "source_ids": initial["sources"]}
            key = self.evidence.put("literature_search", content, initial["sources"])
            row = {"query": query, "evidence": key}
            atomic(self.root / "searches" / f"{digest(normalized_query(query).encode())}.json", canonical(row))
            rows.append(row)
        return {"searches": rows, "source_ids": initial["sources"], "unique_hits": len(initial["sources"]), "unread_hits": 0}


def implementation_reference(root: Path, discovery: dict) -> str:
    store = Evidence(root)
    pointer = root / "svc-parameter-source.json"
    if pointer.exists():
        key = read(pointer)["evidence"]
        store.verify_tree(key)
        return key
    relative = "sklearn/svm/_classes.py"
    name = discovery["selection"]["repository"]
    revision = discovery["commit"]
    code = raw_source(name, revision, relative)
    docs = next(ast.get_docstring(node) for node in ast.parse(code).body
                if isinstance(node, ast.ClassDef) and node.name == "SVC")
    if not all(term in docs for term in ("Regularization parameter", "Specifies the kernel type", "Kernel coefficient")):
        raise ValueError("SVC parameter documentation is missing")
    card = {"title": f"scikit-learn {discovery['library_version']} SVC implementation source and parameter docstring",
            "url": f"https://github.com/{name}/blob/{revision}/{relative}", "abstract": "", "body_excerpt": docs[:4900],
            "type": "official_implementation_documentation", "access": "body_excerpt", "retracted": False,
            "commit": revision, "source_sha256": digest(code), "trust": "implementation semantics only"}
    key = store.put("paper_source", card, [discovery["evidence"]])
    atomic(pointer, canonical({"evidence": key}))
    return key


def run_study(root: Path, interpreter: Path):
    prepare_study(root, interpreter)
    run = root / "study"
    if not (run / "state.json").exists():
        Research.initialize(run, Project.model_validate_json((root / "experiment.json").read_bytes()))
    engine = Research(run)
    discovery = read(root / "discovery.json")
    ids = [*discovery["selection"]["source_ids"], implementation_reference(root, discovery)]
    copy_evidence(Evidence(root), engine.evidence, [*ids, discovery["evidence"]])
    atomic(run / "literature/admitted-sources.json", canonical({"sources": ids, "discovery": discovery["evidence"]}))
    for _ in range(90):
        engine = Research(run)
        engine.radar = StudyRadar(run, engine.evidence)
        state = engine.tick()
        print(f"phase={state.phase} revision={state.revision} experiments={state.used}", flush=True)
        if state.phase == "complete":
            return state
        if state.phase in {"review", "stopped"}:
            raise ValueError(f"Research paused for review: {state.stop_reason} {state.pending_review}")
        time.sleep(0.5)
    raise TimeoutError("Study ticks exhausted")


def verify_predictions(measurement, all_labels):
    indices = measurement["evaluation_indices"]
    predictions = measurement["predictions"]
    if len(indices) != len(predictions) or not set(predictions) <= set(range(10)):
        raise ValueError("Digit predictions have an invalid length or class")
    actual = [all_labels[index] for index in indices]
    correct = [int(y == p) for y, p in zip(actual, predictions, strict=True)]
    f1 = []
    for digit in range(10):
        tp = sum(y == p == digit for y, p in zip(actual, predictions, strict=True))
        den = actual.count(digit) + predictions.count(digit)
        f1.append(2 * tp / den if den else 0)
    metrics = {"accuracy": sum(correct) / len(actual), "macro_f1": sum(f1) / 10}
    if any(not math.isclose(metrics[k], measurement["metrics"][k], abs_tol=1e-12) for k in metrics):
        raise ValueError("Independent metric recomputation disagrees with command output")
    return correct


def audit_study(root: Path, interpreter: Path):
    result_path = root / "study-audit.json"
    store = Evidence(root)
    if result_path.exists():
        result = read(result_path)
        store.verify_tree(result["evidence"])
        return result
    engine = Research(root / "study")
    state = engine.store.read()
    if state.phase != "complete" or not state.final_evidence or state.used != 3:
        raise ValueError("Expected three measured candidates and a complete frozen holdout")
    refs = engine.references()
    for candidate in (state.baseline_evidence, state.best_evidence, state.final_evidence):
        engine.evidence.verify_tree(candidate)
    history = engine.evidence.get(state.baseline_evidence)["content"]
    sealed = engine.evidence.get(state.final_evidence)["content"]
    labels = json.loads(subprocess.check_output([str(interpreter), "-I", "-c", "import json; "
        "from sklearn.datasets import load_digits; print(json.dumps(load_digits().target.tolist()))"], text=True))
    if len(labels) != 1797:
        raise ValueError("Unexpected digits dataset")
    verify_predictions(history, labels)
    if set(sealed["evaluation_indices"]) & (set(history["evaluation_indices"]) | set(history["train_indices"])):
        raise ValueError("Holdout overlaps prior split")
    config = engine.project.command_experiment
    candidate_rows = []
    for task in state.active_tasks:
        if task.status != "succeeded" or not task.code_artifact or not task.receipt:
            raise ValueError("Missing actual candidate measurement")
        raw = engine.evidence.get(task.evidence)["content"]
        engine.verify_validation(raw, state.baseline_evidence)
        verify_predictions(raw, labels)
        code = engine.evidence.get(task.code_artifact)["content"]
        params = json.loads(code["code"])
        if engine.intervention(canonical(config.baseline).decode(), code["code"]) != task.hypothesis.machine_target:
            raise ValueError("Raw candidate code violated single-variable control")
        candidate_rows.append({"hypothesis": task.hypothesis.title, "setting": task.hypothesis.machine_target,
                               "parameters": params, "metrics": raw["metrics"], "evidence": task.evidence,
                               "candidate_git_commit": task.code_commit, "receipt": task.receipt})

    control = run_command(engine.root, config, engine.platform.source, canonical(config.baseline).decode(),
                          engine.project.seed, "test", root / "holdout-control")
    if control["evaluation_indices"] != sealed["evaluation_indices"] or control["data_sha256"] != sealed["data_sha256"]:
        raise ValueError("Control and selected configuration do not share the frozen holdout")
    a, b = (verify_predictions(item, labels) for item in (control, sealed))
    deltas = [new - old for old, new in zip(a, b, strict=True)]
    retained = state.best_code == engine.evidence.get(state.baseline_evidence)["parents"][0]
    interval = None
    if retained:
        if control["predictions"] != sealed["predictions"]:
            raise ValueError("The retained baseline produced inconsistent holdout predictions")
    else:
        rng = random.Random(42)
        bootstrap = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
        interval = [bootstrap[49], bootstrap[1949]]
    replay = reproduce(engine, state.final_evidence)
    if not replay["passed"]:
        raise ValueError("Frozen holdout did not reproduce")
    ids = [state.baseline_evidence, state.best_evidence, state.final_evidence, replay["evidence"],
           *(row["evidence"] for row in candidate_rows), refs["protocol"]]
    copy_evidence(engine.evidence, store, ids)
    control_id = store.put("paper_holdout_baseline", control, [refs["protocol"], state.baseline_evidence])
    result = {"baseline_validation": history["metrics"], "candidates": candidate_rows,
              "baseline_parameters": config.baseline, "selected_parameters": json.loads(engine.evidence.get(state.best_code)["content"]["code"]),
              "selected_validation": state.best_metrics, "selected_validation_evidence": state.best_evidence,
              "control_holdout": control["metrics"], "selected_holdout": sealed["metrics"],
              "holdout_count": len(deltas), "control_correct_selected_wrong": deltas.count(-1),
              "control_wrong_selected_correct": deltas.count(1),
              "retained_baseline": retained, "accuracy_paired_ci_95": interval,
              "accuracy_delta": sum(deltas) / len(deltas), "reproduction": replay["checks"],
              "selected_holdout_evidence": state.final_evidence, "control_holdout_evidence": control_id,
              "source_commit": config.revision, "split_sha256": digest((root / "experiment/split.json").read_bytes())}
    key = store.put("paper_results_audit", result, [*ids, control_id, read(root / "prepared-study.json")["evidence"]])
    result = {**result, "evidence": key}
    atomic(result_path, canonical(result))
    return result


def results_tables(upstream, result):
    def percent(value):
        return f"{100 * value:.2f}%"

    def row(label, params, values):
        return f"| {label} | `{json.dumps(params, ensure_ascii=False, sort_keys=True)}` | {percent(values['accuracy'])} | {percent(values['macro_f1'])} |"

    lines = [f"未修改的上游示例在其原顺序切分上使用 {upstream['train_count']} 个训练样本与 "
        f"{upstream['test_count']} 个测试样本，准确率 {percent(upstream['accuracy'])}，宏平均 F1 {percent(upstream['macro_f1'])}。",
        "", "### 本研究验证集", "", "| 配置 | 参数 | 准确率 | 宏平均 F1 |", "|---|---|---:|---:|",
        row("基线", result["baseline_parameters"], result["baseline_validation"])]
    for candidate in result["candidates"]:
        lines.append(row(candidate["setting"].removeprefix("parameter."), candidate["parameters"], candidate["metrics"]))
    lines.extend(["", "### 本研究冻结测试集", "", "| 配置 | 参数 | 准确率 | 宏平均 F1 |", "|---|---|---:|---:|",
                  row("基线对照", result["baseline_parameters"], result["control_holdout"]),
                  row("保留基线复跑" if result["retained_baseline"] else "验证集选中配置",
                      result["selected_parameters"], result["selected_holdout"])])
    if result["retained_baseline"]:
        lines.extend(["", f"验证集没有支持更换基线，两行是同一配置在 {result['holdout_count']} 个测试样本上的独立运行，逐条预测一致。"
                      "三个未选候选只使用了验证集。本试验因此没有不同配置之间的测试集效应估计。"])
    else:
        lower, upper = (100 * item for item in result["accuracy_paired_ci_95"])
        lines.extend(["", f"在相同的 {result['holdout_count']} 个样本上，选中配置的准确率变化为 "
            f"{100 * result['accuracy_delta']:+.2f} 个百分点。配对样本 bootstrap 95% 区间为 [{lower:+.2f}, {upper:+.2f}] 个百分点。"
            f"其中基线错而选中配置对的样本有 {result['control_wrong_selected_correct']} 个，反向变化有 "
            f"{result['control_correct_selected_wrong']} 个。该区间描述当前切分上的样本不确定性。"])
    return "\n".join(lines)


def write_paper(root: Path, interpreter: Path):
    facts = audit_study(root, interpreter)
    store = Evidence(root)
    path = root / "paper.md"
    receipt_path = root / "paper-receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        store.verify_tree(receipt["evidence"])
        if digest(path.read_bytes()) != receipt["paper_sha256"]:
            raise ValueError("Manuscript changed after review")
        return receipt
    upstream_record = read(root / "upstream-reproduction.json")
    discovery = read(root / "discovery.json")
    design = read(root / "prepared-study.json")
    author_sources = [*discovery["selection"]["source_ids"], implementation_reference(root, discovery)]
    evidence_ids = [facts["evidence"], upstream_record["evidence"], discovery["evidence"], design["evidence"], *author_sources]
    sources = [{k: value for k, value in store.get(key)["content"].items() if k in {"title", "url", "doi", "year"}}
               for key in author_sources]
    table = results_tables(upstream_record["metrics"], facts)
    payload = {"source_ids": evidence_ids, "study_design": (root / "experiment/PROTOCOL.md").read_text(),
               "split_counts": design["split_counts"], "facts": facts, "upstream": upstream_record["metrics"],
               "frozen_tables": table, "upstream_url": discovery["source_url"], "method_references": sources,
               "task": "Write a self-contained Chinese short replication-and-extension research manuscript. All six prose fields "
                       "are separate paragraphs. The controller inserts the exact tables and reference list. Do not repeat tables, invent "
                       "models, sweeps, heldout comparisons for unselected candidates, author claims, novel algorithms or statistical significance. "
                       "Explicitly distinguish the upstream example reproduction and separate frozen 60/20/20 protocol. Explain if accuracy "
                       "improvement was absent or uncertain. If retained_baseline is true, both test rows are the same configuration "
                       "with identical predictions, never infer candidate generalization from them, and do not report an interval. "
                       "Otherwise mention the paired interval only when non-null. Limitations include one fixed split, "
                       "public bundled digits subset, small search, identity groups unavailable, no broad claim. Analysis code uses 2000 paired "
                       "bootstrap draws with seed 42 on test predictions. The final test was not used for candidate selection. Exact data and "
                       "source IDs only. Report data are fixed. Use plain professional Chinese, no emojis."}
    worker = CodexWorker(root, timeout=240)
    parent = facts["evidence"]
    for _ in range(2):
        draft, metadata = worker.run("public-study-paper-writer", payload, PaperDraft)
        if not set(draft.source_ids) <= set(evidence_ids) or not {facts["evidence"], upstream_record["evidence"]} <= set(draft.source_ids):
            raise ValueError("Paper omitted audited results or cited unknown evidence")
        key = store.put("paper_draft", {"draft": draft.model_dump(), "worker": metadata}, [parent, *evidence_ids])
        review, metadata = worker.run("public-study-paper-reviewer", {"draft": draft.model_dump(), "inputs": payload,
            "task": "Independently compare every claim in this draft with the frozen tables and evidence. Check original reproduction vs "
                    "new train/validation/test protocol, limits of observed change, source IDs, and absence of unmeasured comparisons. "
                    "List factual errors in errors. Put ordinary scope limits separately. Mark accurate when factual errors are empty."}, PaperReview)
        parent = store.put("independent_paper_review", {"review": review.model_dump(), "worker": metadata}, [key, facts["evidence"]])
        if review.accurate and not review.errors:
            break
        payload["review_feedback"] = review.errors
    else:
        raise ValueError(f"Manuscript review did not pass: {review.errors}")

    lines = [f"# {draft.title}", "", "## 摘要", "", draft.abstract, "", "## 1 引言", "", draft.introduction,
             "", "## 2 材料与方法", "", draft.method, "", "## 3 结果", "", draft.results, "", table,
             "", "## 4 讨论与局限", "", draft.discussion, "", "## 5 结论", "", draft.conclusion, "", "## 参考文献与实现", ""]
    for reference in sources:
        lines.append(f"- [{reference['title']}]({reference.get('doi') or reference['url']})")
    lines.extend([f"- [scikit-learn 原始示例与固定提交]({discovery['source_url']})",
                  "- [UCI Optical Recognition of Handwritten Digits 数据说明](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)",
                  "", "## 复现材料", "", "- [冻结实验协议](experiment/PROTOCOL.md) · [样本划分](experiment/split.json)",
                  "- [上游示例实跑回执](upstream-reproduction.json) · [候选和冻结测试审计](study-audit.json)",
                  "- [调研和开源项目检索](public-searches.json) · [代码版本记录](discovery.json)", ""])
    raw = "\n".join(lines).encode()
    atomic(path, raw)
    receipt = {"status": "reviewed", "paper": str(path), "paper_sha256": digest(raw), "review": parent,
               "audit": facts["evidence"], "source_commit": discovery["commit"]}
    result = {**receipt, "evidence": store.put("reviewed_manuscript", receipt, [parent, facts["evidence"]])}
    atomic(receipt_path, canonical(result))
    return result


def run_pipeline(root: Path, interpreter: Path):
    run_study(root, interpreter)
    return write_paper(root, interpreter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--brief", help="Submit the research question for a new pilot")
    parser.add_argument("--stage", choices=["discover", "prepare", "run", "audit", "paper", "all"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.brief:
        submit_brief(root, args.brief)
    action = {"discover": discover, "prepare": prepare_study, "run": run_study, "audit": audit_study,
              "paper": write_paper, "all": run_pipeline}[args.stage]
    result = action(root, args.interpreter.resolve(strict=True))
    print(json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, ensure_ascii=False))
