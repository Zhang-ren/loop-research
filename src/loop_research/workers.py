from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from platform import machine

import psutil
from pydantic import BaseModel

from .storage import atomic, canonical, digest


class WorkerFailure(RuntimeError):
    pass


class WorkerCancelled(WorkerFailure):
    pass


def codex_executable() -> str:
    binary = os.environ.get("LOOP_CODEX_BINARY") or shutil.which("codex")
    if not binary:
        raise WorkerFailure("Codex CLI executable is missing")
    path = Path(binary).resolve(strict=True)
    if sys.platform.startswith("linux") and path.suffix == ".js":
        architecture = {"x86_64": ("x64", "x86_64"), "aarch64": ("arm64", "aarch64")}.get(machine())
        if architecture is None:
            raise WorkerFailure("Set LOOP_CODEX_BINARY to the native Codex executable")
        npm_arch, triple_arch = architecture
        path = (path.parent.parent / "node_modules" / "@openai" / f"codex-linux-{npm_arch}" /
                "vendor" / f"{triple_arch}-unknown-linux-musl" / "bin" / "codex").resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise WorkerFailure("Codex CLI binary is not executable")
    return str(path)


def response_schema(model: type[BaseModel], *, source_ids=None, evidence_ids=None, direction_ids=None,
                    machine_targets=None, classifier_names=None, source_paths=None, entrypoint_paths=None) -> dict:
    """Strict structured output requires every property, including nullable ones."""
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: convert(item) for key, item in value.items() if key != "default"}
        if "properties" in result:
            result["required"] = list(result["properties"])
            result["additionalProperties"] = False
            for name, prop in result["properties"].items():
                entrypoint = name in {"baseline_entrypoints", "independent_evaluation_entrypoints"}
                if entrypoint and entrypoint_paths is not None:
                    if entrypoint_paths:
                        prop["items"]["enum"] = list(dict.fromkeys(entrypoint_paths))
                    else:
                        prop["maxItems"] = 0
                elif source_paths and (entrypoint or name in {"paths", "evidence_paths"}):
                    prop["items"]["enum"] = list(dict.fromkeys(source_paths))
                allowed = source_ids if name in {"source_ids", "source_id"} else evidence_ids if name == "evidence_ids" else None
                if allowed:
                    if name in {"source_ids", "evidence_ids"}:
                        prop["items"]["enum"] = list(dict.fromkeys(allowed))
                    else:
                        prop["enum"] = list(dict.fromkeys(allowed))
                allowed = {"direction_id": direction_ids, "machine_target": machine_targets,
                           "classifier_name": classifier_names, "selected_evidence": evidence_ids}.get(name)
                if allowed:
                    if "anyOf" in prop:
                        for branch in prop["anyOf"]:
                            if branch.get("type") == "string":
                                branch["enum"] = list(dict.fromkeys(allowed))
                    elif prop.get("type") == "string":
                        prop["enum"] = list(dict.fromkeys(allowed))
        return result
    return convert(model.model_json_schema())


def restricted_command(command: list[str], protected: Path, workspace: Path) -> list[str]:
    """Use the existing macOS sandbox for actual sibling-read isolation.

    This is a narrow local adapter, not a portable security implementation.
    Other hosts must supply an equivalent runner; never silently downgrade.
    """
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise WorkerFailure("No verified isolation runner on this host")
    protected, workspace = protected.resolve(), workspace.resolve()
    workspace.relative_to(protected)
    def quoted(path):
        return json.dumps(str(path))
    profile = "\n".join([
        "(version 1)", "(allow default)",
        "(deny file-write* (require-all "
        f"(require-not (subpath {quoted(workspace)})) "
        f"(require-not (subpath {quoted(Path.home() / '.codex')})) "
        '(require-not (literal "/dev/null")) (require-not (literal "/dev/tty")) '
        '(require-not (literal "/dev/dtracehelper"))))',
        f"(deny file-read-data file-write* (require-all (subpath {quoted(protected)}) "
        f"(require-not (subpath {quoted(workspace)}))))",
    ])
    return ["/usr/bin/sandbox-exec", "-p", profile, *command]


class CodexWorker:
    def __init__(self, root: Path, model: str | None = None, timeout: int = 240):
        self.root = root.resolve()
        self.model = model
        self.timeout = timeout

    def reap_owned(self, owner_pid: int, owner_created: float):
        for path in (self.root / "workers").glob("*/process.json"):
            if path.with_name("metadata.json").exists() or path.with_name("recovered.json").exists():
                continue
            receipt = json.loads(path.read_bytes())
            if (receipt.get("owner_pid"), receipt.get("owner_created")) != (owner_pid, owner_created):
                continue
            try:
                child = psutil.Process(receipt["pid"])
                if child.create_time() != receipt.get("created") or os.getpgid(child.pid) != child.pid:
                    continue
                os.killpg(child.pid, signal.SIGTERM)
            except (psutil.NoSuchProcess, ProcessLookupError, psutil.AccessDenied, PermissionError):
                continue
            atomic(path.with_name("recovered.json"), canonical({"pid": child.pid, "signal": "SIGTERM",
                                                                  "reason": "decision_owner_exited"}))

    def run(self, role: str, payload: dict, response: type[BaseModel], *, isolated=True,
            cancel_check=None) -> tuple[BaseModel, dict]:
        call_id = f"{role}-{uuid.uuid4().hex[:12]}"
        directory = self.root / "workers" / call_id
        directory.mkdir(parents=True)
        atomic(directory / "input.json", canonical(payload))
        schema = response_schema(response, source_ids=payload.get("source_ids"), evidence_ids=payload.get("evidence_ids"),
            direction_ids=[d["id"] for d in payload.get("direction_pool", [])],
            machine_targets=payload.get("allowed_machine_targets"),
            source_paths=payload.get("source_paths"),
            entrypoint_paths=payload.get("entrypoint_paths"),
            classifier_names=payload.get("available_classifier_names") or
                [d["classifier_name"] for d in payload.get("direction_pool", []) if d.get("classifier_name")])
        atomic(directory / "schema.json", canonical(schema))
        method_name = {"planner": "research-ideation", "coder": "experiment-pipeline", "curator": "evo-memory",
                       "research-query": "paper-navigator/SKILL", "literature-screen": "paper-navigator/SKILL"}.get(role)
        if method_name:
            method = Path(__file__).parent / "_vendor" / "evoskills" / f"{method_name}.md"
            if method.exists():
                payload = {**payload, "upstream_method": method.read_text(),
                           "method_scope": "Reuse relevant research reasoning only. Do not run the upstream's outer workflow, user selection stages, training budgets or direct memory writes. This bounded task's protocol takes precedence."}
                atomic(directory / "input.json", canonical(payload))
        prompt = (
            f"You are the bounded {role} worker of an evidence-driven research workflow. "
            "The task input follows this instruction and is also saved as input.json. It contains all authorized context. "
            "Do not start another research loop, submit training, edit evaluation rules, or publish memory. "
            "Return the requested JSON response. Source/evidence IDs must come from the input. "
            "Distinguish observations from causal explanations. Do not invent metrics. "
            "If asked for code, return the complete artifact text specified in the task in the code field; do not execute training.\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        binary = codex_executable()
        command = [binary, "--no-daemon", "-a", "never", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "-C", str(directory), "--json",
                   "--output-schema", str(directory / "schema.json"),
                   "-c", "features.memories=false", "-c", "memories.use_memories=false",
                   "-c", "memories.generate_memories=false", "-c", "project_doc_max_bytes=0",
                   "-c", 'model_reasoning_effort="medium"']
        # Disable inherited app/plugin/hook and host-skill discovery for bounded research workers.
        for feature in ["apps", "plugins", "remote_plugin", "hooks", "browser_use", "computer_use", "multi_agent",
                        "unified_exec", "daemon_auto_start", "shell_snapshot"]:
            command += ["--disable", feature]
        command += ["--enable", "skip_host_skill_discovery"]
        if isolated:
            # Use the host's native per-tool policy; an outer process sandbox does not
            # constrain tools delegated to an existing service. Do not combine profile
            # selection with legacy --sandbox flags.
            filesystem = '":root"="deny",":minimal"="read",":tmpdir"="deny",":slash_tmp"="deny"'
            if sys.platform.startswith("linux"):
                # bwrap re-executes the native CLI inside its own mount namespace.
                filesystem += f',{json.dumps(binary)}="read"'
            command += ["-c", 'default_permissions="research"',
                        "-c", 'permissions.research.extends=":workspace"',
                        "-c", 'permissions.research.filesystem={' + filesystem + '}',
                        "-c", 'permissions.research.network.enabled=false']
        else:
            command += ["--sandbox", "read-only"]
        if self.model:
            command += ["-m", self.model]
        command.append("-")
        env = dict(os.environ)
        # Use the user-authorized CLI login. Do not forward unrelated API secrets to workers.
        for key in list(env):
            if key.endswith(("_API_KEY", "_SECRET", "_TOKEN")) or (key.startswith("CODEX_") and key != "CODEX_HOME"):
                env.pop(key)
        # Desktop tool bridges are session context, not authentication. Inheriting them would
        # execute worker tools in the host chat and defeat process-level isolation.
        temporary = directory / "tmp"
        temporary.mkdir()
        env["TMPDIR"] = str(temporary)
        started = time.time()
        with (tempfile.TemporaryFile(dir=temporary) as request, (directory / "events.jsonl").open("w") as events,
              (directory / "stderr.log").open("w") as errors):
            request.write(prompt.encode("utf-8"))
            request.seek(0)
            process = subprocess.Popen(command, cwd=directory, stdout=events, stderr=errors,
                                       stdin=request, env=env, start_new_session=True)
            try:
                created = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                created = None
            owner = psutil.Process(os.getpid())
            atomic(directory / "process.json", canonical({"pid": process.pid, "started": started, "created": created,
                                                           "owner_pid": owner.pid, "owner_created": owner.create_time()}))
            reason = None
            while process.poll() is None:
                if cancel_check and cancel_check():
                    reason = "cancelled"
                elif time.time() - started > self.timeout:
                    reason = "timeout"
                if reason:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    break
                time.sleep(0.2)
        finished = time.time()
        metadata = {"call_id": call_id, "role": role, "started": started, "finished": finished,
                    "exit_code": process.returncode, "reason": reason, "directory": str(directory),
                    "usage": None, "thread_id": None, "model_requested": self.model,
                    "isolated": isolated, "native_memory": False,
                    "isolation_backend": "codex-native-permissions/research" if isolated else "read-only",
                    "host_tool_bridge_inherited": False,
                    "input_sha256": digest((directory / "input.json").read_bytes())}
        final_message = None
        for line in (directory / "events.jsonl").read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("type") == "thread.started":
                metadata["thread_id"] = item.get("thread_id")
            if item.get("type") == "turn.completed":
                metadata["usage"] = item.get("usage")
            if item.get("type") == "item.completed" and item.get("item", {}).get("type") == "agent_message":
                final_message = item["item"].get("text")
        atomic(directory / "metadata.json", canonical(metadata))
        if reason or process.returncode != 0 or final_message is None:
            # Keep the original logs locally; avoid printing potentially sensitive service output.
            error_type = WorkerCancelled if reason == "cancelled" else WorkerFailure
            raise error_type(f"{call_id}: {reason or 'failed'}; inspect {directory / 'stderr.log'}")
        result = response.model_validate_json(final_message)
        atomic(directory / "result.json", canonical(result.model_dump()))
        return result, metadata
