from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from .command_candidate import candidate_parts, structural
from .model_policy import validate_proposal
from .storage import atomic, canonical, digest


def measure_command(root, config, source, code, seed, split, logs):
    if config.command == ["isolated-native"]:
        from .native_study_runtime import measure_native
        return measure_native(root, config, source, code, seed, split, logs)
    if structural(config):
        from .prediction_experiment import measure_prediction
        return measure_prediction(root, config, source, code, seed, split, logs)
    from .fold_experiment import measure_command as measure_existing
    return measure_existing(root, config, source, code, seed, split, logs)


class LocalPlatform:
    """Detached real Python experiments with durable idempotency receipts.

    Unknown submission outcomes stay unknown until reconciled; never blindly resubmit.
    """
    def __init__(self, root: Path, command_experiment=None, source=None):
        self.root = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / ".submit.lock"))
        self.command_experiment, self.source = command_experiment, source

    @property
    def module(self):
        if self.command_experiment and self.command_experiment.command == ["isolated-predict"]:
            return "loop_research.prediction_experiment"
        if self.command_experiment and self.command_experiment.command[0] == "isolated-fold":
            return "loop_research.fold_experiment"
        if self.command_experiment and self.command_experiment.command == ["isolated-native"]:
            return "loop_research.native_study_runtime"
        return "loop_research.command_experiment" if self.command_experiment else "loop_research.training"

    @property
    def evaluator_sha256(self):
        evaluator = Path(__file__).with_name(self.module.rsplit(".", 1)[-1] + ".py").read_bytes()
        modules = ()
        if self.module == "loop_research.prediction_experiment":
            modules = ("bridge_runtime", "onboard_replay", "command_candidate", "command_experiment")
        elif self.module == "loop_research.fold_experiment":
            modules = ("fold_runtime", "sealed_stage", "onboard_replay", "bridge_runtime", "command_experiment")
        elif self.module == "loop_research.native_study_runtime":
            modules = ("onboard_native", "sealed_stage", "onboard_replay", "command_experiment", "command_candidate",
                       "native_source_axes")
        elif self.command_experiment and self.command_experiment.bridge_interpreter:
            modules = ("bridge_runtime", "onboard_replay")
        dependencies = (*modules, "runtime_failure")
        checksums = [digest(Path(__file__).with_name(name + ".py").read_bytes()) for name in dependencies]
        return digest(canonical([digest(evaluator), *checksums]))

    def submit(self, task_id: str, code: str, seed: int, *, split="validation") -> str:
        if self.command_experiment:
            candidate_parts(code, self.command_experiment)
        else:
            validate_proposal(code, seed)
        request = {"task_id": task_id, "code": code, "seed": seed, "split": split,
                   "evaluator_sha256": self.evaluator_sha256}
        if self.command_experiment:
            request.update(command_experiment=self.command_experiment.model_dump(), source=self.source)
        key = digest(canonical(request))
        directory = self.root / key
        with self.lock:
            if (directory / "receipt.json").exists():
                return key
            directory.mkdir(parents=True, exist_ok=True)
            atomic(directory / "request.json", canonical(request))
            atomic(directory / "receipt.json", canonical({"key": key, "submitted": time.time(), "pid": None}))
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with (directory / "stdout.log").open("w") as out, (directory / "stderr.log").open("w") as err:
                try:
                    process = subprocess.Popen([sys.executable, "-m", self.module, str(directory)],
                                               stdout=out, stderr=err, env=env, start_new_session=True)
                except OSError as exc:
                    atomic(directory / "result.json", canonical({
                        "status": "infra_failed", "failure_category": "runner_launch_failed",
                        "error": str(exc), "request_sha256": key,
                    }))
                    return key
            # psutil guards against PID reuse during later monitor/cancellation.
            import psutil
            receipt = {"key": key, "submitted": time.time(), "pid": process.pid,
                       "created": psutil.Process(process.pid).create_time()}
            atomic(directory / "receipt.json", canonical(receipt))
        return key

    def poll(self, key: str, required_metrics: list[str] | None = None) -> dict:
        import psutil
        directory = self._directory(key)
        request = json.loads((directory / "request.json").read_bytes())
        if digest(canonical(request)) != key:
            raise ValueError("Training request tampered")
        if request["evaluator_sha256"] != self.evaluator_sha256:
            raise ValueError("Frozen evaluator changed during experiment")
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_bytes())
            if result.get("request_sha256") != key:
                raise ValueError("Result does not belong to submitted request")
            if result.get("status") not in {"succeeded", "code_failed", "infra_failed", "unclassified_failed"}:
                raise ValueError("Invalid training status")
            if result["status"] in {"infra_failed", "unclassified_failed"} and not result.get("failure_category"):
                raise ValueError("A durable runtime fault needs a structured diagnostic")
            if result["status"] == "succeeded":
                metrics = result.get("metrics")
                if not metrics or any(k not in metrics for k in required_metrics or []):
                    raise ValueError("Missing measured metrics")
                if any(type(v) not in (float, int) or not math.isfinite(v) for v in metrics.values()):
                    raise ValueError("Invalid metric value")
                if result.get("code_sha256") != digest(request["code"].encode()):
                    raise ValueError("Wrong model code")
            return result
        receipt = json.loads((directory / "receipt.json").read_bytes())
        if receipt["pid"] is None:
            if receipt.get("reconciled_absent"):
                return {"status": "infra_failed", "failure_category": "runner_never_started",
                        "error": "Local submission reconciled: no process and no result",
                        "request_sha256": key}
            return {"status": "unknown", "error": "Submission receipt incomplete; reconcile before retry"}
        try:
            process = psutil.Process(receipt["pid"])
            if process.create_time() == receipt["created"] and process.status() != psutil.STATUS_ZOMBIE:
                timeout = self.command_experiment.timeout_seconds + 30 if self.command_experiment else 120
                if time.time() - receipt["submitted"] < timeout:
                    return {"status": "running"}
                self.cancel(key)
                return {"status": "code_failed", "failure_category": "training_deadline",
                        "error": f"Training exceeded {timeout}-second budget",
                        "request_sha256": key}
        except psutil.NoSuchProcess:
            pass
        return {"status": "infra_failed", "failure_category": "runner_lost",
                "error": "Process exited without durable result", "request_sha256": key}

    def reconcile(self, key: str, required_metrics: list[str] | None = None) -> dict:
        """Recover the local spawn/receipt gap without launching another training."""
        import psutil
        directory = self._directory(key)
        with self.lock:
            result = self.poll(key, required_metrics)
            if result["status"] != "unknown":
                return result
            path = directory / "receipt.json"
            receipt = json.loads(path.read_bytes())
            uncertain = False
            for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
                try:
                    args = process.info["cmdline"] or []
                    if self.module in args and str(directory) in args:
                        receipt.update(pid=process.pid, created=process.info["create_time"], reconciled=time.time())
                        atomic(path, canonical(receipt))
                        return self.poll(key, required_metrics)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    uncertain = True
            if not uncertain and time.time() - receipt["submitted"] >= 30:
                receipt.update(reconciled_absent=True, reconciled=time.time())
                atomic(path, canonical(receipt))
                return self.poll(key, required_metrics)
            return result

    def cancel(self, key: str):
        import psutil
        receipt = json.loads((self._directory(key) / "receipt.json").read_bytes())
        if receipt["pid"] is None:
            return False
        try:
            process = psutil.Process(receipt["pid"])
            if process.create_time() != receipt["created"]:
                return False
            os.killpg(process.pid, signal.SIGTERM)
            return True
        except (psutil.NoSuchProcess, ProcessLookupError):
            return False

    def _directory(self, key):
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise ValueError("Invalid receipt")
        return self.root / key
