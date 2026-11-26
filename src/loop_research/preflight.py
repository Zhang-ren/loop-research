"""Check the declared runtime and frozen Git inputs before executing a baseline."""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .command_candidate import PREDICTION_FILES
from .command_experiment import read_splits
from .models import Project
from .workers import WorkerFailure, codex_executable


def issue(field: str, question: str, detail: str = "") -> dict:
    return {"field": field, "question": question, "detail": detail}


def git(repository: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "--literal-pathspecs", "-C", str(repository), *args],
                            capture_output=True, timeout=30, check=True)
    return result.stdout


def committed_path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\0" in name:
        raise ValueError("文件路径须位于冻结仓库内")
    return path.as_posix()


def script_input(command: list[str]) -> str | None:
    name = Path(command[0]).name
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?|bash|sh|Rscript|node", name):
        return None
    for arg in command[1:]:
        if arg in {"-c", "-m", "-X", "-W"}:
            return None
        if not arg.startswith("-"):
            return arg if not Path(arg).is_absolute() else None
    return None


def inspect(project: Project, *, check_codex: bool = True) -> dict:
    problems, checks = [], {}
    revision = None
    if check_codex:
        try:
            checks["codex_executable"] = codex_executable()
        except (OSError, WorkerFailure):
            problems.append(issue("codex", "请安装可用的 Codex CLI，或设置 LOOP_CODEX_BINARY 后重试。"))
    config = project.command_experiment
    environment = project.environment
    if not config:
        for name in ("numpy", "scipy", "sklearn"):
            if importlib.util.find_spec(name) is None:
                problems.append(issue("runtime", "请为控制器 Python 安装示例所需依赖。", name))
        return {"status": "blocked" if problems else "ready", "checks": checks, "issues": problems, "revision": revision}

    repository = Path(config.repository).expanduser().resolve()
    if not repository.is_dir() or not shutil.which("git"):
        problems.append(issue("command_experiment.repository", "请提供可读取的 Git 实验目录，并确认 git 可用。"))
    else:
        try:
            revision = git(repository, "rev-parse", "--verify", "--end-of-options", config.revision + "^{commit}").decode().strip()
        except (OSError, subprocess.SubprocessError):
            problems.append(issue("command_experiment.revision", "请提交实验配置和脚本，或提供已有的 Git 提交。"))

    command = config.command
    executable = config.bridge_interpreter or command[0]
    runtime_path = os.defpath + ":" + str(Path(sys.executable).parent)
    runner = shutil.which(executable, path=runtime_path)
    relative_program = executable.startswith("./") and not config.bridge_interpreter
    if not runner and not relative_program:
        field = "command_experiment.bridge_interpreter" if config.bridge_interpreter else "command_experiment.command"
        problems.append(issue(field, "请提供当前运行环境可执行的命令或解释器绝对路径。", executable))
    elif runner:
        checks["experiment_executable"] = runner
    if config.bridge_interpreter:
        bundled = Path(checks["codex_executable"]).parent.parent / "codex-resources" / "bwrap" if "codex_executable" in checks else None
        if not sys.platform.startswith("linux") or not (shutil.which("bwrap") or (bundled and os.access(bundled, os.X_OK))):
            problems.append(issue("runtime", "隔离实验需要 Linux 和可执行的 bubblewrap。请补齐运行环境。"))

    if revision:
        checks["source_revision"] = revision
        try:
            manifest = committed_path(config.split_manifest)
            raw = git(repository, "archive", "--format=tar", revision, "--", manifest)
            with TemporaryDirectory(prefix="loop-preflight-") as folder:
                archive = Path(folder) / "split.tar"
                archive.write_bytes(raw)
                splits = read_splits(archive, manifest)
            checks["split"] = "grouped" if "folds" in splits else "train/validation/test"
            if command == ["isolated-predict"] and "folds" in splits:
                raise ValueError("候选模型接入需要固定的训练、验证、测试划分")
        except (ValueError, OSError, KeyError, tarfile.TarError, subprocess.SubprocessError) as exc:
            detail = str(exc) if isinstance(exc, ValueError) else config.split_manifest
            problems.append(issue("command_experiment.split_manifest", "请提供已提交的划分清单，并修正缺失或重叠的样本划分。", detail))

        required = list(environment.committed_files) if environment else []
        if script := script_input(command):
            required.append(script)
        if relative_program:
            required.append(executable)
        if command == ["isolated-predict"]:
            required.extend(PREDICTION_FILES)
        elif command == ["isolated-native"]:
            required.append("native-study.json")
            required.extend(f"native/{name}" for name in config.mutable_sources)
        elif config.bridge_interpreter:
            required.extend(("fit.py", "predict.py", "score_folds.py") if command[0] == "isolated-fold"
                            else ("predictor.py", "scorer.py", "predict-input.json", "score-input.json"))
        for name in dict.fromkeys(required):
            try:
                member = committed_path(name)
                row = git(repository, "ls-tree", revision, "--", member).decode()
                mode = row.split(" ", 1)[0]
                if mode not in {"100644", "100755"} or (name == executable and mode != "100755"):
                    raise ValueError("需要已提交的普通文件或可执行入口")
            except (ValueError, OSError, subprocess.SubprocessError):
                field = "command_experiment.command" if name in {executable, script_input(command)} else "environment.committed_files"
                problems.append(issue(field, "请补齐仓库中的脚本或输入文件，或填写其实际位置。", name))

    paths = list(environment.local_paths) if environment else []
    if command[0] == "isolated-fold":
        paths.append(command[1])
    for name in dict.fromkeys(paths):
        path = Path(name).expanduser()
        if not path.is_absolute() or not os.access(path, os.R_OK | (os.X_OK if path.is_dir() else 0)):
            problems.append(issue("environment.local_paths", "请提供可读的绝对路径或挂载实验依赖的数据。", name))
    if environment and environment.python_modules and runner:
        for name in dict.fromkeys(environment.python_modules):
            if not name.isidentifier() or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(runner).name):
                problems.append(issue("environment.python_modules", "请声明顶层模块名并配置直接调用的 Python 解释器。", name))
                continue
            try:
                result = subprocess.run([runner, "-I", "-c",
                    "import importlib, sys; importlib.import_module(sys.argv[1])", name],
                    cwd="/", env={"PATH": os.defpath, "HOME": "/", "LANG": "C.UTF-8",
                                  "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
                result.check_returncode()
            except (OSError, subprocess.SubprocessError):
                problems.append(issue("environment.python_modules",
                                      "请检查实验解释器能否导入所需模块，并修复缺失的包或运行库。", name))
    if environment:
        for program in dict.fromkeys(environment.programs):
            if not shutil.which(program, path=runtime_path):
                problems.append(issue("environment.programs", "请安装依赖程序或给出绝对路径。", program))
        if environment.gpu_count:
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                                 stderr=subprocess.DEVNULL, timeout=10)
                available = sum(line.strip().isdigit() for line in output.splitlines())
            except (OSError, subprocess.SubprocessError):
                available = 0
            checks["nvidia_gpus"] = available
            if available < environment.gpu_count:
                problems.append(issue("environment.gpu_count", "请配置所需的 NVIDIA GPU，或确认可接受的设备数。",
                                      f"检测到 {available} 张，配置要求 {environment.gpu_count} 张"))
    return {"status": "blocked" if problems else "ready", "checks": checks, "issues": problems, "revision": revision}
