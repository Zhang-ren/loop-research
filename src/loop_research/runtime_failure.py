"""Classify failed training using the process outcome and trusted runtime diagnostics."""
from __future__ import annotations

import errno
import subprocess
from typing import Literal

FailureStatus = Literal["code_failed", "infra_failed", "unclassified_failed"]
HOST_ERRORS = frozenset({errno.EIO, errno.ENOSPC, errno.EROFS, errno.EMFILE, errno.ENFILE, errno.EDQUOT, errno.ESTALE})


class TrainingFailure(ValueError):
    def __init__(self, message: str, status: FailureStatus, category: str):
        super().__init__(message)
        self.status = status
        self.category = category


def classify_failure(error: Exception) -> tuple[FailureStatus, str]:
    if isinstance(error, TrainingFailure):
        return error.status, error.category
    if isinstance(error, subprocess.TimeoutExpired):
        return "code_failed", "training_deadline"
    if isinstance(error, subprocess.CalledProcessError):
        if error.returncode < 0 or error.returncode in {126, 127}:
            return "unclassified_failed", "command_terminated_or_unavailable"
        return "code_failed", "command_failed"
    if isinstance(error, OSError):
        if error.errno in HOST_ERRORS:
            return "infra_failed", f"os_{errno.errorcode[error.errno].lower()}"
        return "unclassified_failed", "ambiguous_os_error"
    if isinstance(error, (ValueError, TypeError, LookupError, SyntaxError, ArithmeticError, AttributeError, AssertionError)):
        return "code_failed", "execution_or_output_error"
    return "unclassified_failed", "unexpected_exception"


def failed_result(error: Exception, traceback: str) -> dict:
    status, category = classify_failure(error)
    return {"status": status, "failure_category": category,
            "error": f"{type(error).__name__}: {error}", "traceback": traceback}
