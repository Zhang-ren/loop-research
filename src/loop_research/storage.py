from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Callable
from contextlib import contextmanager

from filelock import FileLock

from .models import State


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class Conflict(RuntimeError):
    pass


class StateStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "state.json"
        self.lock = FileLock(str(self.root / ".state.lock"), timeout=10)

    def read(self) -> State:
        return State.model_validate_json(self.path.read_bytes())

    def initialize(self, state: State):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.path.exists():
                raise FileExistsError(self.path)
            atomic(self.path, canonical(state.model_dump()))

    @contextmanager
    def publication_gate(self):
        with self.lock:
            if self.path.is_file() and self.read().pending_review:
                raise ValueError("Resolve the open batch disagreement before publishing research knowledge")
            yield

    def has_event(self, event_id: str, state: State) -> bool:
        if event_id in state.event_ids:
            return True
        key = digest(event_id.encode())
        path = self.root / "trigger-receipts" / f"{key}.json"
        if not path.is_file():
            return False
        revision = json.loads(path.read_bytes())["revision"]
        if revision > state.revision:
            return False
        transition = self.root / "transitions" / f"{revision:08d}.json"
        return transition.is_file() and json.loads(transition.read_bytes()).get("trigger") == key

    def change(self, operation: Callable[[State], None], *, expected: int | None = None, event="update",
               trigger: str | None = None) -> State:
        with self.lock:
            state = self.read()
            if expected is not None and state.revision != expected:
                raise Conflict(f"Expected revision {expected}, found {state.revision}")
            before = state.model_dump()
            trigger_key = digest(trigger.encode()) if trigger and not self.has_event(trigger, state) else None
            operation(state)
            if trigger_key:
                state.event_ids = (state.event_ids + [trigger])[-1000:]
            state.revision += 1
            validated = State.model_validate(state.model_dump())
            # Per-revision receipt is written first. Only state.json commits the revision.
            receipt = {"time": time.time(), "event": event, "before": digest(canonical(before)),
                       "after": digest(canonical(validated.model_dump())), "revision": state.revision}
            if trigger_key:
                receipt["trigger"] = trigger_key
                atomic(self.root / "trigger-receipts" / f"{trigger_key}.json",
                       canonical({"revision": state.revision}))
            atomic(self.root / "transitions" / f"{state.revision:08d}.json", canonical(receipt))
            atomic(self.path, canonical(validated.model_dump()))
            return validated


class Evidence:
    """Content-addressed evidence; verify on every read, never trust worker paths."""

    def __init__(self, root: Path):
        self.root = root / "evidence"
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, kind: str, content, parents: list[str] | None = None) -> str:
        value = {"kind": kind, "content": content, "parents": parents or []}
        for parent in value["parents"]:
            self.get(parent)
        raw = canonical(value)
        key = digest(raw)
        path = self.root / f"{key}.json"
        if not path.exists():
            atomic(path, raw)
        self.get(key)
        return key

    def get(self, key: str) -> dict:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid evidence ID")
        raw = (self.root / f"{key}.json").read_bytes()
        if digest(raw) != key:
            raise ValueError("Evidence integrity failure")
        return json.loads(raw)

    def verify_tree(self, key: str) -> set[str]:
        seen = set()

        def walk(current):
            if current in seen:
                return
            value = self.get(current)
            seen.add(current)
            for parent in value["parents"]:
                walk(parent)

        walk(key)
        return seen

    def import_tree(self, source, key: str) -> None:
        for item in source.verify_tree(key):
            name = f"{item}.json"
            path = self.root / name
            if not path.exists():
                atomic(path, (source.root / name).read_bytes())
            self.get(item)
