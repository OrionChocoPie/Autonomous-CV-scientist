"""Local operation metadata and atomic JSON files inside the project workspace."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


def local_path(workspace, relative):
    # Preserve the root spelling: macOS /var and /private/var are aliases.
    root = Path(workspace.root)
    path = Path(workspace.dir) / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Workspace paths must be relative")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink(): raise ValueError("Workspace output paths must not traverse symlinks")
    return path


def write_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(raw); file.flush(); os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def redact(value):
    """Exclude credentials; callers log metadata rather than prompts or file contents."""
    if isinstance(value, dict):
        return {str(key): "[redacted]" if re.search(r"secret|token|password|credential|api.?key|authorization", str(key), re.I)
                and str(key) not in ("total_tokens", "prompt_tokens", "completion_tokens", "reserved_tokens") else redact(item)
                for key, item in value.items()}
    if isinstance(value, list): return [redact(item) for item in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if secret and len(secret) >= 4 and re.search(r"secret|token|password|api.?key", key, re.I):
                value = value.replace(secret, "[redacted]")
    return value


def record_event(workspace, operation, status, **metadata):
    path = local_path(workspace, "logs/operations.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"version": 1, "id": uuid.uuid4().hex, "time": datetime.now(timezone.utc).isoformat(),
             "operation": operation, "status": status, **redact(metadata)}
    raw = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(raw); file.flush(); os.fsync(file.fileno())
    return event
