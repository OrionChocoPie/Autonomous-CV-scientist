"""Versioned CLI responses and bounded project evidence for external agents."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

from .journal import redact


class CommandError(Exception):
    def __init__(self, code, message, exit_code=2):
        super().__init__(message)
        self.code, self.exit_code = code, exit_code


def json_data(value):
    if hasattr(value, "to_dict"): value = value.to_dict()
    elif is_dataclass(value): value = asdict(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(key): json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_data(item) for item in value]
    return value


def response_json(command, data=None, error=None):
    value = {"schema_version": 1, "command": command, "ok": error is None,
             "data": data if error is None else None,
             "error": {"code": error.code, "message": str(error)} if error else None}
    return json.dumps(redact(json_data(value)), ensure_ascii=False, allow_nan=False)


def history_limit(value):
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("History limit must be between 1 and 100")
    return value


def referenced(value, kind):
    return {**value.to_dict(), "reference": {"kind": kind, "id": value.id}}


def experiment_summary(exp):
    if exp is None: return None
    return {key: value for key, value in referenced(exp, "experiment").items()
            if key in ("id", "name", "parent_id", "status", "metrics", "dataset_version", "code_revision", "run_dir", "reference")}


def research_context(workspace, limit=5):
    limit = history_limit(limit)
    memory = workspace.memory
    status = memory.experiment_summary(workspace.primary_metric)
    status["best"] = experiment_summary(status["best"])
    status["latest"] = experiment_summary(status["latest"])
    status["original_baseline"] = experiment_summary(memory.original_baseline())
    status["working_baseline"] = experiment_summary(memory.working_baseline())
    groups = {
        "recent_experiments": (memory.list_experiments(), "experiment"),
        "recent_observations": (memory.list_observations(), "observation"),
        "open_questions": ([q for q in memory.list_questions() if q.status in ("open", "investigating", "blocked")], "question"),
        "active_hypotheses": (memory.list_hypotheses(status="active"), "hypothesis"),
        "untested_hypotheses": (memory.list_hypotheses(status="untested"), "hypothesis"),
        "recent_conclusions": (memory.list_conclusions(), "conclusion"),
        "recent_notes": (memory.list_notes(), "note"),
        "latest_regressions": (memory.list_regressions(), "regression"),
    }
    data = {"project": {"name": workspace.name, "root": workspace.root, "workspace": workspace.dir,
                        "primary_metric": workspace.primary_metric}, "status": status, "limit_per_group": limit,
            "active_session": memory.active_session(), "truncated": {}}
    for name, (objects, kind) in groups.items():
        selected = list(reversed(objects[-limit:]))
        data[name] = [experiment_summary(obj) if kind == "experiment" else referenced(obj, kind) for obj in selected]
        data["truncated"][name] = len(objects) > limit
    return data


def search_history(workspace, query, limit=5):
    limit = history_limit(limit)
    found = workspace.memory.search(query, limit=limit + 1)
    kinds = {"experiments": "experiment", "observations": "observation", "hypotheses": "hypothesis",
             "conclusions": "conclusion", "questions": "question", "notes": "note", "regressions": "regression"}
    return {"query": query, "limit_per_group": limit,
            "results": {name: [referenced(obj, kinds[name]) for obj in objects[:limit]] for name, objects in found.items()},
            "truncated": {name: len(objects) > limit for name, objects in found.items()}}
