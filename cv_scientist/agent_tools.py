"""Read-only CV research tools with evidence references and image attachments."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path

from .compare import compare_runs
from .diagnose import diagnose
from .query import ProjectQueryService


@dataclass
class ToolResult:
    data: dict
    images: list = field(default_factory=list)


def _tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}


_STRING = {"type": "string"}
TOOLS = [
    _tool("list_experiments", "List imported experiments, their metrics and lineage.", {"limit": {"type": "integer", "minimum": 1, "maximum": 20}}),
    _tool("get_experiment", "Read full configuration, metrics, per-class and slice metrics.", {"experiment_id": _STRING}),
    _tool("compare_experiments", "Compare current to baseline using measured metrics and regressions.", {"baseline_id": _STRING, "current_id": _STRING}),
    _tool("diagnose_experiment", "Run deterministic CV diagnostics on an imported experiment.", {"experiment_id": _STRING}),
    _tool("search_memory", "Find prior experiments, observations, hypotheses, conclusions and user notes.", {"query": _STRING}),
    _tool("get_error_examples", "Read examples from predictions.json, sorted by FP/FN count; returns GT and predicted boxes.",
          {"experiment_id": _STRING, "kind": {"type": "string", "enum": ["fp", "fn", "any"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 8}}),
    _tool("view_error", "View an original image referenced by predictions.json, together with its GT and predicted boxes. Use an image value returned by get_error_examples.",
          {"experiment_id": _STRING, "image": _STRING}),
]


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} must be a nonempty string of at most 2048 characters")
    return value


def _limit(value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"limit must be an integer between 1 and {maximum}")
    return value


class AnalysisTools:
    def __init__(self, workspace):
        self.ws = workspace
        self.memory = workspace.memory
        self.evidence = {}

    def _ref(self, kind, identity, detail=""):
        ref = {"kind": kind, "id": identity, "detail": detail}
        self.evidence[kind, identity] = ref
        return ref

    def _experiment(self, identity):
        identity = _text(identity, "experiment_id")
        exp = self.memory.get_experiment(identity)
        if exp is None: raise ValueError(f"Unknown experiment: {identity}")
        self._ref("experiment", exp.id)
        return exp

    def list_experiments(self, limit):
        experiments = self.memory.list_experiments()[-_limit(limit, 20):]
        return {"experiments": [{"id": e.id, "name": e.name, "parent_id": e.parent_id, "metrics": e.metrics,
                                 "evidence": self._ref("experiment", e.id)} for e in experiments]}

    def get_experiment(self, experiment_id):
        return self._experiment(experiment_id).to_dict()

    def compare_experiments(self, baseline_id, current_id):
        baseline, current = self._experiment(baseline_id), self._experiment(current_id)
        comparison = compare_runs(baseline, current).to_dict()
        # The comparison is calculated on demand; only the stored experiments are cited.
        comparison.pop("id")
        comparison["evidence"] = [self._ref("experiment", e.id) for e in (baseline, current)]
        return comparison

    def diagnose_experiment(self, experiment_id):
        exp = self._experiment(experiment_id)
        baseline = self.memory.working_baseline()
        if baseline: self._ref("experiment", baseline.id)
        return diagnose(exp, baseline if baseline and baseline.id != exp.id else None).to_dict()

    def search_memory(self, query):
        found = ProjectQueryService(self.memory).retrieve(_text(query, "query"))
        kinds = {"experiments": "experiment", "observations": "observation", "hypotheses": "hypothesis",
                 "conclusions": "conclusion", "questions": "question", "notes": "note"}
        result = {}
        for group, objects in found.items():
            result[group] = []
            for obj in objects[:5]:
                result[group].append({**obj.to_dict(), "reference": self._ref(kinds[group], obj.id)})
        return result

    def _records(self, experiment_id):
        exp = self._experiment(experiment_id)
        if not exp.run_dir: raise ValueError("Experiment has no local run directory")
        root = Path(exp.run_dir).resolve()
        path = (root / "predictions.json").resolve()
        if not path.is_relative_to(root): raise ValueError("predictions.json must stay inside the run directory")
        if path.stat().st_size > 8 * 1024 * 1024: raise ValueError("predictions.json exceeds the 8 MiB limit")
        records = json.loads(path.read_text())
        if not isinstance(records, list): raise ValueError("predictions.json must contain a list of image records")
        names = set()
        for row in records:
            if not isinstance(row, dict): raise ValueError("Prediction records must be objects")
            name = _text(row.get("image"), "image")
            if Path(name).is_absolute() or ".." in Path(name).parts or name in names:
                raise ValueError("Prediction image paths must be unique relative paths without ..")
            names.add(name)
            if not isinstance(row.get("truth"), list) or not isinstance(row.get("predictions"), list):
                raise ValueError("Prediction records need truth and predictions lists")
            if any(type(row.get(key)) is not int or row[key] < 0 for key in ("fp", "fn")):
                raise ValueError("Prediction records need nonnegative integer fp and fn")
        return exp, records

    def _example(self, exp, index, row):
        return {**row, "evidence": self._ref("artifact", f"{exp.id}:predictions.json#{index}", row["image"])}

    def get_error_examples(self, experiment_id, kind, limit):
        _limit(limit, 8)
        if kind not in ("fp", "fn", "any"): raise ValueError("kind must be fp, fn or any")
        exp, records = self._records(experiment_id)
        score = lambda row: row["fp"] + row["fn"] if kind == "any" else row[kind]
        ranked = sorted(enumerate(records), key=lambda item: -score(item[1]))
        selected = [(i, row) for i, row in ranked if score(row) > 0][:limit]
        return {"experiment_id": exp.id, "kind": kind,
                "examples": [self._example(exp, i, row) for i, row in selected]}

    def view_error(self, experiment_id, image):
        image = _text(image, "image")
        exp, records = self._records(experiment_id)
        match = next(((i, row) for i, row in enumerate(records) if row["image"] == image), None)
        if match is None: raise ValueError("Image is not referenced by this experiment's predictions.json")
        roots = [Path(exp.run_dir), Path(self.ws.root)]
        dataset_root = exp.config.get("dataset_root") or self.ws.config.get("dataset_root")
        if dataset_root:
            dataset_root = Path(dataset_root)
            roots.append(dataset_root if dataset_root.is_absolute() else Path(self.ws.root) / dataset_root)
        path = None
        for root in roots:
            root = root.resolve()
            candidate = (root / image).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                path = candidate
                break
        if path is None: raise ValueError("Image file not found; configure dataset_root in project.yaml")
        if path.stat().st_size > 2 * 1024 * 1024: raise ValueError("Image exceeds the 2 MiB limit")
        raw = path.read_bytes()
        if raw.startswith(b"\x89PNG\r\n\x1a\n"): mime = "image/png"
        elif raw.startswith(b"\xff\xd8\xff"): mime = "image/jpeg"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP": mime = "image/webp"
        else: raise ValueError("Only PNG, JPEG and WebP images can be sent to the agent")
        index, row = match
        example = self._example(exp, index, row)
        example["image_shown"] = True
        return ToolResult(example, [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}])

    def execute(self, name, arguments):
        if not isinstance(arguments, dict): raise ValueError("Tool arguments must be a JSON object")
        definition = next((tool["function"] for tool in TOOLS if tool["function"]["name"] == name), None)
        if definition is None: raise ValueError("Unknown analysis tool")
        if set(arguments) != set(definition["parameters"]["required"]): raise ValueError(f"Invalid arguments for {name}")
        result = getattr(self, name)(**arguments)
        return result if isinstance(result, ToolResult) else ToolResult(result)
