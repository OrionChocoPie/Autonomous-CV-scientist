"""Execute single-parameter detection evaluations against frozen project inputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess

from .ingestor import LocalExperimentSource
from .predictions import validate_predictions, match_boxes
from .journal import local_path


class EvaluationError(RuntimeError):
    pass


def confidence_value(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number between 0 and 1")
    return round(float(value), 6)


def imgsz_value(value):
    if type(value) is not int or not 32 <= value <= 4096 or value % 32:
        raise ValueError("imgsz must be an integer multiple of 32 between 32 and 4096")
    return value


def _json(path):
    return json.loads(path.read_text())


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Evaluation paths must be relative to the project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError("Evaluation path escapes the project")
    if not path.is_file(): raise FileNotFoundError(path)
    return path


class ConfidenceEvaluator:
    """The model selects a threshold, never a command, executable or output path."""
    FIXED = ("model", "weights_sha256", "imgsz", "iou_match", "device", "seed", "dataset_version", "validation_images", "small_slice")
    PARAMETER = "confidence"
    METADATA = {"experiment_id", "name", "parent", "parent_experiment", "research_signature"}
    validate_value = staticmethod(confidence_value)

    def result_directory(self, run_id): return self.root / "runs" / run_id

    def __init__(self, workspace, baseline):
        self.ws, self.baseline = workspace, baseline
        self.root = Path(workspace.root).resolve()
        missing = [key for key in (*self.FIXED, self.PARAMETER) if key not in baseline.config]
        if missing: raise ValueError("Baseline config is missing: " + ", ".join(missing))
        if not isinstance(baseline.run_dir, str): raise ValueError("Baseline needs a local run directory")
        self.fixed = {key: deepcopy(value) for key, value in baseline.config.items()
                      if key not in self.METADATA | {self.PARAMETER}}
        setattr(self, "baseline_" + self.PARAMETER, self.validate_value(baseline.config[self.PARAMETER]))
        if self.PARAMETER == "imgsz": confidence_value(self.fixed["confidence"])
        self.script = _inside(self.root, workspace.config.get("evaluation_script"))
        python = workspace.config.get("evaluation_python", ".venv/bin/python")
        if not isinstance(python, str) or Path(python).is_absolute() or ".." in Path(python).parts:
            raise ValueError("evaluation_python must be a project-relative virtualenv executable")
        self.python = self.root / python
        if not self.python.is_file(): raise FileNotFoundError(self.python)
        self.weights = _inside(self.root, self.fixed["model"])
        if _digest(self.weights) != self.fixed["weights_sha256"]:
            raise EvaluationError("Baseline weights do not match their recorded SHA256")
        dataset = workspace.config.get("dataset_root")
        if not isinstance(dataset, str): raise ValueError("Configure dataset_root in project.yaml")
        self.manifest_path = _inside(self.root, str(Path(dataset) / "manifest.json"))
        self.dataset = self.manifest_path.parent
        manifest = _json(self.manifest_path)
        val = manifest.get("val")
        if not isinstance(val, list) or not val: raise ValueError("Validation split must not be empty")
        if len(val) != self.fixed["validation_images"]: raise EvaluationError("Validation image count differs from baseline")
        self.truth = {}
        files = [self.script, *sorted(self.script.parent.glob("*.py")), self.weights, self.manifest_path,
                 _inside(self.root, str(Path(dataset) / "data.yaml"))]
        train_names = {Path(row["image"]).name for row in manifest.get("train", [])}
        for row in val:
            image = row["image"]
            if image in self.truth or Path(image).name in train_names: raise EvaluationError("Validation images overlap or are duplicated")
            if not isinstance(row.get("boxes"), list): raise ValueError("Validation records need boxes")
            self.truth[image] = row["boxes"]
            for key in ("image", "label"):
                path = _inside(self.dataset, row[key])
                files.append(path)
        self.files = list(dict.fromkeys(files))
        self.snapshot = {str(path.relative_to(self.root)): _digest(path) for path in self.files}
        self.signature = hashlib.sha256(json.dumps({"baseline_id": baseline.id, "files": self.snapshot, "settings": self.fixed}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self._check_records(_json(Path(baseline.run_dir) / "predictions.json"), baseline.metrics)

    def check_inputs(self):
        for path in self.files:
            if not path.is_file() or _digest(path) != self.snapshot[str(path.relative_to(self.root))]:
                raise EvaluationError("Frozen evaluation inputs changed: " + str(path.relative_to(self.root)))

    def _check_records(self, rows, metrics):
        rows = validate_predictions(rows)
        if not isinstance(rows, list) or len(rows) != len(self.truth): raise EvaluationError("Prediction image count differs from fixed validation")
        found = set()
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for row in rows:
            if not isinstance(row, dict): raise EvaluationError("Prediction records must be objects")
            image = row.get("image")
            if image in found or image not in self.truth or row.get("truth") != self.truth[image]:
                raise EvaluationError("Predictions do not match fixed validation ground truth")
            found.add(image)
            for key in totals:
                value = row.get(key)
                if type(value) is not int or value < 0: raise EvaluationError("Invalid prediction counts")
                totals[key] += value
            if row["tp"] + row["fn"] != len(row["truth"]): raise EvaluationError("Prediction TP/FN counts contradict ground truth")
            if not isinstance(row.get("predictions"), list) or row["tp"] + row["fp"] != len(row["predictions"]):
                raise EvaluationError("Prediction TP/FP counts contradict detections")
            truth, preds, _ = match_boxes(row, self.fixed["iou_match"])
            actual = {"tp": len(preds), "fp": len(row["predictions"]) - len(preds), "fn": len(row["truth"]) - len(truth)}
            if any(row[key] != value for key, value in actual.items()):
                mode = "class-aware" if rows.class_aware else "class-agnostic"
                raise EvaluationError("Exported counts contradict " + mode + " matching")
        for key, value in totals.items():
            if type(metrics.get(key)) not in (int, float) or metrics[key] != value: raise EvaluationError("Exported totals contradict per-image counts")
        tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
        for key, expected in (("precision", tp / (tp + fp) if tp + fp else 0), ("recall", tp / (tp + fn) if tp + fn else 0)):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value - expected) > 1e-9:
                raise EvaluationError("Exported precision/recall contradict detection counts")
        if metrics.get("count") != len(self.truth): raise EvaluationError("Exported image count differs from validation")

    def load_result(self, run, run_id, confidence):
        config, export = _json(run / "config.json"), _json(run / "metrics.json")
        if not isinstance(config, dict) or not isinstance(export, dict): raise EvaluationError("Evaluation exports must be objects")
        if config.get("experiment_id") != run_id or export.get("id") != run_id or config.get("parent") != self.baseline.id:
            raise EvaluationError("Executor returned incorrect experiment identity/lineage")
        if self.validate_value(config.get(self.PARAMETER)) != confidence: raise EvaluationError("Executor changed the requested " + self.PARAMETER)
        if set(config) - (set(self.baseline.config) | self.METADATA | {self.PARAMETER}):
            raise EvaluationError("Executor added an uncontrolled parameter")
        if any(key not in config or config[key] != value for key, value in self.fixed.items()):
            raise EvaluationError("Executor changed a frozen parameter")
        self._check_records(_json(run / "predictions.json"), export.get("metrics", {}))
        return LocalExperimentSource().load_run(str(run))

    def recover_result(self, run_id, confidence):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id): raise ValueError("Invalid evaluation run ID")
        run = self.result_directory(run_id)
        marker = _json(run / ".evaluation-complete.json")
        if marker.get("signature") != self.signature or marker.get("experiment_id") != run_id:
            raise EvaluationError("Evaluation has no matching completion receipt")
        for name in ("config.json", "metrics.json", "predictions.json"):
            if marker.get("sha256", {}).get(name) != _digest(run / name): raise EvaluationError("Completed evaluation artifacts changed")
        self.check_inputs()
        return self.load_result(run, run_id, confidence)

    def evaluate(self, confidence, run_id, timeout, log_path):
        confidence = self.validate_value(confidence)
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id): raise ValueError("Invalid evaluation run ID")
        run = self.root / "runs" / run_id
        destination = self.result_directory(run_id)
        if run.exists() or destination.exists() or self.ws.memory.get_experiment(run_id): raise FileExistsError(run)
        self.check_inputs()
        command = [str(self.python), str(self.script), "evaluate", "--run", run_id,
                   "--parent", self.baseline.id, "--weights", str(self.weights),
                   "--imgsz", str(confidence if self.PARAMETER == "imgsz" else self.fixed["imgsz"]),
                   "--conf", str(confidence if self.PARAMETER == "confidence" else self.fixed["confidence"]), "--device", self.fixed["device"]]
        environment = {key: value for key, value in os.environ.items() if key != "CV_SCIENTIST_API_KEY"}
        try:
            with Path(log_path).open("x") as log:
                with subprocess.Popen(command, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        callback = getattr(self, "on_start", None)
                        if callback: callback(process.pid)
                        process.wait(timeout=timeout)
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
        except subprocess.TimeoutExpired:
            raise EvaluationError(f"Evaluation timed out; log: {log_path}") from None
        if process.returncode: raise EvaluationError(f"Evaluation exited with code {process.returncode}; log: {log_path}")
        self.check_inputs()
        result = self.load_result(run, run_id, confidence)
        result.config["research_signature"] = self.signature
        (run / "config.json").write_text(json.dumps(result.config, indent=2, allow_nan=False) + "\n")
        if destination != run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists(): raise FileExistsError(destination)
            run.rename(destination)
            run = destination
            result.run_dir = str(run)
        receipt = {"signature": self.signature, "experiment_id": run_id,
                   "sha256": {name: _digest(run / name) for name in ("config.json", "metrics.json", "predictions.json")}}
        temporary = run / ".evaluation-complete.tmp"
        temporary.write_text(json.dumps(receipt, allow_nan=False))
        temporary.replace(run / ".evaluation-complete.json")
        return result


class ResolutionEvaluator(ConfidenceEvaluator):
    """Change only input resolution using the same verified evaluation protocol."""
    PARAMETER = "imgsz"
    FIXED = tuple(key for key in ConfidenceEvaluator.FIXED if key != "imgsz") + ("confidence",)
    validate_value = staticmethod(imgsz_value)

    def result_directory(self, run_id): return local_path(self.ws, "runs/" + run_id)
