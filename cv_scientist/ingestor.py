"""Manual experiment ingestion (§45) + ExperimentSource abstraction (§60).

Users keep training however they like (`python train.py ...`), then
`cv-scientist ingest runs/e147`. Initial source: local dirs; W&B/MLflow/TB
implement the same `ExperimentSource` protocol later.
"""
from __future__ import annotations
import glob, hashlib, json, math, os, re
from pathlib import Path
from typing import Protocol, runtime_checkable
import yaml
from .models import Experiment, Provenance

@runtime_checkable
class ExperimentSource(Protocol):
    def list_runs(self) -> list: ...
    def load_run(self, run_ref: str) -> Experiment: ...

def _load_json(path):
    with open(path) as file: value = json.load(file)
    if not isinstance(value, dict): raise ValueError(f"Expected an object in {Path(path).name}")
    return value
def _load_simple_yaml(path) -> dict:
    try:
        with open(path) as file: value = yaml.safe_load(file)
    except yaml.YAMLError as error: raise ValueError(f"Invalid YAML in {Path(path).name}") from error
    if not isinstance(value, dict): raise ValueError(f"Expected an object in {Path(path).name}")
    return value

def _metric_table(value, name):
    if not isinstance(value, dict): raise ValueError(f"{name} must be an object")
    for key, number in value.items():
        if type(number) not in (int, float) or not math.isfinite(number): raise ValueError(f"{name}.{key} must be a finite number")
    return {key: float(number) for key, number in value.items()}

def run_fingerprint(run_dir):
    digest = hashlib.sha256()
    for name in ("config.json", "config.yaml", "config.yml", "metrics.json", "predictions.json", "experiment.json", "source.json"):
        path = Path(run_dir) / name
        digest.update(name.encode())
        if path.exists():
            with path.open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""): digest.update(block)
        else: digest.update(b"<missing>")
    return digest.hexdigest()

def _is_import_copy(run_dir):
    root = Path(run_dir)
    if ".cv-scientist-import" in root.parts: return True
    # Recognize our markers without reserving these filenames in legacy formats.
    markers = (("source.json", rb'"format"\s*:\s*"cv-scientist-import"'),
               ("experiment.json", rb'"source"\s*:\s*"normalized_import"'))
    for name, marker in markers:
        path = root / name
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as file:
                if re.search(marker, file.read(2 * 1024 * 1024)): return True
    return False

def _load_import_copy(run_dir):
    root = Path(run_dir).resolve()
    if (root / "source.json").is_symlink(): raise ValueError("Imported manifest must not be a symlink")
    manifest = json.loads((root / "source.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "cv-scientist-import" or manifest.get("version") != 1:
        raise ValueError("Unsupported normalized import format")
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict) or not {"experiment.json", "config.json", "metrics.json"} <= hashes.keys():
        raise ValueError("Incomplete normalized import manifest")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]*", name): raise ValueError("Invalid import filename")
        path = root / name
        if path.is_symlink(): raise ValueError("Imported copy must not contain symlinks")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError("Imported copy was modified: " + name)
    data = json.loads((root / "experiment.json").read_text())
    if data.get("id") != manifest.get("experiment_id") or data.get("source") != "normalized_import":
        raise ValueError("Invalid imported experiment identity")
    exp = Experiment.from_dict(data)
    exp.run_dir = str(root)
    return exp

class LocalExperimentSource:
    """Reads a run dir: config.(yaml|json), metrics.json, checkpoint, logs."""
    def __init__(self, runs_root: str = "."): self.runs_root = runs_root
    def list_runs(self):
        if not os.path.isdir(self.runs_root): raise FileNotFoundError(self.runs_root)
        if os.path.isfile(os.path.join(self.runs_root, "metrics.json")): return [self.runs_root]
        return sorted(d for d in glob.glob(os.path.join(self.runs_root, "*")) if os.path.isdir(d))
    def load_run(self, run_ref: str) -> Experiment:
        run_dir = run_ref if os.path.isdir(run_ref) else os.path.join(self.runs_root, run_ref)
        if not os.path.isdir(run_dir): raise FileNotFoundError(run_dir)
        if _is_import_copy(run_dir):
            return _load_import_copy(run_dir)
        config = {}
        for pat, loader in (("config.json", _load_json), ("config.yaml", _load_simple_yaml), ("config.yml", _load_simple_yaml)):
            p = os.path.join(run_dir, pat)
            if os.path.exists(p): config = loader(p); break
        metrics_data = _load_json(os.path.join(run_dir, "metrics.json"))
        metadata = {"id", "parent_id", "slices", "per_class", "code_revision", "dataset_version"}
        metrics = _metric_table(metrics_data.get("metrics", {k: v for k, v in metrics_data.items() if k not in metadata}), "metrics")
        groups = {}
        for kind in ("slices", "per_class"):
            table = metrics_data.get(kind, {})
            if not isinstance(table, dict): raise ValueError(f"{kind} must be an object")
            groups[kind] = {name: _metric_table(values, f"{kind}.{name}") for name, values in table.items()}
        if not metrics and not any(values for table in groups.values() for values in table.values()):
            raise ValueError("Run contains no measured metrics")
        ckpts = []
        for ext in ("*.pt", "*.pth", "*.ckpt", "*.onnx"):
            ckpts += glob.glob(os.path.join(run_dir, "**", ext), recursive=True)
        exp_id = config.get("experiment_id") or metrics_data.get("id") or os.path.basename(os.path.abspath(run_dir))
        return Experiment(id=str(exp_id), name=str(config.get("name", exp_id)),
            parent_id=config.get("parent") or config.get("parent_experiment") or metrics_data.get("parent_id"),
            parent_checkpoint=config.get("parent_checkpoint"), checkpoint_path=ckpts[0] if ckpts else None,
            run_dir=os.path.abspath(run_dir), status="completed", config={k: v for k, v in config.items()},
            metrics=metrics, slices=groups["slices"], per_class=groups["per_class"],
            code_revision=config.get("code_revision") or metrics_data.get("code_revision"),
            dataset_version=config.get("dataset_version") or metrics_data.get("dataset_version"),
            source="local", provenance=Provenance.EXTERNAL_IMPORT.value)

# Placeholder stubs so future trackers slot in without redesign (§60)
class WeightsAndBiasesExperimentSource:  # pragma: no cover
    def list_runs(self): raise NotImplementedError("W&B source not yet implemented; use LocalExperimentSource")
    def load_run(self, r): raise NotImplementedError
class MLflowExperimentSource:  # pragma: no cover
    def list_runs(self): raise NotImplementedError("MLflow source not yet implemented")
    def load_run(self, r): raise NotImplementedError
class TensorBoardExperimentSource:  # pragma: no cover
    def list_runs(self): raise NotImplementedError("TensorBoard source not yet implemented")
    def load_run(self, r): raise NotImplementedError

class RunIngestor:
    """ExperimentImporter (§45). Ingest -> memory; analysis is RunAnalysisService's job."""
    def __init__(self, memory, source: ExperimentSource | None = None):
        self.memory = memory; self.source = source or LocalExperimentSource()
    def ingest(self, run_ref: str) -> Experiment:
        exp = self.source.load_run(run_ref)
        with self.memory.transaction():
            existing = self.memory.get_experiment(exp.id)
            if existing:
                exp.created_at = existing.created_at
                if exp.parent_id is None: exp.parent_id = existing.parent_id
                if exp.to_dict() == existing.to_dict(): return existing
            elif not exp.parent_id and exp.source != "normalized_import":
                prev = self.memory.latest_experiment()
                if prev and prev.id != exp.id: exp.parent_id = prev.id
            self.memory.add_experiment(exp)
        return exp
    # alias
    import_run = ingest
ExperimentImporter = RunIngestor
