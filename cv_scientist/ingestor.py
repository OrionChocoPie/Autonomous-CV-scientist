"""Manual experiment ingestion (§45) + ExperimentSource abstraction (§60).

Users keep training however they like (`python train.py ...`), then
`cv-scientist ingest runs/e147`. Initial source: local dirs; W&B/MLflow/TB
implement the same `ExperimentSource` protocol later.
"""
from __future__ import annotations
import glob, json, os
from typing import Protocol, runtime_checkable
from .models import Experiment, Provenance

@runtime_checkable
class ExperimentSource(Protocol):
    def list_runs(self) -> list: ...
    def load_run(self, run_ref: str) -> Experiment: ...

def _load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return {}
def _load_simple_yaml(path) -> dict:
    out = {}
    try:
        for line in open(path):
            line = line.strip()
            if line and ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                v = v.strip()
                try: v = float(v) if "." in v else int(v)
                except ValueError: v = v.strip("'\"")
                out[k.strip()] = v
    except Exception: pass
    return out

class LocalExperimentSource:
    """Reads a run dir: config.(yaml|json), metrics.json, checkpoint, logs."""
    def __init__(self, runs_root: str = "."): self.runs_root = runs_root
    def list_runs(self):
        if os.path.isfile(os.path.join(self.runs_root, "metrics.json")): return [self.runs_root]
        return sorted(d for d in glob.glob(os.path.join(self.runs_root, "*")) if os.path.isdir(d))
    def load_run(self, run_ref: str) -> Experiment:
        run_dir = run_ref if os.path.isdir(run_ref) else os.path.join(self.runs_root, run_ref)
        if not os.path.isdir(run_dir): raise FileNotFoundError(run_dir)
        config = {}
        for pat, loader in (("config.json", _load_json), ("config.yaml", _load_simple_yaml), ("config.yml", _load_simple_yaml)):
            p = os.path.join(run_dir, pat)
            if os.path.exists(p): config = loader(p); break
        metrics_data = _load_json(os.path.join(run_dir, "metrics.json")) if os.path.exists(os.path.join(run_dir, "metrics.json")) else {}
        metrics = metrics_data.get("metrics", metrics_data)
        # flatten numeric metrics only
        metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        ckpts = []
        for ext in ("*.pt", "*.pth", "*.ckpt", "*.onnx"):
            ckpts += glob.glob(os.path.join(run_dir, "**", ext), recursive=True)
        exp_id = config.get("experiment_id") or metrics_data.get("id") or os.path.basename(os.path.abspath(run_dir))
        return Experiment(id=str(exp_id), name=str(config.get("name", exp_id)),
            parent_id=config.get("parent") or config.get("parent_experiment") or metrics_data.get("parent_id"),
            parent_checkpoint=config.get("parent_checkpoint"), checkpoint_path=ckpts[0] if ckpts else None,
            run_dir=os.path.abspath(run_dir), status="completed", config={k: v for k, v in config.items()},
            metrics=metrics, slices=metrics_data.get("slices", {}), per_class=metrics_data.get("per_class", {}),
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
        # Infer parent as previous best if unspecified
        if not exp.parent_id:
            prev = self.memory.latest_experiment()
            if prev and prev.id != exp.id: exp.parent_id = prev.id
        self.memory.add_experiment(exp)
        return exp
    # alias
    import_run = ingest
ExperimentImporter = RunIngestor
