"""Automatic run analysis pipeline (§57): Run -> Metrics -> Comparison ->
Regressions -> Error analysis -> Observations -> Memory. Independent of ResearchEngine."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from .models import Observation, Provenance, ExperimentComparison, RegressionFinding, Diagnosis
from .ingestor import run_fingerprint
from .compare import compare_runs
from .diagnose import diagnose
from . import diagnostics

@dataclass
class RunAnalysis:
    experiment_id: str = ""
    comparison: object = None
    regressions: list = field(default_factory=list)
    diagnosis: object = None
    observations: list = field(default_factory=list)
    def to_dict(self):
        return {"experiment_id": self.experiment_id,
                "comparison": self.comparison.to_dict() if self.comparison else None,
                "regressions": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.regressions],
                "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
                "observations": [o.to_dict() if hasattr(o, "to_dict") else o for o in self.observations]}
    @classmethod
    def from_dict(cls, data):
        return cls(experiment_id=data["experiment_id"],
                   comparison=ExperimentComparison.from_dict(data["comparison"]) if data["comparison"] else None,
                   regressions=[RegressionFinding.from_dict(r) for r in data["regressions"]],
                   diagnosis=Diagnosis.from_dict(data["diagnosis"]) if data["diagnosis"] else None,
                   observations=[Observation.from_dict(o) for o in data["observations"]])

class RunAnalysisService:
    def __init__(self, memory): self.memory = memory
    def analyze(self, experiment_id: str, baseline_id: str | None = None) -> RunAnalysis:
        with self.memory.transaction(): return self._analyze(experiment_id, baseline_id)
    def _analyze(self, experiment_id, baseline_id):
        exp = self.memory.get_experiment(experiment_id)
        if not exp: raise KeyError(experiment_id)
        base = self.memory.get_experiment(baseline_id) if baseline_id else self.memory.working_baseline()
        if base and base.id == exp.id: base = None
        signature = hashlib.sha256(json.dumps({"experiment": exp.to_dict(), "baseline": base.to_dict() if base else None,
            "files": run_fingerprint(exp.run_dir) if exp.run_dir else None}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        cache_key = f"analysis:{exp.id}:{base.id if base else ''}"
        cached = self.memory.get_meta(cache_key)
        if cached:
            data = json.loads(cached)
            if data["signature"] == signature: return RunAnalysis.from_dict(data["result"])
        result = RunAnalysis(experiment_id=exp.id)
        result.diagnosis = diagnose(exp, base)
        for f in result.diagnosis.findings[:5]:
            if f.severity in ("high", "critical", "medium"):
                o = self.memory.add_observation(Observation(title=f.title, description=f"{f.category} {f.metric}={f.value} slice={f.slice}",
                    kind=f.category, provenance=Provenance.DIAGNOSTIC_TOOL.value, evidence=f.evidence, experiment_id=exp.id))
                result.observations.append(o)
        if base:
            result.comparison = compare_runs(base, exp)
            self.memory.add_comparison(result.comparison)
            result.regressions = diagnostics.find_regressions(exp, base)
            for r in result.regressions: self.memory.add_regression(r)
        self.memory.set_meta(cache_key, json.dumps({"signature": signature, "result": result.to_dict()}, allow_nan=False))
        return result
