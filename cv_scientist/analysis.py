"""Automatic run analysis pipeline (§57): Run -> Metrics -> Comparison ->
Regressions -> Error analysis -> Observations -> Memory. Independent of ResearchEngine."""
from __future__ import annotations
from dataclasses import dataclass, field
from .models import Observation, Provenance, EvidenceRef
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

class RunAnalysisService:
    def __init__(self, memory): self.memory = memory
    def analyze(self, experiment_id: str, baseline_id: str | None = None) -> RunAnalysis:
        exp = self.memory.get_experiment(experiment_id)
        if not exp: raise KeyError(experiment_id)
        base = self.memory.get_experiment(baseline_id) if baseline_id else self.memory.working_baseline()
        if base and base.id == exp.id: base = None
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
        return result
