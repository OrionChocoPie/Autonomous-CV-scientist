"""Standalone diagnosis (§48): deterministic diagnostics first, LLM prioritization optional."""
from __future__ import annotations
from .models import Experiment, Diagnosis
from . import diagnostics

def diagnose(exp: Experiment, baseline: Experiment | None = None) -> Diagnosis:
    findings = []
    findings += diagnostics.analyze_small_objects(exp)
    findings += diagnostics.analyze_false_negatives(exp)
    findings += diagnostics.analyze_false_positives(exp)
    findings += diagnostics.cluster_errors(exp)
    findings += diagnostics.audit_annotations(exp)
    if baseline: findings += diagnostics.analyze_dataset_shift(exp, baseline)
    # Dedupe by (category, slice), keep highest share
    best = {}
    for f in findings:
        k = (f.category, f.slice)
        if k not in best or (f.share or 0) > (best[k].share or 0): best[k] = f
    ranked = sorted(best.values(), key=lambda f: ({"critical": 0, "high": 1, "medium": 2, "low": 3}.get(f.severity, 2), -(f.share or 0)))
    return Diagnosis(experiment_id=exp.id, findings=ranked)

def summarize(d: Diagnosis, limit: int = 10) -> str:
    lines = [f"Diagnosis for {d.experiment_id}:"]
    for i, f in enumerate(d.findings[:limit], 1):
        extra = f" {f.metric}={f.value:.3f}" if isinstance(f.value, float) else (f" share={f.share:.0%}" if f.share is not None else "")
        lines.append(f"{i}. [{f.severity}] {f.title}{extra}")
    return "\n".join(lines) if d.findings else lines[0] + "\n(no significant findings)"
