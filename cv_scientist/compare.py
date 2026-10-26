"""Standalone comparison (§47): structured ExperimentComparison, LLM-independent."""
from __future__ import annotations
from .models import Experiment, ExperimentComparison, EvidenceRef
from . import diagnostics

def _delta(a: dict, b: dict) -> dict:
    out = {}
    for k in set(a) | set(b):
        if k in a and k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
            out[k] = {"a": float(a[k]), "b": float(b[k]), "abs": float(b[k] - a[k]),
                      "rel": float((b[k] - a[k]) / abs(a[k])) if a[k] else 0.0}
    return out

def compare_runs(a: Experiment, b: Experiment) -> ExperimentComparison:
    """Compare overall, per-class, per-slice, FP/FN, size/confidence buckets (via slices)."""
    regs = diagnostics.find_regressions(b, a)
    overall, per_class, per_slice = _delta(a.metrics, b.metrics), {}, {}
    for name in set(a.per_class) | set(b.per_class):
        per_class[name] = _delta(a.per_class.get(name, {}), b.per_class.get(name, {}))
    for name in set(a.slices) | set(b.slices):
        per_slice[name] = _delta(a.slices.get(name, {}), b.slices.get(name, {}))
    improvements = []
    for scope, table in (("overall", {"overall": overall}), ("class", per_class), ("slice", per_slice)):
        for name, metrics in table.items():
            for metric, d in metrics.items():
                if metric in ("fp", "fn"):
                    if d["rel"] <= -0.05: improvements.append(f"{scope}:{name} {metric} {d['rel']:+.1%}")
                elif d["rel"] >= 0.02:
                    improvements.append(f"{scope}:{name} {metric} {d['rel']:+.1%}")
    tradeoff = ""
    if improvements and regs:
        tradeoff = (f"{b.id} improves ({'; '.join(improvements[:2])}) but regresses "
                    f"({'; '.join(f'{r.slice} {r.metric} {r.relative_delta:+.1%}' for r in regs[:2])}). "
                    f"Likely trade-off: the change helping the improved slices is harming the regressed slices.")
    return ExperimentComparison(a_id=a.id, b_id=b.id, overall=overall, per_class=per_class, per_slice=per_slice,
        improvements=improvements, regressions=[r.to_dict() for r in regs], likely_tradeoff=tradeoff)

# Module-level alias matching §50 naming
compare_experiments = compare_runs
