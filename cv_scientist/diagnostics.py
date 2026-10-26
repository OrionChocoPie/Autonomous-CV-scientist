"""Diagnostic skills as first-class APIs (§50). Deterministic, LLM-independent.

Usable directly: `diagnostics.analyze_false_positives(exp)`, via CLI, tests,
frontend, and by ScientistAgent tools. Works if the LLM is unavailable (§71).
Input is an Experiment (metrics/slices/per_class) — adapters map tracker data to it.
"""
from __future__ import annotations
from .models import Experiment, DiagnosticFinding, RegressionFinding, EvidenceRef

def _ev(exp, detail=""):
    return [EvidenceRef("experiment", exp.id, detail).to_dict()]

def _sev(rel):  # relative drop magnitude -> severity
    a = abs(rel)
    return "critical" if a >= 0.20 else "high" if a >= 0.10 else "medium" if a >= 0.05 else "low"

def analyze_false_negatives(exp: Experiment) -> list:
    total_fn = sum(s.get("fn", 0) for s in exp.slices.values()) or sum(c.get("fn", 0) for c in exp.per_class.values())
    out = []
    sources = [(f"class:{k}", v) for k, v in exp.per_class.items()] + [(f"slice:{k}", v) for k, v in exp.slices.items()]
    for name, s in sources:
        fn = s.get("fn", 0); rec = s.get("recall")
        if fn or (rec is not None and rec < 0.8):
            out.append(DiagnosticFinding(title=f"False negatives in {name}", category="false_negative",
                slice=name, metric="recall", value=rec, share=(fn / total_fn if total_fn and fn else None),
                severity=_sev(1 - rec) if rec is not None else "medium", evidence=_ev(exp, name)))
    return sorted(out, key=lambda f: -(f.share or 0))

def analyze_false_positives(exp: Experiment) -> list:
    total_fp = sum(s.get("fp", 0) for s in exp.slices.values()) or sum(c.get("fp", 0) for c in exp.per_class.values())
    out = []
    sources = [(f"class:{k}", v) for k, v in exp.per_class.items()] + [(f"slice:{k}", v) for k, v in exp.slices.items()]
    for name, s in sources:
        fp = s.get("fp", 0); prec = s.get("precision")
        if fp or (prec is not None and prec < 0.8):
            out.append(DiagnosticFinding(title=f"False positives in {name}", category="false_positive",
                slice=name, metric="precision", value=prec, share=(fp / total_fp if total_fp and fp else None),
                severity=_sev(1 - prec) if prec is not None else "medium", evidence=_ev(exp, name)))
    return sorted(out, key=lambda f: -(f.share or 0))

def analyze_small_objects(exp: Experiment, threshold: float = 0.10) -> list:
    out = []
    for name, s in exp.slices.items():
        if "small" in name.lower():
            rec = s.get("recall"); avg = exp.metrics.get("recall")
            rel = ((rec - avg) / avg) if rec is not None and avg else 0.0
            if rec is not None and (avg is None or rec < avg):
                out.append(DiagnosticFinding(title=f"Small-object weakness in {name}", category="small_object",
                    slice=name, metric="recall", value=rec, share=s.get("fn_share"),
                    severity=_sev(rel), evidence=_ev(exp, name)))
    return out

def cluster_errors(exp: Experiment) -> list:
    """Group FP/FN by slice+class into error clusters (§47)."""
    clusters = []
    for kind, fn in (("false_positive", analyze_false_positives), ("false_negative", analyze_false_negatives)):
        for f in fn(exp):
            if (f.share or 0) >= 0.05:
                clusters.append(DiagnosticFinding(title=f"Error cluster: {f.slice} ({kind})", category="error_cluster",
                    slice=f.slice, metric=f.metric, value=f.value, share=f.share, severity=f.severity, evidence=f.evidence))
    return sorted(clusters, key=lambda f: -(f.share or 0))

def audit_annotations(exp: Experiment, suspicious_fp_share: float = 0.15) -> list:
    """Heuristic (§48): high-confidence FP clusters often indicate missing annotations."""
    out = []
    for f in analyze_false_positives(exp):
        conf = exp.slices.get(f.slice.replace("slice:", ""), {}).get("mean_confidence") if f.slice.startswith("slice:") else None
        if (f.share or 0) >= suspicious_fp_share:
            out.append(DiagnosticFinding(title=f"Possible annotation issues in {f.slice}", category="annotation",
                slice=f.slice, metric="fp_share", value=f.share, share=f.share, severity="medium", evidence=f.evidence))
    return out

def find_hard_negatives(exp: Experiment, min_fp: int = 1) -> list:
    return [f for f in analyze_false_positives(exp) if (exp.per_class.get(f.slice.replace("class:", ""), {}).get("fp", 0) >= min_fp
            or exp.slices.get(f.slice.replace("slice:", ""), {}).get("fp", 0) >= min_fp)]

def analyze_dataset_shift(exp: Experiment, baseline: Experiment) -> list:
    out = []
    for name, s in exp.slices.items():
        b = baseline.slices.get(name, {})
        if "count" in s and "count" in b and b["count"]:
            rel = (s["count"] - b["count"]) / b["count"]
            if abs(rel) >= 0.20:
                out.append(DiagnosticFinding(title=f"Dataset shift in {name}", category="dataset_shift", slice=name,
                    metric="count", value=float(s["count"]), severity=_sev(rel), evidence=_ev(exp, f"vs {baseline.id}")))
    return out

def find_regressions(current: Experiment, baseline: Experiment, threshold: float = 0.02) -> list:
    """Slice-level regression detection (§49). Does not rely only on primary metric."""
    findings = []
    def check(scope, name, a: dict, b: dict):
        for metric in ("recall", "precision", "map", "map50_95"):
            if metric in a and metric in b and a[metric]:
                rel = (b[metric] - a[metric]) / abs(a[metric])
                if rel <= -threshold:
                    findings.append(RegressionFinding(metric=metric, slice=f"{scope}:{name}" if scope else "overall",
                        previous_value=float(a[metric]), current_value=float(b[metric]),
                        absolute_delta=float(b[metric] - a[metric]), relative_delta=float(rel), severity=_sev(rel),
                        experiment_id=current.id, baseline_id=baseline.id,
                        evidence=[EvidenceRef("experiment", baseline.id).to_dict(), EvidenceRef("experiment", current.id).to_dict()]))
    check("", "overall", baseline.metrics, current.metrics)
    for name in set(baseline.slices) | set(current.slices):
        check("slice", name, baseline.slices.get(name, {}), current.slices.get(name, {}))
    for name in set(baseline.per_class) | set(current.per_class):
        check("class", name, baseline.per_class.get(name, {}), current.per_class.get(name, {}))
    # FP/FN count regressions
    for scope, bd, cd in (("slice", baseline.slices, current.slices), ("class", baseline.per_class, current.per_class)):
        for name in set(bd) | set(cd):
            for metric, worse in (("fp", 1), ("fn", 1)):
                if metric in bd.get(name, {}) and metric in cd.get(name, {}) and bd[name][metric]:
                    rel = (cd[name][metric] - bd[name][metric]) / bd[name][metric]
                    if rel >= threshold:
                        findings.append(RegressionFinding(metric=metric, slice=f"{scope}:{name}",
                            previous_value=float(bd[name][metric]), current_value=float(cd[name][metric]),
                            absolute_delta=float(cd[name][metric] - bd[name][metric]), relative_delta=float(rel),
                            severity=_sev(rel), experiment_id=current.id, baseline_id=baseline.id,
                            evidence=[EvidenceRef("experiment", baseline.id).to_dict(), EvidenceRef("experiment", current.id).to_dict()]))
    return sorted(findings, key=lambda r: -abs(r.relative_delta))
