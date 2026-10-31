"""Diagnostic skills as first-class APIs (§50). Deterministic, LLM-independent.

Usable directly: `diagnostics.analyze_false_positives(exp)`, via CLI, tests,
frontend, and by ScientistAgent tools. Works if the LLM is unavailable (§71).
Input is an Experiment (metrics/slices/per_class) — adapters map tracker data to it.
"""
from __future__ import annotations
import math
from .models import Experiment, DiagnosticFinding, RegressionFinding, EvidenceRef
from .predictions import area, iou, load_predictions, match_boxes, truth_box, classification_errors

def _ev(exp, detail=""):
    return [EvidenceRef("experiment", exp.id, detail).to_dict()]

def _sev(rel):  # relative drop magnitude -> severity
    a = abs(rel)
    return "critical" if a >= 0.20 else "high" if a >= 0.10 else "medium" if a >= 0.05 else "low"

def analyze_prediction_errors(exp: Experiment, iou_threshold=None) -> list:
    """Object groups in pixel area: small <32², medium <96²; score bins .5/.8."""
    threshold = exp.config.get("iou_match", 0.5) if iou_threshold is None else iou_threshold
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Matching IoU must be between 0 (exclusive) and 1")
    groups, totals, findings = {}, {"fp": 0, "fn": 0}, []
    def add(kind, box, confidence, overlap, ref):
        totals[kind] += 1
        size = "small" if area(box) < 32**2 else "medium" if area(box) < 96**2 else "large"
        score = "none" if confidence is None else "low" if confidence < 0.5 else "medium" if confidence < 0.8 else "high"
        for group in ("size:" + size, "confidence:" + score, "overlap:" + overlap):
            bucket = groups.setdefault((kind, group), {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 3: bucket["examples"].append(ref)
    for index, row in enumerate(load_predictions(exp, optional=True)):
        matched_truth, matched_preds, overlaps = match_boxes(row, threshold)
        truth, preds = [truth_box(box) for box in row["truth"]], row["predictions"]
        labelled = any(isinstance(box, dict) for box in row["truth"])
        wrong = classification_errors(row, threshold)[1] if labelled else []
        wrong_truth, wrong_preds = {t for t, _ in wrong}, {p for _, p in wrong}
        def evidence(detail): return EvidenceRef("artifact", f"{exp.id}:predictions.json#{index}", row["image"] + "; " + detail).to_dict()
        actual = {"tp": len(matched_preds), "fp": len(preds) - len(matched_preds), "fn": len(truth) - len(matched_truth)}
        if any(row[key] != actual[key] for key in actual if key in row):
            findings.append(DiagnosticFinding(title=f"Exported counts disagree with box matching: {row['image']}",
                category="artifact_consistency", severity="high", slice=f"artifact:{index}", metric="matching_iou", value=float(threshold),
                evidence=[evidence(f"{'class-aware' if labelled else 'class-agnostic'} matching={actual}")]))
        for p, prediction in enumerate(preds):
            if p in matched_preds: continue
            maximum = max(overlaps[p], default=0)
            overlap = "classification" if p in wrong_preds else "duplicate" if maximum >= threshold else "localization" if maximum > 0 else "background"
            add("fp", prediction["box"], prediction["score"], overlap, evidence(f"prediction#{p}; best_gt_iou={maximum:.6f}"))
        for t, box in enumerate(truth):
            if t in matched_truth: continue
            maximum = max((overlaps[p][t] for p in range(len(preds))), default=0)
            nearest = max(range(len(preds)), key=lambda p: (overlaps[p][t], -p), default=None)
            crowded = any(i != t and iou(box, other) >= threshold for i, other in enumerate(truth))
            overlap = "classification" if t in wrong_truth else "crowded" if crowded else "localization" if maximum > 0 else "uncovered"
            confidence = preds[nearest]["score"] if nearest is not None and maximum > 0 else None
            add("fn", box, confidence, overlap, evidence(f"truth#{t}; best_pred_iou={maximum:.6f}"))
    for (kind, group), bucket in sorted(groups.items()):
        findings.append(DiagnosticFinding(title=f"{kind.upper()} objects in {group}: {bucket['count']}",
            category="false_positive_geometry" if kind == "fp" else "false_negative_geometry", metric=kind,
            slice="objects:" + group, value=float(bucket["count"]), share=bucket["count"] / totals[kind],
            severity="high" if bucket["count"] / totals[kind] >= 0.5 else "medium", evidence=bucket["examples"]))
    return findings


def analyze_classes(exp: Experiment) -> dict:
    records = load_predictions(exp, optional=True)
    result = {"mode": "class_aware" if records.class_aware else "class_agnostic" if records.available else "unavailable",
              "matrix": [], "totals": {"tp": 0, "fp": 0, "fn": 0, "classification": 0}, "limitations": []}
    if not records.class_aware:
        result["limitations"].append("Class diagnostics require predictions version 2 with GT and prediction class_id")
        return result
    cells = {}
    threshold = exp.config.get("iou_match", 0.5)
    def add(truth, prediction): cells[truth, prediction] = cells.get((truth, prediction), 0) + 1
    for row in records:
        pairs, wrong = classification_errors(row, threshold)
        used_truth, used_preds = {t for t, _ in pairs + wrong}, {p for _, p in pairs + wrong}
        result["totals"]["tp"] += len(pairs)
        result["totals"]["fp"] += len(row["predictions"]) - len(pairs)
        result["totals"]["fn"] += len(row["truth"]) - len(pairs)
        result["totals"]["classification"] += len(wrong)
        for t, p in pairs + wrong: add(row["truth"][t]["class_id"], row["predictions"][p]["class_id"])
        for t, obj in enumerate(row["truth"]):
            if t not in used_truth: add(obj["class_id"], None)
        for p, obj in enumerate(row["predictions"]):
            if p not in used_preds: add(None, obj["class_id"])
    result["matrix"] = [{"truth": t, "prediction": p, "count": count}
                        for (t, p), count in sorted(cells.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))]
    return result


def analyze_classification_errors(exp: Experiment) -> list:
    records = load_predictions(exp, optional=True)
    if not records.class_aware: return []
    out = []
    for index, row in enumerate(records):
        _, wrong = classification_errors(row, exp.config.get("iou_match", 0.5))
        for t, p in wrong:
            actual, predicted = row["truth"][t]["class_id"], row["predictions"][p]["class_id"]
            out.append(DiagnosticFinding(title=f"Wrong class: {actual} → {predicted}", category="classification",
                metric="classification_errors", value=1.0, slice=f"class:{actual}→{predicted}:image:{index}:truth:{t}",
                evidence=[EvidenceRef("artifact", f"{exp.id}:predictions.json#{index}", f"{row['image']}; truth#{t}; prediction#{p}").to_dict()]))
    return out

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
    """Slice-level regressions; count growth from zero uses a one-object denominator."""
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
            for metric in ("fp", "fn"):
                if metric in bd.get(name, {}) and metric in cd.get(name, {}):
                    previous, value = bd[name][metric], cd[name][metric]
                    if previous == 0 and value <= 0: continue
                    rel = (value - previous) / (previous or 1)
                    if rel >= threshold or previous == 0:
                        detail = "Baseline count is zero; relative delta normalized to one object" if previous == 0 else ""
                        findings.append(RegressionFinding(metric=metric, slice=f"{scope}:{name}",
                            previous_value=float(previous), current_value=float(value),
                            absolute_delta=float(value - previous), relative_delta=float(rel),
                            severity=_sev(rel), experiment_id=current.id, baseline_id=baseline.id,
                            evidence=[EvidenceRef("experiment", baseline.id, detail).to_dict(), EvidenceRef("experiment", current.id).to_dict()]))
    return sorted(findings, key=lambda r: -abs(r.relative_delta))
