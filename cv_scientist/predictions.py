"""Versioned prediction artifacts and deterministic box/class matching."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _box(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            or value[2] <= value[0] or value[3] <= value[1]):
        raise ValueError("Boxes must contain four finite xyxy coordinates with positive area")
    return value


class PredictionRecords(list):
    def __init__(self, rows=(), class_aware=False, available=True):
        super().__init__(rows)
        self.class_aware, self.available = class_aware, available


def truth_box(truth): return truth["box"] if isinstance(truth, dict) else truth


def _class_id(value):
    if not ((type(value) is int and value >= 0) or (isinstance(value, str) and value.strip() and len(value) <= 128)):
        raise ValueError("class_id must be a nonnegative integer or a nonempty string")
    return value


def load_predictions(exp, optional=False):
    if not exp.run_dir:
        if optional: return PredictionRecords(available=False)
        raise ValueError("Experiment has no local run directory")
    root = Path(exp.run_dir).resolve()
    original = root / "predictions.json"
    if optional and not original.exists() and not original.is_symlink(): return PredictionRecords(available=False)
    path = original.resolve()
    if not path.is_relative_to(root): raise ValueError("predictions.json must stay inside the run directory")
    with path.open("rb") as file: raw = file.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024: raise ValueError("predictions.json exceeds the 8 MiB limit")
    return validate_predictions(json.loads(raw))


def validate_predictions(records):
    class_aware = False
    if isinstance(records, dict):
        if records.get("version") != 2 or type(records.get("version")) is not int or set(records) != {"version", "images"}:
            raise ValueError("Unsupported predictions format; expected version 2 and images")
        class_aware, records = True, records["images"]
    if not isinstance(records, list): raise ValueError("predictions.json must contain a list of image records")
    names, label_types = set(), set()
    for row in records:
        if not isinstance(row, dict): raise ValueError("Prediction records must be objects")
        name = row.get("image")
        if (not isinstance(name, str) or not name.strip() or len(name) > 2048 or Path(name).is_absolute()
                or ".." in Path(name).parts or name in names):
            raise ValueError("Prediction image paths must be unique relative paths without ..")
        names.add(name)
        if not isinstance(row.get("truth"), list) or not isinstance(row.get("predictions"), list):
            raise ValueError("Prediction records need truth and predictions lists")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("fp", "fn")):
            raise ValueError("Prediction records need nonnegative integer fp and fn")
        if "tp" in row and (type(row["tp"]) is not int or row["tp"] < 0): raise ValueError("tp must be a nonnegative integer")
        for box in row["truth"]:
            if class_aware:
                if not isinstance(box, dict): raise ValueError("Version 2 truth needs box and class_id")
                _box(box.get("box"))
                label_types.add(type(_class_id(box.get("class_id"))))
            else: _box(box)
        for pred in row["predictions"]:
            if not isinstance(pred, dict): raise ValueError("Predictions must be objects")
            _box(pred.get("box"))
            score = pred.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction scores must be finite numbers between 0 and 1")
            if class_aware: label_types.add(type(_class_id(pred.get("class_id"))))
            elif "class_id" in pred: raise ValueError("Class labels require version 2 with labelled ground truth")
    if len(label_types) > 1: raise ValueError("Use one class_id type throughout the artifact")
    return PredictionRecords(records, class_aware=class_aware)


def area(box): return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area(a) + area(b) - intersection)


def match_pairs(row, threshold, use_classes=True, truth_indices=None, prediction_indices=None):
    """Greedy matching by descending score; equal scores/IoUs use source order."""
    truth, preds = row["truth"], row["predictions"]
    overlaps = [[iou(p["box"], truth_box(box)) for box in truth] for p in preds]
    available_truth = set(range(len(truth))) if truth_indices is None else set(truth_indices)
    available_preds = range(len(preds)) if prediction_indices is None else prediction_indices
    used_truth, pairs = set(), []
    labelled = any(isinstance(box, dict) for box in truth) or any("class_id" in p for p in preds)
    for index in sorted(available_preds, key=lambda p: (-preds[p]["score"], p)):
        candidates = [i for i in sorted(available_truth - used_truth)
                      if not use_classes or not labelled or truth[i]["class_id"] == preds[index]["class_id"]]
        best = max(candidates, key=lambda i: (overlaps[index][i], -i), default=None)
        if best is not None and overlaps[index][best] >= threshold:
            used_truth.add(best); pairs.append((best, index))
    return pairs, overlaps


def match_boxes(row, threshold):
    pairs, overlaps = match_pairs(row, threshold)
    return {t for t, _ in pairs}, {p for _, p in pairs}, overlaps


def classification_errors(row, threshold):
    """Pair remaining FP/FN geometrically to distinguish wrong labels from misses."""
    pairs, _ = match_pairs(row, threshold)
    used_truth, used_preds = {t for t, _ in pairs}, {p for _, p in pairs}
    wrong, _ = match_pairs(row, threshold, use_classes=False,
        truth_indices=set(range(len(row["truth"]))) - used_truth,
        prediction_indices=set(range(len(row["predictions"]))) - used_preds)
    return pairs, wrong
