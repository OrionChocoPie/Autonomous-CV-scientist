"""Artifact conversion for the approved Threats v1 YOLO experiments."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from cv_scientist.predictions import match_pairs, validate_predictions

CLASSES = ["smoke", "fire", "weapon", "knife"]
RUN_ID = "B0-yolo26n-v1-5ep"


def training_spec(experiment="B0"):
    if experiment == "B0":
        return {"id": RUN_ID, "epochs": 5, "parent": None,
                "dataset_version": None, "pretrained_sha256": None}
    if experiment == "E1":
        return {"id": "E1-yolo26n-v1-20ep", "epochs": 20, "parent": RUN_ID,
                "dataset_version": "threats-v1-11fb60ad98d95abf",
                "pretrained_sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"}
    if experiment == "E2":
        return {**training_spec("E1"), "id": "E2-close-mosaic5",
                "parent": "E1-yolo26n-v1-20ep", "close_mosaic": 5}
    if experiment == "E5":
        return {**training_spec("E1"), "id": "E5-train-reviewed-v2",
                "parent": "E1-yolo26n-v1-20ep"}
    raise ValueError("Experiment must be B0 or E1 or E2 or E5")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yolo_truth(text, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    truth = []
    for line in text.splitlines():
        columns = line.split()
        if len(columns) != 5:
            raise ValueError("YOLO rows must contain five columns")
        class_id = int(columns[0])
        x, y, w, h = map(float, columns[1:])
        if (not 0 <= class_id < len(CLASSES) or not all(math.isfinite(v) for v in [x, y, w, h])
                or not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1
                or x - w/2 < -1e-9 or y - h/2 < -1e-9 or x + w/2 > 1 + 1e-9 or y + h/2 > 1 + 1e-9):
            raise ValueError("Invalid normalized YOLO box or class")
        truth.append({"class_id": class_id, "box": [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
    return truth


def measured_metrics(images, map_metrics, class_maps, *, experiment_id=RUN_ID):
    """Keep fixed-threshold P/R separate from Ultralytics' best-F1 P/R."""
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Experiment ID must be a nonempty string")
    validate_predictions({"version": 2, "images": images})
    totals = Counter(tp=0, fp=0, fn=0)
    per_class = {name: Counter(tp=0, fp=0, fn=0) for name in CLASSES}
    for row in images:
        pairs, _ = match_pairs(row, 0.5)
        used_truth, used_predictions = {t for t, _ in pairs}, {p for _, p in pairs}
        counts = {"tp": len(pairs), "fp": len(row["predictions"]) - len(pairs), "fn": len(row["truth"]) - len(pairs)}
        if any(row[k] != v for k, v in counts.items()):
            raise ValueError("Exported counts disagree with class-aware matching")
        totals.update(counts)
        for index, truth in enumerate(row["truth"]):
            per_class[CLASSES[truth["class_id"]]]["tp" if index in used_truth else "fn"] += 1
        for index, prediction in enumerate(row["predictions"]):
            if index not in used_predictions:
                per_class[CLASSES[prediction["class_id"]]]["fp"] += 1

    def quality(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {**counts, "precision": tp/(tp+fp) if tp+fp else 0.0,
                "recall": tp/(tp+fn) if tp+fn else 0.0,
                "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}

    output = {"id": experiment_id, "metrics": {**quality(totals), **map_metrics},
              "per_class": {name: {**quality(counts), **class_maps.get(name, {})} for name, counts in per_class.items()}}
    json.dumps(output, allow_nan=False)
    return output


def evaluation_spec(experiment):
    values = {"C1": ("C1-e1-parity", 640, 0.25),
              "E3": ("E3-infer960", 960, 0.25), "E4": ("E4-conf010", 640, 0.10)}
    if experiment not in values:
        raise ValueError("Evaluation must be C1, E3 or E4")
    identity, size, confidence = values[experiment]
    return {"id": identity, "imgsz": size, "confidence": confidence,
            "parent": training_spec("E1")["id"], "batch": 32}


def completed_run(run_dir, *, inference_only=False):
    """A completed, unmodified run is a no-op; partial output is an error."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    receipt = run_dir / "completion.json"
    if not receipt.is_file():
        raise ValueError("Run already exists without a completion receipt; inspect it before retrying")
    data = json.loads(receipt.read_text())
    required = {"config.json", "metrics.json", "predictions.json"}
    if inference_only:
        if data.get("mode") != "inference_only":
            raise ValueError("Expected an inference-only receipt")
    else:
        if data.get("mode") == "inference_only":
            raise ValueError("Expected a training receipt")
        required |= {"ultralytics/weights/best.pt", "ultralytics/weights/last.pt"}
    if data.get("status") != "completed" or not required <= data.get("sha256", {}).keys():
        raise ValueError("Incomplete training receipt")
    for relative, expected in data["sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_sha256(run_dir / path) != expected:
            raise ValueError("Completed training artifacts were modified")
    return data
