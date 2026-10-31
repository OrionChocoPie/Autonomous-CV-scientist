"""Create a public synthetic inspection workspace without a model or network."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from cv_scientist.analysis import RunAnalysisService
from cv_scientist.ingestor import LocalExperimentSource, RunIngestor
from cv_scientist.predictions import match_pairs
from cv_scientist.reporting import generate_report
from cv_scientist.workspace import ProjectWorkspace

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = {0: "scratch", 1: "dent"}
SCENES = [
    ("scratch-large", [{"class_id": 0, "box": [80, 100, 300, 135]}], [
        {"class_id": 0, "box": [78, 98, 302, 137], "score": .95},
        {"class_id": 0, "box": [470, 270, 550, 310], "score": .45}]),
    ("scratch-small", [{"class_id": 0, "box": [430, 150, 445, 165]}], [
        {"class_id": 0, "box": [429, 149, 446, 166], "score": .60}]),
    ("dent-large", [{"class_id": 1, "box": [180, 160, 270, 235]}], [
        {"class_id": 1, "box": [182, 162, 268, 233], "score": .93},
        {"class_id": 1, "box": [480, 260, 530, 315], "score": .40}]),
    ("dent-small", [{"class_id": 1, "box": [100, 280, 120, 298]}], [
        {"class_id": 0, "box": [100, 280, 120, 298], "score": .85}]),
    ("clean", [], [{"class_id": 0, "box": [350, 170, 410, 190], "score": .36}]),
]


def _quality(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {**counts, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _metrics(rows):
    totals = Counter(tp=0, fp=0, fn=0)
    classes = {str(key): Counter(tp=0, fp=0, fn=0) for key in CLASS_NAMES}
    for row in rows:
        pairs, _ = match_pairs(row, .5)
        used_truth, used_preds = {t for t, _ in pairs}, {p for _, p in pairs}
        totals.update({key: row[key] for key in ("tp", "fp", "fn")})
        for index, obj in enumerate(row["truth"]):
            classes[str(obj["class_id"])]["tp" if index in used_truth else "fn"] += 1
        for index, obj in enumerate(row["predictions"]):
            if index not in used_preds:
                classes[str(obj["class_id"])]["fp"] += 1
    return _quality(totals), {key: _quality(value) for key, value in classes.items()}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create_demo(destination):
    if not isinstance(destination, (str, Path)) or not str(destination).strip():
        raise ValueError("Choose a nonempty output directory")
    root = Path(destination).expanduser().resolve()
    if root.exists():
        raise FileExistsError("Output already exists; choose a new directory to preserve it")
    root.mkdir(parents=True)
    ws = ProjectWorkspace.init(str(root), name="synthetic-inspection", primary_metric="f1")
    with Path(ws.config_path).open("a") as file:
        file.write("dataset_root: data\n")
    ws.config["dataset_root"] = "data"
    images, labels = root / "data/images/val", root / "data/labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (root / "data/data.yaml").write_text("path: .\nval: images/val\nnames:\n  0: scratch\n  1: dent\n")
    for name, truth, _ in SCENES:
        image = Image.new("RGB", (640, 384), "#e6ebef")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((35, 40, 605, 345), radius=24, fill="#b8c5d0", outline="#526777", width=3)
        for obj in truth:
            x1, y1, x2, y2 = obj["box"]
            if obj["class_id"] == 0:
                draw.line((x1, y2, x2, y1), fill="#672e24", width=max(3, (y2-y1)//3))
            else:
                draw.ellipse((x1, y1, x2, y2), fill="#75818d", outline="#404d5a", width=2)
        image.save(images / (name + ".png"))
        lines = []
        for obj in truth:
            x1, y1, x2, y2 = obj["box"]
            lines.append(f"{obj['class_id']} {(x1+x2)/1280:.8f} {(y1+y2)/768:.8f} {(x2-x1)/640:.8f} {(y2-y1)/384:.8f}")
        (labels / (name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    summary = {"synthetic": True, "trained_model": False, "api_calls": 0, "experiments": {}}
    for identity, confidence in (("B0", .25), ("E1", .70)):
        run = root / "runs" / identity
        run.mkdir(parents=True)
        config = {"experiment_id": identity, "name": identity, "model": "synthetic-fixture",
                  "confidence": confidence, "imgsz": 640, "seed": 0, "iou_match": .5,
                  "dataset_version": "synthetic-inspection-v1", "dataset_root": "data",
                  "class_names": CLASS_NAMES, "synthetic": True,
                  "slice_definition": "small: images whose GT contains a box of area <= 1024 pixels"}
        if identity == "E1":
            config["parent"] = "B0"
        rows = []
        for name, truth, predictions in SCENES:
            row = {"image": f"images/val/{name}.png", "truth": truth,
                   "predictions": [p for p in predictions if p["score"] >= confidence]}
            pairs, _ = match_pairs(row, .5)
            row.update(tp=len(pairs), fp=len(row["predictions"])-len(pairs), fn=len(truth)-len(pairs))
            rows.append(row)
        metrics, per_class = _metrics(rows)
        small = [row for row in rows if any((t["box"][2]-t["box"][0])*(t["box"][3]-t["box"][1]) <= 1024 for t in row["truth"])]
        _write_json(run / "config.json", config)
        _write_json(run / "predictions.json", {"version": 2, "images": rows})
        _write_json(run / "metrics.json", {"id": identity, "metrics": metrics, "per_class": per_class,
                                           "slices": {"small": _metrics(small)[0]}})
        RunIngestor(ws.memory, LocalExperimentSource()).ingest(str(run))
        RunAnalysisService(ws.memory).analyze(identity)
        summary["experiments"][identity] = metrics
    ws.memory.add_user_note("Synthetic illustration: higher precision can hide lost small-defect recall. No model was trained; no mAP was measured.")
    summary["report"] = str(generate_report(ws, "E1", baseline_id="B0"))
    _write_json(root / "demo-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo-workspace")
    args = parser.parse_args(argv)
    try:
        result = create_demo(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    LOGGER.info("Synthetic demo ready: %s", result["report"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
