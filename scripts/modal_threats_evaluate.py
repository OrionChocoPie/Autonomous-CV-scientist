"""Frozen-checkpoint Threats validation experiments; no training or test access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

import modal

from scripts.modal_threats_train import INTEGRATIONS, dataset, image, outputs
from scripts.threats_artifacts import (CLASSES, completed_run, evaluation_spec, file_sha256,
                                      measured_metrics, training_spec, write_json, yolo_truth)

LOGGER = logging.getLogger(__name__)
app = modal.App("threats-frozen-checkpoint-evaluation")


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900,
              retries=0, volumes={"/dataset": dataset, "/results": outputs})
def evaluate(experiment: str):
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from cv_scientist.predictions import match_pairs, validate_predictions

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = evaluation_spec(experiment)
    root = Path("/results") / spec["id"]
    existing = completed_run(root, inference_only=True)
    if existing is not None:
        if existing.get("experiment_id") != spec["id"]:
            raise ValueError("Completed evaluation identity mismatch")
        return existing
    root.mkdir()
    started = time.monotonic()
    write_json(root / "status.json", {"status": "preparing"})
    outputs.commit()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        settings.update(INTEGRATIONS)
        source = Path("/dataset")
        manifest_bytes = (source / "manifest.json").read_bytes()
        version = "threats-v1-" + hashlib.sha256(manifest_bytes).hexdigest()[:16]
        if version != training_spec("E1")["dataset_version"]:
            raise ValueError("Dataset identity mismatch")
        manifest = json.loads(manifest_bytes)
        entries = [entry for entry in manifest["files"] if entry["split"] == "val"]
        if len(entries) != 1024 or manifest["summary"]["classes"] != CLASSES:
            raise ValueError("Unexpected validation population or classes")
        checksums = {relative: digest for digest, relative in
                     (line.split("  ", 1) for line in (source / "checksums.sha256").read_text().splitlines())}
        local = Path(tempfile.mkdtemp(prefix="threats-eval-"))
        paths = ["yolo/data.yaml"] + [entry[key] for entry in entries for key in ("image", "label")]

        def stage(relative):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid dataset path")
            target = local / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / path, target)
            if file_sha256(target) != checksums[relative]:
                raise ValueError("Validation checksum mismatch: " + relative)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(stage, paths))
        control = Path("/results") / spec["parent"]
        control_receipt = json.loads((control / "completion.json").read_text())
        checkpoint = control / "ultralytics/weights/best.pt"
        checkpoint_hash = file_sha256(checkpoint)
        if (control_receipt.get("status") != "completed" or
                checkpoint_hash != control_receipt["sha256"]["ultralytics/weights/best.pt"]):
            raise ValueError("E1 checkpoint integrity failure")
        plan = {"experiment_id": spec["id"], "name": spec["id"], "parent": spec["parent"],
                "operation": "inference_only", "model": "yolo26n.pt", "task": "detect",
                "imgsz": spec["imgsz"], "batch": spec["batch"], "confidence": spec["confidence"],
                "ap_confidence_floor": 0.001, "nms_iou": 0.7, "iou_match": 0.5,
                "max_det": 300, "split": "val", "dataset_version": version,
                "checkpoint_sha256": checkpoint_hash, "checkpoint_source": spec["parent"],
                "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0), "test_used": False}
        write_json(root / "config.json", plan)
        torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(checkpoint), task="detect")
        validation = model.val(data=str(local / "yolo/data.yaml"), split="val", imgsz=spec["imgsz"],
                               batch=spec["batch"], device=0, workers=4, plots=True, verbose=False,
                               conf=0.001, iou=0.7, max_det=300,
                               project=str(root), name="validation", exist_ok=False)
        rows, seen = [], set()
        expected = {str((local / entry["image"]).resolve()) for entry in entries}
        torch.cuda.synchronize()
        prediction_started = time.monotonic()
        speed = {key: 0.0 for key in ("preprocess", "inference", "postprocess")}
        predictions = model.predict(source=str(local / "yolo/images/val"), stream=True,
                                    batch=spec["batch"], imgsz=spec["imgsz"], device=0,
                                    conf=spec["confidence"], iou=0.7, max_det=300,
                                    verbose=False, save=False)
        for prediction in predictions:
            path = Path(prediction.path).resolve()
            if str(path) not in expected or str(path) in seen:
                raise ValueError("Unexpected or duplicate validation prediction")
            seen.add(str(path))
            height, width = prediction.orig_shape
            label = local / "yolo/labels/val" / (path.stem + ".txt")
            row = {"image": path.relative_to(local / "yolo").as_posix(),
                   "truth": yolo_truth(label.read_text(), width, height), "predictions": []}
            for box, score, class_id in zip(prediction.boxes.xyxy.cpu().tolist(),
                                            prediction.boxes.conf.cpu().tolist(), prediction.boxes.cls.cpu().tolist()):
                row["predictions"].append({"box": box, "score": score, "class_id": int(class_id)})
            pairs, _ = match_pairs(row, 0.5)
            row.update(tp=len(pairs), fp=len(row["predictions"]) - len(pairs), fn=len(row["truth"]) - len(pairs))
            rows.append(row)
            for key in speed:
                speed[key] += float(prediction.speed[key])
        torch.cuda.synchronize()
        if seen != expected:
            raise ValueError("Validation coverage mismatch")
        prediction_seconds = time.monotonic() - prediction_started
        rows.sort(key=lambda row: row["image"])
        artifact = {"version": 2, "images": rows}
        validate_predictions(artifact)
        if len(json.dumps(artifact).encode()) > 8 * 1024 * 1024:
            raise ValueError("Prediction artifact exceeds 8 MiB")
        class_maps = {}
        for index, class_id in enumerate(validation.box.ap_class_index):
            result = validation.box.class_result(index)
            class_maps[CLASSES[int(class_id)]] = {"map50": float(result[2]), "map50_95": float(result[3])}
        metrics = measured_metrics(rows, {"map50": float(validation.box.map50), "map50_95": float(validation.box.map),
                                          "ultralytics_precision_best_f1": float(validation.box.mp),
                                          "ultralytics_recall_best_f1": float(validation.box.mr)},
                                   class_maps, experiment_id=spec["id"])
        write_json(root / "metrics.json", metrics)
        write_json(root / "predictions.json", artifact)
        write_json(root / "runtime.json", {"validation_images": len(rows), "prediction_seconds": prediction_seconds,
                   "prediction_ms_per_image_including_io_and_export": prediction_seconds * 1000 / len(rows),
                   "model_ms_per_image": {key: value / len(rows) for key, value in speed.items()},
                   "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
                   "elapsed_seconds": time.monotonic() - started})
        write_json(root / "status.json", {"status": "completed"})
        receipt = {"status": "completed", "mode": "inference_only", "experiment_id": spec["id"],
                   "completed_at": datetime.now(timezone.utc).isoformat(),
                   "checkpoint_sha256": checkpoint_hash, "metrics": metrics["metrics"],
                   "sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}}
        write_json(root / "completion.json", receipt)
        outputs.commit()
        LOGGER.info("Completed %s: %s", spec["id"], json.dumps(metrics["metrics"]))
        return receipt
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error),
                                          "elapsed_seconds": time.monotonic() - started})
        outputs.commit()
        LOGGER.exception("Evaluation failed; batch and resolution were not changed")
        raise


@app.local_entrypoint()
def main(experiment: str = "C1"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluation_spec(experiment)
    result = evaluate.remote(experiment)
    LOGGER.info("Evaluation receipt: %s", json.dumps(result))
