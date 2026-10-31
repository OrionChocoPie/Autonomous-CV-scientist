"""Run an approved Threats YOLO26n experiment on Modal.

From the repository's .venv:
modal run --detach --profile your-profile --env main scripts/modal_threats_train.py
modal run --detach --profile your-profile --env main scripts/modal_threats_train.py --experiment E1
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile

import modal

from scripts.threats_artifacts import CLASSES, completed_run, file_sha256, measured_metrics, training_spec, write_json, yolo_truth

LOGGER = logging.getLogger(__name__)
INTEGRATIONS = {"sync": False, "wandb": False, "mlflow": False, "comet": False,
                "clearml": False, "tensorboard": False}
app = modal.App("threats-yolo26n-first-training")
dataset = modal.Volume.from_name("threats-v1", environment_name="main").read_only()
outputs = modal.Volume.from_name("threats-v1-runs", environment_name="main")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "ultralytics==8.4.193")
    .env({"YOLO_CONFIG_DIR": "/tmp/yolo-settings", "MPLCONFIGDIR": "/tmp/matplotlib"})
    .add_local_python_source("scripts", "cv_scientist")
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800,
              retries=0, volumes={"/dataset": dataset, "/results": outputs})
def train(experiment: str = "B0", label_patch: dict | None = None):
    import csv
    import os
    import time

    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from cv_scientist.predictions import match_pairs, validate_predictions

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = training_spec(experiment)
    if (experiment == "E5") != (label_patch is not None):
        raise ValueError("Only E5 requires a reviewed label patch")
    run_id, epoch_count = spec["id"], spec["epochs"]
    root = Path("/results") / run_id
    existing = completed_run(root)
    if existing is not None:
        if existing.get("experiment_id") != run_id or existing.get("epochs") != epoch_count:
            raise ValueError("Completed run does not match the requested experiment")
        if label_patch is not None:
            from scripts.threats_label_patch import patch_identity
            config = json.loads((root / "config.json").read_text())
            if config["dataset_version"] != patch_identity(label_patch):
                raise ValueError("Completed E5 uses a different reviewed patch")
        LOGGER.info("Reusing verified completed run %s", run_id)
        return existing
    root.mkdir()
    started = time.monotonic()
    write_json(root / "status.json", {"status": "preparing", "started_at": datetime.now(timezone.utc).isoformat()})
    outputs.commit()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; refusing CPU fallback")
        settings.update(INTEGRATIONS)
        source = Path("/dataset")
        manifest_bytes = (source / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        dataset_version = "threats-v1-" + hashlib.sha256(manifest_bytes).hexdigest()[:16]
        if spec["dataset_version"] is not None and dataset_version != spec["dataset_version"]:
            raise ValueError("Dataset differs from the frozen E1 proposal")
        if (manifest["summary"]["model_version"] != "v1" or manifest["summary"]["classes"] != CLASSES
                or manifest["summary"]["splits"] != {"train": 8284, "val": 1024, "test": 1027}):
            raise ValueError("Threats dataset identity or split differs from the approved dataset")
        checksums = dict((relative, digest) for digest, relative in
                         (line.split("  ", 1) for line in (source / "checksums.sha256").read_text().splitlines()))
        local = Path(tempfile.mkdtemp(prefix="threats-v1-"))
        entries = [entry for entry in manifest["files"] if entry["split"] in {"train", "val"}]
        if len(entries) != 9308:
            raise ValueError("Unexpected training/validation image count")
        paths = ["yolo/data.yaml"] + [entry[key] for entry in entries for key in ("image", "label")]

        def stage(relative):
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Invalid dataset path")
            target = local / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative_path, target)
            if file_sha256(target) != checksums[relative]:
                raise ValueError("Dataset checksum mismatch: " + relative)

        with ThreadPoolExecutor(max_workers=16) as pool:
            for index, _ in enumerate(pool.map(stage, paths), 1):
                if index % 4000 == 0:
                    LOGGER.info("Staged and verified %s/%s files", index, len(paths))
        LOGGER.info("Dataset ready: 8284 train, 1024 val; test not used")
        patch_result = None
        if label_patch is not None:
            from scripts.threats_label_patch import apply_label_patch
            patch_result = apply_label_patch(local, label_patch, dataset_version)
            write_json(root / "label-patch.json", label_patch)
            write_json(root / "label-patch-result.json", patch_result)
        plan = {"experiment_id": run_id, "name": run_id, "model": "yolo26n.pt", "task": "detect",
                "epochs": epoch_count, "close_mosaic": spec.get("close_mosaic", 0),
                "imgsz": 640, "batch": 32, "seed": 0, "device": "L4", "workers": 4,
                "confidence": 0.25, "iou_match": 0.5, "nms_iou": 0.7, "split": "val",
                "dataset_version": dataset_version,
                "annotation_source": "Label Studio predictions model_version=v1; not asserted human-reviewed",
                "modal_dataset_volume": "threats-v1", "modal_results_volume": "threats-v1-runs",
                "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0), "timeout_seconds": 1800}
        if spec["parent"] is not None:
            plan["parent"] = spec["parent"]
        if patch_result is not None:
            plan["parent_dataset_version"] = dataset_version
            plan["dataset_version"] = patch_result["dataset_version"]
            plan["annotation_patch"] = patch_result
            plan["annotation_source"] += "; selected duplicate removals visually reviewed by agent"
        write_json(root / "config.json", plan)
        os.chdir(local)
        model = YOLO("yolo26n.pt", task="detect")
        if model.task != "detect":
            raise ValueError("Expected a detection model")
        plan["pretrained_sha256"] = file_sha256(local / "yolo26n.pt")
        if spec["pretrained_sha256"] is not None and plan["pretrained_sha256"] != spec["pretrained_sha256"]:
            raise ValueError("Pretrained weights differ from B0")
        write_json(root / "config.json", plan)

        def record_optimizer(trainer):
            group = trainer.optimizer.param_groups[0]
            resolved = {"name": type(trainer.optimizer).__name__, "lr0": float(trainer.args.lr0),
                        "betas": list(group.get("betas", ())), "warmup_bias_lr": float(trainer.args.warmup_bias_lr)}
            if experiment != "B0" and resolved != {"name": "AdamW", "lr0": 0.00125,
                                                    "betas": [0.9, 0.999], "warmup_bias_lr": 0.0}:
                raise ValueError("Resolved optimizer differs from B0")
            plan["resolved_optimizer"] = resolved
            write_json(root / "config.json", plan)
            outputs.commit()
            LOGGER.info("Resolved optimizer: %s", json.dumps(resolved))

        def checkpoint(trainer):
            write_json(root / "status.json", {"status": "training", "epoch": min(trainer.epoch + 1, epoch_count),
                                               "elapsed_seconds": time.monotonic() - started})
            outputs.commit()
            LOGGER.info("Persisted checkpoint after epoch %s", min(trainer.epoch + 1, epoch_count))

        model.add_callback("on_train_start", record_optimizer)
        model.add_callback("on_fit_epoch_end", checkpoint)
        model.train(data=str(local / "yolo/data.yaml"), task="detect", epochs=epoch_count, imgsz=640, batch=32,
                    device=0, workers=4, seed=0, deterministic=True, cache=False, amp=True,
                    project=str(root), name="ultralytics", exist_ok=False, save=True, save_period=1,
                    val=True, plots=True, patience=0, close_mosaic=spec.get("close_mosaic", 0), verbose=False)
        training = root / "ultralytics"
        with (training / "results.csv").open() as file:
            epochs = list(csv.DictReader(file))
        if len(epochs) != epoch_count:
            raise ValueError(f"Expected {epoch_count} completed epochs, got {len(epochs)}")
        best, last = training / "weights/best.pt", training / "weights/last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError("Training did not produce best.pt and last.pt")
        best_model = YOLO(str(best), task="detect")
        validation = best_model.val(data=str(local / "yolo/data.yaml"), split="val", imgsz=640, batch=32,
                                    device=0, workers=4, plots=True, verbose=False,
                                    project=str(root), name="validation", exist_ok=False)
        rows = []
        expected = {str((local / entry["image"]).resolve()) for entry in entries if entry["split"] == "val"}
        seen = set()
        predictions = best_model.predict(source=str(local / "yolo/images/val"), stream=True, batch=32, imgsz=640,
                                         device=0, conf=0.25, iou=0.7, max_det=300, verbose=False, save=False)
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
        if seen != expected:
            raise ValueError("Not all validation images were evaluated")
        rows.sort(key=lambda row: row["image"])
        artifact = {"version": 2, "images": rows}
        validate_predictions(artifact)
        if len(json.dumps(artifact).encode()) > 8 * 1024 * 1024:
            raise ValueError("Prediction artifact exceeds CV Scientist's 8 MiB limit")
        class_maps = {}
        for index, class_id in enumerate(validation.box.ap_class_index):
            class_result = validation.box.class_result(index)
            class_maps[CLASSES[int(class_id)]] = {"map50": float(class_result[2]), "map50_95": float(class_result[3])}
        metrics = measured_metrics(rows, {"map50": float(validation.box.map50), "map50_95": float(validation.box.map),
                                          "ultralytics_precision_best_f1": float(validation.box.mp),
                                          "ultralytics_recall_best_f1": float(validation.box.mr)}, class_maps,
                                   experiment_id=run_id)
        write_json(root / "predictions.json", artifact)
        write_json(root / "metrics.json", metrics)
        write_json(root / "ultralytics-validation.json", {k: float(v) for k, v in validation.results_dict.items()})
        write_json(root / "status.json", {"status": "completed", "epochs": epoch_count,
                                           "elapsed_seconds": time.monotonic() - started})
        receipt = {"status": "completed", "experiment_id": run_id, "epochs": epoch_count,
                   "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics["metrics"],
                   "sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}}
        write_json(root / "completion.json", receipt)
        outputs.commit()
        LOGGER.info("Completed %s: %s", run_id, json.dumps(metrics["metrics"], allow_nan=False))
        return receipt
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error),
                                           "elapsed_seconds": time.monotonic() - started})
        outputs.commit()
        LOGGER.exception("Training run failed")
        raise


@app.local_entrypoint()
def main(experiment: str = "B0", label_patch: str = ""):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    training_spec(experiment)
    patch = json.loads(Path(label_patch).read_text()) if label_patch else None
    if (experiment == "E5") != (patch is not None):
        raise ValueError("Only E5 requires --label-patch")
    result = train.remote(experiment, patch)
    LOGGER.info("Remote run result: %s", json.dumps(result, allow_nan=False))
