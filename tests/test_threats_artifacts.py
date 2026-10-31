import json
import hashlib

import pytest

from scripts.threats_artifacts import RUN_ID, completed_run, file_sha256, measured_metrics, training_spec, write_json, yolo_truth


@pytest.mark.parametrize("size", [0, 1, 1024 * 1024 + 1])
def test_sha_export_does_not_require_python_311_file_digest(tmp_path, monkeypatch, size):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    content = b"x" * size
    path = tmp_path / "artifact"
    path.write_bytes(content)
    assert file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_approved_experiments_keep_baseline_and_separate_e1_identity():
    assert training_spec() == training_spec("B0")
    assert training_spec()["id"] == RUN_ID
    assert training_spec()["epochs"] == 5
    e1 = training_spec("E1")
    assert e1["id"] == "E1-yolo26n-v1-20ep"
    assert e1["epochs"] == 20
    assert e1["parent"] == RUN_ID
    assert e1["dataset_version"] == "threats-v1-11fb60ad98d95abf"
    assert e1["pretrained_sha256"] == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def test_e2_changes_only_mosaic_and_experiment_lineage():
    e1, e2 = training_spec("E1"), training_spec("E2")
    assert e2.pop("close_mosaic") == 5
    assert e2.pop("id") == "E2-close-mosaic5"
    assert e2.pop("parent") == e1.pop("id")
    e1.pop("parent")
    assert e2 == e1


@pytest.mark.parametrize("experiment", [None, "", "E99", "../E1"])
def test_unapproved_experiments_fail(experiment):
    with pytest.raises(ValueError, match="B0 or E1"):
        training_spec(experiment)


def test_e1_metric_export_does_not_overwrite_baseline_identity():
    result = measured_metrics([], {"map50_95": 0.5}, {}, experiment_id=training_spec("E1")["id"])
    assert result["id"] == "E1-yolo26n-v1-20ep"
    assert measured_metrics([], {}, {})["id"] == RUN_ID


@pytest.mark.parametrize("identity", [None, "", " "])
def test_empty_metric_identity_fails(identity):
    with pytest.raises(ValueError, match="nonempty"):
        measured_metrics([], {}, {}, experiment_id=identity)


def test_yolo_truth_empty_and_box_conversion():
    assert yolo_truth("", 100, 200) == []
    assert yolo_truth("3 0.5 0.5 0.2 0.4\n", 100, 200) == [
        {"class_id": 3, "box": [40.0, 60.0, 60.0, 140.0]}]


@pytest.mark.parametrize("label", [None, "0 0.5", "4 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2",
                                    "0 0.5 0.5 0 0.2", "0 0.99 0.5 0.2 0.2"])
def test_invalid_yolo_labels_fail(label):
    with pytest.raises((ValueError, AttributeError)):
        yolo_truth(label, 100, 200)


def test_metrics_preserve_class_errors_and_fixed_threshold_counts():
    row = {"image": "images/val/a.jpg", "truth": [{"class_id": 0, "box": [0, 0, 20, 20]}],
           "predictions": [{"class_id": 1, "box": [0, 0, 20, 20], "score": 0.9}], "tp": 0, "fp": 1, "fn": 1}
    result = measured_metrics([row], {"map50_95": 0.3}, {"smoke": {"map50_95": 0.2}})
    assert result["metrics"]["map50_95"] == 0.3
    assert result["metrics"]["precision"] == result["metrics"]["recall"] == 0
    assert result["per_class"]["smoke"]["fn"] == 1
    assert result["per_class"]["fire"]["fp"] == 1
    row["tp"] = 1
    with pytest.raises(ValueError, match="counts disagree"):
        measured_metrics([row], {}, {})


def test_empty_images_do_not_create_nonfinite_metrics():
    row = {"image": "images/val/negative.jpg", "truth": [], "predictions": [], "tp": 0, "fp": 0, "fn": 0}
    result = measured_metrics([row], {"map50_95": 0.0}, {})
    assert result["metrics"]["f1"] == 0
    json.dumps(result, allow_nan=False)


def test_completed_run_is_verified_noop_and_partial_run_is_error(tmp_path):
    run = tmp_path / "run"
    assert completed_run(run) is None
    run.mkdir()
    with pytest.raises(ValueError, match="without a completion"):
        completed_run(run)
    files = ["config.json", "metrics.json", "predictions.json", "ultralytics/weights/best.pt", "ultralytics/weights/last.pt"]
    for name in files:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    receipt = {"status": "completed", "sha256": {name: file_sha256(run / name) for name in files}}
    write_json(run / "completion.json", receipt)
    assert completed_run(run) == receipt
    assert completed_run(run) == receipt
    (run / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        completed_run(run)


def test_completed_run_rejects_missing_files_and_incomplete_receipt(tmp_path):
    write_json(tmp_path / "completion.json", {"status": "completed", "sha256": {}})
    with pytest.raises(ValueError, match="Incomplete"):
        completed_run(tmp_path)


def test_atomic_json_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"map50_95": 0.5})
    with pytest.raises(ValueError):
        write_json(path, {"map50_95": float("nan")})
    assert json.loads(path.read_text()) == {"map50_95": 0.5}


@pytest.mark.parametrize("trial,size,confidence", [("C1", 640, .25), ("E3", 960, .25), ("E4", 640, .1)])
def test_fixed_checkpoint_evaluation_changes_only_approved_factor(trial, size, confidence):
    from scripts.threats_artifacts import evaluation_spec
    spec = evaluation_spec(trial)
    assert spec["parent"] == training_spec("E1")["id"]
    assert spec["batch"] == 32
    assert (spec["imgsz"], spec["confidence"]) == (size, confidence)


@pytest.mark.parametrize("trial", [None, "", "E2", "../E3"])
def test_unknown_evaluation_fails(trial):
    from scripts.threats_artifacts import evaluation_spec
    with pytest.raises(ValueError, match="Evaluation must"):
        evaluation_spec(trial)


def test_inference_receipt_cannot_be_reused_as_training(tmp_path):
    files = ["config.json", "metrics.json", "predictions.json"]
    for name in files:
        (tmp_path / name).write_text(name)
    receipt = {"status": "completed", "mode": "inference_only",
               "sha256": {name: file_sha256(tmp_path / name) for name in files}}
    write_json(tmp_path / "completion.json", receipt)
    assert completed_run(tmp_path, inference_only=True) == receipt
    assert completed_run(tmp_path, inference_only=True) == receipt
    with pytest.raises(ValueError, match="training receipt"):
        completed_run(tmp_path)
    (tmp_path / "predictions.json").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        completed_run(tmp_path, inference_only=True)
