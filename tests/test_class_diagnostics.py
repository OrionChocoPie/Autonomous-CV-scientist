import json

import pytest

from cv_scientist.diagnose import diagnose, summarize
from cv_scientist.models import Experiment
from cv_scientist.predictions import load_predictions, match_boxes
from cv_scientist.agent_tools import AnalysisTools
from cv_scientist.workspace import ProjectWorkspace


def artifact(tmp_path, images, version=2):
    value = {"version": version, "images": images} if version else images
    (tmp_path / "predictions.json").write_text(json.dumps(value))
    return Experiment(id="E1", run_dir=str(tmp_path))


def labelled(image="one.png", truth="food", predicted="cup"):
    return {"image": image, "truth": [{"box": [0, 0, 10, 10], "class_id": truth}],
            "predictions": [{"box": [0, 0, 10, 10], "score": 0.9, "class_id": predicted}],
            "tp": int(truth == predicted), "fp": int(truth != predicted), "fn": int(truth != predicted)}


def test_wrong_class_is_fp_and_fn_with_confusion_and_evidence(tmp_path):
    exp = artifact(tmp_path, [labelled()])
    record = load_predictions(exp)[0]
    assert match_boxes(record, 0.5)[:2] == (set(), set())
    diagnosis = diagnose(exp)
    assert diagnosis.classification["mode"] == "class_aware"
    assert diagnosis.classification["matrix"] == [{"truth": "food", "prediction": "cup", "count": 1}]
    assert diagnosis.classification["totals"] == {"tp": 0, "fp": 1, "fn": 1, "classification": 1}
    finding = next(f for f in diagnosis.findings if f.category == "classification")
    assert finding.evidence[0]["id"] == "E1:predictions.json#0"
    assert "class_aware" in summarize(diagnosis)


def test_correct_class_duplicates_missing_and_novel_classes(tmp_path):
    correct = labelled(predicted="food")
    correct["predictions"].append({"box": [0, 0, 10, 10], "score": 0.8, "class_id": "novel"})
    correct.update(fp=1)
    missed = {"image": "missing.png", "truth": [{"box": [0, 0, 10, 10], "class_id": "cup"}],
              "predictions": [], "tp": 0, "fp": 0, "fn": 1}
    data = diagnose(artifact(tmp_path, [correct, missed])).classification
    assert data["totals"] == {"tp": 1, "fp": 1, "fn": 1, "classification": 0}
    assert {tuple(row.values()) for row in data["matrix"]} == {("food", "food", 1), (None, "novel", 1), ("cup", None, 1)}


def test_low_iou_wrong_class_is_localization_not_classification(tmp_path):
    row = labelled()
    row["predictions"][0]["box"] = [8, 0, 18, 10]
    d = diagnose(artifact(tmp_path, [row]))
    assert d.classification["totals"]["classification"] == 0
    assert any(f.slice == "objects:overlap:localization" for f in d.findings)


def test_legacy_format_and_absent_artifact_are_explicit(tmp_path):
    row = {"image": "one.png", "truth": [[0, 0, 10, 10]],
           "predictions": [{"box": [0, 0, 10, 10], "score": 0.9}], "fp": 0, "fn": 0}
    d = diagnose(artifact(tmp_path, [row], version=None))
    assert d.classification["mode"] == "class_agnostic" and d.classification["limitations"]
    assert match_boxes(load_predictions(Experiment(run_dir=str(tmp_path), id="E1"))[0], 0.5)[:2] == ({0}, {0})
    (tmp_path / "predictions.json").unlink()
    assert diagnose(Experiment(id="E1", run_dir=str(tmp_path))).classification["mode"] == "unavailable"


def test_empty_v2_is_valid(tmp_path):
    assert diagnose(artifact(tmp_path, [])).classification["matrix"] == []
    empty = {"image": "empty.png", "truth": [], "predictions": [], "tp": 0, "fp": 0, "fn": 0}
    assert diagnose(artifact(tmp_path, [empty])).classification["totals"] == {"tp": 0, "fp": 0, "fn": 0, "classification": 0}


def test_agent_selects_class_errors_even_if_exported_counts_disagree(tmp_path):
    row = labelled()
    row.update(tp=1, fp=0, fn=0)
    exp = artifact(tmp_path, [row])
    ws = ProjectWorkspace.init(str(tmp_path))
    ws.memory.add_experiment(exp)
    examples = AnalysisTools(ws).get_error_examples("E1", "fp", 2)
    assert examples["examples"][0]["image"] == "one.png"


def test_integer_class_ids_and_exact_iou_boundary(tmp_path):
    row = labelled(truth=0, predicted=1)
    row["predictions"][0]["box"] = [0, 0, 5, 10]
    exp = artifact(tmp_path, [row])
    assert diagnose(exp).classification["totals"]["classification"] == 1
    exp.config["iou_match"] = 0.500001
    assert diagnose(exp).classification["totals"]["classification"] == 0


@pytest.mark.parametrize("change", ["missing_prediction_class", "legacy_gt", "none_class", "bool_class", "empty_class", "mixed_types", "version"])
def test_incomplete_or_invalid_labels_fail(tmp_path, change):
    row = labelled()
    version = 2
    if change == "missing_prediction_class": row["predictions"][0].pop("class_id")
    elif change == "legacy_gt": row["truth"] = [[0, 0, 10, 10]]
    elif change == "mixed_types": row["predictions"][0]["class_id"] = 1
    elif change == "version": version = 99
    else: row["truth"][0]["class_id"] = {"none_class": None, "bool_class": True, "empty_class": ""}[change]
    with pytest.raises(ValueError): load_predictions(artifact(tmp_path, [row], version))
