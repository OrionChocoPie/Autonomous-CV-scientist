import json

import pytest

from cv_scientist import diagnostics
from cv_scientist.agent_tools import AnalysisTools
from cv_scientist.diagnose import diagnose
from cv_scientist.models import Experiment
from cv_scientist.workspace import ProjectWorkspace


def experiment(tmp_path, rows):
    (tmp_path / "predictions.json").write_text(json.dumps(rows))
    return Experiment(id="E1", run_dir=str(tmp_path))


def row(image, truth, predictions, tp, fp, fn):
    return {"image": image, "truth": truth, "predictions": predictions, "tp": tp, "fp": fp, "fn": fn}


def test_groups_errors_by_object_size_confidence_and_overlap(tmp_path):
    rows = [row("small.png", [[0, 0, 10, 10]], [], 0, 0, 1),
            row("medium.png", [[0, 0, 50, 50]], [{"box": [30, 0, 80, 50], "score": 0.9}], 0, 1, 1),
            row("large.png", [[0, 0, 100, 100]], [{"box": [0, 0, 100, 100], "score": 0.95},
                {"box": [0, 0, 100, 100], "score": 0.85}, {"box": [200, 200, 210, 210], "score": 0.2}], 1, 2, 0)]
    findings = diagnostics.analyze_prediction_errors(experiment(tmp_path, rows))
    groups = {(f.category, f.slice): f for f in findings}
    assert groups["false_negative_geometry", "objects:size:small"].value == 1
    assert groups["false_negative_geometry", "objects:size:medium"].value == 1
    assert groups["false_positive_geometry", "objects:overlap:duplicate"].value == 1
    assert groups["false_positive_geometry", "objects:overlap:background"].value == 1
    assert groups["false_positive_geometry", "objects:confidence:high"].value == 2
    assert groups["false_positive_geometry", "objects:confidence:low"].value == 1
    for finding in findings:
        assert finding.evidence and all(ref["kind"] == "artifact" for ref in finding.evidence)
        assert all(ref["id"].startswith("E1:predictions.json#") for ref in finding.evidence)


def test_crowded_fn_and_size_boundaries(tmp_path):
    rows = [row("crowd.png", [[0, 0, 32, 32], [1, 1, 33, 33]], [{"box": [0, 0, 32, 32], "score": 0.9}], 1, 0, 1),
            row("large.png", [[0, 0, 96, 96]], [], 0, 0, 1)]
    findings = diagnostics.analyze_prediction_errors(experiment(tmp_path, rows))
    groups = {f.slice: f.value for f in findings}
    assert groups["objects:size:medium"] == 1 and groups["objects:size:large"] == 1
    assert groups["objects:overlap:crowded"] == 1


def test_geometry_diagnostics_are_available_to_cli_and_agent_with_citable_evidence(tmp_path):
    exp = experiment(tmp_path, [row("one.png", [[0, 0, 10, 10]], [], 0, 0, 1)])
    assert any(f.category == "false_negative_geometry" for f in diagnose(exp).findings)
    ws = ProjectWorkspace.init(str(tmp_path))
    ws.memory.add_experiment(exp)
    tools = AnalysisTools(ws)
    diagnosis = tools.diagnose_experiment("E1")
    refs = [r for f in diagnosis["findings"] for r in f["evidence"] if r["kind"] == "artifact"]
    assert refs and all((r["kind"], r["id"]) in tools.evidence for r in refs)


@pytest.mark.parametrize("records", [None, {}, [None], [row("bad.png", [[0, 0, 0, 1]], [], 0, 0, 1)],
    [row("bad.png", [[0, 0, float("nan"), 1]], [], 0, 0, 1)],
    [row("bad.png", [], [{"box": [0, 0, 1, 1], "score": 1.1}], 0, 1, 0)],
    [row("../escape.png", [], [], 0, 0, 0)]])
def test_invalid_prediction_geometry_is_explicit(tmp_path, records):
    with pytest.raises(ValueError): diagnostics.analyze_prediction_errors(experiment(tmp_path, records))


def test_absent_and_empty_artifacts_are_optional_but_truncated_artifacts_error(tmp_path):
    exp = Experiment(id="E1", run_dir=str(tmp_path))
    assert diagnostics.analyze_prediction_errors(exp) == []
    assert diagnostics.analyze_prediction_errors(experiment(tmp_path, [])) == []
    (tmp_path / "predictions.json").write_text('[{"image":')
    with pytest.raises(ValueError): diagnostics.analyze_prediction_errors(exp)


def test_exported_counts_cannot_hide_matching_disagreements(tmp_path):
    exp = experiment(tmp_path, [row("one.png", [[0, 0, 10, 10]], [], 1, 0, 0)])
    findings = diagnostics.analyze_prediction_errors(exp)
    assert any(f.category == "artifact_consistency" for f in findings)
    assert any(f.category == "false_negative_geometry" and f.value == 1 for f in findings)
