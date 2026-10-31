import json
from pathlib import Path

import pytest

from cv_scientist.confidence import ConfidenceResearch
from cv_scientist.evaluation import EvaluationError, ResolutionEvaluator
from cv_scientist.resolution import ResolutionProposal, ResolutionResearch
from cv_scientist.cli import main
from test_confidence_research import workspace, SCRIPT, Planner


@pytest.fixture
def resolution_workspace(workspace):
    script = SCRIPT.replace("confidence=float(a.conf)", "confidence=float(a.conf),imgsz=int(a.imgsz)")
    script = script.replace("c=float(a.conf)", "c=0.35 if int(a.imgsz)>=960 else 0.5")
    (Path(workspace.root) / "evaluate.py").write_text(script)
    return workspace


def proposal(value=960, **changes):
    return {"action": "evaluate", "parent_id": "F1", "imgsz": value,
            "hypothesis": "Resolution preserves recall with fewer FP", "expected_effect": "FP<28 and recall>=0.95",
            "reason": "Compare measured outcomes", **changes}


def test_adaptive_resolution_only_and_repeat_noop(resolution_workspace):
    ws = resolution_workspace
    planner = Planner([proposal(800), proposal(960)])
    report = ResolutionResearch(ws, planner).run("Reduce FP", "F1", budget=3)
    assert report["parameter"] == "imgsz" and report["status"] == "goal_met"
    assert [a["imgsz"] for a in report["attempts"]] == [800, 960]
    assert [a["status"] for a in report["attempts"]] == ["rejected", "supported"]
    exp = ws.memory.get_experiment(report["selected_run"])
    assert exp.config["imgsz"] == 960 and exp.config["confidence"] == 0.25
    assert all(exp.config[key] == value for key, value in report["fixed"].items())
    assert planner.contexts[1]["history"][-1]["imgsz"] == 800
    assert exp.parent_id == "F1"
    assert Path(exp.run_dir).is_relative_to(Path(ws.dir) / "runs")
    again = ResolutionResearch(ws, Planner([])).run("Same goal", "F1")
    assert again["selected_run"] == exp.id and not again["attempts"]


def test_resolution_duplicates_and_exhausted_budget(resolution_workspace):
    planner = Planner([proposal(640), proposal(800), proposal(800), proposal(832)])
    report = ResolutionResearch(resolution_workspace, planner).run("Reduce FP", "F1", budget=2)
    assert report["status"] == "budget_exhausted" and len(report["attempts"]) == 2
    assert [f["status"] for f in report["feedback"]] == ["duplicate", "duplicate"]
    resumed = ResolutionResearch(resolution_workspace, Planner([])).resume(report["path"])
    assert resumed["usage"]["requests"] == 4 and len(resumed["attempts"]) == 2


def test_resume_resolution_after_evaluation_without_rerun(resolution_workspace, monkeypatch):
    original = ResolutionEvaluator.evaluate
    def interrupt(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(ResolutionEvaluator, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt): ResolutionResearch(resolution_workspace, Planner([proposal()])).run("Reduce FP", "F1")
    path = next((Path(resolution_workspace.dir) / "reports").glob("research-*.json"))
    before = json.loads(path.read_text())
    monkeypatch.setattr(ResolutionEvaluator, "evaluate", original)
    report = ResolutionResearch(resolution_workspace, Planner([])).resume(path)
    assert report["status"] == "goal_met" and report["question_id"] == before["question_id"]
    assert len(report["attempts"]) == 1 and report["usage"]["requests"] == 1
    with pytest.raises(ValueError, match="parameter"): ConfidenceResearch(resolution_workspace, Planner([])).resume(path)


def test_resolution_rejects_changed_confidence_and_keeps_hypothesis_untested(resolution_workspace):
    script = Path(resolution_workspace.root) / "evaluate.py"
    script.write_text(script.read_text().replace("confidence=float(a.conf)", "confidence=0.5"))
    with pytest.raises(EvaluationError, match="frozen parameter"):
        ResolutionResearch(resolution_workspace, Planner([proposal()])).run("Reduce FP", "F1")
    assert len(resolution_workspace.memory.list_experiments()) == 1
    assert resolution_workspace.memory.list_hypotheses()[0].status == "untested"


@pytest.mark.parametrize("value", [None, True, 0, -32, 33, 640.0, 4128, "960", float("nan")])
def test_invalid_resolution_never_parses(value):
    with pytest.raises(ValueError): ResolutionProposal.parse({"content": json.dumps(proposal(value))}, "F1")


@pytest.mark.parametrize("value", [32, 4096])
def test_boundary_resolutions_parse(value):
    assert ResolutionProposal.parse({"content": json.dumps(proposal(value))}, "F1").imgsz == value


def test_stop_and_zero_budget_are_noops(resolution_workspace):
    report = ResolutionResearch(resolution_workspace, Planner([])).run("Reduce FP", "F1", budget=0)
    assert report["status"] == "budget_exhausted" and not report["attempts"]
    report = ResolutionResearch(resolution_workspace, Planner([proposal(None, action="stop")])).run("Reduce FP", "F1")
    assert report["status"] == "agent_stopped" and not report["attempts"]


@pytest.mark.parametrize("wrong_class", [False, True])
def test_class_aware_resolution_outputs_are_validated_and_diagnosed(resolution_workspace, wrong_class):
    ws = resolution_workspace
    root = Path(ws.root)
    artifact = root / "runs" / "F1" / "predictions.json"
    rows = json.loads(artifact.read_text())
    rows[0]["truth"] = [{"box": box, "class_id": "food"} for box in rows[0]["truth"]]
    for pred in rows[0]["predictions"]: pred["class_id"] = "food"
    artifact.write_text(json.dumps({"version": 2, "images": rows}))
    manifest = root / "data" / "manifest.json"
    value = json.loads(manifest.read_text())
    value["val"][0]["boxes"] = rows[0]["truth"]
    manifest.write_text(json.dumps(value))
    script = root / "evaluate.py"
    code = script.read_text().replace("json.loads((base/'predictions.json').read_text())[0]['truth']", "json.loads((base/'predictions.json').read_text())['images'][0]['truth']")
    code = code.replace("{'box':b,'score':0.9}", "{'box':b['box'],'score':0.9,'class_id':b['class_id']}")
    code = code.replace("'score':0.9} for i in range(fp)", "'score':0.9,'class_id':'food'} for i in range(fp)")
    code = code.replace("json.dumps([row])", "json.dumps({'version':2,'images':[row]})")
    if wrong_class: code = code.replace("'class_id':b['class_id']", "'class_id':'cup'")
    script.write_text(code)
    if wrong_class:
        with pytest.raises(EvaluationError, match="class-aware matching"):
            ResolutionResearch(ws, Planner([proposal()])).run("Reduce FP", "F1")
        assert len(ws.memory.list_experiments()) == 1
        return
    report = ResolutionResearch(ws, Planner([proposal()])).run("Reduce FP", "F1")
    from cv_scientist.diagnose import diagnose
    winner = ws.memory.get_experiment(report["selected_run"])
    assert report["status"] == "goal_met"
    assert diagnose(winner).classification["mode"] == "class_aware"


def test_cli_selects_resolution_and_infers_mode_on_resume(resolution_workspace, monkeypatch, capsys):
    ws = resolution_workspace
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "test-key")
    monkeypatch.setattr("cv_scientist.meta.MetaClient", lambda config: Planner([proposal()]))
    assert main(["--project", ws.root, "research", "--agent", "meta", "--parameter", "imgsz", "--baseline", "F1", "--goal", "Reduce FP"]) == 0
    assert "goal_met" in capsys.readouterr().out
    path = next((Path(ws.dir) / "reports").glob("research-*.json"))
    monkeypatch.setattr("cv_scientist.meta.MetaClient", lambda config: Planner([]))
    assert main(["--project", ws.root, "research", "--resume", str(path)]) == 0
    assert "goal_met" in capsys.readouterr().out
    assert main(["--project", ws.root, "research", "--resume", str(path), "--parameter", "confidence"]) == 2
    assert "parameter does not match" in capsys.readouterr().err


def test_cli_rejects_resolution_with_offline_planner(resolution_workspace, capsys):
    assert main(["--project", resolution_workspace.root, "research", "--parameter", "imgsz", "--goal", "Reduce FP"]) == 2
    assert "requires --agent meta" in capsys.readouterr().err


@pytest.mark.parametrize("existing", [False, True])
def test_resolution_freezes_project_specific_parameters(resolution_workspace, existing):
    ws = resolution_workspace
    baseline = ws.memory.get_experiment("F1")
    if existing:
        baseline.config["nms_iou"] = 0.7
        (Path(baseline.run_dir) / "config.json").write_text(json.dumps(baseline.config))
        ws.memory.add_experiment(baseline)
    script = Path(ws.root) / "evaluate.py"
    script.write_text(script.read_text().replace("cfg.update(experiment_id", "cfg['nms_iou']=0.3\ncfg.update(experiment_id"))
    with pytest.raises(EvaluationError, match="uncontrolled parameter|frozen parameter"):
        ResolutionResearch(ws, Planner([proposal()])).run("Reduce FP", "F1")
    assert len(ws.memory.list_experiments()) == 1
