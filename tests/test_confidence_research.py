from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cv_scientist.cli import main
from cv_scientist.confidence import ConfidenceProposal, ConfidenceResearch
from cv_scientist.evaluation import ConfidenceEvaluator, EvaluationError
from cv_scientist.ingestor import RunIngestor
from cv_scientist.meta import MetaAPIError
from cv_scientist.workspace import ProjectWorkspace


SCRIPT = '''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command',choices=['evaluate'])
for key in ('run','parent','weights','imgsz','conf','device'): p.add_argument('--'+key,required=True)
a=p.parse_args()
root=Path(__file__).parent
base=root/'runs'/a.parent
cfg=json.loads((base/'config.json').read_text())
cfg.update(experiment_id=a.run,name=a.run,parent=a.parent,confidence=float(a.conf))
run=root/'runs'/a.run
run.mkdir()
c=float(a.conf)
tp,fp=(0,0) if c==1 else ((18,10) if c>=0.4 else (19,22))
truth=json.loads((base/'predictions.json').read_text())[0]['truth']
predictions=[{'box':b,'score':0.9} for b in truth[:tp]]
predictions += [{'box':[100+i,100,101+i,101],'score':0.9} for i in range(fp)]
row={'image':'images/val/one.png','truth':truth,'predictions':predictions,'tp':tp,'fp':fp,'fn':20-tp}
metrics={'tp':tp,'fp':fp,'fn':20-tp,'count':1,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/20,'map50_95':0.7}
(run/'config.json').write_text(json.dumps(cfg))
(run/'metrics.json').write_text(json.dumps({'id':a.run,'metrics':metrics,'slices':{'small':metrics},'per_class':{'person':metrics}}))
(run/'predictions.json').write_text(json.dumps([row]))
'''


@pytest.fixture
def workspace(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    ws.config.update(dataset_root="data", evaluation_script="evaluate.py", evaluation_python=".venv/bin/python")
    with open(ws.config_path, "a") as file:
        file.write("dataset_root: data\nevaluation_script: evaluate.py\nevaluation_python: .venv/bin/python\n")
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (tmp_path / "evaluate.py").write_text(SCRIPT)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixed-test-weights")
    data = tmp_path / "data"
    (data / "images" / "val").mkdir(parents=True)
    (data / "labels" / "val").mkdir(parents=True)
    (data / "images" / "val" / "one.png").write_bytes(b"test-image")
    (data / "labels" / "val" / "one.txt").write_text("test-label")
    (data / "data.yaml").write_text("val: images/val\n")
    truth = [[i*2, 0, i*2+1, 1] for i in range(20)]
    (data / "manifest.json").write_text(json.dumps({"train": [], "val": [{"image": "images/val/one.png", "label": "labels/val/one.txt", "boxes": truth}]}))
    run = tmp_path / "runs" / "F1"
    run.mkdir(parents=True)
    config = {"experiment_id": "F1", "model": "weights.pt", "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "imgsz": 640, "confidence": 0.25, "iou_match": 0.5, "seed": 42, "device": "cpu",
              "dataset_version": "fixed-v1", "validation_images": 1, "small_slice": "frames"}
    predictions = [{"box": b, "score": 0.9} for b in truth] + [{"box": [100+i, 100, 101+i, 101], "score": 0.9} for i in range(28)]
    metrics = {"tp": 20, "fp": 28, "fn": 0, "count": 1, "precision": 20/48, "recall": 1.0, "map50_95": 0.7}
    (run / "config.json").write_text(json.dumps(config))
    (run / "metrics.json").write_text(json.dumps({"id": "F1", "metrics": metrics}))
    (run / "predictions.json").write_text(json.dumps([{"image": "images/val/one.png", "truth": truth,
        "predictions": predictions, "tp": 20, "fp": 28, "fn": 0}]))
    RunIngestor(ws.memory).ingest(str(run))
    return ws


def proposal(confidence=0.5, **changes):
    return {"action": "evaluate", "parent_id": "F1", "confidence": confidence,
            "hypothesis": "This threshold reduces FP while preserving required recall", "expected_effect": "FP<28 and recall>=0.95",
            "reason": "Choose a measured follow-up", **changes}


class Planner:
    def __init__(self, proposals):
        self.config = SimpleNamespace(model="test-planner")
        self.proposals = iter(proposals)
        self.contexts = []

    def complete(self, messages, tools, response_format, timeout=None):
        self.contexts.append(json.loads(messages[-1]["content"]))
        assert tools == [] and 0 < timeout <= 600
        value = next(self.proposals)
        if isinstance(value, Exception): raise value
        return {"role": "assistant", "content": json.dumps(value)}, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}


def test_adaptive_loop_executes_measures_remembers_and_stops(workspace):
    planner = Planner([proposal(0.5), proposal(0.35)])
    report = ConfidenceResearch(workspace, planner).run("Reduce FP", "F1")
    assert report["status"] == "goal_met" and len(report["attempts"]) == 2
    assert [a["status"] for a in report["attempts"]] == ["rejected", "supported"]
    first = planner.contexts[1]["history"][-1]
    assert first["metrics"]["recall"] == 0.9 and first["metrics"]["fp"] == 10
    assert planner.contexts[1]["remaining_experiments"] == 3
    winner = workspace.memory.get_experiment(report["selected_run"])
    assert winner.metrics["recall"] == 0.95 and winner.metrics["fp"] == 22
    assert winner.parent_id == "F1" and winner.config["confidence"] == 0.35
    assert winner.config["research_signature"] == report["signature"]
    assert workspace.memory.get_question(report["question_id"]).status == "answered"
    assert {h.status for h in workspace.memory.list_hypotheses()} == {"supported", "rejected"}
    assert len(workspace.memory.list_conclusions()) == 3
    assert json.loads(Path(report["path"]).read_text())["selected_run"] == winner.id
    assert not (Path(workspace.dir) / "state" / "confidence.lock").exists()


def test_repeat_reuses_verified_result_without_api_or_execution(workspace):
    first = ConfidenceResearch(workspace, Planner([proposal(0.35)])).run("Reduce FP", "F1")
    ws = ProjectWorkspace(workspace.root)
    planner = Planner([])
    second = ConfidenceResearch(ws, planner).run("Reduce FP again", "F1")
    assert second["selected_run"] == first["selected_run"]
    assert not second["attempts"] and not planner.contexts
    assert second["path"] != first["path"]
    assert len(ws.memory.list_experiments()) == 2


def test_duplicates_are_feedback_not_new_evaluations(workspace):
    planner = Planner([proposal(0.2500000001), proposal(0.5), proposal(0.5), proposal(0.35)])
    report = ConfidenceResearch(workspace, planner).run("Reduce FP", "F1", budget=3)
    assert len(report["attempts"]) == 2
    assert [f["status"] for f in report["feedback"]] == ["duplicate", "duplicate"]
    assert planner.contexts[1]["feedback"][0]["experiment_id"] == "F1"


def test_four_evaluation_limit_and_negative_conclusion(workspace):
    planner = Planner([proposal(value) for value in (0.45, 0.5, 0.7, 1.0)])
    report = ConfidenceResearch(workspace, planner).run("Reduce FP", "F1")
    assert report["status"] == "budget_exhausted" and len(report["attempts"]) == 4
    assert report["selected_run"] is None and report["best_feasible_run"] == "F1"
    assert report["attempts"][-1]["metrics"]["tp"] == 0
    assert "4 new evaluations" in report["reason"]


def test_zero_budget_and_already_satisfied_baseline_are_noops(workspace):
    planner = Planner([])
    runner = ConfidenceResearch(workspace, planner)
    report = runner.run("Reduce FP", "F1", budget=0)
    assert report["status"] == "budget_exhausted" and not report["attempts"]
    report = runner.run("Already enough", "F1", max_fp=29)
    assert report["status"] == "goal_met" and report["selected_run"] == "F1"
    assert not report["attempts"] and not planner.contexts


def test_repeated_bad_proposals_have_bounded_api_budget(workspace):
    planner = Planner([proposal(0.25)] * 8)
    report = ConfidenceResearch(workspace, planner).run("Reduce FP", "F1")
    assert report["status"] == "no_progress" and len(planner.contexts) == 8
    assert not report["attempts"]


def test_invalid_proposal_is_corrected_with_feedback(workspace):
    planner = Planner([proposal(0.35, parent_id="OTHER"), proposal(0.35)])
    report = ConfidenceResearch(workspace, planner).run("Reduce FP", "F1")
    assert report["status"] == "goal_met" and len(report["attempts"]) == 1
    assert planner.contexts[1]["feedback"][0]["status"] == "invalid_proposal"


def test_agent_can_stop_without_faking_an_experiment(workspace):
    report = ConfidenceResearch(workspace, Planner([proposal(None, action="stop", reason="No justified experiment")])).run("Reduce FP", "F1")
    assert report["status"] == "agent_stopped" and not report["attempts"] and report["selected_run"] is None


@pytest.mark.parametrize("changes", [{"confidence": None}, {"confidence": float("nan")}, {"confidence": True},
                                      {"confidence": -1}, {"confidence": 2}, {"parent_id": "OTHER"},
                                      {"action": "train"}, {"command": "anything"}, {"hypothesis": ""}])
def test_bad_structured_proposals(changes):
    with pytest.raises(ValueError): ConfidenceProposal.parse({"content": json.dumps(proposal(**changes))}, "F1")


@pytest.mark.parametrize("arguments", [{"budget": -1}, {"budget": 5}, {"budget": None}, {"min_recall": 1.1},
                                        {"min_recall": float("nan")}, {"max_fp": 0}, {"timeout": 0}, {"timeout": float("inf")}])
def test_invalid_run_limits_never_call_api(workspace, arguments):
    planner = Planner([])
    with pytest.raises(ValueError): ConfidenceResearch(workspace, planner).run("Reduce FP", "F1", **arguments)
    assert not planner.contexts


@pytest.mark.parametrize("goal", ["", " ", None])
def test_empty_goal(workspace, goal):
    with pytest.raises(ValueError): ConfidenceResearch(workspace, Planner([])).run(goal, "F1")


def test_failed_process_saves_partial_report_and_untested_hypothesis(workspace):
    (Path(workspace.root) / "evaluate.py").write_text("raise SystemExit(7)\n")
    with pytest.raises(EvaluationError, match="code 7"):
        ConfidenceResearch(workspace, Planner([proposal()])).run("Reduce FP", "F1")
    report = json.loads(next((Path(workspace.dir) / "reports").glob("research-*.json")).read_text())
    assert report["status"] == "error" and report["attempts"][0]["status"] == "failed"
    assert workspace.memory.list_hypotheses()[0].status == "untested"
    assert len(workspace.memory.list_experiments()) == 1
    assert Path(report["attempts"][0]["log"]).is_file()


def test_subprocess_timeout_stops_owned_evaluation(workspace):
    (Path(workspace.root) / "evaluate.py").write_text("import time\ntime.sleep(10)\n")
    with pytest.raises(EvaluationError, match="timed out|wall-clock"):
        ConfidenceResearch(workspace, Planner([proposal()])).run("Reduce FP", "F1", timeout=0.1)
    assert len(workspace.memory.list_experiments()) == 1


def test_api_failure_is_persisted_without_fallback(workspace):
    with pytest.raises(MetaAPIError, match="HTTP 401"):
        ConfidenceResearch(workspace, Planner([MetaAPIError("Meta API HTTP 401")])).run("Reduce FP", "F1")
    report = json.loads(next((Path(workspace.dir) / "reports").glob("research-*.json")).read_text())
    assert report["status"] == "error" and not report["attempts"]


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 100000, "completion_tokens": 1, "total_tokens": 100001}])
def test_unreported_or_exhausted_api_budget_never_executes(workspace, usage):
    class BudgetPlanner(Planner):
        def complete(self, *args, **kwargs):
            message, _ = super().complete(*args, **kwargs)
            return message, usage
    with pytest.raises(MetaAPIError, match="usage|token budget"):
        ConfidenceResearch(workspace, BudgetPlanner([proposal(0.35)])).run("Reduce FP", "F1")
    assert len(workspace.memory.list_experiments()) == 1


def test_frozen_weights_and_missing_files_are_rejected(workspace):
    baseline = workspace.memory.get_experiment("F1")
    evaluator = ConfidenceEvaluator(workspace, baseline)
    weights = Path(workspace.root) / "weights.pt"
    weights.write_bytes(b"changed")
    with pytest.raises(EvaluationError, match="Frozen"): evaluator.check_inputs()
    with pytest.raises(EvaluationError, match="SHA256"): ConfidenceEvaluator(workspace, baseline)
    weights.unlink()
    with pytest.raises(FileNotFoundError): ConfidenceEvaluator(workspace, baseline)


def test_imported_helpers_are_frozen_too(workspace):
    helper = Path(workspace.root) / "metrics.py"
    helper.write_text("version=1\n")
    evaluator = ConfidenceEvaluator(workspace, workspace.memory.get_experiment("F1"))
    helper.write_text("version=2\n")
    with pytest.raises(EvaluationError, match="metrics.py"): evaluator.check_inputs()


def test_bad_baseline_metadata_fails_before_api(workspace):
    baseline = workspace.memory.get_experiment("F1")
    baseline.config.pop("model")
    with pytest.raises(ValueError, match="missing"): ConfidenceEvaluator(workspace, baseline)


def test_input_change_during_evaluation_invalidates_result(workspace):
    script = Path(workspace.root) / "evaluate.py"
    script.write_text(SCRIPT + "\n(root/'weights.pt').write_bytes(b'changed-during-evaluation')\n")
    with pytest.raises(EvaluationError, match="Frozen"):
        ConfidenceResearch(workspace, Planner([proposal(0.35)])).run("Reduce FP", "F1")
    assert len(workspace.memory.list_experiments()) == 1


def test_invalid_export_not_ingested(workspace):
    script = Path(workspace.root) / "evaluate.py"
    script.write_text(SCRIPT.replace("cfg.update(experiment_id", "cfg['imgsz']=128\ncfg.update(experiment_id"))
    with pytest.raises(EvaluationError, match="frozen parameter"):
        ConfidenceResearch(workspace, Planner([proposal(0.35)])).run("Reduce FP", "F1")
    assert len(workspace.memory.list_experiments()) == 1


def test_existing_run_is_preserved_and_lock_is_not_deleted(workspace):
    evaluator = ConfidenceEvaluator(workspace, workspace.memory.get_experiment("F1"))
    with pytest.raises(FileExistsError): evaluator.evaluate(0.3, "F1", 10, Path(workspace.dir)/"reports"/"log")
    lock = Path(workspace.dir) / "state" / "confidence.lock"
    lock.write_text("another run")
    with pytest.raises(EvaluationError, match="Another"):
        ConfidenceResearch(workspace, Planner([])).run("Reduce FP", "F1")
    assert lock.read_text() == "another run"


def test_cli_missing_baseline_does_not_execute(workspace, capsys):
    assert main(["--project", workspace.root, "research", "--goal", "Reduce FP", "--agent", "meta"]) == 2
    assert "requires --baseline" in capsys.readouterr().err
