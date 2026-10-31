import json
from pathlib import Path

import pytest

from cv_scientist.analysis import RunAnalysisService
from cv_scientist.cli import main
from cv_scientist.ingestor import LocalExperimentSource, RunIngestor
from cv_scientist.sessions import WatchService
from cv_scientist.workspace import ProjectWorkspace


@pytest.fixture
def project(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    run = tmp_path / "runs" / "E1"
    run.mkdir(parents=True)
    (run / "config.json").write_text('{"experiment_id":"E1"}')
    (run / "metrics.json").write_text(json.dumps({"metrics": {"recall": 0.5}, "slices": {"small": {"recall": 0.2, "fn": 8}}}))
    return ws, run


@pytest.mark.parametrize("filename,contents", [("metrics.json", '{"metrics":'), ("config.json", '{"seed":'), ("metrics.json", "null"), ("metrics.json", "[]")])
def test_malformed_sources_do_not_become_empty_experiments(project, filename, contents):
    ws, run = project
    (run / filename).write_text(contents)
    with pytest.raises(ValueError): RunIngestor(ws.memory).ingest(str(run))
    assert ws.memory.list_experiments() == [] and ws.memory.timeline() == []


def test_missing_metrics_are_an_error_not_a_completed_run(project):
    ws, run = project
    (run / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError): RunIngestor(ws.memory).ingest(str(run))
    assert ws.memory.list_experiments() == []


def test_repeated_ingestion_preserves_identity_parent_time_and_events(project):
    ws, run = project
    first = RunIngestor(ws.memory).ingest(str(run))
    second_run = run.parent / "E2"
    second_run.mkdir()
    (second_run / "metrics.json").write_text('{"recall":0.7}')
    RunIngestor(ws.memory).ingest(str(second_run))
    before = len(ws.memory.timeline())
    second = RunIngestor(ws.memory).ingest(str(run))
    assert second.to_dict() == first.to_dict()
    assert len(ws.memory.timeline()) == before


def test_repeated_analysis_is_noop(project):
    ws, run = project
    exp = RunIngestor(ws.memory).ingest(str(run))
    service = RunAnalysisService(ws.memory)
    first = service.analyze(exp.id)
    before = len(ws.memory.timeline())
    second = service.analyze(exp.id)
    assert first.to_dict() == second.to_dict()
    assert len(ws.memory.timeline()) == before


def test_watcher_retries_corrected_json_and_survives_restart(project):
    ws, run = project
    valid = (run / "metrics.json").read_text()
    (run / "metrics.json").write_text('{"metrics":')
    watcher = WatchService(ws, str(run.parent))
    assert watcher.scan_once() == []
    assert not ws.memory.list_experiments()
    (run / "metrics.json").write_text(valid)
    assert [e.id for e in watcher.scan_once()] == ["E1"]
    assert watcher.scan_once() == []
    before = len(ws.memory.timeline())
    restarted = WatchService(ProjectWorkspace(ws.root), str(run.parent))
    assert restarted.scan_once() == []
    assert len(ws.memory.timeline()) == before


def test_analysis_failure_rolls_back_and_retry_succeeds(project, monkeypatch):
    ws, run = project
    watcher = WatchService(ws, str(run.parent))
    original = watcher.analysis.analyze
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("Interrupted after analysis writes")
    monkeypatch.setattr(watcher.analysis, "analyze", fail)
    assert watcher.scan_once() == []
    assert ws.memory.list_experiments() == [] and ws.memory.timeline() == []
    monkeypatch.setattr(watcher.analysis, "analyze", original)
    assert len(watcher.scan_once()) == 1


def test_watcher_waits_for_metrics_and_reimports_updated_result(project):
    ws, run = project
    (run / "metrics.json").unlink()
    watcher = WatchService(ws, str(run.parent))
    assert watcher.scan_once() == []
    (run / "metrics.json").write_text('{"recall":0.7}')
    assert len(watcher.scan_once()) == 1
    assert watcher.scan_once() == []
    (run / "metrics.json").write_text('{"recall":0.8}')
    assert len(watcher.scan_once()) == 1
    assert ws.memory.get_experiment("E1").metrics["recall"] == 0.8


@pytest.mark.parametrize("data", [{"recall": float("nan")}, {"metrics": {"recall": "oops"}}, {"metrics": {"recall": None}}, {"slices": []}])
def test_invalid_metric_values_are_rejected(project, data):
    ws, run = project
    (run / "metrics.json").write_text(json.dumps(data))
    with pytest.raises(ValueError): RunIngestor(ws.memory).ingest(str(run))


def test_yaml_config_preserves_nested_structure(project):
    ws, run = project
    (run / "config.json").unlink()
    (run / "config.yaml").write_text("train:\n  seed: 42\n  img_size: [640, 640]\n")
    exp = RunIngestor(ws.memory).ingest(str(run))
    assert exp.config == {"train": {"seed": 42, "img_size": [640, 640]}}


def test_empty_metrics_placeholder_is_not_a_completed_experiment(project):
    ws, run = project
    (run / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="measured"): RunIngestor(ws.memory).ingest(str(run))
    assert ws.memory.list_experiments() == []


def test_cli_rolls_back_ingestion_when_prediction_analysis_fails(project, capsys):
    ws, run = project
    (run / "predictions.json").write_text('[{"image":')
    assert main(["--project", ws.root, "ingest", str(run)]) == 2
    assert ws.memory.list_experiments() == [] and ws.memory.timeline() == []
    assert "error" in capsys.readouterr().err.lower()
