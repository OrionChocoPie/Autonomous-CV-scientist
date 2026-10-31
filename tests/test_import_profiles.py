from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import pytest

from cv_scientist.cli import main
from cv_scientist.importing import ImportAgent
from cv_scientist.import_source import ImportSource, OUTPUT_DIR
from cv_scientist.ingestor import LocalExperimentSource
from cv_scientist.meta import MetaAPIError
from cv_scientist.workspace import ProjectWorkspace
from test_importing import legacy, Client, conversation


@pytest.fixture
def imported(legacy, tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path / "workspace"))
    report = ImportAgent(ImportSource(str(legacy)), Client(conversation()), workspace=ws).run()
    return ws, legacy, report


def events(ws):
    return [json.loads(line) for line in (Path(ws.dir) / "logs" / "operations.jsonl").read_text().splitlines()]


def test_local_copies_profiles_and_reports_are_self_contained(imported):
    ws, source, report = imported
    assert Path(report["path"]).is_relative_to(Path(ws.dir))
    assert Path(report["runs"][0]["copy"]).is_relative_to(Path(ws.dir) / "import" / "copies")
    profiles = list((Path(ws.dir) / "import" / "profiles").glob("*.json"))
    assert len(profiles) == 1
    assert json.loads(profiles[0].read_text())["recipe"]["metrics"][0]["pointer"] == "/0/mAP_50_95"
    assert not (source / OUTPUT_DIR).exists() and not (source / ".cv-scientist").exists()
    assert any(e["operation"] == "import.tool" for e in events(ws))


def test_repeat_after_restart_uses_profile_with_zero_api_and_immutable_copy(imported):
    ws, source, first = imported
    copy = Path(first["runs"][0]["copy"])
    original = {p.name: p.read_bytes() for p in copy.iterdir()}
    restarted = ProjectWorkspace(ws.root)
    report = ImportAgent(ImportSource(str(source)), None, workspace=restarted).run()
    assert report["profile_reused"] and report["usage"]["requests"] == 0
    assert report["runs"][0]["status"] == "unchanged"
    assert {p.name: p.read_bytes() for p in copy.iterdir()} == original
    assert any(e["status"] == "reused" for e in events(ws))


def test_new_values_and_new_run_reuse_format_without_api(imported):
    ws, source, first = imported
    shutil.copytree(source / "run", source / "run2")
    metrics = source / "run2" / "metrics.csv"
    metrics.write_text(metrics.read_text().replace("2539", "2500"))
    report = ImportAgent(ImportSource(str(source)), None, workspace=ws).run()
    assert len(report["runs"]) == 2 and report["usage"]["requests"] == 0
    second = next(row for row in report["runs"] if row["run_dir"] == "run2")
    exp = LocalExperimentSource().load_run(second["copy"])
    assert exp.metrics["fp"] == 2500 and exp.status == "unknown"
    assert second["experiment_id"] != first["runs"][0]["experiment_id"]


def test_saved_format_is_recognized_in_another_source(imported, tmp_path):
    ws, source, first = imported
    other = tmp_path / "another-source"
    shutil.copytree(source / "run", other / "fresh-run")
    report = ImportAgent(ImportSource(str(other)), None, workspace=ws).run()
    assert report["profile_reused"] and report["usage"]["requests"] == 0
    assert report["runs"][0]["experiment_id"] != first["runs"][0]["experiment_id"]


@pytest.mark.parametrize("change", ["header", "split", "missing", "invalid", "truncated"])
def test_changed_or_damaged_format_fails_explicitly_without_new_copy(imported, change):
    ws, source, first = imported
    metrics = source / "run" / "metrics.csv"
    before = set((Path(ws.dir) / "import" / "copies").glob("I*/*"))
    if change == "missing": metrics.unlink()
    elif change == "header": metrics.write_text(",AP,FPs,recall\nval,0.8,1,0.9\n")
    elif change == "split": metrics.write_text(metrics.read_text().replace("val,", "train,"))
    elif change == "invalid": metrics.write_text(metrics.read_text().replace("2539", "-1"))
    else: metrics.write_text(",mAP_50_95,FPs,recall\nval,0.8\n")
    with pytest.raises((OSError, ValueError)):
        ImportAgent(ImportSource(str(source)), None, workspace=ws).run()
    assert set((Path(ws.dir) / "import" / "copies").glob("I*/*")) == before
    assert events(ws)[-1]["status"] == "failed"


def test_corrupt_profile_is_not_silently_replaced(imported):
    ws, source, _ = imported
    path = next((Path(ws.dir) / "import" / "profiles").glob("*.json"))
    path.write_text('{"version":1}')
    with pytest.raises(ValueError, match="corrupt"):
        ImportAgent(ImportSource(str(source)), None, workspace=ws).run()
    assert path.read_text() == '{"version":1}'


def test_unknown_format_requires_model_but_empty_source_does_not(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    ws = ProjectWorkspace(tmp_path)
    assert ImportAgent(ImportSource(str(source)), None, workspace=ws).run()["runs"] == []
    (source / "metrics.json").write_text('{"recall":0.9}')
    with pytest.raises(MetaAPIError, match="profile"):
        ImportAgent(ImportSource(str(source)), None, workspace=ws).run()


def test_all_cli_commands_append_success_and_failure_without_raw_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "never-record-this-key")
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "note", "never-record-this-key"]) == 0
    assert main(["--project", str(tmp_path), "compare", "missing", "other"]) == 2
    ws = ProjectWorkspace(str(tmp_path))
    log = (Path(ws.dir) / "logs" / "operations.jsonl").read_text()
    assert "never-record-this-key" not in log
    assert [(e["operation"], e["status"]) for e in events(ws)] == [
        ("cli.init", "started"), ("cli.init", "completed"), ("cli.note", "started"), ("cli.note", "completed"),
        ("cli.compare", "started"), ("cli.compare", "failed")]


def test_journal_concurrent_writes_are_complete_and_redacted(tmp_path, monkeypatch):
    ws = ProjectWorkspace.init(str(tmp_path))
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "sensitive-test-value")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: ws.record_event("test", "completed", index=i, api_key="sensitive-test-value", value="sensitive-test-value"), range(40)))
    rows = events(ws)
    assert len(rows) == 40 and {row["index"] for row in rows} == set(range(40))
    assert all(row["api_key"] == "[redacted]" and row["value"] == "[redacted]" for row in rows)


def test_journal_rejects_symlink_outputs(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path / "project"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(ws.dir) / "logs").rmdir()
    (Path(ws.dir) / "logs").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"): ws.record_event("test", "completed")
    assert list(outside.iterdir()) == []


def test_journal_accepts_an_aliased_project_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    ws = ProjectWorkspace.init(str(alias))
    ws.record_event("test", "completed")
    assert events(ws)[-1]["status"] == "completed"


def test_watcher_records_scan_errors_import_errors_and_success(tmp_path):
    from cv_scientist.sessions import WatchService
    ws = ProjectWorkspace.init(str(tmp_path))
    runs = tmp_path / "runs"
    watcher = WatchService(ws, str(runs))
    assert watcher.scan_once() == []
    run = runs / "E1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"recall":')
    assert watcher.scan_once() == []
    (run / "metrics.json").write_text('{"recall":0.9}')
    assert watcher.scan_once()[0].id == "E1"
    assert [(e["operation"], e["status"]) for e in events(ws)] == [
        ("watch.scan", "failed"), ("watch.ingest", "failed"), ("watch.ingest", "completed")]
