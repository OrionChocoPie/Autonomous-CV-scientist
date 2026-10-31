import json
from pathlib import Path

import pytest

from cv_scientist.cli import main
from cv_scientist.models import Experiment, Hypothesis, ResearchConclusion, ResearchQuestion
from cv_scientist.workspace import ProjectWorkspace


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user-config"))
    ws = ProjectWorkspace.init(str(tmp_path))
    for identity, recall in (("B0", 0.95), ("E1", 0.8)):
        run = tmp_path / "runs" / identity
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"experiment_id": identity, "confidence": 0.25}))
        (run / "metrics.json").write_text(json.dumps({"recall": recall, "precision": 0.9}))
        ws.memory.add_experiment(Experiment(id=identity, run_dir=str(run), metrics={"recall": recall, "precision": 0.9},
                                            config={"confidence": 0.25}, slices={"small": {"recall": recall}}))
    return ws


def response(capsys, command, ok=True):
    output = capsys.readouterr()
    value = json.loads(output.out)
    assert set(value) == {"schema_version", "command", "ok", "data", "error"}
    assert value["schema_version"] == 1 and value["command"] == command
    assert value["ok"] is ok
    assert output.err == ""
    if ok:
        assert isinstance(value["data"], dict) and value["error"] is None
    else:
        assert value["data"] is None and set(value["error"]) == {"code", "message"}
    return value


@pytest.mark.parametrize("command", ["status", "experiments", "hypotheses", "conclusions", "timeline", "context"])
@pytest.mark.parametrize("position", ["before", "after"])
def test_json_flag_before_or_after_command(project, capsys, command, position):
    arguments = ["--json", command] if position == "before" else [command, "--json"]
    assert main(arguments) == 0
    response(capsys, command)


def test_compare_and_diagnose_keep_numeric_data_and_evidence(project, capsys):
    assert main(["compare", "B0", "E1", "--json"]) == 0
    data = response(capsys, "compare")["data"]
    assert data["comparison"]["overall"]["recall"]["abs"] == pytest.approx(-0.15)
    assert {item["id"] for item in data["evidence"]} >= {"B0", "E1"}
    assert main(["diagnose", "E1", "--json"]) == 0
    data = response(capsys, "diagnose")["data"]
    assert data["diagnosis"]["experiment_id"] == "E1"
    assert isinstance(data["diagnosis"]["findings"], list)


def test_init_note_ingest_report_and_sessions_are_structured(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--json"]) == 0
    assert response(capsys, "init")["data"]["root"] == str(tmp_path)
    assert main(["note", "Цель: снизить пропуски", "--json"]) == 0
    assert response(capsys, "note")["data"]["note"]["text"] == "Цель: снизить пропуски"
    run = tmp_path / "runs" / "B0"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"recall":0.9}')
    assert main(["ingest", str(run), "--json"]) == 0
    data = response(capsys, "ingest")["data"]
    assert data["experiment"]["id"] == "B0" and data["analysis"]["experiment_id"] == "B0"
    assert main(["report", "B0", "--json"]) == 0
    assert Path(response(capsys, "report")["data"]["path"]).is_file()
    for action, status in (("start", "active"), ("finish", "finished")):
        assert main(["session", action, "--json"]) == 0
        assert response(capsys, "session")["data"]["session"]["status"] == status


def test_context_has_bounded_history_baselines_and_references(project, capsys):
    project.memory.add_user_note("Goal: reduce missed small objects")
    project.memory.add_hypothesis(Hypothesis(statement="Small objects need higher resolution"))
    project.memory.add_question(ResearchQuestion(question="Why are small objects missed?", status="investigating"))
    project.memory.add_conclusion(ResearchConclusion(statement="Confidence affects small objects"))
    before = len(project.memory.timeline())
    assert main(["context", "--limit", "1", "--json"]) == 0
    data = response(capsys, "context")["data"]
    assert data["status"]["total_experiments"] == 2
    assert data["status"]["working_baseline"]["id"] == "B0"
    assert data["recent_experiments"][0]["id"] == "E1"
    assert "config" not in data["recent_experiments"][0]
    assert data["truncated"]["recent_experiments"] is True
    assert data["recent_notes"][0]["reference"]["kind"] == "note"
    assert data["open_questions"][0]["status"] == "investigating"
    assert data["active_hypotheses"] and data["recent_conclusions"]
    assert len(project.memory.timeline()) == before


def test_empty_context_has_null_baselines_without_model_request(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ProjectWorkspace.init(str(tmp_path))
    def forbidden(*args, **kwargs): raise AssertionError("Unexpected model request")
    monkeypatch.setattr("cv_scientist.meta.MetaClient", forbidden)
    assert main(["context", "--json"]) == 0
    data = response(capsys, "context")["data"]
    assert data["status"]["working_baseline"] is None
    assert data["status"]["total_experiments"] == 0 and data["recent_experiments"] == []


def test_search_returns_history_with_refs_and_preserves_no_match(project, capsys):
    project.memory.add_user_note("Camera 17 lens contamination")
    project.memory.add_hypothesis(Hypothesis(statement="Camera lens contamination caused misses", status="rejected"))
    assert main(["search", "lens contamination", "--limit", "1", "--json"]) == 0
    data = response(capsys, "search")["data"]
    assert data["results"]["notes"][0]["reference"]["kind"] == "note"
    assert data["results"]["hypotheses"][0]["status"] == "rejected"
    for query in ("", "   ", "unicorns astronomy zebras"):
        assert main(["search", query, "--json"]) == 0
        assert not any(response(capsys, "search")["data"]["results"].values())


def test_offline_ask_does_not_require_provider_or_key(project, monkeypatch, capsys):
    monkeypatch.delenv("CV_SCIENTIST_API_KEY", raising=False)
    assert main(["ask", "B0", "--agent", "offline", "--json"]) == 0
    data = response(capsys, "ask")["data"]
    assert data["provider"] == "offline" and data["answer"] and data["evidence"]


@pytest.mark.parametrize("arguments, command, code", [
    (["compare", "B0", "missing"], "compare", "experiment_not_found"),
    (["diagnose", "missing"], "diagnose", "experiment_not_found"),
    (["ingest", "missing"], "ingest", "file_not_found"),
    (["session", "finish"], "session", "invalid_data"),
    (["ask", "B0", "--agent", "offline", "--model", "anything"], "ask", "invalid_arguments"),
])
def test_structured_operational_errors(project, capsys, arguments, command, code):
    assert main(arguments + ["--json"]) == 2
    assert response(capsys, command, ok=False)["error"]["code"] == code


@pytest.mark.parametrize("arguments", [
    ["--json"], ["compare", "B0", "--json"], ["unknown", "--json"],
    ["context", "--limit", "0", "--json"], ["context", "--limit", "101", "--json"],
    ["search", "B0", "--limit", "bad", "--json"], ["context", "--unknown", "--json"],
])
def test_parser_errors_are_json_without_usage_text(project, capsys, arguments):
    assert main(arguments) == 2
    value = json.loads(capsys.readouterr().out)
    assert value["ok"] is False and value["error"]["code"] == "invalid_arguments"


def test_missing_workspace_is_structured_and_creates_no_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["context", "--json"]) == 2
    assert response(capsys, "context", ok=False)["error"]["code"] == "workspace_not_found"
    assert not (tmp_path / ".cv-scientist").exists()


def test_json_redacts_secrets_in_nested_config_and_text(project, monkeypatch, capsys):
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "never-emit-this-value")
    exp = project.memory.get_experiment("E1")
    exp.config.update(api_key="nested-private-value", nested={"password": "password-value"})
    exp.name = "never-emit-this-value"
    project.memory.add_experiment(exp)
    assert main(["experiments", "--json"]) == 0
    output = capsys.readouterr().out
    assert all(secret not in output for secret in ("never-emit-this-value", "nested-private-value", "password-value"))
    assert "[redacted]" in output
    json.loads(output)


def test_nonfinite_stored_data_is_an_explicit_json_error(project, capsys):
    exp = project.memory.get_experiment("E1")
    exp.metrics["recall"] = float("nan")
    project.memory.add_experiment(exp)
    assert main(["experiments", "--json"]) == 2
    assert response(capsys, "experiments", ok=False)["error"]["code"] == "invalid_data"


def test_plain_text_mode_is_preserved(project, capsys):
    assert main(["status"]) == 0
    assert capsys.readouterr().out.startswith("Project:")
    assert main(["compare", "B0", "E1"]) == 0
    assert capsys.readouterr().out.startswith("Overall:")
    assert main(["compare", "B0", "missing"]) == 2
    assert "unknown experiment" in capsys.readouterr().err


def test_search_truncation_is_per_kind_and_uses_search_ranking(project, capsys):
    for index in range(3): project.memory.add_user_note(f"Camera contamination finding {index}")
    assert main(["search", "contamination", "--limit", "2", "--json"]) == 0
    data = response(capsys, "search")["data"]
    assert len(data["results"]["notes"]) == 2 and data["truncated"]["notes"] is True
    assert main(["search", "E1", "--json"]) == 0
    assert response(capsys, "search")["data"]["results"]["experiments"][0]["id"] == "E1"


@pytest.mark.parametrize("limit", [None, True, 0, -1, 101])
def test_history_helpers_validate_limits(project, limit):
    from cv_scientist.agent_output import research_context, search_history
    with pytest.raises(ValueError): research_context(project, limit)
    with pytest.raises(ValueError): search_history(project, "B0", limit)


def test_search_rejects_none_and_oversized_input(project, capsys):
    from cv_scientist.agent_output import search_history
    with pytest.raises(ValueError): search_history(project, None)
    assert main(["search", "x" * 2049, "--json"]) == 2
    assert response(capsys, "search", ok=False)["error"]["code"] == "invalid_data"


def test_invalid_artifacts_produce_json_and_ingest_rolls_back(project, capsys):
    run = Path(project.root) / "runs" / "broken"
    run.mkdir()
    (run / "metrics.json").write_text('{"metrics":')
    assert main(["ingest", str(run), "--json"]) == 2
    assert response(capsys, "ingest", ok=False)["error"]["code"] == "invalid_data"
    assert project.memory.get_experiment("broken") is None
    (Path(project.root) / "runs" / "E1" / "predictions.json").write_text("[]broken")
    assert main(["diagnose", "E1", "--json"]) == 2
    assert response(capsys, "diagnose", ok=False)["error"]["code"] == "invalid_data"


def test_meta_ask_success_and_provider_error_are_structured(project, monkeypatch, capsys):
    from test_meta_agent import server, api_response, final
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "private-http-key")
    with server([(200, api_response(final()), {}), (401, b"private-provider-body", {})]) as (url, requests):
        monkeypatch.setenv("META_API_BASE_URL", url)
        assert main(["ask", "Why?", "--agent", "meta", "--json"]) == 0
        data = response(capsys, "ask")["data"]
        assert data["provider"] == "meta" and data["report"]["usage"]["total_tokens"] == 30
        assert main(["ask", "Why?", "--agent", "meta", "--json"]) == 2
        value = response(capsys, "ask", ok=False)
        assert value["error"]["code"] == "provider_error"
        assert "private-provider-body" not in json.dumps(value)
    assert len(requests) == 2


def test_import_empty_source_has_structured_report_without_a_model(project, capsys):
    source = Path(project.root) / "empty-source"
    source.mkdir()
    assert main(["import", str(source), "--json"]) == 0
    data = response(capsys, "import")["data"]
    assert data["runs"] == [] and data["usage"]["requests"] == 0
    assert Path(data["path"]).is_file()


def test_research_reports_unexecuted_proposals_truthfully(project, capsys):
    assert main(["research", "--goal", "Reduce missed objects", "--json"]) == 0
    data = response(capsys, "research")["data"]
    assert data["provider"] == "offline" and data["result"]["iterations"] == 0
    assert data["result"]["experiments"] == []
    assert main(["research", "--agent", "meta", "--goal", "Reduce FP", "--json"]) == 2
    assert response(capsys, "research", ok=False)["error"]["code"] == "invalid_arguments"


def test_regressions_and_watch_final_summary(project, monkeypatch, capsys):
    assert main(["regressions", "E1", "--json"]) == 0
    assert response(capsys, "regressions")["data"]["regressions"]
    def stop(self):
        self.events.append({"type": "scan_failed", "error": "missing directory"})
        raise KeyboardInterrupt()
    monkeypatch.setattr("cv_scientist.sessions.WatchService.watch", stop)
    assert main(["watch", "runs", "--json"]) == 0
    data = response(capsys, "watch")["data"]
    assert data["interrupted"] is True and data["events"][0]["type"] == "scan_failed"


def test_config_json_never_returns_credentials(project, monkeypatch, capsys):
    monkeypatch.setattr("getpass.getpass", lambda prompt: "never-return-this-key")
    assert main(["config", "--json"]) == 0
    value = response(capsys, "config")
    assert "never-return-this-key" not in json.dumps(value)
    assert Path(value["data"]["path"]).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("error, code, exit_code", [(RuntimeError("Unexpected failure"), "internal_error", 1),
                                                    (KeyboardInterrupt(), "interrupted", 130)])
def test_failures_and_interruptions_have_structured_errors(project, monkeypatch, capsys, error, code, exit_code):
    def fail(ws): raise error
    monkeypatch.setattr("cv_scientist.sessions.project_status", fail)
    assert main(["status", "--json"]) == exit_code
    assert response(capsys, "status", ok=False)["error"]["code"] == code
    rows = [json.loads(line) for line in (Path(project.dir) / "logs" / "operations.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] in ("failed", "interrupted")


def test_json_help_and_literal_flag_as_note_text(project, capsys):
    with pytest.raises(SystemExit) as exit:
        main(["context", "--json", "--help"])
    assert exit.value.code == 0
    assert "--limit" in response(capsys, "help")["data"]["help"]
    assert main(["note", "--", "--json"]) == 0
    assert capsys.readouterr().out.startswith("Added note ")
