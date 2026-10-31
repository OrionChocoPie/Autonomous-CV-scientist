from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from cv_scientist.cli import main
from cv_scientist.import_source import ImportSource, OUTPUT_DIR
from cv_scientist.importing import ImportAgent, ImportTools
from cv_scientist.ingestor import LocalExperimentSource, RunIngestor
from cv_scientist.meta import MetaAPIError
from cv_scientist.models import Experiment
from cv_scientist.workspace import ProjectWorkspace


class Client:
    def __init__(self, replies):
        self.config = SimpleNamespace(model="test-model")
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools, response_format):
        self.requests.append(deepcopy(messages))
        reply = next(self.replies)
        if callable(reply): reply = reply()
        return reply, {"total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10}


def calls(*entries):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for identity, name, arguments in entries]}


def finish():
    return {"role": "assistant", "content": json.dumps({"summary": "Converted measured results", "limitations": ["Validation identity unknown"]})}


def reference(column, name, scope="overall", group="", file="run/metrics.csv"):
    target = "/metrics/" + name if scope == "overall" else ("/per_class/" if scope == "class" else "/slices/") + group + "/" + name
    return {"target": target, "file": file, "pointer": "/0/" + column}


def proposal(**changes):
    return {"run_dir": "run", "config_file": "run/config.yaml", "checkpoint": "run/model.pt", "status": "unknown",
            "metrics": [reference("mAP_50_95", "map50_95"), reference("FPs", "fp"), reference("recall", "recall"),
                        reference("FPs_empty_drinking_glass", "fp", "class", "empty_drinking_glass", "run/extended_metrics.csv")],
            "notes": ["Validation dataset identity is not recorded"], **changes}


def conversation():
    return [calls(("list", "list_directory", {"path": "run"})), calls(
        ("config", "read_file", {"path": "run/config.yaml", "pointer": "", "start_line": 1}),
        ("metrics", "read_file", {"path": "run/metrics.csv", "pointer": "", "start_line": 1}),
        ("classes", "read_file", {"path": "run/extended_metrics.csv", "pointer": "", "start_line": 1})),
        calls(("prepare", "prepare_run", proposal())), finish()]


@pytest.fixture
def legacy(tmp_path):
    root = tmp_path / "legacy source"
    run = root / "run"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("exp_name: test\ntrain:\n  batch_size: 128\n  label_to_name:\n    0: food\n  base_lr: ${train.lrs.s}\n")
    (run / "metrics.csv").write_text(",mAP_50_95,FPs,recall\nval,0.8291,2539,0.9689\n")
    (run / "extended_metrics.csv").write_text(",FPs_empty_drinking_glass\nval,267\n")
    (run / "model.pt").write_bytes(b"checkpoint-not-downloaded")
    return root


def loaded_tools(root):
    tools = ImportTools(ImportSource(str(root)))
    tools.list_directory(".")
    tools.list_directory("run")
    for file in ("config.yaml", "metrics.csv", "extended_metrics.csv"):
        tools.read_file("run/" + file)
    return tools


def test_agent_creates_copy_without_profile_source_changes_or_memory(legacy):
    before = {p: p.read_bytes() for p in (legacy / "run").iterdir()}
    client = Client(conversation())
    report = ImportAgent(ImportSource(str(legacy)), client).run()
    assert report["runs"][0]["status"] == "created"
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    exp = LocalExperimentSource().load_run(str(destination))
    assert exp.metrics == {"map50_95": 0.8291, "fp": 2539, "recall": 0.9689}
    assert exp.per_class == {"empty_drinking_glass": {"fp": 267}}
    assert exp.config["train"] == {"batch_size": 128, "label_to_name": {"0": "food"}, "base_lr": "${train.lrs.s}"}
    assert exp.parent_id is None and exp.status == "unknown"
    assert exp.checkpoint_path == str(legacy / "run" / "model.pt")
    assert exp.run_dir == str(destination)
    manifest = json.loads((destination / "source.json").read_text())
    assert manifest["source_created_at"] is None
    assert (destination / manifest["raw_files"]["run/config.yaml"]).read_bytes() == before[legacy / "run" / "config.yaml"]
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not (legacy / ".cv-scientist").exists()
    assert not list((legacy / OUTPUT_DIR).rglob("*profile*"))
    assert not list((legacy / OUTPUT_DIR).rglob("*.pt"))
    assert report["usage"]["requests"] == 4
    ws = ProjectWorkspace.init(str(legacy.parent / "workspace"))
    ws.memory.add_experiment(Experiment(id="unrelated"))
    imported = RunIngestor(ws.memory).ingest(str(destination))
    assert imported.parent_id is None


def test_identical_import_is_noop_and_changed_inputs_create_revision(legacy):
    source = ImportSource(str(legacy))
    first = ImportAgent(source, Client(conversation())).run()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    second = ImportAgent(source, Client(conversation())).run()
    assert second["runs"][0]["status"] == "unchanged"
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    assert len(list((legacy / OUTPUT_DIR).glob("I*/*"))) == 1
    path = legacy / "run" / "metrics.csv"
    path.write_text(path.read_text().replace("2539", "2500"))
    third = ImportAgent(source, Client(conversation())).run()
    assert third["runs"][0]["status"] == "created"
    assert third["runs"][0]["experiment_id"] == first["runs"][0]["experiment_id"]
    assert len(list((legacy / OUTPUT_DIR).glob("I*/*"))) == 2
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


@pytest.mark.parametrize("change", ["pointer", "target", "status", "notes"])
def test_changed_import_recipe_creates_a_revision(legacy, change):
    recipe = proposal(metrics=[reference("recall", "recall")])
    first_tools = loaded_tools(legacy)
    first_tools.prepare_run(**recipe)
    first = first_tools.publish()[0]
    before = {p.name: p.read_bytes() for p in Path(first["copy"]).iterdir()}
    updated = deepcopy(recipe)
    if change == "pointer": updated["metrics"][0]["pointer"] = "/0/mAP_50_95"
    elif change == "target": updated["metrics"][0]["target"] = "/metrics/precision"
    elif change == "status": updated["status"] = "completed"
    else: updated["notes"] = ["Revised interpretation"]
    second_tools = loaded_tools(legacy)
    second_tools.prepare_run(**updated)
    second = second_tools.publish()[0]
    assert second["status"] == "created" and second["copy"] != first["copy"]
    assert second["experiment_id"] == first["experiment_id"]
    exp = LocalExperimentSource().load_run(second["copy"])
    if change == "pointer": assert exp.metrics["recall"] == 0.8291
    elif change == "target": assert exp.metrics == {"precision": 0.9689}
    elif change == "status": assert exp.status == "completed"
    else:
        manifest = json.loads((Path(second["copy"]) / "source.json").read_text())
        assert manifest["notes"] == updated["notes"]
    assert {p.name: p.read_bytes() for p in Path(first["copy"]).iterdir()} == before
    repeated = loaded_tools(legacy)
    repeated.prepare_run(**updated)
    assert repeated.publish()[0]["status"] == "unchanged"


def test_import_metric_order_does_not_change_revision(legacy):
    recipe = proposal()
    first_tools = loaded_tools(legacy)
    first_tools.prepare_run(**recipe)
    first = first_tools.publish()[0]
    reordered = {**recipe, "metrics": list(reversed(recipe["metrics"]))}
    second_tools = loaded_tools(legacy)
    second_tools.prepare_run(**reordered)
    second = second_tools.publish()[0]
    assert second["status"] == "unchanged" and second["copy"] == first["copy"]


def test_skipped_and_empty_sources_are_explicit(legacy, tmp_path):
    replies = [calls(("skip", "skip_run", {"run_dir": "run", "reason": "No final evaluation available"})), finish()]
    report = ImportAgent(ImportSource(str(legacy)), Client(replies)).run()
    assert report["runs"] == [] and report["skipped"] == {"run": "No final evaluation available"}
    empty = tmp_path / "empty"
    empty.mkdir()
    client = Client([])
    report = ImportAgent(ImportSource(str(empty)), client).run()
    assert not client.requests and report["runs"] == []


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "1.5", "hello", ""])
def test_bad_counts_never_create_copy(legacy, value):
    path = legacy / "run" / "metrics.csv"
    path.write_text(",FPs\nval," + value + "\n")
    tools = loaded_tools(legacy)
    with pytest.raises(ValueError): tools.prepare_run(**proposal(metrics=[reference("FPs", "fp")]))
    assert tools.prepared == {} and not (legacy / OUTPUT_DIR).exists()


@pytest.mark.parametrize("value", [False, None, [], {}, float("nan"), -0.01, 1.01])
def test_invalid_json_quality_metrics(legacy, value):
    (legacy / "run" / "values.json").write_text(json.dumps({"recall": value}))
    tools = loaded_tools(legacy)
    ref = {"target": "/metrics/recall", "file": "run/values.json", "pointer": "/recall"}
    try: tools.read_file("run/values.json")
    except ValueError:
        assert isinstance(value, float) and value != value
        return
    with pytest.raises(ValueError): tools.prepare_run(**proposal(metrics=[ref]))
    assert not tools.prepared


def test_source_pointer_evidence_required_and_duplicate_destinations_rejected(legacy):
    tools = loaded_tools(legacy)
    bad = reference("invented", "fp")
    with pytest.raises(ValueError, match="does not exist"): tools.prepare_run(**proposal(metrics=[bad]))
    bad["file"] = "run/unread.json"
    with pytest.raises(ValueError, match="Inspect"): tools.prepare_run(**proposal(metrics=[bad]))
    with pytest.raises(ValueError, match="Duplicate"): tools.prepare_run(**proposal(metrics=[reference("FPs", "fp")] * 2))
    with pytest.raises(ValueError, match="Discover"): tools.prepare_run(**proposal(checkpoint="run/missing.pt"))
    assert not tools.prepared


def test_source_changes_before_publication_abort_all_copies(legacy):
    tools = loaded_tools(legacy)
    tools.prepare_run(**proposal())
    (legacy / "run" / "metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed during import"): tools.publish()
    assert not (legacy / OUTPUT_DIR).exists()


def test_modified_copy_is_never_overwritten_or_ingested(legacy):
    ImportAgent(ImportSource(str(legacy)), Client(conversation())).run()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    (destination / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"): LocalExperimentSource().load_run(str(destination))
    with pytest.raises(ValueError, match="modified"): ImportAgent(ImportSource(str(legacy)), Client(conversation())).run()
    assert (destination / "metrics.json").read_text() == "{}"


@pytest.mark.parametrize("missing", ["source.json", "experiment.json", "metrics.json", "config.json"])
def test_incomplete_copies_cannot_fall_back_to_legacy_loader(legacy, missing):
    ImportAgent(ImportSource(str(legacy)), Client(conversation())).run()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    (destination / missing).unlink()
    with pytest.raises(FileNotFoundError): LocalExperimentSource().load_run(str(destination))


def test_unrelated_legacy_metadata_files_keep_existing_ingest_behavior(legacy):
    folder = legacy / "run"
    (folder / "source.json").write_text('{"tracker":"other"}')
    (folder / "experiment.json").write_text('{"description":"unrelated legacy metadata"}')
    (folder / "metrics.json").write_text('{"metrics":{"recall":0.9}}')
    exp = LocalExperimentSource().load_run(str(folder))
    assert exp.metrics == {"recall": 0.9} and exp.source == "local"


def test_incomplete_hash_manifest_is_not_accepted_as_unchanged(legacy):
    ImportAgent(ImportSource(str(legacy)), Client(conversation())).run()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    path = destination / "source.json"
    manifest = json.loads(path.read_text())
    manifest["output_sha256"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"): ImportAgent(ImportSource(str(legacy)), Client(conversation())).run()


def test_missing_metrics_explicit_skip_and_discovered_run_coverage(tmp_path):
    root = tmp_path / "project"
    (root / "models" / "unfinished").mkdir(parents=True)
    client = Client([
        calls(("models", "list_directory", {"path": "models"})),
        calls(("skip_root", "skip_run", {"run_dir": ".", "reason": "Project container"})),
        finish(),
        calls(("skip", "skip_run", {"run_dir": "models/unfinished", "reason": "Metrics absent; completion unknown"})),
        finish(),
    ])
    report = ImportAgent(ImportSource(str(root)), client).run()
    assert "models/unfinished" in report["skipped"]
    assert "remaining discovered runs" in client.requests[3][-1]["content"]
    assert report["runs"] == []


def test_structured_json_metrics_and_slices_are_copied_exactly(legacy):
    (legacy / "run" / "values.json").write_text('{"eval":{"precision":0.95,"hard":{"fn":0}},"escaped/key":{"~count":3}}')
    tools = loaded_tools(legacy)
    value = tools.read_file("run/values.json", "/eval/hard")
    assert value["data"] == {"fn": 0}
    refs = [{"target": "/metrics/precision", "file": "run/values.json", "pointer": "/eval/precision"},
            {"target": "/slices/hard/fn", "file": "run/values.json", "pointer": "/eval/hard/fn"},
            {"target": "/metrics/fp", "file": "run/values.json", "pointer": "/escaped~1key/~0count"}]
    tools.prepare_run(**proposal(metrics=refs))
    tools.publish()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    exp = LocalExperimentSource().load_run(str(destination))
    assert exp.metrics == {"precision": 0.95, "fp": 3} and exp.slices == {"hard": {"fn": 0}}


@pytest.mark.parametrize("content", [",recall,recall\nval,0.1,0.2\n", ",recall\nval,0.1,0.2\n", ""])
def test_ambiguous_and_empty_csv_rejected(legacy, content):
    (legacy / "run" / "metrics.csv").write_text(content)
    with pytest.raises(ValueError): loaded_tools(legacy)


@pytest.mark.parametrize("reply", [
    {"role": "assistant", "content": "not json"},
    {"role": "assistant", "content": "{}"},
    calls(("duplicate", "list_directory", {"path": "run"}), ("duplicate", "list_directory", {"path": "run"})),
    {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": "{"}}]},
])
def test_invalid_agent_responses_never_publish(legacy, reply):
    with pytest.raises(MetaAPIError): ImportAgent(ImportSource(str(legacy)), Client([reply])).run()
    assert not (legacy / OUTPUT_DIR).exists()


@pytest.mark.parametrize("path", ["../outside", "/absolute", "run/../../outside", ".env", "run/model.pt", "run/secret.conf", ".cv-scientist-import/report.json"])
def test_read_tools_reject_unsafe_paths(legacy, path):
    with pytest.raises(ValueError): ImportTools(ImportSource(str(legacy))).read_file(path)


def test_symlinks_and_missing_sources(legacy, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":1}')
    (legacy / "run" / "linked.json").symlink_to(outside)
    tools = loaded_tools(legacy)
    with pytest.raises(ValueError, match="symlink"): tools.read_file("run/linked.json")
    with pytest.raises(FileNotFoundError): ImportSource(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError): tools.read_file("run/missing.csv")
    output = legacy / OUTPUT_DIR
    output.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"): tools.source.publish(OUTPUT_DIR + "/escape", {"report.json": b"{}"})


@pytest.mark.parametrize("source", [None, "", "ssh://host/", "ssh://user:pass@host/path", "ssh://host/path?x=1", "ssh://-bad/path"])
def test_invalid_source_locations(source, monkeypatch):
    def no_network(*args, **kwargs): raise OSError("No network")
    monkeypatch.setattr(ImportSource, "_ssh", no_network)
    with pytest.raises((ValueError, OSError)): ImportSource(source)


def test_step_tool_and_token_limits_never_publish(legacy):
    call = calls(("list", "list_directory", {"path": "run"}))
    for limits, replies, error in [({"max_steps": 1}, [call], "step"),
                                  ({"max_tool_calls": 1}, [calls(("a", "list_directory", {"path": "run"}), ("b", "list_directory", {"path": "run"}))], "tool-call"),
                                  ({"max_total_tokens": 1}, [call], "token")]:
        with pytest.raises(MetaAPIError, match=error): ImportAgent(ImportSource(str(legacy)), Client(replies), **limits).run()
    assert not (legacy / OUTPUT_DIR).exists()


def test_api_error_after_preparation_never_publishes(legacy):
    def fail(): raise MetaAPIError("Provider unavailable")
    replies = conversation()[:-1] + [fail]
    with pytest.raises(MetaAPIError): ImportAgent(ImportSource(str(legacy)), Client(replies)).run()
    assert not (legacy / OUTPUT_DIR).exists()


def test_tool_errors_are_explicit_and_recoverable(legacy):
    replies = [calls(("bad", "read_file", {"path": "missing.csv", "pointer": "", "start_line": 1}))] + conversation()
    client = Client(replies)
    report = ImportAgent(ImportSource(str(legacy)), client).run()
    assert report["tools"][0]["error"]
    assert report["runs"][0]["status"] == "created"


def test_cli_import_creates_local_profiles_without_initializing_memory(legacy, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "test-key")
    client = Client(conversation())
    monkeypatch.setattr("cv_scientist.meta.MetaClient", lambda config: client)
    assert main(["import", str(legacy), "--agent", "meta"]) == 0
    assert "created:" in capsys.readouterr().out
    assert (tmp_path / ".cv-scientist" / "import" / "profiles").is_dir()
    assert not (tmp_path / ".cv-scientist" / "memory").exists()
    assert not (legacy / OUTPUT_DIR).exists()
    monkeypatch.delenv("CV_SCIENTIST_API_KEY")
    assert main(["import", str(legacy)]) == 0
    assert "unchanged:" in capsys.readouterr().out
    assert len(client.requests) == 4


def test_ssh_transport_quotes_paths_and_does_not_forward_model_credentials(monkeypatch):
    commands = []
    def run(args, **kwargs):
        commands.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"/remote/source with spaces\n", stderr=b"")
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "never-forward")
    monkeypatch.setattr(subprocess, "run", run)
    source = ImportSource("ssh://user@rss:2222/remote/source%20with%20spaces")
    assert source.host == "user@rss" and source.port == 2222
    args, kwargs = commands[0]
    assert "-p" in args and "2222" in args and "StrictHostKeyChecking=yes" in args
    assert "CV_SCIENTIST_API_KEY" not in kwargs["env"] and kwargs["timeout"] == 30
    assert args[-2] == "user@rss" and args[-1].startswith("sh -c ")
    assert source.uri == "ssh://user@rss:2222/remote/source with spaces"


def test_empty_optional_paths_and_text_pointer(legacy):
    (legacy / "README.md").write_text("Legacy results\n")
    tools = loaded_tools(legacy)
    assert tools.read_file("README.md", "")["text"] == "Legacy results"
    tools.prepare_run(**proposal(config_file="", checkpoint=""))
    tools.publish()
    destination = next((legacy / OUTPUT_DIR).glob("I*/*"))
    exp = LocalExperimentSource().load_run(str(destination))
    assert exp.config == {} and exp.checkpoint_path is None


def test_unsafe_yaml_is_data_error_not_code_execution(legacy):
    path = legacy / "run" / "config.yaml"
    path.write_text("!!python/object/apply:builtins.print ['must not execute']")
    with pytest.raises(ValueError, match="structured source"): loaded_tools(legacy)


def test_ssh_failures_are_explicit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=255, stdout=b"", stderr=b"secret detail"))
    with pytest.raises(OSError, match="exit 255") as error: ImportSource("rss:/root/project")
    assert "secret" not in str(error.value)
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired("ssh", 30)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(OSError, match="timed out"): ImportSource("rss:/root/project")
