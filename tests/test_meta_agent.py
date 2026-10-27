import base64
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from cv_scientist.agent_tools import AnalysisTools
from cv_scientist.cli import main
from cv_scientist.investigation import InvestigationAgent
from cv_scientist.meta import MetaAPIError, MetaClient, MetaConfig, read_env
from cv_scientist.models import Experiment
from cv_scientist.workspace import ProjectWorkspace


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII=")


@pytest.fixture
def workspace(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    ws.config["dataset_root"] = "data"
    with open(ws.config_path, "a") as file: file.write("dataset_root: data\n")
    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    (images / "person.png").write_bytes(PNG)
    for name, recall in (("B0", 1.0), ("E1", 0.5)):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        row = {"image": "images/person.png", "truth": [[0, 0, 1, 1]],
               "predictions": [], "tp": 0, "fp": 0, "fn": 1}
        (run / "predictions.json").write_text(json.dumps([row]))
        ws.memory.add_experiment(Experiment(id=name, run_dir=str(run), metrics={"recall": recall, "map50_95": recall},
                                            config={"imgsz": 640 if name == "B0" else 160}, slices={"small": {"recall": recall, "fn": 0 if name == "B0" else 1}}))
    return ws


def tool_call(identity, name, **arguments):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def assistant(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def final(evidence=None):
    return {"role": "assistant", "content": json.dumps({
        "answer": "Recall fell on small objects; the images need review.",
        "evidence": evidence if evidence is not None else [{"kind": "experiment", "id": "E1"}],
        "hypotheses": ["Reduced input resolution may explain missed people."],
        "next_steps": ["Evaluate the same weights at imgsz=320."],
    })}


def api_response(message):
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


@contextmanager
def server(responses):
    received = []
    replies = iter(responses)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
            status, payload, headers = next(replies, (500, b"unexpected request", {}))
            self.send_response(status)
            for key, value in headers.items(): self.send_header(key, value)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode() if isinstance(payload, dict) else payload)

        def do_GET(self):
            received.append({"path": self.path})
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args):
            return

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", received
    finally:
        http.shutdown()
        worker.join(timeout=2)
        http.server_close()


class ScriptedClient:
    def __init__(self, messages):
        self.config = SimpleNamespace(model="test-model")
        self.messages = iter(messages)
        self.requests = []

    def complete(self, messages, tools, response_format):
        self.requests.append(deepcopy(messages))
        return next(self.messages), {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}


def test_real_http_multistep_tools_images_and_report(workspace):
    first = assistant(tool_call("c1", "compare_experiments", baseline_id="B0", current_id="E1"),
                      tool_call("c2", "get_error_examples", experiment_id="E1", kind="fn", limit=1))
    second = assistant(tool_call("c3", "view_error", experiment_id="E1", image="images/person.png"))
    finish = final([{"kind": "experiment", "id": "B0"}, {"kind": "experiment", "id": "E1"},
                    {"kind": "artifact", "id": "E1:predictions.json#0"}])
    before = [e.to_dict() for e in workspace.memory.list_experiments()]
    with server([(200, api_response(m), {}) for m in (first, second, finish)]) as (url, requests):
        report = InvestigationAgent(workspace, MetaClient(MetaConfig("test-secret", base_url=url))).ask("Compare B0 and E1")
    assert len(requests) == 3
    assert all(r["path"] == "/v1/chat/completions" and r["authorization"] == "Bearer test-secret" for r in requests)
    assert requests[0]["body"]["response_format"]["json_schema"]["strict"] is True
    def citation_ids(request):
        return request["body"]["response_format"]["json_schema"]["schema"]["properties"]["evidence"]["items"]["properties"]["id"]["enum"]
    assert set(citation_ids(requests[0])) == {"B0", "E1"}
    assert set(citation_ids(requests[1])) == {"B0", "E1", "E1:predictions.json#0"}
    assert first in requests[1]["body"]["messages"]
    results = [m for m in requests[1]["body"]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]
    assert json.loads(results[0]["content"])["overall"]["recall"]["abs"] == -0.5
    attachments = requests[2]["body"]["messages"][-1]["content"]
    assert attachments[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(attachments[1]["image_url"]["url"].split(",", 1)[1]) == PNG
    saved = Path(report.path).read_text()
    assert "test-secret" not in saved and "base64" not in saved
    assert json.loads(saved)["usage"]["total_tokens"] == 90
    assert [c["name"] for c in report.tool_calls] == ["compare_experiments", "get_error_examples", "view_error"]
    assert len(report.evidence) == 3
    assert [e.to_dict() for e in workspace.memory.list_experiments()] == before


def test_tool_free_structured_request_and_timeout_override():
    with server([(200, api_response(final()), {})]) as (url, requests):
        client = MetaClient(MetaConfig("test-secret", base_url=url))
        message, usage = client.complete([], [], {}, timeout=1)
    assert "tools" not in requests[0]["body"]
    assert message["role"] == "assistant" and usage["total_tokens"] == 30
    with pytest.raises(ValueError, match="timeout"):
        client.complete([], [], {}, timeout=0)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 302])
def test_http_errors_never_leak_body_or_follow_redirect(status):
    with server([(status, b"secret-in-error-body", {"Location": "/other"})]) as (url, requests):
        with pytest.raises(MetaAPIError, match=f"HTTP {status}") as error:
            MetaClient(MetaConfig("test-secret", base_url=url)).complete([], [], {})
    assert "secret" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [
    b"not json", {"choices": []}, {"choices": [{"message": None, "finish_reason": "stop"}]},
    {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}]},
    {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]},
    {**api_response(final()), "usage": {"total_tokens": "unknown"}},
])
def test_invalid_provider_responses(payload):
    with server([(200, payload, {})]) as (url, requests):
        with pytest.raises(MetaAPIError):
            MetaClient(MetaConfig("test-secret", base_url=url)).complete([], [], {})


@pytest.mark.parametrize("failure", [URLError("secret connection detail"), TimeoutError("secret timeout")])
def test_connection_errors_are_explicit_and_sanitized(failure):
    client = MetaClient(MetaConfig("test-secret"))
    def fail(*args, **kwargs): raise failure
    client._opener = SimpleNamespace(open=fail)
    with pytest.raises(MetaAPIError, match="connection failed") as error:
        client.complete([], [], {})
    assert "secret" not in str(error.value)


def test_env_precedence_and_no_environment_mutation(workspace, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    (cwd / ".env").write_text("MODEL_API_KEY=cwd-key\nMETA_MODEL=muse-spark-1.3\n")
    target = Path(workspace.root) / ".env"
    target.write_text("export MODEL_API_KEY='target-key' # comment\n")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL", raising=False)
    config = MetaConfig.from_workspace(workspace)
    assert config.api_key == "target-key" and config.model == "muse-spark-1.3"
    assert "target-key" not in repr(config)
    monkeypatch.setenv("MODEL_API_KEY", "process-key")
    assert MetaConfig.from_workspace(workspace, model="muse-wonder-1.0").api_key == "process-key"
    assert target.read_text() == "export MODEL_API_KEY='target-key' # comment\n"


@pytest.mark.parametrize("entry", ["not-an-assignment", "KEY='unterminated", "KEY=unquoted spaces"])
def test_invalid_env_does_not_echo_contents(tmp_path, entry):
    path = tmp_path / ".env"
    path.write_text(entry)
    with pytest.raises(ValueError) as error: read_env(path)
    assert entry not in str(error.value)
    assert read_env(tmp_path / "missing") == {}


@pytest.mark.parametrize("key", ["", None, "   "])
def test_missing_key_is_an_error(key):
    with pytest.raises(ValueError, match="MODEL_API_KEY"): MetaConfig(key)


@pytest.mark.parametrize("kwargs", [{"base_url": "http://remote.example/v1"}, {"base_url": "https://user:pass@host/v1"},
                                     {"max_tokens": 0}, {"timeout": float("nan")}, {"model": ""}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): MetaConfig("test-secret", **kwargs)


def test_error_tools_and_image_path_limits(workspace):
    tools = AnalysisTools(workspace)
    examples = tools.execute("get_error_examples", {"experiment_id": "E1", "kind": "fn", "limit": 1}).data
    assert examples["examples"][0]["evidence"]["id"] == "E1:predictions.json#0"
    viewed = tools.execute("view_error", {"experiment_id": "E1", "image": "images/person.png"})
    assert viewed.data["image_shown"] and len(viewed.images) == 1
    with pytest.raises(ValueError, match="not referenced"):
        tools.view_error("E1", "../.env")
    with pytest.raises(ValueError, match="Unknown analysis tool"):
        tools.execute("run_shell", {"command": "anything"})
    with pytest.raises(ValueError): tools.execute("list_experiments", {"limit": True})
    with pytest.raises(ValueError): tools.execute("get_experiment", {"experiment_id": None})
    with pytest.raises(ValueError): tools.execute("get_experiment", {"experiment_id": "missing"})
    with pytest.raises(ValueError): tools.execute("list_experiments", None)


def test_all_read_tools(workspace):
    tools = AnalysisTools(workspace)
    assert len(tools.list_experiments(20)["experiments"]) == 2
    assert tools.get_experiment("E1")["config"]["imgsz"] == 160
    assert tools.diagnose_experiment("E1")["findings"]
    assert tools.search_memory("small")["experiments"]


def test_internal_tool_failure_is_not_mislabeled_as_bad_arguments(workspace):
    tools = AnalysisTools(workspace)
    def broken(limit): raise TypeError("internal diagnostic failure")
    tools.list_experiments = broken
    with pytest.raises(ValueError, match="Invalid arguments"):
        tools.execute("list_experiments", {"other": 1})
    with pytest.raises(TypeError, match="internal diagnostic failure"):
        tools.execute("list_experiments", {"limit": 1})


@pytest.mark.parametrize("records", [None, {}, [None], [{"image": "../secret", "truth": [], "predictions": [], "fp": 0, "fn": 0}],
                                     [{"image": "a.png", "truth": [], "predictions": [], "fp": -1, "fn": 0}]])
def test_invalid_prediction_files(workspace, records):
    path = Path(workspace.root) / "runs" / "E1" / "predictions.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError): AnalysisTools(workspace).get_error_examples("E1", "fn", 1)


def test_missing_empty_and_symlinked_artifacts(workspace):
    tools = AnalysisTools(workspace)
    path = Path(workspace.root) / "runs" / "E1" / "predictions.json"
    path.write_text("[]")
    assert tools.get_error_examples("E1", "any", 1)["examples"] == []
    path.unlink()
    with pytest.raises(FileNotFoundError): tools.get_error_examples("E1", "fn", 1)
    path.symlink_to(Path(workspace.root) / "outside.json")
    with pytest.raises(ValueError, match="inside"): tools.get_error_examples("E1", "fn", 1)


def test_missing_and_symlinked_images(workspace, tmp_path):
    tools = AnalysisTools(workspace)
    image = Path(workspace.root) / "data" / "images" / "person.png"
    image.unlink()
    with pytest.raises(ValueError, match="not found"): tools.view_error("E1", "images/person.png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="not found"): tools.view_error("E1", "images/person.png")


@pytest.mark.parametrize("limits", [{"max_steps": 0}, {"max_tool_calls": -1}, {"max_images": True}, {"max_total_tokens": 0}])
def test_invalid_agent_limits(workspace, limits):
    with pytest.raises(ValueError): InvestigationAgent(workspace, ScriptedClient([]), **limits)


def test_step_and_tool_budgets(workspace):
    call = assistant(tool_call("c1", "get_experiment", experiment_id="E1"))
    with pytest.raises(MetaAPIError, match="step budget"):
        InvestigationAgent(workspace, ScriptedClient([call]), max_steps=1).ask("Investigate")
    calls = assistant(tool_call("c1", "get_experiment", experiment_id="E1"), tool_call("c2", "get_experiment", experiment_id="B0"))
    with pytest.raises(MetaAPIError, match="tool-call budget"):
        InvestigationAgent(workspace, ScriptedClient([calls]), max_tool_calls=1).ask("Investigate")
    with pytest.raises(MetaAPIError, match="token budget"):
        InvestigationAgent(workspace, ScriptedClient([final()]), max_total_tokens=1).ask("Investigate")
    assert not list((Path(workspace.dir) / "reports").glob("*.json"))


def test_image_budget_and_tool_errors_are_returned_to_model(workspace):
    client = ScriptedClient([
        assistant(tool_call("c1", "view_error", experiment_id="E1", image="images/person.png"),
                  tool_call("c2", "view_error", experiment_id="E1", image="images/person.png"),
                  tool_call("c3", "unknown_tool")), final(),
    ])
    report = InvestigationAgent(workspace, client, max_images=1).ask("Review errors")
    assert report.tool_calls[0]["result"]["image_shown"]
    assert "budget exhausted" in report.tool_calls[1]["result"]["error"]
    assert "Unknown analysis tool" in report.tool_calls[2]["result"]["error"]
    messages = client.requests[1]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["c1", "c2", "c3"]
    assert len(messages[-1]["content"]) == 2


@pytest.mark.parametrize("question", ["", "  ", None])
def test_empty_questions_never_call_provider(workspace, question):
    client = ScriptedClient([])
    with pytest.raises(ValueError): InvestigationAgent(workspace, client).ask(question)
    assert not client.requests


def test_fabricated_citations_and_duplicate_calls_fail(workspace):
    with pytest.raises(MetaAPIError, match="not received"):
        InvestigationAgent(workspace, ScriptedClient([final([{"kind": "artifact", "id": "invented"}])])).ask("Investigate")
    call = tool_call("duplicate", "get_experiment", experiment_id="E1")
    with pytest.raises(MetaAPIError, match="duplicate tool calls"):
        InvestigationAgent(workspace, ScriptedClient([assistant(call, call)])).ask("Investigate")


@pytest.mark.parametrize("arguments", [None, {}, []])
def test_malformed_tool_arguments_are_provider_errors(workspace, arguments):
    call = tool_call("c1", "get_experiment", experiment_id="E1")
    call["function"]["arguments"] = arguments
    with pytest.raises(MetaAPIError, match="invalid or duplicate tool calls"):
        InvestigationAgent(workspace, ScriptedClient([assistant(call)])).ask("Investigate")


def test_repeated_questions_use_fresh_history_and_unique_reports(workspace):
    client = ScriptedClient([final(), final()])
    agent = InvestigationAgent(workspace, client)
    a, b = agent.ask("First?"), agent.ask("Second?")
    assert a.path != b.path and Path(a.path).is_file() and Path(b.path).is_file()
    assert all(m.get("content") != "First?" for m in client.requests[1])


def test_cli_meta_uses_target_env_and_offline_override(workspace, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_BASE_URL", raising=False)
    with open(workspace.config_path, "a") as file: file.write("agent_provider: meta\n")
    with server([(200, api_response(final()), {})]) as (url, requests):
        (Path(workspace.root) / ".env").write_text(f"MODEL_API_KEY=test-key\nMETA_API_BASE_URL={url}\n")
        assert main(["--project", workspace.root, "ask", "Why?"]) == 0
    output = capsys.readouterr()
    assert "Report:" in output.out and "test-key" not in output.out + output.err
    assert len(requests) == 1
    assert main(["--project", workspace.root, "ask", "small", "--agent", "offline"]) == 0


def test_cli_provider_failure_returns_nonzero(workspace, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_API_BASE_URL", raising=False)
    with server([(401, b"provider-secret", {})]) as (url, requests):
        (Path(workspace.root) / ".env").write_text(f"MODEL_API_KEY=test-key\nMETA_API_BASE_URL={url}\n")
        assert main(["--project", workspace.root, "ask", "Why?", "--agent", "meta"]) == 2
    output = capsys.readouterr()
    assert "HTTP 401" in output.err and not output.out
    assert "provider-secret" not in output.err
