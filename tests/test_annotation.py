import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from cv_scientist.annotation import DatasetAnnotationAgent, _plan
from cv_scientist.cli import main
from cv_scientist.dataset_review import DatasetTools
from cv_scientist.meta import MetaAPIError, MetaClient, MetaConfig
from cv_scientist.sam import SamClient, parse_boxes
from cv_scientist.workspace import ProjectWorkspace
from test_meta_agent import ScriptedClient, api_response, server


def image(path, size=(100, 80), orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", size, "red") as img:
        exif = Image.Exif()
        if orientation: exif[274] = orientation
        img.save(path, exif=exif)


def plan(classes=None):
    return {"classes": classes if classes is not None else [{"name": "человек", "prompt": "person"}], "limitations": []}


def planner_message(value=None):
    return {"role": "assistant", "content": json.dumps(plan() if value is None else value)}


def sam_response(text=""):
    return {"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text}]}]}


def wire(width=100, height=80, x1=10, y1=20, x2=29, y2=39, identity=0):
    return (f"<0f>{identity}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}|>"
            "<|mask;x=0;y=0;data=20,20,!!!!!(QO(0lu8?|>")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return ProjectWorkspace.init(str(tmp_path / "project"))


class RecordingSam:
    def __init__(self, boxes=None, error=None):
        self.requests, self.boxes, self.error = [], boxes or [], error

    def segment(self, image_url, prompt, width, height):
        self.requests.append((image_url, prompt, width, height))
        if self.error: raise self.error
        return self.boxes


def test_inclusive_edges_single_pixel_full_frame_and_no_matches():
    assert parse_boxes(wire(), 100, 80) == [[0.2, 0.375, 0.2, 0.25]]
    assert parse_boxes(wire(x1=0, y1=0, x2=99, y2=79), 100, 80) == [[0.5, 0.5, 1.0, 1.0]]
    assert parse_boxes(wire(x1=99, y1=79, x2=99, y2=79), 100, 80) == [[0.995, 0.99375, 0.01, 0.0125]]
    assert parse_boxes("", 100, 80) == []


def test_multiple_objects_token_order_and_forward_compatible_fields():
    first = wire(identity=10).replace("<0f>", "<0f;extra=1>,").replace(";x1=10", ";extra=future;x1=10")
    second = ("2<|mask;data=1,1,~!!!!!=c=;x=0;y=0|>"
              "<|future;field=1|><|box;h=80;w=100;y2=79;y1=0;x2=99;x1=0;c=0.75|>")
    assert len(parse_boxes(first + "," + second, 100, 80)) == 2
    assert len(parse_boxes(first + second, 100, 80)) == 2


@pytest.mark.parametrize("text", [None, "\n", "<0f>", "no objects", wire(width=99), wire(x1=-1),
    wire(x2=100), wire(y2=80), wire(x1=30), wire().replace("<0f>", "<1f>"),
    wire().replace("x1=10", "x1=1.5"), wire().replace("x1=10", "x1=10;x1=10"),
    wire().split("<|mask")[0], wire() + "," + wire().removeprefix("<0f>"),
    wire() + "garbage", wire().replace(";h=80", ""), wire().replace("w=100", "w=0")])
def test_invalid_boxes_fail_instead_of_becoming_empty_labels(text):
    with pytest.raises(MetaAPIError): parse_boxes(text, 100, 80)


@pytest.mark.parametrize("value", [None, {}, {"classes": [], "limitations": []},
    plan([{"name": "x", "prompt": ""}]), plan([{"name": "x", "prompt": "person, car"}]),
    plan([{"name": "x", "prompt": "a\nb"}]), plan([{"name": "x", "prompt": "person", "id": 1}]),
    plan([{"name": "x", "prompt": "person"}, {"name": "X", "prompt": "car"}]),
    plan([{"name": "x", "prompt": "person"}, {"name": "y", "prompt": "Person"}]),
    {"classes": plan()["classes"], "limitations": "bad"}, {"classes": plan()["classes"], "limitations": [None]}])
def test_invalid_plan_is_provider_error(value):
    with pytest.raises(MetaAPIError): _plan(json.dumps(value))


def test_real_http_entire_multiclass_dataset_exports_valid_yolo(ws, tmp_path):
    source, output = tmp_path / "data", tmp_path / "annotated"
    for name in ("train/a.jpg", "train/a.png", "val/b.png", "test/c.png", "train/d.png"):
        image(source / "images" / name)
    (source / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [old]\n")
    labels = source / "labels/train/a.txt"
    labels.parent.mkdir(parents=True)
    labels.write_text("0 0.5 0.5 0.1 0.1\n")
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    classes = [{"name": "person", "prompt": "person"}, {"name": "car", "prompt": "car"}]
    replies = [(200, api_response(planner_message(plan(classes))), {})]
    replies += [(200, sam_response(wire() if i % 2 == 0 else ""), {}) for i in range(10)]
    with server(replies) as (url, requests):
        config = MetaConfig("test-key", base_url=url)
        agent = DatasetAnnotationAgent(ws, MetaClient(config), SamClient(config))
        report = agent.annotate(str(source), "Разметь людей и автомобили", str(output))
        # A repeat cannot overwrite data or spend another request.
        with pytest.raises(ValueError, match="already exists"):
            agent.annotate(str(source), "Разметь людей и автомобили", str(output))
    assert len(requests) == 11
    assert requests[0]["path"] == "/v1/chat/completions"
    first = requests[0]["body"]
    assert first["messages"][1]["content"] == "Разметь людей и автомобили"
    assert first["response_format"]["json_schema"]["strict"]
    assert sum(p["type"] == "image_url" for p in first["messages"][2]["content"]) == 4
    assert all(r["path"] == "/v1/responses" for r in requests[1:])
    assert all(r["authorization"] == "Bearer test-key" for r in requests)
    assert [r["body"]["input"][0]["content"][0]["text"] for r in requests[1:]] == ["person", "car"] * 5
    for request in requests[1:]:
        payload = request["body"]
        assert payload["model"] == "sam-3.1" and payload["stream"] is False
        raw = base64.b64decode(payload["input"][0]["content"][1]["image_url"].split(",")[1])
        with Image.open(BytesIO(raw)) as uploaded: assert uploaded.size == (100, 80)
    assert report["images"] == 5 and report["objects"] == 5 and report["sam_requests"] == 10
    assert len(report["planner_sample_images"]) == 4
    exported = DatasetTools(ws, str(output))
    assert exported.statistics["images"] == 5 and exported.statistics["objects"] == 5
    assert exported.statistics["invalid_label_rows"] == 0
    assert exported.statistics["splits"] == {"test": 1, "train": 3, "val": 1}
    assert yaml.safe_load((output / "data.yaml").read_text())["names"] == ["person", "car"]
    assert len({r["image"] for r in report["records"]}) == 5
    assert all((output / r["label"]).read_text() == "0 0.2 0.375 0.2 0.25\n" for r in report["records"])
    saved = json.loads((output / "annotation.json").read_text())
    assert saved["classes"] == classes and "base64" not in json.dumps(saved)
    assert "test-key" not in json.dumps(saved)
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_empty_matches_exif_rotation_and_unspecified_split(ws, tmp_path):
    source, output = tmp_path / "data", tmp_path / "out"
    image(source / "rotated.jpg", size=(100, 80), orientation=6)
    client, sam = ScriptedClient([planner_message()]), RecordingSam()
    report = DatasetAnnotationAgent(ws, client, sam).annotate(str(source), "Люди", str(output))
    assert report["empty_images"] == 1 and report["objects"] == 0
    assert report["records"][0]["split"] == "train"
    assert (output / report["records"][0]["label"]).read_text() == ""
    with Image.open(output / report["records"][0]["image"]) as exported:
        assert exported.size == (80, 100) and not exported.getexif().get(274)
    assert sam.requests[0][2:] == (80, 100)
    assert DatasetTools(ws, str(output)).statistics["label_status"] == {"empty": 1}


@pytest.mark.parametrize("field,value", [("source", None), ("source", ""), ("source", "missing"),
    ("description", None), ("description", "  "), ("output", None), ("output", "")])
def test_invalid_inputs_do_not_call_models(ws, tmp_path, field, value):
    source = tmp_path / "data"
    image(source / "a.png")
    inputs = {"source": str(source), "description": "people", "output": str(tmp_path / "out")}
    inputs[field] = value
    client, sam = ScriptedClient([]), RecordingSam()
    with pytest.raises((ValueError, FileNotFoundError)):
        DatasetAnnotationAgent(ws, client, sam).annotate(**inputs)
    assert not client.requests and not sam.requests


@pytest.mark.parametrize("case", ["empty", "corrupt", "missing_ref", "overlap", "nested", "ancestor", "symlink", "existing"])
def test_preflight_failures_never_spend_requests(ws, tmp_path, case):
    source, output = tmp_path / "data", tmp_path / "out"
    source.mkdir()
    if case != "empty": image(source / "a.png")
    if case == "corrupt": (source / "bad.jpg").write_bytes(b"broken")
    if case == "missing_ref":
        (source / "train.txt").write_text("missing.png\n")
        (source / "data.yaml").write_text("train: train.txt\n")
    if case == "overlap": (source / "data.yaml").write_text("train: a.png\nval: a.png\n")
    if case == "nested": output = source / "out"
    if case == "ancestor": output = source.parent
    if case == "symlink": (source / "link.png").symlink_to(source / "a.png")
    if case == "existing": output.mkdir()
    client, sam = ScriptedClient([]), RecordingSam()
    with pytest.raises(ValueError): DatasetAnnotationAgent(ws, client, sam).annotate(str(source), "people", str(output))
    assert not client.requests and not sam.requests


def test_provider_failure_cleans_partial_export_and_does_not_modify_source(ws, tmp_path):
    source, output = tmp_path / "data", tmp_path / "out"
    image(source / "a.png")
    image(source / "b.png")
    class FailSecond(RecordingSam):
        def segment(self, *args):
            if self.requests: raise MetaAPIError("SAM API HTTP 429: retry later")
            return super().segment(*args)
    sam = FailSecond()
    with pytest.raises(MetaAPIError, match="429"):
        DatasetAnnotationAgent(ws, ScriptedClient([planner_message()]), sam).annotate(str(source), "people", str(output))
    assert not output.exists() and not list(tmp_path.glob(".sam-annotation-*"))
    assert len(list(source.rglob("*.png"))) == 2
    assert not list(source.rglob("*.txt"))


def test_changed_image_is_rejected_after_planning(ws, tmp_path):
    source, output = tmp_path / "data", tmp_path / "out"
    image(source / "a.png")
    class ChangingClient(ScriptedClient):
        def complete(self, *args):
            image(source / "a.png", size=(50, 40))
            return super().complete(*args)
    sam = RecordingSam()
    with pytest.raises(ValueError, match="changed"):
        DatasetAnnotationAgent(ws, ChangingClient([planner_message()]), sam).annotate(str(source), "people", str(output))
    assert not sam.requests and not output.exists()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 302])
def test_sam_http_errors_do_not_expose_response_or_key(status):
    with server([(status, b"private-key secret body", {"Location": "http://127.0.0.1:1/"})]) as (url, requests):
        with pytest.raises(MetaAPIError, match=f"SAM API HTTP {status}") as error:
            SamClient(MetaConfig("private-key", base_url=url)).segment("data:image/png;base64,AA==", "person", 100, 80)
    assert "private-key" not in str(error.value) and "secret body" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("body", [b"invalid json", [], {}, {"status": "failed"}, {"status": "incomplete"},
    {"status": "completed", "output": []}, {"status": "completed", "output": None},
    {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": [{"type": "reasoning"}]},
    sam_response("<0f>"), sam_response("not segmentation")])
def test_invalid_sam_responses_are_not_successful_backgrounds(body):
    with server([(200, body, {})]) as (url, _):
        with pytest.raises(MetaAPIError):
            SamClient(MetaConfig("key", base_url=url)).segment("data:image/png;base64,AA==", "person", 100, 80)


def test_real_http_zero_matches():
    with server([(200, sam_response(), {})]) as (url, _):
        assert SamClient(MetaConfig("key", base_url=url)).segment("data:image/png;base64,AA==", "person", 100, 80) == []


def test_cli_json_and_target_config(ws, tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "data", tmp_path / "out"
    image(source / "a.png")
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "test-private-key")
    with server([(200, api_response(planner_message()), {}), (200, sam_response(wire()), {}),
                 (401, b"secret", {})]) as (url, requests):
        monkeypatch.setenv("META_API_BASE_URL", url)
        args = ["--project", ws.root, "annotate-dataset", str(source), "Разметь людей", "--output", str(output), "--json"]
        assert main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] and result["command"] == "annotate-dataset"
        assert result["data"]["provider"] == "meta"
        assert Path(result["data"]["report"]["path"]).is_file()
        assert main(args) == 2
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_data"
        args[args.index("--output") + 1] = str(tmp_path / "new-out")
        assert main(args) == 2
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "provider_error"
    assert len(requests) == 3
