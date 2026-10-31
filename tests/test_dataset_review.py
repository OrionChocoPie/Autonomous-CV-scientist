import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from cv_scientist.cli import main
from cv_scientist.dataset_review import DatasetReviewAgent, DatasetTools
from cv_scientist.meta import MetaAPIError
from cv_scientist.workspace import ProjectWorkspace
from test_meta_agent import ScriptedClient, api_response, assistant, final, server, tool_call


def image(path, color="gray", size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", size, color) as img: img.save(path)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    image(root / "images/train/a.png", "red")
    image(root / "images/train/b.png", "blue", (20, 10))
    image(root / "images/val/c.png", "red")
    (root / "images/val/broken.jpg").write_bytes(b"not a jpeg")
    for name, text in (("train/a", "0 0.5 0.5 0.4 0.5\n"), ("train/b", ""),
                       ("val/c", "1 0.5 0.5 0.1 0.1\n9 0.5 0.5 0.3 0.3\n0 nan 0.5 0.2 0.2\n")):
        label = root / "labels" / (name + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(text)
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [person, car]\n")
    return root


@pytest.fixture
def ws(tmp_path):
    return ProjectWorkspace.init(str(tmp_path / "project"))


def review_final(tools, images=True):
    refs = [{"kind": "artifact", "id": tools.identity + ":statistics"}]
    if images:
        name = tools.get_dataset_examples(4)["examples"][0]["image"]
        refs.append({"kind": "artifact", "id": tools.identity + ":image:" + name})
    return final(refs)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_statistics_decode_images_validate_labels_and_detect_split_leakage(ws, dataset):
    before = snapshot(dataset)
    tools = DatasetTools(ws, str(dataset))
    stats = tools.execute("get_dataset_statistics", {}).data
    assert stats["images"] == 4 and stats["readable_images"] == 3 and stats["unreadable_images"] == 1
    assert stats["splits"] == {"train": 2, "val": 2}
    assert stats["dimensions"]["width"] == {"min": 20, "max": 100, "mean": 220 / 3}
    assert stats["label_status"] == {"present": 1, "empty": 1, "invalid": 1, "missing": 1}
    assert stats["class_objects"] == {"0": 1, "1": 1}
    assert stats["class_objects_by_split"] == {"train": {"0": 1}, "val": {"1": 1}}
    assert stats["objects"] == 2 and stats["invalid_label_rows"] == 2
    assert stats["tiny_boxes_below_32px_area"] == 1
    assert stats["duplicate_groups"] == stats["duplicate_extra_images"] == stats["cross_split_duplicate_groups"] == 1
    assert stats["cross_split_duplicate_examples"] == [["images/train/a.png", "images/val/c.png"]]
    assert stats["screening"]["small_images_below_32px"] == 1
    assert stats["screening"]["near_uniform_contrast_below_2"] == 3
    assert stats["issue_count"] == 2 and stats["evidence"]["kind"] == "artifact"
    assert snapshot(dataset) == before


def test_sampling_is_repeatable_and_images_are_real_decodable_previews(ws, dataset):
    tools = DatasetTools(ws, str(dataset))
    examples = tools.execute("get_dataset_examples", {"limit": 4}).data
    assert examples == tools.get_dataset_examples(4)
    assert {s for r in examples["examples"] for s in r["splits"]} == {"train", "val"}
    assert len(examples["examples"]) == 3
    assert all("image_error" not in r for r in examples["examples"])
    result = tools.execute("view_dataset_image", {"image": "images/val/c.png"})
    url = result.images[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as preview:
        preview.load()
        assert preview.size == (100, 80)
        assert preview.getpixel((0, 0))[0] > 240
    assert result.data["image_shown"] and result.data["boxes"][0]["class_id"] == 1
    image(dataset / "images/val/c.png", "green")
    with pytest.raises(ValueError, match="changed"): tools.view_dataset_image("images/val/c.png")


@pytest.mark.parametrize("source", [None, "", "   "])
def test_invalid_source_never_requests_model(ws, source):
    client = ScriptedClient([])
    with pytest.raises(ValueError): DatasetReviewAgent(ws, client).review(source)
    assert not client.requests


def test_missing_and_empty_sources_never_request_model(ws, tmp_path):
    client = ScriptedClient([])
    agent = DatasetReviewAgent(ws, client)
    with pytest.raises(FileNotFoundError): agent.review(str(tmp_path / "missing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no supported images"): agent.review(str(empty))
    assert not client.requests


def test_image_only_folder_does_not_assume_missing_labels_are_background(ws, tmp_path):
    root = tmp_path / "plain"
    image(root / "one.bmp")
    stats = DatasetTools(ws, str(root)).statistics
    assert not stats["annotations_detected"] and stats["label_status"] == {"missing": 1}
    assert stats["class_names"] is None and stats["objects"] == 0
    assert stats["splits"] == {"unspecified": 1}


@pytest.mark.parametrize("label", ["-1 0.5 0.5 0.1 0.1", "0 0.5 0.5 0 0.1", "0 1 0.5 0.2 0.2",
                                   "0 0.5 0.5 inf 0.1", "0 0.5 0.5 0.2 0.2 0.3", "x 0.5 0.5 0.2 0.2"])
def test_invalid_yolo_rows_are_reported_without_losing_image(ws, tmp_path, label):
    root = tmp_path / "plain"
    image(root / "one.png")
    (root / "one.txt").write_text(label)
    tools = DatasetTools(ws, str(root))
    assert tools.statistics["invalid_label_rows"] == 1
    assert tools.statistics["label_status"] == {"invalid": 1}
    assert tools.statistics["readable_images"] == 1


def test_split_lists_and_shared_image_membership(ws, dataset):
    (dataset / "train.txt").write_text("./images/train/a.png\nimages/val/c.png\nimages/train/missing.png\n")
    (dataset / "data.yaml").write_text("train: [train.txt]\nval: images/val\nnames: {0: person, 1: car}\n")
    stats = DatasetTools(ws, str(dataset)).statistics
    assert stats["images"] == 5 and stats["unreadable_images"] == 2
    assert stats["split_overlap_images"] == 1
    assert stats["split_overlap_examples"] == ["images/val/c.png"]


def test_paths_cannot_escape_and_symlink_images_are_not_transmitted(ws, dataset, tmp_path):
    outside = tmp_path / "outside.png"
    image(outside)
    (dataset / "images/train/link.png").symlink_to(outside)
    tools = DatasetTools(ws, str(dataset))
    assert tools.statistics["unreadable_images"] == 2
    with pytest.raises(ValueError, match="inside"): tools.view_dataset_image("images/train/link.png")
    with pytest.raises(ValueError, match="not in"): tools.view_dataset_image("../outside.png")
    (dataset / "data.yaml").write_text("train: ../\n")
    with pytest.raises(ValueError, match="inside"): DatasetTools(ws, str(dataset))


@pytest.mark.parametrize("config", ["[]", "names: broken", "train: 2", "path: null", "train: ["])
def test_invalid_config_is_an_explicit_error(ws, dataset, config):
    (dataset / "data.yaml").write_text(config)
    with pytest.raises(ValueError): DatasetTools(ws, str(dataset))


def test_real_http_review_sends_statistics_images_and_saves_trace_without_bytes(ws, dataset):
    tools = DatasetTools(ws, str(dataset))
    finish = review_final(tools)
    before, memory_before = snapshot(dataset), ws.memory.timeline()
    messages = [assistant(tool_call("stats", "get_dataset_statistics")), finish]
    from cv_scientist.meta import MetaClient, MetaConfig
    with server([(200, api_response(m), {}) for m in messages]) as (url, requests):
        report = DatasetReviewAgent(ws, MetaClient(MetaConfig("test-key", base_url=url))).review(str(dataset))
    first = requests[0]["body"]
    assert {t["function"]["name"] for t in first["tools"]} == {t["function"]["name"] for t in tools.definitions}
    assert '"unreadable_images": 1' in first["messages"][2]["content"]
    assert sum(i["type"] == "image_url" for i in first["messages"][3]["content"]) == 3
    assert report.usage["requests"] == 2
    assert "Визуально просмотрено 3 из 4" in report.answer
    saved = json.loads(Path(report.path).read_text())
    assert Path(report.path).name.startswith("dataset-review-")
    assert saved["tool_calls"][0]["result"]["images"] == 4
    assert len([c for c in saved["tool_calls"] if c["result"].get("image_shown")]) == 3
    assert "base64" not in Path(report.path).read_text()
    assert snapshot(dataset) == before and ws.memory.timeline() == memory_before


def test_review_requires_statistics_and_viewed_image_citations(ws, dataset):
    tools = DatasetTools(ws, str(dataset))
    with pytest.raises(MetaAPIError, match="viewed image"):
        DatasetReviewAgent(ws, ScriptedClient([review_final(tools, images=False)])).review(str(dataset))
    reference = review_final(tools)
    value = json.loads(reference["content"])
    value["evidence"] = value["evidence"][1:]
    reference["content"] = json.dumps(value)
    with pytest.raises(MetaAPIError, match="statistics"):
        DatasetReviewAgent(ws, ScriptedClient([reference])).review(str(dataset))
    assert not list((Path(ws.dir) / "reports").glob("dataset-review-*.json"))


def test_all_corrupt_images_allow_statistics_only_with_no_fake_view(ws, tmp_path):
    root = tmp_path / "corrupt"
    root.mkdir()
    (root / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
    tools = DatasetTools(ws, str(root))
    client = ScriptedClient([review_final(tools, images=False)])
    report = DatasetReviewAgent(ws, client).review(str(root))
    assert len(client.requests[0]) == 3
    assert not any(t["result"].get("image_shown") for t in report.tool_calls)
    assert report.tool_calls[0]["result"]["unreadable_images"] == 1
    assert "Визуальное качество не проверено" in report.answer


def test_initial_views_count_towards_image_budget_and_reports_are_unique(ws, dataset):
    tools = DatasetTools(ws, str(dataset))
    chosen = tools.get_dataset_examples(1)["examples"][0]["image"]
    finish = final([{"kind": "artifact", "id": tools.identity + ":statistics"},
                    {"kind": "artifact", "id": tools.identity + ":image:" + chosen}])
    client = ScriptedClient([assistant(tool_call("extra", "view_dataset_image", image=chosen)), finish, finish])
    agent = DatasetReviewAgent(ws, client, max_images=1)
    first, second = agent.review(str(dataset)), agent.review(str(dataset))
    assert "Image budget exhausted" in first.tool_calls[-1]["result"]["error"]
    assert first.path != second.path and Path(first.path).is_file() and Path(second.path).is_file()


def test_cli_json_target_project_and_provider_error(ws, dataset, monkeypatch, capsys):
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "private-key")
    tools = DatasetTools(ws, str(dataset))
    with server([(200, api_response(review_final(tools)), {}), (401, b"secret", {})]) as (url, requests):
        monkeypatch.setenv("META_API_BASE_URL", url)
        args = ["--project", ws.root, "review-dataset", str(dataset), "--json"]
        assert main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] and result["command"] == "review-dataset"
        assert result["data"]["provider"] == "meta" and Path(result["data"]["report"]["path"]).is_file()
        assert main(args) == 2
        result = json.loads(capsys.readouterr().out)
        assert result["error"]["code"] == "provider_error"
    assert len(requests) == 2


def test_initial_tool_budget_is_enforced_without_model_request(ws, dataset):
    client = ScriptedClient([])
    with pytest.raises(MetaAPIError, match="tool-call budget"):
        DatasetReviewAgent(ws, client, max_tool_calls=1).review(str(dataset))
    assert not client.requests


def test_image_limits_are_recorded_and_previews_are_bounded(ws, tmp_path, monkeypatch):
    root = tmp_path / "limits"
    image(root / "large.png", size=(2048, 1024))
    tools = DatasetTools(ws, str(root))
    result = tools.view_dataset_image("large.png")
    with Image.open(BytesIO(base64.b64decode(result.images[0]["image_url"]["url"].split(",", 1)[1]))) as preview:
        assert preview.size == (1024, 512)
    monkeypatch.setattr("cv_scientist.dataset_review.MAX_PIXELS", 100)
    stats = DatasetTools(ws, str(root)).statistics
    assert stats["unreadable_images"] == 1 and stats["dimensions"]["width"] is None
    monkeypatch.setattr("cv_scientist.dataset_review.MAX_PIXELS", 40_000_000)
    (root / "oversized.png").write_bytes(b"x" * (32 * 1024 * 1024 + 1))
    stats = DatasetTools(ws, str(root)).statistics
    assert stats["unreadable_images"] == 1
    assert "byte audit limit" in stats["issue_examples"][0]["image_error"]
