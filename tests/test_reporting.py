import json
from pathlib import Path
import struct
import zlib

import pytest

from cv_scientist.cli import main
from cv_scientist.models import Experiment
from cv_scientist.reporting import generate_report
from cv_scientist.workspace import ProjectWorkspace


def png(width=100, height=100):
    def chunk(kind, data): return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    scanlines = b''.join(b'\x00' + bytes([235, 241, 248]) * width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b'')


@pytest.fixture
def project(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    (run / "image.png").write_bytes(png())
    records = [{"image": "image.png", "truth": [{"box": [10, 20, 50, 70], "class_id": "food"}],
        "predictions": [{"box": [12, 20, 52, 70], "score": 0.95, "class_id": "cup"}], "tp": 0, "fp": 1, "fn": 1}]
    (run / "predictions.json").write_text(json.dumps({"version": 2, "images": records}))
    ws.memory.add_experiment(Experiment(id="B0", name="Baseline", config={"imgsz": 640}, metrics={"recall": 0.95, "fp": 3}))
    ws.memory.add_experiment(Experiment(id="E1", name="Candidate", parent_id="B0", run_dir=str(run),
        config={"imgsz": 960}, metrics={"recall": 0.80, "fp": 1}))
    return ws


def test_report_contains_comparison_regressions_config_confusion_and_overlay(project):
    path = generate_report(project, "E1")
    html = path.read_text()
    assert path.is_relative_to(Path(project.dir) / "reports")
    for expected in ("Регрессии", "Изменения конфигурации", "640", "960", "Confusion matrix", "food", "cup", "Ancestry", "B0"):
        assert expected in html
    assert 'class="box gt" x="10" y="20" width="40" height="50"' in html
    assert 'class="box prediction" x="12"' in html and 'data:image/png;base64,' in html
    assert "E1:predictions.json#0" in html and "file://" in html
    assert 'http://' not in html.replace('http://www.w3.org/2000/svg', '') and 'https://' not in html
    assert "naturalWidth" in html and "naturalHeight" in html


def test_report_missing_images_and_legacy_labels_are_explicit(project):
    image = Path(project.root) / "run" / "image.png"
    image.unlink()
    html = generate_report(project, "E1").read_text()
    assert "Изображение недоступно" in html and 'data:image/' not in html
    path = image.parent / "predictions.json"
    records = [{"image": "image.png", "truth": [[0, 0, 5, 5]], "predictions": [], "fp": 0, "fn": 1}]
    path.write_text(json.dumps(records))
    html = generate_report(project, "E1").read_text()
    assert "class_agnostic" in html and "require predictions version 2" in html


def test_report_escapes_untrusted_text_and_redacts_config(project):
    exp = project.memory.get_experiment("E1")
    exp.name = '<script>alert("x")</script>'
    exp.config.update(api_key="never-render-this", label="<img src=x onerror=alert(1)>")
    project.memory.add_experiment(exp)
    html = generate_report(project, "E1").read_text()
    assert '<script>alert' not in html and '<img src=x' not in html
    assert '&lt;script&gt;' in html and '[redacted]' in html and 'never-render-this' not in html


def test_report_repeat_is_deterministic_and_does_not_update_research_memory(project):
    before = len(project.memory.timeline())
    first = generate_report(project, "E1")
    content = first.read_bytes()
    assert generate_report(project, "E1") == first and first.read_bytes() == content
    assert len(project.memory.timeline()) == before


def test_empty_project_and_missing_optional_artifact(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    assert "Нет импортированных экспериментов" in generate_report(ws).read_text()
    ws.memory.add_experiment(Experiment(id="E1", metrics={"recall": 0.9}))
    assert "predictions.json недоступен" in generate_report(ws).read_text()
    with pytest.raises(ValueError, match="Unknown experiment"): generate_report(ws, "absent")
    with pytest.raises(ValueError, match="Unknown baseline"): generate_report(ws, "E1", "absent")


def test_research_report_includes_attempts_and_links_to_journal(project):
    journal = Path(project.dir) / "reports" / "research-Q1.json"
    journal.write_text(json.dumps({"journal_version": 2, "goal": "Reduce FP", "baseline_id": "B0", "status": "budget_exhausted",
        "parameter": "imgsz", "selected_run": None, "best_feasible_run": "B0", "history": [{"experiment_id": "E1"}],
        "attempts": [{"experiment_id": "E1", "imgsz": 960, "status": "rejected"}], "criteria": {"recall_at_least": 0.95}}))
    html = generate_report(project, research_path=journal).read_text()
    assert "budget_exhausted" in html and "rejected" in html and "imgsz" in html
    assert journal.as_uri() in html and "0.95" in html
    with pytest.raises(ValueError, match="Baseline differs"): generate_report(project, baseline_id="E1", research_path=journal)


def test_gallery_limit_does_not_claim_all_images_were_shown(project):
    path = Path(project.root) / "run" / "predictions.json"
    value = json.loads(path.read_text())
    for i in range(3): value["images"].append({**value["images"][0], "image": f"missing{i}.png"})
    path.write_text(json.dumps(value))
    html = generate_report(project, "E1", max_examples=1).read_text()
    assert 'Показано 1 из 4' in html


@pytest.mark.parametrize("value", [0, -1, None, True, 25])
def test_invalid_gallery_limit(project, value):
    with pytest.raises(ValueError): generate_report(project, max_examples=value)


def test_malformed_artifacts_do_not_leave_partial_html(project):
    (Path(project.root) / "run" / "predictions.json").write_text('{"images":')
    with pytest.raises(ValueError): generate_report(project, "E1")
    assert not list((Path(project.dir) / "reports").glob("*.html"))


def test_cli_report_and_failure_are_logged(project, capsys):
    assert main(["--project", project.root, "report", "E1", "--baseline", "B0"]) == 0
    assert ".html" in capsys.readouterr().out
    assert main(["--project", project.root, "report", "absent"]) == 2
    assert "Unknown experiment" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [{"experiment_id": ""}, {"baseline_id": " "}, {"experiment_id": []},
    {"research_path": ""}, {"research_path": 1}])
def test_invalid_report_inputs_fail_explicitly(project, arguments):
    with pytest.raises(ValueError): generate_report(project, **arguments)


def test_malformed_journal_shapes_fail_without_partial_report(project):
    path = Path(project.dir) / "reports" / "research-invalid.json"
    path.write_text(json.dumps({"journal_version": 2, "baseline_id": "B0", "goal": "Reduce FP", "status": "running", "history": "bad"}))
    with pytest.raises(ValueError, match="journal fields"): generate_report(project, research_path=path)
    assert not list((Path(project.dir) / "reports").glob("*.html"))
