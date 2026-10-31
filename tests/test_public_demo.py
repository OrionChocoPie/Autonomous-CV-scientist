import json
from pathlib import Path

import pytest
from PIL import Image

from cv_scientist.cli import main as cli
from cv_scientist.meta import MetaClient
from cv_scientist.predictions import match_pairs, validate_predictions
from cv_scientist.workspace import ProjectWorkspace
from examples.create_demo import create_demo, main


def test_public_demo_is_offline_and_reproduces_visible_tradeoff(tmp_path, monkeypatch, capsys):
    def forbid_api(*args, **kwargs):
        raise AssertionError('The public synthetic demo must not call a paid model')
    monkeypatch.setattr(MetaClient, 'complete', forbid_api)
    root = tmp_path / 'inspection'
    result = create_demo(root)
    assert result['synthetic'] is True and result['trained_model'] is False and result['api_calls'] == 0
    assert result['experiments']['B0']['tp'] == 3
    assert result['experiments']['B0']['fp'] == 4
    assert result['experiments']['B0']['fn'] == 1
    assert result['experiments']['E1']['tp'] == 2
    assert result['experiments']['E1']['fp'] == 1
    assert result['experiments']['E1']['fn'] == 2
    assert result['experiments']['E1']['precision'] > result['experiments']['B0']['precision']
    assert result['experiments']['E1']['recall'] < result['experiments']['B0']['recall']
    ws = ProjectWorkspace(str(root))
    assert ws.memory.original_baseline().id == ws.memory.working_baseline().id == 'B0'
    assert ws.memory.get_experiment('E1').parent_id == 'B0'
    assert ws.memory.get_experiment('E1').slices['small']['recall'] == 0
    for identity in ['B0', 'E1']:
        artifact = json.loads((root / 'runs' / identity / 'predictions.json').read_text())
        for row in validate_predictions(artifact):
            pairs, _ = match_pairs(row, .5)
            assert row['tp'] == len(pairs)
            with Image.open(root / 'data' / row['image']) as image:
                assert image.size == (640, 384)
    report = Path(result['report']).read_text()
    assert 'data:image/png;base64,' in report and 'E1:predictions.json#' in report
    assert cli(['--project', str(root), 'compare', 'B0', 'E1', '--json']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['ok'] and output['data']['comparison']['overall']['recall']['abs'] == -.25


def test_public_demo_repeat_preserves_existing_files(tmp_path):
    root = tmp_path / 'inspection'
    create_demo(root)
    artifact = root / 'runs/E1/predictions.json'
    before = artifact.read_bytes()
    with pytest.raises(FileExistsError, match='already exists'):
        create_demo(root)
    assert artifact.read_bytes() == before
    marker = tmp_path / 'user-file'
    marker.write_text('belongs to user')
    with pytest.raises(FileExistsError):
        create_demo(marker)
    assert marker.read_text() == 'belongs to user'


@pytest.mark.parametrize('destination', [None, '', ' ', 0])
def test_public_demo_invalid_destination_has_no_writes(tmp_path, monkeypatch, destination):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='nonempty'):
        create_demo(destination)
    assert list(tmp_path.iterdir()) == []


def test_public_demo_cli_reports_existing_destination(tmp_path):
    with pytest.raises(SystemExit) as result:
        main(['--output', str(tmp_path)])
    assert result.value.code == 2
