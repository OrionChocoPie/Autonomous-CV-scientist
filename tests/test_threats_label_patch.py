import copy

import pytest

from scripts.threats_artifacts import file_sha256, training_spec
from scripts.threats_label_patch import apply_label_patch, patch_identity


@pytest.fixture
def reviewed_patch(tmp_path):
    root = tmp_path / 'staged'
    parent = training_spec('E1')['dataset_version']
    changes = []
    for index in range(10):
        relative = f'yolo/labels/train/{index}.txt'
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('1 .5 .5 .2 .2\n1 .5 .5 .201 .201\n0 .8 .8 .1 .1\n')
        changes.append({'label': relative, 'original_sha256': file_sha256(path),
                        'reviews': [{'case_id': f'P{index:02d}', 'decision': 'delete_duplicate',
                                     'note': 'Same flame in scene and crop', 'remove_index': 1, 'keep_index': 0}]})
    patch = {'parent_dataset_version': parent, 'policy': 'separate_visible_fragments',
             'reviewer': 'agent_visual_review', 'changes': changes}
    return root, parent, patch


def test_reviewed_fork_preserves_other_objects_val_and_original(reviewed_patch, tmp_path):
    root, parent, patch = reviewed_patch
    original = tmp_path / 'original.txt'
    original.write_bytes((root / patch['changes'][0]['label']).read_bytes())
    val = root / 'yolo/labels/val/a.txt'
    val.parent.mkdir(parents=True)
    val.write_text('untouched')
    result = apply_label_patch(root, patch, parent)
    assert result == {'dataset_version': patch_identity(patch), 'changed_train_labels': 10,
                      'removed_duplicate_boxes': 10, 'val_test_changed': False}
    assert val.read_text() == 'untouched'
    assert len(original.read_text().splitlines()) == 3
    assert (root / patch['changes'][0]['label']).read_text() == '1 .5 .5 .2 .2\n0 .8 .8 .1 .1\n'
    with pytest.raises(ValueError, match='checksum'):
        apply_label_patch(root, patch, parent)


@pytest.mark.parametrize('failure', ['empty', 'too_few', 'val', 'traversal', 'checksum', 'unreviewed',
                                     'different_class', 'separate_fragment', 'same_index', 'none', 'wrong_policy'])
def test_invalid_patch_changes_nothing(reviewed_patch, failure):
    root, parent, patch = reviewed_patch
    before = {p: p.read_bytes() for p in root.rglob('*.txt')}
    patch = copy.deepcopy(patch)
    if failure == 'empty': patch['changes'] = []
    elif failure == 'too_few': patch['changes'].pop()
    elif failure == 'val': patch['changes'][-1]['label'] = 'yolo/labels/val/a.txt'
    elif failure == 'traversal': patch['changes'][-1]['label'] = 'yolo/labels/train/../a.txt'
    elif failure == 'checksum': patch['changes'][-1]['original_sha256'] = 'incorrect'
    elif failure == 'unreviewed': patch['changes'][-1]['reviews'][0]['decision'] = 'iou_only'
    elif failure == 'different_class': patch['changes'][-1]['reviews'][0]['keep_index'] = 2
    elif failure == 'separate_fragment':
        path = root / patch['changes'][-1]['label']
        path.write_text('1 .2 .2 .1 .1\n1 .8 .8 .1 .1\n')
        patch['changes'][-1]['original_sha256'] = file_sha256(path)
        before[path] = path.read_bytes()
    elif failure == 'same_index': patch['changes'][-1]['reviews'][0]['remove_index'] = 0
    elif failure == 'none': patch = None
    elif failure == 'wrong_policy': patch['policy'] = 'merge_all'
    with pytest.raises(ValueError):
        apply_label_patch(root, patch, parent)
    assert {p: p.read_bytes() for p in root.rglob('*.txt')} == before


def test_missing_label_fails(reviewed_patch):
    root, parent, patch = reviewed_patch
    (root / patch['changes'][-1]['label']).unlink()
    with pytest.raises(FileNotFoundError):
        apply_label_patch(root, patch, parent)


def test_e5_keeps_e1_training_recipe():
    e1, e5 = training_spec('E1'), training_spec('E5')
    assert e5.pop('id') == 'E5-train-reviewed-v2'
    assert e5.pop('parent') == e1.pop('id')
    e1.pop('parent')
    assert e5 == e1
