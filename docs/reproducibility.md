# Reproduce the public workflow

## Environment

The core package requires Python 3.10+, PyYAML, Pillow and SQLite with FTS5.
A project-local `.venv` is used throughout. The optional `dev` extra adds pytest;
the optional `cloud` extra adds Modal for the separate private-volume launchers.
Torch and Ultralytics are installed in those launchers' cloud image, not needed
for the public demo or core package.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
python examples/create_demo.py --output demo-workspace
```

Use another destination if `demo-workspace` already exists. The generator refuses
to overwrite an existing file or directory. It makes no network requests and
never calls a model.

## Demo data and expected result

The generator draws five 640×384 industrial-inspection images and writes YOLO
labels, version-2 predictions and run metadata. Labels and predictions are
hand-authored fixtures in the source, not detector outputs or benchmark ground
truth. B0 keeps scores ≥0.25; E1 keeps scores ≥0.70 from the same fixture.
A wrong-class box deliberately remains to exercise class-aware diagnosis.

| Count / metric | B0 | E1 |
|---|---:|---:|
| True positives | 3 | 2 |
| False positives | 4 | 1 |
| False negatives | 1 | 2 |
| Precision | 3/7 | 2/3 |
| Recall | 3/4 | 2/4 |
| Small-image slice recall | 1/2 | 0/2 |

Counts use class-aware greedy matching at IoU=0.5. The small slice contains
images with at least one GT box of area ≤1024 pixels. No AP/mAP or trained
checkpoint is fabricated. Generated assets are original synthetic fixtures
covered by the repository's MIT license.

Inspect the result:

```bash
source .venv/bin/activate
cv-scientist --project demo-workspace compare B0 E1 --json
cv-scientist --project demo-workspace diagnose E1 --json
cv-scientist --project demo-workspace report E1 --baseline B0 --json
cv-scientist --project demo-workspace context --json
```

Reports include embedded synthetic images and box overlays. Evidence links back
to local artifacts will require those local files; moving only the HTML does not
move the linked JSON. Generated workspace state is ignored by Git.

## Offline application evaluations

```bash
source .venv/bin/activate
python -m cv_scientist.offline_eval --output /tmp/cv-scientist-offline-eval.json
python -m cv_scientist.retrieval_eval --output /tmp/cv-scientist-retrieval-eval.json
```

These fixed fixtures evaluate retrieval and guardrails with scripted clients.
They make no paid API requests and do not estimate live-model reasoning quality.
The local test suite also covers empty inputs, missing artifacts, invalid classes,
unsafe paths, interrupted evaluations and repeated ingestion.

## Real data and external evaluation

Real investigations require your images and compatible prediction artifacts.
The [artifact reference](usage.md#error-images) documents their format.
`ask --agent meta` needs a real key and sends selected image inputs to Meta.
The bounded research adapter needs a compatible external evaluation script,
validation manifest and checkpoint; it is not exercised by the synthetic demo.

The [private case study](case-study.md) publishes aggregate results only. Its
original dataset, images, labels, weights and raw model reports are not licensed
for redistribution by this repository. The public checkout cannot independently
retrain that dataset; use your own authorized data and adapt the run inputs.
