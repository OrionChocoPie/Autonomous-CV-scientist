# Measured detector experiments: retaining negative results

The operational question was whether four controlled interventions would improve
a YOLO26n detector relative to a 20-epoch control. The private dataset had 8,284
training images and 1,024 validation images with 1,061 validation objects across
four classes. Its v1 labels are pseudo-labels, not asserted human ground truth.
A separate test split was not used. Private images, labels, model weights,
account details and raw reports are not included here.

## Measurements

| Run | Change | mAP50–95 | Precision | Recall | F1 | Fire FP |
|---|---|---:|---:|---:|---:|---:|
| E1 | Control | 38.89% | 66.98% | 60.41% | 63.53% | 133 |
| E2 | close_mosaic: 0 → 5 | 38.66% | 66.63% | 61.17% | 63.78% | 147 |
| E3 | Same E1 weights; inference 640 → 960 | 36.56% | 64.95% | 60.79% | 62.80% | 135 |
| E4 | Same E1 weights; confidence 0.25 → 0.10 | 38.89% | 47.15% | 69.37% | 56.14% | 367 |
| E5 | Remove 66 visually confirmed duplicate train boxes | 38.62% | 65.89% | 59.00% | 62.26% | 131 |

Exact values and all acceptance checks are in
[case-study-results.json](case-study-results.json). The public figures are
recorded aggregates; the private case cannot be independently retrained from this
checkout. The separate synthetic demo is the public reproduction of the analysis
workflow and is not represented as the source of these measurements.

## Controls and acceptance

E2 and E5 each trained for 20 epochs from the same original COCO initialization,
with batch=32, seed=0, 640 px, the same resolved AdamW and an L4 GPU. E2 changed
only close_mosaic; E5 changed only selected train annotations. E3 and E4 used the
unchanged E1 checkpoint. Each run used the same frozen validation images and GT.
A fresh 640/0.25 control exactly reproduced E1's predictions and metrics.

Precision, recall and F1 use class-aware matching at IoU=0.5, confidence 0.25
except E4 (0.10), with NMS IoU=0.7. AP was calculated separately with the same
0.001 confidence floor. E4's identical AP does not imply an improved model: its
operating point changed but its weights and ranking stayed the same.

Prespecified criteria included reducing fire FP to at most 119 for E2/E5 while
preserving defined recall and mAP floors. E3 required both fire and smoke recall
gains, a precision floor and no mAP loss. E4 required a fire-or-smoke recall gain,
better overall F1, precision ≥0.6398 and fire FP ≤146. None passed every condition.

E2 improved F1 by about 0.26 percentage points but increased fire false positives.
E3 took about twice as long per image including I/O/export in one measured pass
and reduced mAP. E4 recovered positives at a large false-alarm cost. E5 slightly
reduced fire FP but lost recall and failed its mAP floor. These are bounded
negative results, not proof that those techniques never help.

## Annotation review

A coding agent visually reviewed all 89 screened overlapping-box pairs in 88
train images plus 100 empty-label images sampled with seed=0. The user defined
instances as separate visible fragments. The review confirmed 66 repeated boxes
on 65 images and preserved 23 ambiguous pairs; two clear missing-fire images
were flagged without adding boxes. This was agent visual review, not human
adjudication or a full relabeling effort.

The original dataset was retained. Only staged train copies received the explicit
patch; validation/test annotations stayed unchanged. Original train/val files
were checked by SHA. The audit established duplicate labels, but the small
patch's failure on frozen pseudo-label validation leaves the broader annotation
quality hypothesis open.

## What CV Scientist did

- Imported five completed runs, including the parity control, through absolute paths.
- Compared and generated reports explicitly against E1; the separate regressions
  command used the existing working baseline B0.
- Produced diagnostics and stored experiment/hypothesis evidence in local memory.
- Retained three rejected hypotheses and the active annotation-quality question.
- Repeated ingestion without new observations or baseline changes.

The coding agent launched Modal jobs and coordinated the larger visual audit and
label patch. CV Scientist did not autonomously launch those training jobs.
Muse-assisted investigation informed the hypotheses; its dataset review used a
bounded sample, not the entire later audit.

Limitations: one seed, no independent test results, pseudo-label validation,
no new-camera evaluation and no production deployment. No accuracy improvement,
user time saving, GPU-cost saving or broad benchmark superiority is claimed.
