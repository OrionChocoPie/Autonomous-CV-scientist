# Original contributions

## Earlier foundation

The project already had a local research-workspace foundation: experiment and
provenance models, a SQLite history, a lineage graph, deterministic comparison,
sessions, a basic CLI and an offline planning interface. These components support
the submission but are not presented as newly invented hackathon features.
They are visible in foundation commit `1eba2f3`.

## Built during the hackathon

The contributor confirmed the event scope below. Commit `8210366` introduced the
initial Meta agent and bounded confidence evaluator; the submission commit also
captures the subsequent feature work and public demo.

| Contribution | Implementation | Verification |
|---|---|---|
| Multimodal Muse investigator | `meta.py`, `investigation.py`, `agent_tools.py` | `test_meta_agent.py`, scripted HTTP/tool/image exchanges, evidence and budget rejection |
| Bounded adaptive evaluation | `confidence.py`, `resolution.py`, `evaluation.py`, `journal.py` | `test_confidence_research.py`, `test_resolution_research.py`; frozen inputs, receipts, interruption and resume |
| Reliable ingestion and recovery | `ingestor.py`, `analysis.py`, `sessions.py` | `test_ingest_recovery.py`; transactions, incomplete writes, watcher restart and repeat ingestion |
| Class-aware image error analysis | `predictions.py`, `diagnostics.py`, `diagnose.py` | `test_class_diagnostics.py`, `test_prediction_diagnostics.py` |
| Local and SSH metadata import with reusable profiles | `importing.py`, `import_source.py`, `import_profiles.py` | `test_importing.py`, `test_import_profiles.py`; immutable revisions, reuse and invalid mappings |
| Read-only image / YOLO dataset review | `dataset_review.py` | `test_dataset_review.py`; counts, previews, paths, missing labels and actual visual coverage |
| Persistent retrieval and offline agent evaluation | `retrieval.py`, `retrieval_eval.py`, `offline_eval.py`, `benchmark.py` | `test_memory_retrieval.py`, `test_memory_scaling.py`, `test_offline_eval.py` |
| Evidence reports, JSON CLI and credential configuration | `reporting.py`, `agent_output.py`, `credentials.py`, `cli.py` | `test_reporting.py`, `test_cli_json.py`, `test_credentials.py` |
| Controlled private-dataset case study | `scripts/modal_threats_train.py`, `modal_threats_evaluate.py`, `threats_artifacts.py`, `threats_label_patch.py` | Artifact hashes, unchanged validation GT, fixed criteria and [aggregate results](case-study.md) |
| Public synthetic industrial-inspection demo | `examples/create_demo.py` | `test_public_demo.py`; no paid API, correct counts, generated images, refusal to overwrite |

Implementation paths in the first eight rows are relative to `cv_scientist/`;
test paths are relative to `tests/`. The new behavior and refinements are the
contribution, even where they extend an earlier module.

## Attribution in the real experiments

CV Scientist imported, compared, diagnosed and reported the results and retained
hypothesis outcomes. A coding agent invoked the Modal launchers, reviewed the
larger image sample and assembled the duplicate-label patch. The Muse-assisted
investigation and dataset-review commands used bounded image samples. The full
188-image review was not an autonomous CV Scientist dataset audit.

The case study demonstrates analysis of successful and failed runs. It does not
claim autonomous GPU training, improved production accuracy or a benchmark win.
