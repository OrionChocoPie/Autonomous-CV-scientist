# CV Scientist

A daily research workspace for computer vision: research infrastructure +
CV diagnostics + experiment memory + scientist agent (§71).

Two equally important modes (§43): **Copilot** — human drives, the system
analyzes/compares/diagnoses/remembers/recommends; **Autonomous** —
`ResearchEngine` coordinates the same core services toward a goal and budget.
Analysis operations never require `ResearchEngine` or an LLM (§§50, 62).

## Quick start

```bash
source .venv/bin/activate
cv-scientist init
python train.py --config configs/e147.yaml   # your existing workflow, unchanged
cv-scientist ingest runs/e147                # auto-compare, regressions, memory update
cv-scientist status
cv-scientist diagnose                        # deterministic, works with no LLM
cv-scientist compare E145 E146
cv-scientist regressions E146
cv-scientist ask "Why did soup recall drop?"
cv-scientist note "Camera 17 had lens contamination"
cv-scientist session start
cv-scientist watch runs/
cv-scientist research --goal "reduce packaging false positives" --budget 4
```

## Meta research agent

`ask --agent meta` runs a real tool-using, multimodal agent. It can retrieve
project memory, compare experiments, diagnose regressions, select FP/FN examples,
and inspect their original images before answering. It does not launch training
or change experiment data. Offline commands keep their existing behavior.

Set `MODEL_API_KEY` in your environment or a local `.env` (see `.env.example`).
The loader reads `.env` in the current directory, then in the target project;
the target wins, and process environment variables override both. Plain or quoted
`KEY=value` and inline comments are supported; shell expansion is not performed.
The credential file is ignored by Git and keys are not written to agent reports.

```bash
source .venv/bin/activate
cv-scientist ask "Compare B0 and E1. Inspect missed detections and explain the trade-off." \
  --agent meta --model muse-wonder-1.0

# Or choose Spark using the same tools:
cv-scientist ask "Which experiment should I try next?" \
  --agent meta --model muse-spark-1.3
```

The default model is `muse-wonder-1.0`. Selection order: `--model`, `META_MODEL`,
`agent_model` in `.cv-scientist/project.yaml`, then the default. To enable the
agent for ordinary `ask` calls, add `agent_provider: meta` to that file.
`--agent offline` explicitly returns to deterministic memory lookup.
No additional Python packages are required.

For the sibling detection demo, run from this repository so its `.env` is found:

```bash
source .venv/bin/activate
cv-scientist --project ../cv-detection-demo ask \
  "Сравни B0, E1 и F1. Посмотри FN-примеры E1. Как снизить FP при recall >= 0.95?" \
  --agent meta
```

### Error images

The initial artifact adapter reads `predictions.json` in an imported run folder.
It expects the format produced by the detection demo:

```json
[
  {
    "image": "images/val/example.png",
    "truth": [[10, 20, 100, 200]],
    "predictions": [{"box": [12, 22, 98, 198], "score": 0.9}],
    "tp": 1,
    "fp": 0,
    "fn": 0
  }
]
```

Boxes are `xyxy` pixel coordinates. `image` must be a unique relative path.
Image lookup checks the run directory, the project root, and `dataset_root`
from the experiment config or project YAML. For Penn-Fudan, set:

```yaml
dataset_root: data/pennfudan
```

Symlinks cannot escape these roots. The agent sends the selected original image
and box metadata to Meta; PNG, JPEG and WebP are supported, up to 2 MiB per image.
Missing or incompatible artifacts are returned as tool errors for the agent to
explain; metrics-only analysis remains available.

### Limits and reports

Defaults per question: 8 API requests (`--max-steps`), 24 tool calls, 4 images,
4096 maximum completion tokens per request. The loop stops if cumulative
API-reported token usage exceeds 100,000; this check happens after a response
and may overshoot by one request. It is not a monetary spending cap.
HTTP errors, invalid responses and exhausted limits return a nonzero exit code;
they never silently switch to offline output.

Successful answers are saved as `.cv-scientist/reports/ask-<id>.json`, including
the question, model, facts/answer, hypotheses, next steps, validated evidence,
tool results and reported token usage. Reports omit credentials and image bytes.
Evidence IDs in the output schema are restricted to those returned by tools;
the application also checks their kind/id pair. This validates references,
not the truth of every sentence or causal hypothesis. Each question starts a
fresh conversation; previous experiments and notes are accessible through memory.
Hypotheses in this stage are stored in the report, not automatically promoted
to confirmed research conclusions.

API references: [Meta tool calling](https://dev.meta.ai/docs/tool-calling),
[structured output](https://dev.meta.ai/docs/structured-output),
[image understanding](https://dev.meta.ai/docs/image-understanding).

## Executable confidence research

`research --agent meta` closes the loop: a structured Meta proposal chooses a
confidence threshold, a bounded subprocess runs a real evaluation, and measured
results become context for the next proposal. The code decides whether the
acceptance criteria are met; the model cannot declare its experiment successful.

Configure the target project's `.cv-scientist/project.yaml`:

```yaml
dataset_root: data/pennfudan
evaluation_script: task.py
evaluation_python: .venv/bin/python
```

The first executor adapter supports the sibling detection demo's `evaluate`
command and `config.json` / `metrics.json` / `predictions.json` contract. It uses
the target project's virtualenv. Only confidence can vary; baseline weights,
imgsz, device, seed, IoU matching, dataset version, image count and slice
definition stay fixed. Script-directory Python files, weights, dataset YAML,
manifest, validation images and labels are hashed and checked before/after
execution. Baseline GT records must match the validation manifest. These checks
freeze inputs from research startup; legacy exports did not record image hashes
at the time of their original run.

Run from this repository, with its `.env` or `MODEL_API_KEY` configured:

```bash
source .venv/bin/activate
cv-scientist --project ../cv-detection-demo research \
  --agent meta --baseline F1 \
  --goal "Снизить FP при recall не ниже 0.95" \
  --max-fp 28 --min-recall 0.95 --budget 4 --timeout 600
```

`--max-fp` is a **strict** upper bound; recall uses an inclusive lower bound.
The proposal includes its parent, confidence, hypothesis, expected effect and
reason. Confidence is normalized to six decimal places. Exact thresholds under
the same baseline and frozen-input signature are deduplicated, including results
from earlier invocations; neighboring confidence values are distinct experiments.
The loop stops at the first acceptable result. If none is found, it records that
bounded negative result, not a claim that no solution exists.

Limits: 0–4 new evaluations per invocation (failed execution also consumes an
attempt), at most twice that many model requests, 600 seconds by default, and
100,000 API-reported tokens. Zero evaluation budget makes no model request.
Reported tokens are checked after each response, so the limit may overshoot by
one request; this is not a monetary spending cap. Process timeout and API errors
stop research with a nonzero exit code and a partial report. Credentials are not
passed to the evaluation subprocess. `state/confidence.lock` prevents concurrent
confidence loops; after a hard crash, check that no previous evaluation remains
before manually removing a stale lock.

Each proposal and result is journaled to `reports/research-<question-id>.json`;
subprocess logs are stored beside it. Successful evaluations are imported and
compared with the fixed baseline. Questions, hypotheses and measured conclusions
are saved in research memory. A hypothesis's supported/rejected status refers
to the attached acceptance criteria on this fixed validation set, not a proof of
its causal explanation or generalization. Failed execution leaves it untested.
The report includes all outcomes and the best setting satisfying recall, even
when the FP target was not met. Re-running after verified success reuses the
result without another model request or evaluation. No global baseline is
changed and this adapter never launches training.

The original `research` mode without `--agent meta` remains available.

State lives in `.cv-scientist/` (`project.yaml`, `memory/research.db`, plus
`state/ runs/ observations/ experiments/ reports/`) and survives restarts
and manual runs (§44). Integration requires only `project.yaml` plus,
optionally, an adapter mapping your run format to `Experiment` (§61);
trackers sit behind `ExperimentSource` (local implemented; W&B/MLflow/TB stubs) (§60).

## Architecture

```text
                    Core CV capabilities (diagnostics, compare, regressions)
                                   |
              +--------------------+--------------------+
              |                                         |
        Human / Copilot                           ResearchEngine
   compare/diagnose/ask/ingest              autonomous loop (plan: memory
                                            retrieval + dedup check first)
```

| Area | Where |
|---|---|
| Models, provenance, evidence refs (§§52-53, 66-68) | `cv_scientist/models.py` |
| SQLite knowledge base + timeline (§§54, 65) | `cv_scientist/memory.py` |
| Experiment lineage graph (§46) | `cv_scientist/graph.py` |
| Workspace `init` (§44) | `cv_scientist/workspace.py` |
| Diagnostic skills as APIs (§50) | `cv_scientist/diagnostics.py` |
| Compare / Diagnose / Regressions (§§47-49) | `compare.py`, `diagnose.py`, `diagnostics.find_regressions` |
| Ingestion + auto run analysis (§§45, 57) | `ingestor.py`, `analysis.py` |
| Dedup check before proposing (§64) | `dedup.py` |
| Ask over structured memory (§55) | `query.py` |
| Sessions, status, watch (§§51, 56, 58-59) | `sessions.py` |
| Agent (optional LLM layer) / Autonomous engine (§§43, 63) | `agent.py`, `engine.py` |
| Meta API / Tool-using investigation agent / Read-only tools | `meta.py`, `investigation.py`, `agent_tools.py` |
| Adaptive confidence research / Frozen-input evaluation executor | `confidence.py`, `evaluation.py` |

## Product-architecture test (§70)

Workflows A (autonomous), B (human-run + ingest), C (daily diagnosis),
D (research question), E (long-term memory across restarts) are covered in
`tests/test_product.py`:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```
