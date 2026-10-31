# CLI and artifact reference

A daily research workspace for computer vision: research infrastructure +
CV diagnostics + experiment memory + scientist agent.

Two equally important modes: **Copilot** — human drives, the system
analyzes/compares/diagnoses/remembers/recommends; **Autonomous** —
`ResearchEngine` coordinates the same core services toward a goal and budget.
Analysis operations never require `ResearchEngine` or an LLM.

## Install and configure

Install from a checkout into your virtual environment, then configure the Meta
API key once for all projects:

```bash
source .venv/bin/activate
python -m pip install .
cv-scientist config
```

`config` prompts for the key without displaying it. It works outside a project
and saves the key as plaintext in `~/.config/cv-scientist/credentials.json`, with
file permissions `0600`. If `XDG_CONFIG_HOME` is set to an absolute directory,
the file is stored in `$XDG_CONFIG_HOME/cv-scientist/credentials.json` instead.
Run `cv-scientist config` again to replace the saved key. It never writes the
key to project files or prints it.

For CI or a temporary override, set `CV_SCIENTIST_API_KEY` in the process
environment. Key precedence is: process environment, target project's `.env`,
current directory's `.env`, then the global credentials file. An explicitly empty
key is an error. The old `MODEL_API_KEY` name is no longer read; rename it in your
environment or `.env`, or use `cv-scientist config`.

## Structured output for coding agents

Codex, Claude Code, Muse Code and other agents that invoke shell commands can
use the same CLI with `--json`. No provider-specific integration or LLM key is
needed for context, history search, comparison or diagnostics:

```bash
source .venv/bin/activate
cv-scientist context --json
cv-scientist search "small objects recall" --limit 5 --json
cv-scientist experiments --json
cv-scientist compare B0 E1 --json
cv-scientist diagnose E1 --json
cv-scientist ingest runs/E2 --json
# Select a project independently of the shell's current directory:
cv-scientist --project /path/to/project context --limit 3 --json
```

`--json` is accepted before or after the subcommand. stdout contains exactly one
JSON object followed by a newline. Progress logs and interactive prompts use
stderr. Without `--json`, existing commands retain their human-readable output.
Command side effects are unchanged: for example, `ingest` records an experiment
and `compare` saves a comparison. CLI invocations are still logged locally.

The version 1 response envelope always has these five fields:

```json
{
  "schema_version": 1,
  "command": "search",
  "ok": true,
  "data": {
    "query": "E1",
    "limit_per_group": 5,
    "results": {},
    "truncated": {}
  },
  "error": null
}
```

`data` is an object on success and `null` on failure. Failure responses have
`ok: false` and `error: {"code": "...", "message": "..."}`. Agents should branch on
`error.code`, check the process exit code, and ignore unknown additive fields.
Breaking changes to this contract require a new `schema_version`. Numeric values,
booleans and nulls retain their JSON types; nonfinite numbers are explicit errors.
Credential-named fields and sensitive environment values are redacted in JSON.

`context` returns project identity, primary metric, experiment counts, best/latest
and original/working baselines, recent experiments and observations, open questions
(including investigating/blocked), active/untested hypotheses, recent conclusions
and user notes, regressions and the active session. Experiment summaries omit
full configurations; use `experiments` or search by experiment ID to retrieve
them. Goals written as notes remain user notes; they are not inferred requirements.

`search` uses the existing local keyword index and returns `results` grouped by
experiments, observations, hypotheses, conclusions, questions, notes and regressions.
Each record carries `reference: {"kind": "...", "id": "..."}`. Context records also
carry references, and diagnosis findings retain their artifact evidence pointers.
No matches or an empty search string produce successful empty groups. Search
results preserve index ranking; context history is newest first. Both commands
accept `--limit` from 1 to 100, default 5, **per group**, and report omitted records
in `truncated`. An initialized empty project has empty lists and null baselines;
an uninitialized directory returns `workspace_not_found` without creating a workspace.

| Command | Main fields in `data` |
|---|---|
| `init` | `root`, `workspace`, `name` |
| `status` | `status` |
| `context` | `project`, `status`, history groups, `active_session`, `limit_per_group`, `truncated` |
| `search` | `query`, `results`, `limit_per_group`, `truncated` |
| `experiments`, `hypotheses`, `conclusions` | Corresponding array of records with references |
| `compare` | `comparison`, `evidence` |
| `diagnose` | `diagnosis`, `evidence` |
| `regressions` | `regressions`, `experiment_id`, `baseline_id`, `evidence` |
| `ingest` | `experiment`, `analysis`, `evidence` |
| `import` | Full import report, including `runs`, `skipped`, `usage`, `path` |
| `ask --agent offline` | `provider`, `answer`, `evidence`, `confidence` |
| `ask --agent meta`, `research --agent meta` | `provider`, `report` |
| `research --agent offline` | `provider`, `result` (planned hypotheses do not imply execution) |
| `note`, `session`, `timeline` | `note`, `session`, or `events`, respectively |
| `report`, `config` | `path`; `config` still requires hidden interactive input |
| `watch` | Final `events`, `interrupted`; emitted when watching stops, not a live JSON stream |
| `--help` | `help`, inside a response with `command: "help"` |

Exit codes are 0 for success, 2 for argument/data/provider/IO errors, 1 for
unexpected internal errors, and 130 for an interrupted operation. Cancelling the
interactive `config` command retains exit code 2; stopping `watch` retains exit
code 0 with `interrupted: true`. Error codes include `invalid_arguments`,
`workspace_not_found`, `experiment_not_found`, `baseline_not_found`, `invalid_data`,
`file_not_found`, `io_error`, `database_error`, `provider_error`, `input_required`,
`interrupted` and `internal_error`. Parsing failures use `command: null`.

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

Run `cv-scientist config` once, or set `CV_SCIENTIST_API_KEY` in your environment
or a local `.env` (see [`.env.example`](../.env.example)). Local `.env` values override the global
key, and process environment variables override both. Plain or quoted `KEY=value`
and inline comments are supported; shell expansion is not performed.
Project `.env` files are ignored by Git and keys are not written to agent reports.

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

For a project with prediction artifacts, pass its workspace path:

```bash
source .venv/bin/activate
cv-scientist --project /path/to/detection-project ask \
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
The original list format is still supported with explicitly class-agnostic matching.
For class-aware diagnostics use version 2, with a `class_id` on every GT and
prediction object:

```json
{
  "version": 2,
  "images": [{
    "image": "images/val/example.png",
    "truth": [{"box": [10, 20, 100, 200], "class_id": "food"}],
    "predictions": [{"box": [12, 22, 98, 198], "score": 0.9, "class_id": "cup"}],
    "tp": 0, "fp": 1, "fn": 1
  }]
}
```

Class IDs are nonempty strings or nonnegative integers, using one type throughout
the artifact. Novel predicted classes appear in the confusion matrix. Incomplete
labels, mixed class-ID types and unsupported versions are explicit errors.
Matching first pairs same-class boxes by descending confidence; remaining FP/FN
are paired geometrically to identify wrong-class detections. A wrong class at
sufficient IoU contributes one FP, one FN and one off-diagonal confusion entry.
Unpaired detections/misses use `null` for background/missing prediction in the
structured matrix. An overlapping box below the matching IoU is a localization
error rather than an established classification error. `diagnose` and agent tools
expose the matrix, totals, limitations and citable classification findings.

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
Missing API-reported usage is also an error: the agent cannot enforce a token
budget without it. An irrelevant initial catalog does not force unrelated
citations when retrieval finds no evidence; successfully retrieved facts still
require citations, and every supplied kind/id pair is validated.

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

## Dataset quality review

Review a local folder of images, optionally with YOLO detection labels:

```bash
source .venv/bin/activate
cv-scientist review-dataset /path/to/dataset
cv-scientist --project /path/to/project review-dataset /path/to/dataset --json
```

The command requires an initialized workspace and a configured Meta API key, as
for `ask --agent meta`. `--model` and `--max-steps` select the model and request
limit. It reads the dataset without changing images, labels or research memory.

The audit decodes images and measures dimensions, thumbnail brightness/contrast,
readability, exact byte duplicates and duplicate leakage across train/val/test.
For YOLO it counts classes and objects per split, tiny boxes, missing/empty labels,
invalid class IDs and malformed or out-of-bounds normalized boxes. Empty labels
and missing labels remain distinct; missing annotations are not assumed to be
verified backgrounds. Suspicious brightness/contrast is a screening signal.

The dataset directory may contain `data.yaml` or `dataset.yaml`, with `names`,
`path` and `train`/`val`/`test` paths, path lists or image-list text files. All
referenced paths must stay within the selected dataset directory. Without YAML,
split names are inferred from train/val/test directory names. Labels follow
`images/.../name.jpg` → `labels/.../name.txt`, or sit beside images when there
is no `images` path segment. Only detection rows `class cx cy w h` are parsed.

Meta receives the statistics and up to four actual image previews, selected
deterministically from split/class groups, suspicious images and spread-out
paths. Previews preserve aspect ratio within 1024×1024; metadata includes original
dimensions and normalized YOLO boxes. The agent evaluates visual relevance,
noise, exposure and possible annotation issues, separating observed defects from
suspicions. The report records the number viewed versus the total. This limited
sample cannot certify the whole dataset or prove absence of label noise.

Reports are saved to `.cv-scientist/reports/dataset-review-<id>.json`, with the
answer, evidence references, statistics, sample metadata, tool trace and reported
API usage. Image bytes are omitted. `--json` uses the existing response envelope
with `data.provider` and `data.report`. Repeated reviews write separate reports.
Empty folders return an error without a model request; unreadable images remain
in the statistics. If every image is unreadable, the report explicitly marks
visual quality as unverified.

Limits: local PNG/JPEG/WebP/BMP/TIFF files, 100,000 images, 32 MiB and 40 million
pixels per decoded image, 2 MiB per YAML/label/image-list file. Oversized images
are recorded as unreadable. Paths through symlinks are not read; hidden and
symlink directories are excluded. Only exact byte duplicates are detected, not
perceptual near duplicates. COCO, segmentation, pose and SSH are unsupported.
The Meta request, tool, image and token budgets are the same as for `ask`;
the initial statistics/sample/image tools also consume the tool/image budgets.

## Import existing experiment results

`import` recognizes existing result layouts and converts metadata into normalized
**local copies in `.cv-scientist/import/copies/`**. The first recognition uses Meta;
validated field mappings are saved in `.cv-scientist/import/profiles/`. Subsequent
matching runs can be imported without a model request or API key. Original files
are never modified, training is never launched, and import does not open or add
anything to research memory. It creates local profile/report/log directories even
before `init`.

```bash
source .venv/bin/activate
python -m pip install -e .
cv-scientist import /path/to/legacy-project --agent meta
cv-scientist import gpu-host:/srv/experiments/my-detector --agent meta
# Equivalent SSH URL:
cv-scientist import ssh://gpu-host/srv/experiments/my-detector --model muse-wonder-1.0
```

Configure the key with `cv-scientist config` or `CV_SCIENTIST_API_KEY`, as for
`ask`. `--project` selects the local workspace for configuration, profiles, copies,
reports and logs; otherwise the discovered project or current directory is used.
SSH uses your configured aliases and
keys, verified known hosts and noninteractive authentication. Remote file access
requires a Linux host with `sh`, GNU `find`, `head`, `mktemp` and `mv`; no Python
installation or CV Scientist installation is required on that host. Copies and
reports for SSH sources are stored locally; the remote source is read only.

The agent reads JSON, nested YAML (via PyYAML), CSV, documentation, logs and Python
source. It selects original fields by file and JSON Pointer; the converter reads
the numbers directly, validates them and preserves the original configuration,
including unresolved Hydra expressions. There is no execution of generated code.
Per-class and slice metrics are normalized separately. Unknown metadata stays
unknown; the importer never guesses parents, dataset versions or original dates.
Missing/incompatible runs are explicitly skipped with a reason. The report lists
converted/skipped runs, limitations, tool errors, model and reported token usage.

Each converted result is stored as:

```text
<local-project>/.cv-scientist/import/copies/<stable-run-id>/<input-sha256>/
    experiment.json      # full normalized Experiment
    config.json          # nested source configuration
    metrics.json         # overall, per_class, slices
    source.json          # provenance, field evidence, checksums, uncertainties
    raw-000.yaml         # original small files used for the conversion
    raw-001.csv
```

Unchanged inputs reuse the existing copy without overwriting it. Changed inputs
produce a new immutable revision under the same run ID. Every invocation writes
a separate report at `.cv-scientist/reports/import-<id>.json`. Selected source files are
checked again before publication; a changed input aborts publication. Each result
directory is published atomically, with exclusive publication locking. A failure
while publishing multiple runs can leave earlier complete copies; rerunning reuses
them. A modified or incomplete existing copy is an error, not silently repaired.

Ingest is a separate action, using a **local** copy directory:

```bash
source .venv/bin/activate
cv-scientist init
cv-scientist ingest .cv-scientist/import/copies/<run-id>/<revision>
```

SSH imports can be ingested directly from the local copy. Original remote
checkpoint references remain remote; this command does not make weights locally
executable. Imported copies preserve an unknown parent rather than linking to the
previously ingested run. Existing `ingest` baseline selection still applies; verify
the intended baseline before comparing imported results from different datasets.

Profiles store file templates, JSON Pointers, destination metrics and structural
fingerprints. They apply to subsequent runs and unambiguous matching sources in
this local workspace. Numeric values are re-read each time. Changed headers,
split/epoch selectors, missing files and invalid values in a bound run are errors;
they never silently reuse old numbers. New unmatched formats require recognition
by the agent. Unknown completion/dataset identity remains unknown for new runs.
Previously skipped runs are reconsidered through the agent. Existing copies made
by the original low-level importer remain readable; they are not migrated or deleted.

Limits: 64 API requests by default (`--max-steps`), 256 tool calls, 100,000
API-reported tokens (checked after responses; not a monetary spending cap), 2 MiB
per inspected file, and 1,000 entries per directory. Empty directories make no
model request. Hidden files (including `.env`) and virtualenvs are not inspectable.
Large archives may require importing narrower run directories to stay within the
request and token budgets; exceeding a budget returns an error without publishing
prepared copies.
This first version imports experiment **metadata**: Parquet predictions, images,
weights and datasets are not converted/copied. Metrics-only diagnosis remains
available; visual error analysis requires compatible prediction/image artifacts
to be supplied separately. The agent's interpretation of source fields is recorded
as evidence, but semantic correctness of every mapping still requires review.

## Executable confidence research

`research --agent meta` closes the loop: a structured Meta proposal chooses a
confidence threshold, a bounded subprocess runs a real evaluation, and measured
results become context for the next proposal. The code decides whether the
acceptance criteria are met; the model cannot declare its experiment successful.

The same bounded protocol also supports resolution-only research:

```bash
source .venv/bin/activate
cv-scientist research --agent meta --parameter imgsz --baseline F1 \
  --goal "Reduce FP while keeping recall >= 0.95" --budget 4
```

`imgsz` must be an integer multiple of 32 from 32 through 4096. Confidence, weights,
seed, dataset and the other fixed settings stay unchanged. The evaluation adapter
must honor its existing `--imgsz` argument in the exported configuration; otherwise
the result is rejected. No training is launched. Duplicate/baseline resolutions
do not consume an evaluation; failed/interrupted attempts do. Journals record the
selected parameter and retain the same request/token/evaluation/time budgets and
completion-receipt recovery rules as confidence research. Resume automatically
selects the saved mode; a conflicting explicit `--parameter` is an error.
Legacy journals without a parameter field remain confidence journals.
Validated resolution runs are stored in `.cv-scientist/runs/`. The external
evaluation script keeps its existing `runs/<id>` output protocol; the adapter
moves the complete resolution run locally before writing its completion receipt.
Interrupted, unconfirmed script outputs remain untrusted and are not imported.

For version 2 predictions, the fixed validation manifest's `boxes` must contain
the same labelled GT objects, so class identities are frozen and checked too.

Configure the target project's `.cv-scientist/project.yaml`:

```yaml
dataset_root: data/pennfudan
evaluation_script: task.py
evaluation_python: .venv/bin/python
```

The first executor adapter requires a compatible external detection project's `evaluate`
command and `config.json` / `metrics.json` / `predictions.json` contract. It uses
the target project's virtualenv. For confidence research only confidence can vary; baseline weights,
imgsz, device, seed, IoU matching, dataset version, image count and slice
definition stay fixed. Script-directory Python files, weights, dataset YAML,
manifest, validation images and labels are hashed and checked before/after
execution. Baseline GT records must match the validation manifest. These checks
freeze inputs from research startup; legacy exports did not record image hashes
at the time of their original run.

With the package installed in your active virtual environment and the key set
through `cv-scientist config` or `CV_SCIENTIST_API_KEY`, run:

```bash
source .venv/bin/activate
cv-scientist --project /path/to/detection-project research \
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
before recovering a legacy lock. New journals record the owner and evaluation
PID. `research --resume` reclaims a known stale lock only when its recorded owner
and child are no longer alive on this host; unknown/active locks remain intact.

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

### Resume an interrupted research run

```bash
source .venv/bin/activate
cv-scientist --project /path/to/project research \
  --resume /path/to/project/.cv-scientist/reports/research-<question-id>.json
```

Resume uses the original goal, question, baseline, input signature, acceptance
criteria and total evaluation/request/token/active-time budgets. Offline downtime
does not consume active time. The baseline snapshot and fixed inputs must still
match. A validated evaluation has an atomic completion receipt containing hashes
of its config, metrics and predictions. Resume can import that result after a
crash before memory was updated; it does not run the same evaluation again.
Partial/unconfirmed attempts stay in the journal, consume an evaluation attempt
and retain an untested hypothesis; their thresholds are not retried. Confirmed
artifacts are revalidated, including when resuming a finished run.

Requests are journaled before calling the provider. If a response was lost, its
request remains spent and a conservative reservation (configured maximum output,
UTF-8 context bytes and protocol margin) remains charged separately from reported
token usage. This prevents restarting from silently resetting usage; it is not an
exact measurement of the lost response. Old journals without the new resume
metadata cannot be safely resumed. Do not edit journals to reset budgets.

## Reliable ingestion and object diagnostics

Local runs require a valid, nonempty `metrics.json` (overall or grouped metrics).
Existing config files must parse as JSON or nested YAML. Missing required files,
partial JSON, invalid shapes and nonfinite/non-numeric metrics are explicit
errors; an absent optional config is allowed. Repeated ingestion preserves run
identity, original import time and ancestry. Repeated analysis of unchanged
inputs/baseline reuses its recorded result instead of duplicating observations.

`watch` persists successful input fingerprints in SQLite. A failed or changing run
is retried on the next scan; ingestion, analysis and checkpointing form one
transaction. Restarting a watcher skips unchanged successful runs and detects
updated result files. A missing watch directory is logged and retried.

When `predictions.json` exists, `diagnose` also matches boxes greedily by descending
confidence at `config.iou_match` (default 0.5). Version 2 uses class-aware matching;
legacy artifacts remain class-agnostic with an explicit limitation. Errors are grouped by:

- pixel area: small `<32²`, medium `[32²,96²)`, large `>=96²`;
- confidence: low `<0.5`, medium `[0.5,0.8)`, high `>=0.8`; FN confidence refers
  only to an overlapping candidate prediction, otherwise it is unknown;
- overlap: classification, duplicate, localization/background FP;
  classification, crowded/localization/uncovered FN.

Each group counts **objects**, includes up to three source-image evidence
references, and reports its share within FP or FN (groups across dimensions
overlap and must not be summed). Exported counts that disagree with matching are
flagged separately. Missing predictions are optional; malformed predictions are
errors. Grouping is descriptive and does not establish annotation errors or
causal explanations.

## Local HTML reports and operation journal

```bash
source .venv/bin/activate
cv-scientist report E1 --baseline B0
cv-scientist report                         # latest experiment and working baseline
cv-scientist report --research .cv-scientist/reports/research-<question-id>.json
```

Reports are written to `.cv-scientist/reports/report-<id>.html`. Open the printed
path in a browser. They include overall/per-class/per-slice comparisons, config
changes, regressions, lineage, confusion matrices, diagnostic evidence and up to
12 source-image examples across the report. GT and prediction boxes are overlaid
at the original image dimensions; labels/statuses are available in box tooltips
and expandable records. Source images are embedded for offline viewing. Missing
or unsupported images are marked explicitly. Missing optional predictions still
allow metrics-only reports; malformed artifacts abort generation. Research reports
include the criteria, budget, attempts and selection with a link to the journal.
Repeated generation replaces the same report snapshot without changing research
memory. Source-artifact links require the original files to remain accessible.

All parsed CLI commands append start/completion/failure events to
`.cv-scientist/logs/operations.jsonl`. Import requests/tools/profile reuse,
research checkpoints, agent requests/tools and watcher outcomes also append
structured events. Events include UTC timestamps, IDs, statuses, usage/counts
and report paths; they exclude raw CLI input, prompts, image bytes and source
file contents. Secret config keys are masked in HTML and import reports. The
journal serializes concurrent appends and flushes each event to disk. Reports,
profiles and logs remain local for SSH sources too. No rotation/deletion is
automatic; existing state is preserved.

```text
.cv-scientist/
    project.yaml
    memory/research.db
    import/profiles/<format-sha256>.json
    import/sources/<source-sha256>.json
    import/copies/<run-id>/<input-sha256>/...
    runs/<resolution-run-id>/...
    reports/import-<id>.json
    reports/research-<question-id>.json
    reports/report-<id>.html
    logs/operations.jsonl
```

## Offline evaluation and memory performance

Search uses SQLite FTS5 with weighted identity/title/body matching, short exact
experiment IDs, a small documented English/Russian CV vocabulary, and no fallback
to unrelated records. Regressions are searchable and citable. Existing databases
receive a one-time backfill; writes and search-index updates share transactions.
SQLite must include FTS5; no embedding service or paid API is used.

Reproduce the fixtures and benchmarks:

```bash
source .venv/bin/activate
python -m cv_scientist.retrieval_eval --output /tmp/search-evaluation.json
python -m cv_scientist.offline_eval --output /tmp/agent-evaluation.json
python -m cv_scientist.benchmark --experiments 3000 --repeats 7 --output /tmp/memory-benchmark.json
# Reproduce the same fixture on a Git revision in a temporary snapshot:
python -m cv_scientist.benchmark --baseline-revision HEAD --experiments 3000 --repeats 7 --output /tmp/memory-baseline.json
python -m pytest tests/ -q
```

The agent harness injects scripted responses and forbids the paid client. It tests
contradictory evidence, missing evidence/artifacts, tool errors, fabricated or
uncited references, request/tool/token/image limits and an unreachable goal. It
measures runtime guardrails, **not live-model reasoning quality**. The retrieval
fixture is a small fixed corpus, not a broad semantic-search benchmark.
The commands above regenerate local reports; generated reports are excluded from the public repository.

The original `research` mode without `--agent meta` remains available.

State lives in `.cv-scientist/` (`project.yaml`, `memory/research.db`, plus
`state/ runs/ observations/ experiments/ reports/`) and survives restarts
and manual runs. Integration requires only `project.yaml` plus,
optionally, an adapter mapping your run format to `Experiment`;
trackers sit behind `ExperimentSource` (local implemented; W&B/MLflow/TB stubs).

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
| Models, provenance, evidence refs | `cv_scientist/models.py` |
| SQLite knowledge base + timeline | `cv_scientist/memory.py` |
| Experiment lineage graph | `cv_scientist/graph.py` |
| Workspace `init` | `cv_scientist/workspace.py` |
| Diagnostic skills as APIs | `cv_scientist/diagnostics.py` |
| Compare / Diagnose / Regressions | `compare.py`, `diagnose.py`, `diagnostics.find_regressions` |
| Ingestion + auto run analysis | `ingestor.py`, `analysis.py` |
| Dedup check before proposing | `dedup.py` |
| Ask over structured memory | `query.py` |
| Sessions, status, watch | `sessions.py` |
| Agent (optional LLM layer) / Autonomous engine | `agent.py`, `engine.py` |
| Meta API / Tool-using investigation agent / Read-only tools | `meta.py`, `investigation.py`, `agent_tools.py` |
| Adaptive confidence research / Frozen-input evaluation executor | `confidence.py`, `evaluation.py` |

## Product-architecture test

Workflows A (autonomous), B (human-run + ingest), C (daily diagnosis),
D (research question), E (long-term memory across restarts) are covered in
`tests/test_product.py`:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```
