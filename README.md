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

## Product-architecture test (§70)

Workflows A (autonomous), B (human-run + ingest), C (daily diagnosis),
D (research question), E (long-term memory across restarts) are covered in
`tests/test_product.py`:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```
