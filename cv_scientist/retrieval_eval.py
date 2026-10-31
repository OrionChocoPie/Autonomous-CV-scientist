"""Fixed, offline retrieval evidence fixtures and before/after scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .memory import ResearchMemory
from .models import Experiment, Hypothesis, Observation, RegressionFinding, ResearchConclusion, UserNote


CASES = [
    ("exact_short_id", "E1", "experiments", ["E1"]),
    ("id_boundary", "E10", "experiments", ["E10"]),
    ("compare_ids", "compare E1 E2", "experiments", ["E1", "E2"]),
    ("regression", "E2 recall regression", "regressions", ["R-recall"]),
    ("rejected_hypothesis", "rejected oversampling small objects", "hypotheses", ["H-rejected"]),
    ("conclusion", "oversampling weak improvement", "conclusions", ["C-oversampling"]),
    ("observation", "packaging false positives", "observations", ["O-packaging"]),
    ("russian_recall", "E2 регрессия полноты", "regressions", ["R-recall"]),
    ("russian_fn", "пропуски мелких объектов", "observations", ["O-small"]),
    ("note", "camera lens contamination", "notes", ["N-camera"]),
    ("no_match", "unicorns astronomy zebras", None, []),
    ("empty", "", None, []),
]


def seed_memory(memory):
    for identity, name, recall in (("B0", "baseline food detector", 0.9), ("E10", "unrelated camera calibration", 0.9),
                                   ("E1", "packaging false positives", 0.91), ("E2", "small objects recall regression", 0.75)):
        memory.add_experiment(Experiment(id=identity, name=name, metrics={"recall": recall}, created_at=float(len(memory.list_experiments()))))
    memory.add_observation(Observation(id="O-packaging", title="Packaging false positives", description="Hard negatives need review", experiment_id="E1"))
    memory.add_observation(Observation(id="O-small", title="Small objects false negatives", description="Missed detections after resolution reduction", experiment_id="E2"))
    memory.add_hypothesis(Hypothesis(id="H-other", statement="Increase confidence for packaging", status="active"))
    memory.add_hypothesis(Hypothesis(id="H-rejected", statement="Oversampling small objects", status="rejected"))
    memory.add_conclusion(ResearchConclusion(id="C-oversampling", statement="Oversampling small objects gave weak improvement", status="supported"))
    memory.add_regression(RegressionFinding(id="R-recall", experiment_id="E2", baseline_id="B0", metric="recall", slice="small", severity="high",
        previous_value=0.9, current_value=0.75, absolute_delta=-0.15, relative_delta=-1/6))
    memory._put("notes", UserNote(id="N-camera", text="Camera lens contamination"))


def score_retrieval(memory, cases=CASES):
    rows = []
    for name, query, group, expected in cases:
        result = memory.search(query, limit=3)
        ids = {key: [obj.id for obj in objects] for key, objects in result.items()}
        actual = ids.get(group, []) if group else [item for values in ids.values() for item in values]
        hits = sum(item in actual for item in expected)
        recall = hits / len(expected) if expected else float(not actual)
        reciprocal = max((1 / (actual.index(item) + 1) for item in expected if item in actual), default=0.0) if expected else float(not actual)
        rows.append({"case": name, "query": query, "group": group, "expected": expected, "actual": ids,
                     "recall_at_3": recall, "reciprocal_rank": reciprocal, "passed": recall == 1})
    return {"cases": rows, "passed": sum(row["passed"] for row in rows), "total": len(rows),
            "mean_recall_at_3": sum(row["recall_at_3"] for row in rows) / len(rows),
            "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in rows) / len(rows),
            "scope": "Fixed offline fixture, not a general semantic-search evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="cv-retrieval-eval-") as folder:
        memory = ResearchMemory(str(Path(folder) / "memory.db"))
        seed_memory(memory)
        report = score_retrieval(memory)
        memory.close()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Retrieval: {report['passed']}/{report['total']}; recall@3={report['mean_recall_at_3']:.3f}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
