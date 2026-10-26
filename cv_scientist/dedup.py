"""Experiment similarity / deduplication (§64). Structured metadata comparison first."""
from __future__ import annotations
from dataclasses import dataclass, field
from .models import Experiment

@dataclass
class SimilarityResult:
    experiment_id: str = ""
    score: float = 0.0  # 0..1 config similarity
    differing_keys: list = field(default_factory=list)
    matching_keys: list = field(default_factory=list)
    already_tested: bool = False
    result_summary: str = ""

_IGNORE = {"name", "id", "experiment_id", "hypothesis", "goal"}
def config_similarity(a: dict, b: dict) -> SimilarityResult:
    keys = (set(a) | set(b)) - _IGNORE
    if not keys: return SimilarityResult(score=1.0, already_tested=True)
    matching = [k for k in keys if a.get(k) == b.get(k)]
    differing = [k for k in keys if a.get(k) != b.get(k)]
    return SimilarityResult(score=len(matching) / len(keys), matching_keys=matching, differing_keys=differing)

def check_duplicate(memory, proposed_config: dict, threshold: float = 0.85) -> SimilarityResult | None:
    """Return the closest prior experiment if the proposal was already (nearly) tried."""
    best = None
    for exp in memory.list_experiments():
        r = config_similarity(proposed_config, exp.config)
        r.experiment_id = exp.id
        m = exp.primary_metric()
        r.result_summary = f"{exp.id}: primary={m}" + (f", recall={exp.metrics['recall']:+.1%}" if "recall" in exp.metrics else "") if m is not None else exp.id
        r.already_tested = r.score >= threshold
        if best is None or r.score > best.score: best = r
    return best if best and best.score >= threshold else None

def explain_duplicate(result: SimilarityResult) -> str:
    return (f"{result.experiment_id} already tested a near-identical config "
            f"(similarity {result.score:.0%}; differing: {result.differing_keys or 'none'}). "
            f"Result: {result.result_summary}. Avoid repeating it or explicitly justify why this rerun differs.")
