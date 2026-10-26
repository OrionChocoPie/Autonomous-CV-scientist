"""ResearchEngine: AUTONOMOUS mode only (§§43,62-64). Copilot operations
(compare/diagnose/ask/...) never require this class.

Autonomous loop: retrieve prior knowledge -> diagnose -> propose (dedup-checked)
-> (executor runs experiment) -> ingest+analyze -> memory update.
Starting point is existing project knowledge — never rediscovery from scratch (§63).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .models import Hypothesis, Provenance
from .diagnose import diagnose
from .dedup import check_duplicate, explain_duplicate
from .analysis import RunAnalysisService

@dataclass
class ResearchPlan:
    goal: str = ""; proposals: list = field(default_factory=list)
    skipped_duplicates: list = field(default_factory=list); prior_evidence: dict = field(default_factory=dict)

@dataclass
class ResearchRunResult:
    goal: str = ""; iterations: int = 0; experiments: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list); skipped_duplicates: list = field(default_factory=list)
    analyses: list = field(default_factory=list)

class ResearchEngine:
    """`executor(exp_config, proposal)->run_dir|Experiment` is injected; without one,
    proposals are recorded as planned experiments (no training is faked)."""
    def __init__(self, workspace, agent=None, executor=None):
        self.ws = workspace; self.memory = workspace.memory
        from .agent import ScientistAgent
        self.agent = agent or ScientistAgent()
        self.executor = executor
        self.analysis = RunAnalysisService(self.memory)
    def plan(self, goal: str, budget: int = 4) -> ResearchPlan:
        prior = self.memory.search(goal)  # §63: previous experiments/conclusions/rejected hypotheses
        latest = self.memory.latest_experiment()
        plan = ResearchPlan(goal=goal, prior_evidence={
            "experiments": [e.id for e in prior["experiments"]], "conclusions": [c.id for c in prior["conclusions"]],
            "hypotheses": [h.id for h in prior["hypotheses"]]})
        if latest:
            diag = diagnose(latest, self.memory.working_baseline() if self.memory.working_baseline() and self.memory.working_baseline().id != latest.id else None)
            for h in self.agent.propose_hypotheses(diag)[:budget]:
                cfg = dict(latest.config); cfg["hypothesis"] = h.statement; cfg["goal"] = goal
                dup = check_duplicate(self.memory, cfg)
                # Exact-config duplicates: hypothesis key makes configs differ; check base config too
                base_dup = check_duplicate(self.memory, latest.config)
                if dup and dup.already_tested and dup.experiment_id != latest.id:
                    plan.skipped_duplicates.append(explain_duplicate(dup)); continue
                plan.proposals.append({"hypothesis": h.statement, "config": cfg, "based_on": latest.id})
        return plan
    def run(self, goal: str, budget: int = 4) -> ResearchRunResult:
        plan = self.plan(goal, budget)
        result = ResearchRunResult(goal=goal, skipped_duplicates=plan.skipped_duplicates)
        for prop in plan.proposals[:budget]:
            h = self.memory.add_hypothesis(Hypothesis(statement=prop["hypothesis"], provenance=Provenance.LLM.value))
            result.hypotheses.append(h)
            if self.executor is None: continue  # planned only; no fake training
            exp = self.executor(prop["config"], prop)
            exp_id = exp.id if hasattr(exp, "id") else self._ingest(exp)
            result.experiments.append(exp_id); result.analyses.append(self.analysis.analyze(exp_id)); result.iterations += 1
        return result
    def _ingest(self, run_ref):
        from .ingestor import RunIngestor
        return RunIngestor(self.memory).ingest(str(run_ref)).id
