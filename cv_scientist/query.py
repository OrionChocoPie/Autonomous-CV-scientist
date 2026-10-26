"""Ask mode (§55): ProjectQueryService. Retrieves structured evidence, builds
compact context, answers with evidence references. Works without LLM (deterministic
synthesis); if a ScientistAgent is supplied it interprets the same evidence."""
from __future__ import annotations
from .models import AgentStatement, EvidenceRef

class ProjectQueryService:
    def __init__(self, memory, agent=None):
        self.memory = memory; self.agent = agent
    def retrieve(self, question: str) -> dict:
        return self.memory.search(question)
    def build_context(self, evidence: dict) -> str:
        lines = []
        for c in evidence["conclusions"][:5]: lines.append(f"Conclusion {c.id}: {c.statement} (confidence {c.confidence})")
        for e in evidence["experiments"][:8]:
            lines.append(f"Experiment {e.id}: {e.name} metrics={e.metrics} config={e.config}")
        for o in evidence["observations"][:8]: lines.append(f"Observation {o.id}: {o.title} — {o.description}")
        for h in evidence["hypotheses"][:5]: lines.append(f"Hypothesis {h.id} [{h.status}]: {h.statement}")
        for n in evidence["notes"][:3]: lines.append(f"UserNote {n.id}: {n.text}")
        return "\n".join(lines)
    def ask(self, question: str) -> AgentStatement:
        evidence = self.retrieve(question)
        context = self.build_context(evidence)
        if self.agent is not None:
            return self.agent.answer_question(question, context, evidence)
        refs = ([EvidenceRef("conclusion", c.id).to_dict() for c in evidence["conclusions"][:3]]
                + [EvidenceRef("experiment", e.id).to_dict() for e in evidence["experiments"][:4]]
                + [EvidenceRef("observation", o.id).to_dict() for o in evidence["observations"][:4]])
        if not context:
            text = "No relevant project evidence found in ResearchMemory for this question."
        else:
            text = f"Based on project memory:\n{context}"
        return AgentStatement(text=text, evidence=refs, confidence=0.7 if refs else 0.2)
