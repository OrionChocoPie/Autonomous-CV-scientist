"""ScientistAgent: interpretation layer (§§48,55,68). Optional — core works without it.
This default agent is deterministic (no LLM call); swap `llm_fn` for a real LLM.
Every statement carries EvidenceRefs."""
from __future__ import annotations
from .models import AgentStatement, EvidenceRef, Diagnosis

class ScientistAgent:
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn  # callable(prompt:str)->str, optional
    def _llm_or(self, prompt, fallback):
        if self.llm_fn:
            try: return self.llm_fn(prompt)
            except Exception: return fallback
        return fallback
    def prioritize_diagnosis(self, diagnosis: Diagnosis) -> AgentStatement:
        top = diagnosis.findings[:3]
        text = "; ".join(f"{f.title} (severity {f.severity})" for f in top) or "No significant findings."
        ev = [e for f in top for e in f.evidence]
        return AgentStatement(text=f"Top issues: {text}", evidence=ev, confidence=0.8 if top else 0.3)
    def interpret_comparison(self, comparison) -> AgentStatement:
        text = comparison.likely_tradeoff or f"{comparison.a_id} -> {comparison.b_id}: improvements={comparison.improvements}"
        ev = [EvidenceRef("comparison", comparison.id).to_dict(), EvidenceRef("experiment", comparison.b_id).to_dict()]
        return AgentStatement(text=text, evidence=ev, confidence=0.75)
    def answer_question(self, question: str, context: str, evidence: dict) -> AgentStatement:
        fallback = f"Based on project memory:\n{context}" if context else "No relevant evidence in ResearchMemory."
        text = self._llm_or(f"Question: {question}\nEvidence:\n{context}\nAnswer citing evidence IDs.", fallback)
        refs = ([EvidenceRef("experiment", e.id).to_dict() for e in evidence.get("experiments", [])[:4]]
                + [EvidenceRef("conclusion", c.id).to_dict() for c in evidence.get("conclusions", [])[:3]])
        return AgentStatement(text=text, evidence=refs, confidence=0.7 if refs else 0.2)
    def propose_hypotheses(self, diagnosis: Diagnosis) -> list:
        from .models import Hypothesis, Provenance
        out = []
        for f in diagnosis.findings[:3]:
            out.append(Hypothesis(statement=f"{f.category} in {f.slice} is a primary failure driver (observed {f.metric}={f.value})",
                                  provenance=Provenance.LLM.value, evidence=f.evidence))
        return out
