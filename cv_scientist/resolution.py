"""Bounded resolution research sharing confidence research's journal and recovery."""
from __future__ import annotations

from dataclasses import dataclass
import json

from .confidence import ConfidenceProposal, ConfidenceResearch, PLANNER_SYSTEM
from .evaluation import ResolutionEvaluator, imgsz_value


@dataclass(frozen=True)
class ResolutionProposal:
    action: str
    parent_id: str
    imgsz: int | None
    hypothesis: str
    expected_effect: str
    reason: str

    @classmethod
    def parse(cls, message, baseline_id):
        if message.get("tool_calls"): raise ValueError("Planner must return a proposal, not a tool call")
        try: data = json.loads(message["content"])
        except (ValueError, TypeError, KeyError): raise ValueError("Planner returned invalid proposal JSON") from None
        if not isinstance(data, dict) or set(data) != {"action", "parent_id", "imgsz", "hypothesis", "expected_effect", "reason"}:
            raise ValueError("Proposal does not match the allowed schema")
        value = None if data["action"] == "stop" else imgsz_value(data["imgsz"])
        if data["action"] == "stop" and data["imgsz"] is not None: raise ValueError("Stop needs null imgsz")
        # Reuse the existing strict validation of parent, action and explanatory fields.
        shared = {key: item for key, item in data.items() if key != "imgsz"}
        shared["confidence"] = None if data["action"] == "stop" else 0.5
        ConfidenceProposal.parse({"content": json.dumps(shared)}, baseline_id)
        return cls(**{**data, "imgsz": value})


class ResolutionResearch(ConfidenceResearch):
    PARAMETER = "imgsz"
    PROPOSAL = ResolutionProposal
    SYSTEM = PLANNER_SYSTEM.replace(
        "Only confidence may change. Parent, weights, imgsz, seed and validation are fixed by code.",
        "Only imgsz may change: an integer multiple of 32 between 32 and 4096. Parent, weights, confidence, seed and validation are fixed by code."
    ).replace("threshold", "resolution")
    validate_value = staticmethod(imgsz_value)

    def __init__(self, workspace, client, evaluator_factory=ResolutionEvaluator):
        super().__init__(workspace, client, evaluator_factory)
