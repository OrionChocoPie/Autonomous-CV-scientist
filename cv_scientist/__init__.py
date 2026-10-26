"""CV Scientist: research infrastructure + CV diagnostics + experiment memory + scientist agent (§71).

Core capabilities (compare/diagnose/regressions/ask/ingest/status) work without
the LLM and without ResearchEngine. Two modes (§43): COPILOT (human drives,
services assist) and AUTONOMOUS (ResearchEngine coordinates the same services).
"""
from .models import (Experiment, Observation, Hypothesis, ResearchQuestion, ResearchConclusion,
                     RegressionFinding, DiagnosticFinding, Diagnosis, ExperimentComparison,
                     ResearchSession, ResearchEvent, UserNote, EvidenceRef, AgentStatement, Provenance)
from .memory import ResearchMemory
from .workspace import ProjectWorkspace
from .graph import ExperimentGraph
from .ingestor import RunIngestor, ExperimentImporter, ExperimentSource, LocalExperimentSource
from .analysis import RunAnalysisService, RunAnalysis
from .query import ProjectQueryService
from . import diagnostics
from .compare import compare_runs
from .diagnose import diagnose
from .agent import ScientistAgent
from .engine import ResearchEngine

__all__ = ["Experiment", "Observation", "Hypothesis", "ResearchQuestion", "ResearchConclusion",
           "RegressionFinding", "DiagnosticFinding", "Diagnosis", "ExperimentComparison",
           "ResearchSession", "ResearchEvent", "UserNote", "EvidenceRef", "AgentStatement", "Provenance",
           "ResearchMemory", "ProjectWorkspace", "ExperimentGraph", "RunIngestor", "ExperimentImporter",
           "ExperimentSource", "LocalExperimentSource", "RunAnalysisService", "RunAnalysis",
           "ProjectQueryService", "diagnostics", "compare_runs", "diagnose", "ScientistAgent", "ResearchEngine"]
