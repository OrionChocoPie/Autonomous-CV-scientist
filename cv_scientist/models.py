"""Core domain models (§§46-53, 66-68). Stdlib only, LLM-independent."""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field, asdict
from enum import Enum

def _id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"
def now() -> float:
    return time.time()

class Provenance(str, Enum):
    METRIC = "metric"
    DIAGNOSTIC_TOOL = "diagnostic_tool"
    EXPERIMENT = "experiment"
    LLM = "llm_interpretation"
    USER_NOTE = "user_note"
    EXTERNAL_IMPORT = "external_import"

@dataclass
class EvidenceRef:
    """Structured evidence pointer (§68): experiment/observation/comparison/artifact id."""
    kind: str  # experiment|observation|comparison|artifact|conclusion|hypothesis
    id: str
    detail: str = ""
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return EvidenceRef(**d)

@dataclass
class Experiment:
    id: str
    name: str = ""
    parent_id: str | None = None
    parent_checkpoint: str | None = None
    checkpoint_path: str | None = None
    run_dir: str | None = None
    status: str = "completed"  # completed|running|failed
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)  # flat: map50_95, precision, recall...
    slices: dict = field(default_factory=dict)  # slice -> {recall, precision, fp, fn, map}
    per_class: dict = field(default_factory=dict)  # class -> {recall, precision, map, fp, fn}
    code_revision: str | None = None
    dataset_version: str | None = None
    source: str = "local"
    provenance: str = Provenance.EXPERIMENT.value
    created_at: float = field(default_factory=now)
    def primary_metric(self, key: str = "map50_95") -> float | None:
        if key in self.metrics: return float(self.metrics[key])
        for k in ("map", "mAP", "map50"):
            if k in self.metrics: return float(self.metrics[k])
        return None
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d):
        d = dict(d); return Experiment(**{k: d[k] for k in Experiment.__dataclass_fields__ if k in d})

@dataclass
class Observation:
    id: str = field(default_factory=lambda: _id("O"))
    title: str = ""
    description: str = ""
    kind: str = "diagnostic"  # diagnostic|metric|error_cluster|annotation
    provenance: str = Provenance.DIAGNOSTIC_TOOL.value
    evidence: list = field(default_factory=list)  # list[EvidenceRef dict]
    experiment_id: str | None = None
    created_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return Observation(**{k: d[k] for k in Observation.__dataclass_fields__ if k in d})

@dataclass
class Hypothesis:
    id: str = field(default_factory=lambda: _id("H"))
    statement: str = ""
    status: str = "active"  # active|supported|rejected|untested
    provenance: str = Provenance.LLM.value
    evidence: list = field(default_factory=list)
    question_id: str | None = None
    created_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return Hypothesis(**{k: d[k] for k in Hypothesis.__dataclass_fields__ if k in d})

@dataclass
class ResearchQuestion:
    id: str = field(default_factory=lambda: _id("Q"))
    question: str = ""
    status: str = "open"  # open|investigating|answered|blocked|abandoned (§52)
    conclusion: str | None = None
    related_observations: list = field(default_factory=list)
    related_hypotheses: list = field(default_factory=list)
    related_experiments: list = field(default_factory=list)
    created_at: float = field(default_factory=now)
    resolved_at: float | None = None
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return ResearchQuestion(**{k: d[k] for k in ResearchQuestion.__dataclass_fields__ if k in d})

@dataclass
class ResearchConclusion:
    id: str = field(default_factory=lambda: _id("C"))
    statement: str = ""
    confidence: float = 0.5
    status: str = "tentative"  # tentative|supported|strong|contradicted|deprecated (§53)
    supporting_experiments: list = field(default_factory=list)
    supporting_observations: list = field(default_factory=list)
    contradicting_evidence: list = field(default_factory=list)
    provenance: str = Provenance.EXPERIMENT.value
    created_at: float = field(default_factory=now)
    last_updated_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return ResearchConclusion(**{k: d[k] for k in ResearchConclusion.__dataclass_fields__ if k in d})

@dataclass
class RegressionFinding:
    metric: str = ""
    slice: str = "overall"
    previous_value: float = 0.0
    current_value: float = 0.0
    absolute_delta: float = 0.0
    relative_delta: float = 0.0  # fraction, e.g. -0.084
    severity: str = "medium"  # low|medium|high|critical
    evidence: list = field(default_factory=list)
    id: str = field(default_factory=lambda: _id("R"))
    experiment_id: str | None = None
    baseline_id: str | None = None
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return RegressionFinding(**{k: d[k] for k in RegressionFinding.__dataclass_fields__ if k in d})

@dataclass
class DiagnosticFinding:
    title: str = ""
    category: str = ""  # false_negative|false_positive|small_object|low_light|annotation|localization
    severity: str = "medium"
    metric: str = ""
    value: float | None = None
    share: float | None = None  # share of errors, 0..1
    slice: str = ""
    evidence: list = field(default_factory=list)
    id: str = field(default_factory=lambda: _id("D"))
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return DiagnosticFinding(**{k: d[k] for k in DiagnosticFinding.__dataclass_fields__ if k in d})

@dataclass
class Diagnosis:
    experiment_id: str | None = None
    findings: list = field(default_factory=list)  # DiagnosticFinding
    created_at: float = field(default_factory=now)
    id: str = field(default_factory=lambda: _id("DG"))
    def to_dict(self):
        d = asdict(self); return d
    @staticmethod
    def from_dict(d):
        dd = dict(d); dd["findings"] = [DiagnosticFinding.from_dict(f) if isinstance(f, dict) else f for f in dd.get("findings", [])]
        return Diagnosis(**{k: dd[k] for k in Diagnosis.__dataclass_fields__ if k in dd})

@dataclass
class ExperimentComparison:
    a_id: str = ""
    b_id: str = ""
    overall: dict = field(default_factory=dict)  # metric -> {a,b,abs,rel}
    per_class: dict = field(default_factory=dict)
    per_slice: dict = field(default_factory=dict)
    improvements: list = field(default_factory=list)
    regressions: list = field(default_factory=list)  # RegressionFinding dicts
    likely_tradeoff: str = ""
    id: str = field(default_factory=lambda: _id("CMP"))
    created_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return ExperimentComparison(**{k: d[k] for k in ExperimentComparison.__dataclass_fields__ if k in d})

@dataclass
class ResearchSession:
    id: str = field(default_factory=lambda: _id("S"))
    started_at: float = field(default_factory=now)
    finished_at: float | None = None
    status: str = "active"  # active|finished
    start_summary: dict = field(default_factory=dict)
    finish_summary: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return ResearchSession(**{k: d[k] for k in ResearchSession.__dataclass_fields__ if k in d})

@dataclass
class ResearchEvent:
    id: str = field(default_factory=lambda: _id("EV"))
    kind: str = ""  # experiment|observation|hypothesis|conclusion|session|note|regression
    ref_id: str = ""
    description: str = ""
    created_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return ResearchEvent(**{k: d[k] for k in ResearchEvent.__dataclass_fields__ if k in d})

@dataclass
class UserNote:
    id: str = field(default_factory=lambda: _id("N"))
    text: str = ""
    provenance: str = Provenance.USER_NOTE.value  # never mixed with auto observations (§66)
    experiment_id: str | None = None
    created_at: float = field(default_factory=now)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return UserNote(**{k: d[k] for k in UserNote.__dataclass_fields__ if k in d})

@dataclass
class AgentStatement:
    """Agent output carrying evidence IDs (§68)."""
    text: str = ""
    evidence: list = field(default_factory=list)  # EvidenceRef dicts
    confidence: float = 0.5
    def to_dict(self): return asdict(self)
