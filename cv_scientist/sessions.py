"""Research sessions (§51), status (§§58-59), timeline (§65), watch (§56)."""
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from .models import ResearchSession
from .analysis import RunAnalysisService
from .ingestor import RunIngestor, LocalExperimentSource

def _status_dict(ws) -> dict:
    m, metric = ws.memory, ws.primary_metric
    exps = m.list_experiments(); best = m.best_experiment(metric)
    return {"project": ws.name, "total_experiments": len(exps),
            "running": len([e for e in exps if e.status == "running"]),
            "failed": len([e for e in exps if e.status == "failed"]),
            "best": best, "latest": m.latest_experiment(),
            "original_baseline": m.original_baseline(), "working_baseline": m.working_baseline(),
            "open_questions": m.list_questions(status="open"), "active_hypotheses": m.list_hypotheses(status="active"),
            "latest_regressions": m.list_regressions()[-5:]}

@dataclass
class ProjectStatus:
    project: str = ""; best: object = None; latest: object = None
    original_baseline: object = None; working_baseline: object = None
    total_experiments: int = 0; running: int = 0; failed: int = 0
    open_questions: list = field(default_factory=list); active_hypotheses: list = field(default_factory=list)
    latest_regressions: list = field(default_factory=list); primary_metric: str = "map50_95"
    def render(self) -> str:
        def fmt(e): return f"{e.id} ({self.primary_metric}={e.primary_metric(self.primary_metric)})" if e else "—"
        lines = [f"Project: {self.project}", f"Current best: {fmt(self.best)}", f"Latest: {fmt(self.latest)}",
                 f"Original baseline: {fmt(self.original_baseline)}", f"Working baseline: {fmt(self.working_baseline)}"]
        if self.best and self.original_baseline and self.best.primary_metric(self.primary_metric) is not None and self.original_baseline.primary_metric(self.primary_metric) is not None:
            lines.append(f"Change (best vs original): {self.best.primary_metric(self.primary_metric)-self.original_baseline.primary_metric(self.primary_metric):+.3f}")
        lines += [f"Open research questions: {len(self.open_questions)}", f"Active hypotheses: {len(self.active_hypotheses)}",
                  f"Experiments: {self.total_experiments} total, {self.running} running, {self.failed} failed"]
        if self.latest_regressions:
            r = self.latest_regressions[-1]; lines.append(f"Latest regression: {r.slice} {r.metric} {r.relative_delta:+.1%}")
        return "\n".join(lines)

def project_status(ws) -> ProjectStatus:
    d = _status_dict(ws)
    return ProjectStatus(project=d["project"], best=d["best"], latest=d["latest"], original_baseline=d["original_baseline"],
        working_baseline=d["working_baseline"], total_experiments=d["total_experiments"], running=d["running"], failed=d["failed"],
        open_questions=d["open_questions"], active_hypotheses=d["active_hypotheses"], latest_regressions=d["latest_regressions"],
        primary_metric=ws.primary_metric)

class SessionService:
    def __init__(self, workspace): self.ws = workspace
    def start(self) -> ResearchSession:
        st = project_status(self.ws)
        m = self.ws.memory
        prev = m.list_sessions()[-1] if m.list_sessions() else None
        since = prev.finished_at if prev and prev.finished_at else 0
        new_exps = [e for e in m.list_experiments() if e.created_at >= since] if since else m.list_experiments()
        summary = {"new_experiments": [e.id for e in new_exps], "best": st.best.id if st.best else None,
                   "latest": st.latest.id if st.latest else None,
                   "new_regressions": [r.id for r in m.list_regressions() if True][-5:],
                   "open_questions": [q.id for q in st.open_questions], "active_hypotheses": [h.id for h in st.active_hypotheses],
                   "recommended_next": [q.question for q in st.open_questions[:3]]}
        s = ResearchSession(start_summary=summary); m.add_session(s); return s
    def finish(self, session_id: str | None = None) -> ResearchSession:
        m = self.ws.memory
        s = m.get_session(session_id) if session_id else m.active_session()
        if not s: raise ValueError("no active session")
        st = project_status(self.ws)
        during = [e for e in m.list_experiments() if e.created_at >= s.started_at]
        s.finish_summary = {"investigated_experiments": [e.id for e in during],
            "new_observations": [o.id for o in m.list_observations() if o.created_at >= s.started_at],
            "best_checkpoint": st.best.checkpoint_path if st.best else None, "best": st.best.id if st.best else None,
            "open_questions": [q.id for q in st.open_questions],
            "next_recommended": [q.question for q in st.open_questions[:3]]}
        s.finished_at = time.time(); s.status = "finished"; m.update_session(s); return s

class WatchService:
    """Filesystem polling watcher (§56): detect -> ingest -> analyze -> emit event."""
    def __init__(self, workspace, runs_dir: str):
        self.ws = workspace; self.runs_dir = runs_dir; self._seen: set = set()
        self.ingestor = RunIngestor(workspace.memory, LocalExperimentSource(runs_dir))
        self.analysis = RunAnalysisService(workspace.memory); self.events: list = []
    def scan_once(self) -> list:
        ingested = []
        for run in self.ingestor.source.list_runs():
            key = os.path.abspath(run)
            # A run is ready when metrics.json exists
            if key in self._seen or not os.path.exists(os.path.join(run, "metrics.json")): continue
            self._seen.add(key)
            try:
                exp = self.ingestor.ingest(run); self.analysis.analyze(exp.id); ingested.append(exp)
                self.events.append({"type": "run_ingested", "experiment_id": exp.id})
            except Exception as e:
                self.events.append({"type": "ingest_failed", "run": run, "error": str(e)})
        return ingested
    def watch(self, interval: float = 2.0, max_cycles: int | None = None):
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            self.scan_once(); cycles += 1
            if max_cycles is not None and cycles >= max_cycles: break
            time.sleep(interval)
        return self.events
