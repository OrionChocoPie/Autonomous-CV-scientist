"""ResearchMemory: SQLite project knowledge base (§54). No vector DB needed initially.

Clean retrieval interface so semantic retrieval can be added later behind `search()`.
"""
from __future__ import annotations
import json, os, sqlite3
from .models import *

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS hypotheses(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS conclusions(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS regressions(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS comparisons(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY, data TEXT, created_at REAL);
"""

def _j(o):
    return json.dumps(o.to_dict() if hasattr(o, "to_dict") else o)

class ResearchMemory:
    """Structured KB: Questions, Observations, Hypotheses, Experiments,
    Results (inside Experiment), Comparisons, Conclusions, Regressions, Sessions (§54)."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.executescript(_SCHEMA); self._db.commit()

    def close(self): self._db.close()
    # --- generic helpers ---
    def _put(self, table, obj):
        self._db.execute(f"INSERT OR REPLACE INTO {table}(id,data,created_at) VALUES(?,?,?)",
                         (obj.id, _j(obj), getattr(obj, "created_at", now())))
        self._db.commit()
    def _get(self, table, cls, oid):
        r = self._db.execute(f"SELECT data FROM {table} WHERE id=?", (oid,)).fetchone()
        return cls.from_dict(json.loads(r[0])) if r else None
    def _all(self, table, cls):
        rows = self._db.execute(f"SELECT data FROM {table} ORDER BY created_at").fetchall()
        return [cls.from_dict(json.loads(r[0])) for r in rows]
    def _event(self, kind, ref_id, desc):
        ev = ResearchEvent(kind=kind, ref_id=ref_id, description=desc)
        self._put("events", ev); return ev

    # --- meta / baselines (§58) ---
    def set_meta(self, k, v):
        self._db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v))); self._db.commit()
    def get_meta(self, k, default=None):
        r = self._db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return r[0] if r else default

    # --- experiments ---
    def add_experiment(self, exp: Experiment) -> Experiment:
        if not self._get("experiments", Experiment, exp.id):
            self._event("experiment", exp.id, f"{exp.id} {exp.name}".strip())
        self._put("experiments", exp)
        if self.get_meta("original_baseline") is None:
            self.set_meta("original_baseline", exp.id)
        return exp
    def get_experiment(self, eid): return self._get("experiments", Experiment, eid)
    def list_experiments(self): return self._all("experiments", Experiment)
    def latest_experiment(self):
        exps = self.list_experiments(); return exps[-1] if exps else None
    def best_experiment(self, metric="map50_95"):
        exps = [e for e in self.list_experiments() if e.primary_metric(metric) is not None]
        return max(exps, key=lambda e: e.primary_metric(metric)) if exps else None
    def original_baseline(self):
        bid = self.get_meta("original_baseline"); return self.get_experiment(bid) if bid else None
    def working_baseline(self):
        bid = self.get_meta("working_baseline") or self.get_meta("original_baseline")
        return self.get_experiment(bid) if bid else None
    def set_working_baseline(self, eid): self.set_meta("working_baseline", eid)
    def set_original_baseline(self, eid): self.set_meta("original_baseline", eid)

    # --- other entities ---
    def add_observation(self, o: Observation):
        self._put("observations", o); self._event("observation", o.id, o.title or o.description); return o
    def list_observations(self): return self._all("observations", Observation)
    def get_observation(self, oid): return self._get("observations", Observation, oid)
    def add_hypothesis(self, h: Hypothesis):
        self._put("hypotheses", h); self._event("hypothesis", h.id, h.statement); return h
    def list_hypotheses(self, status=None):
        hs = self._all("hypotheses", Hypothesis)
        return [h for h in hs if status is None or h.status == status]
    def add_question(self, q: ResearchQuestion):
        self._put("questions", q); self._event("question", q.id, q.question); return q
    def get_question(self, qid): return self._get("questions", ResearchQuestion, qid)
    def list_questions(self, status=None):
        qs = self._all("questions", ResearchQuestion)
        return [q for q in qs if status is None or q.status == status]
    def update_question(self, q): self._put("questions", q); return q
    def add_conclusion(self, c: ResearchConclusion):
        self._put("conclusions", c); self._event("conclusion", c.id, c.statement); return c
    def list_conclusions(self): return self._all("conclusions", ResearchConclusion)
    def add_regression(self, r: RegressionFinding):
        self._put("regressions", r); self._event("regression", r.id, f"{r.slice} {r.metric} {r.relative_delta:+.1%}"); return r
    def list_regressions(self, experiment_id=None):
        rs = self._all("regressions", RegressionFinding)
        return [r for r in rs if experiment_id is None or r.experiment_id == experiment_id]
    def add_comparison(self, c: ExperimentComparison):
        self._put("comparisons", c); self._event("comparison", c.id, f"{c.a_id} -> {c.b_id}"); return c
    def list_comparisons(self): return self._all("comparisons", ExperimentComparison)
    def add_session(self, s: ResearchSession):
        self._put("sessions", s); self._event("session", s.id, f"session {s.status}"); return s
    def update_session(self, s): self._put("sessions", s); return s
    def get_session(self, sid): return self._get("sessions", ResearchSession, sid)
    def list_sessions(self): return self._all("sessions", ResearchSession)
    def active_session(self):
        ss = [s for s in self.list_sessions() if s.status == "active"]
        return ss[-1] if ss else None
    def add_user_note(self, text: str, experiment_id=None) -> UserNote:
        n = UserNote(text=text, experiment_id=experiment_id)
        self._put("notes", n); self._event("note", n.id, text[:120]); return n
    def list_notes(self): return self._all("notes", UserNote)
    def timeline(self): return self._all("events", ResearchEvent)

    # --- retrieval (§§54-55): keyword structured search, semantic later behind this API ---
    def search(self, query: str, limit: int = 20) -> dict:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        def score(text): return sum(1 for t in terms if t in (text or "").lower())
        def ranked(items, textfn):
            scored = [(score(textfn(i)), i) for i in items]
            positive = [x for x in scored if x[0] > 0]
            chosen = positive if positive else [(0, i) for i in items]
            return [i for _, i in sorted(chosen, key=lambda x: -x[0])[:limit]]
        return {
            "experiments": ranked(self.list_experiments(), lambda e: f"{e.id} {e.name} {json.dumps(e.config)} {json.dumps(e.slices)} {json.dumps(e.per_class)}"),
            "observations": ranked(self.list_observations(), lambda o: f"{o.title} {o.description} {o.kind}"),
            "hypotheses": ranked(self.list_hypotheses(), lambda h: h.statement),
            "conclusions": ranked(self.list_conclusions(), lambda c: c.statement),
            "questions": ranked(self.list_questions(), lambda q: q.question),
            "notes": ranked(self.list_notes(), lambda n: n.text),
        }
