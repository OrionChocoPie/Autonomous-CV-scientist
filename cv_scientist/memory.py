"""ResearchMemory: SQLite project knowledge base (§54). No vector DB needed initially.

Clean retrieval interface so semantic retrieval can be added later behind `search()`.
"""
from __future__ import annotations
from contextlib import contextmanager
import json, os, sqlite3
from .models import *
from .retrieval import terms

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
CREATE INDEX IF NOT EXISTS experiments_created_at ON experiments(created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS search_documents USING fts5(kind UNINDEXED, ref_id UNINDEXED, identity, title, body, tokenize='unicode61');
"""

_SEARCH_TYPES = {"experiments": Experiment, "observations": Observation, "hypotheses": Hypothesis,
                 "conclusions": ResearchConclusion, "questions": ResearchQuestion, "notes": UserNote,
                 "regressions": RegressionFinding}

def _j(o):
    return json.dumps(o.to_dict() if hasattr(o, "to_dict") else o)

class ResearchMemory:
    """Structured KB: Questions, Observations, Hypotheses, Experiments,
    Results (inside Experiment), Comparisons, Conclusions, Regressions, Sessions (§54)."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._transaction_depth = 0
        self._db.executescript(_SCHEMA); self._db.commit()
        if self.get_meta("search_index_version") != "1":
            with self.transaction():
                self._db.execute("DELETE FROM search_documents")
                for table in _SEARCH_TYPES:
                    for raw, in self._db.execute(f"SELECT data FROM {table} ORDER BY created_at"):
                        self._index_record(table, json.loads(raw))
                self.set_meta("search_index_version", "1")

    def close(self): self._db.close()

    @contextmanager
    def transaction(self):
        depth = self._transaction_depth
        savepoint = f"memory_{depth}"
        self._db.execute(f"SAVEPOINT {savepoint}" if depth else "BEGIN IMMEDIATE")
        self._transaction_depth += 1
        try:
            yield self
            self._db.execute(f"RELEASE {savepoint}") if depth else self._db.commit()
        except BaseException:
            if depth:
                self._db.execute(f"ROLLBACK TO {savepoint}")
                self._db.execute(f"RELEASE {savepoint}")
            else: self._db.rollback()
            raise
        finally: self._transaction_depth -= 1

    def _commit(self):
        if not self._transaction_depth: self._db.commit()
    # --- generic helpers ---
    def _put(self, table, obj):
        payload = _j(obj)
        self._db.execute(f"INSERT OR REPLACE INTO {table}(id,data,created_at) VALUES(?,?,?)",
                         (obj.id, payload, getattr(obj, "created_at", now())))
        if table in _SEARCH_TYPES: self._index_record(table, json.loads(payload))
        self._commit()
    def _index_record(self, table, data):
        identity = str(data["id"])
        title = " ".join(str(data.get(key, "")) for key in ("name", "title", "statement", "question", "text", "status"))
        body = json.dumps(data, ensure_ascii=False)
        self._db.execute("DELETE FROM search_documents WHERE kind=? AND ref_id=?", (table, identity))
        self._db.execute("INSERT INTO search_documents(kind,ref_id,identity,title,body) VALUES(?,?,?,?,?)",
                         (table, identity, identity, " ".join(terms(title)), " ".join(terms(body + " " + table))))
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
        self._db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v))); self._commit()
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
        row = self._db.execute("SELECT data FROM experiments ORDER BY created_at DESC,rowid DESC LIMIT 1").fetchone()
        return Experiment.from_dict(json.loads(row[0])) if row else None
    def best_experiment(self, metric="map50_95"):
        paths = ["$.metrics." + json.dumps(key) for key in (metric, "map", "mAP", "map50")]
        row = self._db.execute("""SELECT data FROM experiments
            WHERE coalesce(json_extract(data,?),json_extract(data,?),json_extract(data,?),json_extract(data,?)) IS NOT NULL
            ORDER BY CAST(coalesce(json_extract(data,?),json_extract(data,?),json_extract(data,?),json_extract(data,?)) AS REAL) DESC,
                created_at ASC,rowid ASC LIMIT 1""", (*paths, *paths)).fetchone()
        return Experiment.from_dict(json.loads(row[0])) if row else None
    def experiment_summary(self, metric="map50_95"):
        total, running, failed = self._db.execute("""SELECT count(*),
            coalesce(sum(json_extract(data,'$.status')='running'),0),
            coalesce(sum(json_extract(data,'$.status')='failed'),0) FROM experiments""").fetchone()
        return {"total_experiments": total, "running": running, "failed": failed,
                "best": self.best_experiment(metric), "latest": self.latest_experiment()}
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
    def get_hypothesis(self, hid): return self._get("hypotheses", Hypothesis, hid)
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
    def get_conclusion(self, cid): return self._get("conclusions", ResearchConclusion, cid)
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
        if not isinstance(query, str) or len(query) > 2048: raise ValueError("Search query must be a string of at most 2048 characters")
        if type(limit) is not int or not 1 <= limit <= 1000: raise ValueError("Search limit must be between 1 and 1000")
        tokens = terms(query)
        result = {table: [] for table in _SEARCH_TYPES}
        if not tokens: return result
        expression = " OR ".join('"' + token + '"' for token in tokens)
        exact = ",".join("?" for _ in tokens)
        for table, cls in _SEARCH_TYPES.items():
            rows = self._db.execute(f"""SELECT records.data FROM search_documents
                JOIN {table} AS records ON records.id=search_documents.ref_id
                WHERE search_documents MATCH ? AND search_documents.kind=?
                ORDER BY CASE WHEN lower(search_documents.ref_id) IN ({exact}) THEN 0 ELSE 1 END,
                    bm25(search_documents,0,0,12,4,1), search_documents.rowid LIMIT ?""",
                (expression, table, *tokens, limit))
            result[table] = [cls.from_dict(json.loads(raw)) for raw, in rows]
        return result
