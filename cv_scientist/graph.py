"""ExperimentGraph: lineage without a graph DB (§46). SQLite-backed via ResearchMemory."""
from __future__ import annotations
from .memory import ResearchMemory

class ExperimentGraph:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
    def children(self, eid):
        return [e for e in self.memory.list_experiments() if e.parent_id == eid]
    def descendants(self, eid) -> list:
        out, stack = [], list(self.children(eid))
        while stack:
            cur = stack.pop(); out.append(cur); stack.extend(self.children(cur.id))
        return out
    def ancestors(self, eid) -> list:
        out, cur = [], self.memory.get_experiment(eid)
        while cur and cur.parent_id:
            cur = self.memory.get_experiment(cur.parent_id)
            if cur: out.append(cur)
        return out
    def lineage(self, eid) -> list:
        return list(reversed(self.ancestors(eid))) + ([self.memory.get_experiment(eid)] if self.memory.get_experiment(eid) else [])
    def diff(self, a_id, b_id) -> dict:
        """What changed between two experiments (§46)."""
        a, b = self.memory.get_experiment(a_id), self.memory.get_experiment(b_id)
        if not a or not b: raise KeyError(f"unknown experiment: {a_id if not a else b_id}")
        keys = set(a.config) | set(b.config)
        config_diff = {k: {"from": a.config.get(k), "to": b.config.get(k)} for k in keys if a.config.get(k) != b.config.get(k)}
        mkeys = set(a.metrics) | set(b.metrics)
        metric_diff = {k: {"from": a.metrics.get(k), "to": b.metrics.get(k),
                           "delta": (b.metrics[k] - a.metrics[k]) if k in a.metrics and k in b.metrics else None} for k in mkeys}
        return {"a": a_id, "b": b_id, "config_diff": config_diff, "metric_diff": metric_diff,
                "code_revision": {"from": a.code_revision, "to": b.code_revision},
                "dataset_version": {"from": a.dataset_version, "to": b.dataset_version},
                "parent": {"a": a.parent_id, "b": b.parent_id}}
    def first_appearance(self, predicate) -> object | None:
        """When did a failure first appear? predicate(Experiment)->bool, chronological."""
        for e in self.memory.list_experiments():
            if predicate(e): return e
        return None
    def introduced_by(self, slice_name: str, metric: str = "recall", drop: float = 0.05):
        """Which experiment introduced a slice regression vs its parent?"""
        for e in self.memory.list_experiments():
            if not e.parent_id: continue
            p = self.memory.get_experiment(e.parent_id)
            if not p: continue
            pv = p.slices.get(slice_name, {}).get(metric); cv = e.slices.get(slice_name, {}).get(metric)
            if pv is not None and cv is not None and (pv - cv) >= drop: return e
        return None
