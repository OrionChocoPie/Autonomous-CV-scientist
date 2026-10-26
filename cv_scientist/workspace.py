"""Project workspace (§44): .cv-scientist/ persistent per-repo state."""
from __future__ import annotations
import os
from .memory import ResearchMemory

SUBDIRS = ["state", "memory", "runs", "observations", "experiments", "reports"]

def _write_yaml(path, data: dict):
    with open(path, "w") as f:
        for k, v in data.items(): f.write(f"{k}: {v}\n")
def _read_yaml(path) -> dict:
    out = {}
    if not os.path.exists(path): return out
    for line in open(path):
        line = line.strip()
        if line and ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1); out[k.strip()] = v.strip()
    return out

class ProjectWorkspace:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.dir = os.path.join(self.root, ".cv-scientist")
        self.config_path = os.path.join(self.dir, "project.yaml")
        self.config = _read_yaml(self.config_path)
        self._memory = None
    @property
    def name(self): return self.config.get("name", os.path.basename(self.root))
    @property
    def exists(self): return os.path.isdir(self.dir)
    @classmethod
    def init(cls, root: str, name: str | None = None, primary_metric: str = "map50_95") -> "ProjectWorkspace":
        ws = cls(root)
        for d in SUBDIRS: os.makedirs(os.path.join(ws.dir, d), exist_ok=True)
        if not os.path.exists(ws.config_path):
            _write_yaml(ws.config_path, {"name": name or os.path.basename(ws.root),
                                         "primary_metric": primary_metric, "version": "1"})
            ws.config = _read_yaml(ws.config_path)
        ws.memory  # create DB
        return ws
    @classmethod
    def discover(cls, start: str = ".") -> "ProjectWorkspace | None":
        cur = os.path.abspath(start)
        while True:
            if os.path.isdir(os.path.join(cur, ".cv-scientist")): return cls(cur)
            parent = os.path.dirname(cur)
            if parent == cur: return None
            cur = parent
    @property
    def memory(self) -> ResearchMemory:
        if self._memory is None:
            os.makedirs(os.path.join(self.dir, "memory"), exist_ok=True)
            self._memory = ResearchMemory(os.path.join(self.dir, "memory", "research.db"))
        return self._memory
    @property
    def primary_metric(self): return self.config.get("primary_metric", "map50_95")
