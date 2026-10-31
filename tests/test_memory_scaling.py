import json
import sqlite3

from cv_scientist.benchmark import baseline_benchmark, memory_benchmark
from cv_scientist.memory import ResearchMemory
from cv_scientist.models import Experiment
from cv_scientist.sessions import project_status
from cv_scientist.workspace import ProjectWorkspace


def test_status_best_and_latest_preserve_history_behavior(tmp_path):
    ws = ProjectWorkspace.init(str(tmp_path))
    assert project_status(ws).total_experiments == 0
    records = [Experiment(id="old-best", metrics={"map50_95": 0.9}, created_at=1),
               Experiment(id="same-score", metrics={"map50_95": 0.9}, created_at=2, status="failed"),
               Experiment(id="latest", metrics={"map": 0.8}, created_at=3, status="running")]
    for record in records: ws.memory.add_experiment(record)
    status = project_status(ws)
    assert status.best.id == "old-best" and status.latest.id == "latest"
    assert (status.total_experiments, status.running, status.failed) == (3, 1, 1)
    assert ws.memory.best_experiment("custom").id == "latest"


def test_existing_database_gets_retrieval_backfill_once(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE experiments(id TEXT PRIMARY KEY,data TEXT,created_at REAL)")
        exp = Experiment(id="E1", name="zygomatic")
        connection.execute("INSERT INTO experiments VALUES(?,?,?)", (exp.id, json.dumps(exp.to_dict()), exp.created_at))
    memory = ResearchMemory(str(path))
    assert memory.search("zygomatic")["experiments"][0].id == "E1"
    memory.close()
    memory = ResearchMemory(str(path))
    assert memory._db.execute("SELECT count(*) FROM search_documents").fetchone()[0] == 1
    memory.close()


def test_small_benchmark_is_reproducible_and_checks_behavior():
    first = memory_benchmark(20, 1)
    second = memory_benchmark(20, 1)
    assert first["behavior"] == second["behavior"]
    assert set(first["operations"]) == {"search", "status", "compare"}
    assert all(result["median_ms"] >= 0 for result in first["operations"].values())


def test_baseline_snapshot_uses_same_fixture_without_changing_checkout():
    baseline = baseline_benchmark("HEAD", 20, 1)
    current = memory_benchmark(20, 1)
    assert baseline["behavior"] == current["behavior"]
    assert baseline["source_revision"]
