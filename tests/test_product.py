import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cv_scientist import *
from cv_scientist.dedup import check_duplicate
from cv_scientist.sessions import SessionService, WatchService, project_status
from cv_scientist.cli import main

def make_run(root, eid, metrics, config=None, slices=None, per_class=None, parent=None):
    d = os.path.join(root, eid); os.makedirs(d, exist_ok=True)
    cfg = {"name": eid, **(config or {})}
    if parent: cfg["parent"] = parent
    with open(os.path.join(d, "config.json"), "w") as f: json.dump(cfg, f)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"id": eid, "metrics": metrics, "slices": slices or {}, "per_class": per_class or {}}, f)
    with open(os.path.join(d, "ckpt.pt"), "w") as f: f.write("x")
    return d

BASE_SLICES = {"small": {"recall": 0.60, "precision": 0.70, "fp": 10, "fn": 40}, "overall": {"recall": 0.80, "precision": 0.82, "fp": 50, "fn": 50}}
BASE_CLASS = {"salad": {"recall": 0.80, "precision": 0.85, "fp": 5, "fn": 10}, "packaging": {"recall": 0.9, "precision": 0.60, "fp": 40, "fn": 5}}

def seed(tmp):
    ws = ProjectWorkspace.init(tmp, name="food-detector")
    runs = os.path.join(tmp, "runs"); os.makedirs(runs, exist_ok=True)
    make_run(runs, "B0", {"map50_95": 0.612, "recall": 0.80, "precision": 0.82}, {"lr": 0.01, "crop_p": 0.4}, BASE_SLICES, BASE_CLASS)
    RunIngestor(ws.memory, LocalExperimentSource(runs)).ingest("B0")
    RunAnalysisService(ws.memory).analyze("B0")
    return ws, runs

def test_workflow_b_human_run_ingest_compare_regress(tmp_path):
    ws, runs = seed(str(tmp_path))
    make_run(runs, "E001", {"map50_95": 0.617, "recall": 0.79, "precision": 0.85}, {"lr": 0.01, "crop_p": 0.4, "hard_neg": 0.3},
             {"small": {"recall": 0.51, "precision": 0.72, "fp": 9, "fn": 55}, "overall": {"recall": 0.79, "precision": 0.85, "fp": 35, "fn": 55}},
             {"salad": {"recall": 0.84, "precision": 0.86, "fp": 4, "fn": 8}, "packaging": {"recall": 0.9, "precision": 0.72, "fp": 28, "fn": 5}}, parent="B0")
    exp = RunIngestor(ws.memory, LocalExperimentSource(runs)).ingest("E001")
    assert exp.parent_id == "B0" and exp.checkpoint_path.endswith("ckpt.pt")
    a = RunAnalysisService(ws.memory).analyze("E001", baseline_id="B0")
    assert a.comparison.overall["map50_95"]["abs"] == __import__("pytest").approx(0.005)
    assert any(r.slice == "slice:small" and r.metric == "recall" for r in a.regressions)  # §49 slice regression despite mAP gain
    assert ws.memory.list_regressions(experiment_id="E001")

def test_workflow_c_diagnose_without_llm(tmp_path):
    ws, runs = seed(str(tmp_path))
    b0 = ws.memory.get_experiment("B0")
    d = diagnose(b0)  # no agent, no ResearchEngine (§§48,62,71)
    assert d.findings and any(f.category in ("small_object", "false_negative", "false_positive") for f in d.findings)
    assert diagnostics.analyze_false_positives(b0) and diagnostics.find_hard_negatives(b0)

def test_compare_mode_structured(tmp_path):
    ws, runs = seed(str(tmp_path))
    a = ws.memory.get_experiment("B0")
    b = Experiment(id="E002", metrics={"map50_95": 0.63}, slices=a.slices, per_class=a.per_class, config={"crop_p": 0.15})
    ws.memory.add_experiment(b)
    c = compare_runs(a, b)
    assert c.overall["map50_95"]["b"] == 0.63 and isinstance(c.improvements, list)

def test_workflow_d_and_e_ask_from_persistent_memory(tmp_path):
    ws, runs = seed(str(tmp_path))
    ws.memory.add_conclusion(ResearchConclusion(statement="Oversampling small objects gave weak improvement", confidence=0.7, supporting_experiments=["B0"]))
    ws.memory.add_user_note("Camera 17 had lens contamination")
    ans = ProjectQueryService(ws.memory).ask("did we already try oversampling small objects?")
    assert "Oversampling small objects" in ans.text and ans.evidence  # evidence IDs (§68)
    # Persistence across restart: new workspace object on same dir
    ws2 = ProjectWorkspace(str(tmp_path))
    assert ProjectQueryService(ws2.memory).ask("oversampling small objects").evidence

def test_lineage_graph(tmp_path):
    ws, runs = seed(str(tmp_path))
    for eid, parent in [("E001", "B0"), ("E004", "E001")]:
        ws.memory.add_experiment(Experiment(id=eid, parent_id=parent, config={"x": eid}, metrics={"map50_95": 0.6}))
    g = ExperimentGraph(ws.memory)
    assert [e.id for e in g.descendants("B0")] == ["E001", "E004"] or set(e.id for e in g.descendants("B0")) == {"E001", "E004"}
    assert g.diff("B0", "E001")["config_diff"]

def test_workflow_a_autonomous_uses_memory_and_dedup(tmp_path):
    ws, runs = seed(str(tmp_path))
    eng = ResearchEngine(ws)  # no executor: plans from memory, records hypotheses, fakes no training
    plan = eng.plan("reduce packaging false positives", budget=2)
    assert plan.prior_evidence["experiments"]  # §63 starts from existing knowledge
    r = eng.run("reduce packaging false positives", budget=2)
    assert r.hypotheses and ws.memory.list_hypotheses()
    dup = check_duplicate(ws.memory, {"lr": 0.01, "crop_p": 0.4})
    assert dup is not None and dup.already_tested  # §64

def test_sessions_status_timeline_distinct_baselines(tmp_path):
    ws, runs = seed(str(tmp_path))
    ws.memory.add_experiment(Experiment(id="E151", metrics={"map50_95": 0.638}, checkpoint_path="e151.pt"))
    ws.memory.add_experiment(Experiment(id="E154", metrics={"map50_95": 0.620}))
    ws.memory.set_working_baseline("E151")
    st = project_status(ws)
    assert st.best.id == "E151" and st.latest.id == "E154" and st.original_baseline.id == "B0" and st.working_baseline.id == "E151"  # §58
    s = SessionService(ws).start(); assert s.start_summary["best"] == "E151"
    SessionService(ws).finish(s.id)
    assert ws.memory.timeline()  # §65
    assert ws.memory.list_notes() == []
    ws.memory.add_user_note("lens note")
    assert ws.memory.list_notes()[0].provenance == "user_note"  # §66-67 distinct from observations

def test_watch_ingests_new_run(tmp_path):
    ws, runs = seed(str(tmp_path))
    w = WatchService(ws, runs)
    make_run(runs, "E200", {"map50_95": 0.62, "recall": 0.8, "precision": 0.8}, {}, BASE_SLICES, BASE_CLASS)
    ingested = w.scan_once()
    assert any(e.id == "E200" for e in ingested)

def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    assert main(["init"]) == 0
    ws, runs = seed(str(tmp_path))  # reuses workspace
    assert main(["status"]) == 0
    assert main(["experiments"]) == 0
    assert main(["diagnose", "B0"]) == 0
    assert main(["ask", "what about small objects?"]) == 0
    assert main(["note", "hello note"]) == 0
    assert main(["session", "start"]) == 0
    assert main(["session", "finish"]) == 0
