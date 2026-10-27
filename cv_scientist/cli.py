"""Daily-use CLI (§69). Analysis commands never launch ResearchEngine (§62)."""
from __future__ import annotations
import argparse, json, os, sys

def _ws(args):
    from .workspace import ProjectWorkspace
    ws = ProjectWorkspace.discover(args.project or ".")
    if ws is None:
        print("No .cv-scientist workspace found. Run: cv-scientist init", file=sys.stderr); raise SystemExit(2)
    return ws
def _dump(o):
    print(json.dumps(o.to_dict() if hasattr(o, "to_dict") else o, indent=2, default=str))

def main(argv=None):
    p = argparse.ArgumentParser(prog="cv-scientist")
    p.add_argument("--project", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sp = sub.add_parser("ingest"); sp.add_argument("run")
    sp = sub.add_parser("compare"); sp.add_argument("a"); sp.add_argument("b")
    sp = sub.add_parser("diagnose"); sp.add_argument("run", nargs="?")
    sp = sub.add_parser("regressions"); sp.add_argument("run", nargs="?")
    sp = sub.add_parser("ask"); sp.add_argument("question")
    sp.add_argument("--agent", choices=["offline", "meta"], default=None)
    sp.add_argument("--model", default=None)
    sp.add_argument("--max-steps", type=int, default=8)
    sp = sub.add_parser("note"); sp.add_argument("text")
    sub.add_parser("experiments"); sub.add_parser("hypotheses"); sub.add_parser("conclusions")
    sp = sub.add_parser("session"); sp.add_argument("action", choices=["start", "finish"])
    sp = sub.add_parser("watch"); sp.add_argument("runs_dir")
    sp = sub.add_parser("research"); sp.add_argument("--goal", required=True); sp.add_argument("--budget", type=int, default=4)
    sp.add_argument("--agent", choices=["offline", "meta"], default="offline")
    sp.add_argument("--baseline")
    sp.add_argument("--model")
    sp.add_argument("--min-recall", type=float, default=0.95)
    sp.add_argument("--max-fp", type=int, default=28)
    sp.add_argument("--timeout", type=float, default=600)
    sp = sub.add_parser("timeline")
    args = p.parse_args(argv)

    if args.cmd == "init":
        from .workspace import ProjectWorkspace
        ws = ProjectWorkspace.init(args.project or ".")
        print(f"Initialized CV Scientist workspace at {ws.dir} (project: {ws.name})"); return 0
    ws = _ws(args); m = ws.memory

    if args.cmd == "status":
        from .sessions import project_status; print(project_status(ws).render())
    elif args.cmd == "ingest":
        from .ingestor import RunIngestor, LocalExperimentSource
        from .analysis import RunAnalysisService
        exp = RunIngestor(m, LocalExperimentSource()).ingest(args.run)
        analysis = RunAnalysisService(m).analyze(exp.id)
        print(f"Ingested {exp.id} (parent={exp.parent_id}) metrics={exp.metrics}")
        print(f"Regressions: {len(analysis.regressions)}, observations: {len(analysis.observations)}")
    elif args.cmd == "compare":
        from .compare import compare_runs
        a, b = m.get_experiment(args.a), m.get_experiment(args.b)
        if not a or not b: print("unknown experiment", file=sys.stderr); return 2
        c = compare_runs(a, b); m.add_comparison(c)
        print(f"Overall: {c.overall}"); print("Improvements:"); [print(f"- {x}") for x in c.improvements]
        print("Regressions:"); [print(f"- {r['slice']} {r['metric']} {r['relative_delta']:+.1%}") for r in c.regressions]
        if c.likely_tradeoff: print(f"Likely trade-off: {c.likely_tradeoff}")
    elif args.cmd == "diagnose":
        from .diagnose import diagnose, summarize
        exp = m.get_experiment(args.run) if args.run else m.latest_experiment()
        if not exp: print("no experiment", file=sys.stderr); return 2
        d = diagnose(exp, m.working_baseline()); print(summarize(d))
    elif args.cmd == "regressions":
        from . import diagnostics
        exp = m.get_experiment(args.run) if args.run else m.latest_experiment()
        base = m.working_baseline()
        if not exp or not base: print("need experiment and baseline", file=sys.stderr); return 2
        for r in diagnostics.find_regressions(exp, base): print(f"[{r.severity}] {r.slice} {r.metric}: {r.previous_value} -> {r.current_value} ({r.relative_delta:+.1%})")
    elif args.cmd == "ask":
        provider = args.agent or ws.config.get("agent_provider", "offline")
        if provider == "meta":
            import logging
            from .meta import MetaClient, MetaConfig, MetaAPIError
            from .investigation import InvestigationAgent
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            try:
                report = InvestigationAgent(ws, MetaClient(MetaConfig.from_workspace(ws, args.model)), max_steps=args.max_steps).ask(args.question)
            except (MetaAPIError, ValueError, OSError) as error:
                print(f"Agent error: {error}", file=sys.stderr); return 2
            print(report.answer)
            if report.hypotheses:
                print("Hypotheses:"); [print(f"- {h}") for h in report.hypotheses]
            if report.next_steps:
                print("Next steps:"); [print(f"- {s}") for s in report.next_steps]
            print(f"Evidence: {[e['kind'] + ':' + e['id'] for e in report.evidence]}")
            print(f"Report: {report.path}")
        elif provider == "offline":
            if args.model is not None:
                print("--model requires --agent meta", file=sys.stderr); return 2
            from .query import ProjectQueryService
            ans = ProjectQueryService(m).ask(args.question); print(ans.text); print(f"Evidence: {[e['id'] for e in ans.evidence]}")
        else:
            print("Unknown agent_provider in project.yaml", file=sys.stderr); return 2
    elif args.cmd == "note":
        n = m.add_user_note(args.text); print(f"Added note {n.id}")
    elif args.cmd == "experiments":
        for e in m.list_experiments(): print(f"{e.id}\t{e.name}\tparent={e.parent_id}\tmetrics={e.metrics}")
    elif args.cmd == "hypotheses":
        for h in m.list_hypotheses(): print(f"{h.id}\t[{h.status}] {h.statement}")
    elif args.cmd == "conclusions":
        for c in m.list_conclusions(): print(f"{c.id}\t[{c.status} conf={c.confidence}] {c.statement}")
    elif args.cmd == "session":
        from .sessions import SessionService
        s = SessionService(ws).start() if args.action == "start" else SessionService(ws).finish()
        _dump(s)
    elif args.cmd == "watch":
        from .sessions import WatchService
        print("Watching... (Ctrl+C to stop)")
        try: WatchService(ws, args.runs_dir).watch()
        except KeyboardInterrupt: pass
    elif args.cmd == "research":
        if args.agent == "meta":
            import logging
            from .confidence import ConfidenceResearch
            from .evaluation import EvaluationError
            from .meta import MetaClient, MetaConfig, MetaAPIError
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            if not args.baseline:
                print("Meta research requires --baseline", file=sys.stderr); return 2
            try:
                report = ConfidenceResearch(ws, MetaClient(MetaConfig.from_workspace(ws, args.model))).run(
                    args.goal, args.baseline, budget=args.budget, min_recall=args.min_recall, max_fp=args.max_fp, timeout=args.timeout)
            except (MetaAPIError, EvaluationError, ValueError, OSError, KeyError) as error:
                print(f"Research error: {error}", file=sys.stderr); return 2
            print(f"Status: {report['status']}\n{report['reason']}\nEvaluations: {len(report['attempts'])}/{args.budget}")
            print(f"Selected: {report['selected_run']}\nBest feasible: {report['best_feasible_run']}\nReport: {report['path']}")
        else:
            from .engine import ResearchEngine
            r = ResearchEngine(ws).run(args.goal, budget=args.budget)
            print(f"Goal: {r.goal}\nProposals executed: {r.iterations}, hypotheses: {len(r.hypotheses)}, skipped duplicates: {r.skipped_duplicates}")
    elif args.cmd == "timeline":
        for ev in m.timeline(): print(f"{ev.created_at:.0f}\t{ev.kind}\t{ev.ref_id}\t{ev.description}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
