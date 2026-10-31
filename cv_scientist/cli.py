"""Daily-use CLI (§69). Analysis commands never launch ResearchEngine (§62)."""
from __future__ import annotations
import argparse, json, os, sys

from .agent_output import CommandError, response_json


def _fail(args, code, message):
    if args.json: raise CommandError(code, message)
    print(message, file=sys.stderr)
    return 2


def _human(args, *values, **kwargs):
    if not args.json: print(*values, **kwargs)


def _error(error):
    import sqlite3
    from .meta import MetaAPIError
    if isinstance(error, CommandError): return error
    if isinstance(error, MetaAPIError): code = "provider_error"
    elif isinstance(error, FileNotFoundError): code = "file_not_found"
    elif isinstance(error, OSError): code = "io_error"
    elif isinstance(error, sqlite3.Error): code = "database_error"
    elif isinstance(error, (ValueError, KeyError)): code = "invalid_data"
    else: return CommandError("internal_error", f"{type(error).__name__}: {error}", 1)
    return CommandError(code, str(error))


def _parser(json_mode):
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            if json_mode: raise CommandError("invalid_arguments", message)
            super().error(message)

        def print_help(self, file=None):
            if json_mode: print(response_json("help", {"help": self.format_help()}), file=file)
            else: super().print_help(file)
    return Parser(prog="cv-scientist")

def _ws(args):
    from .workspace import ProjectWorkspace
    ws = ProjectWorkspace.discover(args.project or ".")
    if ws is None:
        _fail(args, "workspace_not_found", "No .cv-scientist workspace found. Run: cv-scientist init")
        raise SystemExit(2)
    return ws
def _dump(o):
    print(json.dumps(o.to_dict() if hasattr(o, "to_dict") else o, indent=2, default=str))

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    options = argv[:argv.index("--")] if "--" in argv else argv
    json_mode = "--json" in options
    state = {"command": None, "json": json_mode}
    try:
        return _main(argv, json_mode, state)
    except KeyboardInterrupt:
        if not state["json"]: raise
        error = CommandError("interrupted", "Operation interrupted", 130)
    except Exception as exception:
        if not state["json"]: raise
        error = _error(exception)
    print(response_json(state["command"], error=error))
    return error.exit_code


def _main(argv, json_mode, state):
    p = _parser(json_mode)
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true", help="Emit one versioned JSON response")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config", help="Save a user-wide API key using a hidden prompt")
    sub.add_parser("init")
    sub.add_parser("status")
    for name in ("context", "search"):
        sp = sub.add_parser(name, help="Read bounded project context" if name == "context" else "Search project history without an LLM")
        if name == "search": sp.add_argument("query")
        sp.add_argument("--limit", type=int, choices=range(1, 101), default=5, metavar="1..100")
    sp = sub.add_parser("ingest"); sp.add_argument("run")
    sp = sub.add_parser("import", help="Import results into .cv-scientist using saved local profiles")
    sp.add_argument("source", help="Local directory, host:/absolute/path, or ssh://host/absolute/path")
    sp.add_argument("--agent", choices=["meta"], default="meta")
    sp.add_argument("--model", default=None)
    sp.add_argument("--max-steps", type=int, default=64)
    sp = sub.add_parser("compare"); sp.add_argument("a"); sp.add_argument("b")
    sp = sub.add_parser("diagnose"); sp.add_argument("run", nargs="?")
    sp = sub.add_parser("report", help="Generate a local HTML evidence report")
    sp.add_argument("run", nargs="?")
    sp.add_argument("--baseline")
    sp.add_argument("--research")
    sp = sub.add_parser("regressions"); sp.add_argument("run", nargs="?")
    sp = sub.add_parser("ask"); sp.add_argument("question")
    sp.add_argument("--agent", choices=["offline", "meta"], default=None)
    sp.add_argument("--model", default=None)
    sp.add_argument("--max-steps", type=int, default=8)
    sp = sub.add_parser("review-dataset", help="Review a local image/YOLO dataset with Meta and save an evidence report")
    sp.add_argument("source", help="Local dataset directory, optionally containing data.yaml or dataset.yaml")
    sp.add_argument("--agent", choices=["meta"], default="meta")
    sp.add_argument("--model", default=None)
    sp.add_argument("--max-steps", type=int, default=8)
    sp = sub.add_parser("note"); sp.add_argument("text")
    sub.add_parser("experiments"); sub.add_parser("hypotheses"); sub.add_parser("conclusions")
    sp = sub.add_parser("session"); sp.add_argument("action", choices=["start", "finish"])
    sp = sub.add_parser("watch"); sp.add_argument("runs_dir")
    sp = sub.add_parser("research"); sp.add_argument("--goal"); sp.add_argument("--budget", type=int, default=4)
    sp.add_argument("--resume", help="Continue an existing research journal with its original budgets")
    sp.add_argument("--agent", choices=["offline", "meta"], default="offline")
    sp.add_argument("--baseline")
    sp.add_argument("--model")
    sp.add_argument("--parameter", choices=["confidence", "imgsz"], default=None)
    sp.add_argument("--min-recall", type=float, default=0.95)
    sp.add_argument("--max-fp", type=int, default=28)
    sp.add_argument("--timeout", type=float, default=600)
    sp = sub.add_parser("timeline")
    for sp in sub.choices.values():
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Emit one versioned JSON response")
    args = p.parse_args(argv)
    state["command"] = args.cmd
    state["json"] = args.json
    if args.cmd == "config":
        import getpass
        import warnings
        from .credentials import save_global_api_key
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                api_key = getpass.getpass("CV Scientist API key (hidden): ")
            path = save_global_api_key(api_key)
        except getpass.GetPassWarning:
            return _fail(args, "input_required", "Configuration requires a terminal; set CV_SCIENTIST_API_KEY for noninteractive use")
        except (EOFError, KeyboardInterrupt):
            return _fail(args, "interrupted", "Configuration cancelled")
        except (OSError, ValueError) as error:
            return _fail(args, _error(error).code, f"Configuration error: {error}")
        if args.json: print(response_json(args.cmd, {"path": path}))
        else: print(f"API key saved to {path}")
        return 0
    from .workspace import ProjectWorkspace
    ws = ProjectWorkspace.discover(args.project or ".")
    if ws is None and args.cmd in ("init", "import"): ws = ProjectWorkspace(args.project or ".")
    if ws is None:
        result = _dispatch(args)
        if args.json:
            print(response_json(args.cmd, result)); return 0
        return result
    invocation = ws.record_event("cli." + args.cmd, "started")["id"]
    try:
        result = _dispatch(args)
        encoded = response_json(args.cmd, result) if args.json else None
        code = 0 if args.json else result
    except BaseException as error:
        ws.record_event("cli." + args.cmd, "failed" if isinstance(error, Exception) else "interrupted", error_type=type(error).__name__, invocation=invocation)
        raise
    ws.record_event("cli." + args.cmd, "completed" if code == 0 else "failed", exit_code=code, invocation=invocation)
    if args.json: print(encoded)
    return code


def _dispatch(args):

    if args.cmd == "init":
        from .workspace import ProjectWorkspace
        ws = ProjectWorkspace.init(args.project or ".")
        _human(args, f"Initialized CV Scientist workspace at {ws.dir} (project: {ws.name})")
        return {"root": ws.root, "workspace": ws.dir, "name": ws.name} if args.json else 0
    if args.cmd == "import":
        import logging
        from .importing import ImportAgent
        from .import_source import ImportSource
        from .meta import MetaClient, MetaConfig, MetaAPIError
        from .workspace import ProjectWorkspace
        from dataclasses import replace
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            # Configuration only: importing copies never opens or modifies ResearchMemory.
            ws = ProjectWorkspace.discover(args.project or ".") or ProjectWorkspace(args.project or ".")
            source = ImportSource(args.source)
            def client_factory():
                return MetaClient(replace(MetaConfig.from_workspace(ws, args.model), max_tokens=16384))
            report = ImportAgent(source, None, max_steps=args.max_steps, workspace=ws, client_factory=client_factory).run()
        except (MetaAPIError, ValueError, OSError) as error:
            return _fail(args, _error(error).code, f"Import error: {error}")
        for run in report["runs"]: _human(args, f"{run['status']}: {run['run_dir']} -> {run['copy']}")
        for run, reason in report["skipped"].items(): _human(args, f"skipped: {run}: {reason}")
        _human(args, report["summary"])
        for limitation in report["limitations"]: _human(args, f"Limitation: {limitation}")
        _human(args, f"Report: {report['path']}")
        return report if args.json else 0
    ws = _ws(args); m = ws.memory

    if args.cmd == "status":
        from .sessions import project_status
        status = project_status(ws)
        data = {"status": status}
        _human(args, status.render())
    elif args.cmd == "context":
        from .agent_output import research_context
        data = research_context(ws, args.limit)
        if not args.json: _dump(data)
    elif args.cmd == "search":
        from .agent_output import search_history
        data = search_history(ws, args.query, args.limit)
        if not args.json: _dump(data)
    elif args.cmd == "ingest":
        from .ingestor import RunIngestor, LocalExperimentSource
        from .analysis import RunAnalysisService
        try:
            with m.transaction():
                exp = RunIngestor(m, LocalExperimentSource()).ingest(args.run)
                analysis = RunAnalysisService(m).analyze(exp.id)
        except (OSError, ValueError) as error:
            return _fail(args, _error(error).code, f"Ingest error: {error}")
        data = {"experiment": exp, "analysis": analysis, "evidence": [{"kind": "experiment", "id": exp.id}]}
        _human(args, f"Ingested {exp.id} (parent={exp.parent_id}) metrics={exp.metrics}")
        _human(args, f"Regressions: {len(analysis.regressions)}, observations: {len(analysis.observations)}")
    elif args.cmd == "compare":
        from .compare import compare_runs
        a, b = m.get_experiment(args.a), m.get_experiment(args.b)
        if not a or not b: return _fail(args, "experiment_not_found", "unknown experiment")
        c = compare_runs(a, b); m.add_comparison(c)
        data = {"comparison": c, "evidence": [{"kind": "experiment", "id": a.id}, {"kind": "experiment", "id": b.id},
                                                {"kind": "comparison", "id": c.id}]}
        _human(args, f"Overall: {c.overall}"); _human(args, "Improvements:"); [_human(args, f"- {x}") for x in c.improvements]
        _human(args, "Regressions:"); [_human(args, f"- {r['slice']} {r['metric']} {r['relative_delta']:+.1%}") for r in c.regressions]
        if c.likely_tradeoff: _human(args, f"Likely trade-off: {c.likely_tradeoff}")
    elif args.cmd == "diagnose":
        from .diagnose import diagnose, summarize
        exp = m.get_experiment(args.run) if args.run else m.latest_experiment()
        if not exp: return _fail(args, "experiment_not_found", "no experiment")
        d = diagnose(exp, m.working_baseline())
        data = {"diagnosis": d, "evidence": [{"kind": "experiment", "id": exp.id}]}
        _human(args, summarize(d))
    elif args.cmd == "report":
        from .reporting import generate_report
        try: path = generate_report(ws, args.run, args.baseline, args.research)
        except (OSError, ValueError, KeyError) as error:
            return _fail(args, _error(error).code, f"Report error: {error}")
        data = {"path": path}
        _human(args, f"Report: {path}")
    elif args.cmd == "regressions":
        from . import diagnostics
        exp = m.get_experiment(args.run) if args.run else m.latest_experiment()
        base = m.working_baseline()
        if not exp or not base: return _fail(args, "experiment_not_found" if not exp else "baseline_not_found", "need experiment and baseline")
        regressions = diagnostics.find_regressions(exp, base)
        data = {"regressions": regressions, "experiment_id": exp.id, "baseline_id": base.id,
                "evidence": [{"kind": "experiment", "id": exp.id}, {"kind": "experiment", "id": base.id}]}
        for r in regressions: _human(args, f"[{r.severity}] {r.slice} {r.metric}: {r.previous_value} -> {r.current_value} ({r.relative_delta:+.1%})")
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
                return _fail(args, _error(error).code, f"Agent error: {error}")
            data = {"provider": provider, "report": report}
            _human(args, report.answer)
            if report.hypotheses:
                _human(args, "Hypotheses:"); [_human(args, f"- {h}") for h in report.hypotheses]
            if report.next_steps:
                _human(args, "Next steps:"); [_human(args, f"- {s}") for s in report.next_steps]
            _human(args, f"Evidence: {[e['kind'] + ':' + e['id'] for e in report.evidence]}")
            _human(args, f"Report: {report.path}")
        elif provider == "offline":
            if args.model is not None:
                return _fail(args, "invalid_arguments", "--model requires --agent meta")
            from .query import ProjectQueryService
            ans = ProjectQueryService(m).ask(args.question)
            data = {"provider": provider, "answer": ans.text, "evidence": ans.evidence, "confidence": ans.confidence}
            _human(args, ans.text); _human(args, f"Evidence: {[e['id'] for e in ans.evidence]}")
        else:
            return _fail(args, "invalid_arguments", "Unknown agent_provider in project.yaml")
    elif args.cmd == "review-dataset":
        import logging
        from .dataset_review import DatasetReviewAgent
        from .meta import MetaClient, MetaConfig, MetaAPIError
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            report = DatasetReviewAgent(ws, MetaClient(MetaConfig.from_workspace(ws, args.model)),
                                        max_steps=args.max_steps).review(args.source)
        except (MetaAPIError, ValueError, OSError) as error:
            return _fail(args, _error(error).code, f"Dataset review error: {error}")
        data = {"provider": "meta", "report": report}
        _human(args, report.answer)
        for hypothesis in report.hypotheses: _human(args, f"Hypothesis: {hypothesis}")
        for step in report.next_steps: _human(args, f"Next step: {step}")
        _human(args, f"Report: {report.path}")
    elif args.cmd == "note":
        n = m.add_user_note(args.text); _human(args, f"Added note {n.id}")
        data = {"note": n}
    elif args.cmd == "experiments":
        from .agent_output import referenced
        experiments = m.list_experiments()
        data = {"experiments": [referenced(e, "experiment") for e in experiments]}
        for e in experiments: _human(args, f"{e.id}\t{e.name}\tparent={e.parent_id}\tmetrics={e.metrics}")
    elif args.cmd == "hypotheses":
        from .agent_output import referenced
        hypotheses = m.list_hypotheses()
        data = {"hypotheses": [referenced(h, "hypothesis") for h in hypotheses]}
        for h in hypotheses: _human(args, f"{h.id}\t[{h.status}] {h.statement}")
    elif args.cmd == "conclusions":
        from .agent_output import referenced
        conclusions = m.list_conclusions()
        data = {"conclusions": [referenced(c, "conclusion") for c in conclusions]}
        for c in conclusions: _human(args, f"{c.id}\t[{c.status} conf={c.confidence}] {c.statement}")
    elif args.cmd == "session":
        from .sessions import SessionService
        s = SessionService(ws).start() if args.action == "start" else SessionService(ws).finish()
        data = {"session": s}
        if not args.json: _dump(s)
    elif args.cmd == "watch":
        from .sessions import WatchService
        _human(args, "Watching... (Ctrl+C to stop)")
        watcher = WatchService(ws, args.runs_dir)
        interrupted = False
        try: events = watcher.watch()
        except KeyboardInterrupt:
            events = watcher.events; interrupted = True
        data = {"events": events, "interrupted": interrupted}
    elif args.cmd == "research":
        if args.parameter and args.agent != "meta" and not args.resume:
            return _fail(args, "invalid_arguments", "--parameter requires --agent meta")
        if args.agent == "meta" or args.resume:
            import logging
            from .confidence import ConfidenceResearch
            from .evaluation import EvaluationError
            from .meta import MetaClient, MetaConfig, MetaAPIError
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            if not args.baseline and not args.resume:
                return _fail(args, "invalid_arguments", "Meta research requires --baseline")
            if not args.goal and not args.resume:
                return _fail(args, "invalid_arguments", "Meta research requires --goal")
            try:
                from pathlib import Path
                from .resolution import ResolutionResearch
                parameter = args.parameter or "confidence"
                if args.resume:
                    path = Path(args.resume)
                    if path.is_symlink() or path.resolve().parent != (Path(ws.dir) / "reports").resolve():
                        raise ValueError("Resume requires a research report in this workspace")
                    saved = json.loads(path.read_text())
                    if not isinstance(saved, dict): raise ValueError("Invalid research journal")
                    parameter = saved.get("parameter", "confidence")
                    if parameter not in ("confidence", "imgsz") or (args.parameter and args.parameter != parameter):
                        raise ValueError("Research parameter does not match the journal")
                research = ResolutionResearch if parameter == "imgsz" else ConfidenceResearch
                runner = research(ws, MetaClient(MetaConfig.from_workspace(ws, args.model)))
                report = runner.resume(args.resume) if args.resume else runner.run(
                    args.goal, args.baseline, budget=args.budget, min_recall=args.min_recall, max_fp=args.max_fp, timeout=args.timeout)
            except (MetaAPIError, EvaluationError, ValueError, OSError, KeyError) as error:
                return _fail(args, "invalid_data" if isinstance(error, EvaluationError) else _error(error).code, f"Research error: {error}")
            data = {"provider": "meta", "report": report}
            _human(args, f"Status: {report['status']}\n{report['reason']}\nEvaluations: {len(report['attempts'])}/{report['experiment_budget']}")
            _human(args, f"Selected: {report['selected_run']}\nBest feasible: {report['best_feasible_run']}\nReport: {report['path']}")
        else:
            if not args.goal:
                return _fail(args, "invalid_arguments", "Research requires --goal")
            from .engine import ResearchEngine
            r = ResearchEngine(ws).run(args.goal, budget=args.budget)
            data = {"provider": "offline", "result": r}
            _human(args, f"Goal: {r.goal}\nProposals executed: {r.iterations}, hypotheses: {len(r.hypotheses)}, skipped duplicates: {r.skipped_duplicates}")
    elif args.cmd == "timeline":
        events = m.timeline()
        data = {"events": events}
        for ev in events: _human(args, f"{ev.created_at:.0f}\t{ev.kind}\t{ev.ref_id}\t{ev.description}")
    return data if args.json else 0

if __name__ == "__main__": raise SystemExit(main())
