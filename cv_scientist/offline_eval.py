"""Offline runtime/guardrail evaluation with scripted model responses, not LLM grading."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from .confidence import ConfidenceResearch
from .investigation import InvestigationAgent
from .meta import MetaAPIError
from .models import Experiment
from .workspace import ProjectWorkspace


def _call(identity, name, **arguments):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _tools(*calls): return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _final(answer="Measured result; causal explanation remains unverified.", refs=None):
    return {"role": "assistant", "content": json.dumps({"answer": answer,
        "evidence": [{"kind": "experiment", "id": "E1"}] if refs is None else refs, "hypotheses": [], "next_steps": []})}


class OfflineClient:
    def __init__(self, responses, usage=20):
        self.config = SimpleNamespace(model="offline-scripted", max_tokens=4096)
        self.responses, self.requests, self.usage = iter(responses), 0, usage

    def complete(self, messages, tools, response_format, timeout=None):
        self.requests += 1
        try: response = next(self.responses)
        except StopIteration: raise RuntimeError("Offline scenario exhausted its scripted responses") from None
        usage = None if self.usage is None else {"prompt_tokens": self.usage, "completion_tokens": 0, "total_tokens": self.usage}
        return response, usage


def _workspace(root):
    ws = ProjectWorkspace.init(str(root))
    run = root / "run"
    run.mkdir()
    (run / "one.png").write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII="))
    (run / "predictions.json").write_text(json.dumps([{"image": "one.png", "truth": [[0, 0, 10, 10]],
        "predictions": [], "tp": 1, "fp": 0, "fn": 0}]))
    ws.memory.add_experiment(Experiment(id="B0", metrics={"map50_95": 0.7, "recall": 0.95, "fp": 20}))
    ws.memory.add_experiment(Experiment(id="E1", run_dir=str(run), metrics={"map50_95": 0.75, "recall": 0.8, "fp": 10}))
    return ws


class OfflineEvaluator:
    """Synthetic evaluator: all proposed settings fail the required recall."""
    def __init__(self, ws, baseline):
        self.ws, self.baseline = ws, baseline
        self.signature, self.snapshot, self.fixed, self.baseline_confidence = "offline-fixed", {}, {}, 0.25

    def check_inputs(self): return None

    def load_result(self, run, run_id, confidence):
        exp = self.ws.memory.get_experiment(run_id)
        if exp is None: raise ValueError("Missing offline result")
        return exp

    def evaluate(self, confidence, run_id, timeout, log_path):
        root = Path(self.ws.root) / "runs" / run_id
        root.mkdir(parents=True)
        Path(log_path).write_text("Synthetic offline evaluation; no subprocess or model inference\n")
        return Experiment(id=run_id, parent_id=self.baseline.id, run_dir=str(root),
            config={"research_signature": self.signature, "confidence": confidence},
            metrics={"tp": 18, "fp": 5, "fn": 2, "count": 1, "precision": 18/23, "recall": 0.9, "map50_95": 0.7})


def evaluate_offline():
    cases = []
    with tempfile.TemporaryDirectory(prefix="cv-agent-offline-") as folder, patch(
            "cv_scientist.meta.MetaClient.complete", side_effect=AssertionError("Paid API calls are forbidden in offline evaluation")) as paid:
        root = Path(folder)
        def run_case(name, responses, *, expected_error=None, check=None, usage=20, limits=None):
            ws = _workspace(root / name)
            client = OfflineClient(responses, usage)
            try:
                report = InvestigationAgent(ws, client, **(limits or {})).ask("Investigate existing evidence and its limitations")
            except (MetaAPIError, RuntimeError) as error:
                passed = expected_error is not None and expected_error in str(error)
                outcome = {"observed": type(error).__name__, "message": str(error), "passed": passed}
            else:
                passed = expected_error is None and (check(report) if check else True)
                outcome = {"observed": "answer", "evidence": [{"kind": r["kind"], "id": r["id"]} for r in report.evidence],
                           "tool_errors": sum("error" in c["result"] for c in report.tool_calls), "passed": bool(passed)}
            finally: ws.memory.close()
            cases.append({"name": name, "requests": client.requests, **outcome})

        run_case("metric_tradeoff", [_tools(_call("compare", "compare_experiments", baseline_id="B0", current_id="E1")),
            _final("mAP improved while recall fell; this is a measured trade-off.", [{"kind": "experiment", "id": x} for x in ("B0", "E1")])],
            check=lambda r: r.tool_calls[0]["result"]["overall"]["recall"]["abs"] < 0
                and r.tool_calls[0]["result"]["overall"]["map50_95"]["abs"] > 0 and len(r.evidence) == 2)
        run_case("contradictory_counts", [_tools(_call("diagnose", "diagnose_experiment", experiment_id="E1")),
            _final("Exported counts disagree with box matching; verify metrics before drawing conclusions.",
                   [{"kind": "artifact", "id": "E1:predictions.json#0"}])],
            check=lambda r: any(f["category"] == "artifact_consistency" for f in r.tool_calls[0]["result"]["findings"]))
        run_case("insufficient_evidence", [_tools(_call("search", "search_memory", query="unicorns astronomy zebras")),
            _final("No relevant evidence is available; a causal answer is unverified.", [])], check=lambda r: r.evidence == [])
        run_case("missing_artifact", [_tools(_call("examples", "get_error_examples", experiment_id="B0", kind="fn", limit=1)),
            _final("Images and prediction artifacts are unavailable; only aggregate metrics can be discussed.")],
            check=lambda r: "error" in r.tool_calls[0]["result"])
        run_case("unknown_tool", [_tools(_call("bad", "launch_training")), _final("Requested tool is unavailable.")],
                 check=lambda r: "error" in r.tool_calls[0]["result"])
        run_case("invalid_arguments", [_tools(_call("bad", "get_experiment", unsupported="E1")), _final("Tool arguments failed validation.")],
                 check=lambda r: "error" in r.tool_calls[0]["result"])
        run_case("fabricated_citation", [_final(refs=[{"kind": "experiment", "id": "invented"}])], expected_error="has not received")
        run_case("wrong_evidence_kind", [_final(refs=[{"kind": "artifact", "id": "E1"}])], expected_error="has not received")
        run_case("uncited_retrieved_facts", [_tools(_call("get", "get_experiment", experiment_id="E1")), _final(refs=[])], expected_error="did not cite")
        duplicate = _call("same", "get_experiment", experiment_id="E1")
        run_case("duplicate_tool_ids", [_tools(duplicate, duplicate)], expected_error="duplicate")
        run_case("step_limit", [_tools(_call("one", "get_experiment", experiment_id="E1"))], expected_error="step budget", limits={"max_steps": 1})
        run_case("tool_limit", [_tools(_call("one", "get_experiment", experiment_id="E1"), _call("two", "get_experiment", experiment_id="B0"))],
                 expected_error="tool-call budget", limits={"max_tool_calls": 1})
        run_case("token_limit", [_final()], usage=100001, expected_error="token budget")
        run_case("unknown_token_usage", [_final()], usage=None, expected_error="usage")
        run_case("image_limit", [_tools(_call("image1", "view_error", experiment_id="E1", image="one.png"),
            _call("image2", "view_error", experiment_id="E1", image="one.png")), _final()], limits={"max_images": 1},
            check=lambda r: r.tool_calls[0]["result"]["image_shown"] and "Image budget" in r.tool_calls[1]["result"]["error"])
        ws = ProjectWorkspace.init(str(root / "unreachable_goal"))
        ws.memory.add_experiment(Experiment(id="F1", metrics={"fp": 28, "recall": 1.0}))
        proposals = [{"role": "assistant", "content": json.dumps({"action": "evaluate", "parent_id": "F1", "confidence": confidence,
            "hypothesis": "Reduce FP", "expected_effect": "Preserve recall", "reason": "Synthetic offline scenario"})} for confidence in (0.4, 0.6)]
        client = OfflineClient(proposals)
        report = ConfidenceResearch(ws, client, evaluator_factory=OfflineEvaluator).run("Reduce FP", "F1", budget=2)
        cases.append({"name": "unreachable_goal", "requests": client.requests, "observed": report["status"], "attempts": len(report["attempts"]),
            "passed": report["status"] == "budget_exhausted" and report["selected_run"] is None and len(report["attempts"]) == 2
                and all(a["status"] == "rejected" for a in report["attempts"])})
        ws.memory.close()
        paid_calls = paid.call_count
    return {"evaluation": "scripted runtime guardrails, not live model answer quality", "cases": cases,
            "passed": sum(case["passed"] for case in cases), "total": len(cases), "paid_api_calls": paid_calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = evaluate_offline()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Offline evaluation: {report['passed']}/{report['total']}; paid API calls={report['paid_api_calls']}")
    return 0 if report["passed"] == report["total"] and not report["paid_api_calls"] else 1


if __name__ == "__main__": raise SystemExit(main())
