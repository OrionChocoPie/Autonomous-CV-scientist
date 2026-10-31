"""Reproducible offline memory benchmark; creates only a temporary workspace."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

from .compare import compare_runs
from .models import Experiment, Hypothesis, Observation
from .sessions import project_status
from .workspace import ProjectWorkspace


def memory_benchmark(experiments=3000, repeats=7):
    if type(experiments) is not int or experiments < 2 or type(repeats) is not int or repeats < 1:
        raise ValueError("Benchmark requires at least two experiments and one repeat")
    with tempfile.TemporaryDirectory(prefix="cv-memory-benchmark-") as folder:
        ws = ProjectWorkspace.init(folder)
        memory = ws.memory
        start = time.perf_counter()
        for i in range(experiments):
            memory.add_experiment(Experiment(id=f"E{i:05d}", name=f"detector fold {i % 5}", created_at=float(i),
                config={"imgsz": 640, "seed": i, "augmentation": "small objects"},
                metrics={"map50_95": 0.5 + (i % 1000) / 2000, "recall": 0.8, "precision": 0.9},
                slices={"small": {"recall": 0.6, "fp": i % 10, "fn": 5}},
                per_class={f"class_{c}": {"recall": 0.8, "precision": 0.9} for c in range(10)}))
            if i % 10 == 0:
                memory.add_observation(Observation(id=f"O{i:05d}", title=f"E{i:05d} small object recall regression", experiment_id=f"E{i:05d}"))
                memory.add_hypothesis(Hypothesis(id=f"H{i:05d}", statement=f"E{i:05d} increase resolution for small objects"))
        seeded_seconds = time.perf_counter() - start
        operations = {"search": lambda: memory.search(f"E{experiments - 1:05d} small recall"),
                      "status": lambda: project_status(ws).render(),
                      "compare": lambda: compare_runs(memory.get_experiment("E00000"), memory.get_experiment(f"E{experiments-1:05d}"))}
        results = {}
        for name, operation in operations.items():
            operation()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter(); operation(); samples.append((time.perf_counter() - start) * 1000)
            profiler = cProfile.Profile(); profiler.runcall(operation)
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(12)
            results[name] = {"median_ms": statistics.median(samples), "max_ms": max(samples),
                             "samples_ms": samples, "profile": stream.getvalue()}
        status = project_status(ws)
        comparison = operations["compare"]()
        report = {"experiments": experiments, "repeats": repeats, "python": platform.python_version(),
                  "platform": platform.system() + " " + platform.machine(), "seed_seconds": seeded_seconds,
                  "operations": results, "behavior": {"total": status.total_experiments,
                    "latest": status.latest.id, "best": status.best.id, "baseline": status.original_baseline.id,
                    "comparison": comparison.overall}}
        memory.close()
        return report


def baseline_benchmark(revision, experiments=3000, repeats=7):
    """Run the same harness against a Git snapshot without changing the checkout."""
    repository = Path(__file__).resolve().parent.parent
    resolved = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"],
                              cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
    archive = subprocess.run(["git", "archive", resolved, "cv_scientist"], cwd=repository, check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory(prefix="cv-memory-baseline-") as folder:
        root = Path(folder).resolve()
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            for member in contents:
                if not member.isfile() or not member.name.endswith(".py"): continue
                target = (root / member.name).resolve()
                if not target.is_relative_to(root): raise ValueError("Invalid repository archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                with contents.extractfile(member) as file: target.write_bytes(file.read())
        (root / "cv_scientist" / "benchmark.py").write_bytes(Path(__file__).read_bytes())
        output = root / "result.json"
        subprocess.run([sys.executable, "-m", "cv_scientist.benchmark", "--experiments", str(experiments),
                        "--repeats", str(repeats), "--output", str(output)], cwd=root, check=True,
                       capture_output=True, timeout=300)
        report = json.loads(output.read_text())
        report["source_revision"] = resolved
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-revision", help="Run against this Git revision in a temporary snapshot")
    args = parser.parse_args(argv)
    report = baseline_benchmark(args.baseline_revision, args.experiments, args.repeats) if args.baseline_revision else memory_benchmark(args.experiments, args.repeats)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in report["operations"].items(): print(f"{name}: median={result['median_ms']:.3f} ms")
    return 0


if __name__ == "__main__": raise SystemExit(main())
