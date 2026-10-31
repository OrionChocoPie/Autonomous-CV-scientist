"""Persist and reuse validated source-field mappings without model requests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import uuid

from .importing import ImportTools, _json_bytes, _structured
from .journal import local_path, write_json, redact


def _hash(value): return hashlib.sha256(_json_bytes(value)).hexdigest()


def _shape(value):
    if isinstance(value, dict):
        shape = {key: _shape(item) for key, item in sorted(value.items())}
        # Row selectors must not silently start pointing at a different split/epoch.
        for key in ("", "split", "fold", "epoch", "dataset", "checkpoint"):
            if key in value and isinstance(value[key], (str, int, float)): shape[key] = [shape[key], value[key]]
        return shape
    if isinstance(value, list): return [_shape(item) for item in value]
    if type(value) in (int, float): return "number"
    if isinstance(value, str):
        try: float(value)
        except ValueError: return "string"
        return "number"
    return type(value).__name__


def _template(path, run):
    if not path: return ""
    try: return "run:" + str(PurePosixPath(path).relative_to(run))
    except ValueError: return "source:" + path


def _expand(path, run):
    if not path: return ""
    if path.startswith("run:"): return str(PurePosixPath(run) / path[4:])
    if path.startswith("source:"): return path[7:]
    raise ValueError("Invalid import profile path")


def _catalog(source):
    root = source.list_directory(".")
    result = {".": [[e["name"], e["kind"]] for e in root]}
    for entry in root:
        if entry["kind"] == "d" and entry["name"] in ("runs", "models", "experiments"):
            result[entry["name"]] = [[e["name"], e["kind"]] for e in source.list_directory(entry["name"])]
    return result


class ImportProfiles:
    def __init__(self, workspace): self.ws = workspace

    def _binding_path(self, source):
        return local_path(self.ws, "import/sources/" + _hash(source.uri) + ".json")

    def _profile(self, identity):
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid import profile ID")
        path = local_path(self.ws, "import/profiles/" + identity + ".json")
        profile = json.loads(path.read_text())
        if not isinstance(profile, dict) or profile.get("version") != 1 or _hash(profile) != identity:
            raise ValueError("Import profile is corrupt or unsupported")
        return profile

    def save(self, tools):
        bindings = {}
        for run, recipe in tools.recipes.items():
            template = deepcopy(recipe)
            template.pop("run_dir")
            # Completion status must be re-established for each new run.
            template["status"] = "unknown"
            template["notes"] = ["Imported using a saved field mapping; completion and dataset identity require verification"]
            for key in ("config_file", "checkpoint"): template[key] = _template(template[key], run)
            for ref in template["metrics"]: ref["file"] = _template(ref["file"], run)
            schemas = {_template(file, run): _hash(_shape(tools.documents[file])) for file in tools.prepared[run]["inputs"]}
            profile = {"version": 1, "recipe": template, "schemas": schemas}
            identity = _hash(profile)
            path = local_path(self.ws, "import/profiles/" + identity + ".json")
            if path.exists(): self._profile(identity)
            else: write_json(path, profile)
            bindings[run] = {"profile": identity, "status": recipe["status"], "notes": recipe["notes"]}
            self.ws.record_event("import.profile", "saved", profile_id=identity)
        if bindings:
            write_json(self._binding_path(tools.source), {"version": 1, "source": tools.source.uri,
                "runs": bindings, "catalog": _catalog(tools.source), "skipped": tools.skipped})

    def _prepare(self, tools, run, profile, binding=None):
        recipe = deepcopy(profile["recipe"])
        recipe["run_dir"] = run
        if binding:
            recipe["status"], recipe["notes"] = binding["status"], binding["notes"]
        if run != ".":
            parent = str(PurePosixPath(run).parent)
            tools.list_directory(parent)
        tools.list_directory(run)
        for template, expected in profile["schemas"].items():
            file = _expand(template, run)
            raw = tools.source.read(file)
            document = _structured(raw, PurePosixPath(file).suffix.lower())
            if _hash(_shape(document)) != expected: raise ValueError("Import source format changed: " + file)
            tools.raw[file], tools.documents[file] = raw, document
        for key in ("config_file", "checkpoint"): recipe[key] = _expand(recipe[key], run)
        checkpoint = recipe["checkpoint"]
        if checkpoint:
            tools.list_directory(str(PurePosixPath(checkpoint).parent))
        for ref in recipe["metrics"]: ref["file"] = _expand(ref["file"], run)
        tools.prepare_run(**recipe)

    def reuse(self, source):
        path = self._binding_path(source)
        if not path.exists(): return self._new_source(source)
        binding = json.loads(path.read_text())
        if not isinstance(binding, dict) or binding.get("version") != 1 or binding.get("source") != source.uri:
            raise ValueError("Invalid saved import source mapping")
        tools = ImportTools(source, self.ws)
        tools.list_directory(".")
        for run, entry in binding["runs"].items():
            self._prepare(tools, run, self._profile(entry["profile"]), entry)
        catalog = _catalog(source)
        if catalog != binding["catalog"]:
            old_dirs = {str(PurePosixPath(parent) / name) for parent, entries in binding["catalog"].items() for name, kind in entries if kind == "d"}
            new_dirs = {str(PurePosixPath(parent) / name) for parent, entries in catalog.items() for name, kind in entries if kind == "d"} - old_dirs
            if not new_dirs: return None
            identities = sorted({entry["profile"] for entry in binding["runs"].values()})
            for run in sorted(new_dirs):
                matches = []
                for identity in identities:
                    trial = ImportTools(source, self.ws)
                    trial.list_directory(".")
                    try: self._prepare(trial, run, self._profile(identity))
                    except (OSError, ValueError, KeyError): continue
                    matches.append((identity, trial.recipes[run]))
                if len(matches) != 1: return None
                identity, recipe = matches[0]
                self._prepare(tools, run, self._profile(identity))
        # Revisit previously skipped runs through the agent: missing results may have arrived.
        if binding.get("skipped"): return None
        outcomes = tools.publish()
        self.save(tools)
        self.ws.record_event("import.profile", "reused", runs=len(outcomes), requests=0)
        return self.save_report({"source": source.uri, "summary": "Applied saved import profiles", "limitations": [],
            "runs": outcomes, "skipped": {}, "usage": {"requests": 0, "total_tokens": 0}, "tools": [], "model": None, "profile_reused": True})

    def _new_source(self, source):
        directory = local_path(self.ws, "import/profiles")
        identities = sorted(path.stem for path in directory.glob("*.json"))
        if not identities: return None
        profiles = {identity: self._profile(identity) for identity in identities}
        # Shared source-level metrics cannot establish the identity of a new run.
        profiles = {identity: profile for identity, profile in profiles.items()
                    if all(ref["file"].startswith("run:") for ref in profile["recipe"]["metrics"])}
        catalog = _catalog(source)
        candidates = {"."}
        for parent, entries in catalog.items():
            candidates.update(str(PurePosixPath(parent) / name) for name, kind in entries if kind == "d" and name not in catalog)
        tools = ImportTools(source, self.ws)
        tools.list_directory(".")
        matched = set()
        for run in sorted(candidates):
            matches = []
            for identity, profile in profiles.items():
                trial = ImportTools(source, self.ws)
                trial.list_directory(".")
                try: self._prepare(trial, run, profile)
                except (OSError, ValueError, KeyError): continue
                matches.append(identity)
            if len(matches) > 1: raise ValueError("Ambiguous saved import profiles for " + run)
            if matches:
                self._prepare(tools, run, profiles[matches[0]])
                matched.add(run)
        if not matched: return None
        # Unknown directories may contain a new format; let the agent account for them.
        if candidates - matched - {"."}: return None
        if "." not in matched and any(kind == "f" and PurePosixPath(name).suffix in (".csv", ".json") for name, kind in catalog["."]):
            return None
        outcomes = tools.publish()
        self.save(tools)
        self.ws.record_event("import.profile", "reused", runs=len(outcomes), requests=0)
        return self.save_report({"source": source.uri, "summary": "Recognized source using saved import profiles", "limitations": [],
            "runs": outcomes, "skipped": {}, "usage": {"requests": 0, "total_tokens": 0}, "tools": [], "model": None, "profile_reused": True})

    def save_report(self, report):
        path = local_path(self.ws, f"reports/import-{uuid.uuid4().hex}.json")
        report["path"] = str(path)
        write_json(path, redact(report))
        self.ws.record_event("import", "completed", runs=len(report["runs"]), report_path=str(path), **report["usage"])
        return report
