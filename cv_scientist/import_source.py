"""Bounded file access and exclusive publication of local or SSH import copies."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit


OUTPUT_DIR = ".cv-scientist-import"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 1000


def relative_path(value, allow_root=False):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("Expected a relative source path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or (str(path) == "." and not allow_root):
        raise ValueError("Source paths must stay inside the selected directory")
    return str(path)


class ImportSource:
    def __init__(self, location, output_dir=OUTPUT_DIR):
        self.output_dir = relative_path(output_dir)
        if not isinstance(location, str) or not location.strip(): raise ValueError("Import source must not be empty")
        self.host, self.port = None, None
        if location.startswith("ssh://"):
            parsed = urlsplit(location)
            if not parsed.hostname or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Expected ssh://[user@]host/absolute/path without credentials")
            self.host = ((parsed.username + "@") if parsed.username else "") + parsed.hostname
            self.port = parsed.port
            root = unquote(parsed.path)
        elif re.match(r"^[\w.@-]+:/", location):
            self.host, root = location.split(":", 1)
        else:
            self.root = Path(location).expanduser().resolve()
            if not self.root.is_dir(): raise FileNotFoundError(self.root)
            self.uri = self.root.as_uri()
            return
        if not re.fullmatch(r"(?:[\w.-]+@)?[\w][\w.-]*", self.host) or not root.startswith("/") or "\x00" in root:
            raise ValueError("Invalid SSH host or source path")
        self.root = self._ssh(f"cd -- {shlex.quote(root)} && pwd -P").decode().strip()
        if not self.root.startswith("/") or "\n" in self.root: raise ValueError("Invalid remote source directory")
        authority = self.host + (f":{self.port}" if self.port else "")
        self.uri = f"ssh://{authority}{self.root}"

    def _ssh(self, command, data=None):
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
                "-o", "ForwardAgent=no"]
        if self.port: args += ["-p", str(self.port)]
        args += ["--", self.host, "sh -c " + shlex.quote(command)]
        try:
            result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                                    env={k: v for k, v in os.environ.items() if k != "CV_SCIENTIST_API_KEY"})
        except subprocess.TimeoutExpired:
            raise OSError("SSH source operation timed out") from None
        if result.returncode: raise OSError(f"SSH source operation failed (exit {result.returncode})")
        return result.stdout

    def _local(self, relative):
        relative = relative_path(relative, allow_root=True)
        path = self.root
        for part in PurePosixPath(relative).parts:
            path = path / part
            if path.is_symlink(): raise ValueError("Import paths must not traverse symlinks")
        return path

    def _remote(self, relative):
        relative = relative_path(relative, allow_root=True)
        path, guards = self.root, []
        for part in PurePosixPath(relative).parts:
            path += "/" + part
            guards.append("[ ! -L " + shlex.quote(path) + " ]")
        return path, " && ".join(guards + ["true"])

    def location(self, relative):
        relative = relative_path(relative, allow_root=True)
        if not self.host: return str(self.root / relative)
        return self.uri.rstrip("/") + ("/" + relative if relative != "." else "")

    def list_directory(self, relative="."):
        if self.host:
            path, guards = self._remote(relative)
            raw = self._ssh(f"{guards} && test -d {shlex.quote(path)} && (find -P {shlex.quote(path)} -mindepth 1 -maxdepth 1 -printf '%f\\0%y\\0%s\\0' | head -c 262145)")
            if len(raw) > 262144: raise ValueError("Directory listing exceeds the import limit")
            fields = raw.decode().split("\0")[:-1]
            if len(fields) % 3: raise ValueError("Invalid SSH directory listing")
            entries = [{"name": fields[i], "kind": fields[i + 1], "size": int(fields[i + 2])} for i in range(0, len(fields), 3)]
        else:
            entries = []
            for path in self._local(relative).iterdir():
                if len(entries) >= MAX_ENTRIES: raise ValueError("Directory has too many entries; select a narrower source")
                entries.append({"name": path.name, "kind": "l" if path.is_symlink() else "d" if path.is_dir() else "f",
                                "size": path.lstat().st_size})
        entries = [e for e in entries if not e["name"].startswith(".") and e["name"] not in ("venv", "node_modules", "__pycache__")]
        if len(entries) > MAX_ENTRIES: raise ValueError("Directory has too many entries; select a narrower source")
        return sorted(entries, key=lambda e: e["name"])

    def read(self, relative):
        if self.host:
            path, guards = self._remote(relative)
            raw = self._ssh(f"{guards} && test -f {shlex.quote(path)} && head -c {MAX_FILE_BYTES + 1} -- {shlex.quote(path)}")
        else:
            with self._local(relative).open("rb") as file: raw = file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES: raise ValueError("Source file exceeds the 2 MiB import limit")
        return raw

    def exists(self, relative):
        if self.host:
            path, guards = self._remote(relative)
            return self._ssh(f"{guards} && if [ -e {shlex.quote(path)} ]; then printf yes; fi") == b"yes"
        return self._local(relative).exists()

    def publish(self, relative, files):
        """Publish a new directory atomically; existing copies are never replaced."""
        relative = relative_path(relative)
        if not relative.startswith(self.output_dir + "/"): raise ValueError("Copies must stay inside the import output directory")
        if any(not re.fullmatch(r"[a-z][a-z0-9_.-]*", name) for name in files): raise ValueError("Invalid copy filename")
        if self.host:
            path, guards = self._remote(relative)
            parent = str(PurePosixPath(path).parent)
            # mkdir is exclusive, and all publishers use the same lock.
            output, output_guards = self._remote(self.output_dir)
            lock = output + "/.write-lock"
            self._ssh(f"{guards} && {output_guards} && mkdir -p -- {shlex.quote(parent)} && mkdir -- {shlex.quote(lock)}")
            stage = None
            try:
                if self.exists(relative): raise FileExistsError(self.location(relative))
                stage = self._ssh(f"mktemp -d {shlex.quote(parent + '/.pending-XXXXXX')}").decode().strip()
                if not stage.startswith(parent + "/.pending-") or "\n" in stage: raise ValueError("Invalid SSH staging directory")
                for name, raw in files.items(): self._ssh("cat > " + shlex.quote(stage + "/" + name), raw)
                self._ssh(f"mv -T --no-clobber -- {shlex.quote(stage)} {shlex.quote(path)} && test ! -d {shlex.quote(stage)}")
            finally:
                if stage is not None and stage.startswith(parent + "/.pending-") and "\n" not in stage:
                    self._ssh("rm -rf -- " + shlex.quote(stage))
                self._ssh("rmdir -- " + shlex.quote(lock))
        else:
            path = self._local(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = self._local(self.output_dir + "/.write-lock")
            lock.mkdir()
            stage = None
            try:
                if path.exists(): raise FileExistsError(path)
                stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=path.parent))
                for name, raw in files.items(): (stage / name).write_bytes(raw)
                stage.rename(path)
            finally:
                if stage is not None and stage.exists(): shutil.rmtree(stage)
                lock.rmdir()
