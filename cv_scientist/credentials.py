"""User-wide API credentials shared by installed CV Scientist projects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def global_credentials_path():
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    if not root.is_absolute(): raise ValueError("XDG_CONFIG_HOME must be an absolute path")
    return root / "cv-scientist" / "credentials.json"


def _validate_key(api_key):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Set CV_SCIENTIST_API_KEY or run: cv-scientist config")


def load_global_api_key():
    path = global_credentials_path()
    try: raw = path.read_text(encoding="utf-8")
    except FileNotFoundError: return ""
    try:
        value = json.loads(raw)
        if not isinstance(value, dict): raise ValueError
        api_key = value.get("api_key")
        _validate_key(api_key)
    except ValueError:
        raise ValueError("Invalid global credentials; run: cv-scientist config") from None
    return api_key


def save_global_api_key(api_key):
    _validate_key(api_key)
    path = global_credentials_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            os.fchmod(file.fileno(), 0o600)
            json.dump({"api_key": api_key}, file)
            file.write("\n"); file.flush(); os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return path
