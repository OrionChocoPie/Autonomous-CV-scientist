"""Meta Model API chat transport and local credential loading. Stdlib only."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class MetaAPIError(RuntimeError):
    """A configured provider failed; never silently fall back to offline output."""


def read_env(path: Path) -> dict:
    if not path.is_file(): return {}
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            raise ValueError(f"Invalid .env entry at line {number}")
        try: parts = shlex.split(value, comments=True)
        except ValueError: raise ValueError(f"Invalid .env quoting at line {number}") from None
        if len(parts) > 1: raise ValueError(f"Quote .env values containing spaces (line {number})")
        values[key.strip()] = parts[0] if parts else ""
    return values


@dataclass
class MetaConfig:
    api_key: str = field(repr=False)
    model: str = "muse-wonder-1.0"
    base_url: str = "https://api.meta.ai/v1"
    timeout: float = 90
    max_tokens: int = 4096

    def __post_init__(self):
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("Set CV_SCIENTIST_API_KEY or run: cv-scientist config")
        if not isinstance(self.model, str) or not self.model.strip(): raise ValueError("Meta model must not be empty")
        parsed = urlsplit(self.base_url)
        local_http = parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if not parsed.hostname or (parsed.scheme != "https" and not local_http) or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Meta base URL must use HTTPS (HTTP is allowed only on localhost)")
        if not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Meta timeout must be positive and finite")
        if type(self.max_tokens) is not int or self.max_tokens < 1: raise ValueError("max_tokens must be positive")

    @classmethod
    def from_workspace(cls, workspace, model=None):
        # Target-project values override cwd .env; process environment wins over both.
        values = read_env(Path.cwd() / ".env")
        values.update(read_env(Path(workspace.root) / ".env"))
        values.update(os.environ)
        from .credentials import load_global_api_key
        api_key = values["CV_SCIENTIST_API_KEY"] if "CV_SCIENTIST_API_KEY" in values else load_global_api_key()
        return cls(api_key=api_key,
                   model=model if model is not None else values.get("META_MODEL") or workspace.config.get("agent_model", "muse-wonder-1.0"),
                   base_url=values.get("META_API_BASE_URL", "https://api.meta.ai/v1"))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward credentials to another host.


class MetaClient:
    def __init__(self, config: MetaConfig):
        self.config = config
        self._opener = build_opener(_NoRedirect())

    def complete(self, messages, tools, response_format, timeout=None):
        timeout = self.config.timeout if timeout is None else min(self.config.timeout, timeout)
        if not math.isfinite(timeout) or timeout <= 0: raise ValueError("Request timeout must be positive and finite")
        payload = {"model": self.config.model, "messages": messages,
                   "response_format": response_format, "max_completion_tokens": self.config.max_tokens,
                   "reasoning_effort": "medium"}
        if tools: payload["tools"] = tools
        request = Request(self.config.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except HTTPError as error:
            status = error.code
            error.close()
            hint = {401: "check CV_SCIENTIST_API_KEY or run cv-scientist config", 403: "check model/account access", 429: "rate limit reached; retry later"}.get(status, "request failed")
            raise MetaAPIError(f"Meta API HTTP {status}: {hint}") from None
        except (URLError, TimeoutError, socket.timeout):
            raise MetaAPIError("Meta API connection failed or timed out") from None
        if len(raw) > 8 * 1024 * 1024: raise MetaAPIError("Meta API response is too large")
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") not in ("stop", "tool_calls"):
                raise MetaAPIError("Meta API did not complete its answer (output limit or refusal)")
            if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("refusal"):
                raise MetaAPIError("Meta API returned no usable assistant message")
            calls = message.get("tool_calls")
            if calls is not None and not isinstance(calls, list): raise TypeError
            if not calls and not isinstance(message.get("content"), str): raise TypeError
            usage = body.get("usage")
            if usage is not None:
                if not isinstance(usage, dict): raise TypeError
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if type(usage.get(key)) is not int or usage[key] < 0: raise TypeError
        except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
            raise MetaAPIError("Meta API returned an invalid response") from None
        return message, usage
