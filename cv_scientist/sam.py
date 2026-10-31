"""SAM 3.1 Responses transport and image-box extraction (masks are not decoded)."""
from __future__ import annotations

import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .meta import MetaAPIError, MetaClient


MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_TOKEN = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(?:;([^<>]*?))?\|>")


def parse_boxes(text, width, height):
    """Read source-pixel boxes; convert inclusive SAM edges to normalized YOLO."""
    if not isinstance(text, str): raise MetaAPIError("SAM returned invalid output text")
    if text == "": return []
    boxes, identities = [], set()
    try:
        for line in text.splitlines():
            frame = re.match(r"<0f(?:;[^<>]*)?>", line)
            if frame is None: raise ValueError
            position = frame.end()
            count = 0
            while position < len(line):
                identity = re.match(r",?([0-9]+)", line[position:])
                if identity is None or identity[1] in identities: raise ValueError
                identities.add(identity[1])
                position += identity.end()
                tokens = {}
                while token := _TOKEN.match(line, position):
                    name, body = token.groups()
                    if name in ("box", "mask"):
                        if name in tokens: raise ValueError
                        tokens[name] = body
                    position = token.end()
                if set(tokens) != {"box", "mask"}: raise ValueError
                fields = {}
                for field in (tokens["box"] or "").split(";"):
                    key, _, value = field.partition("=")
                    key, value = key.strip(), value.strip()
                    if key in ("x1", "y1", "x2", "y2", "w", "h"):
                        if key in fields or not re.fullmatch(r"-?[0-9]+", value): raise ValueError
                        fields[key] = int(value)
                x1, y1, x2, y2 = (fields[key] for key in ("x1", "y1", "x2", "y2"))
                if (fields["w"], fields["h"]) != (width, height): raise ValueError
                if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height): raise ValueError
                boxes.append([(x1 + x2 + 1) / (2 * width), (y1 + y2 + 1) / (2 * height),
                              (x2 - x1 + 1) / width, (y2 - y1 + 1) / height])
                count += 1
            if not count: raise ValueError
    except (ValueError, KeyError):
        raise MetaAPIError("SAM returned malformed boxes or dimensions inconsistent with the image") from None
    return boxes


class SamClient(MetaClient):
    def segment(self, image_url, prompt, width, height):
        payload = {"model": "sam-3.1", "stream": False, "metadata": {"mask_encoding": "one_bit"},
                   "input": [{"type": "message", "role": "user", "content": [
                       {"type": "input_text", "text": prompt},
                       {"type": "input_image", "image_url": image_url}]}]}
        request = Request(self.config.base_url.rstrip("/") + "/responses",
                          data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.config.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            status = error.code
            error.close()
            hint = {401: "check CV_SCIENTIST_API_KEY or run cv-scientist config",
                    403: "check SAM model/account access", 404: "check sam-3.1 account access",
                    429: "rate limit reached; retry later"}.get(status, "request failed")
            raise MetaAPIError(f"SAM API HTTP {status}: {hint}") from None
        except (URLError, TimeoutError, socket.timeout):
            raise MetaAPIError("SAM API connection failed or timed out") from None
        if len(raw) > MAX_RESPONSE_BYTES: raise MetaAPIError("SAM API response is too large")
        try:
            body = json.loads(raw)
            if body.get("status") != "completed" or body.get("error"):
                raise MetaAPIError("SAM API did not complete its answer")
            texts = []
            for item in body["output"]:
                if item.get("type") == "reasoning": continue
                if item.get("type") != "message" or item.get("role") != "assistant": raise ValueError
                if item.get("status") not in (None, "completed"): raise ValueError
                for part in item["content"]:
                    if part.get("type") != "output_text" or not isinstance(part.get("text"), str): raise ValueError
                    texts.append(part["text"])
            if len(texts) != 1: raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError):
            raise MetaAPIError("SAM API returned an invalid response or refusal") from None
        return parse_boxes(texts[0], width, height)
