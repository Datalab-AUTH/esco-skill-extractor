"""
Ollama /api/chat via httpx.

Avoids ``import ollama``: that package instantiates a default ``Client()`` at import
time, which calls ``platform.machine()`` and can hang on Windows (WMI).
"""

from __future__ import annotations

import ipaddress
import os
import urllib.parse
from typing import Any

import httpx

_DEFAULT_TIMEOUT = 600.0


def _parse_ollama_host(host: str | None) -> str:
    """Match ollama-python host parsing (default port 11434, OLLAMA_HOST when host is None)."""
    raw = host if host is not None and str(host).strip() else os.environ.get("OLLAMA_HOST")
    port_default = 11434
    h, port = raw or "", port_default
    scheme, _, hostport = h.partition("://")
    if not hostport:
        scheme, hostport = "http", h
    elif scheme == "http":
        port = 80
    elif scheme == "https":
        port = 443

    split = urllib.parse.urlsplit(f"{scheme}://{hostport}")
    hostname = split.hostname or "127.0.0.1"
    port = split.port or port

    try:
        if isinstance(ipaddress.ip_address(hostname), ipaddress.IPv6Address):
            hostname = f"[{hostname}]"
    except ValueError:
        pass

    if path := split.path.strip("/"):
        return f"{scheme}://{hostname}:{port}/{path}"

    return f"{scheme}://{hostname}:{port}"


def ollama_chat(
    *,
    host: str | None,
    model: str,
    messages: list[dict[str, Any]],
    format: str | None = None,
    options: dict[str, Any] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """POST /api/chat (non-streaming); return assistant message content."""
    base = _parse_ollama_host(host).rstrip("/")
    url = f"{base}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if format is not None:
        payload["format"] = format
    if options is not None:
        payload["options"] = options

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        raise ConnectionError(
            f"Could not reach Ollama at {base}. Is the server running?"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Ollama HTTP {e.response.status_code}: {e.response.text[:500]}"
        ) from e

    msg = data.get("message") or {}
    content = msg.get("content")
    if content is None:
        raise RuntimeError(f"Unexpected Ollama response shape: {data!r}")
    return content
