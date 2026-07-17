"""Bestiary tools — HTTP calls via stdlib urllib only (zero extra deps)."""

from __future__ import annotations

import json
import os
import urllib.request

from tools.registry import tool_error, tool_result

_BESTIARY_URL = lambda: os.environ.get("BESTIARY_URL", "http://localhost:8787").rstrip("/")


def _check_bestiary_available() -> bool:
    try:
        urllib.request.urlopen(f"{_BESTIARY_URL()}/health", timeout=5)
        return True
    except Exception:
        return False


def _handle_bestiary_resolve(args: dict, **kw) -> str:
    query = args.get("query", "")
    body = json.dumps({"input": query}).encode()
    req = urllib.request.Request(
        f"{_BESTIARY_URL()}/v0/resolve",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status < 200 or resp.status >= 300:
                return tool_error(f"bestiary resolve failed: {resp.status}")
            return tool_result(json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        return tool_error(f"bestiary resolve failed: {exc.code}")
    except Exception as exc:
        return tool_error(f"bestiary resolve error: {exc}")


def _handle_bestiary_signals(args: dict, **kw) -> str:
    canonical_token_id = args.get("canonical_token_id", "")
    window_days = int(args.get("window_days") or 30)
    url = f"{_BESTIARY_URL()}/v0/tokens/{canonical_token_id}/signals?windowDays={window_days}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status < 200 or resp.status >= 300:
                return tool_error(f"bestiary signals failed: {resp.status}")
            return tool_result(json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        return tool_error(f"bestiary signals failed: {exc.code}")
    except Exception as exc:
        return tool_error(f"bestiary signals error: {exc}")


def _handle_bestiary_health(args: dict, **kw) -> str:
    try:
        with urllib.request.urlopen(f"{_BESTIARY_URL()}/health", timeout=5) as resp:
            return tool_result(json.loads(resp.read()))
    except Exception as exc:
        return tool_error(f"bestiary health check failed: {exc}")


BESTIARY_RESOLVE_SCHEMA = {
    "name": "bestiary_resolve",
    "description": "Resolve a crypto token by any identifier and return its canonical data from the Bestiary substrate.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The token to look up: a symbol (e.g. 'WETH'), EVM address (e.g. '0x...'), or coingecko_id (e.g. 'weth').",
            },
        },
        "required": ["query"],
    },
}

BESTIARY_SIGNALS_SCHEMA = {
    "name": "bestiary_signals",
    "description": "Fetch windowed social signal aggregates for a token from the Bestiary substrate.",
    "parameters": {
        "type": "object",
        "properties": {
            "canonical_token_id": {
                "type": "string",
                "description": "The canonical token ID as returned by bestiary_resolve.",
            },
            "window_days": {
                "type": "integer",
                "description": "Number of days to look back for signals.",
            },
        },
        "required": ["canonical_token_id"],
    },
}

BESTIARY_HEALTH_SCHEMA = {
    "name": "bestiary_health",
    "description": "Health check for the Bestiary API — confirms the service is reachable.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
