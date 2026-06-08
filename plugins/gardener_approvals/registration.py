"""register(ctx) — wire the pre_llm_call hook. Enabling the plugin
(`hermes plugins enable gardener_approvals`) is the opt-in.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    from .handler import on_pre_llm_call
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("gardener_approvals active: pre_llm_call trigger registered")
