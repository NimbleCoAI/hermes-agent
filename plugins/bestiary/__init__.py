from plugins.bestiary.tools import (
    BESTIARY_RESOLVE_SCHEMA, _handle_bestiary_resolve, _check_bestiary_available,
    BESTIARY_SIGNALS_SCHEMA, _handle_bestiary_signals,
    BESTIARY_HEALTH_SCHEMA, _handle_bestiary_health,
)


def register(ctx) -> None:
    ctx.register_tool("bestiary_resolve", toolset="bestiary", schema=BESTIARY_RESOLVE_SCHEMA,
                      handler=_handle_bestiary_resolve, check_fn=_check_bestiary_available, emoji="🔍")
    ctx.register_tool("bestiary_signals", toolset="bestiary", schema=BESTIARY_SIGNALS_SCHEMA,
                      handler=_handle_bestiary_signals, check_fn=_check_bestiary_available, emoji="📊")
    ctx.register_tool("bestiary_health", toolset="bestiary", schema=BESTIARY_HEALTH_SCHEMA,
                      handler=_handle_bestiary_health, check_fn=_check_bestiary_available, emoji="💚")
