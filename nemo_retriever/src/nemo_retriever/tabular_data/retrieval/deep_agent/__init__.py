"""Benchmark package facade for deep_agent benchmark symbols.

Expose deep-agent entrypoints lazily to avoid importing DuckDB/deepagents until needed.
"""

from __future__ import annotations

import importlib
from typing import Any

def __getattr__(name: str) -> Any:
    """Lazy load deep_agent benchmark symbols to avoid importing DuckDB/deepagents on sql_tool-only use."""
    if name in ("get_deep_agent_sql_response", "get_sql_tool_response"):
        mod = importlib.import_module(
            "nemo_retriever.relational_db.benchmark.deep_agent.generate_sql"
        )
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_deep_agent_sql_response",
    "get_sql_tool_response",
]
