"""Benchmark package facade for **sql_tool** (LanceDB + RAG LLM).

**deep_agent** (DuckDB + Deep Agent): use ``generate_sql.get_deep_agent_sql_response``.
Lazy-loaded via ``__getattr__``.
"""

from __future__ import annotations

import importlib
from typing import Any

from nemo_retriever.relational_db.benchmark.sql_tool.generate_sql import (
    get_sql_tool_response_top_k,
)


def generate_sql(query: str, top_k: int = 15) -> str:
    """Generate SQL via LanceDB retrieval + LLM (**sql_tool** benchmark)."""
    result = get_sql_tool_response_top_k(query, top_k=top_k)
    return (result.get("sql_code") or "").strip() or ""


def __getattr__(name: str) -> Any:
    """Lazy load deep_agent benchmark symbols to avoid importing DuckDB/deepagents on sql_tool-only use."""
    if name in ("get_deep_agent_sql_response", "get_sql_tool_response"):
        mod = importlib.import_module(
            "nemo_retriever.relational_db.benchmark.deep_agent.generate_sql"
        )
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_sql",
    "get_sql_tool_response_top_k",
    "get_deep_agent_sql_response",
    "get_sql_tool_response",
]
