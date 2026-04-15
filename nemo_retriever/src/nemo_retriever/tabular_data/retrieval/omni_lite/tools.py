"""LangChain tools for the Phase 2 SQL Deep Agent.

Phase 2 receives a pre-built ``RetrievalContext`` from Phase 1 and only needs
to generate SQL and validate it.  Retrieval tools and the execute_sql tool
have been removed:
- Retrieval (``extract_entities``, ``retrieve_semantic_candidates``) is now
  handled entirely by Phase 1 (``retrieval_agent_runtime.py``).
- Execution is now a plain Python function in ``main.py`` (Phase 3).

Factory
-------
Use ``build_omni_lite_tools(payload, llm)`` to get a list of bound tools and
an ``ExecutionStore`` for the session.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_validation import (
    query_validation,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution store
# ---------------------------------------------------------------------------


class ExecutionStore:
    """Mutable per-request store written by ``validate_sql``, read by Phase 3.

    ``validate_sql`` writes the last SQL that passed validation.
    ``sql`` is ``None`` until validation succeeds.
    """

    def __init__(self) -> None:
        self.sql: str | None = None

    def as_answer(self) -> dict | None:
        """Return a partial answer dict if validated SQL is present, else ``None``."""
        if self.sql is None:
            return None
        return {
            "sql_code": self.sql,
            "answer": "",
            "result": None,
            "semantic_elements": [],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_sql_fences(sql: str) -> str:
    """Strip markdown code fences from a SQL string.

    The agent is instructed to wrap SQL in triple-backtick fences
    (````sql ... ```) to prevent double-quote characters from breaking
    the JSON tool-call payload.  This helper removes those fences so the
    underlying validation layer always receives plain SQL.

    Handles:
    - ````sql\\n...\\n```` (language tag)
    - ` ``` \\n...\\n``` ` (no language tag)
    - Plain SQL with no fences (returned unchanged)
    """
    stripped = sql.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end]).strip()
    return sql


# ---------------------------------------------------------------------------
# Tool — validate_sql
# ---------------------------------------------------------------------------


def _make_validate_sql_tool(dialects: list[str] | None, store: ExecutionStore):
    """Return a ``validate_sql`` tool bound to *dialects* and *store*."""

    _dialects = dialects or []

    @tool
    def validate_sql(sql: str) -> str:
        """Validate a SQL query for schema correctness and SELECT-only enforcement.

        Call this AFTER generating SQL.  Fix and retry up to 4 times if invalid.

        Args:
            sql: The SQL string to validate.  May be wrapped in triple-backtick
                markdown fences (``\\`\\`\\`sql ... \\`\\`\\``); fences are stripped
                automatically before validation.

        Returns:
            JSON object with:
              - ``valid``: bool
              - ``error``: error message string (empty when valid)
              - ``sql_columns``: list of column IDs referenced by the query
        """
        sql = _strip_sql_fences(sql)
        try:
            result = query_validation(None, sql, _dialects, user_participants=[])
            if result.get("error"):
                return json.dumps(
                    {
                        "valid": False,
                        "error": result["error"],
                        "sql_columns": [],
                    }
                )
            store.sql = sql
            return json.dumps(
                {
                    "valid": True,
                    "error": "",
                    "sql_columns": result.get("sql_columns") or [],
                }
            )
        except Exception as exc:
            logger.warning("validate_sql failed: %s", exc)
            return json.dumps({"valid": False, "error": str(exc), "sql_columns": []})

    return validate_sql


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_omni_lite_tools(payload: AgentPayload, llm: Any) -> tuple[list, ExecutionStore]:
    """Build and return the list of Phase 2 SQL Agent tools and a shared ``ExecutionStore``.

    Phase 2 only needs ``validate_sql``.  Retrieval tools have moved to Phase 1
    (``retrieval_tools.py``) and SQL execution has moved to Phase 3 (plain
    function in ``main.py``).

    Args:
        payload: The ``AgentPayload`` from the caller.
        llm: The LLM client (unused at tool construction time; reserved for
            future LLM-assisted validation).

    Returns:
        A tuple of:
        - ``[validate_sql]`` — list containing the single Phase 2 tool.
        - ``ExecutionStore`` instance populated by ``validate_sql`` as the
          agent runs.
    """
    dialects = payload.get("dialects") or []
    store = ExecutionStore()

    return [
        _make_validate_sql_tool(dialects, store),
    ], store


__all__ = ["build_omni_lite_tools", "ExecutionStore", "_strip_sql_fences"]
